"""Jarvis V3 entry point — constructs every module and wires them together.

Wiring rules (see docs/specs/2026-08-25-jarvis-v3-overhaul.md and the
personal-assistant spec docs/specs/2026-08-26-jarvis-personal-assistant.md,
section 11):
- MainWindow attaches the bus to Tk itself; nothing here may call bus.attach_tk.
- Typed input flows ONLY through services.dispatch_text (MainWindow runs it on
  a worker thread); the CommandBar's UserUtterance(source='typed') event is
  display-only. Discord text enters through the same door with
  source='discord'.
- Voice flows: RecordingStopped -> _process_audio worker -> Transcribed +
  UserUtterance(source='voice') -> commander.
- Speech happens ONLY via JarvisApp._say (talkback-gated); JarvisReply events
  are display-only. Modules that need speech (timekeeper) get _say injected,
  never the TTS.
- Every assistant member of the services namespace may be None when its
  module failed to import or construct (spec 2.2): consumers use
  getattr(services, name, None), and boot never aborts on a tool module.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
import time
from types import SimpleNamespace

from jarvis.config import CONFIG, MACHINE, PATHS
from jarvis.events import (AlarmFired, ApprovalRequested, ApprovalResolved,
                           BriefingReady, ClaudeProgress, ClaudeTaskState,
                           JarvisReply, ModelInfo, RecordingStopped,
                           ReminderFired, Status, Transcribed,
                           UncertainResolved, UncertainUtterance,
                           UserUtterance, bus)
from jarvis.logs import get_logger

from jarvis import brain as brain_mod
from jarvis import desktop as desktop_mod
from jarvis import speak_queue, voice_check
from jarvis.assistant_config import AssistantConfig
from jarvis.brain import JarvisBrain
from jarvis.commander import COURTESY_REPLIES, Commander, parse_yes_no
from jarvis.context import ContextEngine
from jarvis.history import TypedHistory
from jarvis.hotword import Hotword
from jarvis.jarvis_agent import JarvisAgent
from jarvis.memory import JarvisMemory
from jarvis.reader import CONTINUE_PROMPT, ReadAloud
from jarvis.recorder import MicArbiter, Recorder, play_beep
from jarvis.speaker import SpeakerVerifier
from jarvis.tools.registry import ToolRegistry
from jarvis.transcriber import Transcriber
from jarvis.tts import TTS
from jarvis.workflows import Workflows

log = get_logger("app")

# Tool modules registered into the ToolRegistry at boot (spec 4.1). Each
# exposes make_tools(cfg, services) -> list[ToolSpec]; an import or
# registration failure logs and skips that module, never aborts boot.
TOOL_MODULES = (
    "jarvis.tools.location",
    "jarvis.tools.weather",
    "jarvis.tools.calendar",
    "jarvis.tools.timekeeper",
    "jarvis.tools.notes",
    "jarvis.tools.mail",
    "jarvis.tools.briefing",
    "jarvis.tools.spotify",
)

# Fixed persona lines the app itself speaks (spec 3.4); prewarmed. The
# same strings live in jarvis.approvals / jarvis.commander; kept literal
# here so the app never depends on an optional module for a spoken line.
APPROVAL_TIMEOUT_LINE = "No answer in two minutes, sir; I've declined it."
ALLOWED_LINE = "Allowed, sir."
DECLINED_LINE = "Declined, sir."

# Spoken when an answer is slow to arrive. Cached at startup: an uncached XTTS
# line took 12.6 s to render on 2026-08-27, which would make the reassurance
# slower than the answer it is meant to cover.
THINKING_LINES = [
    "Checking right now, sir. One moment.",
    "Looking into it now, sir.",
    "One moment, sir — I'm checking.",
]
THINKING_DELAY_S = 2.0          # only speak up if the answer is slower than this
TURN_TIMEOUT_S = 60.0           # watchdog: a lost reply must not wedge the turn

_YES_WORDS = frozenset({"yes", "y", "yeah", "yep", "yup", "aye", "allow",
                        "allowed", "approve", "approved", "ok", "okay", "sure",
                        "affirmative", "permit", "proceed", "go", "ahead", "do", "it"})
_NO_WORDS = frozenset({"no", "n", "nope", "nah", "deny", "denied", "decline",
                       "declined", "don't", "dont", "negative", "reject", "refuse"})

DISCORD_ACTIVE_S = 600.0        # a Discord exchange stays "active" this long


def yes_no(text: str):
    """True / False for an approval answer, None when the text is neither
    ("yes", "allow it", "no thanks", "deny"). Used for Discord replies to a
    pending permission question (spec 8.2)."""
    t = re.sub(r"[^a-z' ]+", " ", (text or "").lower()).strip()
    if not t:
        return None
    words = [w for w in t.split() if w not in ("please", "sir", "jarvis", "thanks")]
    if not words or len(words) > 4:
        return None
    yes = any(w in _YES_WORDS for w in words)
    no = any(w in _NO_WORDS for w in words)
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None


def _import_optional(modname: str):
    """Import an assistant module; a failure logs with context and yields
    None so the app runs without that feature (spec 2.2)."""
    try:
        return importlib.import_module(modname)
    except Exception:                                  # noqa: BLE001 - boot boundary
        log.exception("assistant module %s failed to import; running without it",
                      modname)
        return None


class JarvisApp:
    def __init__(self):
        # ---- assistant config first: everything below reads it ------------
        self.assistant = AssistantConfig.load()
        self._discord_active_until = 0.0
        self._discord_last_post = ("", 0.0)
        self._last_milestone: dict[str, str] = {}
        self._assistant_started = False
        self._quitting = False

        # ---- intelligence -------------------------------------------------
        self.memory = JarvisMemory()
        self.context = ContextEngine(memory=self.memory)
        self.brain = JarvisBrain(self.context, self.memory)
        # brain.configure(assistant.local_model): module-level in brain.py;
        # env JARVIS_OLLAMA_MODEL wins inside it.
        try:
            brain_mod.configure(self.assistant.local_model)
        except Exception:
            log.exception("brain.configure(%s) failed", self.assistant.local_model)
        self.agent = JarvisAgent()          # retained V1 tools (see spec note)

        # ---- speech -------------------------------------------------------
        # The arbiter is built here, ahead of the mic consumers below, because
        # TTS needs it too: hotword.py's contract lists "TTS talk-back" as a
        # consumer that pauses the wake-word stream, and until it did, the
        # always-on hotword could hear Jarvis's own voice.
        self.arbiter = MicArbiter()
        self.tts = TTS(gpu=0, engine=CONFIG.tts_engine, arbiter=self.arbiter)
        speak_queue.set_sink(self._say)
        speak_queue.start_watcher()
        self.reader = ReadAloud(self.tts)       # "read the clipboard"
        self.history = TypedHistory()           # typed-command history

        # ---- audio in -----------------------------------------------------
        self.speaker = SpeakerVerifier(gpu=0, threshold=CONFIG.speaker_threshold)
        # Load the stored voiceprint now -- one npz read. Without it
        # `speaker.enrolled` stays False and BOTH gates (the wake-word gate
        # below and the transcript filter in _process_audio) silently do
        # nothing no matter how the settings are configured.
        self.speaker.load()
        self.recorder = Recorder(self.arbiter, speaker_verifier=self.speaker)
        self.transcriber = Transcriber()
        # speaker= gates the wake word itself: a non-enrolled voice never
        # reaches _on_hotword, so the TV no longer opens a recording at all.
        self.hotword = Hotword(self.arbiter, self._mic_index, self._on_hotword,
                               speaker=self.speaker)

        # ---- actions ------------------------------------------------------
        self.desktop = desktop_mod.DesktopControl()
        self.workflows = Workflows(say=self._say)
        # workflows.Reminders is superseded by the timekeeper (spec 2.1); its
        # reminders.json is imported once in _make_timekeeper.

        # ---- personal assistant (spec section 11 order) -------------------
        self.tools = ToolRegistry()
        self.timekeeper = self._construct("timekeeper", self._make_timekeeper)
        self.notes = self._construct("notes", self._make_notes)
        self.approvals = self._construct("approvals", self._make_approvals)
        self.claude = self._construct("claude", self._make_claude)
        self.router = self._construct("router", self._make_router)
        self.alerts = self._construct("alerts", self._make_alerts)
        self.discord = self._construct("discord", self._make_discord)
        if self.alerts is not None:
            try:
                self.alerts.attach(self.discord)
            except Exception:
                log.exception("alerts.attach(discord) failed")
            if self.timekeeper is not None:
                # One toaster: the Alerts hub fans AlarmFired / ReminderFired
                # out to notify-send AND Discord, so the timekeeper's own
                # notify-send is switched off (it would double every toast).
                self.timekeeper.notify_enabled = False

        # ---- routing ------------------------------------------------------
        self.services = self._build_services()
        self._register_tools()
        self.commander = Commander(self.services)
        # Without this hook the commander falls back to a bare warn Status --
        # a 4 s toast with no way to answer it, after which the utterance is
        # dropped and resolve_uncertain (and the classifier feedback it feeds)
        # is never reached from the running app at all.
        self.commander.on_uncertain = self._on_uncertain
        self._pending_uncertain: dict = {}      # request_id -> utterance
        self._uncertain_lock = threading.Lock()
        # Set while a captured clip is being transcribed. recorder.recording
        # is already False by then, so it cannot serve as the guard.
        self._audio_busy = threading.Event()
        self._thinking_i = 0             # rotates THINKING_LINES
        # A turn is NOT over when handle() returns: brain.chat runs on a
        # worker thread and the commander says so with done=False. This
        # stays set until the reply actually lands.
        self._turn_busy = threading.Event()
        self._turn_timer = None          # slow-answer filler
        self._turn_watchdog = None

        bus.subscribe(RecordingStopped, self._on_recording_stopped)
        bus.subscribe(ClaudeProgress, self._on_claude_progress)
        bus.subscribe(ClaudeTaskState, self._on_claude_state)
        bus.subscribe(ApprovalRequested, self._on_approval_requested)
        bus.subscribe(ApprovalResolved, self._on_approval_resolved)
        bus.subscribe(AlarmFired, self._on_alarm_fired)
        bus.subscribe(ReminderFired, self._on_reminder_fired)
        bus.subscribe(JarvisReply, self._on_reply_for_discord)

        if CONFIG.target_name:
            self.desktop.restore_target(CONFIG.target_name)

    # ------------------------------------------------------ construction
    def _construct(self, name, factory):
        """Build one assistant member; a failure logs with context and
        leaves the member None so the rest of the app still boots."""
        try:
            obj = factory()
            if obj is None:
                log.warning("assistant: %s unavailable", name)
            return obj
        except Exception:                              # noqa: BLE001 - boot boundary
            log.exception("assistant: %s failed to construct; running without it",
                          name)
            return None

    def _make_timekeeper(self):
        mod = _import_optional("jarvis.tools.timekeeper")
        if mod is None:
            return None
        tk = mod.Timekeeper(PATHS.TIMEKEEPER_DB, say=self._say, cfg=self.assistant)
        try:
            n = tk.import_legacy(PATHS.REMINDERS)      # renames it *.migrated
            if n:
                log.info("timekeeper: imported %d legacy reminder(s) from %s",
                         n, PATHS.REMINDERS)
        except Exception:
            log.exception("timekeeper: legacy reminder import failed")
        return tk

    def _make_notes(self):
        mod = _import_optional("jarvis.tools.notes")
        return None if mod is None else mod.NotesStore(PATHS.NOTES_DB)

    def _make_approvals(self):
        mod = _import_optional("jarvis.approvals")
        if mod is None:
            return None
        return mod.ApprovalBroker(PATHS.APPROVALS_SOCK, self.assistant,
                                  on_request=self._on_approval,
                                  on_resolved=self._on_approval_done)

    def _make_claude(self):
        mod = _import_optional("jarvis.claude_session")
        if mod is None:
            return None
        return mod.ClaudeSessionManager(self.assistant, self.brain, self.approvals,
                                        PATHS.CLAUDE_PROJECTS, PATHS.CLAUDE_TASK_DIR)

    def _make_router(self):
        mod = _import_optional("jarvis.router")
        if mod is None:
            return None
        return mod.Router(self.assistant, classify=self.brain.classify_route)

    def _make_alerts(self):
        mod = _import_optional("jarvis.channels.notify")
        return None if mod is None else mod.Alerts(self.assistant)

    def _make_discord(self):
        mod = _import_optional("jarvis.channels.discord")
        if mod is None:
            return None
        return mod.DiscordChannel(self.assistant, on_message=self._on_discord)

    def _register_tools(self):
        """tools.register_many(m.make_tools(assistant, services)) for every
        tool module; failures log and skip. Then brain.set_registry(tools)."""
        for modname in TOOL_MODULES:
            mod = _import_optional(modname)
            if mod is None:
                continue
            try:
                specs = list(mod.make_tools(self.assistant, self.services) or [])
                self.tools.register_many(specs)
                log.info("tools: %s -> %s", modname.rsplit(".", 1)[-1],
                         ", ".join(s.name for s in specs) or "(none)")
            except Exception:                          # noqa: BLE001 - boot boundary
                log.exception("tools: %s.make_tools failed; skipped", modname)
        try:
            brain_mod.set_registry(self.tools)
        except Exception:
            log.exception("brain.set_registry failed")
        log.info("tools registered: %s", self.tools.names())

    # ---------------------------------------------------------------- speech
    def _say(self, text):
        if text and CONFIG.talkback:
            self.tts.speak(text)

    def interrupt_speech(self) -> bool:
        """Barge-in: cut whatever Jarvis is saying (and any queued lines,
        including a read-aloud in progress). Returns True when something
        was actually cut off. Wired to typed input below; the UI may also
        call it on the first keystroke (see scratchpad ui_hooks_todo.md)."""
        try:
            self.reader.stop() if self.reader.pending_chunks else None
            return self.tts.interrupt()
        except Exception:
            log.exception("interrupt_speech failed")
            return False

    def _canned_phrases(self):
        """Short lines Jarvis says verbatim and often — rendered into the
        speech cache at startup so they play instantly."""
        phrases = [p for lines in COURTESY_REPLIES.values() for p in lines]
        phrases += list(THINKING_LINES)
        phrases += [CONTINUE_PROMPT, "Very good, sir.",
                    "I haven't said anything yet, sir.",
                    "The clipboard is empty, sir.",
                    "Nothing is highlighted, sir.",
                    "Sir, this is your reminder.",
                    APPROVAL_TIMEOUT_LINE, ALLOWED_LINE, DECLINED_LINE]
        try:
            phrases += self.assistant.setup_lines()
        except Exception:
            log.debug("setup lines unavailable", exc_info=True)
        # Fixed lines owned by the assistant modules (spec 3.4).
        for modname, names in (
                ("jarvis.router", ("ROUTER_QUESTION",)),
                # BUSY_LINE is a {project} template — never prewarmed.
                ("jarvis.claude_session", ("CANCELLED_LINE", "NO_PROJECT_LINE",
                                           "OUTSIDE_LINE", "UNSAFE_DIR_LINE",
                                           "NO_SESSION_LINE")),
                ("jarvis.approvals", ("TIMEOUT_LINE", "ALLOWED_LINE",
                                      "DECLINED_LINE")),
                ("jarvis.commander", ("TERMINAL_OPEN_LINE", "TERMINAL_FAIL_LINE")),
                ("jarvis.tools.briefing", ("BRIEFING_OFF_LINE",)),
                ("jarvis.tools.timekeeper", ("NOTHING_RINGING_LINE",
                                             "NO_TIMEKEEPER_LINE")),
                ("jarvis.brain", ("MODEL_DOWN_LINE", "MODEL_SLOW_LINE",
                                  "MODEL_EMPTY_LINE", "TOOL_ONLY_LINE",
                                  "PARTIAL_RESULT_LINE", "INTERNAL_ERROR_LINE",
                                  "NO_CLOCK_LINE", "UNSURE_CLOCK_LINE"))):
            mod = sys.modules.get(modname)
            for name in names:
                line = getattr(mod, name, None) if mod else None
                if isinstance(line, str) and line and "{" not in line:
                    phrases.append(line)
        # Whole lists of fixed lines (the module is already imported when its
        # tools registered; a missing module simply contributes nothing).
        for modname, name in (("jarvis.tools.spotify", "PERSONA_LINES"),):
            lines = getattr(sys.modules.get(modname), name, None)
            if isinstance(lines, (list, tuple)):
                phrases += [ln for ln in lines if isinstance(ln, str) and ln
                            and "{" not in ln]
        seen, out = set(), []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # ------------------------------------------------------------- services
    def _build_services(self):
        dc, app = self.desktop, self

        def handle_action(action):
            if action == "screenshot":
                threading.Thread(target=dc.screenshot, daemon=True).start()
                bus.publish(Status(text="Taking screenshot", kind="busy"))
            elif action == "clear_all":
                bus.publish(Status(text="Cleared", kind="ok"))
            else:
                bus.publish(Status(text=action.replace("_", " "), kind="ok"))

        def move_window_to_monitor(which):
            bus.publish(Status(text="Single monitor — nothing to move", kind="info"))
            return False

        desktop_ns = SimpleNamespace(
            parse_action=desktop_mod.parse_desktop_action,
            execute_actions=dc.execute,
            get_window_list=desktop_mod.list_windows,
            type_text=dc.type_text,
            screenshot=dc.screenshot,
            target_window=dc.target_by_query,
            reset_target=dc.reset_target,
            handle_action=handle_action,
            move_window_to_monitor=move_window_to_monitor,
        )

        # Context adapter: V3 engine first, retained V1 agent for the tool
        # calls the engine never absorbed.
        a, ctx = self.agent, self.context
        context_ns = SimpleNamespace(
            get_last_window=ctx.get_last_window,
            analyze_screen=ctx.capture_screen,
            click_on_text=a.click_on_text,
            list_heavy_processes=a.list_heavy_processes,
            git_summary=a.git_summary,
            check_connectivity=a.check_connectivity,
            find_file=a.find_file,
            recent_files=a.recent_files,
            get_clipboard_history=a.get_clipboard_history,
            paste_from_history=a.paste_from_history,
            answer_question=a.answer_question,
            run_shell=a.run_shell,
            interpret_intent=a.interpret_intent,
        )

        # The legacy "set_reminder(seconds, task)" hook now lands in the
        # timekeeper (workflows.Reminders is no longer constructed).
        def set_reminder(seconds, task):
            tk = app.timekeeper
            if tk is None:
                bus.publish(Status(text="Timekeeper unavailable", kind="warn"))
                return None
            return tk.add_reminder(time.time() + float(seconds), task)

        workflows_ns = SimpleNamespace(
            get=self.workflows.get,
            run=self.workflows.run,
            set_reminder=set_reminder,
            set_trigger=a.set_trigger,
        )

        b = self.brain

        def chat(text, force_tool=None):
            return b.chat(text, callback=app._on_brain_tags, force_tool=force_tool)

        brain_ns = SimpleNamespace(
            think=lambda text: b.think(text, callback=app._on_brain_tags),
            chat=chat,
            # Delegate lazily rather than capturing the bound methods: the
            # namespace is built once at init, so a snapshot here would make
            # `app.brain.<fn> = ...` (tests, and any later brain swap) a no-op
            # that silently reached the real local model instead.
            classify_route=lambda *a, **kw: app.brain.classify_route(*a, **kw),
            summarize=lambda *a, **kw: app.brain.summarize(*a, **kw),
            local_line=lambda *a, **kw: app.brain.local_line(*a, **kw),
            execute_autonomous=lambda task: b.execute_autonomous(
                task, callback=app._on_brain_tags),
        )

        return SimpleNamespace(
            desktop=desktop_ns, context=context_ns, memory=self.memory,
            workflows=workflows_ns, brain=brain_ns, tts=self.tts,
            reader=self.reader, history=self.history,
            # personal assistant (spec 2.2)
            assistant=self.assistant, tools=self.tools, router=self.router,
            timekeeper=self.timekeeper, notes=self.notes, claude=self.claude,
            approvals=self.approvals, alerts=self.alerts,
            # not in the spec 2.2 table; the UI reads discord.status_text()
            # and the commander may post from a handler.
            discord=self.discord,
            # filled / read by the tool modules: calendar.make_tools parks its
            # CalendarSource on `calendar`; briefing.make_tools reads its
            # news cache path.
            calendar=None,
            news_cache_path=PATHS.CACHE_DIR / "news.json",
        )

    # ------------------------------------------------------- brain executor
    def _on_brain_tags(self, tags):
        # Whatever else these tags mean, their arrival ends the turn.
        self._turn_finished()
        """Port of the monolith's _on_brain_response: act on [TAG] tuples.
        A ("BRIEFING", json) tag turns that turn's SPEAK into ONE
        BriefingReady card (no separate JarvisReply) — still spoken."""
        briefing = None
        for tag, content in tags:
            if tag == "BRIEFING":
                try:
                    briefing = json.loads(content) if isinstance(content, str) else content
                    if not isinstance(briefing, dict):
                        briefing = {"text": str(briefing)}
                except (TypeError, ValueError):
                    log.warning("BRIEFING tag carried non-JSON payload")
                    briefing = {}
        for tag, content in tags:
            try:
                if tag == "SPEAK":
                    if briefing is not None:
                        bus.publish(BriefingReady(sections=briefing, spoken=content))
                        briefing = None
                    else:
                        bus.publish(JarvisReply(text=content, speak=True))
                    self._say(content)
                    self.context.add_exchange("", content)
                elif tag == "BRIEFING":
                    pass                               # consumed by the SPEAK
                elif tag == "RUN":
                    def _run(cmd=content):
                        output = self.agent.run_shell(cmd)
                        if output:
                            bus.publish(JarvisReply(text=output[:400], speak=False))
                    threading.Thread(target=_run, daemon=True).start()
                elif tag == "TYPE":
                    self.desktop.type_text(content)
                elif tag == "WINDOW":
                    self.desktop.execute([("window", content)])
                elif tag == "CLICK":
                    self.agent.click_on_text(content)
                elif tag == "DONE" and content and content.strip():
                    # Protocol: "[DONE] text — task complete, speak this"
                    bus.publish(JarvisReply(text=content, speak=True))
                    self._say(content)
                    self.context.add_exchange("", content)
                # SILENT (and bare DONE): nothing to do
            except Exception:
                log.exception("brain tag %s failed", tag)
        if briefing is not None:                       # a card with no SPEAK
            bus.publish(BriefingReady(sections=briefing, spoken=""))

    # ------------------------------------------------------ claude events
    def _alert(self, kind, title, text, request_id=None):
        if self.alerts is None:
            return
        try:
            self.alerts.alert(kind, title, text, request_id=request_id)
        except Exception:
            log.exception("alert %s failed", kind)

    def _on_claude_progress(self, ev):
        if ev.milestone and ev.line:
            self._last_milestone[ev.task_id] = ev.line
            self._say(ev.line)
            self._alert("milestone", f"Claude · {ev.project}", ev.line)

    def _on_claude_state(self, ev):
        if ev.state == "done":
            threading.Thread(target=self._finish_claude_task, args=(ev,),
                             daemon=True, name="claude-summary").start()
        elif ev.state == "failed":
            line = self._spoken_cap(ev.text) or \
                f"Claude's stopped with an error on {ev.project or 'the task'}, sir."
            # The manager may already have spoken this very line as a
            # milestone; do not say it twice.
            if self._last_milestone.pop(ev.task_id, None) != line:
                bus.publish(JarvisReply(text=line, speak=True))
                self._say(line)
            self._alert("blocked", f"Claude · {ev.project}", line)
        elif ev.state == "cancelled":
            self._last_milestone.pop(ev.task_id, None)
            bus.publish(Status(text="Claude task cancelled", kind="info"))

    def _result_text(self, ev) -> str:
        """The task's full final text (manager's Task.result_text), else the
        event's text."""
        mgr = self.claude
        if mgr is not None:
            try:
                task = mgr.task(ev.task_id)
                if task is not None and task.result_text:
                    return str(task.result_text)
            except Exception:
                log.debug("claude.task(%s) failed", ev.task_id, exc_info=True)
        return ev.text or ""

    @staticmethod
    def _spoken_cap(text, n=2) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        return brain_mod.trim_spoken(
            brain_mod.limit_sentences(brain_mod.strip_markdown(text), n))

    def _finish_claude_task(self, ev):
        self._last_milestone.pop(ev.task_id, None)
        text = self._result_text(ev)
        spoken = ""
        try:
            if len(brain_mod.split_sentences(text)) > 2:
                spoken = self.brain.summarize(text, 2) or ""
        except Exception:
            log.exception("claude result summary failed")
        spoken = self._spoken_cap(spoken or text) or \
            f"Claude's finished with {ev.project or 'the task'}, sir."
        bus.publish(JarvisReply(text=spoken, speak=True))
        self._say(spoken)
        self.context.add_exchange("", spoken)
        self._alert("done", f"Claude · {ev.project}", spoken)

    # --------------------------------------------------------- approvals
    def _on_approval(self, req):
        """ApprovalBroker.on_request (socket thread). The broker publishes
        ApprovalRequested itself and the session manager marks the task
        waiting from that event; nothing else to do here."""
        log.info("approval pending: %s %s", req.tool_name, req.detail[:80])

    def _on_approval_done(self, req, allowed, source=""):
        log.info("approval %s %s (%s)", req.request_id,
                 "allowed" if allowed else "declined", source)

    def _on_approval_requested(self, ev):
        self._say(ev.question)
        self._alert("question", f"Claude · {ev.project or 'permission'}",
                    ev.question, request_id=ev.request_id)

    def _on_approval_resolved(self, ev):
        # Typed / voice answers are acknowledged by the commander; the other
        # sources need the spoken line from here.
        if ev.source == "timeout":
            bus.publish(JarvisReply(text=APPROVAL_TIMEOUT_LINE, speak=True))
            self._say(APPROVAL_TIMEOUT_LINE)
        elif ev.source in ("discord", "ui"):
            line = ALLOWED_LINE if ev.allowed else DECLINED_LINE
            bus.publish(JarvisReply(text=line, speak=True))
            self._say(line)

    def approval_answer(self, request_id, allowed) -> bool:
        """UI ALLOW / DENY buttons."""
        if self.approvals is None:
            return False
        try:
            return bool(self.approvals.answer(bool(allowed), request_id=request_id,
                                              source="ui"))
        except Exception:
            log.exception("approval answer failed")
            return False

    # ------------------------------------------------------ alarms / alerts
    def _on_alarm_fired(self, ev):
        title = "Alarm" if ev.kind == "alarm" else ev.kind.title()
        text = " — ".join(p for p in (ev.due_text, ev.label) if p)
        self._alert("alarm", title, text or title)

    def _on_reminder_fired(self, ev):
        self._alert("reminder", "Reminder", ev.text)

    def alarm_action(self, alarm_id, action, minutes=None) -> bool:
        """UI DISMISS / SNOOZE buttons on the alarm modal."""
        tk = self.timekeeper
        if tk is None:
            return False
        try:
            if action == "snooze":
                return bool(tk.snooze(int(minutes or self.assistant.get(
                    "alarms.snooze_min", 10) or 10)))
            return bool(tk.stop_ringing("dismiss"))
        except Exception:
            log.exception("alarm action %s failed", action)
            return False

    # ------------------------------------------------------------ discord
    def _on_discord(self, text, author_id=""):
        """DiscordChannel.on_message (gateway thread): a yes/no answers a
        pending approval; anything else is an ordinary command. Returns
        the worker thread (None when nothing was dispatched)."""
        text = (text or "").strip()
        if not text:
            return None
        self._discord_active_until = time.time() + DISCORD_ACTIVE_S
        mark = getattr(self.discord, "mark_active", None)
        if callable(mark):
            try:
                mark()
            except Exception:
                log.debug("discord.mark_active failed", exc_info=True)
        if self.approvals is not None:
            try:
                pending = self.approvals.pending()
            except Exception:
                log.exception("approvals.pending failed")
                pending = []
            if pending:
                answer = yes_no(text)
                if answer is not None:
                    self.approvals.answer(answer, source="discord")
                    return None
        bus.publish(UserUtterance(text=text, source="discord"))
        t = threading.Thread(target=self.dispatch_text, args=(text, "discord"),
                             daemon=True, name="discord-dispatch")
        t.start()
        return t

    def _discord_is_active(self) -> bool:
        if self.discord is None:
            return False
        fn = getattr(self.discord, "is_active", None)
        if callable(fn):
            try:
                if fn():
                    return True
            except Exception:
                log.debug("discord.is_active failed", exc_info=True)
        return time.time() < self._discord_active_until

    def _on_reply_for_discord(self, ev):
        """While a Discord exchange is active every JarvisReply is posted
        back (deduplicated against the alert hub's own posts)."""
        if not ev.text or not self._discord_is_active():
            return
        last_text, last_at = self._discord_last_post
        if ev.text == last_text and time.time() - last_at < 30:
            return
        self._discord_last_post = (ev.text, time.time())
        post = getattr(self.discord, "post", None)
        if post is None:
            return

        def _post(text=ev.text):
            try:
                post(text)
            except Exception:
                log.exception("discord post failed")
        threading.Thread(target=_post, daemon=True, name="discord-post").start()

    # ------------------------------------------------------------ options
    def get_option(self, key, default=None):
        try:
            return self.assistant.get(key, default)
        except Exception:
            log.exception("get_option %s failed", key)
            return default

    def set_option(self, key, value) -> bool:
        """Settings drawer writes; 'autostart.enabled' also installs or
        removes the GNOME autostart entry."""
        try:
            ok = self.assistant.set(key, value)
        except Exception:
            log.exception("set_option %s failed", key)
            return False
        if key == "autostart.enabled":
            try:
                from jarvis import autostart
                if value:
                    autostart.install(path=PATHS.AUTOSTART_DESKTOP)
                    autostart.disable_gnome_suspend()
                    bus.publish(Status(text="Starts at login", kind="ok"))
                else:
                    autostart.uninstall(path=PATHS.AUTOSTART_DESKTOP)
                    bus.publish(Status(text="Login start removed", kind="ok"))
            except Exception:
                log.exception("autostart change failed")
                bus.publish(Status(text="Autostart change failed", kind="error"))
                return False
        return bool(ok)

    def open_terminal(self, slug=None) -> bool:
        mgr = self.claude
        if mgr is None:
            bus.publish(Status(text="Claude manager unavailable", kind="warn"))
            return False
        try:
            return bool(mgr.open_terminal(slug))
        except Exception:
            log.exception("open_terminal failed")
            bus.publish(Status(text="Could not open the terminal", kind="error"))
            return False

    # ------------------------------------------------------------ voice path
    def _mic_index(self):
        devices = self.recorder.mic_devices
        return devices.get(CONFIG.mic)

    def _on_hotword(self, score):
        # Called from the hotword listener thread.
        if self.recorder.recording:
            return
        if self._audio_busy.is_set() or self._turn_busy.is_set():
            # Transcription of the previous utterance is still running (~20 s
            # for a long clip). Starting a second capture here raced two
            # transcripts into the commander. Say so rather than ignoring it
            # silently -- an unanswered wake word reads as a broken mic.
            log.info("hotword ignored: still transcribing the previous clip")
            bus.publish(Status(text="One moment — still on the last one",
                               kind="warn"))
            return
        if CONFIG.sound:
            threading.Thread(target=play_beep, args=("start",), daemon=True).start()
        threading.Timer(0.2, self.recorder.start).start()

    def _on_recording_stopped(self, ev):
        if ev.reason == "abort":
            return
        audio = self.recorder.last_audio
        if audio is None:
            return
        self._audio_busy.set()
        threading.Thread(target=self._process_audio, args=(audio,),
                         daemon=True).start()

    def _process_audio(self, audio):
        try:
            if CONFIG.speaker_verify and self.speaker.enrolled:
                filtered, stats = self.speaker.filter_segments(audio)
                if filtered is None:
                    bus.publish(Transcribed(
                        text="", accepted=False, reject_reason="speaker",
                        speaker_score=float(stats.get("best_score", 0.0))
                        if isinstance(stats, dict) else 0.0))
                    return
                audio = filtered
            result = self.transcriber.transcribe(audio)
            bus.publish(Transcribed(
                text=result.text, confidence=result.confidence,
                accepted=result.accepted,
                reject_reason="" if result.accepted else "confidence"))
            text = result.text.strip()
            if result.accepted and text:
                bus.publish(UserUtterance(text=text, source="voice"))
                self._dispatch(text, "voice")
        except Exception:
            log.exception("audio processing failed")
            bus.publish(Status(text="Transcription failed", kind="error"))
        finally:
            # Must run on every path: a leaked flag makes every future wake
            # word a no-op, which looks exactly like a dead microphone.
            self._audio_busy.clear()

    # -------------------------------------------------------------- routing
    def _emit_result(self, result):
        """Publish a CommandResult's reply/status. Shared by _dispatch and the
        uncertain-prompt answer, so a YES there runs and SPEAKS exactly like a
        command that had been understood the first time."""
        if result.reply:
            bus.publish(JarvisReply(text=result.reply, speak=result.speak))
            if result.speak:
                self._say(result.reply)
        if result.status:
            bus.publish(Status(text=result.status, kind="info"))
        return result

    _thinking_delay_s = THINKING_DELAY_S
    _turn_timeout_s = TURN_TIMEOUT_S

    def _dispatch(self, text, source):
        # Voice only: a typed answer is visible as it arrives, so being told to
        # wait is just noise.
        if source == "voice":
            self._turn_start()
        try:
            result = self._emit_result(self.commander.handle(text, source))
        except Exception:
            self._turn_finished()
            raise
        # done=False means the answer is still coming on a worker thread (the
        # commander routes local chat that way). Anything else is over now.
        if source != "voice" or getattr(result, "done", True) is not False:
            self._turn_finished()
        return result

    def _turn_start(self):
        """Open a turn: arm the slow-answer filler and a watchdog."""
        self._turn_cancel_timers()
        self._turn_busy.set()
        if CONFIG.talkback:
            self._turn_timer = threading.Timer(self._thinking_delay_s,
                                               self._say_thinking)
            self._turn_timer.daemon = True
            self._turn_timer.start()
        # Without this a reply that never arrives would hold _turn_busy for
        # good, and every later wake word would be a silent no-op --
        # indistinguishable from a dead microphone.
        self._turn_watchdog = threading.Timer(self._turn_timeout_s,
                                              self._turn_timed_out)
        self._turn_watchdog.daemon = True
        self._turn_watchdog.start()

    def _turn_timed_out(self):
        log.warning("turn watchdog fired after %.0fs; releasing the wake word",
                    self._turn_timeout_s)
        self._turn_finished()

    def _turn_cancel_timers(self):
        for name in ("_turn_timer", "_turn_watchdog"):
            t = getattr(self, name, None)
            if t is not None:
                try:
                    t.cancel()
                except Exception:
                    log.debug("timer cancel failed", exc_info=True)
                setattr(self, name, None)

    def _turn_finished(self):
        """The answer landed (or gave up). Also called from the brain callback."""
        self._turn_cancel_timers()
        self._turn_busy.clear()

    def _say_thinking(self):
        """Acknowledge a slow lookup. Rotates so it does not become a tic."""
        try:
            line = THINKING_LINES[self._thinking_i % len(THINKING_LINES)]
            self._thinking_i += 1
            self._say(line)
        except Exception:
            log.exception("thinking line failed")

    # ---------------------------------------------------- uncertain intent
    UNCERTAIN_LISTEN_S = 5.0

    def _on_uncertain(self, text: str):
        """Commander hook: ask a question that can actually be answered."""
        rid = uuid.uuid4().hex[:12]
        with self._uncertain_lock:
            # One open question at a time -- a newer utterance supersedes the
            # old one, or stale cards pile up with no way to tell which is live.
            stale = list(self._pending_uncertain)
            self._pending_uncertain.clear()
            self._pending_uncertain[rid] = text
        for old in stale:
            bus.publish(UncertainResolved(request_id=old, yes=False,
                                          source="superseded"))
        bus.publish(UncertainUtterance(
            request_id=rid, text=text,
            question=f'Was that for me? — "{text[:60]}"'))
        threading.Thread(target=self._ask_uncertain, args=(rid,), daemon=True,
                         name="uncertain-ask").start()

    def _ask_uncertain(self, rid: str):
        """Say it out loud, then listen briefly for a spoken yes/no.

        Blocks on the TTS before recording: talk-back holds the mic arbiter,
        but the arbiter is a depth counter rather than a mutex, so without the
        wait we would happily record Jarvis asking the question.
        """
        try:
            if CONFIG.talkback:
                self.tts.speak("Was that for me?", block=True)
            if not MACHINE.has_mic or self.recorder.recording:
                return
            with self._uncertain_lock:
                if rid not in self._pending_uncertain:
                    return                      # already answered by a click
            audio = self.recorder.record_fixed(self.UNCERTAIN_LISTEN_S)
            if audio is None or len(audio) == 0:
                return
            if CONFIG.speaker_verify and self.speaker.enrolled:
                filtered, _ = self.speaker.filter_segments(audio)
                if filtered is None:
                    log.info("uncertain reply ignored: not the enrolled speaker")
                    return
                audio = filtered
            result = self.transcriber.transcribe(audio)
            heard = result.text if result.accepted else ""
            answer = parse_yes_no(heard)
            log.info("uncertain follow-up heard %r -> %s", heard, answer)
            if answer is None:
                # Never route an unrecognised reply: it could classify as
                # uncertain again and the two prompts would ping-pong. The
                # card stays up for a click instead.
                return
            self.uncertain_answer(rid, answer, source="voice")
        except Exception:
            log.exception("uncertain follow-up failed")

    def uncertain_answer(self, request_id: str, yes: bool, source: str = "ui"):
        """Answer the open prompt. First answer wins -- the card and the
        spoken window race each other, and resolve_uncertain would otherwise
        route the same utterance twice."""
        with self._uncertain_lock:
            text = self._pending_uncertain.pop(request_id, None)
        if text is None:
            return None
        bus.publish(UncertainResolved(request_id=request_id, yes=yes,
                                      source=source))
        return self._emit_result(self.commander.resolve_uncertain(text, yes))

    def dispatch_text(self, text, source="typed"):
        """MainWindow calls this on a worker thread for typed input; the
        Discord channel with source='discord'."""
        text = (text or "").strip()
        if not text:
            return None
        # Barge-in: a typed command while Jarvis is talking cuts him off
        # (the films' JARVIS never talks over Tony), then gets answered.
        self.interrupt_speech()
        if source == "typed":
            self.history.add(text)
        return self._dispatch(text, source)

    # ------------------------------------------------------------ lifecycle
    def start_models(self):
        """Load whisper + XTTS on a worker thread. Called once the avatar's
        full frame cycle is live (main): measured, a 0.5 GB CUDA model load
        that overlaps the bake's frame installs froze the Tk thread for
        2.7 s (322 late slots); the same loads after the bake cost one
        16 ms slot."""
        threading.Thread(target=self._load_models, daemon=True,
                         name="model-loader").start()

    def start_background(self):
        if CONFIG.hotword and MACHINE.has_mic:
            self.hotword.start()
        try:
            self.agent.start_monitoring(speak_func=self._say)
        except Exception:
            log.exception("agent monitoring failed to start")
        self.start_assistant()
        if "--auto-record" in sys.argv and MACHINE.has_mic:
            threading.Timer(1.0, self.recorder.start).start()

    def start_assistant(self, residency=True):
        """Timekeeper (catch-up first), approvals socket, Discord gateway,
        calendar refresh, local-model residency; autostart entry when
        enabled. Idempotent."""
        if self._assistant_started:
            return
        self._assistant_started = True
        for name, obj in (("timekeeper", self.timekeeper),
                          ("approvals", self.approvals),
                          ("discord", self.discord)):
            if obj is None:
                continue
            try:
                obj.start()                # Timekeeper.start() catches up first
            except Exception:
                log.exception("assistant: %s failed to start", name)
                bus.publish(Status(text=f"{name} failed to start", kind="warn"))
        cal = getattr(self.services, "calendar", None)
        if cal is not None:
            try:
                cal.start()
            except Exception:
                log.exception("calendar refresh start failed")
        if residency:
            try:
                # boot warm-up on its own daemon thread, then every 5 min
                brain_mod.start_residency()
            except Exception:
                log.exception("ollama residency thread failed to start")
        if self.assistant.get("autostart.enabled", False):
            try:
                from jarvis import autostart
                autostart.install(path=PATHS.AUTOSTART_DESKTOP)
            except Exception:
                log.exception("autostart install failed")

    def _load_models(self):
        # start_preload() may still be importing torch/whisper; importing the
        # same modules from two threads is what the preload exists to avoid.
        _PRELOAD_DONE.wait(120)
        self.brain.warmup()
        bus.publish(Status(text="Loading speech model…", kind="busy"))
        try:
            backend = self.transcriber.load()
            bus.publish(ModelInfo(text=f"{CONFIG.model} · {backend}"))
            bus.publish(Status(text="Ready", kind="ok"))
        except Exception:
            log.exception("whisper load failed")
            bus.publish(Status(text="Speech model failed to load", kind="error"))
        try:
            self.tts.load()
        except Exception:
            log.exception("tts load failed")
        # Warm the speaker model here so the first wake word does not pay the
        # ~1.4 s CUDA cold start, and so the audio path never has to load it
        # inline. Verification loads lazily too, but that is the fallback.
        try:
            if self.speaker.enrolled:
                self.speaker.load_model()
        except Exception:
            log.exception("speaker model load failed")
        # Honest failure for the speakers: with only a dummy/null sink the
        # playback chain "succeeds" into silence (seen on this machine with
        # no HDMI audio device attached).
        try:
            sink = voice_check.output_sink_state()
            if sink.get("probed") and sink.get("dummy"):
                log.warning("audio output is a dummy sink: %s", sink)
                bus.publish(Status(
                    text="No audio output device — speech will be silent",
                    kind="warn"))
        except Exception:
            log.exception("output sink check failed")
        # Render the canned lines into the speech cache while idle.
        try:
            self.tts.prewarm(self._canned_phrases())
        except Exception:
            log.exception("tts prewarm failed")
        # Both models are resident: freeze their objects too (O(1) — no
        # traversal, so no stall on the frame loop) so no later gen-2 pass
        # walks the model graphs.
        import gc
        gc.freeze()
        log.info("models loaded; heap frozen (%d objects)",
                 gc.get_freeze_count())

    def toggle_hotword(self, enabled):
        if enabled and MACHINE.has_mic:
            self.hotword.start()
        else:
            self.hotword.stop()

    def calibrate_noise(self):
        return self.recorder.calibrate_noise()

    def enroll_speaker(self):
        audio = self.recorder.record_fixed(15)
        if audio is not None and len(audio) > 0:
            ok, n = self.speaker.enroll_from_audio(audio)
            bus.publish(Status(
                text=f"Voice enrolled ({n} samples)" if ok else "Enrollment failed",
                kind="ok" if ok else "error"))

    def train_wakeword(self):
        from jarvis.hotword import train_verifier
        samples = []
        for _ in range(3):
            audio = self.recorder.record_fixed(3)
            if audio is not None and len(audio) > 0:
                samples.append((audio * 32767).astype("int16"))
        if not samples:
            bus.publish(Status(text="No audio captured", kind="error"))
            return
        try:
            train_verifier(samples)
            bus.publish(Status(text="Wake word trained", kind="ok"))
        except Exception:
            log.exception("wake word training failed")
            bus.publish(Status(text="Wake word training failed", kind="error"))

    def stop_assistant(self):
        """Stop every assistant thread (quit, and the tests' teardown)."""
        self._quitting = True
        cal = getattr(self.services, "calendar", None)
        for name, obj in (("discord", self.discord), ("approvals", self.approvals),
                          ("timekeeper", self.timekeeper), ("calendar", cal),
                          ("claude", self.claude)):
            if obj is None:
                continue
            fn = getattr(obj, "stop", None) or getattr(obj, "close", None)
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                log.exception("assistant: %s failed to stop", name)
        for obj in (self.notes,):
            fn = getattr(obj, "close", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    log.exception("notes close failed")

    def quit(self):
        try:
            convo = self.context._conversation[-8:]
            if convo:
                lines = [f"You: {c.get('user', '')} / Jarvis: {c.get('jarvis', '')[:80]}"
                         for c in convo]
                self.memory.save_session(" | ".join(lines)[:1000])
        except Exception:
            log.exception("session summary save failed")
        try:
            self.stop_assistant()
        except Exception:
            log.exception("assistant shutdown failed")
        try:
            self.hotword.stop()
            self.tts.stop()
            speak_queue.stop_watcher()
            self.brain.cancel()
        except Exception:
            log.exception("shutdown cleanup failed")
        try:
            (PATHS.LOG_DIR / "jarvis.pid").unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------ UI hooks
    def ui_service_kwargs(self) -> dict:
        """Everything the UI's Services dataclass may take (spec 9.10);
        main() keeps the fields the installed UI declares."""
        return dict(
            start_recording=self.recorder.start,
            stop_recording=lambda: threading.Thread(
                target=self.recorder.stop, daemon=True).start(),
            dispatch_text=self.dispatch_text,
            toggle_hotword=self.toggle_hotword,
            quit=self.quit,
            calibrate_noise=self.calibrate_noise,
            enroll_speaker=self.enroll_speaker,
            train_wakeword=self.train_wakeword,
            open_terminal=self.open_terminal,
            alarm_action=self.alarm_action,
            approval_answer=self.approval_answer,
            uncertain_answer=self.uncertain_answer,
            get_option=self.get_option,
            set_option=self.set_option,
        )


def build_ui_services(services_cls, kwargs: dict):
    """Instantiate the UI's Services with only the fields it declares (the
    UI item may land after the wiring); dropped names are logged once."""
    import dataclasses
    try:
        names = {f.name for f in dataclasses.fields(services_cls)}
    except TypeError:
        names = set(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in names}
    dropped = sorted(set(kwargs) - names)
    if dropped:
        log.warning("UI Services lacks %s; those hooks stay unwired", dropped)
    return services_cls(**kept)


WM_CLASS = "jarvis"          # tk.Tk(className="jarvis") in ui.main_window.create
FOCUS_WAIT_S = 20.0          # a second click may land while the first still boots
FOCUS_POLL_S = 0.5


def _run(argv, timeout=5):
    """Subprocess seam (spec 3.3): every xdotool / notify-send call in this
    module goes through it so tests can record instead of touching X."""
    return subprocess.run(argv, timeout=timeout, capture_output=True, text=True)


def _sleep(seconds):
    time.sleep(seconds)


def _notify(title, text):
    try:
        _run(["notify-send", "-a", "Jarvis", title, text])
        return True
    except Exception:
        log.debug("notify-send failed", exc_info=True)
        return False


def _raise_window(wait_s: float = 0.0) -> bool:
    """Raise the running instance's window, polling up to wait_s for one to
    appear (the click that starts Jarvis is answered ~15 s before the window
    exists; a silent no-op is what made the launcher look broken).

    Matched by WM_CLASS, not by title: the window is `-type splash`, which
    some window managers refuse to activate, so map + activate + raise are
    all sent."""
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        wid = ""
        try:
            out = _run(["xdotool", "search", "--classname", WM_CLASS])
            ids = [ln.strip() for ln in (out.stdout or "").splitlines()
                   if ln.strip().isdigit()]
            wid = ids[-1] if ids else ""
        except Exception:
            log.debug("xdotool search failed", exc_info=True)
        if wid:
            for verb in ("windowmap", "windowactivate", "windowraise"):
                try:
                    _run(["xdotool", verb, wid])
                except Exception:
                    log.debug("xdotool %s failed", verb, exc_info=True)
            return True
        if time.monotonic() >= deadline:
            return False
        _sleep(FOCUS_POLL_S)


def _focus_running_instance() -> bool:
    """If another Jarvis owns the pid file, raise its window instead of
    starting a second instance (the desktop icon makes this easy to do).
    Returns True when this process should exit."""
    pid_file = PATHS.LOG_DIR / "jarvis.pid"
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)          # alive?
    except (OSError, ValueError):
        try:
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(os.getpid()))
        except OSError:
            log.warning("could not write pid file")
        return False
    log.info("Jarvis already running (pid %s); focusing it", pid)
    import signal
    try:
        os.kill(pid, signal.SIGUSR1)     # asks it to deiconify (tray case)
    except OSError:
        pass
    if _raise_window(FOCUS_WAIT_S):
        return True
    # The window never appeared: say something rather than exit silently.
    log.warning("no Jarvis window after %.0fs; notifying instead", FOCUS_WAIT_S)
    _notify("Jarvis", "Starting up, sir…")
    return True


_PRELOAD_DONE = threading.Event()
_PRELOAD_DONE.set()        # nothing to wait for unless a preload is started


def start_preload() -> threading.Thread:
    """Run _preload_heavy_imports on a worker thread AFTER the window is up.
    Doing it before `create()` cost 15 s of blank screen from the desktop
    icon (the user's "the launcher does nothing"); the frame loop is
    protected by deferring the model loads to cycle-live, not by this."""
    _PRELOAD_DONE.clear()

    def _work():
        try:
            _preload_heavy_imports()
        finally:
            _PRELOAD_DONE.set()

    t = threading.Thread(target=_work, daemon=True, name="preload")
    t.start()
    return t


def _preload_heavy_imports():
    """Import torch/whisper (and the XTTS stack when configured) and create
    the CUDA context BEFORE the window exists. Measured: doing this lazily
    on the model-loader thread held the GIL for ~3.3 s and skipped ~100
    avatar slots over a running sphere (the user's 'random lag spikes' at
    boot); done here it just delays the window by the same amount. The
    heap is then collected once and frozen so torch's ~500k long-lived
    objects never sit in a gen-2 traversal while the frame loop runs."""
    import gc
    t0 = time.monotonic()
    try:
        import torch                                  # noqa: F401
        try:
            if torch.cuda.is_available():
                torch.cuda.init()                     # driver + context
                torch.zeros(1, device="cuda")         # allocator warm
                torch.cuda.synchronize()
        except Exception:
            log.debug("cuda pre-init failed", exc_info=True)
        try:
            import whisper                            # noqa: F401
        except Exception:
            log.debug("whisper pre-import failed", exc_info=True)
        if CONFIG.tts_engine == "xtts":
            try:
                os.environ.setdefault("COQUI_TOS_AGREED", "1")
                # same transformers-5 shim TTS.load() installs: coqui-tts
                # 0.27 imports a helper transformers removed
                import transformers.pytorch_utils as _tpu
                if not hasattr(_tpu, "isin_mps_friendly"):
                    _tpu.isin_mps_friendly = (
                        lambda elements, test_elements: torch.isin(
                            elements, test_elements))
                import TTS.api                        # noqa: F401
            except Exception:
                log.debug("TTS pre-import failed", exc_info=True)
    except Exception:
        log.debug("torch pre-import failed", exc_info=True)
    gc.collect()
    gc.freeze()
    log.info("preloaded torch/whisper + CUDA in %.1fs; heap frozen (%d "
             "objects)", time.monotonic() - t0, gc.get_freeze_count())


def install_autostart_cli() -> int:
    """`python -m jarvis.app --install-autostart`: write the GNOME autostart
    entry, disable automatic suspend, remember the choice, exit."""
    from jarvis import autostart
    target = autostart.install(path=PATHS.AUTOSTART_DESKTOP)
    print(f"autostart entry: {target}")
    ok = autostart.disable_gnome_suspend()
    print(f"gnome suspend on AC: {'disabled' if ok else 'could not set (gsettings?)'}")
    try:
        AssistantConfig.load().set("autostart.enabled", True)
    except Exception:
        log.exception("could not record autostart.enabled")
    return 0


def spotify_login_cli() -> int:
    """`python -m jarvis.app --spotify-login`: the one-time Spotify OAuth
    link (loopback server on 127.0.0.1:8888 + the default browser), then
    exit. `python -m jarvis.tools.spotify --login` does the same standalone."""
    from jarvis.tools.spotify import login_cli
    extra = [a for a in sys.argv[1:] if a in ("--no-browser", "--status")]
    return int(login_cli(["--login"] + extra))


def main():
    if "--install-autostart" in sys.argv:
        sys.exit(install_autostart_cli())
    if "--spotify-login" in sys.argv:
        sys.exit(spotify_login_cli())

    from jarvis.ui.main_window import Services, create

    if _focus_running_instance():
        return
    # Frame-loop hygiene: the GIL switch interval is shortened so a worker
    # thread running Python code hands the GIL to the Tk thread within ~2 ms
    # instead of 5; gen-2 collections are made rarer (the frozen heap makes
    # the remaining ones cheap). The heavy imports themselves happen AFTER
    # the window is mapped (start_preload) — see its docstring.
    import gc
    sys.setswitchinterval(0.002)
    gc.set_threshold(700, 10, 50)
    app = JarvisApp()
    import signal

    def _on_show_signal(*_):
        try:
            window.root.after(0, window.root.deiconify)
        except Exception:
            pass

    signal.signal(signal.SIGUSR1, _on_show_signal)
    window = create(build_ui_services(Services, app.ui_service_kwargs()))
    try:
        window.root.update()          # map it now: the icon click gets a window
    except Exception:
        log.debug("first update() failed", exc_info=True)
    start_preload()                   # torch/CUDA/whisper, off the Tk thread
    app.start_background()
    # Models load only after the avatar's full cycle is live (or 40 s):
    # see JarvisApp.start_models.
    window.reactor.when_cycle_live(app.start_models, timeout_s=40.0)
    from jarvis import perf
    if perf.detail_enabled():          # JARVIS_PERF_DETAIL=1 / PROFILE_SECS
        perf.install_gc_logging()
        perf.log_threads("startup")
    window.root.mainloop()


if __name__ == "__main__":
    main()
