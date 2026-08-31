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
from datetime import datetime, timedelta
import os
import re
import subprocess
import sys
import threading
import uuid
import time
from pathlib import Path
from types import SimpleNamespace

from jarvis.config import CONFIG, MACHINE, PATHS
from jarvis.events import (
    AlarmFired,
    ApprovalRequested,
    ApprovalResolved,
    BriefingReady,
    ClaudeProgress,
    ClaudeTaskState,
    JarvisReply,
    ModelInfo,
    PartialText,
    Presence,
    RecordingStarted,
    RecordingStopped,
    ReminderFired,
    Status,
    Transcribed,
    UncertainResolved,
    UncertainUtterance,
    UserUtterance,
    bus,
    SpeakingState)
from jarvis.logs import get_logger

from jarvis import brain as brain_mod
from jarvis import desktop as desktop_mod
from jarvis import speak_queue, standup, voice_check
from jarvis import vocab as vocab_mod
from jarvis.assistant_config import AssistantConfig
from jarvis.turnclock import TurnLedger
from jarvis import dayreview as dayreview_mod
from jarvis.brain import JarvisBrain
from jarvis.commander import COURTESY_REPLIES, Commander, parse_yes_no
from jarvis.context import ContextEngine
from jarvis.history import TypedHistory
from jarvis.hotword import Hotword
from jarvis.jarvis_agent import JarvisAgent
from jarvis.memory import JarvisMemory
from jarvis.reader import CONTINUE_PROMPT, ReadAloud
from jarvis.tools.docs import doc_paths
from jarvis.recorder import (SAMPLE_RATE, MicArbiter, Recorder,
                             play_beep)
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
    "jarvis.tools.canvas",
    "jarvis.tools.docs",
    "jarvis.tools.screen",
    "jarvis.tools.health",
    "jarvis.tools.journal",
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
# 3.5 not 2.0: local replies measured 2.0-2.3 s, so firing at 2.0 guaranteed
# the filler spoke every time -- and the real answer then queued BEHIND it,
# adding ~2.9 s (2026-08-28 12:54). Only genuinely slow lookups should talk.
#
# 4.5 not 3.5 (2026-08-29): the same collision was still audible, just rarer --
# tool-backed lookups land right around 3.5 s, so the filler started and the
# answer immediately queued behind it. The extra second moves the filler clear
# of the common case, leaving it for lookups that are genuinely slow.
THINKING_DELAY_S = 4.5
SAY_AGAIN_LINE = "Say that again, sir?"
# The "did not catch that" cue (JarvisApp._nudge): a wake-word turn that
# captured nothing usable gets this instead of silence.
NUDGE_LINE = "Sir?"
GUEST_LINE = "I only answer to {name}, sir."
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


def _brain_model_name() -> str:
    try:
        from jarvis.brain import OLLAMA_MODEL
        return str(OLLAMA_MODEL).split(":")[0]
    except Exception:
        return "the local model"


class _Speculation:
    """One speculative decode of the clip being captured (JarvisApp._maybe_speculate).

    ``key`` is the endpointer's last_speech_seconds when the snapshot was
    taken: the pass is only valid for the turn if that is still the last
    word the VAD heard when the recorder stops. ``done`` lets _process_audio
    wait for a pass still in flight instead of starting a second decode
    that would only queue behind it on the model lock.
    """
    __slots__ = ("key", "done", "audio", "stats", "rejected", "result",
                 "error", "started", "finished")

    def __init__(self, key: float):
        self.key = key
        self.done = threading.Event()
        self.audio = None
        self.stats: dict = {}
        self.rejected = False          # the speaker gate dropped the clip
        self.result = None             # TranscribeResult, when it got that far
        self.error = None
        self.started = time.monotonic()
        self.finished: float | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and (self.rejected or self.result is not None)


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
        self.memory = JarvisMemory(
            semantic=bool(self.assistant.get("memory.semantic", True)))
        self.context = ContextEngine(memory=self.memory)
        # The standup and the prompt's git line walk the cleared projects
        # (claude.allowed_dirs + projects_root) and the VSS tree, primary
        # first; the engine's own default is the Jarvis repo + ~/vss_env.
        try:
            repos = standup.default_repo_dirs(self.assistant)
            if repos:
                self.context.repo_dirs = repos
        except Exception:
            log.exception("standup repo list failed; keeping the default")
        self.brain = JarvisBrain(self.context, self.memory)
        # brain.configure(assistant.local_model): module-level in brain.py;
        # env JARVIS_OLLAMA_MODEL wins inside it.
        try:
            brain_mod.configure(self.assistant.local_model)
        except Exception:
            log.exception("brain.configure(%s) failed", self.assistant.local_model)
        self.agent = JarvisAgent()          # retained V1 tools (see spec note)

        # ---- ambient: quiet hours / DND and presence ----------------------
        # Both read services lazily (calendar, presence) because services is
        # built further down; both are started in start_assistant.
        self.presence = self._construct("presence", self._make_presence)
        self.quiet = self._construct("quiet", self._make_quiet)

        # ---- speech -------------------------------------------------------
        # The arbiter is built here, ahead of the mic consumers below, because
        # TTS needs it too: hotword.py's contract lists "TTS talk-back" as a
        # consumer that pauses the wake-word stream, and until it did, the
        # always-on hotword could hear Jarvis's own voice.
        self.arbiter = MicArbiter()
        self.tts = TTS(gpu=0, engine=CONFIG.tts_engine, arbiter=self.arbiter)
        # The speak-queue file is the hooks narrator (Claude Code hooks, VSS,
        # shell): nobody asked for those lines, so quiet hours hold them.
        speak_queue.set_sink(lambda text: self._say(text, proactive=True, kind="message"))
        speak_queue.start_watcher()
        # "read the clipboard" / "read file x" / "explain the handout": the
        # docs folders join the search path so a PDF dropped in
        # ~/Documents/Jarvis Docs resolves by name (spec 11).
        self.reader = ReadAloud(self.tts, search_dirs=[
            Path.cwd(), Path.home(), Path.home() / "Jarvis",
            *doc_paths(self.assistant)])
        self.history = TypedHistory()           # typed-command history

        # ---- audio in -----------------------------------------------------
        self.speaker = SpeakerVerifier(gpu=0, threshold=CONFIG.speaker_threshold)
        # Load the stored voiceprint now -- one npz read. Without it
        # `speaker.enrolled` stays False and BOTH gates (the wake-word gate
        # below and the transcript filter in _process_audio) silently do
        # nothing no matter how the settings are configured.
        self.speaker.load()
        self.recorder = Recorder(self.arbiter, speaker_verifier=self.speaker)
        # prompt_provider: every transcription's initial_prompt is built
        # per turn from his own vocabulary — taught names, corrections,
        # calendar titles, cached Canvas courses (jarvis/vocab.py) —
        # instead of the static warehouse DEFAULT_VOCAB the V3 port
        # carried over from VSS.
        self.transcriber = Transcriber(prompt_provider=vocab_mod.build_prompt)
        # speaker= gates the wake word itself: a non-enrolled voice never
        # reaches _on_hotword, so the TV no longer opens a recording at all.
        self.hotword = Hotword(self.arbiter, self._mic_index, self._on_hotword,
                               speaker=self.speaker, on_guest=self._on_guest)

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
        self.cmdsock = self._construct("cmdsock", self._make_cmdsock)
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
        # After the tools: the session reaches Spotify through the handle
        # spotify.make_tools parks on services.spotify.
        self.focus = self._construct("focus", self._make_focus)
        self.services.focus = self.focus
        self.commander = Commander(self.services)
        # Without this hook the commander falls back to a bare warn Status --
        # a 4 s toast with no way to answer it, after which the utterance is
        # dropped and resolve_uncertain (and the classifier feedback it feeds)
        # is never reached from the running app at all.
        self.commander.on_uncertain = self._on_uncertain
        self.commander.claim_uncertain = self._claim_uncertain
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

        self._wire_turn_clock()
        self._init_assistant_state()
        bus.subscribe(RecordingStopped, self._on_recording_stopped)
        bus.subscribe(RecordingStarted, self._on_recording_started)
        bus.subscribe(ClaudeProgress, self._on_claude_progress)
        bus.subscribe(ClaudeTaskState, self._on_claude_state)
        bus.subscribe(ApprovalRequested, self._on_approval_requested)
        bus.subscribe(ApprovalResolved, self._on_approval_resolved)
        bus.subscribe(AlarmFired, self._on_alarm_fired)
        bus.subscribe(ReminderFired, self._on_reminder_fired)
        bus.subscribe(JarvisReply, self._on_reply_for_discord)
        bus.subscribe(Presence, self._on_presence)

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

    def _make_focus(self):
        mod = _import_optional("jarvis.focus")
        if mod is None:
            return None
        return mod.FocusSession(self.services,
                                state_path=PATHS.MEMORY_DIR / "focus_session.json")

    def _make_presence(self):
        mod = _import_optional("jarvis.presence")
        return None if mod is None else mod.PresenceSentinel(self.assistant)

    def _make_quiet(self):
        mod = _import_optional("jarvis.quiet")
        if mod is None:
            return None
        presence = self.presence
        def _can_speak():
            # The catch-up digest must not land mid-capture or over a turn:
            # a 30 s tick retries until the floor is clear.
            return not (self._turn_busy.is_set() or self._audio_busy.is_set()
                        or getattr(getattr(self, "recorder", None), "recording", False)
                        or getattr(getattr(self, "tts", None), "busy", False) is True)
        kwargs = dict(
            get_calendar=lambda: getattr(getattr(self, "services", None), "calendar", None),
            is_home=(presence.is_home if presence is not None else None),
            say=self._say)                  # the digest is an answer, never held
        # Lazy: self.focus is constructed AFTER the policy, so this must be a
        # late lookup, not the object.
        extra = dict(can_speak=_can_speak,
                     get_focus=lambda: getattr(self, "focus", None))
        try:
            policy = mod.QuietPolicy(self.assistant, **extra, **kwargs)
        except TypeError:                   # quiet.py without the predicates yet
            policy = mod.QuietPolicy(self.assistant, **kwargs)
        try:
            from jarvis.channels import notify
            notify.set_quiet_gate(policy.is_quiet)
        except Exception:
            log.exception("quiet: banner gate not installed")
        return policy

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

    def _make_cmdsock(self):
        mod = _import_optional("jarvis.cmdsock")
        if mod is None:
            return None
        return mod.CommandSocket(PATHS.COMMAND_SOCK, self)

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
    def _say(self, text, proactive=False, kind="message"):
        """The one door to TTS. ``proactive=True`` marks a line Jarvis
        decided to say on his own (watchdog, reminder, heads-up, narrator);
        quiet hours / DND / a running meeting / an empty room hold those
        for the catch-up digest (jarvis/quiet.py). Answers, alarms and
        approval questions pass the default False and are never held."""
        # A quiet CLI turn (python -m jarvis.ask -q) is answered in text
        # only. Per turn, not a global toggle: a voice turn that lands
        # while the CLI answer is still coming resets _last_source.
        if getattr(self, "_quiet_turn", False) and \
                getattr(self, "_last_source", "") == "cli":
            return
        if not text or not CONFIG.talkback:
            return
        quiet = getattr(self, "quiet", None)
        if proactive and quiet is not None:
            try:
                if quiet.should_hold():
                    # False = dropped, not parked (an interval nudge expires
                    # rather than joining the digest); say which happened.
                    kept = quiet.hold(text, kind) is not False
                    bus.publish(Status(text=f"{'Held' if kept else 'Expired'} "
                                       f"({quiet.reason() or 'quiet'}): "
                                       f"{text[:60]}", kind="info"))
                    return
            except Exception:
                log.exception("quiet gate failed; speaking")
        self.tts.speak(text)

    def _async_reply(self, text, speak=True):
        """A Tier 1 handler's answer arriving from its worker thread (an
        explained document, the first quiz question): shown, spoken,
        remembered, and it closes the turn and arms the follow-up window
        exactly as a brain reply does in _on_brain_tags. Without this door
        a done=False command only ended when the 60 s watchdog fired, and
        the next question needed the wake word."""
        at = getattr(self, "_async_turn", None)
        if at is not None:
            gen, src = at
            stale = gen != getattr(self, "_dispatch_gen", 0) or \
                (src == "voice" and not self._turn_busy.is_set())
            if stale:
                # A newer turn owns the floor (or the watchdog closed this
                # one): the late answer is shown, never spoken, and it must
                # not close the live turn or open a mic over it.
                log.info("stale worker reply shown as a card only")
                if text:
                    bus.publish(JarvisReply(text=text, speak=False))
                return
            self._async_turn = None
        self._turn_finished()
        if not text:
            return
        bus.publish(JarvisReply(text=text, speak=speak,
                                turn_id=getattr(self, "_active_turn_id", "")))
        if speak:
            self._say(text)
            if self._last_source == "voice":
                self._followup_after_speech = True
        if getattr(self, "_last_source", "") == "cli":
            self._quiet_turn = False    # this CLI turn's answer is delivered
        try:
            self.context.add_exchange(self._last_user_text, text)
        except Exception:
            log.debug("add_exchange failed", exc_info=True)

    def interrupt_speech(self) -> bool:
        """Barge-in: cut whatever Jarvis is saying (and any queued lines,
        including a read-aloud in progress). Returns True when something
        was actually cut off. Wired to typed input below; the UI may also
        call it on the first keystroke (see scratchpad ui_hooks_todo.md)."""
        try:
            # A paused reading has nothing pending in the TTS but still
            # owns the transport words; stop() clears that too.
            if self.reader.pending_chunks or getattr(self.reader, "active", False):
                self.reader.stop()
            return self.tts.interrupt()
        except Exception:
            log.exception("interrupt_speech failed")
            return False

    def _canned_phrases(self):
        """Short lines Jarvis says verbatim and often — rendered into the
        speech cache at startup so they play instantly."""
        phrases = [p for lines in COURTESY_REPLIES.values() for p in lines]
        phrases += list(THINKING_LINES)
        phrases += [SAY_AGAIN_LINE, NUDGE_LINE, self._guest_line]
        phrases += [CONTINUE_PROMPT, "Very good, sir.", "Welcome back, sir.",
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
                                           "NO_SESSION_LINE", "IDLE_LINE")),
                ("jarvis.approvals", ("TIMEOUT_LINE", "ALLOWED_LINE",
                                      "DECLINED_LINE")),
                ("jarvis.commander", ("TERMINAL_OPEN_LINE", "TERMINAL_FAIL_LINE",
                                      "WEB_LOOKUP_LINE", "WEB_UNAVAILABLE_LINE",
                                      "EXPLAIN_FAIL_LINE", "READ_OFFER_LINE",
                                      "NO_DOCUMENT_LINE")),
                ("jarvis.lecture", ("END_NONE_LINE", "FAIL_LINE")),
                ("jarvis.tools.docs", ("INDEX_DOWN_LINE", "INDEXING_LINE",
                                       "NO_QUESTION_LINE")),
                ("jarvis.tools.quiz", ("NO_DOCS_LINE", "NOTHING_DUE_LINE", "NO_CARDS_LINE",
                                       "QUIZ_STOPPED_EARLY_LINE")),
                ("jarvis.tools.screen", ("NO_SCREEN_LINE", "NO_VISION_LINE")),
                ("jarvis.tools.health", ("UNREADABLE_LINE",)),
                ("jarvis.tools.timekeeper", ("NOTHING_RINGING_LINE",
                                             "NO_TIMEKEEPER_LINE")),
                ("jarvis.brain", ("MODEL_DOWN_LINE", "MODEL_SLOW_LINE",
                                  "MODEL_EMPTY_LINE", "TOOL_ONLY_LINE",
                                  "PARTIAL_RESULT_LINE", "INTERNAL_ERROR_LINE", "WEB_SLOW_LINE", "WEB_FAIL_LINE",
                                  "NO_CLOCK_LINE", "UNSURE_CLOCK_LINE"))):
            mod = sys.modules.get(modname)
            for name in names:
                line = getattr(mod, name, None) if mod else None
                if isinstance(line, str) and line and "{" not in line:
                    phrases.append(line)
        # Whole lists of fixed lines (the module is already imported when its
        # tools registered; a missing module simply contributes nothing).
        for modname, name in (("jarvis.tools.spotify", "PERSONA_LINES"),
                              ("jarvis.tools.canvas", "PERSONA_LINES"),
                              ("jarvis.focus", "PERSONA_LINES")):
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
            # V3 probe: the primary repo (V1's git_summary was hard-coded to
            # ~/vss_env, so "git status" reported the VSS tree, not Jarvis).
            git_summary=ctx.git_summary,
            git_repos=ctx.git_repos,
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

        def chat(text, force_tool=None, force_args=None):
            # Every keyword the real JarvisBrain.chat accepts must be forwarded
            # here: the commander only ever sees this wrapper, and a keyword it
            # does not take raises TypeError inside the handler, which the
            # dispatcher turns into "Command failed: <name>" for the user.
            extra = {"force_args": force_args} if force_args is not None else {}
            if CONFIG.stream_replies:
                extra["on_sentence"] = app._on_stream_sentence
            return b.chat(text, callback=app._on_brain_tags,
                          force_tool=force_tool, **extra)

        def web_answer(question, model="haiku"):
            return b.web_answer(question, callback=app._on_brain_tags, model=model)

        brain_ns = SimpleNamespace(
            think=lambda text: b.think(text, callback=app._on_brain_tags),
            chat=chat,
            web_answer=web_answer,
            # "no, I said ...": the misheard turn's model job is cut so its
            # reply is neither spoken nor remembered (brain job generation).
            cancel=lambda: app.brain.cancel(),
            # Delegate lazily rather than capturing the bound methods: the
            # namespace is built once at init, so a snapshot here would make
            # `app.brain.<fn> = ...` (tests, and any later brain swap) a no-op
            # that silently reached the real local model instead.
            classify_route=lambda *a, **kw: app.brain.classify_route(*a, **kw),
            summarize=lambda *a, **kw: app.brain.summarize(*a, **kw),
            local_line=lambda *a, **kw: app.brain.local_line(*a, **kw),
            explain_text=lambda *a, **kw: app.brain.explain_text(*a, **kw),
            make_quiz=lambda *a, **kw: app.brain.make_quiz(*a, **kw),
            grade_answer=lambda *a, **kw: app.brain.grade_answer(*a, **kw),
            execute_autonomous=lambda task: b.execute_autonomous(
                task, callback=app._on_brain_tags),
            # GPU yield: "lend the GPU" / "take the GPU back" (commander)
            release=lambda: app.brain.release(),
            reclaim=lambda: app.brain.reclaim(),
            is_lent=lambda: app.brain.is_lent(),
        )

        return SimpleNamespace(
            desktop=desktop_ns, context=context_ns, memory=self.memory,
            # the engine itself, for the journal tool and the window
            # sampler (context= above is the narrow V1-shaped adapter)
            context_engine=self.context,
            workflows=workflows_ns, brain=brain_ns, tts=self.tts,
            reader=self.reader, history=self.history,
            # the ContextEngine itself (`context` above is the agent's
            # namespace): corrections drop the misheard exchange from it
            conversation=self.context,
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
            diagnostics=self.diagnostics_text,
            log_triage=self.log_triage_text,
            slow_turn=self.slow_turn_text,
            # "how did yesterday go": the day review, spoken (dayreview.py)
            dayreview=self.day_review_text,
            # the health watchdog resolves this at fire time (talkback-gated).
            # Its warnings are proactive: quiet hours hold them for the digest
            # ("...and a memory warning") instead of waking him at 3 am.
            speak=lambda text, proactive=True, kind="warning": self._say(
                text, proactive=proactive, kind=kind),
            # a Tier 1 worker thread's answer (explain, quiz): see _async_reply
            reply=self._async_reply,
            # docs.make_tools parks its DocsIndex here for quiz mode
            docs=None,
            # quiet hours / DND and the presence sentinel (commander, tools)
            quiet=self.quiet, presence=self.presence,
        )

    # ------------------------------------------------------- brain executor
    def _on_stream_sentence(self, sentence):
        """A sentence of the reply, as the model produces it: speak it now.
        The full reply follows in the tags with a STREAMED marker so it is
        shown, remembered and not spoken again."""
        if getattr(self, "_stream_muted", False):
            return                  # barged in: the rest of this reply is dropped
        # The answer has started, so no "thinking" line is warranted -- and
        # one queued now would play BETWEEN the answer's sentences (the TTS
        # queue is FIFO). Disarm the filler; the watchdog stays.
        t, self._turn_timer = getattr(self, "_turn_timer", None), None
        if t is not None:
            t.cancel()
        if self._last_source == "voice":
            self._followup_after_speech = True
        self._say(sentence)

    def _on_brain_tags(self, tags):
        """Port of the monolith's _on_brain_response: act on [TAG] tuples.
        A ("BRIEFING", json) tag turns that turn's SPEAK into ONE
        BriefingReady card (no separate JarvisReply) — still spoken."""
        # Whatever else these tags mean, their arrival ends the turn.
        self._turn_finished()
        briefing = None
        offer = ""
        streamed = any(tag == "STREAMED" for tag, _ in tags)
        for tag, content in tags:
            if tag == "BRIEFING":
                try:
                    briefing = json.loads(content) if isinstance(content, str) else content
                    if not isinstance(briefing, dict):
                        briefing = {"text": str(briefing)}
                except (TypeError, ValueError):
                    log.warning("BRIEFING tag carried non-JSON payload")
                    briefing = {}
                # The evening preview's "Shall I wake you at seven?" is
                # asked HERE, verbatim, after the spoken preview -- not left
                # to the model, which would phrase (or drop) it. The yes/no
                # is read by commander._try_alarm_offer from the offer the
                # tool parked on services.alarm_offer.
                offer = str(briefing.get("offer") or "").strip()
        for tag, content in tags:
            try:
                if tag == "SPEAK":
                    # Spoken in full, deliberately. Capping this to two
                    # sentences was tried on 2026-08-28 and reverted: cutting
                    # an answer short mid-list is the LEAST human-sounding
                    # option available. What made a long reply feel wrong was
                    # the gap between chunks, and that is fixed in the
                    # splitter (tts._split_sentences), not by saying less.
                    tid = getattr(self, "_active_turn_id", "")
                    if briefing is not None:
                        bus.publish(BriefingReady(sections=briefing, spoken=content,
                                                  turn_id=tid))
                        briefing = None
                    else:
                        bus.publish(JarvisReply(text=content, speak=not streamed,
                                                turn_id=tid))
                    if not streamed:          # streamed sentences already spoke
                        self._say(content)
                    if offer:
                        self._say(offer)
                        offer = ""
                        # the answer window opens whatever the source: the
                        # question was put to him aloud
                        self._followup_after_speech = True
                    # brain._remember has already recorded this exchange;
                    # recording it here too rendered every turn twice.
                    if self._last_source == "voice":
                        self._followup_after_speech = True
                elif tag in ("BRIEFING", "STREAMED"):
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
                    self.context.add_exchange(self._last_user_text, content)
                # SILENT (and bare DONE): nothing to do
            except Exception:
                log.exception("brain tag %s failed", tag)
        if getattr(self, "_last_source", "") == "cli":
            self._quiet_turn = False    # this CLI turn's answer is delivered
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
            # Proactive: quiet hours / DND hold the narration for the digest
            self._say(ev.line, proactive=True)
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
                self._say(line, proactive=True)
            self._alert("blocked", f"Claude · {ev.project}", line)
            self._journal_claude(ev.project, "failed", ev.text or line)
        elif ev.state == "cancelled":
            self._last_milestone.pop(ev.task_id, None)
            bus.publish(Status(text="Claude task cancelled", kind="info"))

    def _journal_claude(self, project, state, text):
        journal = getattr(self.context, "journal_claude", None)
        if journal is None:
            return
        try:
            journal(project, state, text)
        except Exception:
            log.exception("journal_claude failed")

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
        self._say(spoken, proactive=True)
        # The alert queues before the journal writes: a listener waiting on
        # the reply then flushing the alerts must find it queued already.
        self._alert("done", f"Claude · {ev.project}", spoken)
        self.context.add_exchange("", spoken)
        self._journal_claude(ev.project, "done", text)

    # ---------------------------------------------------------- presence
    def _on_presence(self, ev):
        """The phone came back (or left). One greeting per return, then
        whatever was held while he was out -- the quiet policy's own tick
        would read it too, so release() drains atomically and whichever
        gets there first says it."""
        bus.publish(Status(text="Home" if ev.home else "Away", kind="info"))
        if not ev.home or not ev.returned:
            return
        from jarvis.presence import WELCOME_LINE
        quiet = getattr(self, "quiet", None)
        if quiet is not None:
            try:
                reason = quiet.reason()
            except Exception:
                log.exception("presence: quiet gate failed")
                reason = ""
            # "you're out" is quiet.py's away reason: stale by definition on
            # a returned event, so it never defers the greeting. Any OTHER
            # reason (hours, DND, a meeting) does -- the backlog then waits
            # for the policy's own tick.
            if reason and reason != "you're out":
                return
        self._say(WELCOME_LINE)
        if quiet is None:
            return
        try:
            digest = quiet.release()
        except Exception:
            log.exception("presence: digest failed")
            return
        if digest:
            bus.publish(JarvisReply(text=digest, speak=True))
            self._say(digest)

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
        if getattr(ev, "silent", False):
            return              # a focus block / break: the session speaks for it
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

    _WAKE_BEEP_GUARD_S = 0.2   # only when CONFIG.sound plays a chime

    def _on_hotword(self, score):
        # Called from the hotword listener thread.
        if self.recorder.recording:
            return
        # A wake word over Jarvis's own voice is a barge-in ("Jarvis, stop").
        # A streamed reply keeps the turn open for as long as the model is
        # producing sentences, so the turn gate must not refuse it: that
        # made barge-in unreachable exactly when a reply was long.
        # _tts_active only rises at first AUDIO now (the honest ledger
        # mark), so during the render gap it is False while a reply is very
        # much on the way -- the TTS queue is the truthful predicate.
        speaking = getattr(getattr(self, "tts", None), "busy", False) is True or \
            getattr(self, "_tts_active", False)
        barge = CONFIG.barge_in and speaking
        if self._audio_busy.is_set() or (self._turn_busy.is_set() and not barge):
            # Transcription of the previous utterance is still running (~20 s
            # for a long clip). Starting a second capture here raced two
            # transcripts into the commander. Say so rather than ignoring it
            # silently -- an unanswered wake word reads as a broken mic.
            log.info("hotword ignored: still transcribing the previous clip")
            bus.publish(Status(text="One moment — still on the last one",
                               kind="warn"))
            return
        if barge:
            try:
                if self._turn_busy.is_set():
                    # The model is still generating: stop it, drop the
                    # sentences still in flight, and close that turn so its
                    # tags cannot land on this one.
                    self._stream_muted = True
                    cancel = getattr(getattr(self, "brain", None), "cancel", None)
                    if callable(cancel):
                        cancel()
                    self._turn_finished()
                self.interrupt_speech()
            except Exception:
                log.exception("barge-in interrupt failed")
        self.turns.mark("wake")            # accepted: this turn starts now
        self._followup_after_speech = False   # a wake supersedes any follow-up
        self._turn_filler_pending = False     # a stale flag would label this answer a filler
        self._say_again_count = 0
        self._wake_pending = True             # the capture about to open answers a wake word
        if CONFIG.sound:
            threading.Thread(target=play_beep, args=("start",), daemon=True).start()
        # The wake word ends and the user starts talking straight away, so
        # every millisecond before the mic opens is speech thrown away --
        # measured 216-298 ms, median 259, of which this timer was 200.
        #
        # The wait exists so the start chime is not recorded back through the
        # mic, so it belongs only where a chime actually plays. The thread hop
        # is NOT optional and stays either way: start() acquires the arbiter,
        # which pauses the hotword, and this runs ON the hotword listener
        # thread -- pausing it from inside itself would wedge the listener.
        # Timer(0) still hands off to a new thread.
        threading.Timer(self._WAKE_BEEP_GUARD_S if CONFIG.sound else 0.0,
                        self.recorder.start).start()

    # ------------------------------------------------- live transcript
    #
    # PartialText, Transcriber.partial() and the UI's ghost card all shipped
    # with V3; nothing ever connected them, so words only appeared once the
    # user stopped talking. partial()'s own docstring said the live typing
    # "stays in the pipeline" -- this is that missing piece.

    _PARTIAL_INTERVAL_S = 0.9    # re-decode cadence while speaking
    _PARTIAL_MIN_S = 0.7         # below this whisper mostly invents words
    _PARTIAL_MAX_S = 20.0        # decode only the newest span: each pass
                                 # re-decodes the whole buffer, and the final
                                 # transcribe() waits on the same lock

    def _on_recording_started(self, _ev):
        # Consume the wake flag into this capture: a follow-up window or the
        # mic button opens without one, and a stale True from an earlier
        # wake must not make the nudge policy treat those as wake turns.
        self._turn_from_wake, self._wake_pending = self._wake_pending, False
        with self._spec_lock:
            self._speculation = None        # a pass from the last capture is never this one's
        threading.Thread(target=self._partial_loop, name="partial",
                         daemon=True).start()

    def _partial_loop(self):
        """Re-decode the growing buffer and publish PartialText; run the
        speculative decode once the user has paused (_maybe_speculate).

        Deliberately a separate thread: Recorder._poll_loop runs at ~12 Hz
        and owns silence detection, so a decode taking hundreds of ms there
        would delay auto-stop. Everything here is best-effort -- a preview
        must never delay, disturb or fail the real transcription that
        follows. Transcriber.partial() and transcribe() share one lock, so
        the final decode simply waits for at most one preview.

        One thread for both duties on purpose: the speculative pass is a
        full decode, and running it here means the greedy preview is
        suspended while it holds the model instead of queueing behind it.
        """
        last = ""
        due = 0.0                       # next greedy preview (monotonic)
        try:
            while self.recorder.recording:
                if self._maybe_speculate():
                    # The pass just held the model for a full decode and
                    # published its own text: push the preview back a whole
                    # interval rather than re-decoding the same silence.
                    due = time.monotonic() + self._PARTIAL_INTERVAL_S
                    continue
                if time.monotonic() < due:
                    time.sleep(self._SPECULATE_POLL_S)
                    continue
                started = time.monotonic()
                audio = self.recorder.snapshot_audio()
                if audio is not None:
                    audio = audio[-int(SAMPLE_RATE * self._PARTIAL_MAX_S):]
                if audio is not None and len(audio) >= int(
                        SAMPLE_RATE * self._PARTIAL_MIN_S):
                    try:
                        text = (self.transcriber.partial(audio) or "").strip()
                    except Exception:
                        log.debug("partial decode failed", exc_info=True)
                        text = ""
                    # only publish on change: the ghost card redraws on
                    # every event, and whisper often returns the same text.
                    if text and text != last and self.recorder.recording:
                        last = text
                        bus.publish(PartialText(text=text))
                # pace from the END of the decode, so a slow pass backs off
                # instead of queueing up behind itself.
                due = time.monotonic() + max(0.05, self._PARTIAL_INTERVAL_S -
                                             (time.monotonic() - started))
        except Exception:
            log.exception("partial loop died")

    # ------------------------------------------- speculative transcription
    #
    # Every turn paid the full decode strictly AFTER the recorder stopped,
    # while the GPU idled through the 0.8 s endpoint silence (ledger
    # 2026-08-29: dead-air 0.80-0.96 s, then stt 0.40-0.67 s). Once the VAD
    # has heard _SPECULATE_AFTER_S of silence the clip is decoded on the
    # partial thread exactly as _process_audio would decode it -- the same
    # final-clip shaping (recorder.snapshot_final), the same speaker filter,
    # the same full-quality transcribe -- and if no more speech arrives
    # before the stop, _process_audio publishes that result instead of
    # decoding again. The saving is the overlap (~0.4-0.5 s on a short
    # question), not the whole stt figure. A pause that turns out to be
    # mid-sentence ("um...") wastes one decode, bounded like a stray
    # preview; only one pass runs per pause, and one at a time.

    _SPECULATE_AFTER_S = 0.3     # VAD silence before a speculative decode
    _SPECULATE_POLL_S = 0.08     # endpointer poll cadence on the partial thread
    _SPECULATE_JOIN_S = 30.0     # never wedge the real path behind a stuck pass

    # Class-level defaults so a bare object (the preview-thread tests build
    # the app with object.__new__) behaves like an app with no speculation.
    _speculation = None
    _spec_lock = threading.Lock()
    _stop_event = None
    _wake_pending = False
    _turn_from_wake = False
    _last_nudge_ts = -1e9

    def _listening_opt(self, key, default):
        """assistant.json ``listening.<key>``; silent when there is no
        assistant config (get_option logs an exception, and this runs on
        the preview thread several times a second)."""
        cfg = getattr(self, "assistant", None)
        if cfg is None:
            return default
        try:
            value = cfg.get(f"listening.{key}", default)
        except Exception:
            return default
        return default if value is None else value

    def _decode_clip(self, audio):
        """The one decode path for a captured clip: speaker filter, then the
        full transcribe. Returns (audio, stats, rejected, result); rejected
        means the speaker gate dropped the whole clip (result is None)."""
        stats = {}
        if CONFIG.speaker_verify and self.speaker.enrolled:
            filtered, stats = self.speaker.filter_segments(audio)
            if filtered is None:
                return audio, stats, True, None
            audio = filtered
        return audio, stats, False, self.transcriber.transcribe(audio)

    def _maybe_speculate(self) -> bool:
        """Run one speculative decode when the VAD has heard
        _SPECULATE_AFTER_S of silence since a word that has not been
        speculated yet. Returns True when a pass held the model."""
        rec = self.recorder
        ep = getattr(rec, "endpointer", None)
        if ep is None or not CONFIG.endpoint_vad or \
                not self._listening_opt("speculative_stt", True):
            return False
        if not getattr(self.transcriber, "loaded", True):
            return False                # never load whisper from the preview thread
        try:
            gap, key = ep.silence_since_speech, ep.last_speech_seconds
        except Exception:
            return False                # a stub or a broken endpointer: no speculation
        if gap is None or key is None or gap < self._SPECULATE_AFTER_S:
            return False
        with self._spec_lock:
            if not rec.recording:
                return False            # the stop fired: that decode is _process_audio's
            prev = self._speculation
            if prev is not None and prev.key == key:
                return False            # this pause was already decoded
            spec = _Speculation(key)
            self._speculation = spec
        audio = None
        try:
            audio = rec.snapshot_final()
            if audio is not None:
                spec.audio, spec.stats, spec.rejected, spec.result = \
                    self._decode_clip(audio)
        except Exception as exc:        # noqa: BLE001 - best effort by design
            spec.error = exc
            log.debug("speculative decode failed", exc_info=True)
        finally:
            spec.finished = time.monotonic()
            spec.done.set()
        if audio is None:
            return False                # too short to shape: nothing ran
        result = spec.result
        text = (result.text or "").strip() if result is not None else ""
        if text and result.accepted and rec.recording:
            # The speculative text IS the best preview there is.
            bus.publish(PartialText(text=text))
        log.debug("speculative decode at last_speech=%.2fs took %.2fs (%s)",
                  key, spec.finished - spec.started,
                  "rejected" if spec.rejected else repr(text))
        return True

    def _take_speculation(self):
        """The speculative decode for the clip that just stopped, or None.

        Takes the stash (a later pass can never be mistaken for this
        turn), waits for a pass still in flight -- a second decode would
        only queue behind it on the model lock -- and validates it: the
        stop must be the VAD's own (a manual or energy stop may hold frames
        the endpointer never scored) and the last word the VAD heard must
        be the one that was speculated (more speech followed otherwise).
        """
        with self._spec_lock:
            spec, self._speculation = self._speculation, None
        if spec is None:
            return None
        if not spec.done.wait(self._SPECULATE_JOIN_S):
            log.warning("speculative decode still running after %.0fs; decoding normally",
                        self._SPECULATE_JOIN_S)
            return None
        ev = self._stop_event
        ep = getattr(self.recorder, "endpointer", None)
        try:
            key = ep.last_speech_seconds if ep is not None else None
        except Exception:
            key = None
        endpoint = getattr(ev, "endpoint", "") if ev is not None else ""
        if endpoint != "vad":
            log.info("speculative decode discarded: stop=%s", endpoint or "?")
            return None
        if key is None or key != spec.key:
            log.info("speculative decode discarded: speech followed (%.2fs -> %s)",
                     spec.key, "%.2fs" % key if key is not None else "-")
            return None
        if not spec.usable:
            return None
        return spec

    def _install_endpointer(self):
        """Silero VAD for the recorder. Its own step, with no dependency on
        the speaker model: the first version sat inside the enrolment branch,
        so a box with no voiceprint (or a failed ECAPA load) silently kept
        the 2.5 s energy timer while the config promised endpointing."""
        if not CONFIG.endpoint_vad:
            log.info("endpointing off by config; energy timer only")
            return
        try:
            from jarvis.endpoint import VoiceEndpointer
            ep = VoiceEndpointer()
            if ep.warm():
                self.recorder.endpointer = ep
            else:
                log.warning("silero VAD did not load; energy timer only")
        except Exception:
            log.exception("endpointer unavailable; energy timer only")

    # --------------------------------------------------- assistant state
    def _init_assistant_state(self):
        self._app_started = time.monotonic()
        self._last_user_text, self._last_source = "", "voice"
        self._followup_after_speech = False   # arm a wake-word-free listen
        self._reopen_mic = False              # lecture notes: re-open after a silent turn
        self._briefing_pending = False        # first wake of the day
        # -1e9, not 0.0: time.monotonic() counts from boot, so a 0.0 stamp
        # muted the guest line and passive learning for the first 3 / 10
        # minutes after a reboot.
        self._last_guest_ts = -1e9
        self._last_learn_ts = -1e9
        self._say_again_count = 0
        self._stream_muted = False
        self._quiet_turn = False              # CLI turn asked for text only
        self._active_turn_id = ""             # stamps this turn's replies (cli)
        self._async_turn = None               # (gen, source) of a done=False turn
        self._dispatch_gen = 0
        # speculative transcription (_maybe_speculate) and the nudge policy
        self._speculation = None
        self._spec_lock = threading.Lock()
        self._stop_event = None               # the RecordingStopped being processed
        self._wake_pending = False            # a wake word is opening the mic
        self._turn_from_wake = False          # this capture answers a wake word
        self._last_nudge_ts = -1e9            # monotonic; see _last_guest_ts
        name = self.assistant.user_name if self.assistant is not None else "Hunter"
        self._guest_line = GUEST_LINE.format(name=name)

    def _after_dispatch(self, text, source, result):
        """Bookkeeping once a command has been handled synchronously."""
        # "no, I said X" / "that was for you" answered X, not the words
        # said: the exchange is remembered under X.
        text = getattr(result, "corrected", None) or text
        reply = getattr(result, "reply", None)
        done = getattr(result, "done", True) is not False
        status = result.status or ""
        if status.startswith("Noting:"):
            # A lecture line: nothing is spoken, so the follow-up window
            # would never open. Re-open the mic once this turn closes
            # (_dispatch) -- and keep the line out of the conversation
            # memory, it is a note, not an exchange.
            if source == "voice":
                self._reopen_mic = True
            return
        if reply and done and not getattr(result, "ack", False):
            self.context.add_exchange(text, reply)
            if source == "voice" and result.speak and CONFIG.talkback:
                self._followup_after_speech = True
        if source == "voice":
            if status.startswith("Briefing"):
                self._mark_briefing_delivered()
            elif status.startswith(("Was that for me", "Ignored")):
                pass        # no answer here: the briefing waits for a real turn
            elif (reply or not done) and self._briefing_due():
                self._briefing_pending = True

    def _after_speech(self):
        """Jarvis just finished a spoken burst: deliver a pending first-wake
        briefing, else open the follow-up window.

        A burst is not the end of the turn. An ack ("Looking that up, sir"),
        a thinking line and every streamed sentence each end in a falling
        edge, and TTS may already hold the next line -- so a briefing fired
        here landed on the in-flight brain call ("Still on the last one,
        sir") and the follow-up mic opened under the answer. Wait for the
        answer's own edge: nothing pending, no open mic, no queued speech."""
        if self._turn_busy.is_set() or self._audio_busy.is_set() \
                or getattr(self.recorder, "recording", False):
            return                          # the answer is still coming / mic open
        if getattr(self, "_pending_uncertain", None):
            return                          # "Was that for me?" is waiting on a yes/no
        tts = getattr(self, "tts", None)
        if getattr(tts, "is_speaking", False) or getattr(tts, "pending", 0):
            return                          # more speech is queued behind this burst
        if self._briefing_pending:
            self._briefing_pending = False
            self._followup_after_speech = False
            self._deliver_first_wake_briefing()
            return
        if self._followup_after_speech:
            self._followup_after_speech = False
            self._start_followup()

    def _start_followup(self):
        """Listen for a follow-up without the wake word (CONFIG.followup_window)."""
        if CONFIG.followup_window <= 0 or not MACHINE.has_mic:
            return
        if self.recorder.endpointer is None:
            return                      # without a VAD nothing can say "nothing was said"
        if self.recorder.recording or self._audio_busy.is_set() or self._turn_busy.is_set():
            return
        window = self._capture_window()
        log.info("follow-up window: listening %.1fs without a wake word",
                 window or CONFIG.followup_window)
        # The keyword is only passed when a longer window is wanted: the
        # recorder's default is CONFIG.followup_window.
        kw = {"window": window} if window else {}
        self._wake_pending = False    # nothing heard here is a normal outcome
        threading.Timer(0.15, lambda: self.recorder.start(followup=True, **kw)).start()

    def _capture_window(self):
        """A longer wait for the first word while lecture notes are open
        (`lecture.window_s`, default 20 s), else None for the recorder's
        own CONFIG.followup_window. The recorder caps it at half its hard
        cap. This is still one capture per note, not a hands-free mic."""
        commander = getattr(self, "commander", None)
        if not getattr(commander, "lecture_course", None):
            return None
        try:
            window = float(self.assistant.get("lecture.window_s", 20) or 20)
        except (TypeError, ValueError, AttributeError):
            window = 20.0
        return max(window, float(CONFIG.followup_window))

    # ---------------------------------------------------- guests, learning
    def _on_guest(self, score):
        """A clear wake word in a voice that is not the enrolled one."""
        if score < 0.85 or not CONFIG.talkback:
            return
        if getattr(getattr(self, "tts", None), "busy", False) is True or \
                getattr(self, "_tts_active", False):
            return          # under barge-in the listener hears his own voice
        now = time.monotonic()
        if now - self._last_guest_ts < 180.0:
            return
        self._last_guest_ts = now
        log.info("guest wake (score=%.2f): declining politely", score)
        self._say(self._guest_line)

    def _maybe_learn_voice(self, audio, stats):
        """Passive enrolment: an accepted utterance that matched the
        voiceprint comfortably joins the pool, at most once every ten
        minutes, so recognition tracks distance, colds and time of day.
        add_sample() re-checks the score and refuses a pre-trim pool."""
        try:
            scores = list(stats.get("scores") or [])
            if not scores or not CONFIG.speaker_verify:
                return
            if max(scores) < max(CONFIG.speaker_threshold + 0.2, 0.55):
                return
            if len(audio) < int(SAMPLE_RATE * 1.5):
                return
            now = time.monotonic()
            if now - self._last_learn_ts < 600.0:
                return
            self._last_learn_ts = now
            threading.Thread(target=self.speaker.add_sample, args=(audio,),
                             daemon=True, name="voice-learn").start()
        except Exception:
            log.debug("passive learning skipped", exc_info=True)

    # ------------------------------------------------- first-wake briefing
    def _briefing_state_path(self):
        # MEMORY_DIR, not AIWS: the suite redirects MEMORY_DIR, so a test
        # that builds a real app cannot mark the user's real day delivered.
        return PATHS.MEMORY_DIR / "briefing_state.json"

    def _briefing_due(self, now=None):
        try:
            if not self.assistant.get("briefing.on_first_wake", True):
                return False
            after = str(self.assistant.get("briefing.after", "06:00") or "06:00")
            hh, mm = (int(x) for x in after.split(":")[:2])
        except Exception:
            return False
        now = now or datetime.now()
        if (now.hour, now.minute) < (hh, mm):
            return False
        # Quiet hours / a running class: the day stays unmarked, so the first
        # answered turn after the window delivers it instead.
        quiet = getattr(self, "quiet", None)
        try:
            if quiet is not None and quiet.is_quiet():
                return False
        except Exception:
            log.debug("quiet check failed; briefing proceeds", exc_info=True)
        try:
            state = json.loads(self._briefing_state_path().read_text())
        except (OSError, ValueError):
            state = {}
        return state.get("delivered") != now.date().isoformat()

    def _mark_briefing_delivered(self):
        try:
            p = self._briefing_state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")     # a torn write must not eat the day
            tmp.write_text(json.dumps({"delivered": datetime.now().date().isoformat()}))
            os.replace(tmp, p)
        except OSError:
            log.debug("briefing state save failed", exc_info=True)

    def _deliver_first_wake_briefing(self):
        brain = getattr(self.services, "brain", None)
        if brain is None or not hasattr(brain, "chat"):
            return
        if getattr(getattr(self, "brain", None), "is_busy", False):
            # chat() would only say "Still on the last one, sir": leave the
            # day unmarked so the next answered turn delivers it.
            log.info("first-wake briefing: model busy; next turn")
            return
        log.info("first wake of the day: delivering the briefing")
        # Yesterday's self-review first, as its own line: the briefing is a
        # brain.chat(force_tool="get_briefing") call, so nothing can be
        # folded into it "for free" -- and only when there was a yesterday
        # to review (nothing said on a fresh box, or after a day off).
        try:
            review = self._review_line(datetime.now().date() - timedelta(days=1),
                                       label="Yesterday")
        except Exception:
            log.exception("day review for the first wake failed")
            review = ""
        if review:
            self._say(review)
        self._say("Your briefing for today, sir.")
        try:
            brain.chat("my morning briefing", force_tool="get_briefing")
        except Exception:
            log.exception("first-wake briefing failed")
            return
        self._mark_briefing_delivered()     # after the ask, not before

    # -------------------------------------------------------- day review
    def _review_line(self, day, label="Yesterday") -> str:
        """The two-sentence review of `day` (dayreview.spoken_line), from
        the filed digest when the nightly timer has run, else computed
        now. Empty when there is nothing for that day."""
        reviewer = getattr(self, "dayreviewer", None)
        if reviewer is not None:
            digest = reviewer.review(day)
        else:
            digest = dayreview_mod.summarize_day(PATHS.LOG_DIR / "jarvis.log",
                                                 PATHS.LOG_DIR / "turns.jsonl", day)
        name = "sir"
        return dayreview_mod.spoken_line(digest, label=label, name=name)

    def day_review_text(self, which="yesterday") -> str:
        """"How did yesterday go" / "how is today going": the spoken review."""
        today = datetime.now().date()
        if str(which).lower() == "today":
            # today is still open: never read from a filed digest
            digest = dayreview_mod.summarize_day(PATHS.LOG_DIR / "jarvis.log",
                                                 PATHS.LOG_DIR / "turns.jsonl", today)
            line = dayreview_mod.spoken_line(digest, label="So far today")
            return line or "Nothing to report yet today, sir."
        line = self._review_line(today - timedelta(days=1), label="Yesterday")
        return line or "I have no record of yesterday, sir. Either I was off, or the log has gone."

    def _on_review_filed(self, day, digest):
        """The nightly timer filed a digest: the table goes to Discord
        through the Alerts hub (milestone = no desktop banner, and the hub
        only posts when the channel is configured)."""
        self._alert("milestone", f"Day review {day.isoformat()}",
                    dayreview_mod.table(digest))

    # ------------------------------------------------------- diagnostics
    def diagnostics_text(self) -> str:
        """"Run diagnostics": a spoken status in character, from real data."""
        import statistics
        parts = []
        up = time.monotonic() - getattr(self, "_app_started", time.monotonic())
        hours, mins = int(up // 3600), int((up % 3600) // 60)
        uptime = (f"{hours} hour{'s' if hours != 1 else ''} and {mins} minute{'s' if mins != 1 else ''}"
                  if hours else f"{mins} minute{'s' if mins != 1 else ''}")
        engine = getattr(self.tts, "engine", CONFIG.tts_engine)
        parts.append(f"All systems nominal, sir. Up {uptime}; whisper {CONFIG.model} on the GPU, "
                     f"{_brain_model_name()} answering, {engine} speaking"
                     f"{', voice-activity endpointing live' if self.recorder.endpointer else ''}.")
        try:
            waits, n = [], 0
            day = datetime.now().date()
            for line in (PATHS.LOG_DIR / "turns.jsonl").read_text().splitlines():
                rec = json.loads(line)
                if datetime.fromtimestamp(rec.get("at", 0)).date() != day:
                    continue
                if rec.get("outcome") == "abort":
                    continue            # a silent follow-up window, not a turn
                n += 1
                if rec.get("wait") is not None:
                    waits.append(rec["wait"])
            if n:
                med = f", median wait {statistics.median(waits):.1f} seconds" if waits else ""
                parts.append(f"{n} turn{'s' if n != 1 else ''} today{med}.")
        except (OSError, ValueError):
            pass
        try:
            mem = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0])
            free = mem["MemAvailable"] / 1048576
            total = mem["MemTotal"] / 1048576
            gpu = ""
            try:
                # Popen + kill-without-wait (jarvis.tools.health): run() would
                # wait on an nvidia-smi wedged in D-state under a stuck NVRM
                # lock, and this is the command asked in exactly that state.
                from jarvis.tools.health import parse_nvidia_smi, run_nvidia_smi
                reading = parse_nvidia_smi(run_nvidia_smi()) or {}
                if reading.get("temp_c") is not None:
                    gpu = f", GPU at {reading['temp_c']:.0f} degrees"
            except Exception:
                pass
            parts.append(f"Memory {free:.0f} of {total:.0f} gigabytes free{gpu}.")
        except Exception:
            pass
        if self.speaker is not None and self.speaker.enrolled:
            parts.append(f"Your voiceprint holds {self.speaker.num_samples} samples.")
        return " ".join(parts)

    def log_triage_text(self) -> tuple:
        """"Anything wrong in your log?": (spoken, card) from the tail of
        jarvis.log and the turn ledger (jarvis/logtriage.py)."""
        from jarvis.logs import LOG_FILE
        from jarvis.logtriage import (TAIL_LINES, cluster_warnings, read_tail,
                                      read_turns, triage_text, turn_outliers)
        lines = read_tail(LOG_FILE, TAIL_LINES)
        return triage_text(cluster_warnings(lines),
                           turn_outliers(read_turns(PATHS.LOG_DIR / "turns.jsonl")),
                           examined=len(lines))

    def slow_turn_text(self) -> str:
        """"Why was that slow?": the last real turn on the ledger, split
        into silence / transcription / the answer."""
        from jarvis.logtriage import last_turn, slow_text
        return slow_text(last_turn(PATHS.LOG_DIR / "turns.jsonl"))

    # ------------------------------------------------------- turn ledger
    def _wire_turn_clock(self):
        """One "turn:" line per voice turn (jarvis/turnclock.py). Marks read
        the events' publisher-side clock: the bus queues for the Tk thread."""
        self.turns = TurnLedger(jsonl_path=PATHS.LOG_DIR / "turns.jsonl")
        self._tts_active = False              # rising-edge detection for "audio"
        self._turn_filler_pending = False     # the next speech is a filler line
        # "wake" is marked in _on_hotword itself (synchronous, on the hotword
        # thread, after its refusal checks): through the bus the mark arrived
        # after the recorder had already opened and read as refused.
        bus.subscribe(RecordingStarted, lambda ev: self.turns.mark("mic", at=ev.t))
        bus.subscribe(RecordingStopped, self._turn_on_stop)
        bus.subscribe(Transcribed, self._turn_on_transcribed)
        bus.subscribe(SpeakingState, self._turn_on_speaking)

    def _turn_on_stop(self, ev):
        if ev.reason == "abort":
            self.turns.abandon("abort")
            return
        if ev.dead_air_s is not None:
            self.turns.mark("speech_end", at=ev.t - ev.dead_air_s)
        self.turns.mark("stop", at=ev.t, stop=ev.endpoint or ev.reason)

    def _turn_on_transcribed(self, ev):
        # A reused speculative decode makes "stt" the time from the stop to
        # the publish, which is honest -- but only with the note saying why.
        notes = {"decode": "speculative"} if getattr(ev, "speculative", False) else {}
        self.turns.mark("stt", at=ev.t, **notes)
        if not ev.accepted:
            self.turns.abandon(f"rejected:{ev.reject_reason or 'confidence'}")
        elif not (ev.text or "").strip():
            self.turns.abandon("empty")

    def _turn_on_speaking(self, ev):
        # Rising edge only: SpeakingState(active=True) repeats at ~12 Hz for
        # amplitude, and a turn opened while a previous reply is still
        # playing must not be closed by those ticks. A filler line ("Looking
        # into it now, sir") is speech but not the answer.
        was, self._tts_active = self._tts_active, ev.active
        if not ev.active:
            # A filler queued INTO a playing burst never gets its own rising
            # edge; without this the flag outlived the burst and labelled
            # the next answer "filler".
            self._turn_filler_pending = False
            if was:
                self._after_speech()
            return
        if was:
            return
        if self._turn_filler_pending:
            self._turn_filler_pending = False
            self.turns.mark("filler", at=ev.t)
            return
        self.turns.mark("audio", at=ev.t)

    def _turn_after_result(self, result):
        """Close the ledger for a voice turn that will produce no audio."""
        status = result.status or ""
        if status.startswith("Ignored"):
            self.turns.abandon("ignored")
        elif getattr(result, "done", True) is False:
            return                            # the answer is still coming
        elif not (result.reply and result.speak and CONFIG.talkback):
            self.turns.abandon("unspoken")

    def _on_recording_stopped(self, ev):
        if ev.reason == "abort":
            return
        self._stop_event = ev               # _take_speculation / _nudge read it
        audio = self.recorder.last_audio
        if audio is None:
            # "No audio captured" and "Too short" both land here.
            self.turns.abandon("no_audio")
            self._nudge("no_audio")
            return
        self._audio_busy.set()
        threading.Thread(target=self._process_audio, args=(audio,),
                         daemon=True).start()

    def _process_audio(self, audio):
        stats = {}
        try:
            spec = self._take_speculation()
            if spec is not None:
                audio, stats = spec.audio, spec.stats
                rejected, result = spec.rejected, spec.result
                stop_t = getattr(self._stop_event, "t", spec.finished)
                log.info("speculative transcript reused: decode %.2fs, ready %.2fs %s the stop",
                         spec.finished - spec.started, abs(spec.finished - stop_t),
                         "before" if spec.finished <= stop_t else "after")
            else:
                audio, stats, rejected, result = self._decode_clip(audio)
            if rejected:
                bus.publish(Transcribed(
                    text="", accepted=False, reject_reason="speaker",
                    speaker_score=float(stats.get("best_score", 0.0))
                    if isinstance(stats, dict) else 0.0,
                    speculative=spec is not None))
                self._nudge("speaker")
                return
            bus.publish(Transcribed(
                text=result.text, confidence=result.confidence,
                accepted=result.accepted,
                reject_reason="" if result.accepted else "confidence",
                speculative=spec is not None))
            text = result.text.strip()
            if result.accepted and text:
                self._say_again_count = 0
                self._maybe_learn_voice(audio, stats)
                bus.publish(UserUtterance(text=text, source="voice"))
                self._dispatch(text, "voice", confidence=result.confidence)
            elif not result.accepted and text:
                # Garbled, not silent: say so and re-open the mic rather
                # than routing "by Agenda 4.2.6" or going quiet -- once.
                # Twice in a row is not the user mumbling, it is the room
                # (a television, music) reaching the follow-up mic, and
                # asking again re-opens that mic without end.
                self._say_again_count = getattr(self, "_say_again_count", 0) + 1
                if self._say_again_count > 1:
                    log.info("low confidence again (%.2f): %r -> staying quiet",
                             result.confidence, text)
                    self.turns.abandon("rejected:confidence")
                    self._nudge("confidence")
                else:
                    log.info("low confidence (%.2f): %r -> asking again",
                             result.confidence, text)
                    self._say(SAY_AGAIN_LINE)
                    self._followup_after_speech = True
            else:
                # Accepted but empty (the VAD pass found no words): the
                # ledger says "empty"; the user hears the nudge, not silence.
                self._nudge("empty")
        except Exception:
            log.exception("audio processing failed")
            bus.publish(Status(text="Transcription failed", kind="error"))
            self.turns.abandon("error")
        finally:
            # Must run on every path: a leaked flag makes every future wake
            # word a no-op, which looks exactly like a dead microphone.
            self._audio_busy.clear()

    def _nudge(self, reason: str):
        """The "did not catch that" policy: a cue instead of silence.

        A wake word promises a reply, and several outcomes used to end in
        nothing at all -- a clip the speaker gate dropped, an empty
        transcript, "No audio captured" / "Too short" (status text only), a
        second garbled clip -- which reads as a dead microphone. By cause:

        - no_audio / empty: nothing was heard, so a short spoken "Sir?" and
          a follow-up window, so the question can simply be repeated.
        - speaker / confidence: a non-verbal earcon only. A rejected clip
          may be a guest (a spoken line would answer them) or the user
          under a strict threshold (silence would be wrong), and a second
          garbled clip is usually the room reaching the mic, so re-opening
          it would loop.

        Wake-word turns only: a follow-up window that hears nothing is the
        normal case, and the mic button shows its own status. Rate-limited
        (listening.nudge_cooldown_s) so a noisy room cannot make him
        chatter; every outcome still lands in turns.jsonl for tuning.
        """
        ev = self._stop_event
        if ev is not None and getattr(ev, "followup", False):
            return
        if not self._turn_from_wake:
            return
        if not self._listening_opt("nudge", True):
            return
        if getattr(getattr(self, "tts", None), "busy", False) is True or \
                getattr(self, "_tts_active", False):
            return                      # he is talking already (a barge-in wake)
        now = time.monotonic()
        cooldown = float(self._listening_opt("nudge_cooldown_s", 30) or 0)
        if now - self._last_nudge_ts < cooldown:
            log.info("nudge (%s) suppressed: within %.0fs of the last", reason, cooldown)
            return
        self._last_nudge_ts = now
        spoken = reason in ("no_audio", "empty") and CONFIG.talkback
        log.info("nudge (%s): %s", reason, "spoken" if spoken else "earcon")
        if spoken:
            self._say(NUDGE_LINE)
            self._followup_after_speech = True
        else:
            threading.Thread(target=play_beep, args=("nudge",), daemon=True).start()

    # -------------------------------------------------------------- routing
    def _emit_result(self, result):
        """Publish a CommandResult's reply/status. Shared by _dispatch and the
        uncertain-prompt answer, so a YES there runs and SPEAKS exactly like a
        command that had been understood the first time."""
        if result.reply:
            bus.publish(JarvisReply(text=result.reply, speak=result.speak))
            if result.speak:
                if getattr(result, "ack", False):
                    # "Looking that up, sir." is speech, not the answer: the
                    # turn ledger records it as a filler and keeps waiting.
                    # It IS the filler, too: 4.5 s later "Checking right
                    # now, sir" followed it live. Disarm the thinking timer
                    # and leave the watchdog.
                    self._turn_filler_pending = True
                    t, self._turn_timer = getattr(self, "_turn_timer", None), None
                    if t is not None:
                        t.cancel()
                self._say(result.reply)
        if result.status:
            bus.publish(Status(text=result.status, kind="info"))
        return result

    _thinking_delay_s = THINKING_DELAY_S
    _turn_timeout_s = TURN_TIMEOUT_S

    def _dispatch(self, text, source, confidence=None):
        # Voice only: a typed answer is visible as it arrives, so being told to
        # wait is just noise.
        self._last_user_text, self._last_source = text, source
        # A barge's mute must not outlive the barged turn: a typed reply
        # used to arrive silently after a fruitless barge-in capture.
        self._stream_muted = False
        # Every dispatch supersedes older done=False worker replies
        # (_async_reply checks this before speaking a late answer).
        self._dispatch_gen = getattr(self, "_dispatch_gen", 0) + 1
        if source != "cli":
            self._active_turn_id = ""
        if source == "voice":
            self._turn_start()
            self.turns.mark("handle")
        # The Whisper avg_logprob travels only when there is one: typed
        # text has none, and a stand-in commander need not take the keyword.
        kw = {} if confidence is None else {"confidence": confidence}
        try:
            result = self._emit_result(self.commander.handle(text, source, **kw))
            corrected = getattr(result, "corrected", None)
            if corrected:
                self._last_user_text = corrected
            if source == "voice":
                self._turn_after_result(result)
            self._after_dispatch(text, source, result)
        except Exception:
            if source == "voice":
                self._turn_finished()
            raise
        if getattr(result, "done", True) is False:
            self._async_turn = (self._dispatch_gen, source)
        else:
            self._async_turn = None
        # done=False means the answer is still coming on a worker thread (the
        # commander routes local chat that way). Only the voice path opened a
        # turn: closing one here for cli/typed/discord clobbered a live voice
        # turn's watchdog and busy guard from the socket thread.
        if source == "voice" and getattr(result, "done", True) is not False:
            self._turn_finished()
        if getattr(self, "_reopen_mic", False):
            self._reopen_mic = False
            tts = getattr(self, "tts", None)
            if getattr(tts, "is_speaking", False) or getattr(tts, "pending", 0):
                self._followup_after_speech = True   # _after_speech opens it
            else:
                self._start_followup()
        return result

    def _turn_start(self):
        """Open a turn: arm the slow-answer filler and a watchdog."""
        self._turn_cancel_timers()
        self._stream_muted = False
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
        self.turns.abandon("timeout")

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
        if not self._turn_busy.is_set():
            return          # the answer landed while this timer was firing
        with self._uncertain_lock:
            asking = bool(self._pending_uncertain)
        if asking:
            # "Was that for me?" leaves the turn open (done=False) waiting on
            # a yes/no, so this timer kept running and answered the user's
            # question with "Looking into it now, sir." -- for an utterance
            # about to be discarded. Worse, _ask_uncertain is recording the
            # spoken reply by then, and the arbiter is a depth counter rather
            # than a mutex, so Jarvis talked into his own yes/no window.
            log.info("thinking line held: waiting on an answer, not on work")
            return
        try:
            line = THINKING_LINES[self._thinking_i % len(THINKING_LINES)]
            self._thinking_i += 1
            self._turn_filler_pending = True      # the ledger must not call this the answer
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
        # "Was that for me?" is speech but not an answer: close the ledger
        # before the ask thread can publish SpeakingState for it.
        self.turns.abandon("uncertain")
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

    def _claim_uncertain(self, yes: bool) -> bool:
        """Commander hook: a spoken "that was for you" / "that wasn't for
        you" settles the open card. The commander logs the label and routes
        the utterance itself; this only closes the cards and tells the UI.
        Returns whether one was waiting."""
        with self._uncertain_lock:
            stale = list(self._pending_uncertain)
            self._pending_uncertain.clear()
        for rid in stale:
            bus.publish(UncertainResolved(request_id=rid, yes=yes, source="voice"))
        return bool(stale)

    def uncertain_answer(self, request_id: str, yes: bool, source: str = "ui"):
        """Answer the open prompt. First answer wins -- the card and the
        spoken window race each other, and resolve_uncertain would otherwise
        route the same utterance twice."""
        with self._uncertain_lock:
            text = self._pending_uncertain.pop(request_id, None)
        if text is None:
            return None
        # Log the source: a prompt was once answered "no" with no click and no
        # spoken reply, and nothing recorded who did it.
        log.info("uncertain %s answered %s by %s", request_id,
                 "yes" if yes else "no", source)
        bus.publish(UncertainResolved(request_id=request_id, yes=yes,
                                      source=source))
        # The prompt left the turn open (done=False) so this answer could
        # arrive. Re-arm BEFORE routing, exactly as _dispatch does: the brain's
        # busy branch (and a fast failure) invokes _on_brain_tags ->
        # _turn_finished synchronously inside resolve_uncertain, and a
        # _turn_start() placed after it would re-open a turn nothing closes --
        # every wake word refused until the 60 s watchdog. A YES that routes
        # to the brain returns done=False and _on_brain_tags closes it; a NO
        # or a synchronous command is closed right here.
        self._turn_start()
        try:
            result = self._emit_result(self.commander.resolve_uncertain(text, yes))
        except Exception:
            self._turn_finished()
            raise
        if getattr(result, "done", True) is not False:
            self._turn_finished()
        return result

    def dispatch_text(self, text, source="typed", quiet=False, turn_id=""):
        """MainWindow calls this on a worker thread for typed input; the
        Discord channel with source='discord'; the command socket
        (jarvis/cmdsock.py) with source='cli', on the client's thread.
        `quiet` (cli only) answers in text and keeps the soundbar silent;
        `turn_id` (cli) stamps this turn's replies for the socket stream."""
        text = (text or "").strip()
        if not text:
            return None
        # Barge-in: a typed command while Jarvis is talking cuts him off
        # (the films' JARVIS never talks over Tony), then gets answered.
        # NOT for cli: an unattended script or cron call must not cut a
        # reply he is speaking to someone in the room.
        if source != "cli":
            self.interrupt_speech()
        if source == "typed":
            self.history.add(text)
        self._quiet_turn = bool(quiet) and source == "cli"
        self._active_turn_id = turn_id or ""
        result = None
        try:
            result = self._dispatch(text, source)
            return result
        finally:
            # The mute is per turn. A sync answer ends it here; a done=False
            # turn keeps it until _async_reply / the brain tags deliver.
            if result is None or getattr(result, "done", True) is not False:
                self._quiet_turn = False

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
                          ("cmdsock", getattr(self, "cmdsock", None)),
                          ("discord", self.discord)):
            if obj is None:
                continue
            try:
                obj.start()                # Timekeeper.start() catches up first
            except Exception:
                log.exception("assistant: %s failed to start", name)
                bus.publish(Status(text=f"{name} failed to start", kind="warn"))
        focus = getattr(self, "focus", None)
        if focus is not None:
            try:
                # After Timekeeper.start(): its catch-up has already fired or
                # missed whatever came due while the app was down.
                focus.reconcile()
            except Exception:
                log.exception("focus session reconcile failed")
        try:
            # The nightly self-review: files yesterday's digest under
            # MEMORY_DIR/reviews and posts the table to Discord when that
            # channel is configured (dayreview.py).
            self.dayreviewer = dayreview_mod.DayReviewer(
                PATHS.LOG_DIR / "jarvis.log", PATHS.LOG_DIR / "turns.jsonl",
                PATHS.REVIEWS_DIR, on_filed=self._on_review_filed)
            self.dayreviewer.start()
        except Exception:
            log.exception("day reviewer failed to start")
        cal = getattr(self.services, "calendar", None)
        if cal is not None:
            try:
                cal.start()
            except Exception:
                log.exception("calendar refresh start failed")
        wd = getattr(self.services, "health_watchdog", None)
        if wd is not None:
            try:
                wd.start()
            except Exception:
                log.exception("health watchdog failed to start")
        sampler = getattr(self.services, "activity_sampler", None)
        if sampler is not None:
            try:
                # Retention is not optional: exchanges and tool calls are
                # journaled regardless of the sampler switch, so old day
                # files must be pruned even with journal.enabled false.
                sampler.prune()
            except Exception:
                log.exception("journal prune failed")
            if self.assistant.get("journal.enabled", True):
                try:
                    sampler.start()
                except Exception:
                    log.exception("activity sampler failed to start")
        for name, obj in (("presence", self.presence), ("quiet", self.quiet)):
            if obj is None:
                continue
            try:
                obj.start()
            except Exception:
                log.exception("%s failed to start", name)
        try:
            from jarvis.headsup import MeetingHeadsUp
            lead = int(self.assistant.get("calendar.heads_up_min", 10) or 10)
            self.headsup = MeetingHeadsUp(lambda: getattr(self.services, "calendar", None),
                                          self.timekeeper, lead_min=lead,
                                          state_path=PATHS.MEMORY_DIR / "headsup_state.json")
            if self.timekeeper is not None:
                self.headsup.start()
        except Exception:
            log.exception("meeting heads-up failed to start")
        try:
            from jarvis.deadlines import DeadlineHeadsUp
            hours = self.assistant.get("canvas.heads_up_hours", 3)
            self.deadlines = DeadlineHeadsUp(
                self.assistant, self.timekeeper, lead_hours=hours,
                state_path=PATHS.MEMORY_DIR / "deadlines_state.json",
                get_calendar=lambda: getattr(self.services, "calendar", None))
            if self.timekeeper is not None:
                self.deadlines.start()
        except Exception:
            log.exception("deadline heads-up failed to start")
        if residency:
            try:
                # boot warm-up on its own daemon thread, then every 5 min
                brain_mod.start_residency()
            except Exception:
                log.exception("ollama residency thread failed to start")
            # Same gate (residency=False is the tests' "no Ollama"): migrate
            # facts.json into the semantic index and load the embedder off
            # the turn path -- a cold nomic-embed-text is ~7.5 s.
            warm = getattr(self.memory, "warm_index", None)
            if warm is not None:
                threading.Thread(target=warm, daemon=True, name="memory-warm").start()
        if self.assistant.get("autostart.enabled", False):
            try:
                from jarvis import autostart
                autostart.install(path=PATHS.AUTOSTART_DESKTOP)
            except Exception:
                log.exception("autostart install failed")

    def _load_tts(self):
        try:
            self.tts.load()
        except Exception:
            log.exception("tts load failed")

    def _load_models(self):
        # start_preload() may still be importing torch/whisper; importing the
        # same modules from two threads is what the preload exists to avoid.
        _PRELOAD_DONE.wait(120)
        # XTTS is the only model sitting between a finished reply and the
        # first audible word, so it must not queue behind the ones the user
        # never waits on -- it used to load fourth, after a 5.5 s ollama
        # warmup, and was still loading when the user spoke (11.07 s of
        # silence after the reply was ready, 2026-08-28). Running it
        # alongside the rest is safe: TTS.load() serialises against the
        # speak path's own call rather than racing it.
        tts_load = threading.Thread(target=self._load_tts, daemon=True,
                                    name="tts-load")
        tts_load.start()
        self.brain.warmup()
        bus.publish(Status(text="Loading speech model…", kind="busy"))
        try:
            backend = self.transcriber.load()
            bus.publish(ModelInfo(text=f"{CONFIG.model} · {backend}"))
            bus.publish(Status(text="Ready", kind="ok"))
        except Exception:
            log.exception("whisper load failed")
            bus.publish(Status(text="Speech model failed to load", kind="error"))
        # Warm the speaker model here so the first wake word does not pay the
        # ~1.4 s CUDA cold start, and so the audio path never has to load it
        # inline. Verification loads lazily too, but that is the fallback.
        try:
            if self.speaker.enrolled:
                self.speaker.load_model()
        except Exception:
            log.exception("speaker model load failed")
        self._install_endpointer()
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
        # Render the canned lines into the speech cache while idle. The
        # prewarm needs the model, so this is where we finally wait for it.
        tts_load.join(120)
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
                          ("cmdsock", getattr(self, "cmdsock", None)),
                          ("timekeeper", self.timekeeper), ("calendar", cal),
                          ("claude", self.claude),
                          ("health_watchdog", getattr(self.services, "health_watchdog", None)),
                          ("activity_sampler", getattr(self.services, "activity_sampler", None)),
                          ("headsup", getattr(self, "headsup", None)),
                          ("deadlines", getattr(self, "deadlines", None)),
                          ("focus", getattr(self, "focus", None)),
                          ("presence", getattr(self, "presence", None)),
                          ("quiet", getattr(self, "quiet", None)),
                          ("dayreviewer", getattr(self, "dayreviewer", None))):
            if obj is None:
                continue
            fn = getattr(obj, "stop", None) or getattr(obj, "close", None)
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                log.exception("assistant: %s failed to stop", name)
        try:
            # The banner gate holds a reference to this policy; a stopped
            # app (or a test's teardown) must not keep gating banners.
            from jarvis.channels import notify
            notify.set_quiet_gate(None)
        except Exception:
            log.debug("quiet gate not cleared", exc_info=True)
        for obj in (self.notes, getattr(self.commander, "_flashcards", None)):
            fn = getattr(obj, "close", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    log.exception("store close failed")

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
        # A sidecar this process spawned stays resident on purpose (it is
        # what makes the next launch warm) UNLESS jarvis-f5.service has taken
        # the socket over, in which case ours is a few GB answering nobody.
        try:
            release = getattr(self.tts, "release_f5_sidecar", None)
            if callable(release):
                release()
        except Exception:
            log.exception("f5 sidecar release failed")
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
