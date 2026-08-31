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
    ArcChanged,
    BoardCommand,
    BriefingReady,
    HotwordDetected,
    PowerUp,
    ClaudeProgress,
    ClaudeTaskState,
    DeskState,
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

from jarvis import address as address_mod
from jarvis import board as board_mod
from jarvis import brain as brain_mod
from jarvis import debrief as debrief_mod
from jarvis import arrival as arrival_mod
from jarvis import desktop as desktop_mod
from jarvis import earcons
from jarvis import selfstate, speak_queue, standup, voice_check
from jarvis import leavetime as leavetime_mod
from jarvis import vocab as vocab_mod
from jarvis.assistant_config import AssistantConfig
from jarvis.turnclock import TurnLedger
from jarvis import dayreview as dayreview_mod
from jarvis import garden as garden_mod
from jarvis.dialogue import SESSION_WINDOW_S
from jarvis.faults import FaultBoard, FaultLog
from jarvis.brain import JarvisBrain
from jarvis.commander import (COURTESY_BY_REGISTER, COURTESY_REPLIES,
                              DESTRUCTIVE_TTL_S, REGISTER_LINES,
                              CommandResult, Commander, parse_yes_no,
                              strip_jarvis_prefix)
from jarvis.context import ContextEngine
from jarvis.history import TypedHistory
from jarvis.hotword import Hotword
from jarvis.jarvis_agent import JarvisAgent
from jarvis.memory import JarvisMemory
from jarvis.reader import CONTINUE_PROMPT, ReadAloud
from jarvis.tools.briefing import OFFER_TTL_S
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
    # Last, and usually free: oracle.make_tools returns NOTHING unless the
    # Oracle box is switched on AND has a key, so the default install pays
    # no schema for it. The registry is already over its 11-tool budget and
    # every schema is prefill on every model turn.
    "jarvis.tools.oracle",
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

# The Board and the ambient slab (jarvis/board.py, jarvis/ui/*). Canvas is
# the only network source on the Board and Spotify the only one on the
# slab, so both are gated here rather than in the surfaces that draw them.
CANVAS_TTL_S = 300.0          # the Board's Canvas half, cached 5 minutes
BOARD_SESSIONS = 6            # Claude sessions read off disk per poll
ROOM_SPOTIFY_ACTIVE_S = 10.0  # playback poll while something IS playing
ROOM_SPOTIFY_IDLE_S = 60.0    # …and once it has gone quiet
ROOM_GPU_TTL_S = 30.0         # the slab's GPU reading when nobody hands one in
SAY_AGAIN_LINE = "Say that again, sir?"
# The "did not catch that" cue (JarvisApp._nudge): a wake-word turn that
# captured nothing usable gets this instead of silence.
NUDGE_LINE = "Sir?"
GUEST_LINE = "I only answer to {name}, sir."
TURN_TIMEOUT_S = 60.0           # watchdog: a lost reply must not wedge the turn

# The sources that arrive from somewhere other than this desk: a shell /
# SSH / cron client and a clip sent over the command socket
# (jarvis/cmdsock.py, jarvis/intercom.py), and the phone client's page
# (jarvis/webapp.py). They share three rules that the voice, typed and
# Discord sources do not -- their replies are stamped with a turn_id for
# the requesting stream, `quiet` mutes THAT turn's speech, and none of them
# cuts a reply Jarvis is already speaking to someone in the room. The last
# two are why "phone" belongs here: a question asked from bed must not make
# the soundbar answer the house, and must not talk over the room.
SOCKET_SOURCES = ("cli", "intercom", "phone")   # intercom.SOURCE / webapp.SOURCE,
                                                # spelled out so the wiring hub
                                                # need not import either module

_YES_WORDS = frozenset({"yes", "y", "yeah", "yep", "yup", "aye", "allow",
                        "allowed", "approve", "approved", "ok", "okay", "sure",
                        "affirmative", "permit", "proceed", "go", "ahead", "do", "it"})
_NO_WORDS = frozenset({"no", "n", "nope", "nah", "deny", "denied", "decline",
                       "declined", "don't", "dont", "negative", "reject", "refuse"})

DISCORD_ACTIVE_S = 600.0        # a Discord exchange stays "active" this long
# Two probes answer "he's back" -- the phone on the Wi-Fi (jarvis/presence.py)
# and the keyboard (jarvis/deskpresence.py). They cross their thresholds
# minutes apart on the same walk through the door, and that is ONE return:
# both greetings go through _greet_return, which speaks at most once per
# damper. release() already drains atomically, so only the LINE could double.
GREET_DAMPER_S = 600.0


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
    # Class-level defaults for the state __init__ sets, so a test that
    # builds a partial app (tests/test_destructive_readback.py uses
    # object.__new__ and fills in only the commander) still has them. The
    # two features that hang off a turn are optional by construction:
    # without a debrief watch or an aside engine they are simply absent.
    _pending_debrief = None
    aside = None
    debrief = None

    def __init__(self):
        # ---- assistant config first: everything below reads it ------------
        self.assistant = AssistantConfig.load()
        # How often "sir" lands on the ear (jarvis/address.py). Installed as
        # module state, the way earcons.set_config is, because the two join
        # sites that need it -- quiet.digest and speak_queue's watcher -- are
        # module functions with no config of their own. A config edit needs a
        # restart: AssistantConfig.reload_if_changed() has no callers.
        address_mod.set_config(self.assistant)
        self._discord_active_until = 0.0
        self._discord_last_post = ("", 0.0)
        self._last_milestone: dict[str, str] = {}
        self._assistant_started = False
        self._quitting = False
        # The earcon lexicon reads sound.earcons / sound.volume / cooldown
        # from here. Installed the way channels.notify.set_quiet_gate is,
        # because recorder.play_beep() is a module function with no config
        # of its own and the tones must not need one to be auditioned.
        earcons.set_config(self.assistant)

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
        # The register he last asked for, before the first static_system()
        # call: set here it costs nothing, set later it costs a reprocess.
        # assistant.json is the source of truth (memory's preferences.json
        # is the record of what he asked for), same rule as the briefing.
        try:
            brain_mod.set_register(self.assistant.get("persona.register",
                                                      brain_mod.DEFAULT_REGISTER))
        except Exception:
            log.exception("persona.register could not be applied")
        self.agent = JarvisAgent()          # retained V1 tools (see spec note)

        # ---- ambient: quiet hours / DND and presence ----------------------
        # Both read services lazily (calendar, presence) because services is
        # built further down; both are started in start_assistant.
        self.presence = self._construct("presence", self._make_presence)
        # The desk probe needs no configuration at all (GNOME's idle
        # monitor over the session bus), so it is what actually answers
        # "is he there" on this box: presence.phone_ip / phone_mac are
        # unset, so the Wi-Fi sentinel has never fired in the room.
        self.desk = self._construct("desk", self._make_desk)
        self.quiet = self._construct("quiet", self._make_quiet)
        # The arc names the hour and publishes ArcChanged; it is a state
        # source with no side effects, so it is safe to build before the
        # services exist. Room tone is one of its consumers and is OFF
        # unless he has said otherwise.
        self.arc = self._construct("arc", self._make_arc)
        self.roomtone = self._construct("roomtone", self._make_roomtone)

        # ---- the room: display light, scenes and the audio mixer ----------
        # The light and the scenes are inert until spoken to; the Mixer runs
        # itself off the bus (start_assistant). Any state a previous run was
        # holding is restored there too, so a crash at 0.55 brightness or a
        # ducked soundbar heals at the next start.
        self.room_light = self._construct("room light", self._make_room_light)
        self.scenes = self._construct("scenes", self._make_scenes)
        self.mixer = self._construct("mixer", self._make_mixer)
        # The fault lane (jarvis/faults.py): the live fault, held past the
        # 4-6 s Status chip so "what's wrong" can still answer and the
        # board's FAULT row stays lit. Subscribes to FaultRaised itself.
        self.faults = FaultBoard(subscribe=True)

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
        self.webapp = self._construct("webapp", self._make_webapp)
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
        # Same shape as focus: it reaches Spotify and the quiet policy
        # through the services namespace, so it is built after them.
        self.winddown = self._construct("winddown", self._make_winddown)
        self.services.winddown = self.winddown
        # The scene runner reaches quiet hours and Spotify through the same
        # namespace focus does, and Spotify only lands on it once the tools
        # are built (spotify.make_tools parks it there).
        if self.scenes is not None:
            self.scenes.services = self.services
        self.commander = Commander(self.services)
        # Without this hook the commander falls back to a bare warn Status --
        # a 4 s toast with no way to answer it, after which the utterance is
        # dropped and resolve_uncertain (and the classifier feedback it feeds)
        # is never reached from the running app at all.
        self.commander.on_uncertain = self._on_uncertain
        self.commander.claim_uncertain = self._claim_uncertain
        self._pending_uncertain: dict = {}      # request_id -> utterance
        # The open debrief question (jarvis/debrief.py), modelled on
        # _pending_uncertain: a context dict the NEXT transcript is filed
        # against instead of being routed to commander.handle. Cleared on
        # a wake word, and on anything that is plainly a command.
        self._pending_debrief = None
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
        self._wire_roomtone()
        bus.subscribe(DeskState, self._on_desk)
        self._last_greeted = 0.0        # GREET_DAMPER_S, shared by both probes

        if CONFIG.target_name:
            self.desktop.restore_target(CONFIG.target_name)

    def _wire_roomtone(self):
        """The bed must never be inside a capture or under a reply.

        Mic safety is the room tone's design constraint, not a footnote:
        there is no AEC on this box and the Snowball shares the room with
        the speaker, so a bed that is merely QUIET during a capture still
        reaches Whisper, the Silero endpointer and the ECAPA gate. These
        subscriptions assert a HARD mute -- the stream is killed, not faded
        -- for the whole of every capture and every spoken reply, and the
        wake word mutes it before the mic even opens.
        """
        tone = getattr(self, "roomtone", None)
        if tone is None:
            return
        bus.subscribe(HotwordDetected, tone.on_wake)
        bus.subscribe(RecordingStarted, tone.on_mic_open)
        bus.subscribe(RecordingStopped, tone.on_mic_close)
        bus.subscribe(SpeakingState, tone.on_speaking)
        bus.subscribe(ArcChanged, tone.on_arc)

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

    def _make_winddown(self):
        mod = _import_optional("jarvis.winddown")
        if mod is None:
            return None
        return mod.WindDown(self.services,
                            state_path=PATHS.MEMORY_DIR / "winddown.json")

    def _make_presence(self):
        mod = _import_optional("jarvis.presence")
        return None if mod is None else mod.PresenceSentinel(self.assistant)

    def _make_room_light(self):
        mod = _import_optional("jarvis.room")
        if mod is None:
            return None
        # The bedtime wind-down (jarvis/winddown.py) dims the SAME xrandr
        # output through its own state file, so the automatic heals at boot
        # and quit must ask before undoing it. Late-bound on purpose:
        # self.winddown is built further down (_construct order), and the
        # getattr default makes the early state safe.
        return mod.RoomLight(
            state_path=PATHS.MEMORY_DIR / "room_state.json",
            held_by=lambda: bool(
                getattr(getattr(self, "winddown", None), "holding", False)))

    def _make_scenes(self):
        mod = _import_optional("jarvis.scenes")
        if mod is None or self.room_light is None:
            return None
        # services is built further down; the runner reads it lazily, so the
        # attribute is filled in _build_services like focus does.
        return mod.Scenes(None, light=self.room_light,
                          state_path=PATHS.MEMORY_DIR / "scene_state.json")

    def _make_mixer(self):
        mod = _import_optional("jarvis.mixer")
        if mod is None:
            return None
        return mod.RoomMixer(cfg=self.assistant,
                             state_path=PATHS.MEMORY_DIR / "mixer_state.json")
    def _make_desk(self):
        mod = _import_optional("jarvis.deskpresence")
        return None if mod is None else mod.DeskSentinel(self.assistant)

    def _is_home(self) -> bool:
        """The ONE away signal the quiet policy reads (its ``is_home``
        seam, gated by quiet.hold_when_away). Both probes fail OPEN, so
        this is False only once one of them has ESTABLISHED he is gone: an
        unconfigured phone probe and a session with no Mutter interface
        both leave it True, exactly as before this existed."""
        for probe in (getattr(getattr(self, "presence", None), "is_home", None),
                      getattr(getattr(self, "desk", None), "is_at_desk", None)):
            if not callable(probe):
                continue
            try:
                if not probe():
                    return False
            except Exception:  # noqa: BLE001 - a broken probe must not mute him
                log.debug("presence: probe failed", exc_info=True)
        return True

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
            # Both away probes feed the ONE existing hold_when_away seam
            # (see _is_home); a second suppression path would be invisible
            # to every consumer that already reads this one.
            is_home=(self._is_home if (presence is not None or self.desk is not None)
                     else None),
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

    def _make_arc(self):
        mod = _import_optional("jarvis.arc")
        if mod is None:
            return None
        # Late-bound lambdas: focus is built after the services, and the arc
        # must not capture a None that never becomes a session.
        return mod.Arc(self.assistant, quiet=lambda: getattr(self, "quiet", None),
                       presence=lambda: getattr(self, "presence", None),
                       focus=lambda: getattr(self, "focus", None),
                       state_path=PATHS.MEMORY_DIR / mod.STATE_NAME,
                       tick_s=float(self.assistant.get("arc.tick_s", mod.TICK_S) or mod.TICK_S))

    def _make_roomtone(self):
        mod = _import_optional("jarvis.roomtone")
        if mod is None:
            return None
        return mod.RoomTone(self.assistant, arc=lambda: getattr(self, "arc", None),
                            quiet=lambda: getattr(self, "quiet", None),
                            presence=lambda: getattr(self, "presence", None),
                            turns=lambda: getattr(self, "turns", None))

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

    def _make_webapp(self):
        # The phone client. Constructing it binds nothing: PhoneServer.start()
        # is where phone.enabled is read and where a non-private bind address
        # is refused, so a box with the feature off never opens a port.
        mod = _import_optional("jarvis.webapp")
        return None if mod is None else mod.PhoneServer(self)

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
    def _thin_address(self, fragments):
        """The fragments of one spoken burst as they should be SPOKEN: the
        first "sir" survives, later trailing ones go (jarvis/address.py).

        Takes a LIST, never a finished string. That is the whole lesson of
        the first attempt: on a rendered line Jarvis's own words and an
        interpolated mail subject or track title are indistinguishable, and a
        \bsir\b match eats "Sir Isaac Newton". Guarded end to end -- a
        failure here speaks the lines as written, because a doubled courtesy
        is a far smaller bug than a mangled sentence."""
        try:
            return address_mod.thin_fragments(fragments)
        except Exception:
            log.exception("address thinning failed; speaking as written")
            return list(fragments)

    def _say(self, text, proactive=False, kind="message"):
        """The one door to TTS. ``proactive=True`` marks a line Jarvis
        decided to say on his own (watchdog, reminder, heads-up, narrator);
        quiet hours / DND / a running meeting / an empty room hold those
        for the catch-up digest (jarvis/quiet.py). Answers, alarms and
        approval questions pass the default False and are never held.

        Nothing is rewritten on the way through this door. Address thinning
        happens at the JOIN sites, where the fragments of a burst are still
        separate authored lines (jarvis/address.py)."""
        # A quiet socket turn (python -m jarvis.ask -q, or an intercom clip
        # sent without speak) is answered in text only. Per turn, not a
        # global toggle: a voice turn that lands while the socket answer is
        # still coming resets _last_source.
        if getattr(self, "_quiet_turn", False) and \
                getattr(self, "_last_source", "") in SOCKET_SOURCES:
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
        # Nothing is rewritten here. A line arriving at this door is already
        # rendered -- third-party text and all -- and is spoken as written.
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
        if getattr(self, "_last_source", "") in SOCKET_SOURCES:
            self._quiet_turn = False    # this socket turn's answer is delivered
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
        # Every register's courtesy variants, and the register-change
        # acknowledgements: "formal mode" must answer instantly, because
        # the one prefix reprocess it triggers is already running behind it.
        phrases += [p for reg in COURTESY_BY_REGISTER.values()
                    for lines in reg.values() for p in lines]
        phrases += list(REGISTER_LINES.values())
        # The numberless self-state clauses ("how are you?"): a state
        # sentence carrying a figure is an unavoidable cache miss, these
        # are not (jarvis/selfstate.py).
        phrases += list(selfstate.PREWARM_LINES)
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
                              ("jarvis.focus", "PERSONA_LINES"),
                              ("jarvis.mathspeak", "PERSONA_LINES")):
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
            # register ("formal mode" / "banter up"): the ONE production
            # caller of reset_static_prompt, re-warmed off the audio path.
            set_register=lambda name: app.brain.set_register(name),
            register=lambda: app.brain.register(),
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
            # The one flashcard deck: the commander files cards into it, the
            # exam-week briefing counts what is due on it, and the nightly
            # pass (jarvis/studycards.py) fills it from his lecture notes.
            # Filled in start_assistant so building the tools in a test does
            # not open a SQLite file.
            flashcards=None,
            news_cache_path=PATHS.CACHE_DIR / "news.json",
            diagnostics=self.diagnostics_text,
            # the one self-state sheet the courtesy and the readout share
            self_state=self.self_state,
            log_triage=self.log_triage_text,
            slow_turn=self.slow_turn_text,
            # "how did yesterday go": the day review, spoken (dayreview.py)
            dayreview=self.day_review_text,
            # "how was my week": the cross-day trends (dayreview.week_*),
            # and the memory garden's two answers (jarvis/garden.py).
            # Resolved through the app so a garden that has not started
            # yet answers honestly rather than being absent from Tier 1.
            week_review=self.week_review_text,
            garden_report=self.garden_report_text,
            garden_undo=self.garden_undo_text,
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
            quiet=self.quiet, presence=self.presence, desk=self.desk,
            # "what's wrong": the live fault, else the log triage below
            faults=self.faults,
            # Seconds since the last keyboard / mouse event, or None when
            # this session has no idle signal. A callable, not a number.
            desk_idle_s=(self.desk.idle_s if self.desk is not None
                         else (lambda: None)),
            # The learned per-building walks; filled in start_assistant.
            leavetime=None,
            # Reasoned dissent + the Aside (jarvis/objections.py,
            # jarvis/aside.py). Both are parked here rather than reached
            # for by import so a stand-in services namespace in the tests
            # simply has neither and both features go quiet. `deadlines`
            # is the DeadlineHeadsUp itself, read ONLY through its
            # snapshot() -- never its fetch.
            aside=None, deadlines=None,
            # The sink sentinel (jarvis/soundbar.py), filled in
            # start_assistant. Answers "where's my voice coming out" from
            # its LAST tick -- never a pactl call on the reply path.
            soundbar=None,
            journal_objection=self._journal_objection,
            # the arc (a state source; nothing calls it, they subscribe) and
            # the room tone the "room tone on/off" command switches
            arc=self.arc, roomtone=self.roomtone,
            # "bring up the board" / "focus on the sessions": show and hide
            # publish a BoardCommand for the window to act on (the commander
            # runs on worker threads and must never touch Tk); read() answers
            # with the panel's one-line spoken state, or "" for a name that
            # is not a panel at all.
            board=SimpleNamespace(show=self._board_show, hide=self._board_hide,
                                  read=self._board_read),
            # the room: display light/level (jarvis/room.py) and the scenes
            # that compose it with music and quiet hours (jarvis/scenes.py)
            room_light=self.room_light, scenes=self.scenes, mixer=self.mixer,
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
                        # THE JOIN: the briefing's wake-alarm offer is spoken
                        # straight after the reply, so the two are one burst
                        # ("...that's your day, sir. Shall I wake you at
                        # 7:00, sir?"). The reply is the first fragment and
                        # is never rewritten (jarvis/address.py).
                        self._say(self._thin_address([content, offer])[-1])
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
        if getattr(self, "_last_source", "") in SOCKET_SOURCES:
            self._quiet_turn = False    # this socket turn's answer is delivered
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

    def _announce(self, title, text):
        """One unprompted watcher line (jarvis/watchers.py). Both doors, the
        way _on_claude_progress does it: _say(proactive=True) so quiet hours,
        DND and a running lecture hold it for the catch-up digest, and the
        alerts hub so it still reaches Discord when he is out of the room."""
        self._say(text, proactive=True)
        self._alert("milestone", title, text)

    def _on_claude_progress(self, ev):
        if ev.milestone and ev.line:
            self._last_milestone[ev.task_id] = ev.line
            # Proactive: quiet hours / DND hold the narration for the digest
            self._say(ev.line, proactive=True)
            self._alert("milestone", f"Claude · {ev.project}", ev.line)

    def _on_claude_state(self, ev):
        # The Board's sessions panel wants the LIVE states, and this is the
        # only place they pass through the app. A finished task leaves the
        # map so the panel falls back to the session row on disk.
        if ev.project:
            if ev.state in ("queued", "running", "waiting"):
                self._board_tasks[ev.project] = ev.state
            else:
                self._board_tasks.pop(ev.project, None)
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

    def _journal_objection(self, source, reason, row):
        """An objection, in the episodic record. The nightly self-review
        reads the LOG rather than this (dayreview.py parses jarvis.log and
        turns.jsonl and has never touched the journal); this is what makes
        "why did you argue with me about that alarm" answerable later."""
        journal = getattr(self.context, "journal_tool", None)
        if journal is None:
            return
        try:
            journal("objection", {"source": source, "row": row}, True, reason)
        except Exception:
            log.exception("journal objection failed")

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
    def _quiet_reason(self) -> str:
        quiet = getattr(self, "quiet", None)
        if quiet is None:
            return ""
        try:
            return str(quiet.reason() or "")
        except Exception:
            log.exception("presence: quiet gate failed")
            return ""

    def _on_desk(self, ev):
        """He sat down at (or walked away from) the keyboard.

        Routed through the SAME greeter as the phone probe: with both
        configured they would otherwise each say "Welcome back, sir" on the
        same return -- release() drains atomically so the digest cannot
        double, but the greeting line would. The damper in _greet_return is
        what actually prevents it. The board's standby is a separate
        consumer (the main window subscribes to DeskState itself)."""
        bus.publish(Status(text="At the desk" if ev.at_desk else "Desk empty",
                           kind="info"))
        if not (ev.at_desk and ev.returned):
            return
        # When the Wi-Fi probe is configured it owns the greeting: it knows
        # he left the building, which is the return worth marking. Sitting
        # back down after a coffee is not.
        presence = getattr(self, "presence", None)
        if presence is not None and getattr(presence, "configured", False):
            return
        self._greet_return("desk")

    def _greet_return(self, source: str) -> None:
        """One arrival cue per return, shared by the phone and desk probes.

        REGRESSION SITE. The arrival rework moved the choreography into
        jarvis/arrival.py and deleted this method, but _on_desk kept
        calling it: every desk return raised AttributeError inside the bus
        subscriber (swallowed there, so the desk greeting was simply dead),
        and _on_presence greeted without consulting the damper at all, so
        GREET_DAMPER_S and _last_greeted were both orphaned. Both probes
        come through here now: two sentinels crossing their thresholds
        minutes apart on the same walk through the door is ONE return, and
        the damper is what makes that true. The ordered steps still run
        through arrival_mod.run -- the panel/earcon/greeting/catch-up order
        is the feature, not the line.
        """
        now = time.monotonic()
        last = getattr(self, "_last_greeted", 0.0)
        if last and now - last < GREET_DAMPER_S:
            log.info("presence: %s return within the damper; not greeting again",
                     source)
            return
        steps = arrival_mod.arrival_plan(
            returned=True, home=True, quiet_reason=self._quiet_reason(),
            cue=bool(self.assistant.get("presence.arrival_cue", True)))
        if not steps:
            bus.publish(Status(text="Home", kind="info"))
            return
        # Stamped only for a plan that actually SPEAKS. A quiet-hours plan
        # is panel-only (arrival.arrival_plan), and letting that arm the
        # damper would swallow the real greeting when the policy lifts
        # minutes later. Stamped BEFORE run(): it speaks, and the other
        # probe's event can land on another thread while it is still
        # talking.
        if "greeting" in steps:
            self._last_greeted = now
        done = arrival_mod.run(steps, self._arrival_actions())
        log.info("arrival (%s): %s", source, " -> ".join(done) or "(nothing)")

    def _arrival_actions(self) -> dict:
        """The callables behind jarvis/arrival.ARRIVAL_STEPS.

        A step returning False did nothing (there was no backlog), and
        arrival.run() records only what actually happened -- which is what
        the tests assert on.
        """
        from jarvis.presence import WELCOME_LINE

        def panel():
            # The panel coming off its dim is owned by the night surface, not
            # here; what this step owns is the Status that wakes the console.
            bus.publish(Status(text="Home", kind="ok"))
            wake = getattr(self.services, "panel_wake", None)
            if callable(wake):
                wake()
            return True

        def earcon():
            return earcons.play(arrival_mod.ARRIVAL_EARCON)

        # The arrival cue is ONE burst spoken over two _say calls (welcome,
        # then the catch-up digest), so the fragments have to be thinned
        # against each other rather than one at a time. This is that burst's
        # ledger; it lives for the length of one cue.
        burst: list = []

        def greeting():
            burst.append(WELCOME_LINE)
            self._say(WELCOME_LINE)
            return True

        def catch_up():
            quiet = getattr(self, "quiet", None)
            if quiet is None:
                return False
            # release() drains atomically: the policy's own tick would read
            # the same backlog, and whichever gets there first says it.
            frags = quiet.release_fragments()
            if not frags:
                return False
            # THE JOIN (7 sentences, 5 sirs measured): "Welcome back, sir."
            # has already addressed him, so the digest's own later vocatives
            # are the ones that go. The digest arrives as FRAGMENTS and is
            # thinned exactly once, here, against the welcome in front of it
            # -- release() would have joined it into one finished string
            # first, and thinning a finished string is the mode that killed
            # the first attempt (jarvis/address.py). Then both shown and
            # spoken, so the card he reads and the voice he hears agree.
            thinned = self._thin_address(burst + list(frags))
            digest = address_mod.join_thinned(thinned[len(burst):])
            if not digest:
                return False
            burst[:] = thinned
            bus.publish(JarvisReply(text=digest, speak=True))
            self._say(digest)
            return True

        return {"panel": panel, "earcon": earcon, "greeting": greeting,
                "catch-up": catch_up}

    def _on_presence(self, ev):
        """The phone came back, or left.

        Arrival is a composed cue in a fixed order (jarvis/arrival.py):
        panel, earcon, "Welcome back, sir", catch-up. Departure is its
        mirror only in shape -- it is silent, and it waits out a confirm
        window plus a mic-silence veto first, because a sleeping phone
        radio faking a departure while he is in the room is the failure
        worth spending latency on.
        """
        self._cancel_departure()
        if not ev.home:
            bus.publish(Status(text="Away", kind="info"))
            self._arm_departure(ev)
            return
        if ev.returned:
            # The power-up sweep's proper trigger: the away->home edge is
            # the moment he actually sits down. Once a day, latched
            # (_maybe_power_up); the wake-word fallback covers a box where
            # presence is unconfigured.
            try:
                self._maybe_power_up("presence")
            except Exception:                 # noqa: BLE001 - never block a return
                log.debug("power-up check failed", exc_info=True)
            # Through the SAME greeter as the desk probe, so the damper is
            # actually shared: whichever sentinel notices him second must
            # not say "Welcome back, sir" a second time.
            self._greet_return("phone")
            return
        # Home but not a return (a poll that merely confirms he is here):
        # the console gets the state, nothing is spoken.
        bus.publish(Status(text="Home", kind="info"))

    # ------------------------------------------------------ departure
    def _cancel_departure(self) -> None:
        timer, self._departure_timer = getattr(self, "_departure_timer", None), None
        if timer is not None:
            timer.cancel()

    def _arm_departure(self, ev) -> None:
        """Schedule the confirm check. Nothing is spoken, then or later."""
        wait = arrival_mod.confirm_s(self.assistant.get)
        timer = threading.Timer(wait, self._settle_if_gone, args=(ev,))
        timer.daemon = True
        timer.name = "departure-confirm"
        self._departure_timer = timer
        timer.start()
        log.info("departure: confirming in %.0fs", wait)

    def _settle_if_gone(self, ev) -> bool:
        """The confirm window closed: settle the room, or say why not."""
        self._departure_timer = None
        if self._quitting:
            return False
        presence = getattr(self, "presence", None)
        still_away = getattr(presence, "state", "away") == "away" if presence else True
        now = time.time()
        idle = self.turns.idle_s()
        ok, why = arrival_mod.departure_ready(
            now=now, since=getattr(ev, "since", 0.0) or now,
            last_turn=(now - idle) if idle is not None else None,
            confirm=arrival_mod.confirm_s(self.assistant.get),
            mic_silence=arrival_mod.mic_silence_s(self.assistant.get),
            still_away=still_away)
        if not ok:
            log.info("departure: not settling (%s)", why)
            return False
        for step in arrival_mod.departure_plan(home=False):
            log.info("departure: %s (silent by design)", step)
        tone = getattr(self, "roomtone", None)
        if tone is not None:
            try:
                tone.settle()          # the bed drops to its floor; nothing is said
            except Exception:
                log.exception("departure: room tone settle failed")
        return True

    # ------------------------------------------------------- leave times
    def _ask_leave_time(self, key: str, place: str) -> bool:
        """jarvis/leavetime.py asking, once ever, how long a walk is.

        False means "not now": a turn is in flight, or talk-back is off, so
        the question would land on top of something or be silently lost
        with the pending answer armed. The watch retries on the next tick
        and only records the ask when this returns True."""
        if not CONFIG.talkback:
            return False
        if self._turn_busy.is_set() or self._audio_busy.is_set() or \
                getattr(getattr(self, "recorder", None), "recording", False):
            return False
        commander = getattr(self, "commander", None)
        arm = getattr(commander, "ask_leave_time", None)
        if not callable(arm):
            return False
        # The line below is proactive (nobody asked for it), so the quiet
        # gate would HOLD it -- and a held question with an armed pending
        # answer is a trap: he never hears it and the next duration he says
        # gets filed as a walk. Check the gate first and simply try again
        # on a later tick.
        quiet = getattr(self, "quiet", None)
        if quiet is not None:
            try:
                if quiet.should_hold():
                    return False
            except Exception:
                log.exception("leavetime: quiet gate failed")
                return False
        question = leavetime_mod.ASK_LINE.format(place=place)
        # A refusal means another question already owns the floor: the
        # duration would be graded as a flashcard answer or swallowed by a
        # working session, and leavetime._maybe_ask burns its once-ever ask
        # the instant this returns True. Say nothing; the watch retries.
        if arm(key, place) is False:
            return False
        bus.publish(JarvisReply(text=question, speak=False))
        self._say(question, proactive=True, kind="message")
        return True

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

    # Only when a chime will actually play. The gate is the earcon lexicon's
    # own key (sound.earcons, default true), NOT CONFIG.sound: that flag
    # lives in voice_settings.json, defaults False and means "the old
    # start/stop chimes", so hanging the wake acknowledgement off it shipped
    # it mute. Whichever gate says yes, the guard applies -- the bloom fires
    # while the mic is opening and would otherwise be recorded straight back
    # through the Snowball.
    _WAKE_BEEP_GUARD_S = 0.2

    def _wake_chime_enabled(self) -> bool:
        """True when the wake acknowledgement will make a sound."""
        try:
            return bool(CONFIG.sound or earcons.enabled())
        except Exception:                       # noqa: BLE001 - config boundary
            return bool(CONFIG.sound)

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
        # The power-up sweep's fallback trigger. Presence is idle until
        # phone_ip is configured (it is not, on this box), so without this
        # the feature would never fire on the machine that runs it. The date
        # latch keeps it to once a day whichever trigger gets there first.
        try:
            self._maybe_power_up("hotword")
        except Exception:                     # noqa: BLE001 - never block a wake
            log.debug("power-up check failed", exc_info=True)
        self._followup_after_speech = False   # a wake supersedes any follow-up
        # A wake word means he has come back with something of his own; the
        # open debrief question is over, and treating "Jarvis, set a timer"
        # as an answer about the midterm would file nonsense forever.
        self._pending_debrief = None
        self._turn_filler_pending = False     # a stale flag would label this answer a filler
        self._say_again_count = 0
        self._wake_pending = True             # the capture about to open answers a wake word
        chime = self._wake_chime_enabled()
        if chime:
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
        threading.Timer(self._WAKE_BEEP_GUARD_S if chime else 0.0,
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

    def _decode_clip(self, audio, verify=True):
        """The one decode path for a captured clip: speaker filter, then the
        full transcribe. Returns (audio, stats, rejected, result); rejected
        means the speaker gate dropped the whole clip (result is None).

        `verify=False` is the intercom's (jarvis/intercom.py): a clip that
        arrived over the 0600 command socket through the user's own SSH
        session is already authenticated, and a phone codec moves the ECAPA
        embedding far enough that the transcript gate -- which fails SHUT --
        would drop his own voice. The microphone path never passes it."""
        stats = {}
        if verify and CONFIG.speaker_verify and self.speaker.enrolled:
            filtered, stats = self.speaker.filter_segments(audio)
            if filtered is None:
                return audio, stats, True, None
            audio = filtered
        return audio, stats, False, self.transcriber.transcribe(audio)

    def decode_clip(self, audio, verify=True):
        """Public seam for _decode_clip: the command socket's intercom must
        reach the speaker gate and Whisper without reaching into a private."""
        return self._decode_clip(audio, verify=verify)

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
        self._departure_timer = None          # the silent-settle confirm window
        # The Board: the poll thread while it is up, the Canvas cache, and
        # the live Claude task states the sessions panel reads (nothing else
        # in the app kept them — the UI's tracker is a Tk-thread object).
        self._board_feed = None
        self._board_tasks: dict = {}
        self._canvas_due_cache = (-1e9, [])   # monotonic; see _last_nudge_ts
        self._room_gpu_cache = (-1e9, None)   # ditto: the ambient GPU reading
        # The ambient slab's one outbound dependency, on a backoff
        self._room_playing_text = ""
        self._room_playing_ts = -1e9
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
        if done and getattr(result, "speak", False):
            # After the answer, never inside it. _emit_result has already
            # queued the reply, and TTS.speak is FIFO, so this lands as its
            # own beat behind it (jarvis/aside.py).
            self._consider_aside(text, result)
        if source == "voice":
            if status.startswith("Briefing"):
                self._mark_briefing_delivered()
            elif status.startswith(("Was that for me", "Ignored")):
                pass        # no answer here: the briefing waits for a real turn
            elif (reply or not done) and self._briefing_due():
                self._briefing_pending = True

    # ------------------------------------------------------- the debrief
    DEBRIEF_TTL_S = 120.0
    DEBRIEF_FILED_LINE = "Noted, sir."
    DEBRIEF_DECLINED_LINE = "Of course, sir."

    def _ask_debrief(self, cand):
        """DebriefWatch found something that ended: ask, once.

        Called from the watch's own thread. Deliberately NOT proactive: the
        quiet gate was already checked inside tick(), and holding this line
        for the catch-up digest would ask how last night's exam went over
        breakfast, hours after the moment it belonged to."""
        if not CONFIG.talkback:
            return
        if self._turn_busy.is_set() or self._audio_busy.is_set() \
                or getattr(self.recorder, "recording", False):
            return                          # mid-turn: he is talking already
        tts = getattr(self, "tts", None)
        if getattr(tts, "is_speaking", False) or getattr(tts, "pending", 0):
            return
        if self._pending_debrief:
            return                          # one open question at a time
        # ...and the debrief is not the only question in the app. It is the
        # ONE pending stage hoisted ABOVE commander.handle (_dispatch runs
        # _debrief_reply first), so arming over a live flashcard, a working
        # session, a destructive read-back or a ringing alarm files THEIR
        # answer into the episodic record and strands them. This must stay
        # above mark_asked: that is the once-ever promise, and a question
        # withheld was never put. The next tick is five minutes away and
        # the candidate is not consumed by a refusal.
        held = debrief_mod.floor_holder(self)
        if held:
            log.info("debrief held: %s owns the floor", held)
            return
        watch = getattr(self, "debrief", None)
        if watch is not None:
            # Written when the question is PUT, not when it is answered: an
            # unanswered debrief has still been asked, and asking twice is
            # the one thing this feature must never do.
            watch.mark_asked(cand.key)
        self._pending_debrief = {"key": cand.key, "cand": cand,
                                 "at": time.monotonic()}
        log.info("debrief asked: %r", cand.question)
        bus.publish(JarvisReply(text=cand.question, speak=True))
        self._followup_after_speech = True   # answer it without the wake word
        self._say(cand.question)

    def _debrief_reply(self, text, source):
        """The open debrief owns this transcript -- or gives it up.

        Gives it up for anything that is plainly a command (a Tier-1 match,
        a "jarvis" prefix) and for anything that arrives more than
        DEBRIEF_TTL_S later: "set a timer for five minutes" is not how the
        midterm went, and filing it as such would poison the record he is
        meant to be able to trust months from now."""
        pending = self._pending_debrief
        if pending is None or source not in ("voice", "typed"):
            return None
        if time.monotonic() - pending["at"] > self.DEBRIEF_TTL_S:
            self._pending_debrief = None
            return None
        # A holder that appeared AFTER the question was put outranks it:
        # commander._try_ringing and every other pending stage sit BELOW
        # this filter, and the `addressed` escape only probes
        # ASSISTANT_TIER1 -- which has no ringing-alarm words -- so a bare
        # "stop" or a flashcard answer was swallowed and filed as how the
        # midterm went. Return None WITHOUT clearing _pending_debrief: the
        # question survives and is still answerable once the floor is free.
        held = debrief_mod.floor_holder(self)
        if held:
            log.info("debrief stands down: %s owns this turn", held)
            return None
        commander = getattr(self, "commander", None)
        try:
            addressed = strip_jarvis_prefix(text) is not None or \
                bool(commander and commander._match_assistant(text))
        except Exception:  # noqa: BLE001 - a matcher failure must not eat the turn
            addressed = False
        if addressed:
            self._pending_debrief = None
            return None
        self._pending_debrief = None
        cand = pending["cand"]
        if parse_yes_no(text) is False:
            # "Not now" / "never mind": he has been asked, and the ledger
            # already says so, so he is never asked again. Nothing is filed.
            log.info("debrief declined for %r", cand.title)
            return CommandResult(handled=True, reply=self.DEBRIEF_DECLINED_LINE,
                                 speak=True, status="Debrief declined")
        filed = debrief_mod.file_answer(text, cand, memory=self.memory,
                                        context=self.context)
        if not filed:
            return None
        return CommandResult(handled=True, reply=self.DEBRIEF_FILED_LINE,
                             speak=True, status=f"Debrief filed: {cand.word}")

    # ------------------------------------------------------------ asides
    def _consider_aside(self, text, result):
        """One volunteered sentence after the answer, or nothing.

        Queued as a SEPARATE utterance behind the reply, not spliced into
        it: TTS.speak is a FIFO worker, so the answer plays first and the
        aside follows it as its own beat -- which is what makes it read as
        an afterthought rather than as part of the sentence he asked for.
        The engine owns every gate (quiet, budget, the shared said-keys
        ledger); this only carries the structured action result across."""
        engine = getattr(self, "aside", None)
        action = getattr(result, "action", None)
        if engine is None or action is None:
            return
        try:
            line = engine.consider(text, getattr(result, "reply", "") or "", action)
        except Exception:
            log.exception("aside consider failed")
            return
        if not line:
            return
        bus.publish(JarvisReply(text=line, speak=True))
        self._say(line)                  # never proactive: see jarvis/aside.py

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
        """A longer wait for the first word when the last reply left
        something open: a working session (jarvis/dialogue.py,
        ``session.window_s`` ~18 s), lecture notes (`lecture.window_s`,
        20 s), or a question Jarvis just asked -- a flashcard or a yes/no
        read-back (`quiz.window_s`, 15 s). Else None for the recorder's own
        CONFIG.followup_window. The recorder caps it at half its hard cap.
        This is still one capture per answer, not a hands-free mic."""
        commander = getattr(self, "commander", None)
        session = getattr(commander, "_pending_session", None)
        if session is not None and not getattr(session, "finished", False):
            try:
                window = float(getattr(session, "window_s", SESSION_WINDOW_S))
            except (TypeError, ValueError):
                window = SESSION_WINDOW_S
            return max(window, float(CONFIG.followup_window))
        if getattr(commander, "lecture_course", None):
            return self._window_setting("lecture.window_s", 20.0)
        # The open debrief ("How did the midterm go, sir?") arms the mic
        # itself (_followup_after_speech) and is answered in a sentence, not
        # a word. It is checked HERE and deliberately NOT inside
        # _question_open: debrief.floor_holder calls that predicate, so the
        # debrief would name itself as the holder and stand down from its
        # own answer for the whole 120 s.
        pending = self._pending_debrief
        if pending is not None:
            try:
                if time.monotonic() - float(pending["at"]) <= self.DEBRIEF_TTL_S:
                    return self._window_setting("quiz.window_s", 15.0)
            except (TypeError, ValueError, KeyError):
                pass
        if self._question_open(commander):
            return self._window_setting("quiz.window_s", 15.0)
        return None

    def _window_setting(self, key: str, default: float) -> float:
        try:
            window = float(self.assistant.get(key, default) or default)
        except (TypeError, ValueError, AttributeError):
            window = default
        return max(window, float(CONFIG.followup_window))

    def _question_open(self, commander) -> bool:
        """Jarvis asked something and is waiting on the answer.

        The 4 s follow-up window is sized for "...and Tuesday?" -- it is
        far too short for a flashcard (QuizSession.ANSWER_WINDOW_S is 300 s,
        so the session is patient but the mic was not) or for a yes/no the
        user has to think about. Each of these is dropped the moment it
        expires, so the longer window only ever covers a live question.
        """
        if commander is None:
            return False
        # The commander owns the authoritative list -- every rung of
        # handle() that holds the floor, each with its own expiry -- so the
        # study offer and the objection get the 15 s quiz window rather
        # than the 4 s CONFIG.followup_window. The branches below are the
        # older half-list, kept so a slim/duck-typed commander in a test
        # still answers; they are redundant, not wrong.
        probe = getattr(commander, "question_open", None)
        if callable(probe):
            try:
                if probe():
                    return True
            except Exception:
                log.debug("question_open failed", exc_info=True)
        quiz = getattr(commander, "_pending_quiz", None)
        if quiz is not None and not getattr(quiz, "finished", True):
            try:
                if not quiz.stale():
                    return True
            except Exception:
                log.debug("quiz staleness check failed", exc_info=True)
        pend = getattr(commander, "_pending_destructive", None)
        if isinstance(pend, tuple) and len(pend) == 3:
            try:
                if time.monotonic() - float(pend[2]) <= DESTRUCTIVE_TTL_S:
                    return True
            except (TypeError, ValueError):
                pass
        # The wake-alarm offer lives on the services namespace, not on the
        # commander: briefing.make_tools parks it there for
        # _try_alarm_offer.
        offer = getattr(getattr(self, "services", None), "alarm_offer", None)
        if isinstance(offer, dict) and offer:
            try:
                made = float(offer.get("made_at") or 0.0)
            except (TypeError, ValueError):
                made = 0.0
            if not made or time.time() - made <= OFFER_TTL_S:
                return True
        return False

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
        wd = getattr(self, "winddown", None)
        if wd is not None:
            try:
                # The other end of "good night": the first wake of the day is
                # the morning even when he never said the word.
                wd.restore()
            except Exception:
                log.exception("wind-down restore at first wake failed")
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
        # THE JOIN, and the longest burst in the live log: measured at
        # 14:33:49-14:34:29 as 40 seconds of unbroken speech over seven TTS
        # segments carrying FOUR sirs. The review, the weekly lines and the
        # hand-over below are four _say calls with nothing between them, so
        # they are ONE burst and have to be thinned against each other the
        # way the arrival cue is -- fragments, never a joined string.
        burst: list = []

        def say_in_burst(line):
            """Speak ``line`` as the next fragment of this burst."""
            if not line:
                return
            burst.append(line)
            # Stable by construction: the pass reads left to right and never
            # rewrites the first fragment, so re-thinning what was already
            # spoken cannot change it (jarvis/address.py, invariant 1).
            burst[:] = self._thin_address(burst)
            spoken = burst[-1]
            if address_mod.is_speakable(spoken):
                self._say(spoken)

        say_in_burst(review)
        # Then the two weekly lines, if either is owed. Both are produced
        # in the small hours by their own threads and deliberately NOT
        # spoken there: this path is the quiet-gated one (_briefing_due
        # refuses inside quiet hours), so a report written at 3 am is
        # heard at breakfast and never at 3 am.
        for line in (self._pending_week_line(), self._pending_garden_line()):
            say_in_burst(line)
        # The model's own reply lands after this through brain.chat and is
        # NOT in the ledger: it is one authored-shaped line with exactly one
        # sir (24/24 measured), and catching it would mean rewriting at the
        # TTS door, which is the thing this design does not do. Four sirs
        # become two: the review keeps the burst's first, the reply keeps
        # its own.
        say_in_burst("Your briefing for today, sir.")
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
                                                 PATHS.LOG_DIR / "turns.jsonl", day,
                                                 study=self._study_table())
        name = "sir"
        return dayreview_mod.spoken_line(digest, label=label, name=name)

    def _study_table(self) -> dict:
        """focus.study_days() over the live paths, or {} -- the day review
        and the "how much did I study" answer share this one reader."""
        try:
            from jarvis import focus as focus_mod
            tk = getattr(self.services, "timekeeper", None)
            return focus_mod.study_days(
                state_path=PATHS.MEMORY_DIR / "focus_session.json",
                db_path=getattr(tk, "db_path", None))
        except Exception:                    # noqa: BLE001 - source boundary
            log.debug("study ledger unreadable", exc_info=True)
            return {}

    def day_review_text(self, which="yesterday") -> str:
        """"How did yesterday go" / "how is today going": the spoken review."""
        today = datetime.now().date()
        if str(which).lower() == "today":
            # today is still open: never read from a filed digest
            digest = dayreview_mod.summarize_day(PATHS.LOG_DIR / "jarvis.log",
                                                 PATHS.LOG_DIR / "turns.jsonl", today,
                                                 study=self._study_table())
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

    def _pending_week_line(self) -> str:
        """The weekly self-review's two sentences, once, and only when a
        report has been filed that nobody has heard yet."""
        try:
            week = dayreview_mod.pending_week(PATHS.REVIEWS_DIR)
            if week is None:
                return ""
            line = dayreview_mod.week_spoken(week)
            # Marked before speaking, not after: a TTS failure must not
            # make him deliver last week's review again tomorrow.
            dayreview_mod.mark_week_spoken(PATHS.REVIEWS_DIR, week.get("week", ""))
            return line
        except Exception:
            log.exception("pending week review failed")
            return ""

    def _pending_garden_line(self) -> str:
        """"I filed three things from this week, sir." Once per pass."""
        g = getattr(self, "garden", None)
        if g is None:
            return ""
        try:
            line = g.pending_line()
            if line:
                g.mark_spoken()
            return line
        except Exception:
            log.exception("pending garden line failed")
            return ""

    # --------------------------------------------------------- the week
    def _on_week_filed(self, week):
        """The weekly self-review was just aggregated: the table goes to
        Discord like the nightly one, and every recurring warning and
        worsened number is appended to feedback.jsonl. The two spoken
        sentences are NOT said here -- the report is filed in the small
        hours; `pending_week` holds them for the first wake."""
        self._alert("milestone", f"Week review {week.get('week', '')}",
                    dayreview_mod.week_table(week))
        self._file_week_regressions(week)

    def _file_week_regressions(self, week) -> int:
        """One JSON line per regression in feedback.jsonl -- the standing
        bug list Jarvis wrote about himself, ready for the next Claude
        session. `kind: regression` distinguishes these rows from the
        intent-gate labels the commander appends to the same file; nothing
        parses it, and one audit trail beats two. Once per week, because
        week_tick only calls back on the tick that FILED the report."""
        try:
            rows = dayreview_mod.week_regressions(week)
        except Exception:
            log.exception("weekly regressions failed")
            return 0
        if not rows:
            return 0
        try:
            path = Commander.FEEDBACK_LOG
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                for row in rows:
                    fh.write(json.dumps({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "kind": "regression", "week": week.get("week", ""),
                        "text": row.get("text", ""), "how": "weekly-review",
                        "detail": row}) + "\n")
        except Exception:
            log.exception("weekly regression log failed")
            return 0
        log.info("week review: filed %d regressions to feedback.jsonl", len(rows))
        return len(rows)

    def week_review_text(self) -> tuple:
        """"How was my week": (spoken, card) from the newest filed weekly
        report, or an honest excuse before the first one exists."""
        week = dayreview_mod.latest_week(PATHS.REVIEWS_DIR)
        if week is None:
            reviewer = getattr(self, "dayreviewer", None)
            if reviewer is not None:
                try:
                    week = reviewer.week_tick()
                except Exception:
                    log.exception("on-demand week review failed")
        if week is None or not week.get("has_data"):
            return ("I've no full week to review yet, sir; the reports start "
                    "once a week of days has been filed.", "")
        return dayreview_mod.week_spoken(week), dayreview_mod.week_table(week)

    # ------------------------------------------------------ memory garden
    def garden_report_text(self) -> str:
        g = getattr(self, "garden", None)
        if g is None:
            return "The memory garden isn't running, sir."
        return g.report_line()

    def garden_undo_text(self) -> str:
        g = getattr(self, "garden", None)
        if g is None:
            return "The memory garden isn't running, sir."
        return g.undo()

    # ------------------------------------------------------- diagnostics
    def self_state(self, full: bool = True) -> dict:
        """Everything Jarvis knows about himself, as one dict.

        The single source behind "run diagnostics" (plain sheet + film
        register) and "how are you?" (one clause) -- see jarvis/selfstate.py
        for the renderers. Two registers off ONE sheet; never a second
        gatherer that can drift from this one.

        ``full=False`` skips the two probes that cost a subprocess and say
        nothing about his wellbeing (the output sink, his own Claude
        panes), so a courtesy stays the fastest exchange in the system.
        """
        import statistics
        # One monotonic read: the two-call form returned a hair BELOW zero
        # on an app with no _app_started (the fallback is sampled after the
        # minuend), and a negative uptime is a lie even at 1e-7 seconds.
        now = time.monotonic()
        state = {
            "uptime_s": now - getattr(self, "_app_started", now),
            "stt_model": CONFIG.model,
            "brain_model": _brain_model_name(),
            "tts_engine": getattr(getattr(self, "tts", None), "engine",
                                  CONFIG.tts_engine),
            "endpointing": bool(getattr(getattr(self, "recorder", None),
                                        "endpointer", None)),
            "turns_today": 0, "median_wait": None,
            "mem_free_gb": None, "mem_total_gb": None,
            "gpu_temp_c": None, "gpu_mhz": None, "gpu_util_pct": None,
            "lent": False, "enrolled": False, "num_samples": 0,
            "held": 0, "quiet_reason": "",
            "sink": "", "sink_dummy": False,
            "claude_panes": 0, "claude_working": 0, "modes": "",
        }
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
            state["turns_today"] = n
            if waits:
                state["median_wait"] = statistics.median(waits)
        except (OSError, ValueError):
            pass
        try:
            mem = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0])
            # MemAvailable only: the GB10 pool is unified and nvidia-smi
            # reads memory.used/total as N/A, so there is no per-device VRAM
            # figure to quote and inventing one would be a lie.
            state["mem_free_gb"] = mem["MemAvailable"] / 1048576
            state["mem_total_gb"] = mem["MemTotal"] / 1048576
        except Exception:
            log.debug("meminfo unreadable", exc_info=True)
        try:
            # Popen + kill-without-wait (jarvis.tools.health): run() would
            # wait on an nvidia-smi wedged in D-state under a stuck NVRM
            # lock, and this is the command asked in exactly that state.
            from jarvis.tools.health import parse_nvidia_smi, run_nvidia_smi
            reading = parse_nvidia_smi(run_nvidia_smi()) or {}
            state["gpu_temp_c"] = reading.get("temp_c")
            state["gpu_mhz"] = reading.get("sm_mhz")
            state["gpu_util_pct"] = reading.get("util_pct")
        except Exception:
            log.debug("nvidia-smi unreadable", exc_info=True)
        # getattr throughout: "jarvis status" is answered from a
        # half-built app in the tests and from a real one in the field, and
        # a missing collaborator must cost a field, never the whole sheet.
        brain = getattr(self, "brain", None)
        if brain is not None:
            try:
                state["lent"] = bool(brain.is_lent())
            except Exception:
                log.debug("residency unreadable", exc_info=True)
        speaker = getattr(self, "speaker", None)
        if speaker is not None and speaker.enrolled:
            state["enrolled"] = True
            try:
                state["num_samples"] = int(speaker.num_samples)
            except (TypeError, ValueError):
                log.debug("num_samples unreadable", exc_info=True)
        quiet = getattr(self, "quiet", None)
        if quiet is not None:
            try:
                state["held"] = len(quiet.held)
                state["quiet_reason"] = quiet.reason() or ""
            except Exception:
                log.debug("quiet state unreadable", exc_info=True)
        if full:
            try:
                from jarvis.voice_check import output_sink_state
                sinks = output_sink_state()
                state["sink_dummy"] = bool(sinks.get("probed")) and \
                    bool(sinks.get("dummy"))
                suspended = set(sinks.get("suspended") or ())
                live = [s for s in (sinks.get("sinks") or []) if s not in suspended]
                state["sink"] = (live or sinks.get("sinks") or [""])[0]
            except Exception:
                log.debug("sink probe failed", exc_info=True)
            try:
                # own_sessions(), NOT discover_sessions(): the latter globs
                # transcript FILES, so "four sessions on the board" from it
                # would be a figure about the disk, not about live panes.
                claude = getattr(self, "claude", None)
                own = claude.own_sessions() if claude is not None else []
                state["claude_panes"] = len(own)
                state["claude_working"] = sum(1 for _, working in own if working)
            except Exception:
                log.debug("claude panes unreadable", exc_info=True)
        try:
            # The sticky modes are part of the sheet, not a separate one:
            # "status" has always named them and the film register wants
            # them too.
            state["modes"] = self.open_modes_line()
        except Exception:
            log.debug("open modes unreadable", exc_info=True)
        return state

    def diagnostics_text(self) -> str:
        """"Run diagnostics": the plain sheet, from real data. The spoken
        answer is the film-register rendering of the same dict (commander
        _h_diagnostics); this stays the card, the cmdsock "status" reply
        and ask.py --status."""
        return selfstate.diagnostics_line(self.self_state())

    def open_modes_line(self) -> str:
        """The sticky modes that are open, or "".

        They listen to the microphone only (commander._handle_inner), so a
        terminal turn is answered normally instead of being filed -- which
        also means an open mode is invisible from a shell. "status" names
        them, and the end phrase closes them from anywhere.
        """
        c = getattr(self, "commander", None)
        if c is None:
            return ""
        bits = []
        course = getattr(c, "lecture_course", None)
        if course:
            n = getattr(getattr(c, "_lecture", None), "count", 0) or 0
            bits.append(f"lecture notes open for {course}, "
                        f"{n} line{'s' if n != 1 else ''}")
        if getattr(c, "dictation", False):
            bits.append("dictation mode on")
        quiz = getattr(c, "_pending_quiz", None)
        if quiz is not None and not getattr(quiz, "finished", True):
            bits.append(f"a quiz open at question {quiz.index + 1} "
                        f"of {quiz.total}")
        if not bits:
            return ""
        line = bits[0] if len(bits) == 1 else \
            ", ".join(bits[:-1]) + " and " + bits[-1]
        return line[0].upper() + line[1:] + "."

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

    # ------------------------------------------------------------- the Board
    # Providers for jarvis.board.board_state(). Everything expensive is
    # cached or gated HERE and never in the pure layer: health.snapshot()
    # spawns nvidia-smi with a 5 s timeout and canvas_due is a Canvas REST
    # call, while the Board polls every 5 s.
    def _board_canvas_lines(self) -> list:
        """Canvas items due soon, cached for CANVAS_TTL_S.

        Reached through the TOOL REGISTRY, not by import: canvas_due is a
        closure defined inside tools/canvas.make_tools and registered as a
        tool, exactly as tools/briefing.py calls it. Silent (and empty)
        when the token is unset — a box with no Canvas must not nag."""
        now = time.monotonic()
        # The TTL is on the TIMESTAMP, never on the payload: `if lines and
        # ...` could not be satisfied by a stored EMPTY result, so a Canvas
        # with nothing due (or one erroring) re-issued a live REST call on
        # every 5 s Board tick -- ~720 round trips an hour. The sentinel is
        # -1e9 rather than 0.0 because time.monotonic() is uptime-based: a
        # 0.0 default would serve the empty cache for the first 300 s after
        # boot instead of fetching once.
        at, lines = getattr(self, "_canvas_due_cache", (-1e9, []))
        if now - at < CANVAS_TTL_S:
            return lines
        from jarvis.tools.briefing import _due_lines
        try:
            lines = list(_due_lines(self.tools) or [])
        except Exception:                          # noqa: BLE001 - tool boundary
            log.debug("board: canvas_due failed", exc_info=True)
            lines = []
        self._canvas_due_cache = (now, lines)
        return lines

    def _board_sessions(self) -> list:
        from jarvis import claude_session
        return claude_session.discover_sessions(limit=BOARD_SESSIONS)

    def _board_turns(self) -> list:
        from jarvis.logtriage import read_turns
        return read_turns(PATHS.LOG_DIR / "turns.jsonl", board_mod.TURN_WINDOW)

    def _board_schedule(self) -> list:
        tk = self.timekeeper
        return list(tk.list("all")) if tk is not None else []

    def board_state(self):
        """One composed Board. Called on the BoardFeed thread (and on a
        commander worker for the spoken read) — never on the Tk thread."""
        from jarvis.tools import health
        return board_mod.board_state(
            health=health.snapshot,
            sessions=self._board_sessions,
            turns=self._board_turns,
            focus=lambda: getattr(self, "focus", None),
            quiet=lambda: getattr(self, "quiet", None),
            presence=lambda: getattr(self, "presence", None),
            canvas=self._board_canvas_lines,
            schedule=self._board_schedule,
            tasks=lambda: dict(getattr(self, "_board_tasks", {})))

    def board_text(self) -> str:
        """`jarvis board` over SSH — the same state the panel draws."""
        return board_mod.board_text(self.board_state())

    def _board_show(self) -> bool:
        """Raise the Board; True when it was ALREADY up (so the spoken
        answer can say so rather than pretending it just appeared)."""
        feed = getattr(self, "_board_feed", None)
        already = feed is not None and feed.running
        if not already:
            feed = board_mod.BoardFeed(self.board_state)
            feed.start()                  # its first tick is immediate
            self._board_feed = feed
        bus.publish(BoardCommand(action="show"))
        return already

    def _board_hide(self) -> bool:
        feed, self._board_feed = getattr(self, "_board_feed", None), None
        if feed is not None:
            feed.stop()
        bus.publish(BoardCommand(action="hide"))
        return True

    def _board_read(self, panel: str) -> str:
        """"Focus on the sessions": light that panel AND hand back the one
        sentence to speak. An unresolvable name answers "" so the commander
        can fall through to whoever really owns those words."""
        key = board_mod.resolve_panel(panel)
        if not key:
            return ""
        line = board_mod.panel_line(self.board_state(), panel)
        if line:
            bus.publish(BoardCommand(action="focus", panel=key))
        return line

    # ------------------------------------------------ the room (ambient slab)
    def _room_playing(self) -> str:
        """What Spotify is playing, on a BACKOFF.

        spotify.now_playing() is a live REST call with no cache behind it
        (the only cache in that module is the OAuth token), so this is the
        ambient slab's one outbound dependency and it is gated: ROOM_SPOTIFY
        _ACTIVE_S while something was playing, ROOM_SPOTIFY_IDLE_S once it
        went quiet. A permanent 10 s heartbeat to api.spotify.com for a row
        nobody is reading is not a feature."""
        spot = getattr(self.services, "spotify", None)
        if spot is None:
            return ""
        now = time.monotonic()
        gap = ROOM_SPOTIFY_ACTIVE_S if self._room_playing_text \
            else ROOM_SPOTIFY_IDLE_S
        if now - self._room_playing_ts < gap:
            return self._room_playing_text
        self._room_playing_ts = now
        try:
            res = spot.now_playing()
            text = str(getattr(res, "speak", "") or getattr(res, "text", ""))
        except Exception:                          # noqa: BLE001 - tool boundary
            log.debug("room: now_playing failed", exc_info=True)
            text = ""
        if "nothing" in text.lower() or "not playing" in text.lower():
            text = ""
        self._room_playing_text = text.strip()
        return self._room_playing_text

    def _room_next_event(self) -> str:
        cal = getattr(self.services, "calendar", None)
        if cal is None:
            return ""
        try:
            now = datetime.now().astimezone()
            upcoming = [e for e in cal.events()
                        if not e.all_day and e.start > now]
            if not upcoming:
                return ""
            ev = min(upcoming, key=lambda e: e.start)
            return f"{ev.title} {ev.start.strftime('%-I:%M %p').lower()}"
        except Exception:                          # noqa: BLE001 - source boundary
            log.debug("room: calendar read failed", exc_info=True)
            return ""

    def _room_due(self) -> str:
        lines = self._board_canvas_lines()
        return lines[0] if lines else ""

    def room_state(self, gpu_pct=None) -> dict:
        """Room facts for the console's ambient / standby slab, as
        PRE-COMPUTED STRINGS.

        Called on the window's existing 5 s off-Tk-thread probe, never on
        the Tk thread — every value here is a config read, a cached answer
        or a gated call. `presence` is "" unless the sentinel is actually
        configured: presence.is_home() answers True when it is not
        (jarvis/presence.py:159), and a confident false HOME is worse than
        no row at all. `arc` reads the day-arc phase through a getattr seam
        so this lands whichever way that change merges."""
        from jarvis.tools import weather as weather_mod
        pres = getattr(self, "presence", None)
        quiet = getattr(self, "quiet", None)
        arc = getattr(self, "arc", None)
        room = {"playing": self._room_playing(),
                "next": self._room_next_event(),
                "due": self._room_due(),
                "temp": weather_mod.cached_temperature() or "",
                "arc": "", "presence": "", "quiet": "", "gpu": None}
        if pres is not None and getattr(pres, "configured", False):
            room["presence"] = str(getattr(pres, "state", "") or "")
        if quiet is not None:
            try:
                room["quiet"] = str(quiet.reason() or "")
            except Exception:                      # noqa: BLE001 - policy boundary
                log.debug("room: quiet read failed", exc_info=True)
        for name in ("phase", "state"):
            value = getattr(arc, name, None) if arc is not None else None
            if isinstance(value, str) and value:
                room["arc"] = value
                break
        room["gpu"] = self._room_gpu(gpu_pct)
        return room

    def _room_gpu(self, gpu_pct=None):
        """GPU utilisation as 0..1 for the ambient slab, or None.

        The caller's own reading wins. This used to call
        health.snapshot(gpu=True) on the window's 5 s worker -- the very
        pass that had ALREADY forked nvidia-smi for the temps row -- so
        every tick spawned a second nvidia-smi and walked /proc twice more
        (top_processes + find_trainers) for one number the loop was
        throwing away: ~720 extra spawns an hour. `gpu_pct` (0-100) is the
        seam that pass fills; with nothing handed in the reading is a bare
        nvidia-smi, cached for ROOM_GPU_TTL_S, never the whole snapshot.
        """
        if gpu_pct is not None:
            try:
                return max(0.0, min(1.0, float(gpu_pct) / 100))
            except (TypeError, ValueError):        # a caller's bad reading
                return None
        now = time.monotonic()
        at, cached = getattr(self, "_room_gpu_cache", (-1e9, None))
        if now - at < ROOM_GPU_TTL_S:
            return cached
        value = None
        try:
            from jarvis.tools import health
            gpu = (health.parse_nvidia_smi(health.run_nvidia_smi())
                   or {}).get("util_pct")
            value = None if gpu is None else max(0.0, min(1.0, gpu / 100))
        except Exception:                          # noqa: BLE001 - probe boundary
            log.debug("room: gpu read failed", exc_info=True)
        # The timestamp is stamped even for a failure: a box with no
        # nvidia-smi must not pay the spawn every 5 s to learn that again.
        self._room_gpu_cache = (now, value)
        return value

    # ------------------------------------------------------------ power-up
    def _boot_sweep_due(self, idle_s=None) -> bool:
        """Has today earned the power-up sweep?

        The date latch is a `boot_sweep` key in the SAME briefing_state.json
        the first-wake briefing already writes atomically — a second latch
        file would be a second thing to get wrong, and this one already
        survives a crash-relaunch."""
        # Imported here, not at module scope: jarvis.ui.* pulls tkinter, and
        # nothing else in app.py imports the UI outside main().
        from jarvis.ui import console_mode
        try:
            state = json.loads(self._briefing_state_path().read_text())
        except (OSError, ValueError):
            state = {}
        gap = self.assistant.get("console.powerup_gap_h",
                                 console_mode.POWERUP_GAP_H)
        return console_mode.powerup_due(str(state.get("boot_sweep") or ""),
                                        idle_s=idle_s, gap_h=float(gap or 6))

    def _mark_boot_sweep(self) -> None:
        p = self._briefing_state_path()
        try:
            state = json.loads(p.read_text())
            if not isinstance(state, dict):
                state = {}
        except (OSError, ValueError):
            state = {}
        state["boot_sweep"] = datetime.now().date().isoformat()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")     # a torn write must not eat the day
            tmp.write_text(json.dumps(state))
            os.replace(tmp, p)
        except OSError:
            log.debug("boot sweep state save failed", exc_info=True)

    def _maybe_power_up(self, reason: str) -> bool:
        """Fire the sweep on the first activity of the day, once.

        Two triggers, both needed: the presence away->home edge is the
        proper one, and the first HotwordDetected of the day is the
        fallback — presence is idle until phone_ip is configured, and it is
        not configured on this box, so without the fallback the feature
        would be dark on the only machine that runs it."""
        if not self.assistant.get("console.powerup", True):
            return False
        idle = None
        # REGRESSION SITE: this read `getattr(self, "desk_idle_s", None)`,
        # an attribute JarvisApp has never had -- the seam is on the
        # services namespace (_build_services) -- so `idle` was always None
        # and powerup_due's "the machine really was left alone" gate never
        # applied. getattr on services too: the probe must not raise on a
        # half-built app.
        fn = getattr(getattr(self, "services", None), "desk_idle_s", None)
        if callable(fn):
            try:
                idle = fn()
            except Exception:                      # noqa: BLE001 - probe boundary
                idle = None
        if not self._boot_sweep_due(idle):
            return False
        self._mark_boot_sweep()
        log.info("power-up sweep (%s)", reason)
        bus.publish(PowerUp(reason=reason,
                            gap_h=(idle or 0.0) / 3600.0))
        return True

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
        if source not in SOCKET_SOURCES:
            self._active_turn_id = ""
        if source == "voice":
            self._turn_start()
            self.turns.mark("handle")
        # The Whisper avg_logprob travels only when there is one: typed
        # text has none, and a stand-in commander need not take the keyword.
        kw = {} if confidence is None else {"confidence": confidence}
        try:
            # An open debrief question owns this transcript unless it is
            # plainly a command -- the answer is FILED, never routed to the
            # model as chat (jarvis/debrief.py).
            filed = self._debrief_reply(text, source)
            result = self._emit_result(
                filed if filed is not None
                else self.commander.handle(text, source, **kw))
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
        (jarvis/cmdsock.py) with source='cli', on the client's thread, and
        with source='intercom' for a clip sent from the phone.
        `quiet` (socket sources only) answers in text and keeps the soundbar
        silent; `turn_id` stamps this turn's replies for the socket stream."""
        text = (text or "").strip()
        if not text:
            return None
        # Barge-in: a typed command while Jarvis is talking cuts him off
        # (the films' JARVIS never talks over Tony), then gets answered.
        # NOT for the socket sources: an unattended script, a cron call or a
        # clip sent from another room must not cut a reply he is speaking to
        # someone standing in front of him.
        if source not in SOCKET_SOURCES:
            self.interrupt_speech()
        if source == "typed":
            self.history.add(text)
        self._quiet_turn = bool(quiet) and source in SOCKET_SOURCES
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
                          # start() is a no-op unless phone.enabled
                          ("webapp", getattr(self, "webapp", None)),
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
        wd = getattr(self, "winddown", None)
        if wd is not None:
            try:
                # A screen dimmed last night by a Jarvis that has since been
                # restarted has nobody else to brighten it: the autostart
                # entry is not installed on this box, so app start IS the
                # login hook. expired_only, because a restart at two in the
                # morning must not light the room back up.
                wd.restore(expired_only=True)
            except Exception:
                log.exception("wind-down restore at start failed")
        try:
            # The nightly self-review: files yesterday's digest under
            # MEMORY_DIR/reviews and posts the table to Discord when that
            # channel is configured (dayreview.py).
            self.dayreviewer = dayreview_mod.DayReviewer(
                PATHS.LOG_DIR / "jarvis.log", PATHS.LOG_DIR / "turns.jsonl",
                PATHS.REVIEWS_DIR, on_filed=self._on_review_filed,
                study=self._study_table,
                on_week=self._on_week_filed)
            self.dayreviewer.start()
        except Exception:
            log.exception("day reviewer failed to start")
        try:
            # The weekly memory garden (jarvis/garden.py): once the ISO week
            # closes, in the small hours, the week's journal is read by the
            # local model and what it finds is filed as tagged facts. The
            # brain gates are passed as callables so the thread never has to
            # know which brain object is live.
            self.garden = garden_mod.MemoryGarden(
                self.memory, context=self.context, cfg=self.assistant,
                extract=brain_mod.extract_facts,
                busy=lambda: bool(getattr(self.brain, "is_busy", False)),
                lent=brain_mod.is_lent)
            if self.garden.enabled:
                self.garden.start()
        except Exception:
            log.exception("memory garden failed to start")
        cal = getattr(self.services, "calendar", None)
        if cal is not None:
            try:
                cal.start()
            except Exception:
                log.exception("calendar refresh start failed")
        wd = getattr(self.services, "health_watchdog", None)
        if wd is not None:
            try:
                # The fault lane's spoken-once state file (jarvis/faults.py):
                # the watchdog's own latches die with the process, so a
                # restart into a still-tight pool would announce the same
                # episode a second time. Wired HERE rather than in
                # make_tools so a test that builds the tools does not start
                # writing state.
                wd._faults = FaultLog()
            except Exception:
                log.exception("fault state unavailable; alerts will repeat")
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
        for name, obj in (("presence", self.presence), ("desk", self.desk),
                          ("quiet", self.quiet),
                          # arc after those: its first tick should see the
                          # real quiet reason and presence state, not the
                          # "unknown" a sentinel reports before its first probe.
                          ("arc", self.arc), ("roomtone", self.roomtone),
                          ("mixer", self.mixer)):
            if obj is None:
                continue
            try:
                obj.start()
            except Exception:
                log.exception("%s failed to start", name)
        # The price of admission for touching the desktop's light: a run
        # that died holding the display at 0.55 (or warm at 1900 K) heals
        # here, exactly as autostart.disable_gnome_suspend re-asserts its
        # own setting at every start.
        if self.room_light is not None and self.room_light.changed:
            try:
                # healing=True: a restart at two in the morning must not
                # light the room back up while the wind-down still holds
                # it -- the very decision winddown.restore(expired_only=
                # True) made one screenful earlier. The baseline stays on
                # disk for the next "good morning"/"lights up", which pass
                # healing=False and always win.
                self.room_light.restore(healing=True)
                log.info("room: a previous run's display change was restored")
            except Exception:
                log.exception("room light restore at boot failed")
        # The pre-class dossier is built BEFORE the meeting heads-up so the
        # heads-up can stand down for the events it speaks for: both fire at
        # T-lead, and "BIOSENSORS in ten minutes" on top of the dossier would
        # say the title twice.
        try:
            from jarvis.dossier import ClassDossier
            self.dossier = ClassDossier(
                self.assistant, self.timekeeper,
                get_calendar=lambda: getattr(self.services, "calendar", None),
                services=self.services,
                lead_min=int(self.assistant.get("dossier.lead_min", 10) or 10),
                state_path=PATHS.MEMORY_DIR / "dossier_state.json")
            if self.timekeeper is not None:
                self.dossier.start()
        except Exception:
            log.exception("class dossier failed to start")
        try:
            from jarvis.headsup import MeetingHeadsUp
            lead = int(self.assistant.get("calendar.heads_up_min", 10) or 10)
            dossier = getattr(self, "dossier", None)
            self.headsup = MeetingHeadsUp(lambda: getattr(self.services, "calendar", None),
                                          self.timekeeper, lead_min=lead,
                                          state_path=PATHS.MEMORY_DIR / "headsup_state.json",
                                          skip=None if dossier is None else dossier.owns)
            if self.timekeeper is not None:
                self.headsup.start()
        except Exception:
            log.exception("meeting heads-up failed to start")
        try:
            from jarvis.classflow import ClassStager
            self.classflow = ClassStager(
                self.assistant,
                get_calendar=lambda: getattr(self.services, "calendar", None),
                services=self.services,
                get_commander=lambda: self.commander,
                state_path=PATHS.MEMORY_DIR / "classflow_state.json",
                turns_path=PATHS.LOG_DIR / "turns.jsonl")
            self.classflow.start()
        except Exception:
            log.exception("class stager failed to start")
        try:
            from jarvis.deadlines import DeadlineHeadsUp
            hours = self.assistant.get("canvas.heads_up_hours", 3)
            self.deadlines = DeadlineHeadsUp(
                self.assistant, self.timekeeper, lead_hours=hours,
                state_path=PATHS.MEMORY_DIR / "deadlines_state.json",
                get_calendar=lambda: getattr(self.services, "calendar", None))
            if self.timekeeper is not None:
                self.deadlines.start()
            # Read ONLY through snapshot(): jarvis/aside.py and
            # jarvis/objections.py both run inside a spoken turn, where a
            # live canvas.fetch_due would put the network on the reply path.
            self.services.deadlines = self.deadlines
        except Exception:
            log.exception("deadline heads-up failed to start")
        # The three pollers of jarvis/watchers.py. Each is dark-safe (no
        # token / no mailbox / no keywords -> a tick that touches nothing),
        # so they start unconditionally and answer to their watch.* switch.
        try:
            from jarvis.grades import GradeWatch
            self.gradewatch = GradeWatch(
                self.assistant, announce=self._announce,
                state_path=PATHS.MEMORY_DIR / "grades_state.json")
            self.gradewatch.start()
        except Exception:
            log.exception("grade watch failed to start")
        try:
            from jarvis.mailwatch import PeopleMailHeadsUp
            self.mailwatch = PeopleMailHeadsUp(
                self.assistant, people=lambda: self.memory.people(),
                announce=self._announce,
                state_path=PATHS.MEMORY_DIR / "mailwatch_state.json")
            self.mailwatch.start()
        except Exception:
            log.exception("people mail heads-up failed to start")
        try:
            from jarvis.keyword_watch import KeywordWatch
            self.keyword_watch = KeywordWatch(
                self.assistant, announce=self._announce,
                state_path=PATHS.MEMORY_DIR / "keyword_watch_state.json")
            self.keyword_watch.start()
        except Exception:
            log.exception("keyword watch failed to start")
        try:
            from jarvis.aside import AsideEngine
            self.aside = AsideEngine(
                cfg=self.assistant,
                get_calendar=lambda: getattr(self.services, "calendar", None),
                get_deadlines=lambda: getattr(self.services, "deadlines", None),
                quiet=self.quiet,
                state_path=PATHS.MEMORY_DIR / "aside_state.json",
                # The keys the OTHER two doors have already filed a spoken
                # reminder under: whatever is in here is not volunteered.
                filed_paths=(PATHS.MEMORY_DIR / "deadlines_state.json",
                             PATHS.MEMORY_DIR / "headsup_state.json"))
            self.services.aside = self.aside
        except Exception:
            log.exception("aside engine failed to start")
        try:
            from jarvis.debrief import DebriefWatch
            self.debrief = DebriefWatch(
                cfg=self.assistant,
                get_calendar=lambda: getattr(self.services, "calendar", None),
                quiet=self.quiet, presence=self.presence,
                state_path=PATHS.MEMORY_DIR / "debrief_state.json",
                on_candidate=self._ask_debrief)
            self.debrief.start()
        except Exception:
            log.exception("debrief watch failed to start")
        try:
            from jarvis.leavetime import LeadTable, LeaveTimes
            self.leavetime = LeaveTimes(
                lambda: getattr(self.services, "calendar", None),
                self.timekeeper, LeadTable(self.memory),
                ask=self._ask_leave_time, cfg=self.assistant, quiet=self.quiet,
                state_path=PATHS.MEMORY_DIR / "leavetime_state.json")
            self.services.leavetime = self.leavetime
            if self.timekeeper is not None:
                self.leavetime.start()
        except Exception:
            log.exception("leave-time heads-up failed to start")
        try:
            from jarvis.calwatch import CalendarWatch
            self.calwatch = CalendarWatch(
                lambda: getattr(self.services, "calendar", None),
                say=self._say, quiet=self.quiet, cfg=self.assistant,
                state_path=PATHS.MEMORY_DIR / "calwatch_state.json")
            self.calwatch.start()
        except Exception:
            log.exception("calendar anomaly watch failed to start")
        try:
            # The sink sentinel (jarvis/soundbar.py). `speaking` is tts.busy
            # rather than is_speaking: the sink must not move while a burst
            # is still queued either, or the rest of the sentence arrives in
            # a different speaker.
            from jarvis.soundbar import SoundbarSentinel
            self.soundbar = SoundbarSentinel(
                cfg=self.assistant, say=self._say, quiet=self.quiet,
                state_path=PATHS.MEMORY_DIR / "soundbar_state.json",
                speaking=lambda: bool(getattr(self.tts, "busy", False)))
            self.services.soundbar = self.soundbar
            self.soundbar.start()
        except Exception:
            log.exception("sink sentinel failed to start")
        try:
            # The flashcard deck, wired ONCE and shared: the commander built
            # its own lazily (commander._quiz_store) and the briefing read
            # services.flashcards, which nothing ever set -- so the exam-week
            # study section could never see the cards "quiz me" had filed.
            from jarvis.tools.quiz import FlashcardStore
            self.services.flashcards = FlashcardStore()
        except Exception:
            log.exception("flashcard deck unavailable")
        try:
            # Nightly flashcards from his lecture notes (jarvis/studycards.py).
            # The brain gates are callables so the thread never imports it,
            # and the pass is skipped -- never queued -- while the GPU is
            # lent or the model is answering him.
            from jarvis.studycards import NightlyCards
            self.studycards = NightlyCards(
                cfg=self.assistant,
                store=getattr(self.services, "flashcards", None),
                make_quiz=brain_mod.make_quiz,
                busy=lambda: bool(getattr(self.brain, "is_busy", False)),
                lent=brain_mod.is_lent,
                state_path=PATHS.MEMORY_DIR / "studycards_state.json")
            self.studycards.start()
        except Exception:
            log.exception("nightly flashcards failed to start")
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
        # Before the members: a departure confirm pending on a Timer would
        # otherwise fire minutes into the teardown and touch a stopped room.
        self._cancel_departure()
        # Same hazard, other slot: a dissent offer's 60 s Timer RUNS the
        # deferred action and speaks it, so it must not outlive the room.
        # Clearing the slot is the belt for a Timer already past cancel():
        # Commander.objection_timeout returns None on an empty slot.
        try:
            self.commander._objection_cancel_timer()
            self.commander._pending_objection = None
        except Exception:
            log.debug("objection timer not cancelled", exc_info=True)
        cal = getattr(self.services, "calendar", None)
        for name, obj in (("discord", self.discord), ("approvals", self.approvals),
                          ("cmdsock", getattr(self, "cmdsock", None)),
                          ("webapp", getattr(self, "webapp", None)),
                          ("timekeeper", self.timekeeper), ("calendar", cal),
                          ("claude", self.claude),
                          ("health_watchdog", getattr(self.services, "health_watchdog", None)),
                          ("activity_sampler", getattr(self.services, "activity_sampler", None)),
                          ("headsup", getattr(self, "headsup", None)),
                          ("deadlines", getattr(self, "deadlines", None)),
                          ("gradewatch", getattr(self, "gradewatch", None)),
                          ("mailwatch", getattr(self, "mailwatch", None)),
                          ("keyword_watch", getattr(self, "keyword_watch", None)),
                          ("debrief", getattr(self, "debrief", None)),
                          ("leavetime", getattr(self, "leavetime", None)),
                          ("calwatch", getattr(self, "calwatch", None)),
                          ("soundbar", getattr(self, "soundbar", None)),
                          ("studycards", getattr(self, "studycards", None)),
                          ("dossier", getattr(self, "dossier", None)),
                          # stop() also puts the music back and closes any
                          # auto-armed lecture notes: quit must not leave
                          # the desk staged.
                          ("classflow", getattr(self, "classflow", None)),
                          ("focus", getattr(self, "focus", None)),
                          ("winddown", getattr(self, "winddown", None)),
                          ("presence", getattr(self, "presence", None)),
                          ("desk", getattr(self, "desk", None)),
                          ("quiet", getattr(self, "quiet", None)),
                          # roomtone first of the pair: its stop() takes the
                          # paplay stream down, and a bed left playing over a
                          # stopped app is the one failure you can hear.
                          ("roomtone", getattr(self, "roomtone", None)),
                          ("arc", getattr(self, "arc", None)),
                          # mixer stop() restores every stream it ducked
                          ("mixer", getattr(self, "mixer", None)),
                          ("dayreviewer", getattr(self, "dayreviewer", None)),
                          ("board_feed", getattr(self, "_board_feed", None)),
                          ("garden", getattr(self, "garden", None))):
            if obj is None:
                continue
            fn = getattr(obj, "stop", None) or getattr(obj, "close", None)
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                log.exception("assistant: %s failed to stop", name)
        # Never quit holding his display: a Jarvis that is not running
        # cannot be asked for the lights back.
        light = getattr(self, "room_light", None)
        if light is not None and light.changed:
            try:
                # ...unless the wind-down deliberately has it: WindDown.stop()
                # does not brighten the room either, and a quit at midnight
                # that did would be the same 2 a.m. floodlight as the boot heal.
                light.restore(healing=True)
            except Exception:
                log.exception("room light restore at quit failed")
        try:
            # The banner gate holds a reference to this policy; a stopped
            # app (or a test's teardown) must not keep gating banners.
            from jarvis.channels import notify
            notify.set_quiet_gate(None)
        except Exception:
            log.debug("quiet gate not cleared", exc_info=True)
        for obj in (self.notes, getattr(self.commander, "_flashcards", None),
                    # the shared deck start_assistant opened (the commander's
                    # lazy one above stays for a services namespace without it)
                    getattr(self.services, "flashcards", None)):
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
            # The console's ambient / standby surfaces. room_state is called
            # on the window's EXISTING 5 s off-Tk-thread probe (the one that
            # already runs nvidia-smi), never on the Tk thread; desk_idle_s
            # is resolved defensively so it lands whichever way the
            # desk-presence change merges (jarvis/ui/console_mode.py).
            room_state=self.room_state,
            # The Board's own WM close button: without this the window
            # manager's X tore down the toplevel while the app still
            # believed the Board was up, so its feed kept polling. The UI
            # Services dataclass declares the field; build_ui_services drops
            # what it does not, so passing it is safe in either merge order.
            board_closed=self._board_hide,
            # ONE seam for the desk reading: services.desk_idle_s, the
            # DeskSentinel's cached poll. This used to read the name off
            # `self`, where it has never existed, so Services.desk_idle_s
            # was always None and console_mode.resolve_idle_fn fell all the
            # way through to its XScreenSaver probe -- the console's idle
            # clock silently diverged from the rest of the app. Handed over
            # only while the sentinel is actually enabled: a disabled
            # sentinel answers None forever, and resolve_idle_fn accepts
            # ANY callable, which would kill the fallback instead.
            desk_idle_s=(getattr(self.services, "desk_idle_s", None)
                         if getattr(getattr(self, "desk", None), "enabled", False)
                         else None),
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
