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
import contextlib
import threading
import uuid
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from jarvis import gateledger
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
    RoomChanged,
    Status,
    Transcribed,
    UncertainResolved,
    UncertainUtterance,
    UserUtterance,
    bus,
    SpeakingState)
from jarvis.logs import get_logger

from jarvis import address as address_mod
from jarvis import arc as arc_mod
from jarvis import board as board_mod
from jarvis import castview as castview_mod
from jarvis import procnet
from jarvis import brain as brain_mod
from jarvis import debrief as debrief_mod
from jarvis import arrival as arrival_mod
from jarvis import desktop as desktop_mod
from jarvis import earcons
from jarvis import enrolentry
from jarvis import gate as gate_mod
from jarvis import honorific as honorific_mod
from jarvis import passphrase as pp
from jarvis import identity as identity_mod
from jarvis import knightfall_weekly as knightfall_weekly_mod
from jarvis import scope as scope_mod
from jarvis import selfstate, speak_queue, standup, voice_check
from jarvis import leavetime as leavetime_mod
# The one module in the package that may call the mail transport
# (tests/test_send_file.py pins it); Knightfall's rotated code goes out
# through outbox.send_notice.
from jarvis import outbox
from jarvis import relaunch
from jarvis import vocab as vocab_mod
from jarvis.assistant_config import AssistantConfig
from jarvis.turnclock import TurnLedger
from jarvis.previewprobe import PATH_GREEDY, PATH_SPECULATIVE, PreviewProbe
from jarvis import dayreview as dayreview_mod
from jarvis import garden as garden_mod
from jarvis.dialogue import SESSION_WINDOW_S
from jarvis.faults import FaultBoard, FaultLog
from jarvis.brain import JarvisBrain
from jarvis.commander import (BRIEFING_OFFER_TTL_S, COURTESY_BY_REGISTER,
                              COURTESY_REPLIES, DESTRUCTIVE_TTL_S,
                              LEAVE_ANSWER_WINDOW_S, REGISTER_LINES,
                              CommandResult, Commander, parse_yes_no,
                              strip_jarvis_prefix)
from jarvis.context import ContextEngine, _git
from jarvis.endpoint import strip_fillers
from jarvis.history import TypedHistory
from jarvis.relaunch import format_code_status  # noqa: F401 - the drawer's line, re-exported
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
from jarvis.transcriber import Transcriber, shown_words
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
#
# 2026-08-31: 4.5 s is now only the FLOOR and the cold-start default. Every
# tuning above was done against a box whose median wait was 10.68 s, so a
# fixed 4.5 s no longer means "unusually slow", it means "the tail of
# normal". Three of Hunter's reports are that collision:
#   21:02:13 calendar  filler at +4.50 s, answer READY at +5.15 s and queued
#                      behind it -- spoken at +6.92 s, 1.77 s late (#144)
#   21:22:36 Claude MD answer at +0.54 s, filler at +4.50 s -- 3.97 s AFTER
#                      the answer, the moment its audio ended (#89)
#   21:36:36 Oracle    answer at +1.20 s, filler at +5.57 s -- "had the answer
#                      then said checking one moment sir" (#154)
# So the delay now scales with what this box actually costs per turn: speak
# only once a turn is FILLER_WAIT_MULTIPLE times the recent median wait.
THINKING_DELAY_S = 4.5
# x3, from the 105 answered turns in turns.jsonl for 2026-08-31: median wait
# 2.07 s, p90 4.45 s, max 24.6 s. x3 puts the filler at 6.2 s -- clear of the
# p90 and of the 5.15 s calendar answer it used to trample, still ten seconds
# ahead of the 16.5 s mail lookup it exists for. Over that day it would have
# spoken on 4 of 105 turns instead of 10, and the six it drops are the ones
# that were about to answer anyway.
FILLER_WAIT_MULTIPLE = 3.0
# ...and never later than this, or a genuinely stuck turn is just silence.
FILLER_DELAY_MAX_S = 8.0

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
# The SECOND garbled clip in a row. Distinct from SAY_AGAIN_LINE because it
# is not an invitation: the mic is not re-opened, so it must tell him the
# turn is over and that a wake word is how he retries.
NOT_CAUGHT_LINE = "I did not catch that, sir. Do try me again."
# The wake-word refusal, and the gate's, are ONE constant on purpose
# (jarvis/gate.py). "I only answer to Hunter, sir." named no way back in --
# that was the whole complaint -- and named him to a stranger besides.
GUEST_LINE = gate_mod.UNKNOWN_LINE
GUEST_PHRASE_LINE = gate_mod.UNKNOWN_PHRASE_LINE
# What _gate_rescue hands back when the dropped clip WAS the spoken phrase:
# the gate answered it, the window is open, and _process_audio must publish
# neither a rejection nor a transcript (Knightfall, 2026-09-04).
PHRASE_CONSUMED = object()
# The legs on which a clip the speaker filter dropped is let through after
# all: the camera, the phrase's window, the code's window.
RESCUE_HOWS = (gate_mod.HOW_FACE, gate_mod.HOW_GRANT, gate_mod.HOW_CODE)
# ...and the two of those he opened HIMSELF. See _gate_rescue_inner: these
# act in shadow, the camera does not.
WINDOW_HOWS = gate_mod.WINDOW_HOWS
# The typed code (knightfall_code / knightfall_new_code below). The mail
# carries two lines: the code, then this sentence. The Status lines never
# carry a code.
KNIGHTFALL_SUBJECT = "Knightfall"
KNIGHTFALL_BODY_LINE = "Typed only, never spoken. This replaces the old one."
KNIGHTFALL_OK_LINE = "Knightfall accepted, sir; a new code is in your inbox."
KNIGHTFALL_NEW_OK_LINE = "Knightfall: a new code is in your inbox."
KNIGHTFALL_COOLDOWN_LINE = "Knightfall: one code a minute, sir."
# The bootstrap with an empty people book. MEASURED on :93 at S=2
# (2026-09-05): the line this replaced was 1541 px of text in a toast strip
# that holds 894 px at his window, so widgets.ellipsize cut it mid-word --
# "...nobody is enrolled as an owner yet (scri" -- which is the "knightfall
# text doesnt fit in its slot" he reported. The strip is ONE line by design
# (widgets.Toast.STRIP_H), so the fix is a line that fits: 580 px, whole at
# 920x1440 as well. The state it describes is CORRECT and unchanged --
# Knightfall cannot work until somebody is enrolled as an owner -- and the
# command that fixes it goes to the log, which is where a line too long for
# the strip belongs (docs/assistant-setup.md, "If it does not work").
KNIGHTFALL_NO_OWNER_LINE = "Knightfall: enrol an owner first, sir."
KNIGHTFALL_NO_OWNER_LOG = ("knightfall: nobody is enrolled as an owner yet; "
                           "run scripts/jarvis_people.py add <you> "
                           "--role owner")
KNIGHTFALL_COOLDOWN_S = 60.0
# The rotation's failure lines carry the exception's TYPE, never its words,
# because that text came from a transport that had just been handed a code.
# This one is different in a way worth writing down: it is raised by
# mail.notice_destination BEFORE anything reaches a transport, so it can be
# a sentence he can act on. It is still a CONSTANT rather than str(exc), so
# there is no channel from an exception's words to the toast -- and it names
# no address, because the toast is on screen.
KNIGHTFALL_BAD_DESTINATION = ("the notice address in your config is not an "
                              "email address")
# ONE ROTATION AT A TIME. check -> open -> mail -> store is not atomic, and
# the drawer runs each press on its own thread: two presses measured
# (verdict, 2026-09-05) both passed the check against the OLD hash, both
# mailed, and both said "a new code is in your inbox" -- one of which was
# dead on arrival, with nothing to say which. Process-wide because there is
# one keyboard and one drawer; held across the send, so the second press
# reads the state the first one left rather than the state it found.
_KNIGHTFALL_LOCK = threading.RLock()

# ---------------------------------------------- the people book's own lock
# HOW LONG A TYPED OVERRIDE CODE IS GOOD FOR, and it lives HERE rather than
# in the users page because the page is not the guard. The page kept its own
# 120-second dwell and decided from it; a decision made in the UI is not a
# decision at all, because the UI is not what performs the write.
#
# A dwell rather than a prompt per press: the terminal tool authorises ONCE
# per invocation and then acts, and asking on every click trains him to type
# a break-glass code constantly -- one more exposure on his display each
# time. It is re-armed by each successful write and dropped the moment he
# leaves the tab.
PEOPLE_UNLOCK_S = 120.0
# One keyboard, one people book. Held across read-decide-write so two
# presses on two threads cannot both pass a check made before either wrote.
_PEOPLE_LOCK = threading.RLock()
# The first-wake briefing OFFERS itself (Hunter, 2026-09-02: "He should
# offer").
#
# No hour word in it, though the MODEL is still asked for the briefing of
# the hour (_deliver_first_wake_briefing). Jarvis must not say a phrase his
# own grammar refuses: measured against a real Commander, "run my afternoon
# briefing" reaches nothing at all (_BRIEFING_RX, commander.py, has no arc
# words) and "my evening briefing" is _PREVIEW_RX -- TOMORROW's preview,
# the wrong day. All three real incidents landed between 14:29 and 15:00,
# so "your afternoon briefing" is the wording he would have echoed back to
# silence. "my briefing" and "run my briefing" both reach _h_briefing, so
# that is what he is offered.
BRIEFING_OFFER_LINE = "Shall I run your briefing, sir?"
# The arrival catch-up's mail window (_unread_count, _deliver_arrival_catch_up).
# 24 h is mailwatch.SINCE_HOURS' argument, unchanged: an unread mail older
# than a day is not news to be met at the door with, and a homecoming after
# a fortnight away should not be answered with "you've 340 unread emails".
# The limit is the fetch cap, so the COUNT saturates there rather than
# growing without bound -- "25" is honest for anything at or above it and a
# man with 300 unread does not want the true number read out either.
ARRIVAL_MAIL_HOURS = 24
ARRIVAL_MAIL_LIMIT = 25
# HOW LATE A DOORSTEP QUESTION MAY STILL BE ASKED, and it is ENFORCED, not
# quoted: _arrival_catch_up_stale drops the digest past this. The catch-up
# reads the mailbox on a worker, so without a line like this one the only
# thing anybody could say about how late the question can arrive is the
# socket timeout mail.py hands imaplib -- and a SOCKET timeout bounds one
# blocking call, not a mailbox, not a fetch and certainly not the step. It
# was quoted as a worst case three times in this branch and it never was
# one. This IS one, because it is a check and not a claim -- read
# immediately before the words, with nothing between the two that waits on
# anything (the take, a thin, a park, a publish; no I/O, and the only lock
# is quiet.py's own, never held across a socket). Measured at 0.17 ms in
# the harness (tests/test_arrival_app.py's stand-ins, not the field).
#
# The number itself is a JUDGEMENT and has not been measured against
# anything: past about this long the greeting is over, and a mail count
# arriving on its own is an interjection rather than the second half of a
# homecoming. Nothing is lost by the drop -- the mail is still unread, the
# fault is still on the board, and the quiet-hours backlog is still held
# (see speak_catch_up: it is not even taken until this has passed).
ARRIVAL_CATCH_UP_LATENESS_S = 10.0
TURN_TIMEOUT_S = 60.0           # watchdog: a lost reply must not wedge the turn
# The SAME rule for the other flag, and it is not the same number. This one
# bounds a Whisper decode plus the routing behind it, and a long clip on
# this box's CPU path measures ~20 s, so the bar is well clear of an
# honest slow turn and still short enough that he does not sit there
# saying the wake word into a dead microphone. MEASURED 2026-09-06: he
# was locked out for 62 minutes because nothing bounded this at all.
AUDIO_TIMEOUT_S = 90.0          # watchdog: a hung decode must not wedge the mic


def _stuck_thread_stack(thread) -> list:
    """The stack of ``thread`` as it stands NOW, one line per frame, innermost
    last -- or one line saying why there is none. Pure and never raises: a
    diagnostic that can itself fail inside a watchdog is worse than none.

    ``sys._current_frames()`` is the whole trick: it needs no ptrace and no
    root, only the thread's ident, so it works under the account that could
    not run py-spy. The frame is a snapshot and may be a few instructions
    stale by the time it is formatted; for "what is it waiting on" that is
    more than enough.
    """
    import sys
    import traceback
    if thread is None:
        return ["no decode thread recorded"]
    try:
        if not thread.is_alive():
            return ["the decode thread has already exited (name=%s)"
                    % getattr(thread, "name", "?")]
        frame = sys._current_frames().get(thread.ident)
        if frame is None:
            return ["the decode thread is alive but has no frame (ident=%r)"
                    % thread.ident]
        lines = traceback.format_stack(frame)
        out = ["decode thread %r is standing here (innermost last):"
               % getattr(thread, "name", "?")]
        for chunk in lines[-12:]:
            out.extend(ln.rstrip() for ln in chunk.splitlines() if ln.strip())
        return out
    except Exception as exc:                   # noqa: BLE001 - diagnostic
        return ["the decode thread's stack could not be read: %s"
                % type(exc).__name__]

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
# The number lives in jarvis/arrival.py, which owns the choreography and has
# to be able to NAME this gate in a refusal (arrival.greet_refusal); this is
# an alias so the two can never drift apart.
GREET_DAMPER_S = arrival_mod.GREET_DAMPER_S


def _same_clause(a: str, b: str) -> bool:
    """Are these the same sentence, allowing for the full stop?

    Used once, and for one reason: the arrival OFFER speaks the standing
    fault, and the delivery that a "yes" buys must not read it back word
    for word (which is exactly what it did before 2026-09-03).
    """
    def norm(text) -> str:
        return " ".join(str(text or "").split()).rstrip(".!?").casefold()
    return bool(norm(a)) and norm(a) == norm(b)


def yes_no(text: str):
    """True / False for an approval answer, None when the text is neither
    ("yes", "allow it", "no thanks", "deny"). Used for Discord replies to a
    pending permission question (spec 8.2).

    A filled pause comes off first (jarvis.endpoint.strip_fillers): the
    four-word cap below is there to refuse a sentence, and "uh" is not a
    word of the answer -- padding the count with one turned a four-word
    yes into no answer at all."""
    text = strip_fillers(text)
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
    # The reading _gate_admits installs for an admitted VOICE turn, read
    # back by _process_audio one statement later on that same thread.
    _gate_addressee = scope_mod.OWNER

    def __init__(self):
        # ---- assistant config first: everything below reads it ------------
        self.assistant = AssistantConfig.load()
        # THE ONE WRITE. load() is a read everywhere (a script, jarvis-breeze,
        # a test, an agent's import); the app is the process that owns the
        # file, so it alone creates it, recreates a corrupt one, tightens
        # the mode and fills in keys DEFAULTS has gained -- once, here.
        try:
            self.assistant.ensure_defaults()
        except Exception:
            log.exception("assistant config ensure_defaults failed; "
                          "continuing with the in-memory copy")
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
        self._restarting = False
        # The commit this PROCESS started from, read once here (at startup,
        # never at import: a module-level read would stamp whatever tree the
        # test runner or a --help happened to import from). The drawer's
        # code-status line compares it with HEAD on disk at open time.
        self.running_commit = running_commit()
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
        # ...and the model SETTINGS (brain.* -- window, generation budget,
        # temperature, think, the guard) are handed over from the config
        # already loaded above: brain.py reads no config of its own, at
        # import or later (see brain.settings()).
        try:
            brain_mod.configure(self.assistant.local_model,
                                config=self.assistant)
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

        # ---- offline mode: the ONE sensing authority (jarvis/sensing.py) --
        # BEFORE presence, because the room sensor is built inside the
        # sentinel and has to be handed this policy at construction: the
        # enforcement point is the sensor's own read(), not a check in a
        # consumer. On a state file that cannot be read this comes up
        # OFFLINE by design -- his ruling, and the reason the switch does
        # not live in assistant.json, whose loader recreates a corrupt file
        # from DEFAULTS and would therefore fail ONLINE.
        self.sensing = self._construct("sensing", self._make_sensing)

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
        # owner_label: so his migrated gallery label and his voiceprint are
        # read as ONE pool by the per-window filter (speaker.MIGRATED_ALIAS_COSINE).
        self.speaker = SpeakerVerifier(
            gpu=0, threshold=CONFIG.speaker_threshold,
            owner_label=identity_mod.owner_label(self.assistant))
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
        # music_playing= lets that gate relax while Spotify is on: over a
        # vocalist his own "Jarvis" scored 0.135 against the 0.25 bar and got
        # the guest line (2026-09-01).  It is a bound method, not the mixer's,
        # because the mixer is built before this and swapped in tests.
        self.hotword = Hotword(self.arbiter, self._mic_index, self._on_hotword,
                               speaker=self.speaker, on_guest=self._on_guest,
                               music_playing=self._music_playing)

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
        # The Windows cast poller's only way in (jarvis/castview.CastRelay
        # through webapp's one gated /api/cast route). Attached here because
        # this is the first line where both objects exist; with it absent
        # the route answers "none" forever, which is the honest degradation
        # for a startup script polling a Jarvis that cannot cast.
        if self.webapp is not None and self.gesture is not None:
            self.webapp.cast_relay = getattr(self.gesture, "relay", None)
        self._register_tools()
        # The mixer was built at 372, before any tool existed; the Connect
        # ducker it asks on every hold alongside the local one (#72) is the
        # Spotify tool, which only lands on services in _register_tools.
        # Without this line the remote duck is dead code and the mixer ducks
        # only what pactl can see -- on this box, the idle librespot pipe.
        # The same handle is where _music_playing's answer comes from.
        if self.mixer is not None:
            self.mixer.set_remote(getattr(self.services, "spotify", None))
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
        # The spelling hold's "an address is owed" word (jarvis/spelling.py,
        # Recorder.note_partial): read by the recorder ONCE per capture, at
        # start(), from the commander's own pending-question slots.
        self.recorder.address_owed_probe = \
            lambda: self._address_owed(self.commander)
        # ...and the count of those cards, for "clear the transcript": the
        # prompt goes into the SAME TranscriptView._approvals dict the
        # Claude approvals do, clear_all keeps it while it is unanswered,
        # and ApprovalService.pending() has never heard of it. Without this
        # the wipe said "Screen's clear, sir" over a card still on the
        # glass (commander._standing_questions).
        self.commander.uncertain_open = self._uncertain_open
        self._pending_uncertain: dict = {}      # request_id -> utterance
        # The turn sequence number of the turn that is waiting on the open
        # "Was that for me?" -- what an unanswered ask may close, and what
        # _on_hotword names when it refuses a wake word meanwhile.
        self._uncertain_turn = None
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
        # The filler's own state, guarded because the timer fires on its own
        # thread while the answer lands on another. _turn_busy is NOT enough:
        # it means "the router has not reported done", and the Oracle turn at
        # 21:36:36 on 2026-08-31 spoke its answer and left the turn open until
        # the 60 s watchdog -- so the filler passed that guard and spoke
        # "Checking right now, sir" 4.4 s after the answer (#154).
        self._filler_lock = threading.Lock()
        self._turn_seq = 0               # bumped per turn; a stale timer is dropped
        self._turn_answered = False      # this turn has already put audio in the queue

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
        # THE THIRD ARRIVAL TRIGGER. RoomChanged is jarvis/roomfabric.py
        # naming the room he is in; the kitchen sits next to his front
        # door, so that room going occupied after a whole-home absence is
        # the door opening, and it beats the phone leg by however long the
        # phone's radio takes to answer an ARP. It runs through the SAME
        # _greet_return as the phone and the desk, so the damper below is
        # shared and the choreography is not written twice.
        bus.subscribe(RoomChanged, self._on_room_changed)
        self._wire_roomtone()
        bus.subscribe(DeskState, self._on_desk)
        self._last_greeted = 0.0        # GREET_DAMPER_S, shared by both probes
        # When he was last seen, for "welcome back from X" (arrival.outing);
        # 0.0 means no recorded departure, which is a plain welcome.
        self._away_since = 0.0
        self._door = arrival_mod.DoorWatch(
            door=str(self.assistant.get("presence.door_room",
                                        arrival_mod.DEFAULT_DOOR_ROOM)
                     if self.assistant is not None
                     else arrival_mod.DEFAULT_DOOR_ROOM))
        self._warn_door_room_names_nothing()
        # HIS DEPARTURE RULE, as an ordered sequence. Pure and silent; see
        # _departure_seq_room.
        self._departure_seq = self._make_departure_seq()
        # A radar whose gates have gone back to 0 sees 0.75 m and reads the
        # room as EMPTY -- measured twice on real hardware, 2026-09-03. The
        # check is a daemon thread that waits before its first read (the
        # readback lies for a while after a device powers up) and is never
        # joined, so the boot does not pay for it.
        self._room_gate_check = self._start_room_sensor_gate_check()

        # THE OTHER HALF OF THE SAME WALK. The kitchen is the door; the
        # office is where he lands, and it is only reachable through the
        # kitchen. The catch-up is armed at the door and delivered here --
        # see jarvis/arrival.DeskWatch and _settle.
        self._desk = arrival_mod.DeskWatch(
            room=str(self.assistant.get("presence.desk_room",
                                        arrival_mod.DEFAULT_DESK_ROOM)
                     if self.assistant is not None
                     else arrival_mod.DEFAULT_DESK_ROOM),
            zone=str(self.assistant.get("presence.desk_zone",
                                        arrival_mod.DEFAULT_DESK_ZONE)
                     if self.assistant is not None
                     else arrival_mod.DEFAULT_DESK_ZONE))

        # The unread count runs on a worker when a mailbox is configured,
        # so the cue is not paid for on the Tk pump. Held only so a test
        # can join it (see _arrival_actions.catch_up).
        self._arrival_catch_up_thread = None

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

    def _make_sensing(self):
        mod = _import_optional("jarvis.sensing")
        if mod is None:
            return None
        return mod.SensingPolicy(cfg=self.assistant,
                                 path=PATHS.MEMORY_DIR / "sensing.json")

    def _make_presence(self):
        mod = _import_optional("jarvis.presence")
        if mod is None:
            return None
        policy = getattr(self, "sensing", None)
        if policy is None:
            # _construct swallows a constructor failure and hands back None,
            # and a governed sensor whose policy is None decides for itself
            # that nobody is stopping it -- the radar would poll on, with the
            # header badge reading SENSING because there is no state to read.
            # A sensor with no owner does not sense: that is the ruling.
            sens = _import_optional("jarvis.sensing")
            policy = None if sens is None else sens.DENIED
        sentinel = mod.PresenceSentinel(self.assistant, policy=policy)
        legs = getattr(sentinel, "legs", None)
        if legs is not None:
            # Guarded on ``legs`` rather than called unconditionally:
            # tests/test_sensing.py drives this method with a partial self
            # (types.SimpleNamespace) on purpose, and that path has no
            # fabric, so no legs, so nothing to wire.
            self._wire_camera_leg(legs)
            self._wire_mic_leg(legs)
        return sentinel

    def _wire_mic_leg(self, legs) -> bool:
        """Attach the mic leg: SECONDS SINCE A TURN, off the turn ledger.

        The voter's cell 6 -- radar on, phone silent past the grace, camera
        unable to look -- reads identically for a latched radar with him
        out and for him sitting still at his desk with his phone asleep. A
        spoken turn is the one fact that tells those apart, and
        ``arrival.departure_ready`` already reads it off the same ledger
        for its own veto. This hands the voter the same number.

        Late-bound on purpose: ``self.turns`` is built by
        ``_wire_turn_clock`` AFTER ``_construct`` builds presence, so the
        leg looks the ledger up at call time (``_mic_leg``) rather than
        capturing an attribute that does not exist yet.
        """
        if legs is None:
            return False
        legs.mic = self._mic_leg
        return True

    def _mic_leg(self):
        """``TurnLedger.idle_s()`` or None. A number, never audio: the
        ledger holds timestamps, and this reads one of them."""
        turns = getattr(self, "turns", None)
        idle = getattr(turns, "idle_s", None)
        if not callable(idle):
            return None
        try:
            return idle()
        except Exception:  # noqa: BLE001 - the ledger must not cost the vote
            return None

    def _wire_camera_leg(self, legs) -> bool:
        """Attach the camera leg and SAY OUT LOUD whether it can answer.

        His ruling, 2026-09-05: "The camera should be the number one
        understanding for if I'm in. Followed by phone connection then
        sensor." The voter obeys that -- a camera that NAMES him ends the
        vote in every one of the 27 cells.

        BUT NOTHING IN THIS TREE EVER ASSIGNS ``services.camera_feed``
        (grep for "camera_feed ="), so ``_eye_leg`` answers BLIND
        unconditionally today, and BLIND is "could not look", which never
        votes. That makes the running voter a TWO-leg voter wearing a
        three-leg name, and the one thing it must not do is claim
        otherwise: this is his number one signal, and he is entitled to
        know it is dark rather than to find out from a missed greeting.
        So the dark case is a WARNING that names the missing wiring and
        says what the vote is actually standing on.

        The slot is wired either way, so the leg goes live the moment
        something finally attaches a feed -- no second restart.
        """
        if legs is None:
            return False
        legs.eye = self._eye_leg
        live = False
        try:
            identity, faces, live = self._eye_leg()
        except Exception:  # noqa: BLE001 - a broken eye is not a boot failure
            live = False
        if live:
            log.info("presence: the camera leg is live -- his number one "
                     "signal can vote")
            return True
        log.warning(
            "presence: HIS NUMBER ONE SIGNAL IS DARK. Nothing on this tree "
            "assigns services.camera_feed, so the camera leg answers "
            '"could not look" to everything and never votes. The verdict is '
            "standing on the phone and the room sensors only, in that "
            "order. It goes live by itself the moment a feed is attached.")
        return False

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

    def _tell_the_model_who_is_here(self, who: str):
        """Name the addressee to the prompt builder, or clear it, and hand
        back THE READING it installed -- ``(name, honorific)``.

        BELT TO THE SWAP'S BRACES. The swap at ``_say`` is the authority
        and would correct a "sir" gemma4 generated anyway; this stops it
        being generated, and stops the prompt telling the model to call a
        woman "he". Anything that is not a KNOWN non-owner clears the
        addressee, which restores the frozen owner prompt byte for byte.
        """
        try:
            gate = getattr(self, "gate", None)
            owner = identity_mod.owner_label(self.assistant)
            person = (honorific_mod.known_person(gate.registry, who)
                      if gate is not None else None)
            if person is None or who == owner or person.role == \
                    identity_mod.ROLE_OWNER:
                return brain_mod.set_addressee("", honorific_mod.SIR_DEFAULT)
            return brain_mod.set_addressee(person.display(), person.honorific)
        except Exception:                          # noqa: BLE001 - never fatal
            log.exception("honorific: the addressee could not be named to "
                          "the model; using the owner prompt")
            try:
                return brain_mod.set_addressee("", honorific_mod.SIR_DEFAULT)
            except Exception:                      # noqa: BLE001
                return scope_mod.OWNER

    def _the_turn_is_his(self):
        """This turn is the OWNER'S: attribute it to nobody, name the owner
        to the prompt builder, and HAND BACK that reading.

        Called for every source the gate does not judge -- the keyboard,
        the command socket, the phone's intercom clip, Discord -- and
        before every proactive ask (the first-wake briefing). The
        round-2 review (09-04) found the attribution written only by the
        voice path and never cleared: one admitted guest turn and every
        typed turn of his, and his own briefing, ran scoped as the guest.
        The scope is per turn now (jarvis/scope.py); this is the owner's
        half of "per turn", and ``_gate_admits`` is the voice half.

        RETURNS THE READING (round-3, 09-05). Clearing the module state and
        then having the commander look it up again put an unbounded wait
        between the two -- measured 198/200 as HIM being refused his own
        notes when a guest was admitted in that window. The caller carries
        this value into every door of the turn instead.
        """
        self._gate_who, self._gate_how = "", ""
        self._gate_who_ts = -1e9
        self._tell_the_model_who_is_here("")
        return scope_mod.OWNER

    def _honorific(self) -> str:
        """"sir", "ma'am" or "" for WHOEVER THE NEXT LINE IS AIMED AT.

        Resolved from the gate's own attribution and the stored, typed
        value on that person's row -- never from a name, never from a
        guess. Falls back to "sir" whenever there is no registry, no
        attribution, or a stale one, which is the state the live app is in
        today: with no people.json the answer is "sir" on every line and
        the swap below is a no-op.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return honorific_mod.SIR_DEFAULT
        try:
            return honorific_mod.for_addressee(
                gate.registry, lambda: self._gate_who,
                lambda: self._gate_who_ts,
                identity_mod.owner_label(self.assistant), time.monotonic())
        except Exception:                          # noqa: BLE001 - never fatal
            log.exception("honorific: could not be resolved; using sir")
            return honorific_mod.SIR_DEFAULT

    def _address_for_addressee(self, text):
        """The line as the CURRENT ADDRESSEE should hear it.

        THE ONE PLACE THE FORM OF ADDRESS CHANGES on the spoken path. The
        ~1,000 "sir" literals in this tree are authored as written and
        rewritten here; not one of them is edited. For the owner
        ``swap_addresses`` returns the input object unchanged, so his line
        is byte-identical and nothing about his voice can regress through
        this door.

        Guarded exactly as _thin_fragments is: a failure here speaks the
        line as written, because a wrong courtesy is a far smaller bug than
        a lost sentence.
        """
        try:
            return address_mod.swap_addresses(text, self._honorific())
        except Exception:                          # noqa: BLE001
            log.exception("honorific: the swap failed; speaking as written")
            return text

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
        #
        # This is also the only place that sees EVERY spoken line, so it is
        # where the slow-answer filler is disarmed. Individual routes used to
        # cancel the timer themselves (_on_stream_sentence, the ack branch of
        # _emit_result) and every route that forgot -- the Oracle tool, a
        # Claude session ack -- got "Checking right now, sir. One moment."
        # on top of an answer the user had already heard.
        self._note_spoke()
        self.tts.speak(self._address_for_addressee(text))

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
        phrases += [SAY_AGAIN_LINE, NUDGE_LINE, NOT_CAUGHT_LINE, self._guest_line]
        # The owner-gate's fixed lines. An unprewarmed refusal is a
        # three-second pause, which is indistinguishable from being
        # ignored -- the very thing this refusal replaces.
        # KNOWN_SCOPE_LINE is a {name} template and is never prewarmed.
        phrases += list(gate_mod.PREWARM_LINES)
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

        def chat(text, force_tool=None, force_args=None, addressee=None):
            # Every keyword the real JarvisBrain.chat accepts must be forwarded
            # here: the commander only ever sees this wrapper, and a keyword it
            # does not take raises TypeError inside the handler, which the
            # dispatcher turns into "Command failed: <name>" for the user.
            # ``addressee`` is the commander's ONE reading of whose turn this
            # is (jarvis/scope.py). Dropped here, a known person's question
            # would reach the model as his turn with every tool offered --
            # and test_app_wiring pins that this wrapper takes every keyword
            # JarvisBrain.chat takes.
            extra = {"force_args": force_args} if force_args is not None else {}
            if addressee is not None:
                extra["addressee"] = addressee
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

        # Grab and throw (jarvis/gesturecast.py): the courier between the
        # camera's hand stage and the cast sinks, and the voice verbs'
        # target. Built here so it is on the same bag the commander and the
        # console read; the commander itself is built AFTER this bag, so it
        # is handed as a resolver. Side-effect free to construct: no probe,
        # no thread, no device -- a test that builds the services opens
        # nothing.
        self.gesture = self._make_gesture()

        return SimpleNamespace(
            gesture=self.gesture,
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
            # Parked by app._offer_first_wake_briefing, answered by
            # Commander._try_briefing_offer -- declared here so the
            # namespace says the slot exists.
            briefing_offer=None,
            # In-app face enrolment (jarvis/enrolrun.py). The OFFER is parked
            # by Commander._h_face_enrol and answered by _try_enrol; the RUN
            # is the live EnrolRun, which parks and unparks itself. Declared
            # here so the namespace says both slots exist -- and so a box
            # with no camera console still answers getattr with None rather
            # than raising on the first "enrol my face".
            enrol_offer=None,
            enrol_run=None,
            # The console's capture thread and its preview lease, published
            # by ui.main_window once the pane is built. None on a headless
            # box, which is what makes the in-app path refuse rather than
            # reach for a camera nobody is holding.
            preview_worker=None,
            preview_lease=None,
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
            # "sir", "ma'am" or "" for whoever is being addressed right now.
            # A CALLABLE, resolved at speak time: commander._speak bypasses
            # _say, and a value captured at build time would be the owner's
            # for the life of the process.
            honorific=self._honorific,
            # a Tier 1 worker thread's answer (explain, quiz): see _async_reply
            reply=self._async_reply,
            # docs.make_tools parks its DocsIndex here for quiz mode
            docs=None,
            # quiet hours / DND and the presence sentinel (commander, tools)
            quiet=self.quiet, presence=self.presence, desk=self.desk,
            # offline mode: the object the spoken switch and the settings
            # drawer both act on, so there is exactly one path that can
            # take a sensor down (jarvis/sensing.py)
            sensing=self.sensing,
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

    # ---------------------------------------------------------- the cast
    def _make_gesture(self):
        """The grab-and-throw courier, or None with the reason logged. A
        gesture must not be able to stop the app from building."""
        try:
            from jarvis import gesturecast
            try:
                fps = float(self.get_option("camera.preview_fps", 6.0) or 6.0)
            except (TypeError, ValueError):
                fps = 6.0
            return gesturecast.GestureCast(
                get_option=self.get_option, set_option=self.set_option,
                commander=lambda: getattr(self, "commander", None),
                spotify=lambda: getattr(getattr(self, "services", None),
                                        "spotify", None),
                capture=lambda: self.context.capture_screen(),
                identity=self._eye_identity,
                # Not proactive: he is at the desk, gesturing or asking.
                speak=lambda text: self._say(text),
                board_show=self._board_show,
                transfer=self._spotify_transfer,
                view_launch=self._rustdesk_view,
                view_stop=self._rustdesk_close,
                view_alive=self._rustdesk_alive,
                view_connected=self._rustdesk_connected,
                view_served=self._hpcomputer_watching,
                view_serve_arm=self._hpcomputer_watch_arm,
                preview_fps=fps)
        except Exception:                          # noqa: BLE001 - optional lane
            log.exception("gesture courier could not be built; the gesture "
                          "stays off")
            return None

    # -- the screen viewer, the one place a RustDesk window is opened -----
    # NOTHING IN THE DESIGN OR TEST SESSION EVER RAN THESE. They are the
    # injected seam jarvis/castview.py refuses to own: that module holds no
    # process spawner at all, so the only way a viewer window can appear is
    # through these two methods, in the running app, on his say-so.
    #
    # THE ONE THING I COULD NOT VERIFY FROM NUMBERS, and he should read it:
    # I do not know whether the RustDesk viewer steals focus when it opens,
    # whether it can be launched minimised, or whether --connect honours a
    # window-state flag. I did not start a session and would not. It is the
    # same class of harm as the 08-26 desktop freeze, so it is his to try
    # once, deliberately, when he is not mid-sentence in something.
    def _rustdesk_view(self, host: str) -> None:
        """Open the Spark's own RustDesk viewer on ``host``. Outbound only.

        A FIXED argument list built here, never a string from anywhere
        else: the host comes from castview's own constant and the flag is a
        literal. There is no shell, so nothing can be interpolated into
        one.
        """
        import shutil                                       # noqa: PLC0415

        binary = shutil.which("rustdesk") or os.path.expanduser(
            "~/.local/bin/rustdesk")
        if not os.path.exists(binary):
            raise OSError("no rustdesk viewer on this box")
        self._close_rustdesk()
        self._rustdesk = subprocess.Popen(          # noqa: S603 - fixed argv
            [binary, "--connect", str(host)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        log.info("cast: opened a viewer on %s", host)

    def _rustdesk_alive(self) -> bool:
        """Is the viewer THIS app started still running?

        ``Popen`` returning is only evidence that the fork worked, and
        jarvis/castview.py may not call a cast landed on that: a viewer
        that cannot reach the host, cannot open a window or dies on a
        missing display is gone within a moment, and this is what notices.
        ``poll()`` is None while the child lives.
        """
        proc = getattr(self, "_rustdesk", None)
        return proc is not None and proc.poll() is None

    # How long the byte counter is sampled over. Long enough that one
    # frame of a desktop stream is unmistakable, short enough to sit
    # inside the settle the sink already waits.
    _STREAM_SAMPLE_S = 0.35

    def _rustdesk_connected(self) -> Optional[bool]:
        """Is the viewer this app started actually RECEIVING a desktop?

        True / False / None, and None means CANNOT TELL -- which
        jarvis/castview.py turns into an honest "I can't say it landed"
        rather than a landing. ``Popen.poll() is None`` is not evidence: a
        RustDesk viewer sitting on an accept-or-password prompt on the
        Windows side is a perfectly live process, and round 2 called that
        a cast (MEASURED).

        THE EVIDENCE IS LOCAL AND IT IS NUMBERS. Two readings out of
        /proc, both about THIS process and no other:

          * an ESTABLISHED TCP socket that this pid owns, to
            castview.HPCOMPUTER_HOST. That is necessary and it is not
            sufficient -- the password prompt holds one open too.
          * bytes actually arriving. ``/proc/<pid>/io``'s ``rchar`` over a
            short window: a viewer painting a desktop pulls hundreds of
            kilobytes a second, one waiting to be let in pulls a
            keepalive. ``castview.VIEWER_STREAM_BPS`` is the floor and it
            is GUESSED between those two orders of magnitude.

        NOTHING HERE OPENS A WINDOW, A CAPTURE DEVICE OR THE PICTURE, and
        nothing leaves this box. It cannot prove a window is visible or on
        the right monitor; it proves a live connection carrying a stream,
        which is as far as local evidence goes. That last step is his.
        """
        proc = getattr(self, "_rustdesk", None)
        if proc is None or proc.poll() is not None:
            return False
        pid = int(proc.pid)
        try:
            if not procnet.has_socket_to(pid, castview_mod.HPCOMPUTER_HOST):
                return False
            first = procnet.rchar(pid)
            if first is None:
                return None
            time.sleep(self._STREAM_SAMPLE_S)
            second = procnet.rchar(pid)
            if second is None:
                return None
        except Exception:                          # noqa: BLE001 - /proc
            log.debug("cast: the connection probe raised", exc_info=True)
            return None
        rate = (second - first) / self._STREAM_SAMPLE_S
        log.info("cast: viewer pid %d is pulling %.0f B/s", pid, rate)
        return rate >= castview_mod.VIEWER_STREAM_BPS

    # How long the byte counter is sampled over on the SERVING side. The
    # same idea as _STREAM_SAMPLE_S and the same GUESS.
    _SERVE_SAMPLE_S = 0.35

    def _hpcomputer_watch_arm(self) -> None:
        """Remember which RustDesk connections from HPCOMPUTER were ALREADY
        there, so the probe below can answer about THIS cast.

        ROUND 5. ``_hpcomputer_watching`` said True for ANY established
        socket from HPCOMPUTER on a screen-sharing port -- including a
        RustDesk session he opened himself an hour earlier, which is a live
        connection carrying a stream and is not a cast Jarvis landed. The
        sink calls this before it parks the verb; anything in this set is
        not evidence for what happens next.

        THE LIMIT, and it is real: inode numbers are reused, and a
        reconnect of his own session inside the same cast would look new.
        This narrows the probe from "he has RustDesk open" to "a connection
        appeared after I asked" and no further. It reads /proc/net and
        nothing else -- no socket is opened, nothing leaves this box.
        """
        try:
            self._serve_seen = set(procnet.established_to(
                castview_mod.HPCOMPUTER_HOST, castview_mod.RUSTDESK_PORTS))
        except Exception:                          # noqa: BLE001 - /proc
            log.debug("cast: the serving baseline could not be read",
                      exc_info=True)
            self._serve_seen = set()
        log.info("cast: %d RustDesk connection(s) from HPCOMPUTER were "
                 "already up before this cast", len(self._serve_seen))

    def _hpcomputer_watching(self) -> Optional[bool]:
        """Is HPCOMPUTER actually PULLING the Spark's screen?

        True / False / None, and None means CANNOT TELL -- which
        jarvis/castview.py turns into an honest "I can't say it landed"
        rather than a landing. This is the Spark -> HPCOMPUTER direction
        and it is NOT the mirror of ``_rustdesk_connected``: there is no
        viewer process on this box to ask about. The viewer runs in HIS
        Windows session; the Spark is the end being VIEWED. So the
        evidence is what arrives here:

          * an ESTABLISHED TCP socket INBOUND from castview.HPCOMPUTER_HOST
            on one of ``castview.RUSTDESK_PORTS``. THE PORT SCOPING IS THE
            WHOLE POINT: the Windows helper's own long poll is also an
            established socket to that host, held open 25 s at a time
            about 2.4 times a minute, so "is there a connection to
            HPCOMPUTER" is true almost always and is worth nothing. The
            poll's own port (webapp's 8765) is deliberately not in that
            tuple.
          * bytes actually leaving over it. The socket alone is not
            enough for the same reason it was not enough in the other
            direction -- a viewer negotiating a password holds one open --
            so the inode is traced back to the process serving it and its
            ``wchar`` is sampled. ``castview.VIEWER_STREAM_BPS`` is the
            floor and it is GUESSED.

        THE HONEST WEAKNESS, stated rather than buried: ``wchar`` is that
        process's WHOLE output, not this socket's. If his RustDesk were
        serving a second viewer at the same moment, this would read that
        traffic too and could say yes to a cast that is not carrying. It
        cannot say yes to a machine that is not connected at all, which is
        the failure this exists to catch, and per-socket byte counters are
        not in /proc -- reading them means opening a netlink socket, which
        this lane does not do.

        NOTHING HERE OPENS A WINDOW, A SOCKET, A CAPTURE DEVICE OR THE
        PICTURE, and nothing leaves this box. It proves a live connection
        carrying a stream. It does NOT prove a window is visible or on his
        middle monitor. That last step is his.
        """
        try:
            inodes = procnet.established_to(castview_mod.HPCOMPUTER_HOST,
                                            castview_mod.RUSTDESK_PORTS)
            # TIED TO THIS CAST, as far as local evidence reaches: a
            # connection that was already up when the cast was asked for is
            # his own session, not this one. With no baseline taken the
            # behaviour is round 4's exactly.
            inodes = set(inodes) - set(getattr(self, "_serve_seen", ()) or ())
            if not inodes:
                return False
            pid = None
            for inode in sorted(inodes):
                pid = procnet.pid_for_inode(inode)
                if pid is not None:
                    break
            if pid is None:
                return None                # not ours to look into
            first = procnet.wchar(pid)
            if first is None:
                return None
            time.sleep(self._SERVE_SAMPLE_S)
            second = procnet.wchar(pid)
            if second is None:
                return None
        except Exception:                          # noqa: BLE001 - /proc
            log.debug("cast: the serving probe raised", exc_info=True)
            return None
        rate = (second - first) / self._SERVE_SAMPLE_S
        log.info("cast: HPCOMPUTER is pulling %.0f B/s from pid %d",
                 rate, pid)
        return rate >= castview_mod.VIEWER_STREAM_BPS

    def _rustdesk_close(self) -> None:
        self._close_rustdesk()

    def _close_rustdesk(self) -> None:
        """Stop ONLY the viewer this app started. A session he opened
        himself is never touched -- which was the first thing the design
        got wrong: killing every rustdesk process would have closed his
        own window."""
        proc = getattr(self, "_rustdesk", None)
        self._rustdesk = None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:                          # noqa: BLE001 - teardown
            log.debug("cast: the viewer would not close", exc_info=True)

    def _eye_identity(self) -> str:
        """The camera's name for whoever is in frame, "" for no opinion.

        Read off the app's OWN feed when it has one (services.camera_feed
        -> .eye, a state with ``identity`` and ``usable()``). Nothing
        attaches a feed on this tree today and camera.identity ships off,
        so this answers "" -- which cast() reads as NO OPINION, never a
        veto and never a match. An irreversible sink that demands a
        positive identity therefore refuses out loud until both exist.
        """
        feed = getattr(getattr(self, "services", None), "camera_feed", None)
        eye = getattr(feed, "eye", None)
        state = getattr(eye, "state", None)
        if callable(state):
            try:
                state = state()
            except Exception:                      # noqa: BLE001 - the eye
                return ""
        usable = getattr(state, "usable", None)
        try:
            if callable(usable) and not usable():
                return ""
        except Exception:                          # noqa: BLE001 - the eye
            return ""
        return str(getattr(state, "identity", "") or "")

    def _eye_leg(self):
        """(identity, faces, live) for the camera leg of the three-leg
        voter. A NAME AND A COUNT, NEVER A FRAME.

        ``live`` is the whole value of this leg and the reason tonight is
        not covered by his rule 1. A camera that is off, inside its curfew,
        or -- as on this box today -- never attached to
        ``services.camera_feed`` at all has NOT "failed to see him". It did
        not look, and its silence is not evidence of an empty room. Only a
        usable feed reporting a face count has actually looked.

        NOTHING IN THIS TREE EVER ASSIGNS ``services.camera_feed`` (grep
        for "camera_feed ="), so this returns ("", None, False) -- BLIND --
        unconditionally today, and the camera leg is a stub with a real
        shape rather than a leg that votes. campreview builds its own feed
        and logs "no services.camera_feed" instead.
        """
        feed = getattr(getattr(self, "services", None), "camera_feed", None)
        eye = getattr(feed, "eye", None)
        state = getattr(eye, "state", None)
        if callable(state):
            try:
                state = state()
            except Exception:                      # noqa: BLE001 - the eye
                return ("", None, False)
        if state is None:
            return ("", None, False)
        usable = getattr(state, "usable", None)
        try:
            live = bool(usable()) if callable(usable) else bool(usable)
        except Exception:                          # noqa: BLE001 - the eye
            return ("", None, False)
        if not live:
            return ("", None, False)
        faces = getattr(state, "faces", None)
        try:
            faces = None if faces is None else int(faces)
        except (TypeError, ValueError):
            faces = None
        return (str(getattr(state, "identity", "") or ""), faces, True)

    def _spotify_transfer(self, device):
        """HpcomputerSink's one working route: a TRACK moves by Spotify's
        own outbound connection (the firewall does not block it)."""
        sp = getattr(getattr(self, "services", None), "spotify", None)
        if sp is None:
            return None
        return sp.control("transfer", value=device)

    # ------------------------------------------------------- brain executor
    def _on_stream_sentence(self, sentence):
        """A sentence of the reply, as the model produces it: speak it now.
        The full reply follows in the tags with a STREAMED marker so it is
        shown, remembered and not spoken again."""
        if getattr(self, "_stream_muted", False):
            return                  # barged in: the rest of this reply is dropped
        # The answer has started, so no "thinking" line is warranted -- and
        # one queued now would play BETWEEN the answer's sentences (the TTS
        # queue is FIFO). Disarm the filler; the watchdog stays. _say latches
        # it too; this is the earlier of the two.
        self._disarm_filler()
        if self._last_source == "voice":
            self._followup_after_speech = True
        self._say(sentence)

    def _on_brain_tags(self, tags):
        """Port of the monolith's _on_brain_response: act on [TAG] tuples.
        A ("BRIEFING", json) tag turns that turn's SPEAK into ONE
        BriefingReady card (no separate JarvisReply) — still spoken."""
        # Whatever else these tags mean, their arrival ends the turn.
        self._turn_finished()
        # ONE reply never says the same sentence twice. #144: the calendar
        # confirmation ("Added hello, Tuesday at 4:30 PM, to your calendar,
        # sir.") was SPOKEN TWICE -- the add_event tool's own confirmation
        # and the model's reply are the same authored sentence (both are in
        # the log at 21:02:18.205), and a batch carrying both spoke both.
        # Scoped to this one batch of tags on purpose: the same line in a
        # later turn still speaks, and nothing outside a model reply -- a
        # greeting, a timer, a reminder -- is touched by this.
        spoken_here: set = set()
        kept = []
        for tag, content in tags:
            if tag in ("SPEAK", "DONE"):
                line = " ".join(str(content or "").split())
                if line and line in spoken_here:
                    log.info("dropped a repeat of a line this reply already "
                             "speaks: %.60s", content)
                    continue
                spoken_here.add(line)
            kept.append((tag, content))
        tags = kept
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

    def _start_room_sensor_gate_check(self):
        """Check, off the boot thread, that each radar still covers its room.

        See jarvis/sensorcheck.py: it is READ-ONLY (it reports the command
        that fixes a mismatch rather than rewriting his device), it asks
        the sensing policy before any socket, and it is NEVER handed that
        policy -- ``SensingPolicy.attach`` replaces by name, and a second
        sensor attaching as "radar" would take the curfew off the real one.
        A failure here costs the check and nothing else.
        """
        try:
            from jarvis import sensorcheck
            return sensorcheck.start(self.assistant,
                                     policy=getattr(self, "sensing", None))
        except Exception:  # noqa: BLE001 - a check may not cost the boot
            log.debug("sensor check: could not be started", exc_info=True)
            return None

    def _warn_door_room_names_nothing(self) -> None:
        """Say so ONCE at startup when ``presence.door_room`` matches no
        configured room.

        A door room that names nothing is silent: the kitchen trigger
        simply never fires and there is no error anywhere to find. Only
        checked when rooms ARE configured -- on a box with no ``rooms``
        list the whole feature is inert by design and a warning every boot
        would be noise.
        """
        try:
            from jarvis import roomfabric
            specs = roomfabric.room_specs(self.assistant)
            if not specs:
                return
            door = arrival_mod._room_key(self._door.door)
            names = [roomfabric._slug(spec.name) for spec in specs]
            if door and door not in names:
                log.warning("arrival: presence.door_room %r matches no "
                            "configured room (%s); the door trigger can "
                            "never fire", self._door.door, ", ".join(names))
        except Exception:  # noqa: BLE001 - a warning may not cost the boot
            log.debug("arrival: could not check the door room", exc_info=True)

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
        if last:
            why = arrival_mod.greet_refusal(source=source, since_s=now - last,
                                            damper_s=GREET_DAMPER_S)
            if why:
                log.info("arrival: no greeting -- %s", why)
                return
        # GREET AT THE DOOR, ASK AT THE DESK. The catch-up is dropped from
        # the door plan and owed to his desk -- but ONLY if some leg can
        # actually deliver it there (_settle_legs). A deferral nothing can
        # fire is the catch-up silently never happening, which is worse
        # than asking him in the hallway.
        legs = self._settle_legs()
        defer = arrival_mod.defer_catch_up(
            legs=legs,
            enabled=bool(self.assistant.get("presence.settle_at_desk", True)))
        steps = arrival_mod.arrival_plan(
            returned=True, home=True, quiet_reason=self._quiet_reason(),
            cue=bool(self.assistant.get("presence.arrival_cue", True)),
            defer_catch_up=defer)
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
        # THIS arrival, for as long as it owns the floor. The catch-up can
        # finish on a worker seconds after run() returns, and a digest
        # belonging to a homecoming that has since been superseded must not
        # be spoken -- see _arrival_catch_up_stale.
        self._arrival_gen = getattr(self, "_arrival_gen", 0) + 1
        done = arrival_mod.run(steps, self._arrival_actions())
        log.info("arrival (%s): %s", source, self._arrival_ledger(done)
                 or "(nothing)")
        # ARMED ONLY BEHIND A GREETING THAT ACTUALLY HAPPENED. A deferred
        # catch-up is the second half of a welcome; with no welcome (the
        # panel-only quiet plan) there is no homecoming to catch up on and
        # the quiet policy already owns the backlog.
        watch = getattr(self, "_desk", None)
        if defer and watch is not None and "greeting" in done:
            watch.arm()
            log.info("arrival (%s): the catch-up is owed at his desk (%s)",
                     source, " and ".join(legs))

    def _arrival_ledger(self, done) -> str:
        """The cue's one log line, and it must not overstate the last step.

        ``arrival.run``'s contract is "the steps that actually ran", and a
        catch-up that went to a worker has only STARTED: it may yet turn
        out to have nothing to say and log so, and the two lines then
        contradicted each other -- the ledger claiming a step that spoke
        nothing. The whole point of the ledger is recording what happened.
        """
        if getattr(self, "_arrival_catch_up_deferred", False):
            done = ["catch-up (started)" if step == "catch-up" else step
                    for step in done]
        return " -> ".join(done)

    # --------------------------------------- ASK AT THE DESK
    def _settle_legs(self) -> tuple:
        """Which settle legs could actually deliver a deferred catch-up.

        Read at the door, and it decides whether the catch-up is deferred
        at all. IT MAY NOT OVERSTATE: a leg counted here that cannot fire
        is his mail question silently never being asked, so each one is
        gated on the thing that would have to publish it and not on the
        feature being switched on.

        * ``radar``  -- something is feeding zone verdicts
          (``services.zone_source``, whose one job is to call
          ``note_zone_verdict``) AND his desk room's zone map actually
          names the desk band. A verdict lane pointed at a room with no
          map, or a map with no such band, never reaches the desk zone.
        * ``camera`` -- a camera feed is attached
          (``services.camera_feed``, which ``_eye_identity`` reads). It
          answers with a NAME or "", so the curfew and offline mode close
          it by answering "" rather than by being asked about here.

        Nothing attaches either on this tree today, so this is ``()`` on
        the live box and the arrival cue is byte for byte the one that
        shipped. Never raises: a config that cannot be read is no leg.
        """
        watch = getattr(self, "_desk", None)
        if watch is None:
            return ()
        services = getattr(self, "services", None)
        legs = []
        if getattr(services, "zone_source", None) is not None and \
                self._desk_band_configured(watch):
            legs.append(arrival_mod.LEG_RADAR)
        if getattr(services, "camera_feed", None) is not None:
            legs.append(arrival_mod.LEG_CAMERA)
        return tuple(legs)

    def _desk_band_configured(self, watch) -> bool:
        """Does his desk room's zone map name the band he sits in?

        ``zones.zone_map_for`` is the only door to that config and it
        answers None for every way the section can be wrong or switched
        off, which is exactly the answer this wants.
        """
        try:
            from jarvis import zones as zones_mod
            zmap = zones_mod.zone_map_for(self.assistant, watch.room)
            if zmap is None:
                return False
            want = arrival_mod._room_key(watch.zone)
            return any(arrival_mod._room_key(band.name) == want
                       for band in zmap.bands)
        except Exception:  # noqa: BLE001 - an unreadable config is no leg
            log.debug("arrival: could not read the desk zone map", exc_info=True)
            return False

    def _settle(self, *, room="", verdict=None, camera=None) -> str:
        """He has settled at his desk: deliver the catch-up he is owed.

        The one door both legs come through, so "one delivery, never two"
        is enforced in one place (``DeskWatch``) rather than negotiated
        between two lanes. Returns the leg that delivered it, or "".

        THE ORDER HERE IS THE FEATURE. The quiet policy is asked BEFORE
        the latch is spent, because a held catch-up must stay OWED --
        ``_say`` takes the digest with ``proactive=False`` and would
        otherwise pierce a quiet hour that the door plan had respected an
        hour earlier. Then the latch, then the step; nothing between the
        latch and the step can decide not to speak except the step itself,
        which puts the backlog back when it does.

        ``camera`` is an identity label or a ``zones.CameraOpinion``.
        There is no argument here a frame could travel through.
        """
        watch = getattr(self, "_desk", None)
        if watch is None or not watch.armed:
            return ""
        reason = self._quiet_reason()
        steps = arrival_mod.settle_plan(quiet_reason=reason)
        if not steps:
            # Asked only once he has ACTUALLY settled, or every passing
            # kitchen reading would log a hold that never applied to it.
            leg = arrival_mod.settle(room=room, verdict=verdict, camera=camera,
                                     desk_room=watch.room, desk_zone=watch.zone)
            if leg:
                log.info("arrival: he has settled at his desk (by the %s) but "
                         "%s; the catch-up stays owed", leg,
                         reason or "the policy is holding")
            return ""
        leg = watch.observe(room=room, verdict=verdict, camera=camera)
        if not leg:
            return ""
        # RE-ARMED IF THE WORKER DROPS IT. A remote mailbox finishes this
        # step on a thread, and _arrival_catch_up_stale can decide by then
        # that a turn owns the floor. At the door that drop cost nothing
        # (the offer was the last word of a cue that had already spoken);
        # here it would be the whole delivery, so the watch takes the arm
        # back and the next reading at his desk asks again.
        done = arrival_mod.run(steps, self._arrival_actions(missed=watch.arm))
        log.info("arrival: he has settled at his desk, by the %s: %s", leg,
                 self._arrival_ledger(done) or "(nothing)")
        return leg

    def note_zone_verdict(self, verdict, room: str = "") -> str:
        """A ``zones.Verdict`` from whatever drives jarvis/zones.py.

        The radar leg's entry point, and the whole of it. Nothing in this
        tree calls it yet -- zones.py is driven by scripts/zone_log.py --
        so the leg is dark until a verdict lane is attached as
        ``services.zone_source``.
        """
        return self._settle(room=room or str(getattr(verdict, "room", "") or ""),
                            verdict=verdict)

    def note_camera_identity(self, label: str, score: float = 0.0,
                             room: str = "") -> str:
        """The eye recognised somebody in the office: a NAME and a SCORE.

        The camera leg's entry point. It takes what ``eye.identify()``
        returns and it will never take anything else: no frame, no crop,
        no embedding. The score is carried so a caller need not decide
        alone what counts -- ``eye.identify`` already returns ``("", 0.0)``
        for every way it declines, so an empty label is the only refusal
        this has to honour.
        """
        watch = getattr(self, "_desk", None)
        where = room or (watch.room if watch is not None
                         else arrival_mod.DEFAULT_DESK_ROOM)
        if score < 0:
            return ""
        return self._settle(room=where, camera=label)

    def _arrival_catch_up_stale(self, gen: int, turn: int,
                                started: float) -> str:
        """Why this late catch-up must NOT be spoken, or "".

        Moving the unread count to a worker fixed the freeze and cost the
        last arrival step its atomicity: the cue returns, and the offer
        would be spoken and parked whenever the mailbox happened to answer.
        So the worker asks, after the fetch and before anything is
        consumed or said, whether the world it was answering is still
        there.

        FAIL CLOSED. Every check below can only make the guard say "drop
        it", and so can the guard's own failure: a question this could not
        vet is a question that does not get asked. It read the other way
        round first -- one bare ``except`` around the lot, falling through
        to "" -- and "" is PERMISSION. A guard whose own breakage grants
        the thing it exists to withhold is not a guard. There is exactly
        one ``return ""`` in this method and it is the last statement of
        the ``try``, so no failure can reach it.

        What is asked, in order:

        * ``started`` -- HOW LATE IS IT? ``ARRIVAL_CATCH_UP_LATENESS_S``,
          and this line is the whole reason the bound is real: the step
          cannot speak later than this because this drops it, whatever the
          mailbox did. No socket timeout is quoted here any more; a socket
          timeout bounds one blocking call and never bounded this.
        * ``turn`` -- ``_dispatch_gen``, bumped by every dispatch. THIS IS
          THE ONE THAT FIRES IN THE FIELD: he asked Jarvis something while
          the mailbox was thinking, and the answer to THAT owns the floor.
          The same counter ``_async_reply`` reads, for the same purpose.
        * the floor right now: a turn still open, a clip being
          transcribed, the mic recording.
        * ``gen`` -- the arrival this digest belongs to, so a digest can
          never outlive the homecoming it describes.

        SILENCE IS THE CORRECT OUTCOME, not a deferral, and it costs
        nothing that is not still there: the mail is still unread, the
        fault is still on the board, and the quiet-hours backlog has not
        even been taken yet when this is asked (speak_catch_up), so a drop
        leaves it held for the policy's own next tick to read out.

        Racy by construction, and knowingly: this and the ``_say`` are not
        atomic, so a turn beginning in the microseconds between them is
        still spoken over. That window is the same one ``_async_reply``
        and ``_after_speech`` live with; closing it needs a lock on the
        floor that this file does not have.
        """
        try:
            late = time.monotonic() - started
            if late > ARRIVAL_CATCH_UP_LATENESS_S:
                return ("it is %.1f s late, past the %.0f s a doorstep "
                        "question gets" % (late, ARRIVAL_CATCH_UP_LATENESS_S))
            if getattr(self, "_arrival_gen", 0) != gen:
                return "a newer arrival owns the cue"
            if getattr(self, "_dispatch_gen", 0) != turn:
                return "he has taken a turn since"
            busy = getattr(self, "_turn_busy", None)
            if busy is not None and busy.is_set():
                return "a turn is still open"
            audio = getattr(self, "_audio_busy", None)
            if audio is not None and audio.is_set():
                return "a clip is being transcribed"
            if getattr(getattr(self, "recorder", None), "recording", False):
                return "the microphone is open"
            return ""
        except Exception:  # noqa: BLE001 - never raise on the worker...
            # ...and never SPEAK on the strength of a broken guard either.
            log.exception("arrival: the catch-up guard failed; dropping it")
            return "the guard could not tell whether it was still wanted"

    def _take_held_fragments(self):
        """The quiet backlog for one attempt at speaking it, and the way back.

        ``(fragments, put_back)``, straight through to
        ``quiet.take_fragments`` -- see there for why a caller that may
        decide not to speak must never use the one-way drain. This wrapper
        exists for the two cases that are not a live policy: no ``quiet``
        wired at all, and a stand-in that only has the one-way primitive.
        The second is DECLARED, once, rather than quietly losing the lines
        it cannot give back: silence about a degraded guarantee is how the
        original bug got shipped.
        """
        quiet = getattr(self, "quiet", None)
        if quiet is None:
            return [], lambda: 0
        take = getattr(quiet, "take_fragments", None)
        if callable(take):
            frags, put_back = take()
            return list(frags), put_back
        if not getattr(self, "_warned_one_way_quiet", False):
            self._warned_one_way_quiet = True
            log.warning("arrival: %s has no take_fragments, so a catch-up "
                        "that is not spoken cannot give the backlog back",
                        type(quiet).__name__)
        return list(quiet.release_fragments()), lambda: 0

    def _arrival_actions(self, missed=None) -> dict:
        """The callables behind jarvis/arrival.ARRIVAL_STEPS.

        A step returning False did nothing (there was no backlog), and
        arrival.run() records only what actually happened -- which is what
        the tests assert on. The one step that can outlive the call is the
        catch-up: see _arrival_ledger for what "happened" means there.

        ``missed`` is called when the catch-up went to a worker and the
        worker then DROPPED it (a turn took the floor, it ran late, a
        newer arrival superseded it) -- never when it spoke and never when
        it had nothing to say. Only the deferred delivery passes one: at
        the door a drop costs the last word of a cue that has already
        spoken, but at his desk it is the whole delivery, so the watch
        takes its arm back and asks again on his next reading.
        """
        # This cue's own mark, cleared at the top so a previous cue's
        # worker cannot label this one's inline step.
        self._arrival_catch_up_deferred = False

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
            # "Welcome back from the dentist, sir" when the calendar can
            # prove it, and the plain line every other time (_welcome_text).
            line = self._welcome_text()
            burst.append(line)
            self._say(line)
            return True

        def speak_catch_up(guard=None):
            # THE OFFER, and it is the only new thing spoken on the
            # doorstep. His ruling after the 40-second monologue of
            # 2026-09-02: the briefing OFFERS, it does not deliver. So this
            # is a count and a question -- never a sender, never a subject
            # -- and it rides the SAME burst as the digest in front of it so
            # the address thinning is still one pass over the whole cue.
            #
            # Returns "" (nothing worth saying), "spoke", or -- only with a
            # ``guard``, which is only the worker path -- the reason it was
            # dropped. Three outcomes rather than a bool because the log
            # line for "there was no backlog" and the one for "it was too
            # late to say it" are not the same sentence, and the ledger
            # reads both.
            offer, major = self._arrival_offer_fragments()
            # THE SLOW READ IS DONE; ASK THE FLOOR BEFORE CONSUMING
            # ANYTHING. Nothing above this line took something that cannot
            # be given back -- the mailbox and the fault board are reads --
            # and nothing below it runs if the answer is no.
            why = guard() if guard is not None else ""
            if why:
                return why
            # THE BACKLOG IS TAKEN HERE, at the last moment, and it is
            # taken REVERSIBLY. Draining it up front and handing it to a
            # worker that might drop the digest is how the first cut of
            # this destroyed the quiet-hours backlog: his missed lines,
            # gone to say nothing with. Two rules hold it now -- nothing is
            # taken until the guard has passed, and every path out of here
            # that does not queue the words puts it back (the `finally`).
            held, put_back = self._take_held_fragments()
            spoke = False
            try:
                frags = list(held) + list(offer)
                if not frags:
                    return ""
                # A fault-only line is a STATEMENT ("The disk is full.") and
                # there is nothing to go through; only a question may be
                # parked, or a "yes" would hang on nothing.
                asks = bool(offer) and arrival_mod.offers_to_read(offer[-1])
                # THE JOIN (7 sentences, 5 sirs measured): "Welcome back,
                # sir." has already addressed him, so the digest's own later
                # vocatives are the ones that go. The digest arrives as
                # FRAGMENTS and is thinned exactly once, here, against the
                # welcome in front of it -- release() would have joined it
                # into one finished string first, and thinning a finished
                # string is the mode that killed the first attempt
                # (jarvis/address.py). Then both shown and spoken, so the
                # card he reads and the voice he hears agree.
                thinned = self._thin_address(burst + frags)
                digest = address_mod.join_thinned(thinned[len(burst):])
                if not digest:
                    return ""
                burst[:] = thinned
                # Parked BEFORE the words go out, exactly as the first-wake
                # offer marks the day before it speaks: a TTS failure must
                # not leave a question on the floor with nothing listening
                # for the answer, and the answer's window opens off this
                # line's falling edge (_after_speech).
                parked = self._park_arrival_offer(said=major) if asks else None
                bus.publish(JarvisReply(text=digest, speak=True))
                try:
                    self._say(digest)
                    spoke = True
                finally:
                    # THE MIC IS ARMED HERE: after _say, and WHATEVER _say
                    # DID. Two failures, and both are real.
                    #
                    # Arming it at the park, before _say had queued a word,
                    # left a gap the welcome's own falling edge could drain
                    # into on the Tk thread -- _after_speech finds the flag
                    # with tts.pending == 0 and opens a mic for a question
                    # nobody has asked. So: after _say, which is safe in
                    # the other direction because _say only ENQUEUES.
                    #
                    # And moving it after _say without this `finally` swaps
                    # that for the failure the park exists to prevent: a
                    # TTS that raises leaves the question PARKED, on the
                    # card (JarvisReply is already out) and answerable --
                    # with no window to answer it in. The park and the arm
                    # are two halves of one invariant and nothing fallible
                    # gets to separate them.
                    if parked is not None:
                        self._followup_after_speech = True
                # ...and the 60 s TTL is measured from HERE, not from the
                # park. A long quiet-hours backlog in front of the question
                # used to eat most of the window he had to answer it.
                self._restamp_offer(parked)
                return "spoke"
            finally:
                # EVERY WAY OUT OF HERE THAT DID NOT SPEAK GIVES THE
                # BACKLOG BACK: nothing to say, nothing left after the
                # thinning, or an exception anywhere in between. A digest
                # that could not be spoken is worth repeating; one that was
                # destroyed is not recoverable. put_back re-arms the quiet
                # policy's falling edge too, so the lines are read out on
                # its own next tick rather than sitting held and mute.
                if not spoke:
                    put_back()

        def finish_off_thread(gen, turn, started):
            # arrival.run() guards each step and logs what ran; a worker has
            # no such parent, so both come with it. A catch-up that died on
            # the way to the speaker belongs in the log, not in a dead
            # thread -- and so does one that was still worth saying when it
            # was asked for and was not by the time it could be said.
            try:
                outcome = speak_catch_up(
                    guard=lambda: self._arrival_catch_up_stale(gen, turn,
                                                               started))
                if outcome == "spoke":
                    return
                if outcome:
                    # SILENCE IS THE RIGHT OUTCOME FOR A STALE DIGEST, and
                    # it costs nothing that is not still there: the mail is
                    # still unread, the fault is still on the board, and
                    # the held backlog was never taken.
                    log.info("arrival: the catch-up landed too late (%s); "
                             "dropped", outcome)
                    _missed()
                else:
                    log.info("arrival: the catch-up had nothing to say")
            except Exception:  # noqa: BLE001
                log.exception("arrival: the catch-up failed off-thread")
                _missed()

        def _missed():
            # A dropped delivery is still OWED. Guarded because it runs on
            # the worker's own last line: a re-arm that raises must not
            # turn a dropped catch-up into a logged crash.
            if not callable(missed):
                return
            try:
                missed()
            except Exception:  # noqa: BLE001
                log.debug("arrival: could not re-arm the desk watch",
                          exc_info=True)

        def catch_up():
            # NOTHING IS DRAINED ON THIS THREAD. The held lines used to be
            # taken here and handed to the worker, which then had the power
            # to decide not to speak -- and a dropped digest took the
            # backlog down with it. The backlog is now taken inside
            # speak_catch_up, after the guard and immediately before the
            # words, and given back if the words do not happen.
            #
            # OFF THE PUMP THREAD WHEN IT COSTS A SOCKET. bus.publish only
            # queues once Tk is attached, and drain() runs from the UI's
            # own _pump -- so every subscriber here, this step included,
            # executes on the Tk MAIN THREAD. The unread count is an IMAP
            # round trip, and an IMAP round trip has no bound this file can
            # honestly quote: mail.py's IMAP_TIMEOUT is the SOCKET timeout,
            # which bounds one blocking call and not a fetch. What three of
            # his mailboxes cost is NOT MEASURED. That is exactly why it
            # cannot be paid on the pump, where it freezes the window and
            # every event behind it at the moment he walks in -- and why
            # the worker's own lateness is bounded by a check
            # (ARRIVAL_CATCH_UP_LATENESS_S) rather than by a number in a
            # comment. So a configured mailbox finishes the step on a
            # short-lived worker, the way every other mail watcher in this
            # file already does (mailwatch.PeopleMailHeadsUp is its own
            # service). With no mailbox nothing opens a socket, so that
            # path stays inline and the cue is still synchronous end to end.
            if not self._arrival_mail_is_remote():
                # No guard on this path and none needed: inline, the step
                # is still atomic -- nothing can have taken the floor
                # between the read and the words.
                return speak_catch_up() == "spoke"
            # WHOSE ARRIVAL, WHOSE FLOOR, AND WHEN. Read on the pump
            # thread, at the moment the step is taken, and carried to the
            # worker so the worker can tell whether the world it was
            # speaking to is still there when the mailbox finally answers.
            gen = getattr(self, "_arrival_gen", 0)
            turn = getattr(self, "_dispatch_gen", 0)
            started = time.monotonic()
            worker = threading.Thread(target=finish_off_thread,
                                      args=(gen, turn, started),
                                      name="arrival-catchup", daemon=True)
            # Held so a test can join it; nothing in the app waits.
            self._arrival_catch_up_thread = worker
            # The ledger has to say STARTED rather than DONE for this one:
            # see _arrival_ledger.
            self._arrival_catch_up_deferred = True
            worker.start()
            return True

        return {"panel": panel, "earcon": earcon, "greeting": greeting,
                "catch-up": catch_up}

    # ------------------------------------------------- "back from X"
    def _welcome_text(self) -> str:
        """The greeting line: "Welcome back from X, sir", or the plain one.

        X is named ONLY when the calendar can prove it -- an event he was
        out for most of, that ended shortly before he walked in
        (jarvis/arrival.outing). No calendar, an unreachable one, no
        recorded departure, two events that both fit: every one of those is
        the plain "Welcome back, sir", because a guessed event name is
        worse than no event name. There is no new calendar client here:
        this reads the CACHE the parked CalendarSource already holds
        (``events()`` takes no socket), so a homecoming never waits on
        caldav.
        """
        left = float(getattr(self, "_away_since", 0.0) or 0.0)
        if not left or not self.assistant.get("presence.arrival_outing", True):
            return arrival_mod.welcome_line()
        cal = getattr(getattr(self, "services", None), "calendar", None)
        if cal is None:
            return arrival_mod.welcome_line()
        try:
            conf = getattr(cal, "configured", True)
            if callable(conf):
                conf = conf()
            if not conf:
                return arrival_mod.welcome_line()
            what = arrival_mod.outing(
                list(cal.events()),
                left=datetime.fromtimestamp(left).astimezone(),
                back=datetime.now().astimezone())
        except Exception:  # noqa: BLE001 - a name is never worth the greeting
            log.debug("arrival: the calendar could not say where he was",
                      exc_info=True)
            return arrival_mod.welcome_line()
        if what:
            log.info("arrival: he was at %r", what)
        return arrival_mod.welcome_line(what)

    # --------------------------------------------- the catch-up OFFER
    def _arrival_offer_fragments(self):
        """``([fragments], the fault clause inside them)``, both possibly empty.

        "You've 3 unread emails. Shall I go through them, sir?" is one
        fragment; a standing fault is another in front of it, because
        jarvis/address.py thins whole authored LINES and health.py's own
        wording carries a "sir" of its own. The two numbers behind them are
        read HERE and the sentences are built by jarvis/arrival, which is
        pure -- the same split mailwatch makes ("the line is built here,
        never by the model"), for the same reason: this has to work while
        the GPU is lent to a trainer, and a doorstep question is not worth
        a model turn.

        The fault clause comes back as well as going in, so the delivery
        can decline to say it a second time (see _deliver_arrival_catch_up).
        """
        if not self.assistant.get("presence.arrival_offer", True):
            return [], ""
        major = self._major_line()
        frags = arrival_mod.catch_up_fragments(unread=self._unread_count(),
                                               major=major)
        return list(frags), major

    def _arrival_mail_is_remote(self) -> bool:
        """Will the unread count cost an IMAP round trip?

        The one question that decides whether the catch-up step finishes
        on the Tk pump thread or on a worker. No mailbox configured means
        ``fetch_unread`` raises before a socket is opened, and that path is
        cheap enough to stay inline -- which is the live box today, and
        every test that asserts on ``tts.spoken`` the line after ``run()``.
        Never raises: an unreadable config is treated as "no mailbox", and
        the worst that costs is a synchronous call that was going to be
        fast anyway.
        """
        try:
            if not self.assistant.get("presence.arrival_offer", True):
                return False
            from jarvis.tools import mail as mail_mod
            return bool(mail_mod.mail_accounts(self.assistant))
        except Exception:  # noqa: BLE001 - a config read may not cost the cue
            log.debug("arrival: could not tell whether a mailbox is configured",
                      exc_info=True)
            return False

    def _unread_count(self):
        """How many unread emails, or None -- never their contents.

        None is silence about mail, never "no mail": catch_up_offer keeps
        it that way, because a mailbox that timed out must not be
        announced as an empty one. Costs one IMAP round trip, taken AFTER
        the greeting has already been spoken (the catch-up is the last
        arrival step) and, when a mailbox is actually configured, on a
        worker thread rather than on the Tk pump -- see the comment in
        ``catch_up``. No mailbox configured raises MailNotConfigured
        before a socket is opened, which is the dark-safe path every
        watcher here already uses.
        """
        try:
            from jarvis.tools import mail as mail_mod
            mails = mail_mod.fetch_unread(self.assistant,
                                          since_hours=ARRIVAL_MAIL_HOURS,
                                          limit=ARRIVAL_MAIL_LIMIT)
        except Exception:  # noqa: BLE001 - every failure is silence about mail
            log.debug("arrival: the unread count is unavailable", exc_info=True)
            return None
        return len(list(mails))

    def _major_line(self) -> str:
        """One clause on anything MAJOR that happened while he was out.

        The live fault board (jarvis/faults.py) and nothing else: it is
        local, free, already in his own words, and it is the one thing in
        this process that knows the difference between a warning and
        something that actually broke. Only an ERROR counts -- a warning
        that memory is tight is not news to be met at the door with.
        """
        board = getattr(getattr(self, "services", None), "faults", None)
        try:
            fault = getattr(board, "current", None)
            if fault is None or getattr(fault, "kind", "") != "error":
                return ""
            return str(getattr(fault, "line", "") or getattr(fault, "text", "") or "")
        except Exception:  # noqa: BLE001 - the board must not cost the cue
            log.debug("arrival: the fault board could not be read", exc_info=True)
            return ""

    def _park_arrival_offer(self, said: str = ""):
        """Hand the question to the ONE offer protocol. Returns the dict.

        ``services.briefing_offer`` + ``Commander._try_briefing_offer`` is
        the rung that already resolves "Shall I run your briefing, sir?" --
        end-anchored yes/no, a 60 s TTL, a decline that costs nothing, and
        anything not answer-shaped routed as a new subject with the offer
        dropped. Reusing it is the point: a "yes" must not mean different
        things on different rungs, and this question is put with an open
        microphone exactly as that one is.

        ``said`` is the fault clause the OFFER already spoke, carried into
        the delivery so a yes does not get the same sentence read back at
        it. The dict is returned so the caller can re-stamp ``made_at``
        once the words are actually out -- and arm the follow-up mic then
        too. A question nobody listens for is the 2026-09-02 stuck-listen
        bug in miniature, but arming it HERE, before ``_say`` had queued a
        word, opened the mic on the welcome's own falling edge instead of
        the question's. The caller owns that line now.
        """
        offer = {"made_at": time.time(),
                 "deliver": lambda: self._deliver_arrival_catch_up(said=said)}
        try:
            self.services.briefing_offer = offer
        except Exception:  # noqa: BLE001 - a question nobody can answer is worse
            log.exception("arrival: could not park the catch-up offer")
            return None
        return offer

    def _restamp_offer(self, offer) -> None:
        """Start the offer's 60 s TTL from when he could first ANSWER.

        ``BRIEFING_OFFER_TTL_S`` is measured off ``made_at``, and the
        question is parked before the burst goes out (a TTS failure must
        not leave a question on the floor). A released quiet-hours backlog
        can be several sentences, so stamping at the park spent most of
        his window before the question had even been asked. Re-stamped
        only if the offer we parked is still the live one -- a first-wake
        offer that replaced it keeps its own clock.
        """
        if not isinstance(offer, dict):
            return
        try:
            if getattr(self.services, "briefing_offer", None) is offer:
                offer["made_at"] = time.time()
        except Exception:  # noqa: BLE001 - a stale TTL is not worth an exception
            log.debug("arrival: could not re-stamp the catch-up offer",
                      exc_info=True)

    def _deliver_arrival_catch_up(self, said: str = "") -> bool:
        """He said yes: the senders and subjects, built here, not by the model.

        Same rule as mailwatch -- no model turn for a line that is three
        facts -- and the same cap of MAX_LINES, with the rest counted
        rather than read. False means nothing was said, and
        _try_briefing_offer owns telling him so.

        ``said`` is what the offer already spoke about the fault board. It
        used to be read again unconditionally, so with a standing error the
        whole delivery was the sentence he had just heard, word for word.
        A fault is told once.
        """
        from jarvis.mailwatch import MAX_LINES, _subject_words
        lines: list = []
        major = self._major_line()
        if major and _same_clause(major, said):
            major = ""                  # the offer already said it
        if major:
            lines.append(major if major.endswith((".", "!", "?")) else major + ".")
        try:
            from jarvis.tools import mail as mail_mod
            mails = list(mail_mod.fetch_unread(self.assistant,
                                               since_hours=ARRIVAL_MAIL_HOURS,
                                               limit=ARRIVAL_MAIL_LIMIT))
        except Exception:  # noqa: BLE001 - the fault clause still stands alone
            log.debug("arrival: the catch-up could not read the mail",
                      exc_info=True)
            mails = []
        for mail in mails[:MAX_LINES]:
            subject = _subject_words(getattr(mail, "subject", ""))
            who = getattr(mail, "sender", "") or "an unknown sender"
            lines.append(f"{who}, {subject}." if subject else f"{who}.")
        rest = len(mails) - MAX_LINES
        if rest > 0:
            lines.append(f"And {rest} more.")
        if not lines:
            return False
        # ONE burst, thinned once: the same rule the arrival cue itself
        # follows, and the reason release_fragments exists at all.
        thinned = self._thin_address(lines)
        text = address_mod.join_thinned(thinned)
        if not text:
            return False
        bus.publish(JarvisReply(text=text, speak=True))
        self._say(text)
        return True

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
            # WHEN HE LEFT, for "welcome back from X". ev.since is when the
            # away GRACE expired, which on the 12-minute default is twelve
            # minutes after he actually walked out -- long enough to lose a
            # class that ended in between. presence.last_seen is the last
            # time a leg actually saw him, so it is the honest departure and
            # ev.since is only the fallback. Stamped here and never on the
            # return: the sentinel overwrites `since` with the arrival.
            seen = getattr(getattr(self, "presence", None), "last_seen", None)
            self._away_since = float(seen or getattr(ev, "since", 0.0) or 0.0)
            # He is out, so the next kitchen occupancy is a new door opening
            # rather than the same one (arrival.DoorWatch).
            door = getattr(self, "_door", None)
            if door is not None:
                door.left()
            self._departure_seq_phone(ev)
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
        seq = getattr(self, "_departure_seq", None)
        if seq is not None:
            seq.home(at=time.time())      # a genuine second outing is a new one
        # Home but not a return (a poll that merely confirms he is here):
        # the console gets the state, nothing is spoken.
        bus.publish(Status(text="Home", kind="info"))

    def _on_room_changed(self, ev):
        """He is in a new room. Only the DOOR room, only after an absence.

        THE KITCHEN IS A DOOR SENSOR -- his words, "kitchen to see if i
        enter my apartment since the kitchen and door are next to each
        other". jarvis/roomfabric.py already holds a new room occupied for
        ``rooms_enter_hold_s`` (2 s) before it publishes, so a doorway
        pass-through at walking pace does not reach here at all, and its
        stuck-room guard means a fan in the beam cannot pin this on.

        Two guards, and both are needed. ``DoorWatch`` is the rising edge
        -- kitchen, office, kitchen inside one homecoming is ONE arrival.
        ``_greet_return``'s GREET_DAMPER_S is the other, and it is what
        stops the phone sentinel greeting him again ten seconds later when
        its own probe finally catches up. The damper was orphaned once
        before (see _greet_return); this trigger is deliberately routed
        through it rather than around it.

        The away gate is the presence sentinel's own verdict and is read
        as strictly "away": at boot it is "unknown", and a fresh start
        while he is sitting in the office must not welcome him home.

        AND IT IS THE CAMERA LEG'S ONE LIVE TRIGGER. His flat is a
        corridor -- front door, kitchen, office -- so the fabric naming
        the office is the moment to ask the eye whether it can see him
        there, and a deferred catch-up is delivered off that answer. The
        door half runs FIRST: on a box where the door room and the desk
        room are the same, the arm has to exist before the settle can
        spend it.
        """
        self._door_from_room(ev)
        self._departure_seq_room(ev)
        # A NAME, never a frame. _eye_identity answers "" for a camera
        # that is off, blind, or inside its 21:00-07:00 curfew, and "" is
        # no opinion rather than an absence.
        try:
            seen = self._eye_identity()
        except Exception:  # noqa: BLE001 - a broken eye is not a verdict
            log.debug("arrival: the eye could not be asked", exc_info=True)
            return
        self._settle(room=getattr(ev, "room", "") or "", camera=seen)

    def _door_from_room(self, ev) -> None:
        """The door half of _on_room_changed. See its docstring.

        EVERY REFUSAL HERE NAMES ITS GATE, at INFO, once per reason.
        On 2026-09-05 he walked in at 20:43:13, the fabric published the
        kitchen, and nothing happened -- no arrival line, no greeting, and
        no word anywhere on disk about which gate had said no. The refusal
        was arguably correct (the sentinel had never reached "away",
        because a latched office pinned the house occupied); the SILENCE
        was the defect. DoorWatch.observe now writes the reason itself.
        """
        door = getattr(self, "_door", None)
        if door is None:
            return
        # The sentinel's own verdict WORD, not a bool: "home" and "unknown"
        # both refuse here, and telling them apart in the log is the whole
        # point at boot.
        state = getattr(getattr(self, "presence", None), "state", "") or "unknown"
        try:
            if not door.observe(room=getattr(ev, "room", ""),
                                away=(state == "away"), state=state):
                return
        except Exception:  # noqa: BLE001 - the bus must not lose a subscriber
            log.exception("arrival: the door watch failed")
            return
        log.info("arrival: %s is the door and the house was away",
                 getattr(ev, "room", "?"))
        self._greet_return("room:%s" % (getattr(ev, "room", "") or "?"))

    def _make_departure_seq(self):
        """office -> kitchen -> the phone drops. None if it cannot be built:
        a missing sequence costs the log line, never the presence leg."""
        try:
            from jarvis import presencevote
            get = self.assistant.get if self.assistant is not None else \
                (lambda k, d=None: d)
            return presencevote.DepartureSequence(
                # ``presence.desk_room``, NOT ``presence.desk`` -- the
                # latter is deskpresence.py's boolean switch and is True on
                # his box, which made this sequence compare every room
                # against "True" and never arm. Same key ``self._desk``
                # already uses. tests/test_presence_desk_key.py pins it.
                desk_room=str(get("presence.desk_room",
                                  arrival_mod.DEFAULT_DESK_ROOM)
                              or arrival_mod.DEFAULT_DESK_ROOM),
                door_room=str(get("presence.door_room",
                                  arrival_mod.DEFAULT_DOOR_ROOM)
                              or arrival_mod.DEFAULT_DOOR_ROOM))
        except Exception:  # noqa: BLE001 - optional lane
            log.exception("departure: the sequence could not be built")
            return None

    def _departure_seq_room(self, ev) -> None:
        """HIS departure rule: "if you see office sensor then kitchen then
        phone disconnect assume he left the building".

        An ORDERED sequence with a window, not three independent facts --
        three facts that merely happen to be true together would fire on
        him making coffee and then his phone napping in his pocket. The
        ORDER is what makes it a departure, because his flat is a corridor
        and leaving means passing the kitchen after the office and then
        going out of range. See presencevote.DepartureSequence for the two
        windows and why they are 120 s and 900 s.

        SILENT. There is no speak path here and there must never be one:
        arrival.py is explicit that a valediction to an empty room is a
        notification pretending to be a presence.
        """
        seq = getattr(self, "_departure_seq", None)
        if seq is None:
            return
        try:
            seq.room(room=getattr(ev, "room", ""), at=time.time())
        except Exception:  # noqa: BLE001 - the bus must not lose a subscriber
            log.debug("departure: the sequence failed", exc_info=True)

    def _departure_seq_phone(self, ev) -> None:
        """The third step: his phone stopped answering."""
        seq = getattr(self, "_departure_seq", None)
        if seq is None:
            return
        try:
            import jarvis.presencevote as _pv
            if seq.phone_gone(at=time.time()):
                log.info("%s", _pv.departure_note(seq))
        except Exception:  # noqa: BLE001
            log.debug("departure: the sequence failed", exc_info=True)

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
        # ARM THE MIC. This is a QUESTION Jarvis asked, and until 2026-09-02
        # it was the only one that opened nothing: `_after_speech` starts a
        # follow-up only on this flag, `Commander.ask_leave_time` arms just
        # `_pending_leave` (the answering rung, patient for
        # LEAVE_ANSWER_WINDOW_S), and no wake word follows a proactive line.
        # At 08:56:15 he was asked "How long do you need to get to
        # Wisenbaker, sir?", got no mic at all, and answered by TYPING 21 s
        # later (live log: `handle 'about 10 minutes' source=typed`).
        self._followup_after_speech = True
        self._say(question, proactive=True, kind="message")
        # ...and OPEN THE MIC THAT ANSWERS IT. Alone among every question
        # Jarvis asks, this one did not: on 2026-09-02 08:56:15 he asked
        # about the walk to Wisenbaker, no capture was ever opened (no
        # "Recording started", no "Listening…" in the log), and Hunter had
        # to type his answer 21 s later -- "was stuck at speaking and
        # wouldnt let me respond". A question with no mic behind it is not
        # a question. _after_speech opens the window on this burst's own
        # falling edge, sized by _question_open below.
        self._followup_after_speech = True
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

    def unset_option(self, key) -> bool:
        """Remove a settings key entirely. True when the file was written.

        The SENSORS page uses it to retire presence.desk_band_m /
        presence.room_band_m once their bands have been carried into
        zones.rooms: a superseded key left in the file looks exactly like a
        live one.
        """
        try:
            return bool(self.assistant.unset(key))
        except Exception:
            log.exception("unset_option %s failed", key)
            return False

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
        if self._audio_busy.is_set():
            # Transcription of the previous utterance is still running (~20 s
            # for a long clip). Starting a second capture here raced two
            # transcripts into the commander. Say so rather than ignoring it
            # silently -- an unanswered wake word reads as a broken mic.
            log.info("hotword ignored: still transcribing the previous clip")
            bus.publish(Status(text="One moment — still on the last one",
                               kind="warn"))
            return
        if self._turn_busy.is_set() and not barge:
            # THE OTHER FLAG, NAMED AS ITSELF. Until 2026-09-06 this clause
            # shared the line above, and the log of that night's soft lock
            # said "still transcribing" five times about a decode thread
            # that had returned fifteen seconds earlier. What held the
            # floor was an unanswered "Was that for me?" (done=False), and
            # the shared line sent the whole investigation after the wrong
            # flag. Say which one it is.
            if self._asking_uncertain():
                log.info("hotword ignored: waiting on your answer to "
                         "'Was that for me?'")
                bus.publish(Status(text="Waiting on your answer — YES or NO?",
                                   kind="warn"))
            else:
                log.info("hotword ignored: waiting on the reply to the last turn")
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
        # THE MIC IS THE FOURTH LEG, and it is the one corroboration source
        # neither the phone nor the camera can supply: he just spoke, so he
        # is in the flat whatever his phone's radio is doing. It feeds the
        # stuck-room detector (jarvis/stuckroom.py), which is what tells a
        # radar latched on by a fan apart from a man sitting still at his
        # desk. arrival.departure_ready already reads the same ledger for
        # its own 10-minute veto.
        try:
            presence = getattr(self, "presence", None)
            if presence is not None and hasattr(presence, "corroborate"):
                presence.corroborate("mic")
        except Exception:  # noqa: BLE001 - never block a turn
            log.debug("presence: mic corroboration failed", exc_info=True)
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

    @property
    def preview_probe(self) -> PreviewProbe:
        """The ghost-card instrument (jarvis/previewprobe.py), built lazily.

        Lazy, and with a class-level None default, for the same reason the
        attributes above have one: _partial_loop is exercised on apps the
        tests build with object.__new__, which never run __init__. An
        instrument that raises on an uninitialised app would take down the
        preview thread it is supposed to be watching.
        """
        probe = self._preview_probe
        if probe is None:
            probe = PreviewProbe(jsonl_path=PATHS.LOG_DIR / "previews.jsonl")
            self._preview_probe = probe
        return probe

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

        It also RETRACTS the preview when the capture ends, by publishing
        PartialText(""). Nothing else can: the UI drops the ghost card
        when a Transcribed or a UserUtterance replaces it, and the two
        turns that never produce either are exactly the two that leave a
        preview of pure noise on screen --

          * the follow-up window closing on silence, which is
            recorder._check_endpoint -> abort() -> RecordingStopped
            (reason="abort"), and _on_recording_stopped returns on
            "abort" before it publishes anything;
          * a clip the recorder gates out ("Too short" / "No audio
            captured"), which returns there too.

        Both fire after a reply, both run this loop over room noise with
        no VAD and no noise gate for the whole window, and Whisper duly
        invents something. That is Hunter's "2- keeps showing up after a
        response and hes listening back for me": the card was still up
        because nobody had ever been able to take it down.

        THE FILLER HOLD (jarvis/recorder.py, CONFIG.filler_hold): every
        greedy decode here is reported to recorder.note_partial() with the
        capture second the decoded span ENDS at, so a preview ending on
        "um" can hold the stop open. Only the greedy preview reports a
        FILLER. The speculative pass does not: its prompt never carries
        the filler hint and its clean full decode is the one most likely
        to have dropped the um, so letting it overrule the preview would
        defeat the experiment scripts/filler_probe.py exists to run.
        LIMIT: this loop's cadence (_PARTIAL_INTERVAL_S, 0.9 s) is slower
        than the endpoint (CONFIG.endpoint_silence, 0.8 s), and the
        speculative pass parks the preview for a full decode from 0.3 s
        into a pause, so a filler spoken after the last snapshot may never
        be decoded before the stop is due. The recorder does NOT decode
        the tail itself (that would be a decode on every turn's stop): it
        stops as before. How often that happens is a number the probe
        measures.

        THE SPELLING HOLD (jarvis/spelling.py) is fed by BOTH passes
        (09-06): the greedy preview through the same note_partial, and
        the speculative pass through recorder.note_speculative -- which
        only ever ADDS a hold when its decode ends on a run, so the parking
        above cannot hide the letters the hold is waiting for. Measured on
        the recorder's real poll cadence: see Recorder.note_speculative and
        tests/test_spelling_survival.py.
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
                # BEFORE the snapshot, never after: this stamps the note
                # with the capture the audio came from. If the capture
                # turns over between this read and the snapshot the note
                # carries the OLD id and the recorder drops it -- a
                # mismatch may only ever discard, never accept a stale
                # note. Reading it after the decode would do the opposite.
                capture = getattr(self.recorder, "capture_id", None)
                audio = self.recorder.snapshot_audio()
                end_s = 0.0
                if audio is not None:
                    # Where the span ENDS in capture seconds: the whole
                    # buffer, measured BEFORE the trim below. The recorder
                    # compares it with the VAD's last-speech position
                    # (note_partial, the filler hold); the decode wallclock
                    # would be the wrong clock for that.
                    end_s = len(audio) / SAMPLE_RATE
                    audio = audio[-int(SAMPLE_RATE * self._PARTIAL_MAX_S):]
                if audio is not None and len(audio) >= int(
                        SAMPLE_RATE * self._PARTIAL_MIN_S):
                    # Count the decode BEFORE the publish filter below. The
                    # gap between decodes and emissions is the number no
                    # diagnosis of the 09-03 transcript spam could get: the
                    # preview fires ~1/s for the whole capture, and only the
                    # passes that CHANGED the card were ever visible at all.
                    seconds = len(audio) / SAMPLE_RATE
                    self.preview_probe.decoded(PATH_GREEDY)
                    # Announce the decode BEFORE it runs (Recorder.
                    # note_decoding): while it is running and its snapshot
                    # holds the last burst, a stop that falls due waits
                    # for it rather than beating it by a few tens of ms.
                    self._note_decoding(end_s, capture)
                    try:
                        text = (self.transcriber.partial(audio) or "").strip()
                    except Exception:
                        log.debug("partial decode failed", exc_info=True)
                        text = ""
                    # Every decode, changed or not, failed or not: a
                    # failed decode reports "" so a stale um cannot keep
                    # holding the mic open. Guarding this on
                    # recorder.recording is the WRONG fix for the
                    # cross-capture race -- it would drop exactly the note
                    # that clears a stale um. The capture stamp does it.
                    self._note_partial(text, end_s, capture)
                    # only publish on change: the ghost card redraws on
                    # every event, and whisper often returns the same text.
                    if text and text != last and self.recorder.recording:
                        last = text
                        self._partial_shown = True
                        # Record the SHAPE of what is about to be shown.
                        # This call cannot withhold the publish and its
                        # result is deliberately ignored -- the cause of
                        # the spam is unproven, and a preview filter built
                        # on a guess is the 2026-08-31 confidence gate that
                        # ate his commands, wearing a different hat.
                        self.preview_probe.shown(text, path=PATH_GREEDY,
                                                 audio_s=seconds)
                        bus.publish(PartialText(text=text))
                # pace from the END of the decode, so a slow pass backs off
                # instead of queueing up behind itself.
                due = time.monotonic() + max(0.05, self._PARTIAL_INTERVAL_S -
                                             (time.monotonic() - started))
        except Exception:
            log.exception("partial loop died")
        finally:
            # In the finally, not after the loop: a preview that outlives
            # the microphone is the bug, and a died-with-an-exception loop
            # is the last thread that should get to keep one on screen.
            if self._partial_shown:
                self._partial_shown = False
                self.preview_probe.retracted(PATH_GREEDY)
                bus.publish(PartialText(text=""))

    def _drop_partial(self) -> None:
        """Take the ghost card down NOW, whatever is on it.

        The turn that consumes the passphrase has no UserUtterance to
        replace the preview with and no rejection to make the UI clear it,
        so this is the only thing that can. Used by _gate_consumed.

        PUBLISHED UNCONDITIONALLY, and that asymmetry is the whole design:
        a redundant clear costs one no-op call on a pane that is already
        empty, while a clear that did not fire leaves his passphrase
        legible on the glass. The probe's retraction counter is only
        bumped when this process believes a card was up, so the preview
        ledger keeps meaning what it meant.

        Every step is guarded because this runs in front of a redaction:
        an instrument, a bus subscriber or a stand-in without a probe must
        not be able to keep the words on screen by raising.
        """
        try:
            if getattr(self, "_partial_shown", False):
                self._partial_shown = False
                try:
                    self.preview_probe.retracted(PATH_GREEDY)
                except Exception:              # noqa: BLE001 - instrument
                    log.debug("preview retraction failed", exc_info=True)
            bus.publish(PartialText(text=""))
        except Exception:                      # noqa: BLE001 - never fatal
            log.exception("the preview card could not be taken down")

    def _note_partial(self, text: str, end_s: float,
                      capture_id=None) -> None:
        """Hand the preview's newest decode to the recorder's filler hold,
        stamped with the capture its audio came from (Recorder.note_partial
        drops a note from any other). getattr, because the preview-thread
        tests drive _partial_loop with bare recorder fakes -- and a preview
        must never fail the capture."""
        note = getattr(self.recorder, "note_partial", None)
        if note is None:
            return
        try:
            note(text, end_s, capture_id)
        except Exception:
            log.debug("note_partial failed", exc_info=True)

    def _note_decoding(self, end_s: float, capture_id=None) -> None:
        """Tell the recorder a decode is starting on a snapshot that ends
        at ``end_s`` (Recorder.note_decoding), so a stop that falls due
        while it runs can wait for it. getattr, as _note_partial: the
        preview-thread tests drive this loop with bare recorder fakes."""
        note = getattr(self.recorder, "note_decoding", None)
        if note is None:
            return
        try:
            note(end_s, capture_id)
        except Exception:
            log.debug("note_decoding failed", exc_info=True)

    def _note_speculative(self, text: str, end_s: float,
                          capture_id=None) -> None:
        """Hand the speculative pass's decode to the recorder's SPELLING
        hold (Recorder.note_speculative: additive, run-only). getattr for
        the same reason as _note_partial -- the pass is best effort and a
        recorder fake without the seam is a recorder that hears nothing."""
        note = getattr(self.recorder, "note_speculative", None)
        if note is None:
            return
        try:
            note(text, end_s, capture_id)
        except Exception:
            log.debug("note_speculative failed", exc_info=True)

    def _address_owed(self, commander) -> bool:
        """Is the commander waiting on an ADDRESS -- "What is it?" asked,
        or a draft read back that he may correct? Read by the recorder at
        every start() through address_owed_probe, and False on any doubt:
        the only thing it changes is whether a bare spelled first letter
        holds the mic (jarvis.spelling.spelling_run, address_owed)."""
        probe = getattr(commander, "address_owed", None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            log.debug("address_owed failed", exc_info=True)
            return False

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
    _partial_shown = False        # a ghost card is up and needs retracting
    _preview_probe = None         # PreviewProbe, built on first use below
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
        return audio, stats, False, self._clip_decode(audio)

    def _clip_decode(self, audio):
        """THE DECODE, REDACTED WHENEVER IT COULD BE THE SECRET.

        The earlier fix closed jarvis/transcriber.py's `Transcribed: %r`
        line on the RESCUE leg only -- a clip the speaker filter dropped.
        That is not the leg he is on. When the filter MATCHES him (the
        common case, and the one he tests in) this is the decode, and it
        wrote the spoken passphrase to jarvis.log in plaintext at INFO in
        shadow, enforce and off alike -- measured on a fresh copy at
        ecb5abc, and again here by tests/test_knightfall_log_leak.py
        before this method existed.

        So: when an owner has a phrase set, EVERY clip is decoded quietly
        and the words are written down only by ``_log_transcript`` below,
        after the gate has said they are not the phrase. When no owner has
        one there is no secret to protect and nothing changes at all --
        not the decode, not the line, not its position in the log.

        The cost is the one the redaction cannot avoid: on a turn the gate
        consumes, no `Transcribed:` line carries words, because there are
        no words that may be carried. The numbers stay in both cases.
        """
        if self._owner_has_phrase():
            return self._gate_quiet_decode(audio)
        return self.transcriber.transcribe(audio)

    def _loggable(self, text) -> str:
        """``repr(text)`` for a log line -- or the redaction, when an owner
        has a phrase set and nothing has yet ruled these words are not it.

        The rule this whole pass enforces, in one place: WORDS ARE WRITTEN
        DOWN ONLY AFTER SOMETHING HAS SAID THEY ARE NOT THE PHRASE.

        And when they are written, an address in them is masked
        (transcriber.shown_words): this is the speculative-decode line,
        which carried a spelled address raw while the `Transcribed:` line
        was being closed -- the twin the 09-06 verdict did not name.
        """
        if text and self._owner_has_phrase():
            return repr(gate_mod.REDACTED_TEXT)
        return repr(shown_words(text))

    def _log_transcript(self, result) -> None:
        """The `Transcribed:` line the quiet decode deliberately did not
        write, now that the gate has cleared these words.

        Only when the decode was actually redacted -- otherwise
        jarvis/transcriber.py already wrote the line and this would double
        it. Deliberately the same wording and the same numbers as that
        line, so a reader of jarvis.log (and every grep and script written
        against it, scripts/measure_confidence_gate.py included) sees one
        format, not two -- and the same mask over an address in the words
        (transcriber.shown_words), or the quiet decode would be the one
        path on which a spelled address reaches jarvis.log raw.
        """
        if not self._owner_has_phrase():
            return                     # transcriber.py wrote it already
        text = getattr(result, "text", "") or ""
        if not text:
            return                     # nothing was said; nothing to say
        log.info("Transcribed: %r (avg_logprob=%.2f)", shown_words(text),
                 float(getattr(result, "confidence", 0.0) or 0.0))

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
        # BEFORE the snapshot, as _partial_loop reads it: a capture that
        # turns over during the decode leaves this note with the old id,
        # and the recorder drops it.
        capture = getattr(rec, "capture_id", None)
        audio = None
        end_s = 0.0
        try:
            audio = rec.snapshot_final()
            if audio is not None:
                end_s = len(audio) / SAMPLE_RATE
                # Announced BEFORE the decode (Recorder.note_decoding) and
                # reported after it whatever it said (note_speculative,
                # below), so the recorder can wait for this pass when the
                # stop falls due mid-decode and never waits on one that
                # has already returned.
                self._note_decoding(end_s, capture)
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
        # THE SPELLING HOLD hears this pass (09-06): it is the decode the
        # pause itself started, so it is the one that lands between two
        # spelled letters on no clock. Additive only -- Recorder.
        # note_speculative stores nothing unless the text ends on a run
        # -- and NOT gated on `accepted`: a run of single letters is
        # exactly where the confidence gate is least sure, and a wrong
        # hold is 2 s once where a missed one is a lost letter. Reported
        # UNCONDITIONALLY, an empty or rejected decode included: the
        # report is also what tells the recorder the decode it announced
        # has returned, and a recorder still waiting on a pass that came
        # back empty would hold the mic for the cap.
        self._note_speculative(text, end_s, capture)
        if text and result.accepted and rec.recording:
            # The speculative text IS the best preview there is.
            self._partial_shown = True
            # Instrumented too, though this path is already gated and
            # already logged below: both paths publish the SAME PartialText
            # event, so without the path tag a repeat on screen cannot be
            # attributed to an emitter after the fact. That ambiguity is
            # what cost the 09-03 diagnosis its confidence.
            # getattr, not result.audio_seconds: the probe's methods swallow
            # their own exceptions, but ARGUMENT evaluation happens first,
            # and `result` here is duck-typed by callers and tests. An
            # instrument is not allowed to be the thing that breaks the
            # decode path it is measuring.
            self.preview_probe.shown(
                text, path=PATH_SPECULATIVE,
                audio_s=getattr(result, "audio_seconds", None))
            bus.publish(PartialText(text=text))
        # REDACTED FOR THE SAME REASON THE DECODE IS. This is a DEBUG line
        # rather than the INFO one the verdict measured, so it only reaches
        # jarvis.log when he has turned debug logging on -- which is exactly
        # what he does when something is wrong, i.e. the session in which
        # the phrase is most likely to be said and least likely to be
        # noticed. The words reach the log from _log_transcript once the
        # gate has cleared them, and from nowhere else.
        log.debug("speculative decode at last_speech=%.2fs took %.2fs (%s)",
                  key, spec.finished - spec.started,
                  "rejected" if spec.rejected else self._loggable(text))
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
        # Who the owner gate attributed the last turn to, and on which leg.
        # For the pane and the log: "he was refused" is not a bug report,
        # "the face leg named Heather while the voice leg abstained" is.
        self._gate_who = ""
        self._gate_how = ""
        # WHEN that attribution was made (monotonic). The honorific resolver
        # will not use an attribution older than honorific.ADDRESSEE_TTL:
        # one turn in which the camera named Mara must not make tonight's
        # reminder and tomorrow's briefing come out addressed to her.
        self._gate_who_ts = -1e9
        self._canvas_due_cache = (-1e9, [])   # monotonic; see _last_nudge_ts
        self._room_gpu_cache = (-1e9, None)   # ditto: the ambient GPU reading
        # The ambient slab's one outbound dependency, on a backoff
        self._room_playing_text = ""
        self._room_playing_ts = -1e9
        # ---- the owner gate (jarvis/gate.py) ------------------------------
        # WHO Jarvis answers.  He chose to gate EVERYTHING including ordinary
        # chat, so the question is asked ONCE, in _process_audio, and the
        # answer travels with the turn.  It ships in SHADOW (owner.mode):
        # every verdict logged, nothing refused, until the log says the
        # verdicts are right.
        #
        # Constructed defensively because a gate that cannot be built must
        # not be able to stop him talking: any failure here leaves
        # self.gate None and _process_audio admits every turn.
        self.gate = None
        try:
            self.gate = gate_mod.OwnerGate(
                registry=identity_mod.Registry.load(),
                get_option=self.get_option,
                owner=identity_mod.owner_label(self.assistant),
                # THE LEDGER, and shadow mode is worth nothing without it.
                # Every gated verdict lands in gate.jsonl as decisions and
                # scores -- never a word of what was said -- so that
                # scripts/gate_scorecard.py can tell him what enforce would
                # have done to him before he switches it on.
                record=gateledger.writer(PATHS.LOG_DIR / "gate.jsonl"))
        except Exception:                          # noqa: BLE001 - never fatal
            log.exception("owner-gate: could not be built; it is OFF and "
                          "everyone is being answered")
        try:
            # The startup line is its OWN boundary. It reads the speaker and
            # the gallery to say why a leg is dark, and none of that is worth
            # losing the gate over -- an earlier version built the two
            # together and a stubbed app with no `speaker` silently left the
            # gate switched off.
            if self.gate is not None:
                enrolled = getattr(getattr(self, "speaker", None),
                                   "enrolled", False)
                log.info("%s", self.gate.startup_line(
                    voice_ok=bool(CONFIG.speaker_verify and enrolled),
                    face_ok=False, face_why=self._face_leg_why()))
        except Exception:                          # noqa: BLE001 - a line only
            log.exception("owner-gate: the startup line could not be built")
        # One wording for both refusals, and it changes with the registry:
        # offering a passphrase that has never been set would be a dead end
        # of a different shape.
        self._guest_line = (GUEST_PHRASE_LINE if self._owner_has_phrase()
                            else GUEST_LINE)

    def _owner_has_phrase(self) -> bool:
        """Is there a spoken way back in to offer? Never the phrase itself,
        and never anything derived from it -- only whether one exists."""
        try:
            return any(p.phrase_hash for p in self.gate.registry.owners())
        except Exception:                          # noqa: BLE001 - no registry
            return False

    def _face_leg_why(self) -> str:
        """One plain sentence for the startup line saying why the camera
        cannot name anybody, or "" when it can.

        THE ONE THAT IS ABOUT TO HAPPEN: the face model is being swapped
        from SFace (128-D) to ArcFace (512-D) on another branch. The stored
        vectors are not comparable, so his enrolment stops matching and he
        must re-enrol -- and for as long as that lasts the face leg is
        unavailable and VOICE CARRIES THE WHOLE THING. It is a width
        mismatch, which is NO OPINION and never a negative, but he is owed
        the sentence rather than left to work it out from being refused.
        """
        try:
            # THE LIVE MODEL'S WIDTH, not facegallery.EMBED_DIM. That name
            # is a fixed alias for SFace's 128 and says nothing about what
            # this box runs; comparing against it told Hunter on 2026-09-05
            # to re-enrol a CORRECT 512-D ArcFace enrolment, and re-enrolling
            # would have produced another 512-D one. BOTH branches fixed this
            # independently and they disagreed on one point only: which
            # backend to ask about. The config-aware read wins -- it is the
            # documented one-line reversal to SFace, and asking about the
            # DEFAULT would call a correct SFace enrolment stale the moment
            # he takes it. Pinned by tests/test_face_leg_line.py.
            #
            # AND AN UNREADABLE NAME FALLS BACK RATHER THAN GOING SILENT
            # (merge, 09-05). facemodels.backend_for RAISES on a name it does
            # not know -- deliberately, so a typo cannot silently keep the old
            # models -- and letting that escape turned the WHOLE sentence into
            # "the camera could not be asked", which hides a genuinely stale
            # row behind a config typo. A width is not a model load: the
            # default's width is the honest thing to measure against when the
            # configured name cannot be read, and the reason is said out loud
            # in the line's own log.
            from jarvis import facemodels
            want = str(self.get_option("camera.face_backend", "") or "")
            try:
                live_dim = int(facemodels.backend_for(want).embed_dim)
            except Exception:                      # noqa: BLE001 - a name only
                live_dim = int(facemodels.backend_for(None).embed_dim)
                log.warning("owner-gate: camera.face_backend %r is not a "
                            "known backend; the face-leg line measures "
                            "against the default's %d-D", want, live_dim)
            enrolled = [p for p in self.gate.registry.people if p.face]
            stale = [p for p in enrolled if p.face_dim and
                     p.face_dim != live_dim]
            if stale:
                return ("the gallery is %d-D and %s was enrolled at %d-D; "
                        "re-enrol to bring the face leg back"
                        % (live_dim, stale[0].label, stale[0].face_dim))
            if not enrolled:
                return "no face is enrolled in the registry"
            if not self.get_option("camera.identity", False):
                return "camera.identity is off"
            return "nothing on this tree attaches a camera feed yet"
        except Exception:                          # noqa: BLE001 - a line only
            return "the camera could not be asked"

    def _after_dispatch(self, text, source, result, addressee=None):
        """Bookkeeping once a command has been handled synchronously."""
        # ROUND-3, named lower and closed here: this filed a GUEST'S
        # sentence and Jarvis's refusal of it into HIS conversation memory
        # -- measured add_exchange("what's on my to-do list", "That one's
        # Hunter's, Mara Voss. ..."). Nothing of his is disclosed by that,
        # but his record is his: a turn that was not his leaves no trace
        # in it. The mic and follow-up bookkeeping below is I/O, not his
        # data, and still runs for her.
        his_turn = not scope_mod.reading(addressee)[0]
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
            if his_turn:
                self.context.add_exchange(text, reply)
            if source == "voice" and result.speak and CONFIG.talkback:
                self._followup_after_speech = True
        if his_turn and done and getattr(result, "speak", False):
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

    def _debrief_reply(self, text, source, addressee=None):
        """The open debrief owns this transcript -- or gives it up.

        Gives it up for anything that is plainly a command (a Tier-1 match,
        a "jarvis" prefix) and for anything that arrives more than
        DEBRIEF_TTL_S later: "set a timer for five minutes" is not how the
        midterm went, and filing it as such would poison the record he is
        meant to be able to trust months from now."""
        pending = self._pending_debrief
        if pending is None or source not in ("voice", "typed"):
            return None
        # ROUND-3 BLOCKER 3, MEASURED. This method is hoisted ABOVE
        # commander.handle on purpose (see _ask_debrief), so it is the one
        # door of the turn that sits above the commander's scope read --
        # and it took no reading of its own. With the gate naming a guest,
        # "honestly it was a disaster, he ran out of time" was written to
        # BOTH his private sinks (memory.remember + context.journal_debrief)
        # and commander.handle was never called. Worse than an ordinary
        # leak: mark_asked is written when the question is PUT, so she did
        # not merely answer his once-ever question, she SPENT it.
        #
        # It cannot simply move below the commander (jarvis/debrief.py:381
        # explains why), so it asks the same question the commander asks,
        # of the same carried value. Like the floor-holder branch below,
        # this returns None WITHOUT clearing _pending_debrief: the question
        # is his, it was not answered, and it stays open for him.
        who = scope_mod.reading(addressee)[0]
        if who:
            log.info("debrief stands down: this turn is %s's, not his", who)
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
            addressed = strip_jarvis_prefix(strip_fillers(text)) is not None or \
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
            # An OFFER, not the briefing (Hunter, 2026-09-02). Still here,
            # on the settled burst, rather than on some later "natural
            # gap": the only gap this code can actually observe is silence,
            # and speaking into silence out of nowhere is a bigger
            # interruption than one short question at the end of an
            # exchange he started. What it can also observe is another open
            # question -- a flashcard, a read-back, a router ask -- and two
            # questions on the table is how a "yes" lands on the wrong one,
            # so the offer waits for that instead.
            if self._question_open(getattr(self, "commander", None)):
                # Held -- and the question it is held BEHIND still needs
                # its follow-up mic. Returning here dropped the window, so
                # the "yes" to a read-back on the first turn of the day was
                # never heard without a wake word (F35, 09-03; the old
                # branch delivered the briefing and consumed the flag).
                log.debug("briefing offer held: another question is open")
            else:
                self._briefing_pending = False
                self._followup_after_speech = False
                self._offer_first_wake_briefing()
                return
        if self._followup_after_speech:
            self._followup_after_speech = False
            self._start_followup()

    def _start_followup(self):
        """Listen for a follow-up without the wake word (CONFIG.followup_window)."""
        why = ""
        if not MACHINE.has_mic:
            why = "no microphone"
        elif CONFIG.followup_window <= 0:
            why = "the follow-up window is switched off"
        elif self.recorder.endpointer is None:
            # without a VAD nothing can say "nothing was said"
            why = "no VAD to close the window"
        elif self.recorder.recording or self._audio_busy.is_set() \
                or self._turn_busy.is_set():
            why = "the last turn still has the floor"
        if why:
            # LAYER 3 of the 2026-09-02 stuck listen. All four of these used
            # to be a bare `return`: Jarvis asked a question, no mic opened,
            # no status changed, and the board went on reading whatever it
            # last said -- "Speaking". A question waiting on an answer that
            # nothing is listening for must be visible in one glance.
            if self._question_open(getattr(self, "commander", None)):
                log.warning("follow-up window NOT opened (%s) while a question "
                            "is on the table: the answer can only be typed", why)
                bus.publish(Status(text=f"Waiting for your answer — "
                                        f"mic not open ({why})", kind="warn"))
            else:
                log.debug("follow-up window not opened: %s", why)
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
        # "How long do you need to get to Wisenbaker, sir?" -- same shape,
        # and checked HERE rather than in `_question_open` for the same
        # reason as the debrief: `Commander.ask_leave_time` calls
        # `question_open()` as its own do-not-ask guard, so a leave question
        # listed there would stand down from its own answer. A walk duration
        # is said after a pause to think about it, not inside 4 s.
        leave = getattr(commander, "_pending_leave", None)
        if isinstance(leave, tuple) and len(leave) == 3:
            try:
                if time.monotonic() - float(leave[2]) <= LEAVE_ANSWER_WINDOW_S:
                    return self._window_setting("quiz.window_s", 15.0)
            except (TypeError, ValueError):
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

    @staticmethod
    def _leave_pending_age(commander):
        """How long ago the walk question was asked, or None if none stands.

        ``commander._pending_leave`` is ``(key, place, monotonic)`` and is
        cleared lazily -- ``_try_leave_answer`` only drops it on the next
        utterance -- so a stale tuple can sit on the commander for hours.
        Every caller therefore wants the AGE, never the truthiness, and each
        one measures it against its own window: the mic against
        ``quiz.window_s``, the salvage gate against the rung's real
        LEAVE_ANSWER_WINDOW_S.
        """
        pend = getattr(commander, "_pending_leave", None)
        if not isinstance(pend, tuple) or len(pend) != 3:
            return None
        try:
            return time.monotonic() - float(pend[2])
        except (TypeError, ValueError):
            return None

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
        # "How long do you need to get to Wisenbaker, sir?" is a question
        # Jarvis asked too, and commander.question_open() does not count it
        # (it is the one rung that arms itself from outside handle()). Its
        # answer is a sentence -- "about ten minutes" -- not a word, and the
        # 4 s follow-up window is sized for "...and Tuesday?".
        #
        # The expiry here is the MIC's, not the rung's. _pending_leave lives
        # for LEAVE_ANSWER_WINDOW_S (180 s), and review caught what that
        # would have meant: this predicate is not only read by
        # _capture_window, it also gates _salvage_low_confidence, so a
        # three-minute rung would have force-accepted sub-threshold garble
        # for three minutes after a question he may never have heard. The
        # window this branch exists to size is quiz.window_s, so that is
        # what it is measured against; _try_leave_answer keeps its own 180 s
        # for an answer that arrives on a later wake word.
        age = self._leave_pending_age(commander)
        if age is not None and age <= self._window_setting("quiz.window_s", 15.0):
            return True
        # The wake-alarm offer and the first-wake briefing offer live on
        # the services namespace, not on the commander: briefing.make_tools
        # parks one there for _try_alarm_offer and
        # app._offer_first_wake_briefing the other for _try_briefing_offer.
        #
        # The briefing offer's life here is BRIEFING_OFFER_TTL_S, not the
        # wake alarm's 180 s, for the reason written two paragraphs up
        # about _pending_leave: this predicate gates _salvage_low_confidence
        # as well as the mic window, and the briefing offer is put once
        # EVERY day behind an arbitrary request, so three minutes of
        # force-accepted garble after it was the largest instance of
        # exactly that bug.
        services = getattr(self, "services", None)
        for name, ttl in (("alarm_offer", OFFER_TTL_S),
                          ("briefing_offer", BRIEFING_OFFER_TTL_S)):
            offer = getattr(services, name, None)
            if not isinstance(offer, dict) or not offer:
                continue
            try:
                made = float(offer.get("made_at") or 0.0)
            except (TypeError, ValueError):
                made = 0.0
            if not made or time.time() - made <= ttl:
                return True
        return False

    # ---------------------------------------------------- guests, learning
    def _music_playing(self) -> bool:
        """Is music known to be playing?  The mixer's cache read (it holds
        the Spotify tool); False whenever there is no mixer or it breaks, so
        the relaxed wake bar can never become the default by accident."""
        fn = getattr(getattr(self, "mixer", None), "music_playing", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            log.debug("music_playing lookup failed", exc_info=True)
            return False

    def _on_guest(self, score):
        """A clear wake word in a voice that is not the enrolled one."""
        if score < 0.85 or not CONFIG.talkback:
            return
        if getattr(getattr(self, "tts", None), "busy", False) is True or \
                getattr(self, "_tts_active", False):
            return          # under barge-in the listener hears his own voice
        if self._music_playing():
            # A "guest" over music is far more often HIM, scored down by the
            # bed under his voice (0.135 and 0.158 on 2026-09-01), or the
            # vocalist tripping oww.  "I only answer to Hunter, sir" said to
            # either is wrong, and said to him twice in a minute was the
            # complaint.  Stay quiet; the cooldown is not spent.
            log.info("guest-like wake over music, staying quiet (score=%.2f)",
                     score)
            return
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
            # ONLY HIS OWN CLIPS GO INTO HIS VOICEPRINT. filter_segments says
            # whose pool the kept windows cleared the bar on (matched_label,
            # "" for his) and whom the gallery named or guessed (who / top).
            # A KNOWN person's admitted turn -- "what's the time" from Mara
            # -- reaches here too, and learning HER into HIS pool is the
            # walk add_sample's cap exists to bound, one sample at a time.
            mine = str(getattr(self.speaker, "owner_label", "") or "")
            guess = str(stats.get("who") or stats.get("top") or "")
            if stats.get("matched_label") or (guess and guess != mine):
                log.info("passive learning skipped: the clip is %s's, not "
                         "the owner's", stats.get("matched_label") or guess)
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

    def _briefing_gates(self):
        """The conditions the first-wake briefing must pass, cheapest first.

        A LIST of named gates rather than a chain of ifs because Hunter's
        2026-09-02 ruling was "he should offer -- we will have auto
        briefing changed later with presence and camera stuff". That
        rework wants "he is here, he is awake, and it is morning", which
        is one more entry here and one more line in the log that says
        which gate refused. Nothing outside this method knows what the
        gates ARE, so adding one changes no caller.

        Each gate takes ``now`` and returns True to let the briefing
        through, and each swallows its OWN failure in the direction that
        was already established here: an unreadable config refuses (never
        speak on a guess), a broken quiet policy lets through (a bad
        sensor must not silently switch the feature off).
        """
        return (("switched off", self._gate_switched_on),
                ("before the hour", self._gate_after_hour),
                ("quiet hours", self._gate_not_quiet),
                ("already raised today", self._gate_not_raised_today))

    def _gate_switched_on(self, now):
        try:
            return bool(self.assistant.get("briefing.on_first_wake", True))
        except Exception:
            log.debug("briefing.on_first_wake unreadable", exc_info=True)
            return False

    def _gate_after_hour(self, now):
        try:
            after = str(self.assistant.get("briefing.after", "06:00") or "06:00")
            hh, mm = (int(x) for x in after.split(":")[:2])
        except Exception:
            # A typo'd briefing.after must not free-run the day: 06:00 is
            # the DEFAULT, not a floor to fall back to.
            log.debug("briefing.after unreadable", exc_info=True)
            return False
        return (now.hour, now.minute) >= (hh, mm)

    def _gate_not_quiet(self, now):
        # Quiet hours / a running class: the day stays unraised, so the
        # first answered turn after the window makes the offer instead.
        quiet = getattr(self, "quiet", None)
        try:
            return quiet is None or not quiet.is_quiet()
        except Exception:
            log.debug("quiet check failed; briefing proceeds", exc_info=True)
            return True

    def _gate_not_raised_today(self, now):
        try:
            state = json.loads(self._briefing_state_path().read_text())
        except (OSError, ValueError):
            state = {}
        return state.get("delivered") != now.date().isoformat()

    def _briefing_block(self, now=None) -> str:
        """"" when the briefing may be raised, else the name of the first
        gate that refuses it (a log line, not a UI string)."""
        now = now or datetime.now()
        for name, gate in self._briefing_gates():
            try:
                if not gate(now):
                    return name
            except Exception:
                # Belt and braces: every gate above already swallows its
                # own failure, so reaching here means a NEW gate does not.
                # It refuses, because an unbidden 40-second monologue is
                # the thing this whole path exists to stop.
                log.exception("briefing gate %r raised", name)
                return name
        return ""

    def _briefing_due(self, now=None):
        blocked = self._briefing_block(now)
        if blocked:
            log.debug("briefing not due: %s", blocked)
        return not blocked

    def _mark_briefing_delivered(self):
        try:
            p = self._briefing_state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")     # a torn write must not eat the day
            tmp.write_text(json.dumps({"delivered": datetime.now().date().isoformat()}))
            os.replace(tmp, p)
        except OSError:
            log.debug("briefing state save failed", exc_info=True)

    def _offer_first_wake_briefing(self):
        """Ask, once, in one short line -- and stop there.

        THE INCIDENT (2026-09-02 14:29:30): "Say hello to my family." was
        answered with a greeting, then yesterday's turn metrics, then a
        full weather/calendar/deadlines/news briefing. About 40 seconds of
        monologue off five words that had nothing to do with any of it.
        Hunter: "He should offer."

        The offer is parked on ``services.briefing_offer`` and answered by
        ``Commander._try_briefing_offer``, exactly as the wake-alarm and
        exam-week study offers are -- ONE offer protocol, so a yes cannot
        mean different things on different rungs.

        The day is closed HERE, when the question is put, not when it is
        answered. ``_after_dispatch`` re-arms behind every answered turn,
        so an offer that only closed the day on delivery would ask again
        seconds after each decline, and nagging is precisely the failure
        mode being fixed. Asked-and-declined costs him nothing: "my
        briefing" reaches ``_h_briefing`` at any hour.
        """
        offer = {"made_at": time.time(),
                 # The commander cannot reach the app; the callback is how
                 # the wake-alarm offer's park_offer seam works too.
                 "deliver": self._deliver_first_wake_briefing}
        try:
            self.services.briefing_offer = offer
        except Exception:
            log.exception("could not park the briefing offer")
            return
        # The other end of "good night", and it belongs HERE rather than on
        # the delivery: the first wake of the day is the morning whether or
        # not he wants the news read to him, and hanging it off the yes
        # would leave the house in night mode all day on a "no".
        wd = getattr(self, "winddown", None)
        if wd is not None:
            try:
                wd.restore()
            except Exception:
                log.exception("wind-down restore at first wake failed")
        line = BRIEFING_OFFER_LINE
        log.info("first wake of the day: offering the briefing")
        # Marked BEFORE speaking, as the weekly review is: a TTS failure
        # must not turn one question a day into one per utterance.
        self._mark_briefing_delivered()
        # Not proactive=True: _gate_not_quiet already asked the quiet policy,
        # and holding this for the catch-up digest would put a breakfast
        # question to him at lunchtime (the same rule as _ask_debrief).
        bus.publish(JarvisReply(text=line, speak=True))
        self._say(line)
        # A question nobody listens for is the 2026-09-02 stuck-listen bug
        # in miniature: arm the follow-up window so "yes" needs no wake
        # word. _after_speech runs again on THIS line's falling edge and
        # opens the mic there, once the offer itself has finished playing.
        self._followup_after_speech = True

    def _deliver_first_wake_briefing(self) -> bool:
        """He said yes: read the briefing. True when the model was asked.

        False means nothing was said and nothing was marked -- the caller
        owns telling him so, because a yes that vanishes is worse than a
        refusal (Commander._try_briefing_offer).
        """
        brain = getattr(self.services, "brain", None)
        if brain is None or not hasattr(brain, "chat"):
            return False
        if getattr(getattr(self, "brain", None), "is_busy", False):
            # chat() would only say "Still on the last one, sir": leave the
            # day unmarked so the next answered turn delivers it.
            log.info("first-wake briefing: model busy; next turn")
            return False
        log.info("first wake of the day: delivering the briefing")
        # NO DAY REVIEW HERE. It used to open this burst -- "Yesterday: 24
        # turns, median wait 1.4 seconds, worst 6.1. I dropped 7 clips of
        # yours at the speaker gate" -- and it is instrumentation, not
        # news: turn counts and speaker-gate rejections are how the
        # ASSISTANT is doing, which is a thing to ask for and never a thing
        # to be handed. He asked for a briefing; he gets a briefing. It
        # stays one sentence away: "how did yesterday go" (the "day review"
        # command -> services.dayreview -> app.day_review_text) reads the
        # same digest, and the nightly file and the Discord post are
        # untouched.
        # THE JOIN, and once the longest burst in the live log: measured at
        # 14:33:49-14:34:29 as 40 seconds of unbroken speech over seven TTS
        # segments carrying FOUR sirs. What is left -- the weekly lines and
        # the hand-over -- is still consecutive _say calls with nothing
        # between them, so it is ONE burst and has to be thinned against
        # itself the way the arrival cue is: fragments, never a joined
        # string.
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

        # The two weekly lines, if either is owed. Both are produced in the
        # small hours by their own threads and deliberately NOT spoken
        # there: this is the path the quiet gate stands in front of (the
        # offer that leads here refuses inside quiet hours), so a report
        # written at 3 am is heard at breakfast and never at 3 am. On a
        # declined day neither is spoken and neither is lost -- each stays
        # pending until a day he says yes.
        for line in (self._pending_week_line(), self._pending_garden_line()):
            say_in_burst(line)
        # The model's own reply lands after this through brain.chat and is
        # NOT in the ledger: it is one authored-shaped line with exactly one
        # sir (24/24 measured), and catching it would mean rewriting at the
        # TTS door, which is the thing this design does not do. So the
        # burst keeps exactly one sir of its own and the reply keeps its.
        # A briefing that is being DELIVERED retires any parked offer of one.
        # 2026-09-04 15:06: the model answered his briefing inside a compound
        # question, the arrival offer from 14:44 then asked "Shall I run your
        # briefing, sir?", he said yes, and the calendar and the lab were read
        # to him a second time. The offer rung only ever cleared itself when
        # it was answered; nothing cleared it when the thing it offered had
        # already happened.
        try:
            self.services.briefing_offer = None
        except Exception:  # noqa: BLE001 - no services, nothing parked
            pass
        say_in_burst("Your briefing for today, sir.")
        # arc.greeting_word, not the literal "morning": every delivery in
        # the two retained logs (08-31 14:33, 09-01 15:00, 09-02 14:29)
        # asked the model for "my morning briefing" in the AFTERNOON,
        # because briefing.after has a floor and no ceiling. One
        # time-of-day rule for the house (jarvis/arc.py), not a second one
        # here.
        when = arc_mod.greeting_word(datetime.now())
        # Proactive, and his: the calendar and the lab it reads are not
        # scoped by whoever the gate last named (round-2 review, 09-04).
        self._the_turn_is_his()
        try:
            brain.chat(f"my {when} briefing", force_tool="get_briefing",
                       addressee=scope_mod.OWNER)
        except Exception:
            log.exception("first-wake briefing failed")
            return False
        self._mark_briefing_delivered()     # after the ask, not before
        return True

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
        state = self.self_state()
        # The weekly Knightfall caption rides on the card and the socket
        # reply only: this sheet's film-register rendering (what he HEARS)
        # does not read the key, so nothing about a code is ever spoken.
        state.setdefault("knightfall_weekly", self._knightfall_weekly_caption())
        return selfstate.diagnostics_line(state)

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
            tasks=lambda: dict(getattr(self, "_board_tasks", {})),
            # the last thing he threw (jarvis/gesturecast.py), or None
            cast=getattr(getattr(self, "gesture", None), "recent", None))

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
            # The day, or the slab reads "BIOSENSORS 12:45 pm" at 8 in the
            # evening and says nothing about WHICH 12:45 that is. "today" is
            # left off on purpose: the standby face is a clock, so a row
            # with no day word can only mean the day it is already showing.
            from jarvis.tools.calendar import day_label
            when = ev.start.strftime('%-I:%M %p').lower()
            day = day_label(ev.start.date(), now.date())
            return f"{ev.title} {when}" if day == "today" else \
                f"{ev.title} {day} {when}"
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
            state = str(getattr(pres, "state", "") or "")
            # "unknown" is the state before the first poll answers. A WHERE
            # row reading "unknown" is worse than no row, for the same
            # reason the docstring above refuses a confident false HOME.
            room["presence"] = state if state in ("home", "away") else ""
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
        # holds=N only when the filler hold fired: a turn without one keeps
        # the line it always had. spell=N is its own note rather than more
        # holds=N, because a spelled turn is a DIFFERENT wait to explain --
        # a man saying an address one character at a time, not a man
        # thinking after an um -- and a week of turns has to be able to
        # tell the two apart (jarvis/spelling.py, 09-05).
        holds = int(getattr(ev, "filler_holds", 0) or 0)
        spell = int(getattr(ev, "spell_holds", 0) or 0)
        notes = {}
        if holds:
            notes["holds"] = str(holds)
        if spell:
            notes["spell"] = str(spell)
        self.turns.mark("stop", at=ev.t, stop=ev.endpoint or ev.reason, **notes)

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
        if getattr(ev, "amplitude_only", False):
            # The feeder's mouth-close tick. It carries no edge, and taking
            # it as one latched _tts_active True for the rest of the boot --
            # which silently disables the nudge and the guest decline.
            return
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
        self._audio_started()
        t = threading.Thread(target=self._process_audio, args=(audio,),
                             daemon=True, name="audio-decode")
        # Remembered so the watchdog can print WHERE this thread is if it
        # never comes back. On 2026-09-06 nobody could say where the decode
        # hung -- py-spy needs a privilege this account does not have -- and
        # the fix had to be a timeout over an unknown. Next time the log
        # says which line it was standing on.
        self._audio_thread = t
        t.start()

    # ------------------------------------------------------- the owner gate
    def _face_running(self) -> bool:
        """Is the camera leg MEASURING at all? Not "is he in frame".

        A dark camera, the night curfew, no feed attached and identity
        switched off are all NO OPINION -- never a vote against him -- and
        the gate has to be able to tell that apart from a camera that looked
        and saw somebody else.
        """
        feed = getattr(getattr(self, "services", None), "camera_feed", None)
        if getattr(feed, "eye", None) is None:
            return False
        return bool(self.get_option("camera.identity", False))

    def _refuse_politely(self, line: str):
        """Say a refusal once, under the SAME policy _on_guest already uses.

        One cooldown between them (his complaint was hearing it twice in a
        minute), silence over music -- a "guest" over a bed is far more often
        him, scored down -- and silence while Jarvis is speaking, because
        under barge-in the listener hears his own voice.
        """
        if not line or not CONFIG.talkback:
            return
        if getattr(getattr(self, "tts", None), "busy", False) is True or \
                getattr(self, "_tts_active", False):
            return
        if self._music_playing():
            log.info("owner-gate: refusing over music, staying quiet")
            return
        now = time.monotonic()
        if now - self._last_guest_ts < 180.0:
            return
        self._last_guest_ts = now
        self._say(line)

    def _gate_quiet_decode(self, audio):
        """THE DECODE THAT MAY BE A SECRET, and the only one taken behind
        the gate's back. ``transcribe_quiet`` is the same decode with the
        words left out of jarvis/transcriber.py's INFO line -- which is
        upstream of every redaction this module does, and so was where the
        phrase he says in shadow actually landed (verdict, 2026-09-05).

        Duck-typed on purpose: the transcriber is a seam (jarvis/intercom.py
        hands clips in over the command socket, the tests stand in their
        own), and a stand-in without the quiet decode still works -- it
        simply logs what it logged before.
        """
        fn = getattr(self.transcriber, "transcribe_quiet", None)
        if fn is None:
            log.debug("owner-gate: this transcriber has no redacting decode")
            fn = self.transcriber.transcribe
        return fn(audio)

    def _gate_rescue(self, audio, stats, speculative, turn=""):
        """A clip the speaker filter dropped: `(audio, stats, result)` to let
        it through after all, or None to keep today's behaviour.

        ``turn`` IS THE LEDGER'S, NOT THE GATE'S. It changes no decision
        here; it stamps both verdicts of one utterance with one id so the
        scorecard counts a rescued clip as ONE turn (jarvis/gateledger.py).

        THE SPOKEN PASSPHRASE COSTS ONE DECODE, and only here. It arrives as
        a Whisper transcript, and a rejected clip is never transcribed -- so
        without this the phrase could never be heard at all. The decode runs
        only when the gate is actually refusing AND an owner has set a
        phrase, so an ordinary rejected clip pays nothing.

        NOTHING CARRYING THE PHRASE IS EVER PUBLISHED. On a match the turn
        ends right here with a spoken acknowledgement and an open mic: the
        text never reaches the bus, the history, the transcript pane, the
        turn ledger or commander's own log line, which is where a plaintext
        secret would otherwise have landed four times over.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return None
        try:
            return self._gate_rescue_inner(gate, audio, stats,
                                           speculative=speculative,
                                           turn=turn)
        except Exception:                          # noqa: BLE001 - never fatal
            log.exception("owner-gate: the rescue failed; the clip is "
                          "dropped exactly as it was before")
            return None

    def _gate_rescue_inner(self, gate, audio, stats, speculative=False,
                           turn=""):
        face, running = self._eye_identity(), self._face_running()
        # THE PHRASE COSTS ONE DECODE, IN EVERY MODE, and only when an owner
        # has set one. It is the ONE thing that acts in shadow: a rejected
        # clip that is the phrase is consumed (Knightfall, 2026-09-04 --
        # "so he can test it tonight"); everything else shadow still only
        # logs. Judged once, with the words, so the gate can hear it.
        result, text = None, ""
        if self._owner_has_phrase():
            result = self._gate_quiet_decode(audio)
            text = (getattr(result, "text", "") or "").strip()
        d = gate.judge("voice", text, stats=stats, rejected=True,
                       face=face, face_running=running, turn=turn)
        if d.consumed:
            self._gate_consumed(d, getattr(result, "confidence", 0.0),
                                speculative)
            return PHRASE_CONSUMED
        if d.admit and d.how in WINDOW_HOWS:
            # A WINDOW HE OPENED HIMSELF ACTS IN EVERY MODE THAT OPENS ONE,
            # and this is the line that makes "Thank you, sir. I'm
            # listening." true. It used to sit BELOW the shadow return, so
            # in shadow -- HIS LIVE MODE -- the phrase said it was
            # listening and then the speaker filter dropped his very next
            # clip, exactly as it had dropped the one that made him say the
            # phrase. Measured 2026-09-05: dispatched == [] in shadow
            # against [("what time is it", "voice")] in enforce, for the
            # phrase's window and the typed code's window alike.
            #
            # It is not shadow leaking. Shadow's rule is that the gate must
            # not start ACTING ON ITS OWN JUDGEMENT -- a face leg answering
            # clips he never asked it to answer is a behaviour change he did
            # not ask for, and it still only logs, below. This is the
            # opposite thing: he said the phrase, or he typed the code, and
            # the only content of either is "let me in". The phrase already
            # acts in shadow by consuming the turn and speaking a line; the
            # five minutes it buys is the same instruction, and refusing to
            # honour it made the feature inert in the one mode he runs.
            log.info("owner-gate: the %s window rescued a clip the speaker "
                     "filter dropped (%s, mode=%s)", d.how, d.who,
                     gate.effective_mode())
            if result is None:
                result = self.transcriber.transcribe(audio)
            return audio, stats, result
        if gate.effective_mode() != gate_mod.MODE_ENFORCE:
            # SHADOW CHANGES NOTHING ELSE, and that has to include the
            # rescues. A face leg that started answering clips the speaker
            # filter dropped would be a visible change of behaviour he did
            # not ask for yet -- and the whole value of shadow is that it is
            # safe to leave on while the log is read. The verdict is logged
            # inside judge() either way, which is the point of the mode.
            if d.admit and d.how in RESCUE_HOWS:
                log.info("owner-gate: shadow -- the %s leg WOULD have "
                         "rescued this clip for %s", d.how, d.who)
            return None
        if d.admit and d.how in RESCUE_HOWS:
            log.info("owner-gate: the %s leg rescued a clip the speaker "
                     "filter dropped (%s)", d.how, d.who)
            if result is None:
                result = self.transcriber.transcribe(audio)
            return audio, stats, result
        if d.admit:
            return None                # off or blind: nothing changes
        self._refuse_politely(d.line)
        return None

    def _gate_judge(self, text, stats, turn=""):
        """One verdict for the turn, or None when there is no gate or it
        could not decide (both mean: the turn stands, exactly as today).

        ``turn`` only reaches the ledger row -- see ``_gate_rescue``."""
        gate = getattr(self, "gate", None)
        if gate is None:
            return None
        try:
            return gate.judge("voice", text, stats=stats,
                              face=self._eye_identity(),
                              face_running=self._face_running(), turn=turn)
        except Exception:                          # noqa: BLE001 - never fatal
            log.exception("owner-gate: judging failed; the turn stands")
            return None

    def _gate_consumed(self, d, confidence=0.0, speculative=False):
        """The spoken phrase, answered by the gate: say the line, re-open
        the mic, close the turn in the ledger, and publish ONLY the
        redaction -- no UserUtterance, no commander, no model. The log line
        carries who and which path; the text is never anywhere."""
        # THE GLASS FIRST, BEFORE ANYTHING ELSE GETS TO RUN. While he speaks,
        # _partial_loop and _maybe_speculate publish PartialText(<the words>)
        # and the pane draws a ghost card. Publishing the redaction does NOT
        # take it down: MainWindow._ev_transcribed clears the partial only
        # when the event is NOT accepted or is empty, and this event is
        # accepted and non-empty, while add_user -- the pane's other clearer
        # -- never runs because the whole point is that no UserUtterance
        # follows. So the phrase stayed legible on screen after a log line
        # had been carefully redacted in front of it. Measured 2026-09-05 in
        # all three modes: transcript calls were [("show", <the phrase>)]
        # with no ("clear", "").
        self._drop_partial()
        log.info("owner-gate: the phrase turn is consumed for %s; nothing "
                 "dispatched (mode=%s)", d.who, self.gate.effective_mode())
        bus.publish(Transcribed(text=d.redact or gate_mod.REDACTED_TEXT,
                                confidence=float(confidence or 0.0),
                                accepted=True, speculative=bool(speculative)))
        self.turns.abandon("gate:phrase")
        self._say(d.line or gate_mod.PHRASE_OK_LINE)
        self._followup_after_speech = True

    def _gate_admits(self, text, stats, decision=None) -> bool:
        """The admitted path: attribute the turn, and hold a KNOWN person to
        what a known person may ask for. It cannot refuse HIM -- the speaker
        filter has already matched him and this reuses that verdict.
        ``decision`` is the verdict _process_audio already took for this
        turn, so the gate (and its key derivation) runs once, not twice."""
        d = decision if decision is not None else self._gate_judge(text, stats)
        if d is None:
            return True
        if not d.admit:
            # NOT ATTRIBUTED. Round 3 measured this the other way round:
            # the three lines below used to run BEFORE this check, so a
            # voice turn the gate REFUSED still left that person owning
            # the process-wide scope for the full 120 s TTL -- after
            # _gate_admits("read me my mail") returned False,
            # scope.addressee() was ("Heather", "ma'am"). A refused turn
            # is nobody's; whatever held the scope before still holds it.
            self.turns.abandon("gate:%s" % (d.role or "unknown"))
            self._refuse_politely(d.line)
            return False
        self._gate_who, self._gate_how = d.who, d.how
        self._gate_who_ts = time.monotonic()
        # THE READING FOR THIS TURN, taken at the instant the turn is
        # attributed and read back by _process_audio one statement later
        # on this same thread -- never looked up again after a wait.
        self._gate_addressee = self._tell_the_model_who_is_here(d.who)
        if d.line:
            # The sign-in welcome. Only ever set on an admitted turn where
            # a name was CONFIRMED by a leg, and rate-limited inside the
            # gate, so this cannot become a preamble on every sentence.
            self._say(d.line)
        return True

    def _process_audio(self, audio):
        stats = {}
        # ONE STAMP FOR ONE THING HE SAID. A clip the speaker filter dropped
        # and a window rescued is judged TWICE below -- once inside
        # _gate_rescue with rejected=True, and again at `verdict =
        # self._gate_judge(...)` once the rescue has cleared `rejected`.
        # Both verdicts are real and both belong in the ledger; without a
        # shared id the scorecard reports one utterance as two turns and
        # two window admits, and it does so on exactly the path his FIRST
        # Knightfall test takes. A timestamp rather than a counter on
        # purpose: gate.jsonl outlives a restart, and a counter that began
        # again at 1 would fold two boots' turns together.
        tid = "%.4f" % time.time()
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
                # THE ONE PLACE THE OWNER GATE CAN RESCUE A TURN. Until now a
                # clip the speaker filter dropped was silently gone -- which
                # is the right default and is also a dead end. The gate gets
                # to say EITHER LEG SUFFICES (the camera may name him when
                # his voice will not) and to hear the spoken passphrase,
                # which is his way back in when he is ill, in the dark, or
                # turned away. Anything else and today's behaviour stands.
                rescued = self._gate_rescue(audio, stats, spec is not None,
                                            turn=tid)
                if rescued is PHRASE_CONSUMED:
                    return             # the gate answered it; nothing else
                if rescued is None:
                    bus.publish(Transcribed(
                        text="", accepted=False, reject_reason="speaker",
                        speaker_score=float(stats.get("best_score", 0.0))
                        if isinstance(stats, dict) else 0.0,
                        speculative=spec is not None))
                    self._nudge("speaker")
                    return
                audio, stats, result = rescued
                rejected = False
            text = result.text.strip()
            # THE GATE, BEFORE THE TRANSCRIPT CAN REACH THE BUS. The spoken
            # phrase is consumed here in every mode (Knightfall): the turn
            # ends with the line and the redaction, and the words never
            # become a Transcribed, a UserUtterance or a dispatch. Judged
            # once; the admitted path below reuses this verdict. A
            # low-confidence transcript is judged too, when an owner has a
            # phrase: an exact match after normalisation is better evidence
            # than a length-biased score, and not checking would put the
            # phrase on the bus as a rejected transcript.
            verdict = None
            if text and (result.accepted or self._owner_has_phrase()):
                verdict = self._gate_judge(text, stats, turn=tid)
                if verdict is not None and verdict.consumed:
                    self._gate_consumed(verdict, result.confidence,
                                        spec is not None)
                    return
            # THE WORDS ARE WRITTEN DOWN HERE, NOT IN THE DECODE, whenever
            # an owner has a phrase set: _clip_decode redacted the line so
            # that a phrase-shaped clip could be judged before anything
            # logged it, and this is the gate saying it was ordinary.
            self._log_transcript(result)
            # The confidence gate is no longer the last word. It used to
            # fire BEFORE the commander saw a syllable, so a plain "Yes."
            # answering Jarvis's own "Clear all three off your shopping
            # list, sir?" was dropped and nothing happened (2026-08-31,
            # 20:58:35, avg_logprob -0.95). _salvage_low_confidence names
            # the reason to run it anyway, or "".
            accepted = result.accepted
            # A repetition loop is not a length-biased short transcript, and
            # the salvage exists only for those: with a yes/no read-back
            # parked, "yes, yes, yes, yes, yes" would otherwise be salvaged
            # into a destructive confirm. See transcriber.loop_ratio_limit.
            # getattr: decode_clip() is a public seam (jarvis/intercom.py
            # hands a clip in over the command socket) and its result only
            # has to quack like a TranscribeResult.
            looping = bool(getattr(result, "looping", False))
            salvage = ""
            if not accepted and text and not looping:
                salvage = self._salvage_low_confidence(text, result.confidence)
                if salvage:
                    log.info("low confidence (%.2f) overridden -- %s: %r",
                             result.confidence, salvage, text)
                    accepted = True
            bus.publish(Transcribed(
                text=result.text, confidence=result.confidence,
                accepted=accepted,
                reject_reason=("" if accepted else
                               "looping" if looping else "confidence"),
                speculative=spec is not None))
            if accepted and text:
                # The gate, on the admitted path: it attributes the turn and
                # holds a KNOWN person to what a known person may ask for.
                # It cannot refuse HIM here -- the speaker filter has already
                # matched him, and this reuses that verdict rather than
                # taking one of its own.
                if not self._gate_admits(text, stats, decision=verdict):
                    return
                # The reading _gate_admits just installed, carried by
                # value. Only this thread writes it and only this thread
                # reads it, one statement apart, with no lock between --
                # so unlike the module state it cannot be flipped by
                # another turn while this one queues.
                turn_addr = getattr(self, "_gate_addressee", scope_mod.OWNER)
                self._say_again_count = 0
                self._maybe_learn_voice(audio, stats)
                bus.publish(UserUtterance(text=text, source="voice"))
                self._dispatch(text, "voice", confidence=result.confidence,
                               addressee=turn_addr)
            elif text:
                # Garbled, not silent: say so and re-open the mic rather
                # than routing "by Agenda 4.2.6" or going quiet -- once.
                # Twice in a row is not the user mumbling, it is the room
                # (a television, music) reaching the follow-up mic, and
                # asking again re-opens that mic without end.
                self._say_again_count = getattr(self, "_say_again_count", 0) + 1
                if self._say_again_count > 1:
                    # ...but the second strike used to be an EARCON, and
                    # the earcon is cooldown-suppressed, so the honest
                    # outcome was often nothing at all -- indistinguishable
                    # from Jarvis ignoring him (2026-08-31: "Yes (low
                    # confidence, didnt do it)"). Say it out loud instead,
                    # and pointedly WITHOUT _followup_after_speech: it is
                    # the re-opened mic, not the sentence, that lets a
                    # television loop the exchange.
                    log.info("low confidence again (%.2f): %r -> saying so, mic stays shut",
                             result.confidence, text)
                    self.turns.abandon("rejected:confidence")
                    self._say(NOT_CAUGHT_LINE)
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
            # A `finally` covers a RAISE and NOT A HANG, which is why
            # _audio_started arms a watchdog as well -- see _audio_timed_out.
            #
            # REACHED DEFENSIVELY, and that is the point rather than a
            # concession. This whole commit exists because the flag was not
            # released; a release path that can itself raise AttributeError
            # would be the same bug wearing a helper. The direct clear is
            # the floor and always runs. (It is also what lets the many
            # tests that drive this function on a hand-built namespace --
            # one that owns the flag and nothing else -- keep working
            # without teaching each of them about the watchdog.)
            try:
                self._audio_cancel_watchdog()
            except AttributeError:
                pass
            self._audio_busy.clear()

    # The Tier-1 probe is a whole-utterance matcher ("volume 40",
    # "cancel my alarm", "full brightness"), not a keyword search, so a
    # character-salad transcript cannot match one by accident.
    def _salvage_low_confidence(self, text: str, confidence: float = 0.0) -> str:
        """Why a sub-threshold transcript is being run anyway, or "".

        Whisper's avg_logprob is length-biased, so the very utterances the
        gate is worst at are the SHORT ones -- and short is what an answer
        and a command both look like. On 2026-08-31 that cost Hunter nine
        features at once (his #20/#24/#32/#33/#67/#68/#87/#101/#144), all
        with the same symptom: "low confidence", nothing happened.

        Two rescues, both narrow:

        * **Jarvis asked.** A yes/no read-back ("Clear all three off your
          shopping list, sir?"), an objection, a flashcard, a working
          session or the router's "Shall I hand that to Claude, sir?" is
          on the table. Dropping the answer is the worst failure of the
          set: he already committed to acting on the next word, and the
          rungs that read it (``_try_destructive_confirm`` and friends)
          each demand a CLEAR yes or no and let anything else fall through
          as a fresh subject -- so a genuinely garbled answer is still
          safe, it just stops being invisible.

          Deliberately not the open debrief: that one FILES the words as
          the memory of how his midterm went, and filing a garble there is
          worse than asking again (see ``_debrief_reply``).

        * **It is a Tier-1 command verbatim.** ``_match_assistant`` is the
          same probe the intent gate uses to spare exact matches from the
          classifier's guess. If the words ARE "cancel my alarm", a
          confidence score has no business second-guessing them.
        """
        commander = getattr(self, "commander", None)
        # A live flashcard FILES the answer -- _quiz_store.record marks the
        # card wrong and session.settle() burns it -- which is exactly the
        # objection that already excludes the open debrief. Found in review
        # before this shipped: with a card parked (a 300 s window) the
        # session's own character salad at -5.68 would have reached the
        # grader and cost him a permanent wrong mark.
        #
        # Note what is NOT here: a score floor. The danger was never the
        # number, it is what CONSUMES the words. A destructive read-back
        # demands a clear yes or no and lets anything else fall through as
        # a fresh subject, so salvaging a garble there costs a puzzled
        # reply; the rungs that FILE are the ones that must be excluded by
        # name, and they are.
        if getattr(commander, "_pending_quiz", None):
            return ""
        # And the walk question FILES too: _try_leave_answer runs
        # leavetime.learn(key, minutes) -- a permanent walk time -- on
        # anything leavetime.answer_minutes reads as a duration, and it
        # reads "a bow ten minutes" and "uh ten minute" as ten. Its rung
        # stands for LEAVE_ANSWER_WINDOW_S (180 s) after a question Jarvis
        # asked on its own initiative and he may never have heard, so for
        # three minutes any sub-threshold garble containing a numeral would
        # have been run instead of dropped. Excluded BY NAME, per the rule
        # above; the age is checked because _pending_leave is cleared
        # lazily and a stale tuple must not gate the salvage for ever.
        age = self._leave_pending_age(commander)
        if age is not None and age <= LEAVE_ANSWER_WINDOW_S:
            return ""
        try:
            if self._question_open(commander):
                return "answering a question Jarvis asked"
        except Exception:  # noqa: BLE001 - a slim/duck-typed commander
            log.debug("salvage: question_open failed", exc_info=True)
        # The "Was that for me?" card is app-side, not a commander rung,
        # and its whole point is that the next word resolves it.
        if getattr(self, "_pending_uncertain", None):
            return "answering the uncertain-intent prompt"
        probe = getattr(commander, "_match_assistant", None)
        if callable(probe):
            try:
                name = probe(text)
            except Exception:  # noqa: BLE001 - a matcher blew up
                log.debug("salvage: tier-1 probe failed", exc_info=True)
                name = None
            if name:
                return f"exact tier-1 command ({name})"
        return ""

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
        # `display_only` is SHOWN and never SPOKEN (commander.CommandResult):
        # the spoken text below stays exactly `result.reply`, and only the
        # published/displayed text carries the extra line. The clipboard
        # hand-over depends on this -- the command must reach the transcript
        # every time, whether or not the clipboard took it.
        extra = getattr(result, "display_only", None)
        if result.reply or extra:
            shown = result.reply or ""
            if extra:
                shown = "%s\n%s" % (shown, extra) if shown else str(extra)
            bus.publish(JarvisReply(text=shown, speak=result.speak))
            if result.speak and result.reply:
                if getattr(result, "ack", False):
                    # "Looking that up, sir." is speech, not the answer: the
                    # turn ledger records it as a filler and keeps waiting.
                    # It IS the filler, too: 4.5 s later "Checking right
                    # now, sir" followed it live. Disarm the thinking timer
                    # and leave the watchdog.
                    self._turn_filler_pending = True
                    self._disarm_filler()
                self._say(result.reply)
        if result.status:
            bus.publish(Status(text=result.status, kind="info"))
        return result

    _thinking_delay_s = THINKING_DELAY_S
    _turn_timeout_s = TURN_TIMEOUT_S
    # The filler's guards as CLASS defaults, not only __init__ ones. Several
    # test suites build an app with object.__new__ and call _dispatch/_say on
    # it; __init__ shadows all three with per-instance state.
    _filler_lock = threading.Lock()
    _turn_seq = 0
    _turn_answered = False
    _uncertain_turn = None

    def _dispatch(self, text, source, confidence=None, addressee=None):
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
        # THE ONE READING OF WHOSE TURN THIS IS, taken HERE -- where the
        # turn is attributed -- and carried by value into the debrief, the
        # commander and the bookkeeping below. Round 3 (09-05) measured
        # what looking it up downstream costs: the commander re-read this
        # module state inside its own turn lock, i.e. after an unbounded
        # wait, and with the lock contended 200/200 trials answered a
        # guest from his notes and 198/200 refused him his own. Nothing
        # below this line looks the scope up again.
        turn_addr = scope_mod.OWNER
        if source == "voice":
            self._turn_start()
            self.turns.mark("handle")
            turn_addr = scope_mod.reading(addressee)
        elif source not in gate_mod.GATED_SOURCES:
            # Not judged by the gate, so nobody but him: the keyboard, the
            # socket, his phone, Discord. The attribution a guest's voice
            # turn left behind must not scope HIS typed turn.
            turn_addr = self._the_turn_is_his()
        # The Whisper avg_logprob travels only when there is one: typed
        # text has none, and a stand-in commander need not take the keyword.
        kw = {} if confidence is None else {"confidence": confidence}
        try:
            # An open debrief question owns this transcript unless it is
            # plainly a command -- the answer is FILED, never routed to the
            # model as chat (jarvis/debrief.py).
            filed = self._debrief_reply(text, source, turn_addr)
            result = self._emit_result(
                filed if filed is not None
                else self.commander.handle(text, source,
                                           addressee=turn_addr, **kw))
            corrected = getattr(result, "corrected", None)
            if corrected:
                self._last_user_text = corrected
            if source == "voice":
                self._turn_after_result(result)
            self._after_dispatch(text, source, result, turn_addr)
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
        # A new turn: nothing has been said for it yet, and any filler timer
        # still in flight from the previous turn belongs to a turn that is
        # over -- it carries the old sequence number and drops itself.
        # The flag is raised INSIDE the same lock as the sequence bump, so
        # an expiry that checks the sequence under this lock
        # (_uncertain_unanswered) sees either the old number with the old
        # turn or the new number with this one -- never a raised flag it
        # could mistake for the turn it is allowed to close.
        with self._filler_lock:
            self._turn_seq += 1
            self._turn_answered = False
            seq = self._turn_seq
            self._turn_busy.set()
        if CONFIG.talkback:
            self._turn_timer = threading.Timer(self._filler_delay_s(),
                                               self._say_thinking, args=(seq,))
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
        # AND THE OTHER FLAG, because this line promises the wake word back.
        # On 2026-09-06 it cleared _turn_busy, printed exactly that sentence,
        # and left _audio_busy set -- so the wake word stayed dead and every
        # hotword for the next two minutes logged "still transcribing the
        # previous clip". Releasing one half of a two-flag gate is not
        # releasing the gate.
        self._audio_finished(reason="the turn watchdog")
        self.turns.abandon("timeout")

    # ------------------------------------------- the audio flag and its clock
    def _audio_started(self) -> None:
        """Hold the microphone gate for this clip, and arm its release.

        THE FLAG IS THE EASY HALF. `_process_audio` runs on its own daemon
        thread and clears the flag in a `finally`, which covers every way
        that function can RAISE and no way it can HANG. A hang is what
        happened on 2026-09-06: the thread never returned, the finally never
        ran, and `_should_record` dropped five wake words at 0.85-0.98
        confidence over ninety seconds with the line "still transcribing the
        previous clip". He described it as soft locked, and it stayed that
        way for 62 minutes until the process was killed.

        The watchdog is the half that survives a hang. It is armed here
        rather than in the caller so there is ONE place that sets this flag
        and it cannot be set without a release being armed -- pinned by the
        census in tests/test_audio_busy_never_wedges.py.
        """
        self._audio_busy.set()
        self._audio_cancel_watchdog()
        try:
            timeout = float(getattr(self, "_audio_timeout_s", AUDIO_TIMEOUT_S))
            self._audio_watchdog = threading.Timer(timeout,
                                                   self._audio_timed_out)
            self._audio_watchdog.daemon = True
            self._audio_watchdog.start()
        except Exception:      # noqa: BLE001 - a timer we cannot arm is not
            # a reason to refuse the clip; it is a reason to say so. The
            # flag is still cleared by the finally on every non-hang path.
            log.exception("audio watchdog could not be armed; a hung decode "
                          "would wedge the wake word")

    def _audio_finished(self, reason: str = "") -> None:
        """Release the gate and disarm the clock. Safe to call twice."""
        self._audio_cancel_watchdog()
        if reason and self._audio_busy.is_set():
            log.warning("audio flag released by %s", reason)
        self._audio_busy.clear()

    def _audio_cancel_watchdog(self) -> None:
        t = getattr(self, "_audio_watchdog", None)
        if t is not None:
            try:
                t.cancel()
            except Exception:  # noqa: BLE001 - a dead timer is already off
                log.debug("audio watchdog cancel failed", exc_info=True)
            self._audio_watchdog = None

    def _audio_timed_out(self) -> None:
        """The decode never came back. Give him his microphone.

        The orphan thread is left to finish or not: it holds no lock, its
        own finally clears an already-clear flag, and the turn machinery
        stamps replies with a sequence so a late one cannot be mistaken for
        an answer to the next question. Waiting for a thread that is by
        definition stuck is the one thing that must not happen here.
        """
        self._audio_watchdog = None
        if not self._audio_busy.is_set():
            return
        log.error("audio watchdog fired after %.0fs: the last clip never "
                  "finished decoding. Releasing the wake word -- it was "
                  "being dropped as 'still transcribing the previous clip'.",
                  float(getattr(self, "_audio_timeout_s", AUDIO_TIMEOUT_S)))
        # THE INSTRUMENT. Where is the decode thread standing right now?
        # This is the question nobody could answer on 2026-09-06, and it is
        # answerable from inside the process with no privilege at all.
        for line in _stuck_thread_stack(getattr(self, "_audio_thread", None)):
            log.error("audio watchdog: %s", line)
        self._audio_busy.clear()
        try:
            bus.publish(Status(text="Sorry, sir -- I lost that one",
                               kind="warn"))
        except Exception:      # noqa: BLE001 - the UI is not the point here
            log.debug("could not publish the lost-clip status", exc_info=True)
        try:
            self.turns.abandon("audio_timeout")
        except Exception:      # noqa: BLE001 - the ledger is not the point
            log.debug("could not abandon the turn", exc_info=True)

    def _turn_cancel_timers(self):
        for name in ("_turn_timer", "_turn_watchdog"):
            t = getattr(self, name, None)
            if t is not None:
                try:
                    t.cancel()
                except Exception:
                    log.debug("timer cancel failed", exc_info=True)
                setattr(self, name, None)

    def _disarm_filler(self):
        """Cancel the pending "one moment" line, leaving the watchdog armed.

        Timer.cancel() only wins the race it can see: once _say_thinking has
        started running on the timer thread this is a no-op, which is why the
        latch below exists as well."""
        t, self._turn_timer = getattr(self, "_turn_timer", None), None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                log.debug("filler cancel failed", exc_info=True)

    def _note_spoke(self):
        """Record that this turn has already put a line in the TTS queue, so
        no filler may follow it. Called from _say, the one door to speech."""
        with self._filler_lock:
            self._turn_answered = True
        self._disarm_filler()

    @staticmethod
    def _is_busy_tts(tts) -> bool:
        """True only on a REAL busy signal: TTS.is_speaking is a bool and
        TTS.pending an int (jarvis/tts.py). Typed strictly because the test
        doubles are duck-typed stubs whose __getattr__ hands back a callable
        for every name it does not define -- and a bound method is truthy,
        which would silence the filler everywhere instead of only when
        something really is queued ahead of it."""
        if getattr(tts, "is_speaking", False) is True:
            return True
        pending = getattr(tts, "pending", 0)
        return type(pending) is int and pending > 0

    def _tts_busy(self) -> bool:
        return self._is_busy_tts(getattr(self, "tts", None))

    def _filler_delay_s(self) -> float:
        """When to speak "one moment" -- scaled to how fast this box answers.

        A fixed delay ages badly: 4.5 s was "unusually slow" when the median
        wait was 10.68 s and is ordinary now that it is 1.30 s. Three times the
        recent median is the same JUDGEMENT at any speed. Falls back to the
        tuned constant until the ledger has enough answered turns to have an
        opinion, and never drops below it -- the filler must not become more
        eager than the value that was measured by hand."""
        floor = self._thinking_delay_s
        typical = None
        turns = getattr(self, "turns", None)
        if turns is not None:
            try:
                typical = turns.typical_wait_s()
            except Exception:           # noqa: BLE001 - a probe, never fatal
                typical = None
        if typical is None:
            return floor
        return max(floor, min(FILLER_DELAY_MAX_S, typical * FILLER_WAIT_MULTIPLE))

    def _turn_finished(self):
        """The answer landed (or gave up). Also called from the brain callback."""
        self._turn_cancel_timers()
        self._turn_busy.clear()

    def _say_thinking(self, seq=None):
        """Acknowledge a slow lookup. Rotates so it does not become a tic.

        Every guard here is a LAST-MOMENT one, checked on the timer thread
        microseconds before speaking, because threading.Timer.cancel() cannot
        recall a callback that has already started -- and the TTS queue is
        FIFO, so a filler that queues even one millisecond late is spoken
        AFTER the answer instead of ahead of it. That is exactly what Hunter
        heard on 2026-08-31: the Oracle answer at 21:36:37.322 played to
        completion and "Checking right now, sir. One moment." began at
        21:36:41.696, on the very next slot in the queue.
        """
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
        with self._filler_lock:
            if seq is not None and seq != self._turn_seq:
                # This timer belongs to a turn that has already been replaced.
                log.info("thinking line dropped: a newer turn is open")
                return
            if self._turn_answered:
                # _say already ran for this turn: the user HAS the answer
                # (#154 "had the answer then said checking one moment sir").
                log.info("thinking line dropped: the answer is already out")
                return
            if self._tts_busy():
                # Something is mid-burst or queued ahead of us, so this line
                # could only land after it. A filler that arrives late is
                # worse than no filler at all.
                log.info("thinking line dropped: speech is already in the queue")
                return
            # Claim the turn before releasing the lock: an answer arriving now
            # disarms nothing (we are past cancel), but it also must not make
            # a SECOND line think it is still first.
            self._turn_answered = True
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
        # WHICH TURN is waiting on this question. The commander asks only on
        # the voice path, and _dispatch opened that turn (_turn_start)
        # before calling it, so this is the prompt's own sequence number:
        # the one the unanswered exit may close, and the one _on_hotword
        # names when it refuses a wake word in the meantime.
        with self._filler_lock:
            seq = self._turn_seq
            self._uncertain_turn = seq
        for old in stale:
            bus.publish(UncertainResolved(request_id=old, yes=False,
                                          source="superseded"))
        bus.publish(UncertainUtterance(
            request_id=rid, text=text,
            question=f'Was that for me? — "{text[:60]}"'))
        # "Was that for me?" is speech but not an answer: close the ledger
        # before the ask thread can publish SpeakingState for it.
        self.turns.abandon("uncertain")
        # One argument, as before: the ask thread reads the turn number off
        # _uncertain_turn at entry, so a stub bound as `lambda rid: None`
        # (test_app_wiring) still fits the seam.
        threading.Thread(target=self._ask_uncertain, args=(rid,),
                         daemon=True, name="uncertain-ask").start()

    def _ask_uncertain(self, rid: str, seq=None):
        """Say it out loud, then listen briefly for a spoken yes/no.

        Blocks on the TTS before recording: talk-back holds the mic arbiter,
        but the arbiter is a depth counter rather than a mutex, so without the
        wait we would happily record Jarvis asking the question.

        EVERY WAY OUT OF HERE ENDS THE TURN. The prompt left it open
        (done=False) so an answer could arrive; when none does -- an
        enrolment owns the mic, there is no mic, nothing was captured, the
        speaker filter refused the reply, the words were not a yes or a no
        -- nothing else was ever going to close it. On 2026-09-06 at 00:29
        the reply parsed to None, this function returned, and _turn_busy
        held the floor until the 60 s watchdog while five wake words at
        0.85-0.98 were refused (12:25 and 21:53 the same, via the speaker
        filter). The CARD stays up for a click; the TURN does not wait for
        one. `seq` is the asking turn's sequence number; left None it is
        read off _uncertain_turn, where _on_uncertain put it before this
        thread started (a newer prompt overwriting it in between has also
        superseded this rid, so the expiry finds nothing pending and
        leaves the newer turn alone).
        """
        if seq is None:
            with self._filler_lock:
                seq = getattr(self, "_uncertain_turn", None)
        why = "the follow-up window closed without an answer"
        try:
            # ONE CONSUMER ON THE MICROPHONE, asked BEFORE a device opens.
            # The arbiter cannot do this: it is a re-entrant DEPTH COUNTER
            # whose only job is pausing the hotword, so two consumers on two
            # threads both get their context manager and both proceed, and
            # record_fixed's own `self.recording` guard is a flag it never
            # sets. Found by the verdict that cleared the enrolment lane,
            # 2026-09-05: the new exclusion covered the two enrolment runs
            # and `runs_live` appeared nowhere in this file, so the other
            # two doors were never taught to ask. Pinned by
            # tests/test_mic_one_consumer_everywhere.py, whose census fails
            # on any new caller of record_fixed that does not ask.
            try:
                from jarvis.voicerun import runs_live
                live = runs_live(getattr(self, "services", None))
            except Exception:  # noqa: BLE001 - an unreadable seam is not a yes
                log.debug("mic: could not ask which enrolments are live",
                          exc_info=True)
                live = ("unknown",)
            if live:
                log.info("uncertain: not asking aloud -- a %s enrolment is "
                         "running and it owns the microphone", "/".join(live))
                why = "a %s enrolment owns the microphone" % "/".join(live)
                return
            if CONFIG.talkback:
                self.tts.speak("Was that for me?", block=True)
            if not MACHINE.has_mic:
                why = "no microphone to listen on"
                return
            if self.recorder.recording:
                why = "the microphone was already open"
                return
            with self._uncertain_lock:
                if rid not in self._pending_uncertain:
                    why = None                  # answered by a click meanwhile
                    return
            audio = self.recorder.record_fixed(self.UNCERTAIN_LISTEN_S)
            if audio is None or len(audio) == 0:
                why = "nothing was captured"
                return
            if CONFIG.speaker_verify and self.speaker.enrolled:
                filtered, _ = self.speaker.filter_segments(audio)
                if filtered is None:
                    log.info("uncertain reply ignored: not the enrolled speaker")
                    why = "the reply was not the enrolled speaker"
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
                why = ("the reply was not a yes or a no" if heard
                       else "nothing was heard")
                return
            why = None                          # uncertain_answer owns the turn
            self.uncertain_answer(rid, answer, source="voice")
        except Exception:
            log.exception("uncertain follow-up failed")
            why = "the follow-up failed"
        finally:
            if why is not None:
                self._uncertain_unanswered(rid, seq, why)

    def _uncertain_unanswered(self, rid: str, seq, why: str) -> bool:
        """The spoken window closed with no answer: END THE TURN, keep the card.

        Closes only the turn that asked -- the sequence number _on_uncertain
        handed the ask thread -- and only while the card is still his to
        click. A click in the meantime (uncertain_answer pops the id and
        opens its own turn) or a newer utterance (a new sequence number)
        has already taken the floor, and closing THAT turn would be the
        clobber _dispatch guards against for the socket sources. The check
        and the release happen under the lock _turn_start bumps the number
        under, so there is no window between them. Returns whether a turn
        was released. Safe on a stand-in that owns no turn state.
        """
        pending = getattr(self, "_pending_uncertain", None) or {}
        lock = getattr(self, "_uncertain_lock", None)
        if lock is None:
            still = rid in pending
        else:
            with lock:
                still = rid in pending
        if not still or getattr(self, "_turn_busy", None) is None:
            return False
        with self._filler_lock:
            if seq is not None and seq != self._turn_seq:
                log.info("uncertain question expired (%s); a newer turn holds "
                         "the floor, leaving it", why)
                return False
            log.info("uncertain question expired (%s): releasing the turn; "
                     "the card stays up for a click", why)
            self._turn_finished()
        try:
            bus.publish(Status(text="No answer caught — the card's still up",
                               kind="info"))
        except Exception:      # noqa: BLE001 - the pane is not the point here
            log.debug("could not publish the expiry status", exc_info=True)
        return True

    def _asking_uncertain(self) -> bool:
        """Is the turn holding the floor the one waiting on "Was that for
        me?" -- as opposed to a card left up after its window expired, or a
        reply still coming from the router. Safe on a bare stand-in."""
        with self._filler_lock:
            return getattr(self, "_uncertain_turn", None) == self._turn_seq

    def _uncertain_open(self) -> int:
        """Commander hook: how many "Was that for me?" cards are still up.

        One prompt at a time by construction (_on_uncertain supersedes the
        last), so this is 0 or 1 -- but it is counted rather than asserted,
        because the pane counts CARDS and the commander speaks a number.

        Easy to be non-zero with nobody at fault: _ask_uncertain returns
        without publishing UncertainResolved when the 5 s window hears
        nothing, when the transcript is refused, or when there is no mic,
        so the card sits there waiting for a click that may never come.
        """
        with self._uncertain_lock:
            return len(getattr(self, "_pending_uncertain", ()) or ())

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
            # The card is on HIS screen, so this turn is his -- said with
            # the argument rather than left to the ambient scope, which a
            # guest's voice turn may hold when he clicks (round-3: this was
            # a third path around the one scope read).
            result = self._emit_result(self.commander.resolve_uncertain(
                text, yes, addressee=self._the_turn_is_his()))
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
        # sensing first: the clock-driven privacy guard (jarvis/sensing.py)
        # has to be walking the devices before the legs that poll them run.
        for name, obj in (("sensing", getattr(self, "sensing", None)),
                          ("presence", self.presence), ("desk", self.desk),
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
        # One throwaway decode before the user's first word. The weights are
        # already resident (start_preload), but no whisper kernel has run,
        # and the first inference of a process costs 0.92 s against 0.27 s
        # for every one after it (measured 2026-09-02, one 3.3 s capture,
        # fresh process each way). Here rather than earlier so the mic path
        # -- the endpointer above, the speaker model before it -- is in
        # place first. It is NOT free: 0.84 s spent here to save 0.65 s on
        # his first turn, and the prewarm and gc.freeze below land ~0.85 s
        # later because of it. The hotword is already listening by now
        # (start_background runs before start_models), so warmup() gives way
        # to a decode already in flight, and a turn that lands INSIDE the
        # warm-up waits for the rest of it -- at most one ~0.85 s silent
        # decode, then decodes warm (its docstring has the real bound). The
        # one ordering the transcriber cannot see from inside is a capture
        # already open when this line is reached: its decode is seconds
        # away and would only queue behind the throwaway one, so the
        # warm-up is skipped and that turn pays the cold decode it would
        # have paid anyway, minus the wait (F49).
        rec = getattr(self, "recorder", None)
        busy = getattr(self, "_audio_busy", None)
        if getattr(rec, "recording", False) or (busy is not None
                                                and busy.is_set()):
            log.info("whisper warm-up skipped: a capture is in flight")
        else:
            try:
                self.transcriber.warmup()
            except Exception:
                log.exception("whisper warm-up failed")
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

    ENROLL_SPEAKER_GONE = (
        "That button is gone, sir. Enrol your voice on the Users tab.")

    def enroll_speaker(self):
        """REMOVED, and it refuses rather than simply going missing.

        WHAT IT USED TO DO, and it was one click deep in the settings drawer
        next to a slider: record a fixed fifteen seconds of whatever the room
        contained and hand it to ``speaker.enroll_from_audio``, which APPENDS
        one embedding to ``voiceprint.npz``. No loudness floor. No cohesion
        check. No separation check. No consent, no gate, no owner check, no
        label -- and no undo, because that file is written tmp->replace with
        no generations and nothing to roll back to. There was no way to tell
        whose voice it took: the microphone arbiter is a re-entrant depth
        counter rather than a mutex, TTS holds it for talkback, and there is
        no AEC on this box, so pressing it while Jarvis was mid-sentence
        appended JARVIS'S OWN VOICE to the file his identity rests on.

        IT IS A REFUSAL RATHER THAN A DELETION because a method that simply
        vanished would take an AttributeError somewhere at the worst moment,
        and because a stale wiring elsewhere should be told what to use, not
        silently do nothing.

        The replacement is ``voice_enrol_start`` -- jarvis/voicerun.py: the
        microphone taken once for the whole run, a loudness floor per take,
        the pool judged by the SHARED bars before a single write, a stored
        consent attestation, and a generation that can be rolled back.
        """
        log.warning("enroll_speaker: refused -- the unjudged 15 s append was "
                    "removed; use the Users tab (voice_enrol_start)")
        bus.publish(Status(text=self.ENROLL_SPEAKER_GONE, kind="warn"))
        return False, self.ENROLL_SPEAKER_GONE

    def train_wakeword(self):
        # ONE CONSUMER ON THE MICROPHONE, asked BEFORE a device opens. The
        # arbiter cannot do this: it is a re-entrant DEPTH COUNTER whose only
        # job is pausing the hotword, so two consumers on two threads both
        # get their context manager and both proceed, and record_fixed's own
        # `self.recording` guard is a flag it never sets. Found by the
        # verdict that cleared the enrolment lane, 2026-09-05: the new
        # exclusion covered the two enrolment runs and `runs_live` appeared
        # nowhere in this file, so the other two doors were never taught to
        # ask. Pinned by tests/test_mic_one_consumer_everywhere.py, whose
        # census fails on any new caller of record_fixed that does not ask.
        try:
            from jarvis.voicerun import runs_live
            live = runs_live(getattr(self, "services", None))
        except Exception:      # noqa: BLE001 - an unreadable seam is not a yes
            log.debug("mic: could not ask which enrolments are live",
                      exc_info=True)
            live = ("unknown",)
        if live:
            bus.publish(Status(
                text="Finish the %s enrolment first" % "/".join(live),
                kind="error"))
            return
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
                          # Stops the enforcement thread only: quitting is
                          # not consent, so the switch itself is left where
                          # the state file has it.
                          ("sensing", getattr(self, "sensing", None)),
                          ("desk", getattr(self, "desk", None)),
                          ("quiet", getattr(self, "quiet", None)),
                          # roomtone first of the pair: its stop() takes the
                          # paplay stream down, and a bed left playing over a
                          # stopped app is the one failure you can hear.
                          ("roomtone", getattr(self, "roomtone", None)),
                          ("arc", getattr(self, "arc", None)),
                          # mixer stop() restores every stream it ducked
                          ("mixer", getattr(self, "mixer", None)),
                          # ...and after it the Spotify tool's playback
                          # poller, which the mixer's remote duck may have
                          # started, is stopped before it polls a quit app.
                          # Resolved as close(): the tool has no stop(), and
                          # must not grow one -- a transport stop() here
                          # would pause his music every time Jarvis quit.
                          ("spotify", getattr(getattr(self, "services", None),
                                              "spotify", None)),
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

    # ------------------------------------------------------- restart button
    def code_status(self) -> dict:
        """running / disk / behind / dirty for the drawer's line (spec in
        relaunch.format_code_status). Four short git calls in the repo."""
        return probe_code_status(getattr(self, "running_commit", ""),
                                 relaunch.REPO_ROOT)

    def restart(self, spawn=None, sleep=time.sleep, clock=time.monotonic):
        """The Restart button's second press (Hunter, 2026-09-04: "yes,
        button only"). Say the line, let it finish (_tts_busy: the filler's
        strictly-typed TTS.is_speaking / TTS.pending read, bounded at
        RESTART_SAY_WAIT_S -- a ceiling, not a fixed wait, so a gated or
        already-finished line costs nothing), start the detached helper
        with OUR pid to wait on, THEN the normal quit. In that order: a
        quit first would leave nobody to relaunch. Returns the helper's
        pid, or None when it could not be started -- in which case nothing
        is quit, because a Jarvis that is down is worse than one that is
        stale, and the Status says so.

        Runs on the drawer's worker thread; the window close is marshalled
        onto the Tk thread by the hook attach_window installs."""
        if getattr(self, "_restarting", False):
            log.info("restart already in flight")
            return None
        self._restarting = True
        spawn = spawn or relaunch.spawn_relauncher
        try:
            self._say(RESTART_LINE)
        except Exception:                 # noqa: BLE001 - the line is a courtesy
            # Review 2026-09-04: a TTS error here left _restarting set for
            # good and the button dead. He pressed twice; restart anyway.
            log.exception("restart line could not be spoken; restarting anyway")
        deadline = clock() + RESTART_SAY_WAIT_S
        while self._tts_busy() and clock() < deadline:
            sleep(0.1)
        cmd, cwd, env = relaunch.plan()
        log_path = PATHS.LOG_DIR / "relaunch.log"
        try:
            helper = spawn(os.getpid(), cmd, cwd, env, log_path=log_path)
        except Exception as exc:          # noqa: BLE001 - reported, nothing quit
            log.exception("relaunch helper did not start")
            bus.publish(Status(text=f"Restart failed: {exc}"[:120],
                               kind="error"))
            self._restarting = False
            return None
        log.info("relaunch helper pid %s waiting on pid %s (log %s); "
                 "quitting", helper, os.getpid(), log_path)
        self._close_window()
        return helper

    def _close_window(self):
        """The ✕ path when a window is attached (MainWindow._on_close via
        after(0) -- geometry save, tray, services.quit, root.destroy), else
        the app's own quit()."""
        fn = getattr(self, "close_window", None)
        if callable(fn):
            fn()
        else:
            self.quit()

    # ---------------------------------------------- Knightfall, typed
    def knightfall_code(self, code, *, mail=None, smtp=None, now=None) -> str:
        """THE TYPED PATH (Hunter, 2026-09-04: "a and b"). The drawer's
        masked entry lands here, off the Tk thread. Returns the one line
        the drawer toasts; it never contains a code.

        The check is ``gate.check_override_code`` -- the same function the
        people CLI uses, on the gate's OWN code counter (separate from the
        phrase's, so burning one never closes the other). A refusal is its
        reason and nothing else. On success the window opens on the code
        leg exactly as the phrase opens it, and the code ROTATES: a fresh
        one is mailed to his own address from his own first account, and
        only a returned Message-ID lets the new hash be stored -- a mail
        that did not go leaves the old code standing, and the line says
        which happened. The plaintext is deleted the moment it is hashed;
        the log carries who and which path.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return "Knightfall: the gate is not built; see the log"
        with _KNIGHTFALL_LOCK:
            # THE FILE AS IT IS, not the boot copy (2026-09-06). This used
            # to check gate.registry while people_unlock two methods down
            # checked the file, so a code set at a terminal opened the
            # users tab and was refused here until a reload or a restart
            # -- and the weekly issuer writes the file from another
            # process. The boot copy is the fallback ONLY when the file is
            # unusable: a corrupted file must not lock the break-glass
            # that the boot copy would still open.
            registry = self._knightfall_check_registry()
            who, why, leg = gate_mod.check_override_code_leg(
                registry, code, attempts=gate.code_attempts)
            del code
            if not who:
                return "Knightfall: %s" % why
            if leg == gate_mod.LEG_PENDING:
                # TYPED PENDING = THE RECEIPT IN PERSON. He can only have
                # this week's code from Sunday's email, so typing it
                # proves the email arrived: promote it now, then rotate
                # exactly as any accepted code rotates.
                self._knightfall_promote_by_use(who, registry)
            gate.open_window(who, gate_mod.HOW_CODE, now=now)
            line, _mailed = self._knightfall_rotate(who, mail=mail,
                                                    smtp=smtp, accepted=True)
            return line

    def _knightfall_check_registry(self):
        """The registry a TYPED code is checked against: the people file
        re-read now, or -- only when that file is unusable -- the copy the
        gate loaded at boot. See knightfall_code."""
        gate = getattr(self, "gate", None)
        registry, _why = self._people_registry()
        if registry is not None and getattr(registry, "usable", False):
            return registry
        return gate.registry

    def _knightfall_promote_by_use(self, who, registry) -> None:
        """He typed this week's PENDING code: promote it under the file
        lock, tell the weekly lane, reload the live gate. Every failure is
        a log line -- both codes simply stay honoured until the pull."""
        gate = getattr(self, "gate", None)
        person = registry.person(who) if registry is not None else None
        code_id = getattr(person, "pending_code_id", "") if person else ""
        if not code_id:
            return
        path = getattr(getattr(gate, "registry", None), "path", None)
        ok, why = identity_mod.locked_update(
            path, lambda r: r.promote_pending(who, code_id))
        if not ok:
            log.error("knightfall: this week's code (id %s) was typed but could "
                      "not be promoted (%s); both codes stay honoured", code_id, why)
            return
        log.info("knightfall: %s typed this week's code; id %s promoted; the "
                 "previous code is retired", who, code_id)
        try:
            knightfall_weekly_mod.note_promoted_by_use(PATHS.KNIGHTFALL_WEEKLY, code_id)
        except Exception:                          # noqa: BLE001 - narration
            log.exception("knightfall: the weekly state could not be noted")
        try:
            gate.reload()
        except Exception:                          # noqa: BLE001 - a belt
            log.exception("knightfall: the gate could not be reloaded")

    def _knightfall_weekly_caption(self) -> str:
        """The weekly lane's one sentence, off its state file. A file read,
        never a socket; "" when the lane has never run."""
        try:
            return knightfall_weekly_mod.caption(PATHS.KNIGHTFALL_WEEKLY)
        except Exception:                          # noqa: BLE001 - a caption
            log.exception("knightfall: the weekly caption could not be read")
            return ""

    # ------------------------------------------- the people book (USERS tab)
    def people_snapshot(self) -> dict:
        """Everything the USERS tab draws, and NOTHING it must not hold.

        Rows come from ``Person.redacted()``, which replaces both salted
        hashes with a bare yes/no and is already documented as safe for a
        log, a report or the pane. A hash never crosses out of the app, so
        it cannot reach a widget even by accident.

        ``admin`` is ``gate.admin_gate``'s answer -- the SAME decision
        ``scripts/jarvis_people.py`` asks, so the tab and the terminal tool
        cannot come to different conclusions about whether a code is owed.

        The gallery LABELS are listed too, so the tab can flag a face
        pointer that resolves to nothing (that is exactly how the face leg
        silently stops naming anyone). Labels are strings stored beside the
        embeddings; nothing here opens a device or reads a frame.

        THE ROWS AND THE DECISION COME OFF DISK, not from ``gate.registry``.
        That object is rebound only by ``reload()``, which nothing calls
        until a write from this tab succeeds -- so a code, a row or a role
        set at a terminal was invisible here until he happened to change
        something, and the tab's "read the file NOW on open" was not
        reading the file at all. ``gate_line`` is the exception and stays
        the GATE's own sentence, because that one describes what the
        running gate is doing rather than what the file says; when the two
        disagree, the disagreement is the thing worth seeing.

        NEVER RAISES. A page that cannot be painted must not be able to
        take the console down with it.
        """
        out = {"people": [], "gate_line": "", "fault_kind": "",
               "path": "", "gallery": [], "admin": gate_mod.ADMIN_REFUSE,
               "admin_line": ""}
        gate = getattr(self, "gate", None)
        if gate is None:
            out["admin_line"] = ("the owner gate is not built, so nobody "
                                 "can be changed from here")
            return out
        registry, why = self._people_registry()
        if registry is None:
            out["admin_line"] = why
            return out
        try:
            out["people"] = [p.redacted() for p in registry.people]
            out["fault_kind"] = str(getattr(registry, "fault_kind", "") or "")
            out["path"] = str(getattr(registry, "path", "") or "")
            out["admin"], out["admin_line"] = gate_mod.admin_gate(registry)
        except Exception:                          # noqa: BLE001 - a page
            log.exception("users: the people snapshot could not be built")
            return out
        try:
            out["gate_line"] = gate.startup_line(
                voice_ok=bool(CONFIG.speaker_verify),
                face_ok=False, face_why="not asked from this tab")
        except Exception:                          # noqa: BLE001 - a line
            log.exception("users: the gate line could not be built")
        try:
            out["gallery"] = list(gallery_labels())
        except Exception:                          # noqa: BLE001 - optional
            log.exception("users: the gallery labels could not be listed")
        try:
            # THE VOICE GALLERY'S LABELS TOO. Without them the tab cannot say
            # who has a voice pool, and a chip that ASSERTED nobody but the
            # owner could have one is exactly the stale sentence he caught in
            # the foot note, one chip over. Labels are strings stored beside
            # the embeddings; nothing here opens a microphone.
            out["voices"] = list(voice_labels())
        except Exception:                          # noqa: BLE001 - optional
            log.exception("users: the voice labels could not be listed")
        return out

    def _people_registry(self) -> tuple:
        """The people book AS IT IS NOW, re-read from disk. ``(reg, why)``.

        NEVER ``gate.registry``. That object is the copy this process
        loaded at boot and rebinds only when something calls ``reload``,
        and the whole defect this method exists to close is that the tab
        decided from a copy that old: he set an override code at a terminal
        and the in-process copy had never heard of it.

        The window is not closed by this, only shrunk: the terminal tool is
        a separate PROCESS and nothing here can lock against it. What it
        does guarantee is that the decision and the write are made from the
        same read, moments apart, instead of from a snapshot taken when a
        tab was opened.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return None, "the owner gate is not built; see the log"
        try:
            path = getattr(gate.registry, "path", None)
            return identity_mod.Registry.load(path), ""
        except Exception:                          # noqa: BLE001 - a boundary
            log.exception("users: the people file could not be re-read")
            return None, "the people file could not be read; see the log"

    def _people_unlock_left(self, now=None) -> float:
        """Seconds the typed code still buys. Zero is locked.

        0.0 IS A SENTINEL AND IS TESTED AS ONE, never subtracted from. This
        used to be a bare ``max(0.0, until - t)``, and with ``until`` left
        at 0.0 while the tab is locked, any NEGATIVE ``now`` made a dwell
        out of nothing: ``0.0 - (-1.0)`` is 1.0, so
        ``people_forget(..., now=-1.0)`` deleted a row from his people book
        with no override code ever presented, and the book has no history
        and no backup (verdict, 2026-09-05, measured).

        The verdict called this robustness rather than a boundary crossing,
        and it was right: ``now=`` is not on the Services dataclass, and no
        socket, phone or voice rung reaches it, so whoever can pass it
        already holds the app object. It is fixed anyway because a guard
        that decides whether his people book may be written should not be
        forgeable by the clock it is handed. A clock BEFORE the grant is
        also refused, so a negative ``now`` cannot stretch a dwell that is
        genuinely open either. Pinned by
        tests/test_people_dwell_forgery.py.
        """
        t = time.monotonic() if now is None else float(now)
        with _PEOPLE_LOCK:
            until = float(getattr(self, "_people_unlock_until", 0.0) or 0.0)
        if until <= 0.0 or t < 0.0:
            return 0.0
        return max(0.0, until - t)

    def _people_open_unlock(self, now=None) -> None:
        """Arm the dwell. ONLY ever called where a code was just proved --
        ``people_unlock`` on a match, and a successful write that had to
        present one. A bootstrap write must not call it: making the first
        owner proves nothing about a code, and a window left open there is
        exactly the hole a code set seconds later would fall into."""
        t = time.monotonic() if now is None else float(now)
        with _PEOPLE_LOCK:
            self._people_unlock_until = t + PEOPLE_UNLOCK_S

    def people_relock(self) -> None:
        """Shut it now: the Lock button, and leaving the tab.

        THE PAGE RELOCKING ITSELF IS NOT ENOUGH, because the page is not
        the guard. A page that dropped its own dwell and left this one
        standing would look locked and not be."""
        with _PEOPLE_LOCK:
            self._people_unlock_until = 0.0

    def people_admin_state(self, now=None) -> dict:
        """The lock decision, RE-READ FROM DISK, plus what the dwell has
        left. ``{"admin", "admin_line", "unlocked_s"}``. NEVER RAISES.

        This is what lets the tab notice a code set at a terminal while it
        is open. ``people_snapshot`` also carries the decision, but it
        rebuilds every row, the startup line and the gallery listing, which
        is too much to do on a one-second tick; this reads one small JSON
        file and answers.

        IT IS NOT THE GUARD EITHER. It exists so the page can DRAW the
        right thing; ``_people_write`` re-decides for itself at the write.
        """
        out = {"admin": gate_mod.ADMIN_REFUSE, "admin_line": "",
               "unlocked_s": 0.0}
        registry, why = self._people_registry()
        if registry is None:
            out["admin_line"] = why
            return out
        try:
            out["admin"], out["admin_line"] = gate_mod.admin_gate(registry)
        except Exception:                          # noqa: BLE001 - a page
            log.exception("users: the admin state could not be decided")
            return out
        out["unlocked_s"] = self._people_unlock_left(now=now)
        return out

    def people_unlock(self, code, *, now=None) -> tuple:
        """The typed override code, from the tab. ``(ok, line)``.

        AGAINST THE FILE AS IT IS NOW, not the gate's boot-time copy. That
        was a second face of the same staleness and it failed CLOSED in a
        way he could not get out of: set a code at a terminal, come back to
        the running app, type the right code, and the in-memory registry
        carried no ``code_hash`` at all -- so the answer was "no override
        code has been set" and the only cure was restarting Jarvis.

        THIS OPENS NO VOICE WINDOW, and the difference from
        ``knightfall_code`` two methods up is deliberate rather than an
        oversight: Knightfall's whole job is to let the microphone answer
        him for five minutes, and an ADMINISTRATIVE unlock that did the
        same would make Jarvis answer whoever is standing in the room. The
        terminal tool grants no such thing for `add` or `forget`, and
        neither does this. What it opens is a WRITE dwell, checked by
        ``_people_write`` and by nothing else.

        The counter is the gate's OWN ``code_attempts``. A fresh one here
        would silently double the budget from ten tries per five minutes to
        twenty, because the tab -- unlike the CLI -- is inside this process,
        and it is the CODE's counter rather than the spoken phrase's so
        burning one can never close the other.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return False, "the owner gate is not built; see the log"
        registry, why = self._people_registry()
        if registry is None:
            return False, why
        try:
            who, why, leg = gate_mod.check_override_code_leg(
                registry, code, attempts=gate.code_attempts)
        except Exception as exc:                   # noqa: BLE001 - never str
            # Never the exception's text and never a traceback: what that
            # call was handed is a code, and an exception is free to quote
            # its argument back into the log.
            log.error("users: the override check failed (%s)",
                      type(exc).__name__)
            return False, "that check failed; see the log"
        finally:
            del code
        if not who:
            return False, why
        if leg == gate_mod.LEG_PENDING:
            # The same receipt-in-person rule as the drawer: this week's
            # code, typed, is proof the email arrived. No rotate here --
            # an administrative unlock mails nothing.
            self._knightfall_promote_by_use(who, registry)
        self._people_open_unlock(now=now)
        return True, "Unlocked, sir."

    def _people_write(self, what: str, change, *, now=None) -> tuple:
        """Re-read, DECIDE, mutate, save, reload the gate. ``(ok, line)``.

        THIS IS WHERE THE AUTHORITY TO CHANGE THE REGISTRY IS DECIDED, and
        it is decided from the registry as it is at the moment of the
        write. It used to be decided by the users page, from a snapshot
        read when the tab was OPENED, and this seam re-read the file only
        to refuse the CORRUPT case -- so the code case was enforced in
        exactly one place, from possibly-stale data. The sequence the page
        itself teaches walked straight through it: make the first owner in
        the tab (no code, correctly open), go and set a code at a terminal,
        come back to the same open tab, and forget somebody with nothing
        asked for.

        THE RULE IS NOT INVENTED HERE. ``gate.admin_gate`` is the same
        three-and-a-half-case decision ``scripts/jarvis_people.py::
        _authorise`` asks, and a test greps both:

        * no registry or no owner -- anybody at this keyboard may make the
          FIRST owner, because otherwise a fresh install is a brick;
        * an owner with NO code -- allowed, and the page says so out loud;
        * an owner WITH a code -- ``gate.check_override_code`` must have
          admitted one, within ``PEOPLE_UNLOCK_S``;
        * the file is there and BROKEN -- refused outright, and that
          outranks everything above. ``Registry.load`` answers with an
          EMPTY people list for a file that failed to parse, so an
          add-then-save would write a one-row registry over whatever it
          held. There is no history and no backup.

        RE-READ FIRST, EVERY TIME. The terminal tool is a separate process
        and nothing in this one can lock against it, so a write that used a
        registry read minutes ago would silently drop whatever was typed at
        a terminal in between. Re-reading immediately before the mutation
        shrinks that window; it does not close it, and the tab says so.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return False, "the owner gate is not built; see the log"
        with _PEOPLE_LOCK:
            # ...and the cross-process one (identity.registry_lock, an
            # flock), since 2026-09-06 there is a third writer under a
            # timer. The in-process RLock still orders the drawer's
            # threads; the flock orders the processes.
            path = getattr(getattr(gate, "registry", None), "path", None)
            try:
                lock = (identity_mod.registry_lock(path) if path
                        else contextlib.nullcontext())
                with lock:
                    return self._people_write_now(what, change, gate, now)
            except identity_mod.RegistryBusy as exc:
                log.error("users: %s refused: %s", what, exc)
                return False, ("the people book is held by another writer; "
                               "try again in a moment")

    def _people_write_now(self, what, change, gate, now) -> tuple:
        """``_people_write`` with ``_PEOPLE_LOCK`` already held. Split out
        only so the lock is one line rather than a body-wide indent; the
        decision and the reasoning are in ``_people_write``'s docstring."""
        registry, why = self._people_registry()
        if registry is None:
            return False, why
        # THE DECISION IS ASKED ONCE, in one place, by all seven doors now
        # -- see _people_decide. It was written out here and nowhere else,
        # which was right while this was the only door.
        ok, why, state = self._people_decide(registry, what, now)
        if not ok:
            return False, why
        try:
            ok, line = change(registry)
        except Exception:                          # noqa: BLE001 - a boundary
            log.exception("users: the %s failed", what)
            return False, "that did not work, sir; see the log"
        if not ok:
            return False, line
        if not registry.save():
            return False, "the people file could not be written; see the log"
        if state == gate_mod.ADMIN_CODE:
            # A run of edits is one code rather than five -- but ONLY on
            # the leg where a code was actually presented. Re-arming after
            # a bootstrap write would hand a free window to whatever the
            # registry became a moment later.
            self._people_open_unlock(now=now)
        try:
            # LIVE, with no restart. reload() rebinds gate.registry -- one
            # attribute swap, which the audio path then reads. The sensors
            # page says "restart to apply"; this one does not have to.
            gate.reload()
        except Exception:                          # noqa: BLE001 - a boundary
            log.exception("users: the gate could not be reloaded")
            return True, line + " (restart Jarvis for it to take effect)"
        log.info("users: %s -- %s", what, line)
        return True, line

    def people_add(self, *, label, name="", role=identity_mod.ROLE_KNOWN,
                   face="", face_dim=0, voice=False, consent="",
                   confirm_existing_owner=None, now=None) -> tuple:
        """Enrol somebody from the tab. ``(ok, line)``.

        A NON-OWNER WITHOUT A CONSENT RECORD IS REFUSED HERE, not merely
        discouraged in the UI. The record is the only durable evidence that
        the agreement happened at all, and the tab must not become the way
        around the rule the terminal tool enforces.
        """
        who = str(label or "").strip().lower()
        role = (identity_mod.ROLE_OWNER
                if role == identity_mod.ROLE_OWNER else identity_mod.ROLE_KNOWN)
        if role != identity_mod.ROLE_OWNER and not str(consent or "").strip():
            return False, ("adding %s takes their consent, and no consent "
                           "was recorded" % (who or "somebody"))

        def change(registry):
            person = identity_mod.Person(
                label=who, name=str(name or "").strip(), role=role,
                voice=bool(voice), face=str(face or "").strip().lower(),
                face_dim=int(face_dim or 0),
                consent=str(consent or "").strip())
            ok, why = registry.add_person(
                person, confirm_existing_owner=confirm_existing_owner)
            if not ok:
                return False, why
            return True, "%s is enrolled as %s (consent: %s)" % (
                who, role, person.consent or "-")

        return self._people_write("add", change, now=now)

    def people_set_role(self, label, role, *,
                        confirm_existing_owner=None, now=None) -> tuple:
        who = str(label or "").strip().lower()

        def change(registry):
            ok, why = registry.set_role(
                who, role, confirm_existing_owner=confirm_existing_owner)
            return (True, "%s is now %s" % (who, role)) if ok else (False, why)

        return self._people_write("set-role", change, now=now)

    def people_forget(self, label, *, now=None) -> tuple:
        """Remove a row, and SAY WHAT SURVIVED IT -- from the galleries
        themselves, not from a guess.

        MEASURED in ``identity.Registry.forget``: it removes the row and
        nothing else. The face gallery entry is untouched, so a line that
        said only "forgotten" would leave him believing a gallery was
        scrubbed when it was not.

        IT NAMED ONLY THE FACE HALF, and after the voice gallery merged that
        was the same half-truth one store over: a pool under that label could
        outlive the row with nothing in this line to say so. It is asked of
        both galleries now, AFTER the write, and it names the store rather
        than "that command" -- because by the time he reads this the row and
        its buttons are gone, and the terminal is what is left.
        """
        who = str(label or "").strip().lower()

        def change(registry):
            ok, why = registry.forget(who)
            if not ok:
                return False, why
            return True, "%s is forgotten here.%s" % (who,
                                                      _survivors_line(who))

        return self._people_write("forget", change, now=now)

    # ------------------------------------------- the four things the tab may do
    def _people_decide(self, registry, what, now) -> tuple:
        """``(ok, line, state)`` for a registry we have already re-read.

        THE ONE DECISION. It used to live inline in ``_people_write_now`` and
        it is now asked by six more doors -- setting a passphrase, asking for
        a new code, starting a face run, starting a voice run and the two
        gallery purges -- so it is a function rather than a paragraph copied
        seven times. A copy is how one of those doors comes to ask a slightly
        different question, and the door that drifted would be the one nobody
        was watching.
        """
        state, why = gate_mod.admin_gate(registry)
        if state == gate_mod.ADMIN_REFUSE:
            return False, why, state
        if (state == gate_mod.ADMIN_CODE
                and self._people_unlock_left(now=now) <= 0.0):
            # The name of the action and nothing else. What was typed is not
            # here to be logged, and must never become loggable.
            log.info("users: %s refused -- the override code is owed", what)
            return False, gate_mod.ADMIN_CODE_OWED, state
        return True, "", state

    def _people_gate(self, what: str, *, now=None) -> tuple:
        """``(ok, line)`` -- may this keyboard change anything right now?

        RE-READ FROM DISK, every time, for the same reason ``_people_write``
        re-reads: the terminal tool is a separate process and a decision made
        from a memory of the file is a decision about a file that may no
        longer exist in that shape.

        This is what the seams that are NOT registry writes ask -- starting a
        camera, starting a microphone, destroying a gallery. Each of those is
        a write to something at least as irreversible as the people book, so
        none of them gets an easier question than ``forget`` does.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return False, "the owner gate is not built; see the log"
        registry, why = self._people_registry()
        if registry is None:
            return False, why
        ok, line, _state = self._people_decide(registry, what, now)
        return ok, line

    def people_set_phrase(self, label, hashed, *, now=None) -> tuple:
        """Store an ALREADY-HASHED spoken passphrase. ``(ok, line)``.

        THE PLAINTEXT NEVER ARRIVES HERE. Hashing happens in the caller's own
        frame -- ``ui.users_page.UsersSecretControl`` reads two masked boxes,
        empties them, compares, hashes and deletes -- and this seam, like
        ``identity.Registry.set_secret`` below it, cannot leak a passphrase
        because it has never been handed one.

        WHY THIS ONE MAY BE TYPED ON SCREEN WHEN THE OVERRIDE CODE MAY NOT,
        and the distinction is the feature rather than an inconsistency: the
        spoken passphrase is SAID OUT LOUD in normal use, and
        ``scripts/jarvis_people.py`` says so in its own text -- "it can be
        overheard; that is accepted". A secret whose threat model already
        accepts being overheard is not made materially worse by a masked box
        on his own console. The code exists precisely because it never touches
        a microphone, so that argument does not transfer and there is no
        seam here that sets one.

        AN EMPTY HASH IS REFUSED. Clearing a phrase looks exactly like setting
        one from the outside and it takes away a way back in; removing it is
        the terminal's job, where the refusal can be spelled out.
        """
        who = str(label or "").strip().lower()
        hashed = str(hashed or "")
        if not hashed:
            return False, ("nothing was changed, sir: an empty passphrase "
                           "would take away a way back in rather than set "
                           "one")

        def change(registry):
            ok, why = registry.set_secret(who, "phrase_hash", hashed)
            if not ok:
                return False, why
            return True, ("Set, sir. It is stored salted-hashed; nothing "
                          "here can read it back.")

        return self._people_write("set-phrase", change, now=now)

    def people_new_code(self, *, mail=None, smtp=None, now=None) -> tuple:
        """"Send me a new code", from the tab. ``(ok, line)``.

        NO SECOND ROTATE IS WRITTEN. ``_knightfall_rotate`` already gets the
        order right in a way that cannot be reordered by accident -- generate,
        MAIL FIRST, store only on a returned Message-ID, re-read the registry
        immediately before writing, and put the old hash back in memory when
        the store fails -- and every one of its failure paths leaves a code
        that works. This adds the administrative gate in front of it and
        nothing else.

        ``ok`` IS READ OFF THE LINE THE ROTATE RETURNED, never off "the call
        did not raise": ``KNIGHTFALL_NEW_OK_LINE`` is the one answer that
        means a code both left this machine and was stored.

        AND IT CAN CHANGE WHAT THE MICROPHONE DOES. ``gate._mode_unsafe``
        downgrades enforce to SHADOW while no owner row carries a code,
        because a wrong verdict would otherwise have no way back in. Setting
        the FIRST code lifts that downgrade -- so on a box configured
        ``owner.mode=enforce`` but running in shadow for want of a code, this
        press starts refusing turns. It is stated on the panel BEFORE he
        presses (``ui.users_page.code_panel_lines``) and logged here when it
        happens; it is not closed by refusing, because a box with no code is
        the state that most needs one.
        """
        ok, line = self._people_gate("new-code", now=now)
        if not ok:
            return False, line
        before = self._gate_mode_quietly()
        line = self.knightfall_new_code(mail=mail, smtp=smtp, now=now)
        landed = (line == KNIGHTFALL_NEW_OK_LINE)
        after = self._gate_mode_quietly()
        if landed and before != after:
            log.warning("users: the new code lifted the enforce->shadow "
                        "downgrade; the owner gate is now %s (was %s)",
                        after, before)
            line = line + " The gate is now %s: it was running in %s because " \
                          "no code was set." % (after, before)
        return landed, line

    def _gate_mode_quietly(self) -> str:
        """The live gate mode, or "" -- never a raise. Read either side of a
        code being stored so the tab can SAY when the press changed it."""
        gate = getattr(self, "gate", None)
        if gate is None:
            return ""
        try:
            return str(gate.effective_mode())
        except Exception:                          # noqa: BLE001 - a read
            log.debug("users: the gate mode could not be read", exc_info=True)
            return ""

    # ------------------------------------------------------ enrolling, in app
    def face_enrol_start(self, *, now=None) -> tuple:
        """Start the in-app FACE enrolment from the tab. ``(ok, line)``.

        IT TAKES NO LABEL, and that is structural rather than an omission.
        ``enrolrun`` forces ``identity.owner_label`` by construction, for a
        reason a window cannot get round: enrolling somebody else stores a
        measurement of them, so they must read what is kept and type their own
        name, and a dialog is driven by whoever is already logged in. A seam
        that ACCEPTED a label would be the door round that, so it has not got
        one. A guest row hands over the command instead.

        THE BUTTON IS STRICTER THAN THE VOICE PATH TO THE SAME WRITE, and he
        is told rather than left to find it. Asking Jarvis out loud to enrol
        his face still commits on one typed word and never asks for the
        override code; this asks. That asymmetry is the right direction and it
        is written on the tab (``users_page.ENROL_ASYMMETRY``) rather than
        closed by loosening the button.
        """
        ok, line = self._people_gate("face-enrol", now=now)
        if not ok:
            return False, line
        services = getattr(self, "services", None)
        ok, reply, _run = _launch_face_run(
            cfg=getattr(self, "assistant", None), services=services,
            sensing=getattr(self, "sensing", None), say=self._enrol_say)
        return bool(ok), reply

    def face_enrol_stop(self) -> tuple:
        """Stop a running face enrolment. ``(ok, line)``.

        NEVER GATED. Stopping is the safe direction and takes the widest door
        -- the rule ``Commander._enrol_control`` already follows for the
        spoken stop. A code owed must never be the reason a lens stays open.
        """
        return self._stop_run("enrol_run", "the camera is off")

    def voice_enrol_start(self, *, now=None) -> tuple:
        """Start the in-app VOICE enrolment from the tab. ``(ok, line)``.

        Same shape and the same reasoning as the face seam, plus the one the
        camera does not have: the microphone is taken for the whole run, so
        the wake word is deaf and his abort has to be the Stop button. The run
        says that out loud before the first take.
        """
        ok, line = self._people_gate("voice-enrol", now=now)
        if not ok:
            return False, line
        return _launch_voice_run(self, now=now)

    def voice_enrol_stop(self) -> tuple:
        return self._stop_run("voice_run", "the microphone is back")

    def _stop_run(self, slot: str, tail: str) -> tuple:
        services = getattr(self, "services", None)
        run = getattr(services, slot, None)
        if run is None:
            return False, "Nothing is running, sir."
        try:
            run.abort()
        except Exception:                          # noqa: BLE001 - a run
            log.exception("users: %s could not be stopped", slot)
            return False, "That would not stop, sir; see the log."
        return True, "Stopped, sir. Nothing was written and %s." % tail

    def _enrol_say(self, text: str) -> None:
        """The run's spoken channel, through the app's own speaker."""
        try:
            self.speak(text)
        except Exception:                          # noqa: BLE001 - TTS
            log.debug("users: the enrolment line could not be spoken",
                      exc_info=True)

    # ------------------------------------------------- destroying a gallery
    def people_purge_face(self, label, *, now=None) -> tuple:
        """Destroy one person's FACE measurements, everywhere. ``(ok, line)``.

        ``ok`` IS ``complete`` AND NOTHING ELSE. ``facegallery.purge_label``
        destroys no file until the embeddings it held, MINUS theirs, have been
        read back off disk from the new generation by name and by sample
        count -- and it reports, in numbers, every generation it could not
        read, that another model wrote, or that holds a bystander it could not
        carry. A surface that rounded an incomplete purge up to "done" would
        be exactly the class of defect the foot note was just corrected for,
        so the line is built from those numbers and the answer is the
        invariant's own verdict.
        """
        return self._purge("face", label, now=now)

    def people_purge_voice(self, label, *, now=None) -> tuple:
        """The same, for the voice gallery. A face purge that left a voice
        pool behind would be a half-truth of the same shape."""
        return self._purge("voice", label, now=now)

    def _purge(self, kind: str, label, *, now=None) -> tuple:
        who = str(label or "").strip().lower()
        ok, line = self._people_gate("purge-%s" % kind, now=now)
        if not ok:
            return False, line
        if not who:
            return False, "nobody was named, sir, so nothing was touched"
        opener = _open_face_gallery if kind == "face" else _open_voice_gallery
        try:
            gallery = opener()
        except Exception:                          # noqa: BLE001 - a store
            log.exception("users: the %s gallery could not be opened", kind)
            return False, ("the %s gallery could not be opened, sir; nothing "
                           "was touched" % kind)
        if gallery is None:
            return False, ("there is no %s gallery on this box, sir, so there "
                           "is nothing to remove" % kind)
        try:
            report = gallery.purge_label(who, reason="users tab")
        except Exception:                          # noqa: BLE001 - a store
            log.exception("users: the %s purge failed", kind)
            return False, ("that purge did not run, sir; nothing was "
                           "destroyed. See the log.")
        line = enrolentry.purge_line(report, kind=kind)
        complete = bool((report or {}).get("complete"))
        log.info("users: %s purge for %s -- complete=%s, removed=%s",
                 kind, who, complete, (report or {}).get("removed"))
        # THE LIVE PREVIEW HOLDS A GALLERY IN MEMORY. enrolrun stops the
        # worker after a save for exactly this reason: a console that goes on
        # recognising somebody who was destroyed thirty seconds ago is a lie
        # with a face on it. Stop it, or say "restart to apply" -- never
        # neither.
        if kind == "face" and complete and not self._stop_preview_worker():
            line = line + " Restart Jarvis for the live preview to stop "\
                          "matching against the old generation."
        return complete, line

    def _stop_preview_worker(self) -> bool:
        """Stop the preview's capture so it drops the gallery it is holding.
        True if it was stopped; False means the caller must say "restart"."""
        services = getattr(self, "services", None)
        worker = getattr(services, "preview_worker", None)
        if worker is None:
            return False
        try:
            worker.stop()
        except Exception:                          # noqa: BLE001 - a worker
            log.exception("users: the preview worker would not stop")
            return False
        return True

    def knightfall_new_code(self, *, mail=None, smtp=None, now=None) -> str:
        """THE BOOTSTRAP: "Email me a new Knightfall code". The same
        generate -> mail -> store sequence, for the first owner, and it
        opens no window -- mailing a code is not typing one.

        KEYBOARD = OWNER IS THE EXISTING RULE, not a new one: whoever is
        at this keyboard can already edit or delete the registry, which is
        the argument ``check_override_code``'s docstring makes and this
        does not make twice. So there is no second check here -- only a
        cooldown of KNIGHTFALL_COOLDOWN_S, so a stuck button cannot spam
        his inbox.
        """
        gate = getattr(self, "gate", None)
        if gate is None:
            return "Knightfall: the gate is not built; see the log"
        t = time.monotonic() if now is None else float(now)
        with _KNIGHTFALL_LOCK:
            last = getattr(self, "_knightfall_new_ts", None)
            if last is not None and t - last < KNIGHTFALL_COOLDOWN_S:
                return KNIGHTFALL_COOLDOWN_LINE
            try:
                who = gate._owner_label()
            except Exception:                      # noqa: BLE001 - no registry
                who = ""
            if not who:
                log.warning(KNIGHTFALL_NO_OWNER_LOG)
                return KNIGHTFALL_NO_OWNER_LINE
            line, mailed = self._knightfall_rotate(who, mail=mail, smtp=smtp,
                                                   accepted=False)
            # THE CLOCK STARTS ON A CODE THAT ACTUALLY LEFT. It used to be
            # stamped before the send, so a press that mailed nothing --
            # no account configured, the transport refusing -- answered
            # "one code a minute, sir" to the next press and made him wait
            # for a minute to be told the same thing again (R8b).
            if mailed:
                self._knightfall_new_ts = t
            return line

    def knightfall_status(self) -> dict:
        """What the drawer's Knightfall caption needs, for
        ui.views.format_knightfall_status. Three keys:

        * ``to`` -- the MASKED destination the next code would go to, or
          "" when there is none. Masked because the drawer is on screen
          and a full address does not need to be.
        * ``problem`` -- a fixed sentence when his configured destination
          is not usable, "" otherwise.
        * ``setup`` -- where to configure a mailbox, when there is none.

        WHY IT EXISTS AT ALL. The caption under the button was the flat
        sentence "Using it emails you the next one" -- which was false in
        the state he was actually in (zero mail accounts configured,
        measured 2026-09-05): the button mails nothing, and the promise
        was made before he pressed. A read only: no socket, no code.
        """
        from jarvis.tools import mail as mail_mod
        out = {"to": "", "problem": "", "setup": "",
               "weekly": self._knightfall_weekly_caption()}
        try:
            accounts = mail_mod.mail_accounts(self.assistant)
        except Exception:                          # noqa: BLE001 - config
            log.exception("knightfall: the mail accounts could not be read")
            accounts = []
        if not accounts:
            try:
                out["setup"] = mail_mod.setup_line(self.assistant)
            except Exception:                      # noqa: BLE001 - config
                log.exception("knightfall: the setup line could not be read")
            return out
        try:
            out["to"] = mail_mod._mask_address(
                mail_mod.notice_destination(accounts[0]))
        except mail_mod.NoticeAddressInvalid:
            out["problem"] = KNIGHTFALL_BAD_DESTINATION
        except Exception:                          # noqa: BLE001 - config
            log.exception("knightfall: the notice destination is unreadable")
        return out

    def _knightfall_rotate(self, who, *, mail=None, smtp=None,
                           accepted=False):
        """Generate -> mail FIRST -> store ONLY on a Message-ID. Returns
        ``(line, mailed)``: the line to show, and whether a code actually
        left this machine (the bootstrap's cooldown starts on that, not on
        the press). ``mail`` is the mail module (a seam for the tests);
        ``smtp`` is the transport class the mail module takes.

        THE TRANSPORT IS REACHED THROUGH jarvis/outbox.py, like every other
        send in this package, and by its narrowest door: send_notice takes
        no recipient, so a Knightfall code can only ever go to his own
        account's own address.

        NOTHING THE TRANSPORT SAYS IS REPEATED. The reason in the line and
        in the log is the exception's TYPE. It used to be str(exc), which
        trusts a seam to be discreet about a body it was just handed: with
        a transport whose error quoted the first line of the mail, the
        freshly rotated code came back in the line the drawer toasts
        (verdict, 2026-09-05, measured).
        """
        head = "Knightfall accepted, sir; " if accepted else "Knightfall: "
        keep = head + "the code stays as it is (mail: %s)."
        # The REAL module either way: `mail` is the seam a test substitutes
        # for the transport, but the destination and the refusal that goes
        # with it are decided against the real one (see outbox.send_notice).
        from jarvis.tools import mail as mail_mod
        if mail is None:
            mail = mail_mod
        try:
            accounts = mail.mail_accounts(self.assistant)
        except Exception:                          # noqa: BLE001 - config
            log.exception("knightfall: the mail accounts could not be read")
            accounts = []
        if not accounts:
            return keep % "no mail account is configured", False
        account = accounts[0]
        new = pp.new_code()
        body = "%s\n%s\n" % (new, KNIGHTFALL_BODY_LINE)
        try:
            msgid = outbox.send_notice(account, KNIGHTFALL_SUBJECT, body,
                                       smtp=smtp, mail=mail)
        except mail_mod.NoticeAddressInvalid:
            # HIS CONFIG, not the transport: notice_destination refused a
            # destination that is not an address, before a socket was
            # opened. Nothing was sent and nothing is stored, so the old
            # code stands -- and he is told what to fix rather than a
            # class name, because these words are ours and never touched
            # a mail server.
            del new, body
            log.warning("knightfall: the configured notice address is not "
                        "an address; the old code stands")
            return keep % KNIGHTFALL_BAD_DESTINATION, False
        except Exception as exc:                   # noqa: BLE001 - transport
            # MailSendFailed, or anything else the transport did: the old
            # code stands, and only the TYPE of what went wrong is said.
            del new, body
            why = type(exc).__name__
            log.warning("knightfall: the new code could not be mailed (%s); "
                        "the old one stands", why)
            return keep % why, False
        del body
        if not msgid:
            del new
            return keep % "no Message-ID came back", True
        hashed = pp.hash_secret(new)
        del new
        # UNDER THE FILE LOCK (2026-09-06): the weekly issuer is a third
        # writer of people.json from another process, and the flock in
        # identity.registry_lock is what the re-read below serialises
        # against. A lock that cannot be taken is the store failing.
        path = getattr(getattr(self.gate, "registry", None), "path", None)
        try:
            lock = (identity_mod.registry_lock(path) if path
                    else contextlib.nullcontext())
            with lock:
                return self._knightfall_store(who, hashed, head, keep,
                                              accepted=accepted)
        except identity_mod.RegistryBusy:
            log.error("knightfall: the new code was mailed but the people book "
                      "is held by another writer; the old code stands")
            return head + ("the new code could not be stored, so the old one "
                           "stands; the one in your inbox will not work."), True

    def _knightfall_store(self, who, hashed, head, keep, *, accepted=False):
        """The store half of _knightfall_rotate, with the file lock held.

        ``accepted`` rides along because the success line differs: a
        code he TYPED answers "Knightfall accepted"; the "email me a
        new code" button answers with the plain new-code line.
        """
        # RE-READ BEFORE WRITING. self.gate.registry is the copy loaded at
        # BOOT, and the save below writes that whole object over the file --
        # so anything added to the people book since this process started
        # was silently overwritten by a memory that never knew about it.
        # MEASURED 2026-09-05: one press of "Email me a new Knightfall code"
        # DELETED a person enrolled at a terminal since boot, and REVERTED a
        # passphrase set at a terminal to empty. The book has no history and
        # no backup -- jarvis/ui/users_page.py says so on screen -- so the
        # row was simply gone. Same rule the Users tab now follows: a write
        # decides from the file as it is AT THE WRITE, never from a memory
        # of it. Pinned by tests/test_knightfall_stale_registry.py.
        try:
            self.gate.reload()
        except Exception:                          # noqa: BLE001 - a line, not a raise
            log.exception("knightfall: the people book could not be re-read; "
                          "not writing a stale copy over it")
            return keep % "the people book could not be re-read", True
        registry = getattr(self.gate, "registry", None)
        person = registry.person(who) if registry is not None else None
        old = person.code_hash if person is not None else ""
        # EVERY FAILURE HERE IS A LINE, NOT AN EXCEPTION. Registry.save()
        # catches OSError and nothing else, and this is documented to
        # return a line -- the drawer's thread has no other way to tell him
        # what happened, and an exception here would leave the mailed code
        # stored in memory and not on disk (R7).
        try:
            ok, why = (registry.set_secret(who, "code_hash", hashed)
                       if registry is not None else (False, "no registry"))
            stored = bool(ok and registry.save())
        except Exception as exc:                   # noqa: BLE001 - registry
            log.exception("knightfall: storing the new code failed")
            ok, why, stored = False, type(exc).__name__, False
        if stored:
            log.info("knightfall: %s rotated the code at the keyboard; the "
                     "new one is in the mail", who)
            return (KNIGHTFALL_OK_LINE if accepted
                    else KNIGHTFALL_NEW_OK_LINE), True
        # The mail went, the store did not. Put the old hash back so the
        # old code works in memory as it still does on disk: never a state
        # where no code works. The one in the inbox is dead, and he is told.
        if registry is not None:
            try:
                registry.set_secret(who, "code_hash", old)
            except Exception:                      # noqa: BLE001 - registry
                log.exception("knightfall: the old hash could not be put "
                              "back in memory; it is still on disk")
        log.error("knightfall: the new code was mailed but could not be "
                  "stored (%s); the old code stands",
                  why or "the registry could not be written")
        return head + ("the new code could not be stored, so the old one "
                       "stands; the one in your inbox will not work."), True

    # ------------------------------------------------------------ UI hooks
    def ui_service_kwargs(self) -> dict:
        """Everything the UI's Services dataclass may take (spec 9.10);
        main() keeps the fields the installed UI declares."""
        return dict(
            start_recording=self.recorder.start,
            stop_recording=lambda: threading.Thread(
                target=self.recorder.stop, daemon=True).start(),
            # The on-screen mic button turns into a STOP glyph while a
            # capture is open, and he read it as a cancel: "if i press
            # cancel button on the screen when he is waiting on a response
            # he will say i didnt quite get that sir" (2026-08-31). It was
            # wired to recorder.stop, so the half-second of room it had
            # went to Whisper and came back as salad -- live at 20:46:37
            # ("Stopped (manual)" on a 1.2 s clip -> -4.44 -> "Say that
            # again, sir?"), again at 20:46:56 and again at 21:26:06.
            # abort() discards the audio instead: _on_recording_stopped
            # and _turn_on_stop both return early on reason == "abort", so
            # nothing is transcribed, nothing is spoken and the turn is
            # simply abandoned. Cancelling is silent.
            cancel_recording=lambda: threading.Thread(
                target=self.recorder.abort, daemon=True).start(),
            # #135, typed-command history: TypedHistory.prev/next were
            # written FOR the command bar's Up/Down ("arrow keys do
            # nothing") and never wired to it -- the consumer side of the
            # feature was missing, not broken.
            history_prev=self.history.prev,
            history_next=self.history.next,
            dispatch_text=self.dispatch_text,
            toggle_hotword=self.toggle_hotword,
            quit=self.quit,
            # The drawer's Restart button and the code-status line above
            # it (jarvis/relaunch.py). build_ui_services drops both on a
            # UI that does not declare them.
            restart=self.restart,
            code_status=self.code_status,
            # Knightfall: the drawer's masked entry and its bootstrap
            # button. Both return the one line the drawer toasts, and
            # both run off the Tk thread (KnightfallControl).
            knightfall_code=self.knightfall_code,
            knightfall_new_code=self.knightfall_new_code,
            knightfall_status=self.knightfall_status,
            # The USERS tab (jarvis/ui/users_page.py). Seven narrow seams:
            # a redacted snapshot, the administrative unlock, the three
            # writes, and the two that stop the PAGE being the guard --
            # people_admin_state re-reads the file so a code set at a
            # terminal is noticed while the tab is open, people_relock
            # shuts the app's dwell when he leaves it. Each answers ONE
            # line to toast and never a hash. build_ui_services drops them
            # on a UI that does not declare them, so either merge order is
            # safe.
            people_snapshot=self.people_snapshot,
            people_unlock=self.people_unlock,
            people_add=self.people_add,
            people_set_role=self.people_set_role,
            people_forget=self.people_forget,
            people_admin_state=self.people_admin_state,
            people_relock=self.people_relock,
            # The eight the USERS tab's enrolment work adds.
            # people_set_phrase takes a HASH: the plaintext is compared,
            # hashed and deleted in the page's own frame, so it never crosses
            # into the app. There is deliberately NO people_set_code -- the
            # override code's value is that it is GENERATED rather than
            # chosen, and people_new_code() asks the existing mail-first
            # rotate for a fresh one instead.
            people_set_phrase=self.people_set_phrase,
            people_new_code=self.people_new_code,
            face_enrol_start=self.face_enrol_start,
            face_enrol_stop=self.face_enrol_stop,
            voice_enrol_start=self.voice_enrol_start,
            voice_enrol_stop=self.voice_enrol_stop,
            people_purge_face=self.people_purge_face,
            people_purge_voice=self.people_purge_voice,
            calibrate_noise=self.calibrate_noise,
            # enroll_speaker IS DELIBERATELY NOT WIRED any
            # more; see the method for why. The judged
            # replacement is voice_enrol_start, on the
            # USERS tab, under the override code.
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
            # Offline mode: the POLICY, not a callable. The header badge
            # re-reads it on the same 5 s pass as room_state (the curfew
            # edge arrives on the clock, with nothing published), and the
            # drawer's curfew pickers write through it.
            sensing=self.sensing,
            # The Board's own WM close button: without this the window
            # manager's X tore down the toplevel while the app still
            # believed the Board was up, so its feed kept polling. The UI
            # Services dataclass declares the field; build_ui_services drops
            # what it does not, so passing it is safe in either merge order.
            board_closed=self._board_hide,
            # Grab and throw: the courier lends the console its hand stage
            # (rides the preview's capture) and takes the carry chip back.
            gesture=getattr(self, "gesture", None),
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


def voice_labels() -> tuple:
    """Every label the VOICE gallery holds, or () -- never a raise.

    Strings beside the embeddings. Nothing here opens a microphone, reads a
    recording or loads a model, and there is no recording to open: the audio
    became 192 numbers at the microphone and was thrown away.
    """
    try:
        from jarvis import voicegallery as vg
        gallery = vg.VoiceGallery()
        gallery.load()
        return tuple(gallery.labels())
    except Exception:                              # noqa: BLE001 - absent
        log.debug("users: the voice gallery could not be listed",
                  exc_info=True)
        return ()


def _survivors_line(who: str) -> str:
    """"" when nothing is left, else what is STILL on the disk under ``who``,
    named store by store.

    NEVER RAISES and never claims more than it read: a gallery that could not
    be listed is reported as unknown rather than quietly dropped, because
    "forgotten" with a silent omission is the exact shape of the sentence
    this lane exists to stop.
    """
    kept, unknown = [], []
    for kind, lister in (("face measurements", gallery_labels),
                         ("voice pool", voice_labels)):
        try:
            if who in {str(x) for x in lister()}:
                kept.append(kind)
        except Exception:                              # noqa: BLE001 - a store
            log.exception("users: the %s could not be listed after a forget",
                          kind)
            unknown.append(kind)
    out = ""
    if kept:
        out += (" Their %s stay on the disk; the row and its buttons are gone,"
                " so removing them now needs a terminal." % " and their ".join(kept))
    if unknown:
        out += (" I could not read the %s, so I can't say whether anything of"
                " theirs is left there." % " or the ".join(unknown))
    if not kept and not unknown:
        out += " Nothing of theirs is left in either gallery."
    return out


def gallery_labels() -> tuple:
    """The face gallery's LABELS, with no lens involved at all.

    A gallery generation is 128 floats and a string per take; this reads
    the strings so the USERS tab can say whether a row's face pointer
    resolves to anything. It imports no camera module, opens no device and
    touches no frame -- the same split jarvis/enrolentry.py already makes.
    Any failure is an empty tuple: a chip that cannot be drawn is not worth
    a traceback out of a repaint.
    """
    try:
        from jarvis.facegallery import default_gallery
        gallery = default_gallery()
        gallery.load()
        return tuple(gallery.labels())
    except Exception:                              # noqa: BLE001 - optional
        log.debug("users: the face gallery could not be listed",
                  exc_info=True)
        return ()



# ---------------------------------------------------- the in-app enrolments
# MODULE-LEVEL AND THIN ON PURPOSE. These are the places the suite substitutes
# a stand-in so a seam can be exercised with no camera, no microphone, no
# model and no gallery on disk. That is not a convenience: the whole feature
# is verified from NUMBERS -- counts, RMS floors, cosines, generations -- and
# never by looking at a frame or listening to a take, and a method reaching
# straight into jarvis.enrolrun would leave no seam to do that through.
def _launch_face_run(*, cfg, services, sensing, say):
    """One in-app face enrolment. ``(ok, reply, run)``.

    The SEQUENCE lives in ``enrolrun.launch`` so the tab's button and the
    spoken offer cannot drift apart; this only hands it the two test seams.
    """
    from jarvis import enrolrun as er
    return er.launch(cfg=cfg, services=services, sensing=sensing, say=say,
                     preflight_fn=_face_preflight, build_fn=_build_face_run)


def _face_preflight(cfg, *, sensing=None, worker=None, services=None):
    from jarvis import enrolrun as er
    return er.preflight(cfg, sensing=sensing, worker=worker, services=services)


def _build_face_run(**kw):
    from jarvis import enrolrun as er
    return er.EnrolRun(**kw)


def _open_face_gallery():
    """The face gallery at its real root, or None on a box without one."""
    try:
        from jarvis import facegallery as fg
        return fg.FaceGallery()
    except Exception:                              # noqa: BLE001 - absent
        log.exception("users: the face gallery could not be opened")
        return None


def _open_voice_gallery():
    try:
        from jarvis import voicegallery as vg
        return vg.VoiceGallery()
    except Exception:                              # noqa: BLE001 - absent
        log.exception("users: the voice gallery could not be opened")
        return None


def _voiceprint_vectors_quietly():
    """His stored voiceprint, as the anchor the shared bars measure against.
    ``[]`` means "no anchor" and every check that uses it stands down."""
    try:
        from jarvis import voiceenrol as ve
        return ve.voiceprint_vectors(PATHS.VOICEPRINT)
    except Exception:                              # noqa: BLE001 - absent
        log.debug("users: the voiceprint could not be read as an anchor",
                  exc_info=True)
        return []


def _speech_seconds(audio) -> float:
    """Seconds of SPEECH in a take, silence trimmed -- the same measure the
    terminal script stores, so the two enrolments' numbers are comparable."""
    from jarvis.speaker import SAMPLE_RATE, trim_silence
    return float(len(trim_silence(audio))) / float(SAMPLE_RATE)


def _launch_voice_run(app, *, now=None) -> tuple:
    """Preflight, build, park and start ONE in-app voice enrolment.

    THE OWNER ENROLLING HIMSELF IS THE ONLY CASE THIS DOOR OPENS, and the
    attestation stored with the pool says exactly that -- ``consent.HOW_OWNER``
    and never "typed", because a console run claiming a terminal ceremony
    would be a false record on disk and worse than no record at all.
    """
    from jarvis import consent as cs
    from jarvis import earcons as earcons_mod
    from jarvis import voicerun as vr
    services = getattr(app, "services", None)
    cfg = getattr(app, "assistant", None)
    label = identity_mod.owner_label(cfg)
    if not label:
        return False, ("I don't know whose voice that would be, sir -- no "
                       "owner is set.")
    recorder = getattr(app, "recorder", None)
    tts = getattr(app, "tts", None)
    pre = vr.preflight(recorder=recorder, tts=tts, services=services,
                       label=label, owner=label, consent_how=cs.HOW_OWNER)
    if not pre["ok"]:
        return False, str(pre["reply"])
    gallery = _open_voice_gallery()
    if gallery is None:
        return False, ("there is no voice gallery on this box, sir, so there "
                       "is nowhere to put it")
    try:
        gallery.load()
    except Exception:                              # noqa: BLE001 - a store
        log.exception("users: the voice gallery could not be read")
        return False, ("the voice gallery could not be read, sir; nothing "
                       "was recorded")
    speaker = getattr(app, "speaker", None)
    embed = getattr(speaker, "_extract_embedding", None)
    if not callable(embed):
        return False, ("the speaker model isn't loaded, sir, so there is "
                       "nothing to turn a take into numbers")
    run = vr.VoiceRun(
        label=label, gallery=gallery, recorder=recorder, embed=embed,
        tts=tts, services=services, owner=label, consent_how=cs.HOW_OWNER,
        say=app._enrol_say,
        # DISPLAY-ONLY, the same mechanism enrolrun uses: a JarvisReply with
        # speak=False does not go through context.add_exchange, so the card
        # of numbers never enters the plaintext journal.
        card=lambda t: bus.publish(JarvisReply(text=t, speak=False)),
        earcon=lambda name: earcons_mod.play(name, cooldown_s=0.0),
        owner_vectors=_voiceprint_vectors_quietly(),
        speech_seconds=_speech_seconds)
    try:
        services.voice_run = run
    except Exception:                              # noqa: BLE001 - slim
        log.debug("users: could not park the voice run", exc_info=True)
        return False, "that could not start, sir; see the log"
    if not run.start():
        try:
            if getattr(services, "voice_run", None) is run:
                services.voice_run = None
        except Exception:                          # noqa: BLE001 - slim
            log.debug("users: could not unpark the voice run", exc_info=True)
        return False, "that could not start, sir; see the log"
    return True, vr.V1


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


RESTART_LINE = "Back in a moment, sir."
RESTART_SAY_WAIT_S = 6.0      # ceiling on waiting for that line to finish


def running_commit(repo=None, git=None) -> str:
    """Short hash of HEAD in the checkout this process runs from, "" when
    there is no git (a tarball, a box without git): the code-status line
    then reads "Running unknown"."""
    git = git or _git
    return git(str(repo or relaunch.REPO_ROOT), "rev-parse", "--short", "HEAD") or ""


def probe_code_status(running: str, repo, git=None) -> dict:
    """running / disk / behind / dirty, through context._git (stdout or ""
    on any failure, so a missing git degrades to unknowns, never a raise).
    ``behind`` is `git rev-list --count <running>..HEAD` -- but only when
    `HEAD..<running>` is empty, i.e. the running commit is an ANCESTOR of
    HEAD; a rebased-away or unknown hash answers None, because a count
    against a commit HEAD does not contain would be a made-up number."""
    git = git or _git
    repo = str(repo)
    running = running or ""
    disk = git(repo, "rev-parse", "--short", "HEAD") or ""
    behind = None
    if running and disk:
        if running == disk:
            behind = 0
        else:
            own = git(repo, "rev-list", "--count", f"HEAD..{running}")
            count = git(repo, "rev-list", "--count", f"{running}..HEAD")
            if own == "0" and count.isdigit():
                behind = int(count)
    dirty = bool(git(repo, "diff", "HEAD", "--shortstat")) if disk else False
    return {"running": running, "disk": disk, "behind": behind, "dirty": dirty}


def attach_window(app, window) -> None:
    """Give the app the ✕ path: restart() ends by calling this hook from
    a worker thread, and MainWindow._on_close must run on the Tk thread."""
    app.close_window = lambda: window.root.after(0, window._on_close)


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
    attach_window(app, window)        # the Restart button's quit is the ✕'s
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
