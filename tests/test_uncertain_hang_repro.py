"""THE REPRODUCER for the 2026-09-06 00:29 "soft lock".

His words: "jarvis is soft locked ... stuck from saying is that for me and
being in standby mode and wont come out". The immediate net (3865379,
eeb1f34, a4ecf8b) bounds ``_audio_busy`` with a 90 s watchdog on the
theory that the decode thread never returned. This file asks the question
that theory rests on, with a hard bound: DOES the decode thread return,
on exactly the path the log shows, and if not, WHERE is it standing?

Everything here is synthetic. No microphone is opened, no recording is
played, no transcript is real: every clip is ``np.zeros`` and every
transcript is a string written below. The real ``JarvisApp`` is built the
way tests/test_app_wiring.py builds it (real config, tools, brain seams,
commander, gate) with the hardware seams -- recorder, transcriber, speaker,
TTS -- replaced by scripted stand-ins, and the bus with NO Tk attached.

Three variants, straight from the log:

  * ``bool``    the follow-up parses to a yes/no  (14:56, 14:57 -- "fine")
  * ``ignored`` the follow-up is not the enrolled speaker (12:25, 21:53)
  * ``none``    the follow-up parses to None        (00:29 -- THE WEDGE)

Each is run on the direct decode path and on the SPECULATIVE path the 00:29
log shows ("speculative transcript reused"), with an owner enrolled and a
passphrase SET so the quiet-decode / redaction branch is the one taken.

Then the collaborator shapes: a ``speak()`` that blocks, a ``record_fixed()``
that blocks, a bus with a Tk root attached (publish only queues), and a
``publish()`` that blocks on the decode thread's LAST publish. Each is
bounded: the block is an Event the test releases in ``finally``.

The measurement is one number per run: seconds until ``_audio_busy`` is
clear AND the decode thread has exited, with a 10 s ceiling. When it does
not clear, the thread's stack is dumped with ``sys._current_frames`` --
the same instrument a4ecf8b put in the watchdog -- and printed verbatim.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis import identity as identity_mod
from jarvis import passphrase as pp
from jarvis.commander import IntentClassifier
from jarvis.config import CONFIG
from jarvis.events import RecordingStopped, Status, bus
from jarvis.transcriber import TranscribeResult
# The fixtures are IMPORTED, not copied: pytest discovers fixture objects in a
# module's globals, so `build` here is the very same factory the wiring
# suite uses (real modules on tmp paths, hardware stubbed, no Tk).
from tests.test_app_wiring import FakeTTS, build, paths, seams  # noqa: F401

SR = 16000
WAIT_S = 10.0            # the hard bound on "does the flag clear"
BLOCK_S = 12.0           # a blocking collaborator is held this long, then let go
JOIN_S = 15.0            # every thread join in this file

# The 00:29 transcript, verbatim from the log, and the three replies.
UTTERANCE = "Mald, so wouldn't worry about it."
REPLY_NONE = "Nudge, ROS, thank you"          # parse_yes_no -> None (00:29)
REPLY_YES = "yes"                             # -> True  (14:57)
REPLY_NO = "no"                               # -> False (14:56)

# What the speaker filter reported on the utterance at 00:29:27.869:
# "segment filter: 2/2 windows matched on 'voiceprint' (scores: 0.53, 0.37)"
UTTER_STATS = {"total": 2, "matched": 2, "scores": [0.53, 0.37],
               "best_score": 0.53, "who": "hunter", "who_is_owner": True,
               "who_scores": {"hunter": 0.53}, "labels": ("hunter",),
               "matched_label": "", "top": "hunter"}
# ...and on the follow-up at 00:29:34.428: "1/3 windows matched".
REPLY_STATS = {"total": 3, "matched": 1, "scores": [0.50, 0.04, -0.01],
               "best_score": 0.50, "who": "hunter", "who_is_owner": True,
               "who_scores": {"hunter": 0.50}, "labels": ("hunter",),
               "matched_label": "", "top": "hunter"}
# 12:25 / 21:53: "0/3 windows matched" -> filter returns None.
REJECT_STATS = {"total": 3, "matched": 0, "scores": [0.14, 0.16, 0.18],
                "best_score": 0.18, "who": "", "who_is_owner": False,
                "who_scores": {}, "labels": ("hunter",),
                "matched_label": "", "top": ""}


def _clip(seconds: float) -> np.ndarray:
    """A synthetic clip: silence. Nothing here has ever been a microphone."""
    return np.zeros(int(SR * seconds), dtype=np.float32)


# ----------------------------------------------------------------- stand-ins
class ScriptedTranscriber:
    """Returns the scripted (text, avg_logprob) pairs in order. Has the
    quiet decode too, so the redaction branch is the one the app takes."""
    loaded = True

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[tuple[str, str]] = []      # (door, text)

    def _next(self, door, audio):
        text, conf = self.script.pop(0) if self.script else ("", -3.0)
        self.calls.append((door, text))
        return TranscribeResult(text=text, confidence=conf,
                                segments=[(text, conf)],
                                audio_seconds=len(audio) / SR)

    def transcribe(self, audio, *, redact=False):
        return self._next("transcribe", audio)

    def transcribe_quiet(self, audio):
        return self._next("transcribe_quiet", audio)

    def partial(self, *a, **kw):
        return None


class ScriptedSpeaker:
    """filter_segments() answers from a script of (match: bool, stats)."""
    enrolled = True
    owner_label = "hunter"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def filter_segments(self, audio):
        self.calls += 1
        match, stats = self.script.pop(0) if self.script else (True, UTTER_STATS)
        return (audio if match else None), dict(stats)

    def add_sample(self, audio):
        return None


class FakeRecorder:
    """last_audio is synthetic; record_fixed returns synthetic silence at
    once, or waits on ``block`` (bounded) when a test asks for that shape."""

    def __init__(self):
        self.last_audio = None
        self.recording = False
        self.mic_available = True
        self.endpointer = None
        self.mic_devices: dict = {}
        self.fixed_calls: list[float] = []
        self.started = 0
        self.block: threading.Event | None = None

    def record_fixed(self, seconds):
        self.fixed_calls.append(float(seconds))
        if self.block is not None:
            self.block.wait(BLOCK_S)
        return _clip(seconds)

    def start(self):
        self.started += 1

    def stop(self, *a, **kw):
        return None

    def abort(self):
        return None

    def snapshot_final(self):
        return None


class RecordingTTS(FakeTTS):
    """FakeTTS plus the predicates the app reads, and an optional block
    on the one line the ask thread speaks with block=True."""
    busy = False
    is_speaking = False
    pending = 0
    last_text = ""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.block: threading.Event | None = None
        self.spoke_on: list[tuple[str, str]] = []   # (thread name, text)

    def speak(self, text, block=False):
        self.spoken.append(text)
        self.spoke_on.append((threading.current_thread().name, text))
        self.last_text = text
        if self.block is not None and text == "Was that for me?":
            self.block.wait(BLOCK_S)
        return None


# ------------------------------------------------------------------ helpers
def _stack(t: threading.Thread) -> list[str]:
    """The thread's stack NOW, innermost last -- the a4ecf8b instrument."""
    if t is None:
        return ["<no thread>"]
    frame = sys._current_frames().get(t.ident)
    if frame is None:
        return ["<thread %r has no frame; alive=%s>" % (t.name, t.is_alive())]
    out = []
    for chunk in traceback.format_stack(frame):
        out.extend(ln.rstrip() for ln in chunk.splitlines() if ln.strip())
    return out


def _thread_named(name: str) -> threading.Thread | None:
    for t in threading.enumerate():
        if t.name == name:
            return t
    return None


def _wait_clear(app, t, timeout=WAIT_S, drain=False) -> float | None:
    """Seconds until _audio_busy is clear AND the decode thread has exited,
    or None at the ceiling. ``drain`` plays the Tk pump on this thread."""
    t0 = time.monotonic()
    end = t0 + timeout
    while time.monotonic() < end:
        if drain:
            bus.drain()
        if not app._audio_busy.is_set() and not t.is_alive():
            return time.monotonic() - t0
        time.sleep(0.02)
    return None


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        try:
            self.lines.append("%s %s %s" % (threading.current_thread().name,
                                            record.name, record.getMessage()))
        except Exception:                          # noqa: BLE001 - a log only
            pass


@pytest.fixture
def lines():
    h = _Lines()
    root = logging.getLogger()
    root.addHandler(h)
    try:
        yield h
    finally:
        root.removeHandler(h)


@pytest.fixture
def make(build, monkeypatch, tmp_path):
    """The 00:29 app: owner enrolled with a passphrase, gate in shadow,
    speaker verification on, mic present, classifier saying UNCERTAIN."""
    made = []

    def _make(*, reply=REPLY_NONE, reply_match=True):
        app = build(TTS=RecordingTTS)
        monkeypatch.setattr(CONFIG, "speaker_verify", True)
        monkeypatch.setattr(CONFIG, "talkback", True)
        monkeypatch.setattr(CONFIG, "barge_in", True)
        monkeypatch.setattr(CONFIG, "endpoint_vad", True)
        monkeypatch.setattr(app_mod.MACHINE, "has_mic", True)
        # Never a sound and never a power-up sweep from a unit test.
        monkeypatch.setattr(app_mod, "play_beep", lambda *a, **kw: None)
        monkeypatch.setattr(app, "_wake_chime_enabled", lambda: False)
        monkeypatch.setattr(app, "_maybe_power_up", lambda *a, **kw: None)
        app.recorder = FakeRecorder()
        app.transcriber = ScriptedTranscriber([(UTTERANCE, -0.53),
                                               (reply, -0.51)])
        app.speaker = ScriptedSpeaker([(True, UTTER_STATS),
                                       (reply_match,
                                        REPLY_STATS if reply_match
                                        else REJECT_STATS)])
        # The gate: one owner, a phrase SET (its plaintext is a string
        # invented here and never the real one), shadow mode.
        reg = identity_mod.Registry(path=tmp_path / "people.json")
        ok, why = reg.add_person(identity_mod.Person(
            label="hunter", name="Hunter", honorific="sir",
            role=identity_mod.ROLE_OWNER, voice=True, consent="owner"))
        assert ok, why
        ok, why = reg.set_secret("hunter", "phrase_hash",
                                 pp.hash_secret("a phrase invented for a test"))
        assert ok, why
        app.gate = gate_mod.OwnerGate(registry=reg, get_option=app.get_option,
                                      owner="hunter", record=None)
        assert app._owner_has_phrase(), "the redaction branch must be reachable"
        assert app.gate.effective_mode() == gate_mod.MODE_SHADOW
        # The classifier's verdict at 00:29:28.323: UNCERTAIN at 0.50.
        app.commander.intent.classify = \
            lambda text: (IntentClassifier.UNCERTAIN, 0.5)
        app._turn_from_wake = True
        made.append(app)
        return app

    yield _make
    for app in made:
        try:
            app._audio_cancel_watchdog()
            app._turn_cancel_timers()
        except Exception:                          # noqa: BLE001 - teardown
            pass
        bus._root = None


def _drive(app, *, speculative: bool):
    """The RecordingStopped of 00:29:28.269, on the direct or the
    speculative path. Returns the decode thread."""
    audio = _clip(3.7)
    app.recorder.last_audio = audio
    # An ACCEPTED wake word opens the ledger's turn (_on_hotword, then the
    # RecordingStarted subscriber); the "turn[uncertain]" line needs it.
    app.turns.mark("wake")
    app.turns.mark("mic")
    if speculative:
        # _maybe_speculate ran at 00:29:28.100 (last_speech=2.82 s); the
        # stop at 28.234 was the VAD's and no more speech followed, so
        # _take_speculation reuses it -- "speculative transcript reused".
        spec = app_mod._Speculation(2.82)
        spec.audio, spec.stats, spec.rejected, spec.result = \
            app._decode_clip(audio)
        spec.finished = time.monotonic()
        spec.done.set()
        app._speculation = spec
        app.recorder.endpointer = SimpleNamespace(last_speech_seconds=2.82,
                                                  silence_since_speech=0.86)
    ev = RecordingStopped(reason="silence", endpoint="vad", dead_air_s=0.864,
                          followup=False, t=time.monotonic())
    app._on_recording_stopped(ev)
    t = app._audio_thread
    assert t is not None and t.name == "audio-decode"
    return t


def _report(tag, app, t, took, lines, extra=""):
    ask = _thread_named("uncertain-ask")
    print("\n[%s] audio_busy cleared in %s; decode thread alive=%s; "
          "turn_busy=%s; ask-thread alive=%s; spoke=%r%s" % (
              tag, ("%.3fs" % took) if took is not None else "NOT WITHIN %.0fs" % WAIT_S,
              t.is_alive(), app._turn_busy.is_set(),
              ask.is_alive() if ask else None, app.tts.spoken, extra))
    if took is None:
        print("[%s] decode thread stack (innermost last):" % tag)
        for ln in _stack(t):
            print("    " + ln)
    if ask is not None and ask.is_alive():
        print("[%s] uncertain-ask thread stack (innermost last):" % tag)
        for ln in _stack(ask):
            print("    " + ln)
    for ln in lines.lines:
        if any(k in ln for k in ("Uncertain intent", "turn[", "uncertain",
                                 "gate:", "Transcribed:", "speculative",
                                 "hotword ignored", "watchdog",
                                 "audio flag")):
            print("    log: " + ln)


# ------------------------------------------------- the three log variants
VARIANTS = {
    "bool-yes": (REPLY_YES, True),
    "bool-no": (REPLY_NO, True),
    "ignored": (REPLY_NONE, False),      # speaker mismatch: reply never parsed
    "none": (REPLY_NONE, True),          # parsed, unreadable -> None (00:29)
}


@pytest.mark.parametrize("path", ["direct", "speculative"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_decode_thread_returns_on_every_variant(make, lines, variant, path):
    reply, match = VARIANTS[variant]
    app = make(reply=reply, reply_match=match)
    t = _drive(app, speculative=(path == "speculative"))
    took = _wait_clear(app, t)
    ask = _thread_named("uncertain-ask")
    if ask is not None:
        ask.join(JOIN_S)
    _report("%s/%s" % (variant, path), app, t, took, lines)
    assert took is not None, "WEDGED: _audio_busy did not clear -- see the stack above"
    assert not t.is_alive()
    # The path taken is the log's path, not a shortcut around it.
    joined = "\n".join(lines.lines)
    assert "Uncertain intent (conf=0.50)" in joined
    assert "turn[uncertain]" in joined
    assert "gate: hunter (owner) on the voice leg" in joined
    if path == "speculative":
        assert "speculative transcript reused" in joined
    assert app.tts.spoken[:1] == ["Was that for me?"]
    assert app.recorder.fixed_calls == [app.UNCERTAIN_LISTEN_S]
    if variant == "ignored":
        assert "uncertain reply ignored: not the enrolled speaker" in joined
    elif variant == "none":
        assert "uncertain follow-up heard %r -> None" % REPLY_NONE in joined
    else:
        assert "uncertain follow-up heard %r -> %s" % (
            reply, "True" if variant == "bool-yes" else "False") in joined


# ---------------------------------------- what he actually saw, without a hang
@pytest.mark.parametrize("variant", ["ignored", "none"])
def test_an_unresolved_prompt_no_longer_holds_the_turn(make, lines, variant):
    """THE SIGNATURE IN THE LOG, REPRODUCED WITH THE DECODE THREAD GONE --
    and then closed.

    Five wake words at 00:29:43-00:30:00 were refused with "hotword ignored:
    still transcribing the previous clip". Before the fix that line was
    printed for EITHER flag: `_on_hotword` refused on `_audio_busy` OR on
    `_turn_busy` (unless Jarvis was mid-speech). The uncertain prompt
    returns done=False, so `_turn_busy` stayed set until an answer or the
    60 s turn watchdog -- and an unresolved prompt has no answer. As first
    written, this test asserted exactly that: turn still open, wake word
    refused with the transcribing line, released only by _turn_timed_out().

    It now asserts the inverse: with the decode thread gone and its flag
    clear, the unresolved prompt has released the turn BY ITSELF, the log
    says the question expired, and the very next wake word opens the mic.
    The full contract (every unanswered exit, the truthful refusal while
    the window is open, the card kept for a click) is in
    tests/test_uncertain_unanswered_releases_the_turn.py.
    """
    reply, match = VARIANTS[variant]
    app = make(reply=reply, reply_match=match)
    t = _drive(app, speculative=True)
    took = _wait_clear(app, t)
    ask = _thread_named("uncertain-ask")
    if ask is not None:
        ask.join(JOIN_S)
    assert took is not None and not t.is_alive()
    assert not app._audio_busy.is_set(), "the decode thread released its flag"
    end = time.monotonic() + 2.0
    while time.monotonic() < end and app._turn_busy.is_set():
        time.sleep(0.02)
    assert not app._turn_busy.is_set(), \
        "the unresolved prompt still holds the turn: the 00:29 minute"
    assert any("uncertain question expired" in ln for ln in lines.lines)
    before = len(lines.lines)
    app._on_hotword(0.896)                     # 00:29:43.581, score 0.896
    assert not any("hotword ignored" in ln for ln in lines.lines[before:])
    end = time.monotonic() + 2.0
    while time.monotonic() < end and app.recorder.started == 0:
        time.sleep(0.02)
    assert app.recorder.started == 1, "the wake word opens the mic once the question expired"
    _report("signature/%s" % variant, app, t, took, lines,
            extra="; turn released by the expiry, wake accepted")


# ---------------------------------------------- the collaborators' shapes
def _run_shape(app, lines, tag, *, drain=False):
    t = _drive(app, speculative=True)
    took = _wait_clear(app, t, drain=drain)
    _report(tag, app, t, took, lines)
    return t, took


def test_shape_speak_blocks(make, lines):
    """The TTS sidecar socket never answers 'Was that for me?'."""
    app = make()
    gate = threading.Event()
    app.tts.block = gate
    try:
        t, took = _run_shape(app, lines, "shape/speak-blocks")
        ask = _thread_named("uncertain-ask")
        assert took is not None, "a blocked speak() must not hold _audio_busy"
        assert ask is not None and ask.is_alive(), "the block lands on the ASK thread"
        assert any("tts.speak" in ln or "speak(" in ln for ln in _stack(ask))
    finally:
        gate.set()
        ask = _thread_named("uncertain-ask")
        if ask is not None:
            ask.join(JOIN_S)


def test_shape_record_fixed_blocks(make, lines):
    """The recorder device never returns from the 5 s follow-up capture."""
    app = make()
    gate = threading.Event()
    app.recorder.block = gate
    try:
        t, took = _run_shape(app, lines, "shape/record_fixed-blocks")
        ask = _thread_named("uncertain-ask")
        assert took is not None, "a blocked record_fixed() must not hold _audio_busy"
        assert ask is not None and ask.is_alive(), "the block lands on the ASK thread"
        assert any("record_fixed" in ln for ln in _stack(ask))
    finally:
        gate.set()
        ask = _thread_named("uncertain-ask")
        if ask is not None:
            ask.join(JOIN_S)


def test_shape_tk_attached_publish_only_queues(make, lines):
    """The live shape: a Tk root is attached, so publish() only queues and
    the subscribers run on the (test) thread that pumps, never on the
    decode thread."""
    app = make()
    bus._root = SimpleNamespace(after=lambda *a, **kw: None)
    try:
        t, took = _run_shape(app, lines, "shape/tk-attached", drain=True)
        assert took is not None
        ask = _thread_named("uncertain-ask")
        if ask is not None:
            ask.join(JOIN_S)
    finally:
        bus._root = None
        bus.drain()


def test_shape_publish_blocks_on_the_decode_threads_last_publish(make, lines):
    """The ONLY shape that wedges the flag: the decode thread's last
    publish -- _emit_result's Status("Was that for me?") -- never returns.
    The log rules this out: that Status reached the pane at 00:29:28.358,
    so with Tk attached the put returned. Kept as the demonstration of what
    a real wedge looks like under the a4ecf8b instrument."""
    app = make()
    gate = threading.Event()
    real = bus.publish

    def blocking(ev):
        if (threading.current_thread().name == "audio-decode"
                and isinstance(ev, Status)
                and ev.text == "Was that for me?"):
            gate.wait(BLOCK_S)
        return real(ev)

    bus.publish = blocking
    try:
        t, took = _run_shape(app, lines, "shape/publish-blocks")
        assert took is None, "this is the one shape that must wedge"
        assert app._audio_busy.is_set()
        frames = _stack(t)
        assert any("_emit_result" in ln for ln in frames)
        assert any("_process_audio" in ln for ln in frames)
    finally:
        gate.set()
        bus.publish = real
        t.join(JOIN_S)
        ask = _thread_named("uncertain-ask")
        if ask is not None:
            ask.join(JOIN_S)
        assert not t.is_alive() and not app._audio_busy.is_set()
