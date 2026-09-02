"""The 2026-09-02 08:56 stuck listen, and the three holes it went through.

WHAT HE SAW (his words):

    "How long do you need to get to Wisenbaker, sir? was stuck at speaking
     and wouldnt let me respond"

He answered by TYPING 21 s later -- /tmp/vss_voice/jarvis.log 08:56:36.530
``handle 'about 10 minutes' source=typed``.

WHAT THE LOG SAYS.  The window, verbatim:

    08:56:15.455 tts      speaking (f5): How long do you need to get to Wisenbaker, sir?
    08:56:15.465 leavetime leavetime: asked for the walk to Wisenbaker
    08:56:16.157 mixer    mixer: ducked 1 stream(s) to 30%
    08:56:18.754 tts      speech complete
    08:56:18.843 mixer    mixer: restored 1 stream(s)
    08:56:19.079 mixer    mixer: ducked 1 stream(s) to 30%      <-- never lifted
    08:56:36.530 commander handle 'about 10 minutes' source=typed

The duck at 19.079 was NOT a recording.  ``Recorder.start`` logs "Mic
native rate" and "Recording started" and publishes ``Status("Listening...")``
which the window logs -- all three appear on every one of the other captures
in that log, and NONE of them appears here.  Nothing opened a mic.

Two separate defects, and each one hid the other:

1.  THE QUESTION NEVER OPENED A MIC AT ALL.  ``App._ask_leave_time`` speaks
    the question and arms the commander's pending answer, and -- alone among
    every question Jarvis asks -- never sets ``_followup_after_speech``.  So
    ``_after_speech`` had nothing to do and the answer had nowhere to land.
    ``commander.question_open()`` does not count ``_pending_leave`` either,
    so even once a window opens it would be the 4 s one, not the 15 s one
    an answer needs.

2.  THE ROOM STAYED DUCKED AND THE BOARD STAYED ON "SPEAKING".  The TTS
    amplitude feeder's last tick is published as
    ``SpeakingState(active=True, amplitude=0.0)`` (jarvis/tts.py:1900), and
    over a long line the feeder's 80 ms sleeps drift behind the player: here
    it landed 325 ms AFTER the worker's ``SpeakingState(active=False)``.
    Every subscriber that reads ``active`` as a level latched on True with no
    falling edge left to come -- the mixer's "speaking" hold, the window's
    pill, ``app._tts_active``.  Exactly one orphan duck exists in the whole
    boot and it is this one.

The fix here is in three layers, because layer 2 is a class of bug and not a
line: the ask opens a window (and a question-sized one); nothing -- a hold or
a capture -- may hold the room or the mic down past a maximum without saying
so at WARNING and putting the world back; and a question with no mic behind
it says so on the board instead of looking like speech.

STILL OPEN, deliberately not touched here (both outside this branch's files):

  * jarvis/tts.py:1900 is the SOURCE.  ``_feed_amp`` signs off with
    ``bus.publish(SpeakingState(active=True, amplitude=0.0))``; the value
    that belongs there is ``active=self._speaking``, or nothing at all --
    the falling edge is the worker's to publish, not the feeder's.  Until
    that lands, the watchdogs below are what stands between a lost tick and
    a room left at 30 %.
  * jarvis/ui/main_window.py:1678 ``_ev_speaking`` reads ``ev.active`` as a
    level, so the pill latches on "Speaking" from the same tick -- which is
    literally the thing he reported seeing.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import mixer as mx
from jarvis import recorder as rec_mod
from jarvis.config import CONFIG, MACHINE
from jarvis.events import (RecordingStarted, RecordingStopped, Status, bus)
from jarvis.recorder import MicArbiter, Recorder

JARVIS_PID = 3708595            # the paplay stream in the pactl fixture
MUSIC_INDEX = "92"              # the librespot stream it must put back


# ====================================================================
# harnesses
# ====================================================================
class Clock:
    """A hand-driven wall clock for the mixer's hold watchdog."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _mixer(tmp_path, run=None, now=None, **cfg):
    from tests.test_mixer import FakeRun
    settings = {"audio.duck": True, "audio.duck_level": 30,
                "audio.duck_ramp_ms": 0}
    settings.update(cfg)
    conf = SimpleNamespace(get=lambda k, d=None: settings.get(k, d))
    m = mx.RoomMixer(cfg=conf, run=run or FakeRun(),
                     state_path=tmp_path / "mixer.json",
                     registry=mx.PidRegistry(), ppid_of=lambda pid: None,
                     sleep=lambda s: None, now=now or Clock(), remote=None)
    m._registry.add(JARVIS_PID)
    return m


def _volume_of(run, index=MUSIC_INDEX):
    """The last level the mixer wrote to a stream, as an int percent."""
    writes = [c for c in run.writes if c[2] == index]
    return int(writes[-1][-1].rstrip("%")) if writes else None


class FakeStream:
    """sounddevice.InputStream: opens, never delivers a frame."""

    def __init__(self, **kw):
        self.kw = kw
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


class FakeSounddevice:
    """Just enough of the module Recorder.start() imports."""

    def __init__(self):
        self.streams = []

    def query_devices(self, *a, **kw):
        if a:
            return {"default_samplerate": 16000}
        return [{"name": "fake mic", "max_input_channels": 1,
                 "default_samplerate": 16000}]

    def InputStream(self, **kw):
        s = FakeStream(**kw)
        self.streams.append(s)
        return s


@pytest.fixture
def mic(monkeypatch):
    """A recorder that can open a mic and will never be given any audio."""
    import sys
    sd = FakeSounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(MACHINE, "has_mic", True)
    monkeypatch.setattr(CONFIG, "sound", False)
    monkeypatch.setattr(rec_mod, "play_beep", lambda kind: None)
    return sd


def _wired(recorder, mixer):
    """Subscribe the real mixer to the real recorder's events."""
    subs = [(RecordingStarted, bus.subscribe(RecordingStarted,
                                             mixer.on_recording_started)),
            (RecordingStopped, bus.subscribe(RecordingStopped,
                                             mixer.on_recording_stopped))]
    return subs


def _unwire(subs):
    for etype, fn in subs:
        bus.unsubscribe(etype, fn)


# ====================================================================
# 1. the incident: a hold nothing will ever lift
# ====================================================================
def test_the_orphan_duck_of_08_56_19_is_let_go(tmp_path):
    """Replay of the exact edges, with the feeder's trailing tick last.

    On af64264 the room stays at 30 % for as long as Jarvis runs: there is
    no falling edge left in the world for the "speaking" hold, and the only
    thing that ever cleared it in the live log was the NEXT utterance's own
    ending 22 s later.
    """
    from tests.test_mixer import FakeRun
    run, clock = FakeRun(), Clock()
    m = _mixer(tmp_path, run=run, now=clock)

    m.on_speaking(SimpleNamespace(active=True, amplitude=0.0))   # 08:56:16.157
    m.pump()
    for _ in range(34):                                          # ~12 Hz ticks
        clock.advance(0.08)
        m.on_speaking(SimpleNamespace(active=True, amplitude=0.4))
        m.pump()
    assert _volume_of(run) == 30, "the room should be ducked while he talks"

    clock.advance(0.09)
    m.on_speaking(SimpleNamespace(active=False, amplitude=0.0))  # 08:56:18.754
    m.pump()
    assert _volume_of(run) == 100, "the burst ended: the room comes back"

    # jarvis/tts.py:1900 -- the feeder's last tick, 325 ms late.
    clock.advance(0.325)
    m.on_speaking(SimpleNamespace(active=True, amplitude=0.0))   # 08:56:19.079
    m.pump()
    assert _volume_of(run) == 30, "the incident: re-ducked with nothing playing"

    # Nothing more is ever published. The room must not stay down.
    clock.advance(mx.HOLD_MAX_S + 1.0)
    m.pump()
    assert _volume_of(run) == 100, \
        "a hold that outlives the speech it was taken for must be let go"
    assert not m.holds, "and the hold itself must be gone, not just the volume"


def test_the_hold_watchdog_names_the_hold_at_warning(tmp_path, caplog):
    """An unbounded wait that says nothing is worse than a short one."""
    from tests.test_mixer import FakeRun
    run, clock = FakeRun(), Clock()
    m = _mixer(tmp_path, run=run, now=clock)
    m.on_recording_started()
    m.pump()
    clock.advance(mx.HOLD_MAX_S + 1.0)
    with caplog.at_level("WARNING"):
        m.pump()
    text = " ".join(r.getMessage() for r in caplog.records
                    if r.levelname == "WARNING")
    assert "recording" in text, "the WARNING must name which hold it was"
    assert _volume_of(run) == 100


def test_speech_that_keeps_arriving_is_never_cut_short(tmp_path):
    """A read-aloud is one hold for many minutes. The watchdog must expire
    a hold that has gone SILENT, not a burst that is still speaking: the
    12 Hz SpeakingState ticks are the proof of life."""
    from tests.test_mixer import FakeRun
    run, clock = FakeRun(), Clock()
    m = _mixer(tmp_path, run=run, now=clock)
    for _ in range(int((mx.HOLD_MAX_S * 3) / 0.08)):
        m.on_speaking(SimpleNamespace(active=True, amplitude=0.4))
        clock.advance(0.08)
    m.pump()
    assert _volume_of(run) == 30, "he is still talking"
    m.on_speaking(SimpleNamespace(active=False, amplitude=0.0))
    m.pump()
    assert _volume_of(run) == 100


# ====================================================================
# 2. the backstop: no capture outlives its cap
# ====================================================================
def test_a_capture_whose_endpointer_never_fires_hands_everything_back(
        mic, tmp_path):
    """The damning one.

    Every exit a capture has -- the VAD endpoint, the energy timer, the
    60 s cap -- lives on ONE thread, ``Recorder._poll_loop``.  There is no
    second one.  If that thread dies, is starved or never starts, the mic
    stays open, the arbiter stays held (which mutes the wake word: Jarvis
    goes deaf), and the room stays ducked -- with nothing logged and
    nothing on screen.  A path with no timeout at all is the whole bug.
    """
    from tests.test_mixer import FakeRun
    run = FakeRun()
    m = _mixer(tmp_path, run=run)
    arb = MicArbiter()
    paused = []
    arb.register_hotword(lambda: paused.append("pause"),
                         lambda: paused.append("resume"))
    rec = Recorder(arb)
    rec.endpointer = None                      # the VAD never says a word
    rec.watchdog_cap_s = 0.4                   # the 60 s cap, shortened
    # The one thread that owns every ordinary exit is gone.
    rec._poll_loop = lambda: None

    stops = []
    subs = _wired(rec, m)
    f = bus.subscribe(RecordingStopped, stops.append)
    try:
        rec.start(followup=True)
        m.pump()
        assert rec.recording and arb.held_by == "recorder"
        assert paused == ["pause"], "the wake word is off while the mic is open"
        assert _volume_of(run) == 30

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and rec.recording:
            time.sleep(0.05)
        m.pump()

        assert not rec.recording, "the capture must end on its own"
        assert arb.held_by == "" and paused == ["pause", "resume"], \
            "a stranded acquire leaves Jarvis permanently deaf"
        assert stops, "RecordingStopped is what lifts the duck and the UI"
        assert _volume_of(run) == 100, "and the room must come back"
    finally:
        bus.unsubscribe(RecordingStopped, f)
        _unwire(subs)
        rec.abort()


def test_the_watchdog_says_who_was_holding_the_mic(mic, tmp_path, caplog):
    arb = MicArbiter()
    rec = Recorder(arb)
    rec.endpointer = None
    rec.watchdog_cap_s = 0.3
    rec._poll_loop = lambda: None
    try:
        with caplog.at_level("WARNING"):
            rec.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and rec.recording:
                time.sleep(0.05)
        text = " ".join(r.getMessage() for r in caplog.records
                        if r.levelname == "WARNING")
        assert "recorder" in text and "watchdog" in text.lower()
    finally:
        rec.abort()


def test_stop_hands_the_world_back_even_when_finalising_explodes(mic):
    """``stop()`` is a happy path today: everything after the raise is
    skipped, so the arbiter comes back (its own try/finally) but
    ``RecordingStopped`` is never published -- and that event is the ONLY
    thing that lifts the mixer's duck and clears "Listening…" from the
    board.  A silent, permanent listening state, which is the incident's
    exact shape."""
    arb = MicArbiter()
    rec = Recorder(arb)
    rec.endpointer = None
    rec._poll_loop = lambda: None
    rec._finalize_audio = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    stops = []
    f = bus.subscribe(RecordingStopped, stops.append)
    try:
        rec.start()
        with pytest.raises(RuntimeError):
            rec.stop(reason="silence")
        assert arb.held_by == "", "the mic must come back"
        assert stops, "and the world must be told the capture ended"
        assert rec.recording is False
    finally:
        bus.unsubscribe(RecordingStopped, f)
        rec.abort()


# ====================================================================
# 3. the specific path: the walk question opens a mic for its answer
# ====================================================================
def _leave_app(tmp_path, monkeypatch, **over):
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(CONFIG, "followup_window", 4.0)
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: d, user_name="Hunter")
    a._init_assistant_state()
    a.starts = []
    a.recorder = SimpleNamespace(
        endpointer=object(), recording=False,
        start=lambda followup=False, **kw: a.starts.append((followup, kw)))
    a._audio_busy = threading.Event()
    a._turn_busy = threading.Event()
    a._tts_active = False
    a._pending_uncertain = None
    a._pending_debrief = None
    a._briefing_pending = False
    a._followup_after_speech = False
    a._wake_pending = False
    a.tts = SimpleNamespace(is_speaking=False, pending=0, busy=False)
    a.quiet = SimpleNamespace(should_hold=lambda *x, **k: False,
                              hold=lambda *x, **k: True, reason=lambda: "")
    a.said = []
    a._say = lambda text, proactive=False, kind="message": a.said.append(text)
    a.commander = SimpleNamespace(_pending_leave=None, _pending_session=None,
                                  lecture_course=None, _pending_quiz=None,
                                  _pending_destructive=None)
    a.services = SimpleNamespace(alarm_offer=None)

    def _arm(key, place):
        a.commander._pending_leave = (key, place, time.monotonic())
        return True
    a.commander.ask_leave_time = _arm
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_the_walk_question_opens_a_mic_for_its_answer(tmp_path, monkeypatch):
    """He should never have had to type. Every other question Jarvis asks
    arms the follow-up window; this one, alone, did not."""
    monkeypatch.setattr(app_mod, "MACHINE", SimpleNamespace(has_mic=True))
    a = _leave_app(tmp_path, monkeypatch)
    assert a._ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker") is True
    assert a._followup_after_speech is True, \
        "the walk question must arm the mic that answers it"
    a._after_speech()
    deadline = time.monotonic() + 3.0            # _start_followup's 0.15 s hop
    while time.monotonic() < deadline and not a.starts:
        time.sleep(0.02)
    assert [s[0] for s in a.starts] == [True], \
        "and the window must actually open, without a wake word"
    assert a.starts[0][1].get("window", 0) >= 15.0, \
        "and it must be the question window, not the 4 s \"...and Tuesday?\" one"


def test_the_walk_answer_gets_the_question_window_not_four_seconds(
        tmp_path, monkeypatch):
    """"About ten minutes, I suppose" is a sentence, not a word. A live
    leave question is a question on the table like any other."""
    a = _leave_app(tmp_path, monkeypatch)
    assert a._capture_window() is None            # nothing on the table yet
    a.commander._pending_leave = ("Wisenbaker Engineering Bldg",
                                  "Wisenbaker", time.monotonic())
    assert a._question_open(a.commander) is True
    assert (a._capture_window() or 0) >= 15.0


def test_a_question_with_no_mic_behind_it_says_so(tmp_path, monkeypatch,
                                                  caplog):
    """LAYER 3.  When the window cannot open, the board must not be left
    reading "Speaking" while Jarvis waits for an answer that can only be
    typed.  Silence here is what made a dead question look like a live one.
    """
    monkeypatch.setattr(app_mod, "MACHINE", SimpleNamespace(has_mic=False))
    a = _leave_app(tmp_path, monkeypatch)
    a.commander._pending_leave = ("Wisenbaker Engineering Bldg",
                                  "Wisenbaker", time.monotonic())
    a._followup_after_speech = True
    seen = []
    f = bus.subscribe(Status, seen.append)
    try:
        with caplog.at_level("WARNING"):
            a._after_speech()
    finally:
        bus.unsubscribe(Status, f)
    assert a.starts == []
    text = " ".join(r.getMessage() for r in caplog.records
                    if r.levelname == "WARNING")
    assert "follow-up" in text.lower()
    assert seen and any(s.kind in ("warn", "busy") for s in seen), \
        "he must be able to see, in one glance, that nothing is listening"


def test_the_orphan_duck_is_the_only_one_in_the_whole_boot():
    """Provenance, so the next reader does not have to re-derive it.

    Every "ducked" line in /tmp/vss_voice/jarvis.log is preceded within 2 s
    by a "speaking (f5)" or a "Recording started" -- except one, at
    08:56:19.079, which is the incident. That is what makes it a race in
    the TTS amplitude feeder (jarvis/tts.py:1900) and not a policy.

    Kept as a fixture, not a read of the live log: the log is scratch
    (/tmp is wiped at boot) and the suite must never touch the running
    app's files.
    """
    window = [("08:56:15.455", "SPEAK"), ("08:56:16.157", "DUCK"),
              ("08:56:18.754", "DONE"), ("08:56:18.843", "RESTORE"),
              ("08:56:19.079", "DUCK"), ("08:56:36.531", "SPEAK")]

    def sec(stamp):
        h, m, rest = stamp.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    orphans = [t for i, (t, kind) in enumerate(window) if kind == "DUCK"
               and not any(k in ("SPEAK", "REC") and 0 <= sec(t) - sec(u) < 2.0
                           for u, k in window[:i])]
    assert orphans == ["08:56:19.079"]
