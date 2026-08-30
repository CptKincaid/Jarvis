"""Speculative transcription during the endpoint silence (jarvis/app.py,
JarvisApp._maybe_speculate / _take_speculation).

Every turn paid the full decode strictly AFTER the recorder stopped, while
the GPU idled through the 0.8 s endpoint silence (ledger: dead-air
0.80-0.96 s, then stt 0.40-0.67 s). The partial thread now decodes the clip
once the VAD has heard 0.3 s of silence; if no more speech follows before
the stop, _process_audio reuses that result. These tests pin what makes
that safe: the speculative clip is shaped exactly like the final one, one
pass per pause, the real path WAITS for a pass in flight rather than
decoding twice, a pass is discarded when more speech followed or the stop
was not the VAD's, and the ledger says when a turn's stt was speculative.
The recorder is the real class with only its hardware stubbed, as in
tests/test_endpoint.py.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np

import jarvis.app as app_mod
import jarvis.events as events_mod
import jarvis.recorder as recorder_mod
from jarvis.config import CONFIG
from jarvis.events import PartialText, RecordingStopped, Transcribed, UserUtterance
from jarvis.recorder import SAMPLE_RATE, MicArbiter, Recorder
from jarvis.turnclock import TurnLedger


# ------------------------------------------------------------------ stubs
class FakeEP:
    """The two endpointer readings the speculation uses."""

    def __init__(self, gap=None, last=None):
        self.silence_since_speech, self.last_speech_seconds = gap, last


class FakeTranscriber:
    loaded = True

    def __init__(self, text="what time is it", block=None):
        self.text, self.block = text, block
        self.calls, self.partials = [], 0

    def transcribe(self, audio):
        self.calls.append(np.array(audio, copy=True))
        if self.block is not None:
            self.block.wait(5)
        return SimpleNamespace(text=self.text, confidence=-0.2, accepted=True)

    def partial(self, audio):
        self.partials += 1
        return "partial preview"


def _recorder(ep, seconds=2.0, rate=16000):
    """A recorder mid-capture with `seconds` of audio, as test_endpoint builds it."""
    rec = object.__new__(Recorder)
    rec.recording = True
    rec.endpointer = ep
    rec._record_rate = rate
    rec._resample_to_16k = lambda a: a
    n = int(rate * seconds)
    rng = np.random.default_rng(1)
    rec._audio_frames = [(0.2 * rng.standard_normal(n)).astype(np.float32).reshape(-1, 1)]
    rec.last_audio = None
    return rec


def _app(monkeypatch, rec, tr=None, speaker=None):
    monkeypatch.setattr(CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(CONFIG, "noise_gate", True)
    monkeypatch.setattr(CONFIG, "speaker_verify", speaker is not None)
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: d, user_name="Hunter")
    a._init_assistant_state()
    a.recorder, a.transcriber = rec, tr or FakeTranscriber()
    a.speaker = speaker or SimpleNamespace(enrolled=False)
    a._audio_busy = threading.Event()
    a._tts_active = False
    a.said, a.dispatched, a.marks = [], [], []
    a._say = a.said.append
    a._dispatch = lambda text, source: a.dispatched.append((text, source))
    a._maybe_learn_voice = lambda audio, stats: None
    a.turns = SimpleNamespace(mark=lambda *x, **k: a.marks.append(x),
                              abandon=lambda r: a.marks.append(("abandon", r)))
    a.events = []
    monkeypatch.setattr(events_mod.bus, "publish", a.events.append)
    return a


def _stop(a, endpoint="vad"):
    """The recorder stopped: what _on_recording_stopped leaves behind."""
    a.recorder.recording = False
    a._stop_event = RecordingStopped(reason="silence", endpoint=endpoint, dead_air_s=0.8)


def _transcribed(a):
    return [e for e in a.events if isinstance(e, Transcribed)]


# ------------------------------------------------------ the shaped clip
def test_snapshot_final_is_what_finalize_audio_would_return(monkeypatch):
    """The reuse is only honest if whisper saw the same clip: snapshot_audio()
    skips the noise gate (which zeroes quiet 100 ms blocks) and the 60 s cap."""
    monkeypatch.setattr(CONFIG, "noise_gate", True)
    monkeypatch.setattr(events_mod.bus, "publish", lambda ev: None)
    rec = _recorder(FakeEP(), seconds=61.0)
    quiet = rec._audio_frames[0]
    quiet[SAMPLE_RATE:SAMPLE_RATE + 1600] = 0.0001          # one block under the gate
    spec = rec.snapshot_final()
    final = rec._finalize_audio()
    assert len(spec) == SAMPLE_RATE * 60, "the 60 s cap was skipped"
    assert np.array_equal(spec, final)
    assert not spec[SAMPLE_RATE:SAMPLE_RATE + 1600].any(), "the noise gate was skipped"
    assert spec[:SAMPLE_RATE].any(), "the gate zeroed loud audio too"
    short = _recorder(FakeEP(), seconds=0.2)
    assert short.snapshot_final() is None and short._finalize_audio() is None


def test_a_pause_of_0_3s_triggers_one_decode_of_the_shaped_clip(monkeypatch):
    rec = _recorder(FakeEP(gap=0.1, last=1.5))
    a = _app(monkeypatch, rec)
    assert a._maybe_speculate() is False and a.transcriber.calls == []
    rec.endpointer.silence_since_speech = 0.3
    assert a._maybe_speculate() is True
    assert len(a.transcriber.calls) == 1
    assert np.array_equal(a.transcriber.calls[0], rec.snapshot_final())
    # the speculative text is the live preview too
    assert [e.text for e in a.events if isinstance(e, PartialText)] == ["what time is it"]
    # the same pause is never decoded twice, however often the loop polls
    for _ in range(5):
        assert a._maybe_speculate() is False
    assert len(a.transcriber.calls) == 1


def test_the_stop_reuses_a_matching_pass_and_marks_it(monkeypatch):
    rec = _recorder(FakeEP(gap=0.3, last=1.5))
    a = _app(monkeypatch, rec)
    assert a._maybe_speculate()
    _stop(a, "vad")
    a._process_audio(rec.snapshot_final())
    assert len(a.transcriber.calls) == 1, "the real path decoded again"
    (ev,) = _transcribed(a)
    assert ev.speculative is True and ev.text == "what time is it" and ev.accepted
    assert [e.text for e in a.events if isinstance(e, UserUtterance)] == ["what time is it"]
    assert a.dispatched == [("what time is it", "voice")]
    assert a._speculation is None, "the stash must not outlive the turn"
    assert not a._audio_busy.is_set()


def test_more_speech_after_the_snapshot_discards_the_pass(monkeypatch):
    """"What time is it... in Tokyo": the last word moved, so the speculated
    clip is not the clip."""
    rec = _recorder(FakeEP(gap=0.3, last=1.5))
    a = _app(monkeypatch, rec)
    assert a._maybe_speculate()
    rec.endpointer.last_speech_seconds = 2.4
    _stop(a, "vad")
    a._process_audio(rec.snapshot_final())
    assert len(a.transcriber.calls) == 2
    (ev,) = _transcribed(a)
    assert ev.speculative is False


def test_a_stop_that_is_not_the_vads_never_reuses_a_pass(monkeypatch):
    """A manual or energy stop may hold frames the endpointer never scored."""
    for endpoint in ("energy", "manual", "cap"):
        rec = _recorder(FakeEP(gap=0.3, last=1.5))
        a = _app(monkeypatch, rec)
        assert a._maybe_speculate()
        _stop(a, endpoint)
        a._process_audio(rec.snapshot_final())
        assert len(a.transcriber.calls) == 2, endpoint
        assert _transcribed(a)[0].speculative is False


def test_the_real_path_waits_for_a_pass_in_flight(monkeypatch):
    """The trigger at 0.3 s plus a 0.4-0.6 s decode usually finishes right
    around the 0.8 s stop: starting a second decode then would only queue
    behind the first on the model lock, so the real path joins it."""
    release = threading.Event()
    tr = FakeTranscriber(block=release)
    rec = _recorder(FakeEP(gap=0.3, last=1.5))
    a = _app(monkeypatch, rec, tr)
    spec_thread = threading.Thread(target=a._maybe_speculate, daemon=True)
    spec_thread.start()
    deadline = time.monotonic() + 2.0
    while not tr.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(tr.calls) == 1, "the speculative decode did not start"
    _stop(a, "vad")
    real = threading.Thread(target=a._process_audio, args=(rec.snapshot_final(),), daemon=True)
    real.start()
    time.sleep(0.15)
    assert real.is_alive() and _transcribed(a) == [], "did not wait for the pass"
    assert len(tr.calls) == 1, "a second decode was started"
    release.set()
    real.join(3.0)
    spec_thread.join(1.0)
    assert not real.is_alive()
    assert len(tr.calls) == 1
    assert _transcribed(a)[0].speculative is True


def test_a_speaker_rejected_pass_is_reused_as_the_rejection(monkeypatch):
    """The speaker filter is part of the speculated work (its ECAPA cost is
    not in the ledger's stt figure); a clip it dropped stays dropped."""
    seen = []

    class Speaker:
        enrolled = True

        def filter_segments(self, audio):
            seen.append(len(audio))
            return None, {"best_score": 0.12, "scores": [0.12]}
    rec = _recorder(FakeEP(gap=0.3, last=1.5))
    a = _app(monkeypatch, rec, speaker=Speaker())
    assert a._maybe_speculate()
    assert a.transcriber.calls == [], "transcribed a clip the speaker gate dropped"
    _stop(a, "vad")
    a._process_audio(rec.snapshot_final())
    assert len(seen) == 1, "the speaker filter ran again"
    (ev,) = _transcribed(a)
    assert (ev.accepted, ev.reject_reason, ev.speculative) == (False, "speaker", True)
    assert abs(ev.speaker_score - 0.12) < 1e-9


def test_a_pass_that_failed_is_not_reused(monkeypatch):
    class Boom(FakeTranscriber):
        def transcribe(self, audio):
            if not self.calls:
                self.calls.append(audio)
                raise RuntimeError("cuda hiccup")
            return super().transcribe(audio)
    rec = _recorder(FakeEP(gap=0.3, last=1.5))
    a = _app(monkeypatch, rec, Boom())
    assert a._maybe_speculate() is True         # it ran, and failed quietly
    _stop(a, "vad")
    a._process_audio(rec.snapshot_final())
    assert len(a.transcriber.calls) == 2
    assert _transcribed(a)[0].speculative is False and a.dispatched


def test_nothing_speculates_without_a_vad_or_with_it_switched_off(monkeypatch):
    rec = _recorder(None)
    a = _app(monkeypatch, rec)
    assert a._maybe_speculate() is False
    rec.endpointer = FakeEP(gap=0.5, last=1.0)
    monkeypatch.setattr(CONFIG, "endpoint_vad", False)
    assert a._maybe_speculate() is False
    monkeypatch.setattr(CONFIG, "endpoint_vad", True)
    a.assistant = SimpleNamespace(get=lambda k, d=None: False if k == "listening.speculative_stt" else d)
    assert a._maybe_speculate() is False
    a.transcriber.loaded = False                # never load whisper from the preview thread
    a.assistant = SimpleNamespace(get=lambda k, d=None: d)
    assert a._maybe_speculate() is False
    assert a.transcriber.calls == []
    # and the endpointer of the wiring tests' _Stub recorder is a callable,
    # not an endpointer: that must read as "no speculation", not a crash
    rec.endpointer = lambda *a, **k: None
    a.transcriber.loaded = True
    assert a._maybe_speculate() is False


def test_a_new_capture_drops_the_previous_captures_pass(monkeypatch):
    rec = _recorder(FakeEP(gap=0.3, last=1.5))
    a = _app(monkeypatch, rec)
    assert a._maybe_speculate()
    rec.recording = False                       # (keeps the preview thread it starts idle)
    a._on_recording_started(None)               # the next capture opens
    assert a._speculation is None


def test_the_partial_loop_runs_the_pass_and_rests_the_preview(monkeypatch):
    """One thread for both: while the speculative pass holds the model the
    greedy preview is suspended, and after it the preview backs off a whole
    interval instead of re-decoding the same silence."""
    rec = _recorder(FakeEP(gap=None, last=None))
    a = _app(monkeypatch, rec)
    a._PARTIAL_INTERVAL_S, a._PARTIAL_MIN_S, a._SPECULATE_POLL_S = 0.02, 0.5, 0.005
    ticks = {"n": 0}
    orig = a.transcriber.partial

    def partial(audio):
        ticks["n"] += 1
        if ticks["n"] == 2:                     # the user pauses after two previews
            rec.endpointer.silence_since_speech, rec.endpointer.last_speech_seconds = 0.3, 1.5
        return orig(audio)
    a.transcriber.partial = partial
    orig_spec = a._maybe_speculate

    def spec():
        ran = orig_spec()
        if ran:
            threading.Timer(0.005, lambda: setattr(rec, "recording", False)).start()
        return ran
    a._maybe_speculate = spec
    a._partial_loop()
    assert len(a.transcriber.calls) == 1
    assert ticks["n"] == 2, "the preview kept decoding through the pass"
    texts = [e.text for e in a.events if isinstance(e, PartialText)]
    assert texts == ["partial preview", "what time is it"], texts


def test_the_ledger_stamps_a_speculative_stt_honestly():
    """"stt 12ms" would read as a faster model; the note says why."""
    clock = {"t": 100.0}
    got = []
    turns = TurnLedger(clock=lambda: clock["t"], emit=got.append)
    a = object.__new__(app_mod.JarvisApp)
    a.turns = turns
    turns.mark("wake")
    clock["t"] = 100.1
    turns.mark("mic")
    turns.mark("speech_end", at=102.0)
    turns.mark("stop", at=102.8, stop="vad")
    a._turn_on_transcribed(Transcribed(text="what time is it", speculative=True, t=102.81))
    turns.mark("handle", at=102.9)
    turns.mark("audio", at=103.2)
    (rec,) = got
    assert abs(rec["stt"] - 0.01) < 1e-6 and rec["decode"] == "speculative"
    assert "decode=speculative" in TurnLedger.format(rec)
    turns.mark("wake")
    a._turn_on_transcribed(Transcribed(text="x", speculative=False))
    turns.abandon("empty")
    assert "decode" not in got[1]


def test_the_beep_kinds_include_the_nudge(monkeypatch, tmp_path):
    """Sanity for the earcon the nudge policy plays (recorder.play_beep)."""
    monkeypatch.setattr(recorder_mod.PATHS, "LOG_DIR", tmp_path)
    monkeypatch.setattr(recorder_mod, "_BEEP_FILES", {})
    recorder_mod._init_beeps()
    assert set(recorder_mod._BEEP_FILES) == {"start", "stop", "nudge"}
    assert (tmp_path / "beep_nudge.wav").stat().st_size > 44


def test_recording_stopped_says_how_the_session_opened(monkeypatch):
    """The nudge policy must know a follow-up window from a wake turn, and
    the recorder is the only party that does (recorder._followup)."""
    monkeypatch.setattr(CONFIG, "sound", False)
    rec = Recorder(MicArbiter())
    got = []
    monkeypatch.setattr(events_mod.bus, "publish", got.append)
    for followup, method in ((True, "stop"), (False, "stop"), (True, "abort")):
        got.clear()
        rec.recording, rec._followup = True, followup
        rec._stream, rec._poll_thread, rec._session_ctx = None, None, None
        rec._audio_frames = []
        getattr(rec, method)()
        assert rec.followup is followup
        (ev,) = [e for e in got if isinstance(e, RecordingStopped)]
        assert ev.followup is followup, (method, followup)
        assert ev.reason == ("abort" if method == "abort" else "manual")
