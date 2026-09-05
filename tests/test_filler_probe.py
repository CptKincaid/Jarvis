"""scripts/filler_probe.py -- the numbers-only instrument for the filler
hold, and the rules it runs under.

The probe is HIS to run: it is the one place the hold meets a microphone,
and it opens the mic only when he runs it from a shell. Nothing here does.
Every test drives run_take/summarize/main with fakes: a scripted recorder
that never touches a device, a scripted transcriber, a fake clock, fake
input()/print(). What is pinned without a device:

* this test file and the probe's CODE name no sounddevice stream and no
  audio writer (token-level, so the docstrings may discuss them);
* the takes alternate hint OFF / hint ON;
* per take it collects the numbers the design asks for -- fillers seen in
  the preview (word, capture second), the last speech second, the gap that
  ended the capture, holds fired, whether the tail was ever decoded, his
  own y/n on being cut off, the transcript's LENGTH -- and prints the
  transcript's words only under --show;
* it mirrors the app's speculative pass (which parks the preview) unless
  told not to, so the measurement is of the live timing;
* a capture that outlives max_take_s is stopped;
* the summary is four lines he can paste back;
* JARVIS_DEBUG_AUDIO is cleared before any capture (never saves audio).
"""
from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import tokenize
from types import SimpleNamespace

import numpy as np

import jarvis.events as events_mod
from jarvis.config import CONFIG
from jarvis.events import RecordingStopped

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_HERE, "scripts", "filler_probe.py")
_SPEC = importlib.util.spec_from_file_location("filler_probe", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["filler_probe"] = probe
_SPEC.loader.exec_module(probe)


def _code_names(path: str) -> set:
    """NAME tokens of the file's code: strings and comments excluded."""
    names = set()
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type == tokenize.NAME:
                names.add(tok.string)
    return names


# ------------------------------------------------------------ the rules
def test_neither_this_file_nor_the_probe_names_a_stream_or_an_audio_writer():
    forbidden = {"sounddevice", "InputStream", "RawInputStream", "rec",
                 "soundfile", "wave", "save", "savez", "tofile", "write",
                 "play", "paplay", "aplay"}
    for path in (_PATH, os.path.abspath(__file__)):
        hit = _code_names(path) & forbidden
        assert not hit, f"{os.path.basename(path)} names {sorted(hit)}"


def test_the_probe_says_it_is_his_to_run_and_never_saves_audio():
    doc = probe.__doc__ or ""
    assert re.search(r"\bhis to run\b", doc, re.I), "the docstring must say whose it is"
    assert re.search(r"never (saves|writes) audio", doc, re.I)


def test_forbid_audio_dumps_clears_the_debug_env(monkeypatch):
    monkeypatch.setenv("JARVIS_DEBUG_AUDIO", "1")
    assert probe.forbid_audio_dumps() is True
    assert "JARVIS_DEBUG_AUDIO" not in os.environ
    assert probe.forbid_audio_dumps() is False


def test_the_takes_alternate_hint_off_and_on():
    assert probe.plan(4) == [False, True, False, True]
    assert probe.plan(1) == [False]
    assert probe.plan(0) == []


# ------------------------------------------------------------ fakes
class FakeEP:
    def __init__(self, last=None, gap=None):
        self.last_speech_seconds, self.silence_since_speech = last, gap


class FakeRecorder:
    """Never a device: start() flips a flag, note_partial() counts decodes
    and ends the capture on the Nth by publishing RecordingStopped, exactly
    as the real poll thread would from its own side."""

    def __init__(self, stop_after=3, holds=1, gap=2.3, endpoint="vad",
                 last_speech=2.5, auto_stop=True):
        self.recording = False
        self.endpointer = FakeEP(last=last_speech)
        self.notes, self.stops = [], []
        self.decodes, self.starts = 0, 0
        self.last_audio = None
        self._stop_after, self._holds, self._gap = stop_after, holds, gap
        self._endpoint, self._auto = endpoint, auto_stop
        self.capture_id = 1              # as the real Recorder: bumped by start()

    def start(self):
        self.starts += 1
        self.capture_id += 1
        self.recording = True

    def snapshot_audio(self):
        return np.zeros(int(16000 * (1.0 + 0.9 * self.decodes)), dtype=np.float32)

    def snapshot_final(self):
        return self.snapshot_audio()

    def note_partial(self, text, audio_end_s, capture_id=None):
        # The real Recorder drops a note stamped with a finished capture.
        assert capture_id == self.capture_id, (capture_id, self.capture_id)
        self.notes.append((text, audio_end_s))
        self.decodes += 1
        if self._auto and self.decodes >= self._stop_after:
            self._end()

    def _end(self):
        self.recording = False
        self.last_audio = np.zeros(16000 * 3, dtype=np.float32)
        events_mod.bus.publish(RecordingStopped(
            reason="silence", endpoint=self._endpoint, dead_air_s=self._gap,
            filler_holds=self._holds))

    def stop(self, reason="manual", **kw):
        self.stops.append(reason)
        self._end()


class FakeTranscriber:
    def __init__(self, texts, final="remind me to call the dentist"):
        self.texts, self.i, self.final = list(texts), 0, final
        self.hints, self.finals = [], 0

    def partial(self, audio):
        self.hints.append(CONFIG.filler_prompt_hint)
        t = self.texts[min(self.i, len(self.texts) - 1)]
        self.i += 1
        return t

    def transcribe(self, audio):
        self.finals += 1
        return SimpleNamespace(text=self.final, accepted=True)


class FakeClock:
    def __init__(self, step=0.1):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


TEXTS = ["set a timer for", "set a timer for, um", "set a timer for, um, ten minutes"]


def _pipe(recorder=None, tr=None):
    return SimpleNamespace(recorder=recorder or FakeRecorder(),
                           transcriber=tr or FakeTranscriber(TEXTS))


def _run(pipe, hint=False, answers=("", "y"), show=False, speculate=True,
         clock=None, n=1, max_take_s=20.0):
    said, answers = [], list(answers)
    take = probe.run_take(pipe, n=n, hint=hint,
                          ask=lambda prompt: answers.pop(0) if answers else "",
                          say=said.append, show=show, speculate=speculate,
                          cadence_s=0.0, min_s=0.7, max_take_s=max_take_s,
                          clock=clock or FakeClock(), sleep=lambda s: None)
    return take, said


# ------------------------------------------------------------ one take
def test_run_take_collects_the_numbers_from_fakes(monkeypatch):
    monkeypatch.setattr(CONFIG, "filler_prompt_hint", False)
    pipe = _pipe()
    take, said = _run(pipe, hint=True)
    recorder, tr = pipe.recorder, pipe.transcriber
    assert recorder.starts == 1
    # every decode reached the recorder's seam with the buffer's END
    assert recorder.notes == [(TEXTS[0], 1.0), (TEXTS[1], 1.9), (TEXTS[2], 2.8)]
    assert take.hint is True and tr.hints == [True, True, True]
    assert take.fillers == [("um", 1.9)]          # (word, capture second)
    assert take.partial_ends == [1.0, 1.9, 2.8]
    assert take.last_speech_s == 2.5
    assert take.gap_s == 2.3 and take.endpoint == "vad"
    assert take.holds == 1
    assert take.cut is True                       # he answered y
    assert take.transcript_len == len("remind me to call the dentist")
    assert take.tail_decoded is True              # 2.8 >= 2.5 - 0.6
    assert take.spec_passes == 0                  # the fake VAD never paused
    # the take's hint is restored afterwards: the probe changes CONFIG in
    # memory only, and only for the take
    assert CONFIG.filler_prompt_hint is False


def test_the_tail_undecoded_case_is_reported(monkeypatch):
    """The um was spoken after the last snapshot: no partial reaches the
    last speech, which is the limit the probe exists to count."""
    monkeypatch.setattr(CONFIG, "filler_prompt_hint", False)
    pipe = _pipe(recorder=FakeRecorder(last_speech=3.9, holds=0))
    take, _ = _run(pipe)
    assert take.tail_decoded is False and take.holds == 0


def test_run_take_never_prints_the_transcript_unless_show(monkeypatch):
    monkeypatch.setattr(CONFIG, "filler_prompt_hint", False)
    _, said = _run(_pipe())
    joined = "\n".join(said)
    # the instruction line quotes an EXAMPLE command by design; what must
    # never appear is what whisper heard
    assert "dentist" not in joined, joined
    assert "29" in joined                          # its length is printed
    _, said = _run(_pipe(), show=True)
    assert "remind me to call the dentist" in "\n".join(said)


def test_run_take_mirrors_the_speculative_pass_that_parks_the_preview(monkeypatch):
    """Live, _maybe_speculate runs a full decode 0.3 s into every pause and
    the preview waits behind it; the probe does the same so its catch rate
    is the live one, and --no-speculate measures the difference."""
    monkeypatch.setattr(CONFIG, "filler_prompt_hint", False)
    recorder = FakeRecorder(stop_after=3)
    recorder.endpointer = FakeEP(last=1.9, gap=0.5)
    pipe = _pipe(recorder=recorder)
    take, _ = _run(pipe, speculate=True)
    assert take.spec_passes == 1                  # one pause, one pass
    assert pipe.transcriber.finals == 2           # the pass + the final decode
    recorder = FakeRecorder(stop_after=3)
    recorder.endpointer = FakeEP(last=1.9, gap=0.5)
    pipe = _pipe(recorder=recorder)
    take, _ = _run(pipe, speculate=False)
    assert take.spec_passes == 0 and pipe.transcriber.finals == 1


def test_a_capture_that_outlives_max_take_s_is_stopped(monkeypatch):
    monkeypatch.setattr(CONFIG, "filler_prompt_hint", False)
    recorder = FakeRecorder(auto_stop=False)
    pipe = _pipe(recorder=recorder)
    take, said = _run(pipe, max_take_s=2.0)
    assert recorder.stops == ["manual"]
    assert take is not None and take.endpoint == "vad"   # the fake's stop event


def test_a_mic_that_does_not_open_ends_the_take_with_a_reason(monkeypatch):
    monkeypatch.setattr(CONFIG, "filler_prompt_hint", False)

    class NoMic(FakeRecorder):
        def start(self):
            self.starts += 1               # recording stays False

    take, said = _run(_pipe(recorder=NoMic()))
    assert take is None
    assert any("mic" in line.lower() for line in said)


# ------------------------------------------------------------ the summary
def _take(hint, fillers, holds, cut, tail, endpoint="vad", gap=2.3, length=30):
    return probe.Take(n=1, hint=hint, fillers=fillers, partial_ends=[1.0, 1.9],
                      last_speech_s=1.9 if tail else 3.0, gap_s=gap,
                      endpoint=endpoint, holds=holds, cut=cut,
                      transcript_len=length, transcript="x" * length,
                      spec_passes=1)


def test_the_summary_is_four_lines_he_can_paste_back():
    takes = [_take(False, [], 0, True, False),
             _take(False, [("um", 1.9)], 1, False, True, endpoint="energy", gap=2.5),
             _take(True, [("um", 1.9)], 1, False, True),
             _take(True, [("uh", 1.9)], 1, False, True, length=34)]
    lines = probe.summarize(takes, speculate=True)
    assert len(lines) == 4, lines
    assert lines[0].startswith("filler-probe:") and "4 takes" in lines[0]
    assert lines[1].startswith("hint off:") and "1/2" in lines[1] and "holds 1" in lines[1]
    assert lines[2].startswith("hint on :") and "2/2" in lines[2] and "holds 2" in lines[2]
    assert "cut off 1/2" in lines[1] and "cut off 0/2" in lines[2]
    assert "tail undecoded 1/2" in lines[1] and "tail undecoded 0/2" in lines[2]
    assert "vad 3" in lines[3] and "energy 1" in lines[3]
    assert "31 chars" in lines[3]                  # mean of 30,30,30,34
    for line in lines:
        assert "x" * 30 not in line               # never the words


def test_the_summary_survives_no_takes():
    lines = probe.summarize([], speculate=False)
    assert len(lines) == 4 and "0 takes" in lines[0]


# ------------------------------------------------------------ main, with fakes
def test_main_runs_the_planned_takes_and_prints_the_summary(monkeypatch):
    monkeypatch.setattr(CONFIG, "filler_prompt_hint", False)
    monkeypatch.setenv("JARVIS_DEBUG_AUDIO", "1")
    built, said, answers = [], [], [""] * 20

    def build():
        p = _pipe()
        built.append(p)
        return p

    rc = probe.main(["--takes", "2"], build=build,
                    ask=lambda prompt: answers.pop(0), say=said.append,
                    clock=FakeClock(), sleep=lambda s: None)
    assert rc == 0 and len(built) == 1
    assert "JARVIS_DEBUG_AUDIO" not in os.environ
    assert built[0].recorder.starts == 2
    summary = [line for line in said if line.startswith(("filler-probe:", "hint off:",
                                                         "hint on :", "stops:"))]
    assert len(summary) == 4, said
    assert "2 takes (1 hint off, 1 hint on)" in summary[0]
