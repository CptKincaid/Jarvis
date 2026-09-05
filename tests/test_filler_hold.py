"""The filler hold: "um" and "uh" buy the user more time (Hunter, 08-31,
approved again 09-04).

The recorder stops CONFIG.endpoint_silence (0.8 s) after the VAD last
heard speech. A thinking pause after "set a timer for, um..." is longer
than that, so the capture ended mid-sentence. Now the live preview's
newest decode is reported to the recorder (Recorder.note_partial); when
its last word is a filler AND the VAD heard nothing after that span, the
capture waits CONFIG.filler_hold_s longer, at most CONFIG.filler_max_holds
distinct pauses per capture.

Everything here runs on synthetic arrays and scripted fakes: no device is
opened, no audio is played, nothing is saved (tests/conftest.py's firewall
covers the rest).
"""
from __future__ import annotations

import jarvis.endpoint as ep_mod


# ------------------------------------------------------------ (1) one list
def test_the_commander_reads_the_endpoint_filler_list():
    """One list, not two: commander.py used to own a copy nothing read."""
    import jarvis.commander as commander_mod
    assert commander_mod.FILLER_WORDS is ep_mod.FILLER_WORDS


def test_the_filler_list_is_pure_fillers_only():
    assert {"um", "uh", "hmm", "er"} <= set(ep_mod.FILLER_WORDS)
    # never common words: "like", "so", "well" end real sentences
    assert not {"like", "so", "well", "okay"} & set(ep_mod.FILLER_WORDS)


def test_trailing_filler_finds_a_final_um_through_punctuation_and_case():
    tf = ep_mod.trailing_filler
    assert tf("set a timer for um") == "um"
    assert tf("set a timer for, um,") == "um"
    assert tf("set a timer for uh...") == "uh"
    assert tf("Hmm?") == "hmm"
    assert tf("what is, Umm") == "umm"


def test_trailing_filler_skips_a_trailing_token_that_is_only_punctuation():
    """whisper writes 'for um ...' with the dots as their own token."""
    assert ep_mod.trailing_filler("set a timer for um ...") == "um"
    assert ep_mod.trailing_filler("set a timer for, um -") == "um"


def test_trailing_filler_is_empty_when_the_last_word_is_not_a_filler():
    tf = ep_mod.trailing_filler
    assert tf("set a timer for ten minutes") == ""
    assert tf("set a timer for, um, ten minutes") == ""     # mid-sentence um
    assert tf("uh-huh") == ""                                # an answer, not a filler
    assert tf("hummus") == ""
    assert tf("") == ""
    assert tf("...") == ""
    assert tf(None) == ""


# ------------------------------------------------------------ (2) config
def test_config_ships_the_filler_hold_settings_with_the_agreed_defaults():
    from jarvis.config import Config
    cfg = Config()
    assert cfg.filler_hold is True
    assert cfg.filler_hold_s == 1.5
    assert cfg.filler_max_holds == 3
    # UNMEASURED, so it ships OFF: scripts/filler_probe.py decides.
    assert cfg.filler_prompt_hint is False


def test_the_filler_settings_load_from_the_settings_file(tmp_path, monkeypatch):
    import json
    from jarvis import config
    path = tmp_path / "settings.json"
    monkeypatch.setattr(config.PATHS, "SETTINGS_FILE", path)
    path.write_text(json.dumps({"filler_hold_s": 2, "filler_prompt_hint": True,
                                "filler_max_holds": 1, "filler_hold": False}))
    cfg = config.Config.load()
    assert cfg.filler_hold_s == 2.0 and isinstance(cfg.filler_hold_s, float)
    assert cfg.filler_prompt_hint is True
    assert cfg.filler_max_holds == 1
    assert cfg.filler_hold is False


# ------------------------------------------------------------ (3) recorder
import inspect          # noqa: E402
import logging          # noqa: E402
import time             # noqa: E402

import numpy as np      # noqa: E402

import jarvis.events as events_mod                 # noqa: E402
import jarvis.recorder as recorder_mod             # noqa: E402
from jarvis.endpoint import CHUNK, SAMPLE_RATE, VoiceEndpointer   # noqa: E402
from jarvis.events import RecordingStopped         # noqa: E402
from jarvis.recorder import MicArbiter, Recorder   # noqa: E402


class ScriptedVAD:
    """probability per 512-sample chunk, from a list; last value repeats."""

    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def __call__(self, chunk):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p

    def reset(self):
        self.i = 0


def _chunks(n):
    return np.zeros(n * CHUNK, dtype=np.float32)


def _recorder(endpointer):
    """A recorder mid-capture, as tests/test_endpoint.py builds one: bare
    object, scripted stop, no device anywhere."""
    rec = object.__new__(Recorder)
    rec.recording = True
    rec.endpointer = endpointer
    rec._ep_cursor = 0
    rec._record_rate = SAMPLE_RATE
    rec._record_start_time = time.monotonic() - 10.0   # well past the grace
    rec._audio_frames = []
    rec._stop_endpoint, rec._stop_dead_air = "", None
    rec._voice_stopped = False
    rec._followup = False
    rec.stops = []

    def stop(reason="manual", endpoint="", dead_air=None):
        rec.stops.append((reason, endpoint, dead_air))
        rec.recording = False
    rec.stop = stop
    return rec


def _push(rec, n_chunks):
    rec._audio_frames.append(_chunks(n_chunks).reshape(-1, 1))


def _hold_config(monkeypatch, **over):
    cfg = dict(endpoint_vad=True, endpoint_silence=0.8, silence_grace=0.0,
               filler_hold=True, filler_hold_s=1.5, filler_max_holds=3)
    cfg.update(over)
    for k, v in cfg.items():
        monkeypatch.setattr(recorder_mod.CONFIG, k, v)


def _speaks_then_pauses(monkeypatch, **over):
    """0.96 s of speech (30 chunks) followed by silence, checked once so the
    VAD has heard the speech; the caller then reports a partial and pushes
    the pause."""
    _hold_config(monkeypatch, **over)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.9] * 30 + [0.0] * 400)))
    _push(rec, 30)
    assert rec._check_endpoint() is False
    assert abs(rec.endpointer.last_speech_seconds - 0.96) < 1e-9
    return rec


def test_a_trailing_filler_in_the_newest_partial_holds_the_stop(monkeypatch):
    """"set a timer for, um" then a thinking pause: the 0.8 s endpoint
    would have cut him off; the hold waits filler_hold_s longer."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96)     # decoded through the um
    _push(rec, 26)                                     # 0.83 s gap: the old stop
    assert rec._check_endpoint() is False and rec.stops == []
    assert rec._filler_holds == 1
    _push(rec, 40)                                     # 2.11 s: inside 0.8 + 1.5
    assert rec._check_endpoint() is False and rec.stops == []
    _push(rec, 7)                                      # 2.34 s: the hold is spent
    assert rec._check_endpoint() is True
    (reason, endpoint, dead_air), = rec.stops
    assert (reason, endpoint) == ("silence", "vad") and dead_air >= 2.3


def test_a_filler_mid_sentence_does_not_hold(monkeypatch):
    """"for, um, ten minutes": the um was not the last word."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um, ten minutes", 0.96)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._filler_holds == 0
    assert rec.stops[0][2] < 1.0


def test_a_partial_older_than_the_last_speech_by_more_than_the_slack_does_not_hold(monkeypatch):
    """The preview decoded "for, um" and then the VAD heard MORE speech
    (0.92 s of it after that span): whatever followed the um was never
    decoded, so nothing here can claim the um was the last word. The
    recorder does not decode the tail itself -- that would add a decode to
    every turn -- it stops as before. This is the stated limit."""
    _hold_config(monkeypatch)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.9] * 60 + [0.0] * 400)))
    _push(rec, 60)                                     # 1.92 s of speech
    assert rec._check_endpoint() is False
    rec.note_partial("set a timer for, um", 1.0)       # 0.92 s before the last speech
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._filler_holds == 0


def test_a_partial_within_the_slack_of_the_last_speech_counts_as_last(monkeypatch):
    """FILLER_SLACK_S: the span may end inside the um's own tail (whisper
    writes "um" from its first half), the VAD's hysteresis hangs on a chunk
    or two after the voice fades and the resampler lags 10 ms -- so a
    partial that ends up to 0.6 s before the last speech chunk still
    counts. A wrong hold costs one hold_s once; a missed one cuts him off."""
    rec = _speaks_then_pauses(monkeypatch)
    assert recorder_mod.FILLER_SLACK_S == 0.6
    rec.note_partial("set a timer for, um", 0.96 - 0.5)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1


def test_a_hold_is_counted_once_per_pause_not_per_tick_or_per_redecode(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1
    for _ in range(6):                                 # six more poll ticks
        _push(rec, 1)
        assert rec._check_endpoint() is False
    assert rec._filler_holds == 1
    # the preview re-decodes the SAME pause 0.9 s later and still ends on um
    rec.note_partial("set a timer for, um", rec.endpointer.audio_seconds)
    _push(rec, 1)
    assert rec._check_endpoint() is False
    assert rec._filler_holds == 1


def test_the_hold_is_logged_once_with_the_word_and_the_wait(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="jarvis.recorder")
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96)
    _push(rec, 26)
    rec._check_endpoint()
    for _ in range(4):
        _push(rec, 1)
        rec._check_endpoint()
    lines = [r.getMessage() for r in caplog.records if "filler hold" in r.getMessage()]
    assert lines == ["filler hold 1/3: 'um' at 1.0s, waiting 1.5s"], lines


def test_at_most_filler_max_holds_distinct_pauses_per_capture(monkeypatch):
    """Three pauses each ending on um; only the first two are held."""
    _hold_config(monkeypatch, filler_max_holds=2)
    script = ([0.9] * 30 + [0.0] * 30) * 2 + [0.9] * 30 + [0.0] * 400
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD(script)))
    ep = rec.endpointer
    for pause in (1, 2):
        _push(rec, 30)                                 # speech
        assert rec._check_endpoint() is False
        rec.note_partial("set a timer for, um", ep.last_speech_seconds)
        _push(rec, 30)                                 # 0.96 s pause: held
        assert rec._check_endpoint() is False and rec.stops == []
        assert rec._filler_holds == pause
    _push(rec, 30)                                     # speech again
    assert rec._check_endpoint() is False
    rec.note_partial("set a timer for, um", ep.last_speech_seconds)
    _push(rec, 30)                                     # third pause: the cap
    assert rec._check_endpoint() is True
    assert rec._filler_holds == 2 and rec.stops[0][2] < 1.0


def test_the_hold_is_off_by_config(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch, filler_hold=False)
    rec.note_partial("set a timer for, um", 0.96)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._filler_holds == 0


def test_the_hold_never_applies_before_the_endpoint_silence_itself(monkeypatch):
    """The hold lengthens a stop that was about to happen; it is not
    counted while the user is still inside the ordinary 0.8 s."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96)
    _push(rec, 10)                                     # 0.32 s gap
    assert rec._check_endpoint() is False and rec._filler_holds == 0


def test_a_new_capture_forgets_the_last_ones_partials_and_holds():
    rec = object.__new__(Recorder)
    rec._latest_partial = ("um", 3.0, 1.0)
    rec._filler_holds, rec._filler_hold_key = 2, 3.0
    rec._reset_filler_hold()
    assert rec._latest_partial is None
    assert rec._filler_holds == 0 and rec._filler_hold_key is None
    # ...and start() is where that reset lives, with the other per-capture state
    assert "self._reset_filler_hold()" in inspect.getsource(Recorder.start)


def test_the_stop_event_carries_the_hold_count(monkeypatch):
    """RecordingStopped.filler_holds, so a week of turns can say how often
    the hold fired without anyone reading a transcript."""
    assert RecordingStopped().filler_holds == 0
    monkeypatch.setattr(recorder_mod.CONFIG, "sound", False)
    rec = Recorder(MicArbiter())          # built first: its constructor publishes MicState
    published = []
    monkeypatch.setattr(events_mod.bus, "publish", published.append)
    rec.recording = True
    rec._audio_frames = [np.zeros((SAMPLE_RATE, 1), dtype=np.float32)]
    rec._filler_holds = 2
    rec.stop(reason="silence", endpoint="vad", dead_air=2.3)
    ev = [e for e in published if isinstance(e, RecordingStopped)][-1]
    assert ev.filler_holds == 2 and ev.endpoint == "vad" and ev.dead_air_s == 2.3


def test_the_turn_ledger_line_carries_the_hold_count():
    """`turn: ... (stop=vad holds=1)` in the log and turns.jsonl -- and a
    turn with no hold stays exactly as it was."""
    import jarvis.app as app_mod
    from jarvis.turnclock import TurnLedger
    emitted = []
    led = TurnLedger(clock=lambda: 99.0, emit=emitted.append)
    a = object.__new__(app_mod.JarvisApp)
    a.turns = led

    def turn(holds):
        emitted.clear()
        led.mark("wake", at=0.0)
        led.mark("mic", at=0.1)
        a._turn_on_stop(RecordingStopped(reason="silence", endpoint="vad",
                                         dead_air_s=2.3, t=5.0, filler_holds=holds))
        led.mark("stt", at=5.4)
        led.mark("audio", at=6.0)
        (rec,) = emitted
        return rec

    rec = turn(1)
    assert rec["holds"] == "1" and rec["stop"] == "vad"
    assert "holds=1" in TurnLedger.format(rec)
    rec = turn(0)
    assert "holds" not in rec and "holds" not in TurnLedger.format(rec)


# ------------------------------------------------ (3b) the two 09-05 holes
#
# Both were MEASURED by the round-1 verdict against this branch, both are
# one-sided-guard bugs in _filler_hold_extra, and neither can cut him off
# sooner than today -- they corrupt the ledger and accept a partial the
# endpointer could not have heard.
def test_a_partial_from_the_future_never_buys_a_hold(monkeypatch):
    """MEASURED 09-05: note_partial("um", 999.0) against last_speech 0.96
    fired a hold. The guard `end_s < last - FILLER_SLACK_S` had no upper
    bound, so a preview decode that lands after a stop -- carrying the
    OLD capture's position -- claimed the um was the last thing heard in
    a pause it never covered. A span may not reach past the audio the
    endpointer has actually been fed (plus the same slack)."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("um", 999.0)
    _push(rec, 26)                                     # 0.83 s: the ordinary stop
    assert rec._check_endpoint() is True and rec._filler_holds == 0
    assert rec.stops[0][2] < 1.0


def test_a_partial_a_little_past_the_fed_audio_still_holds(monkeypatch):
    """The bound is a bound, not a trap: the preview snapshots the buffer
    while the poll thread is between feeds, and feed() only consumes whole
    512-sample chunks, so end_s legitimately runs a fraction of a second
    ahead of ep.audio_seconds. One slack either side."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", rec.endpointer.audio_seconds + 0.5)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1


def test_a_hold_that_delays_nothing_is_not_counted_and_not_logged(monkeypatch, caplog):
    """MEASURED 09-05: one starved poll tick arriving with 3.84 s of
    silence already banked stopped on that very tick (the gap is past
    0.8 + 1.5 = 2.3 s) and still recorded filler_holds == 1. holds=N is
    the number the design added so a week of turns can say how often the
    hold fired -- an increment that delayed nothing makes it an upper
    bound instead of a count. Count and log only when the hold changes
    the outcome."""
    caplog.set_level(logging.INFO, logger="jarvis.recorder")
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96)
    _push(rec, 120)                                    # 3.84 s in one tick
    assert rec._check_endpoint() is True
    (reason, endpoint, dead_air), = rec.stops
    assert (reason, endpoint) == ("silence", "vad") and dead_air > 2.3
    assert rec._filler_holds == 0
    assert [r.getMessage() for r in caplog.records if "filler hold" in r.getMessage()] == []


def test_a_hold_that_did_delay_the_stop_is_still_counted_when_it_expires(monkeypatch):
    """The other side of the same rule: a hold counted on the tick it
    delayed stays counted on the tick it expires. holds=1 must survive
    the stop it lengthened, or the ledger under-counts instead."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1
    _push(rec, 50)                                     # 2.43 s: the hold is spent
    assert rec._check_endpoint() is True and rec._filler_holds == 1
    assert rec.stops[0][2] > 2.3


def test_the_hold_count_is_the_number_of_holds_that_delayed_a_stop(monkeypatch):
    """What holds=N means, pinned in one line so the docs and the ledger
    cannot drift apart again."""
    import jarvis.recorder as rm
    doc = rm.Recorder._filler_hold_extra.__doc__ or ""
    assert "delay" in doc.lower()


# ------------------------------------------------------------ (4) partial loop
def _pipeline_class():
    import jarvis.app as app_mod
    for name in dir(app_mod):
        obj = getattr(app_mod, name)
        if isinstance(obj, type) and hasattr(obj, "_partial_loop"):
            return obj
    raise AssertionError("no class owns _partial_loop")


def test_the_partial_loop_reports_every_decode_to_the_recorder_with_the_buffer_end(monkeypatch):
    """Every decode, not only a changed one (a hold must survive whisper
    returning the same "…um" twice), stamped with where the decoded span
    ENDS in capture seconds -- the whole buffer's length, even though only
    the newest _PARTIAL_MAX_S of it was decoded."""
    monkeypatch.setattr(events_mod.bus, "publish", lambda ev: None)
    texts = ["set a timer for", "set a timer for", "set a timer for, um"]
    state = {"i": 0}

    class Rec:
        recording = True
        notes = []

        def snapshot_audio(self):
            return np.zeros(int(SAMPLE_RATE * 2.0), dtype=np.float32)

        def note_partial(self, text, audio_end_s):
            Rec.notes.append((text, audio_end_s))

    class Tr:
        def partial(self, audio):
            assert len(audio) == SAMPLE_RATE, "only the newest 1.0 s is decoded"
            i = min(state["i"], len(texts) - 1)
            state["i"] += 1
            if state["i"] >= len(texts):
                Rec.recording = False
            return texts[i]

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S, pipe._PARTIAL_MAX_S = 0.01, 0.5, 1.0
    pipe._partial_loop()
    assert Rec.notes == [(t, 2.0) for t in texts], Rec.notes


def test_the_partial_loop_reports_a_failed_decode_as_no_filler(monkeypatch):
    """A decode that raised tells the recorder nothing ended on a filler;
    it is not allowed to leave a stale um holding the mic open."""
    monkeypatch.setattr(events_mod.bus, "publish", lambda ev: None)
    state = {"n": 0}

    class Rec:
        recording = True
        notes = []

        def snapshot_audio(self):
            state["n"] += 1
            if state["n"] > 2:
                Rec.recording = False
            return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

        def note_partial(self, text, audio_end_s):
            Rec.notes.append((text, audio_end_s))

    class Tr:
        def partial(self, audio):
            raise RuntimeError("model exploded")

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.5
    pipe._partial_loop()
    assert Rec.notes and all(t == "" for t, _ in Rec.notes), Rec.notes


def test_the_partial_loop_docstring_states_the_cadence_limit():
    src = _pipeline_class()._partial_loop.__doc__ or ""
    assert "filler" in src.lower() and "cadence" in src.lower()


# ------------------------------------------------------------ (5) prompt hint
import sys                                     # noqa: E402
from types import SimpleNamespace              # noqa: E402

import pytest                                  # noqa: E402

import jarvis.transcriber as tr_mod            # noqa: E402
from jarvis.config import PATHS                # noqa: E402
from jarvis.transcriber import Transcriber     # noqa: E402

AUDIO = np.zeros(1600, dtype="float32")


class FakeGpuWhisper:
    def __init__(self):
        self.prompts = []

    def transcribe(self, audio, **kw):
        self.prompts.append(kw.get("initial_prompt"))
        return {"segments": [{"text": "hello", "avg_logprob": -0.3}],
                "language": "en", "text": "hello"}


class FakeFasterWhisper:
    def __init__(self):
        self.prompts = []

    def transcribe(self, audio, **kw):
        self.prompts.append(kw.get("initial_prompt"))
        seg = SimpleNamespace(text="hello", avg_logprob=-0.3)
        return iter([seg]), SimpleNamespace(language="en")


@pytest.fixture
def firewall(tmp_path, monkeypatch):
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None)))
    return tmp_path


def _gpu(provider):
    tr = Transcriber(prompt_provider=provider)
    tr._model, tr._gpu, tr._backend = FakeGpuWhisper(), True, "GPU fp16"
    return tr


def _cpu(provider):
    tr = Transcriber(prompt_provider=provider)
    tr._model, tr._gpu, tr._backend = FakeFasterWhisper(), False, "CPU int8"
    return tr


@pytest.mark.parametrize("make", [_gpu, _cpu], ids=["gpu", "cpu"])
def test_the_hint_reaches_the_preview_prompt_and_never_the_final_one(firewall, monkeypatch, make):
    monkeypatch.setattr(tr_mod.CONFIG, "filler_prompt_hint", True)
    tr = make(lambda: "Peyrovi, BIOSENSORS")
    assert tr.partial(AUDIO) == "hello"
    tr.transcribe(AUDIO)
    assert tr._model.prompts == ["Peyrovi, BIOSENSORS Um, uh, hmm, er.",
                                 "Peyrovi, BIOSENSORS"], tr._model.prompts


@pytest.mark.parametrize("make", [_gpu, _cpu], ids=["gpu", "cpu"])
def test_the_hint_is_absent_from_both_passes_when_off(firewall, monkeypatch, make):
    monkeypatch.setattr(tr_mod.CONFIG, "filler_prompt_hint", False)
    tr = make(lambda: "Peyrovi, BIOSENSORS")
    tr.partial(AUDIO)
    tr.transcribe(AUDIO)
    assert tr._model.prompts == ["Peyrovi, BIOSENSORS", "Peyrovi, BIOSENSORS"]


def test_the_hint_names_only_fillers_the_endpoint_list_knows():
    words = {w.strip(".,").lower() for w in tr_mod.FILLER_PROMPT_HINT.split()}
    assert words and words <= set(ep_mod.FILLER_WORDS), words
