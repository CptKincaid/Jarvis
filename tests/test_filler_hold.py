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
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)     # decoded through the um
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
    rec.note_partial("set a timer for, um, ten minutes", 0.96, rec.capture_id)
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
    rec.note_partial("set a timer for, um", 1.0, rec.capture_id)       # 0.92 s before the last speech
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
    rec.note_partial("set a timer for, um", 0.96 - 0.5, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1


def test_a_hold_is_counted_once_per_pause_not_per_tick_or_per_redecode(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1
    for _ in range(6):                                 # six more poll ticks
        _push(rec, 1)
        assert rec._check_endpoint() is False
    assert rec._filler_holds == 1
    # the preview re-decodes the SAME pause 0.9 s later and still ends on um
    rec.note_partial("set a timer for, um", rec.endpointer.audio_seconds, rec.capture_id)
    _push(rec, 1)
    assert rec._check_endpoint() is False
    assert rec._filler_holds == 1


def test_the_hold_is_logged_once_with_the_word_and_the_wait(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="jarvis.recorder")
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)
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
        rec.note_partial("set a timer for, um", ep.last_speech_seconds, rec.capture_id)
        _push(rec, 30)                                 # 0.96 s pause: held
        assert rec._check_endpoint() is False and rec.stops == []
        assert rec._filler_holds == pause
    _push(rec, 30)                                     # speech again
    assert rec._check_endpoint() is False
    rec.note_partial("set a timer for, um", ep.last_speech_seconds, rec.capture_id)
    _push(rec, 30)                                     # third pause: the cap
    assert rec._check_endpoint() is True
    assert rec._filler_holds == 2 and rec.stops[0][2] < 1.0


def test_the_hold_is_off_by_config(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch, filler_hold=False)
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._filler_holds == 0


def test_the_hold_never_applies_before_the_endpoint_silence_itself(monkeypatch):
    """The hold lengthens a stop that was about to happen; it is not
    counted while the user is still inside the ordinary 0.8 s."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)
    _push(rec, 10)                                     # 0.32 s gap
    assert rec._check_endpoint() is False and rec._filler_holds == 0


def test_a_new_capture_forgets_the_last_ones_partials_and_holds():
    rec = object.__new__(Recorder)
    rec._latest_partial = ("um", 0, 3.0, 1.0)   # (filler, spell_n, end_s, wall)
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
    rec.note_partial("um", 999.0, rec.capture_id)
    _push(rec, 26)                                     # 0.83 s: the ordinary stop
    assert rec._check_endpoint() is True and rec._filler_holds == 0
    assert rec.stops[0][2] < 1.0


def test_a_partial_a_little_past_the_fed_audio_still_holds(monkeypatch):
    """The bound is a bound, not a trap: the preview snapshots the buffer
    while the poll thread is between feeds, and feed() only consumes whole
    512-sample chunks, so end_s legitimately runs a fraction of a second
    ahead of ep.audio_seconds. One slack either side."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", rec.endpointer.audio_seconds + 0.5, rec.capture_id)
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
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)
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
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1
    _push(rec, 50)                                     # 2.43 s: the hold is spent
    assert rec._check_endpoint() is True and rec._filler_holds == 1
    assert rec.stops[0][2] > 2.3


def test_the_hold_count_is_the_number_of_holds_that_delayed_a_stop(monkeypatch):
    """What holds=N means, pinned in one line so the docs and the ledger
    cannot drift apart again."""
    import jarvis.recorder as rm
    # The rule moved into _counted_hold on 09-05 when the spelling hold
    # joined this seam (jarvis/spelling.py); both holds obey it now.
    doc = (rm.Recorder._counted_hold.__doc__ or "") + \
          (rm.Recorder._hold_extra.__doc__ or "")
    assert "delay" in doc.lower()


# ----------------------------------------- (3c) the 09-05 cross-capture race
#
# BLOCKER B, verdict round 2 finding 1, reproduced at SHIPPED DEFAULTS. The
# preview snapshots capture N and the model is still running when capture N
# stops and capture N+1 starts (start() -> _reset_filler_hold()). The decode
# then lands on the NEW capture carrying the OLD one's words and the OLD
# one's position, and buys a hold on capture N+1's FIRST pause -- measured
# `filler hold 1/3: 'um' at 1.0s, waiting 1.5s` on a capture in which he had
# said no filler at all.
#
# The round-1 bounds do not cover it. They reject a stale span that is far
# from the new capture's last-speech mark (`end_s < last - FILLER_SLACK_S`)
# or past the audio this endpointer has been fed (`end_s > audio_seconds +
# FILLER_SLACK_S`) -- but a SHORT capture N leaves end_s squarely inside
# both, and the preview only decodes from _PARTIAL_MIN_S (0.7 s) up, so
# every short capture lands there.
#
# The obvious one-line fix -- guarding the note on self.recorder.recording,
# the guard the publish two lines below already carries -- is WRONG: it
# reads a flag at a DIFFERENT MOMENT from the snapshot, and it turns design
# item 4 red (a decode that returns after the stop must still report, or a
# stale um keeps holding). The note's identity belongs to the capture its
# AUDIO came from, so the preview stamps each note with the capture id it
# read before snapshotting, and the recorder drops a note from any other.
def test_a_new_capture_gets_a_new_id_before_the_old_note_is_cleared():
    """Order matters: the id must move FIRST. A note landing between the
    two statements reads the new id and is dropped; clearing first would
    leave a window where the old id still matches."""
    rec = object.__new__(Recorder)
    first = rec.capture_id
    rec._reset_filler_hold()
    assert rec.capture_id != first
    src = inspect.getsource(Recorder._reset_filler_hold)
    assert src.index("_capture_id") < src.index("_latest_partial"), src


def test_a_note_stamped_with_a_finished_capture_never_holds(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 1.4, rec.capture_id - 1)
    _push(rec, 26)                                   # 0.83 s: the ordinary stop
    assert rec._check_endpoint() is True and rec._filler_holds == 0
    assert rec.stops[0][2] < 1.0


def test_a_decode_from_the_previous_capture_cannot_hold_the_next_ones_pause(monkeypatch):
    """The race itself: capture 1 was 1.4 s long, its decode lands after
    capture 2's start() has already reset the hold state."""
    rec = _speaks_then_pauses(monkeypatch)
    one = rec.capture_id
    rec._reset_filler_hold()                         # capture 2 start()
    rec.note_partial("set a timer for, um", 1.4, one)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._filler_holds == 0


def test_an_unstamped_note_is_dropped_rather_than_vouched_for(monkeypatch):
    """No "trust me" default. A stamp that is not the open capture's id --
    None included -- is not this capture's, so it is dropped. A default
    meaning "the caller vouches" is how this hole gets reopened by the next
    caller that forgets to stamp."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96, None)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._filler_holds == 0


def test_a_note_from_the_capture_it_was_taken_in_still_holds(monkeypatch):
    """The stamp may only ever DISCARD. The ordinary path is untouched."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("set a timer for, um", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._filler_holds == 1


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
        capture_id = 4
        notes = []

        def snapshot_audio(self):
            return np.zeros(int(SAMPLE_RATE * 2.0), dtype=np.float32)

        def note_partial(self, text, audio_end_s, capture_id=None):
            assert capture_id == 4, capture_id     # stamped with THIS capture
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
        capture_id = 4
        notes = []

        def snapshot_audio(self):
            state["n"] += 1
            if state["n"] > 2:
                Rec.recording = False
            return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

        def note_partial(self, text, audio_end_s, capture_id=None):
            assert capture_id == 4, capture_id
            Rec.notes.append((text, audio_end_s))

    class Tr:
        def partial(self, audio):
            raise RuntimeError("model exploded")

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.5
    pipe._partial_loop()
    assert Rec.notes and all(t == "" for t, _ in Rec.notes), Rec.notes


def test_the_preview_stamps_its_note_with_the_capture_it_snapshotted(monkeypatch):
    """Read BEFORE the snapshot, deliberately: if the capture turns over
    between the read and the snapshot, the note carries the OLD id and is
    dropped. A mismatch may only ever discard a note, never accept a stale
    one -- reading it after would do the opposite."""
    monkeypatch.setattr(events_mod.bus, "publish", lambda ev: None)
    seen = []

    class Rec:
        recording = True
        capture_id = 7

        def snapshot_audio(self):
            Rec.capture_id = 8            # the capture turns over right here
            return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

        def note_partial(self, text, audio_end_s, capture_id=None):
            seen.append(capture_id)
            Rec.recording = False

    class Tr:
        def partial(self, audio):
            return "set a timer for, um"

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S, pipe._PARTIAL_MAX_S = 0.01, 0.5, 1.0
    pipe._partial_loop()
    assert seen == [7], seen


def test_the_probe_stamps_its_note_too():
    """scripts/filler_probe.py runs the same loop by hand; an unstamped
    note there would reopen the hole in the one instrument he runs."""
    src = inspect.getsource(__import__("scripts.filler_probe",
                                       fromlist=["run_take"]).run_take)
    assert "note_partial(text, end_s, capture)" in src, src


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


# ------------------------------------- (6) the hint and the prompt-echo gate
#
# jarvis-v3 (fix-prompt-echo ecc3666) hoisted `prompt = self._prompt()` to
# ONE call per preview pass and feeds that same string to a new echo gate,
# _preview_text(text, prompt), which blanks a preview that is just the
# prompt read back. This branch rewrote the same two lines to
# `initial_prompt=self._partial_prompt()`. Keeping both sides literally --
# the resolution the conflict invites, because the hunks look independent
# -- costs a second prompt build per preview AND makes the hint invisible
# to the gate that most needs it: "Um, uh, hmm, er." is a comma-separated
# list, exactly the shape fix-prompt-echo's own comment names as what a
# greedy decoder runs away on. Measured 09-05 (verdict round 1).
def test_the_preview_builds_its_prompt_once_even_with_the_hint_on(firewall, monkeypatch):
    """The provider is jarvis.vocab.build_prompt: a calendar cache and the
    people store, several times a second while he is talking. One build
    per pass, hint or no hint."""
    monkeypatch.setattr(tr_mod.CONFIG, "filler_prompt_hint", True)
    calls = []

    def provider():
        calls.append(1)
        return "Peyrovi, BIOSENSORS"

    tr = _gpu(provider)
    tr.partial(AUDIO)
    assert calls == [1]


@pytest.mark.parametrize("make", [_gpu, _cpu], ids=["gpu", "cpu"])
def test_an_echoed_hint_is_blanked_by_the_preview_gate(firewall, monkeypatch, make):
    """The decoder is given the hint, so the gate must judge against the
    HINTED prompt. Measured under the naive resolution: partial() returned
    'Um, uh, hmm, er. Um, uh, hmm, er. Um, uh, hmm, er.' verbatim onto the
    ghost card -- and trailing_filler of that is 'er', so note_partial then
    bought a 1.5 s hold on a decoder runaway."""
    monkeypatch.setattr(tr_mod.CONFIG, "filler_prompt_hint", True)
    echo = "Um, uh, hmm, er. Um, uh, hmm, er. Um, uh, hmm, er."
    tr = make(lambda: "Peyrovi, BIOSENSORS")

    def transcribe(audio, **kw):
        tr._model.prompts.append(kw.get("initial_prompt"))
        if tr._gpu:
            return {"segments": [{"text": echo}], "language": "en", "text": echo}
        return iter([SimpleNamespace(text=echo, avg_logprob=-0.3)]), \
            SimpleNamespace(language="en")

    tr._model.transcribe = transcribe
    assert tr.partial(AUDIO) == ""
    assert tr._model.prompts == ["Peyrovi, BIOSENSORS Um, uh, hmm, er."]


@pytest.mark.parametrize("make", [_gpu, _cpu], ids=["gpu", "cpu"])
def test_the_echo_gate_is_handed_the_very_string_the_decoder_was_given(
        firewall, monkeypatch, make):
    """The DIRECT pin on the merge resolution, one assertion deep rather
    than through a behaviour: the hint text must be INSIDE what the echo
    gate sees, and what it sees must be the identical string the model was
    given. The naive resolution -- keeping both conflict sides literally --
    hands `self._partial_prompt()` to initial_prompt and `self._prompt()`
    to _preview_text, so this fails on both halves at once: the hint is
    absent from the gate's prompt AND the two strings differ. Measured
    red on that resolution 09-05 (see the merge commit)."""
    monkeypatch.setattr(tr_mod.CONFIG, "filler_prompt_hint", True)
    seen = []

    def spy(text, prompt):
        seen.append(prompt)
        return text

    monkeypatch.setattr(Transcriber, "_preview_text", staticmethod(spy))
    tr = make(lambda: "Peyrovi, BIOSENSORS")
    tr.partial(AUDIO)

    assert len(seen) == 1, seen
    assert tr_mod.FILLER_PROMPT_HINT in seen[0], seen[0]
    # ...and it is the SAME string, not a second build that merely matches.
    assert seen == tr._model.prompts, (seen, tr._model.prompts)


def test_the_hinted_prompt_does_not_blank_a_real_command(firewall, monkeypatch):
    """The gate is judged against a LONGER prompt with the hint on, so
    check it costs nothing: none of his ordinary commands -- including one
    with a real um in it -- is caught as an echo of the hinted prompt."""
    hinted = "Peyrovi, BIOSENSORS " + tr_mod.FILLER_PROMPT_HINT
    caught = {said: tr_mod.prompt_echo(said, hinted)
              for said in ("set a timer for ten minutes",
                           "set a timer for, um, ten minutes",
                           "um, what is the weather",
                           "stop stop stop",
                           "turn it up turn it up turn it up",
                           "yes")
              if tr_mod.prompt_echo(said, hinted)[1]}
    assert caught == {}
    # ...while a pure stutter still is, and only in the preview.
    assert tr_mod.prompt_echo("um um um", hinted)[1] >= 2


# --------------------------- (6b) the hint runaway UNDER three repeats
#
# BLOCKER C, verdict round 2 FINDING 2. prompt_echo will not call a preview
# an echo until a unit repeats ECHO_MIN_REPEATS (3) times, and that floor is
# right for HIS words: "yes yes" is a man being emphatic, not a decoder
# looping. The hint is NOT his words. "Um, uh, hmm, er." is a four-word
# string this module injects into the PREVIEW's prompt so whisper will write
# his fillers down -- so read back even ONCE it is a runaway. At k=1 and k=2
# prompt_echo passes it: four words he never said land on the ghost card,
# and trailing_filler of them is "er", so note_partial buys a 1.5 s hold on
# a decoder runaway. Closing it must not touch ECHO_MIN_REPEATS (that would
# blank his real insistence) and must not touch the hold (a genuine
# trailing um must still hold) -- so the gate is narrowed to exactly what
# this module itself put in the prompt.
def test_the_repeat_floor_is_why_a_short_hint_runaway_got_through():
    """The underlying fact, pinned so the fix is not mistaken for a
    prompt_echo bug: at one and two repeats prompt_echo says 'not an echo',
    and it is RIGHT to -- three is its floor for everything else."""
    hint = tr_mod.FILLER_PROMPT_HINT
    prompt = "Peyrovi, BIOSENSORS " + hint
    assert tr_mod.ECHO_MIN_REPEATS == 3
    assert tr_mod.prompt_echo(hint, prompt)[1] == 0
    assert tr_mod.prompt_echo(f"{hint} {hint}", prompt)[1] == 0
    assert tr_mod.prompt_echo(f"{hint} {hint} {hint}", prompt)[1] >= 3


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_a_hint_runaway_is_blanked_at_every_repeat_count(k):
    hint = tr_mod.FILLER_PROMPT_HINT
    prompt = "Peyrovi, BIOSENSORS " + hint
    assert tr_mod.Transcriber._preview_text(" ".join([hint] * k), prompt) == ""


def test_a_hint_runaway_cut_mid_pass_is_blanked_too():
    """sample_len stops the decode anywhere: one whole hint plus part of
    the next, including a word cut in half."""
    hint = tr_mod.FILLER_PROMPT_HINT
    prompt = "Peyrovi, BIOSENSORS " + hint
    for text in (f"{hint} Um,", f"{hint} Um, uh,", f"{hint} Um, uh, hmm",
                 f"{hint} {hint} Um, u"):
        assert tr_mod.Transcriber._preview_text(text, prompt) == "", text


def test_a_real_trailing_um_still_reaches_the_card_and_the_hold():
    """The hold's whole input. Narrowing the gate may not cost this."""
    hint = tr_mod.FILLER_PROMPT_HINT
    prompt = "Peyrovi, BIOSENSORS " + hint
    for text in ("set a timer for, um", "um", "uh", "um, what is the weather",
                 "set a timer for ten minutes"):
        assert tr_mod.Transcriber._preview_text(text, prompt) == text, text
    assert ep_mod.trailing_filler("set a timer for, um") == "um"


def test_the_runaway_gate_only_removes_what_this_module_put_in_the_prompt():
    """No hint in the prompt the model was given -> this gate says nothing
    and prompt_echo's ordinary rules decide. Same principle as the merge:
    judge the text against the prompt the model ACTUALLY had."""
    hint = tr_mod.FILLER_PROMPT_HINT
    plain = "Peyrovi, BIOSENSORS"
    assert tr_mod.Transcriber._preview_text(hint, plain) == hint
    assert tr_mod.hint_echo(hint, plain) == 0
    assert tr_mod.hint_echo(f"{hint} {hint}", plain + " " + hint) == 2


@pytest.mark.parametrize("make", [_gpu, _cpu], ids=["gpu", "cpu"])
@pytest.mark.parametrize("k", [1, 2])
def test_a_short_hint_runaway_buys_no_hold(firewall, monkeypatch, make, k):
    """End to end: the decoder runs away twice, partial() reports "", and
    the recorder's newest partial has no trailing filler -- so the capture
    ends at the ordinary endpoint instead of waiting 1.5 s on words he
    never said."""
    monkeypatch.setattr(tr_mod.CONFIG, "filler_prompt_hint", True)
    runaway = " ".join([tr_mod.FILLER_PROMPT_HINT] * k)
    tr = make(lambda: "Peyrovi, BIOSENSORS")

    def transcribe(audio, **kw):
        tr._model.prompts.append(kw.get("initial_prompt"))
        if tr._gpu:
            return {"segments": [{"text": runaway}], "language": "en",
                    "text": runaway}
        return iter([SimpleNamespace(text=runaway, avg_logprob=-0.3)]), \
            SimpleNamespace(language="en")

    tr._model.transcribe = transcribe
    shown = tr.partial(AUDIO)
    assert shown == "", shown
    assert ep_mod.trailing_filler(shown) == ""

    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial(shown, 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._filler_holds == 0
    assert rec.stops[0][2] < 1.0


# ------------------------------------------------ (7) the one invariant
@pytest.mark.parametrize("end_s", [0.0, 0.5, 0.96, 1.2, 5.0, 999.0])
@pytest.mark.parametrize("text", ["set a timer for, um", "set a timer for uh",
                                  "set a timer for ten minutes", "", "...",
                                  "Um, uh, hmm, er. Um, uh, hmm, er."])
def test_no_partial_of_any_shape_ends_a_capture_before_the_ordinary_endpoint(
        monkeypatch, text, end_s):
    """The whole feature rests on one invariant: whatever the preview
    reports -- a stale span, a span from another capture, a decoder
    runaway, nothing at all -- the capture never ends SOONER than the
    0.8 s it would have ended at with the hold switched off. A wrong hold
    costs him one filler_hold_s once; a wrong stop cuts him off
    mid-sentence. Both 09-05 guards only ever REMOVE an unearned hold, so
    this is the test that says they cannot have gone the other way.

    WHAT THIS TEST ACTUALLY GUARDS, measured by mutation on 09-05 rather
    than assumed. Making _filler_hold_extra return -0.5 does NOT turn it
    red: _check_endpoint returns early on `gap < CONFIG.endpoint_silence`
    before the filler code is reached, so that early return -- not the
    hold arithmetic -- is the floor. It goes red (`assert 0.32 >= 0.8`)
    only when BOTH floors are moved, which is exactly the refactor this
    test exists to catch: someone hoisting the filler check above the
    ordinary endpoint.

    Bounded by an iteration count, never a condition: an unbounded push
    loop against this harness is what OOM-killed the box on 09-05 (only
    _check_endpoint feeds the endpointer, so audio_seconds cannot move
    inside a loop that merely appends frames)."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial(text, end_s, rec.capture_id)
    for _ in range(200):
        _push(rec, 5)
        if rec._check_endpoint():
            break
    assert rec.stops, "the capture never ended in 200 poll ticks"
    (reason, endpoint, dead_air), = rec.stops
    assert (reason, endpoint) == ("silence", "vad")
    assert dead_air >= recorder_mod.CONFIG.endpoint_silence
