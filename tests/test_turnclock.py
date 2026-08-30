"""The per-turn timing ledger (jarvis/turnclock.py).

Every latency claim so far came from subtracting log timestamps by hand.
These pin the arithmetic so the "turn:" line can be trusted: which marks
open a turn, what each duration means, and that off-turn marks (typed
commands, alarms, the speak queue) never produce a report.
"""
from jarvis.turnclock import STAGES, TurnLedger


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt
        return self.t


def _ledger():
    clock, out = Clock(), []
    return TurnLedger(clock=clock, emit=out.append), clock, out


def test_a_full_turn_reports_once_with_every_span():
    led, clock, out = _ledger()
    led.mark("wake")
    clock.tick(0.06)
    led.mark("mic")
    clock.tick(1.9)                                  # the user talks
    speech_end = clock.t
    clock.tick(0.8)
    led.mark("speech_end", at=speech_end)
    led.mark("stop", stop="vad")
    clock.tick(0.37)
    led.mark("stt")
    clock.tick(0.01)
    led.mark("handle")
    clock.tick(1.2)
    led.mark("audio")
    clock.tick(0.1)
    led.mark("audio")  # amplitude ticks: ignored
    assert len(out) == 1
    r = out[0]
    assert r["outcome"] == "audio" and r["stop"] == "vad"
    assert abs(r["wake_to_mic"] - 0.06) < 1e-9
    assert abs(r["speech"] - 1.9) < 1e-9
    assert abs(r["dead_air"] - 0.8) < 1e-9
    assert abs(r["stt"] - 0.37) < 1e-9
    assert abs(r["route"] - 1.2) < 1e-9
    # what the user feels: last word -> first audio
    assert abs(r["wait"] - (0.8 + 0.37 + 0.01 + 1.2)) < 1e-9
    assert not led.open


def test_the_line_reads_like_the_docstring():
    led, clock, out = _ledger()
    led.mark("wake")
    clock.tick(0.061)
    led.mark("mic")
    clock.tick(2.0)
    led.mark("speech_end", at=clock.t)
    clock.tick(0.8)
    led.mark("stop", stop="vad")
    clock.tick(0.37)
    led.mark("stt")
    led.mark("handle")
    clock.tick(1.2)
    led.mark("audio")
    line = TurnLedger.format(out[0])
    assert line.startswith("turn: wake→mic 61ms · speech 2.00s · dead-air 800ms · stt 370ms · route 1.20s · wait 2.37s")
    assert "stop=vad" in line


def test_marks_outside_a_turn_are_ignored():
    """Typed commands, alarms and the speak queue all reach the TTS; none of
    them is a voice turn and none may produce a report."""
    led, clock, out = _ledger()
    led.mark("handle")
    led.mark("audio")
    led.mark("stt")
    assert out == [] and not led.open


def test_a_button_press_opens_a_turn_without_a_wake():
    led, clock, out = _ledger()
    led.mark("mic")                                   # the UI mic button
    clock.tick(1.0)
    led.mark("stop", stop="manual")
    clock.tick(0.3)
    led.mark("stt")
    led.mark("handle")
    clock.tick(0.5)
    led.mark("audio")
    assert len(out) == 1
    assert out[0]["wake_to_mic"] is None and out[0]["total"] is None
    assert abs(out[0]["wait"] - 0.8) < 1e-9          # speech_end falls back to stop


def test_a_rejected_clip_reports_as_such_and_closes():
    led, clock, out = _ledger()
    led.mark("wake")
    led.mark("mic")
    clock.tick(2.0)
    led.mark("stop", stop="energy")
    clock.tick(0.3)
    led.mark("stt")
    led.abandon("rejected:speaker")
    assert out[0]["outcome"] == "rejected:speaker" and out[0]["wait"] is None
    assert TurnLedger.format(out[0]).startswith("turn[rejected:speaker]:")
    assert not led.open
    led.mark("audio")                                 # nothing follows
    assert len(out) == 1


def test_a_new_wake_supersedes_an_unfinished_turn():
    led, clock, out = _ledger()
    led.mark("wake")
    led.mark("mic")
    clock.tick(5.0)
    led.mark("wake")  # user gave up, tried again
    assert len(out) == 1 and out[0]["outcome"] == "superseded"
    assert led.open


def test_speech_end_takes_the_last_value_and_may_be_backdated():
    """The recorder knows when speech ended only once it stops; the mark is
    backdated and a later, later speech_end wins over an earlier one."""
    led, clock, out = _ledger()
    led.mark("wake")
    led.mark("mic")
    led.mark("speech_end", at=101.0)
    led.mark("speech_end", at=102.5)
    clock.t = 103.0
    led.mark("stop")
    clock.t = 104.0
    led.mark("audio")
    assert abs(out[0]["dead_air"] - 0.5) < 1e-9


def test_unknown_stage_is_a_programming_error():
    led, _, _ = _ledger()
    try:
        led.mark("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unknown stage")
    assert "audio" in STAGES


def test_emit_failure_never_propagates(monkeypatch):
    def boom(rec):
        raise RuntimeError("disk full")
    led = TurnLedger(clock=Clock(), emit=boom)
    led.mark("wake")
    led.mark("mic")
    led.mark("audio")  # must not raise
    assert not led.open


def test_events_carry_their_own_clock():
    """The bus queues events for the Tk thread when the UI is attached, so a
    subscriber's clock reads drain time -- one live turn showed wake->mic
    0 ms because both events drained in one tick. The five turn events stamp
    themselves at publish time and the ledger reads that."""
    import time
    from jarvis.events import (HotwordDetected, RecordingStarted, RecordingStopped,
                               SpeakingState, Transcribed)
    before = time.monotonic()
    evs = [HotwordDetected(score=0.9), RecordingStarted(), RecordingStopped(),
           Transcribed(text="x"), SpeakingState(active=True)]
    after = time.monotonic()
    for ev in evs:
        assert before <= ev.t <= after, type(ev).__name__
