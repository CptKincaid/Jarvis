"""ROUND 7: A STOP THAT SUCCEEDS INSIDE THE LANDING PROBE MUST NOT LAND.

Named by the round-6 adversary as "remaining (1)", not a blocker, fix one
line. ``deliver`` parks the verb, gets the receipt, then spends the settle
and the connection probe (~0.35 s of real probing on the Spark side, a
modelled round trip on the HP side) before saying it landed. A stop that
arrives INSIDE that window -- typed, from the phone, over the socket;
voice turns are serialised so it cannot come by voice -- succeeds: the
viewer closes, the deck is released, Jarvis says "Cast stopped, sir." And
then ``deliver`` returns from the probe with ``None`` and says "The Spark's
screen is on HPCOMPUTER, sir." with nothing on any monitor. The watchdog
sees the deck free and stays silent, so it is never retracted. Deck is
free, nothing strands -- but a sentence that is not true was said.

THE FIX IS THE RE-CHECK the round-5 ``_confirm`` already does after ITS
probe: is the deck still mine? If a stop took it during the window, the
landing is not claimed.

EVERY launcher, wait, settle, probe and helper is an INJECTED RECORDER from
tests/test_castview_round5.py. Nothing starts a RustDesk session, opens a
process, a socket, a window, a camera or a capture device; nothing contacts
HPCOMPUTER; nothing touches the running Jarvis or his desktop. The Windows
side is MODELLED and every number from it is a modelled number.
"""
from __future__ import annotations

from jarvis import castview as cv
from tests.test_castview_round5 import (Clock, Helper, Later, hp_subj,
                                        live_relay)


def hp_sink_with_probe(clock, helper, relay, probe, said):
    """The round-5 builder with ONE seam swapped: the connection probe."""
    return cv.HpViewSink(
        relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
        connected=probe, later=Later(), retract=said.append,
        settle=lambda s: None,
        wait=lambda s: helper.round_trip(at=clock.t),
        now=clock)


class TestAStopInsideTheLandingProbe:

    def test_hp_a_stop_that_lands_during_the_probe_is_not_a_landing(self):
        """THE CASE. The probe itself is where the stop arrives."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        said = []
        box = {}

        def probe():
            # THE ORDER IS THE RACE. The probe reads the truth (viewer up,
            # bytes flowing -> True). THEN a typed "stop the cast" lands
            # and succeeds: the helper kills it, the deck is released.
            # deliver is handed the answer that was true a moment ago.
            answer = helper.connected()
            box["stop"] = box["sink"].stop_cast()
            return answer

        box["sink"] = hp_sink_with_probe(clock, helper, relay, probe, said)
        res = box["sink"].deliver(hp_subj())
        print("\n  MODELLED HPCOMPUTER: viewers up %d" % helper.windows_up)
        print("  the stop said: %r (stopped=%s)"
              % (box["stop"].line, bool(box["stop"])))
        print("  deliver said: %r  landed=%s held=%s"
              % (res.spoken, res.landed, res.held))
        assert bool(box["stop"]), "the premise: the stop succeeded"
        assert helper.windows_up == 0, "the premise: nothing is on his monitor"
        assert not res.landed, (
            "deliver claimed a landing for a cast that was stopped during "
            "its own probe: %r" % res.spoken)
        assert box["sink"].state.live == "", "the deck must stay free"
        assert not (res.landed and res.held)

    def test_hp_the_landing_line_is_never_spoken_after_a_stop(self):
        """The two sentences he must never hear back to back."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        said = []
        box = {}

        def probe():
            answer = helper.connected()
            box["stop"] = box["sink"].stop_cast()
            return answer

        box["sink"] = hp_sink_with_probe(clock, helper, relay, probe, said)
        res = box["sink"].deliver(hp_subj())
        landing_line = cv.SHOWN_LINE.format(
            What=cv._cap(cv.VIEW_SCREENS[cv.HPCOMPUTER]),
            target=box["sink"].label)
        assert res.spoken != landing_line, (
            "the landing line was spoken after a successful stop")

    def test_hp_an_ordinary_landing_is_unchanged(self):
        """The re-check must not cost a real landing."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        said = []
        sink = hp_sink_with_probe(clock, helper, relay, helper.connected, said)
        res = sink.deliver(hp_subj())
        assert res.landed and not res.held
        assert sink.state.live == sink.name


def spark_sink_with_probe(clock, probe, said, box):
    """A Spark viewer whose process we own, with the probe swapped."""
    alive = {"up": True}

    def launch(host):
        alive["up"] = True

    def stop():
        alive["up"] = False

    box["alive"] = alive
    return cv.SparkViewSink(
        launch=launch, stop=stop, alive=lambda: alive["up"],
        connected=probe, settle=lambda s: None, later=Later(),
        retract=said.append, state=cv.ViewState(now=clock), now=clock)


class TestAStopInsideTheLandingProbeSparkSide:

    def test_spark_a_stop_that_lands_during_the_probe_is_not_a_landing(self):
        from tests.test_castview_round5 import spark_subj
        clock = Clock()
        said = []
        box = {}

        def probe():
            answer = True if box["alive"]["up"] else False
            box["stop"] = box["sink"].stop_cast()
            return answer

        box["sink"] = spark_sink_with_probe(clock, probe, said, box)
        res = box["sink"].deliver(spark_subj())
        print("\n  viewer alive: %s   the stop: %r (stopped=%s)"
              % (box["alive"]["up"], box["stop"].line, bool(box["stop"])))
        print("  deliver said: %r  landed=%s held=%s"
              % (res.spoken, res.landed, res.held))
        assert bool(box["stop"]), "the premise: the stop succeeded"
        assert not box["alive"]["up"], "the premise: the viewer is gone"
        assert not res.landed, (
            "deliver claimed a landing for a cast that was stopped during "
            "its own probe: %r" % res.spoken)
        assert box["sink"].state.live == ""

    def test_spark_an_ordinary_landing_is_unchanged(self):
        from tests.test_castview_round5 import spark_subj
        clock = Clock()
        said = []
        box = {}
        box["sink"] = spark_sink_with_probe(
            clock, lambda: True if box["alive"]["up"] else False, said, box)
        res = box["sink"].deliver(spark_subj())
        assert res.landed and not res.held
        assert box["sink"].state.live == box["sink"].name
