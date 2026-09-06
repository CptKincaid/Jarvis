"""ROUND 6: A STOP THAT FAILS LEAVES THE DECK HELD WITH NOTHING WATCHING.

ONE LINE, BOTH DIRECTIONS, AND IT BLOCKS. Round 5 made the stop verb wait
for its receipt and keep the deck when the stop did not happen -- both
right. But ``_ViewSink._stop()`` cancels the confirm watchdog and moves the
generation on EVERY call, ``stop_cast()`` calls it with confirm=True, and
when the stop fails nothing re-arms. MEASURED on the modelled helper,
before the fix: after the failed stop, watchdogs armed 1, live 0,
cancelled 1, deck held. Then the cast ends by any route other than another
successful stop -- he closes the RustDesk window himself -- and over
3000 s of MODELLED wall time: windows up 0, retractions 0, deck still held,
every "cast my screen" refused as busy. Jarvis's last word was "There's a
viewer still up on HPCOMPUTER that I couldn't close, sir" and it never
took it back. Identical on HPCOMPUTER -> Spark. Round 4 had no such path,
because round 4 released the deck unconditionally.

THE FIX IS ONE CALL, on the shared base: when the stop failed,
``stop_cast`` re-arms the confirm watchdog before it returns the failed
report, logging (as ``_confirm`` already does) when it will not arm. These
tests were run red first; the numbers above are that run.

EVERY launcher, wait, settle, probe, timer and helper here is an INJECTED
RECORDER, borrowed from tests/test_castview_round5.py. Nothing starts a
RustDesk session, opens a process, a socket, a window, a camera or a
capture device; nothing contacts HPCOMPUTER; nothing touches the running
Jarvis or his desktop. The Windows side is MODELLED and every number
quoted from it is a modelled number.
"""
from __future__ import annotations

from jarvis import castview as cv
from tests.test_castview_round5 import (Clock, Helper, Later, hp_sink,
                                        hp_subj, live_relay, spark_subj)

# Modelled wall time to run the clock after the cast dies. Hard-bounded:
# 3000 s in four-second looks is 750 iterations, never more.
WATCH_S = 3000.0


def run_the_clock(clock, later, said, *, keep_alive=None, budget_s=WATCH_S):
    """Advance modelled time in confirm-sized steps, firing whatever the
    injected scheduler holds. ``(seconds until the first retraction or
    None, looks fired)``. ``keep_alive`` is called each step so a modelled
    helper can keep polling -- the only thing that changes in these
    scenarios is the window he closed, never the helper's presence."""
    t0 = clock.t
    fired_at = None
    looks = 0
    steps = int(budget_s / cv.VIEWER_CONFIRM_S)
    for _ in range(steps):
        if keep_alive is not None:
            keep_alive()
        clock.t += cv.VIEWER_CONFIRM_S
        looks += later.fire()
        if said and fired_at is None:
            fired_at = clock.t - t0
    return fired_at, looks


class TestAFailedStopKeepsWatching:
    """The deck is kept by a stop that did not happen -- right -- and the
    thing that would notice the cast ending any other way must be kept
    with it."""

    def test_spark_to_hp_a_window_he_closes_himself_is_still_spoken(self):
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        later = Later()
        said = []
        sink = hp_sink(clock, helper, relay, later=later, said=said)
        assert sink.deliver(hp_subj()).landed
        helper.can_kill = False               # the viewer will not close
        out = sink.stop_cast()
        assert not out and out.line == cv.HELPER_FAIL_LINES["stop-failed"]
        assert sink.state.live == "hp-view"
        live = len(later.jobs)
        print("\n  after the failed stop: watchdogs armed %d, live %d, "
              "cancelled %d, deck %r"
              % (later.armed, live, later.cancelled, sink.state.live))
        # He closes the RustDesk window on HPCOMPUTER himself. The helper
        # keeps polling; only the window is gone.
        helper.windows_up = 0
        fired_at, looks = run_the_clock(
            clock, later, said,
            keep_alive=lambda: helper.poll_once(at=clock.t))
        print("  %.0f s later (MODELLED): windows up %d, looks %d, "
              "retractions %d, deck %r"
              % (WATCH_S, helper.windows_up, looks, len(said),
                 sink.state.live))
        print("  retraction came after: %s"
              % ("%.1f s" % fired_at if fired_at is not None else "never"))
        helper.can_kill = True
        nxt = sink.deliver(hp_subj(at=clock.t))
        print("  the next \"cast my screen\": landed=%s held=%s %r"
              % (nxt.landed, nxt.held, nxt.spoken))
        assert live == 1, "a failed stop must leave the watchdog armed"
        assert said == [cv.CAST_GONE_LINE], "exactly one retraction"
        assert fired_at is not None and fired_at <= cv.VIEWER_CONFIRM_S
        assert nxt.landed, "the deck must have come back"

    def test_hp_to_spark_a_viewer_that_dies_later_is_still_spoken(self):
        clock = Clock()
        state = {"alive": True}
        later = Later()
        said = []

        def launch(_host):
            state["alive"] = True

        sink = cv.SparkViewSink(launch=launch,
                                stop=lambda: None,        # does nothing
                                alive=lambda: state["alive"],
                                connected=lambda: state["alive"],
                                settle=lambda s: None, later=later,
                                retract=said.append,
                                state=cv.ViewState(now=clock), now=clock)
        assert sink.deliver(spark_subj()).landed
        out = sink.stop_cast()                # the viewer survives the kill
        assert not out and out.line == cv.STOP_FAILED_LINE
        assert sink.state.live == "spark-view"
        live = len(later.jobs)
        print("\n  after the failed stop: watchdogs armed %d, live %d, "
              "cancelled %d, deck %r"
              % (later.armed, live, later.cancelled, sink.state.live))
        state["alive"] = False                # he closed it by hand
        fired_at, looks = run_the_clock(clock, later, said)
        print("  %.0f s later (MODELLED): viewer alive %s, looks %d, "
              "retractions %d, deck %r"
              % (WATCH_S, state["alive"], looks, len(said),
                 sink.state.live))
        print("  retraction came after: %s"
              % ("%.1f s" % fired_at if fired_at is not None else "never"))
        nxt = sink.deliver(spark_subj(at=clock.t))
        print("  the next \"cast my screen\": landed=%s held=%s %r"
              % (nxt.landed, nxt.held, nxt.spoken))
        assert live == 1, "a failed stop must leave the watchdog armed"
        assert said == [cv.CAST_GONE_LINE], "exactly one retraction"
        assert fired_at is not None and fired_at <= cv.VIEWER_CONFIRM_S
        assert nxt.landed, "the deck must have come back"

    def test_a_re_arm_that_will_not_start_is_logged_and_the_deck_is_kept(
            self, monkeypatch):
        """The shipped scheduler is ``threading.Timer(...).start()``, which
        raises under thread exhaustion. A failed stop whose watchdog will
        not re-arm is still a failed stop: the deck is kept, the line is
        spoken, and the log says the cast can no longer be watched -- the
        same degradation ``_confirm`` already reports. It must not raise."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        later = Later()
        sink = hp_sink(clock, helper, relay, later=later)
        assert sink.deliver(hp_subj()).landed
        helper.can_kill = False
        later.breaks = True                   # after the landing, not before
        warned = []
        monkeypatch.setattr(cv.log, "warning",
                            lambda msg, *a, **k: warned.append(str(msg)))
        out = sink.stop_cast()
        print("\n  stop failed AND the timer would not arm: %r" % out.line)
        print("  deck %r, pending %d, warnings %d"
              % (sink.state.live, len(later.jobs), len(warned)))
        assert not out and out.line == cv.HELPER_FAIL_LINES["stop-failed"]
        assert sink.state.live == "hp-view"
        assert later.jobs == []
        assert any("would not arm" in w for w in warned)
        assert any("no longer be watched" in w for w in warned)

    def test_the_second_stop_once_it_can_close_ends_the_watch_quietly(self):
        """The other half of the ledger. A watchdog kept alive by a failed
        stop must still be ended by the stop that works, and must not say
        "the cast has dropped" about a stop he asked for."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        later = Later()
        said = []
        sink = hp_sink(clock, helper, relay, later=later, said=said)
        assert sink.deliver(hp_subj()).landed
        helper.can_kill = False
        assert not sink.stop_cast()
        assert later.jobs, "re-armed by the failed stop"
        clock.t += cv.VIEWER_CONFIRM_S
        helper.poll_once(at=clock.t)
        later.fire()                          # the cast is still well
        assert later.jobs, "and it keeps watching"
        helper.can_kill = True                # now it can be closed
        again = sink.stop_cast()
        print("\n  second stop: %r  pending %d  said %r  deck %r"
              % (again.line, len(later.jobs), said, sink.state.live))
        assert again and again.line == cv.STOPPED_LINE
        assert later.jobs == [] and said == [] and sink.state.live == ""
