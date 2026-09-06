"""ROUND 5: THE STOP VERB SAYS A STOP HAPPENED THAT DID NOT.

Round 4 fixed PARKED-IS-NOT-RECEIVED for the SHOW verb and did not apply
it to the STOP verb, and the stop is the one he reaches for when a window
has landed on a screen he was reading. MEASURED against the real relay and
a faithful model of scripts/windows/cast-poll.ps1:

  * he says "stop the cast"; ``_ViewSink.stop_cast`` calls ``_stop()``,
    releases the deck and returns True UNCONDITIONALLY, so Jarvis says
    "Cast stopped, sir." at the instant the verb is PARKED. At that moment
    the viewer is still up on HPCOMPUTER and the helper has attempted
    nothing.
  * the helper then fails to kill it and reports ``stop-failed``. NOTHING
    SPEAKS IT: the deck is already released and ``failed_for(seq)`` is read
    by nobody.
  * asking again does not help. ``state.live`` is already "", so the second
    "stop the cast" answers "There's nothing cast, sir." while the window
    is still on his middle monitor.

The machinery to fix it already existed one method away: ``await_ack``,
``failed_for`` and ``HELPER_FAIL_LINES["stop-failed"]`` are all round 4's,
and today that line is only ever reachable through a LATER show-spark.

HPCOMPUTER -> SPARK is the same class: ``_stop()`` swallowed any exception
from ``_stop_here()``, so a kill that raised still returned True and still
said "Cast stopped, sir." with the viewer alive, the deck released, and
nothing tracking it.

AND FOUR SMALLER TRUTH DEFECTS, each measured here:
  * the second look was ONE-SHOT (deck held for ever by a dead cast);
  * a landing was claimed when the watchdog COULD NOT BE ARMED;
  * a "stop the cast" during the 0.35 s probe still got the retraction
    sentence, and pushed the suppression window forward;
  * the refusal diagnostic reported the fastest step of the WHOLE CARRY,
    not the speed inside the fling window the bar was compared against --
    so it could tell him he threw at 2.61 against a bar of 2.45 and was
    refused on speed.

EVERY launcher, wait, settle, probe, timer and helper here is an INJECTED
RECORDER. Nothing starts a RustDesk session, opens a process, a socket, a
window, a camera or a capture device; nothing contacts HPCOMPUTER; and
nothing touches the running Jarvis or his desktop. The Windows side is
MODELLED, and every number quoted from it is a modelled number.
"""
from __future__ import annotations

from jarvis import castview as cv


class Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


class Later:
    """The injected scheduler for the second look. Runs nothing until a
    test says so, and can be cancelled like the real timer."""

    def __init__(self, *, breaks=False):
        self.jobs = []
        self.cancelled = 0
        self.armed = 0
        self.breaks = bool(breaks)

    def __call__(self, delay_s, fn):
        if self.breaks:
            raise RuntimeError("can't start new thread")
        self.armed += 1
        self.jobs.append((float(delay_s), fn))
        return self

    def cancel(self):
        self.cancelled += 1
        self.jobs = []

    def fire(self):
        jobs, self.jobs = list(self.jobs), []
        for _d, fn in jobs:
            fn()
        return len(jobs)


def hp_subj(at=1000.0):
    return cv.view_subject(cv.HPCOMPUTER, at=at)


def spark_subj(at=1000.0):
    return cv.view_subject(cv.SPARK, at=at)


class Helper:
    """A faithful model of scripts/windows/cast-poll.ps1 AS IT SHIPS.

    MODELLED, not run: nothing here contacts HPCOMPUTER. It commits the
    receipt only after an action it has watched succeed, and reports a
    code from ``HELPER_FAILS`` otherwise -- including ``stop-failed``,
    which the script has sent since round 4 and which nothing on the
    Jarvis side has ever read.
    """

    def __init__(self, relay, *, launch_works=True, survives=True,
                 streams=True, can_kill=True):
        self.relay = relay
        self.seq = -1
        self.failseq = -1
        self.launch_works = launch_works
        self.survives = survives
        self.streams = streams
        self.can_kill = can_kill
        self.windows_up = 0
        self.launch_attempts = 0
        self.kill_attempts = 0
        self._fail = ""

    def poll_once(self, at=None):
        self.relay.note(mon=1, layout="0,1920,1920,1920", at=at,
                        fail=self._fail, failseq=self.failseq)
        out = self.relay.poll(self.seq, timeout_s=0.0)
        new, verb = int(out["seq"]), str(out["verb"])
        if new != self.seq and new != self.failseq:
            ok = True
            if verb == cv.VERB_SHOW:
                self.launch_attempts += 1
                if not self.launch_works:
                    self._fail, ok = "launch-failed", False
                elif not self.survives:
                    self._fail, ok = "viewer-exited", False
                else:
                    self.windows_up += 1
            elif verb == cv.VERB_STOP:
                self.kill_attempts += 1
                if self.can_kill:
                    self.windows_up = 0
                else:
                    # Stop-Cast's own reported failure: the viewer would
                    # not die, so the receipt does NOT move.
                    self._fail, ok = "stop-failed", False
            if ok:
                self.seq, self._fail, self.failseq = new, "", -1
            else:
                self.failseq = new
        return out

    def round_trip(self, at=None):
        """The real shape: one poll TAKES the verb and acts on it, the
        next poll QUOTES it back as the receipt."""
        self.poll_once(at=at)
        self.poll_once(at=at)

    def deaf(self, at=None):
        """The helper is inside its 25 s long poll and does not come back
        inside the ack budget. Nothing is attempted."""
        return None

    def connected(self):
        return bool(self.windows_up) and bool(self.streams)


def live_relay(clock):
    relay = cv.CastRelay(now=clock)
    relay.note(mon=1, layout="0,1920,1920,1920", at=clock.t)
    return relay


def hp_sink(clock, helper, relay, *, later=None, said=None, wait=None):
    return cv.HpViewSink(
        relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
        connected=helper.connected, later=later if later is not None
        else Later(), retract=(said.append if said is not None else None),
        settle=lambda s: None,
        wait=wait if wait is not None
        else (lambda s: helper.round_trip(at=clock.t)),
        now=clock)


# ==================================================== 1. THE BLOCKER
class TestTheStopVerbWaitsForItsReceipt:
    """PARKED IS NOT RECEIVED, applied to the verb round 4 left behind."""

    def test_a_stop_the_helper_could_not_do_is_not_reported_as_done(self):
        """THE BLOCKER, measured. The viewer will not die on HPCOMPUTER.
        The helper says so with the code round 4 added. Jarvis must repeat
        it and must NOT say "Cast stopped, sir."."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        sink = hp_sink(clock, helper, relay)
        assert sink.deliver(hp_subj()).landed
        helper.can_kill = False              # the viewer will not close
        out = sink.stop_cast()
        print("\n  MODELLED HPCOMPUTER: viewers up %d, kill attempts %d"
              % (helper.windows_up, helper.kill_attempts))
        print("  Jarvis says: %r  (stopped=%s)" % (out.line, bool(out)))
        assert helper.kill_attempts == 1, "the helper must have been asked"
        assert not out, "a stop that did not happen is not a stop"
        assert out.line == cv.HELPER_FAIL_LINES["stop-failed"]
        assert helper.windows_up == 1, "the window is still on his monitor"

    def test_the_deck_is_not_released_by_a_stop_that_did_not_happen(self):
        """The deck is the record of what is on his screens. Giving it
        back for a viewer that is still up is what made the SECOND "stop
        the cast" answer "There's nothing cast, sir."."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay, can_kill=False)
        sink = hp_sink(clock, helper, relay)
        assert sink.deliver(hp_subj()).landed
        first = sink.stop_cast()
        print("\n  first stop: %r" % first.line)
        print("  deck after a failed stop: %r" % sink.state.live)
        assert sink.state.live == "hp-view"
        helper.can_kill = True               # he closed it by hand, or it died
        second = sink.stop_cast()
        print("  asking again once it CAN be closed: %r" % second.line)
        assert second and second.line == cv.STOPPED_LINE
        assert sink.state.live == ""

    def test_the_helper_never_coming_back_is_not_a_stop_either(self):
        """Silence is not a receipt. The helper is inside its long poll
        and answers nothing within the ack budget."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        sink = hp_sink(clock, helper, relay)
        assert sink.deliver(hp_subj()).landed
        sink._wait = helper.deaf             # nobody answers from here on
        out = sink.stop_cast()
        print("\n  no receipt for the stop: %r" % out.line)
        print("  deck: %r   verb still parked: %r"
              % (sink.state.live, relay.verb))
        assert not out
        assert out.line == cv.NO_STOP_RECEIPT_LINE
        assert sink.state.live == "hp-view"
        assert relay.verb == cv.VERB_STOP, "the verb stays on for a slow helper"

    def test_a_stop_that_worked_still_says_so(self):
        """The other side of the ledger: a fix that makes a working stop
        unspeakable is also a failure."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        sink = hp_sink(clock, helper, relay)
        assert sink.deliver(hp_subj()).landed
        out = sink.stop_cast()
        print("\n  MODELLED: viewers up %d after the stop; Jarvis: %r"
              % (helper.windows_up, out.line))
        assert out and out.line == cv.STOPPED_LINE
        assert helper.windows_up == 0 and sink.state.live == ""

    def test_a_kill_that_raises_on_the_spark_is_never_a_true(self):
        """HPCOMPUTER -> SPARK, the same class. ``_stop()`` swallowed the
        exception, so a kill that raised still returned True, still said
        "Cast stopped, sir.", released the deck, and left a viewer alive
        that nothing was tracking -- the next cast opened a second one on
        top of it."""
        clock = Clock()
        state = {"alive": True}

        def stop():
            raise OSError("no such process")

        sink = cv.SparkViewSink(launch=lambda h: None, stop=stop,
                                alive=lambda: state["alive"],
                                connected=lambda: True, settle=lambda s: None,
                                later=Later(), state=cv.ViewState(now=clock),
                                now=clock)
        assert sink.deliver(spark_subj()).landed
        out = sink.stop_cast()
        print("\n  the kill raised: stopped=%s  Jarvis: %r"
              % (bool(out), out.line))
        print("  deck: %r  (the viewer is still alive)" % sink.state.live)
        assert not out
        assert out.line == cv.STOP_FAILED_LINE
        assert sink.state.live == "spark-view"

    def test_a_viewer_that_survives_the_kill_is_not_a_stop(self):
        """The mirror of the helper's ``stop-failed``, and it is the same
        probe the landing claim already uses: after asking the viewer to
        die, ask whether it did."""
        clock = Clock()
        state = {"alive": True}
        sink = cv.SparkViewSink(launch=lambda h: None,
                                stop=lambda: None,           # does nothing
                                alive=lambda: state["alive"],
                                connected=lambda: True, settle=lambda s: None,
                                later=Later(), state=cv.ViewState(now=clock),
                                now=clock)
        assert sink.deliver(spark_subj()).landed
        out = sink.stop_cast()
        print("\n  viewer still alive after the kill: stopped=%s %r"
              % (bool(out), out.line))
        assert not out and sink.state.live == "spark-view"
        state["alive"] = False
        again = sink.stop_cast()
        print("  once it really died: %r" % again.line)
        assert again and sink.state.live == ""

    def test_nothing_of_ours_is_still_nothing_of_ours(self):
        clock = Clock()
        sink = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                                alive=lambda: True, connected=lambda: True,
                                settle=lambda s: None, later=Later(),
                                state=cv.ViewState(now=clock), now=clock)
        out = sink.stop_cast()
        print("\n  nothing up: %s / %r" % (bool(out), out.line))
        assert not out and out.mine is False
        assert out.line == cv.NOTHING_UP_LINE

    def test_the_courier_speaks_the_failure_rather_than_nothing_is_cast(self):
        """The whole point, at the level he actually hears. The courier
        used to fall through to "There's nothing cast, sir." for a stop
        that failed, because a False from the sink was indistinguishable
        from "not ours"."""
        from tests.test_gesturecast import build
        out = build()
        courier = out.courier

        class Stuck:
            name = "hp-view"
            label = "HPCOMPUTER"

            def stop_cast(self):
                return cv.StopReport(
                    False, True, cv.HELPER_FAIL_LINES["stop-failed"],
                    "stop-failed")

        courier.views = {cv.HPCOMPUTER: Stuck()}
        courier.view_state.live = "hp-view"
        line = courier.stop_cast()
        print("\n  courier says: %r" % line)
        assert line == cv.HELPER_FAIL_LINES["stop-failed"]
        assert courier.view_state.live == "hp-view"


# ============================================ 2. the four truth defects
class TestTheSecondLookRepeats:
    """MEASURED: the confirm fired at 4.0 s, found everything well, and NO
    timer was ever armed again. Running the clock 200 s with the viewer
    gone left the deck held, 0 retractions, 0 timers, and the next genuine
    cast refused as busy."""

    def test_a_cast_that_dies_after_the_first_confirm_is_still_caught(self):
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        later = Later()
        said = []
        sink = hp_sink(clock, helper, relay, later=later, said=said)
        assert sink.deliver(hp_subj()).landed
        looks = 0
        for _ in range(50):                  # 200 s of four-second looks
            if not later.jobs:
                break
            clock.t += cv.VIEWER_CONFIRM_S
            looks += later.fire()
        print("\n  looks taken while the cast was well: %d" % looks)
        assert looks >= 5, "one look is not a watch"
        helper.windows_up = 0                # NOW it falls over
        clock.t += cv.VIEWER_CONFIRM_S
        later.fire()
        print("  retraction after it died: %r  deck %r"
              % (said, sink.state.live))
        assert said == [cv.CAST_GONE_LINE]
        assert sink.state.live == ""

    def test_the_spark_direction_repeats_too(self):
        clock = Clock()
        state = {"alive": True}
        later = Later()
        said = []
        sink = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                                alive=lambda: state["alive"],
                                connected=lambda: state["alive"],
                                settle=lambda s: None, later=later,
                                retract=said.append,
                                state=cv.ViewState(now=clock), now=clock)
        assert sink.deliver(spark_subj()).landed
        looks = 0
        for _ in range(150):                 # ten minutes
            if not later.jobs:
                break
            clock.t += cv.VIEWER_CONFIRM_S
            looks += later.fire()
        print("\n  looks over ten minutes: %d" % looks)
        assert looks >= 100
        state["alive"] = False
        clock.t += cv.VIEWER_CONFIRM_S
        later.fire()
        print("  retraction: %r  deck %r" % (said, sink.state.live))
        assert said == [cv.CAST_GONE_LINE] and sink.state.live == ""

    def test_the_watch_stops_when_he_stops_the_cast(self):
        """A repeating timer that repeats for ever is a leak. His stop
        must end it."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        later = Later()
        said = []
        sink = hp_sink(clock, helper, relay, later=later, said=said)
        assert sink.deliver(hp_subj()).landed
        clock.t += cv.VIEWER_CONFIRM_S
        later.fire()
        assert later.jobs, "it should have re-armed"
        assert sink.stop_cast()
        print("\n  after his stop: pending %d, cancelled %d, said %r"
              % (len(later.jobs), later.cancelled, said))
        assert later.jobs == [] and later.cancelled >= 1 and said == []


class TestALandingNeedsAWatchdog:
    """``_arm_confirm`` swallowed the exception from ``_later`` and
    returned; ``deliver`` still returned landed=True and still held the
    deck. The shipped ``_later`` is ``threading.Timer(...).start()``,
    which raises under thread exhaustion -- not hypothetical on a box that
    has had an OOM kill. Failing to arm the watchdog must HOLD."""

    def test_the_hp_cast_holds_when_the_timer_will_not_arm(self):
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        sink = hp_sink(clock, helper, relay, later=Later(breaks=True))
        res = sink.deliver(hp_subj())
        print("\n  timer would not arm: %r (landed=%s held=%s)"
              % (res.spoken, res.landed, res.held))
        assert not res.landed and res.held
        assert res.spoken == cv.NO_WATCHDOG_LINE
        assert sink.state.live == "", "the deck must not be held"
        assert relay.verb == cv.VERB_STOP, "the verb must be taken back"

    def test_the_spark_cast_holds_too(self):
        clock = Clock()
        stopped = []
        sink = cv.SparkViewSink(launch=lambda h: None,
                                stop=lambda: stopped.append(1),
                                alive=lambda: True, connected=lambda: True,
                                settle=lambda s: None,
                                later=Later(breaks=True),
                                state=cv.ViewState(now=clock), now=clock)
        res = sink.deliver(spark_subj())
        print("\n  %r (landed=%s)  viewer stopped %d time(s)"
              % (res.spoken, res.landed, len(stopped)))
        assert not res.landed and res.held
        assert res.spoken == cv.NO_WATCHDOG_LINE
        assert sink.state.live == "" and stopped == [1]


class TestNoRetractionForAStopHeAskedFor:
    """``_confirm`` read ``state.live``, then spent 0.35 s inside the real
    probe. A "stop the cast" in that window still got "The cast has
    dropped, sir. What I told you a moment ago is no longer true." -- and
    it pushed ``_last_end`` forward, so the cast he asked for next was
    refused as too soon."""

    def test_a_stop_during_the_probe_is_not_a_dropped_cast(self):
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        later = Later()
        said = []
        box = {}

        def slow_probe():
            # The real probe sleeps 0.35 s between two byte readings. He
            # says "stop the cast" inside that window.
            if box.get("armed"):
                box["armed"] = False
                box["stop"] = box["sink"].stop_cast()
            return helper.connected()

        sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                             state=cv.ViewState(now=clock),
                             connected=slow_probe, later=later,
                             retract=said.append, settle=lambda s: None,
                             wait=lambda s: helper.round_trip(at=clock.t),
                             now=clock)
        box["sink"] = sink
        assert sink.deliver(hp_subj()).landed
        box["armed"] = True
        clock.t += cv.VIEWER_CONFIRM_S
        later.fire()
        print("\n  his stop landed mid-probe: %r" % box["stop"].line)
        print("  retractions: %r" % (said,))
        assert bool(box["stop"]), box["stop"].line
        assert said == [], "nothing may retract a cast he ended himself"

    def test_it_does_not_push_the_suppression_window_forward(self):
        """The second symptom, and the one he would feel: a spurious
        retraction re-stamps ``_last_end``, so the cast he asks for next
        is refused as 'a cast ended less than 3s ago'."""
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        later = Later()
        said = []
        box = {}

        def slow_probe():
            if box.get("armed"):
                box["armed"] = False
                box["sink"].stop_cast()
                clock.t += 0.35          # the real probe's own sample window
            return helper.connected()

        sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                             state=cv.ViewState(now=clock),
                             connected=slow_probe, later=later,
                             retract=said.append, settle=lambda s: None,
                             wait=lambda s: helper.round_trip(at=clock.t),
                             now=clock)
        box["sink"] = sink
        assert sink.deliver(hp_subj()).landed
        box["armed"] = True
        stop_at = clock.t + cv.VIEWER_CONFIRM_S
        clock.t = stop_at
        later.fire()
        clock.t = stop_at + cv.CAST_SUPPRESS_S + 0.01
        print("  clock ran %.2fs inside the probe" % (clock.t - stop_at))
        ok, why = sink.state.take("probe")
        print("\n  a cast %.2fs after HIS stop: ok=%s (%s)"
              % (cv.CAST_SUPPRESS_S + 0.01, ok, why))
        assert ok, why


class TestTheHpProbeIsTiedToThisCast:
    """``_hpcomputer_watching`` said True for ANY established socket from
    HPCOMPUTER on a screen-sharing port -- including a RustDesk session he
    opened himself before Jarvis was asked for anything. The sink now ARMS
    the probe before it starts the cast, and the probe answers about what
    appeared after that."""

    def test_the_sink_arms_the_probe_before_it_starts_the_cast(self):
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)
        order = []

        def arm():
            order.append("arm")

        sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                             state=cv.ViewState(now=clock),
                             connected=helper.connected, arm=arm,
                             later=Later(), settle=lambda s: None,
                             wait=lambda s: (order.append("verb"),
                                             helper.round_trip(at=clock.t)),
                             now=clock)
        assert sink.deliver(hp_subj()).landed
        print("\n  order of operations: %s" % (order,))
        assert order and order[0] == "arm", order

    def test_an_arm_that_raises_never_stops_a_cast(self):
        clock = Clock()
        relay = live_relay(clock)
        helper = Helper(relay)

        def arm():
            raise OSError("/proc/net/tcp went away")

        sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                             state=cv.ViewState(now=clock),
                             connected=helper.connected, arm=arm,
                             later=Later(), settle=lambda s: None,
                             wait=lambda s: helper.round_trip(at=clock.t),
                             now=clock)
        res = sink.deliver(hp_subj())
        print("\n  the arm raised: %r (landed=%s)" % (res.spoken, res.landed))
        assert res.landed

    def test_a_session_he_had_open_already_is_not_this_cast(self):
        """The app-level probe, against a MODELLED /proc. His own
        RustDesk session from HPCOMPUTER is up before Jarvis is asked for
        anything, and it is pouring bytes. That must not read as a cast
        Jarvis landed."""
        from types import SimpleNamespace

        import jarvis.app as appmod

        rows = {"inodes": {4001}, "wchar": 0}

        def wchar(_pid):
            rows["wchar"] += 10 ** 6
            return rows["wchar"]

        fake = SimpleNamespace(
            established_to=lambda host, ports: set(rows["inodes"]),
            pid_for_inode=lambda inode: 777,
            wchar=wchar, rchar=lambda pid: None,
            has_socket_to=lambda pid, host: True)
        app = appmod.JarvisApp.__new__(appmod.JarvisApp)
        app._serve_seen = set()
        old, appmod.procnet = appmod.procnet, fake
        old_sleep, appmod.time = appmod.time, SimpleNamespace(
            sleep=lambda s: None, monotonic=lambda: 0.0)
        try:
            app._hpcomputer_watch_arm()
            before = app._hpcomputer_watching()
            print("\n  his OWN session, armed against it: %r" % (before,))
            assert before is False
            rows["inodes"] = {4001, 4002}       # the cast's own connection
            after = app._hpcomputer_watching()
            print("  a NEW connection after the arm: %r" % (after,))
            assert after is True
        finally:
            appmod.procnet = old
            appmod.time = old_sleep
