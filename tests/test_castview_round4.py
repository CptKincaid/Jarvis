"""ROUND 4: the SPOKEN cast, which is the one he can hit today.

The gesture ships ``enabled: False``. The spoken cast does not, so a
sentence it says wrongly is live exposure and a false-fire rate behind a
switch he has not thrown is not. This file is about the sentence.

ROUND 3 FIXED ONE DIRECTION AND NOT THE OTHER. It diagnosed "a live
RustDesk process is not a connection" correctly, gave ``SparkViewSink`` a
tri-state ``connected`` probe, a second look and a spoken retraction --
and left ``HpViewSink`` exactly as it was. HpViewSink is the
Spark -> HPCOMPUTER direction, which is his PRIMARY one. Measured against
the real relay and the real sink, with a faithful in-process model of
scripts/windows/cast-poll.ps1: a viewer that launches on HPCOMPUTER,
survives the script's own 700 ms settle and then sits on an
accept-or-password prompt produced landed=True, held=False, "The Spark's
screen is on HPCOMPUTER, sir." with nothing on any monitor -- and nothing
ever re-checked it, so ``state.live`` stayed held by a cast that was not
there and the next genuine cast was refused as busy.

WHAT THE EVIDENCE CAN AND CANNOT SAY, and it does not change here: a
socket carrying a stream is proof of a live connection. It is NOT proof
that a window is visible, or on the monitor he expects. That last step has
no local evidence and is his to look at once.

EVERY launcher, wait, settle, probe, timer and helper here is an INJECTED
RECORDER. Nothing starts a RustDesk session, opens a process, a socket, a
window, a camera or a capture device; nothing contacts HPCOMPUTER; and
nothing touches the running Jarvis or his desktop. The Windows side is
MODELLED.
"""
from __future__ import annotations

import inspect
import re

from jarvis import castview as cv


class Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


class Later:
    """The injected scheduler for the second look. Runs nothing until a
    test says so, and can be cancelled like the real timer."""

    def __init__(self):
        self.jobs = []
        self.cancelled = 0

    def __call__(self, delay_s, fn):
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


class Round3Helper:
    """A faithful model of scripts/windows/cast-poll.ps1 AS ROUND 3 WROTE
    IT -- the corrected script, not the broken one.

    It commits the receipt only after a launch it has WATCHED survive its
    own 700 ms settle, and reports a code from ``HELPER_FAILS`` otherwise.
    That is the script working exactly as designed. ``streams`` is the
    thing the script cannot see and never could: whether the viewer it
    started is showing a desktop or sitting on an accept-or-password
    prompt.
    """

    def __init__(self, relay, *, launch_works=True, survives=True,
                 streams=True):
        self.relay = relay
        self.seq = -1
        self.failseq = -1
        self.launch_works = launch_works
        self.survives = survives
        self.streams = streams
        self.windows_up = 0
        self.launch_attempts = 0

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
                self.windows_up = 0
            if ok:
                self.seq, self._fail, self.failseq = new, "", -1
            else:
                self.failseq = new
        return out

    _fail = ""

    def round_trip(self, at=None):
        """The real shape: one poll TAKES the verb and acts on it, the
        next poll QUOTES it back as the receipt."""
        self.poll_once(at=at)
        self.poll_once(at=at)

    # What is actually on his middle monitor. The script cannot see this.
    def connected(self):
        return bool(self.windows_up) and bool(self.streams)


def live_relay(clock):
    relay = cv.CastRelay(now=clock)
    relay.note(mon=1, layout="0,1920,1920,1920", at=clock.t)
    return relay


# ============================================ 1. up is not connected, both ways
def test_a_viewer_on_a_password_prompt_is_not_the_spark_on_hpcomputer():
    """THE ROUND-3 GAP, in one test. The helper did everything right --
    it launched, it watched the viewer survive 700 ms, it sent the
    receipt -- and the viewer is sitting on an accept-or-password prompt
    with nothing on the monitor. A receipt is proof the script ACTED. It
    is not proof that a desktop arrived."""
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay, streams=False)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=helper.connected, later=Later(),
                         settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    res = sink.deliver(hp_subj())
    print("\n  helper: %d launch attempts, %d viewers up, streaming=%s"
          % (helper.launch_attempts, helper.windows_up, helper.streams))
    print("  Jarvis says: %r  (landed=%s held=%s)"
          % (res.spoken, res.landed, res.held))
    assert not res.landed and res.held, res
    assert sink.state.live == "", "the deck must be given back"
    assert relay.verb == cv.VERB_STOP, "the verb must be taken back"


def test_a_viewer_that_IS_streaming_still_lands():
    """The other side of the ledger: a fix that kills the feature is also
    a failure."""
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=helper.connected, later=Later(),
                         settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    res = sink.deliver(hp_subj())
    print("\n  windows up %d, streaming: %r" % (helper.windows_up, res.spoken))
    assert res.landed and not res.held, res
    assert sink.state.live == "hp-view"


def test_the_hp_sink_with_no_connection_probe_reports_itself_unavailable():
    """The rule ``SparkViewSink`` is already held to, applied to the
    direction that actually matters. No honest local evidence that the
    cast arrived means no landing may be claimed -- so the sink says it
    is unavailable rather than saying a sentence it cannot support."""
    clock = Clock()
    relay = live_relay(clock)
    sink = cv.HpViewSink(relay=relay, state=cv.ViewState(now=clock), now=clock)
    ok, why = sink.available()
    print("\n  no connection probe wired: available=%s (%s)" % (ok, why))
    assert not ok and why == cv.NO_CONNECTION_PROBE_REASON
    res = sink.deliver(hp_subj())
    print("  and it says: %r" % res.spoken)
    assert not res.landed and res.held
    assert relay.seq == 0, "nothing may be parked by an unavailable sink"


def test_the_hp_sink_that_cannot_tell_says_so_rather_than_claiming():
    """A probe with NO OPINION is not a yes. /proc is unreadable on some
    ownerships and the byte counter can simply be absent."""
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=lambda: None, later=Later(),
                         settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    res = sink.deliver(hp_subj())
    print("\n  probe has no opinion: %r" % res.spoken)
    assert not res.landed and res.held
    assert res.spoken == cv.CANNOT_TELL_LINE
    assert sink.state.live == "" and relay.verb == cv.VERB_STOP


def test_a_probe_that_raises_is_never_a_yes():
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay)

    def boom():
        raise OSError("/proc went away")

    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=boom, later=Later(),
                         settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    res = sink.deliver(hp_subj())
    print("\n  probe raised: %r" % res.spoken)
    assert not res.landed and res.held


# ================================================== 2. the second look
def test_the_hp_cast_is_looked_at_again_and_retracted_out_loud():
    """A cast that is gone at four seconds was claimed at one and never
    revisited: the deck stayed held by something that was not there and
    the next genuine cast was refused as busy. The second look gives the
    deck back AND takes the sentence back, out loud."""
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay)
    later = Later()
    said = []
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=helper.connected, later=later,
                         retract=said.append, settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    res = sink.deliver(hp_subj())
    assert res.landed, res
    print("\n  claimed: %r" % res.spoken)
    assert later.jobs and later.jobs[0][0] == cv.VIEWER_CONFIRM_S, later.jobs
    helper.windows_up = 0                    # the viewer fell over
    fired = later.fire()
    print("  second look fired %d job(s); retraction: %r" % (fired, said))
    assert said == [cv.CAST_GONE_LINE], said
    assert sink.state.live == "", "the deck must come back"
    assert relay.verb == cv.VERB_STOP


def test_the_second_look_leaves_a_live_cast_alone():
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay)
    later = Later()
    said = []
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=helper.connected, later=later,
                         retract=said.append, settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    assert sink.deliver(hp_subj()).landed
    later.fire()
    print("\n  still streaming: retractions %r, deck %r"
          % (said, sink.state.live))
    assert said == [] and sink.state.live == "hp-view"


def test_the_second_look_notices_the_helper_stopped_polling():
    """A machine that has stopped answering is a machine that is gone,
    whatever a socket says. ``alive()`` is a minute wide, so this is the
    slow half of the same evidence."""
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay)
    later = Later()
    said = []
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=lambda: True, later=later,
                         retract=said.append, settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    assert sink.deliver(hp_subj()).landed
    clock.t += cv.HELPER_ALIVE_S + 1.0       # HPCOMPUTER went away
    later.fire()
    print("\n  helper silent %.0fs: retraction %r, deck %r"
          % (cv.HELPER_ALIVE_S + 1.0, said, sink.state.live))
    assert said == [cv.CAST_GONE_LINE]
    assert sink.state.live == ""


def test_stopping_a_cast_cancels_the_second_look():
    """He said stop. Nothing may retract a sentence about a cast he ended
    himself."""
    clock = Clock()
    relay = live_relay(clock)
    helper = Round3Helper(relay)
    later = Later()
    said = []
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clock),
                         connected=helper.connected, later=later,
                         retract=said.append, settle=lambda s: None,
                         wait=lambda s: helper.round_trip(at=clock.t),
                         now=clock)
    assert sink.deliver(hp_subj()).landed
    assert sink.stop_cast()
    print("\n  cancelled %d timer(s); pending jobs %d"
          % (later.cancelled, len(later.jobs)))
    assert later.cancelled >= 1 and later.jobs == []
    assert said == []


# ============================================== 3. one shape, not two
def test_the_two_sinks_share_one_honesty_and_do_not_repeat_it():
    """THE ANSWER TO 'should that shape be shared'. Yes. A second copy of
    'probe, give the deck back, retract' is a second place for the two to
    drift apart -- which is exactly how round 3 shipped one honest sink
    and one that lied. The probe, the tri-state, the second look, the
    retraction and the give-back live on the shared base."""
    base = cv._ViewSink
    for name in ("_is_connected", "_arm_confirm", "_confirm", "_give_back",
                 "_still_there", "available"):
        assert hasattr(base, name), name
    shared = []
    for name in ("_is_connected", "_arm_confirm", "_confirm", "_give_back"):
        a = getattr(cv.SparkViewSink, name)
        b = getattr(cv.HpViewSink, name)
        c = getattr(base, name)
        assert a is c and b is c, name
        shared.append(name)
    print("\n  shared on _ViewSink, not copied: %s" % ", ".join(shared))
    # ...and the one thing that legitimately differs: what "still there"
    # means for a process we own versus a machine that polls us.
    assert cv.SparkViewSink._still_there is not base._still_there
    assert cv.HpViewSink._still_there is not base._still_there
    print("  overridden per direction: _still_there")


def test_both_sinks_refuse_to_land_without_a_connection_probe():
    clock = Clock()
    relay = live_relay(clock)
    pairs = [("spark-view", cv.SparkViewSink(launch=lambda h: None,
                                             alive=lambda: True,
                                             settle=lambda s: None,
                                             now=clock)),
             ("hp-view", cv.HpViewSink(relay=relay, now=clock))]
    print()
    for name, sink in pairs:
        ok, why = sink.available()
        print("  %-11s with no connection probe: available=%s (%s)"
              % (name, ok, why))
        assert not ok and why == cv.NO_CONNECTION_PROBE_REASON


def test_the_module_still_holds_no_transport_of_its_own():
    """The line this whole file lives behind: castview.py spawns nothing
    and opens nothing. The probes are injected callables."""
    src = inspect.getsource(cv)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    for word in ("subprocess", "Popen(", "socket.", "requests", "urllib",
                 "cv2", "VideoCapture", "rustdesk", "os.system"):
        assert word not in code, word
    print("\n  jarvis/castview.py: no transport, no lens, no spawner")
