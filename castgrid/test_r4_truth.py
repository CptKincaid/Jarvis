"""ROUND 4's Block-C instrument: does the SPOKEN cast still lie?

The same question the round-3 adversary asked, asked of both directions
this time, and asked of the direction that is actually exposed: the
gesture ships ``enabled: False``, the spoken cast does not.

HOW TO RUN IT. This directory is deliberately OUTSIDE pytest.ini's
``testpaths``, so ``pytest -q`` does not collect it. From the repo root:

    ~/.local/bin/memcap timeout 900 ~/vss_env/bin/python -m pytest -q \
        -p no:cacheprovider castgrid/test_r4_truth.py -s

EVERY launcher, wait, settle, probe, timer and helper is an INJECTED
RECORDER. Nothing starts a RustDesk session, opens a process, a socket, a
window, a camera or a capture device; nothing contacts HPCOMPUTER at all.
THE WINDOWS SIDE IS MODELLED, and every number that comes out of the model
is labelled MODELLED where it is quoted.
"""
from __future__ import annotations

from jarvis import castview as cv


class Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


class Later:
    def __init__(self):
        self.jobs = []

    def __call__(self, delay_s, fn):
        self.jobs.append((float(delay_s), fn))
        return self

    def cancel(self):
        self.jobs = []

    def fire(self):
        jobs, self.jobs = list(self.jobs), []
        for _d, fn in jobs:
            fn()
        return len(jobs)


class Helper:
    """MODELLED: scripts/windows/cast-poll.ps1 as ROUND 4 writes it.

    Faithful to the file, statement for statement, in the parts that
    decide a receipt:

      * a show-spark first stops the viewer it has, and does NOT start a
        new one if that stop did not take (round 4);
      * a launch is watched for 700 ms and only then receipted;
      * a stop is receipted only when the process is actually gone
        (round 4);
      * a sequence it has failed is never retried.

    ``streams`` is the one thing the script cannot see and never could:
    whether the viewer it started is showing a desktop or sitting on an
    accept-or-password prompt.
    """

    def __init__(self, *, launch_works=True, survives=True, streams=True,
                 stop_works=True):
        self.seq = -1
        self.failseq = -1
        self.fail = ""
        self.launch_works = launch_works
        self.survives = survives
        self.streams = streams
        self.stop_works = stop_works
        self.viewers = 0                # how many are up on HIS monitor
        self.launch_attempts = 0
        self.stop_attempts = 0

    # -- the two functions ------------------------------------------------
    def stop_cast(self) -> bool:
        if not self.viewers:
            return True                 # already gone: nothing to report
        self.stop_attempts += 1
        if not self.stop_works:
            self.fail = "stop-failed"
            return False
        self.viewers = 0
        return True

    def start_viewer(self) -> bool:
        self.launch_attempts += 1
        if not self.launch_works:
            self.fail = "launch-failed"
            return False
        if not self.survives:
            self.fail = "viewer-exited"
            return False
        self.viewers += 1
        return True

    # -- the loop ---------------------------------------------------------
    def poll_once(self, relay, at):
        relay.note(mon=1, layout="0,1920,1920,1920", at=at, fail=self.fail,
                   failseq=self.failseq)
        out = relay.poll(self.seq, timeout_s=0.0)
        new, verb = int(out["seq"]), str(out["verb"])
        if new != self.seq and new != self.failseq:
            ok = True
            if verb == cv.VERB_SHOW:
                ok = self.stop_cast() and self.start_viewer()
            elif verb == cv.VERB_STOP:
                ok = self.stop_cast()
            if ok:
                self.seq, self.fail, self.failseq = new, "", -1
            else:
                self.failseq = new
        return out

    def round_trip(self, relay, at):
        self.poll_once(relay, at)
        self.poll_once(relay, at)

    def connected(self):
        """What the SPARK can see out of /proc: an inbound socket from
        HPCOMPUTER on a screen-sharing port, carrying a stream."""
        return bool(self.viewers) and bool(self.streams)


def bench(**kw):
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920,1920,1920", at=clk.t)
    helper = Helper(**kw)
    later = Later()
    said = []
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clk),
                         connected=helper.connected, later=later,
                         retract=said.append, settle=lambda s: None,
                         wait=lambda s: helper.round_trip(relay, clk.t),
                         now=clk)
    return clk, relay, helper, later, said, sink


def show(tag, res, helper, sink, said, truth):
    print("\n  %s" % tag)
    print("      helper (MODELLED): launches=%d stops=%d viewers_up=%d "
          "streaming=%s" % (helper.launch_attempts, helper.stop_attempts,
                            helper.viewers, helper.streams))
    print("      Jarvis SAYS : %r" % res.spoken)
    print("      landed=%s held=%s   deck=%r  verb=%r"
          % (res.landed, res.held, sink.state.live, sink.relay.verb))
    if said:
        print("      RETRACTED   : %r" % said[-1])
    print("      TRUE at that moment? %s" % truth)


# ================================== the round-3 defect, measured after the fix
def test_1_a_viewer_on_a_password_prompt_is_not_a_landing():
    print("\n=== 1. HPCOMPUTER launched it, and it never connected ===")
    print("  ROUND 3 (d4def0b), MEASURED on this same model: landed=True, "
          "held=False,")
    print('  "The Spark\'s screen is on HPCOMPUTER, sir." with nothing on '
          "any monitor,")
    print("  and state.live stayed 'hp-view' so the next genuine cast was "
          "refused as busy.")
    _clk, _relay, helper, _later, said, sink = bench(streams=False)
    res = sink.deliver(cv.view_subject(cv.HPCOMPUTER, at=1000.0))
    show("round 4", res, helper, sink, said,
         "TRUE - it says nothing is cast, and nothing is")
    assert not res.landed and res.held
    assert sink.state.live == "" and sink.relay.verb == cv.VERB_STOP


def test_2_a_viewer_that_is_streaming_still_lands():
    print("\n=== 2. the same round trip, and the desktop is there ===")
    _clk, _relay, helper, _later, said, sink = bench()
    res = sink.deliver(cv.view_subject(cv.HPCOMPUTER, at=1000.0))
    show("round 4", res, helper, sink, said, "TRUE - there is a desktop up")
    assert res.landed and helper.viewers == 1


def test_3_a_cast_that_dies_after_it_was_claimed_is_taken_back_out_loud():
    print("\n=== 3. it landed, and four seconds later it was gone ===")
    _clk, _relay, helper, later, said, sink = bench()
    res = sink.deliver(cv.view_subject(cv.HPCOMPUTER, at=1000.0))
    assert res.landed
    print("\n  at 0.0 s: %r  (deck %r)" % (res.spoken, sink.state.live))
    helper.viewers = 0                       # RustDesk fell over
    fired = later.fire()
    print("  the second look ran %d job(s) at %.1f s"
          % (fired, cv.VIEWER_CONFIRM_S))
    print("  RETRACTED : %r" % (said[-1] if said else None))
    print("  deck after: %r  (round 3 left it 'hp-view' for ever)"
          % sink.state.live)
    assert said == [cv.CAST_GONE_LINE] and sink.state.live == ""


def test_4_the_cases_where_it_cannot_tell():
    print("\n=== 4. no evidence is not a landing ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    blind = cv.HpViewSink(relay=relay, state=cv.ViewState(now=clk), now=clk)
    ok, why = blind.available()
    print("  no probe wired      : available=%s (%s)" % (ok, why))
    res = blind.deliver(cv.view_subject(cv.HPCOMPUTER, at=clk.t))
    print("                        %r" % res.spoken)
    assert not ok and not res.landed and relay.seq == 0

    _clk, relay2, helper, _later, _said, sink = bench()
    sink._connected = lambda: None
    res2 = sink.deliver(cv.view_subject(cv.HPCOMPUTER, at=1000.0))
    print("  probe has no opinion: %r" % res2.spoken)
    print("                        deck=%r verb=%r"
          % (sink.state.live, relay2.verb))
    assert not res2.landed and res2.held and sink.state.live == ""


# ================================= the stop that said it happened and did not
def test_5_a_stop_that_did_not_happen_is_not_a_receipt():
    print("\n=== 5. Stop-Cast could not kill the viewer ===")
    print("  ROUND 3: the empty catch swallowed it, Stop-Cast returned 0,")
    print("  $ok stayed true and the receipt said the stop had happened -- ")
    print("  with the viewer still up on his middle monitor.")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    helper = Helper(stop_works=False)
    helper.viewers = 1                       # one is already up
    seq = relay.set_verb(cv.VERB_STOP)
    helper.round_trip(relay, clk.t)
    print("\n  round 4 (MODELLED): stop attempts=%d viewers still up=%d"
          % (helper.stop_attempts, helper.viewers))
    print("  receipt for seq %d? %s     helper reports %r for seq %d"
          % (seq, relay.acked_through(seq), relay.fail, relay.failseq))
    assert not relay.acked_through(seq)
    assert relay.failed_for(seq) == "stop-failed"
    assert "stop-failed" in cv.HELPER_FAIL_LINES
    print("  and Jarvis can say it: %r" % cv.HELPER_FAIL_LINES["stop-failed"])


def test_6_a_show_does_not_stack_a_second_viewer():
    print("\n=== 6. a show-spark over a viewer that would not close ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    helper = Helper(stop_works=False)
    helper.viewers = 1
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, state=cv.ViewState(now=clk),
                         connected=helper.connected, later=Later(),
                         settle=lambda s: None,
                         wait=lambda s: helper.round_trip(relay, clk.t),
                         now=clk)
    res = sink.deliver(cv.view_subject(cv.HPCOMPUTER, at=clk.t))
    print("\n  launches attempted=%d  viewers up=%d  (round 3 would have "
          "made it 2)" % (helper.launch_attempts, helper.viewers))
    print("  Jarvis SAYS: %r" % res.spoken)
    assert helper.launch_attempts == 0 and helper.viewers == 1
    assert not res.landed and res.held
    assert sink.state.live == ""


# ============================== what no local evidence can reach
def test_7_what_this_still_cannot_prove():
    print("\n=== 7. the honest limit, carried forward unchanged ===")
    print("  The probe proves a live socket to that host on a")
    print("  screen-sharing port, carrying a stream. It does NOT prove:")
    print("    * that a window is VISIBLE;")
    print("    * that it is on the monitor he expects;")
    print("    * that it is not in front of what he was reading.")
    print("  Nothing local can. That step is his to look at once.")
    print("  And the byte counter is the SERVING PROCESS's whole output,")
    print("  not this socket's -- /proc has no per-socket counter and")
    print("  reading one means opening a netlink socket, which this lane")
    print("  does not do. It cannot say yes to a machine that is not")
    print("  connected; it could say yes to the wrong stream from the")
    print("  right machine.")
