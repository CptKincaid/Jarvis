"""ATTACK on the claim: can a cast say it landed when nothing landed?

Every launcher, every wait and every probe is an injected recorder.
NOTHING HERE STARTS A RUSTDESK SESSION, opens a process, a socket, a
camera or a capture device, and nothing touches the running Jarvis.
"""
from __future__ import annotations

import pathlib
import re

from jarvis.cast import CastResult, CastSubject
from jarvis import castview as cv


class Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def subj(at=1000.0):
    return cv.view_subject(cv.SPARK, at=at)


def hp_subj(at=1000.0):
    return cv.view_subject(cv.HPCOMPUTER, at=at)


def say(res):
    return "landed" if res.landed else ("held" if res.held else "neither")


def report(name, res, truth):
    print("  %-44s\n      Jarvis SAYS : %r\n      landed=%s held=%s\n"
          "      TRUE at that moment? %s"
          % (name, res.spoken, res.landed, res.held, truth))
    return res


# =============================== the Windows helper, as SHIPPED ============
class ShippedHelper:
    """A faithful model of scripts/windows/cast-poll.ps1.

    THE ORDER IS THE POINT and it is copied from the file:

        if ($newSeq -ne $seq) {
          $seq = $newSeq                     <-- the receipt is committed HERE
          if ($verb -eq 'show-spark') { ... Start-Process ... }
        }

    with ``$ErrorActionPreference = 'SilentlyContinue'`` at the top.  So the
    sequence Jarvis reads back as a RECEIPT is committed BEFORE the launch
    is attempted and WHETHER OR NOT it succeeds.
    """

    def __init__(self, relay, *, launch_works=True):
        self.relay = relay
        self.seq = -1
        self.launch_works = launch_works
        self.windows_up = 0
        self.launch_attempts = 0

    def poll_once(self, at=None):
        self.relay.note(mon=1, layout="0,1920,1920,1920", at=at)
        out = self.relay.poll(self.seq, timeout_s=0.0)
        new, verb = int(out["seq"]), str(out["verb"])
        if new != self.seq:
            self.seq = new                       # <-- exactly as shipped
            if verb == cv.VERB_SHOW:
                self.launch_attempts += 1
                if self.launch_works:
                    self.windows_up += 1
            elif verb == cv.VERB_STOP:
                self.windows_up = 0
        return out

    def round_trip(self, at=None):
        """The real shape of it: one poll TAKES the verb, the next poll
        QUOTES it back.  The script sleeps 200 ms between them."""
        self.poll_once(at=at)
        self.poll_once(at=at)


def hp_sink(relay, state=None, now=None):
    return cv.HpViewSink(relay=relay, state=state,
                         now=now or Clock())


# ================================================== 1. the viewer dies at once
def test_1_the_viewer_is_killed_the_instant_it_is_launched():
    print("\n=== 1. kill the viewer immediately after launch ===")
    alive = {"up": False}          # Popen forked, then the child died
    launched = []
    stopped = []
    sink = cv.SparkViewSink(launch=lambda h: launched.append(h),
                            stop=lambda: stopped.append(1),
                            alive=lambda: alive["up"],
                            settle=lambda s: None, now=Clock())
    res = report("viewer dead at the settle probe", sink.deliver(subj()),
                 "TRUE - it says nothing is cast, and nothing is")
    assert not res.landed and res.held
    assert launched == [cv.HPCOMPUTER_HOST] and stopped == [1]
    assert sink.state.live == "", "the deck must be given back"


def test_1b_the_viewer_dies_just_AFTER_the_settle_window():
    """The floor the fixer names: a viewer alive at 0.7 s and gone at 3 s."""
    print("\n=== 1b. the viewer dies a moment after the settle probe ===")
    alive = {"up": True}
    sink = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                            alive=lambda: alive["up"],
                            settle=lambda s: None, now=Clock())
    res = report("viewer alive at 0.7 s, dead at 3 s", sink.deliver(subj()),
                 "TRUE when spoken; FALSE 2.3 s later, and never revisited")
    alive["up"] = False
    print("      after it dies: state.live=%r, nothing re-checks it, "
          "nothing retracts the line" % sink.state.live)
    assert res.landed and sink.state.live == "spark-view"


def test_1c_a_viewer_that_is_UP_but_never_CONNECTED():
    """THE ONE THE MODULE'S OWN DOCSTRING PREDICTS.  ``SparkViewSink`` says:
    without direct-IP access and an unattended password on HPCOMPUTER, "a
    direct-IP connect waits for someone to accept it on the Windows side and
    the cast silently does nothing".  A RustDesk viewer sitting on a
    password prompt is a LIVE PROCESS.  ``app._rustdesk_alive`` is
    ``Popen.poll() is None`` and answers yes."""
    print("\n=== 1c. the viewer is up, but showing a password prompt ===")
    sink = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                            alive=lambda: True,      # the process lives
                            settle=lambda s: None, now=Clock())
    res = report("process alive, no desktop on screen", sink.deliver(subj()),
                 "FALSE - HPCOMPUTER's screen is NOT on the Spark")
    assert res.landed, "it claims a landing off process liveness alone"


# ============================================== 2. the helper never polls
def test_2_the_helper_never_polls_at_all():
    print("\n=== 2. HPCOMPUTER's helper never polls ===")
    relay = cv.CastRelay(now=Clock())
    sink = hp_sink(relay)
    res = report("helper never seen", sink.deliver(hp_subj()),
                 "TRUE - it refuses before parking anything")
    assert not res.landed and res.held
    assert relay.seq == 0, "nothing may be queued for a machine not there"


def test_2b_the_helper_polled_once_and_then_stopped():
    """``alive()`` is a MINUTE wide.  A helper that polled 59 s ago and then
    died is 'alive'.  This is the case the receipt was added for."""
    print("\n=== 2b. the helper polled 59 s ago and then died ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    clk.t += 59.0
    waits = []
    sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                         wait=lambda s: waits.append(s), now=clk)
    res = report("alive() says yes, nobody is home", sink.deliver(hp_subj()),
                 "TRUE - no receipt, so it holds")
    assert not res.landed and res.held
    assert relay.verb == cv.VERB_STOP, relay.verb
    assert waits == [2.0], waits


# ================================================ 3. the helper polls stale
def test_3_the_helper_polls_but_quotes_an_OLD_sequence():
    print("\n=== 3. the helper polls, quoting a stale sequence ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    relay.set_verb(cv.VERB_STOP)          # seq 1, and the helper acks it
    relay.poll(1, timeout_s=0.0)
    assert relay.acked == 1

    def stale_wait(_s):
        relay.poll(1, timeout_s=0.0)      # it keeps quoting seq 1
        relay.poll(1, timeout_s=0.0)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, wait=stale_wait, now=clk)
    res = report("helper stuck on an older seq", sink.deliver(hp_subj()),
                 "TRUE - a stale quote is not a receipt")
    assert not res.landed and res.held


# =========================================== 4. the relay accepts and drops
def test_4_the_relay_accepts_the_verb_and_windows_drops_it():
    """THE SHIPPED SCRIPT COMMITS THE RECEIPT BEFORE IT LAUNCHES ANYTHING,
    and swallows the failure.  RustDesk not installed at the hard-coded
    path, a blocked Start-Process, a locked session: the ack still comes
    back and Jarvis calls it a landing."""
    print("\n=== 4. HPCOMPUTER receives the verb and the launch fails ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    helper = ShippedHelper(relay, launch_works=False)
    helper.poll_once(at=clk.t)            # it is at the door
    sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                         wait=lambda s: helper.round_trip(at=clk.t), now=clk)
    res = report("verb received, Start-Process silently failed",
                 sink.deliver(hp_subj()),
                 "FALSE - the Spark's screen is NOT on HPCOMPUTER")
    print("      helper: %d launch attempts, %d windows actually up"
          % (helper.launch_attempts, helper.windows_up))
    assert res.landed and helper.windows_up == 0


def test_4b_the_same_round_trip_when_the_launch_DOES_work():
    print("\n=== 4b. the same round trip, launch works ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    helper = ShippedHelper(relay, launch_works=True)
    helper.poll_once(at=clk.t)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                         wait=lambda s: helper.round_trip(at=clk.t), now=clk)
    res = report("verb received and launched", sink.deliver(hp_subj()),
                 "TRUE - there is a window up")
    assert res.landed and helper.windows_up == 1


# ================================================ 5. Windows answers late
def test_5_the_windows_side_answers_after_the_deadline():
    print("\n=== 5. HPCOMPUTER answers AFTER the 2 s deadline ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    helper = ShippedHelper(relay, launch_works=True)
    helper.poll_once(at=clk.t)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                         wait=lambda s: None, now=clk)      # nothing arrives
    res = report("no receipt inside 2 s", sink.deliver(hp_subj()),
                 "TRUE at that moment - nothing is up yet")
    assert not res.landed and res.held
    print("      relay verb after the give-up: %r (seq %d)"
          % (relay.verb, relay.seq))
    # ...and now the slow helper finally answers.
    helper.poll_once(at=clk.t)
    print("      the slow helper then polls: windows up = %d"
          % helper.windows_up)
    helper.poll_once(at=clk.t)
    print("      and on its NEXT poll it takes the stop: windows up = %d"
          % helper.windows_up)
    assert helper.windows_up == 0


def test_5b_an_unrelated_receipt_ends_the_wait_early():
    """``await_ack`` waits ONCE on a shared event.  Any other fresh receipt
    -- the ack for the PREVIOUS cast's stop, say -- wakes it, and the wait
    is then spent.  It reports a hold for a helper that was about to
    answer."""
    print("\n=== 5b. an unrelated receipt spends the whole ack wait ===")
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    relay.set_verb(cv.VERB_STOP)                    # seq 1, unacked
    calls = []

    def wait(_s):
        calls.append(_s)
        relay.poll(1, timeout_s=0.0)                # the OLD stop is acked
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, wait=wait, now=clk)
    res = report("woken by the previous verb's receipt",
                 sink.deliver(hp_subj()),
                 "TRUE that nothing landed - but the helper was answering")
    assert not res.landed and relay.acked == 1
    print("      the wait was entered %d time(s); the receipt it woke on "
          "was for seq 1, not seq %d" % (len(calls), 2))


# ======================================= landed and held cannot be inferred
def test_landed_and_held_are_never_both_and_never_inferred():
    print("\n=== landed / held ===")
    try:
        CastResult(landed=True, held=True, spoken="x")
        raise AssertionError("both were accepted")
    except ValueError as exc:
        print("  both at once: refused (%s)" % exc)
    r = CastResult(landed=False, held=False, spoken="x")
    print("  NEITHER is representable: landed=%s held=%s -> cast() reports "
          "%r" % (r.landed, r.held, say(r)))
    # every sink result in this file
    print("  every result measured above kept them disjoint.")


# ============================== nothing here can open a real thing
def test_no_test_can_open_a_camera_or_a_rustdesk_session():
    """Source-level, over the files this attack drives."""
    print("\n=== can a test open a real viewer or lens? ===")
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "jarvis" / "castview.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    for word in ("subprocess", "Popen(", "socket", "requests", "urllib",
                 "cv2", "VideoCapture", "rustdesk", "os.system", "exec("):
        assert word not in code, word
    print("  jarvis/castview.py holds no transport (code, not prose): OK")
    sink = cv.SparkViewSink(now=Clock())
    ok, why = sink.available()
    print("  SparkViewSink with nothing wired: available=%s (%s)" % (ok, why))
    assert not ok
    res = sink.deliver(subj())
    print("  and it %s: %r" % (say(res), res.spoken))
    assert not res.landed
    probe_only = cv.SparkViewSink(launch=lambda h: None, now=Clock())
    ok2, why2 = probe_only.available()
    print("  launcher but NO aliveness probe: available=%s (%s)" % (ok2, why2))
    assert not ok2
    # the default settle is a real sleep -- a test that forgets to inject it
    # sleeps for 0.7 s of wall clock.
    d = cv.SparkViewSink(launch=lambda h: None, alive=lambda: True,
                         now=Clock())
    print("  default settle callable: %r (a real sleep if not injected)"
          % d._settle)
    body = (root / "tests" / "test_castview.py").read_text()
    n = len(re.findall(r"SparkViewSink\(", body))
    m = len(re.findall(r"settle=", body))
    print("  tests/test_castview.py builds SparkViewSink %d times and "
          "injects settle= %d times" % (n, m))
