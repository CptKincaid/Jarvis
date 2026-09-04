"""The frame tap (jarvis/enroltap.py) on its own.

WHAT THESE TESTS ARE ALLOWED TO DO. Every "frame" here is a small array this
process made up, or a bare sentinel string; nothing opens a device and nothing
looks at pixels. The properties under test are all structural -- is the slot
empty, did the blocked reader wake, did the deny stick -- which is exactly the
shape the privacy rule demands: the tap has to be provable from state, not
from looking at what went through it.

THE ONE THING THIS FILE IS REALLY FOR. The tap is the only object in Jarvis
that holds a full camera frame outside the capture loop. The contract is that
it holds at most one, for at most one read cycle, and that every exit path
empties it. So most of these tests assert on ``tap._slot`` being None, which
is deliberate white-box testing: the invariant IS the private field.
"""
import threading
import time

import numpy as np
import pytest

from jarvis import enroltap as et


def frame(value=7):
    """A frame-shaped array this process generated. Not a picture."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


# ---------------------------------------------------------- the handover
def test_an_offer_with_nobody_waiting_is_dropped_on_the_floor():
    """The capture thread's common case. With no enrolment running the tap
    must cost the preview nothing and, above all, must not START holding a
    frame -- a tap that buffered "just in case" would be a picture of him
    sitting in memory for as long as the app runs."""
    tap = et.FrameTap()
    assert tap.offer(frame()) is False
    assert tap._slot is None
    assert tap.numbers()["handed"] == 0


def test_a_read_gets_the_offered_frame_exactly_once():
    tap = et.FrameTap(timeout_s=1.0)
    sent = frame(9)
    threading.Thread(target=lambda: (time.sleep(0.02), tap.offer(sent)),
                     daemon=True).start()
    ok, got = tap.read()
    assert ok is True
    assert got is sent          # by reference: no copy is made of a frame
    # ...and it is gone. A second read with nobody offering must time out
    # rather than hand the same frame over twice.
    assert tap._slot is None
    ok2, got2 = tap.read(timeout=0.05)
    assert (ok2, got2) == (False, None)


def test_the_slot_is_empty_the_instant_the_read_returns():
    """THE CORE PRIVACY INVARIANT. The frame lives in this object for one
    read cycle and not one microsecond longer."""
    tap = et.FrameTap(timeout_s=1.0)
    threading.Thread(target=lambda: (time.sleep(0.02), tap.offer(frame())),
                     daemon=True).start()
    ok, _got = tap.read()
    assert ok is True
    assert tap._slot is None
    for value in vars(tap).values():
        assert not isinstance(value, np.ndarray)


def test_a_second_offer_before_the_read_does_not_pile_up():
    """LATEST-WINS WITH ONE SLOT, and no queue. The second offer finds no
    want outstanding and is dropped, so two frames can never be resident."""
    tap = et.FrameTap(timeout_s=1.0)
    ready = threading.Event()

    def reader():
        ready.set()
        tap.read()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    ready.wait(1.0)
    time.sleep(0.05)
    assert tap.offer(frame(1)) is True
    assert tap.offer(frame(2)) is False        # nobody is asking any more
    thread.join(timeout=1.0)
    assert tap._slot is None


# --------------------------------------------------------------- the deny
def test_deny_wakes_a_blocked_reader_with_false_and_a_reason():
    """The sensing edge. run_enrolment treats a False read as fatal, so this
    is what ends a run when the curfew starts mid-capture -- and it must not
    take the two-second timeout to do it."""
    tap = et.FrameTap(timeout_s=30.0)
    started = threading.Event()
    out = []

    def reader():
        started.set()
        out.append(tap.read())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    started.wait(1.0)
    time.sleep(0.05)
    t0 = time.monotonic()
    tap.deny("sensing said no")
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert out == [(False, None)]
    assert tap.reason == "sensing said no"
    # Woken, not timed out: the tap's own timeout is 30 s here.
    assert time.monotonic() - t0 < 2.0


def test_deny_drops_a_frame_that_was_already_pending():
    """A frame offered a moment before the deny must not survive it. The
    whole meaning of a deny is "no more pictures", and one still sitting in
    the slot would be the exception that swallows the rule."""
    tap = et.FrameTap(timeout_s=1.0)
    ready = threading.Event()
    done = threading.Event()

    def reader():
        ready.set()
        # Hold the want open long enough for the offer below to land.
        tap.read(timeout=0.5)
        done.set()

    threading.Thread(target=reader, daemon=True).start()
    ready.wait(1.0)
    time.sleep(0.02)
    tap.offer(frame())
    tap.deny("the camera was handed back")
    done.wait(2.0)
    assert tap._slot is None


def test_a_denied_tap_never_delivers_again():
    """One-way. A tap is not revived; a new run builds a new one -- which is
    what stops a stale enrolment from resuming against a camera that was
    taken away and given back for some other reason."""
    tap = et.FrameTap(timeout_s=0.2)
    tap.deny("stopped")
    assert tap.read() == (False, None)
    assert tap.offer(frame()) is False
    assert tap.read() == (False, None)
    assert tap._slot is None


def test_the_first_reason_is_the_one_that_is_kept():
    """A second deny (stop() after the loop already exited, say) must not
    overwrite the true first cause with a generic one."""
    tap = et.FrameTap()
    tap.deny("the curfew started")
    tap.deny("the preview stopped")
    assert tap.reason == "the curfew started"


def test_abort_from_another_thread_unwinds_a_blocked_read():
    """"Jarvis, stop" arrives on the commander's thread while the enrolment
    thread is blocked here. It must come back promptly, because the run
    cannot release the camera until it does."""
    tap = et.FrameTap(timeout_s=30.0)
    started = threading.Event()
    out = []
    thread = threading.Thread(
        target=lambda: (started.set(), out.append(tap.read())), daemon=True)
    thread.start()
    started.wait(1.0)
    time.sleep(0.05)
    tap.abort("he said stop")
    thread.join(timeout=2.0)
    assert out == [(False, None)]
    assert tap.reason == "he said stop"


# ------------------------------------------------------------ the release
def test_release_empties_the_slot_and_claims_no_reason():
    """A clean finish is not a denial: the run's finally calls this, and it
    must empty the slot just as hard without inventing a failure."""
    tap = et.FrameTap(timeout_s=1.0)
    ready = threading.Event()

    def reader():
        ready.set()
        tap.read(timeout=0.5)

    threading.Thread(target=reader, daemon=True).start()
    ready.wait(1.0)
    time.sleep(0.02)
    tap.offer(frame())
    tap.release()
    assert tap._slot is None
    assert tap.reason == ""
    assert tap.dead is True


def test_a_read_that_nobody_answers_reports_a_timeout_not_a_frame():
    tap = et.FrameTap(timeout_s=0.05)
    assert tap.read() == (False, None)
    assert tap.numbers()["timeouts"] == 1
    assert tap.reason == et.TIMED_OUT


# ------------------------------------------------------------- the report
def test_the_numbers_are_counts_and_nothing_else():
    """What a log line may say about a run. If a frame could reach the log it
    would be through here, so this returns ints and bools only."""
    from jarvis import visionrig as vr
    tap = et.FrameTap(timeout_s=1.0)
    threading.Thread(target=lambda: (time.sleep(0.02), tap.offer(frame())),
                     daemon=True).start()
    tap.read()
    numbers = tap.numbers()
    vr.assert_numbers_only(numbers)
    assert numbers["reads"] == 1 and numbers["handed"] == 1


@pytest.mark.parametrize("finish", ["deny", "abort", "release"])
def test_every_exit_path_leaves_the_slot_empty(finish):
    """The invariant stated once over all three endings, so a fourth ending
    added later without emptying the slot fails here."""
    tap = et.FrameTap(timeout_s=1.0)
    ready = threading.Event()
    threading.Thread(
        target=lambda: (ready.set(), tap.read(timeout=0.4)),
        daemon=True).start()
    ready.wait(1.0)
    time.sleep(0.02)
    tap.offer(frame())
    getattr(tap, finish)(*(() if finish == "release" else ("done",)))
    assert tap._slot is None
    assert tap.dead is True
