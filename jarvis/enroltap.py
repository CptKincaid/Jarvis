"""The rendezvous that lets enrolment read frames from the preview's camera
without opening a second one.

THE PROBLEM THIS SOLVES. V4L2 capture is exclusive. While the console's
preview is running it holds /dev/video0, so ``scripts/face_enrol.py`` cannot
open the camera at all; and if Jarvis is stopped to free the device, the
sensing policy then refuses on the grounds that the app is offline. Both
doors were locked on 2026-09-03 and the only way through was to kill Jarvis,
enrol, and restart -- a dance he asked to be rid of.

WHY A TAP AND NOT A HANDOVER. The preview already owns a GATED ``CameraFeed``
(jarvis/campreview.py), which is the object the curfew, offline mode and
``SensingPolicy`` all reach. Anything that opened its own device would be a
second lens outside that gate: ``SensingPolicy.attach`` replaces devices by
NAME, so a second feed registered as the camera would silently take the
curfew away from the preview's, and one registered under any other name would
be governed by the offline switch alone with the 21:00-07:00 curfew never
reaching it. So enrolment does not get a device. It gets a frame the preview
had already captured, on the enrolment thread, through this object.

PULL, NOT PUSH, AND A SINGLE SLOT. ``read()`` is the enrolment thread saying
"I want the next one"; ``offer()`` is the capture thread checking a flag that
is unset almost always and returning in microseconds when it is. There is no
queue and no ring buffer, deliberately:

* A QUEUE WOULD BE A PILE OF HIS FACE. The privacy contract of enrolment is
  that no frame is displayed, saved, described or written, and the smallest
  honest reading of that is that at most one frame exists outside the capture
  loop at any moment. One slot, cleared on take, cleared on deny, cleared on
  abort, cleared on release.
* A QUEUE WOULD ALSO BE STALE. Enrolment asks at most ~3 times a second
  against 7.5-15 fps delivered; a buffered frame is a picture of where his
  head WAS, judged against the station he is in now.

THE DENY IS THE PRIVACY EDGE, AND IT IS ONE-WAY. When the preview lets go of
its pipeline -- sensing said no, the curfew started, he said "offline mode",
the worker stopped -- ``deny()`` drops any pending frame and makes this tap
dead for good: the blocked reader wakes with ``(False, None)``, and so does
every later read. ``faceenrol.run_enrolment`` already treats a False read as
fatal and does not retry, so a deny ends the run rather than stalling it. A
tap is never revived; a new run builds a new one.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from jarvis.logs import get_logger

log = get_logger("enroltap")

# How long ``read()`` waits for the capture thread to hand one over. Sized
# against the preview's own rate: at the configured floor of 7.5 fps a frame
# is due every 133 ms, so 2 s is roughly fifteen missed frames -- long enough
# that a slow detector or a re-opened device is not mistaken for a dead one,
# short enough that a wedged capture thread cannot hold an enrolment (and the
# lens) open indefinitely.
READ_TIMEOUT_S = 2.0

# How many times read() waits READ_TIMEOUT_S before calling the source
# dead. Three is ~6 s of patience: far longer than any frame interval
# this preview delivers (268 ms at the 3.7 fps measured 2026-09-04, 133 ms
# at the configured floor), and still short enough that a genuinely
# wedged capture thread cannot hold the lens open through a whole run.
READ_ATTEMPTS = 3

# What a reader is told when nobody ever denied it and no frame arrived.
TIMED_OUT = "the preview stopped delivering frames"


class FrameTap:
    """A one-shot rendezvous for ONE frame at a time.

    ``read()`` is cv2.VideoCapture's contract -- ``(ok, frame)`` -- which is
    exactly what ``faceenrol.run_enrolment`` takes as its ``source``, so the
    in-app run drives the same station loop as the CLI with no second copy.
    """

    def __init__(self, timeout_s: float = READ_TIMEOUT_S):
        self.timeout_s = float(timeout_s)
        self._want = threading.Event()
        self._ready = threading.Event()
        self._dead = threading.Event()
        self._lock = threading.Lock()
        self._slot = None
        # Why the tap died, in the words the run will repeat to him. A plain
        # attribute rather than a property: ``run_enrolment`` reads it
        # through getattr on a source it does not otherwise know.
        self.reason = ""
        # Counters, so a log line and a test can say what happened without
        # anything having to look at a frame.
        self.reads = 0
        self.offers = 0
        self.handed = 0
        self.timeouts = 0

    # ------------------------------------------------------------ reading
    @property
    def dead(self) -> bool:
        return self._dead.is_set()

    def read(self, timeout: Optional[float] = None) -> Tuple[bool, object]:
        """The next frame the preview captures, or ``(False, None)``.

        Called from the ENROLMENT thread and from nowhere else. The frame
        comes back by reference and the slot is cleared in the same locked
        step, so this object is holding nothing the moment it returns.

        A TIMEOUT IS NOT A DEAD SOURCE, and the difference matters because
        ``faceenrol.run_enrolment`` treats the first not-ok read as FATAL
        with no retry. One slow moment -- a GC pause, the preview reopening
        its device, this box under agent load -- would otherwise end a
        two-minute run he is sitting through. Only the tap can tell the two
        apart, so the retry lives here and the caller keeps its simple
        contract. A DENY still returns immediately: that one really is fatal.
        """
        self.reads += 1
        wait = self.timeout_s if timeout is None else float(timeout)
        # An explicit timeout means a caller that wants exactly one attempt
        # (the tests, and anything polling); the default is the enrolment
        # path, which wants patience.
        attempts = 1 if timeout is not None else READ_ATTEMPTS
        for attempt in range(1, attempts + 1):
            if self._dead.is_set():
                return False, None
            # Order matters: _ready is cleared BEFORE the want is announced,
            # or a very fast capture thread can set _ready between the two and
            # have it cleared out from under it -- which costs this read its
            # frame and the next one a stale wake.
            self._ready.clear()
            self._want.set()
            got = self._ready.wait(wait)
            with self._lock:
                # _want is cleared in the SAME locked step that takes the
                # frame. It used to be cleared after the lock was released,
                # and an offer() arriving in that window wrote into a slot the
                # reader had already passed -- so the frame was not lost, it
                # was handed over one interval STALE on the next read. During
                # a station that is a sample of him mid-move, not held still.
                frame, self._slot = self._slot, None
                self._want.clear()
            if self._dead.is_set():
                return False, None
            if got and frame is not None:
                self.handed += 1
                return True, frame
            self.timeouts += 1
            if attempt < attempts:
                # Logged, never silent: a silent retry is how a degrading
                # camera looks healthy right up to the moment it is not.
                log.info("enroltap: no frame in %.1fs (attempt %d of %d), "
                         "retrying", wait, attempt, attempts)
        # Out of patience. NOT fatal on its own -- the caller decides -- but
        # it must not be reported as a frame, and a reason is set so a caller
        # that does give up has something true to say.
        self.reason = self.reason or TIMED_OUT
        return False, None

    # ------------------------------------------------------------ offering
    def offer(self, frame) -> bool:
        """Hand ``frame`` over if somebody is waiting. True if it was taken.

        Called from the PREVIEW'S CAPTURE THREAD, inside its grab, and it is
        on that thread's critical path -- so the common case (nobody is
        enrolling) is one Event read and a return, with no lock taken.
        """
        self.offers += 1
        if self._dead.is_set() or not self._want.is_set():
            return False
        with self._lock:
            if self._dead.is_set() or not self._want.is_set():
                return False
            self._slot = frame
        # Clearing _want here rather than in the reader is what makes this
        # latest-wins with a single producer: the next offer sees no want and
        # returns at once instead of overwriting a frame nobody has taken.
        self._want.clear()
        self._ready.set()
        return True

    # -------------------------------------------------------------- ending
    def deny(self, reason: str) -> None:
        """The preview gave the camera back. This tap is over.

        Wired into ``campreview.PreviewWorker._close_pipeline`` -- the one
        point every handback converges on -- so the sensing deny, the curfew
        edge, ``stop()`` and the loop's own exit all reach it without four
        copies of this call.
        """
        self._end(reason)

    def abort(self, reason: str) -> None:
        """He said stop, or the run failed. Same mechanics, different voice."""
        self._end(reason)

    def _end(self, reason: str) -> None:
        was = self._dead.is_set()
        with self._lock:
            # THE PENDING FRAME DIES HERE. A tap that kept its slot after a
            # deny would be a picture of him surviving the exact event whose
            # entire meaning is "no more pictures".
            self._slot = None
            if not was:
                self.reason = str(reason or "")
        self._dead.set()
        self._want.clear()
        # Last, and always: a reader blocked in wait() must come back even
        # though there is nothing for it.
        self._ready.set()
        if not was:
            log.info("enrol tap closed: %s", self.reason or "no reason given")

    def release(self) -> None:
        """Done with it. Drops any frame without claiming a reason.

        Called in the run's ``finally``. Separate from ``deny`` because a
        clean finish is not a denial and must not log like one -- but the
        slot is emptied just as hard, because a live object still holding a
        frame after the run is exactly what the privacy contract forbids.
        """
        with self._lock:
            self._slot = None
        self._dead.set()
        self._want.clear()
        self._ready.set()

    # -------------------------------------------------------------- report
    def numbers(self) -> dict:
        """Counts only -- what a log line or a test may say about a run."""
        return {"reads": int(self.reads), "offers": int(self.offers),
                "handed": int(self.handed), "timeouts": int(self.timeouts),
                "dead": bool(self._dead.is_set())}
