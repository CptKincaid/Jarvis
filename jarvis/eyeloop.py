"""The producer that fills the eye, so HIS NUMBER ONE SIGNAL can vote.

His ruling, 2026-09-05: "The camera should be the number one understanding
for if I'm in. Followed by phone connection then sensor." The voter obeys
that -- ``presencevote`` gives a camera that NAMES him the vote outright in
all 27 cells -- and the leg has been DARK since the day it shipped, because
nothing on the tree ever assigned ``services.camera_feed`` and nothing ever
filled a reading for ``app._eye_leg`` to read. The app said so itself at
every boot. This module is the missing half.

WHAT IT PRODUCES, AND WHAT IT CANNOT PRODUCE. A count of faces, a label, a
score and an age -- an ``eye.Attention``, whose fields cannot hold a pixel.
The frame goes ``feed.capture()`` -> ``detector.detect`` -> rows ->
``FaceIdentifier.identify`` -> ``(label, score)``, and is dropped inside the
same call. Nothing is written, nothing is kept, nothing is thumbnailed and
no frame is ever handed to a consumer: ``Eye.publish`` refuses anything that
is not an ``Attention`` BY TYPE, so this is a property of the code and not
of a review. There is no cv2, no imaging library and no encoder in this file, and
``tests/test_eyeloop.py`` greps the source to keep it that way.

IT NEVER FREE-RUNS. The thread sleeps on an Event and a caller ARMS it:

  * the fabric naming his office (``app._on_room_changed``) -- the lens is
    in the office and his flat is a corridor, so that is the moment there
    is something to look at;
  * the two cells his rule 1 cannot reach today -- a room reading occupied
    with the phone silent (cell 6) or unaskable (cell 9). Those are exactly
    the votes that went wrong on 2026-09-05 and again on 09-06, and they
    are the ones where a lens ends the argument.

Between bursts the device is closed and the lamp is dark. ``camera.idle_fps``
exists in his config and is used NOWHERE in the tree; it stays that way. A
lens lit around the clock in his office is his decision to make, not a
fixer's.

THE CURFEW IS TAKEN FOR HIM, IN THE RESTRICTIVE DIRECTION ONLY. His ruling
of 2026-09-02 was "camera OK in his OFFICE, OFF 21:00-07:00, ALL-LOCAL", and
``camera.curfew_enabled`` / ``camera.curfew_start`` / ``camera.curfew_end``
are unset on his box, so this module applies 21:00-07:00 by default and
publishes BLIND inside it -- the leg does not vote and the device is never
opened. It does NOT read or write ``sensing.curfew.*``: he has explicitly
set that one to disabled and changing a privacy control on his behalf is
the wrong move even when the code default agrees with his ruling. The
sensing policy still governs the device open underneath this, which is two
independent gates on one rule and deliberate.

ONE DEVICE, SHARED WITH THE CONSOLE. When the preview is capturing, this
takes ZERO frames of its own and reads the numbers off the shot the console
already has (``PreviewFace`` carries ``name``, ``id_score``, ``id_ran`` --
all numbers). When it is not, the burst holds the feed by name and drops
it, and only the LAST holder closes: a close from here would bump the gate
epoch under an in-flight preview read.

ALL LOCAL. Nothing in this file, or anything it reaches, opens a socket.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable, Optional

from jarvis.eye import Attention, PRESENCE_MAX_AGE_S
from jarvis.logs import get_logger
from jarvis.visionrig import DETECT_COLS, IDX_SCORE

log = get_logger("eyeloop")

# HIS 2026-09-02 RULING, as the default whenever camera.curfew* is unset.
# DEFAULT TAKEN FOR HIM -- flagged in the report, and reversible from his
# config without touching this file.
DEFAULT_CURFEW_START = (21, 0)
DEFAULT_CURFEW_END = (7, 0)

# One burst. 3.0 s at the armed tier is ~24 frames; his own preview line of
# 02:18:13 measured grab 1.8 ms + detect 6.1 ms (and the 10:48 boot run 4.2
# + 7.0), with ArcFace embed at 5.2 ms, so a burst costs about a quarter of
# a second of CPU on two threads. The face lane is CPU-ONLY by construction
# (jarvis/facedetect.py pins CPUExecutionProvider), so the GPU cost is zero.
BURST_S = 3.0
# ``camera.armed_fps`` -- declared in his config since the vision lane
# landed and, until now, executed NOWHERE. Treat 8.0 as a starting point
# and report the DELIVERED rate, not the requested one: the preview's own
# history (7.5 requested, 3.7-13.0 delivered) is the cautionary tale.
DEFAULT_ARMED_FPS = 8.0
# What a caller on the presence thread may spend waiting for a look. The
# vote is worth ~1.5 s and no more; past that the leg answers BLIND, which
# is the honest answer and today's behaviour.
WAIT_S = 1.5


def _num(cfg, key: str, default: float) -> float:
    get = getattr(cfg, "get", None)
    if not callable(get):
        return float(default)
    try:
        value = get(key, default)
        out = float(default if value is None else value)
    except Exception:  # noqa: BLE001 - a bad key is not a reason to look
        return float(default)
    return out


def _hhmm(value, default):
    """``"21:00"`` -> ``(21, 0)``. Anything unusable is the DEFAULT and not
    "no curfew": a typo must not quietly delete a privacy control, which is
    the rule ``sensing.curfew`` already follows."""
    try:
        hh, mm = str(value).strip().split(":")
        h, m = int(hh), int(mm)
    except Exception:  # noqa: BLE001
        return default
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return default
    return (h, m)


def _in_window(start, end, now) -> bool:
    """Does ``now`` fall in [start, end)? Wraps midnight."""
    s = start[0] * 60 + start[1]
    e = end[0] * 60 + end[1]
    t = now[0] * 60 + now[1]
    if s == e:
        return False
    return (s <= t < e) if s < e else (t >= s or t < e)


class EyeLoop:
    """Arms, looks, publishes, sleeps. Owns one daemon thread and no socket.

    ``feed`` is the app's ONE ``camera.CameraFeed`` -- the same object
    ``campreview.resolve_feed`` borrows -- so the curfew and offline mode
    reach the lens through it whether the console is up or not.

    The detector and the identifier are built LAZILY, on the first burst.
    Loading ONNX weights at construction would put the whole face lane on
    the startup path of a box that may never arm a burst.
    """

    def __init__(self, feed, *, detector=None, identifier=None, cfg=None,
                 make_detector: Optional[Callable] = None,
                 make_identifier: Optional[Callable] = None,
                 burst_s: float = BURST_S, fps: Optional[float] = None,
                 curfew=None, clock: Optional[Callable] = None,
                 now: Callable[[], float] = time.monotonic,
                 sleep: Optional[Callable[[float], None]] = None,
                 preview: Optional[Callable] = None,
                 on_named: Optional[Callable] = None,
                 name: str = "presence"):
        self.feed = feed
        self.name = str(name)
        self._detector = detector
        self._identifier = identifier
        self._make_detector = make_detector
        self._make_identifier = make_identifier
        self._built = detector is not None or make_detector is None
        self.burst_s = max(0.0, float(burst_s))
        # An explicit rate wins; otherwise ``camera.armed_fps`` -- which has
        # been declared in his config since the vision lane landed and has
        # executed NOWHERE until now, so 8.0 is a starting point and not a
        # measurement.
        if fps is not None:
            self.fps = max(0.5, float(fps))
        elif cfg is not None:
            self.fps = max(0.5, _num(cfg, "camera.armed_fps", DEFAULT_ARMED_FPS))
        else:
            self.fps = DEFAULT_ARMED_FPS
        self.curfew = self._curfew_from(cfg) if curfew is None else curfew
        self._clock = clock or (lambda: (datetime.now().hour,
                                         datetime.now().minute))
        self._now = now
        self._sleep = sleep or time.sleep
        self._preview = preview
        # Called with the LABEL (a string, never a frame) at the end of a
        # burst that named somebody. It is how the arrival settle gets the
        # answer to a burst it armed: the arming call runs on the Tk or
        # fabric thread and must return at once, so the answer has to come
        # back on THIS thread rather than be waited for on that one.
        self._on_named = on_named
        self._arm = threading.Event()
        self._stop = threading.Event()
        self._looked = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # NUMBERS, for status() and the report. No array is ever kept.
        self.bursts = 0
        self.frames = 0            # frames this producer grabbed itself
        self.faces_seen = 0
        self.named = 0
        self.blind_bursts = 0
        self.borrowed = 0          # bursts served off the console's own shot
        self.curfew_skips = 0
        self.arms = 0
        self.waits = 0
        self.wait_timeouts = 0
        self.detect_errors = 0
        self.published = 0
        self.last_faces = 0
        self.last_score = 0.0
        self.last_ms = 0.0
        self.last_reason = "no look yet"
        self._named_this_burst = ""

    # ------------------------------------------------------------ curfew
    @staticmethod
    def _curfew_from(cfg):
        """His 21:00-07:00 ruling unless ``camera.curfew*`` says otherwise.

        Read from the CAMERA keys and never from ``sensing.curfew.*``: his
        sensing curfew is explicitly disabled in his own config and this
        module may only ever ADD a restriction, never remove one.
        """
        if cfg is None:
            return DEFAULT_CURFEW_START, DEFAULT_CURFEW_END
        get = getattr(cfg, "get", None)
        try:
            enabled = get("camera.curfew_enabled", None) if callable(get) else None
        except Exception:  # noqa: BLE001
            enabled = None
        if enabled is False:
            return None
        try:
            start = get("camera.curfew_start", None) if callable(get) else None
            end = get("camera.curfew_end", None) if callable(get) else None
        except Exception:  # noqa: BLE001
            start = end = None
        return (_hhmm(start, DEFAULT_CURFEW_START),
                _hhmm(end, DEFAULT_CURFEW_END))

    def curfew_now(self) -> bool:
        win = self.curfew
        if not win:
            return False
        try:
            return _in_window(win[0], win[1], self._clock())
        except Exception:  # noqa: BLE001 - an unreadable clock is not consent
            log.debug("eyeloop: the curfew clock failed; staying shut",
                      exc_info=True)
            return True

    # ------------------------------------------------------------- arming
    def arm(self, why: str = "") -> bool:
        """Ask for a burst. Returns False when nothing will happen.

        Never blocks and never opens anything itself: the thread does the
        work, so a bus subscriber or the Tk thread can call this.
        """
        if self._stop.is_set():
            return False
        if self.curfew_now():
            self.curfew_skips += 1
            self._publish_dark("inside the camera curfew")
            return False
        self.arms += 1
        if why:
            log.debug("eyeloop: armed (%s)", why)
        self._arm.set()
        return True

    def wait_for_look(self, timeout_s: float = WAIT_S, why: str = "") -> bool:
        """Arm and WAIT, bounded, for the next reading. For the presence
        thread's two ambiguous cells only.

        Bounded on purpose and by a small number: this runs on the presence
        daemon and the vote it improves is worth about a second and a half.
        A timeout is not a failure -- the leg answers BLIND, which is
        exactly today's behaviour.
        """
        self.waits += 1
        self._looked.clear()
        if not self.arm(why or "an ambiguous vote"):
            return False
        if self.running:
            got = self._looked.wait(max(0.0, float(timeout_s)))
        else:
            # No thread (a test, or a build that only wants the leg): do the
            # burst here rather than wait for nobody.
            self._arm.clear()
            self.burst()
            got = True
        if not got:
            self.wait_timeouts += 1
        return got

    # ------------------------------------------------------------ thread
    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        if self.feed is None:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="eyeloop",
                                        daemon=True)
        self._thread.start()
        log.info("eyeloop: the camera leg's producer is up -- bursts of "
                 "%.1f s at %.1f fps when armed, nothing in between%s",
                 self.burst_s, self.fps,
                 ("; curfew %02d:%02d-%02d:%02d" %
                  (self.curfew[0][0], self.curfew[0][1],
                   self.curfew[1][0], self.curfew[1][1])) if self.curfew
                 else "; no curfew")
        return True

    def stop(self) -> None:
        self._stop.set()
        self._arm.set()          # wake the thread so it can see the stop
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        try:
            if self.feed is not None:
                self.feed.drop(self.name)
        except Exception:  # noqa: BLE001 - teardown
            log.debug("eyeloop: the feed would not release", exc_info=True)

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._arm.wait()
            if self._stop.is_set():
                return
            self._arm.clear()
            try:
                self.burst()
            except Exception:  # noqa: BLE001 - a burst may not end the thread
                log.exception("eyeloop: the burst failed")

    # ------------------------------------------------------------- a burst
    def burst(self) -> int:
        """One look. Returns the number of frames this producer grabbed.

        Zero is not a failure: the console may have been capturing, in
        which case the numbers came off its shot and no second reader ever
        touched the device.
        """
        if self.curfew_now():
            self.curfew_skips += 1
            self._publish_dark("inside the camera curfew")
            return 0
        feed = self.feed
        if feed is None:
            self._publish_dark("no camera feed")
            return 0
        if self._from_preview():
            self.bursts += 1
            self.borrowed += 1
            self._tell_named()
            return 0
        self._build_once()
        began = self._now()
        got = 0
        feed.hold(self.name)
        try:
            step = 1.0 / self.fps
            # HARD-BOUNDED, both ways: a wall-clock deadline AND a frame
            # ceiling, so a device that returns instantly cannot spin and a
            # device that hangs cannot hold the lens open.
            ceiling = int(max(1, min(64, round(self.burst_s * self.fps) + 1)))
            deadline = began + self.burst_s
            for _ in range(ceiling):
                if self._stop.is_set():
                    break
                frame = feed.capture()
                if frame is None:
                    # Denied, no device, an unplug, a failed grab, or offline
                    # set mid-grab. All five read the same way and the safe
                    # reading of all five is "I did not look".
                    self._publish_dark("no frame came back")
                else:
                    got += 1
                    self._look(frame)
                    frame = None
                if self._now() >= deadline:
                    break
                self._sleep(step)
        finally:
            self._release(feed)
        self.bursts += 1
        self.frames += got
        self.last_ms = round((self._now() - began) * 1000.0, 1)
        if got == 0:
            self.blind_bursts += 1
        self._tell_named()
        return got

    def _release(self, feed) -> None:
        """Give the claim back -- and DO NOT close if the console started
        capturing while we were looking. Closing bumps the gate epoch and
        the preview's in-flight handle would refuse its next read."""
        try:
            feed.drop(self.name, close=not self._preview_running())
        except Exception:  # noqa: BLE001 - a stub feed in a test
            log.debug("eyeloop: the feed would not release", exc_info=True)

    def _build_once(self) -> None:
        if self._built:
            return
        self._built = True
        for what, make, slot in (("detector", self._make_detector, "_detector"),
                                 ("identifier", self._make_identifier,
                                  "_identifier")):
            if make is None:
                continue
            try:
                setattr(self, slot, make())
            except Exception:  # noqa: BLE001 - absence is not a crash
                log.info("eyeloop: no face %s; the leg looks but cannot %s",
                         what, "count" if what == "detector" else "name",
                         exc_info=True)
                setattr(self, slot, None)

    # ------------------------------------------------------------ one frame
    def _look(self, frame) -> None:
        """Frame in, NUMBERS out. The frame does not survive this call."""
        det = self._detector
        if det is None:
            self._publish_dark("no face detector on this box")
            return
        try:
            rows = det.detect(frame)
        except Exception:  # noqa: BLE001 - a broken model is not an empty room
            self.detect_errors += 1
            log.debug("eyeloop: the detector failed", exc_info=True)
            self._publish_dark("the detector failed")
            return
        rows = [] if rows is None else list(rows)
        faces = len(rows)
        label, score, conf = "", 0.0, 0.0
        best = self._best(rows)
        if best is not None:
            conf = self._conf(best)
            ident = self._identifier
            if ident is not None:
                try:
                    label, score = ident.identify(frame, best)
                except Exception:  # noqa: BLE001 - identity is never fatal
                    log.debug("eyeloop: identity failed", exc_info=True)
                    label, score = "", 0.0
        self.faces_seen += faces
        self.last_faces = faces
        self.last_score = float(score or 0.0)
        if label:
            self.named += 1
        self.last_reason = "looked"
        if label:
            self._named_this_burst = str(label)
        self._publish(Attention(faces=faces, identity=str(label or ""),
                                id_score=float(score or 0.0), conf=conf))

    @staticmethod
    def _best(rows):
        """The strongest detection -- ONE identity per look, the same
        one-subject rule the preview follows and for the same reason."""
        best, top = None, None
        for row in rows:
            try:
                score = float(row[IDX_SCORE]) if len(row) == DETECT_COLS else None
            except Exception:  # noqa: BLE001 - a strange row is not a face
                score = None
            if score is None:
                continue
            if top is None or score > top:
                best, top = row, score
        return best

    @staticmethod
    def _conf(row) -> float:
        try:
            return round(float(row[IDX_SCORE]), 4)
        except Exception:  # noqa: BLE001
            return 0.0

    # -------------------------------------------------------- the console
    def _preview_running(self) -> bool:
        worker = self._worker()
        return worker is not None and bool(getattr(worker, "running", False))

    def _worker(self):
        get = self._preview
        if get is None:
            return None
        try:
            return get()
        except Exception:  # noqa: BLE001
            return None

    def _from_preview(self) -> bool:
        """Take the numbers off the console's own shot, or return False.

        Zero extra frames and zero extra detection while the pane is up,
        and -- the point -- no second reader on the v4l2 node, so the
        drain's delivered-interval arithmetic is not corrupted and neither
        reader gets half the stream.
        """
        if not self._preview_running():
            return False
        worker = self._worker()
        try:
            shot = worker.latest()
        except Exception:  # noqa: BLE001 - a stub worker
            return False
        seq = int(getattr(shot, "seq", 0) or 0)
        at = float(getattr(shot, "at", 0.0) or 0.0)
        age = max(0.0, self._now() - at) if at else None
        if seq <= 0 or age is None or age > PRESENCE_MAX_AGE_S:
            # The console is capturing but has nothing fresh to say. Taking
            # our OWN frames here is exactly the two-reader hazard, so the
            # honest answer is that we did not look.
            self._publish_dark("the console is capturing and its shot is "
                               "not fresh")
            return True
        faces = tuple(getattr(shot, "faces", ()) or ())
        label, score = "", 0.0
        for face in faces:
            if getattr(face, "id_ran", False) and str(getattr(face, "name", "")):
                label = str(face.name)
                try:
                    score = float(getattr(face, "id_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                break
        self.last_faces = len(faces)
        self.last_score = score
        self.last_reason = "borrowed from the console"
        if label:
            self.named += 1
            self._named_this_burst = label
        self._publish(Attention(faces=len(faces), identity=label,
                                id_score=score, age_s=age))
        return True

    # ------------------------------------------------------- publishing
    def _publish(self, attention: Attention) -> None:
        eye = getattr(self.feed, "eye", None)
        publish = getattr(eye, "publish", None)
        if not callable(publish):
            return
        try:
            publish(attention)
        except Exception:  # noqa: BLE001 - a stub eye in a test
            log.debug("eyeloop: the eye took no reading", exc_info=True)
            return
        self.published += 1
        self._looked.set()

    def _publish_dark(self, why: str) -> None:
        """"I could not look" -- which is NEVER "I looked and saw nobody".

        ``visionrig``'s rule 3, verbatim: zero faces with ok False means the
        rig was blind. It is the whole reason CAM_BLIND and CAM_LOOKED are
        two different values, and publishing a live zero here would turn
        every curfew minute into evidence that his flat is empty.
        """
        self.last_reason = why
        self.last_faces = 0
        self.last_score = 0.0
        self._publish(Attention(dark=True))

    def _tell_named(self) -> None:
        """Hand the LABEL to whoever asked, once per burst. Never raises,
        and never hands over anything but a string."""
        label, self._named_this_burst = self._named_this_burst, ""
        cb = self._on_named
        if not label or cb is None:
            return
        try:
            cb(str(label))
        except Exception:  # noqa: BLE001 - a consumer may not break the loop
            log.debug("eyeloop: the naming consumer raised", exc_info=True)

    # ---------------------------------------------------------- the report
    def status(self) -> dict:
        """NUMBERS AND STRINGS ONLY -- ``visionrig.assert_numbers_only``
        passes on this, and tests/test_eyeloop.py checks that it does."""
        return {
            "bursts": self.bursts, "frames": self.frames,
            "faces_seen": self.faces_seen, "named": self.named,
            "blind_bursts": self.blind_bursts, "borrowed": self.borrowed,
            "curfew_skips": self.curfew_skips, "arms": self.arms,
            "waits": self.waits, "wait_timeouts": self.wait_timeouts,
            "detect_errors": self.detect_errors, "published": self.published,
            "last_faces": self.last_faces, "last_score": self.last_score,
            "last_ms": self.last_ms, "last_reason": self.last_reason,
            "burst_s": self.burst_s, "fps": self.fps,
            "running": self.running,
            "curfew": ("%02d:%02d-%02d:%02d" % (self.curfew[0][0],
                                                self.curfew[0][1],
                                                self.curfew[1][0],
                                                self.curfew[1][1]))
            if self.curfew else "",
            "in_curfew": self.curfew_now(),
        }
