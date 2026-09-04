"""The hand stage: the one reducer that turns a preview frame into a grab.

It rides INSIDE ``PreviewPipeline.grab()`` (jarvis/campreview.py), on the
frame the preview already pulled, on the capture thread, after the face
reduction and before the frame is shrunk and dropped. It is not a second
consumer of the camera and it opens nothing: ``SensingPolicy.attach``
replaces a device by name, so a second ``CameraFeed`` would silently take
the curfew away from the first (the frames lane measured that path and
ruled it out). Riding the existing grab is also what gives this stage the
six shutdown paths for free -- camera.enabled, camera.preview, the
ACTIVE->AMBIENT edge, the curfew and offline mode, Eye's own two checks per
frame, and quit -- with no seventh copy of a privacy control to disagree.

WHAT CROSSES OUT OF HERE IS SCALARS. ``observe()`` takes the frame and the
faces and returns a frozen :class:`HandShot` of floats, ints, bools and short
strings, or ``None``. It keeps no frame, no crop, no landmark array and no
detector output on any attribute; ``jarvis/handpose.HandTracker.detect`` is
a pure function of the array handed in, and the 21 landmarks it returns are
reduced to four numbers by ``gesture.observe_hand`` inside this call and then
dropped. tests/test_campreview.py greps this file for the vocabulary of a
frame leaving the process, and tests/test_gesturecast.py asserts that no
ndarray survives on the stage after a call.

``None`` MEANS NO OPINION, AND THAT IS NOT ``HandShot(present=False)``. The
stage answers None when it is switched off, when the hand models are not on
disk, or when the tracker raised; it answers ``present=False`` only when it
actually looked and found no hand. The VSS yunet path that silently fell
back and never once ran is the reason the distinction lives in the type.

THE PRECONDITION STACK (safety lane), cheapest and most-vetoing first:

  0. ``gesture.enabled`` (assistant.json), default FALSE.
  1. the lens is allowed -- structural, never checked here: a denied lens
     produces no frame, so this code does not run.
  2. the console is ACTIVE -- also structural: the preview worker is
     stopped on every ACTIVE->AMBIENT edge.
  3. no question is on the floor (``commander.question_open``): a grab
     must not land while a read-back is waiting on a yes, and a carry that
     is live when a question opens is put down.
  4. a FACE BASELINE exists: the median interocular distance over the last
     ``face_window_s`` with at least ``face_min_samples`` samples. No
     baseline, no scale, no grab -- a refusal, never a default. A RUN of
     faces -- the median of the last ``FACE_JUMP_SAMPLES`` -- more than
     ``FACE_JUMP_FRAC`` away from that median is a lean toward the lens,
     and is no opinion for the frame, because the median lags the lean
     and the reach ratio would inherit the error. A single wild sample is
     not a lean and is absorbed, which is what the median is for.
  5. ATTENTION IS LATCHED, not sampled per frame: the reaching arm crosses
     the face at exactly the moment the gesture matters. The stage arms
     when one face was ATTENDING within ``attend_latch_s`` and stays armed
     for the life of a reach or a carry.
  6..10. the depth, the fist, the dwell and the fling are
     ``gesture.CastGesture``'s, from coordinates only.

THE DELIVERED FRAME RATE, NOT THE CONFIGURED ONE. His live config asks for
camera.preview_fps 15; the LifeCam delivers ~7.5 (measured 09-02, 133 ms a
frame). ``CastThresholds.for_fps`` scales the frame counters for a faster
feed, and applied to the CONFIGURED 15 it would double the dwell to six
frames -- 800 ms of holding a fist still before anything happens. So the
stage measures the interval between its own calls and applies ``for_fps``
to that, re-fitting only while the machine is idle and through
``CastGesture.retune`` (which also rebuilds the open-history window --
assigning ``t`` alone left it at its construction-time length, and at
30 fps delivered that grabbed 0/48). The wall-clock stall backstop is set
from the same measurement, and until there is one it is floored at
``STALL_FLOOR_S`` rather than trusting the configured rate.

ATTEND_LATCH_S IS A GUESS. The frames lane proposed 3.0 s and the safety
lane 1.0 s; 3.0 is used because a reach, a close and a three-frame dwell is
already ~1 s at 7.5 fps, most of it with an arm across the face. It is in
the config block beside every other number here.
"""
from __future__ import annotations

import dataclasses
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jarvis.gesture import CastGesture, CastState, CastThresholds, observe_hand
from jarvis.logs import get_logger

log = get_logger("handstage")

# assistant.json keys. The switch defaults OFF, like every other lens key.
OPTION_ENABLED = "gesture.enabled"
OPTION_PREFIX = "gesture."
OPTION_THREADS = "gesture.hand_threads"
OPTION_MODEL_DIR = "gesture.model_dir"
OPTION_MIRRORED = "gesture.mirrored"
OPTION_LATCH = "gesture.attend_latch_s"
OPTION_FACE_WINDOW = "gesture.face_window_s"
OPTION_FACE_MIN = "gesture.face_min_samples"

ATTEND_LATCH_S = 3.0          # GUESSED: frames lane 3.0, safety lane 1.0
FACE_WINDOW_S = 5.0           # safety lane: a 5 s rolling median
FACE_MIN_SAMPLES = 3
DEFAULT_THREADS = 2           # matches camera.threads; see handpose.py
RETRY_MODELS_S = 60.0         # how often a missing model is looked for again
FPS_WINDOW = 24               # frames over which the delivered rate is read
FPS_REFIT_DELTA = 0.5         # re-fit the counters when the rate moves this much
# The stall backstop until a rate has been MEASURED. His config asks for
# 15 fps, which seeds the engine's three-period bar at 0.2 s; in low light
# the LifeCam delivers 3.8 fps (263 ms a frame), so every one of the first
# eight frames read as a stall and reset the reach -- a gesture begun
# inside ~2 s of the worker starting was lost (MEASURED: 0/24 with a
# 3-frame lead against 24/24 at 7.5 fps). Three frames at 3.8 fps is
# 0.79 s; the measured rate replaces this the moment there is one.
STALL_FLOOR_S = 0.8
# A face this much bigger or smaller than its own 5 s median is a LEAN,
# and the reach ratio measured against the lagging median is wrong by the
# same fraction: leaning in from 700 to 450 mm while closing an open hand
# onto the chin read R ~2.5 and grabbed 6 of 6 sub-frame phases through
# the wired path (MEASURED). No opinion for that frame instead. 20% is
# above landmark jitter and a 30-degree head turn (cos 30 = 0.87) and
# below any lean that changes the answer.
FACE_JUMP_FRAC = 0.20
# ...and the near side of that comparison is itself a MEDIAN, over the
# last three samples, not the one frame that just arrived. Comparing a
# single raw sample against the median throws away the only thing the
# median is there for: one wild YuNet interocular (400 px is a face
# 156 mm from the lens) is absorbed by the median and must not be read as
# a lean. MEASURED, one 400 px sample injected at each of the 14 frames
# of a his-left throw: against the raw frame 13 of 14 throws survived,
# against the median of three 14 of 14, and the lean stays refused 6/6
# either way.
FACE_JUMP_SAMPLES = 3

# Why the stage did not look this frame. "" when it did.
REASON_OFF = "off"
REASON_QUESTION = "question on the floor"
REASON_UNARMED = "not attending"
REASON_NO_MODELS = "no hand models"


@dataclass(frozen=True)
class HandShot:
    """What the hand stage is allowed to say. SCALARS ONLY, the discipline
    ``campreview.PreviewFace`` holds: no landmark array, no crop, no
    embedding, and no field that could hold one."""

    present: bool = False
    conf: float = 0.0
    cx: float = 0.0            # palm centroid, CAPTURE px (PreviewFace space)
    cy: float = 0.0
    palm_diag: float = 0.0     # the hand-unit, capture px
    closed: float = 0.0        # C: fist <= 0.70, open >= 0.85
    reach: float = 0.0         # R: palm_diag / interocular; 0.0 = no opinion
    state: str = "idle"        # gesture.CastState.value
    held: str = ""             # the spoken name of what is being carried
    event: str = ""            # grab | throw | drop, on the frame it fired
    toward: str = ""           # the measured sector of a carry end
    armed: bool = False        # the tracker ran this frame
    reason: str = ""           # why it did not, else ""
    hand_ms: float = 0.0       # the stage's cost this frame

    def as_dict(self) -> dict:
        return {"present": self.present, "conf": round(self.conf, 3),
                "cx": round(self.cx, 1), "cy": round(self.cy, 1),
                "palm_diag": round(self.palm_diag, 1),
                "closed": round(self.closed, 3), "reach": round(self.reach, 3),
                "state": self.state, "held": self.held, "event": self.event,
                "toward": self.toward, "armed": self.armed,
                "reason": self.reason, "hand_ms": round(self.hand_ms, 2)}


# ------------------------------------------------------------- the config
def _option(get_option: Optional[Callable], key: str, default):
    if not callable(get_option):
        return default
    try:
        value = get_option(key, default)
    except TypeError:                     # a reader that takes only the key
        try:
            value = get_option(key)
        except Exception:                 # noqa: BLE001 - config boundary
            return default
    except Exception:                     # noqa: BLE001 - config boundary
        return default
    return default if value is None else value


def gesture_enabled(get_option: Optional[Callable]) -> bool:
    return bool(_option(get_option, OPTION_ENABLED, False))


def thresholds_from_options(get_option: Optional[Callable],
                            base: Optional[CastThresholds] = None
                            ) -> CastThresholds:
    """``gesture.<field>`` overrides on top of the design defaults, key by
    key and coerced to the field's own type, so a hand-typed "3" in the
    file cannot turn a frame counter into a string. Unknown keys are not
    an error; a value that will not coerce is logged and skipped."""
    base = base or CastThresholds()
    over = {}
    for field in dataclasses.fields(base):
        raw = _option(get_option, OPTION_PREFIX + field.name, None)
        if raw is None:
            continue
        default = getattr(base, field.name)
        try:
            if isinstance(default, tuple):
                over[field.name] = tuple(str(s).strip().lower() for s in raw)
            elif isinstance(default, bool):
                over[field.name] = bool(raw)
            elif isinstance(default, int):
                over[field.name] = int(raw)
            else:
                over[field.name] = float(raw)
        except (TypeError, ValueError):
            log.warning("gesture: %s%s=%r ignored (not a %s)", OPTION_PREFIX,
                        field.name, raw, type(default).__name__)
    return dataclasses.replace(base, **over) if over else base


def _float_option(get_option, key: str, default: float) -> float:
    try:
        return float(_option(get_option, key, default))
    except (TypeError, ValueError):
        return float(default)


def _int_option(get_option, key: str, default: int) -> int:
    try:
        return int(_option(get_option, key, default))
    except (TypeError, ValueError):
        return int(default)


def default_tracker_factory(get_option: Optional[Callable]) -> Callable:
    """A zero-arg maker for the real tracker, resolved lazily so this
    module loads on a tree with no cv2, no onnxruntime and no weights."""
    def make():
        from jarvis import handpose                  # noqa: PLC0415 - lazy lane
        model_dir = str(_option(get_option, OPTION_MODEL_DIR, "") or "") or None
        threads = _int_option(get_option, OPTION_THREADS, DEFAULT_THREADS)
        return handpose.HandTracker(model_dir=model_dir, threads=threads)
    return make


# --------------------------------------------------------------- the stage
class HandStage:
    """Frames in, a ``HandShot`` out, on the capture thread.

    ``gesture`` is the ``CastGesture`` the courier built (it owns the
    payload and the event sink); ``make_tracker`` is a zero-arg callable
    returning something with ``detect(frame) -> rows`` where each row has
    ``lm`` (21x2, capture px) and ``conf``; ``hold_off`` answers True while
    a question holds the floor. Everything is injected so the suite can
    drive the whole stage with a fake tracker and generated arrays.
    """

    def __init__(self, gesture: CastGesture, *,
                 get_option: Optional[Callable] = None,
                 make_tracker: Optional[Callable[[], Any]] = None,
                 hold_off: Optional[Callable[[], bool]] = None,
                 now: Callable[[], float] = time.monotonic,
                 attend_latch_s: Optional[float] = None,
                 face_window_s: Optional[float] = None,
                 face_min_samples: Optional[int] = None) -> None:
        self.gesture = gesture
        self._get = get_option
        self._make = make_tracker or default_tracker_factory(get_option)
        self._hold_off = hold_off
        self._now = now
        self.attend_latch_s = float(
            attend_latch_s if attend_latch_s is not None
            else _float_option(get_option, OPTION_LATCH, ATTEND_LATCH_S))
        self.face_window_s = float(
            face_window_s if face_window_s is not None
            else _float_option(get_option, OPTION_FACE_WINDOW, FACE_WINDOW_S))
        self.face_min_samples = max(1, int(
            face_min_samples if face_min_samples is not None
            else _int_option(get_option, OPTION_FACE_MIN, FACE_MIN_SAMPLES)))
        self._base = gesture.t
        # The configured rate is an ASK the camera has not met; do not let
        # it set a stall bar shorter than a low-light frame period.
        gesture.stall_s = max(float(gesture.stall_s), STALL_FLOOR_S)
        self._tracker = None
        self._tracker_reason = ""
        self._tracker_tried = -1e9
        self._tracker_logged = False
        self._attended_at = -1e9
        self._eyes: deque = deque()
        self._ticks: deque = deque(maxlen=FPS_WINDOW)
        self.fps_measured = 0.0
        self._fps_applied = 0.0
        self.frames = 0
        self.looked = 0
        self.events = 0
        self.last: Optional[HandShot] = None

    # ------------------------------------------------------------ reads
    @property
    def enabled(self) -> bool:
        return gesture_enabled(self._get)

    @property
    def tracker_reason(self) -> str:
        return self._tracker_reason

    def status(self) -> dict:
        """Numbers and strings only; goes into the worker's status dict."""
        last = self.last.as_dict() if self.last is not None else {}
        return {"enabled": self.enabled, "frames": self.frames,
                "looked": self.looked, "events": self.events,
                "fps_measured": round(self.fps_measured, 2),
                "fps_applied": round(self._fps_applied, 2),
                "tracker": self._tracker is not None,
                "tracker_reason": self._tracker_reason,
                "baseline_samples": len(self._eyes),
                "machine": self.gesture.status(), "last": last}

    # ------------------------------------------------------------ guts
    def _tick_fps(self, t: float) -> None:
        """Read the delivered rate off this stage's own call times and
        re-fit the frame counters to it while nothing is in flight."""
        self._ticks.append(t)
        n = len(self._ticks)
        if n < 8:
            return
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0.0:
            return
        fps = (n - 1) / span
        self.fps_measured = fps
        if self.gesture.state is not CastState.IDLE:
            return
        if abs(fps - self._fps_applied) < FPS_REFIT_DELTA:
            return
        self._fps_applied = fps
        # Counters, the open-history window and the three-period stall
        # bar, all at the rate actually delivered, in one locked step.
        self.gesture.retune(CastThresholds.for_fps(fps, self._base), fps=fps)

    def _baseline(self, faces, t: float) -> float:
        """The interocular distance the reach ratio is measured against: a
        rolling median over ``face_window_s``, and 0.0 (no opinion) until
        ``face_min_samples`` have been seen. Exactly one face, with its
        landmarks, feeds it -- two faces is not this gesture."""
        if len(faces) == 1:
            f = faces[0]
            px = float(getattr(f, "eye_px", 0.0) or 0.0)
            if px > 0.0 and bool(getattr(f, "landmarks_ok", True)):
                self._eyes.append((t, px))
        cutoff = t - self.face_window_s
        while self._eyes and self._eyes[0][0] < cutoff:
            self._eyes.popleft()
        if len(self._eyes) < self.face_min_samples:
            return 0.0
        median = float(statistics.median(px for _t, px in self._eyes))
        recent = [px for _t, px in list(self._eyes)[-FACE_JUMP_SAMPLES:]]
        near = float(statistics.median(recent)) if recent else 0.0
        if near > 0.0 and median > 0.0 and \
                abs(near - median) / median > FACE_JUMP_FRAC:
            # A lean, not a reach: the median has not caught up with the
            # face and the ratio would be measured against the wrong
            # scale. A LEAN MOVES A RUN OF SAMPLES; a bad detection moves
            # one, and ``near`` being a median is what tells them apart.
            return 0.0
        return median

    def _arm(self, faces, t: float) -> bool:
        if len(faces) == 1 and bool(getattr(faces[0], "attending", False)):
            self._attended_at = t
        if self.gesture.state in (CastState.REACHING, CastState.CLOSING,
                                  CastState.CARRYING):
            return True
        return (t - self._attended_at) <= self.attend_latch_s

    def _tracker_or_none(self, t: float):
        if self._tracker is not None:
            return self._tracker
        if t - self._tracker_tried < RETRY_MODELS_S:
            return None
        self._tracker_tried = t
        try:
            self._tracker = self._make()
            self._tracker_reason = ""
            log.info("gesture: hand tracker ready")
        except Exception as exc:                     # noqa: BLE001 - no fallback
            self._tracker = None
            self._tracker_reason = "%s: %s" % (type(exc).__name__, exc)
            if not self._tracker_logged:
                self._tracker_logged = True
                log.warning("gesture: no hand tracker -- %s",
                            self._tracker_reason)
        return self._tracker

    def _fit(self, frame_w: int, frame_h: int) -> None:
        fw, fh = float(frame_w), float(frame_h)
        g = self.gesture
        if (g.frame_w, g.frame_h) != (fw, fh) and g.state is CastState.IDLE:
            g.frame_w, g.frame_h = fw, fh

    def _shot(self, obs, ev, armed: bool, reason: str, t0: float,
              eye_px: float) -> HandShot:
        g = self.gesture
        best = obs[0] if obs else None
        shot = HandShot(
            present=best is not None,
            conf=float(best.conf) if best else 0.0,
            cx=float(best.cx) if best else 0.0,
            cy=float(best.cy) if best else 0.0,
            palm_diag=float(best.palm_diag) if best else 0.0,
            closed=float(best.closed) if best else 0.0,
            reach=float(best.reach) if best else 0.0,
            state=g.state.value, held=g.held,
            event=ev.kind if ev is not None else "",
            toward=ev.toward if ev is not None else "",
            armed=armed, reason=reason,
            hand_ms=(self._now() - t0) * 1000.0)
        if ev is not None:
            self.events += 1
        self.last = shot
        return shot

    # ---------------------------------------------------------- the call
    def observe(self, frame, faces, frame_w: int, frame_h: int,
                seq: int = 0) -> Optional[HandShot]:
        """One frame. Returns scalars, or None for NO OPINION.

        ``faces`` are the ``PreviewFace`` rows of THIS frame (biggest
        first, attention verdict on the first). ``frame`` is borrowed for
        the duration of the call and nothing about it is kept.
        """
        t0 = self._now()
        self.frames += 1
        g = self.gesture
        if not self.enabled:
            if g.state is not CastState.IDLE:
                g.cancel("gesture switched off")
            return None
        self._fit(frame_w, frame_h)
        self._tick_fps(t0)
        eye_px = self._baseline(faces, t0)
        armed = self._arm(faces, t0)
        if self._hold_off is not None:
            try:
                busy = bool(self._hold_off())
            except Exception:                        # noqa: BLE001 - a predicate
                busy = False
            if busy:
                if g.state is CastState.CARRYING:
                    g.cancel("a question is on the floor")
                ev = g.update((), eye_px, seq)
                return self._shot((), ev, False, REASON_QUESTION, t0, eye_px)
        if not armed:
            # A miss, not a skipped call: the machine's clock keeps
            # running, so a carry that loses attention still times out.
            ev = g.update((), eye_px, seq)
            return self._shot((), ev, False, REASON_UNARMED, t0, eye_px)
        tracker = self._tracker_or_none(t0)
        if tracker is None:
            return None
        try:
            rows = tracker.detect(frame)
        except Exception:                            # noqa: BLE001 - one bad frame
            log.debug("gesture: the hand tracker raised", exc_info=True)
            return None
        self.looked += 1
        obs = []
        for row in rows or ():
            try:
                obs.append(observe_hand(row.lm, eye_px, float(row.conf)))
            except (ValueError, AttributeError, TypeError):
                continue
        rows = None                                  # nothing of the frame survives
        ev = g.update(obs, eye_px, seq)
        return self._shot(obs, ev, True, "", t0, eye_px)

    def cancel(self, why: str = "cancelled"):
        """The spoken "drop it", the chip click, shutdown: any thread."""
        return self.gesture.cancel(why)


__all__ = [
    "ATTEND_LATCH_S", "DEFAULT_THREADS", "FACE_JUMP_FRAC",
    "FACE_JUMP_SAMPLES", "FACE_MIN_SAMPLES", "FACE_WINDOW_S", "STALL_FLOOR_S",
    "HandShot", "HandStage", "OPTION_ENABLED", "OPTION_PREFIX",
    "REASON_NO_MODELS", "REASON_OFF", "REASON_QUESTION", "REASON_UNARMED",
    "default_tracker_factory", "gesture_enabled", "thresholds_from_options",
]
