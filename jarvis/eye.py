"""The camera-side interface: who holds the device, and what may leave it.

There is no camera on this machine yet (``/dev/video*`` does not exist), so
this module is the half of the design that can be built and tested anyway --
the gate, the fusion rule and the body anchor. The inference half (YuNet,
SFace) lives in a sidecar process and is specified in
``scratchpad/ideas/vision.md`` and ``docs/vision.md``; nothing here imports
cv2, torch or a model, so the whole file runs in the suite with no camera, no
display and no GPU.

THE ONE RULE THIS MODULE EXISTS TO ENFORCE. ``Eye`` is the only thing in the
process that opens the video device, and it asks the sensing-state owner for
permission twice on every single frame: once BEFORE the device is opened, and
once AFTER the grab and before the frame is handed to anyone. Hunter's ruling
was that offline mode must be "enforced so the device is not opened at all --
not merely a software flag that a later code path could ignore", and the two
checks are what make that true in both directions:

* the first means a denied gate costs zero device opens, so there is nothing
  to ignore;
* the second means offline set *mid-pipeline* discards the frame already in
  flight instead of recognising a face captured a millisecond earlier.

FAIL TO OFFLINE, which he chose over persisting state and over failing online.
``allow`` returning anything but exactly ``True`` -- False, None, a truthy 1, a
missing callable, or an exception out of the owner -- is a NO. A camera that
watches because nobody told it not to is precisely the bug the ruling was
about, and "the sensing owner is broken" is not evidence that watching is
wanted.

WHAT CROSSES THE BOUNDARY. ``Attention`` -- counts, booleans and floats. Never
a frame, never a landmark array, never an embedding. The privacy decision is
made at the call site rather than inside the consumer, the same shape VSS's
``aiws_system/face_obscurer.py:20-23`` uses, and the consequence is that no
future consumer can acquire pixels by accident because no code path produces
them.

THE FUSION RULE (``resolve_wake``) is the reason to own a camera at all, and
it is deliberately one-directional: the camera can promote a wake the audio
gate suppressed, and can never do the reverse. ``jarvis/hotword.py:373-375``
already states why -- "a wake word that cannot be triggered is worse than one
that triggers too often" -- and a second sensor must not quietly undo that.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from jarvis.facegallery import cosine
from jarvis.logs import get_logger

log = get_logger("eye")

# A FLOOR on the dwell another process measured, not the dwell policy itself.
# The sidecar decides when to enter ATTENDING (camera.dwell_s, 0.6 s = five
# frames at the armed tier's 8 fps) and this is the sanity bar underneath it,
# because the number arrives over a socket from a process that can be
# reconfigured, restarted or wrong. A head turn to a new target completes in
# 200-400 ms, so below 400 ms it is a glance passing the lens on the way to a
# mug -- never an address, whatever the sender claims.
MIN_DWELL_S = 0.4
# Older than this and the camera is describing a different moment than the
# microphone is. The wake buffer is 2 s; 1.5 s keeps them the same event.
MAX_AGE_S = 1.5
# Body re-identification is dominated by clothing, so an anchor taken this
# morning says nothing this afternoon. Fifteen minutes is short enough that he
# has not plausibly changed and left and come back unseen.
BODY_TTL_S = 900.0
# Cosine over the 768-D YoutuReID vector. Deliberately high: this signal is
# only ever used to CARRY an identity a face already established, never to
# establish one, so a miss costs a re-look and a false match costs trust.
BODY_MATCH_MIN = 0.75


@dataclass
class Attention:
    """Everything the camera is allowed to say. No pixels, by construction.

    ``dark`` and a stale ``age_s`` both mean NO OPINION, which is not the same
    as "not attending" -- the same three-valued contract
    ``jarvis/roomsensor.py:27-32`` writes down for the mmWave leg, where
    ``None`` is "no opinion, NEVER empty". A camera that cannot see must
    degrade to exactly today's behaviour, not to a negative vote."""

    faces: int = 0
    attending: bool = False
    dwell_s: float = 0.0
    conf: float = 0.0
    identity: str = ""        # "" = identity not run or not confident
    id_score: float = 0.0
    dark: bool = False
    age_s: float = 0.0

    def usable(self, max_age_s: float = MAX_AGE_S) -> bool:
        """Is this a measurement, or an absence of one?"""
        return not self.dark and self.age_s <= max_age_s


@dataclass
class WakeVerdict:
    """What the fusion rule decided, and why -- ``evidence`` goes on the
    existing one-line ``wake candidate:`` log (jarvis/hotword.py:502)."""

    verdict: str
    ok: bool
    guest_ok: bool = True
    evidence: str = ""


class Eye:
    """Owner of the video device. Nothing else in the process opens it.

    ``allow`` is the sensing-state owner's read -- the single source of truth
    for offline mode and the curfew, which lives in its own module so this one
    neither duplicates the schedule nor can drift from it. ``open_device`` is
    the transport seam, the way ``RoomSensor(get=...)`` is one next door
    (jarvis/roomsensor.py); tests pass a fake and never touch /dev/video*.

    The returned device needs only ``read() -> (ok, frame)`` and ``release()``
    -- which is cv2.VideoCapture's own interface, so the production opener is
    a one-liner."""

    def __init__(self, allow: Optional[Callable[[], bool]],
                 open_device: Callable[[], object]):
        self._allow = allow
        self._open_device = open_device
        self._device = None
        self.opens = 0        # devices actually opened; the UI lamp reads this
        self.denials = 0
        self.reads_dropped = 0    # frames grabbed and then thrown away

    # ------------------------------------------------------------ the gate
    def permitted(self) -> bool:
        """May the device be open right now?

        Anything other than exactly True is a no, including an exception: the
        sensing owner failing is not permission, and there is no state here to
        remember a yes with (he chose fail-to-offline over persistence)."""
        allow = self._allow
        if not callable(allow):
            log.debug("no sensing-state owner wired; staying offline")
            return False
        try:
            return allow() is True
        except Exception:
            log.warning("sensing state unavailable; staying offline",
                        exc_info=True)
            return False

    @property
    def device_open(self) -> bool:
        """True only while a device handle is actually held. The UI's EYE lamp
        must be driven by this and never by a config value, so that a dark
        lamp means the camera is closed by construction."""
        return self._device is not None

    def close(self) -> None:
        """Release the device. Idempotent; never raises."""
        dev, self._device = self._device, None
        if dev is None:
            return
        try:
            dev.release()
        except Exception:
            log.debug("device release failed", exc_info=True)

    # --------------------------------------------------------- the capture
    def capture(self):
        """One frame, or None meaning NO OPINION.

        None covers every way this can decline -- denied, no device, an unplug,
        a failed grab, offline set mid-grab -- because a consumer that has to
        tell those apart will eventually get one of them wrong, and the safe
        reading of all five is identical: behave exactly as if there were no
        camera."""
        if not self.permitted():
            self.denials += 1
            self.close()       # a deny closes a device that is already open
            return None

        if self._device is None:
            try:
                self._device = self._open_device()
                self.opens += 1
            except Exception:
                # No /dev/video*, or the desktop session grabbed it: no
                # opinion. "Unplug it and nothing changes" is the promise
                # jarvis/roomsensor.py:33-40 makes for the mmWave leg.
                log.debug("could not open the camera", exc_info=True)
                self._device = None
                return None

        try:
            ok, frame = self._device.read()
        except Exception:
            log.debug("camera read failed", exc_info=True)
            ok, frame = False, None
        if not ok or frame is None:
            self.close()       # so the next tick re-opens rather than latching
            return None

        # THE SECOND CHECK. He may have said "offline mode" while this grab
        # was in flight. The frame exists; it must not be used.
        if not self.permitted():
            self.denials += 1
            self.reads_dropped += 1
            self.close()
            log.info("offline set mid-capture: frame dropped, device released")
            return None
        return frame


# ----------------------------------------------------------- the fusion
def resolve_wake(verdict: str, ok: bool, eye: Optional[Attention],
                 owner: str = "hunter",
                 max_age_s: float = MAX_AGE_S) -> WakeVerdict:
    """Fold the camera into the audio wake gate's verdict.

    Called with what ``Hotword._speaker_ok`` already computed
    (jarvis/hotword.py:487-501) and the camera's latest reading. It may turn a
    False into a True. It may NEVER turn a True into a False -- asserted
    exhaustively in tests/test_eye.py, because that invariant is the entire
    safety argument for adding a second sensor to a gate that deliberately
    fails open.

    WHAT IT ACTUALLY BUYS, stated so it can be falsified. The gate's only
    remaining False is "clean audio, enough measured speech, no music known,
    score under the bar" -- and jarvis/hotword.py:428-448 documents the case
    where that verdict is WRONG on his own voice: under an unflagged bed (a
    television, a browser -- anything that is not Spotify, since music_playing
    is Spotify's cache alone) the trim widens to cover the room, so the
    too-little-speech abstention disengages, and hey_jarvis_05 reports 1.46 s
    and scores 0.183 -> suppress, on exactly the 0.52 s of him that scores
    0.277 and accepts when dry. Audio has no more information to give there.
    The camera does: a television cannot put a face in the chair.

    The cost, equally plainly: a stranger sitting at his desk, facing the
    camera, whose voice scores under the bar, now wakes Jarvis. That is why
    the promotion needs ``faces == 1`` (a second person could be the one who
    spoke), a real dwell (a glance past the lens is not an address), and -- as
    soon as the gallery exists -- an identity that is not someone else's. The
    transcript gate in jarvis/app.py still fails shut behind all of it."""
    out = WakeVerdict(verdict=verdict, ok=ok)
    if eye is None or not eye.usable(max_age_s):
        return out       # no opinion: today's behaviour, byte for byte

    # A second person in the room is the one thing the camera knows that makes
    # "I only answer to Hunter, sir" (jarvis/app.py:2575) worse than silence.
    if eye.faces >= 2:
        out.guest_ok = False

    vouches = (eye.faces == 1 and eye.attending
               and eye.dwell_s >= MIN_DWELL_S
               and eye.identity in ("", owner))
    if not vouches:
        return out

    evidence = ("%s, attending" % owner) if eye.identity == owner \
        else "one attending face"
    if ok:
        # An abstention already wakes him; the camera cannot improve the
        # outcome, only the record. That record is the only calibration data
        # this gate will ever get, so it is worth the log line.
        if verdict.startswith("abstain"):
            out.verdict = "%s (eye: %s)" % (verdict, evidence)
        out.evidence = evidence
        return out

    out.verdict = "accept (eye)"
    out.ok = True
    out.guest_ok = False        # it woke; there is no guest to decline
    out.evidence = evidence
    return out


# ------------------------------------------------------- the body anchor
class SessionIdentity:
    """Who the body in the chair is, for as long as a face vouched for it.

    BE HONEST ABOUT WHAT THIS IS WORTH. Person re-identification models are
    trained to match the same person across cameras minutes apart, and what
    they mostly encode is CLOTHING. YoutuReID (opencv_zoo, Apache-2.0,
    106,878,407 bytes, 768-D) will happily call a stranger in a similar dark
    hoodie "him", and will not recognise him at all tomorrow in a different
    shirt. It is not an identity model and must never be used as one.

    So it is used only to CARRY an identity that a face already established:
    a face sighting anchors the body vector, and the anchor answers "still
    him" while he faces his monitor -- which at a desk is most of the time,
    and is exactly the gap he asked about. Two clocks bound the damage:

    * ``ttl_s`` -- an anchor expires, because clothes change;
    * ``room_empty()`` -- an anchor is dropped the moment nobody is in frame,
      because that is precisely the window in which a different person can sit
      down wearing anything at all.

    Nothing is persisted. A body vector never reaches the disk; there is no
    body gallery to leak, back up or delete."""

    def __init__(self, ttl_s: float = BODY_TTL_S,
                 match_min: float = BODY_MATCH_MIN,
                 now: Callable[[], float] = time.monotonic):
        self.ttl_s = float(ttl_s)
        self.match_min = float(match_min)
        self._now = now
        self._label = ""
        self._vec = None
        self._at = 0.0

    def anchor(self, label: str, body_vec) -> None:
        """A face said who this body is. Raises on a vector that cannot be
        one, so a failed crop cannot become a confident anchor."""
        arr = np.asarray(body_vec, dtype=np.float32).ravel()
        if arr.size == 0 or not np.all(np.isfinite(arr)) or \
                float(np.linalg.norm(arr)) == 0.0 or float(np.std(arr)) < 1e-6:
            raise ValueError("refusing to anchor a degenerate body vector")
        self._label, self._vec, self._at = label, arr.copy(), self._now()

    def room_empty(self) -> None:
        """Nobody in frame: forget who was."""
        self._label, self._vec, self._at = "", None, 0.0

    def identify(self, body_vec) -> Tuple[str, float]:
        """("hunter", score) while the anchor holds, else ("", 0.0)."""
        if self._vec is None:
            return "", 0.0
        if self._now() - self._at > self.ttl_s:
            self._label, self._vec = "", None
            return "", 0.0
        arr = np.asarray(body_vec, dtype=np.float32).ravel()
        if arr.size != self._vec.size or not np.all(np.isfinite(arr)):
            return "", 0.0
        score = cosine(arr, self._vec)
        if score < self.match_min:
            return "", 0.0
        return self._label, score
