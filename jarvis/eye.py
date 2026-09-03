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

A deny also fires ``on_blind`` on the EDGE, which is how ``SessionIdentity``'s
"dropped the instant the room empties" bound acquires an owner: offline mode
and the curfew are unbounded blind windows, and a body anchor that survives
one vouches for whoever is in the chair when the lens re-opens. It also means
the body vector does not sit in RAM after he has said "offline mode".

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

WHO IT IS (``FaceIdentifier``) is the enrolled gallery asked one question,
under one rule: identity is only ever computed from a detection that already
cleared the detector's confidence bar, because SFace scores garbage
CONFIDENTLY rather than low. And a name may only ever REMOVE capability or
ADD a name -- never grant capability the existing gates do not already grant,
which is why ``resolve_wake`` promotes exactly as much for a recognised him
as it does for an anonymous attending face, and less for a recognised
stranger.

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

from jarvis.facegallery import SFACE_COSINE_SAME, cosine
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
# Cosine over the 768-D YoutuReID vector, WHICH IS THE ONLY VECTOR THIS
# NUMBER IS FOR. Deliberately high: this signal is only ever used to CARRY an
# identity a face already established, never to establish one, so a miss costs
# a re-look and a false match costs trust.
#
# It does NOT transfer to the colour histogram docs/vision.md section 4 offers
# as the phase-1 stand-in, and the arithmetic says so exactly. Cosine over
# NON-NEGATIVE vectors lives in a compressed range: for iid uniform components
# the expected cosine of two UNRELATED vectors is E[x]^2/E[x^2] = 0.25/(1/3) =
# 0.750 -- measured here 2026-09-02 over 2000 pairs at d=256/768/4096, mean
# 0.750 every time, and 50% of unrelated pairs at or above 0.75. A histogram
# anchor at this bar is a coin flip. Sparse HSV histograms of nine unlike
# shirts did better (median 0.000) but still put 3 of 36 unlike pairs over
# 0.75 -- and, worse in the other direction, the SAME shirt under a changed
# desk lamp scored 0.692-0.706, i.e. BELOW the bar. The histogram is
# miscalibrated both ways, which is why ``match_min`` is required rather than
# defaulted: a caller must state which vector it is holding.
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
                 open_device: Callable[[], object],
                 on_blind: Optional[Callable[[], None]] = None):
        self._allow = allow
        self._open_device = open_device
        self._on_blind = on_blind
        self._device = None
        # True until a frame is actually returned: nothing has been seen yet,
        # so there is nothing for a first deny to invalidate.
        self._blind = True
        self.opens = 0        # devices actually opened; the UI lamp reads this
        self.denials = 0
        self.reads_dropped = 0    # frames grabbed and then thrown away
        self.blind_edges = 0      # deny transitions; on_blind fired this often

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

    def _go_blind(self) -> None:
        """Release the device and, on the EDGE only, tell the consumer.

        WHY THIS EXISTS. ``SessionIdentity`` holds a body vector that a face
        vouched for, and its own bound -- "dropped the instant the room
        empties" -- had no owner: ``room_empty()`` had no caller anywhere
        outside its tests. Offline mode and the 21:00-07:00 curfew are exactly
        the unbounded blind windows the bound was written for, and they are
        the two this class is the sole witness to. So the deny edge is where
        the call belongs.

        The EDGE, not every frame: at the armed tier a standing deny would
        otherwise fire this eight times a second for no new information.

        A failed grab is deliberately NOT a blind edge. One dropped frame is a
        hiccup, not an absence, and clearing the anchor on every hiccup would
        make the anchor useless; the TTL is the backstop for a camera that
        quietly stops working. A deny is different in kind -- somebody, or the
        clock, said stop, and nobody says how long for."""
        self.close()
        if self._blind:
            return
        self._blind = True
        self.blind_edges += 1
        cb = self._on_blind
        if cb is None:
            return
        try:
            cb()
        except Exception:
            # A consumer that raises must not keep the device open or turn a
            # deny into an exception out of capture(). Fail to offline.
            log.warning("on_blind consumer raised; the device is closed "
                        "regardless", exc_info=True)

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
            self._go_blind()   # a deny closes a device that is already open
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
            self._go_blind()
            log.info("offline set mid-capture: frame dropped, device released")
            return None
        self._blind = False
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
    transcript gate in jarvis/app.py still fails shut behind all of it.

    SO THE TIEBREAKER DIES WITH THE MOUNT, and docs/vision.md section 11.1
    used to say otherwise. ``attending`` is a hard requirement above, not a
    bonus term: if the $0 tape-and-photos test in section 9 shows the geometry
    cannot separate "looking at Jarvis" from "reading the tab bar", this
    function promotes nothing and there is no weaker version of it worth
    having. A faces-only promotion would wake Jarvis for anyone whose face is
    in frame while he is suppressed -- which is the cost paragraph above with
    its only mitigation removed. What survives an unusable mount is presence,
    not the tiebreaker. Pinned by
    tests/test_eye.py::test_the_tiebreaker_promotes_nothing_without_attention."""
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


# ------------------------------------------------------- the face identity
class FaceIdentifier:
    """The enrolled gallery, asked "who is this", under one hard rule.

    IDENTITY IS ONLY EVER COMPUTED FROM A DETECTION THAT ALREADY CLEARED THE
    DETECTOR BAR, AND THAT IS ENFORCED HERE IN CODE. SFace's 128-D embedding
    collapses on out-of-distribution input -- measured on this box 2026-09-02,
    unrelated NON-FACE crops match each other at mean cosine 0.66-0.92, with
    94-100% of pairs above the 0.363 "same person" bar
    (jarvis/facemodels.py). A false-positive box, a motion blur or a bad
    alignment therefore does not score LOW against the gallery, it scores
    CONFIDENTLY. The embedding is not a second opinion on whether this is a
    face and must never be used as one, so a row under ``min_conf`` returns
    ("", 0.0) with no embedding computed at all, and ``gated_out`` counts how
    often that happened so the number is visible in a report rather than
    inferred. ``SFaceRecogniser.embed`` refuses the same row underneath
    (jarvis/facedetect.py:212-226); two gates on one rule is deliberate.

    ``match_min`` is OpenCV's own documented SFace cosine for "same person",
    0.363, exported by ``jarvis/facegallery.SFACE_COSINE_SAME`` and settable
    from ``camera.identity_min``. It is keyword-only for the same reason
    ``SessionIdentity.match_min`` is: a threshold calibrated for one vector
    silently applied to another is how a gate stops meaning anything.

    WHAT A NAME MAY DO, which is his standing ruling and not a preference:
    **identity may REMOVE capability or ADD a name; it must never GRANT
    capability the existing gates do not already grant.** ``resolve_wake``
    above is written that way -- ``eye.identity in ("", owner)`` means a
    recognised stranger BLOCKS a promotion an anonymous face would have got,
    while a recognised owner promotes exactly what an anonymous single
    attending face already promoted, and only the log line differs. This
    class does the same to the body anchor: recognising him anchors it,
    recognising somebody else drops it. Pinned by
    tests/test_eye.py::test_recognising_him_grants_nothing_an_anonymous_face_lacked.

    Nothing is retained. The frame belongs to the caller, the crop lives
    inside the recogniser, and the embedding is dropped before this returns:
    a body vector never reaches the disk and neither does a face crop.
    """

    def __init__(self, gallery, recogniser, *, min_conf: float,
                 match_min: float = SFACE_COSINE_SAME, owner: str = "hunter",
                 session: Optional["SessionIdentity"] = None):
        self.gallery = gallery
        self.recogniser = recogniser
        self.min_conf = float(min_conf)
        self.match_min = float(match_min)
        self.owner = str(owner)
        self.session = session
        self.calls = 0
        self.gated_out = 0        # rows under the bar; no embedding computed
        self.errors = 0
        self.matched = 0
        self.unknown = 0

    def enrolled(self) -> int:
        try:
            return int(self.gallery.total())
        except Exception:  # noqa: BLE001 - a broken gallery is no opinion
            return 0

    def identify(self, frame, row, body_vec=None) -> Tuple[str, float]:
        """``(label, score)``, or ``("", 0.0)`` meaning NO OPINION.

        Every way this can decline -- under the bar, no gallery, a recogniser
        that raised, a score under ``match_min`` -- returns the same pair,
        because a consumer that has to tell them apart will get one of them
        wrong, and the safe reading of all four is "the camera does not know
        who this is", which is today's behaviour byte for byte.
        """
        self.calls += 1
        try:
            conf = float(np.asarray(row).ravel()[-1])
        except Exception:  # noqa: BLE001
            self.errors += 1
            return "", 0.0
        if conf < self.min_conf:
            # THE GATE. Nothing below this line runs on a weak detection.
            self.gated_out += 1
            return "", 0.0
        if self.enrolled() == 0:
            return "", 0.0
        try:
            vec = self.recogniser.embed(frame, row)
            label, score = self.gallery.match(vec)
        except Exception:  # noqa: BLE001 - a broken model is not an identity
            self.errors += 1
            log.debug("eye: identity failed on a detection", exc_info=True)
            return "", 0.0
        finally:
            vec = None
        if not label or score < self.match_min:
            self.unknown += 1
            # A face that is not confirmed to be him ends the body anchor.
            # Below the bar the nearest LABEL means nothing -- the gallery
            # always has a nearest member -- so "matched him weakly" and
            # "matched somebody else" are the same state, and the anchor may
            # not survive either. Removing an identity is the direction this
            # is allowed to act in; granting one is not.
            if self.session is not None:
                self.session.room_empty()
            return "", 0.0
        self.matched += 1
        if self.session is not None and body_vec is not None and \
                label == self.owner:
            try:
                self.session.anchor(label, body_vec)
            except (ValueError, TypeError):
                # A degenerate body vector is not an anchor. It is also not a
                # reason to lose the face identification.
                log.debug("eye: the body vector could not anchor",
                          exc_info=True)
        return label, float(score)

    def status(self) -> dict:
        """Numbers only, for the self-check and the enrolment report."""
        return {"enrolled": self.enrolled(), "calls": self.calls,
                "gated_out": self.gated_out, "matched": self.matched,
                "unknown": self.unknown, "errors": self.errors,
                "min_conf": self.min_conf, "match_min": self.match_min,
                "owner": self.owner}


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
      down wearing anything at all. ``Eye`` calls it on the deny edge (see
      ``on_blind``): offline mode and the curfew are unbounded blind windows,
      and an anchor that survives one is an anchor vouching for whoever is
      sitting there when the lens re-opens.

    Nothing is persisted. A body vector never reaches the disk; there is no
    body gallery to leak, back up or delete -- and after a deny there is none
    in memory either."""

    def __init__(self, ttl_s: float = BODY_TTL_S, *,
                 match_min: float,
                 now: Callable[[], float] = time.monotonic):
        # match_min is keyword-only and has NO default on purpose. The one
        # number here that cannot be chosen without knowing which vector the
        # sidecar is producing is this one (see BODY_MATCH_MIN above: 0.75 is
        # right for YoutuReID and a coin flip for a colour histogram), and a
        # default is how a threshold calibrated for one vector ends up
        # silently applied to another.
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
