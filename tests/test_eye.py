"""The camera-side interface (jarvis/eye.py), tested with no camera.

Three things are pinned here, and each is a promise the design cannot keep by
intention alone:

* **The device gate.** ``Eye`` is the only thing in the process that holds
  /dev/video*, and it asks the sensing-state owner for permission BEFORE it
  opens and AGAIN before a captured frame is allowed out. Offline set
  mid-pipeline therefore drops the frame in flight rather than processing it.
  Fail-to-offline: an owner that raises, or is missing, is a NO.
* **The fusion rule.** The camera may promote a suppressed wake to an accepted
  one; it may never do the reverse. That is asserted exhaustively rather than
  by example, because "the camera never vetoes" is the whole safety argument
  (jarvis/hotword.py:373-375: a wake word that cannot be triggered is worse
  than one that triggers too often).
* **The body anchor.** A body embedding is only ever as good as the face
  sighting that vouched for it, and it expires.

``FakeDevice`` stands in for cv2.VideoCapture the way ``Http`` stands in for
urllib in tests/test_roomsensor.py. No cv2, no /dev/video*, no display.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from jarvis import eye as eye_mod
from jarvis.eye import Attention, Eye, SessionIdentity, resolve_wake


class FakeDevice:
    """A frame source: hands out numbered frames, records release()."""

    def __init__(self, frames=None, fail_read: bool = False):
        self.frames = list(frames) if frames is not None else None
        self.fail_read = fail_read
        self.released = 0
        self.reads = 0
        self.opened = True

    def read(self):
        self.reads += 1
        if self.fail_read:
            return False, None
        if self.frames is not None:
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)
        return True, np.full((4, 4, 3), self.reads, dtype=np.uint8)

    def release(self):
        self.released += 1
        self.opened = False


class Opener:
    """Records how many times a device was actually opened."""

    def __init__(self, device_factory=FakeDevice):
        self.opens = 0
        self.factory = device_factory
        self.last = None

    def __call__(self):
        self.opens += 1
        self.last = self.factory()
        return self.last


# ------------------------------------------------------------- the gate
def test_a_denied_gate_never_opens_the_device_at_all():
    """Not 'opens it and ignores the frames'. Hunter's ruling is that the
    device is not opened -- a software flag a later code path could ignore is
    the thing he specifically refused."""
    opener = Opener()
    eye = Eye(allow=lambda: False, open_device=opener)
    assert eye.capture() is None
    assert opener.opens == 0
    assert eye.device_open is False
    assert eye.denials == 1


def test_an_allowed_gate_opens_once_and_reuses_the_device():
    opener = Opener()
    eye = Eye(allow=lambda: True, open_device=opener)
    assert eye.capture() is not None
    assert eye.capture() is not None
    assert opener.opens == 1
    assert eye.device_open is True


def test_going_offline_mid_pipeline_drops_the_frame_in_flight():
    """The gate is re-read AFTER the grab and BEFORE the frame is returned, so
    a frame captured a millisecond before he said "offline mode" is discarded
    rather than recognised."""
    state = {"allow": True}
    seen = []

    class Racing(FakeDevice):
        def read(self):
            # Offline is set by another thread while this grab is in flight.
            state["allow"] = False
            return super().read()

    opener = Opener(device_factory=Racing)
    eye = Eye(allow=lambda: state["allow"], open_device=opener)
    frame = eye.capture()
    seen.append(frame)
    assert frame is None                      # nothing reached the caller
    assert opener.last.reads == 1             # it WAS grabbed...
    assert opener.last.released == 1          # ...and the device was released
    assert eye.device_open is False


def test_a_deny_closes_a_device_that_is_already_open():
    state = {"allow": True}
    opener = Opener()
    eye = Eye(allow=lambda: state["allow"], open_device=opener)
    eye.capture()
    dev = opener.last
    assert eye.device_open is True
    state["allow"] = False
    assert eye.capture() is None
    assert dev.released == 1
    assert eye.device_open is False


def test_it_fails_to_offline_when_the_owner_raises():
    """Privacy beats convenience: he chose fail-to-offline over failing
    online. A sensing owner that throws is not a reason to keep watching."""
    def boom():
        raise RuntimeError("sensing state unavailable")

    opener = Opener()
    eye = Eye(allow=boom, open_device=opener)
    assert eye.capture() is None
    assert opener.opens == 0


def test_it_fails_to_offline_with_no_owner_wired_at_all():
    """A camera that watches because nobody told it not to is the bug."""
    opener = Opener()
    assert Eye(allow=None, open_device=opener).capture() is None
    assert opener.opens == 0


@pytest.mark.parametrize("answer", [None, 1, "yes", "", 0])
def test_only_a_real_true_is_permission(answer):
    """A truthy 1 or a string is a wiring mistake, not a decision."""
    opener = Opener()
    assert Eye(allow=lambda: answer, open_device=opener).capture() is None
    assert opener.opens == 0


def test_a_device_that_cannot_be_opened_is_no_opinion_not_an_error():
    def cannot():
        raise OSError("No such file or directory: /dev/video0")

    eye = Eye(allow=lambda: True, open_device=cannot)
    assert eye.capture() is None
    assert eye.device_open is False


def test_a_failed_read_releases_the_device_so_an_unplug_recovers():
    """Unplug it and nothing breaks -- the promise jarvis/roomsensor.py:33-40
    already makes for the mmWave leg."""
    opener = Opener(device_factory=lambda: FakeDevice(fail_read=True))
    eye = Eye(allow=lambda: True, open_device=opener)
    assert eye.capture() is None
    assert eye.device_open is False
    assert eye.capture() is None
    assert opener.opens == 2          # it retries; it does not latch off


def test_close_is_idempotent_and_reports_the_truth():
    opener = Opener()
    eye = Eye(allow=lambda: True, open_device=opener)
    eye.capture()
    eye.close()
    eye.close()
    assert opener.last.released == 1
    assert eye.device_open is False


# ------------------------------------------------------------- the fusion
def att(**kw) -> Attention:
    base = dict(faces=1, attending=True, dwell_s=0.8, conf=0.95,
                identity="", id_score=0.0, dark=False, age_s=0.2)
    base.update(kw)
    return Attention(**base)


def test_the_camera_promotes_a_suppressed_wake_when_it_can_see_him():
    """The measured failure this pays for. jarvis/hotword.py:428-448: over an
    UNFLAGGED bed (a TV, a browser -- anything that is not Spotify) the trim
    widens to cover the room, music= is False, and hey_jarvis_05 reports
    1.46 s and scores 0.183 -> suppress, on exactly the 0.52 s of him that
    scores 0.277 and accepts when dry. A television cannot touch the camera."""
    out = resolve_wake("suppress", False, att())
    assert out.ok is True
    assert out.verdict == "accept (eye)"
    assert out.evidence == "one attending face"


def test_the_camera_never_vetoes_an_accepted_wake():
    for eye in (att(faces=0, attending=False), att(faces=3), att(dark=True), None):
        out = resolve_wake("accept", True, eye)
        assert out.ok is True


def test_the_camera_cannot_subtract_over_the_whole_input_space():
    """The safety argument in one assertion. Anything the camera can say, on
    top of any verdict the audio gate can reach, must leave ok >= the audio's
    own ok."""
    verdicts = [("accept", True), ("abstain", True),
                ("abstain (too little speech)", True),
                ("abstain (music)", True), ("suppress", False)]
    eyes = [None]
    for faces, attending, dark, age, ident in itertools.product(
            (0, 1, 2, 5), (True, False), (True, False), (0.1, 9.0),
            ("", "hunter", "guest")):
        eyes.append(att(faces=faces, attending=attending, dark=dark,
                        age_s=age, identity=ident,
                        id_score=0.0 if not ident else 0.7))
    for (verdict, ok), eye in itertools.product(verdicts, eyes):
        out = resolve_wake(verdict, ok, eye)
        assert out.ok >= ok, (verdict, eye)
        if ok:
            assert out.ok is True


@pytest.mark.parametrize("eye,why", [
    (att(faces=0, attending=False), "nobody in frame"),
    (att(faces=2), "a second person could be the one who spoke"),
    (att(attending=False), "he is not facing it"),
    (att(dark=True), "a dark frame is no opinion"),
    (att(age_s=9.0), "a stale reading is no opinion"),
    (None, "no camera at all"),
])
def test_a_suppression_stands_when_the_camera_cannot_vouch(eye, why):
    out = resolve_wake("suppress", False, eye)
    assert out.ok is False, why
    assert out.verdict == "suppress"


def test_a_recognised_stranger_is_not_promoted():
    """Positive identification of someone else is the one case where the
    camera has real evidence AGAINST -- and it still only declines to promote.
    Suppression was already the audio gate's verdict; the camera does not get
    to add a veto it would not otherwise have."""
    out = resolve_wake("suppress", False, att(identity="guest", id_score=0.7))
    assert out.ok is False and out.verdict == "suppress"


def test_the_owner_being_recognised_is_the_strongest_evidence():
    out = resolve_wake("suppress", False, att(identity="hunter", id_score=0.62))
    assert out.ok is True
    assert "hunter" in out.evidence


def test_a_second_face_suppresses_the_guest_line_because_it_is_true():
    """"I only answer to Hunter, sir" said into an empty room is a bug; said
    to an actual second person it is at best rude. jarvis/app.py:2575 fires it
    only on a suppression, so this is the flag that call site reads."""
    assert resolve_wake("suppress", False, att(faces=2)).guest_ok is False
    assert resolve_wake("suppress", False, att(faces=1, attending=False)).guest_ok is True
    assert resolve_wake("suppress", False, None).guest_ok is True


def test_an_abstention_gains_evidence_without_changing_the_outcome():
    """Today an abstention wakes him on a shrug (jarvis/hotword.py:497-498,
    'wake speaker check unavailable -- waking anyway'). The camera does not
    change that -- it makes the line say WHY, which is the only calibration
    data this gate will ever get."""
    out = resolve_wake("abstain", True, att())
    assert out.ok is True
    assert out.verdict == "abstain (eye: one attending face)"
    out = resolve_wake("abstain", True, att(faces=0, attending=False))
    assert out.verdict == "abstain"


def test_a_dwell_that_is_too_short_is_a_glance_not_an_address():
    """Below ~400 ms you fire on a head passing the lens on the way to a mug."""
    assert resolve_wake("suppress", False, att(dwell_s=0.1)).ok is False
    assert resolve_wake("suppress", False,
                        att(dwell_s=eye_mod.MIN_DWELL_S)).ok is True


# -------------------------------------------------------- the body anchor
def bvec(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def test_a_body_is_only_who_a_face_recently_said_it_was():
    clock = {"t": 100.0}
    s = SessionIdentity(ttl_s=600.0, match_min=eye_mod.BODY_MATCH_MIN, now=lambda: clock["t"])
    body = bvec(1)
    assert s.identify(body) == ("", 0.0)          # nothing vouched yet
    s.anchor("hunter", body)
    label, score = s.identify(body + bvec(2) * 0.05)
    assert label == "hunter" and score > 0.9


def test_the_body_anchor_expires_because_clothes_change():
    """Body re-ID is dominated by clothing. An anchor that outlives the outfit
    it was taken from is a confident wrong answer, so it has a clock."""
    clock = {"t": 0.0}
    s = SessionIdentity(ttl_s=600.0, match_min=eye_mod.BODY_MATCH_MIN, now=lambda: clock["t"])
    body = bvec(3)
    s.anchor("hunter", body)
    clock["t"] = 599.0
    assert s.identify(body)[0] == "hunter"
    clock["t"] = 601.0
    assert s.identify(body) == ("", 0.0)


def test_the_room_emptying_drops_the_anchor_immediately():
    """The window in which a different person can sit down wearing anything is
    exactly the window in which nobody was in frame."""
    s = SessionIdentity(ttl_s=600.0, match_min=eye_mod.BODY_MATCH_MIN, now=lambda: 0.0)
    body = bvec(4)
    s.anchor("hunter", body)
    s.room_empty()
    assert s.identify(body) == ("", 0.0)


def test_a_different_body_is_not_him_even_inside_the_ttl():
    s = SessionIdentity(ttl_s=600.0, match_min=eye_mod.BODY_MATCH_MIN, now=lambda: 0.0)
    s.anchor("hunter", bvec(5))
    assert s.identify(bvec(6)) == ("", 0.0)


def test_a_degenerate_body_vector_is_refused():
    s = SessionIdentity(ttl_s=600.0, match_min=eye_mod.BODY_MATCH_MIN, now=lambda: 0.0)
    s.anchor("hunter", bvec(7))
    assert s.identify(np.zeros(768, dtype=np.float32)) == ("", 0.0)
    with pytest.raises(ValueError):
        s.anchor("hunter", np.zeros(768, dtype=np.float32))


# ---------------------------------------------------------- config keys
def test_the_camera_is_off_until_he_turns_it_on():
    from jarvis.assistant_config import DEFAULTS
    cam = DEFAULTS["camera"]
    assert cam["enabled"] is False
    assert cam["identity"] is False       # a face gallery on disk is opt-in
    assert cam["debug_frame"] is False    # the one JPEG path, off by default


def test_the_camera_section_carries_no_schedule_of_its_own():
    """The curfew and offline mode belong to the sensing-state owner, which is
    the single place that can be enforced at the device. A second copy of the
    window here is a copy that can disagree with the first, and the one that
    disagrees quietly is the one that keeps the lens open at 22:00."""
    from jarvis.assistant_config import DEFAULTS
    keys = set(DEFAULTS["camera"])
    # MATCHED PER UNDERSCORE-SEGMENT, not as a substring. "end" inside
    # "face_backend" is not a schedule, and the substring form failed on that
    # key the day it was added -- a rule that cries wolf gets deleted, and
    # this one is guarding the lens at 22:00.
    schedule = {"curfew", "hours", "offline", "start", "end", "until"}
    assert not {k for k in keys if schedule & set(k.split("_"))}


def test_the_detect_size_cannot_drift_from_the_capture_aspect():
    """The defect this replaces: 320x240 (4:3) shipped as the detect target
    for a 1080p (16:9) capture, which is a 1.33x anisotropic horizontal squash
    of every face in the frame.

    Measured 2026-09-02 on a 1080p frame carrying the 161 px face that
    docs/vision.md section 9's own arithmetic produces at the recommended
    mount: 320x240 scored 0.703 against the then-default 0.700 bar, while the
    aspect-correct 320x180 scored 0.840 on FEWER pixels and 2.0 ms less
    resize+detect (1920x1080 -> 320x180 is an exact 6:1 in both axes). Two
    other reconstructions of the same scene put 320x240 at 0.62 and at no
    detection at all -- the old default straddled its own threshold.

    Pinning the RATIO rather than the number is the point: the two settings
    are one decision and the first version let them drift apart silently.
    This is the no-drift half of the guard and it passes on the OLD defaults
    too (640x480 with 320x240 is consistently 4:3); the defect itself is
    caught by the capture-resolution test below. They are complementary: one
    fixes the scale, the other forbids the squash."""
    from jarvis.assistant_config import DEFAULTS
    cam = DEFAULTS["camera"]
    cap = cam["width"] / cam["height"]
    det = cam["detect_width"] / cam["detect_height"]
    assert abs(cap - det) < 0.01, "detect %dx%d squashes a %dx%d capture" % (
        cam["detect_width"], cam["detect_height"], cam["width"], cam["height"])
    assert DEFAULTS["camera"]["idle_fps"] < DEFAULTS["camera"]["armed_fps"]


def test_the_capture_resolution_is_the_one_the_mount_arithmetic_needs():
    """A 16 cm face at the 95 cm mount must survive to SFace's 112x112 input.

    THIS TEST USED TO CARRY THE BUG IT WAS GUARDING. It hard-coded a 191 cm
    span, which is what ~90 deg horizontal gives at 95 cm -- and 90 deg is
    the 98 deg-diagonal Arducam docs/vision.md section 9 recommends BUYING,
    not the 65.6 deg LifeCam Cinema he owns. With the span nailed to one
    camera, changing width/height to a mode the LifeCam actually has made
    the test fail on a config that is MORE correct, not less.

    So the span now comes from the configured field of view, which is the
    whole point of camera.hfov_deg existing: 1280 px across a 122 cm span is
    167 px on the face, better than the 162 the Arducam manages across 191
    cm at 1920. 640x480, which shipped first, would give 53 px -- what
    section 9 calls "far too small"."""
    from jarvis.assistant_config import DEFAULTS
    from jarvis.facemodels import Lens
    cam = DEFAULTS["camera"]
    lens = Lens(cam["width"], cam["height"], cam["hfov_deg"])
    face_px = lens.face_px(95.0)
    assert face_px >= 112.0, "a 16 cm face is %.0f px at %dx%d, %.1f deg" % (
        face_px, cam["width"], cam["height"], cam["hfov_deg"])
    # And it must still clear YuNet's floor after the downscale the detector
    # actually sees -- the step the two cameras do NOT cancel at.
    assert lens.detect_face_px(95.0, cam["detect_width"]) >= 10.0


def test_the_config_states_a_field_of_view_instead_of_implying_one():
    """The 90 deg the old comment reasoned from lived in prose, so nothing
    could disagree with it and nothing could be wrong. Every real candidate
    is 65-82 deg."""
    from jarvis.assistant_config import DEFAULTS
    cam = DEFAULTS["camera"]
    assert 0.0 < cam["hfov_deg"] < 180.0
    assert cam["hfov_deg"] < 85.0, "no camera he is considering is that wide"
    assert (cam["width"], cam["height"]) == (1280, 720), \
        "the LifeCam has no 1080p mode; asking for one gets a silent fallback"


def test_the_confidence_bar_leaves_the_measured_face_a_margin():
    """0.7 was not a bar the recommended mount clears. At the aspect-correct
    detect size the same face scores 0.840, so 0.6 leaves 0.24 of margin
    rather than 0.003 -- and the asymmetry is why the margin goes on this
    side: a miss is SILENT and disables the feature outright, while a false
    face still has to survive faces==1 and dwell_s >= camera.dwell_s before it
    can promote anything."""
    from jarvis.assistant_config import DEFAULTS
    assert DEFAULTS["camera"]["min_conf"] <= 0.65


# --------------------------------------- the holes an adversarial read found
def test_a_deny_tells_the_consumer_it_has_gone_blind_once_per_edge():
    """``SessionIdentity.room_empty()`` had no caller anywhere outside its own
    tests, so its stated bound -- "dropped the instant the room empties" -- was
    unowned. Offline mode and the 21:00-07:00 curfew are the unbounded blind
    windows that bound exists for, and ``Eye`` is the only witness to them.

    The EDGE, not every frame: at 8 fps a standing deny would otherwise fire
    this eight times a second saying nothing new."""
    gate = {"ok": True}
    blinds = []
    e = Eye(lambda: gate["ok"], lambda: FakeDevice(), on_blind=lambda: blinds.append(1))

    assert e.capture() is not None
    gate["ok"] = False
    for _ in range(8):                      # a whole second of armed-tier ticks
        assert e.capture() is None
    assert len(blinds) == 1, "fired %d times for one deny" % len(blinds)
    assert e.device_open is False

    gate["ok"] = True                        # he comes back
    assert e.capture() is not None
    gate["ok"] = False
    assert e.capture() is None
    assert len(blinds) == 2 and e.blind_edges == 2


def test_a_dropped_frame_is_a_hiccup_and_does_not_drop_the_anchor():
    """A failed grab is deliberately NOT a blind edge. Clearing the body
    anchor on every transient read failure would make the anchor useless; the
    TTL is the backstop for a camera that quietly stops working. A deny is
    different in kind -- somebody, or the clock, said stop."""
    blinds = []
    e = Eye(lambda: True, lambda: FakeDevice(fail_read=True),
            on_blind=lambda: blinds.append(1))
    for _ in range(5):
        assert e.capture() is None
    assert blinds == []


def test_a_blind_consumer_that_raises_still_leaves_the_device_shut():
    """Fail to offline: a broken consumer is not a reason to keep watching,
    and must not turn a deny into an exception out of capture()."""
    def boom():
        raise RuntimeError("the sidecar died")

    gate = {"ok": True}
    e = Eye(lambda: gate["ok"], lambda: FakeDevice(), on_blind=boom)
    assert e.capture() is not None
    gate["ok"] = False
    assert e.capture() is None               # no exception escapes
    assert e.device_open is False


def test_going_offline_clears_the_body_vector_from_memory_too():
    """The end-to-end shape of the two fixes above: after "offline mode" there
    is no body vector left in RAM vouching for whoever is in the chair when
    the lens re-opens."""
    s = SessionIdentity(ttl_s=900.0, match_min=eye_mod.BODY_MATCH_MIN,
                        now=lambda: 0.0)
    gate = {"ok": True}
    e = Eye(lambda: gate["ok"], lambda: FakeDevice(), on_blind=s.room_empty)
    e.capture()
    body = bvec(11)
    s.anchor("hunter", body)
    assert s.identify(body)[0] == "hunter"

    gate["ok"] = False                       # "Jarvis, offline mode"
    assert e.capture() is None
    assert s.identify(body) == ("", 0.0)


def test_the_body_threshold_must_be_stated_not_inherited():
    """0.75 is right for the 768-D YoutuReID vector and a coin flip for the
    colour histogram docs/vision.md section 4 offers as the phase-1 stand-in.

    The arithmetic, not an opinion: cosine over NON-NEGATIVE vectors is
    compressed, and for iid uniform components two UNRELATED vectors have
    expected cosine E[x]^2/E[x^2] = 0.25/(1/3) = 0.750 exactly -- measured
    2026-09-02 over 2000 pairs at d=256, 768 and 4096, mean 0.750 every time
    with ~50% at or above 0.75. Sparse HSV histograms of nine unlike shirts
    still put 3 of 36 unlike pairs over the bar, and the same shirt under a
    changed desk lamp scored 0.692-0.706, i.e. UNDER it. Miscalibrated both
    ways, so the caller has to say which vector it is holding."""
    with pytest.raises(TypeError):
        SessionIdentity(ttl_s=600.0, now=lambda: 0.0)      # no match_min

    d = 768
    rng = np.random.default_rng(5)
    from jarvis.facegallery import cosine
    unrelated = [cosine(rng.random(d), rng.random(d)) for _ in range(400)]
    assert abs(float(np.mean(unrelated)) - 0.75) < 0.02, (
        "the compressed range this guard is about has moved: %.3f"
        % float(np.mean(unrelated)))
    assert eye_mod.BODY_MATCH_MIN == 0.75      # the YoutuReID number, unchanged


def test_a_non_finite_body_threshold_is_refused_and_fails_shut():
    """The body bar is the one the docstring above says must be STATED, and a
    stated NaN is not a bar. ``score < self.match_min`` is False for every
    score when ``match_min`` is NaN, so the anchor answers "still him" for
    any body at all -- the exact opposite of what the guard is for (F17,
    reproduced 2026-09-03 on the face side of the same spelling)."""
    with pytest.raises(ValueError):
        SessionIdentity(ttl_s=600.0, match_min=float("nan"), now=lambda: 0.0)
    with pytest.raises(ValueError):
        SessionIdentity(ttl_s=600.0, match_min=float("inf"), now=lambda: 0.0)

    s = SessionIdentity(ttl_s=600.0, match_min=eye_mod.BODY_MATCH_MIN,
                        now=lambda: 0.0)
    s.anchor("hunter", bvec(11))
    s.match_min = float("nan")                # set past the constructor
    assert s.identify(bvec(12)) == ("", 0.0)


def test_the_tiebreaker_promotes_nothing_without_attention():
    """docs/vision.md section 11.1 used to offer "build the wake tiebreaker and
    presence only" as the fallback if the mount geometry cannot separate
    "looking at Jarvis" from "reading the tab bar". It cannot: ``attending``
    is a hard requirement, so with no usable attention signal this function
    promotes nothing, and a faces-only version would be the cost paragraph in
    resolve_wake's docstring with its only mitigation removed.

    Pinned exhaustively so the fallback cannot be quietly re-invented."""
    for faces, dwell, ident, age in itertools.product(
            (0, 1, 2, 5), (0.0, 0.5, 5.0, 900.0), ("", "hunter"), (0.0, 1.4)):
        eye = att(faces=faces, attending=False, dwell_s=dwell,
                  identity=ident, age_s=age)
        out = resolve_wake("suppress", False, eye)
        assert out.ok is False, (faces, dwell, ident, age)
        assert out.evidence == ""
