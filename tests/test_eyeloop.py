"""The producer that finally makes HIS NUMBER ONE SIGNAL vote.

Every image in this file is a synthetic numpy array the test made. Nothing
here opens /dev/video*: ``camera.build`` takes a fake opener, which is the
seam that already exists. No frame is written, saved, thumbnailed or looked
at -- the producer is verified from NUMBERS ONLY (face counts, identity
scores, opens, denials, ages), which is the whole method and not a
consolation prize.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from jarvis import camera as camera_mod
from jarvis import eye as eye_mod
from jarvis import presencevote as pv


# ------------------------------------------------------------- the fakes
class Policy:
    """A sensing owner that says yes or no, and counts nothing else."""

    def __init__(self, allow=True):
        self.allow = allow
        self.attached = {}

    def allowed(self, kind):
        return bool(self.allow) is True and self.allow is True

    def attach(self, name, stop, present=None, resume=None):
        self.attached[name] = stop


class Device:
    """A capture handle. Returns a SYNTHETIC frame this test made."""

    def __init__(self, ok=True, h=8, w=8):
        self.ok = ok
        self.reads = 0
        self.released = 0
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)

    def read(self):
        self.reads += 1
        return (True, self._frame.copy()) if self.ok else (False, None)

    def release(self):
        self.released += 1


class Detector:
    """Rows in the 15-column shape jarvis/visionrig.py declares."""

    def __init__(self, faces=1, conf=0.9, boom=False):
        self.faces = faces
        self.conf = conf
        self.boom = boom
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.boom:
            raise RuntimeError("the detector fell over")
        rows = []
        for i in range(self.faces):
            row = np.zeros(15, dtype=np.float32)
            row[14] = self.conf - 0.01 * i
            rows.append(row)
        return rows


class Recogniser:
    def __init__(self):
        self.embeds = 0

    def embed(self, frame, row):
        self.embeds += 1
        return np.arange(8, dtype=np.float32)


class Gallery:
    def __init__(self, label="hunter", score=0.8):
        self.label = label
        self.score = score

    def total(self):
        return 13

    def labels(self):
        return (self.label,)

    def match(self, vec):
        return self.label, self.score


class Cfg:
    def __init__(self, **kw):
        self._d = dict(kw)

    def get(self, key, default=None):
        return self._d.get(key, default)


def make_feed(policy=None, device=None, on_blind=None):
    policy = Policy() if policy is None else policy
    device = Device() if device is None else device
    cfg = Cfg(**{"camera.enabled": True, "camera.device": "/dev/video-fake",
                 "camera.width": 1280, "camera.height": 720,
                 "camera.hfov_deg": 65.6})
    feed, why = camera_mod.build(cfg, policy, opener=lambda: device,
                                 on_blind=on_blind)
    assert feed is not None, why
    return feed, device, policy


def make_loop(feed, detector=None, identifier=None, **kw):
    from jarvis import eyeloop as mod
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("burst_s", 0.0)          # one frame per burst unless asked
    # NOON. The default curfew is his 21:00-07:00 and the default clock is
    # the wall: every arm() in this file refused after nine at night, and
    # 15 tests were red for the hour they were run in (measured 2026-09-12
    # 21:20). The curfew has its own tests; these are not them.
    kw.setdefault("clock", lambda: (12, 0))
    return mod.EyeLoop(feed, detector=detector, identifier=identifier, **kw)


def eye_leg_of(feed):
    """Exactly what ``app._eye_leg`` does, without booting an app."""
    from jarvis import app as app_mod
    a = object.__new__(app_mod.JarvisApp)
    a.services = type("S", (), {"camera_feed": feed})()
    return a._eye_leg()


# ------------------------------------------------------- 1. it fills the leg
def test_the_leg_answers_from_a_real_look():
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=2))
    loop.burst()
    ident, faces, live = eye_leg_of(feed)
    assert (ident, faces, live) == ("", 2, True)
    assert pv.camera_leg(identity=ident, faces=faces, live=live) == pv.CAM_SAW


def test_an_empty_room_is_looked_at_not_blind():
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=0))
    loop.burst()
    ident, faces, live = eye_leg_of(feed)
    assert (ident, faces, live) == ("", 0, True)
    assert pv.camera_leg(identity=ident, faces=faces, live=live) == pv.CAM_LOOKED


def test_a_name_reaches_the_leg_and_ends_the_vote():
    feed, dev, _ = make_feed()
    ident = eye_mod.FaceIdentifier(Gallery(), Recogniser(), min_conf=0.6,
                                   match_min=0.5, owner="hunter")
    loop = make_loop(feed, detector=Detector(faces=1), identifier=ident)
    loop.burst()
    name, faces, live = eye_leg_of(feed)
    assert (name, faces, live) == ("hunter", 1, True)
    assert pv.camera_leg(identity=name, faces=faces, live=live) == pv.CAM_SAW


# --------------------------------------------- 2. BOTH services get the feed
def test_both_service_objects_carry_the_feed_the_app_owns():
    """THE GUARD-ONE-HALF PIN, asserted in ONE test so the mirror cannot be
    forgotten. ``app.services`` is a SimpleNamespace and the UI's is a
    dataclass; setting only one leaves either the leg dark or the preview
    building a second feed, which is a privacy hole (SensingPolicy.attach
    replaces by name, so the curfew only reaches the newer one)."""
    import dataclasses
    import inspect

    from jarvis import app as app_mod
    from jarvis.ui.main_window import Services

    names = {f.name for f in dataclasses.fields(Services)}
    assert "camera_feed" in names, "the UI Services must declare the slot"

    src = inspect.getsource(app_mod.JarvisApp._build_services)
    assert re.search(r"camera_feed\s*=", src), \
        "app.services (the SimpleNamespace _eye_leg reads) needs camera_feed"

    src = inspect.getsource(app_mod.JarvisApp.ui_service_kwargs)
    assert re.search(r"camera_feed\s*=", src), \
        "the UI half needs it too, or campreview builds a SECOND feed"

    sentinel = object()
    built = app_mod.build_ui_services(Services, {"camera_feed": sentinel})
    assert built.camera_feed is sentinel


# ------------------------------------------------- 3. denied costs no opens
@pytest.mark.parametrize("why", ["offline", "curfew"])
def test_a_denied_policy_costs_zero_opens_and_the_leg_stays_blind(why):
    policy = Policy(allow=False)
    feed, dev, _ = make_feed(policy=policy)
    loop = make_loop(feed, detector=Detector(faces=1))
    loop.burst()
    assert feed.opens == 0
    assert feed.denials > 0
    assert dev.reads == 0
    assert feed.eye.state().dark is True
    assert eye_leg_of(feed) == ("", None, False)


def test_the_curfew_the_producer_owns_never_opens_the_lens():
    """DEFAULT TAKEN FOR HIM: 21:00-07:00 whenever camera.curfew* is unset
    (his 2026-09-02 ruling). Inside it the leg is BLIND and never votes."""
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=1),
                     clock=lambda: (23, 30))
    assert loop.curfew_now() is True
    loop.burst()
    assert feed.opens == 0
    assert dev.reads == 0
    assert eye_leg_of(feed) == ("", None, False)
    loop2 = make_loop(feed, detector=Detector(faces=1), clock=lambda: (14, 5))
    assert loop2.curfew_now() is False


# ----------------------------------------- 4. blind is never "saw nobody"
@pytest.mark.parametrize("kind", ["no-detector", "detector-raises", "no-frame"])
def test_a_blind_burst_is_never_published_as_an_empty_room(kind):
    device = Device(ok=(kind != "no-frame"))
    feed, dev, _ = make_feed(device=device)
    det = None if kind == "no-detector" else Detector(boom=(kind == "detector-raises"))
    loop = make_loop(feed, detector=det)
    loop.burst()
    state = feed.eye.state()
    assert state.dark is True
    assert eye_leg_of(feed) == ("", None, False)
    ident, faces, live = eye_leg_of(feed)
    assert pv.camera_leg(identity=ident, faces=faces, live=live) == pv.CAM_BLIND


# ------------------------------------------------------------ 5. staleness
def test_a_reading_older_than_the_presence_bound_stops_voting():
    clock = {"t": 1000.0}
    feed, dev, _ = make_feed()
    feed.eye._now = lambda: clock["t"]
    loop = make_loop(feed, detector=Detector(faces=1), now=lambda: clock["t"])
    loop.burst()
    assert eye_leg_of(feed)[2] is True
    clock["t"] += eye_mod.PRESENCE_MAX_AGE_S + 1.0
    assert feed.eye.state().usable(eye_mod.PRESENCE_MAX_AGE_S) is False
    assert eye_leg_of(feed) == ("", None, False)


def test_the_wake_bar_is_untouched_and_a_presence_look_promotes_nothing():
    """``eye.MAX_AGE_S`` is the WAKE bar, matched to the 2 s wake buffer. The
    presence tolerance is a SECOND number; two consumers, two bars."""
    assert eye_mod.MAX_AGE_S == 1.5
    assert eye_mod.PRESENCE_MAX_AGE_S > eye_mod.MAX_AGE_S
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=1))
    loop.burst()
    out = eye_mod.resolve_wake("suppress", False, feed.eye.state())
    assert out.ok is False, "a presence burst has no dwell and must promote nothing"


# ------------------------------------------------------- 6. deny mid-burst
def test_a_deny_during_the_burst_drops_the_frame_and_the_reading():
    calls = {"n": 0}
    dropped = {"n": 0}

    class Flipper(Policy):
        def allowed(self, kind):
            calls["n"] += 1
            # Eye asks once, the gate asks once on the open, and Eye asks
            # AGAIN after the grab -- which is the check that must say no.
            return calls["n"] <= 5

    feed, dev, policy = make_feed(policy=Flipper(),
                                  on_blind=lambda: dropped.__setitem__("n", dropped["n"] + 1))
    loop = make_loop(feed, detector=Detector(faces=1), burst_s=0.3, fps=8.0)
    loop.burst()
    assert feed.eye.reads_dropped == 1
    assert feed.eye.blind_edges == 1
    assert dropped["n"] == 1
    assert feed.eye.state().dark is True
    assert eye_leg_of(feed) == ("", None, False)


# -------------------------------------------------------- 7. identity gate
def test_a_row_under_the_detector_bar_computes_no_embedding_at_all():
    feed, dev, _ = make_feed()
    rec = Recogniser()
    ident = eye_mod.FaceIdentifier(Gallery(), rec, min_conf=0.6, match_min=0.5)
    loop = make_loop(feed, detector=Detector(faces=1, conf=0.2), identifier=ident)
    loop.burst()
    assert ident.gated_out == 1
    assert rec.embeds == 0, "no embedding may be computed under the detector bar"
    name, faces, live = eye_leg_of(feed)
    assert (name, faces, live) == ("", 1, True)


def test_a_score_under_the_match_bar_is_no_name_and_drops_the_anchor():
    feed, dev, _ = make_feed()
    session = eye_mod.SessionIdentity(match_min=eye_mod.BODY_MATCH_MIN)
    session.anchor("hunter", np.arange(8, dtype=np.float32))
    ident = eye_mod.FaceIdentifier(Gallery(score=0.2), Recogniser(), min_conf=0.6,
                                   match_min=0.5, session=session)
    loop = make_loop(feed, detector=Detector(faces=1), identifier=ident)
    loop.burst()
    assert eye_leg_of(feed)[0] == ""
    assert session.identify(np.arange(8, dtype=np.float32)) == ("", 0.0)


# ------------------------------------------------------------- 8. ONE device
def test_a_running_preview_is_borrowed_rather_than_raced_for_the_device():
    feed, dev, _ = make_feed()

    class Face:
        name, id_score, id_ran = "hunter", 0.71, True

    class Shot:
        faces = (Face(),)
        seq = 7
        at = 100.0

    class Worker:
        running = True

        def latest(self):
            return Shot()

    clock = {"t": 100.0}
    feed.eye._now = lambda: clock["t"]
    loop = make_loop(feed, detector=Detector(faces=1), now=lambda: clock["t"],
                     preview=lambda: Worker())
    loop.burst()
    assert feed.opens == 0, "the console owns the device; take zero frames"
    assert dev.reads == 0
    assert eye_leg_of(feed) == ("hunter", 1, True)


def test_with_no_preview_the_burst_opens_once_and_closes_after():
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=1), burst_s=0.3, fps=8.0)
    loop.burst()
    assert feed.opens == 1
    assert feed.device_open is False, "the lamp must be dark between bursts"


def test_the_producer_never_closes_a_device_another_holder_is_using():
    feed, dev, _ = make_feed()
    feed.hold("console")
    loop = make_loop(feed, detector=Detector(faces=1))
    loop.burst()
    assert feed.device_open is True, "another holder is still reading"
    assert feed.drop("console") is True
    assert feed.device_open is False


# ------------------------------------------------- 9. no pixels, structurally
def test_the_producer_reports_numbers_only_and_keeps_no_array():
    from jarvis import visionrig
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=2))
    loop.burst()
    visionrig.assert_numbers_only(loop.status())
    for name, value in vars(loop).items():
        assert not isinstance(value, np.ndarray), \
            "%s holds an array; the producer must keep no pixels" % name


def test_the_producer_source_cannot_write_a_frame_anywhere():
    from jarvis import eyeloop as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("imwrite", "imsave", "tofile", "np.save", "savez",
                   "pickle", "PIL", "Image."):
        assert banned not in src, "%s in the producer would put a frame somewhere" % banned


def test_the_published_reading_is_an_attention_and_cannot_be_a_frame():
    feed, dev, _ = make_feed()
    with pytest.raises(TypeError):
        feed.eye.publish(np.zeros((4, 4, 3), dtype=np.uint8))


# --------------------------------- the answer comes back on the RIGHT thread
def test_a_burst_that_names_him_hands_the_label_back():
    """The arming call runs on the Tk thread (bus.publish queues for it once
    a window is attached) or, headless, inline on the room fabric's thread
    inside its own lock. Neither may block for a second and a half, so the
    arm returns at once and the ANSWER comes back through this callback on
    the producer's own thread."""
    named = []
    feed, dev, _ = make_feed()
    ident = eye_mod.FaceIdentifier(Gallery(), Recogniser(), min_conf=0.6,
                                   match_min=0.5, owner="hunter")
    loop = make_loop(feed, detector=Detector(faces=1), identifier=ident,
                     on_named=named.append)
    loop.burst()
    assert named == ["hunter"]
    assert all(isinstance(n, str) for n in named), "a NAME, never a frame"


def test_an_anonymous_burst_hands_nothing_back():
    named = []
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=1), on_named=named.append)
    loop.burst()
    assert named == []


def test_a_consumer_that_raises_does_not_break_the_producer():
    def boom(label):
        raise RuntimeError("the settle fell over")

    feed, dev, _ = make_feed()
    ident = eye_mod.FaceIdentifier(Gallery(), Recogniser(), min_conf=0.6,
                                   match_min=0.5, owner="hunter")
    loop = make_loop(feed, detector=Detector(faces=1), identifier=ident,
                     on_named=boom)
    loop.burst()
    assert loop.bursts == 1
    assert eye_leg_of(feed) == ("hunter", 1, True)


def test_arming_never_blocks_and_never_opens_anything_itself():
    feed, dev, _ = make_feed()
    loop = make_loop(feed, detector=Detector(faces=1))
    assert loop.arm("a room change") is True
    assert feed.opens == 0, "arming is a flag, not a capture"
    assert dev.reads == 0
    assert loop.arms == 1


def test_arming_inside_the_curfew_publishes_blind_rather_than_nothing():
    """A stale HOME reading surviving into the curfew would be the leg
    voting off a look the curfew has since forbidden."""
    feed, dev, _ = make_feed()
    live = make_loop(feed, detector=Detector(faces=1), clock=lambda: (14, 0))
    live.burst()
    assert eye_leg_of(feed) == ("", 1, True)
    dark = make_loop(feed, detector=Detector(faces=1), clock=lambda: (22, 0))
    assert dark.arm("a room change") is False
    assert dark.curfew_skips == 1
    assert eye_leg_of(feed) == ("", None, False)
