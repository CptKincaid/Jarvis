"""A camera leg that is attached but has nothing to see SAYS SO, once.

HIS REPORT, 2026-09-11, one minute after I told him the camera leg was
live: "camera is not connected" -- and /dev/video* did not exist at all.

The leg itself degraded exactly right: every vote that hour reads
"cam-blind", and presencevote's whole asymmetry is that a camera which
CANNOT LOOK must never be read as an empty room. Nothing was broken.
What was missing was the sentence. The build before this one had a loud
line for it ("HIS NUMBER ONE SIGNAL IS DARK") and it only fired when no
feed was attached at all; once the feed was wired, an absent DEVICE fell
into the quiet "attached and votes from its first look" branch, which is
a promise it could not keep.

That is the defect this project has paid for over and over --
arrival._door_from_room, the mic-ledger wording, the outing refusals --
so it is pinned here in both directions.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace


import jarvis.app as app_mod
from jarvis import camera as camera_mod


class _Legs:
    eye = None
    look = None


def _app(monkeypatch, *, nodes, live):
    a = SimpleNamespace(camera_feed=object(), eyeloop=None)
    a._eye_leg = lambda: (("hunter", 1, True) if live else ("", None, False))
    monkeypatch.setattr(camera_mod, "device_nodes", lambda: list(nodes))
    return a


def _wire(a, legs=None):
    return app_mod.JarvisApp._wire_camera_leg(a, legs if legs is not None else _Legs())


def test_no_device_says_so_at_warning(monkeypatch, caplog):
    a = _app(monkeypatch, nodes=[], live=False)
    with caplog.at_level(logging.WARNING, logger=app_mod.log.name):
        assert _wire(a) is False
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "THERE IS NO CAMERA" in said
    assert camera_mod.DEVICE_GLOB in said


def test_it_says_the_blind_leg_is_the_RIGHT_answer(monkeypatch, caplog):
    """The line must not read as a fault he has to fix before presence
    works: the verdict is still sound without it."""
    a = _app(monkeypatch, nodes=[], live=False)
    with caplog.at_level(logging.WARNING, logger=app_mod.log.name):
        _wire(a)
    said = " ".join(r.getMessage() for r in caplog.records).lower()
    assert "never votes" in said and "not a wrong one" in said
    assert "phone" in said and "rooms" in said


def test_a_device_that_exists_keeps_the_quiet_line(monkeypatch, caplog):
    a = _app(monkeypatch, nodes=["/dev/video0"], live=False)
    with caplog.at_level(logging.INFO, logger=app_mod.log.name):
        assert _wire(a) is False
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "votes from its first look" in said
    assert "THERE IS NO CAMERA" not in said


def test_a_leg_that_can_already_see_is_untouched(monkeypatch, caplog):
    a = _app(monkeypatch, nodes=["/dev/video0"], live=True)
    with caplog.at_level(logging.INFO, logger=app_mod.log.name):
        assert _wire(a) is True
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "the camera leg is live" in said


def test_the_leg_is_still_wired_even_with_no_camera(monkeypatch):
    """The slot must stay attached: the docstring promises it goes live
    the moment a feed can see, with no restart. Refusing to wire it would
    break that."""
    legs = _Legs()
    _wire(_app(monkeypatch, nodes=[], live=False), legs)
    assert callable(legs.eye)


def test_a_broken_device_probe_does_not_stop_the_boot(monkeypatch, caplog):
    def boom():
        raise OSError("no /dev")
    a = SimpleNamespace(camera_feed=object(), eyeloop=None)
    a._eye_leg = lambda: ("", None, False)
    monkeypatch.setattr(camera_mod, "device_nodes", boom)
    with caplog.at_level(logging.WARNING, logger=app_mod.log.name):
        assert app_mod.JarvisApp._wire_camera_leg(a, _Legs()) is False
    assert "THERE IS NO CAMERA" in " ".join(r.getMessage() for r in caplog.records)
