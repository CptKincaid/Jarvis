"""The camera leg is HIS NUMBER ONE SIGNAL, and today it is dark.

His ruling, 2026-09-05: "The camera should be the number one understanding
for if I'm in. Followed by phone connection then sensor." The voter obeys
that order -- a camera that NAMES him ends the vote in every one of the 27
cells. But nothing in this tree ever assigns ``services.camera_feed``
(grep for "camera_feed ="), so the leg answers BLIND unconditionally on
his box, and BLIND is "could not look", which never votes.

That is a two-leg voter wearing a three-leg name, and the one thing it must
never do is claim otherwise at boot. These tests pin the boot line: it says
LIVE only when the eye can actually answer, and says DARK -- loudly, and
naming the missing wiring -- when it cannot.
"""
from __future__ import annotations

import logging

import pytest

from jarvis import app as app_mod


class Legs:
    """Stands in for presence.ThreeLegProbe: it owns an ``eye`` slot."""

    def __init__(self):
        self.eye = None


def bare_app(eye_leg):
    a = object.__new__(app_mod.JarvisApp)
    a._eye_leg = eye_leg
    return a


DARK = lambda: ("", None, False)                       # noqa: E731
LIVE_EMPTY = lambda: ("", 0, True)                     # noqa: E731
LIVE_NAMED = lambda: ("hunter", 1, True)               # noqa: E731


@pytest.fixture()
def logs(caplog):
    caplog.set_level(logging.INFO, logger="app")
    return caplog


def test_the_eye_is_attached_whether_or_not_it_can_answer(logs):
    """The slot is wired either way -- the leg must go live the moment
    something finally attaches a feed, without another restart."""
    legs = Legs()
    a = bare_app(DARK)
    a._wire_camera_leg(legs)
    assert legs.eye == a._eye_leg


def test_a_dark_camera_leg_is_announced_and_names_the_missing_wiring(logs):
    a = bare_app(DARK)
    assert a._wire_camera_leg(Legs()) is False
    text = logs.text.lower()
    assert "dark" in text
    assert "camera_feed" in text, "the boot line must name the wiring that is missing"
    assert any(r.levelno >= logging.WARNING for r in logs.records), \
        "his number one signal being unavailable is not an INFO detail"


def test_the_dark_line_says_what_the_voter_is_actually_running_on(logs):
    """No quietly shipping a two-leg voter and calling it his design."""
    a = bare_app(DARK)
    a._wire_camera_leg(Legs())
    text = logs.text.lower()
    assert "phone" in text and ("room" in text or "sensor" in text)


def test_a_dark_leg_is_never_described_as_wired_or_live(logs):
    a = bare_app(DARK)
    a._wire_camera_leg(Legs())
    text = logs.text.lower()
    assert "leg is live" not in text
    assert "wired to the app's eye" not in text


def test_a_live_camera_leg_is_announced_as_live(logs):
    a = bare_app(LIVE_EMPTY)
    assert a._wire_camera_leg(Legs()) is True
    text = logs.text.lower()
    assert "live" in text
    assert "dark" not in text


def test_a_live_leg_that_has_already_named_him_is_still_only_live(logs):
    """The boot line reports the LEG's health, not tonight's verdict."""
    a = bare_app(LIVE_NAMED)
    assert a._wire_camera_leg(Legs()) is True
    assert "dark" not in logs.text.lower()


def test_an_eye_that_raises_at_boot_is_dark_rather_than_fatal(logs):
    def boom():
        raise RuntimeError("no feed")

    a = bare_app(boom)
    assert a._wire_camera_leg(Legs()) is False
    assert "dark" in logs.text.lower()


def test_an_eye_that_returns_nonsense_is_dark_rather_than_fatal(logs):
    a = bare_app(lambda: "not a triple")
    assert a._wire_camera_leg(Legs()) is False
    assert "dark" in logs.text.lower()


def test_the_boot_line_carries_no_frame_and_no_image_word(logs):
    """The leg is a NAME and a COUNT. Nothing here may carry a picture."""
    a = bare_app(LIVE_NAMED)
    a._wire_camera_leg(Legs())
    for record in logs.records:
        assert "frame" not in record.getMessage().lower()


def test_no_legs_object_is_not_an_error(logs):
    """The old two-leg composition (presence.three_legs false) has no
    ``legs`` at all, and boot must not care."""
    a = bare_app(DARK)
    assert a._wire_camera_leg(None) is False
