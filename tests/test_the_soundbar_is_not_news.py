"""A speaker that dropped and came back is not something he missed.

HIS BUG, 2026-09-11, and this is a real held backlog read back to him
verbatim on a return he never made:

    "i never left and he said While you were out: three warnings. The
     soundbar is back; I'm still coming out of the soundbar. I'm coming
     out of the monitor; the soundbar has dropped. The soundbar is back;
     I'm still coming out of the soundbar."

Three "warnings", all about the same speaker, none of them true any more.
jarvis/soundbar.py published one line per ROUTE TRANSITION as
``kind="warning"``, and jarvis/quiet.py queues a warning for the digest.
The module already had the right idea for this -- ``EPHEMERAL_KINDS``, "a
kind that is only worth hearing AT its moment" -- and audio had never
been put in it.

The test that matters most is the last one: the digest is for things he
MISSED, and he cannot miss where the sound is coming from.
"""
from __future__ import annotations


import pytest

from jarvis import quiet as quiet_mod
from jarvis import soundbar as soundbar_mod
from tests.test_quiet import FakeCfg


@pytest.fixture
def held():
    """A QuietPolicy with a readable backlog, built the way tests/test_quiet
    builds one."""
    return quiet_mod.QuietPolicy(FakeCfg())


def test_audio_route_is_an_ephemeral_kind():
    assert "audio-route" in quiet_mod.EPHEMERAL_KINDS
    assert "nudge" in quiet_mod.EPHEMERAL_KINDS, "the original must survive"


def test_a_route_line_is_expired_not_queued(held):
    kept = held.hold("I'm coming out of the monitor, sir; "
                     "the soundbar has dropped.", kind="audio-route")
    assert kept is False
    assert held.held == []


def test_his_three_warnings_leave_nothing_behind(held):
    """His exact backlog, as three transitions. Nothing survives to be
    read back."""
    for line in ("The soundbar is back; I'm still coming out of the soundbar.",
                 "I'm coming out of the monitor, sir; the soundbar has dropped.",
                 "The soundbar is back; I'm still coming out of the soundbar."):
        held.hold(line, kind="audio-route")
    assert held.held == []
    frags, _put_back = held.take_fragments()
    assert list(frags) == []


def test_a_real_warning_is_still_kept(held):
    """The change is scoped to audio. A disk or memory warning is exactly
    the thing the digest exists for."""
    assert held.hold("The disk is nearly full, sir.", kind="warning") is True
    assert len(held.held) == 1


@pytest.mark.parametrize("kind", ["message", "reminder", "timer", "alarm", "warning"])
def test_every_other_kind_still_queues(held, kind):
    assert held.hold("something worth keeping", kind=kind) is True


def test_the_soundbar_publishes_the_ephemeral_kind():
    """The producer and the gate have to agree, or the fix is half of one
    -- which is the defect this project has paid for seven times."""
    said = []
    bar = soundbar_mod.SoundbarSentinel.__new__(soundbar_mod.SoundbarSentinel)
    bar._say = lambda text, proactive=True, kind="warning": said.append((text, kind))
    bar._speak("I'm coming out of the monitor, sir; the soundbar has dropped.")
    assert said and said[0][1] == "audio-route", said
    assert said[0][1] in quiet_mod.EPHEMERAL_KINDS


def test_the_line_is_still_spoken_at_its_moment():
    """Expiring the digest must not silence the transition itself: the
    value of the line is entirely in hearing it as it happens."""
    said = []
    bar = soundbar_mod.SoundbarSentinel.__new__(soundbar_mod.SoundbarSentinel)
    bar._say = lambda text, proactive=True, kind="warning": said.append(text)
    bar._speak("The soundbar is back, sir.")
    assert said == ["The soundbar is back, sir."]


# ---- HIS RULING, 2026-09-11: a moment that has gone is not read back ----
#
# "missed reminders shouldnt be read back if the reminder is 30 mins
# outdated", and the half of his report that prompted it: "If the event has
# passed then he doesnt need to tell me about it if i come back."
#
# EPHEMERAL_KINDS above never lets a line into the backlog. This is the
# other half: a line that was worth holding when it fired and went stale
# waiting for him.
class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def aging():
    clock = _Clock()
    q = quiet_mod.QuietPolicy(FakeCfg(), now=clock)
    q.clock = clock
    return q


@pytest.mark.parametrize("kind", quiet_mod.MOMENT_KINDS)
def test_a_moment_that_has_gone_is_not_read_back(aging, kind):
    aging.hold("You want to be walking in 5 minutes, sir.", kind=kind)
    aging.clock.t += quiet_mod.STALE_AFTER_S + 1.0
    frags, _put_back = aging.take_fragments()
    assert list(frags) == [], kind


@pytest.mark.parametrize("kind", quiet_mod.MOMENT_KINDS)
def test_the_same_line_inside_the_window_is_still_read_back(aging, kind):
    aging.hold("You want to be walking in 5 minutes, sir.", kind=kind)
    aging.clock.t += quiet_mod.STALE_AFTER_S - 60.0
    frags, _put_back = aging.take_fragments()
    assert any("walking" in f for f in frags), kind


@pytest.mark.parametrize("kind", ["warning", "message"])
def test_a_kind_whose_value_is_not_its_moment_still_keeps(aging, kind):
    """Scoped deliberately: "the disk is nearly full" is as true an hour
    later, and a message he was sent does not expire."""
    aging.hold("The disk is nearly full, sir.", kind=kind)
    aging.clock.t += quiet_mod.STALE_AFTER_S * 4
    frags, _put_back = aging.take_fragments()
    assert any("disk" in f for f in frags), kind


def test_a_stale_line_is_not_handed_back_to_the_queue(aging):
    """put_back exists so a caller that might not speak can return what it
    took. A line nothing should ever say is not one to give back."""
    aging.hold("stale reminder", kind="reminder")
    aging.hold("The disk is nearly full, sir.", kind="warning")
    aging.clock.t += quiet_mod.STALE_AFTER_S + 1.0
    frags, put_back = aging.take_fragments()
    assert put_back() == 1                       # the warning only
    assert len(aging.held) == 1 and aging.held[0][2] == "warning"


def test_the_boundary_is_thirty_minutes():
    assert quiet_mod.STALE_AFTER_S == 30 * 60.0


def test_a_nudge_never_reaches_the_window_because_it_never_queues():
    """The two rules do not overlap, and this says so: a nudge is expired
    at HOLD time by EPHEMERAL_KINDS, so it has no staleness window to
    reach and is not listed as if it had one."""
    assert "nudge" in quiet_mod.EPHEMERAL_KINDS
    assert "nudge" not in quiet_mod.MOMENT_KINDS


def test_a_line_that_cannot_date_itself_is_kept(aging):
    """Never lose a line to a bad timestamp: unknown age keeps."""
    with aging._lock:
        aging._held.append(("not a time", "something he missed", "reminder"))
    frags, _put_back = aging.take_fragments()
    assert any("missed" in f for f in frags)
