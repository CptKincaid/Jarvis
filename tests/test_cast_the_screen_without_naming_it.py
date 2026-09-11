""""Cast my screen" casts his screen.

HIS BUG, 2026-09-11: "Jarvis does not cast the screen when asked."

TWO reasons, and the first one is why he heard NOTHING rather than a
refusal. `_CAST_SCREEN_RX` REQUIRED a named destination, so the three
ways anybody actually says it -- "cast the screen", "cast my screen",
"cast screen" -- matched no command at all and fell through to the model.
Nothing cast, and nothing said why.

The handler's own docstring already carried the reason a bare form is
safe: "there are two machines", so a sentence that names no destination
names the other one by elimination.

THE SECOND REASON IS HIS AND IS NOT FIXED HERE: `gesture.cast_screens`
ships OFF and is off on his box, so a correctly-phrased cast answers
"Screen casting is off, sir." That is a deliberate default about putting
a desktop on a monitor in his office, and it is his to flip -- the test
at the end pins that the refusal still happens and still says so.
"""
from __future__ import annotations

import pytest

from jarvis import screens as screens_mod
from jarvis.commander import _CAST_SCREEN_RX


BARE = ["cast the screen", "cast my screen", "cast screen",
        "mirror my desktop", "show my screen",
        "cast the desktop", "mirror the screen"]
# NOT covered, and said so rather than quietly claimed: a trailing particle
# with no destination ("put the screen up", "throw the screen over"). The
# optional tail wants a preposition AND a machine or nothing at all, so a
# dangling "up" leaves text the pattern will not eat. Worth adding the day
# he says one of them; not worth widening the pattern on a guess.
NOT_COVERED = ["put the screen up", "throw the screen over"]
NAMED = ["cast the screen to hpcomputer", "cast the spark to hpcomputer",
         "put my screen on hpcomputer", "mirror the desktop over to the pc"]


@pytest.mark.parametrize("said", BARE)
def test_the_bare_form_is_a_cast_command_at_all(said):
    """Before this, every one of these matched nothing."""
    m = _CAST_SCREEN_RX.match(said)
    assert m is not None, said
    assert m.groupdict().get("where") is None, said


@pytest.mark.parametrize("said", NAMED)
def test_naming_the_destination_still_works_and_still_wins(said):
    m = _CAST_SCREEN_RX.match(said)
    assert m is not None and m.group("where"), said


@pytest.mark.parametrize("said", NOT_COVERED)
def test_a_dangling_particle_is_not_covered_and_this_says_so(said):
    assert _CAST_SCREEN_RX.match(said) is None, said


@pytest.mark.parametrize("said", [
    "cast a spell", "cast your mind back", "show me the weather",
    "put the kettle on", "mirror mirror on the wall",
])
def test_it_does_not_swallow_ordinary_sentences(said):
    """The bare form is wider, so this is the half that matters: a
    sentence that is not about a screen must still reach the model."""
    assert _CAST_SCREEN_RX.match(said) is None, said


def test_there_are_exactly_two_machines_which_is_why_the_bare_form_is_safe():
    """The premise the whole change rests on, pinned. If a third machine
    is ever added, an unnamed destination stops being unambiguous and this
    fails rather than guessing."""
    assert len(screens_mod.MACHINES) == 2, screens_mod.MACHINES


class _Courier:
    """The courier's resolution, with its own gate open."""

    screen_cast_on = True
    this_machine = screens_mod.SPARK

    def __init__(self):
        self.views = {screens_mod.HPCOMPUTER: object()}
        self.cast = []

    def _cast_view(self, plan, by):
        self.cast.append(plan)
        return "Casting, sir.", "cast"


def _cast(target):
    from jarvis import gesturecast
    c = _Courier()
    line, status = gesturecast.GestureCast.cast_screen(c, target)
    return c, line, status


def test_an_unnamed_destination_resolves_to_the_other_machine():
    c, _line, status = _cast("")
    assert status == "cast"
    assert c.cast and c.cast[0][1] == screens_mod.HPCOMPUTER


def test_a_named_destination_is_unchanged():
    c, _line, status = _cast("hpcomputer")
    assert status == "cast" and c.cast[0][1] == screens_mod.HPCOMPUTER


def test_an_unknown_destination_is_still_refused_by_name():
    _c, line, status = _cast("the toaster")
    assert status == "refused" and "toaster" in line


def test_the_off_switch_still_refuses_and_says_so():
    """His, not mine: gesture.cast_screens ships off. A bare cast now
    REACHES this refusal instead of falling through in silence, which is
    the difference between "it doesn't work" and "it's switched off"."""
    from jarvis import gesturecast

    class _Off(_Courier):
        screen_cast_on = False

    line, status = gesturecast.GestureCast.cast_screen(_Off(), "")
    assert status == "refused"
    assert line == gesturecast.NO_SCREEN_CAST_LINE
