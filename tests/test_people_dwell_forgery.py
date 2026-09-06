"""The people-tab dwell must not be forgeable by the clock it is handed.

FOUND by the users-tab verdict, 2026-09-05, and reported by it as a
robustness defect rather than a boundary crossing -- correctly, and it is
fixed here for the same reason the verdict raised it: `now=` is a seam on
every people seam, and a security guard should not be forgeable at all,
whoever can reach it today.

THE DEFECT.  `_people_unlock_left` computed `max(0.0, until - now)` with
`until` left at 0.0 while the tab is LOCKED.  So any negative `now` made
`0.0 - (-1.0)` = 1.0 s of dwell out of nothing, and `people_forget(...,
now=-1.0)` deleted a row from his people book with no override code ever
presented.  His people book has no history and no backup.

WHY IT WAS NOT CALLED A BLOCKER, kept here so nobody re-escalates it: `now=`
is not on the Services dataclass, and no socket, phone or voice rung reaches
it.  Whoever can pass it already holds the app object and could call
`_people_open_unlock()` directly.  It is defence in depth, not a hole that
was open to the room.

THE RULE: a dwell that was never opened is zero seconds long, whatever clock
it is asked about.  0.0 is the locked sentinel, and a sentinel must be tested
as a sentinel, never subtracted from.
"""
import pytest

from jarvis import app as app_mod


class _App:
    """Just the two methods under test, over a real _PEOPLE_LOCK."""

    def __init__(self, until=0.0):
        self._people_unlock_until = until

    _people_unlock_left = app_mod.JarvisApp._people_unlock_left


@pytest.mark.parametrize("now", [-1.0, -0.001, -1e9, -120.0])
def test_a_locked_tab_is_locked_at_any_clock_it_is_handed(now):
    """The forgery, straight from the verdict: negative `now`, locked tab."""
    assert _App()._people_unlock_left(now=now) == 0.0


@pytest.mark.parametrize("until", [0.0, -0.0, None])
def test_the_locked_sentinel_is_tested_not_subtracted_from(until):
    """0.0 means locked. None and -0.0 land on the same sentinel."""
    a = _App(until=until)
    assert a._people_unlock_left(now=-5.0) == 0.0
    assert a._people_unlock_left(now=5.0) == 0.0


def test_a_real_dwell_is_unharmed_and_still_counts_down():
    """The fix must not shorten a dwell that was genuinely opened."""
    a = _App(until=1000.0)
    assert a._people_unlock_left(now=880.0) == pytest.approx(120.0)
    assert a._people_unlock_left(now=999.5) == pytest.approx(0.5)
    assert a._people_unlock_left(now=1000.0) == 0.0
    assert a._people_unlock_left(now=1001.0) == 0.0


def test_a_negative_clock_cannot_stretch_a_real_dwell_either():
    """A dwell that IS open must report what it has left, not more. Before
    the fix this returned 1120.0 for a 120 s grant."""
    a = _App(until=1000.0)
    assert a._people_unlock_left(now=-120.0) == 0.0
