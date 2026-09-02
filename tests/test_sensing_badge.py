"""The console's sensing indicator, tested display-free.

House pattern (tests/test_holo_geometry.py): every decision the badge
makes is a pure function, so the one thing a screenshot review would argue
about -- is this legible, and can the three states be told apart at a
glance -- is asserted here without ever creating a toplevel window. The
2026-08-26 desktop freeze came from UI window churn; nothing here churns.
"""
import pytest

from jarvis.ui import theme
from jarvis.ui.sensing_badge import (TONE_CURFEW, TONE_OFF, TONE_ON,
                                     badge_caption, badge_colors, badge_tone,
                                     badge_word, normalise)


@pytest.fixture(autouse=True)
def _restore_look():
    yield
    theme.select_look(theme.DEFAULT_LOOK)


def _state(**kw):
    from jarvis.sensing import SensingState
    base = dict(camera=True, radar=True, offline=False, reason="")
    base.update(kw)
    return SensingState(**base)


# ------------------------------------------------------------------- tone
def test_the_three_states_map_to_three_tones():
    assert badge_tone(_state()) == TONE_ON
    assert badge_tone(_state(camera=False, reason="curfew")) == TONE_CURFEW
    assert badge_tone(_state(camera=False, radar=False, offline=True,
                             reason="offline")) == TONE_OFF
    assert badge_tone(_state(camera=False, radar=False, offline=True,
                             reason="failsafe")) == TONE_OFF


def test_the_word_says_the_state_not_the_feature_name():
    assert badge_word(_state()) == "SENSING"
    assert badge_word(_state(camera=False, reason="curfew")) == "CAMERA OFF"
    assert badge_word(_state(camera=False, radar=False, offline=True,
                             reason="offline")) == "OFFLINE"


def test_a_dict_from_the_policy_status_reads_the_same_as_the_state():
    from jarvis.sensing import SensingPolicy
    st = _state(camera=False, reason="curfew")
    assert normalise(st) == normalise({"camera": False, "radar": True,
                                       "offline": False, "reason": "curfew"})
    assert badge_tone(SensingPolicy.status(_Policy(st))) == TONE_CURFEW


class _Policy:
    """A stand-in with just the attribute SensingPolicy.status() reads."""

    def __init__(self, state):
        self._state = state

    def state(self):
        return self._state


# ------------------------------------------------------------------ colour
@pytest.mark.parametrize("look", theme.LOOKS)
def test_every_tone_is_legible_in_both_looks(look):
    """The indicator has to read in classic AND holo -- they have different
    grounds, so a colour picked against one of them is the usual way an
    overlay goes invisible."""
    theme.select_look(look)
    for tone in (TONE_ON, TONE_CURFEW, TONE_OFF):
        c = badge_colors(tone)
        assert c["ink"] != theme.BG, (look, tone)
        assert c["dot"] != theme.BG, (look, tone)
        # the ink is one of the light text tokens, never structure cyan
        assert c["ink"] in (theme.FOCAL, theme.INK), (look, tone)


@pytest.mark.parametrize("look", theme.LOOKS)
def test_the_three_tones_are_told_apart_without_reading_the_word(look):
    theme.select_look(look)
    signatures = {(badge_colors(t)["dot"], badge_colors(t)["filled"])
                  for t in (TONE_ON, TONE_CURFEW, TONE_OFF)}
    assert len(signatures) == 3, look


def test_offline_is_the_only_hollow_dot():
    """Colour alone would not survive a bad monitor or a colour-blind
    glance; "nothing is lit" is the shape that carries it."""
    assert badge_colors(TONE_OFF)["filled"] is False
    assert badge_colors(TONE_ON)["filled"] is True
    assert badge_colors(TONE_CURFEW)["filled"] is True


def test_the_colours_follow_the_look_at_call_time():
    """theme.X is re-derived by select_look, so a captured colour would
    freeze the import-time look -- the trap the theme docstring names."""
    theme.select_look("classic")
    classic = badge_colors(TONE_ON)["ink"]
    theme.select_look("holo")
    assert badge_colors(TONE_ON)["ink"] != classic or theme.FOCAL == classic


# ----------------------------------------------------------------- caption
def test_the_caption_says_why_and_until_when():
    import datetime as dt
    end = dt.datetime(2026, 9, 2, 14, 0).timestamp()
    assert "2 pm" in badge_caption(_state(camera=False, radar=False,
                                          offline=True, reason="timed",
                                          until=end))
    assert "7 am" in badge_caption(_state(camera=False, reason="curfew",
                                          curfew=((21, 0), (7, 0))))
    assert badge_caption(_state(camera=False, radar=False, offline=True,
                                reason="failsafe")).lower().count("state") == 1
    assert "camera" in badge_caption(_state()).lower()


def test_an_unsaved_switch_is_visible_on_the_console_too():
    """A switch that could not be written is a switch that will not survive
    a restart. He is told out loud; the badge has to say it as well, or the
    console would quietly disagree with the spoken line."""
    cap = badge_caption(_state(camera=False, radar=False, offline=True,
                               reason="offline", persisted=False))
    assert "not saved" in cap.lower()


# ------------------------------------------------------ the settings window
def test_the_settings_choices_cover_the_hours_a_curfew_is_worth_setting():
    from jarvis.sensing import (CURFEW_END_CHOICES, CURFEW_START_CHOICES,
                                DEFAULT_CURFEW_END, DEFAULT_CURFEW_START,
                                fmt_hhmm, parse_hhmm)
    assert fmt_hhmm(DEFAULT_CURFEW_START) in CURFEW_START_CHOICES
    assert fmt_hhmm(DEFAULT_CURFEW_END) in CURFEW_END_CHOICES
    for value in CURFEW_START_CHOICES + CURFEW_END_CHOICES:
        assert parse_hhmm(value) is not None, value
    # the two lists must not overlap, or the picker could build a window
    # whose ends are the same time (which curfew() reads as "no curfew")
    assert not set(CURFEW_START_CHOICES) & set(CURFEW_END_CHOICES)
