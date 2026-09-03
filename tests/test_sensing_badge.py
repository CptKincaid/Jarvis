"""The console's sensing indicator, tested display-free.

House pattern (tests/test_holo_geometry.py): every decision the badge
makes is a pure function, so the one thing a screenshot review would argue
about -- is this legible, and can the three states be told apart at a
glance -- is asserted here without ever creating a toplevel window. The
2026-08-26 desktop freeze came from UI window churn; nothing here churns.
"""
import pytest

from jarvis.ui import theme
from jarvis.ui.sensing_badge import (DOT_DISC, DOT_HALF, DOT_RING,
                                     TONE_CURFEW, TONE_OFF, TONE_ON,
                                     badge_caption, badge_colors, badge_tone,
                                     badge_word, normalise,
                                     sensing_failsafe_state)


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
    assert badge_word(_state(camera=False, reason="curfew")) == "CAM OFF"
    assert badge_word(_state(camera=False, radar=False, offline=True,
                             reason="offline")) == "OFFLINE"


def test_every_word_is_seven_glyphs_so_the_chip_does_not_twitch():
    """'CAMERA OFF' became 'CAM OFF' on 2026-09-03 to fit a header that
    may not shrink its wordmark. Seven glyphs in every tone means the
    capsule barely changes width between states (135/140/141 px at S=2,
    measured on Xvfb through the real widget), so it reads as one steady
    mark rather than a thing that jumps -- and
    it is still a WORD, not a bare dot, which is the whole reason this
    readout is in the header. The unabbreviated reason stays reachable on
    badge_caption, which MainWindow feeds the badge's tooltip."""
    from jarvis.ui.sensing_badge import WORDS
    assert {len(word) for word in WORDS.values()} == {7}
    curfew = _state(camera=False, reason="curfew", curfew=((21, 0), (7, 0)))
    assert badge_word(curfew) == "CAM OFF"
    assert badge_caption(curfew).startswith("camera off until")


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
    """Each channel on its own separates at least two of the three, and
    the SHAPE separates all three by itself: CAM OFF and OFFLINE are both
    amber and both seven glyphs, so with colour gone the dot's shape is
    the only tell left that is not the word."""
    theme.select_look(look)
    tones = (TONE_ON, TONE_CURFEW, TONE_OFF)
    assert len({badge_colors(t)["shape"] for t in tones}) == 3, look
    assert len({badge_colors(t)["dot"] for t in tones}) == 2, look
    assert len({(badge_colors(t)["dot"], badge_colors(t)["shape"])
                for t in tones}) == 3, look


def test_the_dot_is_full_half_or_empty_by_how_much_is_lit():
    """Colour alone would not survive a bad monitor or a colour-blind
    glance. Everything lit is a disc; the curfew -- radar on, camera off
    -- is HALF a disc, half the sensors lit; offline is a hollow ring,
    nothing lit. Before 2026-09-03 the curfew dot was a full amber disc,
    6 px of chip width and one word away from OFFLINE's ring."""
    assert badge_colors(TONE_ON)["shape"] == DOT_DISC
    assert badge_colors(TONE_CURFEW)["shape"] == DOT_HALF
    assert badge_colors(TONE_OFF)["shape"] == DOT_RING
    assert len({DOT_DISC, DOT_HALF, DOT_RING}) == 3
    assert "filled" not in badge_colors(TONE_ON)      # the old bool is gone


@pytest.mark.parametrize("look", theme.LOOKS)
def test_the_badge_never_wears_the_pills_edge(look):
    """The two header chips are one face and one geometry since
    2026-09-03, so the badge's edge -- the capsule outline in holo, the
    1 px catch-light in classic -- is its own token, never the pill's
    GLASS_EDGE (widgets.StatePill._fit draws GLASS_EDGE in both looks).
    Honest about its weight: live, CYAN_DIM sits 1.17:1 (holo) / 1.41:1
    (classic) from GLASS_EDGE, a hue on a hairline that this token-level
    test can read and a glance cannot; restricted, WARN is plain. The
    glance-level tell in holo is the pill's corner tick (next test)."""
    theme.select_look(look)
    for tone in (TONE_ON, TONE_CURFEW, TONE_OFF):
        edge = badge_colors(tone)["edge"]
        assert edge != theme.GLASS_EDGE, (look, tone)
        assert edge != theme.BG, (look, tone)
    assert badge_colors(TONE_ON)["edge"] == theme.CYAN_DIM
    assert badge_colors(TONE_CURFEW)["edge"] == theme.WARN
    assert badge_colors(TONE_OFF)["edge"] == theme.WARN
    assert _contrast(theme.CYAN_DIM, theme.GLASS_EDGE) < 1.5    # hue-only
    assert _contrast(theme.WARN, theme.GLASS_EDGE) > 2          # amber reads


def _contrast(a: str, b: str) -> float:
    """WCAG relative-luminance contrast of two #rrggbb tokens."""
    def lum(h):
        def ch(i):
            c = int(h[i:i + 2], 16) / 255
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * ch(1) + 0.7152 * ch(3) + 0.0722 * ch(5)
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_in_holo_the_pills_corner_tick_is_the_glance_tell_and_the_badge_draws_none():
    """Reviewed 2026-09-03: the badge's CYAN_DIM outline against the pill's
    GLASS_EDGE is a hue shift (1.17:1 in holo), so what tells the chips
    apart at a glance is the BRIGHT tick on the pill's left cut, which
    the badge does not draw. That absence is DELIBERATE and pinned here:
    a tick in any token the badge owns would sit a hue from the pill's
    (BRIGHT vs CYAN 1.31:1, vs WARN 1.18:1, computed from the tokens; the
    cut pixels confirmed off a private Xvfb at S=1 and S=2), and two
    near-equal ticks would erase the one luminance-level asymmetry, not
    add a channel. Source-inspected, the house way: no window is made."""
    import inspect

    from jarvis.ui.sensing_badge import SensingBadge
    from jarvis.ui.widgets import StatePill
    theme.select_look("holo")
    assert "theme.BRIGHT" in inspect.getsource(StatePill._fit)
    assert "theme.BRIGHT" not in inspect.getsource(SensingBadge._fit)
    for tone in (TONE_ON, TONE_CURFEW, TONE_OFF):
        assert theme.BRIGHT not in badge_colors(tone).values(), tone
    # the tick is a real mark against the ground, and nothing the badge
    # owns would be a real mark against the tick
    assert _contrast(theme.BRIGHT, theme.BG) > 9
    for own in (theme.CYAN, theme.WARN, theme.CYAN_DIM):
        assert _contrast(theme.BRIGHT, own) < 1.7, own


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


def test_with_no_policy_at_all_the_badge_reads_offline():
    """A header still reading SENSING because the owner failed to construct
    would be the console asserting the one thing nobody can check -- and
    the app hands the sensors a denying stand-in in exactly that case, so
    OFFLINE is also the truth."""
    st = sensing_failsafe_state()
    assert badge_tone(st) == TONE_OFF
    assert badge_word(st) == "OFFLINE"
    assert "state" in badge_caption(st).lower()


# ------------------------------------------------------ the settings window
def test_the_privacy_rows_are_re_read_every_time_the_drawer_opens():
    """Every OTHER row in the drawer is only ever changed from the drawer,
    so building it once is enough. This one is not: offline mode's primary
    control is VOICE, so the switch flips with the drawer shut and a toggle
    read at construction then shows the opposite of the badge two inches
    away in the same header.
    """
    import inspect
    import types

    from jarvis.sensing import SensingState
    from jarvis.ui.views import SettingsDrawer

    assert "_refresh_privacy" in inspect.getsource(SettingsDrawer.open), \
        "the refresh has to run on open(), not only at construction"

    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    class Tog:
        value = None
        animated = None

        def set(self, value, animate=True):
            self.value, self.animated = bool(value), animate

    class Pol:
        def state(self):
            return SensingState(camera=False, radar=False, offline=True,
                                reason="offline")

        def curfew(self):
            return ((22, 0), (6, 30))

    drawer = types.SimpleNamespace(
        _sensing=Pol, _sensing_toggle=Tog(), _curfew_quiet=False,
        _curfew_start=Var("21:00"), _curfew_end=Var("07:00"))
    SettingsDrawer._refresh_privacy(drawer)
    assert drawer._sensing_toggle.value is True
    assert drawer._sensing_toggle.animated is False, \
        "animate=False is also what stops the read-back writing the switch"
    assert drawer._curfew_start.get() == "22:00"
    assert drawer._curfew_end.get() == "06:30"
    assert drawer._curfew_quiet is False, "the guard has to be put back"


def test_the_read_back_does_not_write_the_curfew_it_just_read():
    """Setting a picker's StringVar fires its write trace, so the refresh
    would save the window straight back -- and toast about it."""
    import types

    from jarvis.ui.views import SettingsDrawer

    def boom():
        raise AssertionError("the read-back wrote the config back")

    SettingsDrawer._curfew_changed(
        types.SimpleNamespace(_curfew_quiet=True, _sensing=boom))



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
