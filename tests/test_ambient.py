"""The standby / ambient slab's pure geometry and palette helpers
(jarvis/ui/ambient.py) plus the board's call-time tone lookup.

Everything here is a pure function of its inputs: no Tk widget is created,
no clock is read, no display is opened. The RoomSlab widget itself is
covered by the Xvfb screenshot rounds (scratchpad/holo/w2/C/), which no
unit test may reproduce (conftest: no Tk in unit tests).

Written with the W2 holo overhaul, 2026-09-01 (lane C). The rules under
test are the ones the shots taught us: an empty room draws no frame, a
frame fits its content rather than the whole slab, brackets never cross on
a short edge, and every colour is read from theme at CALL time so the
look chosen at start-up is the look that renders.
"""
from __future__ import annotations

import pytest

from jarvis.ui import ambient
from jarvis.ui import board as ui_board
from jarvis.ui import console_mode as cm
from jarvis.ui import theme


@pytest.fixture(autouse=True)
def _restore_look():
    """A look switch inside a test must never leak into the next file:
    the suite imports with holo selected and every module reads theme.X
    at call time, so a stray classic would silently change other tests'
    expectations."""
    yield
    theme.select_look(theme.DEFAULT_LOOK)


# ------------------------------------------------------------- tracked
def test_tracked_letter_spaces_caps_with_real_spaces():
    # Chakra Petch has no thin-space glyph, so the tracking is a real
    # space between letters and a wider one between words.
    assert ambient.tracked("NEXT") == "N E X T"
    assert ambient.tracked("LAB REPORT") == "L A B   R E P O R T"


def test_tracked_collapses_whitespace_so_a_pre_spaced_value_cannot_double():
    assert ambient.tracked("  DUE   NOW ") == "D U E   N O W"
    assert ambient.tracked("N E X T") == "N   E   X   T"   # already spaced = words
    assert ambient.tracked("") == ""
    assert ambient.tracked(None) == ""


def test_tracked_gap_and_word_gap_are_tunable():
    assert ambient.tracked("AB CD", gap="-", word_gap="|") == "A-B|C-D"


# ----------------------------------------------------------- frame_box
def test_frame_box_is_none_for_an_empty_room():
    # Same rule as the rows: nothing to say, nothing drawn — not an empty
    # box in the middle of the standby screen.
    assert ambient.frame_box(600, 100, 0, 26, frac=0.64, min_w=300,
                             side_pad=20, inset_y=8) is None
    assert ambient.frame_box(600, 100, -3, 26, frac=0.64, min_w=300,
                             side_pad=20, inset_y=8) is None


def test_frame_box_is_centred_and_takes_its_share_of_the_slab():
    x0, y0, x1, y1 = ambient.frame_box(1000, 100, 3, 26, frac=0.64, min_w=300,
                                       side_pad=20, inset_y=8)
    assert x1 - x0 == 640
    assert x0 == (1000 - 640) // 2 and x1 == x0 + 640
    # half a row past the first and last text centre, plus the inset
    assert y0 == 100 - 13 - 8
    assert y1 == 100 + 2 * 26 + 13 + 8


def test_frame_box_never_narrower_than_min_w_nor_wider_than_the_slab():
    x0, _, x1, _ = ambient.frame_box(400, 100, 1, 26, frac=0.5, min_w=300,
                                     side_pad=20, inset_y=0)
    assert x1 - x0 == 300                       # min_w wins over 0.5*400
    x0, _, x1, _ = ambient.frame_box(200, 100, 1, 26, frac=0.5, min_w=300,
                                     side_pad=20, inset_y=0)
    assert x1 - x0 == 200 - 2 * 20              # the slab less side_pad wins
    assert x0 == 20 and x1 == 180


def test_frame_box_single_row_is_one_row_tall_plus_insets():
    _, y0, _, y1 = ambient.frame_box(600, 50, 1, 30, frac=0.64, min_w=300,
                                     side_pad=20, inset_y=5)
    assert (y0, y1) == (50 - 15 - 5, 50 + 15 + 5)


# ------------------------------------------------ hud_frame_points
def test_hud_frame_cuts_all_four_corners():
    pts = ambient.hud_frame_points(0, 0, 100, 50, 7)
    assert len(pts) == 16
    xs, ys = pts[0::2], pts[1::2]
    assert min(xs) == 0 and max(xs) == 100 and min(ys) == 0 and max(ys) == 50
    # no corner point sits ON a corner: every one is `cut` along an edge
    corners = {(0, 0), (100, 0), (100, 50), (0, 50)}
    assert not corners & set(zip(xs, ys))
    assert (7, 0) in zip(xs, ys) and (0, 7) in zip(xs, ys)
    assert (93, 50) in zip(xs, ys) and (100, 43) in zip(xs, ys)


def test_hud_frame_cut_is_clamped_so_a_tiny_box_still_closes():
    pts = ambient.hud_frame_points(0, 0, 10, 8, 40)
    xs, ys = pts[0::2], pts[1::2]
    assert min(xs) >= 0 and max(xs) <= 10 and min(ys) >= 0 and max(ys) <= 8
    # cut clamps to min(w//2, h//2) = 4
    assert (4, 0) in zip(xs, ys) and (6, 0) in zip(xs, ys)
    assert ambient.hud_frame_points(0, 0, 100, 50, 0)[:4] == [0, 0, 100, 0]


def test_hud_frame_negative_cut_is_treated_as_none():
    assert ambient.hud_frame_points(0, 0, 100, 50, -5) == \
        ambient.hud_frame_points(0, 0, 100, 50, 0)


# ---------------------------------------------- hud_bracket_points
def test_hud_brackets_are_four_polylines_following_the_diagonal():
    brs = ambient.hud_bracket_points(0, 0, 100, 50, 7, 16)
    assert len(brs) == 4
    for br in brs:
        assert len(br) == 8                     # four points, flat
    tl, tr, brr, bl = brs
    # top-left: down the left edge, across the diagonal, along the top
    assert tl == [0, 23, 0, 7, 7, 0, 23, 0]
    assert tr == [77, 0, 93, 0, 100, 7, 100, 23]
    assert brr == [100, 27, 100, 43, 93, 50, 77, 50]
    assert bl == [23, 50, 7, 50, 0, 43, 0, 27]


def test_hud_bracket_arms_never_cross_on_a_short_edge():
    # 40 wide, cut 7: each arm may be at most 20 - 7 = 13 so the two
    # top brackets meet at the centre instead of overlapping.
    brs = ambient.hud_bracket_points(0, 0, 40, 200, 7, 60)
    tl, tr = brs[0], brs[1]
    assert tl[6] == 20 and tr[0] == 20         # arm ends meet at x=20
    assert tl[1] == 7 + 13                      # vertical arm clamped too


def test_hud_bracket_points_stay_inside_the_box():
    for w, h, cut, ln in [(100, 50, 7, 16), (30, 30, 7, 16), (12, 300, 7, 40),
                          (5, 5, 7, 16)]:
        for br in ambient.hud_bracket_points(10, 20, 10 + w, 20 + h, cut, ln):
            xs, ys = br[0::2], br[1::2]
            assert min(xs) >= 10 and max(xs) <= 10 + w
            assert min(ys) >= 20 and max(ys) <= 20 + h


# --------------------------------------------------------- separator_ys
def test_separators_sit_between_rows_and_never_after_the_last():
    assert ambient.separator_ys(100, 3, 26) == [113, 139]
    assert ambient.separator_ys(100, 1, 26) == []
    assert ambient.separator_ys(100, 0, 26) == []
    assert ambient.separator_ys(100, -2, 26) == []


# ---------------------------------------------------------- glow_span
def test_glow_span_is_ninety_percent_of_the_digits_centred():
    x0, x1 = ambient.glow_span(500, 400, 120, 900)
    assert x1 - x0 == 360 and (x0 + x1) // 2 == 500


def test_glow_span_is_clamped_both_ways():
    x0, x1 = ambient.glow_span(500, 20, 120, 900)      # '1:11' still gets a line
    assert x1 - x0 == 120
    x0, x1 = ambient.glow_span(500, 5000, 120, 900)    # never past the slab
    assert x1 - x0 == 900
    x0, x1 = ambient.glow_span(500, 5000, 120, 50)     # max below min: min wins
    assert x1 - x0 == 120


# --------------------------------------------------------- glow_steps
def test_glow_steps_nest_around_one_centre_longest_first():
    spans = ambient.glow_steps(100, 300)
    assert len(spans) == len(ambient.GLOW_STEPS)
    assert spans[0] == (100, 300)
    widths = [b - a for a, b in spans]
    assert widths == sorted(widths, reverse=True)
    for a, b in spans:
        assert (a + b) // 2 == 200 and a >= 100 and b <= 300


def test_glow_steps_never_collapse_to_nothing():
    for a, b in ambient.glow_steps(100, 101, steps=(1.0, 0.1, 0.0)):
        assert b - a >= 2                       # a 1px glow still draws
    assert ambient.glow_steps(0, 100, steps=(2.0,)) == [(0, 100)]  # >1 clamps


# --------------------------------------------------------- band_frame
def test_band_frame_fits_its_rows_rather_than_the_whole_slab():
    # r1 shot (09-01): a frame around the empty lower half read as a box
    # around nothing, so the frame stops half a row past the last row.
    x0, y0, x1, y1 = ambient.band_frame(800, 900, 10, 60, 100, 3, 26, 14)
    assert (x0, y0, x1) == (10, 10, 790)
    assert y1 == 100 + 2 * 26 + 13 + 14
    assert y1 < 900 - 10


def test_band_frame_with_no_rows_closes_just_under_the_header_rail():
    _, _, _, y1 = ambient.band_frame(800, 900, 10, 60, 100, 0, 26, 14)
    assert y1 == 60 + 14


def test_band_frame_is_clamped_to_a_short_canvas():
    _, _, _, y1 = ambient.band_frame(800, 120, 10, 60, 100, 6, 26, 14)
    assert y1 == 120 - 10


# ------------------------------------------------------ rail_segments
def test_rail_segments_split_the_lit_lead_from_the_dim_run():
    assert ambient.rail_segments(0, 100, 0.22) == ((0, 22), (22, 100))
    assert ambient.rail_segments(50, 150, 0.5) == ((50, 100), (100, 150))


def test_rail_segments_clamp_the_lit_fraction():
    assert ambient.rail_segments(0, 100, 5.0) == ((0, 100), (100, 100))
    assert ambient.rail_segments(0, 100, -1.0) == ((0, 0), (0, 100))


# --------------------------------------------- palette at call time
def test_quiet_ink_is_read_from_the_theme_when_asked(monkeypatch):
    assert ambient.quiet_ink() == theme.WARN
    # Both looks happen to share the ember today, so the proof that it is
    # a call-time read is a swapped token, not a look switch: the old
    # `QUIET_INK = theme.WARN` would still answer the import-time value.
    monkeypatch.setattr(theme, "WARN", "#123456")
    assert ambient.quiet_ink() == "#123456"
    assert ambient.slab_ink("quiet")[0] == "#123456"


def test_slab_ink_quiet_hours_take_the_whole_palette_ember():
    head, body, faint = ambient.slab_ink("quiet")
    assert head == theme.WARN
    assert body == cm.dim(theme.WARN, 0.7)
    assert faint == theme.FAINT


def test_slab_ink_away_drops_the_contrast():
    assert ambient.slab_ink("away") == (theme.MUTED, theme.MUTED, theme.FAINT)


def test_slab_ink_normal_is_the_focal_ladder_in_classic():
    theme.select_look("classic")
    assert ambient.slab_ink("normal") == (theme.FOCAL, theme.INK, theme.MUTED)
    # and the explicit look argument beats the module state
    theme.select_look("holo")
    assert ambient.slab_ink("normal", look="classic")[0] == theme.FOCAL


def test_slab_ink_holo_clock_is_focal_the_top_of_the_ladder():
    """09-01 stepped the holo clock to CORE_BANDS[2] (#a8e9ff) for a cool
    cast; measured on 17-standby (U09, 09-03) that made the clock the
    third brightest text on its own slab. FOCAL now, in both looks."""
    theme.select_look("holo")
    head, body, faint = ambient.slab_ink("normal")
    assert head == theme.FOCAL
    assert head not in theme.CORE_BANDS
    assert (body, faint) == (theme.INK, theme.MUTED)
    assert not hasattr(ambient, "HOLO_CLOCK_BAND")


def test_slab_ink_reads_the_theme_when_called_not_when_imported():
    theme.select_look("classic")
    classic = ambient.slab_ink("normal")
    theme.select_look("holo")
    holo = ambient.slab_ink("normal")
    assert classic != holo                  # INK / MUTED differ per look
    assert holo == (theme.FOCAL, theme.INK, theme.MUTED)


# ------------------------------------------------- board tone colours
def test_board_tone_colour_is_resolved_at_call_time():
    theme.select_look("holo")
    assert ui_board.tone_color("ok") == theme.FOCAL
    assert ui_board.tone_color("warn") == theme.WARN
    assert ui_board.tone_color("error") == theme.ERR
    assert ui_board.tone_color("idle") == theme.CYAN_DIM
    assert ui_board.tone_color("off") == theme.FAINT
    # FAINT is one of the tokens the two looks disagree on, so the "off"
    # tone is where a frozen dict would show: the old TONE_COLORS answered
    # the import-time look for the whole session.
    holo_off = theme.FAINT
    theme.select_look("classic")
    assert ui_board.tone_color("off") == theme.FAINT
    assert ui_board.tone_color("off") != holo_off


def test_board_unknown_tone_falls_back_to_the_idle_cyan():
    assert ui_board.tone_color("nonsense") == theme.CYAN_DIM
    assert ui_board.tone_color(None) == theme.CYAN_DIM


def test_every_board_tone_token_exists_on_the_theme():
    for token in ui_board.TONE_TOKENS.values():
        assert hasattr(theme, token), token


# ------------------------------------------------- measured-width memo
def _fake_measure(calls):
    def measure(font_spec, text):
        calls.append((font_spec, text))
        return 10 * len(text)                 # 10 px a character
    return measure


def test_measured_asks_tk_once_per_font_and_text(monkeypatch):
    calls = []
    monkeypatch.setattr(ambient, "measure", _fake_measure(calls))
    monkeypatch.setattr(ambient, "_MEASURE_CACHE", {})
    font = ("Chakra Petch", -24, "bold")
    assert ambient.measured(font, "7:30") == 40
    assert ambient.measured(font, "7:30") == 40
    assert ambient.measured(font, "7:31") == 40
    # a different spec (another scale, another family) is a miss, never a
    # stale width borrowed from the old font
    assert ambient.measured(("Chakra Petch", -48, "bold"), "7:30") == 40
    assert len(calls) == 3


def test_measured_clears_rather_than_grows_without_bound(monkeypatch):
    calls = []
    monkeypatch.setattr(ambient, "measure", _fake_measure(calls))
    monkeypatch.setattr(ambient, "_MEASURE_CACHE", {})
    monkeypatch.setattr(ambient, "_MEASURE_CACHE_MAX", 4)
    for i in range(10):
        ambient.measured(("f", -10, "normal"), str(i))
    assert len(ambient._MEASURE_CACHE) <= 4


def test_measured_raises_what_tk_raises(monkeypatch):
    def boom(_font, _text):
        raise RuntimeError("no default root")
    monkeypatch.setattr(ambient, "measure", boom)
    monkeypatch.setattr(ambient, "_MEASURE_CACHE", {})
    with pytest.raises(RuntimeError):
        ambient.measured(("f", -10, "normal"), "x")


def test_fitted_matches_ellipsize_and_survives_a_dead_tk(monkeypatch):
    calls = []
    monkeypatch.setattr(ambient, "measure", _fake_measure(calls))
    monkeypatch.setattr(ambient, "_MEASURE_CACHE", {})
    font = ("f", -10, "normal")
    assert ambient.fitted("LAB REPORT", font, 200) == "LAB REPORT"
    # 10 px a char: 'LAB…' is 40 px, the widest that fits 45
    assert ambient.fitted("LAB REPORT", font, 45) == "LAB…"
    n = len(calls)
    assert ambient.fitted("LAB REPORT", font, 45) == "LAB…"
    assert len(calls) == n                   # the second ask cost nothing

    def boom(_font, _text):
        raise RuntimeError("no default root")
    monkeypatch.setattr(ambient, "measure", boom)
    monkeypatch.setattr(ambient, "_MEASURE_CACHE", {})
    assert ambient.fitted("LAB REPORT", font, 45) == "LAB REPORT"
