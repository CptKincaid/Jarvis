"""Display-free geometry of the holo stage decor (jarvis/ui/reactor.py).

The helpers under test are pure, so the numbers a screenshot review
argued over on 2026-09-01 can be asserted without a Tk window. Three
came out of that review and all three are here: the engine card's right
edge sat 1 design px short of the rail's long ticks (the card read as a
toothed comb glued to the frame), the radar sweep's lead and outer orbit
crossed the frame's LEFT rule once a revolution, and the ruler caption
"600 F" read as a Fahrenheit temperature next to the strip's "CPU 76°".

Classic has no frame and no rail, so every one of these must leave it at
its pre-overhaul value -- it is the fallback that renders exactly as the
08-31 console did.
"""
import pytest

from jarvis.ui import theme
from jarvis.ui.reactor import (AV_FRAMES, AV_PERIOD, CARD_RAIL_GAP, CARD_W,
                               DECOR_MARGIN, FRAME_INSET, FRAME_KEEP,
                               MAX_RENDER, MAX_SIZE, MIN_SIZE, RAIL_TICK,
                               STAGE_CX, SWEEP_LEAD, dynamic_r_limit,
                               engine_card_x1, ruler_caption, sweep_radii)
from jarvis.ui.widgets import get_scale, px, set_scale

# The stage (w, h) per UI scale, MEASURED off the judge harness: his
# desktop runs JARVIS_UI_SCALE 2.0 and the reactor gets 918x520; the same
# window at S=1.0 gets 918x403 (the console keeps its pixel geometry, the
# design units halve). Both are checked because the 09-01 review found the
# frame crossing proportionally WORSE at S=1.
STAGE = {1.0: (918, 403), 2.0: (918, 520)}


@pytest.fixture(autouse=True)
def _restore():
    """The look and the UI scale are module globals; put both back."""
    scale = get_scale()
    yield
    set_scale(scale)
    theme.select_look(theme.DEFAULT_LOOK)
    theme.apply_scale(scale)


def _stage(scale: float):
    """(w, h, size, Rr, cx) the way Reactor lays the stage out: the base
    size _maybe_rescale settles on, the ruler radius, and the cluster
    centre _cluster_xy clamps against the engine card."""
    set_scale(scale)
    theme.apply_scale(scale)
    w, h = STAGE[scale]
    avail = min(int(w * 2 * STAGE_CX), h)
    size = max(px(MIN_SIZE),
               min(min(px(MAX_SIZE), MAX_RENDER), avail - px(DECOR_MARGIN)))
    rr = size * 0.5 + px(6)
    card_x0 = engine_card_x1(w) - px(CARD_W)
    cx = min(round(w * STAGE_CX), int(card_x0 - px(12) - (rr + px(22))))
    return w, h, size, rr, cx


# ------------------------------------------------------ the ruler caption
def test_the_ruler_caption_cannot_be_read_as_a_temperature():
    assert ruler_caption(600, 10.0) == "600 FRAMES  ·  60 HZ"
    assert "F " not in ruler_caption() and "°" not in ruler_caption()
    # it defaults to the cycle this build actually bakes
    assert ruler_caption() == "%d FRAMES  ·  %d HZ" % (
        AV_FRAMES, round(AV_FRAMES / AV_PERIOD))


# ---------------------------------------------------- the engine card's x1
@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_the_engine_card_clears_the_rail_ticks_in_holo_only(scale):
    w, h, size, rr, cx = _stage(scale)
    tip = w - px(FRAME_INSET) - px(RAIL_TICK[1])      # long-tick tips
    assert engine_card_x1(w, "holo") == tip - px(CARD_RAIL_GAP)
    assert tip - engine_card_x1(w, "holo") >= px(3)   # not a comb
    # classic: PAD from the stage edge, exactly as it always was
    assert engine_card_x1(w, "classic") == w - theme.PAD
    theme.select_look("classic")
    assert engine_card_x1(w) == w - theme.PAD         # follows theme.LOOK
    theme.select_look("holo")
    assert engine_card_x1(w) == tip - px(CARD_RAIL_GAP)


# ------------------------------------------- the sweep vs the frame's rule
@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_no_dynamic_element_reaches_the_holo_frames_left_rule(scale):
    """The lead's tip, the trail arcs and the outermost orbit all live at
    r_lim from the cluster centre; in holo that has to stay FRAME_KEEP
    inside the frame line at px(FRAME_INSET), or the blade crosses a 1px
    hairline every revolution (09-01 review measured the tip at x=4 with
    the rule at x=20)."""
    w, h, size, rr, cx = _stage(scale)
    r_lim = dynamic_r_limit(h, cx, rr + px(8), "holo")
    assert cx - r_lim >= px(FRAME_INSET) + px(FRAME_KEEP)
    assert r_lim <= h // 2 - px(4)                    # stage bound still holds
    lo, hi = sweep_radii(rr, r_lim, "holo")
    assert cx - hi >= px(FRAME_INSET)                 # the tip stays inside
    assert hi - lo == px(SWEEP_LEAD)                  # ...at full length
    assert lo > size * 0.5                            # ...and off the sphere


@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_classic_keeps_the_pre_overhaul_radii_to_the_pixel(scale):
    w, h, size, rr, cx = _stage(scale)
    r_lim = dynamic_r_limit(h, cx, rr + px(8), "classic")
    assert r_lim == h // 2 - px(4)                    # d38b493
    assert sweep_radii(rr, r_lim, "classic") == (rr + px(8),
                                                 min(rr + px(20), r_lim))


def test_the_limit_never_inverts_the_lead_on_a_narrow_stage():
    """A stage narrow enough that the frame rule is inside the ring band
    (a half-width window, the capture tool) must not hand back an outer
    radius below the inner one -- the lead would draw backwards."""
    set_scale(1.0)
    for cx in (0, 20, 40, 80, 160):
        rr = 100.0
        r_lim = dynamic_r_limit(400, cx, rr + px(8), "holo")
        lo, hi = sweep_radii(rr, r_lim, "holo")
        assert hi >= lo, cx
    # and where there IS room the stage bound wins, unchanged from classic
    assert dynamic_r_limit(400, 900, 108.0, "holo") == 400 // 2 - px(4)


def test_the_limit_follows_theme_look_when_no_look_is_passed():
    set_scale(1.0)
    theme.select_look("classic")
    assert dynamic_r_limit(520, 252, 208.0) == 520 // 2 - px(4)
    assert sweep_radii(208.0, 256.0) == (216.0, 228.0)
    theme.select_look("holo")
    assert dynamic_r_limit(520, 252, 208.0) < 520 // 2 - px(4)
    assert sweep_radii(208.0, 256.0) == (216.0, 228.0)   # room: unchanged
    assert sweep_radii(208.0, 220.0) == (208.0, 220.0)   # clamped: slid in
