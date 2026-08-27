"""FOUND 2026-08-26 (adversarial UI re-verification): two defects on
plan_strip()'s PROJECT path — the branch a previous sweep left untested
because no Claude project was ever active on the live window.

jarvis/ui/views.py:137-162.

1. MEMORY stays hidden after the chip itself has yielded.  The yield loop
   appends "MEMORY" to `hidden`, retries, and when the value STILL does
   not fit it returns `(0, hidden)` — chip not drawn, MEMORY gone anyway.
   The bar then shows strictly LESS than it would with no project at all
   (slug_len == 0 returns `(0, [])` and keeps every segment), and the
   space MEMORY vacated is left empty.  Measured at S=2.0 with the live
   telemetry widths: at w=920 with slug 'jarvis',
   plan_strip -> (0, ['MEMORY']); with no project -> (0, []).

2. The chip gets LONGER as the window gets NARROWER.  The value budget
   shrinks with the width until it drops under PROJECT_MIN_CHARS, at
   which point MEMORY yields and the freed 274 px go straight back into
   the value.  Measured with slug 'a-very-long-project-slug-name':
   w=1220 -> 'A-VER…' (6 chars); w=1200 -> 'A-VERY-LONG-PROJECT-SLU…'
   (24 chars).  Dragging the window narrower makes the project name jump
   four times longer.

Both are pure-function defects; no Tk root is needed to see them.  The
fix for both is the same shape: after a segment yields, re-clamp the
value budget to what it was allowed before the yield (or hide the chip
first and only then let telemetry yield), and never leave a segment
hidden once the chip has been dropped.
"""
import pytest

from jarvis.ui.views import fmt_project_chip, plan_strip

# device px measured at S=2.0 against the live 1040x1760 window
LEFT_W = 267                                    # '○ WAKE WORD OFF' + PAD_S
SEG_WS = [("CPU", 218), ("GPU", 218), ("MEMORY", 274)]
PROJ_FIXED = 16 + 116 + 12 + 16                 # PAD_S + 'PROJECT' + gap + PAD_S
CHAR_W = 14                                     # JetBrains Mono at SIZE_CAPTION


@pytest.mark.xfail(reason="plan_strip returns (0, hidden) — it drops the chip "
                          "but never puts the yielded MEMORY segment back",
                   strict=True)
def test_memory_returns_when_the_chip_itself_yields():
    chars, hidden = plan_strip(920, LEFT_W, PROJ_FIXED, CHAR_W, len("jarvis"),
                               SEG_WS)
    assert chars == 0                     # the chip did not fit and is dropped
    assert hidden == [], (
        "the PROJECT chip is not drawn, so the bar is identical to the "
        f"no-project bar — which keeps every segment — yet hidden={hidden}")


@pytest.mark.xfail(reason="once MEMORY yields, the freed width goes back into "
                          "the chip, so a narrower bar shows a longer slug",
                   strict=True)
def test_chip_never_grows_as_the_bar_narrows():
    slug = "a-very-long-project-slug-name"
    prev = None
    for w in range(1500, 1000, -20):
        chars, _hidden = plan_strip(w, LEFT_W, PROJ_FIXED, CHAR_W, len(slug),
                                    SEG_WS)
        chip = fmt_project_chip(slug, chars)
        if prev is not None:
            assert len(chip) <= len(prev), (
                f"bar narrowed to {w} px and the chip GREW from "
                f"{len(prev)} to {len(chip)} chars ({prev!r} -> {chip!r})")
        prev = chip
