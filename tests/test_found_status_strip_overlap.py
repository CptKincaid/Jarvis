"""FOUND 2026-08-26 (UI sweep): the status strip's telemetry cluster runs
UNDER the wake-word text when the window is narrow and no Claude project
is active.

StatusStrip._layout() lays the CPU / GPU / MEMORY cluster out right->left
from `w - PAD - PAD_S` and only hides a segment when plan_strip() says so
(jarvis/ui/views.py ~1700-1740).  plan_strip() short-circuits on
`if slug_len <= 0: return 0, []` (jarvis/ui/views.py ~150) — so with no
PROJECT chip on the bar, MEMORY is never asked to yield and the cluster
simply keeps marching left past the wake-word segment.

The StatusStrip docstring promises "nothing ever overlaps or clips".

Measured on the live 1040x1760 window (S=2.0): wake-word hairline x=268,
CPU hairline x=329, i.e. the whole bar wants 979 device px.  The window's
own minimum is 920x1440 (px(MIN_W=460)), so the user can always drag into
the collision — at 920 the CPU label is drawn on top of "OFF"
(scratchpad shot s2_narrow613_win.png / s2_statusbar_zoom.png).

Fix belongs in plan_strip: the no-project path must still be allowed to
hide MEMORY (and then GPU) when left_w + right > total_w.

2026-09-01: the holo look sidesteps it in StatusStrip._layout with
plan_telemetry (a compact 'CPU 53° | GPU 44° | 47.5 GB' level when the
full cluster would run under the wake word; tests/test_ui_chrome.py). The
classic look deliberately kept level 0 only -- it is the fallback that
must render exactly as the 08-31 console did, this collision included --
so the plan_strip path below was still the open defect for classic.

2026-09-02: classic elides too (views.levels_for_look). Answering "is
that free or used?" changed the level-0 memory value in BOTH looks, so
classic's pixels moved anyway and its cluster grew 710 -> 718 px; keeping
the contract would have meant an overprint 8 px DEEPER rather than none.
The strip no longer reaches the collision in either look. plan_strip
itself is unfixed, and the assertion below still pins that: it is the
PROJECT-chip path, where a no-project bar still refuses to let MEMORY
yield.
"""
import pytest

from jarvis.ui.views import plan_strip


@pytest.mark.xfail(reason="plan_strip returns early when slug_len <= 0, so "
                          "the telemetry cluster never yields and overlaps "
                          "the wake-word text on a narrow bar",
                   strict=True)
def test_no_project_telemetry_yields_before_overlapping_wake_word():
    # device px measured off the live window at S=2.0
    left_w = 268                       # '○ WAKE WORD OFF' + PAD_S hairline
    seg_ws = [("CPU", 218), ("GPU", 218), ("MEMORY", 275)]   # 711 total
    total_w = 920                      # the app's own minimum width
    chars, hidden = plan_strip(total_w, left_w, 0, 14, 0, seg_ws)
    right = sum(w for name, w in seg_ws if name not in hidden)
    assert left_w + right <= total_w, (
        f"telemetry cluster needs {right}px and the wake-word segment "
        f"{left_w}px in a {total_w}px bar; hidden={hidden}")
