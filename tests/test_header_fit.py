"""The header's width budget, tested display-free.

2026-09-02, verbatim: "the word sensing is underneath the ready symbol".
2026-09-03, verbatim: "dont make jarvis smaller, just make ready and
sensing smaller to fit".

DIAGNOSIS (unchanged, and still the reason this file exists). Nothing is
misaligned vertically -- SensingBadge and StatePill declare the SAME
26-design-unit height, both are packed with the default anchor into a
56-unit header, so pack centres them on the same line and a baseline
mismatch is impossible. The header is over-SUBSCRIBED, and what Tk does
with an over-subscribed bar is not overlap: generic/tkPack.c
ArrangePacking clamps the cavity at zero (752-756) so a child's frame
SHRINKS, clips the child to its frame less padding (786-790), and UNMAPS
a child left with no width (846-850). The last child packed is the one
that loses, and that child is the sensing badge. views.header_spans
transcribes those three passages; the assertion that matters is
spans_clipped() == [], because an absent badge and a badge reading
SENSING must not look the same from across the room.

WHAT CHANGED. The 09-02 remedy took the pixels out of the WORDMARK,
stepping it down "J A R V I S" -> "JARVIS" -> "J". He rejected that: the
wordmark stays whole in every state, and the two chips give up the space
instead. So the budget is now fixed on both ends --

    header at his 920-px window (shell inset 1 px a side)   918
    PAD + wordmark 'J A R V I S'                          - 322
    close + min + gear and their padding                  - 252
    the two chips' own PAD_S pads                          - 32
    ------------------------------------------------------------
    left for the pill and the badge, in EVERY state         312

-- and 'LISTENING…' + 'CAMERA OFF' asked 485 of those 312.

HOW THE 173 px CAME OUT, SECOND CUT. The badge is the privacy readout,
so it gave up the least: it keeps its type size, its dot, its capsule
and all three of its tell-apart channels, and loses only chip padding
and one word ('CAMERA OFF' -> 'CAM OFF', with the unabbreviated reason
still on the tooltip via badge_caption). The pill drops to the badge's
SIZE_CAPTION face, so the two header chips are one type size and one
geometry, and loses its ellipsis. The FIRST cut (ff3fcf9) also cut the
pill's words to bare imperatives (LISTEN / THINK / SPEAK / WAIT / WORK:
280 of 312) -- on a voice console those read as orders to the user, and
his words were "smaller", not "reworded". MEASURED, the full participles
fit at PAD_X 4 / GAP 4 on both chips at every scale 1.0-3.0 in both
looks (margin +8 / +9 / +6 / +13 / +13 / +31 / +28 px at S = 1.0 / 1.25
/ 1.5 / 1.75 / 2.0 / 2.5 / 3.0; PAD_X 5 / GAP 4 clips 2 px at S=1.5):
271+214 -> 158+141 = 299 of 312, 13 px to spare at his window.

Every width below was MEASURED on a PRIVATE Xvfb (:94, never his :1) at
JARVIS_UI_SCALE 2.0 off the rendered faces ('Chakra Petch SemiBold' -59
for the wordmark, -35 for the old pill, -24 for both chips now; 'Inter'
for the chrome buttons), through the real widgets' winfo_reqwidth, and
kept as literals so the file stays Tk-free, the same house pattern as
tests/test_ui_chrome.py.
"""
import pytest

from jarvis.ui import main_window
from jarvis.ui.main_window import STATE_WORDS, WORDMARK
from jarvis.ui.sensing_badge import WORDS as BADGE_WORDS
from jarvis.ui.sensing_badge import SensingBadge, badge_caption
from jarvis.ui.views import header_spans, spans_clipped
from jarvis.ui.widgets import StatePill

HEADER_W = 918          # his 920-px window, less the shell's 1-px inset
DEFAULT_W = 1038        # the 520-design-unit default window
PAD, PAD_S = 32, 16     # theme.PAD / theme.PAD_S at S=2

# wordmark canvas width (px(2) lead-in + the text item's bbox + the 1-px
# fringe ghost), as _draw_wordmark ends up configuring it. ONE form now.
WORDMARK_W = 290
CLOSE, MIN, GEAR = 69, 75, 76

# --- what the chips cost BEFORE (SIZE_LABEL pill, PAD_X 12/10, GAP 6) ---
PILL_WAS = {"READY": 188, "ERROR": 190, "WAITING": 214, "WORKING": 232,
            "SPEAKING": 241, "THINKING…": 255, "LISTENING…": 271}
BADGE_WAS = {"OFFLINE": 163, "SENSING": 168, "CAMERA OFF": 214}
# --- the first cut (ff3fcf9): PAD_X 6 / GAP 5 and bare imperatives ---
FIRST_CUT_WORST = 129 + 151                  # LISTEN + CAM OFF

# --- and NOW: both chips at SIZE_CAPTION, PAD_X 4, DOT 8, GAP 4, so
# every chip is its measured text + 2*px(4) + px(8) + px(4) = text + 40 ---
PILL = {"READY": 116, "ERROR": 119, "WAITING": 135, "THINKING": 148,
        "WORKING": 148, "SPEAKING": 153, "LISTENING": 158}
BADGE = {"OFFLINE": 135, "SENSING": 140, "CAM OFF": 141}
# slack at his window with the worst pair: what separates the badge from
# the wordmark canvas there (the canvas ends ~2 px past its last glyph)
SLACK = 13


def _right(pill: str, badge: str) -> list:
    """The header's right cluster in PACKING order as header_spans takes
    it -- (name, requested width, pad left, pad right), matching the
    padx each child is packed with in main_window._build_header."""
    return [("close", CLOSE, 0, PAD_S), ("min", MIN, 0, 0),
            ("gear", GEAR, 0, PAD_S), ("pill", PILL[pill], 0, PAD_S),
            ("sensing", BADGE[badge], 0, PAD_S)]


def _spans(total_w: int, pill: str, badge: str, wordmark: int = WORDMARK_W):
    return header_spans(total_w, [("wordmark", wordmark, PAD, 0)],
                        _right(pill, badge))


def _named(spans, name):
    return [s for s in spans if s[0] == name][0]


def _chips_budget(total_w: int = HEADER_W) -> int:
    """Pixels left for the two chips THEMSELVES once the wordmark, the
    window chrome and every pack pad have been paid for."""
    return (total_w - (PAD + WORDMARK_W)
            - (CLOSE + PAD_S + MIN + GEAR + PAD_S) - 2 * PAD_S)


# ------------------------------------------------ the budget, both ends
def test_the_wordmark_is_whole_and_there_is_no_ladder_left():
    """His decision. The 09-02 remedy is gone: no forms list, no fitter,
    no reserve-driven redraw -- one wordmark, drawn once, in every state
    at every width the window allows."""
    assert WORDMARK == "J A R V I S"
    assert not hasattr(main_window, "WORDMARK_FORMS")
    for gone in ("_fit_wordmark", "_wordmark_options"):
        assert not hasattr(main_window.MainWindow, gone)


def test_what_is_left_for_the_two_chips_is_312_px():
    assert _chips_budget() == 312
    assert PAD + WORDMARK_W == 322          # the wordmark's fixed claim
    assert CLOSE + PAD_S + MIN + GEAR + PAD_S == 252     # fixed chrome


def test_the_old_chips_asked_485_of_those_312():
    """The reproduction. At rest the old pair was 44 px over and Tk cut
    the badge -- packed last -- to 124 px of its 168, capsule and the
    tail of the word sheared off, flush against the wordmark; in the
    state he is in most it was 41 px of 214, a dot and a sliver."""
    assert max(PILL_WAS.values()) + max(BADGE_WAS.values()) == 485
    assert 485 - _chips_budget() == 173
    was = header_spans(HEADER_W, [("wordmark", WORDMARK_W, PAD, 0)],
                       [("close", CLOSE, 0, PAD_S), ("min", MIN, 0, 0),
                        ("gear", GEAR, 0, PAD_S),
                        ("pill", PILL_WAS["READY"], 0, PAD_S),
                        ("sensing", BADGE_WAS["SENSING"], 0, PAD_S)])
    assert spans_clipped(was) == [("sensing", 124, 168)]
    assert _named(was, "sensing")[1] == _named(was, "wordmark")[2]  # flush
    worst = header_spans(HEADER_W, [("wordmark", WORDMARK_W, PAD, 0)],
                         [("close", CLOSE, 0, PAD_S), ("min", MIN, 0, 0),
                          ("gear", GEAR, 0, PAD_S),
                          ("pill", PILL_WAS["LISTENING…"], 0, PAD_S),
                          ("sensing", BADGE_WAS["CAMERA OFF"], 0, PAD_S)])
    assert spans_clipped(worst) == [("sensing", 41, 214)]


def test_the_new_chips_ask_299_and_leave_13_px():
    """The worst case is what fits: LISTENING + CAM OFF, not READY +
    SENSING. Less than the first cut left (32 px) and on purpose: the
    words were the wrong thing to spend, the padding was the right one."""
    assert max(PILL.values()) + max(BADGE.values()) == 299
    assert _chips_budget() - 299 == SLACK
    assert FIRST_CUT_WORST < 299 < 485
    assert max(PILL.values()) / max(PILL_WAS.values()) < 0.6
    assert max(BADGE.values()) / max(BADGE_WAS.values()) < 0.7


# ------------------------------- the cross-product that actually matters
@pytest.mark.parametrize("badge", sorted(BADGE))
@pytest.mark.parametrize("pill", sorted(PILL))
def test_no_child_is_clipped_or_unmapped_in_any_pair(pill, badge):
    """Every pill state x every badge state, at his window and at the
    520-unit default, with the FULL wordmark. Nothing clipped, nothing
    unmapped, and the badge never flush against the wordmark: SLACK
    clear of it in the worst pair at his window, a PAD_S or more at any
    wider one."""
    for width in (HEADER_W, DEFAULT_W, 1400):
        spans = _spans(width, pill, badge)
        assert spans_clipped(spans) == []
        assert all(mapped for _n, _a, _b, _r, mapped in spans)
        gap = _named(spans, "sensing")[1] - _named(spans, "wordmark")[2]
        assert gap >= (SLACK if width == HEADER_W else PAD_S)


def test_the_badge_comes_back_whole_and_beside_the_state():
    """His worst case, laid out: 141 px of 141, one PAD_S from the pill
    rather than 116 px adrift of it, and 13 px clear of the wordmark."""
    spans = _spans(HEADER_W, "LISTENING", "CAM OFF")
    assert spans_clipped(spans) == []
    _n, bx0, bx1, req, mapped = _named(spans, "sensing")
    assert (bx0, bx1, req, mapped) == (335, 476, 141, True)
    assert _named(spans, "pill")[1] - bx1 == PAD_S
    assert bx0 - _named(spans, "wordmark")[2] == SLACK


def test_the_vocabulary_is_at_capacity_so_a_longer_word_must_be_measured():
    """The 13 px of slack is less than ONE glyph at this face ('ING' is
    39 px, measured: 13 px a glyph). A state word one glyph wider than
    LISTENING clips the badge at his window. Whoever adds a longer
    state word re-measures on Xvfb first; this is the test that will
    say so."""
    assert (PILL["LISTENING"] - 119) == 39          # LISTENING - LISTEN
    longer = max(PILL.values()) + 14
    spans = header_spans(HEADER_W, [("wordmark", WORDMARK_W, PAD, 0)],
                         [("close", CLOSE, 0, PAD_S), ("min", MIN, 0, 0),
                          ("gear", GEAR, 0, PAD_S), ("pill", longer, 0, PAD_S),
                          ("sensing", max(BADGE.values()), 0, PAD_S)])
    assert spans_clipped(spans) == [("sensing", 140, 141)]
    assert all(len(word) <= len("LISTENING") for word in STATE_WORDS.values())


def test_the_460_unit_minimum_is_still_the_floor_that_holds():
    """His 920-px window IS the app's minimum (MIN_W 460 at S=2), so the
    worst pair has to fit there -- and 905 px is where it stops."""
    assert main_window.MIN_W * 2 - 2 == HEADER_W
    need = (PAD + WORDMARK_W + max(BADGE.values()) + PAD_S
            + max(PILL.values()) + PAD_S + GEAR + PAD_S + MIN + CLOSE + PAD_S)
    assert need == 905 and HEADER_W - need == SLACK
    assert spans_clipped(_spans(905, "LISTENING", "CAM OFF")) == []
    assert spans_clipped(_spans(904, "LISTENING", "CAM OFF")) != []


# ------------------------------------------- the badge is still the loud one
def test_the_badge_kept_its_word_and_the_pill_kept_its_vocabulary():
    """Constraint 1. The badge is a privacy readout: it keeps its type
    size and a seven-glyph word in EVERY tone, so it stays unmistakable
    across the room and cannot be confused with an absent badge -- and,
    every word being seven glyphs, the chip barely twitches between
    states (135/140/141). The pill keeps its participles: the state is
    said as a state (LISTENING), never as an order (LISTEN); only the
    ellipsis went."""
    assert set(BADGE) == set(BADGE_WORDS.values())
    assert all(len(word) == 7 for word in BADGE_WORDS.values())
    assert max(BADGE.values()) - min(BADGE.values()) <= 6
    assert STATE_WORDS == {"idle": "READY", "listening": "LISTENING",
                           "thinking": "THINKING", "speaking": "SPEAKING",
                           "waiting": "WAITING", "working": "WORKING",
                           "error": "ERROR"}
    assert not any("…" in word or "." in word for word in STATE_WORDS.values())


def test_the_two_header_chips_are_one_type_size_and_one_geometry():
    """Asserted through the widgets themselves so the formula cannot
    drift; both calls are pure and need no root. PAD_X 4 / GAP 4 is the
    measured floor: PAD_X 5 clips 2 px at S=1.5."""
    assert StatePill.font() == SensingBadge.font()
    assert StatePill.chip_w(0) == SensingBadge.chip_w(0)
    assert ((StatePill.HEIGHT, StatePill.PAD_X, StatePill.DOT, StatePill.GAP)
            == (SensingBadge.HEIGHT, SensingBadge.PAD_X, SensingBadge.DOT,
                SensingBadge.GAP) == (26, 4, 8, 4))
    assert StatePill.chip_w(100) - StatePill.chip_w(0) == 100


def test_two_state_words_measure_the_same_width():
    """THINKING and WORKING are 148 px each at S=2 (and equal again at
    S=1, 1.25 and 2.5). StatePill._fit returns early when the width is
    unchanged, so it never redraws between them -- which is fine ONLY
    because set_state re-applies the word with itemconfigure after _fit.
    Pinned so the early return cannot be made load-bearing by accident."""
    assert PILL["THINKING"] == PILL["WORKING"]
    import inspect
    src = inspect.getsource(StatePill.set_state)
    assert src.index("self._fit(word)") < src.index("itemconfigure(self._word")


def test_the_abbreviated_chip_still_says_which_reason_it_is_blocked():
    """Constraint 2. 'CAM OFF' is seven glyphs; the unabbreviated reason
    -- and the hour it ends -- stays on badge_caption, which is what
    MainWindow feeds the badge's tooltip on every refresh."""
    assert BADGE_WORDS["curfew"] == "CAM OFF"
    caption = badge_caption({"camera": False, "radar": True, "offline": False,
                             "reason": "curfew", "curfew": ((21, 0), (7, 0))})
    assert caption.startswith("camera off until") and "curfew" in caption
    assert "7" in caption
    off = badge_caption({"camera": False, "radar": False, "offline": True,
                         "reason": "offline"})
    assert "camera" in off and "radar" in off


# ------------------------------------------------- the pure helpers
def test_header_spans_transcribes_the_packer_it_is_named_after():
    """tkPack.c: clamp the cavity, shrink the frame, clip the child, unmap
    it when nothing is left. Verified against real Tk 8.6 on Xvfb :95."""
    spans = header_spans(100, [("a", 30, 0, 0)],
                         [("b", 20, 0, 0), ("c", 20, 0, 0)])
    assert spans == [("a", 0, 30, 30, True), ("b", 80, 100, 20, True),
                     ("c", 60, 80, 20, True)]
    assert spans_clipped(spans) == []
    # cavity exhausted: the frame shrinks to what is left and the child is
    # CLIPPED to it -- pack does not wrap and it does not overlap
    tight = header_spans(40, [("a", 30, 0, 0)], [("b", 20, 0, 0)])
    assert tight == [("a", 0, 30, 30, True), ("b", 30, 40, 20, True)]
    assert spans_clipped(tight) == [("b", 10, 20)]
    # padding comes out of the frame, so a child can be unmapped with the
    # cavity still nominally open
    padded = header_spans(40, [("a", 30, 0, 0)], [("b", 20, 0, 10)])
    assert padded == [("a", 0, 30, 30, True), ("b", 30, 30, 20, False)]
    assert spans_clipped(padded) == [("b", 0, 20)]
    assert header_spans(0, [], []) == [] and spans_clipped([]) == []
    assert spans_clipped(None) == []


def test_both_chips_can_report_their_widest_width_without_a_display():
    """The header's own budget reads the cluster through the widgets, so
    the two formulas cannot drift apart -- and neither call needs a root."""
    assert StatePill.widest_w() > 0
    assert SensingBadge.widest_w() > 0
    assert set(PILL) == set(STATE_WORDS.values())
    # the pill's budget follows the WINDOW's vocabulary, which is wider
    # than its own class constant (WAITING / WORKING are Claude-task states)
    assert set(StatePill.WORDS) < set(STATE_WORDS.values())
    assert StatePill.widest_w(STATE_WORDS.values()) >= StatePill.widest_w()


# ------------------------------------- what is left of the 09-02 remedy
class _FakeChild:
    """The two things _cluster_w reads off a header child."""

    def __init__(self, name, req, padx=0):
        self._name, self._req, self._padx = name, req, padx

    def __str__(self):
        return self._name

    def winfo_reqwidth(self):
        return self._req

    def pack_info(self):
        return {"padx": self._padx}


class _FakeHeader:
    def __init__(self, children, width):
        self._children, self._width = children, width

    def pack_slaves(self):
        return self._children

    def winfo_width(self):
        return self._width


def _stub_window(header_w):
    win = main_window.MainWindow.__new__(main_window.MainWindow)
    win._wordmark = _FakeChild("wordmark", WORDMARK_W, (PAD, 0))
    win.pill = _FakeChild("pill", 1, (0, PAD_S))
    win.sensing_badge = _FakeChild("sensing", 1, (0, PAD_S))
    kids = [win._wordmark, _FakeChild("close", CLOSE, (0, PAD_S)),
            _FakeChild("min", MIN, 0), _FakeChild("gear", GEAR, (0, PAD_S)),
            win.pill, win.sensing_badge]
    win._header = _FakeHeader(kids, header_w)
    win._header_short = None
    return win


def test_the_cluster_budget_walks_the_pack_list_and_takes_the_widest_word():
    """_cluster_w is measured through the widgets, not off a hand-written
    list: the header child most likely to be added next is a camera
    preview on a sibling branch, and a budget that missed it is how the
    badge gets cut again. Both chips count at their WIDEST word, so a
    bar that fits READY/SENSING cannot come apart at 21:00 on CAM OFF.
    And NO reserve is seeded in: the sum is the clipping cliff itself
    (the first cut seeded PAD_S, which would have warned about his own
    918-px window once the slack fell to 13)."""
    win = _stub_window(HEADER_W)
    expect = (CLOSE + PAD_S + MIN + GEAR + PAD_S
              + StatePill.widest_w(STATE_WORDS.values()) + PAD_S
              + SensingBadge.widest_w() + PAD_S)
    assert win._cluster_w() == expect
    # padx is Tk's own: a scalar pads BOTH sides, a pair only the named one
    assert main_window.MainWindow._padx_total(_FakeChild("x", 1, 7)) == 14
    assert main_window.MainWindow._padx_total(_FakeChild("x", 1, (0, 7))) == 7
    assert main_window.MainWindow._padx_total(object()) == 0


def test_a_header_that_ran_short_says_so_once_instead_of_shrinking_jarvis(caplog):
    """All that survives of the remedy he rejected. The wordmark is never
    touched; the log names the shortfall so the next header child does
    not silently put the privacy badge back under Tk's knife -- and it
    says it ONCE per width, not on every <Configure> of a drag. The line
    fires at the cliff (need - 1), not a PAD_S early."""
    win = _stub_window(HEADER_W)
    need = win._cluster_w() + main_window.theme.PAD + WORDMARK_W
    with caplog.at_level("WARNING", logger="jarvis.ui.main_window"):
        win._check_header_fit(need)          # exactly enough: silence
        assert caplog.records == []
        win._check_header_fit(need - 1)
        assert len(caplog.records) == 1
        assert "1 px short" in caplog.records[0].getMessage()
        win._check_header_fit(need - 1)      # same width: still one line
        assert len(caplog.records) == 1
        win._check_header_fit(need - 40)
        assert len(caplog.records) == 2
    assert win._wordmark.winfo_reqwidth() == WORDMARK_W   # never touched


def test_the_fit_check_is_silent_at_his_window_with_the_real_widths(caplog):
    """The point of dropping the PAD_S seed: at 918 px with the measured
    chip widths there are 13 px to spare, and a diagnostic that warned
    about a bar which fits would be the log crying wolf at every start.
    Stubbed at the S=2 literals since _cluster_w reads the chips' widest
    word through theme (scale 1.0 here, no root)."""
    win = _stub_window(HEADER_W)
    real = (CLOSE + PAD_S + MIN + GEAR + PAD_S + max(PILL.values()) + PAD_S
            + max(BADGE.values()) + PAD_S)
    assert HEADER_W - (real + PAD + WORDMARK_W) == SLACK
    win._cluster_w = lambda: real
    # theme.PAD is the scale-1.0 token in a test process, so the cliff is
    # rebuilt from it rather than from the S=2 literal above
    cliff = real + main_window.theme.PAD + WORDMARK_W
    with caplog.at_level("WARNING", logger="jarvis.ui.main_window"):
        win._check_header_fit(cliff + SLACK)            # his window: silent
        win._check_header_fit(cliff)                    # the cliff: silent
        assert caplog.records == []
        win._check_header_fit(cliff - 1)                # one past it
        assert len(caplog.records) == 1
        assert "1 px short" in caplog.records[0].getMessage()
