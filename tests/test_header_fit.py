"""The header's width budget, tested display-free.

2026-09-02, verbatim: "the word sensing is underneath the ready symbol".

DIAGNOSIS. Nothing is misaligned vertically -- SensingBadge and StatePill
declare the SAME 26-design-unit height, both are packed with the default
anchor into a 56-unit header, so pack centres them on the same line and a
baseline mismatch is impossible. The header is over-SUBSCRIBED, and what
Tk does with an over-subscribed bar is not what the first pass at this
file claimed.

WHAT TK ACTUALLY DOES. generic/tkPack.c, ArrangePacking: the cavity is
clamped at zero (752-756) so a child's frame SHRINKS rather than the
cavity going negative; the child is then clipped to its frame less its
padding (786-790); and a child left with no width is UNMAPPED (846-850).
Right-packed siblings therefore cannot overlap -- `frameX` is
`cavityX + cavityWidth` and cavityWidth is never below zero. Confirmed
against real Tk 8.6 on Xvfb :95: at his 918-px header with the tracked
wordmark and READY/SENSING the badge lands at x=322..446 -- 124 px of the
168 it asked for, TRUNCATED, torn 116 px off the pill it belongs beside
and jammed against the wordmark. In his worst state (LISTENING… +
CAMERA OFF) it is 41 px of 214: a dot and a sliver. Push one child
further -- a wider word, the extra header child a sibling branch is
adding -- and Tk stops drawing it at all. views.header_spans transcribes
those three passages and reproduces every mapped case exactly; the
assertion that matters is spans_clipped() == [], because an absent badge
and a badge reading SENSING must not look the same from across the room.

THE FIX is alignment-preserving, not a relocation: the badge stays in the
header beside the state pill, which is deliberate, and the WORDMARK --
the one element in the bar carrying no information -- gives up the
pixels, tracked form first, then untracked, then a monogram.

Every width below was MEASURED on Xvfb :95 at JARVIS_UI_SCALE 2.0 off the
rendered faces ('Chakra Petch SemiBold' -59 for the wordmark, -35 for the
pill, -24 for the badge; 'Inter' for the chrome buttons) and kept as
literals so the file stays Tk-free, the same house pattern as
tests/test_ui_chrome.py.
"""
import pytest

from jarvis.ui.main_window import STATE_WORDS, WORDMARK_FORMS
from jarvis.ui.sensing_badge import WORDS as BADGE_WORDS
from jarvis.ui.sensing_badge import SensingBadge
from jarvis.ui.views import fit_placeholder, header_spans, spans_clipped
from jarvis.ui.widgets import StatePill

HEADER_W = 918          # his 920-px window, less the shell's 1-px inset
DEFAULT_W = 1038        # the 520-design-unit default window
PAD, PAD_S = 32, 16     # theme.PAD / theme.PAD_S at S=2

# wordmark canvas widths (px(2) lead-in + the text item's bbox + the 1-px
# fringe ghost), as _draw_wordmark ends up configuring them
WORDMARK = {"J A R V I S": 290, "JARVIS": 210, "J": 42, "": 1}
# StatePill slabs: text + 2*px(12) + px(8) dot + px(6) gap. The vocabulary
# is main_window.STATE_WORDS, NOT the shorter StatePill.WORDS: WAITING and
# WORKING (a Claude task) only exist in the window's map.
PILL = {"READY": 188, "ERROR": 190, "WAITING": 214, "WORKING": 232,
        "SPEAKING": 241, "THINKING…": 255, "LISTENING…": 271}
# SensingBadge chips: text + 2*px(10) + px(8) dot + px(6) gap
BADGE = {"SENSING": 168, "OFFLINE": 163, "CAMERA OFF": 214}
CLOSE, MIN, GEAR = 69, 75, 76

WORDMARK_WS = [(t, WORDMARK[t]) for t in WORDMARK_FORMS]


def _right(pill: str, badge: str) -> list:
    """The header's right cluster in PACKING order as header_spans takes
    it -- (name, requested width, pad left, pad right), matching the
    padx each child is packed with in main_window._build_header."""
    return [("close", CLOSE, 0, PAD_S), ("min", MIN, 0, 0),
            ("gear", GEAR, 0, PAD_S), ("pill", PILL[pill], 0, PAD_S),
            ("sensing", BADGE[badge], 0, PAD_S)]


def _reserve() -> int:
    """What _wordmark_reserve() budgets: every non-wordmark child at its
    WIDEST, its padding, and a PAD_S gap the wordmark keeps clear."""
    return PAD_S + sum(w + pl + pr for _n, w, pl, pr
                       in _right(max(PILL, key=PILL.get),
                                 max(BADGE, key=BADGE.get)))


def _spans(total_w: int, wordmark: str, pill: str, badge: str) -> list:
    return header_spans(total_w, [("wordmark", WORDMARK[wordmark], PAD, 0)],
                        _right(pill, badge))


def _named(spans, name):
    return [s for s in spans if s[0] == name][0]


# ------------------------------------------------------- the reproduction
def test_the_badge_is_the_child_tk_truncates_at_his_window():
    """What he photographed. The tracked wordmark plus the right cluster
    ask for more than the header has, and the badge -- packed last -- is
    the child Tk cuts down: 124 px of the 168 it asked for, its capsule
    and the tail of its word sheared off, sitting FLUSH against the
    wordmark with no gap at all."""
    need = PAD + WORDMARK["J A R V I S"] + sum(
        w + pl + pr for _n, w, pl, pr in _right("READY", "SENSING"))
    assert need == 962 and need > HEADER_W          # 44 px over, at rest
    spans = _spans(HEADER_W, "J A R V I S", "READY", "SENSING")
    assert spans_clipped(spans) == [("sensing", 124, 168)]
    _n, bx0, bx1, _req, mapped = _named(spans, "sensing")
    assert (bx0, bx1, mapped) == (322, 446, True)   # measured on real Tk
    assert bx1 - bx0 == 124 and 168 - 124 == 44     # 44 px of word gone
    assert bx0 == _named(spans, "wordmark")[2]      # flush, zero gap
    # right-packed pack siblings never overlap; that was never the defect
    drawn = [(x0, x1) for _n, x0, x1, _r, m in spans if m]
    assert all(a[1] <= b[0] or b[1] <= a[0]
               for i, a in enumerate(drawn) for b in drawn[i + 1:])
    # and it is far worse in the state he is in most
    worst = _spans(HEADER_W, "J A R V I S", "LISTENING…", "CAMERA OFF")
    assert spans_clipped(worst) == [("sensing", 41, 214)]


def test_one_child_further_and_tk_stops_drawing_the_badge_entirely():
    """The hazard the budget exists to prevent, and the reason the
    assertion is "nothing is clipped" rather than "nothing collides".

    Tk unmaps a child whose frame has nothing left after its padding, so
    the privacy readout does not degrade -- it disappears, and an absent
    badge is indistinguishable from a badge that never said OFFLINE. The
    app's own 460-unit minimum keeps his window clear of this, which is
    exactly what a new header child would spend."""
    spans = _spans(877, "J A R V I S", "LISTENING…", "CAMERA OFF")
    assert _named(spans, "sensing")[4] is False     # unmapped: gone
    assert spans_clipped(spans) == [("sensing", 0, 214)]
    # one pixel wider and Tk keeps a sliver of it
    assert _named(_spans(878, "J A R V I S", "LISTENING…",
                         "CAMERA OFF"), "sensing")[4] is True


def test_no_window_width_holds_the_tracked_wordmark_in_every_state():
    """Why the wordmark has to yield rather than the header just being
    given a wider minimum: the worst state asks 1091 px on its own, and
    1107 with the gap the wordmark keeps clear -- so a minimum that fitted
    it would be 554 design units, wider than the 520-unit DEFAULT window,
    let alone the 460-unit minimum."""
    bare = PAD + WORDMARK["J A R V I S"] + sum(
        w + pl + pr for _n, w, pl, pr in _right("LISTENING…", "CAMERA OFF"))
    assert bare == 1091 > DEFAULT_W
    assert PAD + WORDMARK["J A R V I S"] + _reserve() == 1107


# ------------------------------------------------------------- the fix
def _fitted(total_w: int) -> str:
    return fit_placeholder(total_w - _reserve() - PAD, WORDMARK_WS)


@pytest.mark.parametrize("pill", sorted(PILL))
@pytest.mark.parametrize("badge", sorted(BADGE))
def test_nothing_in_the_header_is_clipped_once_the_wordmark_yields(pill, badge):
    for width in (HEADER_W, DEFAULT_W, 1107, 1400):
        assert spans_clipped(_spans(width, _fitted(width), pill, badge)) == []


def test_the_badge_comes_back_whole_and_beside_the_state():
    """The fix in his own worst case: 214 px of 214, one PAD_S gap from
    the pill rather than 116 px adrift of it."""
    spans = _spans(HEADER_W, _fitted(HEADER_W), "LISTENING…", "CAMERA OFF")
    assert spans_clipped(spans) == []
    _n, bx0, bx1, req, mapped = _named(spans, "sensing")
    assert (bx0, bx1, req, mapped) == (149, 363, 214, True)
    assert _named(spans, "pill")[1] - bx1 == PAD_S
    # ...and the wordmark is not left flush against it either
    assert bx0 - _named(spans, "wordmark")[2] >= PAD_S


def test_the_wordmark_steps_down_by_window_width_and_comes_back():
    assert _fitted(HEADER_W) == "J"                 # his window today
    assert _fitted(DEFAULT_W) == "JARVIS"           # the default window
    assert _fitted(1107) == "J A R V I S"           # ~554 design units
    assert _fitted(1106) == "JARVIS"
    # monotone: a wider window never shows a SMALLER mark
    rung = [WORDMARK_FORMS.index(_fitted(w)) for w in range(600, 1400, 8)]
    assert rung == sorted(rung, reverse=True)


def test_the_budget_is_the_widest_words_and_not_the_current_ones():
    """Budgeting against READY/SENSING would fit 'JARVIS' at his window --
    and cut the badge in half the moment he speaks."""
    at_rest = sum(w + pl + pr for _n, w, pl, pr in _right("READY", "SENSING"))
    naive = fit_placeholder(HEADER_W - at_rest - PAD, WORDMARK_WS)
    assert naive == "JARVIS"
    assert spans_clipped(_spans(HEADER_W, naive, "LISTENING…",
                                "CAMERA OFF")) == [("sensing", 121, 214)]


def test_the_empty_form_is_the_last_resort_and_the_minimum_window_is_the_floor():
    """Below the app's own minimum even an empty wordmark cannot save the
    badge -- 802 px is where the right cluster alone stops fitting. The
    460-design-unit minimum (918 px here) is what keeps him above it."""
    assert WORDMARK_FORMS[0] == "J A R V I S" and WORDMARK_FORMS[-1] == ""
    assert _fitted(700) == ""
    assert spans_clipped(_spans(802, "", "LISTENING…", "CAMERA OFF")) == []
    assert spans_clipped(_spans(801, "", "LISTENING…", "CAMERA OFF")) != []
    assert HEADER_W > 802


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
    """The header measures the cluster through the widgets themselves, so
    the two formulas cannot drift apart -- and neither call needs a root."""
    assert StatePill.widest_w() > 0
    assert SensingBadge.widest_w() > 0
    # the widest WORD is what each is budgeting for
    assert max(BADGE, key=BADGE.get) in BADGE_WORDS.values()
    assert set(PILL) == set(STATE_WORDS.values())
    # ...and the pill's budget follows the window's vocabulary, which is
    # wider than its own class constant
    assert set(StatePill.WORDS) < set(STATE_WORDS.values())
    assert StatePill.widest_w(STATE_WORDS.values()) >= StatePill.widest_w()
