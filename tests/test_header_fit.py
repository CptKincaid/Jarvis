"""The header's width budget, tested display-free.

2026-09-02, verbatim: "the word sensing is underneath the ready symbol".

DIAGNOSIS. Nothing is misaligned vertically -- SensingBadge and StatePill
declare the SAME 26-design-unit height, both are packed with the default
anchor into a 56-unit header, so pack centres them on the same line and a
baseline mismatch is impossible. The header is over-SUBSCRIBED. At his
window (920x1440, saved; JARVIS_UI_SCALE 2.0 -> a 918-px header) the
children request 951 px with the pill on READY and 1082 px in the worst
state, and Tk's packer neither wraps nor stops: it hands every right-packed
child its full parcel out of a shrinking cavity, so the LAST one packed --
the sensing badge -- is handed a parcel that starts left of where the
cavity began, on top of its neighbour. header_spans() reproduces exactly
that, and spans_collide() names the pair, without creating a window (the
standing rule since the 2026-08-26 desktop freeze came out of UI window
churn).

THE FIX is alignment-preserving, not a relocation: the badge stays in the
header beside the state pill, which is deliberate (an absent badge and a
badge reading SENSING must not look the same from across the room), and
the WORDMARK -- the one element in the bar carrying no information -- gives
up the pixels, tracked form first, then untracked, then a monogram.

Every width below was measured off the rendered faces at JARVIS_UI_SCALE
2.0 ('Chakra Petch SemiBold' -59 for the wordmark, -35 for the pill, -24
for the badge; 'Inter'/'DejaVu Sans' -35/-40 for the chrome buttons) and
kept as literals so the file stays Tk-free, the same house pattern as
tests/test_ui_chrome.py.
"""
import pytest

from jarvis.ui.main_window import STATE_WORDS, WORDMARK_FORMS
from jarvis.ui.sensing_badge import WORDS as BADGE_WORDS
from jarvis.ui.sensing_badge import SensingBadge
from jarvis.ui.views import fit_placeholder, header_spans, spans_collide
from jarvis.ui.widgets import StatePill

HEADER_W = 918          # his 920-px window, less the shell's 1-px inset
DEFAULT_W = 1038        # the 520-design-unit default window
PAD, PAD_S = 32, 16     # theme.PAD / theme.PAD_S at S=2

# wordmark canvas widths (px(2) lead-in + text + the 1-px fringe ghost)
WORDMARK = {"J A R V I S": 286, "JARVIS": 207, "J": 41, "": 1}
# StatePill slabs: text + 2*px(12) + px(8) dot + px(6) gap. The vocabulary
# is main_window.STATE_WORDS, NOT the shorter StatePill.WORDS: WAITING and
# WORKING (a Claude task) only exist in the window's map.
PILL = {"READY": 187, "ERROR": 191, "WAITING": 214, "WORKING": 233,
        "SPEAKING": 243, "THINKING…": 257, "LISTENING…": 272}
# SensingBadge chips: text + 2*px(10) + px(8) dot + px(6) gap
BADGE = {"SENSING": 168, "OFFLINE": 162, "CAMERA OFF": 214}
CLOSE, MIN, GEAR = 67, 73, 74

WORDMARK_WS = [(t, WORDMARK[t]) for t in WORDMARK_FORMS]


def _right(pill: str, badge: str) -> list:
    """The header's right cluster in PACKING order, each width including
    the padx pack gives it (main_window._build_header)."""
    return [("close", CLOSE + PAD_S), ("min", MIN), ("gear", GEAR + PAD_S),
            ("pill", PILL[pill] + PAD_S), ("sensing", BADGE[badge] + PAD_S)]


def _reserve() -> int:
    """What _wordmark_reserve() budgets: the cluster at its WIDEST."""
    return sum(w for _n, w in _right(max(PILL, key=PILL.get),
                                     max(BADGE, key=BADGE.get)))


def _spans(total_w: int, wordmark: str, pill: str, badge: str) -> list:
    return header_spans(total_w, [("wordmark", PAD + WORDMARK[wordmark])],
                        _right(pill, badge))


# ------------------------------------------------------- the reproduction
def test_the_header_overflows_at_his_window_and_the_badge_is_what_lands_on_top():
    """What he photographed. The tracked wordmark plus the right cluster
    ask for more than the header has, and the badge -- packed last -- is
    the child pack pushes back over its neighbour."""
    need = PAD + WORDMARK["J A R V I S"] + sum(w for _n, w in
                                               _right("READY", "SENSING"))
    assert need == 951 and need > HEADER_W          # 33 px over, at rest
    hit = spans_collide(_spans(HEADER_W, "J A R V I S", "READY", "SENSING"))
    assert hit == [("wordmark", "sensing")]
    # and it is not a near miss in the state he is in most: the moment the
    # pill says LISTENING… the badge is pushed 118 px back
    _n, x0, _x1 = [s for s in _spans(HEADER_W, "J A R V I S", "LISTENING…",
                                     "SENSING") if s[0] == "sensing"][0]
    assert PAD + WORDMARK["J A R V I S"] - x0 == 118


def test_no_window_width_holds_the_tracked_wordmark_in_every_state():
    """Why the wordmark has to yield rather than the header just being
    given a wider minimum: the worst state needs 1082 px, so a minimum
    that fitted it would be 541 design units -- wider than the 520-unit
    DEFAULT window, let alone the 460-unit minimum."""
    worst = PAD + WORDMARK["J A R V I S"] + _reserve()
    assert worst == 1082 > DEFAULT_W


# ------------------------------------------------------------- the fix
def _fitted(total_w: int) -> str:
    return fit_placeholder(total_w - _reserve() - PAD, WORDMARK_WS)


@pytest.mark.parametrize("pill", sorted(PILL))
@pytest.mark.parametrize("badge", sorted(BADGE))
def test_nothing_in_the_header_overlaps_in_any_state_once_the_wordmark_yields(
        pill, badge):
    assert spans_collide(_spans(HEADER_W, _fitted(HEADER_W), pill, badge)) == []
    assert spans_collide(_spans(DEFAULT_W, _fitted(DEFAULT_W), pill,
                                badge)) == []


def test_the_wordmark_steps_down_by_window_width_and_comes_back():
    assert _fitted(HEADER_W) == "J"                 # his window today
    assert _fitted(DEFAULT_W) == "JARVIS"           # the default window
    assert _fitted(1120) == "J A R V I S"           # ~560 design units
    # monotone: a wider window never shows a SMALLER mark
    rung = [WORDMARK_FORMS.index(_fitted(w)) for w in range(600, 1400, 8)]
    assert rung == sorted(rung, reverse=True)


def test_the_budget_is_the_widest_words_and_not_the_current_ones():
    """Budgeting against READY/SENSING would fit 'JARVIS' at his window --
    and put the overlap straight back the moment he speaks."""
    at_rest = sum(w for _n, w in _right("READY", "SENSING"))
    naive = fit_placeholder(HEADER_W - at_rest - PAD, WORDMARK_WS)
    assert naive == "JARVIS"
    assert spans_collide(_spans(HEADER_W, naive, "LISTENING…",
                                "CAMERA OFF")) == [("wordmark", "sensing")]


def test_the_empty_form_is_the_last_resort_and_still_never_collides():
    assert WORDMARK_FORMS[0] == "J A R V I S" and WORDMARK_FORMS[-1] == ""
    assert _fitted(700) == ""
    assert spans_collide(_spans(700, "", "READY", "SENSING")) == []


# ------------------------------------------------- the pure helpers
def test_header_spans_models_pack_including_the_negative_remainder():
    spans = header_spans(100, [("a", 30)], [("b", 20), ("c", 20)])
    assert spans == [("a", 0, 30), ("b", 80, 100), ("c", 60, 80)]
    assert spans_collide(spans) == []
    # cavity exhausted: pack does not wrap, the parcel goes left of 0
    assert header_spans(40, [("a", 30)], [("b", 20)]) == [("a", 0, 30),
                                                          ("b", 20, 40)]
    assert spans_collide(header_spans(40, [("a", 30)], [("b", 20)])) == [
        ("a", "b")]
    assert header_spans(0, [], []) == [] and spans_collide([]) == []


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
