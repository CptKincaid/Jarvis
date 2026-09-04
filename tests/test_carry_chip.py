"""The carry chip's words and colours (jarvis/ui/carry_chip.py), display-free.

House pattern (tests/test_sensing_badge.py): every decision the chip makes
is a pure function, so legibility in both looks is asserted here without a
window. The widget itself is never constructed -- UI window churn on the
live display froze his desktop on 2026-08-26.
"""
import pytest

from jarvis.ui import theme
from jarvis.ui.carry_chip import (
    MAX_CHARS,
    STATE_DROPPED,
    STATE_HELD,
    STATE_HOLDING,
    STATE_IDLE,
    STATE_LANDED,
    STATE_THROWN,
    backstop_due,
    chip_colors,
    chip_name,
    chip_tone,
    chip_word,
    rule_fraction,
)


@pytest.fixture(autouse=True)
def _restore_look():
    yield
    theme.select_look(theme.DEFAULT_LOOK)


def test_the_name_is_tail_truncated_the_way_a_partial_is():
    assert chip_name("the thesis draft") == "the thesis draft"
    long = "a very long window title that goes on and on and on"
    short = chip_name(long)
    assert len(short) <= MAX_CHARS and short.endswith("…")
    assert chip_name("  two   spaces  ") == "two spaces"


def test_each_state_reads_as_itself():
    assert chip_word(STATE_HOLDING, "the thesis draft").endswith("the thesis draft")
    assert chip_word(STATE_THROWN, "the thesis draft", "left").endswith("←")
    assert chip_word(STATE_THROWN, "the thesis draft", "right").endswith("→")
    assert chip_word(STATE_LANDED, target="the board") == "→ THE BOARD"
    assert "HPCOMPUTER" in chip_word(STATE_HELD, target="HPCOMPUTER")
    assert chip_word(STATE_DROPPED) == "dropped"
    assert chip_word(STATE_IDLE) == ""


def test_a_timeout_looks_identical_to_a_deliberate_drop():
    """Same outcome, same word: pretending otherwise would be a lie."""
    assert chip_word(STATE_DROPPED, "x", "down") == chip_word(STATE_DROPPED)


def test_the_tones():
    assert chip_tone(STATE_HOLDING) == "live"
    assert chip_tone(STATE_THROWN) == chip_tone(STATE_LANDED) == "ok"
    assert chip_tone(STATE_HELD) == "warn"
    assert chip_tone(STATE_DROPPED) == chip_tone(STATE_IDLE) == "muted"


@pytest.mark.parametrize("look", ["holo", "classic"])
def test_every_tone_is_legible_in_both_looks(look):
    theme.select_look(look)
    for tone in ("live", "ok", "warn", "muted"):
        c = chip_colors(tone)
        assert c["ink"] != theme.BG and c["edge"] != theme.BG, (look, tone)
        # the ink is a text token, never structure cyan (the badge's rule)
        assert c["ink"] in (theme.FOCAL, theme.INK,
                            getattr(theme, "MUTED", theme.INK)), (look, tone)
        assert c["rule"]


def test_held_is_told_apart_from_holding_without_reading_the_word():
    for look in ("holo", "classic"):
        theme.select_look(look)
        assert chip_colors("warn")["edge"] != chip_colors("live")["edge"]


def test_the_rule_depletes_over_the_ttl_and_clamps():
    assert rule_fraction(0.0, 0.0, 8.0) == 1.0
    assert rule_fraction(4.0, 0.0, 8.0) == 0.5
    assert rule_fraction(9.0, 0.0, 8.0) == 0.0
    assert rule_fraction(-1.0, 0.0, 8.0) == 1.0
    assert rule_fraction(1.0, 0.0, 0.0) == 0.0


def test_the_backstop_is_the_chips_own_timer_and_zero_means_none():
    """When no frame arrives to end a carry the chip puts it down itself
    at the wall-clock cap (MEASURED before: 60 s of silence, still
    holding). No backstop asked for is never "due at once"."""
    assert backstop_due(1000.0, 1000.0, 8.0) is False
    assert backstop_due(1007.9, 1000.0, 8.0) is False
    assert backstop_due(1008.1, 1000.0, 8.0) is True
    assert backstop_due(1060.0, 1000.0, 0.0) is False
    assert backstop_due(1060.0, 1000.0, -1.0) is False
