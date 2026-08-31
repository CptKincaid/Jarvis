"""Continuity callbacks: "that would be the third coffee timer, sir."

The counters come off the activity journal, which is the ONLY store where a
tool's argument survives (habits.json rows carry the user's utterance and
no tool name at all). They are rendered into the DYNAMIC user turn --
never the static prefix, which must stay byte-identical for Ollama's cache.

The live journal is two rows deep, so every case here is built from a
synthetic day rather than read from disk.
"""
import json
from datetime import datetime, timedelta

import pytest

from jarvis.context import ContextEngine
from jarvis.tools.journal import (CONTINUITY_HEAD, CONTINUITY_RULE,
                                  journal_repeats)


def rows(*specs, start=None):
    """(kind, payload) pairs -> journal rows with _when filled in."""
    when = start or datetime.now().replace(hour=8, minute=0, second=0,
                                           microsecond=0)
    out = []
    for i, (kind, payload) in enumerate(specs):
        row = dict(payload)
        row["kind"] = kind
        row["_when"] = when + timedelta(minutes=3 * i)
        row["time"] = row["_when"].isoformat(timespec="seconds")
        out.append(row)
    return out


def tool(name, **args):
    return ("tool", {"name": name, "args": args, "ok": True})


def ex(user, jarvis="Very good, sir."):
    return ("exchange", {"user": user, "jarvis": jarvis})


# ------------------------------------------------------------ counters
def test_the_third_coffee_timer_is_counted_by_its_label():
    block = journal_repeats(rows(
        tool("set_timer", minutes=4, label="coffee"),
        tool("set_timer", minutes=4, label="coffee"),
        tool("set_timer", minutes=4, label="coffee"),
    ))
    assert block.startswith(CONTINUITY_HEAD)
    assert '3rd "coffee" timer' in block
    # the block carries its own number rule: VOICE_RULES bans reciting file
    # names and unperformed checks but says NOTHING about counts, and the
    # measured number suppression lives in a few-shot pool sampled once
    assert block.endswith(CONTINUITY_RULE)
    assert len(block.splitlines()) == 2


def test_a_different_label_is_a_different_timer():
    assert journal_repeats(rows(
        tool("set_timer", minutes=4, label="coffee"),
        tool("set_timer", minutes=4, label="pasta"),
    )) == ""


def test_the_duration_is_not_the_thing_that_repeats():
    """"the third five-minute timer" is a coincidence; the label is the
    fact about his morning. Two unlabelled timers still count as timers."""
    block = journal_repeats(rows(
        tool("set_timer", minutes=5),
        tool("set_timer", minutes=5),
    ))
    assert "2nd timer" in block and "5" not in block.split("\n")[0]


def test_a_defaulted_selector_is_never_quoted():
    """when="today" / range="today" is the tool's boilerplate, not his
    subject: quoting it produced 'the 2nd "today" weather check'."""
    block = journal_repeats(rows(
        tool("get_weather", when="today", location=""),
        tool("get_weather", when="today", location=""),
    ))
    assert "2nd weather check" in block and '"today"' not in block


def test_a_failed_call_is_not_something_he_did_twice():
    bad = rows(tool("system_health"), tool("system_health"))
    for row in bad:
        row["ok"] = False
    assert journal_repeats(bad) == ""


def test_a_single_occurrence_is_not_a_callback():
    assert journal_repeats(rows(tool("set_timer", label="coffee"))) == ""
    assert journal_repeats([]) == ""
    assert journal_repeats([None, "nonsense", 3]) == ""


def test_a_repeat_the_model_can_already_see_is_not_reported():
    """format_for_prompt renders the last four exchanges, so a question
    asked twice in a row needs no help. The callback is the one from
    EARLIER: it only counts once its first asking has fallen out."""
    inside = rows(ex("what's on my calendar"), ex("what's on my calendar"),
                  ex("thanks"), ex("right"))
    assert journal_repeats(inside) == ""
    outside = rows(ex("what's on my calendar"), ex("set a timer"),
                   ex("thanks"), ex("what's the weather"), ex("right"),
                   ex("what's on my calendar"))
    assert 'he asked "what\'s on my calendar" twice' in journal_repeats(outside)


def test_the_wake_word_and_punctuation_do_not_split_a_repeat():
    said = rows(ex("Jarvis, what's on my calendar?"), ex("one"), ex("two"),
                ex("three"), ex("four"), ex("what's on my calendar"))
    assert "twice" in journal_repeats(said)


def test_the_block_is_bounded_to_two_clauses_and_the_loudest_win():
    many = rows(
        *[tool("set_timer", minutes=4, label="coffee")] * 4,
        *[tool("get_weather", location="Austin")] * 3,
        *[tool("get_mail")] * 2,
    )
    block = journal_repeats(many)
    head = block.splitlines()[0]
    assert head.count(";") == 1                      # two clauses, one line
    assert '4th "coffee" timer' in head and '3rd "Austin" weather check' in head
    assert "mail" not in head
    assert journal_repeats(many, max_clauses=1).splitlines()[0].count(";") == 0


def test_ordinals_do_not_go_wrong_in_the_teens():
    from jarvis.tools.journal import _ordinal
    assert [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 111)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
         "23rd", "111th"]


# ------------------------------------------------------- the engine seam
@pytest.fixture
def engine(tmp_path):
    return ContextEngine(journal_dir=tmp_path / "journal")


def test_the_engine_reads_todays_rows_and_caches_the_render(engine, monkeypatch):
    for _ in range(3):
        engine.journal_tool("set_timer", {"minutes": 4, "label": "coffee"})
    assert '3rd "coffee" timer' in engine.continuity_block()

    reads = []
    real = engine.journal_rows
    monkeypatch.setattr(engine, "journal_rows",
                        lambda *a, **k: (reads.append(1), real(*a, **k))[1])
    engine.continuity_block()
    engine.continuity_block()
    assert reads == []                     # the day file is not re-read per turn

    # ... but a journal write drops the cache, so the timer he just set is
    # counted on the very next turn rather than up to 30 seconds later
    engine.journal_tool("set_timer", {"minutes": 4, "label": "coffee"})
    assert '4th "coffee" timer' in engine.continuity_block()
    assert reads == [1]


def test_yesterdays_repeats_do_not_leak_into_today(engine):
    day = engine.journal_dir()
    day.mkdir(parents=True, exist_ok=True)
    yesterday = datetime.now() - timedelta(days=1)
    lines = []
    for _ in range(3):
        lines.append(json.dumps({"time": yesterday.isoformat(timespec="seconds"),
                                 "kind": "tool", "name": "set_timer",
                                 "args": {"label": "coffee"}, "ok": True}))
    (day / f"{yesterday:%Y-%m-%d}.jsonl").write_text("\n".join(lines) + "\n")
    assert engine.continuity_block() == ""


def test_an_unreadable_journal_never_reaches_the_turn(engine, monkeypatch):
    def boom(*a, **k):
        raise OSError("journal on fire")
    monkeypatch.setattr(engine, "journal_rows", boom)
    assert engine.continuity_block() == ""


# --------------------------------------------------- the prompt placement
def test_the_block_rides_the_user_turn_and_never_the_static_prefix():
    """The whole point: continuity is per-turn background. In the system
    prompt it would evict Ollama's prefix cache on every single question."""
    import jarvis.brain as brain_mod

    class FakeContext:
        def get_context(self, level):
            return {"time": "ten past nine"}

        def format_for_prompt(self, ctx, spoken=False):
            return f"Current time: {ctx['time']}"

        def continuity_block(self):
            return 'Earlier today: 3rd "coffee" timer.\nThose counts are background.'

    b = brain_mod.JarvisBrain.__new__(brain_mod.JarvisBrain)
    b._context = FakeContext()
    b._memory = None
    ctx_text, mem_text = b._dynamic_context("another timer, please")
    assert '3rd "coffee" timer' in ctx_text
    turn = brain_mod.build_user_turn(ctx_text, mem_text, "another timer, please")
    assert '3rd "coffee" timer' in turn
    assert "Earlier today" not in brain_mod.static_system()


def test_a_context_engine_without_the_block_still_answers():
    """Old engines (and every test double) have no continuity_block."""
    import jarvis.brain as brain_mod

    class Old:
        def get_context(self, level):
            return {"time": "ten"}

        def format_for_prompt(self, ctx, spoken=False):
            return "Current time: ten"

    b = brain_mod.JarvisBrain.__new__(brain_mod.JarvisBrain)
    b._context, b._memory = Old(), None
    assert b._dynamic_context("hello") == ("Current time: ten", "")


def test_a_block_that_raises_is_logged_and_dropped():
    import jarvis.brain as brain_mod

    class Angry:
        def get_context(self, level):
            return {"time": "ten"}

        def format_for_prompt(self, ctx, spoken=False):
            return "Current time: ten"

        def continuity_block(self):
            raise RuntimeError("no")

    b = brain_mod.JarvisBrain.__new__(brain_mod.JarvisBrain)
    b._context, b._memory = Angry(), None
    assert b._dynamic_context("hello") == ("Current time: ten", "")
