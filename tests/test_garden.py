"""The weekly memory garden (jarvis/garden.py): the Sunday-night pass that
promotes what Jarvis OBSERVED in the journal into what he KNOWS in
facts.json, its provenance tag, its Monday line and its undo.

Real JarvisMemory (semantic=False, so no Ollama), a real ContextEngine
writing a real journal tree under tmp_path, and the model replaced by a
plain function. Nothing here loads a model or touches the network.
"""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis import garden as garden_mod
from jarvis.commander import (Commander, IntentClassifier, _GARDEN_REPORT_RX,
                              _GARDEN_UNDO_RX, _h_garden_report, _h_garden_undo)
from jarvis.config import CONFIG
from jarvis.context import ContextEngine
from jarvis.garden import SOURCE, MemoryGarden, filed_line, week_key, week_window
from jarvis.memory import JarvisMemory

# A Monday at 02:00 — the hour the pass is meant to run.
MONDAY = datetime(2026, 8, 31, 2, 0)
PROPOSALS = [{"key": "morning medication",
              "value": "Hunter takes a thyroid tablet at seven every morning."},
             {"key": "briefing length",
              "value": "Hunter prefers short briefings in the morning."},
             {"key": "thesis work", "value": "Hunter works on his thesis most evenings."}]


@pytest.fixture
def memory(tmp_path):
    return JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "none",
                        semantic=False)


@pytest.fixture
def context(tmp_path):
    return ContextEngine(project_dir=tmp_path, vss_dir=tmp_path / "novss",
                         journal_dir=tmp_path / "journal")


def _fill_journal(context, when=MONDAY, rows=20):
    """A week's worth of rows, inside the closed week's window."""
    since, _until = week_window(when)
    d = context.journal_dir()
    d.mkdir(parents=True, exist_ok=True)
    for i in range(rows):
        stamp = since + timedelta(days=i % 7, hours=9 + (i % 5))
        with open(d / f"{stamp:%Y-%m-%d}.jsonl", "a") as fh:
            fh.write(json.dumps({"time": stamp.isoformat(timespec="seconds"),
                                 "kind": "exchange", "user": f"line {i}",
                                 "jarvis": "Very good, sir."}) + "\n")


def _garden(memory, context, tmp_path, proposals=None, **kw):
    seen = {}

    def extract(text, known=None, limit=4):
        seen["text"], seen["known"], seen["limit"] = text, known, limit
        return list(PROPOSALS if proposals is None else proposals)

    g = MemoryGarden(memory, context=context, extract=extract,
                     state_path=tmp_path / "garden.json",
                     now=lambda: MONDAY, **kw)
    g.seen = seen
    return g


# ------------------------------------------------------------ the week
def test_the_week_key_names_the_week_that_closed():
    # Monday 31 Aug 2026 belongs to W36; the week just closed is W35.
    assert week_key(MONDAY) == "2026-W35"
    # Any later day of the same week answers the same key, so a late pass
    # cannot file the same week twice under two names.
    assert week_key(MONDAY + timedelta(days=3)) == "2026-W35"
    assert week_key(MONDAY + timedelta(days=7)) == "2026-W36"


def test_the_window_is_monday_to_monday():
    since, until = week_window(MONDAY)
    assert (since.weekday(), until.weekday()) == (0, 0)
    assert until - since == timedelta(days=7)
    assert until <= MONDAY and since.hour == 0


# ------------------------------------------------------------ the pass
def test_a_pass_files_tagged_facts_and_records_them(memory, context, tmp_path):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path)
    rec = g.run_pass()
    assert rec["week"] == "2026-W35" and rec["proposed"] == 3
    assert [f["key"] for f in rec["filed"]] == [p["key"] for p in PROPOSALS]
    facts = memory.get_all_facts()
    assert facts["morning medication"]["source"] == SOURCE
    assert dict(memory.facts_from(SOURCE)).keys() == {p["key"] for p in PROPOSALS}
    # the state file is on disk and reloadable
    assert json.loads((tmp_path / "garden.json").read_text())["week"] == "2026-W35"


def test_the_pass_never_writes_into_the_live_fact_store(memory, context, tmp_path):
    """get_all_facts() returns JarvisMemory's own dict. The pass stages
    each filed fact so the next proposal can be diffed against it; doing
    that in the live dict overwrote the entry remember() had just written
    and dropped its "time", so facts.json and the store disagreed and
    recall(since=...) silently lost the date."""
    _fill_journal(context)
    _garden(memory, context, tmp_path).run_pass()
    stored = memory.get_all_facts()
    on_disk = json.loads((tmp_path / "mem" / "facts.json").read_text())
    assert stored == on_disk
    assert all("time" in e for e in stored.values())


def test_a_pass_files_two_facts_that_say_the_same_thing_only_once(memory, context, tmp_path):
    _fill_journal(context)
    twins = [{"key": "meds", "value": "Hunter takes a thyroid tablet at seven."},
             {"key": "medication", "value": "Hunter takes a thyroid tablet at seven."}]
    g = _garden(memory, context, tmp_path, proposals=twins)
    assert [f["key"] for f in g.run_pass()["filed"]] == ["meds"]


def test_the_prompt_gets_the_journal_and_the_keys_it_must_not_repeat(memory, context, tmp_path):
    _fill_journal(context)
    memory.remember("dentist", "my dentist is Dr Patel")
    g = _garden(memory, context, tmp_path)
    g.run_pass()
    assert "line 3" in g.seen["text"]              # the digest, not the raw rows
    assert any("dentist" in k for k in g.seen["known"])


def test_a_hand_told_fact_is_never_overwritten(memory, context, tmp_path):
    _fill_journal(context)
    memory.remember("morning medication", "he told me this himself")
    g = _garden(memory, context, tmp_path)
    rec = g.run_pass()
    assert "morning medication" not in [f["key"] for f in rec["filed"]]
    assert memory.get_all_facts()["morning medication"]["value"] == \
        "he told me this himself"
    assert "source" not in memory.get_all_facts()["morning medication"]


def test_a_restatement_of_a_known_fact_is_not_filed_twice(memory, context, tmp_path):
    _fill_journal(context)
    memory.remember("meds", "Hunter takes a thyroid tablet at seven every morning")
    g = _garden(memory, context, tmp_path)
    rec = g.run_pass()
    assert [f["key"] for f in rec["filed"]] == ["briefing length", "thesis work"]


def test_the_cap_holds_however_much_the_model_offers(memory, context, tmp_path):
    _fill_journal(context)
    many = [{"key": f"fact {i}", "value": f"Hunter does thing {i}."} for i in range(9)]
    g = _garden(memory, context, tmp_path, proposals=many,
                cfg=SimpleNamespace(get=lambda k, d=None: 2 if "max_facts" in k else d))
    assert len(g.run_pass()["filed"]) == 2


def test_a_thin_week_is_filed_as_done_without_calling_the_model(memory, context, tmp_path):
    _fill_journal(context, rows=3)                 # under MIN_ROWS
    g = _garden(memory, context, tmp_path)
    rec = g.run_pass()
    assert rec["skipped"] == "thin week" and rec["filed"] == []
    assert "text" not in g.seen                    # the model was never asked
    assert g.due(MONDAY) is False                  # and the week is not retried


@pytest.mark.parametrize("gate,why", [("lent", "lent to a trainer"), ("busy", "busy")])
def test_a_busy_or_lent_model_skips_the_pass_and_retries_next_tick(
        memory, context, tmp_path, gate, why):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path, **{gate: lambda: True})
    rec = g.run_pass()
    assert rec["skipped"] == why and rec["filed"] == []
    assert memory.get_all_facts() == {}
    assert not (tmp_path / "garden.json").exists()  # NOT recorded: try again
    assert g.due(MONDAY) is True


def test_a_model_that_raises_loses_the_week_not_the_process(memory, context, tmp_path):
    _fill_journal(context)
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    g = MemoryGarden(memory, context=context, extract=boom,
                     state_path=tmp_path / "g.json", now=lambda: MONDAY)
    assert g.run_pass()["filed"] == []


def test_a_memory_that_cannot_take_provenance_still_files(context, tmp_path):
    """An older memory object (or a stub) whose remember() takes two
    arguments must not lose the week -- it files untagged and says so."""
    _fill_journal(context)
    calls = []
    mem = SimpleNamespace(get_all_facts=lambda: {},
                          remember=lambda k, v: calls.append((k, v)),
                          facts_from=lambda s: [])
    g = _garden(mem, context, tmp_path)
    assert len(g.run_pass()["filed"]) == 3 and len(calls) == 3


# ----------------------------------------------------------- scheduling
def test_due_waits_for_the_small_hours_early_in_the_week(memory, context, tmp_path):
    g = _garden(memory, context, tmp_path)
    assert g.due(MONDAY) is True                         # 02:00 Monday
    assert g.due(MONDAY.replace(hour=11)) is False       # Monday lunchtime
    # ... but a week still ungardened by Wednesday runs at any hour.
    assert g.due((MONDAY + timedelta(days=2)).replace(hour=11)) is True


def test_a_filed_week_is_not_gardened_again(memory, context, tmp_path):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path)
    g.run_pass()
    assert g.due(MONDAY) is False
    assert g.tick() is None
    # the next week is due again
    assert g.due(MONDAY + timedelta(days=7)) is True


def test_disabled_in_config_is_never_due(memory, context, tmp_path):
    g = _garden(memory, context, tmp_path,
                cfg=SimpleNamespace(get=lambda k, d=None: False if "enabled" in k else d))
    assert g.enabled is False and g.due(MONDAY) is False


# ------------------------------------------------------------- speaking
@pytest.mark.parametrize("n,expected", [
    (0, ""),
    (1, "I filed one thing from this week, sir; say memory report to hear it."),
    (3, "I filed three things from this week, sir; say memory report to hear them."),
])
def test_the_monday_line(n, expected):
    assert filed_line(n) == expected


def test_the_line_is_owed_once_then_never_again(memory, context, tmp_path):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path)
    g.run_pass()
    assert g.pending_line().startswith("I filed three things")
    g.mark_spoken()
    assert g.pending_line() == ""
    # and it survives a restart: the flag is in the state file
    again = MemoryGarden(memory, context=context, state_path=tmp_path / "garden.json",
                         now=lambda: MONDAY)
    assert again.pending_line() == ""


def test_a_pass_that_filed_nothing_says_nothing(memory, context, tmp_path):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path, proposals=[])
    g.run_pass()
    assert g.pending_line() == ""


def test_the_report_reads_back_what_is_still_filed(memory, context, tmp_path):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path)
    g.run_pass()
    line = g.report_line()
    assert line.startswith("From last week I filed three, sir:")
    assert "thyroid tablet" in line


def test_the_report_before_the_first_pass_is_honest(memory, context, tmp_path):
    g = _garden(memory, context, tmp_path)
    assert g.report_line() == "I've filed nothing from the journal yet, sir."


# ---------------------------------------------------------------- undo
def test_undo_removes_exactly_what_the_pass_filed(memory, context, tmp_path):
    _fill_journal(context)
    memory.remember("dentist", "my dentist is Dr Patel")
    g = _garden(memory, context, tmp_path)
    g.run_pass()
    line = g.undo()
    assert line == "Forgotten, sir; three things out of memory."
    assert list(memory.get_all_facts()) == ["dentist"]      # his own fact stays
    assert g.undo() == "There's nothing from the last pass to forget, sir."


def test_undo_leaves_a_fact_he_has_since_re_told(memory, context, tmp_path):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path)
    g.run_pass()
    # He corrected one by voice: it is his now, not the garden's.
    memory.remember("briefing length", "actually I want the long one")
    g.undo()
    facts = memory.get_all_facts()
    assert facts["briefing length"]["value"] == "actually I want the long one"
    assert "morning medication" not in facts


def test_undo_silences_the_pending_line(memory, context, tmp_path):
    _fill_journal(context)
    g = _garden(memory, context, tmp_path)
    g.run_pass()
    g.undo()
    assert g.pending_line() == ""


# ------------------------------------------------------------- thread
def test_start_and_stop_are_idempotent_and_join(memory, context, tmp_path):
    g = _garden(memory, context, tmp_path)
    g.start()
    g.start()                                   # alive-guard, not a second thread
    assert g.running
    g.stop()
    assert not g.running
    g.start()                                   # a stopped garden restarts
    g.stop()


def test_a_corrupt_state_file_is_a_fresh_start(memory, context, tmp_path):
    (tmp_path / "garden.json").write_text("{not json")
    g = _garden(memory, context, tmp_path)
    assert g.pending_line() == "" and g.due(MONDAY) is True


# ------------------------------------------------------------ commands
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "il.json")
    monkeypatch.setattr(CONFIG, "talkback", False)
    svc = SimpleNamespace(tts=MagicMock(), memory=MagicMock(),
                          garden_report=lambda: "From last week I filed two, sir: a; b.",
                          garden_undo=lambda: "Forgotten, sir; two things out of memory.")
    return Commander(svc)


@pytest.mark.parametrize("said", [
    "memory report", "what did you file this week", "what have you filed",
    "what did you learn about me this week"])
def test_report_phrasings(cmdr, said):
    assert _GARDEN_REPORT_RX.match(said)
    res = _h_garden_report(cmdr, said, None)
    assert res.handled and res.speak and "I filed two" in res.reply


@pytest.mark.parametrize("said", [
    "forget the last garden pass", "forget the last memory pass",
    "undo the last garden pass", "forget what you filed this week"])
def test_undo_phrasings(cmdr, said):
    assert _GARDEN_UNDO_RX.match(said)
    assert _h_garden_undo(cmdr, said, None).reply.startswith("Forgotten, sir")


def test_undo_is_matched_before_the_report(cmdr):
    # "forget what you filed" must never be answered by reading it out.
    assert cmdr._match_assistant("forget what you filed this week") == "garden undo"
    assert cmdr._match_assistant("memory report") == "garden report"


def test_a_handler_that_raises_is_an_excuse(cmdr):
    cmdr.services.garden_undo = lambda: 1 / 0
    assert "couldn't undo" in _h_garden_undo(cmdr, "forget the last garden pass",
                                             None).reply


def test_without_the_service_both_fall_through(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "il2.json")
    c = Commander(SimpleNamespace(tts=MagicMock()))
    assert _h_garden_report(c, "memory report", None) is None
    assert _h_garden_undo(c, "forget the last garden pass", None) is None


def test_the_module_docstring_promise_holds():
    """SOURCE is the one tag the whole feature turns on; a rename that
    misses memory.facts_from would silently orphan every filed fact."""
    assert garden_mod.SOURCE == "garden"
