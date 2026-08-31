"""The study ledger: focus_history.jsonl, the merge with the timekeeper's
own focus-block rows, the two spoken answers, and the backfill script.

Real modules: a real FocusSession over a tmp state path with a real
Timekeeper store behind it, a real Commander for the phrases. No Tk, no
audio, no Spotify (the music mode is off).
"""
import json
import time
from datetime import timedelta
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.focus as focus_mod
from jarvis.commander import ASSISTANT_TIER1, Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.focus import (NO_STUDY_LINE, STREAK_NONE_LINE, STREAK_ONE_LINE,
                          FocusSession, read_history, streak_days, streak_line,
                          study_days, summary_line, time_words, week_start)
from jarvis.tools.timekeeper import Timekeeper

DAY = 86_400.0


@pytest.fixture
def memory(tmp_path):
    d = tmp_path / "jarvis_memory"
    d.mkdir()
    return d


@pytest.fixture
def tk(memory):
    keeper = Timekeeper(memory / "timekeeper.db")
    yield keeper
    keeper.stop() if hasattr(keeper, "stop") else None


@pytest.fixture
def session(memory, tk):
    clock = {"t": time.time()}
    svc = SimpleNamespace(timekeeper=tk, spotify=None, assistant=None, speak=None)
    s = FocusSession(svc, state_path=memory / "focus_session.json",
                     now=lambda: clock["t"], bg=lambda fn: fn())
    s.clock = clock
    yield s
    s.stop()


def _row(started, blocks=2, block_min=25, label="thesis"):
    return {"date": focus_mod._day(started), "started": started,
            "ended": started + blocks * block_min * 60, "label": label,
            "blocks": blocks, "block_min": block_min}


# ------------------------------------------------------------- the write
def test_a_finished_session_lands_in_the_ledger(session):
    session.start("thesis", block_min=25, break_min=5)
    session.state["blocks_done"] = 3
    line = session.end()
    assert "three blocks" in line
    rows = read_history(session.history_path)
    assert len(rows) == 1
    assert rows[0]["blocks"] == 3 and rows[0]["block_min"] == 25
    assert rows[0]["label"] == "thesis"
    assert rows[0]["date"] == focus_mod._day(rows[0]["started"])


def test_the_state_file_is_still_overwritten_but_the_ledger_is_not(session):
    """The bug this fixes: focus_session.json is the CURRENT session only."""
    session.start("thesis")
    session.state["blocks_done"] = 2
    session.end()
    session.clock["t"] += 3600            # the dedupe key is the start stamp
    session.start("reading")
    session.state["blocks_done"] = 1
    session.end()
    state = json.loads(Path(session._state_path).read_text())
    assert state.get("blocks_done") == 1                 # only the last one
    rows = read_history(session.history_path)
    assert [r["label"] for r in rows] == ["thesis", "reading"]


def test_a_session_with_no_full_block_logs_nothing(session):
    session.start("thesis")
    session.end()
    assert read_history(session.history_path) == []


def test_the_lapsed_path_cannot_file_the_same_night_twice(session):
    """reconcile() closes a lapsed session with end("lapsed"); a crash and
    restart must not count it again."""
    session.start("thesis")
    session.state["blocks_done"] = 2
    session.end()
    before = read_history(session.history_path)
    # a restart that finds the same session still marked live
    session.state["phase"] = "block"
    session.end("lapsed")
    assert read_history(session.history_path) == before


def test_a_ledger_the_process_cannot_write_is_not_fatal(session, monkeypatch):
    session.start("thesis")
    session.state["blocks_done"] = 2

    def _boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "open", _boom)
    assert "two blocks" in session.end()                 # still speaks


def test_no_state_path_means_no_ledger(tk):
    svc = SimpleNamespace(timekeeper=tk, spotify=None, assistant=None, speak=None)
    s = FocusSession(svc, bg=lambda fn: fn())
    try:
        assert s.history_path is None
        s.start("thesis")
        s.state["blocks_done"] = 1
        assert s.end()                                   # no crash, no file
    finally:
        s.stop()


# ------------------------------------------------------------- the merge
def test_ledger_rows_become_days():
    now = time.time()
    days = study_days(rows=[_row(now - DAY, blocks=2), _row(now, blocks=3)], blocks=[])
    assert days[focus_mod._day(now)] == {"blocks": 3, "minutes": 75}
    assert days[focus_mod._day(now - DAY)] == {"blocks": 2, "minutes": 50}


def test_a_timekeeper_block_inside_a_logged_session_is_not_counted_twice():
    now = time.time()
    row = _row(now, blocks=2, block_min=25)
    inside = [{"when": row["started"] + 60, "minutes": 25},
              {"when": row["started"] + 30 * 60, "minutes": 25}]
    days = study_days(rows=[row], blocks=inside)
    assert days[row["date"]] == {"blocks": 2, "minutes": 50}


def test_a_timekeeper_block_outside_every_session_counts_as_itself():
    """A session that ended before the ledger existed, or one the app died
    in: the block row is all that survives, and it is real study."""
    now = time.time()
    old = now - 10 * DAY
    days = study_days(rows=[_row(now)], blocks=[{"when": old, "minutes": 50}])
    assert days[focus_mod._day(old)] == {"blocks": 1, "minutes": 50}


def test_junk_rows_are_skipped_not_raised(memory):
    path = memory / focus_mod.HISTORY_NAME
    good = _row(time.time())
    path.write_text("not json\n" + json.dumps(good) + "\n{}\n\n[]\n")
    assert [r["blocks"] for r in read_history(path)] == [good["blocks"]]
    assert study_days(state_path=memory / "focus_session.json", db_path=None)


def test_timekeeper_blocks_reads_the_real_store(session, tk, memory):
    """End to end: a block filed by a live session is readable back out of
    timekeeper.db with the right length."""
    session.start("thesis", block_min=30, break_min=5)
    item = tk.get(session.state["block_id"])
    tk._update(item.id, state="done", fired_at=time.time())
    blocks = focus_mod.timekeeper_blocks(memory / "timekeeper.db")
    assert [b["minutes"] for b in blocks] == [30]
    # the halfway and break items share the "focus:" prefix and must not count
    assert len(blocks) == 1


def test_a_missing_or_broken_store_reads_as_nothing(tmp_path):
    assert focus_mod.timekeeper_blocks(tmp_path / "nope.db") == []
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"this is not sqlite")
    assert focus_mod.timekeeper_blocks(bad) == []
    assert focus_mod.timekeeper_blocks(None) == []


# -------------------------------------------------------------- the words
def test_time_words():
    # count_words stops at twelve, exactly as focus.left_words already does
    assert time_words(0) == "no minutes"
    assert time_words(1) == "a minute"
    assert time_words(9) == "nine minutes"
    assert time_words(50) == "50 minutes"
    assert time_words(60) == "an hour"
    assert time_words(75) == "an hour and 15 minutes"
    assert time_words(190) == "three hours and ten minutes"


def test_week_start_is_monday():
    assert week_start(date(2026, 8, 30)).isoformat() == "2026-08-24"   # a Sunday
    assert week_start(date(2026, 8, 24)).isoformat() == "2026-08-24"


def test_summary_line_counts_only_the_window():
    today = date.today()
    days = {today.isoformat(): {"blocks": 3, "minutes": 75},
            (today - timedelta(days=1)).isoformat(): {"blocks": 2, "minutes": 50},
            (today - timedelta(days=40)).isoformat(): {"blocks": 9, "minutes": 300}}
    line = summary_line(days, today - timedelta(days=1))
    assert line == "Two hours and five minutes this week, sir, over five blocks on two days."
    assert summary_line({}, today) == NO_STUDY_LINE.format(when="this week")
    assert "one day" in summary_line(days, today)


def test_streak_counts_back_from_today_or_yesterday():
    today = date.today()

    def days(*offsets):
        return {(today - timedelta(days=o)).isoformat(): {"blocks": 1, "minutes": 25}
                for o in offsets}

    assert streak_days(days(0, 1, 2), today) == 3
    # yesterday still counts: at 9 am last night's streak is not broken yet
    assert streak_days(days(1, 2), today) == 2
    assert streak_days(days(2, 3), today) == 0
    assert streak_days({}, today) == 0
    assert streak_line({}, today) == STREAK_NONE_LINE
    assert streak_line(days(0), today) == STREAK_ONE_LINE
    assert streak_line(days(0, 1), today) == "two days running, sir."


# ----------------------------------------------------------- the commander
@pytest.fixture
def cmdr(session, tk, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    svc = SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
                          context=MagicMock(), tts=MagicMock(), focus=session,
                          timekeeper=tk)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock())
    return Commander(svc)


def test_the_ledger_phrases_are_tier_one():
    names = {c.name for c in ASSISTANT_TIER1}
    assert {"study total", "study streak"} <= names


@pytest.mark.parametrize("text", [
    "how much did i study this week", "how much have i studied this week",
    "how long did i study today", "how much study time did i log",
    "what's my study time", "how much did i study yesterday",
])
def test_study_total_phrases(cmdr, text):
    assert cmdr._match_assistant(text) == "study total"


@pytest.mark.parametrize("text", [
    "what's my streak", "how's my study streak", "streak",
    "what is my streak", "how's the streak going",
])
def test_streak_phrases(cmdr, text):
    assert cmdr._match_assistant(text) == "study streak"


def test_the_answer_reads_the_ledger(cmdr, session, monkeypatch):
    # Boundary-proof: at 00:10 on a Monday, "yesterday" is last ISO week and
    # the second row silently fell out (caught live 2026-08-31). Widen the
    # window seam instead of trusting the wall clock's weekday.
    from jarvis import focus as focus_mod
    monkeypatch.setattr(focus_mod, "week_start", lambda d: d - timedelta(days=6))
    now = time.time()
    session.history_path.write_text(
        json.dumps(_row(now, blocks=2, block_min=25)) + "\n" +
        json.dumps(_row(now - DAY, blocks=1, block_min=50)) + "\n")
    res = cmdr.handle("how much did i study this week", "typed")
    assert res.handled and res.speak
    assert "three blocks" in res.reply and "hour" in res.reply


def test_an_empty_ledger_says_so(cmdr):
    res = cmdr.handle("how much did i study this week", "typed")
    assert res.reply == NO_STUDY_LINE.format(when="this week")
    assert cmdr.handle("what's my streak", "typed").reply == STREAK_NONE_LINE


def test_yesterday_does_not_include_today(cmdr, session):
    now = time.time()
    session.history_path.write_text(
        json.dumps(_row(now, blocks=4, block_min=25)) + "\n" +
        json.dumps(_row(now - DAY, blocks=1, block_min=25)) + "\n")
    res = cmdr.handle("how much did i study yesterday", "typed")
    assert "one block" in res.reply and "four" not in res.reply


def test_a_session_still_running_is_counted(cmdr, session):
    session.start("thesis", block_min=25, break_min=5)
    session.state["blocks_done"] = 2
    res = cmdr.handle("how much did i study today", "typed")
    assert "two blocks" in res.reply
    assert cmdr.handle("what's my streak", "typed").reply == STREAK_ONE_LINE


# ------------------------------------------------------------- the script
def test_backfill_writes_only_the_orphans(session, tk, memory):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "backfill", Path(__file__).resolve().parent.parent /
        "scripts" / "backfill_focus_history.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    now = time.time()
    logged = _row(now, blocks=1, block_min=25)
    session.history_path.write_text(json.dumps(logged) + "\n")
    state = Path(session._state_path)
    db = memory / "timekeeper.db"
    # one block inside the logged session, one from a week the ledger never saw
    tk._update(tk.add_silent_timer(1500, "block 1").id, state="done",
               fired_at=logged["started"] + 300)
    old_item = tk.add_silent_timer(1500, "block 1")
    tk._update(old_item.id, state="done", fired_at=now - 9 * DAY)

    orphans = mod.orphan_blocks(state, db)
    assert len(orphans) == 1 and abs(orphans[0]["when"] - (now - 9 * DAY)) < 1

    rows = mod.as_rows(orphans)
    assert rows[0]["blocks"] == 1 and rows[0]["block_min"] == 25
    assert rows[0]["source"] == "backfill"

    # idempotent: once those rows are in the ledger there is nothing left
    with session.history_path.open("a") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    assert mod.orphan_blocks(state, db) == []


# --------------------------------------------------------- the day review
def test_the_nightly_review_says_how_much_was_studied(tmp_path):
    import jarvis.dayreview as dr
    day = date.today() - timedelta(days=1)
    table = {day.isoformat(): {"blocks": 4, "minutes": 100}}
    digest = dr.summarize_day(tmp_path / "nope.log", tmp_path / "nope.jsonl", day,
                              study=table)
    assert digest["study_blocks"] == 4 and digest["study_minutes"] == 100
    # a day he studied is a day worth reviewing even with no Jarvis log
    assert digest["has_data"]
    line = dr.spoken_line(digest)
    assert "You studied four blocks, an hour and 40 minutes." in line
    assert "study blocks" in dr.table(digest)


def test_a_day_with_no_study_reads_exactly_as_before(tmp_path):
    import jarvis.dayreview as dr
    day = date.today() - timedelta(days=1)
    plain = dr.summarize_day(tmp_path / "nope.log", tmp_path / "nope.jsonl", day)
    with_empty = dr.summarize_day(tmp_path / "nope.log", tmp_path / "nope.jsonl", day,
                                  study={})
    assert plain["study_blocks"] == 0 and plain == with_empty
    assert not plain["has_data"] and dr.spoken_line(plain) == ""


def test_a_study_reader_that_raises_never_stops_a_review(tmp_path):
    import jarvis.dayreview as dr

    def boom():
        raise RuntimeError("ledger on fire")

    r = dr.DayReviewer(tmp_path / "nope.log", tmp_path / "nope.jsonl",
                       tmp_path / "reviews", study=boom)
    assert r.review(date.today() - timedelta(days=1))["study_blocks"] == 0
