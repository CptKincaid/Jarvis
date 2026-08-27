"""Tests for jarvis.memory — remember/recall/habits round-trip in tmpdirs,
plus the one-time jarvis_data legacy migration merge."""
import json

import pytest

from jarvis.memory import JarvisMemory


@pytest.fixture
def mem_dir(tmp_path):
    return tmp_path / "jarvis_memory"


@pytest.fixture
def legacy_dir(tmp_path):
    return tmp_path / "jarvis_data"


def make(mem_dir, legacy_dir):
    return JarvisMemory(memory_dir=mem_dir, legacy_dir=legacy_dir)


# ---------------------------------------------------------------- facts
def test_remember_recall_roundtrip(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.remember("project", "training uses batch size 16")

    # Fresh instance from the same dir → persisted
    mem2 = make(mem_dir, legacy_dir)
    hits = mem2.recall("batch size")
    assert len(hits) == 1
    assert hits[0]["key"] == "project"
    assert hits[0]["value"] == "training uses batch size 16"

    # Recall by key substring too
    assert mem2.recall("proj")[0]["key"] == "project"


def test_forget(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.remember("thing", "value")
    mem.forget("thing")
    assert mem.recall("thing") == []
    assert make(mem_dir, legacy_dir).recall("thing") == []


def test_recall_no_match(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.remember("a", "b")
    assert mem.recall("zzz-not-there") == []


# --------------------------------------------------------------- habits
def test_habits_roundtrip_and_suggest(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    for _ in range(12):
        mem.log_habit("check gpu")

    # Persisted across instances
    mem2 = make(mem_dir, legacy_dir)
    assert mem2.suggest_by_habit() == "check gpu"


def test_suggest_needs_history(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.log_habit("check gpu")
    assert mem.suggest_by_habit() is None  # < 10 entries


def test_habit_cap(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    for i in range(JarvisMemory.MAX_HABITS + 20):
        mem.log_habit(f"cmd {i}")
    assert len(json.loads((mem_dir / "habits.json").read_text())) == \
        JarvisMemory.MAX_HABITS


def test_habit_summary(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    assert "No habits" in mem.get_habit_summary()
    mem.log_habit("deploy")
    summary = mem.get_habit_summary()
    assert "deploy" in summary and "1x" in summary


# ------------------------------------------------------------- sessions
def test_save_session_roundtrip(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.save_session("worked on the detector pipeline")

    mem2 = make(mem_dir, legacy_dir)
    sessions = mem2.get_recent_sessions()
    assert len(sessions) == 1
    assert sessions[0]["summary"] == "worked on the detector pipeline"
    assert "worked on the detector" in mem2.format_sessions_for_prompt()


def test_save_session_ignores_empty(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.save_session("")
    assert mem.get_recent_sessions() == []


# ---------------------------------------------------------------- notes
def test_notes_roundtrip(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    path = mem.save_note("remember to rotate the dataset")
    assert path
    notes = make(mem_dir, legacy_dir).get_notes()
    assert len(notes) == 1
    assert "rotate the dataset" in notes[0]["content"]


# ---------------------------------------------------------- preferences
def test_preferences_roundtrip(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.set_preference("voice", "edge")
    assert make(mem_dir, legacy_dir).get_preference("voice") == "edge"
    assert mem.get_preference("missing", "dflt") == "dflt"


# ------------------------------------------------------------ intent log
def test_intent_log(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.log_intent("open firefox", True)
    mem.log_intent("talking to my friend", False)
    entries = make(mem_dir, legacy_dir).get_intent_log()
    assert entries[0]["label"] == "yes"
    assert entries[1]["label"] == "no"


# ------------------------------------------------------------- migration
def _seed_legacy(legacy_dir, n_habits=3):
    legacy_dir.mkdir(parents=True)
    habits = [
        {"time": f"2026-01-0{i + 1}T09:00:00", "hour": 9, "day": "Monday",
         "command": "morning check", "context": None}
        for i in range(n_habits)
    ]
    (legacy_dir / "habits.json").write_text(json.dumps(habits))
    notes = legacy_dir / "voice_notes"
    notes.mkdir()
    (notes / "note_2026-01-01_09-00-00.txt").write_text(
        "[2026-01-01 09:00]\nlegacy note body\n")


def test_migration_merges_habits_copies_notes_renames_dir(
        mem_dir, legacy_dir):
    _seed_legacy(legacy_dir, n_habits=3)
    # Pre-existing habits in the NEW store (must survive the merge)
    mem_dir.mkdir(parents=True)
    existing = [
        {"time": "2026-08-25T10:00:00", "hour": 10, "day": "Tuesday",
         "command": "check gpu", "context": None}
        for _ in range(2)
    ]
    (mem_dir / "habits.json").write_text(json.dumps(existing))

    mem = make(mem_dir, legacy_dir)

    # Habit entries merged (3 legacy + 2 existing) → merged counts
    habits = json.loads((mem_dir / "habits.json").read_text())
    assert len(habits) == 5
    cmds = [h["command"] for h in habits]
    assert cmds.count("morning check") == 3
    assert cmds.count("check gpu") == 2

    # Notes copied into the single store
    notes = mem.get_notes()
    assert any("legacy note body" in n["content"] for n in notes)

    # Old dir renamed .migrated
    assert not legacy_dir.exists()
    assert legacy_dir.with_name(legacy_dir.name + ".migrated").is_dir()


def test_migration_runs_once(mem_dir, legacy_dir):
    _seed_legacy(legacy_dir, n_habits=3)
    make(mem_dir, legacy_dir)
    # Second construction: legacy dir is gone, no duplicate merge
    make(mem_dir, legacy_dir)
    habits = json.loads((mem_dir / "habits.json").read_text())
    assert len(habits) == 3


def test_no_legacy_dir_is_fine(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    mem.remember("k", "v")
    assert mem.recall("k")


# ----------------------------------------------------- format_for_context
def test_format_for_context(mem_dir, legacy_dir):
    mem = make(mem_dir, legacy_dir)
    assert mem.format_for_context() == ""
    mem.remember("gpu", "GB10 unified memory")
    mem.save_session("tested memory module")
    text = mem.format_for_context()
    assert "GB10 unified memory" in text
    assert "tested memory module" in text
