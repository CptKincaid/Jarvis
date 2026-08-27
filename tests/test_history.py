"""Tests for jarvis.history — typed-command history."""
import json

from jarvis.history import TypedHistory


def test_add_recent_and_persist(tmp_path):
    h = TypedHistory(tmp_path / "h.jsonl")
    assert h.add("what time is it")
    assert h.add("open the terminal")
    assert h.recent() == ["open the terminal", "what time is it"]
    assert h.last() == "open the terminal"
    lines = (tmp_path / "h.jsonl").read_text().splitlines()
    assert [json.loads(line)["text"] for line in lines] == [
        "what time is it", "open the terminal"]
    assert json.loads(lines[0])["source"] == "typed"
    # reload
    again = TypedHistory(tmp_path / "h.jsonl")
    assert len(again) == 2 and again.recent(1) == ["open the terminal"]


def test_consecutive_duplicates_collapse(tmp_path):
    h = TypedHistory(tmp_path / "h.jsonl")
    assert h.add("status") is True
    assert h.add("status") is False
    assert h.add("  status ") is False
    assert len(h) == 1
    assert h.add("git status") is True
    assert h.add("status") is True                 # not consecutive any more


def test_empty_ignored(tmp_path):
    h = TypedHistory(tmp_path / "h.jsonl")
    assert h.add("") is False and h.add("   ") is False and len(h) == 0
    assert h.prev() is None and h.next() is None and h.last() is None


def test_cursor_navigation_like_a_shell(tmp_path):
    h = TypedHistory(tmp_path / "h.jsonl")
    for t in ["one", "two", "three"]:
        h.add(t)
    assert h.prev() == "three"
    assert h.prev() == "two"
    assert h.prev() == "one"
    assert h.prev() == "one"                        # clamps at oldest
    assert h.next() == "two"
    assert h.next() == "three"
    assert h.next() is None                         # past the newest: empty
    assert h.prev() == "three"                      # starts again from newest
    h.add("four")                                   # any add resets
    assert h.prev() == "four"


def test_search_prefix_newest_first_dedup(tmp_path):
    h = TypedHistory(tmp_path / "h.jsonl")
    for t in ["open terminal", "open browser", "what time", "open terminal"]:
        h.add(t)
    assert h.search("open") == ["open terminal", "open browser"]
    assert h.search("OPEN", n=1) == ["open terminal"]
    assert h.search("zzz") == []


def test_max_items_trims_and_rewrites(tmp_path):
    h = TypedHistory(tmp_path / "h.jsonl", max_items=3)
    for i in range(5):
        h.add(f"cmd {i}")
    assert h.recent() == ["cmd 4", "cmd 3", "cmd 2"]
    lines = (tmp_path / "h.jsonl").read_text().splitlines()
    assert len(lines) == 3


def test_corrupt_lines_skipped(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"text": "good"}\nnot json\n{"nope": 1}\n')
    h = TypedHistory(p)
    assert h.recent() == ["good"]


def test_clear(tmp_path):
    h = TypedHistory(tmp_path / "h.jsonl")
    h.add("x")
    h.clear()
    assert len(h) == 0 and (tmp_path / "h.jsonl").read_text() == ""
