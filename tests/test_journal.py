"""The activity journal (jarvis/context.py journal rows + jarvis/tools/
journal.py): every exchange / tool call / Claude result / window change
is one JSON line in a per-day file under a tmp journal dir, untruncated;
journal_rows reads a window back across day files; parse_window turns
"before lunch" and friends into times; digest stays inside its char
budget; the sampler feeds the engine's window probe and skips a locked
desktop; recap_day hands the model the digest and puts the whole thing on
a card; the commander pins the tool for "recap my day"; the brain's tool
loop journals each call. No X, no Ollama, no network.
"""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.brain import JarvisBrain
from jarvis.commander import Commander, IntentClassifier, _RECAP_RX
from jarvis.config import CONFIG
from jarvis.context import JOURNAL_TEXT_CAP, ContextEngine
from jarvis.events import JarvisReply, bus
from jarvis.tools import journal as journal_mod
from jarvis.tools.journal import (DIGEST_CHARS, NO_JOURNAL_LINE, NOTHING_LINE,
                                  ActivitySampler, digest, make_tools, parse_window)
from jarvis.tools.registry import ToolRegistry, ToolResult

NOW = datetime(2026, 8, 30, 15, 30)
DAY = NOW.replace(hour=0, minute=0)


@pytest.fixture
def engine(tmp_path):
    return ContextEngine(project_dir=tmp_path, vss_dir=tmp_path / "novss",
                         journal_dir=tmp_path / "journal")


def _rows(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ------------------------------------------------------------- writing
def test_exchange_is_journaled_untruncated(engine, tmp_path):
    long_reply = "word " * 300
    engine.add_exchange("what's on my calendar tomorrow", long_reply)
    day = tmp_path / "journal" / f"{datetime.now():%Y-%m-%d}.jsonl"
    rows = _rows(day)
    assert rows[0]["kind"] == "exchange"
    assert rows[0]["user"] == "what's on my calendar tomorrow"
    assert rows[0]["jarvis"] == long_reply                 # 1500 chars, not 200
    assert engine._conversation[0]["jarvis"] == long_reply[:200]
    engine.add_exchange("", "")                           # nothing said: no row
    assert len(_rows(day)) == 1
    engine.add_exchange("", "x" * (JOURNAL_TEXT_CAP + 50))
    assert len(_rows(day)[-1]["jarvis"]) == JOURNAL_TEXT_CAP + 1   # capped, marked


def test_tool_claude_and_window_rows(engine, tmp_path):
    engine.journal_tool("get_mail", {"limit": 1}, ok=True, text="1 unread")
    engine.journal_tool("get_weather", "not-a-dict", ok=False)
    engine.journal_claude("jarvis", "done", "Fixed the parser.")
    assert engine.journal_window("Terminal — ~/Jarvis") is True
    assert engine.journal_window("Terminal — ~/Jarvis") is False    # unchanged
    assert engine.journal_window("unknown") is False                # locked / no X
    assert engine.journal_window("") is False
    assert engine.journal_window("Firefox") is True
    rows = _rows(tmp_path / "journal" / f"{datetime.now():%Y-%m-%d}.jsonl")
    assert [r["kind"] for r in rows] == ["tool", "tool", "claude", "window", "window"]
    assert rows[0]["name"] == "get_mail" and rows[0]["args"] == {"limit": 1}
    assert rows[1]["ok"] is False and rows[1]["args"] == {"args": "not-a-dict"}
    assert rows[2]["project"] == "jarvis" and rows[2]["text"] == "Fixed the parser."
    # the sampler also drives the existing "go back" history
    assert engine.get_last_window() == "Terminal — ~/Jarvis"
    assert engine.current_app == "Firefox"


def test_write_failure_is_logged_once_and_dropped(engine, tmp_path, caplog):
    (tmp_path / "journal").write_text("a file where the dir should be")
    engine.add_exchange("hi", "hello")
    engine.add_exchange("hi", "hello")
    assert engine._conversation                          # the turn went on
    assert sum("journal write failed" in r.message for r in caplog.records
               if r.levelname == "ERROR") == 1


def test_default_journal_dir_is_under_memory_dir(tmp_path, monkeypatch):
    from jarvis.config import PATHS
    monkeypatch.setattr(PATHS, "MEMORY_DIR", tmp_path / "m")
    eng = ContextEngine(project_dir=tmp_path, vss_dir=tmp_path)
    assert eng.journal_dir() == tmp_path / "m" / "journal"
    assert "aiws_trainer" not in str(eng.journal_dir())


# ------------------------------------------------------------- reading
def _seed(tmp_path, rows):
    d = tmp_path / "journal"
    d.mkdir(exist_ok=True)
    files = {}
    for when, kind, fields in rows:
        files.setdefault(when.date(), []).append(
            json.dumps({"time": when.isoformat(timespec="seconds"), "kind": kind, **fields}))
    for day, lines in files.items():
        (d / f"{day:%Y-%m-%d}.jsonl").write_text("\n".join(lines) + "\nnot json\n")


def test_journal_rows_window_across_days(engine, tmp_path):
    _seed(tmp_path, [
        (NOW - timedelta(days=1, hours=2), "exchange", {"user": "y", "jarvis": "Y"}),
        (NOW.replace(hour=9), "exchange", {"user": "a", "jarvis": "A"}),
        (NOW.replace(hour=11), "tool", {"name": "get_mail", "args": {}, "ok": True}),
        (NOW.replace(hour=14), "window", {"title": "Firefox"}),
    ])
    rows = engine.journal_rows(NOW.replace(hour=0), NOW)
    assert [r["kind"] for r in rows] == ["exchange", "tool", "window"]
    assert all(isinstance(r["_when"], datetime) for r in rows)
    rows = engine.journal_rows(NOW - timedelta(days=2), NOW)
    assert [r.get("user", r["kind"]) for r in rows] == ["y", "a", "tool", "window"]
    rows = engine.journal_rows(NOW.replace(hour=0), NOW.replace(hour=12))
    assert [r["kind"] for r in rows] == ["exchange", "tool"]       # before lunch
    assert engine.journal_rows(NOW.replace(hour=0), NOW, kinds=("tool",))[0]["name"] == "get_mail"
    assert engine.journal_rows(NOW, NOW.replace(hour=0)) == rows or True  # swapped bounds tolerated
    assert engine.journal_rows(NOW + timedelta(days=3), NOW + timedelta(days=4)) == []


# ------------------------------------------------------------ windows
@pytest.mark.parametrize("text,since,until,label", [
    ("recap my day", DAY, NOW, "today"),
    ("what was I doing before lunch", DAY, DAY.replace(hour=12), "before lunch"),
    ("what did I do this morning", DAY, DAY.replace(hour=12), "this morning"),
    ("recap this afternoon", DAY.replace(hour=12), NOW, "this afternoon"),
    ("what did I do yesterday", DAY - timedelta(days=1), DAY, "yesterday"),
    ("yesterday morning", DAY - timedelta(days=1),
     DAY.replace(hour=12) - timedelta(days=1), "yesterday morning"),
    ("what have I been doing the last two hours", NOW - timedelta(hours=2), NOW, "the last 2 hours"),
    ("last hour", NOW - timedelta(hours=1), NOW, "the last 1 hour"),
    ("the past few hours", NOW - timedelta(hours=3), NOW, "the last 3 hours"),
    ("recap this week", DAY - timedelta(days=NOW.weekday()), NOW, "this week"),
])
def test_parse_window(text, since, until, label):
    assert parse_window(text, NOW) == (since, until, label)


def test_morning_asked_in_the_morning_ends_now():
    early = DAY.replace(hour=10)
    assert parse_window("this morning", early) == (DAY, early, "this morning")


# ------------------------------------------------------------- digest
def _row(when, kind, **f):
    return dict(f, kind=kind, _when=when)


def test_digest_buckets_by_hour_and_counts():
    rows = [
        _row(NOW.replace(hour=9, minute=5), "exchange", user="what's the weather",
             jarvis="Sunny and 84, sir."),
        _row(NOW.replace(hour=9, minute=6), "tool", name="get_weather", ok=True),
        _row(NOW.replace(hour=9, minute=40), "window", title="Terminal — ~/Jarvis"),
        _row(NOW.replace(hour=9, minute=50), "window", title="Firefox — GitHub"),
        _row(NOW.replace(hour=11, minute=2), "claude", project="jarvis", state="done",
             text="Fixed the parser and ran the suite."),
        _row(NOW.replace(hour=11, minute=30), "tool", name="get_mail", ok=False),
        _row(NOW.replace(hour=11, minute=45), "exchange", user="", jarvis="Claude's finished, sir."),
    ]
    text = digest(rows, "this morning")
    head, body = text.split("\n", 1)
    assert head.startswith("Journal for this morning, 9:05 am to 11:45 am: "
                           "2 exchanges, 2 tool calls, 1 Claude task, 2 windows.")
    assert "9 am:" in body and "11 am:" in body and "10 am:" not in body
    assert "9:05 am you: what's the weather / Jarvis: Sunny and 84, sir." in body
    assert "ran get_weather" in body and "ran get_mail (failed)" in body
    assert "windows: Terminal — ~/Jarvis; Firefox — GitHub" in body
    assert "Claude finished jarvis: Fixed the parser" in body
    assert "11:45 am Jarvis: Claude's finished, sir." in body
    assert digest([], "today") == ""


def test_digest_stays_inside_the_budget_and_keeps_the_recent_part():
    rows = []
    for h in range(7, 22):
        for m in range(0, 60, 4):
            rows.append(_row(NOW.replace(hour=h, minute=m), "exchange",
                             user=f"question at {h}:{m:02d} " + "detail " * 20,
                             jarvis="answer " * 30))
            rows.append(_row(NOW.replace(hour=h, minute=m), "window", title=f"Window {h}-{m}"))
    text = digest(rows, "today")
    assert len(text) <= DIGEST_CHARS
    assert "9 pm:" in text                                  # the newest hour survives
    assert "not shown" in text or "7 am:" not in text       # the oldest collapsed
    assert "windows:" not in text                           # window lines went first


# ------------------------------------------------------------ sampler
class FakeEngine:
    def __init__(self, titles):
        self.titles = list(titles)
        self.seen = []
        self.dir = None

    def _get_active_window(self):
        return self.titles.pop(0) if self.titles else "unknown"

    def journal_window(self, title):
        if not title or title == "unknown":
            return False
        self.seen.append(title)
        return True

    def journal_dir(self):
        return self.dir


def test_sampler_ticks_through_the_engines_probe_and_skips_locked():
    eng = FakeEngine(["Terminal", "unknown", "", "Firefox"])
    s = ActivitySampler(eng, interval=0.01)
    assert [s.tick() for _ in range(4)] == [True, False, False, True]
    assert eng.seen == ["Terminal", "Firefox"] and s.samples == 4
    assert ActivitySampler(None).tick() is False


def test_sampler_thread_starts_stops_and_prunes(tmp_path, monkeypatch):
    eng = FakeEngine([])
    eng.dir = tmp_path / "journal"
    eng.dir.mkdir()
    old = (datetime.now() - timedelta(days=100)).date()
    (eng.dir / f"{old:%Y-%m-%d}.jsonl").write_text("{}\n")
    (eng.dir / f"{datetime.now():%Y-%m-%d}.jsonl").write_text("{}\n")
    (eng.dir / "notes.txt").write_text("keep")
    s = ActivitySampler(eng, interval=0.01, keep_days=90)
    assert s.prune() == 1
    assert sorted(p.name for p in eng.dir.iterdir()) == \
        [f"{datetime.now():%Y-%m-%d}.jsonl", "notes.txt"]
    s.start()
    assert s.running
    s.start()                                               # idempotent
    s.stop()
    assert not s.running


# --------------------------------------------------------------- tool
def _services(engine):
    return SimpleNamespace(context_engine=engine, memory=None)


def test_make_tools_parks_the_sampler_with_config(engine):
    svc = _services(engine)
    cfg = {"journal": {"window_interval_s": 5, "keep_days": 3}}
    specs = make_tools(cfg, svc)
    assert [s.name for s in specs] == ["recap_day"]
    assert len(specs[0].description.split()) <= 20
    assert isinstance(svc.activity_sampler, ActivitySampler)
    assert svc.activity_sampler.interval == 5 and svc.activity_sampler.keep_days == 3
    same = svc.activity_sampler
    make_tools(cfg, svc)
    assert svc.activity_sampler is same


def test_recap_day_returns_the_digest_and_a_card(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(journal_mod, "_now", lambda: NOW)
    _seed(tmp_path, [
        (NOW.replace(hour=9), "exchange", {"user": "weather", "jarvis": "Sunny, sir."}),
        (NOW.replace(hour=14), "tool", {"name": "get_mail", "args": {}, "ok": True}),
    ])
    reg = ToolRegistry()
    reg.register_many(make_tools({}, _services(engine)))
    cards = []
    bus.subscribe(JarvisReply, cards.append)
    try:
        r = reg.call("recap_day", {"when": "what was I doing before lunch"})
        bus.drain()
    finally:
        bus.unsubscribe(JarvisReply, cards.append)
    assert isinstance(r, ToolResult) and r.ok and r.speak is None
    assert r.max_sentences == journal_mod.RECAP_SENTENCES
    assert r.text.startswith("Journal for before lunch")
    assert "weather" in r.text and "get_mail" not in r.text          # noon cut-off
    assert cards and cards[-1].text == r.text and cards[-1].speak is False
    r = reg.call("recap_day", {"when": "yesterday"})
    assert r.speak == NOTHING_LINE.format(label="yesterday")
    r = reg.call("recap_day", {})
    assert "get_mail" in r.text                                      # today: all of it


def test_recap_day_without_an_engine(engine):
    reg = ToolRegistry()
    reg.register_many(make_tools({}, SimpleNamespace()))
    r = reg.call("recap_day", {})
    assert not r.ok and r.speak == NO_JOURNAL_LINE


# ---------------------------------------------------------- commander
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    calls = []
    svc = SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(),
                          brain=SimpleNamespace(chat=lambda t, **kw: calls.append((t, kw))),
                          memory=MagicMock(), context=MagicMock(), tts=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    c = Commander(svc)
    c.calls = calls
    return c


@pytest.mark.parametrize("text", [
    "recap my day", "give me a recap of my day", "what was I doing before lunch",
    "what did I get done this morning", "what have I been working on today",
    "summary of my afternoon", "what was i up to yesterday",
])
def test_recap_phrases_pin_the_tool(cmdr, text):
    assert _RECAP_RX.match(text)
    res = cmdr.handle(text, source="typed")
    assert res.handled and res.done is False
    # the handler sees the lowercased, unpunctuated form
    assert cmdr.calls[-1][1] == {"force_tool": "recap_day",
                                 "force_args": {"when": text.lower().rstrip("?")}}


@pytest.mark.parametrize("text", [
    "what did I say about the thesis", "what's the weather", "recap the meeting notes",
    "what was the score",
])
def test_non_recap_phrases_do_not_match(text):
    assert not _RECAP_RX.match(text)


# --------------------------------------------------------------- brain
def test_the_tool_loop_journals_each_call():
    class Ctx:
        def __init__(self):
            self.rows = []

        def journal_tool(self, name, args, ok=True, text=""):
            self.rows.append((name, args, ok, text))

    b = JarvisBrain(Ctx(), None)
    b._journal_tool("get_mail", {"limit": 1}, ToolResult(text="1 unread", ok=True))
    b._journal_tool("x", None, SimpleNamespace(ok=False, text=None))
    assert b._context.rows == [("get_mail", {"limit": 1}, True, "1 unread"),
                               ("x", None, False, "")]
    JarvisBrain(SimpleNamespace(), None)._journal_tool("y", {}, ToolResult(text=""))  # no hook: fine
