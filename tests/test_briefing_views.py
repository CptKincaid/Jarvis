"""The briefing's two new views and the explicit preferences (backlog 7,
12, 33): the good-night preview of tomorrow with its wake-up offer, the
week forecast day by day, per-section switches ("no news in the morning"),
the verbosity knob, the card rows for the new keys, and the yes/no that
turns the offer into an alarm.

Firewall as tests/test_briefing.py: tmp JARVIS_ASSISTANT_CONFIG, no
network, no Tk. Tools are the real modules with fake registries.
"""
import json
import os
import threading
import time
import types
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.commander as commander
from jarvis.commander import (COURTESY_REPLIES, GOODNIGHT_PREVIEW_LINE, NO_ALARM_LINE,
                              PREVIEW_ASK, Commander, IntentClassifier)
from jarvis.config import CONFIG
from jarvis.events import BriefingReady, bus
from jarvis.tools import briefing as br
from jarvis.tools.briefing import (build_briefing, build_preview, build_week,
                                   coerce_when, section_on, verbosity_cap, wake_offer)
from jarvis.tools.calendar import Event
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec
from jarvis.ui.views import briefing_rows


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(br, "_fetch", _no_network)
    yield


# Sunday evening, the planning hour.
NOW = datetime(2026, 8, 30, 21, 30).astimezone()
TZ = NOW.tzinfo
TOMORROW = NOW.date() + timedelta(days=1)          # Monday 31 August

CAL_TOMORROW = ("Tomorrow: 8:00 am Biosensors for an hour at ZACH 350, "
                "2:30 pm standup; nothing else.")
CAL_WEEK = ("This week: today 6:00 pm dinner with Sam; Tuesday 9:00 am Biosensors "
            "for an hour at ZACH 350, 11:00 am Lecture, 2:00 pm Lab; "
            "Thursday all day: Holiday; nothing else.")
CANVAS_1 = "Due in the next day (1):\n1) CSCE 421 - Lab 3, tomorrow 11:59 pm"
CANVAS_7 = ("Due this week (2):\n1) CSCE 421 - Lab report, Tue 11:59 pm\n"
            "2) MATH 308 - Quiz 2, Fri 11:59 pm\nand 1 more")


class Reg(ToolRegistry):
    """get_weather / get_calendar / canvas_due answering from a table keyed
    by (name, the one argument that matters)."""

    def __init__(self, calendar_ok=True, weather_ok=True, canvas=True):
        super().__init__()
        self.calls = []

        def w(when="now", **kw):
            self.calls.append(("get_weather", {"when": when}))
            if not weather_ok:
                return ToolResult(text="weather unreachable", ok=False)
            return ToolResult(text=f"{when.capitalize()}: high 91, low 74, sunny.")

        def c(range="today", **kw):
            self.calls.append(("get_calendar", {"range": range}))
            text = {"tomorrow": CAL_TOMORROW, "week": CAL_WEEK}.get(range, "Nothing on today, sir.")
            return ToolResult(text=text, ok=calendar_ok) if calendar_ok else \
                ToolResult(text="calendar unreachable", ok=False)

        def d(days=7, **kw):
            self.calls.append(("canvas_due", {"days": days}))
            return ToolResult(text=CANVAS_1 if days == 1 else CANVAS_7)
        self.register(ToolSpec("get_weather", "w", handler=w))
        self.register(ToolSpec("get_calendar", "c", handler=c))
        if canvas:
            self.register(ToolSpec("canvas_due", "d", handler=d))


class Notes:
    def __init__(self, todos=("buy milk", "email the professor")):
        self.todos = list(todos)

    def list(self, kind, limit=10, include_done=False):
        assert kind == "todo"
        return [{"text": t} for t in self.todos[-limit:]]

    def count(self, kind, include_done=False):
        return len(self.todos)


class Item:
    def __init__(self, kind, due, label="", repeat=""):
        self.kind, self.due, self.label, self.repeat = kind, float(due), label, repeat
        self.state = "pending"

    @property
    def effective_due(self):
        return self.due


class TK:
    def __init__(self, items=()):
        self.items = list(items)
        self.alarms = []

    def list(self, kind="all", include_done=False):
        return [i for i in self.items if kind == "all" or i.kind == kind]

    def add_alarm(self, due, label="", repeat="once"):
        self.alarms.append((due, label, repeat))
        return Item("alarm", due, label)


def cfg(**over):
    data = {"briefing": {"enabled": False, "hn_items": 3, "news_feeds": [],
                         "sports_feeds": [], "stock_symbols": [], **over},
            "canvas": {"token": "tok"}}
    return data


def services(reg=None, notes=None, tk=None, calendar=None, tmp_path=None):
    return SimpleNamespace(tools=reg, notes=notes, timekeeper=tk, calendar=calendar,
                           news_cache_path=(tmp_path / "news.json") if tmp_path else None)


def _at(day, hh, mm=0):
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=TZ)


# ============================================================ preview
def test_build_preview_sections_sheet_and_wake_offer():
    reg = Reg()
    svc = services(reg, Notes(), TK())
    sections, sheet, offer = build_preview(cfg(), reg, svc, NOW)
    assert sections["weather"] == "Tomorrow: high 91, low 74, sunny."
    assert sections["calendar"] == [CAL_TOMORROW]
    assert sections["canvas"] == ["CSCE 421 - Lab 3, tomorrow 11:59 pm"]
    assert sections["todos"] == ["buy milk", "email the professor"]
    assert sections["alarm"] == "none set"
    # 8:00 am first thing, no alarm: offer an hour before, on the quarter hour
    assert offer["time"] == "7:00 am" and offer["label"] == "wake up"
    assert offer["line"] == "Shall I wake you at 7:00 am, sir?"
    assert datetime.fromtimestamp(offer["due"], TZ) == _at(TOMORROW, 7)
    assert offer["made_at"] == NOW.timestamp()
    assert sections["offer"] == offer["line"]
    assert sheet == (
        "Preview for tomorrow, Monday 31 August.\n"
        "Weather tomorrow: Tomorrow: high 91, low 74, sunny.\n"
        f"First up: {CAL_TOMORROW}\n"
        "Canvas due tomorrow: CSCE 421 - Lab 3, tomorrow 11:59 pm\n"
        "To-dos (open): buy milk; email the professor\n"
        "Alarm tomorrow: none set")
    # the offer is asked by the app, verbatim: the model never sees it
    assert "wake you" not in sheet
    assert reg.calls == [("get_weather", {"when": "tomorrow"}),
                         ("get_calendar", {"range": "tomorrow"}),
                         ("canvas_due", {"days": 1})]


def test_preview_offer_is_withheld_when_covered_late_or_off():
    reg = Reg()
    # an alarm already rings before the first event
    tk = TK([Item("alarm", _at(TOMORROW, 6, 30).timestamp(), "wake up")])
    sections, _sheet, offer = build_preview(cfg(), reg, services(reg, Notes(), tk), NOW)
    assert offer is None and "offer" not in sections
    assert sections["alarm"] == "6:30 am (wake up)"
    # an alarm AFTER the first event does not cover it
    tk = TK([Item("alarm", _at(TOMORROW, 9).timestamp())])
    _s, _sheet, offer = build_preview(cfg(), reg, services(reg, Notes(), tk), NOW)
    assert offer is not None and offer["time"] == "7:00 am"
    # a late first event needs no waking; the toggle wins over everything
    late = _at(TOMORROW, 10)
    assert wake_offer(cfg(), late, [], NOW) is None
    assert wake_offer(cfg(wake_offer=False), _at(TOMORROW, 8), [], NOW) is None
    assert wake_offer(cfg(sections={"alarms": False}), _at(TOMORROW, 8), [], NOW) is None
    # the lead and the threshold are config; 8:20 less 60 min lands on 7:15
    o = wake_offer(cfg(wake_lead_min=60), _at(TOMORROW, 8, 20), [], NOW)
    assert o["time"] == "7:15 am"
    o = wake_offer(cfg(early_before="08:00"), _at(TOMORROW, 8, 20), [], NOW)
    assert o is None
    # an offer in the past (asked after the wake time) is no offer
    assert wake_offer(cfg(), _at(TOMORROW, 8), [], _at(TOMORROW, 7, 30)) is None


def test_preview_prefers_the_parked_calendar_source():
    reg = Reg()
    events = [Event(start=_at(TOMORROW, 0), end=_at(TOMORROW + timedelta(days=1), 0),
                    all_day=True, title="Labor Day"),
              Event(start=_at(TOMORROW, 8, 30), end=_at(TOMORROW, 9, 20),
                    title="Biosensors", location="ZACH 350")]
    cal = SimpleNamespace(configured=True, events=lambda: events)
    sections, sheet, offer = build_preview(cfg(), reg, services(reg, Notes(), TK(), cal), NOW)
    assert sections["calendar"] == ["all day: Labor Day", "8:30 am Biosensors at ZACH 350"]
    assert offer["time"] == "7:30 am"                 # from the timed event, not the all-day
    assert ("get_calendar", {"range": "tomorrow"}) not in reg.calls
    # an unconfigured source falls back to the tool's text
    cal = SimpleNamespace(configured=False, events=lambda: events)
    sections, _sheet, _o = build_preview(cfg(), reg, services(reg, Notes(), TK(), cal), NOW)
    assert sections["calendar"] == [CAL_TOMORROW]


def test_preview_degrades_and_skips_unset_canvas():
    reg = Reg(calendar_ok=False, weather_ok=False, canvas=False)
    c = cfg()
    c["canvas"]["token"] = ""
    sections, sheet, offer = build_preview(c, reg, services(reg, Notes([]), TK()), NOW)
    assert sections["calendar"] == [] and sections["canvas"] == [] and offer is None
    assert "Canvas" not in sheet, "no token: Canvas is not mentioned at bedtime"
    assert "Weather tomorrow: weather unreachable" in sheet
    assert "Calendar: calendar unreachable" in sheet
    assert "To-dos: none open" in sheet
    assert ("canvas_due", {"days": 1}) not in reg.calls
    # switched-off sections vanish from both the card and the sheet
    c = cfg(sections={"weather": False, "todos": False})
    sections, sheet, _o = build_preview(c, Reg(), services(Reg(), Notes(), TK()), NOW)
    assert sections["weather"] == "" and sections["todos"] == []
    assert "Weather" not in sheet and "To-dos" not in sheet


def test_get_briefing_when_tomorrow_parks_the_offer(tmp_path):
    reg = Reg()
    svc = services(reg, Notes(), TK(), tmp_path=tmp_path)
    (spec,) = br.make_tools(cfg(), svc)
    r = spec.handler(when="tonight")                  # loose value coerced
    assert r.ok and r.text.startswith("Preview for tomorrow")
    assert r.max_sentences == 5 and r.speak is None
    assert r.card["offer"].startswith("Shall I wake you")
    assert svc.alarm_offer["time"].endswith(" am")
    # briefer: the allowance halves
    (spec,) = br.make_tools(cfg(verbosity="brief"), svc)
    assert spec.handler(when="tomorrow").max_sentences == 2
    # no offer -> nothing parked (a stale offer must not linger)
    tk = TK([Item("alarm", (datetime.now().astimezone() + timedelta(days=1)).replace(
        hour=6, minute=0).timestamp())])
    svc = services(reg, Notes(), tk, tmp_path=tmp_path)
    (spec,) = br.make_tools(cfg(), svc)
    spec.handler(when="tomorrow")
    assert svc.alarm_offer is None
    # nothing reachable at all
    reg = Reg(calendar_ok=False, weather_ok=False, canvas=False)
    svc = services(reg, Notes([]), TK(), tmp_path=tmp_path)
    (spec,) = br.make_tools(cfg(), svc)
    r = spec.handler(when="tomorrow")
    assert not r.ok and r.speak == br.PREVIEW_NOTHING_LINE


def test_coerce_when():
    assert coerce_when("") == "today" and coerce_when(None) == "today"
    assert coerce_when("tomorrow") == "tomorrow" and coerce_when("Tonight") == "tomorrow"
    assert coerce_when("this week") == "week" and coerce_when("next 7 days") == "week"
    assert coerce_when("morning") == "today"


# =============================================================== week
def test_build_week_buckets_by_day_and_names_heavy_and_clear():
    reg = Reg()
    tue = NOW.date() + timedelta(days=2)
    tk = TK([Item("reminder", _at(tue, 8).timestamp(), "call the dentist"),
             Item("reminder", (NOW + timedelta(days=30)).timestamp(), "far away"),
             Item("alarm", _at(tue, 7).timestamp(), "wake up")])
    sections, sheet = build_week(cfg(), reg, services(reg, Notes(["buy milk"]), tk), NOW)
    days = sections["days"]
    assert [d["label"] for d in days] == ["Today", "Tomorrow", "Tuesday", "Wednesday",
                                          "Thursday", "Friday", "Saturday"]
    assert days[0]["items"] == ["6:00 pm dinner with Sam"]
    assert days[2]["items"] == ["9:00 am Biosensors for an hour at ZACH 350",
                                "11:00 am Lecture", "2:00 pm Lab",
                                "due: CSCE 421 - Lab report, Tue 11:59 pm",
                                "8:00 am reminder: call the dentist"]
    assert days[4]["items"] == ["all day: Holiday"]
    assert days[5]["items"] == ["due: MATH 308 - Quiz 2, Fri 11:59 pm"]
    assert days[1]["items"] == [] and days[3]["items"] == [] and days[6]["items"] == []
    assert sections["summary"] == "Heaviest: Tuesday (5); Clear: tomorrow, Wednesday, Saturday."
    assert sections["todos"] == ["buy milk"] and sections["unavailable"] == []
    assert sheet.splitlines()[:3] == [
        "Week ahead from Sunday 30 August: 8 items over 7 days.",
        "Heaviest: Tuesday (5); Clear: tomorrow, Wednesday, Saturday.",
        "Today: 6:00 pm dinner with Sam"]
    assert "Wednesday: clear" in sheet and "To-dos (open): buy milk" in sheet
    assert ("get_calendar", {"range": "week"}) in reg.calls
    assert ("canvas_due", {"days": 7}) in reg.calls
    assert not any(c[0] == "get_weather" for c in reg.calls), "no weather in a forecast"


def test_build_week_structured_calendar_and_light_week():
    reg = Reg(canvas=False)
    wed = NOW.date() + timedelta(days=3)
    events = [Event(start=_at(wed, 10), end=_at(wed, 11), title="Office hours",
                    location="ETB 2005")]
    cal = SimpleNamespace(configured=True, events=lambda: events)
    c = cfg()
    c["canvas"]["token"] = ""
    sections, sheet = build_week(c, reg, services(reg, Notes([]), TK(), cal), NOW)
    assert sections["days"][3]["items"] == ["10:00 am Office hours at ETB 2005"]
    assert sections["summary"].startswith("A light week")
    assert "Clear: today, tomorrow, Tuesday, Thursday, Friday, Saturday." in sections["summary"]
    assert ("get_calendar", {"range": "week"}) not in reg.calls
    # nothing at all
    cal = SimpleNamespace(configured=True, events=lambda: [])
    sections, sheet = build_week(c, reg, services(reg, Notes([]), TK(), cal), NOW)
    assert sections["summary"] == "Nothing on the week at all."
    assert sheet.startswith("Week ahead from Sunday 30 August: 0 items over 7 days.")


def test_get_briefing_when_week_card_cap_and_unreachable(tmp_path):
    reg = Reg()
    svc = services(reg, Notes(), TK(), tmp_path=tmp_path)
    (spec,) = br.make_tools(cfg(), svc)
    r = spec.handler(when="week")
    assert r.ok and r.max_sentences == 8 and len(r.card["days"]) == 7
    assert r.text.startswith("Week ahead from ")
    (spec,) = br.make_tools(cfg(verbosity="brief"), svc)
    assert spec.handler(when="this week").max_sentences == 4
    reg = Reg(calendar_ok=False, canvas=False)
    svc = services(reg, Notes([]), TK(), tmp_path=tmp_path)
    (spec,) = br.make_tools(cfg(), svc)
    r = spec.handler(when="week")
    assert not r.ok and r.speak == br.WEEK_NOTHING_LINE


def test_canvas_and_week_text_parsers():
    today = NOW.date()
    items = br._canvas_items(" ".join(CANVAS_7.split()), today)
    assert items == [(today + timedelta(days=2), "CSCE 421 - Lab report, Tue 11:59 pm"),
                     (today + timedelta(days=5), "MATH 308 - Quiz 2, Fri 11:59 pm")]
    # a dated form beyond the week, and a December ask about January
    (day, _t), = br._canvas_items("1) X - Y, Tue 15 Sep 11:59 pm", today)
    assert day == datetime(2026, 9, 15).date()
    (day, _t), = br._canvas_items("1) X - Y, Tue 5 Jan 11:59 pm", datetime(2026, 12, 28).date())
    assert day == datetime(2027, 1, 5).date()
    assert br._canvas_items("nothing due in the next 7 days", today) == []
    week = br._week_text_items(CAL_WEEK + " That's as of 9:10 am.", today)
    assert week[0] == (today, "6:00 pm dinner with Sam")
    assert week[-1] == (today + timedelta(days=4), "all day: Holiday")
    assert len(week) == 5


# ================================================== sections + verbosity
def test_no_news_in_the_morning_drops_the_section(tmp_path):
    calls = []

    def fetch(url, timeout=None, headers=None):
        calls.append(url)
        raise OSError("no route")
    reg = Reg()
    c = cfg(news_feeds=[br.DEFAULT_NEWS_FEEDS[0]], sections={"news": False})
    sections, sheet = build_briefing(c, reg, fetch, NOW, tmp_path / "news.json")
    assert calls == [], "news switched off: nothing was fetched"
    assert sections["news"] == [] and "News" not in sheet
    assert sheet.splitlines()[1].startswith("Weather:")
    # the other switches: no weather call, no calendar line
    reg = Reg()
    c = cfg(sections={"weather": False, "calendar": False, "news": False, "canvas": False})
    sections, sheet = build_briefing(c, reg, fetch, NOW, tmp_path / "news.json")
    assert reg.calls == [] and sheet == "Briefing for Sunday 30 August, 9:30 pm."
    # a stock section switched off never fetches even with symbols set
    c = cfg(stock_symbols=["NVDA"], sections={"stocks": False, "news": False})
    _s, sheet = build_briefing(c, Reg(), fetch, NOW, tmp_path / "news.json")
    assert calls == [] and "Stocks" not in sheet


def test_section_on_and_verbosity_cap_defaults():
    assert section_on(None, "news") is True
    assert section_on({"briefing": {"sections": {"news": "off"}}}, "news") is False
    assert section_on({"briefing": {"sections": {"news": "on"}}}, "news") is True
    assert section_on({"briefing": {"sections": {}}}, "nonsense") is True
    assert verbosity_cap(None, 6) == 6
    assert verbosity_cap({"briefing": {"verbosity": "brief"}}, 6) == 3
    assert verbosity_cap({"briefing": {"verbosity": "brief"}}, 3) == 2
    assert verbosity_cap({"briefing": {"verbosity": "normal"}}, 8) == 8


def test_real_config_carries_the_switches(tmp_path):
    from jarvis.assistant_config import DEFAULTS, AssistantConfig
    b = DEFAULTS["briefing"]
    assert b["sections"] == {k: True for k in br.SECTIONS}
    assert b["verbosity"] == "normal" and b["wake_offer"] is True
    assert b["early_before"] == "09:00" and b["wake_lead_min"] == 60
    c = AssistantConfig.load(tmp_path / "assistant.json")
    assert section_on(c, "news") is True
    c.set("briefing.sections.news", False)
    assert section_on(c, "news") is False
    assert section_on(AssistantConfig.load(tmp_path / "assistant.json"), "news") is False
    c.set("briefing.verbosity", "brief")
    assert verbosity_cap(c, 6) == 3


# ============================================================= the card
def test_briefing_rows_render_the_preview_and_week_keys():
    rows = briefing_rows({"weather": "Tomorrow: sunny.", "calendar": [CAL_TOMORROW],
                          "canvas": ["CSCE 421 - Lab 3, tomorrow 11:59 pm"],
                          "todos": ["buy milk", "email the professor"],
                          "alarm": "none set", "offer": "Shall I wake you at 7:00 am, sir?"})
    assert rows == [("WEATHER", "Tomorrow: sunny."), ("CALENDAR", CAL_TOMORROW),
                    ("CANVAS", "CSCE 421 - Lab 3, tomorrow 11:59 pm"),
                    ("TO-DOS", "buy milk"), ("", "email the professor"),
                    ("ALARM", "none set"), ("OFFER", "Shall I wake you at 7:00 am, sir?")]
    rows = briefing_rows({"summary": "Heaviest: Tuesday (5); Clear: Wednesday.",
                          "days": [{"label": "Today", "items": ["6:00 pm dinner"]},
                                   {"label": "Tuesday", "items": ["9:00 am Lab", "2:00 pm Lab"]},
                                   {"label": "Wednesday", "items": []}],
                          "todos": [], "unavailable": ["canvas"], "reminders": ["8:00 am x"]})
    assert rows == [("WEEK", "Heaviest: Tuesday (5); Clear: Wednesday."),
                    ("TODAY", "6:00 pm dinner"),
                    ("TUESDAY", "9:00 am Lab"), ("", "2:00 pm Lab"),
                    ("WEDNESDAY", "clear"), ("REMINDERS", "8:00 am x")]
    assert briefing_rows({"days": ["not a dict"], "offer": ""}) == []


# ========================================================== the commander
class Brain:
    def __init__(self):
        self.chats = []

    def chat(self, text, force_tool=None, force_args=None):
        self.chats.append((text, force_tool, force_args))


class Cfg:
    def __init__(self, **data):
        self.data = dict(data)
        self.sets = []

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.sets.append((key, value))
        self.data[key] = value
        return True


class Memory:
    def __init__(self):
        self.prefs = {}

    def set_preference(self, key, value):
        self.prefs[key] = value

    def log_habit(self, text):
        pass


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    svc = types.SimpleNamespace(brain=Brain(), assistant=Cfg(), memory=Memory(),
                                timekeeper=TK(), tts=None, router=None)
    return Commander(svc)


def test_good_night_previews_tomorrow_when_briefings_are_on(cmdr):
    svc = cmdr.services
    # off (the default): a good night is a good night
    res = cmdr.handle("jarvis good night")
    assert res.reply in COURTESY_REPLIES["goodnight"] and svc.brain.chats == []
    res = cmdr.handle("good night", source="typed")
    assert res.reply in COURTESY_REPLIES["goodnight"] and svc.brain.chats == []
    # on: the night line now, the preview through the tool, turn left open
    svc.assistant.data["briefing.enabled"] = True
    for text in ("jarvis good night", "Good night, Jarvis.", "goodnight"):
        res = cmdr.handle(text, source="typed")
        assert res.handled and res.speak and res.ack and res.done is False
        assert res.reply == GOODNIGHT_PREVIEW_LINE and res.status == "Preview…"
    assert svc.brain.chats == [(PREVIEW_ASK, "get_briefing", {"when": "tomorrow"})] * 3
    # no brain: still a good night, never an error
    svc.brain = None
    res = cmdr.handle("good night", source="typed")
    assert res.reply in COURTESY_REPLIES["goodnight"]


def test_explicit_preview_and_week_asks_ignore_the_toggle(cmdr):
    svc = cmdr.services
    for text in ("what does tomorrow look like", "how's tomorrow looking?",
                 "preview tomorrow", "tomorrow's briefing", "the evening preview"):
        res = cmdr.handle(text, source="typed")
        assert res.handled and res.done is False and res.status == "Preview…", text
        assert svc.brain.chats[-1] == (text.lower().rstrip("?"), "get_briefing",
                                       {"when": "tomorrow"})
    for text in ("how's my week looking?", "how is my week looking", "how does my week look",
                 "what's the week ahead", "weekly forecast", "how busy is my week",
                 "brief me on my week", "what's my week looking like, jarvis"):
        res = cmdr.handle(text, source="typed")
        assert res.handled and res.status == "Week ahead…", text
        assert svc.brain.chats[-1][1:] == ("get_briefing", {"when": "week"})
    # calendar questions stay calendar questions (the docs' phrasing)
    for text in ("what does my week look like", "what's on this week", "what's on tomorrow",
                 "what's the weather tomorrow"):
        assert commander._WEEK_RX.match(text) is None, text
        assert commander._PREVIEW_RX.match(text) is None, text


def test_section_preferences_write_the_config_and_the_memory(cmdr):
    svc = cmdr.services
    res = cmdr.handle("no news in the morning", source="typed")
    assert res.handled and res.speak and "no news" in res.reply
    assert svc.assistant.sets == [("briefing.sections.news", False)]
    assert svc.memory.prefs == {"briefing.sections.news": False}
    cases = {
        "I don't want the weather in my morning briefing": ("weather", False),
        "skip the sports in my briefing": ("sports", False),
        "put the news back in the briefing": ("news", True),
        "don't read the stocks in the daily briefing": ("stocks", False),
        "leave out canvas in the evening preview": ("canvas", False),
        "include the to-dos in my briefing again": ("todos", True),
        "drop the reminders from my weekly forecast, please": ("reminders", False),
        "no study in the morning": ("study", False),
        "put the flashcards back in my briefing": ("study", True),
    }
    for text, (name, on) in cases.items():
        res = cmdr.handle(text, source="typed")
        assert res.handled and res.speak, text
        assert svc.assistant.sets[-1] == (f"briefing.sections.{name}", on), text
        assert svc.memory.prefs[f"briefing.sections.{name}"] is on
    assert "to-dos" in cmdr.handle("include the to-dos in my briefing", source="typed").reply
    # an unknown section is not a preference at all
    assert commander._PREF_SECTION_RX.match("no jokes in the morning") is None
    # verbosity
    for text, mode, line in (("be briefer", "brief", commander.BRIEFER_LINE),
                             ("shorter briefings please", "brief", commander.BRIEFER_LINE),
                             ("the full briefing", "normal", commander.FULL_LENGTH_LINE),
                             ("more detail, jarvis", "normal", commander.FULL_LENGTH_LINE)):
        res = cmdr.handle(text, source="typed")
        assert res.reply == line and res.speak, text
        assert svc.assistant.sets[-1] == ("briefing.verbosity", mode)
        assert svc.memory.prefs["briefing.verbosity"] == mode
    assert svc.brain.chats == [], "preferences never cost a model turn"
    # without an assistant config there is no store the briefing reads, so
    # the phrase is not a preference at all (it falls through to the router)
    svc.assistant = None
    res = cmdr.handle("no news in the morning", source="typed")
    assert res.status != "Briefing: news off"
    assert svc.memory.prefs["briefing.sections.news"] is True, "unchanged"


def test_a_yes_to_the_wake_offer_sets_the_alarm(cmdr):
    svc = cmdr.services
    due = _at(TOMORROW, 7).timestamp()

    def offer(age=0.0):
        return {"due": due, "time": "7:00 am", "label": "wake up",
                "line": "Shall I wake you at 7:00 am, sir?", "made_at": time.time() - age}
    svc.alarm_offer = offer()
    res = cmdr.handle("yes please")
    assert res.reply == "Alarm at 7:00 am, sir." and res.speak
    assert svc.timekeeper.alarms == [(due, "wake up", "once")]
    assert svc.alarm_offer is None
    # no: dropped politely
    svc.alarm_offer = offer()
    res = cmdr.handle("no thanks")
    assert res.reply == NO_ALARM_LINE and len(svc.timekeeper.alarms) == 1
    # a new subject drops the offer and is handled as itself
    svc.alarm_offer = offer()
    res = cmdr.handle("jarvis what time is it")
    assert svc.alarm_offer is None and res.status != "Alarm 7:00 am"
    assert len(svc.timekeeper.alarms) == 1
    # an expired offer never turns a stray yes into an alarm
    svc.alarm_offer = offer(age=br.OFFER_TTL_S + 5)
    res = cmdr.handle("yes", source="typed")
    assert len(svc.timekeeper.alarms) == 1 and svc.alarm_offer is None
    # the offer is a yes/no BEFORE the voice intent gate: a bare spoken "yes"
    # must not be dropped as background chat
    svc.alarm_offer = offer()
    res = cmdr.handle("yes")
    assert res.reply == "Alarm at 7:00 am, sir." and len(svc.timekeeper.alarms) == 2
    # no timekeeper: the excuse, not a crash
    svc.timekeeper = None
    svc.alarm_offer = offer()
    assert cmdr.handle("yes").reply == commander.TIMEKEEPER_SETUP_LINE


# ================================================================ the app
def _app():
    a = object.__new__(app_mod.JarvisApp)
    a._turn_busy = threading.Event()
    a._turn_timer = a._turn_watchdog = None
    a._last_source, a._last_user_text = "voice", "good night"
    a._followup_after_speech = False
    a.said = []
    a._say = a.said.append
    a.context = SimpleNamespace(add_exchange=lambda u, j: None)
    return a


def test_the_app_asks_the_wake_offer_after_the_spoken_preview():
    a = _app()
    cards = []
    bus.subscribe(BriefingReady, cards.append)
    try:
        card = {"weather": "sunny", "calendar": [CAL_TOMORROW], "alarm": "none set",
                "offer": "Shall I wake you at 7:00 am, sir?"}
        a._on_brain_tags([("BRIEFING", json.dumps(card)),
                          ("SPEAK", "Biosensors at eight, sir; no alarm set.")])
        assert a.said == ["Biosensors at eight, sir; no alarm set.",
                          "Shall I wake you at 7:00 am, sir?"]
        assert a._followup_after_speech, "the answer window opens for the yes/no"
        assert len(cards) == 1 and cards[0].sections["offer"].startswith("Shall I")
        # a card without an offer asks nothing
        a.said.clear()
        a._followup_after_speech = False
        a._last_source = "typed"
        a._on_brain_tags([("BRIEFING", json.dumps({"weather": "sunny"})),
                          ("SPEAK", "Sunny, sir.")])
        assert a.said == ["Sunny, sir."] and not a._followup_after_speech
    finally:
        bus.unsubscribe(BriefingReady, cards.append)


# ---------------------------------------------- Canvas without a token
# The preview and the week forecast used to gate on `canvas.token` alone,
# so on a box whose university blocks personal tokens (his does) they went
# dark while the coursework sat in a calendar he had already subscribed to.
# The gate is about a SOURCE now (briefing._canvas_source).
LAB1 = ("Lab 1:  Introduction to the AD2 SDK [BMEN-427:501,502,503,504,"
        "BMEN-627:600,601,602,603]")


def _coursework_cal(*titles):
    events = [Event(start=_at(TOMORROW, 0), end=_at(TOMORROW + timedelta(days=1), 0),
                    all_day=True, title=t) for t in titles]
    return SimpleNamespace(configured=True, events=lambda: events)


def test_canvas_source_is_the_token_or_a_coursework_feed():
    with_token = cfg()
    assert br._canvas_source(with_token, services()) is True
    no_token = cfg()
    no_token["canvas"]["token"] = ""
    assert br._canvas_source(no_token, services()) is False
    assert br._canvas_source(no_token, services(calendar=_coursework_cal("Chiro"))) is False
    assert br._canvas_source(no_token, services(calendar=_coursework_cal(LAB1))) is True


def test_preview_and_week_read_canvas_from_the_feed_without_a_token():
    c = cfg()
    c["canvas"]["token"] = ""
    reg = Reg()
    cal = _coursework_cal(LAB1)
    sections, sheet, _offer = build_preview(c, reg, services(reg, Notes(), TK(), cal), NOW)
    assert sections["canvas"] == ["CSCE 421 - Lab 3, tomorrow 11:59 pm"]
    assert "Canvas due tomorrow: CSCE 421 - Lab 3, tomorrow 11:59 pm" in sheet
    assert ("canvas_due", {"days": 1}) in reg.calls
    reg = Reg()
    sections, sheet = build_week(c, reg, services(reg, Notes([]), TK(), cal), NOW)
    assert any("due: CSCE 421 - Lab report" in i
               for d in sections["days"] for i in d["items"])
    assert ("canvas_due", {"days": 7}) in reg.calls
