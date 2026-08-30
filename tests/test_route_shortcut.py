"""Route short-cuts (2026-08-30): when the router's reason names the tool
and the utterance carries its arguments, the commander forces the call
(brain.chat force_tool/force_args) so only the render turn runs.

Live, "What's on my calendar tomorrow?" spent 0.98 s letting the model
choose get_calendar and 0.48 s rendering; "What's the time in London?"
1.45 s + 0.73 s. The first chunk is what these tests remove -- and what
they must NOT remove: a write, a second clause, a lowercase city.

The router is the real one; the brain is a mock. No Ollama, no network.
"""
import types
from unittest.mock import MagicMock

import pytest

from jarvis.commander import (Commander, IntentClassifier, calendar_range, forced_call,
                              place_in, weather_when)
from jarvis.config import CONFIG
from jarvis.router import Router


class Cfg:
    def __init__(self, **over):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def rich(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    # raising=False: FEEDBACK_LOG arrives with the voice-feedback feature,
    # and this file must run on the commits before it too.
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "feedback.jsonl",
                        raising=False)
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=MagicMock(), notes=MagicMock(), approvals=MagicMock(),
        claude=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    return Commander(svc), svc


def _only_chat(svc):
    assert svc.brain.chat.call_count == 1, svc.brain.chat.call_args_list
    return svc.brain.chat.call_args


# ------------------------------------------------------------ forced
@pytest.mark.parametrize("said,tool,args", [
    ("what's on my calendar tomorrow?", "get_calendar", {"range": "tomorrow"}),
    ("What's on my schedule today?", "get_calendar", {"range": "today"}),
    ("when's my next meeting?", "get_calendar", {"range": "next"}),
    ("do I have any meetings on Monday?", "get_calendar", {"range": "monday"}),
    ("what's the weather tomorrow?", "get_weather", {"when": "tomorrow", "location": ""}),
    ("what's the weather like in London?", "get_weather", {"when": "now", "location": "London"}),
    ("will it rain this week?", "get_weather", {"when": "week", "location": ""}),
    ("what's the time in London?", "get_time", {"location": "London"}),
    ("what time is it in New York right now?", "get_time", {"location": "New York"}),
])
def test_the_named_tool_is_forced_and_only_the_render_turn_runs(rich, said, tool, args):
    c, svc = rich
    res = c.handle(said, source="typed")
    assert res.handled and res.done is False
    call = _only_chat(svc)
    assert call.args == (said,)
    assert call.kwargs == {"force_tool": tool, "force_args": args}


def test_the_forced_call_never_carries_the_address(rich):
    """The model sees "what's on my calendar tomorrow?", not "Jarvis, ..."
    -- the address strip that the plain path already does."""
    c, svc = rich
    c.handle("Jarvis, what's on my calendar tomorrow?", source="typed")
    call = _only_chat(svc)
    assert call.args == ("what's on my calendar tomorrow?",)
    assert call.kwargs["force_tool"] == "get_calendar"


def test_by_voice_the_short_cut_still_fires(rich, monkeypatch):
    c, svc = rich
    monkeypatch.setattr(c.intent, "classify", lambda text: (IntentClassifier.YES, 0.9))
    c.handle("What's on my calendar tomorrow?", source="voice")
    assert _only_chat(svc).kwargs == {"force_tool": "get_calendar",
                                      "force_args": {"range": "tomorrow"}}


# ----------------------------------------------------- left to the loop
@pytest.mark.parametrize("said", [
    "book a meeting on Monday",                           # a write
    "schedule a meeting with Bob tomorrow at 3",          # a write
    "can you add a meeting tomorrow at noon",             # a write, led by "can you"
    "what's on my calendar tomorrow and what's the weather",   # two intents
    "what's the weather in london",                       # lowercase city: not trusted
    "what year is it",                                    # local:clock with no city
])
def test_a_write_a_second_clause_or_an_unsure_city_is_not_forced(rich, said):
    c, svc = rich
    c.handle(said, source="typed")
    for call in svc.brain.chat.call_args_list:
        assert "force_tool" not in call.kwargs, (said, call)


def test_a_write_to_the_calendar_takes_the_full_loop_not_a_read(rich):
    """The router's local:calendar cue also matches "schedule/book/put ...
    meeting": forcing get_calendar there would answer a write with a
    read, confidently."""
    c, svc = rich
    c.handle("put a dentist appointment on my calendar for Friday", source="typed")
    call = _only_chat(svc)
    assert call.kwargs == {}


# ------------------------------------------------------------ helpers
@pytest.mark.parametrize("text,place", [
    ("what's the weather in Paris tomorrow", "Paris"),
    ("what's the time in Salt Lake City?", "Salt Lake City"),
    ("what's the weather for tomorrow", ""),          # a time word, not a place
    ("what's the weather for Tuesday", ""),           # capitalised weekday
    ("is it cold at the moment", ""),
    ("what's it like at home", ""),
    ("what's the weather", ""),
    ("what's the weather in london", None),           # unsure: not forced
    ("will it rain in a while", ""),
])
def test_place_in(text, place):
    assert place_in(text) == place


@pytest.mark.parametrize("text,when", [
    ("what's the weather", "now"),
    ("is it raining", "now"),
    ("what's the weather tonight", "today"),
    ("what's the forecast", "today"),
    ("what's the weather tomorrow", "tomorrow"),
    ("what's the weather like this weekend", "week"),
    ("will it rain this week", "week"),
])
def test_weather_when(text, when):
    assert weather_when(text) == when


@pytest.mark.parametrize("text,rng", [
    ("what's on my calendar", "today"),
    ("when's my meeting", "next"),               # a "when" with no day: the next one
    ("when's my meeting today", "today"),
    ("what's on this week", "week"),
    ("anything on my calendar tomorrow", "tomorrow"),
])
def test_calendar_range(text, rng):
    assert calendar_range(text) == rng


def test_forced_call_only_for_the_three_reasons():
    assert forced_call("local:question", "what's on my calendar tomorrow") is None
    assert forced_call("local:calendar", "") is None
    assert forced_call("answer", "what's the weather") is None
    assert forced_call("local:weather", "what's the weather") == \
        ("get_weather", {"when": "now", "location": ""})
    assert forced_call("local:clock", "what's the time in Tokyo") == \
        ("get_time", {"location": "Tokyo"})
