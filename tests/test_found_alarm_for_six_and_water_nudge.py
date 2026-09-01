""""Set an alarm for six" became six MINUTES, and a water nudge died on
"I couldn't make out the time".

Two features from Hunter's 155-feature voice session, 2026-08-31. Both were
heard perfectly -- avg_logprob -0.63 and -0.31, no mis-hear anywhere -- and
both still went wrong inside the timekeeper.

#17 alarm (his note, verbatim):

    "Set an alarm for six (from me), I would advise against a 8:49 pm alarm,
     sir; you're MAGNETIC RESONANCE ENGR until 8:50 pm then. Shall I set it
     anyway?, Cancel (from me) setting an alarm for 6 mins from now anyway
     sir. Nothing's running, sir."

/tmp/vss_voice/jarvis.log has it end to end:

    20:43:25 Transcribed: 'Set an alarm for six' (avg_logprob=-0.63)
    20:43:26 objection quiet conflict: ... (row: quiet MAGNETIC RESONANCE ...)
    20:43:26 speaking: I would advise against a 8:49 pm alarm, sir; ...
    20:43:39 Transcribed: 'Cancel'
    20:43:39 timekeeper: alarm '' due 2026-08-31 20:49 (once)
    20:43:39 speaking: Setting it anyway, sir. Alarm in 6 minutes, sir.

8:49 pm is six MINUTES after 8:43. The commander strips the sentence to the
single word "six" and hands it to parse_when; a lone number went down the
bare-duration path and came out as minutes. "6" would have been safe --
commander._norm_when prefixes "at " to a `when` that starts with a DIGIT --
so the bug only bites the spelled-out hour a person actually says.

The SECOND fault in that turn ("Cancel did not cancel") is not in this file
and not in the timekeeper: commander.parse_yes_no("Cancel") returns None,
not False, so _try_objection_confirm fell through to its "anything else ->
run it anyway" arm. "no", "never mind" and "forget it" all resolve to False;
"cancel" is a one-word hole in that vocabulary.

#22 interval nudges (his note): the transcript was right --

    20:47:22 Transcribed: 'Set a reminder to drink water every 45 minutes'
             (avg_logprob=-0.31)
    20:47:24 tool set_reminder -> ok=False could not understand the time 'now'
    20:47:24 chat reply: I couldn't make out the time, sir; try 'in ten
             minutes' or 'at seven'.

-- and the parser was right too: parse_when_full reads "every 45 minutes"
perfectly. The model simply split the sentence across the arguments, and the
tool only ever looked at one of them (~/.aiws_trainer/jarvis_memory/journal/
2026-08-31.jsonl):

    {"name": "set_reminder", "args": {"repeat": "every 45 minutes",
     "text": "drink water", "when": "now"}, "ok": false,
     "text": "could not understand the time 'now'"}
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from jarvis.tools import timekeeper as tkm
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.timekeeper import (
    CANT_PARSE_LINE, Timekeeper, make_tools, parse_when,
)

# 8:43 pm on the Monday, the minute he said it.
EVENING = datetime(2026, 8, 31, 20, 43)


@pytest.fixture
def tk(tmp_path):
    t = Timekeeper(tmp_path / "tk.db", say=lambda *a, **k: None,
                   cfg=SimpleNamespace(get=lambda *a, **k: None),
                   now=lambda: EVENING.timestamp(), run=lambda *a, **k: None,
                   tick_s=0.01, ring=False, cache_dir=tmp_path / "cache")
    yield t
    t.close()


@pytest.fixture
def tools(tk):
    reg = ToolRegistry()
    reg.register_many(make_tools(tk.cfg, SimpleNamespace(timekeeper=tk)))
    return reg


# ------------------------------------------------------ #17 "for six"
def test_an_alarm_for_six_is_six_oclock_not_six_minutes():
    """The exact turn: at 8:43 pm, "six" must not mean 8:49 pm."""
    got = parse_when("six", EVENING, prefer="morning")
    assert got == datetime(2026, 9, 1, 6, 0), (
        f"'Set an alarm for six' resolved to {got} -- 20:49 is the six-MINUTE "
        "reading that set the alarm he cancelled")


def test_a_lone_digit_reads_the_same_as_the_lone_word():
    """"6" only escaped this because commander._norm_when prefixes "at " to
    a digit. The parser must not need that crutch to agree with itself."""
    assert parse_when("6", EVENING) == parse_when("six", EVENING)
    assert parse_when("6", EVENING) == parse_when("at 6", EVENING)


@pytest.mark.parametrize("text,expected", [
    ("seven", datetime(2026, 9, 1, 7, 0)),
    ("12", datetime(2026, 9, 1, 12, 0)),        # bare 12 is noon, still
    ("18", datetime(2026, 9, 1, 18, 0)),        # 24-hour readings survive
])
def test_other_lone_hours(text, expected):
    assert parse_when(text, EVENING, prefer="morning") == expected


@pytest.mark.parametrize("text,expected", [
    ("in six", datetime(2026, 8, 31, 20, 49)),      # "in" still means minutes
    ("in 5", datetime(2026, 8, 31, 20, 48)),
    ("six minutes", datetime(2026, 8, 31, 20, 49)),  # so does a unit
    ("10 minutes", datetime(2026, 8, 31, 20, 53)),
    ("45", datetime(2026, 8, 31, 21, 28)),          # cannot BE an hour
    ("1.5", datetime(2026, 8, 31, 20, 44, 30)),
])
def test_the_minute_readings_are_untouched(text, expected):
    """Only a lone 0-24 changed meaning. Everything that was already a
    duration -- the "in", the unit, the number no clock could show -- stays
    one, or "How long for, sir?" -> "five" stops working."""
    assert parse_when(text, EVENING) == expected


def test_the_alarm_tool_sets_six_oclock(tools, tk):
    r = tools.call("set_alarm", {"when": "six"})
    assert r.ok
    due = datetime.fromtimestamp(tk.list("alarm")[0].due)
    assert due == datetime(2026, 9, 1, 6, 0)
    assert "6:00 am" in r.text


# ------------------------------------------- #22 the interval in `repeat`
def test_the_interval_is_read_out_of_repeat_when_when_is_useless(tools, tk):
    """His exact turn, with the exact arguments the model sent."""
    r = tools.call("set_reminder", {"repeat": "every 45 minutes",
                                    "text": "drink water", "when": "now"})
    assert r.ok, f"still {r.text!r}"
    assert r.speak != CANT_PARSE_LINE
    item = tk.list("reminder")[0]
    assert item.label == "drink water"
    assert item.repeat == "45m"
    # The first nudge is one interval out, not this instant: a "drink water
    # every 45 minutes" that fires while he is still talking is a bug report.
    assert datetime.fromtimestamp(item.due) == EVENING + timedelta(minutes=45)
    assert ", every 45 minutes" in r.text


def test_the_label_does_not_become_the_word_now(tools, tk):
    """With no `text` to fall back on, the leftover from parsing "now" must
    not be filed as the thing to be reminded of."""
    r = tools.call("set_reminder", {"repeat": "every 30 minutes", "when": "now"})
    assert r.ok and tk.list("reminder")[0].label != "now"


def test_an_alarm_can_split_the_same_way(tools, tk):
    """set_alarm's schema advertises "an interval like 'every 90 minutes'"
    to the same model, so it can make the same split."""
    r = tools.call("set_alarm", {"when": "now", "repeat": "every 90 minutes"})
    assert r.ok and tk.list("alarm")[0].repeat == "90m"


def test_an_unparseable_time_with_no_interval_still_apologises(tools):
    """The fallback must not turn every bad time into a reminder."""
    r = tools.call("set_reminder", {"when": "whenever", "text": "x"})
    assert not r.ok and r.speak == CANT_PARSE_LINE
    r = tools.call("set_reminder", {"when": "now", "text": "x", "repeat": "once"})
    assert not r.ok and r.speak == CANT_PARSE_LINE


def test_a_mis_heard_interval_is_still_refused(tools):
    """"every 2 seconds" is a mis-hear, and the bounds that keep it out of
    the store must not be bypassed by the new `repeat` route."""
    r = tools.call("set_reminder", {"when": "now", "text": "x",
                                    "repeat": "every 2 seconds"})
    assert not r.ok and r.speak == CANT_PARSE_LINE


def test_firewall():
    from jarvis import logs
    assert str(logs.LOG_DIR) != "/tmp/vss_voice"
    assert tkm.MIN_INTERVAL_MIN == 1 and tkm.MAX_INTERVAL_MIN == 24 * 60
