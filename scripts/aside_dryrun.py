#!/usr/bin/env python
"""Offline dry run for jarvis/aside.py -- READ IT AS A TRANSCRIPT.

    ~/vss_env/bin/python scripts/aside_dryrun.py

Twenty turns across two days, in the shape the real turn ledger records
(/tmp/vss_voice/turns.jsonl has timings only, so the words are drawn from
the `handle '...'` lines of the live log and from the phrasings the alarm
and timer matchers actually accept). Each turn carries the STRUCTURED
action result its handler would have produced -- a timekeeper Item with a
real epoch, or None where nothing resolved -- because that is the only
input `consider()` is allowed to anchor on.

Nothing here touches the network, the live app or the user's state: the
calendar and the deadline snapshot are fakes, the state file is a temp
file, and JARVIS_LOG_DIR is redirected before jarvis is imported.

What to look for, in order:
  * every SAID line must read like something a person would be glad to
    hear once. If more than a couple make you wince, the templates are
    wrong and no amount of ledger correctness saves it;
  * the budget must feel like restraint, not like a broken feature -- two
    a day, forty-five minutes apart, and the QUIET lines dropped rather
    than saved up;
  * "no more asides" must end the day.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="aside-dryrun-"))
os.environ["JARVIS_LOG_DIR"] = str(_TMP)
os.environ["JARVIS_ASSISTANT_CONFIG"] = str(_TMP / "assistant.json")
os.environ["JARVIS_MEMORY_DIR"] = str(_TMP / "memory")
os.environ["JARVIS_CACHE_DIR"] = str(_TMP / "cache")

from jarvis.aside import AsideEngine, kill_phrase          # noqa: E402
from jarvis.tools.calendar import Event                    # noqa: E402

TZ = ZoneInfo("America/Chicago")
DAY1 = datetime(2026, 9, 14, tzinfo=TZ)                    # a Monday
DAY2 = DAY1 + timedelta(days=1)


def at(day: datetime, hh: int, mm: int = 0) -> datetime:
    return day.replace(hour=hh, minute=mm)


# --------------------------------------------------------------- sources
# The calendar cache, as CalendarSource.events() would hand it over.
EVENTS = [
    Event(start=at(DAY1, 9, 10), end=at(DAY1, 10, 30), title="BIOSENSORS lecture",
          calendar="school"),
    Event(start=at(DAY1, 14, 0), end=at(DAY1, 15, 0), title="advisor meeting",
          calendar="school"),
    Event(start=at(DAY2, 9, 10), end=at(DAY2, 10, 30), title="BIOSENSORS lecture",
          calendar="school"),
    Event(start=at(DAY2, 13, 0), end=at(DAY2, 15, 0), title="Midterm 1",
          calendar="school"),
    Event(start=at(DAY1, 0, 0), end=at(DAY2, 0, 0), title="Rosh Hashanah",
          all_day=True, calendar="holidays"),
]

# The snapshot DeadlineHeadsUp stashes on its own thread: Canvas planner rows.
ITEMS = [
    {"course": "BIOSENSORS", "title": "Lab 3 report", "due": at(DAY1, 23, 59)},
    {"course": "BIOSENSORS", "title": "Reading quiz 4", "due": at(DAY2, 23, 59)},
    {"course": "THERMO", "title": "Problem set 6", "due": at(DAY2, 17, 0)},
]


class Cal:
    configured = True

    def events(self):
        return list(EVENTS)


class Deadlines:
    def snapshot(self):
        return list(ITEMS)


class Quiet:
    """Stands in for jarvis/quiet.py: 23:00-07:00 plus the lecture block."""

    def __init__(self):
        self.now = DAY1

    def reason(self) -> str:
        if self.now.hour >= 23 or self.now.hour < 7:
            return "quiet hours until 7:00 am"
        for ev in EVENTS:
            if not ev.all_day and ev.start <= self.now < ev.end \
                    and "lecture" in ev.title.lower():
                return f"{ev.title} until {ev.end:%H:%M}"
        return ""

    def should_hold(self) -> bool:
        return bool(self.reason())


class Cfg(dict):
    def get(self, key, default=None):          # dotted, like AssistantConfig
        node = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def item(kind: str, due: datetime, label: str = ""):
    """What the timekeeper hands back -- the anchor `consider()` needs."""
    return SimpleNamespace(id="x", kind=kind, label=label,
                           due=due.timestamp(), created=0.0, repeat="once")


# ----------------------------------------------------------------- turns
# (clock, utterance, action result). Deliberately a mix: alarms and timers
# that resolve, questions that resolve nothing, and one turn inside quiet
# hours -- the interesting cases are the ones he stays quiet for.
TURNS = [
    (at(DAY1, 8, 12), "what time is it", None),
    (at(DAY1, 8, 13), "set a timer for ten minutes", item("timer", at(DAY1, 8, 23))),
    (at(DAY1, 8, 20), "play some miles davis", None),
    (at(DAY1, 8, 41), "remind me to email my advisor at noon",
     item("reminder", at(DAY1, 12, 0), "email my advisor")),
    (at(DAY1, 9, 30), "what's due this week", None),
    (at(DAY1, 9, 40), "set a timer for twenty minutes", item("timer", at(DAY1, 10, 0))),
    (at(DAY1, 11, 5), "wake me at seven tomorrow",
     item("alarm", at(DAY2, 7, 0))),
    (at(DAY1, 11, 20), "set a timer for five minutes", item("timer", at(DAY1, 11, 25))),
    (at(DAY1, 13, 15), "remind me to hand in the lab report at nine tonight",
     item("reminder", at(DAY1, 21, 0), "hand in the lab report")),
    (at(DAY1, 16, 2), "drop my needle", None),
    (at(DAY1, 16, 30), "set an alarm for eight",
     item("alarm", at(DAY2, 8, 0))),
    (at(DAY1, 23, 10), "set an alarm for six thirty",
     item("alarm", at(DAY2, 6, 30))),
    (at(DAY2, 7, 5), "set a timer for forty minutes", item("timer", at(DAY2, 7, 45))),
    (at(DAY2, 7, 50), "what was my last email about", None),
    (at(DAY2, 9, 15), "set a timer for fifteen minutes", item("timer", at(DAY2, 9, 30))),
    (at(DAY2, 10, 40), "remind me to print the problem set at four",
     item("reminder", at(DAY2, 16, 0), "print the problem set")),
    (at(DAY2, 10, 45), "no more asides", None),
    (at(DAY2, 11, 30), "set an alarm for seven tomorrow",
     item("alarm", at(DAY2 + timedelta(days=1), 7, 0))),
    (at(DAY2, 12, 0), "how's my week looking", None),
    (at(DAY2, 12, 30), "set a timer for an hour", item("timer", at(DAY2, 13, 30))),
]


def main() -> None:
    clock = SimpleNamespace(now=DAY1)
    quiet = Quiet()
    cfg = Cfg({"aside": {"enabled": True, "per_day": 2, "gap_min": 45,
                         "horizon_hours": 18}})
    engine = AsideEngine(cfg=cfg, get_calendar=Cal(), get_deadlines=Deadlines(),
                         quiet=quiet, state_path=_TMP / "aside_state.json",
                         filed_paths=(_TMP / "deadlines_state.json",),
                         now=lambda: clock.now)
    said = 0
    day = None
    for when, text, action in TURNS:
        clock.now = quiet.now = when
        if day != when.date():
            day = when.date()
            print(f"\n=== {when:%A %-d %B} " + "=" * 40)
        print(f"\n{when:%H:%M}  you: {text}")
        if kill_phrase(text):
            print(f"       jarvis: {engine.silence(text)}")
            continue
        line, why = engine.explain(text, "", action)
        if line is None:
            print(f"       (quiet: {why})")
            continue
        engine.consider(text, "", action)
        said += 1
        print(f"       jarvis: ... {line}")
    print(f"\n{said} aside(s) across {len(TURNS)} turns. State: {_TMP}")


if __name__ == "__main__":
    main()
