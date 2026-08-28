"""Deciding when Jarvis may add an event without asking.

Writing is a different category from everything else Jarvis does with the
calendar: a misheard time becomes a real object on the user's phone. The
chosen policy is "add outright when the parse is unambiguous, confirm when it
is not", so the whole safety of the feature rests on this predicate being
conservative.

Ambiguity here is about the SPOKEN TEXT, not the parser's willingness to
return something. parse_when_full always produces a datetime for "at four" --
it guesses an hour from a heuristic. A guess is exactly the case that must be
confirmed, so the rule looks for explicit evidence of BOTH a day and a time
rather than trusting that a datetime came back.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from jarvis.tools.calendar import (PROTECTED_CALENDARS, event_confidence,
                                   pick_write_calendar)

NOW = datetime(2026, 8, 28, 9, 0).astimezone()


# ------------------------------------------------- add without asking
@pytest.mark.parametrize("text", [
    "electrical design lab monday at 4:10 pm",
    "dentist tomorrow at 2 pm",
    "lab presentation on friday at 9 am",
    "meeting with Dr Chen on september 3rd at 11 am",
    "study group tomorrow at noon",
    "biosensors exam next thursday at 8:30 am",
])
def test_explicit_day_and_time_is_confident(text):
    ok, _ = event_confidence(text, NOW)
    assert ok is True, f"{text!r} should not need confirming"


# ------------------------------------------------------- must confirm
@pytest.mark.parametrize("text,why", [
    ("dentist at 4", "bare hour -- am or pm is a guess"),
    ("meeting on friday", "no time at all"),
    ("lab presentation", "no day and no time"),
    ("something at half past", "no day, vague time"),
    ("", "nothing to parse"),
])
def test_anything_underspecified_gets_confirmed(text, why):
    ok, reason = event_confidence(text, NOW)
    assert ok is False, f"{text!r} should be confirmed: {why}"
    assert reason, "an unconfident verdict must say why"


def test_a_bare_hour_is_never_confident_even_though_the_parser_resolves_it():
    """parse_when_full turns 'at four' into a real datetime by heuristic.
    That heuristic is a guess, and guesses are what confirmation is for."""
    from jarvis.tools.timekeeper import parse_when_full
    dt, _, _ = parse_when_full("dentist at four", NOW)
    assert dt is not None, "precondition: the parser does resolve it"
    assert event_confidence("dentist at four", NOW)[0] is False


def test_24_hour_and_noon_forms_count_as_explicit():
    assert event_confidence("standup tomorrow at 09:15", NOW)[0] is True
    assert event_confidence("lunch tomorrow at midnight", NOW)[0] is True


def test_a_title_is_required_even_when_the_time_is_perfect():
    ok, reason = event_confidence("tomorrow at 2 pm", NOW)
    assert ok is False and "title" in reason.lower()


# ------------------------------------------------- choosing the target
class FakeCal:
    def __init__(self, name, comps=("VEVENT",)):
        self.name = name              # real caldav Calendar objects expose .name
        self._name = name
        self._comps = comps
        self.saved = []

    def get_supported_components(self):
        return list(self._comps)

    def save_event(self, ical):
        self.saved.append(ical)
        return object()


ICLOUD = [
    FakeCal("Navigate360 - Courses"), FakeCal("Family", ("VTODO",)),
    FakeCal("Hunter Peyrovi Calendar (Canvas)"), FakeCal("Calendar"),
    FakeCal("Home"), FakeCal("Reminders", ("VTODO",)), FakeCal("Work"),
]


def test_default_target_is_the_plain_calendar():
    assert pick_write_calendar(ICLOUD, None)._name == "Calendar"


def test_a_named_calendar_wins():
    assert pick_write_calendar(ICLOUD, "work")._name == "Work"
    assert pick_write_calendar(ICLOUD, "Home")._name == "Home"


def test_university_feeds_are_never_written_to():
    """Navigate360 and Canvas mirror official data Jarvis does not own."""
    for name in ("Navigate360 - Courses", "Hunter Peyrovi Calendar (Canvas)"):
        assert name in PROTECTED_CALENDARS or any(
            p.lower() in name.lower() for p in PROTECTED_CALENDARS), name
        with pytest.raises(ValueError, match="read-only|protected"):
            pick_write_calendar(ICLOUD, name)


def test_a_reminders_only_list_is_not_an_event_target():
    with pytest.raises(ValueError, match="does not take events"):
        pick_write_calendar(ICLOUD, "Reminders")


def test_an_unknown_name_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="don't have a calendar"):
        pick_write_calendar(ICLOUD, "Quidditch")
