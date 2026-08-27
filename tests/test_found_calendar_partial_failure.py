"""FOUND 2026-08-26 (adversarial re-check of the assistant-tools sweep):
when one of several calendar sources fails, get_calendar answers ok=True
with "... ; nothing else." and never mentions that a whole calendar is
missing.

jarvis/tools/calendar.py _refresh() keeps a failing source's previous
events and records the failure in ``CalendarSource.errors``; get_calendar
(:589-604) only looks at ``snap.fetched_at`` and ``snap.stale``, so as
long as ONE source answered, the reply is a confident, complete-sounding
sentence built from a partial event set.  ``errors`` is never surfaced.

The realistic trigger is a revoked Google secret-ICS link: it keeps
returning HTTP 200, but with an HTML sign-in page, so parse_ics raises
ValueError rather than the fetch failing.

Reproduced in-process (two configured ical urls, one serving a real ICS,
one serving Google's sign-in HTML):

    refresh() -> True, errors ['google-2: ValueError']
    get_calendar(range="today") -> ok=True
        'Today: 11:00 pm Work standup; nothing else.'

This test asserts the reply admits the gap; it fails today.
"""
from datetime import date
from types import SimpleNamespace

import pytest

from jarvis.assistant_config import AssistantConfig
from jarvis.tools.calendar import CalendarSource, make_tools
from jarvis.tools.registry import ToolRegistry

# The event must fall on the day the test runs: it was written with the date
# hard-coded, so it silently stopped exercising the bug (and XPASSed strict)
# at the first midnight after it was written.
_TODAY = date.today().strftime("%Y%m%d")
GOOD_ICS = (b"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nX-WR-CALNAME:Work\r\n"
            b"BEGIN:VEVENT\r\nUID:g1\r\nDTSTART:" + _TODAY.encode() +
            b"T230000\r\nDTEND:" + _TODAY.encode() +
            b"T233000\r\nSUMMARY:Work standup\r\n"
            b"END:VEVENT\r\nEND:VCALENDAR\r\n")
SIGN_IN_HTML = (b"<!DOCTYPE html><html><head><title>Sign in - Google Accounts"
                b"</title></head><body>...</body></html>")


@pytest.mark.xfail(reason="get_calendar ignores CalendarSource.errors when "
                          "any one source answered", strict=True)
def test_a_broken_calendar_is_not_answered_as_if_it_were_empty(tmp_path):
    cfg = AssistantConfig({"google_ical_urls": ["https://good.test/a.ics",
                                                "https://revoked.test/b.ics"]})

    def fetch(url, **_):
        return GOOD_ICS if "good" in url else SIGN_IN_HTML

    source = CalendarSource(cfg, cache_path=tmp_path / "cal.json", fetch=fetch)
    assert source.refresh() is True
    assert source.errors == ["google-2: ValueError"], source.errors

    services = SimpleNamespace(calendar=source)
    reg = ToolRegistry()
    reg.register_many(make_tools(cfg, services))
    result = reg.call("get_calendar", {"range": "today"})

    # either say so, or do not claim the day is accounted for
    assert not (result.ok and "nothing else" in result.text.lower()), result.text
