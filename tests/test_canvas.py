"""Canvas LMS tools: canned planner / courses / announcements JSON through
a fake ``_fetch`` -> exact fact sheets; Link-header pagination; the
unconfigured token; 401; transport failure; the planner-off fallback to
per-course assignments; the 10-min course cache; spoken-friendly dates.

Firewall: no network in any test (urlopen is booby-trapped), tmp
JARVIS_LOG_DIR (conftest), tmp JARVIS_ASSISTANT_CONFIG.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from jarvis.tools import canvas as cv
from jarvis.tools.registry import DESCRIPTION_WORD_CAP, ToolRegistry

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 31, 9, 0, tzinfo=TZ)          # Monday morning
BASE = "https://canvas.tamu.edu"
TOKEN = "7~abcDEF123secret"


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(cv, "_now", lambda: NOW)
    monkeypatch.setattr(cv, "_clock", lambda: 1000.0)
    cv.clear_cache()
    yield
    cv.clear_cache()


def _iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _local(day_offset, hour, minute=0):
    return (NOW + timedelta(days=day_offset)).replace(hour=hour, minute=minute)


def _course(cid, name, code="", score=None, grade=None, **extra):
    enr = {"type": "student", "computed_current_score": score,
           "computed_current_grade": grade}
    return {"id": cid, "name": name, "course_code": code, "enrollments": [enr], **extra}


COURSES = [
    _course(101, "BMEN 420 500 BIOSENSORS FA26", "BMEN-420-500", 94.25, "A"),
    _course(102, "ECEN 214 Electrical Circuit Theory FA26", "ECEN-214", 81.0, "B-"),
    _course(103, "KINE 199 Required Physical Activity", "KINE-199"),
    _course(104, "Old term", access_restricted_by_date=True),
    {"id": 105, "name": "Observer only", "enrollments": [{"type": "teacher"}]},
]


def _planner(ptype, cid, title, due, submitted=False, complete=False, context="BIOSENSORS"):
    return {"plannable_type": ptype, "course_id": cid, "context_name": context,
            "plannable": {"title": title, "due_at": _iso(due) if due else None},
            "plannable_date": _iso(due) if due else None,
            "submissions": {"submitted": submitted, "graded": False} if ptype != "calendar_event" else False,
            "planner_override": {"marked_complete": complete} if complete else None}


PLANNER_P1 = [
    _planner("assignment", 101, "Lab 3 report", _local(1, 23, 59),
             context="BMEN 420 500 BIOSENSORS FA26"),
    _planner("quiz", 102, "Quiz 2", _local(0, 17, 0),
             context="ECEN 214 Electrical Circuit Theory FA26"),
    _planner("assignment", 101, "Already in", _local(2, 23, 59), submitted=True,
             context="BMEN 420 500 BIOSENSORS FA26"),
    _planner("calendar_event", 101, "Lecture", _local(1, 10, 0)),
]
PLANNER_P2 = [
    _planner("discussion_topic", 102, "Discussion: op-amps", _local(4, 23, 59),
             context="ECEN 214 Electrical Circuit Theory FA26"),
    _planner("assignment", 101, "Done by hand", _local(3, 8, 0), complete=True),
    _planner("assignment", 101, "Far off", _local(20, 23, 59),
             context="BMEN 420 500 BIOSENSORS FA26"),
]

ANNOUNCEMENTS = [
    {"title": "Lab safety reminder", "context_code": "course_101",
     "posted_at": _iso(_local(0, 8, 15)),
     "message": "<p>Please <b>wear</b> goggles &amp; closed shoes.<br>Thanks!</p>"},
    {"title": "Exam 1 moved", "context_code": "course_102",
     "posted_at": _iso(_local(-1, 16, 30)), "message": "<div>Now on Friday.</div>"},
    {"title": "Stale one", "context_code": "course_102",
     "posted_at": _iso(_local(-10, 9, 0)), "message": "old"},
]


class FakeFetch:
    """url-prefix -> (status, headers, body-object) | Exception. Records
    every (url, headers, timeout) so tests can check the auth header and
    that the token never leaks into a URL."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, headers, timeout))
        assert 0 < timeout <= cv.REQUEST_TIMEOUT
        # longest matching prefix wins: "?page=2" must beat the bare "?"
        for prefix, val in sorted(self.table.items(), key=lambda kv: -len(kv[0])):
            if url.startswith(prefix):
                if isinstance(val, Exception):
                    raise val
                status, hdrs, body = val
                if not isinstance(body, bytes):
                    body = json.dumps(body).encode()
                return status, hdrs, body
        raise OSError(f"no route to {url}")

    def urls(self):
        return [c[0] for c in self.calls]


def _ok(body, next_url=None):
    hdrs = {"link": f'<{next_url}>; rel="next", <x>; rel="current"'} if next_url else {}
    return 200, hdrs, body


PLANNER_URL = BASE + "/api/v1/planner/items"
COURSES_URL = BASE + "/api/v1/courses?"
ANN_URL = BASE + "/api/v1/announcements"


def canned():
    return {
        PLANNER_URL + "?": _ok(PLANNER_P1, PLANNER_URL + "?page=2&per_page=50"),
        PLANNER_URL + "?page=2": _ok(PLANNER_P2),
        COURSES_URL: _ok(COURSES),
        ANN_URL: _ok(ANNOUNCEMENTS),
    }


def _cfg(token=TOKEN, base=None):
    data = {"canvas": {"token": token}}
    if base is not None:
        data["canvas"]["base_url"] = base
    return data


def _tools(cfg, fetch, monkeypatch):
    monkeypatch.setattr(cv, "_fetch", fetch)
    reg = ToolRegistry()
    reg.register_many(cv.make_tools(cfg, None))
    return reg


# ---------------------------------------------------------------- spec
def test_tool_names_and_description_budget():
    specs = cv.make_tools(_cfg(), None)
    assert [s.name for s in specs] == ["canvas_due", "canvas_grades", "canvas_announcements"]
    for s in specs:
        assert s.description_words() <= DESCRIPTION_WORD_CAP, s.name
        assert s.schema()["function"]["parameters"]["type"] == "object"
    assert "days" in specs[0].parameters["properties"]
    assert specs[1].parameters["properties"] == {}


def test_settings_and_placeholders():
    assert cv.canvas_settings(_cfg()) == {"base_url": BASE, "token": TOKEN}
    assert cv.canvas_settings(_cfg(base="https://x.instructure.com/")) == \
        {"base_url": "https://x.instructure.com", "token": TOKEN}
    assert cv.canvas_settings(_cfg(base="canvas.example.edu"))["base_url"] == \
        "https://canvas.example.edu"
    for bad in ("", "  ", "<paste your token here>", "your-token", "changeme",
                "xxxxxxxx", "PASTE-TOKEN", None, 42):
        assert cv.canvas_settings(_cfg(token=bad)) is None, bad
    assert cv.canvas_settings(None) is None and cv.canvas_settings({}) is None


def test_unconfigured_token_speaks_setup_line_without_a_request(monkeypatch):
    fetch = FakeFetch(canned())
    for cfg in (None, {}, _cfg(token=""), _cfg(token="<paste here>")):
        reg = _tools(cfg, fetch, monkeypatch)
        for name in ("canvas_due", "canvas_grades", "canvas_announcements"):
            r = reg.call(name, {})
            assert not r.ok and r.speak == cv.SETUP_LINE
            assert r.speak == ("I'll need a Canvas access token set up, sir; "
                               "the notes are in docs/assistant-setup.md.")
    assert fetch.calls == []


def test_real_assistant_config_reads_the_token(tmp_path, monkeypatch):
    """AssistantConfig.get on a key that DEFAULTS does not know yet returns
    the default -- the tool must read the token straight rather than via
    is_configured(), which answers False for unknown sections."""
    from jarvis.assistant_config import AssistantConfig

    cfg = AssistantConfig.load(tmp_path / "assistant.json")
    assert cv.canvas_settings(cfg) is None
    cfg.set("canvas.token", TOKEN)
    assert cv.canvas_settings(cfg) == {"base_url": BASE, "token": TOKEN}
    fetch = FakeFetch(canned())
    reg = _tools(cfg, fetch, monkeypatch)
    assert reg.call("canvas_grades", {}).ok
    assert TOKEN not in fetch.urls()[0]


# ----------------------------------------------------------------- due
def test_canvas_due_paginates_filters_and_formats(monkeypatch):
    fetch = FakeFetch(canned())
    reg = _tools(_cfg(), fetch, monkeypatch)
    r = reg.call("canvas_due", {})
    assert r.ok and r.speak is None and r.max_sentences == 4
    assert r.text == (
        "Due this week (3):\n"
        "1) Electrical Circuit Theory - Quiz 2, today 5:00 pm\n"
        "2) BIOSENSORS - Lab 3 report, tomorrow 11:59 pm\n"
        "3) Electrical Circuit Theory - Discussion: op-amps, Fri 11:59 pm")
    # one planner call plus its second page; no course walk
    urls = fetch.urls()
    assert len(urls) == 2 and urls[1] == PLANNER_URL + "?page=2&per_page=50"
    assert urls[0].startswith(PLANNER_URL + "?start_date=2026-08-31T14%3A00%3A00Z"
                              "&end_date=2026-09-07T14%3A00%3A00Z")
    assert all(h["Authorization"] == f"Bearer {TOKEN}" for _, h, _ in fetch.calls)
    assert all(TOKEN not in u for u in urls)


def test_canvas_due_days_argument_and_far_dates(monkeypatch):
    fetch = FakeFetch(canned())
    reg = _tools(_cfg(), fetch, monkeypatch)
    r = reg.call("canvas_due", {"days": "30"})
    assert r.text.startswith("Due in the next 30 days (4):")
    assert r.text.endswith("4) BIOSENSORS - Far off, Sun 20 Sep 11:59 pm")
    assert "end_date=2026-09-30T14" in fetch.urls()[0]
    # garbage / out of range days fall back or clamp instead of raising
    assert reg.call("canvas_due", {"days": "soon"}).text.startswith("Due this week")
    r = reg.call("canvas_due", {"days": 500})
    assert "end_date=2026-09-30T14" in fetch.urls()[-2]
    r = reg.call("canvas_due", {"days": 0})
    assert "end_date=2026-09-01T14" in fetch.urls()[-2]
    assert r.text.startswith("Due in the next day (1):")


def test_canvas_due_empty(monkeypatch):
    table = canned()
    table[PLANNER_URL + "?"] = _ok([])
    reg = _tools(_cfg(), FakeFetch(table), monkeypatch)
    r = reg.call("canvas_due", {})
    assert r.ok and r.speak == cv.NOTHING_DUE_LINE == "Nothing due this week, sir."
    r = reg.call("canvas_due", {"days": 3})
    assert r.ok and r.speak == "Nothing due in the next 3 days, sir."
    r = reg.call("canvas_due", {"days": 1})
    assert r.speak == "Nothing due in the next day, sir."


def test_canvas_due_bad_token_and_unreachable(monkeypatch):
    table = canned()
    table[PLANNER_URL + "?"] = (401, {}, {"errors": [{"message": "Invalid access token."}]})
    reg = _tools(_cfg(), FakeFetch(table), monkeypatch)
    r = reg.call("canvas_due", {})
    assert not r.ok and r.speak == cv.BAD_TOKEN_LINE == "Canvas rejected the token, sir."
    table[PLANNER_URL + "?"] = OSError("Name or service not known")
    reg = _tools(_cfg(), FakeFetch(table), monkeypatch)
    r = reg.call("canvas_due", {})
    assert not r.ok and r.speak == cv.UNREACHABLE_LINE == "I can't reach Canvas, sir."
    table[PLANNER_URL + "?"] = (500, {}, b"<html>oops")
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_due", {})
    assert not r.ok and r.speak == cv.UNREACHABLE_LINE and "500" in r.text
    table[PLANNER_URL + "?"] = (200, {}, b"not json")
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_due", {})
    assert not r.ok and r.speak == cv.UNREACHABLE_LINE


def test_canvas_due_falls_back_to_course_assignments_when_planner_is_off(monkeypatch):
    table = canned()
    table[PLANNER_URL + "?"] = (404, {}, {"errors": [{"message": "not found"}]})
    a101 = BASE + "/api/v1/courses/101/assignments"
    a102 = BASE + "/api/v1/courses/102/assignments"
    a103 = BASE + "/api/v1/courses/103/assignments"
    table[a101] = _ok([{"name": "Lab 3 report", "due_at": _iso(_local(1, 23, 59))},
                       {"name": "No date", "due_at": None},
                       {"name": "Next month", "due_at": _iso(_local(25, 23, 59))}])
    table[a102] = _ok([{"name": "Quiz 2", "due_at": _iso(_local(0, 17, 0))}])
    table[a103] = _ok([])
    fetch = FakeFetch(table)
    reg = _tools(_cfg(), fetch, monkeypatch)
    r = reg.call("canvas_due", {})
    assert r.ok and r.text == (
        "Due this week (2):\n"
        "1) Electrical Circuit Theory - Quiz 2, today 5:00 pm\n"
        "2) BIOSENSORS - Lab 3 report, tomorrow 11:59 pm")
    urls = fetch.urls()
    assert urls[0].startswith(PLANNER_URL)
    assert urls[1].startswith(COURSES_URL) and "enrollment_state=active" in urls[1]
    assert [u.split("?")[0] for u in urls[2:]] == [a101, a102, a103]
    assert all("bucket=upcoming" in u for u in urls[2:])
    # a 403 planner does the same; a 500 does NOT (that is an outage)
    table[PLANNER_URL + "?"] = (403, {}, b"{}")
    assert _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_due", {}).ok
    table[PLANNER_URL + "?"] = (500, {}, b"{}")
    assert not _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_due", {}).ok


def test_pagination_stops_at_the_budget_and_page_cap(monkeypatch):
    """A slow Canvas returns what it has so far rather than an excuse."""
    # budget init, page-1 timeout, then the clock jumps before the page-2 check
    ticks = iter([0.0, 0.0] + [5.8] * 20)
    monkeypatch.setattr(cv, "_monotonic", lambda: next(ticks))
    fetch = FakeFetch(canned())
    reg = _tools(_cfg(), fetch, monkeypatch)
    r = reg.call("canvas_due", {})
    assert r.ok and len(fetch.calls) == 1, "page 2 was fetched with no budget left"
    assert r.text.startswith("Due this week (2):")
    # the page cap: a page that always points at another page
    table = {PLANNER_URL: _ok(PLANNER_P1[:1], PLANNER_URL + "?page=next")}
    monkeypatch.setattr(cv, "_monotonic", lambda: 0.0)
    fetch = FakeFetch(table)
    reg = _tools(_cfg(), fetch, monkeypatch)
    r = reg.call("canvas_due", {})
    assert r.ok and len(fetch.calls) == cv.MAX_PAGES
    # budget already gone before the first request -> unreachable
    monkeypatch.setattr(cv, "_monotonic", lambda: 100.0)
    budget = cv._Budget()
    monkeypatch.setattr(cv, "_monotonic", lambda: 200.0)
    with pytest.raises(cv.CanvasError) as info:
        budget.timeout()
    assert info.value.kind == "unreachable"


def test_next_link_parsing():
    link = ('<https://c.edu/api/v1/planner/items?page=2&per_page=50>; rel="next",'
            '<https://c.edu/api/v1/planner/items?page=1&per_page=50>; rel="current",'
            '<https://c.edu/api/v1/planner/items?page=3&per_page=50>; rel="last"')
    assert cv._next_link({"link": link}) == \
        "https://c.edu/api/v1/planner/items?page=2&per_page=50"
    assert cv._next_link({"link": '<https://c.edu/x?page=1>; rel="current"'}) is None
    assert cv._next_link({}) is None


# -------------------------------------------------------------- grades
def test_canvas_grades_and_course_cache(monkeypatch):
    fetch = FakeFetch(canned())
    reg = _tools(_cfg(), fetch, monkeypatch)
    r = reg.call("canvas_grades", {})
    assert r.ok and r.max_sentences == 4 and r.text == (
        "Current grades (3 courses):\n"
        "1) BIOSENSORS 94.2% (A)\n"
        "2) Electrical Circuit Theory 81% (B-)\n"
        "3) Required Physical Activity no score yet")
    url = fetch.urls()[0]
    assert "include%5B%5D=total_scores" in url and "enrollment_state=active" in url
    # cached for ten minutes, per base URL
    reg.call("canvas_grades", {})
    assert len(fetch.calls) == 1
    monkeypatch.setattr(cv, "_clock", lambda: 1000.0 + 9 * 60)
    reg.call("canvas_grades", {})
    assert len(fetch.calls) == 1
    monkeypatch.setattr(cv, "_clock", lambda: 1000.0 + 11 * 60)
    reg.call("canvas_grades", {})
    assert len(fetch.calls) == 2
    other = _tools(_cfg(base="https://other.instructure.com"), fetch, monkeypatch)
    other.call("canvas_grades", {})
    assert len(fetch.calls) == 3


def test_canvas_grades_nothing_posted_and_errors(monkeypatch):
    table = canned()
    table[COURSES_URL] = _ok([_course(1, "MATH 152 Engineering Math II")])
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_grades", {})
    assert r.ok and r.speak == cv.NO_GRADES_LINE
    cv.clear_cache()
    table[COURSES_URL] = _ok([])
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_grades", {})
    assert r.ok and r.speak == cv.NO_GRADES_LINE
    cv.clear_cache()
    table[COURSES_URL] = (401, {}, b"{}")
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_grades", {})
    assert not r.ok and r.speak == cv.BAD_TOKEN_LINE
    table[COURSES_URL] = TimeoutError("timed out")
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_grades", {})
    assert not r.ok and r.speak == cv.UNREACHABLE_LINE
    # letter grade only / score only
    assert cv._score_words({"score": None, "grade": "A"}) == "A"
    assert cv._score_words({"score": 88.0, "grade": None}) == "88%"
    assert cv._score_words({"score": "bad", "grade": None}) == "no score yet"


# ------------------------------------------------------- announcements
def test_canvas_announcements(monkeypatch):
    fetch = FakeFetch(canned())
    reg = _tools(_cfg(), fetch, monkeypatch)
    r = reg.call("canvas_announcements", {})
    assert r.ok and r.max_sentences == 4 and r.text == (
        "Announcements in the last 3 days (2):\n"
        "1) BIOSENSORS - Lab safety reminder (today 8:15 am): "
        "Please wear goggles & closed shoes. Thanks!\n"
        "2) Electrical Circuit Theory - Exam 1 moved (yesterday 4:30 pm): Now on Friday.")
    urls = fetch.urls()
    assert urls[0].startswith(COURSES_URL)          # roster first (then cached)
    ann = urls[1]
    assert ann.startswith(ANN_URL + "?")
    assert "context_codes%5B%5D=course_101" in ann and "context_codes%5B%5D=course_102" in ann
    # date-restricted courses drop out; a non-student enrolment still gets
    # its announcements
    assert "course_104" not in ann and "course_105" in ann
    assert "start_date=2026-08-28" in ann and "end_date=2026-09-01" in ann
    reg.call("canvas_announcements", {"days": 7})
    assert len(fetch.calls) == 3 and "start_date=2026-08-24" in fetch.urls()[2]


def test_canvas_announcements_empty_and_errors(monkeypatch):
    table = canned()
    table[ANN_URL] = _ok([])
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_announcements", {})
    assert r.ok and r.speak == "No announcements in the last 3 days, sir."
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_announcements", {"days": 7})
    assert r.speak == "No announcements in the last week, sir."
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_announcements", {"days": 1})
    assert r.speak == "No announcements in the last day, sir."
    cv.clear_cache()
    table[COURSES_URL] = _ok([])
    fetch = FakeFetch(table)
    r = _tools(_cfg(), fetch, monkeypatch).call("canvas_announcements", {})
    assert r.ok and r.speak.startswith("No announcements") and len(fetch.calls) == 1
    cv.clear_cache()
    table[COURSES_URL] = _ok(COURSES)
    table[ANN_URL] = (401, {}, b"{}")
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_announcements", {})
    assert not r.ok and r.speak == cv.BAD_TOKEN_LINE
    table[ANN_URL] = ConnectionResetError()
    r = _tools(_cfg(), FakeFetch(table), monkeypatch).call("canvas_announcements", {})
    assert not r.ok and r.speak == cv.UNREACHABLE_LINE


def test_snippet_strips_html():
    assert cv.snippet("<p>Hi <b>there</b>&nbsp;all.</p><style>x{}</style><br>Bye") == \
        "Hi there all. Bye"
    long = "word " * 60
    out = cv.snippet(long)
    assert out.endswith("…") and len(out) <= cv.SNIPPET_CHARS + 1
    assert cv.snippet(None) == ""


# --------------------------------------------------------------- words
def test_when_words_is_local_and_spoken_friendly():
    utc_due = datetime(2026, 9, 1, 4, 59, tzinfo=timezone.utc)   # 11:59 pm Mon, Chicago
    assert cv.when_words(utc_due, NOW) == "today 11:59 pm"
    assert cv.when_words(_local(1, 0, 5), NOW) == "tomorrow 12:05 am"
    assert cv.when_words(_local(-1, 12, 0), NOW) == "yesterday 12:00 pm"
    assert cv.when_words(_local(3, 15, 30), NOW) == "Thu 3:30 pm"
    assert cv.when_words(_local(6, 9, 0), NOW) == "Sun 9:00 am"
    assert cv.when_words(_local(7, 9, 0), NOW) == "Mon 7 Sep 9:00 am"
    assert cv.when_words(_local(-3, 9, 0), NOW) == "Fri 28 Aug 9:00 am"
    assert cv._parse_iso("2026-09-01T04:59:00Z") == utc_due
    assert cv._parse_iso("2026-09-01T04:59:00") == utc_due     # naive -> UTC
    assert cv._parse_iso("") is None and cv._parse_iso("soon") is None


def test_tidy_course():
    assert cv.tidy_course("BMEN 420 500 BIOSENSORS FA26") == "BIOSENSORS"
    assert cv.tidy_course("BMEN-420-500-BIOSENSORS-FA26") == "BIOSENSORS"
    assert cv.tidy_course("ECEN 214 Electrical Circuit Theory Fall 2026") == \
        "Electrical Circuit Theory"
    assert cv.tidy_course("MATH 152: Engineering Math II (SP27)") == "Engineering Math II"
    assert cv.tidy_course("Introduction to Biology") == "Introduction to Biology"
    assert cv.tidy_course("BMEN 420") == "BMEN 420"        # never strips to nothing
    assert cv.tidy_course("  ") == ""
    assert len(cv.tidy_course("x" * 100)) == 48


def test_handlers_never_raise_through_the_registry(monkeypatch):
    """A handler that blows up is the registry's problem, but a broken
    payload shape must not even get that far."""
    table = canned()
    table[PLANNER_URL + "?"] = _ok([{"plannable_type": "assignment"}, "junk", None,
                                   {"plannable_type": "quiz", "plannable": {"title": "Q",
                                                                            "due_at": "never"}}])
    table[COURSES_URL] = _ok([{"id": None, "name": "x"}, "junk",
                              {"id": 7, "name": "MATH 152 Calc", "enrollments": "odd"}])
    table[ANN_URL] = _ok([{"title": "T", "context_code": "weird", "posted_at": None,
                           "message": None}, 3])
    reg = _tools(_cfg(), FakeFetch(table), monkeypatch)
    assert reg.call("canvas_due", {}).speak == cv.NOTHING_DUE_LINE
    assert reg.call("canvas_grades", {}).speak == cv.NO_GRADES_LINE
    r = reg.call("canvas_announcements", {})
    assert r.ok and r.text.endswith("1) Canvas - T")
