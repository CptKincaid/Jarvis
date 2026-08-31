"""New-grade watch (jarvis/grades.py): the first tick is a silent baseline,
a moved total_score is announced once and named from one submissions call,
a token-less box never touches the network, and the snapshot survives a
restart. Fakes only: the Canvas HTTP seam is a table of canned pages.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.canvas as cv
from jarvis.grades import GradeWatch
from jarvis.tools.canvas import CanvasError

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)
BASE = "https://canvas.tamu.edu"
CFG = {"canvas": {"token": "7~abcDEF123secret", "base_url": BASE}}
NO_TOKEN = {"canvas": {"token": ""}}
COURSES_URL = BASE + "/api/v1/courses?"
SUBS_URL = BASE + "/api/v1/courses/101/students/submissions"


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    # active_courses caches per base_url for 10 minutes, process-wide.
    cv.clear_cache()
    yield
    cv.clear_cache()


def _course(cid, name, score=None, grade=None, student=True):
    enr = {"type": "student" if student else "teacher",
           "computed_current_score": score, "computed_current_grade": grade}
    return {"id": cid, "name": name, "course_code": "", "enrollments": [enr]}


def _sub(name, graded_at):
    return {"workflow_state": "graded", "assignment": {"name": name},
            "graded_at": graded_at.astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")}


class _Fetch:
    """url-prefix -> body (or Exception). Records every url so a test can
    assert the submissions call happened only on a delta."""

    def __init__(self, table):
        self.table = table
        self.urls = []

    def __call__(self, url, headers, timeout):
        self.urls.append(url)
        for prefix, val in sorted(self.table.items(), key=lambda kv: -len(kv[0])):
            if url.startswith(prefix):
                if isinstance(val, Exception):
                    raise val
                return 200, {}, json.dumps(val).encode()
        raise OSError(f"no route to {url}")

    def subs_calls(self):
        return [u for u in self.urls if u.startswith(SUBS_URL)]


def _make(tmp_path, courses, subs=None, cfg=CFG, now=NOW, state="grades.json"):
    table = {COURSES_URL: courses}
    if subs is not None:
        table[SUBS_URL] = subs
    fetch = _Fetch(table)
    said = []
    w = GradeWatch(cfg, announce=lambda title, text: said.append((title, text)),
                   state_path=tmp_path / state, now=lambda: now, fetch=fetch)
    return w, said, fetch


# ----------------------------------------------------------- baseline
def test_the_first_tick_is_a_silent_baseline(tmp_path):
    """A fresh install must not read out every course's standing."""
    w, said, _f = _make(tmp_path, [_course(101, "BIOSENSORS", 94.0, "A"),
                                   _course(102, "CIRCUITS", 81.0, "B-")])
    assert w.tick() == 0 and said == []
    saved = json.loads((tmp_path / "grades.json").read_text())
    assert saved["101"]["score"] == 94.0 and saved["102"]["grade"] == "B-"


def test_a_moved_score_is_announced_once_and_named(tmp_path):
    courses = [_course(101, "BMEN 420 500 BIOSENSORS FA26", 90.0, "A-")]
    w, said, fetch = _make(tmp_path, courses)
    assert w.tick() == 0                                  # baseline
    assert fetch.subs_calls() == [], "no submissions call on a quiet tick"
    fetch.table[COURSES_URL] = [_course(101, "BMEN 420 500 BIOSENSORS FA26",
                                        94.25, "A")]
    fetch.table[SUBS_URL] = [_sub("Quiz 2", NOW - timedelta(hours=2)),
                             _sub("Ancient homework", NOW - timedelta(days=40))]
    cv.clear_cache()
    assert w.tick() == 1
    (title, line), = said
    assert title == "Canvas grade"
    assert line == ("Quiz 2 posted for BIOSENSORS, sir: "
                    "the course is now 94.2% (A), up from 90%.")
    assert len(fetch.subs_calls()) == 1
    # said once: the snapshot moved with it
    cv.clear_cache()
    assert w.tick() == 0 and len(said) == 1


def test_a_drop_says_down_and_an_unnamed_delta_still_speaks(tmp_path):
    w, said, fetch = _make(tmp_path, [_course(102, "ECEN 214 Circuits", 81.0, "B-")],
                           subs=[])
    w.tick()
    fetch.table[COURSES_URL] = [_course(102, "ECEN 214 Circuits", 74.0, "C")]
    cv.clear_cache()
    assert w.tick() == 1
    # tidy_course drops the department number for speech
    assert said[0][1] == ("A grade posted for Circuits, sir: "
                          "the course is now 74% (C), down from 81%.")


def test_a_first_score_on_a_known_course_is_news_but_a_new_course_is_not(tmp_path):
    w, said, fetch = _make(tmp_path, [_course(101, "BIOSENSORS")], subs=[])
    w.tick()
    fetch.table[COURSES_URL] = [_course(101, "BIOSENSORS", 100.0, "A"),
                                _course(103, "KINE 199", 95.0, "A")]
    cv.clear_cache()
    assert w.tick() == 1, "the newly-enrolled course is not a grade event"
    assert said[0][1] == "A grade posted for BIOSENSORS, sir: the course is now 100% (A)."


def test_a_recomputed_last_decimal_is_not_news(tmp_path):
    w, said, fetch = _make(tmp_path, [_course(101, "BIOSENSORS", 94.25, "A")], subs=[])
    w.tick()
    fetch.table[COURSES_URL] = [_course(101, "BIOSENSORS", 94.27, "A")]
    cv.clear_cache()
    assert w.tick() == 0 and said == []


def test_a_stale_graded_at_does_not_name_the_wrong_assignment(tmp_path):
    w, said, fetch = _make(tmp_path, [_course(101, "BIOSENSORS", 90.0, "A-")],
                           subs=[_sub("Homework 1", NOW - timedelta(days=30))])
    w.tick()
    fetch.table[COURSES_URL] = [_course(101, "BIOSENSORS", 92.0, "A-")]
    cv.clear_cache()
    assert w.tick() == 1
    assert said[0][1].startswith("A grade posted for BIOSENSORS")


def test_a_broken_submissions_call_still_speaks_the_course_line(tmp_path):
    w, said, fetch = _make(tmp_path, [_course(101, "BIOSENSORS", 90.0, "A-")])
    w.tick()
    fetch.table[COURSES_URL] = [_course(101, "BIOSENSORS", 92.0, "A-")]
    fetch.table[SUBS_URL] = CanvasError("unreachable", "timeout")
    cv.clear_cache()
    assert w.tick() == 1
    assert said[0][1].startswith("A grade posted for BIOSENSORS")


# ------------------------------------------------------------- silence
def test_silent_without_a_token(tmp_path):
    calls = []

    def fetch(url, headers, timeout):
        calls.append(url)
        raise AssertionError("a token-less box must not call Canvas")

    said = []
    w = GradeWatch(NO_TOKEN, announce=lambda t, x: said.append(x),
                   state_path=tmp_path / "g.json", now=lambda: NOW, fetch=fetch)
    assert w.tick() == 0 and said == [] and calls == []
    assert not (tmp_path / "g.json").exists()


def test_the_switch_turns_it_off(tmp_path):
    cfg = dict(CFG, watch={"grades": False})
    w, said, fetch = _make(tmp_path, [_course(101, "BIOSENSORS", 90.0)], cfg=cfg)
    assert w.tick() == 0 and fetch.urls == []


def test_a_canvas_outage_is_swallowed(tmp_path):
    w, said, fetch = _make(tmp_path, [_course(101, "BIOSENSORS", 90.0, "A-")])
    w.tick()
    fetch.table[COURSES_URL] = CanvasError("unreachable", "timeout")
    cv.clear_cache()
    assert w.tick() == 0 and said == []


def test_non_student_enrolments_are_ignored(tmp_path):
    w, said, _f = _make(tmp_path, [_course(105, "Observer only", 50.0, "F",
                                           student=False)])
    assert w.tick() == 0 and said == []
    # a roster with no student enrolment leaves no snapshot behind at all
    assert not (tmp_path / "grades.json").exists()


# --------------------------------------------------------------- state
def test_a_restart_reads_the_snapshot_and_does_not_re_announce(tmp_path):
    w, _said, _f = _make(tmp_path, [_course(101, "BIOSENSORS", 94.0, "A")])
    w.tick()
    cv.clear_cache()
    w2, said2, _f2 = _make(tmp_path, [_course(101, "BIOSENSORS", 94.0, "A")])
    assert w2.tick() == 0 and said2 == []


def test_a_junk_state_file_is_a_baseline_not_a_crash(tmp_path):
    (tmp_path / "grades.json").write_text('{"101": "not a dict", "x": 3}')
    w, said, _f = _make(tmp_path, [_course(101, "BIOSENSORS", 94.0, "A")])
    assert w.tick() == 0 and said == []


def test_a_sink_failure_does_not_re_announce_next_tick(tmp_path):
    """The snapshot is saved before speaking, so a Discord outage costs one
    line, not a repeat every fifteen minutes."""
    def boom(title, text):
        raise RuntimeError("discord down")

    fetch = _Fetch({COURSES_URL: [_course(101, "BIOSENSORS", 90.0, "A-")],
                    SUBS_URL: []})
    w = GradeWatch(CFG, announce=boom, state_path=tmp_path / "g.json",
                   now=lambda: NOW, fetch=fetch)
    w.tick()
    fetch.table[COURSES_URL] = [_course(101, "BIOSENSORS", 92.0, "A-")]
    cv.clear_cache()
    assert w.tick() == 0                       # spoke nothing: the sink threw
    cv.clear_cache()
    assert w.tick() == 0                       # and does not try again


# -------------------------------------------------------------- thread
def test_the_thread_is_joinable_and_restartable(tmp_path):
    w, _said, _f = _make(tmp_path, [])
    w.start()
    t1 = w._thread
    assert t1 is not None and t1.is_alive()
    w.start()
    assert w._thread is t1, "start is idempotent"
    w.stop()
    assert not t1.is_alive(), "stop() joins the thread out"
    w.start()
    assert w._thread is not t1 and w._thread.is_alive()
    w.stop()
