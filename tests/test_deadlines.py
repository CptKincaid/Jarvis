"""Deadline heads-up (jarvis/deadlines.py): one reminder per Canvas
deadline `lead_hours` before it, the evening-before exam call from Canvas
items and calendar events alike, silence without a token, a CanvasError
that never takes the calendar half down, and the filed-state file that
survives a restart. Fakes only: fetch_due is a callable seam, the
timekeeper records add_reminder calls, the calendar is a stub.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import jarvis.deadlines as dl
from jarvis.deadlines import DeadlineHeadsUp
from jarvis.tools.canvas import CanvasError

TZ = ZoneInfo("America/Chicago")
CFG = {"canvas": {"token": "7~abcDEF123secret"}}
NO_TOKEN = {"canvas": {"token": ""}}


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)


def _item(title, due, course="BIOSENSORS"):
    return {"course": course, "title": title, "due": due.astimezone(timezone.utc)}


class _Cal:
    def __init__(self, events, configured=True):
        self._events = events
        self._configured = configured

    @property
    def configured(self):
        return self._configured

    def events(self):
        return self._events


def _ev(title, start, all_day=False):
    return SimpleNamespace(title=title, start=start, all_day=all_day, calendar="Canvas")


def _make(tmp_path, now, items=None, cfg=CFG, cal=None, lead=3, fetch=None):
    filed = []
    tk = SimpleNamespace(add_reminder=lambda due, text, repeat="": filed.append((due, text)))
    fetch_due = fetch or (lambda settings, days, fetch, now: list(items or []))
    h = DeadlineHeadsUp(cfg, tk, lead_hours=lead, state_path=tmp_path / "deadlines.json",
                        now=lambda: now, get_calendar=lambda: cal, fetch_due=fetch_due)
    return h, filed


# ------------------------------------------------------------ deadlines
def test_files_one_reminder_lead_hours_before_a_deadline(tmp_path):
    now = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)          # Monday 8:30 pm
    items = [_item("Lab 3 report", now.replace(hour=23, minute=59)),
             _item("Far off", now + timedelta(days=1, hours=6)),
             _item("Gone", now - timedelta(minutes=5))]
    h, filed = _make(tmp_path, now, items)
    assert h.tick() == 1
    (due, text), = filed
    assert text == "Lab 3 report for BIOSENSORS is due in 3 hours"
    assert abs(due - (now.replace(hour=20, minute=59)).timestamp()) < 1
    assert h.tick() == 0, "filed twice"
    # a restart reads the state file and does not file the deadline again
    h2, filed2 = _make(tmp_path, now, items)
    assert h2.tick() == 0 and filed2 == []
    saved = json.loads((tmp_path / "deadlines.json").read_text())
    assert list(saved) == ["BIOSENSORS|Lab 3 report|2026-09-01T04:59:00+00:00"]


def test_inside_the_lead_says_the_real_time_left_and_fires_now(tmp_path):
    now = datetime(2026, 8, 31, 22, 30, tzinfo=TZ)
    items = [_item("Quiz 2", now + timedelta(minutes=45), course="CIRCUITS"),
             _item("Essay", now + timedelta(hours=1, minutes=30), course="")]
    h, filed = _make(tmp_path, now, items)
    assert h.tick() == 2
    assert [t for _d, t in filed] == ["Quiz 2 for CIRCUITS is due in 45 minutes",
                                      "Essay is due in an hour and a half"]
    assert all(0 < d - now.timestamp() < 10 for d, _t in filed)


def test_a_far_deadline_is_not_filed_early(tmp_path):
    """Filing days ahead would announce work already handed in; the
    reminder cannot be withdrawn once the timekeeper has it."""
    now = datetime(2026, 8, 31, 9, 0, tzinfo=TZ)
    h, filed = _make(tmp_path, now, [_item("Lab 3 report", now.replace(hour=23, minute=59))])
    assert h.tick() == 0 and filed == []
    later = now.replace(hour=20, minute=30)
    h._now = lambda: later
    assert h.tick() == 1


def test_silent_without_a_token(tmp_path):
    now = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)
    calls = []

    def fetch(settings, days, fetch, now):
        calls.append(days)
        return [_item("Lab 3 report", now.replace(hour=23, minute=59))]
    h, filed = _make(tmp_path, now, cfg=NO_TOKEN, fetch=fetch)
    assert h.tick() == 0 and filed == [] and calls == []
    assert not (tmp_path / "deadlines.json").exists()
    # the placeholder the docs template leaves behind counts as unset too
    h, filed = _make(tmp_path, now, cfg={"canvas": {"token": "<paste the token>"}}, fetch=fetch)
    assert h.tick() == 0 and calls == []


def test_a_canvas_error_is_swallowed_and_the_calendar_half_still_runs(tmp_path):
    now = datetime(2026, 8, 31, 18, 40, tzinfo=TZ)           # Monday evening

    def fetch(settings, days, fetch, now):
        assert days == dl.DUE_DAYS
        raise CanvasError("unreachable", "URLError")
    cal = _Cal([_ev("Midterm 1", now.replace(day=1, month=9, hour=9, minute=0))])
    h, filed = _make(tmp_path, now, cal=cal, fetch=fetch)
    assert h.tick() == 1
    (due, text), = filed
    assert text == "Midterm 1 is tomorrow at 9:00 am"
    assert abs(due - now.replace(hour=19, minute=0).timestamp()) < 1, "the 7 pm eve call"


def test_no_timekeeper_means_no_work(tmp_path):
    now = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)
    h = DeadlineHeadsUp(CFG, None, state_path=tmp_path / "s.json", now=lambda: now,
                        fetch_due=lambda *a: (_ for _ in ()).throw(AssertionError("fetched")))
    assert h.tick() == 0


# ------------------------------------------------------------ exam eves
def test_exam_eve_from_canvas_names_the_course_and_dedupes_the_calendar_copy(tmp_path):
    now = datetime(2026, 8, 31, 18, 30, tzinfo=TZ)
    exam_at = datetime(2026, 9, 1, 9, 0, tzinfo=TZ)
    items = [_item("Midterm 1", exam_at)]
    # the iCloud "Canvas" subscription mirrors the same exam
    cal = _Cal([_ev("Midterm 1", exam_at), _ev("Lecture", exam_at + timedelta(hours=2))])
    h, filed = _make(tmp_path, now, items, cal=cal)
    assert h.tick() == 1
    (due, text), = filed
    assert text == "Midterm 1 for BIOSENSORS is tomorrow at 9:00 am"
    assert abs(due - now.replace(hour=19, minute=0).timestamp()) < 1
    assert h.tick() == 0
    key = "exam|BIOSENSORS|Midterm 1|2026-09-01T14:00:00+00:00"
    assert key in json.loads((tmp_path / "deadlines.json").read_text())


def test_exam_eve_past_seven_is_said_now_and_not_before_the_evening(tmp_path):
    exam_at = datetime(2026, 9, 1, 14, 0, tzinfo=TZ)
    cal = _Cal([_ev("Final exam", exam_at)])
    # 3 pm the day before: too early, nothing filed
    h, filed = _make(tmp_path, datetime(2026, 8, 31, 15, 0, tzinfo=TZ), cal=cal, cfg=NO_TOKEN)
    assert h.tick() == 0
    # booted at 9 pm: the 7 pm slot is gone, say it in five seconds
    now = datetime(2026, 8, 31, 21, 0, tzinfo=TZ)
    h, filed = _make(tmp_path, now, cal=cal, cfg=NO_TOKEN)
    assert h.tick() == 1
    (due, text), = filed
    assert text == "Final exam is tomorrow at 2:00 pm"
    assert 0 < due - now.timestamp() < 10
    # the day of: the meeting heads-up owns it; nothing here
    h, filed = _make(tmp_path, datetime(2026, 9, 1, 8, 0, tzinfo=TZ), cal=cal, cfg=NO_TOKEN)
    assert h.tick() == 0


def test_all_day_exam_event_reads_without_a_clock_time(tmp_path):
    now = datetime(2026, 8, 31, 19, 10, tzinfo=TZ)
    cal = _Cal([_ev("Quiz 3", datetime(2026, 9, 1, 0, 0, tzinfo=TZ), all_day=True)])
    h, filed = _make(tmp_path, now, cal=cal, cfg=NO_TOKEN)
    assert h.tick() == 1
    assert filed[0][1] == "Quiz 3 is tomorrow"


def test_unconfigured_or_broken_calendar_is_ignored(tmp_path):
    now = datetime(2026, 8, 31, 19, 10, tzinfo=TZ)
    exam_at = datetime(2026, 9, 1, 9, 0, tzinfo=TZ)
    h, filed = _make(tmp_path, now, cal=_Cal([_ev("Midterm 1", exam_at)], configured=False),
                     cfg=NO_TOKEN)
    assert h.tick() == 0

    class Broken:
        configured = True

        def events(self):
            raise RuntimeError("cache corrupt")
    h, filed = _make(tmp_path, now, [_item("Lab", now + timedelta(hours=2))], cal=Broken())
    assert h.tick() == 1, "the Canvas half still files"


# ---------------------------------------------------------------- state
def test_state_prunes_old_entries_drops_junk_and_saves_atomically(tmp_path, monkeypatch):
    now = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)
    state = tmp_path / "deadlines.json"
    state.write_text(json.dumps({"Old|x|2026-08-01T09:00:00+00:00": "2026-08-01T09:00:00+00:00",
                                 "Bad": None, "Junk|y": 7,
                                 "Kept|z|2026-09-02T09:00:00+00:00": "2026-09-02T09:00:00+00:00"}))
    replaced = []
    real = dl.os.replace
    monkeypatch.setattr(dl.os, "replace",
                        lambda src, dst: replaced.append((str(src), str(dst))) or real(src, dst))
    h, filed = _make(tmp_path, now, [_item("Lab 3 report", now.replace(hour=23, minute=59))])
    assert h.tick() == 1
    saved = json.loads(state.read_text())
    assert "Kept|z|2026-09-02T09:00:00+00:00" in saved
    assert "Old|x|2026-08-01T09:00:00+00:00" not in saved and "Bad" not in saved
    assert "Junk|y" not in saved
    assert all(isinstance(v, str) for v in saved.values())
    assert replaced and all(dst == str(state) and src != dst for src, dst in replaced)
    assert not list(tmp_path.glob("*.tmp"))
    state.write_text("{not json")
    h2, filed2 = _make(tmp_path, now, [_item("Lab 3 report", now.replace(hour=23, minute=59))])
    assert h2.tick() == 1 and json.loads(state.read_text())


def test_lead_hours_is_clamped_and_config_typed(tmp_path):
    now = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)
    tk = SimpleNamespace(add_reminder=lambda *a, **k: None)
    assert DeadlineHeadsUp(CFG, tk, lead_hours="junk").lead_hours == dl.DEFAULT_LEAD_HOURS
    assert DeadlineHeadsUp(CFG, tk, lead_hours=0).lead_hours == 0.25
    assert DeadlineHeadsUp(CFG, tk, lead_hours="1.5").lead_hours == 1.5
    h, filed = _make(tmp_path, now, [_item("Lab", now + timedelta(hours=2))], lead=1.5)
    assert h.tick() == 1 and filed[0][1].endswith("is due in an hour and a half")


def test_thread_start_is_idempotent_and_stops(tmp_path):
    h, _ = _make(tmp_path, datetime(2026, 8, 31, 20, 30, tzinfo=TZ), cfg=NO_TOKEN)
    h.start()
    t = h._thread
    h.start()
    assert h._thread is t
    h.stop()
    t.join(timeout=5)
    assert not t.is_alive()


def test_the_deadline_thread_is_joinable_and_restartable(tmp_path):
    h, _filed = _make(tmp_path, datetime(2026, 8, 31, 9, 0).astimezone())
    h.start()
    t1 = h._thread
    assert t1 is not None and t1.is_alive()
    h.stop()
    assert not t1.is_alive(), "stop() joins the thread out"
    h.start()
    assert h._thread is not t1 and h._thread.is_alive()
    h.stop()


def test_the_meeting_heads_up_thread_is_joinable_and_restartable(tmp_path):
    from jarvis.headsup import MeetingHeadsUp
    h = MeetingHeadsUp(lambda: None, None, state_path=tmp_path / "h.json")
    h.start()
    t1 = h._thread
    assert t1 is not None and t1.is_alive()
    h.stop()
    assert not t1.is_alive()
    h.start()
    assert h._thread is not t1 and h._thread.is_alive()
    h.stop()
