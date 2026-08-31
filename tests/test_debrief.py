"""The debrief (jarvis/debrief.py): "How did the midterm go, sir?"

The hard half is knowing something real ENDED, and it is hard in three
ways that each get a test here:

  * the cancellation case -- an event that only ever appeared in the cache
    after it had started must never be asked about. calendar._window
    fetches from yesterday midnight, so a row added retroactively this
    afternoon looks exactly like one that happened;
  * the restart case -- the `asked` ledger is the promise never to ask
    twice, and it has to survive the process;
  * the quiet-hours case -- an exam that ends at eleven at night is held,
    not dropped, and asked in the morning.

Plus the filing half: the answer goes to the journal and to memory, and
NEVER to the model as chat. Fakes only; no network, no speech path (this
module has none by design).
"""
import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from jarvis.debrief import (DebriefWatch, Candidate, file_answer, fact_key,
                            matched_word, question_for)

TZ = ZoneInfo("America/Chicago")
DAY = datetime(2026, 9, 14, tzinfo=TZ)


def at(hh, mm=0, day=DAY):
    return day.replace(hour=hh, minute=mm)


@pytest.fixture(autouse=True)
def _firewall(monkeypatch):
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)


class Cfg(dict):
    def get(self, key, default=None):
        node = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


class Cal:
    configured = True

    def __init__(self, events=()):
        self._events = list(events)

    def events(self):
        return list(self._events)


def _ev(title, start, minutes=90, all_day=False):
    return SimpleNamespace(title=title, start=start,
                           end=start + timedelta(minutes=minutes), all_day=all_day)


MIDTERM = _ev("BIOSENSORS Midterm 1", at(13, 0), minutes=90)   # ends 14:30


def _watch(tmp_path, clock, events=(MIDTERM,), quiet=None, presence=None,
           cfg=None, seen=None):
    return DebriefWatch(cfg=cfg if cfg is not None else Cfg({}),
                        get_calendar=Cal(events), quiet=quiet, presence=presence,
                        state_path=tmp_path / "debrief.json",
                        now=lambda tz=None: clock.now)


# --------------------------------------------------------------- the ask
def test_an_event_seen_ahead_then_ended_is_asked_about_once(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock)
    assert w.tick() is None                       # still ahead: only recorded
    assert list(w.seen)                            # …but recorded it is
    clock.now = at(15, 10)                         # 40 minutes after it ended
    cand = w.tick()
    assert cand is not None and cand.question == "How did the midterm go, sir?"
    w.mark_asked(cand.key)
    assert w.tick() is None                        # never twice


def test_a_retroactively_added_event_is_never_asked_about(tmp_path):
    """The cancellation case. The cache reaches back a day, so something
    that was moved, cancelled or entered after the fact looks identical to
    something that happened -- except that this watch never saw it coming."""
    clock = SimpleNamespace(now=at(15, 10))
    w = _watch(tmp_path, clock)
    assert w.tick() is None
    assert w.seen == {}


def test_the_asked_ledger_survives_a_restart(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock)
    w.tick()
    clock.now = at(15, 10)
    cand = w.tick()
    w.mark_asked(cand.key)
    fresh = _watch(tmp_path, clock)                # a new process, same file
    assert fresh.tick() is None
    assert cand.key in fresh.asked


def test_too_soon_and_too_late_are_both_silence(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock)
    w.tick()
    clock.now = at(14, 35)                         # 5 minutes: still walking out
    assert w.tick() is None
    clock.now = at(20, 0)                          # 5½ hours: the moment passed
    assert w.tick() is None


def test_only_exam_and_interview_words_earn_a_question(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock, events=(_ev("lunch with Sam", at(13, 0)),))
    w.tick()
    clock.now = at(15, 10)
    assert w.tick() is None


def test_whole_words_only():
    """"contest" is not a test and "finalise the slides" is not a final."""
    kws = ("test", "final", "interview")
    assert matched_word("Driving test", kws) == "test"
    assert matched_word("Robotics contest", kws) == ""
    assert matched_word("Finalise the slides", kws) == ""
    assert matched_word("Final exam - Rm 214", kws) == "final"


def test_the_question_names_the_keyword_not_the_whole_title():
    assert question_for("BIOSENSORS Midterm 1 - Rm 214", "midterm") == \
        "How did the midterm go, sir?"


def test_an_all_day_row_is_never_a_candidate(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock,
               events=(_ev("Final exams week", DAY, minutes=1440, all_day=True),))
    w.tick()
    clock.now = at(15, 10)
    assert w.tick() is None


def test_one_question_at_a_time(tmp_path):
    """Two in a row is an interrogation; the next tick is five minutes away."""
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock,
               events=(MIDTERM, _ev("Interview with Acme", at(13, 15), minutes=60)))
    w.tick()
    clock.now = at(15, 10)
    assert w.tick() is not None
    assert len([c for c in [w.tick()] if c is not None]) == 1


# -------------------------------------------------------------- the gates
def test_quiet_hours_hold_the_question_to_the_morning(tmp_path):
    """Not dropped: an exam that ends at eleven at night is exactly the one
    worth asking about, and the 180-minute window would swallow it."""
    late = _ev("BIOSENSORS Final", at(21, 0), minutes=120)     # ends 23:00
    quiet = SimpleNamespace(should_hold=lambda: True,
                            reason=lambda: "quiet hours until 7:00 am")
    clock = SimpleNamespace(now=at(20, 0))
    w = _watch(tmp_path, clock, events=(late,), quiet=quiet)
    w.tick()
    clock.now = at(23, 30)
    assert w.tick() is None
    assert list(w._state["held"])                  # parked, not forgotten
    # Morning, quiet lifted.
    quiet.should_hold = lambda: False
    quiet.reason = lambda: ""
    clock.now = at(7, 30, DAY + timedelta(days=1))
    cand = w.tick()
    assert cand is not None and cand.word == "final"


def test_a_hold_does_not_last_forever(tmp_path):
    late = _ev("BIOSENSORS Final", at(21, 0), minutes=120)
    quiet = SimpleNamespace(should_hold=lambda: True, reason=lambda: "quiet hours")
    clock = SimpleNamespace(now=at(20, 0))
    w = _watch(tmp_path, clock, events=(late,), quiet=quiet)
    w.tick()
    clock.now = at(23, 30)
    w.tick()
    quiet.should_hold = lambda: False
    clock.now = at(20, 0, DAY + timedelta(days=1))   # 21 hours later
    assert w.tick() is None


def test_he_is_not_asked_while_he_is_out(tmp_path):
    presence = SimpleNamespace(configured=True, state="away")
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock, presence=presence)
    w.tick()
    clock.now = at(15, 10)
    assert w.tick() is None
    presence.state = "home"
    assert w.tick() is not None


def test_unconfigured_presence_counts_as_home(tmp_path):
    """Correction: phone_ip is unset on this box, so "known to be away" is
    the gate. The other way round the feature would never fire at all."""
    presence = SimpleNamespace(configured=False, state="unknown")
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock, presence=presence)
    w.tick()
    clock.now = at(15, 10)
    assert w.tick() is not None


def test_it_can_be_switched_off(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock, cfg=Cfg({"debrief": {"enabled": False}}))
    assert w.tick() is None


def test_the_callback_gets_the_candidate(tmp_path):
    seen = []
    clock = SimpleNamespace(now=at(12, 0))
    w = DebriefWatch(cfg=Cfg({}), get_calendar=Cal([MIDTERM]),
                     state_path=tmp_path / "d.json",
                     now=lambda tz=None: clock.now, on_candidate=seen.append)
    w.tick()
    clock.now = at(15, 10)
    w.tick()
    assert len(seen) == 1 and seen[0].title == "BIOSENSORS Midterm 1"


def test_a_callback_that_raises_does_not_kill_the_tick(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))

    def boom(cand):
        raise RuntimeError("no speech path here")
    w = DebriefWatch(cfg=Cfg({}), get_calendar=Cal([MIDTERM]),
                     state_path=tmp_path / "d.json",
                     now=lambda tz=None: clock.now, on_candidate=boom)
    w.tick()
    clock.now = at(15, 10)
    assert w.tick() is not None


# --------------------------------------------------------------- state
def test_a_hand_edited_state_file_does_not_raise(tmp_path):
    (tmp_path / "debrief.json").write_text(
        '{"seen": {"k": null, "ok": "2026-09-14T13:00:00-05:00"},'
        ' "asked": 7, "held": {"h": "2026-09-14T13:00:00-05:00"}}')
    clock = SimpleNamespace(now=at(15, 10))
    w = _watch(tmp_path, clock)
    assert w.seen == {"ok": "2026-09-14T13:00:00-05:00"}
    assert w.asked == {}


def test_the_state_write_is_atomic(tmp_path):
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock)
    w.tick()
    assert json.loads((tmp_path / "debrief.json").read_text())["seen"]
    assert not (tmp_path / "debrief.json.tmp").exists()


def test_asked_outlives_seen(tmp_path):
    """`seen` is working memory; `asked` is a promise, and it is kept."""
    clock = SimpleNamespace(now=at(12, 0))
    w = _watch(tmp_path, clock)
    w.tick()
    clock.now = at(15, 10)
    w.mark_asked(w.tick().key)
    clock.now = at(12, 0, DAY + timedelta(days=10))
    w.tick()
    assert w.seen == {} and w.asked


# ---------------------------------------------------------------- filing
def _cand():
    return Candidate("k", "BIOSENSORS Midterm 1", "midterm", at(14, 30))


def test_the_answer_is_filed_to_both_sinks_and_never_to_the_model():
    rows, facts = [], {}
    context = SimpleNamespace(
        journal_debrief=lambda title, word, text, ended: rows.append(
            (title, word, text, ended)),
        # A debrief must never become a conversational turn: an exchange
        # would be answered with sympathy and forgotten at the end of the
        # follow-up window, which is the opposite of the point.
        add_exchange=lambda *a, **k: pytest.fail("a debrief was routed as chat"))
    memory = SimpleNamespace(remember=lambda k, v: facts.__setitem__(k, v))
    assert file_answer("it went badly, I ran out of time", _cand(),
                       memory=memory, context=context)
    assert rows[0][:3] == ("BIOSENSORS Midterm 1", "midterm",
                           "it went badly, I ran out of time")
    assert facts[fact_key("BIOSENSORS Midterm 1", at(14, 30))] == \
        "it went badly, I ran out of time"


def test_an_empty_answer_files_nothing():
    assert not file_answer("   ", _cand(), memory=SimpleNamespace(), context=None)


def test_one_sink_failing_does_not_lose_the_other():
    facts = {}
    context = SimpleNamespace(journal_debrief=lambda *a, **k: 1 / 0)
    memory = SimpleNamespace(remember=lambda k, v: facts.__setitem__(k, v))
    assert file_answer("fine", _cand(), memory=memory, context=context)
    assert facts


def test_the_filed_line_matches_the_day_review_counter():
    from jarvis.dayreview import COUNTERS
    rx = dict(COUNTERS)["debriefs"]
    assert rx.search("debrief filed for 'BIOSENSORS Midterm 1': it went badly")


def test_the_journal_renders_and_pins_a_debrief():
    from jarvis.tools.journal import digest
    rows = [{"kind": "debrief", "_when": datetime(2026, 9, 14, 14, 40),
             "title": "BIOSENSORS Midterm 1", "word": "midterm",
             "text": "it went badly, I ran out of time on the last question"}]
    out = digest(rows)
    assert "debrief — BIOSENSORS Midterm 1: it went badly" in out
    # …and it survives the collapse that eats the oldest hours first.
    noise = [{"kind": "exchange", "_when": datetime(2026, 9, 14, 15, m % 60),
              "user": "x" * 200, "jarvis": "y" * 200} for m in range(200)]
    tight = digest(rows + noise, max_chars=2000)
    assert "debrief — BIOSENSORS Midterm 1" in tight


# ------------------------------------------- who owns the next words
# The debrief is the one pending question hoisted ABOVE commander.handle,
# so it jumps the queue every other open question waits in. Two failures
# came out of that: a bare "stop" said to a ringing alarm inside the 120 s
# debrief window was filed as how the midterm went (commander._try_ringing
# runs BELOW the debrief filter and "stop" is in no ASSISTANT_TIER1
# matcher), and a flashcard answer said mid-quiz was filed the same way
# while the quiz never advanced. floor_holder is the debrief's side of the
# yield: it must name every holder, and it must never invent one.
import jarvis.app as _app_mod                                    # noqa: E402
from jarvis.debrief import FLOOR_FREE, floor_holder              # noqa: E402


class _FloorApp:
    """The slice of JarvisApp floor_holder actually probes, with the REAL
    _question_open bound in -- the TTL-aware predicate app.py already owned
    and wired into the mic window but never into the debrief gate."""
    _question_open = _app_mod.JarvisApp._question_open

    def __init__(self, **kw):
        self.timekeeper = kw.pop("timekeeper", None)
        self._pending_uncertain = kw.pop("uncertain", None)
        self.commander = kw.pop("commander", SimpleNamespace(
            _pending_session=None, _pending_leave=None,
            _pending_quiz=None, _pending_destructive=None))
        self.services = kw.pop("services", SimpleNamespace(alarm_offer=None))
        assert not kw, kw


def test_a_free_floor_is_free():
    assert floor_holder(_FloorApp()) == FLOOR_FREE
    assert floor_holder(None) == FLOOR_FREE


def test_a_ringing_alarm_owns_the_next_words():
    """The regression: "stop" inside the debrief window was written into
    the episodic record while the alarm went on ringing."""
    app = _FloorApp(timekeeper=SimpleNamespace(ringing=SimpleNamespace(id=1)))
    assert floor_holder(app) == "a ringing alarm"
    app.timekeeper = SimpleNamespace(ringing=None)
    assert floor_holder(app) == FLOOR_FREE


def test_the_timekeeper_is_also_found_on_the_services_namespace():
    app = _FloorApp()
    app.timekeeper = None
    app.services = SimpleNamespace(alarm_offer=None,
                                   timekeeper=SimpleNamespace(ringing=object()))
    assert floor_holder(app) == "a ringing alarm"


def test_a_live_flashcard_owns_the_next_words():
    """debrief.INTERVAL_S (300 s) and quiz ANSWER_WINDOW_S (300 s) overlap
    by design, so a tick landing on a card on the table is routine."""
    quiz = SimpleNamespace(finished=False, stale=lambda: False)
    app = _FloorApp(commander=SimpleNamespace(
        _pending_session=None, _pending_leave=None,
        _pending_quiz=quiz, _pending_destructive=None))
    assert floor_holder(app) == "an open question"
    quiz.stale = lambda: True                    # expired: the floor is free
    assert floor_holder(app) == FLOOR_FREE


def test_a_destructive_read_back_owns_the_next_words():
    import time as _t
    pend = ("delete", object(), _t.monotonic())
    app = _FloorApp(commander=SimpleNamespace(
        _pending_session=None, _pending_leave=None,
        _pending_quiz=None, _pending_destructive=pend))
    assert floor_holder(app) == "an open question"
    app.commander._pending_destructive = ("delete", object(), _t.monotonic() - 10_000)
    assert floor_holder(app) == FLOOR_FREE       # stale: no longer a floor


def test_a_working_session_and_the_leave_question_own_the_next_words():
    import time as _t
    from jarvis.commander import LEAVE_ANSWER_WINDOW_S
    cmdr = SimpleNamespace(_pending_session=SimpleNamespace(finished=False),
                           _pending_leave=None, _pending_quiz=None,
                           _pending_destructive=None)
    app = _FloorApp(commander=cmdr)
    assert floor_holder(app) == "a working session"
    cmdr._pending_session = SimpleNamespace(finished=True)
    assert floor_holder(app) == FLOOR_FREE
    # "twelve minutes" is the shape of BOTH answers, so whichever filter
    # runs first wins the wrong one.
    cmdr._pending_leave = ("Wisenbaker", "Wisenbaker", _t.monotonic())
    assert floor_holder(app) == "the leave-time question"
    cmdr._pending_leave = ("Wisenbaker", "Wisenbaker",
                           _t.monotonic() - LEAVE_ANSWER_WINDOW_S - 1)
    assert floor_holder(app) == FLOOR_FREE


def test_the_was_that_for_me_prompt_owns_the_next_words():
    assert floor_holder(_FloorApp(uncertain={"text": "hello"})) == \
        "the was-that-for-me prompt"


def test_a_raising_probe_reads_as_a_free_floor():
    """A spurious hold means the debrief is never asked at all, which is
    worse than one asked at a bad moment -- so every probe fails open."""
    class Boom:
        @property
        def ringing(self):
            raise RuntimeError("db on fire")

    class Angry:
        def __getattr__(self, name):
            raise RuntimeError("commander on fire")

    app = _FloorApp(timekeeper=Boom(), commander=Angry())
    assert floor_holder(app) == FLOOR_FREE
    app2 = _FloorApp()
    app2._question_open = lambda c: (_ for _ in ()).throw(RuntimeError("boom"))
    assert floor_holder(app2) == FLOOR_FREE
