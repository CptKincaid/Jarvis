"""FOUND 2026-09-02 10:10 — "what is my next class" opened a coding session.

From /tmp/vss_voice/jarvis.log, twice in five seconds::

    10:10:33.886 jarvis.commander INFO route claude (code-cue) 'what is my next class'
    10:10:33      jarvis.claude_session INFO task 20260902-101033-f268e2 [test] model=opus
    10:10:38.290 jarvis.commander INFO route claude (code-cue) 'what is my next class'

A billed Opus session, which then told him it had no access to his
schedule.  Two independent defects stacked.

**The router.**  An engineering student's diary is written in
``CODE_OBJECTS``: class, classes, lab, test, module.  ``code_cues`` sets
``code_strong`` whenever ``_QUESTION_OBJECT_RX`` matches and the sentence
opens with a wh-word -- which every diary question does -- with no coding
verb anywhere in it.  Measured before the fix, nine phrasings became
``claude (code-cue)``, four became ``local (local:code-lookup)`` (worse:
``code_paths(cfg)`` falls back to ``claude.allowed_dirs`` = ~/Jarvis, so
the index would have answered a lecture question out of his own Python),
three dead-ended in the tie-break, and "what time is my class" never
reached the router at all -- ``clock_kind`` claimed it at REGISTRY index 4
and answered "It's 10:25 in the morning, sir."

The nouns cannot come out of the code tables: "what does this class do"
and "what is in that module" have no other cue and would collapse to the
tie-break.  What separates the readings is POSSESSION plus a DIARY FRAME
(``router.class_diary``).

**The answer.**  ``courses.next_class`` (jarvis/courses.py:136) existed
with ZERO callers anywhere in the repo, and ``get_calendar(range="next")``
is no substitute -- it answers the next EVENT of any kind.  Measured
against his live cache that morning: at 10:25 both agree, but at 09:00 the
next day ``format_events(evs, "next")`` says "Hunter Peyrovi and
ValerieAnne Staffeldt ... tamu.zoom.us" (an MBA admissions Zoom) while
``next_class`` says ELECTRICAL DESIGN LAB II.
"""
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.router import Router, class_diary
from jarvis.tools.location import clock_words


# ------------------------------------------------------------------ router
# His own words, transcribed the way he says them.  Every one of these
# routed to Claude, to the code index or to the tie-break before the fix.
HIS_TIMETABLE = [
    "what is my next class",
    "what's my next class",
    "when is my next class",
    "when is my class today",
    "do I have class tomorrow",
    "when is my next test",
    "when is my biosensors class",
    "is my class cancelled",
    "when does my class start",
    "what do I have after this class",
    "what class do I have next",
    "what class is next",
    "where is my class",
    "what are my classes today",
    "do I have any classes today",
    "how many classes do I have today",
    "what time is my class",
    "when is my lab",
    "when is my next lab session",
    "what is my next lecture",
    "when is my electrical design lab",
    "what time is my next class",
    "where is my next class",
    "what's my next lecture",
]

# The record the fix must not regress.  Three of these turn on exactly the
# words that had to change: "what does this class do" and "what is in that
# module" have NO cue other than the code object.
STILL_CLAUDE = [
    "what does this class do",
    "what is in that module",
    "why is the test failing",
    "explain this function",
    "run the tests",
    "fix the failing test",
    "add a test for the router",
    "fix the weather module",
    "fix the weather handler",
    "fix the mail parser",
    "why is my test failing",
    "write a test for my parser",
    "how long do my tests take",
]

# The local code index keeps its own lookups (router.is_code_lookup).
STILL_THE_CODE_INDEX = [
    "what class handles the router",
    "what module owns the mic arbiter",
    "which file has the speaker gate",
    "where is the router module",
]


@pytest.mark.parametrize("text", HIS_TIMETABLE)
def test_his_own_class_is_a_lecture_not_a_python_class(text):
    d = Router(None, classify=None).route(text)
    assert d.kind == "local", (text, d.kind, d.reason)
    assert d.reason != "local:code-lookup", (text, d.reason)


@pytest.mark.parametrize("text", STILL_CLAUDE)
def test_a_real_coding_question_still_goes_to_claude(text):
    d = Router(None, classify=None).route(text)
    assert d.kind == "claude" and d.reason == "code-cue", (text, d.kind, d.reason)


@pytest.mark.parametrize("text", STILL_THE_CODE_INDEX)
def test_the_code_index_keeps_its_own_lookups(text):
    d = Router(None, classify=None).route(text)
    assert d.reason == "local:code-lookup", (text, d.kind, d.reason)


def test_the_exam_family_is_not_forced_onto_get_calendar():
    """"when is my next exam" must keep reaching the full tool loop: the
    Tier-1 exam handler returns None with no Canvas token so the model can
    call canvas_due, whose setup line names what is missing."""
    d = Router(None, classify=None).route("when is my next exam")
    assert d.kind == "local"
    assert commander.forced_call(d.reason, "when is my next exam") is None


@pytest.mark.parametrize("text,want", [
    ("what is my next class", True),
    ("when is my class today", True),
    ("what does this class do", False),
    ("fix my failing test", False),
    ("what is my lab", False),          # no diary frame: already local:question
    ("rerun my class tests in tests/test_router.py", False),
])
def test_class_diary_needs_possession_and_a_frame(text, want):
    assert class_diary(text) is want, text


# FOUND in review, all measured base(a414152) -> HEAD as `claude code-cue`
# -> `local local:calendar`.  Two holes: three branches of _CLASS_MINE_RX
# are built on "do i have", which _CLASS_WHEN_RX also carried, so one
# substring satisfied both halves of a two-part test; and the class_diary
# early return in code_cues fires before verbs, objects or paths are
# examined, so "file", "module", "repo", "lines", "subclass" and
# "decorator" never got a vote.
STILL_CODING_DESPITE_THE_WORD_CLASS = [
    "how many classes do I have in this file",
    "do I have a class for the mixer",
    "do I have a class for that yet",
    "do I have any classes without tests",
    "when does my class start in the file",
    "how long is my class in lines",
    "how many lines does my class have",
    "what do I have after this class in the module",
    "is my class cancelled by the decorator",
    "how many labs do I have in the repo",
    "do I have a course module",
    "do I have any classes to clean up today",
    "which classes do I have that subclass Command",
]


@pytest.mark.parametrize("text", STILL_CODING_DESPITE_THE_WORD_CLASS)
def test_a_coding_sentence_that_says_class_is_not_a_timetable(text):
    assert class_diary(text) is False, text
    assert Router(None, classify=None).route(text).reason != "local:calendar"


def test_a_reminder_whose_tail_names_a_class_stays_a_reminder():
    """The class_diary early return sat ahead of the wrapper table, so
    "remind me to email my professor after my class" was relabelled
    local:timekeeper -> local:calendar (measured).  local_tool_clause reads
    that reason and _compound_hijack makes a Tier-1 command stand down when
    the other clause names a local tool."""
    text = "remind me to email my professor after my class"
    d = Router(None, classify=None).route(text)
    assert d.kind == "local" and d.reason == "local:timekeeper", d
    assert commander.local_tool_clause("after my class") == "calendar"


# ------------------------------------------------------- the answer layer
NOW = datetime.now().astimezone()
WISENBAKER = "College Station Wisenbaker Engineering Bldg 049"
ETB = "College Station Emerging Technologies Building 1003"


def _ev(title, start, location="", all_day=False):
    return types.SimpleNamespace(title=title, start=start,
                                 end=start + timedelta(hours=1),
                                 all_day=all_day, location=location,
                                 description="", calendar="icloud")


def _course(title, start, location="", weeks=3):
    """The same slot on ``weeks`` distinct dates -- what makes a title a
    course to jarvis/courses.recurring_courses (MIN_DATES = 2)."""
    return [_ev(title, start + timedelta(days=7 * i), location)
            for i in range(-1, weeks - 1)]


def _cal(events):
    return types.SimpleNamespace(configured=True, events=lambda: list(events))


def _commander(calendar):
    c = object.__new__(Commander)
    c.services = types.SimpleNamespace(calendar=calendar)
    return c


def _ask(calendar, text):
    m = commander._NEXT_CLASS_RX.match(text)
    assert m, text
    return commander._h_next_class(_commander(calendar), text, m)


# -- the matcher ------------------------------------------------------
@pytest.mark.parametrize("text", [
    "what is my next class",
    "what's my next class",
    "when is my next class",
    "what time is my next class",
    "what time is my class",
    "where's my next lecture",
    "when is my biosensors class",
    "when is my electrical design lab",
    "when is my next lab session",
    "what class do I have next",
    "what class is next",
    "when does my class start",
    "when does my next class start",
    "what's my next class?",
])
def test_the_next_class_matcher_claims_his_phrasings(text):
    assert commander._NEXT_CLASS_RX.match(text), text


@pytest.mark.parametrize("text", [
    "what time is it",
    "what does this class do",
    "what class handles the router",
    "when is my next exam",
    "when is my next quiz",
    "when is my next meeting",
    "what's on my calendar",
    "what is my next class assignment",
    # A DAY is not "next": the handler read neither, so frozen at Wed
    # 08:00 on his real cache "when is my class tomorrow" answered
    # 'BIOSENSORS in an hour, at 9:10 am' -- that class is TODAY -- and
    # frozen on Saturday "when is my class today" named Monday's.  The
    # router's local:calendar route and get_calendar honour a range.
    "when is my class today",
    "when is my class tomorrow",
    "when is my class tonight",
    "what time is my class tomorrow",
    "when is my lab this afternoon",
    # ...and a LIST question is not one row.
    "what are my classes today",
    "what are my classes",
    "when are my classes",
])
def test_the_next_class_matcher_leaves_the_clock_and_code_alone(text):
    assert not commander._NEXT_CLASS_RX.match(text), text


@pytest.mark.parametrize("text,kind", [
    ("when is my next lab", "lab"),
    ("what is my next lecture", "lecture"),
    ("when is my next lab session", "lab"),
    ("where is my lab", "lab"),
    ("what is my next seminar", "seminar"),
    ("what is my next class", ""),
    ("what are my classes", None),          # no match at all
    ("what class is next", ""),
])
def test_the_kind_noun_he_said_is_captured(text, kind):
    """It was thrown away, and "when is my next lab" answered with the
    12:40 lecture."""
    m = commander._NEXT_CLASS_RX.match(text)
    if kind is None:
        assert m is None, text
        return
    assert m, text
    got = commander._class_kind(m.group("k1") or m.group("k2") or
                                m.group("k3") or "")
    assert got == kind, (text, got)


# -- the answer -------------------------------------------------------
def test_the_next_class_is_named_with_its_time_and_room():
    start = NOW + timedelta(hours=2)
    events = _course("MAGNETIC RESONANCE ENGR", start, ETB)
    res = _ask(_cal(events), "what is my next class")
    assert res.speak and res.handled
    assert res.reply == (
        "Your next class is MAGNETIC RESONANCE ENGR in 2 hours, "
        f"at {clock_words(start)}, in Emerging Technologies 1003, sir.")


def test_the_nearer_of_two_courses_wins():
    near = NOW + timedelta(hours=2)
    far = NOW + timedelta(hours=5)
    events = _course("BIOSENSORS", near, WISENBAKER) + \
        _course("ELECTRICAL DESIGN LAB II", far, ETB)
    res = _ask(_cal(events), "when is my next class")
    assert "BIOSENSORS" in res.reply
    assert "ELECTRICAL" not in res.reply
    assert "Wisenbaker 049" in res.reply


def test_an_all_day_entry_is_never_offered_as_the_next_class():
    """An all-day row sorts ahead of every timed one.  courses.slot()
    refuses it, which is why next_class is the right lookup and
    get_calendar(range="next") is not."""
    tomorrow = (NOW + timedelta(days=1)).replace(hour=9, minute=10,
                                                 second=0, microsecond=0)
    holiday = [_ev("READING DAY", tomorrow.replace(hour=0, minute=0),
                   all_day=True),
               _ev("READING DAY", tomorrow.replace(hour=0, minute=0) +
                   timedelta(days=7), all_day=True)]
    res = _ask(_cal(holiday + _course("BIOSENSORS", tomorrow, WISENBAKER)),
               "what is my next class")
    assert "READING DAY" not in res.reply
    assert "BIOSENSORS" in res.reply


def test_a_timetable_that_has_run_out_says_so_rather_than_inventing_one():
    past = NOW - timedelta(days=1)
    events = _course("BIOSENSORS", past, WISENBAKER, weeks=2)
    res = _ask(_cal(events), "what is my next class")
    assert res.reply == commander.NO_CLASS_LINE
    assert "BIOSENSORS" not in res.reply


def test_no_calendar_falls_through_instead_of_claiming_an_empty_schedule():
    """Nothing to read is not the same as nothing on: the router's model
    turn reaches get_calendar, whose setup line says what is missing."""
    assert _ask(None, "what is my next class") is None
    assert _ask(_cal([]), "what is my next class") is None


def test_a_calendar_with_no_recurring_course_falls_through():
    one_off = [_ev("Dentist", NOW + timedelta(hours=1), "")]
    assert _ask(_cal(one_off), "what is my next class") is None


def test_a_named_course_answers_for_that_course_not_the_nearer_one():
    near = NOW + timedelta(hours=2)
    far = NOW + timedelta(hours=5)
    events = _course("MAGNETIC RESONANCE ENGR", near, ETB) + \
        _course("BIOSENSORS", far, WISENBAKER)
    res = _ask(_cal(events), "when is my biosensors class")
    assert res.reply == (
        "Your next BIOSENSORS is in 5 hours, "
        f"at {clock_words(far)}, in Wisenbaker 049, sir.")


def test_a_course_he_does_not_have_falls_through_rather_than_naming_another():
    events = _course("BIOSENSORS", NOW + timedelta(hours=2), WISENBAKER)
    assert _ask(_cal(events), "when is my thermodynamics class") is None
    # A day worn as a modifier reads as a course name and degrades the same
    # way, so no day-scoped phrasing survives to be answered undated.
    assert _ask(_cal(events), "when is my monday class") is None


def test_a_lab_question_is_answered_with_the_lab_not_the_nearer_lecture():
    """FOUND in review, reproduced on his real cache (30 events, courses
    BIOSENSORS / MAGNETIC RESONANCE ENGR / ELECTRICAL DESIGN LAB II) at Wed
    2026-09-02 11:03: "when is my next lab" answered 'Your next class is
    MAGNETIC RESONANCE ENGR ... at 12:40 pm', a LECTURE, while his own
    ELECTRICAL DESIGN LAB II sat at 16:10.  The matched noun was discarded
    and every course was a candidate."""
    lecture = NOW + timedelta(hours=2)
    lab = NOW + timedelta(hours=5)
    events = _course("MAGNETIC RESONANCE ENGR", lecture, ETB) + \
        _course("ELECTRICAL DESIGN LAB II", lab, WISENBAKER)
    for text in ("when is my next lab", "when is my next lab session",
                 "where is my lab", "what's my next lab"):
        res = _ask(_cal(events), text)
        assert res is not None, text
        assert "ELECTRICAL DESIGN LAB II" in res.reply, (text, res.reply)
        assert "MAGNETIC" not in res.reply, (text, res.reply)
        assert res.reply.startswith("Your next lab is"), res.reply
    # ...and the generic word still means any course.
    res = _ask(_cal(events), "what is my next class")
    assert "MAGNETIC RESONANCE ENGR" in res.reply


def test_a_kind_he_has_no_course_for_falls_through_rather_than_guessing():
    """Same honest degradation an unknown course name already gets: none of
    his three courses is titled "lecture" or "seminar", so the model looks
    instead of the timetable naming a lab."""
    events = _course("ELECTRICAL DESIGN LAB II", NOW + timedelta(hours=2), ETB)
    assert _ask(_cal(events), "what is my next lecture") is None
    assert _ask(_cal(events), "when is my next seminar") is None
    assert _ask(_cal(events), "when is my next tutorial") is None


def test_a_named_lab_course_still_wins_over_the_kind_filter():
    """"my electrical design lab" names a course; the kind is a fallback."""
    events = _course("BIOSENSORS", NOW + timedelta(hours=1), WISENBAKER) + \
        _course("ELECTRICAL DESIGN LAB II", NOW + timedelta(hours=5), ETB)
    res = _ask(_cal(events), "when is my electrical design lab")
    assert res.reply.startswith("Your next ELECTRICAL DESIGN LAB II is")


def test_a_class_in_progress_answers_about_the_room_he_is_standing_in():
    """FOUND in review.  Frozen Wed 12:50, ten minutes into his 12:40-13:30
    MAGNETIC RESONANCE ENGR in ETB 1003, "where is my class" answered
    'ELECTRICAL DESIGN LAB II ... in Emerging Technologies 1020' -- a
    different room from the one he is in and asking about."""
    started = NOW - timedelta(minutes=10)
    later = NOW + timedelta(hours=3)
    events = _course("MAGNETIC RESONANCE ENGR", started, ETB) + \
        _course("ELECTRICAL DESIGN LAB II", later, WISENBAKER)
    res = _ask(_cal(events), "where is my class")
    assert res.reply == (
        "MAGNETIC RESONANCE ENGR is on now until "
        f"{clock_words(started + timedelta(hours=1))}, "
        "in Emerging Technologies 1003, sir.")
    # ...but "NEXT" is still the next one, not the one he is sitting in.
    nxt = _ask(_cal(events), "what is my next class")
    assert "ELECTRICAL DESIGN LAB II" in nxt.reply


def test_a_sooner_unlisted_session_hands_the_turn_to_the_calendar():
    """courses.recurring_courses needs MIN_DATES = 2, so a one-off sitting
    is invisible and a LATER class was named "your next class" with no word
    about it.  Probed: one SENIOR DESIGN SEMINAR at now+1h beside a
    recurring lecture at now+2h answered with the lecture."""
    seminar = _ev("SENIOR DESIGN SEMINAR", NOW + timedelta(hours=1),
                  "Zachry 110")
    events = [seminar] + _course("MAGNETIC RESONANCE ENGR",
                                 NOW + timedelta(hours=2), ETB)
    assert _ask(_cal(events), "what is my next class") is None


def test_a_canvas_assignment_row_is_not_a_sitting():
    """Every Canvas row in his cache wears the section tag -- 'Prelab for
    Lab 3 (canvas quiz) [BMEN-427:501,502,503,504,BME...]' is timed at
    exactly his 12:40 slot -- and coursework must not cost him a model
    turn on "what is my next class"."""
    prelab = _ev("Prelab for Lab 3 (canvas quiz) [BMEN-427:501,502]",
                 NOW + timedelta(hours=1))
    events = [prelab] + _course("MAGNETIC RESONANCE ENGR",
                                NOW + timedelta(hours=2), ETB)
    res = _ask(_cal(events), "what is my next class")
    assert res is not None and "MAGNETIC RESONANCE ENGR" in res.reply


def test_a_naive_datetime_in_the_feed_degrades_instead_of_raising():
    """courses.recurring_courses sorts on each slot's first start, so ONE
    naive datetime raised TypeError straight out of Commander.handle.  A
    source that cannot be read is the model's problem."""
    naive = [_ev("BIOSENSORS", datetime.now() + timedelta(hours=1)),
             _ev("BIOSENSORS", datetime.now() + timedelta(days=7, hours=1))]
    aware = _course("MAGNETIC RESONANCE ENGR", NOW + timedelta(hours=2), ETB)
    assert _ask(_cal(naive + aware), "what is my next class") is None


def test_a_class_on_a_link_says_the_class_without_inventing_a_room():
    start = NOW + timedelta(hours=2)
    events = _course("BIOSENSORS", start, "https://tamu.zoom.us/j/1234")
    res = _ask(_cal(events), "what is my next class")
    assert res.reply.endswith(f"at {clock_words(start)}, sir.")


# -------------------------------------------------------------- the wiring
def test_the_next_class_command_is_registered_before_the_clock():
    """REGISTRY index 4 is Command("clock", ...), and clock_kind claims
    "what time is my class" -- live, "It's 10:25 in the morning, sir."."""
    names = [c.name for c in commander.REGISTRY]
    assert names.index("next class") < names.index("clock")
    assert "next class" in [c.name for c in commander.ASSISTANT_TIER1]


@pytest.mark.parametrize("text", [
    "what time is my class",
    "what time is my next lecture",
    "what time is my exam",
    "what time is my meeting",
    # FOUND in review: the guard was wired only for the word "time", so
    # "what date is my exam" still answered with today's wall date -- one
    # word from the phrasing it did catch.  Built from the same
    # time|date|day alternation _CLOCK_KINDS carries.
    "what date is my exam",
    "what date is my lab",
    "what date is my class",
])
def test_the_wall_clock_declines_a_question_about_his_own_diary(text):
    """Belt and braces for the box with no calendar, where the Tier-1
    answer falls through: "It's 10:25 in the morning, sir." is a confident
    wrong answer whether or not the schedule can be read."""
    assert commander.clock_kind(text) is None, text


@pytest.mark.parametrize("text", [
    "what time is it", "what's the time", "what time is it in London",
    # FOUND in review: `.*?` between "what time" and the possessive let a
    # room named LATER in the sentence disable the wall clock, and these
    # two really are asking the current time.  Both measured 'time' ->
    # None; forced_call finds no city either, so they reached brain.chat.
    "what time is it in the lab",
    "what time is it now that the lab is over",
    "what time is it before my class",
])
def test_the_wall_clock_still_answers_the_clock(text):
    assert commander.clock_kind(text) == ("time" if "London" not in text
                                          else None)


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(),
        calendar=_cal(_course("BIOSENSORS", NOW + timedelta(hours=2),
                              WISENBAKER)))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    return Commander(svc)


def test_what_time_is_my_class_no_longer_gets_the_wall_clock(cmdr):
    res = cmdr.handle("what time is my class", source="typed")
    assert res.status != "Clock"
    assert "BIOSENSORS" in res.reply
    cmdr.services.brain.chat.assert_not_called()


def test_the_prefixed_form_is_answered_by_the_same_command(cmdr):
    res = cmdr.handle("Jarvis, what's my next class", source="voice")
    assert "BIOSENSORS" in res.reply
