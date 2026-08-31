"""Time to leave (jarvis/leavetime.py) and its three commander doors.

The building normaliser is the feature's correctness core and is tested
against the real shapes his calendar writes -- including the Zoom URL and
the empty locations, which must NEVER be asked about. Everything else (the
table, the ask-once, the filed reminder, "make that ten next time") hangs
off it.
"""
from __future__ import annotations

import json
import time
import types
from datetime import datetime, timedelta, timezone

import pytest

from jarvis import commander
from jarvis import leavetime as lt_mod
from jarvis.commander import (ASSISTANT_TIER1, LEAVE_ANSWER_WINDOW_S,
                              LEAVE_DROPPED_LINE, REGISTRY, Commander,
                              CommandResult)
from jarvis.leavetime import (LeadTable, LeaveTimes, answer_minutes,
                              building_key, leave_amend_kind, leave_line,
                              leave_query_kind, leave_set_kind, match_key,
                              parse_minutes, speech_name)
from jarvis.tools.calendar import Event

# The exact strings in ~/.cache/jarvis/calendar_cache.json (2026-08-30).
LIVE_LOCATIONS = [
    "College Station Wisenbaker Engineering Bldg 049",
    "College Station Emerging Technologies Building 1003",
    "College Station Emerging Technologies Building 1020",
    "College Station Zachry Engineering Ed. Complex 330",
    "College Station Zachry Engineering Ed. Complex 110",
    "College Station Jack E. Brown Chem Engn Bldg 731A",
    "https://tamu.zoom.us/j/94324046592?pwd=Ic6ifN6gKbxvhRCvfFypmo6hSM0kYw.1",
    "",
    "",
]
TZ = timezone(timedelta(hours=-5))
DAY = datetime(2026, 8, 30, 8, 0, tzinfo=TZ)


# ------------------------------------------------------------- the key
def test_the_live_locations_collapse_to_exactly_four_buildings():
    keys = [building_key(loc) for loc in LIVE_LOCATIONS]
    assert keys.count(None) == 3                  # the Zoom URL and two blanks
    assert sorted(set(k for k in keys if k)) == [
        "Emerging Technologies Building",
        "Jack E. Brown Chem Engn Bldg",
        "Wisenbaker Engineering Bldg",
        "Zachry Engineering Ed. Complex",
    ]


def test_two_rooms_in_one_building_are_one_walk():
    assert building_key(LIVE_LOCATIONS[1]) == building_key(LIVE_LOCATIONS[2])
    assert building_key(LIVE_LOCATIONS[3]) == building_key(LIVE_LOCATIONS[4])


@pytest.mark.parametrize("loc", [
    "", "   ", None,
    "https://tamu.zoom.us/j/94324046592",
    "http://meet.example/x",
    "www.zoom.us/j/123",
])
def test_a_url_or_an_empty_location_is_never_a_building(loc):
    """Asking "how long is the walk to https://tamu.zoom.us/..." even once
    is the failure that kills trust in the feature."""
    assert building_key(loc) is None


def test_a_trailing_room_token_comes_off_but_a_name_does_not():
    assert building_key("Wisenbaker 049") == "Wisenbaker"
    assert building_key("Jack E. Brown Chem Engn Bldg 731A") == \
        "Jack E. Brown Chem Engn Bldg"
    assert building_key("Building 8") == "Building"
    assert building_key("Halbouty") == "Halbouty"


def test_the_spoken_name_drops_what_a_building_is_and_keeps_which_one():
    assert speech_name("Wisenbaker Engineering Bldg") == "Wisenbaker"
    assert speech_name("Emerging Technologies Building") == "Emerging Technologies"
    assert speech_name("Zachry Engineering Ed. Complex") == "Zachry"
    assert speech_name("Jack E. Brown Chem Engn Bldg") == "Jack E. Brown"
    assert speech_name("Building") == "Building"      # never emptied


def test_match_key_finds_the_building_he_names():
    keys = [building_key(loc) for loc in LIVE_LOCATIONS if building_key(loc)]
    assert match_key("wisenbaker", keys) == "Wisenbaker Engineering Bldg"
    assert match_key("Emerging Technologies", keys) == "Emerging Technologies Building"
    assert match_key("the moon", keys) is None


# -------------------------------------------------------------- durations
@pytest.mark.parametrize("text,minutes", [
    ("about ten minutes", 10), ("15", 15), ("five", 5),
    ("it takes 4 minutes", 4), ("a quarter of an hour", 15),
    ("twenty five minutes", 25), ("half an hour", 30), ("2 hours", 120),
])
def test_durations_parse(text, minutes):
    assert parse_minutes(text) == minutes


@pytest.mark.parametrize("text", ["no idea", "", "zero minutes", "300 minutes"])
def test_a_non_duration_is_none(text):
    assert parse_minutes(text) is None


@pytest.mark.parametrize("text,minutes", [
    ("ten minutes", 10), ("about ten minutes or so", 10), ("five", 5),
    ("it's a 12 minute walk", 12), ("maybe 20 minutes sir", 20),
])
def test_an_answer_is_a_duration_and_little_else(text, minutes):
    assert answer_minutes(text) == minutes


@pytest.mark.parametrize("text", [
    "set a timer for five minutes",
    "remind me in ten minutes to call mum",
    "how long until my next break",
    "what's on my calendar",
])
def test_a_command_containing_minutes_is_not_an_answer(text):
    """The question stays open for minutes; a loose match would steal a
    timer out of the middle of it."""
    assert answer_minutes(text) is None


def test_the_heads_up_line_reads_like_the_pitch():
    assert leave_line(4, "Wisenbaker", 12) == \
        "You want to be walking in 4 minutes, sir; Wisenbaker is a 12 minute walk."
    assert leave_line(1, "Zachry", 6).startswith("You want to be walking in 1 minute,")
    assert leave_line(0, "Zachry", 6).startswith("You want to be walking now,")


# ---------------------------------------------------------- the utterances
def test_teaching_a_walk_parses():
    assert leave_set_kind("it takes ten minutes to get to Wisenbaker") == \
        ("Wisenbaker", 10)
    assert leave_set_kind("it's a 12 minute walk to the ETB") == ("the ETB", 12)
    assert leave_set_kind("the walk to Zachry is eight minutes") == ("Zachry", 8)


def test_a_half_heard_number_never_reaches_the_table():
    assert leave_set_kind("it takes ages to get to Wisenbaker") is None
    assert leave_set_kind("what time is it") is None


def test_make_that_ten_next_time_is_its_own_pattern():
    """correction_kind() reads only "no, I said X" / "not X, Y"; this is
    new work, not a free ride on that regex."""
    from jarvis.commander import correction_kind
    assert correction_kind("make that ten next time") is None
    assert leave_amend_kind("make that ten next time") == 10
    assert leave_amend_kind("make it 15") == 15
    assert leave_amend_kind("make that better") is None


def test_the_query_needs_a_destination_and_spares_how_long_left():
    assert leave_query_kind("how long to Wisenbaker") == "Wisenbaker"
    assert leave_query_kind("how far is it to the ETB") == "the ETB"
    assert leave_query_kind("how long left") is None          # focus session
    assert leave_query_kind("how long until my next break") is None


# ---------------------------------------------------------------- table
class FakeMemory:
    def __init__(self):
        self.prefs = {}

    def set_preference(self, key, value):
        self.prefs[key] = value

    def get_preference(self, key, default=None):
        return self.prefs.get(key, default)

    def get_all_preferences(self):
        return dict(self.prefs)


def test_the_table_round_trips_through_memory_preferences():
    mem = FakeMemory()
    table = LeadTable(mem)
    assert table.get("Wisenbaker Engineering Bldg") is None
    table.set("Wisenbaker Engineering Bldg", 12)
    assert mem.prefs == {"leave_lead.Wisenbaker Engineering Bldg": 12}
    assert table.get("Wisenbaker Engineering Bldg") == 12
    assert table.keys() == ["Wisenbaker Engineering Bldg"]
    table.forget("Wisenbaker Engineering Bldg")
    assert table.get("Wisenbaker Engineering Bldg") is None and table.keys() == []


def test_a_broken_memory_store_keeps_the_answer_for_the_session():
    class Boom:
        def set_preference(self, *a):
            raise RuntimeError("store on fire")

        def get_preference(self, *a, **k):
            raise RuntimeError("store on fire")
    table = LeadTable(Boom())
    table.set("Zachry Engineering Ed. Complex", 7)
    assert table.get("Zachry Engineering Ed. Complex") == 7


def test_absurd_values_are_clamped_and_junk_reads_as_unknown():
    mem = FakeMemory()
    table = LeadTable(mem)
    assert table.set("X", 9999) == lt_mod.MAX_LEAD_MIN
    mem.prefs["leave_lead.Y"] = "soon"
    assert table.get("Y") is None


# ----------------------------------------------------------------- watch
class FakeTk:
    def __init__(self):
        self.reminders = []

    def add_reminder(self, due, text):
        self.reminders.append((due, text))


class FakeCal:
    def __init__(self, events):
        self.configured = True
        self._events = list(events)

    def events(self):
        return list(self._events)


class Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def event(mins_ahead, location, title="BIOSENSORS", now=DAY):
    start = now + timedelta(minutes=mins_ahead)
    return Event(start=start, end=start + timedelta(minutes=50), title=title,
                 location=location)


def _watch(events, tmp_path, table=None, ask=None, cfg=None, now=DAY, **kw):
    cal = FakeCal(events)
    tk = FakeTk()
    w = LeaveTimes(lambda: cal, tk, table if table is not None else LeadTable(FakeMemory()),
                   ask=ask, cfg=cfg if cfg is not None else Cfg(),
                   state_path=tmp_path / "leave.json",
                   now=lambda tz=None: now.astimezone(tz) if tz else now, **kw)
    w.cal, w.tk = cal, tk
    return w


def test_a_known_walk_files_one_leave_heads_up(tmp_path):
    table = LeadTable(FakeMemory())
    table.set("Wisenbaker Engineering Bldg", 12)
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path, table=table)
    assert w.tick() == 1
    due, text = w.tk.reminders[0]
    # 40 min out, a 12 min walk, 5 min of notice -> fires in 23 minutes
    assert due == pytest.approx((DAY + timedelta(minutes=23)).timestamp())
    assert text == ("You want to be walking in 5 minutes, sir; "
                    "Wisenbaker is a 12 minute walk.")
    assert w.tick() == 0                       # filed once, never twice


def test_coming_in_late_says_how_much_notice_is_actually_left(tmp_path):
    table = LeadTable(FakeMemory())
    table.set("Wisenbaker Engineering Bldg", 12)
    w = _watch([event(15, LIVE_LOCATIONS[0])], tmp_path, table=table)
    assert w.tick() == 1
    due, text = w.tk.reminders[0]
    assert due == pytest.approx((DAY + timedelta(seconds=5)).timestamp())
    assert "walking in 3 minutes" in text


def test_a_walk_already_started_files_nothing(tmp_path):
    table = LeadTable(FakeMemory())
    table.set("Wisenbaker Engineering Bldg", 12)
    w = _watch([event(5, LIVE_LOCATIONS[0])], tmp_path, table=table)
    assert w.tick() == 0 and w.tk.reminders == []


def test_an_unknown_building_gets_no_line_at_all(tmp_path):
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    assert w.tick() == 0 and w.tk.reminders == []


def test_the_ask_lands_once_inside_the_heads_up_window(tmp_path):
    asked = []
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path,
               ask=lambda key, place: asked.append((key, place)) or True)
    w.tick()
    assert asked == [("Wisenbaker Engineering Bldg", "Wisenbaker")]
    w.tick()
    assert len(asked) == 1                     # once ever, even years later
    assert LeaveTimes(lambda: w.cal, w.tk, w.table, ask=lambda *a: True,
                      state_path=tmp_path / "leave.json",
                      now=lambda tz=None: DAY).tick() == 0


def test_the_ask_waits_until_the_heads_up_window(tmp_path):
    asked = []
    w = _watch([event(90, LIVE_LOCATIONS[0])], tmp_path,
               ask=lambda key, place: asked.append(key) or True)
    assert w.tick() == 0 and asked == []


def test_a_zoom_link_and_an_empty_location_are_never_asked_about(tmp_path):
    asked = []
    w = _watch([event(8, LIVE_LOCATIONS[6]), event(8, ""), event(8, "   ")],
               tmp_path, ask=lambda key, place: asked.append(key) or True)
    assert w.tick() == 0 and asked == []


def test_a_refused_ask_is_retried_not_recorded(tmp_path):
    """The app answers False while a turn is in flight; the question must
    still be asked later."""
    tries = []

    def refuse(key, place):
        tries.append(key)
        return False

    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path, ask=refuse)
    w.tick()
    assert tries == ["Wisenbaker Engineering Bldg"]
    w.tick()
    assert len(tries) == 2


def test_a_heads_down_moment_defers_the_ask(tmp_path):
    class Quiet:
        def should_hold(self):
            return True
    asked = []
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path,
               ask=lambda key, place: asked.append(key) or True, quiet=Quiet())
    assert w.tick() == 0 and asked == []


def test_learning_stores_the_walk_and_remembers_the_last_building(tmp_path):
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path)
    assert w.learn("Wisenbaker Engineering Bldg", 12) == 12
    assert w.table.get("Wisenbaker Engineering Bldg") == 12
    assert w.last_key == "Wisenbaker Engineering Bldg"
    stored = json.loads((tmp_path / "leave.json").read_text())
    assert stored["last_key"] == "Wisenbaker Engineering Bldg"


def test_known_keys_span_the_table_and_the_calendar(tmp_path):
    table = LeadTable(FakeMemory())
    table.set("Halbouty", 3)
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path, table=table)
    assert w.known_keys() == ["Halbouty", "Wisenbaker Engineering Bldg"]
    assert w.resolve("wisenbaker") == "Wisenbaker Engineering Bldg"
    assert w.resolve("the moon") is None


def test_a_hand_mangled_state_file_does_not_break_the_tick(tmp_path):
    (tmp_path / "leave.json").write_text(json.dumps(
        {"filed": {"a": 1}, "asked": "nope", "last_key": None}))
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    assert w.tick() == 0 and w._filed == {} and w._asked == {}


def test_the_watch_can_be_switched_off(tmp_path):
    table = LeadTable(FakeMemory())
    table.set("Wisenbaker Engineering Bldg", 12)
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path, table=table,
               cfg=Cfg({"calendar.leave_times": False}))
    assert w.tick() == 0 and w.tk.reminders == []


def test_start_stop_joins_the_thread(tmp_path):
    w = _watch([], tmp_path)
    w.start()
    assert w._thread is not None and w._thread.is_alive()
    w.start()
    w.stop()
    assert not w._thread.is_alive()


# ------------------------------------------------------------- commander
class Svc:
    def __init__(self, leavetime):
        self.leavetime = leavetime


def _commander(watch):
    c = object.__new__(Commander)
    Commander.__init__(c, Svc(watch))
    return c


def test_the_three_commands_are_tier_one_and_registered():
    names = {c.name for c in REGISTRY}
    tier1 = {c.name for c in ASSISTANT_TIER1}
    for name in ("leave time", "leave time amend", "leave time query"):
        assert name in names and name in tier1


def test_teaching_by_voice_stores_the_walk(tmp_path):
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    res = c.handle("it takes twelve minutes to get to wisenbaker", source="voice")
    assert res.handled and res.speak and "12 minutes" in res.reply
    assert w.table.get("Wisenbaker Engineering Bldg") == 12


def test_make_that_ten_next_time_edits_the_last_building(tmp_path):
    """Taught by voice, then amended in the same breath -- the amend only
    reaches back over a building THIS conversation named (see
    test_a_stale_last_key_is_never_amended)."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c.handle("it takes twelve minutes to get to wisenbaker", source="voice")
    res = c.handle("make that ten next time", source="voice")
    assert res.handled and w.table.get("Wisenbaker Engineering Bldg") == 10


def test_an_unknown_building_is_never_invented(tmp_path):
    """The handler returns None so the utterance reaches the model rather
    than storing a walk to a place he does not go."""
    w = _watch([], tmp_path)
    c = _commander(w)
    assert _h_of("leave time")(c, "it takes ten minutes to get to mordor",
                               ("mordor", 10)) is None
    assert _h_of("leave time query")(c, "how long to mordor", "mordor") is None


def _h_of(name):
    return next(cmd.handler for cmd in REGISTRY if cmd.name == name)


def test_the_query_answers_or_admits_it_does_not_know(tmp_path):
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    res = c.handle("how long to wisenbaker", source="voice")
    assert res.handled and lt_mod.UNKNOWN_LINE in res.reply
    w.learn("Wisenbaker Engineering Bldg", 12)
    res = c.handle("how long to wisenbaker", source="voice")
    assert "Wisenbaker is a 12 minute walk, sir." == res.reply


def test_asking_back_arms_the_answer_instead_of_dead_ending(tmp_path):
    """"How long to Wisenbaker?" on an unlearned building asks back -- and
    the answer must LAND. An unanswerable question is the dead end this
    repo has been bitten by before."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    res = c.handle("how long to wisenbaker", source="voice")
    assert res.handled and "How long do you need to get to Wisenbaker" in res.reply
    assert c._pending_leave is not None
    c.handle("about twelve minutes", source="voice")
    assert w.table.get("Wisenbaker Engineering Bldg") == 12


def test_the_asked_question_is_answered_by_a_plain_duration(tmp_path):
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c.ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker")
    res = c.handle("about twelve minutes", source="voice")
    assert isinstance(res, CommandResult) and res.speak
    assert w.table.get("Wisenbaker Engineering Bldg") == 12
    assert c._pending_leave is None


def test_a_real_command_inside_the_window_is_still_that_command(tmp_path):
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c.ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker")
    assert c._try_leave_answer("set a timer for five minutes") is None
    assert c._pending_leave is not None        # the question is still standing
    assert w.table.get("Wisenbaker Engineering Bldg") is None


def test_a_decline_closes_the_question_for_good(tmp_path):
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c.ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker")
    res = c._try_leave_answer("no idea")
    assert res is not None and res.reply == LEAVE_DROPPED_LINE
    assert c._pending_leave is None
    assert w.table.get("Wisenbaker Engineering Bldg") is None


def test_the_question_expires(tmp_path, monkeypatch):
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c.ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker")
    key, place, ts = c._pending_leave
    c._pending_leave = (key, place, ts - LEAVE_ANSWER_WINDOW_S - 1)
    assert c._try_leave_answer("twelve minutes") is None
    assert c._pending_leave is None


# ----------------------------------------------------- forgetting a walk
# A walk is taught from a half-heard spoken number, so a wrong one is
# routine. FORGOT_LINE and LeadTable.forget() shipped with no matcher and
# no command at all, which made a mistaught lead unrevocable by voice.
@pytest.mark.parametrize("text,place", [
    ("forget the walk to Wisenbaker", "Wisenbaker"),
    ("Forget the walk to the ETB.", "the ETB"),
    ("forget the drive to Zachry", "Zachry"),
    ("forget how long to Wisenbaker", "Wisenbaker"),
    ("forget how long it takes to get to Zachry", "Zachry"),
])
def test_the_forget_matcher_names_the_building(text, place):
    assert lt_mod.leave_forget_kind(text) == place


@pytest.mark.parametrize("text", [
    "forget it", "forget that", "scratch that", "forget the last one",
    "forget what you filed", "never mind", "how long to Wisenbaker",
    "it takes ten minutes to get to Wisenbaker",
])
def test_the_forget_matcher_leaves_every_other_dismissal_alone(text):
    """The bare undo words belong to _UNDO_RX and the no-phrases; a matcher
    that swallowed them would eat every dismissal in the app."""
    assert lt_mod.leave_forget_kind(text) is None


def test_forgetting_drops_the_walk_and_re_arms_the_ask(tmp_path):
    """LeadTable.forget() alone leaves the building unknown AND unaskable:
    `_asked` is what stops the proactive question from ever firing twice,
    so the walk could never be re-learned in passing."""
    asked = []
    table = LeadTable(FakeMemory())
    table.set("Wisenbaker Engineering Bldg", 12)
    w = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path, table=table,
               ask=lambda key, place: (asked.append(key), True)[1])
    w.tick()
    assert asked == []                          # known walks are never asked about
    w.forget("Wisenbaker Engineering Bldg")
    assert table.get("Wisenbaker Engineering Bldg") is None
    assert "Wisenbaker Engineering Bldg" not in w._asked
    w2 = _watch([event(8, LIVE_LOCATIONS[0])], tmp_path, table=table,
                ask=lambda key, place: (asked.append(key), True)[1])
    assert w2._asked == {}                      # survives the restart, too
    w2.tick()
    assert asked == ["Wisenbaker Engineering Bldg"]


def test_nothing_else_in_the_registry_claims_forget_the_walk():
    """Guard for the pending wiring: "forget the walk to X" must reach the
    leave-time door, not the garden undo, the undo window or the generic
    "forget it" dismissal."""
    from jarvis.commander import parse_yes_no, undo_kind
    text = "forget the walk to wisenbaker"

    def _hit(cmd):
        if cmd.matcher(text) is True:
            return False                        # the catch-all handlers self-select
        try:
            return bool(cmd.matcher(text))
        except Exception:                       # noqa: BLE001 - a picky matcher
            return False
    hits = {cmd.name for cmd in REGISTRY if _hit(cmd)}
    assert hits <= {"leave time forget"}, f"claimed by {sorted(hits)}"
    assert not undo_kind(text) and parse_yes_no(text) is None
    # ...and the three existing leave-time matchers keep their hands off it.
    assert leave_set_kind(text) is None and leave_query_kind(text) is None
    assert leave_amend_kind(text) is None


# ------------------------------------------- the amend's recency window
# "make that ten" carries no building of its own: it amends LeaveTimes.last_key,
# which is RESTORED FROM DISK at construction and is also re-pointed by the
# background reminder tick and by _maybe_ask, neither of which is a user turn.
# Without a window, a bare "make it twenty" said after a timer -- or on a fresh
# boot -- silently rewrote a stored walk for a building last touched days ago.
def test_a_stale_last_key_is_never_amended(tmp_path):
    """A disk-loaded last_key is not a subject: nothing was said about it."""
    (tmp_path / "leave.json").write_text(
        json.dumps({"last_key": "Wisenbaker Engineering Bldg"}))
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    w.table.set("Wisenbaker Engineering Bldg", 12)
    assert w.last_key == "Wisenbaker Engineering Bldg"      # loaded, not spoken
    c = _commander(w)
    c.handle("make it twenty", source="voice")
    assert w.table.get("Wisenbaker Engineering Bldg") == 12  # untouched


def test_a_background_tick_repointing_the_key_does_not_arm_the_amend(tmp_path):
    """leavetime._file_reminder / _maybe_ask call note_key from a thread with
    no user turn behind it. That must not make "make it twenty" mean that
    building."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    w.table.set("Zachry Engineering Ed. Complex", 9)
    c = _commander(w)
    w.note_key("Zachry Engineering Ed. Complex")            # the tick, not him
    c.handle("make it twenty", source="voice")
    assert w.table.get("Zachry Engineering Ed. Complex") == 9


def test_asking_how_long_arms_the_amend_that_follows(tmp_path):
    """The query names the building out loud, so the amend right after it is
    plainly about that one."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    w.table.set("Wisenbaker Engineering Bldg", 12)
    c = _commander(w)
    c.handle("how long to wisenbaker", source="voice")
    res = c.handle("make that ten next time", source="voice")
    assert res.handled and w.table.get("Wisenbaker Engineering Bldg") == 10


def test_the_amend_window_expires(tmp_path, monkeypatch):
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c.handle("it takes twelve minutes to get to wisenbaker", source="voice")
    # There are TWO clocks on this window and the amend needs both to be
    # stale: the commander stamps the turn, LeaveTimes.learn stamps the
    # teaching. Winding only one forward is not the passage of time.
    c._leave_touch -= commander.LEAVE_AMEND_WINDOW_S + 1
    w.last_touch -= commander.LEAVE_AMEND_WINDOW_S + 1
    c.handle("make that ten next time", source="voice")
    assert w.table.get("Wisenbaker Engineering Bldg") == 12


def test_teaching_through_the_store_alone_arms_the_amend(tmp_path):
    """LeaveTimes.learn() is the teaching event, so it stamps last_touch
    itself. Without that the amend window depended entirely on the
    commander handler stamping on its way past, and a walk taught through
    any other path -- a tick, a tool, a test -- left "make that ten next
    time" refusing an edit to the walk it had just stored."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    w.learn("Wisenbaker Engineering Bldg", 12)     # no commander turn at all
    assert w.last_touch > 0.0
    c.handle("make that ten next time", source="voice")
    assert w.table.get("Wisenbaker Engineering Bldg") == 10


def test_note_key_alone_never_stamps_the_teaching_clock(tmp_path):
    """The other half: note_key is called by the background reminder tick
    and by _maybe_ask, so stamping there (rather than in learn) would make
    "make it twenty" mean whatever building a thread last filed for."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    w.table.set("Zachry Engineering Ed. Complex", 9)
    w.note_key("Zachry Engineering Ed. Complex")
    assert w.last_touch == 0.0
    c = _commander(w)
    c.handle("make it twenty", source="voice")
    assert w.table.get("Zachry Engineering Ed. Complex") == 9


def test_a_leavetime_last_touch_stamp_is_honoured_if_the_store_grows_one(tmp_path):
    """Seam for jarvis/leavetime.py: the store sees the touches the commander
    never handles (an answered heads-up filed from the tick). If it ever
    stamps its own monotonic last_touch, that counts too."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    w.table.set("Wisenbaker Engineering Bldg", 12)
    w.note_key("Wisenbaker Engineering Bldg")
    c = _commander(w)
    c.handle("make that ten next time", source="voice")
    assert w.table.get("Wisenbaker Engineering Bldg") == 12   # no stamp: refused
    w.last_touch = time.monotonic()
    c.handle("make that ten next time", source="voice")
    assert w.table.get("Wisenbaker Engineering Bldg") == 10


# --------------------------------------- one question on the table at a time
# _try_leave_answer is the LAST pending rung, so a duration said while a quiz
# or a working session is open is graded as THAT answer -- a flashcard marked
# wrong and the walk never learned. Worse, leavetime._maybe_ask burns its
# once-ever ask the moment the asker returns True, so the collision spent the
# question for good rather than delaying it.
def _open_session():
    return types.SimpleNamespace(finished=False, stale=lambda: False,
                                 name="week plan")


def test_a_leave_question_is_not_armed_while_a_session_is_open(tmp_path):
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c._pending_session = _open_session()
    assert c.question_open() is True
    assert c.ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker") is False
    assert c._pending_leave is None


def test_a_leave_question_is_armed_when_the_floor_is_free(tmp_path):
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    assert c.ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker") is True
    assert c._pending_leave[0] == "Wisenbaker Engineering Bldg"
    res = c.handle("about twelve minutes", source="voice")
    assert res.handled and w.table.get("Wisenbaker Engineering Bldg") == 12


def test_the_query_does_not_ask_back_when_another_question_owns_the_floor(tmp_path):
    """Asking a question nothing is listening for is the dead end this repo
    has been bitten by before: say what he asked and stop."""
    w = _watch([event(40, LIVE_LOCATIONS[0])], tmp_path)
    c = _commander(w)
    c._pending_session = _open_session()
    res = _h_of("leave time query")(c, "how long to wisenbaker", "wisenbaker")
    assert res.reply == lt_mod.UNKNOWN_LINE
    assert c._pending_leave is None
