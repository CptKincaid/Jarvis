"""The pre-class dossier (jarvis/dossier.py) and class-start staging
(jarvis/classflow.py), both hanging off course identity taken from the
calendar (jarvis/courses.py).

The events here are the shapes his live cache actually holds on 2026-08-30:
BIOSENSORS at Wisenbaker 049 on Monday and Wednesday 09:10, MAGNETIC
RESONANCE ENGR, ELECTRICAL DESIGN LAB II (whose 08-31 occurrence is titled
"…LAB II- Presentation"), and three one-off appointments -- one of which
carries a tamu.zoom.us URL where the room should be.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis import classflow as cf
from jarvis import courses as courses_mod
from jarvis import desk as desk_mod
from jarvis import dossier as dossier_mod
from jarvis.dossier import ClassDossier, build_dossier, room_words
from jarvis.events import ReminderFired, bus
from jarvis.headsup import MeetingHeadsUp
from jarvis.tools.calendar import Event
from jarvis.ui.views import briefing_rows

TZ = timezone(timedelta(hours=-5))
MON = datetime(2026, 8, 31, tzinfo=TZ)


def ev(title, day_offset, hour, minute, location="", minutes=50):
    start = MON + timedelta(days=day_offset, hours=hour, minutes=minute)
    return Event(start=start, end=start + timedelta(minutes=minutes),
                 title=title, location=location, calendar="icloud")


BIO_ROOM = "College Station Wisenbaker Engineering Bldg 049"
MRE_ROOM = "College Station Emerging Technologies Building 1003"
EDL_ROOM = "College Station Emerging Technologies Building 1020"
ZOOM = "https://tamu.zoom.us/j/94324046592?pwd=Ic6ifN6gKbxvhRCvfFypmo6hSM0kYw.1"


def live_events():
    """The live cache, rebuilt: two weeks of four courses plus one-offs."""
    out = []
    for week in (0, 7):
        out += [
            ev("BIOSENSORS", week, 9, 10, BIO_ROOM),
            ev("MAGNETIC RESONANCE ENGR", week, 12, 40, MRE_ROOM),
            # the 08-31 occurrence carries a suffix the other weeks do not
            ev("ELECTRICAL DESIGN LAB II" + ("- Presentation" if week == 0 else ""),
               week, 16, 10, EDL_ROOM),
            ev("MAGNETIC RESONANCE ENGR", week, 18, 0,
               "College Station Zachry Engineering Ed. Complex 330"),
            ev("BIOSENSORS", week + 1, 12, 45,
               "College Station Jack E. Brown Chem Engn Bldg 731A"),
            ev("BIOSENSORS", week + 2, 9, 10, BIO_ROOM),
            ev("MAGNETIC RESONANCE ENGR", week + 2, 12, 40, MRE_ROOM),
            ev("ELECTRICAL DESIGN LAB II", week + 2, 16, 10, EDL_ROOM),
            ev("ELECTRICAL DESIGN LAB II", week + 3, 14, 20,
               "College Station Zachry Engineering Ed. Complex 110"),
        ]
    out += [ev("Chiro", 1, 16, 0),
            ev("Hunter Peyrovi and Manuel Suarez", 4, 15, 0, ZOOM),
            ev("Your Brightside appointment with Nikkala Kordzik, LPC", 5, 14, 0)]
    return out


class Cfg:
    """The `get(dotted, default)` slice of AssistantConfig."""

    def __init__(self, **sections):
        self.data = {}
        for k, v in sections.items():
            self.data.setdefault(k, {}).update(v)

    def get(self, dotted, default=None):
        obj = self.data
        for part in dotted.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]
        return obj


class FakeCal:
    configured = True

    def __init__(self, events):
        self._events = events

    def events(self):
        return list(self._events)


# ------------------------------------------------------- course identity
def test_the_recurring_courses_are_his_three_and_nothing_else():
    """The first_step's pass condition, against the live cache's shape."""
    got = courses_mod.recurring_courses(live_events())
    assert got == ["BIOSENSORS", "MAGNETIC RESONANCE ENGR",
                   "ELECTRICAL DESIGN LAB II"]


def test_a_one_off_meeting_is_never_a_course():
    got = courses_mod.recurring_courses(live_events())
    for one_off in ("Chiro", "Hunter Peyrovi and Manuel Suarez",
                    "Your Brightside appointment with Nikkala Kordzik, LPC"):
        assert one_off not in got
        assert courses_mod.course_for(one_off, got) == ""


def test_one_weeks_variant_title_still_names_the_course():
    got = courses_mod.recurring_courses(live_events())
    assert courses_mod.course_for("ELECTRICAL DESIGN LAB II- Presentation",
                                  got) == "ELECTRICAL DESIGN LAB II"


def test_a_single_occurrence_is_not_enough():
    once = [ev("ONE OFF SEMINAR", 0, 10, 0)]
    assert courses_mod.recurring_courses(once) == []


def test_all_day_events_are_never_courses():
    day = Event(start=MON, end=MON + timedelta(days=1), all_day=True,
                title="Reading day")
    assert courses_mod.recurring_courses([day, day]) == []


# ------------------------------------------------------------- wording
@pytest.mark.parametrize("raw,said", [
    (BIO_ROOM, "Wisenbaker 049"),
    (MRE_ROOM, "Emerging Technologies 1003"),
    ("College Station Zachry Engineering Ed. Complex 330", "Zachry 330"),
    ("College Station Jack E. Brown Chem Engn Bldg 731A", "Jack E. Brown 731A"),
    ("", ""),
    (ZOOM, ""),
])
def test_room_words_says_the_building_and_the_number(raw, said):
    assert room_words(raw) == said


# -------------------------------------------------------- the dossier
def test_every_section_dark_still_gives_him_the_room():
    """~/Documents does not exist and canvas.token is empty: three of the
    four sections cannot fire, and the line is still worth saying."""
    sections, spoken = build_dossier(ev("BIOSENSORS", 0, 9, 10, BIO_ROOM), Cfg(),
                                     course="BIOSENSORS")
    assert spoken == "Your 9:10 is BIOSENSORS, Wisenbaker 049, sir."
    assert sections["room"] == BIO_ROOM, "the card keeps the raw string"
    assert "join" not in sections and "due" not in sections and "mail" not in sections


def test_a_url_location_becomes_a_join_link_not_a_room():
    zoom = ev("Hunter Peyrovi and Manuel Suarez", 4, 15, 0, ZOOM)
    sections, spoken = build_dossier(zoom, Cfg())
    assert sections["join"] == ZOOM and "room" not in sections
    assert spoken == ("Your 3:00 is Hunter Peyrovi and Manuel Suarez, sir; "
                      "the link's on the card.")


def test_a_link_in_the_body_is_the_join_row_too():
    """His Canvas feed leaves LOCATION empty and puts the Zoom link in the
    DESCRIPTION, so the JOIN row was blank for the classes that are online."""
    online = ev("Office hours", 4, 15, 0)
    online.description = f"Join Zoom Meeting\n{ZOOM}\nPasscode: 123456"
    sections, spoken = build_dossier(online, Cfg())
    assert sections["join"] == ZOOM and "room" not in sections
    assert spoken == "Your 3:00 is Office hours, sir; the link's on the card."


def test_a_hybrid_class_gets_the_room_and_the_link():
    """A room AND a link is exactly when he needs to know both are on offer."""
    hybrid = ev("BIOSENSORS", 0, 9, 10, BIO_ROOM)
    hybrid.description = f"Zoom for anyone at home: {ZOOM}"
    sections, spoken = build_dossier(hybrid, Cfg(), course="BIOSENSORS")
    assert sections["room"] == BIO_ROOM and sections["join"] == ZOOM
    # The room still leads the spoken line: it is the thing he has to act on.
    assert spoken == "Your 9:10 is BIOSENSORS, Wisenbaker 049, sir."


def test_an_ordinary_body_is_not_a_join_link():
    lesson = ev("BIOSENSORS", 0, 9, 10, BIO_ROOM)
    lesson.description = "Read chapter 4. Assignment: https://canvas.tamu.edu/a/1"
    sections, _ = build_dossier(lesson, Cfg(), course="BIOSENSORS")
    assert "join" not in sections


def test_the_full_dossier_reads_as_one_sentence_after_the_head():
    lesson = ev("BIOSENSORS", 0, 9, 10, BIO_ROOM)
    now = lesson.start - timedelta(minutes=10)
    due = [{"title": "Lab 3 report", "course": "BIOSENSORS",
            "due": lesson.start + timedelta(days=3)}]
    mail = [SimpleNamespace(from_name="Priya (TA)", from_addr="ta@tamu.edu",
                            subject="Lab 3 questions", snippet="biosensors lab")]
    sections, spoken = build_dossier(
        lesson, Cfg(), course="BIOSENSORS",
        notes={"path": "/n/biosensors-2026-08-26.md", "date": "2026-08-26",
               "topic": "electrode drift"},
        due=due, mail=mail, now=now)
    assert spoken == ("Your 9:10 is BIOSENSORS, Wisenbaker 049, sir. "
                      "Last time you noted electrode drift; Lab 3 report is "
                      "due Thu 9:10 am, and there's unread mail from "
                      "Priya (TA).")
    assert sections["last"].startswith("electrode drift (biosensors-2026-08-26.md")
    assert sections["due"] == ["Lab 3 report — Thu 9:10 am"]
    assert sections["mail"] == ["Priya (TA) — Lab 3 questions"]


def test_the_card_renders_every_dossier_row():
    sections, _ = build_dossier(
        ev("BIOSENSORS", 0, 9, 10, BIO_ROOM), Cfg(), course="BIOSENSORS",
        notes={"path": "/n/biosensors-2026-08-26.md", "topic": "electrode drift"},
        mail=[SimpleNamespace(from_name="Priya", from_addr="", subject="Lab 3")])
    labels = [label for label, _text in briefing_rows(sections)]
    assert labels == ["CALENDAR", "ROOM", "LAST TIME", "MAIL"]


def test_briefing_rows_keeps_the_old_order_when_a_dossier_key_rides_along():
    """WEEK / WEATHER / CALENDAR / DUE / EXAM is what the briefing and the
    evening preview depend on; ROOM must not push DUE above CALENDAR."""
    rows = briefing_rows({"summary": "busy", "weather": "warm",
                          "calendar": ["9:10 BIOSENSORS"], "room": "Wisenbaker",
                          "due": ["Lab 3"], "exam": "Midterm Friday"})
    assert [label for label, _ in rows] == ["WEEK", "WEATHER", "CALENDAR",
                                            "ROOM", "DUE", "EXAM"]


# ------------------------------------------------------------ sources
def test_last_notes_finds_the_previous_session_not_todays(tmp_path):
    notes = tmp_path / "docs" / "notes"
    notes.mkdir(parents=True)
    (notes / "biosensors-2026-08-24.md").write_text("# BIOSENSORS\n\n- 09:12  old news\n")
    (notes / "biosensors-2026-08-26.md").write_text("# BIOSENSORS\n\n- 09:15  electrode drift\n"
                                                    "- 09:40  and calibration\n")
    (notes / "biosensors-2026-08-31.md").write_text("# BIOSENSORS\n\n")
    cfg = Cfg(docs={"paths": [str(tmp_path / "docs")]})
    got = dossier_mod.last_notes(cfg, "BIOSENSORS", MON + timedelta(hours=9))
    assert got is not None
    assert got["topic"] == "electrode drift" and got["date"] == "2026-08-26"


def test_a_header_only_file_is_not_a_previous_session(tmp_path):
    """The stager primes an empty dated file; that is not something he
    'covered last time'."""
    notes = tmp_path / "docs" / "notes"
    notes.mkdir(parents=True)
    (notes / "biosensors-2026-08-26.md").write_text("# BIOSENSORS — 2026-08-26\n\n")
    cfg = Cfg(docs={"paths": [str(tmp_path / "docs")]})
    assert dossier_mod.last_notes(cfg, "BIOSENSORS", MON + timedelta(hours=9)) is None


def test_a_missing_notes_folder_is_a_dark_section_not_a_crash(tmp_path):
    cfg = Cfg(docs={"paths": [str(tmp_path / "nope")]})
    assert dossier_mod.last_notes(cfg, "BIOSENSORS", MON) is None


def test_mail_needs_both_the_course_and_a_person_he_knows():
    people = {"ta": {"name": "Priya", "email": "ta@tamu.edu"}}
    from_ta = SimpleNamespace(from_name="Priya", from_addr="ta@tamu.edu",
                              subject="Biosensors lab 3", snippet="")
    stranger = SimpleNamespace(from_name="Deals", from_addr="spam@x.com",
                               subject="Biosensors news", snippet="")
    off_topic = SimpleNamespace(from_name="Priya", from_addr="ta@tamu.edu",
                                subject="lunch?", snippet="")
    got = dossier_mod.course_mail([from_ta, stranger, off_topic], "BIOSENSORS", people)
    assert got == [from_ta]


def test_only_this_courses_deadlines_reach_the_card():
    items = [{"title": "Lab 3", "course": "BIOSENSORS", "due": MON},
             {"title": "HW 2", "course": "MAGNETIC RESONANCE ENGR", "due": MON}]
    assert [i["title"] for i in dossier_mod.course_due(items, "BIOSENSORS")] == ["Lab 3"]


# ------------------------------------------------------------- the filer
class FakeTimekeeper:
    def __init__(self):
        self.silent = []

    def add_silent_timer(self, seconds, label=""):
        item = SimpleNamespace(id=f"id{len(self.silent)}", label=label,
                               seconds=seconds)
        self.silent.append(item)
        return item


@pytest.fixture
def desk_world(tmp_path):
    """A dossier over the live calendar with every network source stubbed."""
    said, cards = [], []
    services = SimpleNamespace(speak=lambda text, **kw: said.append(text),
                               memory=SimpleNamespace(people=lambda: {}))
    cfg = Cfg(docs={"paths": [str(tmp_path / "docs")]}, dossier={"mail": False})
    tk = FakeTimekeeper()
    now = MON + timedelta(hours=8, minutes=30)
    d = ClassDossier(cfg, tk, get_calendar=lambda: FakeCal(live_events()),
                     services=services, state_path=tmp_path / "d.json",
                     now=lambda tzinfo=None: now.astimezone(tzinfo) if tzinfo else now,
                     bg=lambda fn: fn(), publish=cards.append,
                     fetch_due=lambda *a, **k: [])
    yield SimpleNamespace(d=d, tk=tk, said=said, cards=cards, cfg=cfg,
                          now=now, tmp=tmp_path, services=services)
    d.stop()


def test_the_dossier_files_one_silent_timer_per_class(desk_world):
    assert desk_world.d.tick() == 1, "only the 09:10 class is inside the horizon"
    item, = desk_world.tk.silent
    assert "BIOSENSORS" in item.label
    assert abs(item.seconds - 30 * 60) < 1, "T-10 for a 09:10 class at 08:30"
    assert desk_world.d.tick() == 0, "filed twice"


def test_a_restart_does_not_re_file_the_same_class(desk_world):
    desk_world.d.tick()
    twin = ClassDossier(desk_world.cfg, desk_world.tk,
                        get_calendar=lambda: FakeCal(live_events()),
                        state_path=desk_world.tmp / "d.json",
                        now=lambda tzinfo=None: desk_world.now)
    try:
        assert twin.tick() == 0
    finally:
        twin.stop()


def test_the_timer_firing_speaks_the_line_and_publishes_the_card(desk_world):
    desk_world.d.tick()
    item, = desk_world.tk.silent
    bus.publish(ReminderFired(text=item.label, item_id=item.id, silent=True))
    assert desk_world.said == ["Your 9:10 is BIOSENSORS, Wisenbaker 049, sir."]
    card, = desk_world.cards
    assert card.sections["room"] == BIO_ROOM
    assert card.spoken == desk_world.said[0]


def test_a_body_link_survives_the_state_file_to_the_card(desk_world, tmp_path):
    """tick() files a timer and forgets the event; only the state file
    reaches the delivery ten minutes later, so the link has to be in it."""
    events = live_events()
    for e in events:                       # the 09:10 class, moved online
        if e.title == "BIOSENSORS" and e.start.hour == 9:
            e.location, e.description = "", f"Join Zoom Meeting {ZOOM}"
    d = ClassDossier(desk_world.cfg, desk_world.tk,
                     get_calendar=lambda: FakeCal(events),
                     services=desk_world.services, state_path=tmp_path / "d2.json",
                     now=lambda tzinfo=None: desk_world.now,
                     bg=lambda fn: fn(), publish=desk_world.cards.append,
                     fetch_due=lambda *a, **k: [])
    try:
        assert d.tick() == 1
        item = desk_world.tk.silent[-1]
        bus.publish(ReminderFired(text=item.label, item_id=item.id, silent=True))
    finally:
        d.stop()
    assert desk_world.cards[-1].sections["join"] == ZOOM


def test_somebody_elses_silent_timer_is_left_alone(desk_world):
    desk_world.d.tick()
    bus.publish(ReminderFired(text="focus: block 1", item_id="not-mine", silent=True))
    assert desk_world.said == [] and desk_world.cards == []


def test_a_class_the_dossier_speaks_for_silences_the_bare_heads_up(desk_world, tmp_path):
    """Both fire at T-10; "BIOSENSORS in ten minutes" on top of the dossier
    would say the title twice."""
    filed = []
    tk = SimpleNamespace(add_reminder=lambda due, text, repeat="": filed.append(text))
    now = desk_world.now
    h = MeetingHeadsUp(lambda: FakeCal(live_events()), tk, lead_min=10,
                       state_path=tmp_path / "h.json",
                       now=lambda tzinfo=None: now.astimezone(tzinfo) if tzinfo else now,
                       skip=desk_world.d.owns)
    assert h.tick() == 0
    assert filed == []
    # …and a one-off meeting still gets one
    chiro = [ev("Chiro", 0, 9, 0)]
    h2 = MeetingHeadsUp(lambda: FakeCal(chiro), tk, lead_min=10,
                        state_path=tmp_path / "h2.json",
                        now=lambda tzinfo=None: now.astimezone(tzinfo) if tzinfo else now,
                        skip=desk_world.d.owns)
    assert h2.tick() == 1 and filed == ["Chiro in 10 minutes"]


def test_the_dossier_switched_off_leaves_the_heads_up_alone(desk_world, tmp_path):
    desk_world.cfg.data.setdefault("dossier", {})["enabled"] = False
    assert desk_world.d.tick() == 0
    assert desk_world.d.owns(ev("BIOSENSORS", 0, 9, 10, BIO_ROOM)) is False


# ----------------------------------------------------------- the stager
class FakeSpotify:
    def __init__(self, state="playing"):
        self.calls = []
        self.state = state

    def playback_state(self):
        return self.state

    def control(self, action, value=None, device=None):
        self.calls.append(action)
        self.state = "paused" if action == "pause" else "playing"
        return SimpleNamespace(ok=True)


class FakeQuiet:
    def __init__(self, why="", calendar_why="in class"):
        self.why, self.calendar_why = why, calendar_why

    def reason(self, now=None, calendar=True):
        return self.calendar_why if (calendar and self.calendar_why) else self.why


@pytest.fixture
def stage_world(tmp_path, monkeypatch):
    monkeypatch.setattr(desk_mod, "at_desk", lambda *a, **k: True)
    said, cards = [], []
    commander = SimpleNamespace(lecture_course=None, _lecture=None)
    spotify = FakeSpotify()
    services = SimpleNamespace(speak=lambda text, **kw: said.append(text),
                               spotify=spotify, quiet=FakeQuiet(), notes=None)
    cfg = Cfg(docs={"paths": [str(tmp_path / "docs")]})
    clock = {"now": MON + timedelta(hours=9, minutes=10, seconds=20)}
    s = cf.ClassStager(cfg, get_calendar=lambda: FakeCal(live_events()),
                       services=services, state_path=tmp_path / "cf.json",
                       now=lambda tz=None: clock["now"].astimezone(tz) if tz
                       else clock["now"],
                       get_commander=lambda: commander, bg=lambda fn: fn(),
                       publish=cards.append)
    yield SimpleNamespace(s=s, said=said, cards=cards, cfg=cfg, clock=clock,
                          spotify=spotify, commander=commander, tmp=tmp_path,
                          services=services)
    s.stop()


def notes_file(world):
    return world.tmp / "docs" / "notes" / "biosensors-2026-08-31.md"


def test_the_desk_is_set_at_the_moment_the_class_starts(stage_world):
    assert stage_world.s.tick() == 1
    path = notes_file(stage_world)
    assert path.exists() and path.read_text().startswith("# BIOSENSORS — 2026-08-31")
    assert stage_world.spotify.calls == ["pause"]
    card, = stage_world.cards
    assert card.sections["class"] == ["BIOSENSORS staged — notes ready",
                                      "notes: biosensors-2026-08-31.md",
                                      "music paused for the hour"]
    assert card.spoken == "", "staging is silent unless notes were armed"


def test_a_staged_class_is_not_staged_again_on_the_next_tick(stage_world):
    stage_world.s.tick()
    stage_world.clock["now"] += timedelta(seconds=45)
    assert stage_world.s.tick() == 0
    assert stage_world.spotify.calls == ["pause"]


def test_capture_is_not_armed_unless_he_asked_for_it(stage_world):
    stage_world.s.tick()
    assert stage_world.commander.lecture_course is None
    assert stage_world.commander._lecture is None


def test_auto_notes_arms_capture_and_says_so(stage_world):
    stage_world.cfg.data.setdefault("class_flow", {})["auto_notes"] = True
    stage_world.s.tick()
    assert stage_world.commander.lecture_course == "BIOSENSORS"
    assert stage_world.said == ["Taking notes for BIOSENSORS, sir; "
                                "say end notes when you're done."]


def test_the_end_of_the_class_puts_everything_back(stage_world):
    stage_world.cfg.data.setdefault("class_flow", {})["auto_notes"] = True
    stage_world.s.tick()
    stage_world.clock["now"] += timedelta(hours=2)
    stage_world.s.tick()
    assert stage_world.spotify.calls == ["pause", "resume"]
    assert stage_world.commander.lecture_course is None
    assert stage_world.commander._lecture is None


def test_quit_puts_the_music_back_even_mid_class(stage_world):
    stage_world.s.tick()
    stage_world.s.restore_all()
    assert stage_world.spotify.calls == ["pause", "resume"]


def test_a_wake_word_after_the_class_is_the_safety_net(stage_world):
    stage_world.s.tick()
    stage_world.clock["now"] += timedelta(hours=2)
    stage_world.s._on_wake()
    assert stage_world.spotify.calls == ["pause", "resume"]


def test_a_wake_word_during_the_class_leaves_the_music_down(stage_world):
    stage_world.s.tick()
    stage_world.clock["now"] += timedelta(minutes=5)
    stage_world.s._on_wake()
    assert stage_world.spotify.calls == ["pause"], "resumed mid-lecture"


def test_do_not_disturb_stages_nothing(stage_world):
    stage_world.services.quiet = FakeQuiet(why="do not disturb until 10:00",
                                           calendar_why="")
    assert stage_world.s.tick() == 0
    assert not notes_file(stage_world).exists()
    assert stage_world.spotify.calls == []


def test_the_class_being_a_quiet_window_does_not_block_its_own_staging(stage_world):
    """quiet.py now treats a running course as a quiet window, so the full
    reason() is non-empty the instant the class starts. The stager asks
    without the calendar leg or it could never fire."""
    stage_world.services.quiet = FakeQuiet(why="", calendar_why="BIOSENSORS until 10:00")
    assert stage_world.s.tick() == 1


def test_an_empty_desk_is_not_staged(stage_world, monkeypatch):
    monkeypatch.setattr(desk_mod, "at_desk", lambda *a, **k: False)
    assert stage_world.s.tick() == 0
    assert not notes_file(stage_world).exists()


def test_walking_in_two_minutes_late_still_gets_a_staged_desk(stage_world,
                                                              monkeypatch):
    """A refused gate is not remembered: he may sit down a minute into the
    hour, and GRACE_MIN already bounds how long the offer stands."""
    away = {"v": True}
    monkeypatch.setattr(desk_mod, "at_desk", lambda *a, **k: not away["v"])
    assert stage_world.s.tick() == 0
    away["v"] = False
    stage_world.clock["now"] += timedelta(minutes=2)
    assert stage_world.s.tick() == 1
    assert notes_file(stage_world).exists()


def test_six_minutes_late_is_no_longer_a_class_starting(stage_world, monkeypatch):
    monkeypatch.setattr(desk_mod, "at_desk", lambda *a, **k: False)
    stage_world.s.tick()
    monkeypatch.setattr(desk_mod, "at_desk", lambda *a, **k: True)
    stage_world.clock["now"] += timedelta(minutes=6)
    assert stage_world.s.tick() == 0


def test_nothing_playing_means_nothing_to_put_back(stage_world):
    stage_world.spotify.state = "idle"
    stage_world.s.tick()
    assert stage_world.spotify.calls == []
    stage_world.clock["now"] += timedelta(hours=2)
    stage_world.s.tick()
    assert stage_world.spotify.calls == [], "resumed music he never had on"


def test_switched_off_it_does_nothing(stage_world):
    stage_world.cfg.data.setdefault("class_flow", {})["enabled"] = False
    assert stage_world.s.tick() == 0
    assert not notes_file(stage_world).exists()


def test_saying_notes_for_biosensors_lands_in_the_file_already_staged(stage_world):
    """The whole point of priming: the page he opens by voice is the page
    that is already there, header and all, not a second one beside it."""
    from jarvis.lecture import LectureNotes
    stage_world.s.tick()
    staged = notes_file(stage_world)
    capture = LectureNotes(stage_world.cfg, "biosensors",
                           now=lambda: MON + timedelta(hours=9, minutes=12))
    assert capture.path == staged
    capture.add("electrode drift")
    text = staged.read_text()
    assert text.count("# BIOSENSORS") == 1, "a second header was written"
    assert "electrode drift" in text


def test_a_one_off_meeting_is_never_staged(stage_world):
    stage_world.clock["now"] = MON + timedelta(days=1, hours=16, minutes=1)
    assert stage_world.s.tick() == 0


# --------------------------------------------------------- desk presence
def test_desk_presence_fails_open_when_nothing_can_measure_it(monkeypatch):
    monkeypatch.setattr(desk_mod, "x_idle_ms", lambda display=None: None)
    assert desk_mod.at_desk(15, turns_path=None) is True


def test_the_turn_ledger_answers_when_x_cannot(tmp_path, monkeypatch):
    monkeypatch.setattr(desk_mod, "x_idle_ms", lambda display=None: None)
    ledger = tmp_path / "turns.jsonl"
    ledger.write_text('{"outcome": "audio", "at": 1000.0}\n'
                      '{"outcome": "audio", "at": 2000.0}\n')
    assert desk_mod.last_turn_at(ledger) == 2000.0
    assert desk_mod.at_desk(15, turns_path=ledger, now=2300.0) is True
    assert desk_mod.at_desk(15, turns_path=ledger, now=4000.0) is False


def test_a_torn_ledger_line_is_skipped(tmp_path):
    ledger = tmp_path / "turns.jsonl"
    ledger.write_text('{"at": 1000.0}\n{"at": broken\n')
    assert desk_mod.last_turn_at(ledger) == 1000.0
