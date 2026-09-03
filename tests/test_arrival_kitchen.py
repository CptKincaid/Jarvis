"""The kitchen as the door, "welcome back from X", and the catch-up OFFER.

Three additions to jarvis/arrival.py, and all three are the same promise
written three times: **Jarvis says less rather than saying something it
cannot stand behind.** A kitchen trip mid-evening is not an arrival, a
guessed event name is worse than no event name, and a catch-up that reads
his mail at him is worse than one that asks first.

Every test here is pure. No phone, no radar, no mailbox, no window --
which is the property the module was built for and the reason the
ordering has stayed correct through three reworks.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

from jarvis import arrival
from jarvis.arrival import (DEFAULT_DOOR_ROOM, DoorWatch, catch_up_offer,
                            door_arrival, outing, welcome_line)
from jarvis.presence import WELCOME_LINE


def ev(title, start_h, end_h, day=None, all_day=False):
    """One calendar event on 2026-09-03, hours as floats."""
    day = day or datetime(2026, 9, 3)

    def at(h):
        return day + timedelta(hours=h)

    return SimpleNamespace(title=title, start=at(start_h), end=at(end_h),
                           all_day=all_day, location="", calendar="")


def window(left_h, back_h, day=None):
    day = day or datetime(2026, 9, 3)
    return {"left": day + timedelta(hours=left_h),
            "back": day + timedelta(hours=back_h)}


# ====================================================================
# 1. THE KITCHEN IS A DOOR SENSOR
# ====================================================================
def test_the_kitchen_going_present_after_an_absence_is_an_arrival():
    """His words: "kitchen to see if i enter my apartment since the kitchen
    and door are next to each other"."""
    assert door_arrival(room="kitchen", away=True) is True


def test_a_kitchen_trip_mid_evening_is_not_an_arrival():
    """THE failure this gate exists for. He is home, he walks in to make
    coffee, and the room fabric makes the kitchen active -- if that were an
    arrival he would be welcomed home twice an evening."""
    assert door_arrival(room="kitchen", away=False) is False


def test_walking_from_the_kitchen_to_the_office_is_not_a_second_arrival():
    """Only the DOOR room triggers. The office becoming active seconds
    later is the same walk through the same door."""
    assert door_arrival(room="office", away=True) is False


def test_the_door_room_is_configurable_and_defaults_to_the_kitchen():
    assert DEFAULT_DOOR_ROOM == "kitchen"
    assert door_arrival(room="entry", door="entry", away=True) is True
    assert door_arrival(room="kitchen", door="entry", away=True) is False


def test_the_room_name_is_matched_case_and_space_insensitively():
    """roomfabric slugs its names; a hand-written config key may not."""
    assert door_arrival(room=" Kitchen ", door="kitchen", away=True) is True


def test_an_unnamed_room_is_never_an_arrival():
    assert door_arrival(room="", away=True) is False


def test_the_door_watch_fires_once_per_return():
    """The rising edge, not the level: the fabric re-publishing the same
    room must not put a second welcome on the floor. (The app's own
    GREET_DAMPER_S is the second guard; this is the first.)"""
    watch = DoorWatch()
    assert watch.observe(room="kitchen", away=True) is True
    assert watch.observe(room="kitchen", away=True) is False
    assert watch.observe(room="office", away=True) is False
    # He left again, and came back through the kitchen: that IS a new arrival.
    watch.left()
    assert watch.observe(room="kitchen", away=True) is True


def test_the_door_watch_never_fires_while_he_is_home():
    watch = DoorWatch()
    for _ in range(5):
        assert watch.observe(room="kitchen", away=False) is False


# ====================================================================
# 2. "WELCOME BACK FROM X" -- ONLY ON HONEST EVIDENCE
# ====================================================================
def test_with_no_calendar_at_all_he_still_gets_the_plain_welcome():
    assert welcome_line() == WELCOME_LINE
    assert welcome_line("") == WELCOME_LINE
    assert outing([], **window(9, 11)) == ""
    assert outing(None, **window(9, 11)) == ""


def test_an_event_that_covered_the_absence_is_named():
    events = [ev("Organic Chemistry", 9.0, 10.5)]
    assert outing(events, **window(8.8, 10.7)) == "Organic Chemistry"
    assert welcome_line("Organic Chemistry") == \
        "Welcome back from Organic Chemistry, sir."


def test_an_event_he_came_straight_home_from_is_named():
    """Ended shortly before he walked in -- the commonest shape there is."""
    events = [ev("the dentist", 14.0, 15.0)]
    assert outing(events, **window(13.5, 15.4)) == "the dentist"


def test_an_event_that_ended_hours_before_he_got_back_is_not_named():
    """He was at a lecture at nine and came home at six. Whatever he was
    doing at half five, the lecture is not it -- and "welcome back from
    Organic Chemistry" at 6 pm is a guess wearing a fact's clothes."""
    events = [ev("Organic Chemistry", 9.0, 10.5)]
    assert outing(events, **window(8.8, 18.0)) == ""


def test_an_event_he_was_only_out_for_the_tail_of_is_not_named():
    """The event ran 1 pm to 4 pm and he left at half three: he attended a
    sixth of it. Naming it claims he was there all afternoon."""
    events = [ev("Lab section", 13.0, 16.0)]
    assert outing(events, **window(15.5, 16.2)) == ""


def test_two_events_that_both_fit_are_ambiguous_and_neither_is_named():
    """A GUESSED EVENT NAME IS WORSE THAN NO EVENT NAME. This is the case
    that pins it: the calendar honestly cannot say which one he went to."""
    events = [ev("Organic Chemistry", 9.0, 10.5), ev("Advisor meeting", 9.0, 10.5)]
    assert outing(events, **window(8.8, 10.7)) == ""


def test_the_same_event_from_two_calendars_is_one_event_not_an_ambiguity():
    """iCloud and a subscribed feed both carrying the class must not
    silently disqualify a match that is really only one."""
    events = [ev("Organic Chemistry", 9.0, 10.5), ev("organic chemistry", 9.0, 10.5)]
    assert outing(events, **window(8.8, 10.7)) == "Organic Chemistry"


def test_an_all_day_event_is_never_an_outing():
    """"Fall break" spans the absence perfectly and is not somewhere he
    went."""
    events = [ev("Fall break", 0.0, 24.0, all_day=True)]
    assert outing(events, **window(8.8, 10.7)) == ""


def test_an_untitled_event_is_not_named():
    events = [ev("   ", 9.0, 10.5)]
    assert outing(events, **window(8.8, 10.7)) == ""


def test_a_title_too_long_to_speak_is_not_named():
    """A calendar title can be a paragraph. Reading one out as the tail of
    a two-second greeting is worse than the plain line."""
    events = [ev("Zoom meeting with the department about " + "x" * 80,
                 9.0, 10.5)]
    assert outing(events, **window(8.8, 10.7)) == ""


def test_a_two_minute_absence_is_not_an_outing():
    """He took the bins out. There is nothing to be back FROM."""
    events = [ev("Advisor meeting", 9.0, 10.5)]
    assert outing(events, **window(10.4, 10.44)) == ""


def test_an_event_still_running_when_he_walked_in_is_named():
    """He left the meeting early. He was still at it, and it is still the
    honest answer to "back from what"."""
    events = [ev("Advisor meeting", 9.0, 12.0)]
    assert outing(events, **window(8.9, 11.5)) == "Advisor meeting"


def test_a_calendar_that_raises_costs_the_name_and_never_the_greeting():
    """Every failure here is a plain welcome, never an exception into the
    arrival cue."""
    class Exploding:
        all_day = False

        @property
        def start(self):
            raise RuntimeError("caldav is down")

    assert outing([Exploding()], **window(8.8, 10.7)) == ""


def test_a_missing_absence_window_is_a_plain_welcome():
    """A return with no recorded departure (Jarvis restarted while he was
    out) knows nothing about where he was."""
    assert outing([ev("Advisor meeting", 9.0, 10.5)],
                  left=None, back=datetime(2026, 9, 3, 10, 42)) == ""


# ====================================================================
# 3. THE CATCH-UP OFFERS, IT DOES NOT DELIVER
# ====================================================================
def test_the_offer_asks_and_never_reads():
    """He has ruled on this: the briefing must OFFER, not deliver. The
    line carries a COUNT and a question, and no subject line, no sender
    and no body."""
    line = catch_up_offer(unread=3)
    assert "3 unread" in line and line.endswith("?")


def test_an_empty_inbox_and_a_clear_board_offer_nothing():
    """Nothing to ask about is silence, not "you have no email, sir"."""
    assert catch_up_offer(unread=0) == ""
    assert catch_up_offer(unread=None) == ""
    assert catch_up_offer(unread=0, major="") == ""


def test_a_mailbox_that_could_not_be_reached_says_nothing_about_mail():
    """None is "I could not look", and it must never become "no mail"."""
    assert "unread" not in catch_up_offer(unread=None, major="The GPU is throttling.")


def test_one_clause_on_anything_major():
    line = catch_up_offer(unread=None, major="The GPU is throttling.")
    assert "GPU is throttling" in line and line.endswith("?")


def test_the_major_clause_and_the_count_ride_one_question():
    """One question, not two: two questions in a row on the doorstep is
    the 40-second monologue in miniature."""
    line = catch_up_offer(unread=2, major="The disk is nearly full.")
    assert line.count("?") == 1
    assert "2 unread" in line and "disk is nearly full" in line


def test_one_unread_email_is_singular():
    """Noun AND pronoun. A line that says "one email... shall I go through
    them" is the tell that it was assembled rather than written."""
    line = catch_up_offer(unread=1)
    assert "1 unread email" in line and "emails" not in line
    assert "through it, sir?" in line
    assert "through them, sir?" in catch_up_offer(unread=1,
                                                  major="The disk is full.")


def test_the_offer_addresses_him_once():
    """It is spoken as the tail of a burst that already said "sir" -- the
    thinning pass in app._arrival_actions takes the second one, but the
    line itself must not arrive carrying two."""
    assert catch_up_offer(unread=3, major="The disk is nearly full.").count("sir") <= 1


# ====================================================================
# 4. EVERYTHING DEGRADES
# ====================================================================
def test_no_calendar_no_mail_no_kitchen_still_welcomes_him_home():
    """The whole brief in one assertion: with every new source absent he
    gets exactly what he got before any of this was built."""
    assert welcome_line(outing([], **window(9, 11))) == WELCOME_LINE
    assert catch_up_offer(unread=None, major="") == ""
    assert door_arrival(room="", away=True) is False
    assert arrival.arrival_plan(returned=True) == list(arrival.ARRIVAL_STEPS)


def test_a_sensor_with_no_opinion_is_not_an_arrival():
    """RoomSensor.read() returns None for a dead radar, and roomfabric
    turns that into no active room at all. None is never a door."""
    watch = DoorWatch()
    assert watch.observe(room=None, away=True) is False
