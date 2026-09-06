"""Every refusal on the arrival path names the gate that said no.

Tonight he got silence. Five gates sit between a door opening and a
"Welcome back, sir", and on 2026-09-05 one of them declined without
writing a word: the whole day's log holds seven "presence: home" lines,
zero "away" lines and no arrival line at all. Silence is the defect.
"""
from __future__ import annotations

import logging

from jarvis import arrival


def test_a_door_opening_that_would_fire_has_no_refusal():
    assert arrival.door_refusal(room="kitchen", door="kitchen",
                                state="away") == ""


def test_the_away_gate_names_itself_and_says_what_it_saw():
    why = arrival.door_refusal(room="kitchen", door="kitchen", state="home")
    assert why
    assert "away" in why
    assert "home" in why


def test_unknown_is_refused_as_loudly_as_home_and_says_which_it_was():
    """A fresh start while he sits in the office must not welcome him --
    but the log has to show that "unknown" is what declined, because that
    is the boot case and it is indistinguishable from tonight otherwise."""
    why = arrival.door_refusal(room="kitchen", door="kitchen", state="unknown")
    assert "unknown" in why


def test_the_wrong_room_names_itself_and_both_rooms():
    why = arrival.door_refusal(room="office", door="kitchen", state="away")
    assert "office" in why and "kitchen" in why


def test_no_room_at_all_is_its_own_named_gate():
    why = arrival.door_refusal(room="", door="kitchen", state="away")
    assert why and "no room" in why.lower()
    assert arrival.door_refusal(room=None, door="kitchen", state="away")


def test_a_door_room_matching_no_configured_room_is_named(caplog):
    why = arrival.door_refusal(room="kitchen", door="scullery", state="away")
    assert "scullery" in why


def test_the_refusal_survives_the_slugging_both_sides():
    assert arrival.door_refusal(room="  Kitchen! ", door="kitchen",
                                state="away") == ""


def test_the_door_watch_records_why_it_declined():
    watch = arrival.DoorWatch()
    assert watch.observe(room="kitchen", away=False) is False
    assert "away" in watch.last_refusal


def test_the_second_kitchen_of_one_homecoming_names_the_latch():
    watch = arrival.DoorWatch()
    assert watch.observe(room="kitchen", away=True) is True
    assert watch.last_refusal == ""
    assert watch.observe(room="kitchen", away=True) is False
    assert "already" in watch.last_refusal


def test_the_watch_clears_its_refusal_when_it_fires():
    watch = arrival.DoorWatch()
    watch.observe(room="office", away=True)
    assert watch.last_refusal
    watch.observe(room="kitchen", away=True)
    assert watch.last_refusal == ""


def test_the_refusal_reaches_the_log_at_info(caplog):
    """The point of the whole file: he must be able to read the answer to
    "why did he not greet me" out of jarvis.log without a debugger."""
    watch = arrival.DoorWatch()
    with caplog.at_level(logging.INFO, logger="arrival"):
        watch.observe(room="kitchen", away=False)
    assert any("away" in r.getMessage() for r in caplog.records)
    assert all(r.levelno >= logging.INFO for r in caplog.records)


def test_a_refusal_is_logged_once_per_reason_not_once_per_poll(caplog):
    watch = arrival.DoorWatch()
    with caplog.at_level(logging.INFO, logger="arrival"):
        for _ in range(20):
            watch.observe(room="kitchen", away=False)
    assert len(caplog.records) == 1


def test_a_new_reason_is_logged_even_after_an_old_one(caplog):
    watch = arrival.DoorWatch()
    with caplog.at_level(logging.INFO, logger="arrival"):
        watch.observe(room="kitchen", away=False)
        watch.observe(room="office", away=True)
    assert len(caplog.records) == 2


def test_the_greet_damper_refusal_is_named_too():
    why = arrival.greet_refusal(source="phone", since_s=10.0,
                                damper_s=arrival.GREET_DAMPER_S)
    assert "damper" in why and "phone" in why
    assert arrival.greet_refusal(source="phone", since_s=10_000.0,
                                 damper_s=arrival.GREET_DAMPER_S) == ""


def test_the_damper_is_the_one_arrival_py_already_owns():
    assert arrival.GREET_DAMPER_S == 600.0


def test_every_gate_on_the_path_has_a_name():
    """The gate names are a closed set so nothing can refuse anonymously."""
    assert set(arrival.DOOR_GATES) == {
        "no-room", "wrong-room", "not-away", "already-greeted", "damper"}


def test_departure_ready_already_names_its_gates_and_still_does():
    ok, why = arrival.departure_ready(now=100.0, since=0.0, last_turn=99.0,
                                      confirm=60.0, mic_silence=600.0)
    assert ok is False and "mic" in why
    ok, why = arrival.departure_ready(now=10.0, since=0.0, last_turn=None,
                                      confirm=600.0, mic_silence=600.0)
    assert ok is False and "grace" in why
