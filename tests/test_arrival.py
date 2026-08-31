"""Arrival and departure choreography (jarvis/arrival.py).

The delta is ordering and asymmetry, so that is what these assert: the four
arrival steps in their fixed order, and a departure that is silent, late
and vetoable.
"""
import pytest

from jarvis import arrival
from jarvis.arrival import (ARRIVAL_STEPS, arrival_plan, confirm_s, departure_plan,
                            departure_ready, mic_silence_s, run)


def cfg_get(data):
    def get(key, default=None):
        return data.get(key, default)
    return get


# ---------------------------------------------------------------- arrival
def test_the_arrival_order_is_panel_earcon_greeting_catch_up():
    assert arrival_plan(returned=True) == ["panel", "earcon", "greeting", "catch-up"]
    assert ARRIVAL_STEPS == ("panel", "earcon", "greeting", "catch-up")


def test_a_departure_event_gets_no_arrival_cue():
    assert arrival_plan(returned=False, home=False) == []


def test_a_home_event_that_is_not_a_return_gets_no_cue():
    """Presence publishes home=True on the first probe of a fresh boot with
    returned=False; that is not a door opening."""
    assert arrival_plan(returned=False, home=True) == []


def test_quiet_hours_hold_the_voice_but_still_wake_the_panel():
    """The pre-existing rule kept: any reason other than "you're out" defers
    the greeting to the quiet policy's own tick."""
    assert arrival_plan(returned=True, quiet_reason="quiet hours until 7:00 am") == ["panel"]
    assert arrival_plan(returned=True, quiet_reason="a meeting until 2:00 pm") == ["panel"]


def test_the_away_reason_never_defers_the_greeting():
    """"you're out" is stale by definition on a returned event."""
    assert arrival_plan(returned=True, quiet_reason="you're out") == list(ARRIVAL_STEPS)


def test_the_cue_can_be_turned_off_without_losing_the_greeting():
    assert arrival_plan(returned=True, cue=False) == ["panel", "greeting", "catch-up"]


def test_run_performs_the_steps_in_order():
    order = []
    done = run(ARRIVAL_STEPS, {name: (lambda n=name: order.append(n) or True)
                               for name in ARRIVAL_STEPS})
    assert order == list(ARRIVAL_STEPS) and done == list(ARRIVAL_STEPS)


def test_a_step_that_did_nothing_is_not_reported():
    done = run(["panel", "catch-up"], {"panel": lambda: True,
                                       "catch-up": lambda: False})
    assert done == ["panel"]


def test_a_broken_step_never_costs_him_the_greeting():
    """A missing earcon must not swallow "Welcome back, sir"."""
    spoken = []

    def boom():
        raise RuntimeError("no pulse")

    done = run(ARRIVAL_STEPS, {"panel": lambda: True, "earcon": boom,
                               "greeting": lambda: spoken.append("hi") or True,
                               "catch-up": lambda: True})
    assert done == ["panel", "greeting", "catch-up"] and spoken == ["hi"]


def test_a_step_with_no_action_is_skipped():
    assert run(ARRIVAL_STEPS, {"greeting": lambda: True}) == ["greeting"]


# -------------------------------------------------------------- departure
def test_departure_is_silent():
    """The whole point of the mirror: no speech step exists to run."""
    assert departure_plan(home=False) == ["settle"]
    assert "greeting" not in departure_plan(home=False)


def test_a_home_event_never_settles():
    assert departure_plan(home=True) == []


def test_the_mic_is_a_hard_veto():
    """A sleeping phone radio must not settle the room around a man who
    just spoke."""
    ok, why = departure_ready(now=10_000.0, since=0.0, last_turn=9_900.0,
                              confirm=300.0, mic_silence=600.0)
    assert ok is False and "mic heard him" in why


def test_the_confirm_window_must_elapse():
    ok, why = departure_ready(now=1_100.0, since=1_000.0, last_turn=None,
                              confirm=300.0, mic_silence=600.0)
    assert ok is False and "past the away grace" in why


def test_coming_back_cancels_the_settle():
    ok, why = departure_ready(now=10_000.0, since=0.0, last_turn=None,
                              confirm=300.0, mic_silence=600.0, still_away=False)
    assert ok is False and why == "he came back"


def test_all_three_gates_clear():
    ok, why = departure_ready(now=10_000.0, since=1_000.0, last_turn=1_000.0,
                              confirm=300.0, mic_silence=600.0)
    assert ok is True and why == ""


def test_a_mic_that_never_spoke_does_not_block():
    ok, _why = departure_ready(now=10_000.0, since=1_000.0, last_turn=None,
                               confirm=300.0, mic_silence=600.0)
    assert ok is True


def test_departure_is_asymmetric_against_the_away_grace():
    """away_after_min already clamps at 60 s; the confirm window is strictly
    additional, so the total is always longer than the grace alone."""
    assert confirm_s(cfg_get({"presence.departure_confirm_min": 0.1})) == 60.0
    assert confirm_s(cfg_get({"presence.departure_confirm_min": 5})) == 300.0
    assert confirm_s(cfg_get({})) == arrival.DEFAULT_CONFIRM_MIN * 60.0


def test_nonsense_windows_fall_back_to_the_defaults():
    assert confirm_s(cfg_get({"presence.departure_confirm_min": "soon"})) == \
        arrival.DEFAULT_CONFIRM_MIN * 60.0
    assert mic_silence_s(cfg_get({"presence.departure_mic_silence_min": None})) == \
        arrival.DEFAULT_MIC_SILENCE_MIN * 60.0


# ----------------------------------------------------- presence cadence
def test_presence_polls_fast_only_while_he_is_out():
    """Correction 1: at the 60 s default the room notices the door up to a
    minute late, by which time a staged cue reads as late, not composed."""
    from jarvis.presence import PresenceSentinel

    class Cfg:
        def get(self, key, default=None):
            return {"presence.poll_s": 60, "presence.poll_s_away": 10}.get(key, default)

    p = PresenceSentinel(Cfg())
    p.home = True
    assert p.poll_s == 60.0
    p.home = False
    assert p.poll_s == 10.0
    p.home = None                      # unknown: the slow cadence, not the fast one
    assert p.poll_s == 60.0


def test_the_fast_cadence_still_respects_the_five_second_floor():
    from jarvis.presence import PresenceSentinel

    class Cfg:
        def get(self, key, default=None):
            return 0.1

    p = PresenceSentinel(Cfg())
    p.home = False
    assert p.poll_s == 5.0


# ------------------------------------------------------- the turn ledger
def test_the_ledger_reports_how_long_the_mic_has_been_idle():
    """The departure veto keys off the EXISTING ledger, not a new field."""
    from jarvis.turnclock import TurnLedger
    now = [100.0]
    led = TurnLedger(clock=lambda: now[0], emit=lambda rec: None)
    assert led.idle_s() is None
    led.mark("wake")
    now[0] += 30.0
    assert led.idle_s() == pytest.approx(30.0)
    led.mark("mic")
    assert led.idle_s() == pytest.approx(0.0)
