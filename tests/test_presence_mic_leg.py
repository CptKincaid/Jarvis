"""The mic is the FOURTH leg, and the app hands the voter ONE NUMBER.

``arrival.departure_ready`` already vetoes a departure off the turn ledger
-- "the mic heard him 40s ago", read through ``TurnLedger.idle_s()``. Cell
6 of the voter (radar on, phone silent past the grace, camera unable to
look) needs the same fact for the same reason: it is the one cell whose
three legs read identically for a latched radar with him out and for him
sitting still at his desk with his phone asleep, and a spoken turn is the
one thing that tells them apart. A radar can latch and a phone can nap; a
microphone does not hallucinate a sentence.

These tests pin the wiring in jarvis/app.py:

  * the slot is attached whenever the voter is built;
  * it is LATE-BOUND -- ``self.turns`` is built after presence in
    ``__init__`` (``_wire_turn_clock`` runs after ``_construct``), so the
    leg must look the ledger up at call time and answer None before it
    exists rather than capture a missing attribute at boot;
  * what crosses the seam is seconds, never audio, and the source of the
    two methods names no capture device.
"""
from __future__ import annotations

import inspect

from jarvis import app as app_mod, presencevote as pv


class Legs:
    """Stands in for presence.ThreeLegProbe: it owns the two slots."""

    def __init__(self):
        self.eye = None
        self.mic = None


class Turns:
    """Stands in for turnclock.TurnLedger: idle_s() and nothing else."""

    def __init__(self, idle):
        self._idle = idle

    def idle_s(self):
        return self._idle


class Cfg:
    """AssistantConfig's read shape."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def bare_app(turns=None):
    a = object.__new__(app_mod.JarvisApp)
    if turns is not None:
        a.turns = turns
    return a


# ------------------------------------------------------------ the slot
def test_the_mic_slot_is_attached_to_the_voter():
    legs, a = Legs(), bare_app()
    assert a._wire_mic_leg(legs) is True
    assert legs.mic == a._mic_leg


def test_no_legs_object_is_not_an_error():
    """The old two-leg composition (presence.three_legs false) has no
    ``legs`` at all, and boot must not care."""
    assert bare_app()._wire_mic_leg(None) is False


# -------------------------------------------------------- late binding
def test_before_the_ledger_exists_the_leg_answers_none_not_an_error():
    """``self.turns`` is built AFTER presence in __init__."""
    a = bare_app()                          # no .turns at all
    assert a._mic_leg() is None


def test_the_leg_reads_idle_s_off_the_ledger_at_call_time():
    a = bare_app(Turns(42.0))
    assert a._mic_leg() == 42.0
    a.turns = Turns(7.0)                    # rebound: still read live
    assert a._mic_leg() == 7.0


def test_a_ledger_that_never_heard_a_turn_is_none():
    assert bare_app(Turns(None))._mic_leg() is None


# --------------------------------------------------------- the boundary
def test_the_wiring_names_no_capture_device_and_carries_no_audio():
    """The mic leg is a NUMBER. Nothing on this seam may open a device,
    read a sample, or reach for a dump."""
    src = (inspect.getsource(app_mod.JarvisApp._mic_leg) +
           inspect.getsource(app_mod.JarvisApp._wire_mic_leg)).lower()
    for word in ("sounddevice", "pyaudio", "record", "stream", "wav",
                 "debug_audio", "capture"):
        assert word not in src, word


# ------------------------------------------------------- end to end
def his_config_with_the_voter_on():
    # TEST-NET addresses (RFC 5737): even if something did poll, nothing
    # of his is at the far end. Construction never polls in any case.
    return Cfg(**{
        "presence.room_sensor_enabled": True,
        "presence.rooms": [{"name": "office", "url": "http://192.0.2.1"},
                           {"name": "kitchen", "url": "http://192.0.2.2"}],
        "presence.phone_ip": "192.0.2.34",
        "presence.three_legs": True,
    })


def test_make_presence_wires_the_mic_when_the_voter_is_on():
    a = bare_app()
    a.assistant = his_config_with_the_voter_on()
    a._eye_leg = lambda: ("", None, False)          # dark, as on his box
    s = a._make_presence()
    assert s is not None and s.legs is not None
    assert s.legs.mic == a._mic_leg


def test_the_number_reaches_the_voters_mic_leg_once_the_ledger_exists():
    """Boot order, replayed: presence first, the ledger after, then a turn."""
    a = bare_app()
    a.assistant = his_config_with_the_voter_on()
    a._eye_leg = lambda: ("", None, False)
    s = a._make_presence()
    assert s.legs._mic_leg() == (pv.MIC_UNKNOWN, None)      # no ledger yet
    a.turns = Turns(30.0)                                    # _wire_turn_clock
    assert s.legs._mic_leg() == (pv.MIC_HEARD, 30.0)
    a.turns = Turns(30.0 * 60.0)                             # half an hour on
    assert s.legs._mic_leg() == (pv.MIC_SILENT, 1800.0)
