"""THE 10:49:03 LINE MUST BE IMPOSSIBLE.

Five seconds after a restart on 2026-09-06, with him standing in the flat,
the voter published:

  10:49:03.489  presence: away (cell 24) -- phone-only away past the grace,
                no room readable [phone phone-no, camera cam-blind,
                rooms rooms-unreachable]

Two independent causes, both pinned here.

  * The rooms fabric had not completed its first poll. It started at
    10:49:02.4 and named the office at 10:49:04.9 -- 1.4 s AFTER the
    verdict. ``Room.value`` starts None, so the leg read UNREACHABLE.
  * ``ThreeLegProbe.grace_s`` was 0.0 and nothing ever wrote it, so
    ``0.0 >= 0.0`` made the FIRST unanswered ping PHONE_NO. The verdict's
    own sentence -- "his phone did not answer past the grace" -- was false.

And the third boot defect, from the same log: ``_agreed_s_ago`` aged a run
that was ALREADY LATCHED before the restart from its post-boot start, which
handed a stuck radar a full 15-minute HOME window.
"""
from __future__ import annotations

import logging
import time

import pytest

from jarvis import presence as pres
from jarvis import presencevote as pv
from jarvis import roomfabric as rf
from jarvis import stuckroom as sr


class Cfg:
    def __init__(self, **kw):
        self._d = dict(kw)

    def get(self, key, default=None):
        return self._d.get(key, default)


class Reader:
    """A room reader whose answers a test scripts, poll by poll."""

    def __init__(self, values):
        self.values = list(values)
        self.reads = 0
        self.configured = True

    def read(self):
        self.reads += 1
        if not self.values:
            return None
        return self.values[0] if len(self.values) == 1 else self.values.pop(0)


def fabric(office=(True,), kitchen=(False,), poll_s=2.0, now=None):
    fab = rf.from_readers([("office", Reader(office)),
                           ("kitchen", Reader(kitchen))],
                          poll_s=poll_s, enter_hold_s=2.0,
                          **({"now": now} if now else {}))
    return fab


# =====================================================================
# (a) THE PHONE GRACE
# =====================================================================
def test_the_first_unanswered_ping_is_unknown_not_no():
    """An iPhone routinely ignores a single ``ping -W 1``. The 12-minute
    grace exists for exactly that, and the voter never applied it."""
    clock = {"t": 5000.0}
    probe = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(),
                               phone=lambda ip, mac: False,
                               now=lambda: clock["t"])
    probe("10.0.0.5", "")
    assert probe.legs["phone"] == pv.PHONE_UNKNOWN
    assert probe.grace_s >= 700.0, "the grace must be away_after_min, not 0.0"


def test_a_genuine_hours_long_absence_still_reaches_phone_no():
    """11:05's verdict must be unchanged -- only its sentence becomes true."""
    clock = {"t": 5000.0}
    probe = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(),
                               phone=lambda ip, mac: False,
                               now=lambda: clock["t"])
    probe("10.0.0.5", "")
    clock["t"] += 3600.0
    probe("10.0.0.5", "")
    assert probe.legs["phone"] == pv.PHONE_NO


def test_cell_24_cannot_be_reached_inside_the_grace_of_a_fresh_start():
    """The exact 10:49:03 sequence: no room has answered yet AND the phone
    has never answered. Cell 24 is AWAY; it must be unreachable."""
    clock = {"t": 5000.0}
    fab = fabric(office=(None,), kitchen=(None,))
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(),
                               phone=lambda ip, mac: False,
                               now=lambda: clock["t"], boot_grace_s=0.0)
    out = probe("10.0.0.5", "")
    assert probe.verdict.cell != 24
    assert probe.verdict.state != pv.AWAY
    assert out is None, "unknown must hold, not vote"


def test_the_grace_clock_does_not_start_on_a_genuinely_blank_answer():
    """A phone that could not be ASKED at all -- nothing was learned, so
    ``PresenceSentinel._started_at`` must still be None. The grace has to
    start from the first real answer, or a leg that was down for an hour
    would earn an instant away the moment it came back."""
    clock = {"t": 5000.0}
    fab = fabric(office=(None,), kitchen=(None,))
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(),
                               phone=lambda ip, mac: None,
                               now=lambda: clock["t"], boot_grace_s=0.0)
    s = pres.PresenceSentinel(Cfg(**{"presence.phone_ip": "10.0.0.5"}),
                              publish=lambda ev: None, probe_fn=probe,
                              now=lambda: clock["t"])
    assert s.tick() is None
    assert probe.grace_running is False
    assert s._started_at is None


def test_a_phone_asked_and_silent_does_start_the_clock_while_the_grace_runs():
    """THE OTHER HALF, and the reason the two graces cannot stack. Silence
    from a phone that WAS asked is evidence, not a blank -- so the
    sentinel's own clock starts now rather than when this leg's grace
    expires. MEASURED both ways: away at minute 13 with this, 25 without."""
    clock = {"t": 5000.0}
    fab = fabric(office=(None,), kitchen=(None,))
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(),
                               phone=lambda ip, mac: False,
                               now=lambda: clock["t"], boot_grace_s=0.0)
    s = pres.PresenceSentinel(Cfg(**{"presence.phone_ip": "10.0.0.5"}),
                              publish=lambda ev: None, probe_fn=probe,
                              now=lambda: clock["t"])
    assert s.tick() is None, "still no verdict -- the grace has not run out"
    assert probe.grace_running is True
    assert s._started_at == 5000.0
    assert s.last_seen is None, "he has NOT been seen; only the clock started"


# =====================================================================
# (b) THE BOUNDED WAIT FOR THE ROOMS' FIRST POLL
# =====================================================================
def test_no_verdict_before_the_rooms_have_answered_once():
    fab = fabric(office=(True,), kitchen=(False,))
    assert fab.ready is False, "nothing has answered yet"
    fab.tick()
    assert fab.ready is True
    assert fab.polls == 1


def test_the_boot_wait_is_bounded_and_says_so_once(caplog):
    """A fabric that never answers must not hang the presence thread, and
    must not print the line once per poll."""
    caplog.set_level(logging.INFO, logger="presence")
    fab = fabric(office=(None,), kitchen=(None,))
    fab._thread = type("T", (), {"is_alive": staticmethod(lambda: True)})()
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(),
                               phone=lambda ip, mac: True, boot_grace_s=0.05)
    began = time.monotonic()
    probe("10.0.0.5", "")
    probe("10.0.0.5", "")
    probe("10.0.0.5", "")
    assert time.monotonic() - began < 1.0, "the wait must be bounded"
    said = [r for r in caplog.records if "without the rooms" in r.getMessage()]
    assert len(said) == 1, "once, not once per poll"


def test_a_fabric_that_answers_inside_the_wait_is_used():
    fab = fabric(office=(True,), kitchen=(False,))
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(),
                               phone=lambda ip, mac: True, boot_grace_s=0.5)
    probe("10.0.0.5", "")
    assert probe.legs["rooms"] == pv.ROOMS_ON


# =====================================================================
# (c) A RUN THAT BEGAN BEFORE BOOT
# =====================================================================
def test_a_latched_radar_at_boot_is_a_pre_existing_run_with_no_honest_age(tmp_path):
    clock = {"t": 1788709803.5}
    stuck = sr.StuckRooms(path=tmp_path / "roomruns.json",
                          now=lambda: clock["t"])
    stuck.observe("office", True)
    row = stuck.status()["office"]
    assert row["pre_existing"] is True
    assert row["run_s"] is None, \
        "the radar's latch may be seconds or hours old and the box cannot know"


def test_a_rising_edge_after_boot_is_an_ordinary_run(tmp_path):
    clock = {"t": 1788709803.5}
    stuck = sr.StuckRooms(path=tmp_path / "roomruns.json",
                          now=lambda: clock["t"])
    stuck.observe("office", False)
    clock["t"] += 30.0
    stuck.observe("office", True)
    row = stuck.status()["office"]
    assert row["pre_existing"] is False
    assert row["run_s"] == 0.0


def test_cell6_holds_unknown_on_a_pre_existing_run_nothing_agrees_with():
    """NEVER home -- that is an unearned 15-minute window. NEVER away -- a
    false away on a man at his desk after a restart is the worse error."""
    state, reason = pv.cell6(agreed_s_ago=None, pre_existing=True,
                             mic=pv.MIC_SILENT)
    assert state == pv.UNKNOWN
    assert "restart" in reason or "before" in reason
    out = pv.decide(phone=pv.PHONE_NO, camera=pv.CAM_BLIND, rooms=pv.ROOMS_ON,
                    agreed_s_ago=None, agreed_pre_existing=True,
                    mic=pv.MIC_SILENT)
    assert out.state == pv.UNKNOWN and out.hold is True
    assert out.state not in (pv.HOME, pv.AWAY)


def test_one_corroboration_hands_a_pre_existing_run_back_to_the_arithmetic(tmp_path):
    clock = {"t": 1788709803.5}
    stuck = sr.StuckRooms(path=tmp_path / "roomruns.json",
                          now=lambda: clock["t"])
    stuck.observe("office", True)
    clock["t"] += 10.0
    stuck.corroborate("phone")
    row = stuck.status()["office"]
    assert row["pre_existing"] is False
    assert row["corroborated_s_ago"] == 0.0
    state, _ = pv.cell6(agreed_s_ago=0.0, pre_existing=False, mic=pv.MIC_SILENT)
    assert state == pv.HOME


def test_the_probe_reports_the_pre_existing_flag_to_the_cell(tmp_path):
    clock = {"t": 1788709803.5}
    stuck = sr.StuckRooms(path=tmp_path / "roomruns.json",
                          now=lambda: clock["t"])
    fab = fabric(office=(True,), kitchen=(False,))
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), stuck=stuck,
                               phone=lambda ip, mac: False,
                               now=lambda: clock["t"], boot_grace_s=0.0)
    probe("10.0.0.5", "")       # starts the grace clock, opens the run
    # 20 min: past the 12-minute phone grace, well under the stuck
    # detector's 45, so this really is cell 6 and not cell 15.
    clock["t"] += 1200.0
    probe("10.0.0.5", "")
    assert probe.legs["phone"] == pv.PHONE_NO
    assert probe.verdict.cell == 6
    assert probe.verdict.state == pv.UNKNOWN


# =====================================================================
# (4) THE MIC WORDING: THREE CAUSES, THREE SENTENCES
# =====================================================================
def test_the_three_mic_silences_are_three_different_sentences():
    def why(**kw):
        return pv.cell6(agreed_s_ago=9999.0, mic=pv.MIC_UNKNOWN, **kw)[1]

    no_reader = why(mic_reason=pv.MIC_WHY_NO_READER)
    unreadable = why(mic_reason=pv.MIC_WHY_UNREADABLE)
    no_turn = why(mic_reason=pv.MIC_WHY_NO_TURN)
    assert len({no_reader, unreadable, no_turn}) == 3
    assert "could not be read" not in no_turn, \
        "a ledger with no turn yet is not an unreadable ledger"
    assert "since Jarvis started" in no_turn
    assert "wired" in no_reader
    assert "could not be read" in unreadable


def test_the_probe_names_which_of_the_three_it_is():
    fab = fabric()
    none_probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), mic=None)
    assert none_probe._mic_leg()[2] == pv.MIC_WHY_NO_READER

    def boom():
        raise RuntimeError("no ledger")

    raiser = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), mic=boom)
    assert raiser._mic_leg()[2] == pv.MIC_WHY_UNREADABLE

    fresh = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), mic=lambda: None)
    leg, ago, reason = fresh._mic_leg()
    assert leg == pv.MIC_UNKNOWN and ago is None
    assert reason == pv.MIC_WHY_NO_TURN


def test_a_live_ledger_with_a_turn_is_still_heard():
    fab = fabric()
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), mic=lambda: 30.0)
    leg, ago, reason = probe._mic_leg()
    assert leg == pv.MIC_HEARD and ago == 30.0 and reason == ""


# =====================================================================
# THE AMBIGUOUS CELLS ARE WHERE THE CAMERA EARNS "NUMBER ONE"
# =====================================================================
@pytest.mark.parametrize("answer,leg", [(False, pv.PHONE_NO),
                                        (None, pv.PHONE_UNKNOWN)])
def test_the_lens_is_armed_for_the_two_cells_his_rule_one_cannot_reach(answer, leg):
    clock = {"t": 5000.0}
    looks = {"n": 0}
    seen = {"v": ("", None, False)}

    def look(timeout_s=1.5, why=""):
        looks["n"] += 1
        seen["v"] = ("hunter", 1, True)
        return True

    probe = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(),
                               phone=lambda ip, mac: answer,
                               eye=lambda: seen["v"], look=look,
                               now=lambda: clock["t"], boot_grace_s=0.0)
    clock["t"] += 3600.0
    probe("10.0.0.5", "")
    assert looks["n"] == 1, "rooms ON with no phone answer must ask the lens"
    assert probe.legs["camera"] == pv.CAM_SAW
    assert probe.verdict.state == pv.HOME


def test_a_clear_house_never_lights_the_lamp():
    """The producer must not free-run: no burst when the vote is not close."""
    looks = {"n": 0}
    probe = pres.ThreeLegProbe(
        fabric=fabric(office=(False,), kitchen=(False,)), cfg=Cfg(),
        phone=lambda ip, mac: True, eye=lambda: ("", None, False),
        look=lambda timeout_s=1.5, why="": looks.__setitem__(
            "n", looks["n"] + 1))
    probe("10.0.0.5", "")
    assert looks["n"] == 0


# =====================================================================
# THE REPLAY: 2026-09-06, 10:48:58 -> 10:49:04.9, second by second
# =====================================================================
def test_the_exact_boot_sequence_of_2026_09_06_produces_no_away():
    """His log, replayed on a fake clock at the real offsets:

      10:48:58.0  presence: three-leg voter active   (t+0.0)
      10:49:02.4  roomfabric: watching office, kitchen every 2.0s
      10:49:03.5  presence: away (cell 24)   <-- the line under test
      10:49:04.9  roomfabric: office          (1.4 s AFTER the verdict)

    He was standing in the flat. Two independent causes, and the replay
    has to kill BOTH: the fabric had not polled, and the phone leg called
    one unanswered ping a departure.
    """
    clock = {"t": 0.0}
    published = []
    # The fabric's readers answer only once the fabric has "started" --
    # exactly the state Room.value is in before the first poll.
    started = {"on": False}

    class Late:
        configured = True

        def __init__(self, value):
            self.value = value

        def read(self):
            return self.value if started["on"] else None

    fab = rf.from_readers([("office", Late(True)), ("kitchen", Late(False))],
                          poll_s=2.0, enter_hold_s=2.0,
                          now=lambda: clock["t"])
    stuck_path = None
    probe = pres.ThreeLegProbe(
        fabric=fab, cfg=Cfg(**{"presence.away_after_min": 12}),
        phone=lambda ip, mac: False,          # his iPhone ignored the ping
        now=lambda: clock["t"], boot_grace_s=6.0)
    del stuck_path
    s = pres.PresenceSentinel(
        Cfg(**{"presence.phone_ip": "192.168.50.34",
               "presence.enabled": True}),
        publish=published.append, probe_fn=probe,
        now=lambda: clock["t"], poll_s=60.0)

    clock["t"] = 5.5                      # 10:49:03.5 -- the offending tick
    assert s.tick() is None
    assert s.state == "unknown", "a boot must not decide he is out"
    assert probe.verdict.cell != 24
    assert probe.verdict.state != pv.AWAY
    assert published == [], "no Presence event may leave this tick"

    # 10:49:04.9: the fabric finally answers, and the office is occupied.
    started["on"] = True
    clock["t"] = 6.9
    fab.tick()
    s.tick()
    assert s.state != "away"
    assert probe.legs["rooms"] == pv.ROOMS_ON
    assert probe.legs["phone"] == pv.PHONE_UNKNOWN, \
        "one unanswered ping is a napping radio, not a departure"


def test_the_14_49_47_restart_at_the_desk_produces_no_away():
    """THE SECOND HALF OF THE SAME DEFECT, and the reason it is a separate
    test: 10:49:03 was a cold boot, and 14:49:47 was HIM RESTARTING JARVIS
    WHILE SITTING AT HIS DESK, five seconds earlier. That is the case where
    the box has the MOST evidence and got the answer most wrong:

      14:49:42   boot
      14:49:47   presence: away (cell 24) -- phone-only away past the
                 grace, no room readable

    Three things were true at once and every one of them had to be read as
    "no": the fabric had not polled, his phone had ignored one ping, and
    the office radar had been latched since before the restart. So this
    replay drives all three -- and adds the fourth leg he actually had, a
    keyboard he had touched seconds before, which by itself now makes the
    line impossible.
    """
    clock = {"t": 0.0}
    published = []
    started = {"on": False}

    class Late:
        configured = True
        blocked = ""

        def __init__(self, value):
            self.value = value

        def read(self):
            return self.value if started["on"] else None

    fab = rf.from_readers([("office", Late(True)), ("kitchen", Late(False))],
                          poll_s=2.0, enter_hold_s=2.0,
                          now=lambda: clock["t"])
    probe = pres.ThreeLegProbe(
        fabric=fab, cfg=Cfg(**{"presence.away_after_min": 12}),
        phone=lambda ip, mac: False,        # his iPhone ignored the ping
        desk=lambda: 8.0,                   # he had just typed the restart
        now=lambda: clock["t"], boot_grace_s=6.0)
    s = pres.PresenceSentinel(
        Cfg(**{"presence.phone_ip": "192.168.50.34"}),
        publish=published.append, probe_fn=probe,
        now=lambda: clock["t"], poll_s=60.0)

    clock["t"] = 5.0                        # 14:49:47 -- the offending tick
    assert s.tick() is None
    assert probe.verdict.state != pv.AWAY
    assert probe.legs["desk"] == pv.DESK_AT
    assert published == [], "nothing may reach the greeter from this tick"

    # And the latched radar that comes with a restart at the desk: it must
    # not buy a 15-minute HOME window either.
    started["on"] = True
    clock["t"] = 7.0
    fab.tick()
    s.tick()
    assert s.state != "away"
    assert probe.legs["rooms"] == pv.ROOMS_ON


def test_the_verdict_sentence_about_the_grace_is_now_true():
    """The 10:49:03 line said "phone-only away past the grace" when the
    grace was 0.0. Whenever PHONE_NO is reported now, the phone really has
    been silent for the whole grace."""
    clock = {"t": 100.0}
    probe = pres.ThreeLegProbe(fabric=fabric(office=(False,), kitchen=(False,)),
                               cfg=Cfg(), phone=lambda ip, mac: False,
                               now=lambda: clock["t"])
    for elapsed in (0.0, 1.0, probe.grace_s - 1.0):
        clock["t"] = 100.0 + elapsed
        probe("10.0.0.5", "")
        assert probe.legs["phone"] == pv.PHONE_UNKNOWN, elapsed
    clock["t"] = 100.0 + probe.grace_s
    probe("10.0.0.5", "")
    assert probe.legs["phone"] == pv.PHONE_NO
    assert "past the grace" in probe.verdict.reason
