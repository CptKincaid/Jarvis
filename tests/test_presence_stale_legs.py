"""THREE WAYS A LEG KEPT ANSWERING AFTER IT HAD STOPPED KNOWING.

Every image in this file is a synthetic numpy array this file made, handed
to a FAKE opener; nothing here opens /dev/video*, no frame is written,
saved or looked at, and every camera claim below is a COUNT (bursts,
frames, device opens). Every utterance and every idle number is one this
file wrote. Nothing reads his config, his mailbox or his calendar.

  BLOCK 1  A PERMANENT FALSE HOME WHILE HE IS OUT.
           ``DeskSentinel.idle_s()`` handed out ``self.last_idle``, a cache
           written ONLY on a successful poll. A None reading returned early
           and left it untouched; ``enabled`` False skipped the write
           entirely. So the last good number stood FOR EVER: 400 no-signal
           polls later -- three hours and twenty minutes of fake clock --
           it still answered 5.0 s, which the voter reads as DESK_AT, which
           vetoes away in every cell. At the sentinel, with the rooms clear
           and his phone gone from t=0, AWAY NEVER FIRED.

           Its twin cannot do this. ``turnclock.TurnLedger.idle_s`` computes
           ``clock() - last_mark``, so a dead microphone AGES to SILENT.
           THE GUARD-ONE-HALF SHAPE, seventh occurrence on this project in
           two days: a guard built for one of a symmetric pair and never
           applied to its twin.

  BLOCK 2  THE LENS OPENED IN AN EMPTY OFFICE. The producer was armed
           whenever ANY room read occupied, and the lens is in the OFFICE.
           His flat is a corridor: him in the kitchen with the office empty
           armed 90 bursts an hour -- 2250 synthetic frames and 90 device
           opens -- to look at a room he was not in. It is his lamp and his
           lens, and CAM_LOOKED from the office is scored as "looked and
           saw nobody" for the whole flat.

  BLOCK 3  THE WITNESS AGE WAS APPLIED TO ONE HALF. ``decide``'s away guard
           and ``decide_rooms_only`` both stamp ``Verdict.witness_s_ago``;
           cell 6's own witness branch returned a bare ``(state, reason)``
           and left it None. With a 660 s keystroke, cell 15 said 660.0,
           rooms-only said 660.0, and cell 6 said None -- so the sentinel
           read cell 6's HOME as "seen NOW", restarted its own 12-minute
           grace every poll, and away landed at 26.8 min on a latched
           office instead of 15.0.

           AND THE MIRROR TEST COULD NOT SEE IT. The existing symmetry pin
           compares the mic against the desk, and in cell 6 they were
           EQUALLY wrong, so they matched. A mirror test that cannot see a
           one-sided defect is the pattern wearing the costume of its own
           cure; the cross-CELL pin below is what actually catches it.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from jarvis import camera as camera_mod
from jarvis import deskpresence as desk_mod
from jarvis import eye as eye_mod
from jarvis import presence as pres
from jarvis import presencevote as pv
from jarvis import roomfabric as rf
from jarvis.deskpresence import DeskSentinel


class Cfg:
    def __init__(self, **kw):
        self._d = dict(kw)

    def get(self, key, default=None):
        return self._d.get(key, default)


class Reader:
    def __init__(self, value=None, blocked=""):
        self.value = value
        self.blocked = blocked
        self.configured = True

    def read(self):
        return None if self.blocked else self.value


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def now(self):
        return self.t

    def advance(self, s):
        self.t += float(s)


@pytest.fixture(autouse=True)
def _desk_enabled(monkeypatch):
    """conftest switches desk presence off for the whole suite (the session
    bus is the developer's real desktop). Every sentinel here is driven by
    a fake idle function this file wrote, so the switch goes back on -- and
    the one case that tests the switch itself sets it again by hand."""
    monkeypatch.delenv(desk_mod.ENV_OFF, raising=False)


def _sentinel(idle=0.0, cfg=None, **kw):
    """A REAL DeskSentinel over a fake idle function. ``answers['idle']``
    is what the next poll returns; None is "no signal"."""
    clock = Clock()
    answers = {"idle": idle}
    s = DeskSentinel(cfg if cfg is not None else Cfg(),
                     publish=lambda ev: None,
                     idle_fn=lambda: answers["idle"],
                     now=clock.now, **kw)
    s.clock, s.answers = clock, answers
    return s


# =====================================================================
# BLOCK 1 -- A READING WITH NO AGE IS A PERMANENT FALSE HOME
# =====================================================================
def test_a_fresh_reading_is_still_handed_out():
    """The guard must not cost the ordinary case: a poll, then the number."""
    s = _sentinel(idle=5.0, poll_s=30.0)
    s.tick()
    assert s.idle_s() == 5.0
    assert s.idle_age_s() == 0.0


def test_four_hundred_no_signal_polls_do_not_leave_a_five_second_reading():
    """THE MEASUREMENT. One good poll at 5.0 s, then 400 consecutive
    no-signal polls -- 3 h 20 min at his 30 s cadence. The cache said 5.0 s
    for every one of them, and 5.0 s is DESK_AT, which vetoes away."""
    s = _sentinel(idle=5.0, poll_s=30.0)
    s.tick()
    assert pv.desk_leg(idle_s=s.idle_s()) == pv.DESK_AT
    s.answers["idle"] = None
    for _ in range(400):
        s.clock.advance(30.0)
        s.tick()
    assert s.clock.t - 1_000_000.0 == 12_000.0            # 3 h 20 min
    assert s.idle_s() is None, "a reading nothing has refreshed is not news"
    assert pv.desk_leg(idle_s=s.idle_s()) == pv.DESK_UNKNOWN


def test_the_reading_dies_two_poll_periods_after_it_was_taken():
    """The bound, both sides of it, at his ``presence.desk_poll_s`` of 30."""
    s = _sentinel(idle=5.0, poll_s=30.0)
    s.tick()
    assert s.stale_after_s == 60.0
    s.clock.advance(59.0)
    assert s.idle_s() == 5.0
    s.clock.advance(2.0)
    assert s.idle_s() is None


def test_a_switched_off_sentinel_hands_out_nothing():
    """``presence.desk`` False skipped the WRITE and left the last number
    standing, so switching the leg off made it permanent instead of
    silent."""
    cfg = Cfg(**{"presence.desk": True})
    s = _sentinel(idle=5.0, poll_s=30.0, cfg=cfg)
    s.tick()
    assert s.idle_s() == 5.0
    cfg._d["presence.desk"] = False
    assert s.enabled is False
    assert s.idle_s() is None


def test_a_stopped_sentinel_hands_out_nothing():
    s = _sentinel(idle=5.0, poll_s=30.0)
    s.tick()
    s.stop()
    assert s.idle_s() is None


def test_the_process_wide_switch_silences_the_reading_too(monkeypatch):
    """JARVIS_DESK_PRESENCE=0 is what the suite itself sets, and it is the
    switch a developer throws to keep the voter off a live session bus."""
    s = _sentinel(idle=5.0, poll_s=30.0)
    s.tick()
    monkeypatch.setenv(desk_mod.ENV_OFF, "0")
    assert s.enabled is False
    assert s.idle_s() is None


def test_the_desk_ages_the_way_its_twin_the_mic_already_did():
    """THE SYMMETRY THIS BLOCK IS ABOUT. ``TurnLedger.idle_s`` is
    ``clock() - last_mark``: a microphone that stops marking AGES, so it
    reaches SILENT on its own. The desk must reach UNKNOWN the same way."""
    from jarvis import turnclock

    clock = Clock()
    led = turnclock.TurnLedger(clock=clock.now)
    led.mark("wake")
    desk = _sentinel(idle=5.0, poll_s=30.0)
    desk.tick()
    assert pv.mic_leg(s_ago=led.idle_s()) == pv.MIC_HEARD
    assert pv.desk_leg(idle_s=desk.idle_s()) == pv.DESK_AT

    clock.advance(3600.0)          # both legs stop being fed
    desk.clock.advance(3600.0)
    assert pv.mic_leg(s_ago=led.idle_s()) == pv.MIC_SILENT
    assert pv.desk_leg(idle_s=desk.idle_s()) == pv.DESK_UNKNOWN


def test_away_fires_at_the_sentinel_with_a_dead_idle_monitor():
    """END TO END, AND THIS IS THE SERIOUS ONE. Rooms clear, his phone gone
    from t=0, six hours of polls, and a desk sentinel whose idle monitor
    answered ONCE and then went silent. Away has to arrive."""
    clock = Clock(t=0.0)
    desk = DeskSentinel(Cfg(), publish=lambda ev: None,
                        idle_fn=lambda: 5.0, now=clock.now, poll_s=30.0)
    desk.tick()                       # the one good reading, at t=0
    desk._idle = lambda: None         # the monitor goes away for good

    fab = rf.from_readers([("office", Reader(False)),
                           ("kitchen", Reader(False))],
                          poll_s=2.0, enter_hold_s=2.0)
    probe = pres.ThreeLegProbe(
        fabric=fab, cfg=Cfg(**{"presence.away_after_min": 12}),
        phone=lambda ip, mac: False, desk=desk.idle_s,
        desk_age=desk.idle_age_s, now=clock.now,
        grace_s=0.0, boot_grace_s=0.0)
    sentinel = pres.PresenceSentinel(
        Cfg(**{"presence.phone_ip": "10.0.0.5"}), publish=lambda e: None,
        probe_fn=probe, now=clock.now, poll_s=10.0)

    away_at = None
    while clock.t < 6 * 3600.0:
        desk.tick()
        sentinel.tick()
        if sentinel.state == "away":
            away_at = clock.t / 60.0
            break
        clock.advance(10.0)
    assert away_at is not None, "AWAY NEVER FIRED -- he was out for six hours"
    assert away_at < 20.0, away_at


# ------------------------------------------------- the belt, at the voter
def test_the_voter_refuses_a_stale_reading_even_if_one_reaches_it():
    """BELT. ``Attention.usable`` already bounds the camera by age; the
    desk gets the same bound, so a reader that forgets to expire its own
    cache cannot manufacture a home."""
    assert pv.desk_leg(idle_s=5.0, age_s=0.0) == pv.DESK_AT
    assert pv.desk_leg(idle_s=5.0, age_s=59.0) == pv.DESK_AT
    assert pv.desk_leg(idle_s=5.0, age_s=3600.0) == pv.DESK_UNKNOWN
    for junk in ("banana", float("nan"), -1.0, float("inf")):
        assert pv.desk_leg(idle_s=5.0, age_s=junk) == pv.DESK_UNKNOWN


def test_the_desk_age_bound_is_two_of_deskpresence_s_own_poll_periods():
    """A NUMBER IN SHIPPED SOURCE MUST BE REPRODUCIBLE IN THIS TREE. This
    one is 2 x ``deskpresence.DEFAULT_POLL_S``, and here is the arithmetic."""
    assert desk_mod.DEFAULT_POLL_S == 30.0
    assert pv.DESK_MAX_AGE_S == 2.0 * desk_mod.DEFAULT_POLL_S == 60.0


def test_the_probe_hands_the_voter_the_age_it_was_given():
    """The belt is only a belt if the live path actually buckles it."""
    fab = rf.from_readers([("office", Reader(True)), ("kitchen", Reader(False))],
                          poll_s=2.0, enter_hold_s=2.0)
    fresh = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), desk=lambda: 5.0,
                               desk_age=lambda: 10.0)
    assert fresh._desk_leg() == (pv.DESK_AT, 5.0)
    stale = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), desk=lambda: 5.0,
                               desk_age=lambda: 3600.0)
    assert stale._desk_leg() == (pv.DESK_UNKNOWN, None)


def test_no_age_reader_at_all_is_the_old_behaviour():
    """A box that cannot say how old its reading is keeps voting; the
    sentinel's own expiry is the guard there, and this is the belt."""
    fab = rf.from_readers([("office", Reader(True)), ("kitchen", Reader(False))],
                          poll_s=2.0, enter_hold_s=2.0)
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), desk=lambda: 5.0)
    assert probe._desk_leg() == (pv.DESK_AT, 5.0)


def test_an_age_reader_that_raises_costs_the_freshness_not_the_vote():
    def boom():
        raise RuntimeError("no monitor")

    fab = rf.from_readers([("office", Reader(True)), ("kitchen", Reader(False))],
                          poll_s=2.0, enter_hold_s=2.0)
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(), desk=lambda: 5.0,
                               desk_age=boom)
    assert probe._desk_leg() == (pv.DESK_AT, 5.0)


# =====================================================================
# BLOCK 2 -- THE LENS MAY ONLY OPEN FOR THE ROOM IT CAN SEE
# =====================================================================
class Device:
    """A capture handle over a SYNTHETIC array this file made."""

    def __init__(self, h=8, w=8):
        self.reads = 0
        self.released = 0
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)

    def read(self):
        self.reads += 1
        return True, self._frame.copy()

    def release(self):
        self.released += 1


class Detector:
    """Rows in the 15-column shape jarvis/visionrig.py declares."""

    def __init__(self, faces=0):
        self.faces = faces
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        rows = []
        for i in range(self.faces):
            row = np.zeros(15, dtype=np.float32)
            row[14] = 0.9 - 0.01 * i
            rows.append(row)
        return rows


class Policy:
    def __init__(self, allow=True):
        self.allow = allow
        self.attached = {}

    def allowed(self, kind):
        return self.allow is True

    def attach(self, name, stop, present=None, resume=None):
        self.attached[name] = stop


def _lens(clock):
    """A feed behind a FAKE opener plus the producer, on a fake clock."""
    from jarvis import eyeloop as eyeloop_mod

    device = Device()
    cfg = Cfg(**{"camera.enabled": True, "camera.device": "/dev/video-fake",
                 "camera.width": 1280, "camera.height": 720,
                 "camera.hfov_deg": 65.6})
    feed, why = camera_mod.build(cfg, Policy(), opener=lambda: device)
    assert feed is not None, why
    # The eye's clock is the seam its age arithmetic runs on; without it a
    # reading published in this test never ages and the camera leg answers
    # CAM_LOOKED for ever, which is a different test.
    feed.eye._now = clock.now
    loop = eyeloop_mod.EyeLoop(feed, detector=Detector(faces=0),
                               curfew=None, sleep=lambda s: None,
                               now=clock.now)
    return feed, device, loop


def _eye_leg_of(feed):
    """Exactly what ``app._eye_leg`` does, without booting an app."""
    from jarvis import app as app_mod
    a = object.__new__(app_mod.JarvisApp)
    a.services = type("S", (), {"camera_feed": feed})()
    return a._eye_leg()


def _drive(office, kitchen, votes=90, step_s=40.0):
    """``votes`` votes at ``step_s`` apart -- 90 x 40 s is one hour, the
    span the arming cost was measured over. Returns the NUMBERS."""
    clock = Clock()
    feed, device, loop = _lens(clock)
    fab = rf.from_readers([("office", Reader(office)),
                           ("kitchen", Reader(kitchen))],
                          poll_s=2.0, enter_hold_s=0.0)
    probe = pres.ThreeLegProbe(
        fabric=fab, cfg=Cfg(),
        phone=lambda ip, mac: None,               # unaskable
        eye=lambda: _eye_leg_of(feed),
        look=loop.wait_for_look, now=clock.now,
        grace_s=0.0, boot_grace_s=0.0)
    for _ in range(votes):
        probe("10.0.0.5", "")
        clock.advance(step_s)
    return {"bursts": loop.bursts, "frames": loop.frames,
            "opens": feed.eye.opens, "reads": device.reads,
            "arms": loop.arms}


def test_the_lens_does_not_open_for_a_room_it_cannot_see():
    """HIS FLAT IS A CORRIDOR AND THE LENS IS IN THE OFFICE. Him in the
    kitchen, the office empty, his phone unaskable: an hour of votes used
    to cost 90 bursts, 2250 frames and 90 device opens, to look at a room
    he was not in."""
    got = _drive(office=False, kitchen=True)
    assert got == {"bursts": 0, "frames": 0, "opens": 0, "reads": 0,
                   "arms": 0}, got


def test_the_lens_still_opens_for_the_room_it_is_in():
    """The other half, and the reason the leg exists at all: the office
    reads occupied with the phone quiet is cell 6, and cell 6 is exactly
    where a look ends the argument."""
    got = _drive(office=True, kitchen=False)
    assert got["bursts"] == 90, got
    assert got["frames"] == 90 * 25, got
    assert got["opens"] == 90, got


def test_no_room_occupied_opens_nothing():
    assert _drive(office=False, kitchen=False)["bursts"] == 0


def test_a_desk_room_that_is_not_a_configured_sensor_never_arms():
    """A box whose desk room has no radar cannot be told the lens has
    something to look at, so it does not guess. Said out loud in the log
    rather than left as a silent dark leg."""
    clock = Clock()
    feed, device, loop = _lens(clock)
    fab = rf.from_readers([("kitchen", Reader(True))],
                          poll_s=2.0, enter_hold_s=0.0)
    probe = pres.ThreeLegProbe(
        fabric=fab, cfg=Cfg(), desk_room="office",
        phone=lambda ip, mac: None, eye=lambda: _eye_leg_of(feed),
        look=loop.wait_for_look, now=clock.now,
        grace_s=0.0, boot_grace_s=0.0)
    for _ in range(20):
        probe("10.0.0.5", "")
        clock.advance(40.0)
    assert (loop.bursts, feed.eye.opens) == (0, 0)


# =====================================================================
# BLOCK 3 -- THE WITNESS AGE, ON EVERY CELL THAT USES A WITNESS
# =====================================================================
TRIPLES = tuple(itertools.product(
    (pv.ROOMS_ON, pv.ROOMS_CLEAR, pv.ROOMS_UNREACHABLE),
    (pv.PHONE_YES, pv.PHONE_NO, pv.PHONE_UNKNOWN),
    (pv.CAM_SAW, pv.CAM_LOOKED, pv.CAM_BLIND)))


@pytest.mark.parametrize("witness", ("mic", "desk"))
def test_every_cell_a_witness_rescued_says_how_old_that_witness_is(witness):
    """THE CROSS-CELL PIN, and the one the mic-vs-desk mirror could not
    make. Cell 6 has its OWN witness branch; the away guard has another.
    Both must stamp the age, or the sentinel reads a stale keystroke as
    "seen now" and its 12-minute grace restarts every poll."""
    ago = 660.0
    kw = ({"mic": pv.MIC_HEARD, "mic_s_ago": ago} if witness == "mic"
          else {"desk": pv.DESK_AT, "desk_s_ago": ago})
    rescued = 0
    for rooms, phone, camera in TRIPLES:
        out = pv.decide(rooms=rooms, phone=phone, camera=camera,
                        agreed_s_ago=9999.0, **kw)
        bare = pv.decide(rooms=rooms, phone=phone, camera=camera,
                         agreed_s_ago=9999.0)
        if out.state == pv.HOME and bare.state != pv.HOME:
            rescued += 1
            assert out.witness_s_ago == ago, (rooms, phone, camera, out.cell)
    assert rescued >= 6, rescued


def test_cell_six_reports_its_witness_age_exactly_as_its_neighbours_do():
    """MEASURED end to end with a 660 s keystroke: cell 15 -> 660.0,
    rooms-only -> 660.0, cell 6 -> None. Three code paths, one question,
    two of them answered."""
    kw = {"desk": pv.DESK_AT, "desk_s_ago": 660.0}
    six = pv.decide(rooms=pv.ROOMS_ON, phone=pv.PHONE_NO,
                    camera=pv.CAM_BLIND, agreed_s_ago=9999.0, **kw)
    fifteen = pv.decide(rooms=pv.ROOMS_CLEAR, phone=pv.PHONE_NO,
                        camera=pv.CAM_BLIND, agreed_s_ago=9999.0, **kw)
    plain = pv.decide_rooms_only(rooms=pv.ROOMS_CLEAR, **kw)
    assert six.cell == 6 and fifteen.cell == 15
    assert (six.state, fifteen.state, plain.state) == (pv.HOME,) * 3
    assert fifteen.witness_s_ago == 660.0
    assert plain.witness_s_ago == 660.0
    assert six.witness_s_ago == 660.0, "cell 6 is the half that was missed"


def test_cell_six_returns_its_own_witness_age():
    """At the cell itself, so the next caller cannot lose it again."""
    state, reason, ago = pv.cell6(agreed_s_ago=9999.0, mic=pv.MIC_SILENT,
                                  desk=pv.DESK_AT, desk_s_ago=660.0)
    assert (state, ago) == (pv.HOME, 660.0)
    assert "11 min" in reason
    assert pv.cell6(agreed_s_ago=9999.0, mic=pv.MIC_SILENT)[2] is None
    assert pv.cell6(agreed_s_ago=60.0, mic=pv.MIC_SILENT)[2] is None


def _away_at(latched, keystroke_s, tmp_path):
    """When does away arrive, in minutes after he walked out, with the desk
    vouching for him from a keystroke ``keystroke_s`` old and never
    refreshed? ``latched`` pins the office occupied the whole time."""
    from jarvis import stuckroom as sr

    clock = Clock(t=0.0)
    left = 600.0
    office = Reader(False)
    fab = rf.from_readers([("office", office), ("kitchen", Reader(False))],
                          poll_s=2.0, enter_hold_s=0.0)
    stuck = sr.StuckRooms(path=tmp_path / "roomruns.json", now=clock.now,
                          publish=lambda ev: None) if latched else None
    probe = pres.ThreeLegProbe(
        fabric=fab, cfg=Cfg(**{"presence.away_after_min": 12}), stuck=stuck,
        phone=lambda ip, mac: clock.t < left,
        # The keystroke is a fixed moment: it ages with the clock and is
        # never refreshed, which is what "he walked out" means.
        desk=lambda: (clock.t - left) + keystroke_s,
        now=clock.now, boot_grace_s=0.0)
    sentinel = pres.PresenceSentinel(
        Cfg(**{"presence.phone_ip": "10.0.0.5"}), publish=lambda e: None,
        probe_fn=probe, now=clock.now, poll_s=10.0)
    if latched:
        probe("10.0.0.5", "")
        office.value = True
    while clock.t < left + 4 * 3600.0:
        sentinel.tick()
        if sentinel.state == "away":
            return round((clock.t - left) / 60.0, 1)
        clock.advance(10.0)
    return None


def test_the_stacked_clocks_cost_measured_on_a_latched_office(tmp_path):
    """THE PRICE OF THE MISSING STAMP, END TO END AND IN MINUTES.

    He walks out; his phone stops answering; his last keystroke is the
    moment he left, so the desk vouches for him for its 15-minute window
    and then stops. MEASURED in this tree, away after he walked out:

        rooms clear    stamped (cell 15)   15.0 min
        office latched cell 6, no stamp    26.8 min      <- the defect
        office latched cell 6, stamped     15.0 min

    The two paths have to AGREE, because it is one question asked in two
    cells, and the sentinel's 12-minute grace must run from the keystroke
    in both. 26.8 is what the grace restarting at every poll costs.
    """
    latched = _away_at(latched=True, keystroke_s=0.0, tmp_path=tmp_path)
    clear = _away_at(latched=False, keystroke_s=0.0, tmp_path=tmp_path)
    assert clear == 15.0, clear
    assert latched == 15.0, latched


def test_a_keystroke_already_stale_when_he_leaves_costs_nothing(tmp_path):
    """The other end of the same measurement, and the one an earlier
    comment in presencevote.py attached the 26.8 to by mistake: a keystroke
    ALREADY 11 minutes old when he walks out has four minutes of window
    left, so away lands at the phone's own grace and the desk is free."""
    assert _away_at(latched=False, keystroke_s=660.0,
                    tmp_path=tmp_path) == 11.8


# =====================================================================
# SMALLER -- A READING TAKEN WHILE PERMITTED KEPT VOTING AFTER HE SAID STOP
# =====================================================================
def test_a_reading_stops_voting_the_moment_he_switches_sensing_off():
    """``Eye.state()`` read ``_reading_dark``, a flag stamped at PUBLISH
    time, and ``_go_blind`` -- which clears it -- only ever runs from
    ``capture()``. Between bursts nothing captures BY DESIGN ("the lens is
    dark in between"), so a reading published a moment before he switched
    sensing off went on voting for the full 30 s presence window with
    nobody left to invalidate it. The permission is re-read here."""
    allow = {"ok": True}
    eye = eye_mod.Eye(allow=lambda: allow["ok"],
                      open_device=lambda: Device(), now=Clock().now)
    eye.publish(eye_mod.Attention(faces=1, identity="hunter", id_score=0.8))
    assert eye.state().usable(eye_mod.PRESENCE_MAX_AGE_S) is True

    allow["ok"] = False              # "camera off for ten minutes"
    assert eye.state().dark is True
    assert eye.state().usable(eye_mod.PRESENCE_MAX_AGE_S) is False


def test_a_sensing_owner_that_raises_is_not_permission_either():
    def boom():
        raise RuntimeError("the policy is gone")

    eye = eye_mod.Eye(allow=lambda: True, open_device=lambda: Device(),
                      now=Clock().now)
    eye.publish(eye_mod.Attention(faces=1))
    assert eye.state().dark is False
    eye._allow = boom
    assert eye.state().dark is True
