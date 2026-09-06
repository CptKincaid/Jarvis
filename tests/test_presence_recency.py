"""Cell 6 must be honest about TIME, and a false away must be hard to reach.

Both defects this file pins live on one line of the first cut:

    corroborated = any(v.get("corroborated_s_ago") is not None ...)

That asks whether the occupied run has EVER been agreed with. The sentence
it printed said RECENTLY. Two opposite failures follow from the same line.

DEFECT 1 -- 2026-09-05 SURVIVES FOR SHORT TRIPS. The realistic latch is a
radar that saw him, he left, and it never cleared: the run STARTED while he
was home, so his phone stamped it dozens of times, so "ever corroborated"
stays true for ever. MEASURED on the sentinel end to end before the fix
(scripts/presence_cliff.py --ever): a trip of <= 55 minutes leaves the
sentinel reading home and nobody greets him; 56 minutes is the first that
does (the brief's harness read 57 -- one 60 s poll of phase). That is the
45-minute stuck fault plus the 12-minute away grace, and a shop run
reproduces 2026-09-05 exactly. After: 26 minutes, the 15-minute window
plus the grace less the one poll that was still fresh.

DEFECT 2 -- A FALSE AWAY, WHICH IS THE WORSE DIRECTION. He is at his desk,
the office radar correctly ON, the camera dark, his phone napping. MEASURED
before the fix: the voter said AWAY on poll 2 and the sentinel flipped at
13 minutes, and his next walk to the kitchen fired the full greeting at a
man who never left.

They are THE SAME CELL with opposite truths and no combination of the three
legs separates them. What separates them here is a fourth leg that already
exists in the tree -- the turn ledger arrival.departure_ready already reads
-- plus an honest clock. The asymmetry is the design: cell 6 reaches HOME
when EITHER clock is fresh and AWAY only when BOTH have run out.
"""
from __future__ import annotations

import pathlib
import uuid

import pytest

from jarvis import arrival, presence, presencevote as pv, roomfabric, stuckroom

Y, N, U = pv.PHONE_YES, pv.PHONE_NO, pv.PHONE_UNKNOWN
S, L, X = pv.CAM_SAW, pv.CAM_LOOKED, pv.CAM_BLIND
ON, CLR, UNR = pv.ROOMS_ON, pv.ROOMS_CLEAR, pv.ROOMS_UNREACHABLE


class FakeSensor:
    configured = True
    blocked = ""
    paused = False

    def __init__(self, value=None):
        self.value = value

    def read(self):
        return self.value

    def status(self):
        return {}


class Cfg:
    """AssistantConfig's read shape."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


# ====================================================================
# THE FOURTH LEG: the mic, from the ledger the tree already keeps
# ====================================================================
def test_a_turn_inside_the_window_is_heard_and_one_outside_it_is_silence():
    assert pv.mic_leg(s_ago=0.0, window_s=600.0) == pv.MIC_HEARD
    assert pv.mic_leg(s_ago=599.0, window_s=600.0) == pv.MIC_HEARD
    assert pv.mic_leg(s_ago=600.0, window_s=600.0) == pv.MIC_SILENT
    assert pv.mic_leg(s_ago=6000.0, window_s=600.0) == pv.MIC_SILENT


def test_no_ledger_is_unknown_and_never_silence():
    """The same rule as the camera: a leg that could not be READ has not
    heard nothing. An empty box has no turns.jsonl at all."""
    assert pv.mic_leg(s_ago=None, window_s=600.0) == pv.MIC_UNKNOWN
    assert pv.mic_leg(s_ago="banana", window_s=600.0) == pv.MIC_UNKNOWN
    assert pv.mic_leg(s_ago=-5.0, window_s=600.0) == pv.MIC_UNKNOWN


# ====================================================================
# CELL 6, HONEST ABOUT TIME
# ====================================================================
def test_corroboration_goes_stale_and_the_sentence_says_so():
    """The bug in one assertion: one hit at t=0 kept cell 6 HOME for ever."""
    fresh = pv.cell6(agreed_s_ago=60.0, recency_s=900.0, mic=pv.MIC_SILENT)
    stale = pv.cell6(agreed_s_ago=2640.0, recency_s=900.0, mic=pv.MIC_SILENT)
    assert fresh[0] == pv.HOME
    assert stale[0] == pv.AWAY


@pytest.mark.parametrize("s_ago,want", [
    (0.0, pv.HOME), (899.0, pv.HOME), (900.0, pv.AWAY), (901.0, pv.AWAY),
])
def test_the_window_is_a_real_edge_not_a_mood(s_ago, want):
    assert pv.cell6(agreed_s_ago=s_ago, recency_s=900.0,
                    mic=pv.MIC_SILENT)[0] == want


def test_the_sentence_matches_the_question_the_code_asks():
    """The first cut printed 'recently' while asking 'ever'. Whatever the
    sentence claims, a number for it has to be in the sentence."""
    home = pv.cell6(agreed_s_ago=60.0, recency_s=900.0, mic=pv.MIC_SILENT)[1]
    away = pv.cell6(agreed_s_ago=2640.0, recency_s=900.0, mic=pv.MIC_SILENT)[1]
    assert "1 min" in home or "60 s" in home, home
    assert "44 min" in away, away
    for line in (home, away):
        assert "15 min" in line or "15-min" in line, line


def test_a_run_with_no_history_at_all_is_not_held_against_the_radar():
    """Absence of history is not evidence of a fault -- the rule the first
    cut had and the one worth keeping."""
    assert pv.cell6(agreed_s_ago=None, recency_s=900.0,
                    mic=pv.MIC_SILENT)[0] == pv.HOME


# ====================================================================
# THE ASYMMETRY: either clock fresh is HOME, both run out for AWAY
# ====================================================================
def test_the_mic_alone_keeps_him_home_with_the_corroboration_long_stale():
    """He spoke four minutes ago. Nothing else has agreed for an hour."""
    state, reason = pv.cell6(agreed_s_ago=3600.0, recency_s=900.0,
                             mic=pv.MIC_HEARD, mic_s_ago=240.0)
    assert state == pv.HOME
    assert "mic" in reason


def test_away_needs_both_clocks_run_out_and_home_needs_only_one():
    """The whole asymmetry in one table. HOME on either; AWAY on neither."""
    for mic in (pv.MIC_HEARD, pv.MIC_SILENT, pv.MIC_UNKNOWN):
        assert pv.cell6(agreed_s_ago=10.0, recency_s=900.0,
                        mic=mic)[0] == pv.HOME, mic
    assert pv.cell6(agreed_s_ago=9999.0, recency_s=900.0,
                    mic=pv.MIC_HEARD)[0] == pv.HOME
    for mic in (pv.MIC_SILENT, pv.MIC_UNKNOWN):
        assert pv.cell6(agreed_s_ago=9999.0, recency_s=900.0,
                        mic=mic)[0] == pv.AWAY, mic


def test_cell_six_away_is_reachable_from_fewer_states_than_before():
    """COUNTED, not asserted by feel. Six (corroboration x mic) states; the
    first cut had the mic no vote at all, so away took 3 of 6. Now 2."""
    states = [(agreed, mic)
              for agreed in (10.0, 9999.0)
              for mic in (pv.MIC_HEARD, pv.MIC_SILENT, pv.MIC_UNKNOWN)]
    assert len(states) == 6
    away = [s for s in states
            if pv.cell6(agreed_s_ago=s[0], recency_s=900.0, mic=s[1])[0] == pv.AWAY]
    assert len(away) == 2, away


def test_a_named_face_still_ends_the_vote_before_cell_six_is_reached():
    """P1 is untouched: this is a change to cell 6, not to his ordering."""
    v = pv.decide(phone=N, camera=S, rooms=ON, agreed_s_ago=9999.0,
                  mic=pv.MIC_SILENT)
    assert v.state == pv.HOME
    assert v.cell == 4


def test_the_default_recency_is_named_and_its_provenance_is_written_down():
    assert pv.DEFAULT_RECENCY_S == 15 * 60.0
    assert "away_after_min" in pv.RECENCY_S_PROVENANCE
    assert "measured" in pv.RECENCY_S_PROVENANCE.lower()


# ====================================================================
# THE VOTER, WIRED
# ====================================================================
@pytest.fixture()
def clocked(tmp_path):
    now = {"t": 1_000_000.0}
    sr = stuckroom.StuckRooms(path=tmp_path / ("r-%s.json" % uuid.uuid4().hex),
                              publish=lambda e: None, now=lambda: now["t"])
    return sr, now


def build_fabric(office=None, kitchen=None, now=None):
    clock = {"t": 1000.0}
    fab = roomfabric.from_readers(
        [("office", FakeSensor(office)), ("kitchen", FakeSensor(kitchen))],
        now=(now or (lambda: clock["t"])), enter_hold_s=0.0, poll_s=2.0)
    fab._clock = clock
    return fab


def test_the_probe_reads_the_freshest_agreement_across_open_runs(clocked):
    stuck, now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False,
                                 mic=lambda: None, recency_s=900.0)
    stuck.observe("office", True)
    now["t"] += 60.0
    assert leg("1.2.3.4", "") is True          # 60 s old: fresh
    now["t"] += 1800.0
    assert leg("1.2.3.4", "") is False         # half an hour: stale
    assert leg.verdict.cell == 6


def test_the_mic_leg_is_reported_alongside_the_other_three(clocked):
    stuck, _now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False,
                                 mic=lambda: 30.0)
    leg("1.2.3.4", "")
    assert leg.legs["mic"] == pv.MIC_HEARD
    assert set(leg.legs) == {"phone", "camera", "rooms", "mic"}


def test_a_mic_reader_that_throws_is_unknown_and_never_breaks_the_vote(clocked):
    stuck, _now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()

    def boom():
        raise RuntimeError("no ledger")

    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True, mic=boom)
    assert leg("1.2.3.4", "") is True
    assert leg.legs["mic"] == pv.MIC_UNKNOWN


def test_the_mic_reader_never_receives_audio_only_a_number(clocked):
    """The boundary, pinned. The leg takes SECONDS SINCE A TURN off the
    ledger. Nothing here opens a device or reads a sample."""
    stuck, _now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    seen = []

    def reader():
        seen.append(True)
        return 42.0

    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False, mic=reader)
    leg("1.2.3.4", "")
    assert seen, "the mic leg was never asked"
    import inspect
    src = inspect.getsource(presence.ThreeLegProbe._mic_leg)
    for word in ("sounddevice", "pyaudio", "record", "stream", "wav"):
        assert word not in src.lower(), word


def test_a_heard_turn_stamps_the_run_at_the_turn_not_at_the_poll(clocked):
    """The fault detector and the voter must see the SAME evidence. Found
    by the instrument: with the mic hearing him every five minutes and
    nothing stamping the run, the 45-minute stuck fault dropped the office
    anyway, the rooms read CLEAR, and cell 15 -- which the mic never sees
    -- called him away at minute 57. The stamp is placed AT THE TURN: a
    turn nine minutes ago is not an agreement now."""
    stuck, now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    stuck.observe("office", True)
    now["t"] += 600.0
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False,
                                 mic=lambda: 300.0)
    leg("1.2.3.4", "")
    assert stuck.status()["office"]["corroborated_s_ago"] == pytest.approx(300.0)


def test_a_silent_or_unknown_mic_stamps_nothing(clocked):
    stuck, now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    stuck.observe("office", True)
    now["t"] += 600.0
    for reader in (lambda: 6000.0, lambda: None):
        leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                     phone=lambda ip, mac: False, mic=reader)
        leg("1.2.3.4", "")
        assert stuck.status()["office"]["corroborated_s_ago"] is None


def test_the_stuck_detector_takes_an_age_as_well_as_a_time():
    """``corroborate(ago=)`` is the turn ledger's shape (seconds since),
    and it must land on the detector's OWN clock, not the caller's."""
    from jarvis import stuckroom
    now = {"t": 5_000.0}
    sr = stuckroom.StuckRooms(path=pathlib.Path("/nonexistent/never-written.json"),
                              publish=lambda e: None, now=lambda: now["t"])
    sr.observe("office", True)
    now["t"] += 900.0
    sr.corroborate("mic", ago=120.0)
    assert sr.status()["office"]["corroborated_s_ago"] == pytest.approx(120.0)
    sr.corroborate("mic", ago=600.0)                 # older: ignored
    assert sr.status()["office"]["corroborated_s_ago"] == pytest.approx(120.0)
    sr.corroborate("mic", ago=-5.0)                  # the future is now
    assert sr.status()["office"]["corroborated_s_ago"] == pytest.approx(0.0)


# ====================================================================
# DEFECT 2, END TO END: he never left
# ====================================================================
def desk_run(tmp_path, *, warm_min, mic_s_ago, recency_min=15.0, limit=180):
    """He is in the office the whole time. Returns the minute the sentinel
    flipped to away, or None if it never did inside ``limit`` minutes."""
    clock = {"t": 2_000_000.0}
    now = lambda: clock["t"]
    sr = stuckroom.StuckRooms(path=tmp_path / ("d-%s.json" % uuid.uuid4().hex),
                              publish=lambda e: None, now=now)
    fab = roomfabric.from_readers(
        [("office", FakeSensor(True)), ("kitchen", FakeSensor(False))],
        now=now, enter_hold_s=0.0, poll_s=2.0)
    fab.tick()
    phone = {"up": warm_min > 0}
    leg = presence.ThreeLegProbe(fabric=fab, stuck=sr,
                                 phone=lambda i, m: phone["up"],
                                 mic=(None if mic_s_ago is None
                                      else (lambda: mic_s_ago)),
                                 recency_s=recency_min * 60.0)
    s = presence.PresenceSentinel(
        Cfg(**{"presence.phone_ip": "1.2.3.4", "presence.enabled": True}),
        publish=lambda e: None, probe_fn=leg, now=now, poll_s=60.0)
    for _ in range(int(warm_min)):
        clock["t"] += 60.0
        fab.tick()
        s.tick()
    phone["up"] = False
    for minute in range(1, limit + 1):
        clock["t"] += 60.0
        fab.tick()
        s.tick()
        if s.state == "away":
            return minute
    return None


def test_a_turn_in_the_last_ten_minutes_never_lets_him_be_called_out(tmp_path):
    """He is at his desk and he spoke five minutes ago. MEASURED before the
    fix: away at minute 13 and the kitchen fired a full greeting."""
    assert desk_run(tmp_path, warm_min=0.0, mic_s_ago=300.0) is None


def test_the_desk_with_a_napping_phone_and_no_mic_is_the_honest_residual(tmp_path):
    """No mic evidence at all, and this is the exposure that remains.

    MEASURED (scripts/presence_cliff.py, scenario B): 26 minutes -- the
    15-minute window, then the 12-minute away grace, less the one poll
    that was still fresh. The first cut called him away at minute 13 on a
    run nothing had stamped, and at 56 on a stamped one (the stuck fault).
    jarvis-v3 never does while a room reads on. So this is NOT closed for a
    man who sits silent for 26 minutes with his phone asleep and the
    camera dark; it is closed by any turn inside that time (the tests
    around this one), and by the camera going live (his rule 1). The
    number is pinned exactly so a change to either window shows up here.
    """
    flipped = desk_run(tmp_path, warm_min=30.0, mic_s_ago=None)
    assert flipped == 26, flipped


def test_he_is_never_greeted_at_the_kitchen_while_the_mic_still_hears_him(tmp_path):
    """The harm the false away actually does: the full door cue fired at a
    man walking to make a coffee."""
    clock = {"t": 2_000_000.0}
    now = lambda: clock["t"]
    sr = stuckroom.StuckRooms(path=tmp_path / "k.json",
                              publish=lambda e: None, now=now)
    fab = roomfabric.from_readers(
        [("office", FakeSensor(True)), ("kitchen", FakeSensor(False))],
        now=now, enter_hold_s=0.0, poll_s=2.0)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=sr,
                                 phone=lambda i, m: False, mic=lambda: 120.0)
    s = presence.PresenceSentinel(
        Cfg(**{"presence.phone_ip": "1.2.3.4", "presence.enabled": True}),
        publish=lambda e: None, probe_fn=leg, now=now, poll_s=60.0)
    for _ in range(60):
        clock["t"] += 60.0
        fab.tick()
        s.tick()
    watch = arrival.DoorWatch()
    assert watch.observe(room="kitchen", away=(s.home is False),
                         state=s.state) is not True


# ====================================================================
# DEFECT 3: the default is OFF
# ====================================================================
def his_config(**extra):
    values = {
        "presence.room_sensor_enabled": True,
        "presence.rooms": [{"name": "office", "url": "http://192.168.50.51"},
                           {"name": "kitchen", "url": "http://192.168.50.52"}],
        "presence.phone_ip": "192.168.50.34",
        "presence.door_room": "kitchen",
    }
    values.update(extra)
    return Cfg(**values)


def test_a_merge_and_a_restart_cannot_change_his_presence_behaviour():
    """``presence.three_legs`` is absent from his config today. With the
    voter defaulting ON, merging this branch and restarting would have
    changed what 'away' means on a live box with no decision from him."""
    s = presence.PresenceSentinel(his_config(), publish=lambda e: None)
    assert s.legs is None
    assert s.stuck is None


def test_turning_the_key_on_builds_the_voter():
    s = presence.PresenceSentinel(his_config(**{"presence.three_legs": True}),
                                  publish=lambda e: None)
    assert s.legs is not None


def test_the_key_is_documented_in_one_plain_sentence():
    import inspect
    src = inspect.getsource(presence.PresenceSentinel.__init__)
    assert "presence.three_legs" in src
    assert "off by default" in src.lower() or "defaults off" in src.lower()


def test_the_recency_key_is_one_edit_and_reaches_the_voter():
    s = presence.PresenceSentinel(
        his_config(**{"presence.three_legs": True,
                      "presence.corroboration_recency_min": 25}),
        publish=lambda e: None)
    assert s.legs.recency_s == 1500.0


def test_the_recency_key_falls_back_to_the_default_when_it_is_nonsense():
    s = presence.PresenceSentinel(
        his_config(**{"presence.three_legs": True,
                      "presence.corroboration_recency_min": "soon"}),
        publish=lambda e: None)
    assert s.legs.recency_s == pv.DEFAULT_RECENCY_S


def test_the_mic_window_reuses_the_key_arrival_already_has():
    """One number, one meaning: 'how long a turn counts as proof he is in
    the flat'. arrival.departure_ready reads the same key."""
    s = presence.PresenceSentinel(
        his_config(**{"presence.three_legs": True,
                      "presence.departure_mic_silence_min": 20}),
        publish=lambda e: None)
    assert s.legs.mic_window_s == 1200.0
