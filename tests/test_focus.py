"""Focus / study sessions (jarvis/focus.py) on the real Timekeeper.

The timekeeper runs on a fake clock with tick() driven by hand (no thread),
the session's speech is a list, Spotify is a fake SpotifyTool that records
control()/play() calls or raises SpotifyError. Nothing here touches the
network, audio or the live app: PATHS are firewalled by conftest and every
store is under tmp_path.
"""
from __future__ import annotations

import json
import types
from datetime import datetime
from types import SimpleNamespace

import pytest

import jarvis.focus as focus_mod
from jarvis import commander as cm
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.events import ReminderFired, bus
from jarvis.focus import (ALREADY_LINE, BREAK_LINE, COMPLETE_LINE, END_LINE, END_NONE_LINE,
                          HALFWAY_LINE, LAPSED_LINE, NO_SESSION_LINE, RESUME_LINE,
                          FocusSession, left_words)
from jarvis.tools.spotify import NO_DEVICE_LINE, NOT_LINKED_LINE, SpotifyError
from jarvis.tools.timekeeper import (SILENT_PREFIX, Timekeeper, display_label,
                                     is_silent)

NOW = datetime(2026, 9, 3, 14, 0, 0)


class FakeClock:
    def __init__(self, start=NOW):
        self.t = start.timestamp()

    def now(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds
        return self.t


class Cfg:
    """The `get(dotted, default)` slice of AssistantConfig."""

    def __init__(self, **sections):
        self.data = {"focus": {"block_min": 25, "break_min": 5, "halfway": True,
                               "max_blocks": 4, "music": "pause", "playlist": ""}}
        for k, v in sections.items():
            self.data.setdefault(k, {}).update(v)

    def get(self, dotted, default=None):
        obj = self.data
        for part in dotted.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]
        return obj


class FakeSpotify:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail                  # SpotifyError to raise, or None

    def control(self, action, value=None, device=None):
        self.calls.append(("control", action))
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(ok=True, text=action)

    def play(self, query, kind="auto", device=None):
        self.calls.append(("play", query, kind))
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(ok=True, text=query)


@pytest.fixture
def world(tmp_path):
    """Timekeeper + services + a FocusSession on one clock; music inline."""
    clock = FakeClock()
    tk_said = []
    tk = Timekeeper(tmp_path / "tk.db", say=tk_said.append, cfg={}, now=clock.now,
                    run=lambda *a, **k: None, ring=False, notify=False,
                    cache_dir=tmp_path / "cache")
    said = []
    spotify = FakeSpotify()
    services = SimpleNamespace(timekeeper=tk, speak=said.append, assistant=Cfg(),
                               spotify=spotify)
    fs = FocusSession(services, state_path=tmp_path / "focus.json", now=clock.now,
                      bg=lambda fn: fn())
    w = types.SimpleNamespace(clock=clock, tk=tk, tk_said=tk_said, said=said,
                              spotify=spotify, services=services, fs=fs,
                              state=tmp_path / "focus.json", tmp=tmp_path)
    yield w
    fs.stop()
    tk.close()


def _pending(tk):
    return [i for i in tk.list("timer")]


# ------------------------------------------------------- timekeeper side
def test_silent_timer_fires_without_the_generic_line(world):
    tk, clock = world.tk, world.clock
    got = []
    bus.subscribe(ReminderFired, got.append)
    try:
        it = tk.add_silent_timer(60, "block 1")
        assert is_silent(it.label) and it.label == f"{SILENT_PREFIX} block 1"
        clock.advance(60)
        assert tk.tick() == 1
        assert world.tk_said == [], "the timekeeper must not speak a silent item"
        ev = [e for e in got if e.item_id == it.id]
        assert ev and ev[0].silent and ev[0].kind == "timer" and not ev[0].late
        assert tk.get(it.id).state == "done"
    finally:
        bus.unsubscribe(ReminderFired, got.append)


def test_a_normal_timer_still_speaks_and_carries_its_id(world):
    tk, clock = world.tk, world.clock
    got = []
    bus.subscribe(ReminderFired, got.append)
    try:
        it = tk.add_timer(60, "tea")
        clock.advance(60)
        tk.tick()
        assert world.tk_said == ["Sir, your 1-minute tea timer is up."]
        assert got[-1].item_id == it.id and not got[-1].silent
    finally:
        bus.unsubscribe(ReminderFired, got.append)


def test_schedule_readout_hides_the_colon(world):
    world.tk.add_silent_timer(600, "block 1")
    assert display_label("focus: block 1") == "focus block 1"
    assert "focus block 1 in 10 minutes" in world.tk.list_text("timer")


def test_missed_silent_items_are_not_read_out_at_boot(world):
    tk, clock = world.tk, world.clock
    tk.add_silent_timer(60, "block 1")
    tk.add_reminder(clock.now() + 60, "call mum")
    clock.advance(2 * 3600)
    tk.catch_up()
    assert len(world.tk_said) == 1 and "call mum" in world.tk_said[0]
    assert "block" not in world.tk_said[0]


# --------------------------------------------------------- the session
def test_full_cycle_block_halfway_break_resume_end(world):
    fs, tk, clock, said = world.fs, world.tk, world.clock, world.said
    line = fs.start("biosensors")
    assert line == "25 minutes on biosensors, sir; I'll call the halfway mark and the break."
    assert fs.active and fs.phase == "block"
    labels = sorted(i.label for i in _pending(tk))
    assert labels == ["focus: block 1", "focus: halfway 1"]
    assert world.spotify.calls == [("control", "pause")]
    assert json.loads(world.state.read_text())["phase"] == "block"

    clock.advance(12 * 60 + 30)
    tk.tick()
    assert said == [HALFWAY_LINE]
    assert fs.time_left() == "Thirteen minutes left in block 1, sir." or \
        fs.time_left() == "13 minutes left in block 1, sir."

    clock.advance(12 * 60 + 30)
    tk.tick()
    assert said[-1] == BREAK_LINE.format(n=1, m="five")
    assert fs.phase == "break" and fs.blocks_done == 1
    assert [i.label for i in _pending(tk)] == ["focus: break 1"]
    assert world.spotify.calls[-1] == ("control", "resume")
    assert fs.time_left() == "Five minutes of break left, sir."

    clock.advance(5 * 60)
    tk.tick()
    assert said[-1] == RESUME_LINE.format(n=2)
    assert fs.phase == "block" and world.spotify.calls[-1] == ("control", "pause")

    clock.advance(60)
    assert fs.end() == END_LINE.format(blocks="one block", n=25)
    assert not fs.active and _pending(tk) == []
    assert world.spotify.calls[-1] == ("control", "resume")
    assert world.tk_said == [], "the timekeeper never spoke during the session"
    assert fs.time_left() == NO_SESSION_LINE and fs.end() == NO_SESSION_LINE


def test_fifty_minute_session_and_no_halfway_under_ten(world):
    fs = world.fs
    assert fs.start("", 50, 10).startswith("50 minutes of focus, sir")
    assert {i.label for i in _pending(world.tk)} == {"focus: block 1", "focus: halfway 1"}
    fs.end()
    assert fs.start("", 5).startswith("5 minutes of focus, sir; I'll call the break")
    assert {i.label for i in _pending(world.tk)} == {"focus: block 1"}


def test_ending_in_the_first_block_counts_nothing(world):
    world.fs.start("signals")
    assert world.fs.end() == END_NONE_LINE


def test_starting_twice_reports_the_running_block(world):
    fs = world.fs
    fs.start("signals")
    world.clock.advance(60)
    line = fs.start("something else")
    assert line == ALREADY_LINE.format(left="24 minutes", n=1)
    assert fs.label == "signals"


def test_max_blocks_ends_the_session_itself(world):
    world.services.assistant.data["focus"]["max_blocks"] = 1
    fs, tk, clock = world.fs, world.tk, world.clock
    fs.start("signals", 10)
    clock.advance(10 * 60)
    tk.tick()
    assert world.said[-1] == COMPLETE_LINE.format(blocks="one block")
    assert not fs.active and _pending(tk) == []
    assert world.spotify.calls == [("control", "pause"), ("control", "resume")]


# --------------------------------------------------------------- music
def test_no_device_is_a_session_without_music(world):
    world.spotify.fail = SpotifyError(NO_DEVICE_LINE.format(default="X"), "device")
    fs, tk, clock = world.fs, world.tk, world.clock
    fs.start("signals", 10)
    assert fs.active and world.said == []          # nothing apologised
    clock.advance(10 * 60)
    tk.tick()
    assert fs.phase == "break"
    # resume is not attempted: nothing was paused
    assert world.spotify.calls == [("control", "pause")]


def test_unlinked_account_switches_music_off_for_the_session(world):
    world.spotify.fail = SpotifyError(NOT_LINKED_LINE, "auth")
    fs, tk, clock = world.fs, world.tk, world.clock
    fs.start("signals", 10)
    assert fs.state["music_off"] is True
    clock.advance(10 * 60)
    tk.tick()
    clock.advance(5 * 60)
    tk.tick()
    assert world.spotify.calls == [("control", "pause")], "no retry every block"


def test_playlist_mode_plays_for_the_block_and_pauses_for_the_break(world):
    world.services.assistant.data["focus"].update({"music": "playlist",
                                                   "playlist": "Deep Focus"})
    fs, tk, clock = world.fs, world.tk, world.clock
    fs.start("signals", 10)
    assert world.spotify.calls == [("play", "Deep Focus", "playlist")]
    clock.advance(10 * 60)
    tk.tick()
    assert world.spotify.calls[-1] == ("control", "pause")
    fs.end()
    assert world.spotify.calls[-1] == ("control", "pause") and len(world.spotify.calls) == 2


def test_music_off_never_touches_spotify(world):
    world.services.assistant.data["focus"]["music"] = "off"
    world.fs.start("signals", 10)
    world.fs.end()
    assert world.spotify.calls == []


def test_without_a_spotify_handle_nothing_breaks(world):
    world.services.spotify = None
    assert world.fs.start("signals", 10).startswith("10 minutes on signals")


# ------------------------------------------------------------- restart
def _reopen(world):
    """A second app: same clock, same db, same state file."""
    world.fs.stop()
    world.tk.close()
    tk = Timekeeper(world.tmp / "tk.db", say=world.tk_said.append, cfg={},
                    now=world.clock.now, run=lambda *a, **k: None, ring=False,
                    notify=False, cache_dir=world.tmp / "cache")
    services = SimpleNamespace(timekeeper=tk, speak=world.said.append,
                               assistant=Cfg(), spotify=world.spotify)
    fs = FocusSession(services, state_path=world.state, now=world.clock.now,
                      bg=lambda fn: fn())
    world.tk, world.fs, world.services = tk, fs, services
    return fs, tk


def test_session_survives_a_restart(world):
    world.fs.start("biosensors")
    world.clock.advance(5 * 60)
    fs, tk = _reopen(world)
    assert fs.active and fs.label == "biosensors"
    assert fs.reconcile() is None and fs.time_left() == "20 minutes left in block 1, sir."
    world.clock.advance(20 * 60)
    tk.tick()
    assert world.said[-1].startswith("Time for a break, sir")


def test_block_that_came_due_while_down_moves_to_the_break(world):
    world.fs.start("biosensors")
    world.clock.advance(30 * 60)                    # block due 5 min ago, halfway too
    fs, tk = _reopen(world)
    tk.start(catch_up=True)                         # the app's boot order
    tk.stop()
    fs.reconcile()
    assert world.said == [BREAK_LINE.format(n=1, m="five")], world.said
    assert fs.phase == "break" and fs.blocks_done == 1
    # the halfway item fired late in the same catch-up: not spoken, and gone
    assert [i.label for i in _pending(tk)] == ["focus: break 1"]
    assert world.tk_said == []


def test_session_missed_by_hours_lapses_with_the_count(world):
    world.fs.start("biosensors")
    world.clock.advance(3 * 3600)
    fs, tk = _reopen(world)
    tk.catch_up()
    assert fs.reconcile() == LAPSED_LINE.format(blocks="no blocks")
    assert world.said == [LAPSED_LINE.format(blocks="no blocks")]
    assert not fs.active and not json.loads(world.state.read_text()).get("phase")


def test_a_stale_state_file_without_a_phase_is_ignored(tmp_path):
    (tmp_path / "f.json").write_text(json.dumps({"phase": "", "blocks_done": 3}))
    fs = FocusSession(SimpleNamespace(), state_path=tmp_path / "f.json")
    try:
        assert not fs.active and fs.time_left() == NO_SESSION_LINE
    finally:
        fs.stop()


def test_other_owners_silent_items_are_ignored(world):
    world.fs.start("signals", 10)
    before = dict(world.fs.state)
    bus.publish(ReminderFired(text="x", item_id="not-ours", silent=True))
    assert world.fs.state == before and world.said == []


@pytest.mark.parametrize("seconds,words", [
    (10, "under a minute"), (50, "a minute"), (61, "two minutes"),
    (12 * 60, "twelve minutes"), (13 * 60, "13 minutes"), (0, "under a minute")])
def test_left_words(seconds, words):
    assert left_words(seconds) == words


# ----------------------------------------------------------- commander
@pytest.fixture
def cmdr(world, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = world.services
    svc.focus = world.fs
    svc.desktop = SimpleNamespace(parse_action=lambda part: None)
    svc.tts = None
    return Commander(svc)


@pytest.mark.parametrize("text,label,n,b", [
    ("study session biosensors", "biosensors", None, None),
    ("jarvis, study session for biosensors", "biosensors", None, None),
    ("start a fifty-minute focus session", "", 50, None),
    ("start a 25 minute study session for signals with a ten minute break",
     "signals", 25, 10),
    ("pomodoro", "", None, None),
    ("focus session on the thesis", "the thesis", None, None),
])
def test_start_phrasings(cmdr, world, text, label, n, b):
    res = cmdr.handle(text, "voice")
    assert res.handled and res.speak and world.fs.active, text
    assert world.fs.label == label
    assert world.fs.state["block_min"] == (n or 25)
    assert world.fs.state["break_min"] == (b or 5)
    world.fs.end()


def test_a_bare_session_is_not_a_focus_session(cmdr, world):
    assert cm._m_focus_start("start a session") is None
    assert cm._m_focus_start("resume the session") is None
    assert cm._m_focus_start("study session") is not None


def test_course_keeps_its_casing(cmdr, world):
    cmdr.handle("Jarvis, study session for Biosensors", "voice")
    assert world.fs.label == "Biosensors"


def test_how_long_left_and_end_the_session(cmdr, world):
    cmdr.handle("study session biosensors", "typed")
    world.clock.advance(3 * 60)
    res = cmdr.handle("how long left", "voice")
    assert res.reply == "22 minutes left in block 1, sir." and res.speak
    res = cmdr.handle("jarvis, how much time is left in the block", "voice")
    assert res.reply.startswith("22 minutes left")
    res = cmdr.handle("end the session", "voice")
    assert res.reply == END_NONE_LINE and res.status == "Focus: ended"
    assert not world.fs.active


def test_with_no_session_how_long_left_reads_the_timers(cmdr, world):
    res = cmdr.handle("how long left", "typed")
    assert res.reply == "No timers running, sir."
    world.tk.add_timer(600, "tea")
    res = cmdr.handle("how long left", "typed")
    assert "tea in 10 minutes" in res.reply


def test_end_the_session_without_one_is_not_ours(cmdr, world):
    """It may be a Claude session: the registry must fall through."""
    assert cm._FOCUS_END_RX.match("stop the session")
    assert cmdr._try_registry("end the session") is None


def test_the_focus_commands_are_tier_one(cmdr):
    names = {c.name for c in cm.ASSISTANT_TIER1}
    assert {"focus start", "focus left", "focus end"} <= names


def test_persona_lines_are_fixed_strings():
    for line in focus_mod.PERSONA_LINES:
        assert "{" not in line and line.endswith(".")
