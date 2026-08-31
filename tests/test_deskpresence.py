"""Desk presence (jarvis/deskpresence.py): the pure idle probe through the
`run` seam (no subprocess ever runs), the sentinel's threshold on a fake
clock, the DeskState event, start/stop with a joined thread, and the two
rules everything else leans on -- no signal is NEVER away, and away is
never spoken, only suppressed.

Also covers the board standby helper (jarvis/ui/views.standby_alpha), which
is a pure function precisely so it can be tested without a Tk root.
"""
from __future__ import annotations

import subprocess
import threading
from types import SimpleNamespace

import pytest

from jarvis import deskpresence as desk_mod
from jarvis.deskpresence import (DEFAULT_AWAY_MIN, DeskSentinel, desk_idle_s,
                                 parse_idletime)
from jarvis.events import DeskState
from jarvis.ui.views import STANDBY_ALPHA_FLOOR, standby_alpha

LIVE_STDOUT = "(uint64 13478107,)\n"


@pytest.fixture(autouse=True)
def _desk_enabled(monkeypatch):
    """conftest switches desk presence off for the whole suite (the session
    bus is the developer's real desktop). These cases drive the sentinel
    through a fake idle function, so they turn the switch back on."""
    monkeypatch.delenv(desk_mod.ENV_OFF, raising=False)


class Runner:
    """Fake subprocess.run: records argv, answers from a table."""

    def __init__(self, rc=0, stdout=LIVE_STDOUT, raises=None):
        self.calls = []
        self.rc, self.stdout, self.raises = rc, stdout, raises

    def __call__(self, argv, timeout=None, **kw):
        self.calls.append((list(argv), timeout))
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(returncode=self.rc, stdout=self.stdout)


# ------------------------------------------------------------------ parse
def test_parse_reads_the_gdbus_tuple_as_seconds():
    assert parse_idletime(LIVE_STDOUT) == pytest.approx(13478.107)
    assert parse_idletime("(uint64 0,)") == 0.0


def test_parse_of_anything_else_is_none_never_zero():
    """0.0 would read as "sitting right there"; unknown must stay unknown."""
    for junk in ("", None, "garbage", "(int32 5,)", "uint64 ,"):
        assert parse_idletime(junk) is None


# ------------------------------------------------------------------ probe
def test_a_valid_tuple_gives_seconds_and_calls_the_idle_monitor():
    run = Runner()
    assert desk_idle_s(run=run) == pytest.approx(13478.107)
    argv, timeout = run.calls[0]
    assert argv[0] == "gdbus" and "org.gnome.Mutter.IdleMonitor" in argv
    assert argv[-1].endswith("GetIdletime") and timeout == desk_mod.CALL_TIMEOUT_S


def test_a_nonzero_return_code_is_unknown():
    assert desk_idle_s(run=Runner(rc=1, stdout=LIVE_STDOUT)) is None


def test_garbage_stdout_is_unknown():
    assert desk_idle_s(run=Runner(stdout="Error: no such interface")) is None


def test_a_missing_gdbus_or_a_timeout_is_unknown_not_an_exception():
    assert desk_idle_s(run=Runner(raises=FileNotFoundError("gdbus"))) is None
    assert desk_idle_s(
        run=Runner(raises=subprocess.TimeoutExpired("gdbus", 4))) is None


def test_the_real_runner_never_runs_by_default_in_tests(monkeypatch):
    """Belt and braces, as in test_presence.py: the module default must be
    subprocess.run with a timeout, and nothing here reaches it."""
    calls = []
    monkeypatch.setattr(
        desk_mod.subprocess, "run",
        lambda argv, **kw: calls.append((argv, kw)) or
        SimpleNamespace(returncode=1, stdout=""))
    assert desk_idle_s() is None
    assert calls and calls[0][1]["timeout"] == desk_mod.CALL_TIMEOUT_S
    assert calls[0][1]["capture_output"] is True


# --------------------------------------------------------------- sentinel
class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


class DictCfg(dict):
    def get(self, key, default=None):          # dotted keys stored flat
        return dict.get(self, key, default)


def _sentinel(cfg=None, idle=0.0, **kw):
    clock = Clock()
    events, answers = [], {"idle": idle}
    s = DeskSentinel(cfg if cfg is not None else DictCfg(),
                     publish=events.append,
                     idle_fn=lambda: answers["idle"],
                     now=clock.now, **kw)
    s.clock, s.events, s.answers = clock, events, answers
    return s


def test_unknown_until_a_reading_lands_and_unknown_is_at_the_desk():
    s = _sentinel()
    assert s.state == "unknown" and s.is_at_desk() and s.idle_s() is None
    ev = s.tick()
    assert isinstance(ev, DeskState) and ev.at_desk and not ev.returned
    assert s.state == "at-desk" and s.idle_s() == 0.0
    assert s.tick() is None                    # no crossing, no event


def test_the_chair_empties_only_past_the_threshold_and_the_return_is_marked():
    s = _sentinel()
    s.tick()
    s.answers["idle"] = DEFAULT_AWAY_MIN * 60 - 1
    assert s.tick() is None and s.is_at_desk()          # still reading, not gone
    s.answers["idle"] = DEFAULT_AWAY_MIN * 60
    ev = s.tick()
    assert ev is not None and ev.at_desk is False and not s.is_at_desk()
    assert ev.idle_s == DEFAULT_AWAY_MIN * 60
    assert s.tick() is None                             # one event per crossing
    s.answers["idle"] = 2.0
    ev = s.tick()
    assert ev.at_desk and ev.returned and s.is_at_desk()
    assert [e.returned for e in s.events] == [False, False, True]


def test_no_signal_never_moves_the_state_and_never_reads_as_away():
    """The Mutter interface is not guaranteed; a failure must degrade to
    silence, not to a wrong answer."""
    s = _sentinel(idle=None)
    assert s.tick() is None and s.state == "unknown" and s.is_at_desk()
    s.answers["idle"] = DEFAULT_AWAY_MIN * 60
    s.tick()
    assert s.state == "away"
    s.answers["idle"] = None
    assert s.tick() is None and s.state == "away"       # unchanged, not "back"
    assert s.events == [e for e in s.events if e.at_desk is False]


def test_a_probe_that_raises_is_survived():
    def boom():
        raise RuntimeError("session bus gone")
    s = DeskSentinel(DictCfg(), publish=lambda e: None, idle_fn=boom)
    assert s.tick() is None and s.state == "unknown"


def test_a_publish_failure_does_not_lose_the_state_change():
    def boom(_ev):
        raise RuntimeError("bus down")
    s = DeskSentinel(DictCfg(), publish=boom, idle_fn=lambda: 0.0)
    assert s.tick() is not None and s.state == "at-desk"


def test_config_fallbacks_and_the_threshold_floor():
    s = _sentinel(DictCfg({"presence.desk_away_after_min": "nope"}))
    assert s.away_after_s == DEFAULT_AWAY_MIN * 60
    s = _sentinel(DictCfg({"presence.desk_away_after_min": 0}))
    assert s.away_after_s == 60.0                       # a floor, never 0
    s = _sentinel(DictCfg({"presence.desk_poll_s": 1}))
    assert s.poll_s == 5.0                              # a floor, never a spin
    s = _sentinel(DictCfg({"presence.desk_poll_s": "nope"}))
    assert s.poll_s == desk_mod.DEFAULT_POLL_S


def test_a_broken_config_object_does_not_stop_the_probe():
    class Boom:
        def get(self, key, default=None):
            raise RuntimeError("config on fire")
    s = _sentinel(Boom())
    assert s.enabled is True and s.away_after_s == DEFAULT_AWAY_MIN * 60
    assert s.tick() is not None


def test_disabled_by_config_ticks_nothing_and_starts_no_thread():
    s = _sentinel(DictCfg({"presence.desk": False}))
    assert s.tick() is None and s.events == []
    s.start()
    assert not s.running


def test_the_env_switch_wins_over_the_config(monkeypatch):
    """The suite's firewall (tests/conftest.py): a test that builds the real
    App must never read the developer's live session bus."""
    s = _sentinel(DictCfg({"presence.desk": True}))
    for off in ("0", "off", "FALSE", "no"):
        monkeypatch.setenv(desk_mod.ENV_OFF, off)
        assert s.enabled is False and s.tick() is None
    monkeypatch.setenv(desk_mod.ENV_OFF, "1")
    assert s.enabled is True and s.tick() is not None


# ----------------------------------------------------------------- thread
def test_start_stop_joins_the_thread():
    seen = threading.Event()
    s = DeskSentinel(DictCfg(), publish=lambda e: seen.set(),
                     idle_fn=lambda: 0.0, poll_s=0.01)
    s.start()
    assert seen.wait(2.0) and s.running
    s.start()                                           # alive-guard: no second
    s.stop()
    assert not s.running
    s.start()                                           # a stopped one restarts
    assert s.running
    s.stop()


# ---------------------------------------------------------- board standby
def test_standby_dims_only_an_established_empty_chair():
    assert standby_alpha(True) == 1.0
    assert standby_alpha(None) == 1.0                   # unknown looks like today
    assert standby_alpha(False) < 1.0


def test_standby_can_be_switched_off_and_is_never_invisible():
    assert standby_alpha(False, enabled=False) == 1.0
    assert standby_alpha(False, dim=0.0) == STANDBY_ALPHA_FLOOR
    assert standby_alpha(False, dim=5) == 1.0
    assert standby_alpha(False, dim="nonsense") == standby_alpha(False)
