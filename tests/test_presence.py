"""Presence sentinel (jarvis/presence.py): the pure probe through the `run`
seam (no subprocess ever runs), the sentinel's away hysteresis and boot
grace on a fake clock, the Presence event, and start/stop with a joined
thread. Nothing here touches the network.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from jarvis import presence as presence_mod
from jarvis.events import Presence
from jarvis.presence import (DEFAULT_AWAY_MIN, PresenceSentinel, parse_neigh,
                             probe)

NEIGH = """192.168.50.1 dev wlP9s9 lladdr 3c:37:86:aa:bb:cc REACHABLE
192.168.50.35 dev wlP9s9 lladdr 90:a8:22:d7:d0:8a STALE
192.168.50.114 dev wlP9s9 lladdr 60:CF:84:AD:FD:91 DELAY
192.168.50.77 dev wlP9s9  FAILED
fe80::1 dev wlP9s9 lladdr 3c:37:86:aa:bb:cc router REACHABLE
"""


class Runner:
    """Fake subprocess.run: records argv, answers from a table."""

    def __init__(self, neigh=NEIGH, ping_rc=0, raise_on=None):
        self.calls = []
        self.neigh, self.ping_rc, self.raise_on = neigh, ping_rc, raise_on

    def __call__(self, argv, timeout=None, **kw):
        self.calls.append(list(argv))
        if self.raise_on and argv[0] == self.raise_on:
            raise subprocess.TimeoutExpired(argv, timeout or 0)
        if argv[0] == "ip":
            return SimpleNamespace(returncode=0, stdout=self.neigh)
        return SimpleNamespace(returncode=self.ping_rc, stdout="")

    @property
    def pinged(self):
        return [c[-1] for c in self.calls if c[0] == "ping"]


# ---------------------------------------------------------------- parse
def test_parse_neigh_reads_state_mac_and_skips_junk():
    rows = parse_neigh(NEIGH + "\ngarbage line\n")
    by_ip = {r["ip"]: r for r in rows}
    assert by_ip["192.168.50.35"]["state"] == "STALE"
    assert by_ip["192.168.50.114"]["mac"] == "60:cf:84:ad:fd:91"     # lower-cased
    assert by_ip["192.168.50.77"]["mac"] == "" and by_ip["192.168.50.77"]["state"] == "FAILED"
    assert by_ip["fe80::1"]["state"] == "REACHABLE"                   # flags before the state
    assert "garbage" not in by_ip


# ---------------------------------------------------------------- probe
def test_reachable_neighbour_is_present_without_a_ping():
    run = Runner()
    assert probe("192.168.50.1", run=run) is True
    assert run.pinged == []


def test_delay_counts_as_present():
    run = Runner()
    assert probe("192.168.50.114", run=run) is True and run.pinged == []


def test_stale_is_never_absence_it_triggers_the_ping():
    run = Runner(ping_rc=0)
    assert probe("192.168.50.35", run=run) is True
    assert run.pinged == ["192.168.50.35"]
    run = Runner(ping_rc=1)
    assert probe("192.168.50.35", run=run) is False
    assert run.calls[-1][:5] == ["ping", "-c", "1", "-W", "1"]


def test_missing_entry_pings_the_configured_ip():
    run = Runner(ping_rc=0)
    assert probe("192.168.50.200", run=run) is True
    assert run.pinged == ["192.168.50.200"]


def test_mac_only_config_matches_case_insensitively_and_pings_what_arp_knows():
    run = Runner(ping_rc=1)
    assert probe(mac="90:A8:22:D7:D0:8A", run=run) is False
    assert run.pinged == ["192.168.50.35"]
    run = Runner()
    assert probe(mac="60:cf:84:ad:fd:91", run=run) is True and run.pinged == []


def test_mac_only_with_no_arp_entry_has_nothing_to_ping():
    run = Runner()
    assert probe(mac="aa:aa:aa:aa:aa:aa", run=run) is False
    assert run.pinged == []


def test_nothing_configured_is_absent_without_running_anything():
    run = Runner()
    assert probe("", "", run=run) is False and run.calls == []


def test_subprocess_failures_read_as_absent_not_as_exceptions():
    assert probe("192.168.50.35", run=Runner(raise_on="ip", ping_rc=0)) is True   # ping still ran
    assert probe("192.168.50.35", run=Runner(raise_on="ping")) is False


def test_the_real_probe_never_runs_by_default_in_tests(monkeypatch):
    """Belt and braces: the module's default runner must be subprocess.run
    with a timeout, and nothing in this file reaches it."""
    calls = []
    monkeypatch.setattr(presence_mod.subprocess, "run",
                        lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=1, stdout=""))
    assert probe("10.0.0.1") is False
    assert calls and calls[0][1]["timeout"] == presence_mod.PING_TIMEOUT_S
    assert calls[0][1]["capture_output"] is True


# ------------------------------------------------------------- sentinel
class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


def _sentinel(cfg=None, present=None, clock=None, **kw):
    clock = clock or Clock()
    events = []
    answers = {"present": True if present is None else present}
    s = PresenceSentinel(cfg if cfg is not None else {"presence.phone_ip": "10.0.0.2"},
                         publish=events.append,
                         probe_fn=lambda ip, mac: answers["present"],
                         now=clock.now, **kw)
    s.clock, s.events, s.answers = clock, events, answers
    return s


class DictCfg(dict):
    def get(self, key, default=None):          # dotted keys stored flat
        return dict.get(self, key, default)


@pytest.fixture
def cfg():
    return DictCfg({"presence.phone_ip": "10.0.0.2", "presence.away_after_min": 12})


def test_unknown_until_the_first_probe_and_home_means_is_home(cfg):
    s = _sentinel(cfg)
    assert s.state == "unknown" and s.is_home()              # unknown = do not mute him
    ev = s.tick()
    assert isinstance(ev, Presence) and ev.home and not ev.returned
    assert s.state == "home" and s.is_home()
    assert s.tick() is None                                  # no transition, no event


def test_away_needs_the_grace_to_elapse_and_home_is_immediate(cfg):
    s = _sentinel(cfg)
    s.tick()
    s.answers["present"] = False
    for _ in range(11):
        s.clock.tick(60)
        assert s.tick() is None and s.is_home()              # a nap, not a departure
    s.clock.tick(60)                                          # 12 min unseen
    ev = s.tick()
    assert ev is not None and ev.home is False and s.state == "away" and not s.is_home()
    s.clock.tick(3600)
    assert s.tick() is None                                   # still away: one event only
    s.answers["present"] = True
    ev = s.tick()
    assert ev.home and ev.returned and s.is_home()
    assert [e.returned for e in s.events] == [False, False, True]


def test_a_phone_asleep_at_boot_earns_no_spurious_welcome(cfg):
    s = _sentinel(cfg, present=False)
    assert s.tick() is None and s.state == "unknown" and s.is_home()
    s.clock.tick(5 * 60)
    s.answers["present"] = True
    ev = s.tick()
    assert ev.home and not ev.returned                        # home, but he never "left"


def test_absent_from_boot_becomes_away_after_the_grace(cfg):
    s = _sentinel(cfg, present=False)
    s.tick()
    s.clock.tick(DEFAULT_AWAY_MIN * 60)
    ev = s.tick()
    assert ev is not None and ev.home is False


def test_grace_floor_and_config_fallbacks():
    s = _sentinel(DictCfg({"presence.phone_ip": "10.0.0.2", "presence.away_after_min": "nope"}))
    assert s.away_after_s == DEFAULT_AWAY_MIN * 60
    s = _sentinel(DictCfg({"presence.phone_ip": "10.0.0.2", "presence.away_after_min": 0}))
    assert s.away_after_s == 60
    s = _sentinel(DictCfg({"presence.phone_ip": "10.0.0.2", "presence.poll_s": 1}))
    assert s.poll_s == 5.0


def test_unconfigured_or_disabled_sentinel_never_probes_or_starts():
    probes = []
    s = PresenceSentinel(DictCfg({}), publish=lambda e: None,
                         probe_fn=lambda ip, mac: probes.append(1) or True)
    assert not s.configured and s.tick() is None and probes == []
    s.start()
    assert not s.running
    off = PresenceSentinel(DictCfg({"presence.phone_ip": "10.0.0.2", "presence.enabled": False}),
                           publish=lambda e: None, probe_fn=lambda ip, mac: True)
    assert not off.configured


def test_a_probe_that_raises_does_not_change_state(cfg):
    s = _sentinel(cfg)
    s.tick()

    def boom(ip, mac):
        raise RuntimeError("ip(8) missing")
    s._probe = boom
    s.clock.tick(3600)
    assert s.tick() is None and s.state == "home"


def test_publish_failure_is_logged_not_raised(cfg):
    s = PresenceSentinel(cfg, publish=lambda e: 1 / 0, probe_fn=lambda ip, mac: True)
    assert s.tick() is not None and s.state == "home"


def test_start_and_stop_join_the_thread(cfg):
    s = _sentinel(cfg, poll_s=0.01)
    s.start()
    assert s.running
    s.start()                                                 # idempotent
    s.stop()
    assert not s.running
    assert s.events and s.events[0].home                      # it ticked at least once
