"""The phone leg learns to say "I could not ask".

``probe()`` returns a plain bool and has no way to say unknown. Its own
comment says a missing or timed-out ip(8) is "unknown" -- and then the
function falls through to the ping and returns False on any exception.
False means AWAY, so the Spark's own Wi-Fi dropping, the router rebooting,
ping missing from PATH or the subnet changing all read as "he left the
flat". ``probe_state`` is the honest version.
"""
from __future__ import annotations

import subprocess

from jarvis import presence


class _Res:
    def __init__(self, rc=0, out=""):
        self.returncode, self.stdout = rc, out


NEIGH_STALE = "192.168.50.34 dev wlan0 lladdr aa:bb:cc:dd:ee:ff STALE\n"
NEIGH_REACHABLE = "192.168.50.34 dev wlan0 lladdr aa:bb:cc:dd:ee:ff REACHABLE\n"
ROUTE = "default via 192.168.50.1 dev wlan0 proto dhcp metric 600\n"


def _runner(neigh="", route=ROUTE, pings=None, raise_on=()):
    """A fake ``run``. ``pings`` maps ip -> returncode."""
    pings = dict(pings or {})

    def run(argv, timeout=None):
        head = argv[0]
        if head in raise_on:
            raise OSError("no such binary: %s" % head)
        if head == "ip" and "neigh" in argv:
            if "neigh" in raise_on:
                raise subprocess.TimeoutExpired(argv, timeout or 1)
            return _Res(0, neigh)
        if head == "ip" and "route" in argv:
            return _Res(0, route)
        if head == "ping":
            ip = argv[-1]
            if ip in raise_on:
                raise subprocess.TimeoutExpired(argv, timeout or 1)
            rc = pings.get(ip, 1)
            if isinstance(rc, list):      # a SEQUENCE: first packet, then the retry
                rc = rc.pop(0) if rc else 1
            return _Res(rc)
        return _Res(1)
    return run


# ------------------------------------------------------------- present
def test_a_reachable_arp_row_is_present_without_a_ping():
    run = _runner(neigh=NEIGH_REACHABLE)
    assert presence.probe_state(ip="192.168.50.34", run=run) is True


def test_a_stale_row_falls_through_to_the_ping_and_that_is_the_nap_case():
    """Measured on his box: only some neighbours sit REACHABLE and his
    phone's row is STALE between conversations. STALE is never absence."""
    run = _runner(neigh=NEIGH_STALE, pings={"192.168.50.34": 0})
    assert presence.probe_state(ip="192.168.50.34", run=run) is True


# --------------------------------------------------------------- absent
# HIS MEASURED FAILURE, 2026-09-11. One `ping -c1 -W1` with no reply was
# read as "he left the flat". An iPhone on Wi-Fi power-save routinely
# drops a single unsolicited echo -- this module's own docstring says so
# -- and the log shows eleven consecutive single-packet misses while he
# sat in the office, followed by a reply on the very next poll. Seven
# false departures in two days came out of that one line.
#
# "Asked, and no answer" is still False. The fix is what counts as ASKED.
def test_one_lost_packet_is_not_an_answer_and_the_probe_asks_again():
    run = _runner(neigh=NEIGH_STALE, pings={"192.168.50.34": [1, 0],
                                            "192.168.50.1": 0})
    assert presence.probe_state(ip="192.168.50.34", run=run) is True


def test_a_phone_that_misses_the_retry_TOO_is_a_real_no():
    run = _runner(neigh=NEIGH_STALE, pings={"192.168.50.34": [1, 1],
                                            "192.168.50.1": 0})
    assert presence.probe_state(ip="192.168.50.34", run=run) is False


def test_the_retry_really_does_ask_HARDER_than_the_first_packet():
    """Anti-drift. Nothing stops a later edit making the retry identical
    to the probe it exists to second-guess, and the suite would stay
    green while the bug came back."""
    sent = []

    def run(argv, timeout=None):
        if argv[0] == "ping":
            sent.append(argv)
            return _Res(1)
        return _Res(0, NEIGH_STALE if "neigh" in argv else ROUTE)

    presence.probe_state(ip="192.168.50.34", run=run)
    first, retry = sent[0], sent[1]

    def count(argv, flag):
        return float(argv[argv.index(flag) + 1])

    assert count(retry, "-c") > count(first, "-c"), "the retry sends no more packets"
    assert count(retry, "-W") > count(first, "-W"), "the retry waits no longer"


def test_the_retry_costs_nothing_when_the_first_packet_lands():
    pings = []

    def run(argv, timeout=None):
        if argv[0] == "ping":
            pings.append(argv)
            return _Res(0)
        return _Res(0, NEIGH_STALE if "neigh" in argv else ROUTE)

    assert presence.probe_state(ip="192.168.50.34", run=run) is True
    assert len(pings) == 1, "a phone that answered was pinged twice"


def test_a_retry_that_could_not_RUN_is_unknown_and_never_away():
    """Same rule as the first packet: could-not-ask is not an answer."""
    calls = []

    def run(argv, timeout=None):
        if argv[0] == "ping":
            calls.append(argv)
            if len(calls) == 1:
                return _Res(1)
            raise subprocess.TimeoutExpired(argv, timeout or 1)
        return _Res(0, NEIGH_STALE if "neigh" in argv else ROUTE)

    assert presence.probe_state(ip="192.168.50.34", run=run) is None


# -------------------------------------------------------------- unknown
def test_ip_neigh_failing_is_unknown_and_never_away():
    run = _runner(neigh=NEIGH_STALE, raise_on=("neigh",))
    assert presence.probe_state(ip="192.168.50.34", run=run) is None


def test_a_missing_ping_binary_is_unknown_and_never_away():
    run = _runner(neigh=NEIGH_STALE, raise_on=("ping",))
    assert presence.probe_state(ip="192.168.50.34", run=run) is None


def test_the_gateway_canary_turns_a_dead_network_into_unknown():
    """The Spark's own Wi-Fi dropped. That is not a departure."""
    run = _runner(neigh=NEIGH_STALE, pings={"192.168.50.34": [1, 1],
                                            "192.168.50.1": 1})
    assert presence.probe_state(ip="192.168.50.34", run=run) is None


def test_the_canary_only_costs_a_packet_on_the_negative_path():
    seen = []

    def run(argv, timeout=None):
        seen.append(argv[-1] if argv[0] == "ping" else argv[0])
        if argv[0] == "ip" and "neigh" in argv:
            return _Res(0, NEIGH_REACHABLE)
        return _Res(0, ROUTE)
    assert presence.probe_state(ip="192.168.50.34", run=run) is True
    assert "192.168.50.1" not in seen


def test_no_address_configured_is_unknown_rather_than_absent():
    """The leg is not configured. That says nothing about where he is."""
    assert presence.probe_state(run=_runner()) is None


def test_a_mac_only_config_pings_whatever_arp_says_that_mac_is_on():
    run = _runner(neigh=NEIGH_STALE, pings={"192.168.50.34": 0})
    assert presence.probe_state(mac="AA:BB:CC:DD:EE:FF", run=run) is True


def test_a_mac_that_arp_has_never_seen_is_unknown_not_away():
    run = _runner(neigh="", pings={})
    assert presence.probe_state(mac="aa:bb:cc:dd:ee:ff", run=run) is None


# ------------------------------------------------- the old shape stays
def test_the_bool_probe_is_unchanged_for_every_existing_caller():
    run = _runner(neigh=NEIGH_STALE, pings={"192.168.50.34": 0})
    assert presence.probe(ip="192.168.50.34", run=run) is True
    run = _runner(neigh=NEIGH_STALE, pings={"192.168.50.34": 1})
    assert presence.probe(ip="192.168.50.34", run=run) is False


def test_the_bool_probe_still_flattens_unknown_to_false():
    """Documented, not fixed: every existing caller expects a bool, and
    ``probe_state`` is the seam the new path uses instead."""
    run = _runner(neigh=NEIGH_STALE, raise_on=("ping",))
    assert presence.probe(ip="192.168.50.34", run=run) is False
    assert presence.probe_state(ip="192.168.50.34", run=run) is None


def test_room_or_phone_passes_an_honest_unknown_through(monkeypatch):
    """``bool(self.phone(...))`` flattened a None to False on the way out,
    which would undo the fix at the last line."""
    class Sensor:
        def read(self):
            return None
    leg = presence.RoomOrPhone(Sensor(), phone=lambda ip, mac: None)
    assert leg("192.168.50.34", "") is None


def test_room_or_phone_still_lets_a_room_beat_a_silent_phone():
    class Sensor:
        def read(self):
            return True
    leg = presence.RoomOrPhone(Sensor(), phone=lambda ip, mac: False)
    assert leg("1.2.3.4", "") is True
