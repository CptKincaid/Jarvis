"""Bedtime wind-down (jarvis/winddown.py) and its two hooks in the commander.

The one subprocess seam (`winddown._run`) is a recorder, so xrandr and
gsettings are never actually run; Spotify is a fake that can fail the way
SpotifyError does. The property under test throughout is REVERSIBILITY: the
restore record is written before anything changes, any failure puts the
screen back, and a second "good night" can never record the dimmed screen as
the thing to restore.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import jarvis.winddown as wd_mod
from jarvis.commander import Commander, CommandResult
from tests.test_app_wiring import build, paths, seams  # noqa: F401 - fixtures

XRANDR_OUT = """Screen 0: minimum 8 x 8, current 3840 x 2160, maximum 32767 x 32767
HDMI-0 connected primary 3840x2160+0+0 (0x18d) normal (normal left inverted right) 596mm x 335mm
\tIdentifier: 0x18b
\tTimestamp:  9464928
\tBrightness: 1.0
\tCTM: 0
USB-C-0 disconnected (normal left inverted right x axis y axis)
\tIdentifier: 0x18c
DP-1 connected 1920x1080+0+0 (0x19a) normal
\tBrightness: 0.80
"""


class FakeSpotify:
    def __init__(self, volume=63, fail=False):
        self.calls: list[tuple] = []
        self.volume = volume
        self.fail = fail

    def resolve_device(self, named=None, prefer_active=False):
        if self.fail:
            raise RuntimeError("no device")
        return SimpleNamespace(id="d1", name="HPCOMPUTER", volume=self.volume,
                               is_active=True)

    def control(self, action="", value=None, device=None):
        if self.fail:
            raise RuntimeError("no device")
        self.calls.append((action, value))
        return SimpleNamespace(text="ok")


class FakeQuiet:
    def __init__(self, end=None):
        self.end = end
        self.dnd: list[float] = []

    def hours_end(self, now=None):
        return self.end

    def set_dnd(self, seconds, now=None):
        self.dnd.append(seconds)
        return (now or time.time()) + seconds


@pytest.fixture
def runs(monkeypatch):
    """Record every xrandr / gsettings call and answer them plausibly."""
    seen: list[list] = []
    state = {"night_light": "false\n"}

    def fake_run(argv, timeout=5.0):
        seen.append(list(argv))
        if argv[0] == "xrandr" and "--verbose" in argv:
            return True, XRANDR_OUT
        if argv[0] == "gsettings" and argv[1] == "get":
            return True, state["night_light"]
        if argv[0] == "gsettings" and argv[1] == "set":
            state["night_light"] = argv[-1] + "\n"
            return True, ""
        return True, ""

    monkeypatch.setattr(wd_mod, "_run", fake_run)
    return seen


def make(tmp_path, runs=None, spotify=None, quiet=None, **cfg):
    """A WindDown on a tmp state file with the config it is given."""
    settings = {"wind_down.enabled": True, "wind_down.fade_s": 0.05,
                "wind_down.brightness": 0.5}
    settings.update({f"wind_down.{k}": v for k, v in cfg.items()})
    services = SimpleNamespace(
        assistant=SimpleNamespace(get=lambda k, d=None: settings.get(k, d)),
        spotify=spotify, quiet=quiet)
    return wd_mod.WindDown(services, state_path=tmp_path / "winddown.json")


def wait_idle(wd, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        t = wd._thread
        if t is None or not t.is_alive():
            return True
        time.sleep(0.02)
    return False


def argv_of(seen, *head):
    return [a for a in seen if a[:len(head)] == list(head)]


# ------------------------------------------------------------- reading
def test_brightness_is_read_only_for_connected_outputs(runs):
    assert wd_mod.screen_brightness() == {"HDMI-0": 1.0, "DP-1": 0.8}


def test_night_light_reads_the_gnome_key(runs):
    assert wd_mod.night_light() is False
    assert argv_of(runs, "gsettings", "get")[0][2:] == [
        wd_mod.NIGHT_LIGHT_SCHEMA, wd_mod.NIGHT_LIGHT_KEY]


def test_an_unreadable_screen_is_no_screen_at_all(monkeypatch):
    monkeypatch.setattr(wd_mod, "_run", lambda argv, timeout=5.0: (False, ""))
    assert wd_mod.screen_brightness() == {}
    assert wd_mod.night_light() is None


# --------------------------------------------------------------- start
def test_off_by_default(tmp_path, runs):
    wd = make(tmp_path, enabled=False)
    assert wd.start() is False
    assert runs == [] and not (tmp_path / "winddown.json").exists()


def test_good_night_dims_warms_fades_and_arms(tmp_path, runs):
    spotify, quiet = FakeSpotify(volume=70), FakeQuiet(end=time.time() + 8 * 3600)
    wd = make(tmp_path, spotify=spotify, quiet=quiet)
    assert wd.start() is True
    assert wait_idle(wd)

    # the screen: every connected output, at the configured level
    dims = argv_of(runs, "xrandr", "--output")
    assert {a[2] for a in dims} == {"HDMI-0", "DP-1"}
    assert {a[4] for a in dims} == {"0.50"}
    assert argv_of(runs, "gsettings", "set")[-1][-1] == "true"

    # the music: stepped down to zero, paused, and the volume put back so a
    # manual play in the night is not silent
    actions = [c for c in spotify.calls]
    assert actions[0][0] == "volume" and actions[0][1] < 70
    assert ("volume", 0) in actions
    assert actions[-2] == ("pause", None) and actions[-1] == ("volume", 70)

    # DND for the length of the quiet window he is joining
    assert len(quiet.dnd) == 1 and 7.5 * 3600 < quiet.dnd[0] < 8.1 * 3600


def test_the_restore_record_is_written_before_anything_changes(tmp_path, runs,
                                                               monkeypatch):
    """A crash mid-fade must still leave a file that says how to undo it."""
    order = []
    real = wd_mod._run

    def watching(argv, timeout=5.0):
        if argv[:2] == ["xrandr", "--output"]:
            order.append(("dim", (tmp_path / "winddown.json").exists()))
        return real(argv, timeout)

    monkeypatch.setattr(wd_mod, "_run", watching)
    wd = make(tmp_path, spotify=FakeSpotify())
    wd.start()
    assert wait_idle(wd)
    assert order and all(existed for _, existed in order)
    state = json.loads((tmp_path / "winddown.json").read_text())
    assert state["brightness"] == {"HDMI-0": 1.0, "DP-1": 0.8}
    assert state["night_light"] is False and state["volume"] == 63


def test_a_second_good_night_never_snapshots_the_dimmed_screen(tmp_path, runs):
    wd = make(tmp_path, spotify=FakeSpotify())
    assert wd.start() and wait_idle(wd)
    before = json.loads((tmp_path / "winddown.json").read_text())
    assert wd.start() is False
    assert json.loads((tmp_path / "winddown.json").read_text()) == before
    assert before["brightness"]["HDMI-0"] == 1.0


def test_a_failure_to_dim_puts_the_screen_straight_back(tmp_path, monkeypatch):
    calls = []

    def flaky(argv, timeout=5.0):
        calls.append(list(argv))
        if argv[0] == "xrandr" and "--verbose" in argv:
            return True, XRANDR_OUT
        if argv[0] == "gsettings" and argv[1] == "get":
            return True, "false\n"
        if argv[:2] == ["xrandr", "--output"] and argv[2] == "DP-1":
            return False, ""          # the second monitor refuses
        return True, ""

    monkeypatch.setattr(wd_mod, "_run", flaky)
    spotify = FakeSpotify()
    wd = make(tmp_path, spotify=spotify)
    assert wd.start() and wait_idle(wd)
    # ...and the half-dimmed screen went back to what it was
    dims = [a for a in calls if a[:2] == ["xrandr", "--output"]]
    assert ["xrandr", "--output", "HDMI-0", "--brightness", "1.00"] in dims
    # DP-1 refuses the put-back as well, so the record is KEPT -- pruned to
    # the one output that did not come back. Deleting it here (the old
    # unconditional _clear_state) is what made a failed restore
    # unrecoverable and still reported it as a success.
    kept = json.loads((tmp_path / "winddown.json").read_text())
    assert kept["brightness"] == {"DP-1": 0.8}
    # the fade never started; the only Spotify call is the restore's own
    assert spotify.calls == [("volume", 63)]


def test_no_spotify_device_is_a_night_without_music_not_an_error(tmp_path, runs):
    wd = make(tmp_path, spotify=FakeSpotify(fail=True))
    assert wd.start() and wait_idle(wd)
    assert argv_of(runs, "xrandr", "--output")          # the screen still dimmed
    state = json.loads((tmp_path / "winddown.json").read_text())
    assert state["volume"] is None


def test_brightness_is_floored_well_above_black(tmp_path, runs):
    wd = make(tmp_path, brightness=0.0)
    assert wd.start() and wait_idle(wd)
    assert argv_of(runs, "xrandr", "--output")[0][4] == \
        f"{wd_mod.MIN_BRIGHTNESS:.2f}"


def test_each_half_has_its_own_switch(tmp_path, runs):
    spotify, quiet = FakeSpotify(), FakeQuiet(end=time.time() + 3600)
    wd = make(tmp_path, spotify=spotify, quiet=quiet,
              music=False, night_light=False, dnd=False)
    assert wd.start() and wait_idle(wd)
    assert spotify.calls == [] and quiet.dnd == []
    assert argv_of(runs, "gsettings", "set") == []
    assert argv_of(runs, "xrandr", "--output")          # the dim still happened


def test_without_quiet_hours_dnd_runs_to_the_configured_morning(tmp_path, runs):
    quiet = FakeQuiet(end=None)
    now = datetime.now().astimezone().replace(hour=23, minute=0, second=0,
                                              microsecond=0)
    wd = make(tmp_path, spotify=None, quiet=quiet, morning="07:00")
    wd._now = lambda: now.timestamp()
    assert wd.start() and wait_idle(wd)
    assert len(quiet.dnd) == 1 and abs(quiet.dnd[0] - 8 * 3600) < 1


def test_the_clock_helper_walks_to_tomorrow(tmp_path):
    at = datetime.now().astimezone().replace(hour=23, minute=30, second=0,
                                             microsecond=0)
    assert abs(wd_mod._clock_seconds(at.timestamp(), "07:00") - 7.5 * 3600) < 1
    early = at.replace(hour=5)                      # 05:30, so 07:00 is today
    assert abs(wd_mod._clock_seconds(early.timestamp(), "07:00") - 1.5 * 3600) < 1
    # nonsense falls back to a morning rather than raising into the turn
    assert wd_mod._clock_seconds(at.timestamp(), "not a clock") > 0


# ------------------------------------------------------------- restore
def test_good_morning_puts_everything_back(tmp_path, runs):
    spotify = FakeSpotify(volume=70)
    wd = make(tmp_path, spotify=spotify)
    assert wd.start() and wait_idle(wd)
    runs.clear()
    spotify.calls.clear()

    assert wd.restore() is True
    dims = {a[2]: a[4] for a in argv_of(runs, "xrandr", "--output")}
    assert dims == {"HDMI-0": "1.00", "DP-1": "0.80"}
    assert argv_of(runs, "gsettings", "set")[-1][-1] == "false"
    assert spotify.calls == [("volume", 70)]
    assert not (tmp_path / "winddown.json").exists()
    # idempotent: nothing to undo twice
    assert wd.restore() is False


def test_a_restart_in_the_night_leaves_the_room_dark(tmp_path, runs):
    wd = make(tmp_path, spotify=None, quiet=FakeQuiet(end=time.time() + 6 * 3600))
    assert wd.start() and wait_idle(wd)
    runs.clear()
    assert wd.restore(expired_only=True) is False
    assert runs == [] and (tmp_path / "winddown.json").exists()


def test_a_restart_after_the_window_brightens_it(tmp_path, runs):
    wd = make(tmp_path, spotify=None, quiet=FakeQuiet(end=time.time() + 6 * 3600))
    assert wd.start() and wait_idle(wd)
    wd._now = lambda: time.time() + 7 * 3600      # the next morning
    runs.clear()
    assert wd.restore(expired_only=True) is True
    assert {a[2]: a[4] for a in argv_of(runs, "xrandr", "--output")} == \
        {"HDMI-0": "1.00", "DP-1": "0.80"}


def test_a_forgotten_state_file_is_always_restored(tmp_path, runs):
    """A file from some night days ago -- whatever its `until` says, a
    screen must not stay dim because a clock went backwards."""
    (tmp_path / "winddown.json").write_text(json.dumps({
        "at": time.time() - 5 * 86400, "until": time.time() + 10 * 86400,
        "brightness": {"HDMI-0": 1.0}, "night_light": False, "volume": None}))
    wd = make(tmp_path)
    assert wd.restore(expired_only=True) is True


def test_a_corrupt_state_file_is_not_a_crash(tmp_path, runs):
    (tmp_path / "winddown.json").write_text("{not json")
    wd = make(tmp_path)
    assert wd.restore() is False and wd.active is False


def test_stopping_the_app_abandons_the_fade_but_keeps_the_record(tmp_path, runs):
    spotify = FakeSpotify(volume=100)
    wd = make(tmp_path, spotify=spotify, fade_s=30)      # 6 steps of 5 s
    assert wd.start()
    time.sleep(0.3)
    wd.stop()
    assert wait_idle(wd, timeout=3)
    assert len(spotify.calls) < 6                        # it did not run to zero
    assert (tmp_path / "winddown.json").exists()         # the morning can undo it


# ------------------------------------------------- the commander hooks
class Cmd:
    """The two commander helpers only need _svc."""

    def __init__(self, wd=None, **svc):
        self.services = SimpleNamespace(winddown=wd, **svc)

    def _svc(self, name):
        return getattr(self.services, name, None)


def test_good_night_starts_it_even_with_briefings_off(monkeypatch):
    from jarvis import commander as cmd_mod
    started = []
    c = Cmd(wd=SimpleNamespace(start=lambda: bool(started.append(1)) or True),
            assistant=SimpleNamespace(get=lambda k, d=None: False))
    res = cmd_mod._h_courtesy(c, "good night", "goodnight")
    assert started == [1]
    # the preview bailed (briefing.enabled false) and the courtesy line stands
    assert isinstance(res, CommandResult) and res.speak and res.reply


def test_a_start_that_raises_never_reaches_the_user(monkeypatch):
    from jarvis import commander as cmd_mod

    def boom():
        raise RuntimeError("xrandr exploded")

    c = Cmd(wd=SimpleNamespace(start=boom),
            assistant=SimpleNamespace(get=lambda k, d=None: False))
    res = cmd_mod._h_courtesy(c, "good night", "goodnight")
    assert res.handled and res.reply


def test_good_morning_restores_through_both_doors():
    from jarvis import commander as cmd_mod
    for handler, text, kind in ((cmd_mod._h_greeting, "good morning", "greeting"),
                                (cmd_mod._h_briefing, "good morning", None)):
        calls = []
        c = Cmd(wd=SimpleNamespace(restore=lambda: bool(calls.append(1)) or True),
                assistant=SimpleNamespace(get=lambda k, d=None: False),
                brain=None)
        handler(c, text, kind)
        assert calls == [1], handler.__name__


def test_a_greeting_that_is_not_a_morning_leaves_it_alone():
    from jarvis import commander as cmd_mod
    calls = []
    c = Cmd(wd=SimpleNamespace(restore=lambda: bool(calls.append(1)) or True))
    cmd_mod._h_greeting(c, "how are you", "wellbeing")
    assert calls == []


def test_the_commander_survives_a_box_without_the_module():
    from jarvis import commander as cmd_mod
    c = Cmd(wd=None, assistant=SimpleNamespace(get=lambda k, d=None: False))
    assert cmd_mod._h_courtesy(c, "good night", "goodnight").handled
    assert cmd_mod._h_greeting(c, "good morning", "greeting").handled


# --------------------------------------------------------- quiet policy
def test_quiet_hours_end_is_a_public_seam(tmp_path):
    from jarvis.quiet import QuietPolicy
    store = {"quiet.hours": {"start": "23:00", "end": "07:00"}}
    pol = QuietPolicy(SimpleNamespace(
        get=lambda k, d=None: store.get(k, d), set=store.__setitem__))
    at = datetime.now().astimezone().replace(hour=23, minute=30, second=0,
                                             microsecond=0)
    assert abs(pol.hours_end(at.timestamp()) - (at + timedelta(hours=7.5)).timestamp()) < 2
    store["quiet.hours"] = {"start": "", "end": ""}
    assert pol.hours_end(at.timestamp()) is None


# ---------------------------------------------------------- the wiring
def test_the_app_builds_it_and_restores_at_start(build, monkeypatch):  # noqa: F811
    calls = []
    monkeypatch.setattr(wd_mod, "_run", lambda argv, timeout=5.0: (True, ""))
    app = build()
    assert type(app.winddown).__name__ == "WindDown"
    assert app.services.winddown is app.winddown
    assert isinstance(app.commander, Commander)
    app.winddown.restore = lambda expired_only=False: calls.append(expired_only)
    app.start_assistant(residency=False)
    assert calls == [True], "app start must undo a screen dimmed last night"
    app.stop_assistant()


def test_the_state_file_lives_under_the_memory_dir(build):  # noqa: F811
    from jarvis.config import PATHS
    app = build()
    assert app.winddown._state_path == PATHS.MEMORY_DIR / "winddown.json"


# ------------------------------------------- a restore that did not take
def test_a_restore_the_desktop_refuses_keeps_the_snapshot(tmp_path, monkeypatch):
    """The regression: restore() discarded both setters' booleans, called
    _clear_state() unconditionally and returned True, so ONE attempt made
    against a dead X -- the app-start call runs before the session is
    necessarily up, and `--restore` is usually typed from a TTY -- deleted
    the snapshot the module's docstring promises and left the screen dim
    with nothing left to put it back."""
    live = {"up": True}
    seen: list[list] = []

    def flaky(argv, timeout=5.0):
        seen.append(list(argv))
        if not live["up"]:
            return False, ""
        if argv[0] == "xrandr" and "--verbose" in argv:
            return True, XRANDR_OUT
        if argv[0] == "gsettings" and argv[1] == "get":
            return True, "false\n"
        return True, ""

    monkeypatch.setattr(wd_mod, "_run", flaky)
    spotify = FakeSpotify(volume=70)
    wd = make(tmp_path, spotify=spotify)
    assert wd.start() and wait_idle(wd)
    before = json.loads((tmp_path / "winddown.json").read_text())

    live["up"] = False                      # X is gone: a TTY, or a boot
    assert wd.restore() is False, "a restore that did not take must say so"
    kept = json.loads((tmp_path / "winddown.json").read_text())
    assert kept["brightness"] == before["brightness"]
    assert kept["night_light"] is False

    live["up"] = True                       # the session comes back
    del seen[:]
    assert wd.restore() is True
    assert {a[2]: a[4] for a in argv_of(seen, "xrandr", "--output")} == \
        {"HDMI-0": "1.00", "DP-1": "0.80"}
    assert not (tmp_path / "winddown.json").exists()


def test_the_kept_snapshot_is_pruned_to_what_failed(tmp_path, monkeypatch):
    """Only DP-1 refuses, so only DP-1 is left to retry -- the record can
    never drift back over an output that already came home."""
    def one_bad(argv, timeout=5.0):
        if argv[0] == "xrandr" and "--verbose" in argv:
            return True, XRANDR_OUT
        if argv[0] == "gsettings" and argv[1] == "get":
            return True, "false\n"
        if argv[:2] == ["xrandr", "--output"] and argv[2] == "DP-1":
            return False, ""
        return True, ""

    monkeypatch.setattr(wd_mod, "_run", one_bad)
    (tmp_path / "winddown.json").write_text(json.dumps({
        "at": time.time(), "until": time.time() - 1,
        "brightness": {"HDMI-0": 1.0, "DP-1": 0.8},
        "night_light": False, "volume": None}))
    wd = make(tmp_path)
    assert wd.restore() is False
    kept = json.loads((tmp_path / "winddown.json").read_text())
    assert kept["brightness"] == {"DP-1": 0.8}
    assert "night_light" not in kept, "the gsettings write did come back"


# ------------------------------------------------------- who owns the room
def test_holding_is_the_open_window_not_merely_a_state_file(tmp_path, runs):
    """`holding` is the seam jarvis/room.py's boot and quit heals and
    jarvis/scenes.py ask before touching the same panel, so it has to mean
    "deliberately held", not "a file exists"."""
    wd = make(tmp_path, spotify=None, quiet=FakeQuiet(end=time.time() + 6 * 3600))
    assert wd.holding is False
    assert wd.start() is True
    # Raised synchronously inside start(): the caller decides on the very
    # next line whether the scene should also run, and the worker has not
    # snapshotted anything yet.
    assert wd.holding is True
    assert wait_idle(wd)
    assert wd.holding is True and wd.active is True

    wd._now = lambda: time.time() + 7 * 3600          # the next morning
    assert wd.holding is False, "an expired window is not a hold"
    assert wd.active is True, "...but there is still something to put back"


def test_a_forgotten_state_file_is_not_a_hold(tmp_path, runs):
    (tmp_path / "winddown.json").write_text(json.dumps({
        "at": time.time() - 5 * 86400, "until": time.time() + 10 * 86400,
        "brightness": {"HDMI-0": 1.0}, "night_light": False, "volume": None}))
    wd = make(tmp_path)
    assert wd.active is True and wd.holding is False


def test_a_scene_that_already_has_the_room_stands_the_wind_down_down(tmp_path,
                                                                    runs):
    """ONE owner. A "power down the workshop" earlier in the evening already
    dimmed this panel through jarvis/scenes.py; snapshotting now would
    record ITS dimmed screen as the brightness to go back to."""
    wd = make(tmp_path, spotify=FakeSpotify())
    wd.services.scenes = SimpleNamespace(active=lambda: "wind down")
    assert wd.start() is False
    assert runs == [] and not (tmp_path / "winddown.json").exists()

    wd.services.scenes = SimpleNamespace(active=lambda: "")
    assert wd.start() is True and wait_idle(wd)
    assert (tmp_path / "winddown.json").exists()


def test_a_broken_scene_probe_never_costs_him_his_bedtime(tmp_path, runs):
    def boom():
        raise RuntimeError("scene state unreadable")

    wd = make(tmp_path, spotify=None)
    wd.services.scenes = SimpleNamespace(active=boom)
    assert wd.start() is True and wait_idle(wd)
