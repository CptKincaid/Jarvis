"""Display light and level (jarvis/room.py).

The xrandr and gsettings text below is verbatim from this box on :1
(read-only probes, 2026-08-30): HDMI-0 connected primary at Brightness 1.0,
night light off, temperature 2700, schedule-automatic TRUE -- which is why
"warmer" has to write the schedule keys and not just the enable flag.
"""
from types import SimpleNamespace

from jarvis import room

QUERY = """Screen 0: minimum 8 x 8, current 3840 x 2160, maximum 32767 x 32767
HDMI-0 connected primary 3840x2160+0+0 (normal left inverted right x axis y axis) 596mm x 335mm
   3840x2160     60.00*+  59.94    50.00
USB-C-0 disconnected (normal left inverted right x axis y axis)
USB-C-1 disconnected (normal left inverted right x axis y axis)
"""

VERBOSE = """Screen 0: minimum 8 x 8, current 3840 x 2160, maximum 32767 x 32767
HDMI-0 connected primary 3840x2160+0+0 (0x18d) normal (normal left inverted right x axis y axis) 596mm x 335mm
\tIdentifier: 0x1c9
\tTimestamp:  9532
\tGamma:      1.0:1.0:1.0
\tBrightness: 1.0
\tCRTC:       0
USB-C-0 disconnected (normal left inverted right x axis y axis)
\tIdentifier: 0x1ca
"""

LIVE_NIGHT = {"automatic": True, "from": 20.0, "to": 6.0,
              "temperature": 2700, "enabled": False}
LIVE = {"output": "HDMI-0", "brightness": 1.0, "night_light": dict(LIVE_NIGHT)}


class FakeRun:
    """The subprocess seam: records argv, answers the two probes."""

    def __init__(self, query=QUERY, verbose=VERBOSE, night=None, rc=0,
                 fail_on=""):
        self.query, self.verbose = query, verbose
        self.night = dict(night or LIVE_NIGHT)
        self.rc, self.fail_on = rc, fail_on
        self.calls: list[list] = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if self.fail_on and self.fail_on in " ".join(argv):
            return SimpleNamespace(returncode=1, stdout="", stderr="nope")
        out = ""
        if argv[:2] == ["xrandr", "--query"]:
            out = self.query
        elif argv[:2] == ["xrandr", "--verbose"]:
            out = self.verbose
        elif argv[:2] == ["gsettings", "get"]:
            key = argv[3]
            name = {room.KEY_AUTO: "automatic", room.KEY_FROM: "from",
                    room.KEY_TO: "to", room.KEY_TEMP: "temperature",
                    room.KEY_ENABLED: "enabled"}[key]
            value = self.night[name]
            if name == "temperature":
                out = f"uint32 {value}"
            elif isinstance(value, bool):
                out = "true" if value else "false"
            else:
                out = str(value)
        return SimpleNamespace(returncode=self.rc, stdout=out, stderr="")

    @property
    def writes(self):
        return [c for c in self.calls
                if c[:2] == ["gsettings", "set"] or "--brightness" in c]


def light(tmp_path, run=None):
    return room.RoomLight(run=run or FakeRun(),
                          state_path=tmp_path / "room.json",
                          now=lambda: 1000.0)


# ---------------------------------------------------------------- parsing
def test_primary_output_is_discovered_not_hardcoded():
    assert room.primary_output(QUERY) == "HDMI-0"
    assert [o["name"] for o in room.parse_outputs(QUERY)] == \
        ["HDMI-0", "USB-C-0", "USB-C-1"]
    assert room.primary_output("") == ""


def test_primary_output_falls_back_to_the_first_connected():
    text = QUERY.replace(" primary", "")
    assert room.primary_output(text) == "HDMI-0"
    only_off = "DP-1 disconnected (normal)\n"
    assert room.primary_output(only_off) == ""


def test_parse_brightness_reads_the_right_output_block():
    assert room.parse_brightness(VERBOSE, "HDMI-0") == 1.0
    assert room.parse_brightness(VERBOSE, "USB-C-0") is None
    assert room.parse_brightness("", "HDMI-0") is None


def test_parse_gvalue_handles_every_night_light_type():
    assert room.parse_gvalue("uint32 2700") == 2700
    assert room.parse_gvalue("true") is True
    assert room.parse_gvalue("false") is False
    assert room.parse_gvalue("20.0") == 20.0
    assert room.parse_gvalue("'x'") == "x"
    assert room.parse_gvalue("") is None


def test_gvalue_formats_for_the_command_line():
    assert room.gvalue(True) == "true" and room.gvalue(False) == "false"
    assert room.gvalue(2700) == "2700"
    assert room.gvalue(23.99) == "23.99"


# --------------------------------------------------------------- clamping
def test_brightness_never_reaches_black():
    assert room.clamp_brightness(0.0) == room.MIN_BRIGHTNESS
    assert room.clamp_brightness(-5) == room.MIN_BRIGHTNESS
    assert room.clamp_brightness(2.0) == 1.0
    assert room.clamp_brightness("dark") == 1.0


def test_stepping_stops_at_the_floor_and_the_ceiling():
    assert room.step_brightness(1.0, -1) == 0.85
    assert room.step_brightness(0.6, -1) == room.MIN_BRIGHTNESS
    assert room.step_brightness(1.0, 1) == 1.0
    assert room.step_kelvin(2700, -1) == 1900
    assert room.step_kelvin(1800, -1) == room.MIN_KELVIN
    assert room.step_kelvin(6500, 1) == room.MAX_KELVIN


# --------------------------------------------------------------- planning
def test_night_light_is_written_before_brightness():
    """gsd-color rewrites the CRTC gamma ramp on a night-light change, so a
    dim applied first would be silently undone."""
    plan = room.plan_light(LIVE, {"brightness": 0.7,
                                  "night_light": room.night_light_target(True)})
    kinds = ["gsettings" if c[0] == "gsettings" else "xrandr" for c in plan]
    assert kinds == sorted(kinds, key=lambda k: 0 if k == "gsettings" else 1)
    assert kinds[-1] == "xrandr"


def test_warming_forces_the_schedule_off():
    """schedule-automatic is TRUE on this box: enabling night light at
    three in the afternoon would otherwise do nothing at all."""
    plan = room.plan_light(LIVE, {"night_light": room.night_light_target(True)})
    keys = [c[3] for c in plan if c[0] == "gsettings"]
    assert room.KEY_AUTO in keys and room.KEY_FROM in keys and room.KEY_TO in keys
    assert keys.index(room.KEY_ENABLED) == len(keys) - 1
    assert ["gsettings", "set", room.SCHEMA, room.KEY_AUTO, "false"] in plan


def test_plan_only_sets_what_differs():
    same = {"output": "HDMI-0", "brightness": 1.0,
            "night_light": dict(LIVE_NIGHT)}
    assert room.plan_light(same, {"brightness": 1.0}) == []
    assert room.plan_light(same, {"night_light": {"temperature": 2700}}) == []


def test_brightness_is_reasserted_after_a_night_light_change():
    dimmed = {"output": "HDMI-0", "brightness": 0.7,
              "night_light": dict(LIVE_NIGHT)}
    plan = room.plan_light(dimmed, {"night_light": room.night_light_target(True)})
    assert plan[-1] == ["xrandr", "--output", "HDMI-0", "--brightness", "0.70"]


def test_no_reassert_when_the_display_is_already_at_full():
    plan = room.plan_light(LIVE, {"night_light": room.night_light_target(True)})
    assert all(c[0] == "gsettings" for c in plan)


def test_plan_without_an_output_writes_no_xrandr():
    headless = {"output": "", "brightness": None, "night_light": dict(LIVE_NIGHT)}
    assert all(c[0] == "gsettings"
               for c in room.plan_light(headless, {"brightness": 0.6}))


# ------------------------------------------------------------------ verbs
def test_dim_lowers_the_display_and_says_so(tmp_path):
    run = FakeRun()
    lt = light(tmp_path, run)
    state, line = lt.dim()
    assert line == room.DIM_LINE
    assert state["brightness"] == 0.85
    assert ["xrandr", "--output", "HDMI-0", "--brightness", "0.85"] in run.calls


def test_dim_stops_at_the_floor_rather_than_going_black(tmp_path):
    run = FakeRun(verbose=VERBOSE.replace("Brightness: 1.0", "Brightness: 0.55"))
    lt = light(tmp_path, run)
    state, line = lt.dim()
    assert line == room.AT_FLOOR_LINE
    assert run.writes == []


def test_brighten_at_full_says_so(tmp_path):
    run = FakeRun()
    assert light(tmp_path, run).brighten()[1] == room.AT_FULL_LINE
    assert run.writes == []


def test_warmer_writes_the_five_night_light_keys(tmp_path):
    run = FakeRun()
    state, line = light(tmp_path, run).warmer()
    assert line == room.WARM_LINE
    keys = [c[3] for c in run.writes]
    assert keys == [room.KEY_AUTO, room.KEY_FROM, room.KEY_TO,
                    room.KEY_TEMP, room.KEY_ENABLED]
    assert state["night_light"]["enabled"] is True


def test_cooler_from_daylight_says_it_is_already_there(tmp_path):
    run = FakeRun()
    assert light(tmp_path, run).cooler()[1] == room.AT_COOLEST_LINE


def test_no_display_is_admitted_not_faked(tmp_path):
    run = FakeRun(query="Screen 0: minimum 8 x 8\n")
    lt = light(tmp_path, run)
    assert lt.dim() == (None, room.NO_DISPLAY_LINE)
    assert lt.warmer() == (None, room.NO_DISPLAY_LINE)
    assert not (tmp_path / "room.json").exists()


# -------------------------------------------------------- reversibility
def test_the_baseline_is_what_he_had_before_the_first_change(tmp_path):
    run = FakeRun()
    lt = light(tmp_path, run)
    lt.dim()
    base = lt.baseline()
    assert base["brightness"] == 1.0
    run.verbose = VERBOSE.replace("Brightness: 1.0", "Brightness: 0.85")
    lt.dim()
    assert lt.baseline()["brightness"] == 1.0, "the baseline drifted"


def test_restore_puts_back_brightness_and_the_schedule_flag(tmp_path):
    run = FakeRun()
    lt = light(tmp_path, run)
    lt.warmer()
    run.night = {"automatic": False, "from": 0.0, "to": 23.99,
                 "temperature": 1900, "enabled": True}
    del run.calls[:]
    ok, line = lt.restore()
    assert ok and line == room.RESTORED_LINE
    assert ["gsettings", "set", room.SCHEMA, room.KEY_AUTO, "true"] in run.calls
    assert ["gsettings", "set", room.SCHEMA, room.KEY_ENABLED, "false"] in run.calls
    assert run.calls[-1] == ["xrandr", "--output", "HDMI-0",
                             "--brightness", "1.00"]
    assert not (tmp_path / "room.json").exists()


def test_restore_writes_brightness_after_the_night_light_keys(tmp_path):
    run = FakeRun()
    lt = light(tmp_path, run)
    lt.warmer()
    del run.calls[:]
    lt.restore()
    kinds = [c[0] for c in run.calls]
    assert kinds[-1] == "xrandr" and "gsettings" in kinds


def test_restore_with_nothing_held_is_a_no_op(tmp_path):
    run = FakeRun()
    ok, line = light(tmp_path, run).restore()
    assert ok and line == room.NOTHING_TO_RESTORE_LINE
    assert run.writes == []


def test_a_crash_at_low_brightness_heals_at_the_next_start(tmp_path):
    """The state file is the whole point: a new RoomLight, a fresh process,
    restores what the dead one was holding."""
    run = FakeRun()
    light(tmp_path, run).dim()
    assert (tmp_path / "room.json").exists()
    run2 = FakeRun(verbose=VERBOSE.replace("Brightness: 1.0", "Brightness: 0.85"))
    ok, _ = light(tmp_path, run2).restore()
    assert ok
    assert run2.calls[-1] == ["xrandr", "--output", "HDMI-0",
                              "--brightness", "1.00"]


def test_a_failed_command_restores_immediately(tmp_path):
    run = FakeRun(fail_on="--brightness")
    lt = light(tmp_path, run)
    state, line = lt.dim()
    assert state is None and line == room.FAILED_LINE
    # The put-back's own --brightness write failed too (the same command is
    # refused), so the baseline is KEPT for the boot heal to retry. Dropping
    # it here was how a display held dim lost the only record of his level.
    assert lt.baseline() is not None
    assert lt.restore()[0] is False


def test_returning_to_full_stops_holding_the_baseline(tmp_path):
    run = FakeRun()
    lt = light(tmp_path, run)
    lt.dim()
    assert lt.changed
    run.verbose = VERBOSE.replace("Brightness: 1.0", "Brightness: 0.85")
    lt.brighten()
    assert not lt.changed


def test_state_file_is_written_before_the_first_command(tmp_path):
    seen = []
    inner = FakeRun()

    def run(argv, **kw):
        seen.append((list(argv), (tmp_path / "room.json").exists()))
        return inner(argv, **kw)

    light(tmp_path, run).dim()
    writes = [(a, existed) for a, existed in seen if "--brightness" in a]
    assert writes and all(existed for _, existed in writes)


def test_an_unreadable_state_file_is_ignored(tmp_path):
    (tmp_path / "room.json").write_text("{not json")
    lt = light(tmp_path, FakeRun())
    assert lt.baseline() is None
    assert lt.restore()[1] == room.NOTHING_TO_RESTORE_LINE


# ------------------------------------------------- a restore that failed
def test_a_restore_that_did_not_take_keeps_the_baseline_and_says_so(tmp_path):
    """The regression: restore() used to _forget() unconditionally and
    always answer RESTORED_LINE, so one attempt against a dead X (the boot
    heal runs before the session is necessarily up) destroyed the only
    record of his brightness AND reported success -- leaving the panel held
    at 0.55 with every retry path dead, because they all gate on
    ``changed``."""
    run = FakeRun()
    lt = light(tmp_path, run)
    lt.dim()
    assert lt.changed
    # X goes away between the dim and the put-back.
    run.fail_on = "--brightness"
    ok, line = lt.restore()
    assert ok is False
    assert line == room.FAILED_RESTORE_LINE, "a failed restore must not claim one"
    assert lt.changed, "the baseline is the retry; it must survive"
    # ...and the retry, once the display answers again, actually works.
    run.fail_on = ""
    ok, line = lt.restore()
    assert ok and line == room.RESTORED_LINE and not lt.changed


def test_a_failed_night_light_write_also_keeps_the_baseline(tmp_path):
    run = FakeRun()
    lt = light(tmp_path, run)
    lt.warmer()
    run.fail_on = "gsettings set"
    ok, line = lt.restore()
    assert ok is False and line == room.FAILED_RESTORE_LINE
    assert lt.baseline() is not None


# ------------------------------------------ the wind-down's hold on boot
def test_the_boot_heal_stands_down_while_the_wind_down_holds_the_room(tmp_path):
    """The regression: start_assistant honours winddown.restore(
    expired_only=True)'s "still inside the window, leave it dark" and then,
    ninety lines later, healed the room light anyway -- driving the same
    xrandr brightness straight back up at two in the morning."""
    held = {"now": True}
    run = FakeRun()
    lt = room.RoomLight(run=run, state_path=tmp_path / "room.json",
                        now=lambda: 1000.0, held_by=lambda: held["now"])
    lt.dim()
    del run.calls[:]

    ok, _ = lt.restore(healing=True)
    assert ok is False
    assert run.writes == [], "the boot heal must not touch a held display"
    assert lt.changed, "and it must leave the baseline for the morning"

    # A deliberate "lights up" is NOT a heal and always wins, even at 2 a.m.
    ok, line = lt.restore()
    assert ok and line == room.RESTORED_LINE and not lt.changed


def test_without_a_hold_the_heal_behaves_exactly_as_before(tmp_path):
    lt = light(tmp_path)                      # no held_by seam at all
    lt.dim()
    assert lt.restore(healing=True)[0] is True and not lt.changed


def test_a_hold_probe_that_raises_never_strands_the_display(tmp_path):
    def boom():
        raise RuntimeError("winddown is not built on this box")

    lt = room.RoomLight(run=FakeRun(), state_path=tmp_path / "room.json",
                        now=lambda: 1000.0, held_by=boom)
    lt.dim()
    assert lt.restore(healing=True)[0] is True and not lt.changed
