"""Scenes and the room-control commands (jarvis/scenes.py + commander).

Nothing here shells out: the light is a jarvis.room.RoomLight with a fake
``run`` (the same seam tests/test_room.py uses), and Spotify and the quiet
policy are stand-ins that record what they were asked to do.
"""
import types
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
from jarvis import room, scenes
from jarvis.commander import Commander, IntentClassifier, REGISTRY
from jarvis.config import CONFIG
from tests.test_room import VERBOSE, FakeRun


DOWN = [
    {"do": "temperature", "kelvin": 2700},
    {"do": "brightness", "level": 0.6},
    {"do": "music", "action": "pause"},
    {"do": "quiet_hours", "start": "22:00", "end": "07:00"},
    {"do": "say", "line": "Powering down the workshop, sir."},
]


class FakeSpotify:
    def __init__(self, boom=False):
        self.calls = []
        self.boom = boom

    def control(self, action="", value=None, device=None):
        self.calls.append(action)
        if self.boom:
            raise RuntimeError("no credentials")


class FakeQuiet:
    def __init__(self):
        self.hours = []

    def set_hours(self, start, end):
        self.hours.append((start, end))


def make(tmp_path, steps=None, cfg=None, spotify=None, quiet=None, run=None):
    settings = {"room.enabled": True,
                "room.scenes": {scenes.WIND_DOWN: steps if steps is not None else DOWN},
                "quiet.hours": {"start": "", "end": ""}}
    settings.update(cfg or {})
    assistant = types.SimpleNamespace(get=lambda k, d=None: settings.get(k, d))
    services = types.SimpleNamespace(
        assistant=assistant,
        spotify=FakeSpotify() if spotify is None else spotify,
        quiet=FakeQuiet() if quiet is None else quiet)
    light = room.RoomLight(run=run or FakeRun(),
                           state_path=tmp_path / "room.json",
                           now=lambda: 1000.0)
    return scenes.Scenes(services, light=light,
                         state_path=tmp_path / "scene.json",
                         now=lambda: 1000.0), services, light


# ----------------------------------------------------------------- pure
def test_parse_hhmm():
    assert scenes.parse_hhmm("22:00") == (22, 0)
    assert scenes.parse_hhmm("7:05") == (7, 5)
    for bad in ("", "25:00", "22:70", "ten", None, "2200"):
        assert scenes.parse_hhmm(bad) is None


def test_normalize_drops_what_it_cannot_run_or_reverse():
    steps = scenes.normalize_steps([
        {"do": "brightness", "level": 0.6},
        {"do": "brightness"},                       # no level
        {"do": "temperature", "kelvin": "warm"},    # not a number
        {"do": "music", "action": "explode"},
        {"do": "quiet_hours", "start": "22:00", "end": "22:00"},
        {"do": "teleport"},
        "not a step",
        {"do": "say", "line": "  Good  night, sir. "},
    ])
    assert [s["do"] for s in steps] == ["brightness", "say"]
    assert steps[1]["line"] == "Good night, sir."


def test_normalize_clamps_to_the_room_limits():
    steps = scenes.normalize_steps([{"do": "brightness", "level": 0.0},
                                    {"do": "temperature", "kelvin": 500}])
    assert steps[0]["level"] == room.MIN_BRIGHTNESS      # never black
    assert steps[1]["kelvin"] == room.MIN_KELVIN


def test_light_target_collects_both_knobs_into_one_change():
    """plan_light is what knows night light must be written before
    brightness -- it can only know it if it sees both at once."""
    target = scenes.light_target(scenes.normalize_steps(DOWN))
    assert target["brightness"] == 0.6
    assert target["night_light"]["enabled"] is True
    assert target["night_light"]["automatic"] is False


def test_scene_line_is_the_last_say_step():
    assert scenes.scene_line(scenes.normalize_steps(DOWN)).startswith("Powering down")
    assert scenes.scene_line([]) == ""


# ---------------------------------------------------------------- apply
def test_the_wind_down_moves_every_knob(tmp_path):
    run = FakeRun()
    sc, services, _ = make(tmp_path, run=run)
    res = sc.apply()
    assert res.ok and res.failed == []
    assert set(res.applied) == {"light", "music", "quiet_hours"}
    assert res.line == "Powering down the workshop, sir."
    assert services.spotify.calls == ["pause"]
    assert services.quiet.hours == [((22, 0), (7, 0))]
    assert ["xrandr", "--output", "HDMI-0", "--brightness", "0.60"] in run.calls
    assert sc.active() == scenes.WIND_DOWN


def test_the_panel_is_dimmed_never_blanked(tmp_path):
    """DPMS off would take Jarvis's own console with it."""
    run = FakeRun()
    sc, _, _ = make(tmp_path, run=run)
    sc.apply()
    for argv in run.calls:
        assert "xset" not in argv[0]
        assert "dpms" not in " ".join(argv).lower()


def test_night_light_is_written_before_brightness_in_a_scene(tmp_path):
    run = FakeRun()
    sc, _, _ = make(tmp_path, run=run)
    sc.apply()
    writes = [c for c in run.calls
              if c[:2] == ["gsettings", "set"] or "--brightness" in c]
    assert writes[-1][0] == "xrandr"
    assert all(c[0] == "gsettings" for c in writes[:-1])


def test_a_missing_spotify_does_not_stop_the_scene(tmp_path):
    sc, services, _ = make(tmp_path, spotify=FakeSpotify(boom=True))
    res = sc.apply()
    assert "music" in res.failed
    assert "light" in res.applied and "quiet_hours" in res.applied


def test_an_unknown_scene_says_so(tmp_path):
    sc, _, _ = make(tmp_path)
    res = sc.apply("teleport the workshop")
    assert not res.ok and res.line == scenes.UNKNOWN_LINE


def test_scenes_can_be_switched_off(tmp_path):
    sc, _, _ = make(tmp_path, cfg={"room.enabled": False})
    res = sc.apply()
    assert not res.ok and res.line == scenes.DISABLED_LINE
    assert sc.active() == ""


def test_the_scene_is_recorded_before_the_first_change(tmp_path):
    seen = []
    inner = FakeRun()

    def run(argv, **kw):
        seen.append((tmp_path / "scene.json").exists())
        return inner(argv, **kw)

    sc, _, _ = make(tmp_path, run=run)
    sc.apply()
    assert seen and all(seen[1:]), "a crash mid-scene would leave nothing to reverse"


# -------------------------------------------------------------- restore
def test_good_morning_puts_every_knob_back(tmp_path):
    run = FakeRun()
    sc, services, light = make(tmp_path, run=run)
    sc.apply()
    run.night = {"automatic": False, "from": 0.0, "to": 23.99,
                 "temperature": 2700, "enabled": True}
    run.verbose = VERBOSE.replace("Brightness: 1.0", "Brightness: 0.60")
    del run.calls[:]
    res = sc.restore()
    assert res.ok and res.line == scenes.RESTORED_LINE
    assert services.spotify.calls == ["pause", "resume"]
    assert services.quiet.hours[-1] == (None, None)     # none were set before
    assert run.calls[-1] == ["xrandr", "--output", "HDMI-0", "--brightness", "1.00"]
    assert sc.active() == ""
    assert not light.changed


def test_restore_returns_quiet_hours_he_had_set_himself(tmp_path):
    sc, services, _ = make(tmp_path,
                           cfg={"quiet.hours": {"start": "23:30", "end": "06:30"}})
    sc.apply()
    sc.restore()
    assert services.quiet.hours == [((22, 0), (7, 0)), ((23, 30), (6, 30))]


def test_restore_with_nothing_running_is_a_no_op(tmp_path):
    sc, services, _ = make(tmp_path)
    res = sc.restore()
    assert res.line == scenes.NOTHING_TO_RESTORE_LINE
    assert services.spotify.calls == [] and services.quiet.hours == []


def test_restore_works_from_the_state_file_after_a_restart(tmp_path):
    run = FakeRun()
    sc, _, _ = make(tmp_path, run=run)
    sc.apply()
    fresh, services, _ = make(tmp_path, run=run)       # a new process
    assert fresh.active() == scenes.WIND_DOWN
    fresh.restore()
    assert services.spotify.calls == ["resume"]


def test_restore_lifts_a_bare_dim_with_no_scene(tmp_path):
    run = FakeRun()
    sc, _, light = make(tmp_path, run=run)
    light.dim()
    assert light.changed
    res = sc.restore()
    assert res.line == scenes.RESTORED_LINE and not light.changed


# ------------------------------------------------------ commander wiring
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    sc, services, light = make(tmp_path)
    services.desktop = MagicMock()
    services.desktop.parse_action = lambda part: None
    services.room_light = light
    services.scenes = sc
    services.brain = MagicMock()
    services.memory = MagicMock()
    services.context = MagicMock()
    services.context.answer_question.return_value = None
    services.context.get_last_window.return_value = None
    services.memory.suggest_by_habit.return_value = None
    return Commander(services), services, sc, light


def test_the_scene_command_outranks_the_light_command():
    names = [c.name for c in REGISTRY]
    assert names.index("scene") < names.index("room light")


@pytest.mark.parametrize("phrase,kind", [
    ("dim it a little", "dim"),
    ("dim the screen", "dim"),
    ("lights down", "dim"),
    ("turn the lights down", "dim"),
    ("it's too bright", "dim"),
    ("brighter", "brighter"),
    ("brighten the display", "brighter"),
    ("lights up", "up"),
    ("full brightness", "up"),
    ("warmer", "warm"),
    ("warm the screen", "warm"),
    ("night light on", "warm"),
    ("cooler", "cool"),
    ("back to daylight", "cool"),
])
def test_light_phrases(phrase, kind):
    assert commander.room_light_kind(phrase) == kind


@pytest.mark.parametrize("phrase", [
    "what's the weather", "dim sum", "turn the volume down",
    "is it warmer today", "read me the lights chapter",
])
def test_light_matcher_leaves_other_sentences_alone(phrase):
    assert commander.room_light_kind(phrase) is None


@pytest.mark.parametrize("phrase,kind", [
    ("power down the workshop", "down"),
    ("shut down the lab", "down"),
    ("wind down", "down"),
    ("lights out", "down"),
    ("call it a night", "down"),
    ("wake up the workshop", "up"),
    ("bring the workshop back up", "up"),
])
def test_scene_phrases(phrase, kind):
    assert commander.scene_kind(phrase) == kind


def test_dim_it_a_little_dims_the_display(cmdr):
    c, _, _, light = cmdr
    res = c.handle("jarvis dim it a little", source="voice")
    assert res.handled and res.reply == room.DIM_LINE and res.speak
    assert light.changed


def test_power_down_the_workshop_runs_the_scene(cmdr):
    c, services, sc, _ = cmdr
    res = c.handle("power down the workshop", source="voice")
    assert res.handled and res.reply.startswith("Powering down")
    assert services.spotify.calls == ["pause"]
    assert sc.active() == scenes.WIND_DOWN


def test_lights_up_reverses_the_scene(cmdr):
    c, services, sc, light = cmdr
    c.handle("power down the workshop", source="voice")
    res = c.handle("lights up", source="voice")
    assert res.reply == scenes.RESTORED_LINE
    assert services.spotify.calls == ["pause", "resume"]
    assert sc.active() == "" and not light.changed


def test_lights_up_after_a_bare_dim_restores_the_baseline(cmdr):
    c, _, _, light = cmdr
    c.handle("dim it a little", source="voice")
    res = c.handle("lights up", source="voice")
    assert res.reply in (room.RESTORED_LINE, scenes.RESTORED_LINE)
    assert not light.changed


def test_good_night_does_not_wind_down_unless_asked(cmdr):
    """commander._goodnight_preview already owns "good night"; a scene is a
    change to his desktop he did not ask for by saying it."""
    c, services, sc, light = cmdr
    res = c.handle("jarvis good night", source="voice")
    assert res.handled and res.speak and sc.active() == ""
    assert not light.changed and services.spotify.calls == []


def test_good_night_winds_down_when_he_asked_for_it(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "talkback", True)
    sc, services, light = make(tmp_path, cfg={"room.wind_down_on_goodnight": True})
    services.desktop = MagicMock()
    services.desktop.parse_action = lambda part: None
    services.room_light, services.scenes = light, sc
    c = Commander(services)
    res = c.handle("jarvis good night", source="voice")
    assert sc.active() == scenes.WIND_DOWN
    assert services.spotify.calls == ["pause"]
    assert res.reply.startswith("Powering down")


def test_good_morning_brings_the_room_back(cmdr):
    c, services, sc, light = cmdr
    c.handle("power down the workshop", source="voice")
    res = c.handle("good morning", source="voice")
    assert res.handled
    assert sc.active() == "" and not light.changed
    assert services.spotify.calls == ["pause", "resume"]


def test_room_commands_bypass_the_intent_gate(cmdr):
    c, _, _, _ = cmdr
    assert c._match_assistant("dim it a little") == "room light"
    assert c._match_assistant("power down the workshop") == "scene"


def test_no_display_is_admitted_rather_than_faked(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "talkback", True)
    sc, services, light = make(tmp_path,
                               run=FakeRun(query="Screen 0: minimum 8 x 8\n"))
    services.desktop = MagicMock()
    services.desktop.parse_action = lambda part: None
    services.room_light, services.scenes = light, sc
    res = Commander(services).handle("dim it a little", source="voice")
    assert res.reply == room.NO_DISPLAY_LINE


# --------------------------------------------- a restore that half failed
def test_a_failed_resume_keeps_the_scene_state_and_stops_claiming_success(tmp_path):
    """The regression: restore() recorded the failure in res.failed and then
    called _save({}) -- which UNLINKS the file -- outside any success check,
    while still answering RESTORED_LINE. The music really is still paused
    (state["music"] is only written when the pause took), so that deleted
    the one record that could ever resume it."""
    spotify = FakeSpotify()
    sc, services, _ = make(tmp_path, spotify=spotify)
    sc.apply()
    assert sc.active() == scenes.WIND_DOWN
    spotify.boom = True                       # Spotify falls over overnight
    res = sc.restore()
    assert "music" in res.failed and res.ok is False
    assert res.line == scenes.PARTLY_RESTORED_LINE, "do not claim a clean restore"
    assert sc.active() == scenes.WIND_DOWN, "the morning must keep something to retry"

    spotify.boom = False                      # ...and the retry finishes it
    res = sc.restore()
    assert res.ok and res.line == scenes.RESTORED_LINE
    assert spotify.calls[-1] == "resume" and sc.active() == ""


def test_a_failed_quiet_write_keeps_the_window_he_had(tmp_path):
    """quiet.set_hours persists to assistant.json, so a failed restore left
    the scene's 22:00-07:00 window written there with his own deleted."""
    class Flaky(FakeQuiet):
        boom = False

        def set_hours(self, start, end):
            if self.boom:
                raise RuntimeError("config is read-only")
            super().set_hours(start, end)

    quiet = Flaky()
    sc, _, _ = make(tmp_path, quiet=quiet,
                    cfg={"quiet.hours": {"start": "23:30", "end": "06:30"}})
    sc.apply()
    quiet.boom = True
    res = sc.restore()
    assert "quiet_hours" in res.failed and not res.ok
    kept = sc._load()
    assert kept["quiet_hours_prior"] == {"start": "23:30", "end": "06:30"}
    assert kept["name"] == scenes.WIND_DOWN, "_load needs a name to find it"

    quiet.boom = False
    assert sc.restore().ok
    assert quiet.hours[-1] == ((23, 30), (6, 30))
    assert sc.active() == ""


def test_a_clean_restore_still_forgets_everything(tmp_path):
    sc, _, _ = make(tmp_path)
    sc.apply()
    res = sc.restore()
    assert res.ok and res.line == scenes.RESTORED_LINE
    assert not (tmp_path / "scene.json").exists()


# ------------------------------------------- one owner per "good night"
def _winddown(tmp_path, services, monkeypatch, **cfg):
    """A REAL jarvis.winddown.WindDown on the same services namespace the
    app gives it, with its one subprocess seam stubbed."""
    import jarvis.winddown as wd_mod

    def fake_run(argv, timeout=5.0):
        if argv[0] == "xrandr" and "--verbose" in argv:
            return True, ("HDMI-0 connected primary 3840x2160+0+0 normal\n"
                          "\tBrightness: 1.0\n")
        return True, "false\n"

    monkeypatch.setattr(wd_mod, "_run", fake_run)
    wd = wd_mod.WindDown(services, state_path=tmp_path / "winddown.json")
    services.winddown = wd
    return wd


def test_the_wind_down_and_the_scene_never_both_take_the_night(tmp_path,
                                                               monkeypatch):
    """ONE owner. jarvis/winddown.py fades Spotify to nothing over 60 s and
    only THEN pauses; the scene pauses instantly. With both switched on the
    scene's pause used to land a minute before the fade's first step --
    deterministically defeating the fade the user configured -- and the two
    modules raced to snapshot the same xrandr brightness into two separate
    state files, so the loser recorded an already-dimmed screen as the
    level to restore."""
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "talkback", True)
    sc, services, light = make(tmp_path, cfg={
        "room.wind_down_on_goodnight": True,
        "wind_down.enabled": True, "wind_down.fade_s": 60})
    services.desktop = MagicMock()
    services.desktop.parse_action = lambda part: None
    services.room_light, services.scenes = light, sc
    wd = _winddown(tmp_path, services, monkeypatch)
    try:
        res = Commander(services).handle("jarvis good night", source="voice")
        assert wd.holding is True, "the wind-down took the night"
        # ...so the scene stood down: no instant pause over the fade, and no
        # second snapshot of the same brightness.
        assert services.spotify.calls == []
        assert sc.active() == ""
        assert not (tmp_path / "scene.json").exists()
        assert "Powering down" not in (res.reply or "")
        # ...and the courtesy never even reached for the scene: asking a
        # held Scenes to apply answers with WIND_DOWN_HELD_LINE, which
        # would then be spoken as if it were the good-night line.
        assert scenes.WIND_DOWN_HELD_LINE not in (res.reply or "")
    finally:
        wd.stop()


def test_the_scene_still_runs_when_the_wind_down_is_switched_off(tmp_path,
                                                                monkeypatch):
    """The other half of the rule: exactly ONE of the two, never neither."""
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "talkback", True)
    sc, services, light = make(tmp_path, cfg={
        "room.wind_down_on_goodnight": True, "wind_down.enabled": False})
    services.desktop = MagicMock()
    services.desktop.parse_action = lambda part: None
    services.room_light, services.scenes = light, sc
    wd = _winddown(tmp_path, services, monkeypatch)
    try:
        res = Commander(services).handle("jarvis good night", source="voice")
        assert sc.active() == scenes.WIND_DOWN
        assert services.spotify.calls == ["pause"]
        assert res.reply.startswith("Powering down")
    finally:
        wd.stop()


def test_an_explicit_scene_over_a_live_wind_down_is_refused_out_loud(tmp_path,
                                                                    monkeypatch):
    sc, services, _ = make(tmp_path, cfg={"wind_down.enabled": True})
    wd = _winddown(tmp_path, services, monkeypatch)
    try:
        assert wd.start() is True
        res = sc.apply()
        assert not res.ok and res.failed == ["winddown"]
        assert res.line == scenes.WIND_DOWN_HELD_LINE
        assert services.spotify.calls == [] and sc.active() == ""
    finally:
        wd.stop()


# ------------------- the scene line survives the good-night preview
class _PreviewBrain:
    """Just enough brain for _goodnight_preview to take the turn."""

    def __init__(self):
        self.asked = []

    def chat(self, text, force_tool=None, force_args=None):
        self.asked.append((force_tool, force_args))


def _preview_services(tmp_path, monkeypatch, **cfg):
    settings = {"room.wind_down_on_goodnight": True, "briefing.enabled": True}
    settings.update(cfg)
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "talkback", True)
    sc, services, light = make(tmp_path, cfg=settings)
    services.desktop = MagicMock()
    services.desktop.parse_action = lambda part: None
    services.room_light, services.scenes = light, sc
    services.brain = _PreviewBrain()
    return sc, services, light


def test_good_night_preview_still_says_the_scene_moved_the_room(tmp_path,
                                                               monkeypatch):
    """The scene line is the ONLY announcement that the desktop moved --
    _start_winddown is deliberately silent -- so returning the briefing
    preview unchanged dimmed the screen, paused the music and rearmed quiet
    hours with nothing said about any of it."""
    sc, services, _ = _preview_services(tmp_path, monkeypatch)
    res = Commander(services).handle("jarvis good night", source="voice")
    assert res.status == "Preview…"
    assert sc.active() == scenes.WIND_DOWN
    assert res.reply.startswith("Powering down")
    assert commander.GOODNIGHT_PREVIEW_LINE in res.reply


def test_go_to_sleep_gets_the_same_night_as_good_night(tmp_path, monkeypatch):
    """"good night"/"goodnight" never reach _h_goodnight -- the courtesy
    entry matches them first and never returns None -- so the two phrases
    that DO reach it used to get neither the wind-down nor the scene."""
    sc, services, _ = _preview_services(tmp_path, monkeypatch)
    res = Commander(services).handle("jarvis go to sleep", source="voice")
    assert res.status == "Preview…"
    assert sc.active() == scenes.WIND_DOWN
    assert res.reply.startswith("Powering down")
    assert services.spotify.calls == ["pause"]


def test_go_to_sleep_hands_the_night_to_the_wind_down(tmp_path, monkeypatch):
    """One owner, on this path too: with the wind-down switched on it takes
    the night and the scene is never reached for."""
    sc, services, _ = _preview_services(tmp_path, monkeypatch,
                                        **{"wind_down.enabled": True,
                                           "briefing.enabled": False})
    wd = _winddown(tmp_path, services, monkeypatch)
    try:
        res = Commander(services).handle("jarvis go to sleep", source="voice")
        assert wd.holding is True
        assert sc.active() == "" and services.spotify.calls == []
        assert scenes.WIND_DOWN_HELD_LINE not in (res.reply or "")
        assert "Powering down" not in (res.reply or "")
    finally:
        wd.stop()
