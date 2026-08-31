"""Scenes: "power down the workshop", and the morning that undoes it.

One sentence moves several knobs at once -- the display warms and dims, the
music stops, quiet hours arm -- and one sentence puts every one of them
back.  A scene is DATA (``room.scenes`` in assistant.json): an ordered list
of the primitives from jarvis/room.py and the services already shipped, so
a new scene is a config edit, not a code change::

    {"do": "temperature", "kelvin": 2700}
    {"do": "brightness",  "level": 0.6}
    {"do": "music",       "action": "pause"}
    {"do": "quiet_hours", "start": "22:00", "end": "07:00"}
    {"do": "say",         "line": "Powering down the workshop, sir."}

Four decisions worth stating out loud, because each is a place this could
have gone wrong:

**This does not own "good night".**  commander._goodnight_preview already
turns "good night" into a spoken wind-down and tomorrow's preview.  Scenes
are a SERVICE that call site (and the backlogged bedtime wind-down) can ask
for, gated on ``room.wind_down_on_goodnight`` -- off by default, because a
scene is a change to his desktop and he did not ask for one by saying good
night.  The new spoken verbs are "power down the workshop", "lights down"
and "lights up".

**The panel is never blanked.**  DPMS off would take Jarvis's own console
with it -- the cards, the reactor, an alarm prompt -- so the down scene
dims to ``room.MIN_BRIGHTNESS`` and stops.  Nothing here touches ``xset``.
That is the answer to "what happens when a reminder fires against a
blanked panel": the panel is not blanked, so the card is still readable and
no wake-the-display rule is needed.

**Every step is reversible and the reversal is remembered.**  The light's
baseline lives in jarvis/room.py's state file; what the scene itself did
(the music it paused, the quiet hours it armed and what was set before)
lives in an atomic state file of its own, so ``restore()`` works after a
restart and a half-applied scene still reverses.

**A step that cannot run is not fatal.**  No Spotify credentials, no
display: the rest of the scene still applies, the failure is logged, and
``SceneResult.failed`` names it.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from jarvis import room
from jarvis.logs import get_logger

log = get_logger("scenes")

WIND_DOWN = "wind down"
# The scenes shipped in assistant_config.DEFAULTS live under this key.
CONFIG_KEY = "room.scenes"

APPLIED_LINE = "Powering down the workshop, sir."
RESTORED_LINE = "The workshop's back up, sir."
NOTHING_TO_RESTORE_LINE = "Nothing to bring back, sir."
UNKNOWN_LINE = "I've no scene by that name, sir."
DISABLED_LINE = "The room controls are switched off, sir."

STEPS = ("brightness", "temperature", "music", "quiet_hours", "say")
_CLOCK_RX = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$")


@dataclass
class SceneResult:
    name: str = ""
    line: str = ""
    applied: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    ok: bool = True


# ------------------------------------------------------------------ pure
def parse_hhmm(text) -> Optional[tuple]:
    """"22:00" -> (22, 0). None for anything else (quiet.set_hours takes
    a tuple, and a malformed scene must not arm a nonsense window)."""
    m = _CLOCK_RX.match(str(text or "").strip())
    if m is None:
        return None
    h, minute = int(m.group("h")), int(m.group("m"))
    if not (0 <= h < 24 and 0 <= minute < 60):
        return None
    return h, minute


def normalize_steps(raw) -> list[dict]:
    """Keep the steps this module knows how to run and reverse, in order.

    Hand-edited config is the normal case here, so an unknown verb or a
    malformed value is dropped with a log line rather than raising into
    whatever asked for the scene."""
    out: list[dict] = []
    for item in raw or ():
        if not isinstance(item, dict):
            log.info("scenes: step %r is not an object; skipped", item)
            continue
        do = str(item.get("do") or "").strip().lower()
        if do not in STEPS:
            log.info("scenes: unknown step %r; skipped", do or item)
            continue
        step = {"do": do}
        if do == "brightness":
            try:
                step["level"] = room.clamp_brightness(float(item["level"]))
            except (KeyError, TypeError, ValueError):
                log.info("scenes: brightness step needs a level; skipped")
                continue
        elif do == "temperature":
            try:
                # float() first: clamp_kelvin answers "daylight" for
                # anything it cannot read, and a typo must not silently
                # become a scene step that turns the night light OFF.
                step["kelvin"] = room.clamp_kelvin(float(item["kelvin"]))
            except (KeyError, TypeError, ValueError):
                log.info("scenes: temperature step needs kelvin; skipped")
                continue
        elif do == "music":
            action = str(item.get("action") or "").strip().lower()
            if action not in ("pause", "resume"):
                log.info("scenes: music step needs pause/resume; skipped")
                continue
            step["action"] = action
        elif do == "quiet_hours":
            if item.get("off"):
                step["off"] = True
            else:
                start, end = parse_hhmm(item.get("start")), parse_hhmm(item.get("end"))
                if start is None or end is None or start == end:
                    log.info("scenes: quiet_hours step needs HH:MM start/end; skipped")
                    continue
                step["start"], step["end"] = start, end
        else:                                   # say
            line = " ".join(str(item.get("line") or "").split())
            if not line:
                continue
            step["line"] = line
        out.append(step)
    return out


def light_target(steps) -> dict:
    """The ONE light change a scene's brightness/temperature steps add up
    to.  Collected rather than applied one at a time: jarvis/room.py's
    plan_light is what knows that the night-light keys must be written
    before brightness, and it can only know that if it sees both."""
    target: dict = {}
    for step in steps or ():
        if step.get("do") == "brightness":
            target["brightness"] = step["level"]
        elif step.get("do") == "temperature":
            kelvin = step["kelvin"]
            target["night_light"] = room.night_light_target(
                kelvin < room.NEUTRAL_KELVIN, kelvin)
    return target


def scene_line(steps) -> str:
    """The dry line a scene ends on ("" when it has no say step)."""
    for step in reversed(list(steps or ())):
        if step.get("do") == "say":
            return step["line"]
    return ""


# ---------------------------------------------------------------- runner
class Scenes:
    """The room-state layer: apply a named scene, and put it all back.

    ``services`` is the app's namespace (assistant / quiet / spotify), the
    same seam jarvis/focus.py uses; ``light`` is a jarvis.room.RoomLight.
    Every collaborator is optional -- a box with no Spotify credentials
    still dims and arms quiet hours."""

    def __init__(self, services=None, light: Optional[room.RoomLight] = None,
                 state_path: Optional[Path] = None,
                 now: Callable[[], float] = time.time):
        self.services = services
        self.light = light
        self._state_path = Path(state_path) if state_path else None
        self._now = now
        self._lock = threading.RLock()

    # ------------------------------------------------------------ access
    def _svc(self, name: str):
        return getattr(self.services, name, None)

    def _cfg(self, key: str, default=None):
        cfg = self._svc("assistant")
        get = getattr(cfg, "get", None)
        if callable(get):
            try:
                value = get(key, default)
                return default if value is None else value
            except Exception:  # noqa: BLE001 - a config hiccup is not fatal
                log.debug("scenes: cfg.get(%s) failed", key, exc_info=True)
        return default

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("room.enabled", True))

    def names(self) -> list[str]:
        table = self._cfg(CONFIG_KEY, {}) or {}
        return sorted(table) if isinstance(table, dict) else []

    def steps(self, name: str) -> list[dict]:
        table = self._cfg(CONFIG_KEY, {}) or {}
        if not isinstance(table, dict):
            return []
        return normalize_steps(table.get(name))

    # ------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict) and data.get("name"):
                    return data
        except (OSError, ValueError):
            log.debug("scene state unreadable", exc_info=True)
        return {}

    def _save(self, data: dict) -> None:
        if not self._state_path:
            return
        try:
            if not data:
                self._state_path.unlink(missing_ok=True)
                return
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, self._state_path)   # atomic: never half a file
        except OSError:
            log.debug("scene state save failed", exc_info=True)

    def active(self) -> str:
        return str(self._load().get("name") or "")

    # ------------------------------------------------------------- apply
    def apply(self, name: str = WIND_DOWN) -> SceneResult:
        with self._lock:
            if not self.enabled:
                return SceneResult(name=name, line=DISABLED_LINE, ok=False)
            steps = self.steps(name)
            if not steps:
                return SceneResult(name=name, line=UNKNOWN_LINE, ok=False)
            res = SceneResult(name=name, line=scene_line(steps) or APPLIED_LINE)
            # Written BEFORE the first change: a crash halfway through must
            # still leave "good morning" something to reverse.
            state = {"name": name, "t": self._now()}
            self._save(state)

            target = light_target(steps)
            if target:
                self._step_light(target, res)
            for step in steps:
                do = step["do"]
                if do == "music":
                    self._step_music(step["action"], res, state)
                elif do == "quiet_hours":
                    self._step_quiet(step, res, state)
            self._save(state)
            log.info("scene %r applied: %s%s", name, ", ".join(res.applied),
                     f" (failed: {', '.join(res.failed)})" if res.failed else "")
            return res

    def _step_light(self, target: dict, res: SceneResult) -> None:
        if self.light is None:
            res.failed.append("light")
            return
        try:
            state = self.light.apply(brightness=target.get("brightness"),
                                     night_light=target.get("night_light"))
        except Exception:  # noqa: BLE001 - a scene step is never fatal
            log.exception("scene: light step failed")
            state = None
        if state is None:
            res.failed.append("light")
            res.ok = False
        else:
            res.applied.append("light")

    def _step_music(self, action: str, res: SceneResult, state: dict) -> None:
        spotify = self._svc("spotify")
        if spotify is None or not hasattr(spotify, "control"):
            res.failed.append("music")
            return
        try:
            spotify.control(action)
        except Exception as exc:  # noqa: BLE001 - SpotifyError or worse
            log.info("scene: spotify %s skipped: %s", action,
                     getattr(exc, "text", exc))
            res.failed.append("music")
            return
        # Only a pause is reversed: a scene that started the music is not
        # undone by silence.
        if action == "pause":
            state["music"] = "paused"
        res.applied.append("music")

    def _step_quiet(self, step: dict, res: SceneResult, state: dict) -> None:
        quiet = self._svc("quiet")
        if quiet is None or not hasattr(quiet, "set_hours"):
            res.failed.append("quiet_hours")
            return
        # The window in force BEFORE the scene, so the morning restores what
        # he set rather than assuming the defaults.
        prior = self._cfg("quiet.hours", {}) or {}
        if isinstance(prior, dict):
            state.setdefault("quiet_hours_prior",
                             {"start": str(prior.get("start") or ""),
                              "end": str(prior.get("end") or "")})
        try:
            if step.get("off"):
                quiet.set_hours(None, None)
            else:
                quiet.set_hours(step["start"], step["end"])
        except Exception:  # noqa: BLE001
            log.exception("scene: quiet hours step failed")
            res.failed.append("quiet_hours")
            return
        res.applied.append("quiet_hours")

    # ----------------------------------------------------------- restore
    def restore(self) -> SceneResult:
        """"Good morning" / "lights up": every knob back where he had it."""
        with self._lock:
            state = self._load()
            light_held = self.light is not None and self.light.changed
            if not state and not light_held:
                return SceneResult(line=NOTHING_TO_RESTORE_LINE)
            res = SceneResult(name=str(state.get("name") or ""),
                              line=RESTORED_LINE)
            if self.light is not None:
                try:
                    ok, _ = self.light.restore()
                except Exception:  # noqa: BLE001
                    log.exception("scene: light restore failed")
                    ok = False
                (res.applied if ok else res.failed).append("light")
                res.ok = res.ok and ok
            if state.get("music") == "paused":
                spotify = self._svc("spotify")
                try:
                    if spotify is None or not hasattr(spotify, "control"):
                        raise RuntimeError("no spotify")
                    spotify.control("resume")
                    res.applied.append("music")
                except Exception as exc:  # noqa: BLE001
                    log.info("scene: spotify resume skipped: %s",
                             getattr(exc, "text", exc))
                    res.failed.append("music")
            prior = state.get("quiet_hours_prior")
            if isinstance(prior, dict):
                quiet = self._svc("quiet")
                start, end = parse_hhmm(prior.get("start")), parse_hhmm(prior.get("end"))
                try:
                    if quiet is None or not hasattr(quiet, "set_hours"):
                        raise RuntimeError("no quiet policy")
                    if start is None or end is None:
                        quiet.set_hours(None, None)
                    else:
                        quiet.set_hours(start, end)
                    res.applied.append("quiet_hours")
                except Exception:  # noqa: BLE001
                    log.exception("scene: quiet hours restore failed")
                    res.failed.append("quiet_hours")
            self._save({})
            log.info("scene %r restored: %s", res.name, ", ".join(res.applied))
            return res
