"""The room answers: light and level on a box with no bulbs.

"Dim it a little" / "lights down" / "warmer" / "lights up".  The 43-inch
panel on :1 IS the light in that room, so Jarvis moves the display's
brightness and colour temperature -- and says so honestly.  **The screen
dims; the room does not.**  There is no backlight to reach: ``ddcutil`` is
dead twice over here (``/dev/i2c-*`` is root-only and no ``i2c`` group
exists on the box at all).

Two knobs, verified read-only on :1 before a line of this was written:

* ``xrandr --output HDMI-0 --brightness N`` -- X's gamma SCALING, reported
  by ``xrandr --verbose`` as ``Brightness: 1.0``.  The output name is
  discovered from ``xrandr --query``, never hardcoded.
* the three-and-a-bit ``org.gnome.settings-daemon.plugins.color``
  night-light keys (enabled / temperature / schedule-automatic / from /
  to), all ``gsettings writable`` = true.

**They are the same knob.**  Both end up in the CRTC gamma ramp: night
light is a gamma ramp written by gsd-color, and ``--brightness`` is gamma
scaling, not a backlight.  Last writer wins, and gsd-color re-asserts its
ramp whenever the temperature or the schedule changes.  So every plan here
writes the night-light keys FIRST, brightness SECOND, and re-asserts
brightness after any night-light change (``plan_light``).

**The schedule flag is not optional.**  ``night-light-schedule-automatic``
is true on this box, which means flipping ``night-light-enabled`` at three
in the afternoon does nothing at all -- GNOME only warms between its own
sunset and sunrise.  Forcing the warmth means writing automatic=false plus
a from/to window, and putting all three back afterwards.

**Reversibility is the price of admission**, exactly as quiet hours pays it
(capabilities row: quiet hours silences banners in-process, "no gsettings:
a crash mid-window would leave the desktop mute").  Brightness and night
light ARE that hazard, so:

* the values in force before the FIRST change are written atomically to a
  state file, and ``restore()`` puts them back;
* the app calls ``restore(healing=True)`` at boot, so a crash at 0.55 heals
  at the next start, and again at quit -- and those two automatic calls
  stand down while jarvis/winddown.py is deliberately holding the same
  panel dim (``held_by``), because a restart at two in the morning must not
  light the room;
* a restore that did NOT take keeps the baseline and says so
  (``FAILED_RESTORE_LINE``), so the next heal retries rather than leaving
  the display held dim with nothing left to undo it;
* any failure mid-plan restores immediately;
* the floor is ``MIN_BRIGHTNESS`` (0.55), well clear of black.  Jarvis's own
  console lives on this panel: a black screen would take the cards, the
  reactor and any alarm prompt with it.

Seams for tests: ``run`` (the one subprocess call) and ``state_path``.  The
parsing and planning halves are pure functions, so the suite records argv
with no display attached.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("room")

SCHEMA = "org.gnome.settings-daemon.plugins.color"
KEY_ENABLED = "night-light-enabled"
KEY_TEMP = "night-light-temperature"
KEY_AUTO = "night-light-schedule-automatic"
KEY_FROM = "night-light-schedule-from"
KEY_TO = "night-light-schedule-to"

# 0.55, not 0.0: the panel is Jarvis's own console (see the module note).
MIN_BRIGHTNESS = 0.55
MAX_BRIGHTNESS = 1.0
BRIGHTNESS_STEP = 0.15
# GNOME's own night-light slider runs 1700-4700 K; 6500 K is daylight, i.e.
# no warming at all, and is what "cooler" walks back towards.
MIN_KELVIN = 1700
MAX_KELVIN = 6500
NEUTRAL_KELVIN = 6500
WARM_KELVIN = 2700                 # the box's own configured temperature
KELVIN_STEP = 800
# The forced window: with schedule-automatic off, from/to decide when the
# warmth applies, and "now, whatever the hour" is 00:00 to 23:59.
FORCE_FROM = 0.0
FORCE_TO = 23.99
GSETTINGS_TIMEOUT_S = 10.0

# Persona lines (short, dry, and honest about what moved).
DIM_LINE = "Dimming the display, sir."
BRIGHTEN_LINE = "Bringing the display back up, sir."
WARM_LINE = "Warming the screen, sir."
COOL_LINE = "Cooling the screen down, sir."
AT_FLOOR_LINE = "That's as low as I'll take the display, sir."
AT_FULL_LINE = "The display is already at full, sir."
AT_WARMEST_LINE = "The screen is as warm as it goes, sir."
AT_COOLEST_LINE = "The screen is back to daylight, sir."
RESTORED_LINE = "Display restored, sir."
# Said when the put-back itself did not take (X not up yet at boot, the
# output renamed, gsettings missing). Never RESTORED_LINE: the baseline is
# kept for the next attempt, and claiming success is what used to leave him
# holding a dim screen with nothing left to undo it.
FAILED_RESTORE_LINE = "I couldn't put the display back, sir; I'll try again."
NOTHING_TO_RESTORE_LINE = "The display is as you left it, sir."
NO_DISPLAY_LINE = "I've no display to work with, sir."
FAILED_LINE = "The display wouldn't take that, sir; I've put it back."

_OUTPUT_RX = re.compile(r"^([\w-]+) (connected|disconnected)\b(.*)$")
_BRIGHTNESS_RX = re.compile(r"^\s*Brightness:\s*([\d.]+)\s*$")


# ------------------------------------------------------------------ pure
def parse_outputs(text: str) -> list[dict]:
    """``xrandr --query`` -> [{"name", "connected", "primary"}] in order."""
    out = []
    for line in (text or "").splitlines():
        m = _OUTPUT_RX.match(line)
        if m is None:
            continue
        out.append({"name": m.group(1),
                    "connected": m.group(2) == "connected",
                    "primary": " primary" in m.group(3)})
    return out


def primary_output(text: str) -> str:
    """The output to dim: the primary connected one, else the first
    connected one, else "" (no display -- say so, do not guess HDMI-0)."""
    outputs = [o for o in parse_outputs(text) if o["connected"]]
    for o in outputs:
        if o["primary"]:
            return o["name"]
    return outputs[0]["name"] if outputs else ""


def parse_brightness(text: str, output: str = "") -> Optional[float]:
    """The ``Brightness:`` line of ``xrandr --verbose`` for one output.

    Every connected output has one, so the block for ``output`` has to be
    found first; with no name given the first Brightness wins."""
    seen = not output
    for line in (text or "").splitlines():
        m = _OUTPUT_RX.match(line)
        if m is not None:
            seen = (m.group(1) == output)
            continue
        if not seen:
            continue
        b = _BRIGHTNESS_RX.match(line)
        if b is not None:
            try:
                return float(b.group(1))
            except ValueError:
                return None
    return None


def parse_gvalue(text: str):
    """``gsettings get`` output -> a Python value.

    'uint32 2700' -> 2700, 'true' -> True, '20.0' -> 20.0, "'x'" -> 'x'."""
    s = (text or "").strip()
    if not s:
        return None
    for prefix in ("uint32 ", "int32 ", "int64 ", "uint64 ", "double "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def gvalue(value) -> str:
    """A Python value as ``gsettings set`` expects it on the command line."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 4))
    return str(value)


def clamp_brightness(level) -> float:
    try:
        level = float(level)
    except (TypeError, ValueError):
        return MAX_BRIGHTNESS
    return round(max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, level)), 2)


def clamp_kelvin(kelvin) -> int:
    try:
        kelvin = int(round(float(kelvin)))
    except (TypeError, ValueError):
        return NEUTRAL_KELVIN
    return max(MIN_KELVIN, min(MAX_KELVIN, kelvin))


def step_brightness(current, steps: int = -1,
                    step: float = BRIGHTNESS_STEP) -> float:
    """One notch darker (steps<0) or brighter (steps>0), clamped."""
    base = clamp_brightness(current)
    return clamp_brightness(base + steps * step)


def step_kelvin(current, steps: int = -1, step: int = KELVIN_STEP) -> int:
    """steps<0 warms (fewer kelvin), steps>0 cools."""
    return clamp_kelvin(clamp_kelvin(current) + steps * step)


def brightness_argv(output: str, level) -> list[str]:
    return ["xrandr", "--output", str(output), "--brightness",
            f"{clamp_brightness(level):.2f}"]


def gset_argv(key: str, value) -> list[str]:
    return ["gsettings", "set", SCHEMA, key, gvalue(value)]


def gget_argv(key: str) -> list[str]:
    return ["gsettings", "get", SCHEMA, key]


def night_light_target(warm: bool, kelvin: int = WARM_KELVIN) -> dict:
    """The full night-light state for "warm the screen" / "back to daylight".

    ``schedule-automatic`` false plus a from/to window is what makes a
    warm-up at three in the afternoon do anything at all."""
    if not warm:
        return {"enabled": False, "automatic": True}
    return {"enabled": True, "temperature": clamp_kelvin(kelvin),
            "automatic": False, "from": FORCE_FROM, "to": FORCE_TO}


_NL_ORDER = (("automatic", KEY_AUTO), ("from", KEY_FROM), ("to", KEY_TO),
             ("temperature", KEY_TEMP), ("enabled", KEY_ENABLED))


def plan_light(current: dict, target: dict) -> list[list[str]]:
    """The argv to get from ``current`` to ``target``, in the ONE order
    that survives gsd-color.

    Night-light keys first (schedule before temperature before enabled),
    brightness second, and brightness re-asserted last whenever a
    night-light key moved while a dim is in force -- gsd-color rewrites the
    CRTC gamma ramp on those changes and would silently undo the dim."""
    current = current or {}
    cur_nl = dict(current.get("night_light") or {})
    want_nl = dict(target.get("night_light") or {})
    cmds: list[list[str]] = []
    nl_changed = False
    for name, key in _NL_ORDER:
        if name not in want_nl:
            continue
        value = want_nl[name]
        if name in cur_nl and cur_nl[name] == value:
            continue                      # "only set when it differs"
        cmds.append(gset_argv(key, value))
        nl_changed = True

    output = str(current.get("output") or "")
    cur_b = current.get("brightness")
    want_b = target.get("brightness")
    if want_b is not None and output:
        want_b = clamp_brightness(want_b)
        if cur_b is None or abs(float(cur_b) - want_b) >= 0.005:
            cmds.append(brightness_argv(output, want_b))
        elif nl_changed and want_b < MAX_BRIGHTNESS:
            cmds.append(brightness_argv(output, want_b))     # re-assert
    elif nl_changed and output and cur_b is not None and \
            float(cur_b) < MAX_BRIGHTNESS:
        cmds.append(brightness_argv(output, cur_b))          # re-assert
    return cmds


def _run(argv: list[str], timeout: float = GSETTINGS_TIMEOUT_S):
    """The one subprocess seam (tests replace it with a recorder).

    ``JARVIS_ROOM_CONTROL=0`` (see mixer.blocked) suppresses every real
    call: the suite builds the REAL app, and nothing in it may move the
    user's live desktop. It is his kill switch too."""
    from jarvis.mixer import blocked
    if blocked():
        log.debug("room: %s suppressed (JARVIS_ROOM_CONTROL)", argv[:2])
        return subprocess.CompletedProcess(argv, 1, "", "room control off")
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------- room light
class RoomLight:
    """Display brightness and colour temperature, reversibly.

    Every mutation remembers the state that was in force before the FIRST
    one (the baseline) in an atomic state file, so ``restore()`` -- at
    "lights up", at boot, at quit, and on any failure -- puts the room back
    exactly as he had it."""

    def __init__(self, run: Optional[Callable] = None,
                 state_path: Optional[Path] = None,
                 now: Callable[[], float] = time.time,
                 held_by: Optional[Callable[[], bool]] = None):
        self._run = run or _run
        self._state_path = Path(state_path) if state_path else None
        self._now = now
        # "Is somebody else deliberately holding this display?" -- the
        # bedtime wind-down (jarvis/winddown.py) dims the SAME xrandr output
        # through its own state file, and the automatic heals below (boot
        # and quit) would otherwise drive the brightness straight back up at
        # two in the morning, undoing a guard winddown.restore() had just
        # honoured. Only the healing paths consult it; "lights up" is a
        # deliberate request and always wins.
        self._held_by = held_by
        self._lock = threading.RLock()
        self._output = ""

    # ------------------------------------------------------------- probe
    def _cmd(self, argv: list[str]) -> Optional[str]:
        try:
            out = self._run(argv)
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("room: %s failed: %s", " ".join(argv[:2]), exc)
            return None
        if getattr(out, "returncode", 1) != 0:
            log.debug("room: %s rc=%s: %s", " ".join(argv[:2]),
                      getattr(out, "returncode", None),
                      (getattr(out, "stderr", "") or "").strip()[:120])
            return None
        return getattr(out, "stdout", "") or ""

    def output(self) -> str:
        """The connected primary output, discovered once per instance."""
        if self._output:
            return self._output
        text = self._cmd(["xrandr", "--query"])
        self._output = primary_output(text or "")
        if not self._output:
            log.info("room: no connected output; the light knobs stand down")
        return self._output

    def snapshot(self) -> Optional[dict]:
        """What the display is doing right now, or None with no display."""
        output = self.output()
        if not output:
            return None
        verbose = self._cmd(["xrandr", "--verbose"])
        brightness = parse_brightness(verbose or "", output)
        night = {}
        for name, key in _NL_ORDER:
            value = self._cmd(gget_argv(key))
            if value is None:
                continue
            night[name] = parse_gvalue(value)
        return {"output": output, "brightness": brightness,
                "night_light": night}

    # ------------------------------------------------------------- state
    def baseline(self) -> Optional[dict]:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict) and data.get("output"):
                    return data
        except (OSError, ValueError):
            log.debug("room state unreadable", exc_info=True)
        return None

    def _remember(self, snap: dict) -> None:
        """First change only: the baseline is what he had before Jarvis
        touched anything, not the last step on the way down."""
        if not self._state_path or self.baseline() is not None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            payload = dict(snap)
            payload["t"] = self._now()
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self._state_path)   # atomic: never half a file
        except OSError:
            log.debug("room state save failed", exc_info=True)

    def _forget(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.unlink(missing_ok=True)
        except OSError:
            log.debug("room state clear failed", exc_info=True)

    @property
    def changed(self) -> bool:
        """True while Jarvis is holding the display away from his own."""
        return self.baseline() is not None

    # ------------------------------------------------------------- apply
    def apply(self, brightness=None, night_light: Optional[dict] = None,
              snap: Optional[dict] = None) -> Optional[dict]:
        """Move the light. Returns the state now in force, or None when
        there is no display or a command failed (in which case the room has
        already been put back)."""
        with self._lock:
            snap = snap if snap is not None else self.snapshot()
            if snap is None:
                return None
            target = {}
            if brightness is not None:
                target["brightness"] = clamp_brightness(brightness)
            if night_light:
                target["night_light"] = dict(night_light)
            cmds = plan_light(snap, target)
            if not cmds:
                return snap
            self._remember(snap)
            for argv in cmds:
                if self._cmd(argv) is None:
                    log.warning("room: %s failed; restoring", " ".join(argv[:3]))
                    self.restore()
                    return None
            now = {"output": snap["output"],
                   "brightness": target.get("brightness", snap.get("brightness")),
                   "night_light": {**(snap.get("night_light") or {}),
                                   **(target.get("night_light") or {})}}
            log.info("room: brightness %s, night light %s",
                     now["brightness"], now["night_light"].get("enabled"))
            return now

    # ----------------------------------------------------------- verbs
    def dim(self, steps: int = 1) -> tuple[Optional[dict], str]:
        return self._step_brightness(-abs(steps))

    def brighten(self, steps: int = 1) -> tuple[Optional[dict], str]:
        return self._step_brightness(abs(steps))

    def _step_brightness(self, steps: int) -> tuple[Optional[dict], str]:
        snap = self.snapshot()
        if snap is None:
            return None, NO_DISPLAY_LINE
        current = snap.get("brightness")
        current = MAX_BRIGHTNESS if current is None else current
        want = step_brightness(current, steps)
        if abs(want - clamp_brightness(current)) < 0.005:
            return snap, AT_FLOOR_LINE if steps < 0 else AT_FULL_LINE
        state = self.apply(brightness=want, snap=snap)
        if state is None:
            return None, FAILED_LINE
        # Back at full with nothing else held: stop holding the baseline.
        if want >= MAX_BRIGHTNESS and not self._night_light_held(state):
            self._forget()
        return state, DIM_LINE if steps < 0 else BRIGHTEN_LINE

    def warmer(self, steps: int = 1) -> tuple[Optional[dict], str]:
        return self._step_kelvin(-abs(steps))

    def cooler(self, steps: int = 1) -> tuple[Optional[dict], str]:
        return self._step_kelvin(abs(steps))

    def _step_kelvin(self, steps: int) -> tuple[Optional[dict], str]:
        snap = self.snapshot()
        if snap is None:
            return None, NO_DISPLAY_LINE
        night = snap.get("night_light") or {}
        # Night light off means the screen is at daylight, whatever
        # temperature the disabled key happens to remember.
        current = clamp_kelvin(night.get("temperature", WARM_KELVIN)) \
            if night.get("enabled") else NEUTRAL_KELVIN
        want = step_kelvin(current, steps)
        if want == current:
            return snap, AT_WARMEST_LINE if steps < 0 else AT_COOLEST_LINE
        if want >= NEUTRAL_KELVIN:
            state = self.apply(night_light=night_light_target(False), snap=snap)
            line = COOL_LINE
        else:
            state = self.apply(night_light=night_light_target(True, want),
                               snap=snap)
            line = WARM_LINE if steps < 0 else COOL_LINE
        if state is None:
            return None, FAILED_LINE
        if want >= NEUTRAL_KELVIN and \
                clamp_brightness(state.get("brightness") or MAX_BRIGHTNESS) >= MAX_BRIGHTNESS:
            self._forget()
        return state, line

    def _night_light_held(self, state: dict) -> bool:
        base = self.baseline()
        if base is None:
            return False
        was = (base.get("night_light") or {}).get("enabled")
        now = (state.get("night_light") or {}).get("enabled")
        return bool(now) != bool(was)

    # ----------------------------------------------------------- restore
    def held_elsewhere(self) -> bool:
        """True while another module is deliberately holding the display.

        Reads the ``held_by`` seam defensively: a probe that raises must
        never stop the light coming back."""
        probe = self._held_by
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 - a broken probe must not hold him
            log.debug("room: display-hold probe failed", exc_info=True)
            return False

    def restore(self, healing: bool = False) -> tuple[bool, str]:
        """Put back what he had before Jarvis touched the light.

        Called at "lights up", at boot (so a crash at 0.55 heals), at quit,
        and by ``apply`` when a command fails.

        ``healing`` marks the two AUTOMATIC calls (boot and quit). Those
        stand down while the bedtime wind-down still holds the room: a
        restart at two in the morning must not light it back up, which is
        the very decision ``winddown.restore(expired_only=True)`` had just
        made one screenful earlier in ``start_assistant``."""
        with self._lock:
            base = self.baseline()
            if base is None:
                return True, NOTHING_TO_RESTORE_LINE
            if healing and self.held_elsewhere():
                log.info("room: the wind-down still holds the display; "
                         "leaving the baseline for the morning")
                return False, NOTHING_TO_RESTORE_LINE
            output = str(base.get("output") or "") or self.output()
            night = base.get("night_light") or {}
            ok = True
            # Same order as plan_light, and for the same reason: gsd-color
            # rewrites the ramp, so brightness is written after it and last.
            for name, key in _NL_ORDER:
                if name not in night:
                    continue
                if self._cmd(gset_argv(key, night[name])) is None:
                    ok = False
            brightness = base.get("brightness")
            if output and brightness is not None:
                if self._cmd(brightness_argv(output, brightness)) is None:
                    ok = False
            # Keep the baseline when the put-back did not take, and stop
            # claiming it did. Forgetting here used to destroy the ONLY
            # record of his brightness the moment a restore ran against a
            # dead X -- and every retry path (the boot heal, the quit heal,
            # "lights up", scenes.restore) gates on `changed`, so the screen
            # stayed held dim with nothing left to put it back.
            if ok:
                self._forget()
            log.info("room: display %s", "restored" if ok else
                     "NOT restored; keeping the baseline to retry")
            return ok, RESTORED_LINE if ok else FAILED_RESTORE_LINE
