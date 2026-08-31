"""Bedtime wind-down: "good night" makes the room agree with him.

"Good night, sir. I'll be here." was the whole of it -- a line, and a
briefing preview when briefings are on. This turns it into something
physical, on the three things Jarvis can already reach without sudo:

* Spotify fades to nothing over `fade_s` and then pauses (the volume is put
  straight back, so a manual play at three in the morning is not silent);
* the screen warms (GNOME night light) and dims (`xrandr --brightness`);
* do-not-disturb is armed until quiet hours close, so nothing proactive is
  spoken into a dark room.

"Good morning" -- and, failing that, the next app start after the window --
puts all of it back.

**Everything is reversible and nothing is guessed.** The state that will be
restored is snapshotted and written to disk BEFORE the first thing changes,
so a crash mid-fade still restores; any failure in the desktop half restores
immediately rather than leaving a man with a black screen; and a second
"good night" while a wind-down is live is a no-op, because snapshotting then
would record the DIMMED screen as the brightness to go back to. When all
else fails there is a door from the outside::

    ~/vss_env/bin/python -m jarvis.winddown --restore

Off by default (`wind_down.enabled`): a box that never asked for this gets
the courtesy line and nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from jarvis.logs import get_logger

log = get_logger("winddown")

DEFAULT_FADE_S = 60.0
DEFAULT_BRIGHTNESS = 0.5
DEFAULT_MORNING = "07:00"        # when quiet hours are not configured
STEP_S = 5.0                     # one Spotify volume write per step
# xrandr accepts 0.0, which is a black screen with no way back except this
# module. The floor is what a dimmed screen means, not what it allows.
MIN_BRIGHTNESS = 0.2
MAX_STATE_AGE_S = 36 * 3600      # older than this and "until" is not trusted

NIGHT_LIGHT_SCHEMA = "org.gnome.settings-daemon.plugins.color"
NIGHT_LIGHT_KEY = "night-light-enabled"

# "HDMI-0 connected primary 3840x2160+0+0 ..." then, further down its block,
# "\tBrightness: 1.0". Verified against xrandr 1.5 on this box's :1.
_OUTPUT_RX = re.compile(r"^(\S+)\s+connected\b")
_BRIGHTNESS_RX = re.compile(r"^\s*Brightness:\s*([0-9.]+)\s*$")


def _run(argv, timeout: float = 5.0):
    """One subprocess -> (ok, stdout). The desktop seam; tests replace it."""
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:              # noqa: BLE001 - missing binary, X gone
        log.info("wind-down: %s failed: %s", argv[0], exc)
        return False, ""
    if res.returncode != 0:
        log.info("wind-down: %s exited %d: %.120s", argv[0], res.returncode,
                 (res.stderr or "").strip())
    return res.returncode == 0, res.stdout or ""


def screen_brightness() -> dict:
    """{output: brightness} for every CONNECTED output, from xrandr."""
    ok, out = _run(["xrandr", "--verbose"], timeout=8.0)
    if not ok:
        return {}
    found, current = {}, ""
    for line in out.splitlines():
        m = _OUTPUT_RX.match(line)
        if m:
            current = m.group(1)
            continue
        if current:
            m = _BRIGHTNESS_RX.match(line)
            if m:
                try:
                    found[current] = float(m.group(1))
                except ValueError:
                    pass
                current = ""
    return found


def set_brightness(output: str, value: float) -> bool:
    ok, _ = _run(["xrandr", "--output", output, "--brightness", f"{value:.2f}"])
    return ok


def night_light() -> Optional[bool]:
    """The current night-light setting, or None when gsettings cannot say."""
    ok, out = _run(["gsettings", "get", NIGHT_LIGHT_SCHEMA, NIGHT_LIGHT_KEY])
    if not ok:
        return None
    val = out.strip().lower()
    return True if val == "true" else False if val == "false" else None


def set_night_light(on: bool) -> bool:
    ok, _ = _run(["gsettings", "set", NIGHT_LIGHT_SCHEMA, NIGHT_LIGHT_KEY,
                  "true" if on else "false"])
    return ok


def _clock_seconds(now_ts: float, hhmm: str) -> float:
    """Seconds from `now_ts` to the next "HH:MM" (tomorrow if it has gone)."""
    try:
        hh, mm = (int(x) for x in str(hhmm).split(":")[:2])
    except (TypeError, ValueError):
        hh, mm = 7, 0
    dt = datetime.fromtimestamp(now_ts).astimezone()
    end = dt.replace(hour=hh % 24, minute=mm % 60, second=0, microsecond=0)
    if end <= dt:
        end += timedelta(days=1)
    return max(60.0, end.timestamp() - now_ts)


class WindDown:
    """The choreography. One instance, owned by the app."""

    def __init__(self, services, state_path=None, now=None):
        self.services = services
        self._state_path = Path(state_path) if state_path else None
        self._now = now or time.time
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ config
    def _svc(self, name):
        return getattr(self.services, name, None)

    def _cfg(self, key: str, default):
        cfg = self._svc("assistant")
        if cfg is None:
            return default
        try:
            val = cfg.get(f"wind_down.{key}", default)
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("wind_down.%s unreadable", key, exc_info=True)
            return default
        return default if val is None else val

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("enabled", False))

    @property
    def active(self) -> bool:
        return self._read_state() is not None

    # ------------------------------------------------------------- state
    def _read_state(self) -> Optional[dict]:
        if not self._state_path:
            return None
        try:
            data = json.loads(self._state_path.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _write_state(self, state: dict) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.warning("wind-down state could not be saved; not dimming",
                        exc_info=True)
            raise

    def _clear_state(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.unlink()
        except OSError:
            pass

    # -------------------------------------------------------- the wind-down
    def start(self) -> bool:
        """"Good night": fade / dim / arm, all on a worker thread.

        Nothing blocking happens here -- reading the screen costs an xrandr
        and the Spotify device costs a round trip, and the good-night line
        should not wait for either. False when it is off or already wound
        down."""
        if not self.enabled:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._read_state() is not None:
                # A second "good night". Snapshotting now would record the
                # DIMMED screen as the brightness to restore -- the one way
                # this feature could leave him in the dark for good.
                log.info("wind-down: already wound down; nothing to do")
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._wind, daemon=True,
                                            name="winddown")
            self._thread.start()
        return True

    def _snapshot(self) -> dict:
        now = self._now()
        state = {"at": now, "until": now + self._dnd_seconds(now),
                 "brightness": screen_brightness(), "night_light": night_light(),
                 "volume": None}
        spotify = self._svc("spotify")
        if spotify is not None and bool(self._cfg("music", True)):
            try:
                dev = spotify.resolve_device(prefer_active=True)
                state["volume"] = dev.volume
                state["device"] = dev.name
            except Exception as exc:      # noqa: BLE001 - SpotifyError or worse
                log.info("wind-down: no Spotify device (%s)",
                         getattr(exc, "text", exc))
        log.info("wind-down: snapshot %s", state)
        return state

    def _wind(self) -> None:
        """Snapshot, record, then act -- in that order, so a crash between
        any two steps still leaves a file that says how to undo them.

        Do-not-disturb first (it is instant and it is the half that must not
        wait for a fade), then the screen, then the slow music fade."""
        if self._stop.is_set():
            return                        # the app quit between start and here
        try:
            state = self._snapshot()
            self._write_state(state)
        except Exception:                 # noqa: BLE001 - no record, no changes
            log.exception("wind-down: nothing could be saved; not touching "
                          "the desktop")
            return
        try:
            self._arm_dnd(state)
            self._dim(state)
            self._fade_music(state)
        except Exception:                 # noqa: BLE001 - never leave it dark
            log.exception("wind-down failed; putting the desktop back")
            self.restore()

    # ------------------------------------------------------------- music
    def _fade_music(self, state: dict) -> None:
        """Step the Spotify volume to zero, pause, and put the volume back.

        The volume is restored immediately rather than in the morning: the
        wind-down's lasting effect is the PAUSE, and a device left at zero
        is indistinguishable from a broken speaker to anyone who presses
        play in the night. Every SpotifyError is swallowed, as in focus.py:
        a night without music is not worth an apology."""
        start = state.get("volume")
        spotify = self._svc("spotify")
        if spotify is None or start is None or not bool(self._cfg("music", True)):
            return
        try:
            fade_s = max(0.0, float(self._cfg("fade_s", DEFAULT_FADE_S)))
        except (TypeError, ValueError):
            fade_s = DEFAULT_FADE_S
        steps = max(1, int(round(fade_s / STEP_S)))
        start = max(0, min(100, int(start)))
        try:
            for i in range(1, steps + 1):
                if self._stop.wait(fade_s / steps):
                    return                # stopping the app abandons the fade
                spotify.control("volume", int(round(start * (1 - i / steps))))
            spotify.control("pause")
            spotify.control("volume", start)
            log.info("wind-down: music faded from %d%% over %.0fs and paused",
                     start, fade_s)
        except Exception as exc:          # noqa: BLE001 - SpotifyError or worse
            log.info("wind-down: music skipped: %s", getattr(exc, "text", exc))

    # ------------------------------------------------------------ screen
    def _dim(self, state: dict) -> None:
        """Night light on, brightness down -- and the caller restores if
        either half fails, so a screen is never left half-dimmed."""
        if bool(self._cfg("night_light", True)) and state.get("night_light") is not None:
            if not set_night_light(True):
                raise RuntimeError("night light could not be set")
        try:
            level = float(self._cfg("brightness", DEFAULT_BRIGHTNESS))
        except (TypeError, ValueError):
            level = DEFAULT_BRIGHTNESS
        level = max(MIN_BRIGHTNESS, min(1.0, level))
        for output in state.get("brightness") or {}:
            if not set_brightness(output, level):
                raise RuntimeError(f"brightness could not be set on {output}")
        log.info("wind-down: screen at %.2f on %s", level,
                 ", ".join(state.get("brightness") or {}) or "no output")

    # --------------------------------------------------------------- dnd
    def _dnd_seconds(self, now_ts: float) -> float:
        """How long to hold his tongue: until quiet hours close when they are
        configured (the same window the digest is read back at), else until
        `wind_down.morning`."""
        quiet = self._svc("quiet")
        if quiet is not None:
            try:
                end = quiet.hours_end(now_ts)
                if end:
                    return max(60.0, float(end) - now_ts)
            except Exception:             # noqa: BLE001 - policy boundary
                log.debug("wind-down: quiet hours unreadable", exc_info=True)
        return _clock_seconds(now_ts, self._cfg("morning", DEFAULT_MORNING))

    def _arm_dnd(self, state: dict) -> None:
        if not bool(self._cfg("dnd", True)):
            return
        quiet = self._svc("quiet")
        if quiet is None or not hasattr(quiet, "set_dnd"):
            return
        try:
            seconds = max(60.0, float(state.get("until", 0)) - self._now())
            quiet.set_dnd(seconds)
        except Exception:                 # noqa: BLE001 - policy boundary
            log.exception("wind-down: do-not-disturb could not be armed")

    # ----------------------------------------------------------- restore
    def restore(self, expired_only: bool = False) -> bool:
        """Put the screen (and the volume) back. Idempotent: no state means
        nothing to undo. `expired_only` is app startup's -- a restart at two
        in the morning must not brighten the room, but a restart after the
        window, or with a state file from some forgotten night, must."""
        with self._lock:
            state = self._read_state()
            if state is None:
                return False
            now = self._now()
            until = float(state.get("until") or 0)
            stale = now - float(state.get("at") or 0) > MAX_STATE_AGE_S
            if expired_only and now < until and not stale:
                log.info("wind-down: still inside its window; leaving the "
                         "screen as it is")
                return False
            for output, value in (state.get("brightness") or {}).items():
                try:
                    set_brightness(output, max(0.0, min(1.0, float(value))))
                except (TypeError, ValueError):
                    set_brightness(output, 1.0)
            was = state.get("night_light")
            if was is not None and bool(self._cfg("night_light", True)):
                set_night_light(bool(was))
            volume = state.get("volume")
            spotify = self._svc("spotify")
            if spotify is not None and volume is not None:
                try:
                    spotify.control("volume", int(volume))
                except Exception as exc:  # noqa: BLE001 - SpotifyError or worse
                    log.info("wind-down: volume not restored: %s",
                             getattr(exc, "text", exc))
            self._clear_state()
        log.info("wind-down: restored")
        return True

    # -------------------------------------------------------- lifecycle
    def stop(self) -> None:
        """App shutdown. The fade is abandoned; the screen is NOT restored
        (a quit at midnight must not light the room), and the state file
        stays so the next start or "good morning" puts it back."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)


def main(argv=None) -> int:
    """`python -m jarvis.winddown --restore`: the door from the outside,
    for a screen left dim by a crash with no app to ask."""
    ap = argparse.ArgumentParser(prog="jarvis.winddown", description=(
        "put back the screen a wind-down dimmed"))
    ap.add_argument("--restore", action="store_true", help="restore now")
    ap.add_argument("--status", action="store_true", help="print the state")
    args = ap.parse_args(argv)
    from types import SimpleNamespace

    from jarvis.config import PATHS
    wd = WindDown(SimpleNamespace(), state_path=PATHS.MEMORY_DIR / "winddown.json")
    state = wd._read_state()
    if args.restore:
        # No config here, so night_light() decides for itself and the
        # brightness in the file is authoritative.
        print("restored" if wd.restore() else "nothing to restore")
        return 0
    print(json.dumps(state, indent=2) if state else "no wind-down is active")
    print(f"screen: {screen_brightness()}  night light: {night_light()}")
    if not args.status:
        ap.print_usage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
