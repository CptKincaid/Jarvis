"""The sink sentinel: knowing which speaker Jarvis is actually talking into.

2026-08-30, the night this was written: the Bluetooth soundbar's battery
died mid-evening.  PipeWire did what PipeWire does -- it moved the default
sink to the HDMI monitor -- and Jarvis carried on speaking, for hours, into
a monitor behind the desk.  Nothing in the app noticed, and the only signal
Hunter got was mystery noise from the wrong side of the room.

So this thread answers one question every ``interval_s``: **where is my
voice landing, and is that where it should be?**

* ``pactl list short sinks`` is the roster, ``pactl get-default-sink`` the
  destination.  Speech lands on ``config.playback_device`` when that is
  set, otherwise on the default sink -- exactly the rule tts._play uses.
* The PREFERRED sink is ``audio.preferred_sink`` from assistant.json when
  set (an exact name or any substring of one, so "soundbar" or a MAC
  fragment both work), and otherwise the Bluetooth sink this box has seen,
  remembered in the state file.  Learning is silent: a sink appearing is
  not news, a sink *vanishing* is.
* One line per transition, never per tick: "I'm coming out of the monitor,
  sir; the soundbar has dropped", and one when it comes back.  The state
  file carries the latch, so a restart into a still-dead soundbar does not
  say it again -- the same rule jarvis/faults.py has about telling him
  twice.

**The one thing it may change.**  ``pactl set-default-sink <preferred>``,
and only ever back to the preferred sink, only when that sink is really in
the roster, only when ``audio.restore_sink`` is on (it is OFF by default),
and never while Jarvis is mid-burst -- moving the default out from under a
playing paplay is how you get half a sentence in each speaker.  The sink it
moved away from is recorded in the state file (``restored_from``) so the
change is reversible by hand, and it is skipped entirely when
``config.playback_device`` pins an explicit device: with a pinned device
the default sink is not where his voice goes, so moving it would be a
change to his desktop that buys Jarvis nothing.

Everything else here is read-only, and a box with no ``pactl`` at all goes
quiet rather than guessing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
# One definition of "a sink that is not really an output": voice_check's
# --sinks probe already had to know that auto_null means speech into
# silence, and two lists that drift would disagree about a dead box.
from jarvis.voice_check import DUMMY_SINK_NAMES

log = get_logger("soundbar")

INTERVAL_S = 30.0              # cheap: two short pactl calls
FIRST_TICK_S = 45.0            # PipeWire is still enumerating bluez at boot
PACTL_TIMEOUT_S = 5.0
STATE_VERSION = 1

# What the sink NAME says the thing is. PipeWire's names are stable enough
# to read: bluez_output.* is a Bluetooth device, alsa_output.*hdmi* is the
# monitor, and a USB DAC carries "usb" in the ALSA id.
BLUETOOTH, HDMI, USB, ANALOG, DUMMY, OTHER = (
    "bluetooth", "hdmi", "usb", "analog", "dummy", "other")

# The spoken name for each kind. "the soundbar" is his: the only Bluetooth
# sink on this box is the soundbar, and "the Bluetooth output" is not how
# anyone says it out loud.
WORDS = {BLUETOOTH: "the soundbar", HDMI: "the monitor", USB: "the USB speakers",
         ANALOG: "the speakers", DUMMY: "a dead output", OTHER: "another output"}

DROP_LINE = "I'm coming out of {now}, sir; {preferred} has dropped."
DROP_LINE_NO_TARGET = "{Preferred} has dropped, sir; I've nowhere to speak from."
BACK_LINE = "{Preferred} is back, sir; I'm still coming out of {now}."
BACK_MOVED_LINE = "{Preferred} is back, sir; I've moved my voice across."
DEAD_LINE = "I've no working audio output at all, sir."
OK_LINE = "I'm coming out of {now}, sir."
OK_NAMED_LINE = "I'm coming out of {now}, sir, which is where I should be."
BLIND_LINE = "I can't see this machine's audio devices, sir."
PERSONA_LINES = [DEAD_LINE, BLIND_LINE]

_SPACE_RX = re.compile(r"\s+")


def _head(words: str) -> str:
    """"the soundbar" -> "The soundbar": a sentence opener, without
    str.capitalize() flattening the rest of the phrase."""
    words = str(words or "")
    return (words[0].upper() + words[1:]) if words else words


# ------------------------------------------------------------------ pure
def parse_sinks(text: str) -> list[dict]:
    """``pactl list short sinks`` -> one dict per sink.

    voice_check.parse_sinks answers a different question (is this box deaf
    at all?) and throws the per-sink state away; the sentinel needs the
    rows themselves, so this is a second parser rather than a rework of a
    diagnostic every other probe depends on.
    """
    out: list[dict] = []
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        if not name:
            continue
        try:
            index = int(parts[0].strip())
        except (TypeError, ValueError):
            index = -1
        out.append({"index": index, "name": name,
                    "driver": parts[2].strip() if len(parts) > 2 else "",
                    "state": parts[4].strip() if len(parts) > 4 else ""})
    return out


def classify(name) -> str:
    low = str(name or "").strip().lower()
    if not low:
        return OTHER
    if any(d in low for d in DUMMY_SINK_NAMES):
        return DUMMY
    if low.startswith("bluez") or "bluez" in low or ".bluetooth" in low:
        return BLUETOOTH
    if "hdmi" in low or "displayport" in low:
        return HDMI
    if "usb" in low:
        return USB
    if low.startswith("alsa_output") or "analog" in low:
        return ANALOG
    return OTHER


def describe(name) -> str:
    """The words for a sink, for the spoken line. Never the raw PipeWire
    name: "alsa_output.platform-NVDA2014_00.hdmi-stereo" is unspeakable."""
    if not str(name or "").strip():
        return "nothing"
    return WORDS.get(classify(name), WORDS[OTHER])


def match_sink(names, wanted) -> str:
    """The sink ``wanted`` names, or "".

    Exact first, then case-folded, then substring either way -- so
    ``audio.preferred_sink`` can be the full PipeWire name, "soundbar", or
    a fragment of the MAC, and a soundbar whose sink index changes between
    pairings still resolves.
    """
    want = _SPACE_RX.sub("", str(wanted or "")).lower()
    if not want:
        return ""
    names = [str(n or "") for n in (names or ())]
    for name in names:
        if name == wanted:
            return name
    for name in names:
        if name.lower() == want:
            return name
    for name in names:
        low = name.lower()
        if want in low or low in want:
            return name
    return ""


def real_sinks(sinks) -> list[dict]:
    """The sinks speech could actually come out of: null sinks are not
    outputs, they are the shape of a box with nothing plugged in."""
    return [s for s in (sinks or []) if classify(s.get("name")) != DUMMY]


def _cfg(cfg, dotted: str, default):
    if cfg is None:
        return default
    try:
        value = cfg.get(dotted, default)
    except Exception:                          # noqa: BLE001 - config boundary
        return default
    return default if value is None else value


def _run_pactl(argv, timeout: float = PACTL_TIMEOUT_S):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


class SoundbarSentinel:
    """Watches the sinks; speaks once per transition; moves the default
    sink back only when told it may.

    Seams: ``run`` (the ONE subprocess call), ``say``, ``speaking``,
    ``playback_device``, ``now`` and ``state_path``. Tests drive a fake
    runner -- nothing here may touch a live PipeWire from the suite.
    """

    def __init__(self, cfg=None, say: Optional[Callable] = None, quiet=None,
                 run: Optional[Callable] = None, now: Optional[Callable] = None,
                 state_path: Optional[Path] = None,
                 playback_device: Optional[Callable] = None,
                 speaking: Optional[Callable] = None):
        self._cfg = cfg
        self._say = say
        self._quiet = quiet
        self._run = run or _run_pactl
        self._now = now or datetime.now
        self._state_path = Path(state_path) if state_path else None
        self._playback_device = playback_device
        self._speaking = speaking
        self._lock = threading.RLock()
        self._state = self._load()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # The last tick's reading, for status_line(): the spoken answer must
        # never fork a subprocess on the reply path (jarvis/aside.py's rule).
        self._seen: dict = {}

    # ------------------------------------------------------------- config
    @property
    def enabled(self) -> bool:
        return bool(_cfg(self._cfg, "audio.sink_watch", True))

    @property
    def interval_s(self) -> float:
        try:
            return max(5.0, float(_cfg(self._cfg, "audio.sink_poll_s", INTERVAL_S)))
        except (TypeError, ValueError):
            return INTERVAL_S

    @property
    def restore_enabled(self) -> bool:
        """OFF by default, and deliberately so: set-default-sink changes HIS
        desktop, not just Jarvis's voice."""
        return bool(_cfg(self._cfg, "audio.restore_sink", False))

    @property
    def configured_preferred(self) -> str:
        return " ".join(str(_cfg(self._cfg, "audio.preferred_sink", "") or "").split())

    # -------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict):
                    return data
        except (OSError, ValueError):
            log.debug("soundbar state unreadable", exc_info=True)
        return {"version": STATE_VERSION}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._state, indent=1))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("soundbar state save failed", exc_info=True)

    # ------------------------------------------------------------- probes
    def _pactl(self, *args) -> Optional[str]:
        """stdout, or None when pactl could not answer. None is the honest
        "I don't know" -- it must never read as "there are no sinks"."""
        try:
            # The timeout is not optional: a wedged PipeWire must not hold
            # this thread past stop()'s 2 s join.
            proc = self._run(["pactl", *args], timeout=PACTL_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            log.debug("soundbar: pactl %s failed", args[0], exc_info=True)
            return None
        if getattr(proc, "returncode", 1) != 0:
            return None
        return str(getattr(proc, "stdout", "") or "")

    def sinks(self) -> Optional[list[dict]]:
        text = self._pactl("list", "short", "sinks")
        return None if text is None else parse_sinks(text)

    def default_sink(self) -> str:
        text = self._pactl("get-default-sink")
        if text is not None and text.strip():
            return text.strip().splitlines()[0].strip()
        # Older pactl has no get-default-sink; `pactl info` always has the
        # line, and a box that answers neither is one we say nothing about.
        info = self._pactl("info") or ""
        for line in info.splitlines():
            if line.lower().startswith("default sink:"):
                return line.split(":", 1)[1].strip()
        return ""

    def _pinned_device(self) -> str:
        """``config.playback_device`` -- the sink tts._play passes to paplay
        with --device. Read through a seam so the tests never import the
        live voice settings."""
        if self._playback_device is not None:
            try:
                return " ".join(str(self._playback_device() or "").split())
            except Exception:                      # noqa: BLE001
                log.debug("soundbar: playback device unreadable", exc_info=True)
                return ""
        try:
            from jarvis.config import CONFIG
            return " ".join(str(getattr(CONFIG, "playback_device", "") or "").split())
        except Exception:                          # noqa: BLE001
            return ""

    def _busy(self) -> bool:
        try:
            return bool(self._speaking()) if callable(self._speaking) else False
        except Exception:                          # noqa: BLE001
            log.debug("soundbar: speaking probe failed", exc_info=True)
            return False

    # ---------------------------------------------------------- reasoning
    def preferred(self, names) -> tuple[str, bool]:
        """(the preferred sink's identity, present) for this roster.

        The identity is the live sink name when the preference resolves to
        one, otherwise the remembered/configured string -- an absent
        soundbar has to keep its name, or its absence could not be noticed
        at all.
        """
        names = [str(n or "") for n in (names or ())]
        want = self.configured_preferred
        if want:
            hit = match_sink(names, want)
            return (hit or want), bool(hit)
        remembered = " ".join(str(self._state.get("preferred") or "").split())
        if remembered:
            hit = match_sink(names, remembered)
            return (hit or remembered), bool(hit)
        # Nothing configured and nothing learned: the Bluetooth sink, if
        # this box has one. Only Bluetooth -- a monitor does not "drop",
        # and learning the monitor would make every headphone unplug a
        # false alarm.
        for name in names:
            if classify(name) == BLUETOOTH:
                return name, True
        return "", False

    def _learn(self, name: str) -> None:
        if not name or self.configured_preferred:
            return                                 # config wins; never overwritten
        with self._lock:
            if self._state.get("preferred") == name:
                return
            self._state["preferred"] = name
            self._state["learned_at"] = self._now().isoformat(timespec="seconds")
            self._save()
        log.info("soundbar: learned %r as the preferred sink", name)

    def read(self) -> dict:
        """One reading of the audio world. ``status`` is one of:
        blind (no pactl) | dead (no real sink) | missing (the preferred sink
        is gone) | elsewhere (it is back but speech is not on it) | ok."""
        sinks = self.sinks()
        if sinks is None:
            return {"status": "blind", "sinks": [], "target": "", "preferred": "",
                    "default": ""}
        names = [s["name"] for s in sinks]
        default = self.default_sink()
        pinned = self._pinned_device()
        # tts._play: --device when pinned, the default sink otherwise. A
        # pinned device that is not in the roster is landing nowhere, which
        # reads as "missing" below exactly like a vanished preferred sink.
        target = match_sink(names, pinned) if pinned else default
        pref, present = self.preferred(names)
        if present and pref:
            self._learn(pref)
        reading = {"status": "ok", "sinks": names, "default": default,
                   "pinned": pinned, "target": target, "preferred": pref,
                   "preferred_present": present}
        if not real_sinks(sinks):
            reading["status"] = "dead"
        elif pref and not present:
            reading["status"] = "missing"
        elif pref and target and match_sink([target], pref) != target:
            reading["status"] = "elsewhere"
        elif not target:
            # Pinned at a device this box does not have: speech is falling
            # through paplay to aplay, or nowhere at all.
            reading["status"] = "missing" if pinned else "dead"
        return reading

    # ------------------------------------------------------------ speaking
    def _speak(self, line: str) -> None:
        if not line:
            return
        if not callable(self._say):
            log.info("soundbar: %s", line)
            return
        try:
            # proactive: quiet hours hold it for the digest like any other
            # line he did not ask for. A dead speaker at 3 am is not urgent
            # -- he is not listening to it either way.
            self._say(line, proactive=True, kind="warning")
        except Exception:                          # noqa: BLE001 - the tick survives TTS
            log.exception("soundbar: speak failed")

    def _restore(self, reading: dict) -> bool:
        """The one mutation. Every gate is a separate line on purpose: each
        one is a reason a reviewer (or Hunter, in the log) can check."""
        if not self.restore_enabled:
            return False
        if reading.get("pinned"):
            return False                           # the default is not where his voice goes
        pref = reading.get("preferred") or ""
        if not pref or not reading.get("preferred_present"):
            return False
        if reading.get("default") == pref:
            return False                           # already there
        if self._busy():
            # Mid-burst: switching the default under a playing paplay
            # splits the sentence across two speakers. It waits a tick.
            log.info("soundbar: holding the sink move; Jarvis is speaking")
            return False
        text = self._pactl("set-default-sink", pref)
        if text is None:
            log.warning("soundbar: set-default-sink %r failed", pref)
            return False
        with self._lock:
            # Reversible by hand: what it moved away from, and when.
            self._state["restored_from"] = reading.get("default") or ""
            self._state["restored_at"] = self._now().isoformat(timespec="seconds")
            self._save()
        log.info("soundbar: default sink moved %r -> %r",
                 reading.get("default") or "?", pref)
        return True

    def tick(self) -> dict:
        """One pass. Returns the reading, with ``spoken`` and ``moved``."""
        if not self.enabled:
            return {"status": "off", "spoken": "", "moved": False}
        reading = self.read()
        reading["spoken"], reading["moved"] = "", False
        self._seen = dict(reading)
        status = reading["status"]
        if status == "blind":
            # No pactl, or it refused to answer: say nothing at all. A
            # broken probe that talks is worse than a silent one.
            return reading
        with self._lock:
            previous = str(self._state.get("status") or "")
        if status in ("elsewhere", "ok") and previous in ("missing", "dead"):
            reading["moved"] = self._restore(reading)
            if reading["moved"]:
                reading["status"] = status = "ok"
                reading["default"] = reading["target"] = reading["preferred"]
        line = self._line_for(reading, previous)
        if line:
            self._speak(line)
            reading["spoken"] = line
        if status != previous:
            with self._lock:
                self._state["status"] = status
                self._state["at"] = self._now().isoformat(timespec="seconds")
                self._save()
            log.info("soundbar: %s (target %r, preferred %r)", status,
                     reading.get("target") or "-", reading.get("preferred") or "-")
        return reading

    def _line_for(self, reading: dict, previous: str) -> str:
        """The ONE line a transition earns, or "". Nothing is said for a
        state that has not changed, and nothing for the very first sight of
        a healthy box."""
        status = reading["status"]
        if status == previous:
            return ""
        pref = self._preferred_words(reading)
        now_words = describe(reading.get("target"))
        if status == "dead":
            return DEAD_LINE
        if status == "missing":
            if not reading.get("target"):
                return DROP_LINE_NO_TARGET.format(Preferred=_head(pref))
            return DROP_LINE.format(now=now_words, preferred=pref)
        if previous in ("missing", "dead"):
            if reading.get("moved"):
                return BACK_MOVED_LINE.format(Preferred=_head(pref))
            return BACK_LINE.format(Preferred=_head(pref), now=now_words)
        return ""                                  # ok <-> elsewhere is his own doing

    @staticmethod
    def _preferred_words(reading: dict) -> str:
        """What to CALL the speaker that should have his voice. Normally the
        preferred sink; with a pinned playback_device and no preference it is
        that device, so "nothing has dropped" can never be said."""
        pref = reading.get("preferred") or ""
        if pref:
            return describe(pref)
        pinned = reading.get("pinned") or ""
        return describe(pinned) if pinned else "the speaker I was told to use"

    # ------------------------------------------------------------- answer
    def status_line(self) -> str:
        """"Where's my audio going" -- from the last tick's reading, never
        a fresh subprocess: this runs inside a spoken turn."""
        reading = dict(self._seen)
        if not reading:
            reading = self.read() if self.enabled else {"status": "off"}
            self._seen = dict(reading)
        status = reading.get("status")
        if status in ("blind", "off", None):
            return BLIND_LINE
        if status == "dead":
            return DEAD_LINE
        now_words = describe(reading.get("target"))
        pref = self._preferred_words(reading)
        if status == "missing":
            if not reading.get("target"):
                return DROP_LINE_NO_TARGET.format(Preferred=_head(pref))
            return DROP_LINE.format(now=now_words, preferred=pref)
        if status == "elsewhere":
            return (f"I'm coming out of {now_words}, sir, though {pref} "
                    f"is available.")
        if reading.get("preferred"):
            return OK_NAMED_LINE.format(now=now_words)
        return OK_LINE.format(now=now_words)

    # ------------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again (tests, a config reload), and a dead thread must
        # not block a fresh one.
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="soundbar")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)                    # a pactl call is bounded at 5 s

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        if self._stop.wait(FIRST_TICK_S):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("soundbar tick failed")
            if self._stop.wait(self.interval_s):
                return


__all__ = ["SoundbarSentinel", "parse_sinks", "classify", "describe",
           "match_sink", "real_sinks", "BLIND_LINE", "DEAD_LINE"]
