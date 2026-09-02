"""The Room Mixer: every non-Jarvis stream bows under his voice.

On ``SpeakingState(active=True)`` and ``RecordingStarted`` every audio
stream on the box that is NOT Jarvis's own slides down to a floor
(``audio.duck_level``, 30 % by default) over ~200 ms, and slides back when
he stops talking and the capture ends.  So he can answer at conversational
volume while the music keeps playing, and -- the half that actually earns
its keep -- Whisper, the Silero endpointer and the ECAPA gate hear the
Snowball instead of the soundbar.

Two rules make this safe on THIS box, and both are load-bearing:

* **Never the sink.**  ``config.playback_device`` defaults to "", i.e. the
  default sink, which IS the bluez soundbar the music comes out of.
  Lowering the sink would lower Jarvis with it.  Only
  ``pactl set-sink-input-volume`` (per stream) is ever issued here; the
  master volume is left exactly where he put it.
* **Jarvis's own streams are known by PID, never by name.**  librespot
  pipes through ``pacat`` and so does ``paplay`` -- the live dump in
  ``tests/fixtures/pactl_sink_inputs.txt`` has BOTH, each with
  ``application.process.binary = "pacat"``.  Only the PID tells them
  apart, so tts.py and tools/timekeeper.py register the players they spawn
  (``register_own_pid``), and their descendants are exempt too.  The one
  exception is ``AEC_PLAYBACK_NAME``: the echo canceller's playback leg is
  his voice too, but it belongs to the pipewire process, so it can only be
  known by name.

The stream-restore hazard is why this module writes a state file.  A ducked
stream carries ``module-stream-restore.id =
"sink-input-by-application-name:pacat"``: PipeWire remembers the volume we
wrote *keyed by application name*, so a Jarvis killed mid-duck would leave
every FUTURE pacat stream at 30 % -- a librespot restarted tomorrow
included.  ``heal()`` therefore keeps unmatched entries in the file until
the stream they belong to reappears, instead of clearing them at boot.

When the music is on his phone or the HPCOMPUTER Connect target, pactl
cannot reach it.  That case is not rare -- it is his NORMAL one, and it is
why "he cant discern my voice from the vocalists in the music" (#72) sits
in the log right beside ``mixer: ducked 1 stream(s) to 30%``: the one
stream pactl could see was the idle librespot pipe, and the music he could
actually hear was coming out of HPCOMPUTER's own speakers.  So the Mixer
also asks the ``remote`` ducker (the Spotify tool -- ``duck()`` /
``unduck()`` over the Connect volume endpoint) on every hold, WHETHER OR
NOT a local stream was found.  It used to ask only when pactl found
nothing, and on 2026-09-01 that decided the whole incident: the librespot
pipe is an always-open, uncorked, SILENT sink-input (the Spark has never
been picked as a device), so pactl always found exactly one stream, ducked
it -- theatre -- and the remote duck never fired once.  The remote side
skips the case where the active device IS this box's librespot, so local
music is never ducked twice.  ``remote`` is None until something wires it,
and a remote that fails to DUCK is ignored -- but one that fails to RESTORE
is not: the duck stays on the books, the pump retries it with a backoff,
stop() forces one last attempt, and what is still down goes into the state
file so the next start can heal it.  That is the same protection the local
streams have had all along, and it became necessary the moment the remote
duck started firing on every hold instead of never.

``music_playing()`` is a pass-through to the remote's own cache, for the
wake gate in hotword.py: it relaxes its speaker threshold while music is
known to be playing, because a "Jarvis" said over a vocalist scores like a
stranger (0.135 and 0.158 against a 0.25 bar in that same log) and the guest
line answered him.  A duck of the mixer's own is deliberately NOT counted as
music -- it only proves Spotify still lists a device as active, which stays
true long after a pause, and counting it would relax that bar after every
turn in a silent room.

Seams for tests: ``run`` (the one subprocess call), ``sleep``, ``ppid_of``
and ``state_path``.  The parsing and planning halves are pure functions.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from jarvis.events import RecordingStarted, RecordingStopped, SpeakingState, bus
from jarvis.logs import get_logger

log = get_logger("mixer")

DEFAULT_FLOOR_PCT = 30
DEFAULT_RAMP_MS = 200
RAMP_STEPS = 4
# Streams already this quiet are left alone: ducking a whisper to 30 % of
# a whisper is noise, and every write skipped is one less stream-restore
# entry to heal later.
MIN_TOUCH_PCT = 5
# A stale duck older than this is abandoned rather than healed -- by then
# the volume he is looking at is one he set himself.
STALE_MAX_S = 7 * 24 * 3600.0
PACTL_TIMEOUT_S = 5.0
# THE BACKSTOP (2026-09-02).  A hold is a promise that something is going to
# lift it, and on 08:56:19 nothing did: the TTS amplitude feeder's last tick
# is published as SpeakingState(active=True) (jarvis/tts.py:1900) and drifted
# 325 ms PAST the worker's active=False, so the "speaking" hold was taken
# again with no falling edge left in the world.  His music sat at 30 % and
# the board read "Speaking" while he typed his answer.
#
# So: no hold may hold the room down longer than this without saying so.
# The "speaking" hold is refreshed by every one of its ~12 Hz ticks, so this
# measures SILENCE, not the length of a burst -- a half-hour read-aloud is
# never cut short, and a hold whose publisher has stopped talking to us is
# always let go.  The "recording" hold has no such heartbeat, so for it this
# is an absolute cap; it sits above the recorder's own watchdog
# (recorder.MAX_RECORDING_SECONDS + WATCHDOG_GRACE_S = 65 s), which should
# always get there first.
HOLD_MAX_S = 90.0
# The remote's version of the three numbers above.  A Connect device left
# down by a crash cannot be healed for free: every attempt is a Web API
# request, and a device that is switched off never answers.  So the remote
# record expires in an hour rather than a week (after that the slider he is
# looking at is one he has since touched himself) and is retried once a
# minute, which bounds the whole affair at 60 requests.
REMOTE_STALE_MAX_S = 3600.0
REMOTE_HEAL_RETRY_S = 60.0
# A restore that FAILS keeps the duck on the books and tries again, because
# the alternative is his phone stuck at 30 % of his volume with nothing left
# that remembers the other 70 %.  The worker wakes every second, so the
# retry backs off -- 5 s doubling to 5 min -- rather than hammering a device
# that is simply gone.
REMOTE_RETRY_S = 5.0
REMOTE_RETRY_MAX_S = 300.0
# The ONE exemption by name.  With echo cancellation on, Jarvis's own speech
# no longer reaches the sink from a paplay he spawned: it goes into the
# filter-chain's capture side and comes back out as a sink-input owned by the
# pipewire process, named as the chain declares it.  That PID is not his and
# never descends from him, so the registry cannot see it, and a PID-only
# rule would duck his own voice to 30 % under his own voice -- the failure
# this whole module exists to prevent.  Matched exactly against node.name,
# media.name and application.name, because which of the three carries the
# name depends on how the chain was declared.
AEC_PLAYBACK_NAME = "jarvis_aec_playback"

_INDEX_RX = re.compile(r"^Sink Input #(\d+)")
_PROP_RX = re.compile(r'^\s+([\w.-]+) = "(.*)"\s*$')
_PCT_RX = re.compile(r"(\d+)%")


# ------------------------------------------------------------------ pure
def parse_sink_inputs(text: str) -> list[dict]:
    """Parse ``pactl list sink-inputs`` into one dict per stream.

    Modelled on voice_check.parse_sinks: a pure parser the probes wrap.
    Unknown fields keep their empty default -- pactl's property block
    differs between a pacat pipe and a paplay file."""
    out: list[dict] = []
    cur: Optional[dict] = None
    for line in (text or "").splitlines():
        m = _INDEX_RX.match(line)
        if m:
            cur = {"index": int(m.group(1)), "sink": "", "corked": False,
                   "mute": False, "volume_pct": None, "app_name": "",
                   "media_name": "", "node_name": "", "pid": None,
                   "restore_key": ""}
            out.append(cur)
            continue
        if cur is None:
            continue
        stripped = line.strip()
        if stripped.startswith("Sink:"):
            cur["sink"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Corked:"):
            cur["corked"] = stripped.split(":", 1)[1].strip() == "yes"
        elif stripped.startswith("Mute:"):
            cur["mute"] = stripped.split(":", 1)[1].strip() == "yes"
        elif stripped.startswith("Volume:") and cur["volume_pct"] is None:
            pcts = [int(p) for p in _PCT_RX.findall(stripped)]
            if pcts:
                # The loudest channel: restoring that number keeps the
                # stream's balance where its owner set it.
                cur["volume_pct"] = max(pcts)
        else:
            pm = _PROP_RX.match(line)
            if pm is None:
                continue
            key, value = pm.group(1), pm.group(2)
            if key == "application.name":
                cur["app_name"] = value
            elif key == "media.name":
                cur["media_name"] = value
            elif key == "node.name":
                cur["node_name"] = value
            elif key == "module-stream-restore.id":
                cur["restore_key"] = value
            elif key == "application.process.id":
                try:
                    cur["pid"] = int(value)
                except ValueError:
                    pass
    return out


def descends_from(pid, root: int, ppid_of: Callable, max_depth: int = 8) -> bool:
    """True when ``pid`` is ``root`` or one of its descendants.

    The safety net under the explicit PID registry: a player Jarvis spawns
    through a wrapper is still his, and ducking his own voice under his own
    voice is the one failure this module must never have."""
    try:
        cur = int(pid)
    except (TypeError, ValueError):
        return False
    root = int(root)
    for _ in range(max(1, int(max_depth))):
        if cur == root:
            return True
        if cur <= 1:
            return False
        nxt = ppid_of(cur)
        if nxt is None:
            return False
        try:
            nxt = int(nxt)
        except (TypeError, ValueError):
            return False
        if nxt == cur:                    # a lying /proc must not spin
            return False
        cur = nxt
    return False


def is_aec_playback(inp: dict) -> bool:
    """Is this sink-input the echo canceller's playback leg (Jarvis's own
    voice, see AEC_PLAYBACK_NAME)?  Name match only, exact, any of the three
    name properties."""
    return any(inp.get(k) == AEC_PLAYBACK_NAME
               for k in ("node_name", "media_name", "app_name"))


def duck_targets(inputs: Iterable[dict], exempt_pids: Iterable,
                 floor_pct: int = DEFAULT_FLOOR_PCT) -> list[dict]:
    """The streams worth moving: not Jarvis's (by PID, or the AEC leg by
    name), not corked, not already at or below the floor."""
    pids = set()
    for p in exempt_pids or ():
        try:
            pids.add(int(p))
        except (TypeError, ValueError):
            continue
    floor = max(int(floor_pct), MIN_TOUCH_PCT)
    out = []
    for inp in inputs or ():
        if not isinstance(inp, dict):
            continue
        if inp.get("pid") in pids:
            continue
        if is_aec_playback(inp):
            continue
        if inp.get("corked"):
            continue
        vol = inp.get("volume_pct")
        if not isinstance(vol, int) or vol <= floor:
            continue
        out.append(inp)
    return out


def volume_argv(index, pct) -> list[str]:
    return ["pactl", "set-sink-input-volume", str(int(index)),
            f"{max(0, min(150, int(round(pct))))}%"]


def plan_duck(inputs: Iterable[dict], exempt_pids: Iterable,
              floor_pct: int = DEFAULT_FLOOR_PCT) -> list[list[str]]:
    """The pactl commands that put every non-Jarvis stream at the floor."""
    return [volume_argv(i["index"], floor_pct)
            for i in duck_targets(inputs, exempt_pids, floor_pct)]


def plan_restore(saved: Iterable[dict]) -> list[list[str]]:
    """The commands that put back what ``plan_duck`` moved."""
    out = []
    for entry in saved or ():
        if not isinstance(entry, dict):
            continue
        idx, vol = entry.get("index"), entry.get("volume_pct")
        if idx is None or vol is None:
            continue
        out.append(volume_argv(idx, vol))
    return out


def ramp(start_pct: int, end_pct: int, steps: int = RAMP_STEPS) -> list[int]:
    """The volumes to write: inclusive of the end, exclusive of the start."""
    steps = max(1, int(steps))
    start, end = int(start_pct), int(end_pct)
    return [int(round(start + (end - start) * (i + 1) / steps))
            for i in range(steps)]


# -------------------------------------------------------- own-PID registry
class PidRegistry:
    """Every player Jarvis spawns, so the Mixer never ducks his own voice.

    One attribute (tts._play_proc) is not enough: tools/timekeeper.py runs
    its own ``paplay --volume=`` alarm loop, and an alarm ducked under the
    spoken alarm line is exactly the bug this registry prevents."""

    def __init__(self):
        self._pids: set[int] = set()
        self._lock = threading.Lock()

    def add(self, pid) -> None:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._pids.add(pid)

    def discard(self, pid) -> None:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._pids.discard(pid)

    def snapshot(self) -> set:
        with self._lock:
            return set(self._pids)


OWN_PIDS = PidRegistry()


def register_own_pid(pid) -> None:
    """Called by tts.py / tools/timekeeper.py when they spawn a player."""
    OWN_PIDS.add(pid)


def forget_own_pid(pid) -> None:
    OWN_PIDS.discard(pid)


def _proc_ppid(pid) -> Optional[int]:
    try:
        with open(f"/proc/{int(pid)}/stat", "r", encoding="utf-8",
                  errors="replace") as fh:
            data = fh.read()
    except (OSError, ValueError):
        return None
    # comm can contain spaces and parentheses, so the fields are counted
    # from the LAST closing paren, not from the start of the line.
    close = data.rfind(")")
    if close < 0:
        return None
    parts = data[close + 2:].split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def blocked() -> bool:
    """JARVIS_ROOM_CONTROL=0 forbids every real pactl/xrandr/gsettings call.

    tests/conftest.py sets it: the suite builds the REAL app, and a
    SpeakingState published in a test would otherwise duck the volume of
    whatever the user is actually listening to.  It doubles as his kill
    switch.  Only the default seam checks it -- an injected ``run`` is a
    test double and moves nothing."""
    return (os.environ.get("JARVIS_ROOM_CONTROL") or "").strip().lower() in \
        ("0", "off", "false", "no")


def _run(argv: list[str], timeout: float = PACTL_TIMEOUT_S):
    """The one subprocess seam (tests replace it with a recorder)."""
    if blocked():
        log.debug("mixer: %s suppressed (JARVIS_ROOM_CONTROL)", argv[:2])
        return subprocess.CompletedProcess(argv, 1, "", "room control off")
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# ----------------------------------------------------------------- mixer
class RoomMixer:
    """Ducks the room while Jarvis speaks or listens.

    ``start()`` subscribes it to the bus; ``stop()`` restores whatever it
    moved and unsubscribes.  Every pactl call happens on this object's own
    worker thread -- the bus delivers on the Tk thread, and a 200 ms ramp
    there would stutter the console."""

    def __init__(self, cfg=None, run: Optional[Callable] = None,
                 state_path: Optional[Path] = None,
                 registry: Optional[PidRegistry] = None,
                 ppid_of: Optional[Callable] = None,
                 sleep: Optional[Callable] = None,
                 now: Callable[[], float] = time.time,
                 remote: Optional[object] = None):
        self._cfg = cfg
        self._run = run or _run
        self._state_path = Path(state_path) if state_path else None
        self._registry = registry or OWN_PIDS
        self._ppid_of = ppid_of or _proc_ppid
        self._sleep = sleep or time.sleep
        self._now = now
        # Anything with duck(pct)/unduck(); jarvis.tools.spotify.SpotifyTool is
        # the one that exists.  See the module docstring (#72).
        self._remote = remote
        self._remote_ducked = False
        # The hold generation the remote was last asked on.  Without it a
        # hold with no active Connect device would re-ask every second (the
        # worker wakes on a 1 s timeout), two Web API calls a time.
        self._remote_tried = -1
        # What the remote is holding down right now, as it goes into the
        # state file, and when the next failed restore may be retried.
        self._remote_entry: Optional[dict] = None
        self._remote_retry_at = 0.0
        self._remote_retry_s = REMOTE_RETRY_S
        self._lock = threading.RLock()
        self._holds: set[str] = set()          # "speaking" | "recording"
        # When each hold was last vouched for.  See HOLD_MAX_S.
        self._hold_at: dict[str, float] = {}
        self._ducked: list[dict] = []          # what we moved, with originals
        self._generation = 0                   # bumped on every edge
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscribed = False
        # A remote duck the last run did not live to lift; healed from the
        # worker on the first pump, never from start() -- it is a network
        # call, and start() runs on the thread building the app.
        self._stale, self._remote_stale = self._load()
        self._remote_heal_at = 0.0

    # ------------------------------------------------------------ config
    def _get(self, key: str, default=None):
        get = getattr(self._cfg, "get", None)
        if callable(get):
            try:
                value = get(key, default)
                return default if value is None else value
            except Exception:  # noqa: BLE001 - a config hiccup must not duck him
                log.debug("mixer: cfg.get(%s) failed", key, exc_info=True)
        return default

    @property
    def enabled(self) -> bool:
        return bool(self._get("audio.duck", True))

    @property
    def floor_pct(self) -> int:
        try:
            value = int(self._get("audio.duck_level", DEFAULT_FLOOR_PCT))
        except (TypeError, ValueError):
            value = DEFAULT_FLOOR_PCT
        # 0 would mute the room outright and 100 is not a duck at all.
        return max(MIN_TOUCH_PCT, min(95, value))

    @property
    def ramp_ms(self) -> int:
        try:
            value = int(self._get("audio.duck_ramp_ms", DEFAULT_RAMP_MS))
        except (TypeError, ValueError):
            value = DEFAULT_RAMP_MS
        return max(0, min(2000, value))

    # ------------------------------------------------------------- state
    def _load(self) -> tuple[list[dict], Optional[dict]]:
        """(local stream entries, the remote duck entry) from the file."""
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict):
                    entries = data.get("streams")
                    remote = data.get("remote")
                    return ([e for e in entries if isinstance(e, dict)]
                            if isinstance(entries, list) else [],
                            remote if isinstance(remote, dict) else None)
        except (OSError, ValueError):
            log.debug("mixer state unreadable", exc_info=True)
        return [], None

    def _save(self, entries: list[dict]) -> None:
        """Write what is held down.  The remote half rides along from
        ``_remote_entry`` (live) or ``_remote_stale`` (still unhealed), so
        every existing caller persists it without knowing it exists."""
        if not self._state_path:
            return
        remote = self._remote_entry or self._remote_stale
        try:
            if not entries and not remote:
                self._state_path.unlink(missing_ok=True)
                return
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            payload = {"t": self._now(), "streams": entries}
            if remote:
                payload["remote"] = remote
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self._state_path)   # atomic: never half a file
        except OSError:
            log.debug("mixer state save failed", exc_info=True)

    def _persist(self) -> None:
        """Rewrite the file from what is currently held down, both halves.
        The remote duck lands after the local one, so it needs its own write."""
        with self._lock:
            live = list(self._ducked)
        self._save(self._merge_stale(live) if live else list(self._stale))

    # ------------------------------------------------------------- pactl
    def _pactl(self, argv: list[str]) -> Optional[str]:
        what = argv[1] if len(argv) > 1 else argv
        try:
            out = self._run(argv)
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("mixer: %s failed: %s", what, exc)
            return None
        if getattr(out, "returncode", 1) != 0:
            log.debug("mixer: %s rc=%s", what, getattr(out, "returncode", None))
            return None
        return getattr(out, "stdout", "") or ""

    def sink_inputs(self) -> list[dict]:
        text = self._pactl(["pactl", "list", "sink-inputs"])
        return parse_sink_inputs(text) if text is not None else []

    def exempt_pids(self, inputs: Iterable[dict]) -> set:
        """Jarvis's own PIDs: the registry, this process, and any stream
        whose process descends from it."""
        pids = self._registry.snapshot()
        mine = os.getpid()
        pids.add(mine)
        for inp in inputs or ():
            pid = inp.get("pid") if isinstance(inp, dict) else None
            if pid is None or pid in pids:
                continue
            if descends_from(pid, mine, self._ppid_of):
                pids.add(pid)
        return pids

    # -------------------------------------------------------- stale heal
    def heal(self) -> int:
        """Put back any volume a crashed Jarvis left written into the
        stream-restore database.  Entries whose stream has not reappeared
        are KEPT: the poison is keyed by application name, so a librespot
        restarted tomorrow inherits it and needs healing then."""
        if not self._stale:
            return 0
        cutoff = self._now() - STALE_MAX_S
        live = parse_sink_inputs(self._pactl(["pactl", "list", "sink-inputs"]) or "")
        by_key: dict[str, list[dict]] = {}
        for inp in live:
            key = inp.get("restore_key") or f"index:{inp.get('index')}"
            by_key.setdefault(key, []).append(inp)
        kept, healed = [], 0
        floor = self.floor_pct
        for entry in self._stale:
            try:
                stamp = float(entry.get("t") or 0)
            except (TypeError, ValueError):
                stamp = 0.0
            if stamp < cutoff:
                continue                       # too old to still be ours
            key = entry.get("restore_key") or f"index:{entry.get('index')}"
            matches = by_key.get(key) or []
            if not matches:
                kept.append(entry)             # heal it when it comes back
                continue
            want = entry.get("volume_pct")
            for inp in matches:
                cur = inp.get("volume_pct")
                # Only heal a stream still sitting at (or under) the floor:
                # anything louder is a volume he has since chosen himself.
                if want is None or cur is None or cur > floor + 2:
                    continue
                if self._pactl(volume_argv(inp["index"], want)) is not None:
                    healed += 1
        self._stale = kept
        self._save(kept)
        if healed:
            log.info("mixer: healed %d stream(s) left ducked by a previous run",
                     healed)
        return healed

    # -------------------------------------------------------------- edges
    def _set_hold(self, name: str, on: bool) -> None:
        with self._lock:
            before = bool(self._holds)
            if on:
                self._holds.add(name)
                # Stamped on EVERY tick, not only on the edge: this is the
                # hold's proof of life for _expire_holds, and SpeakingState
                # repeats at ~12 Hz for the whole of a burst.
                self._hold_at[name] = self._now()
            else:
                self._holds.discard(name)
                self._hold_at.pop(name, None)
            if before == bool(self._holds):
                return
            self._generation += 1
        self._wake.set()

    def _expire_holds(self) -> None:
        """Let go of any hold nothing has vouched for in HOLD_MAX_S.

        WARNING, and it names the hold: an unbounded wait that says nothing
        is worse than a short one that does.  This is the last line between
        a publisher that stops publishing and a room left at 30 % for the
        rest of the day (2026-09-02 08:56:19)."""
        now = self._now()
        with self._lock:
            stale = sorted(n for n in self._holds
                           if now - self._hold_at.get(n, now) > HOLD_MAX_S)
            if not stale:
                return
            ages = {n: now - self._hold_at.get(n, now) for n in stale}
            for name in stale:
                self._holds.discard(name)
                self._hold_at.pop(name, None)
            self._generation += 1
        for name in stale:
            log.warning("mixer: the %r hold has said nothing for %.0fs; "
                        "letting the room back up (nothing lifted it)",
                        name, ages[name])

    @property
    def holds(self) -> set:
        """What is holding the room down right now (read-only)."""
        with self._lock:
            return set(self._holds)

    def on_speaking(self, ev) -> None:
        """SpeakingState arrives ~12 Hz while he talks; only the edge counts."""
        self._set_hold("speaking", bool(getattr(ev, "active", False)))

    def on_recording_started(self, ev=None) -> None:
        self._set_hold("recording", True)

    def on_recording_stopped(self, ev=None) -> None:
        self._set_hold("recording", False)

    # -------------------------------------------------------------- work
    def set_remote(self, remote) -> None:
        """Wire the Connect ducker after construction.

        app.py builds the mixer at line 372 but the tools only at 446, so
        ``services.spotify`` does not exist yet at __init__ time.  This is the
        seam for the single line that connects them (#72):
        ``self.mixer.set_remote(getattr(self.services, "spotify", None))``
        after ``_register_tools()``."""
        self._remote = remote

    def _remote_duck(self, floor: int, generation: int) -> None:
        """Duck the Spotify Connect device, once per hold.

        Guarded by blocked() as well as by ``remote`` being None: a test that
        wired a real SpotifyTool must not reach across the network and turn
        down music he is actually listening to.  Runs on the mixer thread --
        the two Web API calls take 0.3-0.6 s, which is why the local ramp
        goes first and why the hold is re-checked afterwards: a duck that
        lands after he has stopped talking is put straight back.  The cost
        of sharing the worker is that a hung call delays the LOCAL restore
        too, bounded by spotify.API_TIMEOUT_S per call; on this box the local
        stream is the silent librespot pipe, so that delay is inaudible."""
        if self._remote is None or self._remote_ducked or blocked():
            return
        if self._remote_tried == generation:
            return
        self._remote_tried = generation
        try:
            ok = bool(self._remote.duck(floor))
        except Exception:  # noqa: BLE001 - a remote hiccup must not break the duck
            log.debug("mixer: remote duck failed", exc_info=True)
            return
        if not ok:
            return
        self._remote_ducked = True
        self._remote_retry_at = 0.0
        self._remote_retry_s = REMOTE_RETRY_S
        name = getattr(self._remote, "ducked_device", None) or "(unnamed)"
        # Take ownership of the record: whatever a previous run left unhealed
        # for this device is superseded by the volume we just read.  Stamped
        # here, not by the tool, because it is this file's age that decides
        # whether a heal on the next start is still his volume or his choice.
        state = getattr(self._remote, "ducked_state", None)
        if isinstance(state, dict):
            self._remote_entry = {**state, "t": self._now()}
            self._remote_stale = None
            self._persist()
        log.info("mixer: ducked the Spotify Connect device %s to %d%% of its volume",
                 name, floor)
        if self._stop.is_set():
            # stop() restored while this call was in flight and the worker
            # is about to exit: nobody else will lift it.  A hold that merely
            # ended is handled by the next pump, which the edge already woke.
            self._remote_restore(force=True)

    def _remote_restore(self, force: bool = False) -> None:
        """Lift the remote duck, and KEEP IT ON THE BOOKS if that fails.

        A failed unduck used to clear the flag and walk away, which left his
        phone or HPCOMPUTER at 30 % of his volume with nothing that would
        ever put it back.  Now the duck stands until the write is confirmed:
        the next pump tries again (backing off, because the worker wakes every
        second and the device may simply be gone), stop() tries once more with
        ``force``, and what is still down stays in the state file for the next
        start's heal."""
        if self._remote is None or not self._remote_ducked:
            return
        now = self._now()
        if not force and now < self._remote_retry_at:
            return
        name = getattr(self._remote, "ducked_device", None) or "(unnamed)"
        try:
            ok = bool(self._remote.unduck())
        except Exception:  # noqa: BLE001 - see _remote_duck
            log.debug("mixer: remote unduck failed", exc_info=True)
            ok = False
        if not ok:
            wait = self._remote_retry_s
            self._remote_retry_at = now + wait
            self._remote_retry_s = min(wait * 2, REMOTE_RETRY_MAX_S)
            log.warning("mixer: could not restore the Spotify Connect device %s; "
                        "it is still down, retrying in %.0f s", name, wait)
            return
        self._remote_ducked = False
        self._remote_retry_at = 0.0
        self._remote_retry_s = REMOTE_RETRY_S
        self._remote_entry = None
        self._persist()
        log.info("mixer: restored the Spotify Connect device %s", name)

    def _remote_heal(self) -> None:
        """Put back a Connect volume a crashed run left down.

        The remote half of heal().  It cannot run there: heal() happens on
        start(), on the thread assembling the app, and this is a Web API
        call -- so the worker does it on its first pump instead.  Retried
        once a minute while the device is not listed (it may be switched
        off), abandoned after REMOTE_STALE_MAX_S."""
        entry = self._remote_stale
        if entry is None or self._remote is None or blocked():
            return
        now = self._now()
        try:
            stamp = float(entry.get("t") or 0)
        except (TypeError, ValueError):
            stamp = 0.0
        if now - stamp > REMOTE_STALE_MAX_S:
            self._forget_remote_stale()
            return
        if now < self._remote_heal_at:
            return
        self._remote_heal_at = now + REMOTE_HEAL_RETRY_S
        fn = getattr(self._remote, "restore_volume", None)
        if not callable(fn):
            self._forget_remote_stale()
            return
        try:
            # Same rule as the local heal: only a device still at or under
            # the floor can still be ours.
            ok = fn(entry.get("device"), int(entry.get("volume_pct") or 0),
                    at_or_below=self.floor_pct + 2)
        except Exception:  # noqa: BLE001 - a remote hiccup must not break the pump
            log.debug("mixer: remote heal failed", exc_info=True)
            return
        if ok is None:
            return            # not listed: ask again while the record lasts
        if ok:
            log.info("mixer: healed the Spotify Connect device %s left ducked "
                     "by a previous run", entry.get("name") or "(unnamed)")
        self._forget_remote_stale()

    def _forget_remote_stale(self) -> None:
        self._remote_stale = None
        self._persist()

    def music_playing(self) -> bool:
        """Is music known to be playing?  For the wake gate (hotword.py):
        a cache read on the remote, never a request, and False without one.

        A duck of our own is deliberately NOT counted.  The duck fires on
        every hold and only proves a Connect device is listed active, which
        Spotify keeps true long after a pause -- counting it would relax the
        wake bar during every turn in a silent room."""
        fn = getattr(self._remote, "music_playing", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001 - a broken remote is not music
            log.debug("mixer: remote music_playing failed", exc_info=True)
            return False

    def _duck(self, generation: int) -> None:
        floor = self.floor_pct
        inputs = self.sink_inputs()
        targets = duck_targets(inputs, self.exempt_pids(inputs), floor)
        if targets:
            saved = [{"index": t["index"], "volume_pct": t["volume_pct"],
                      "restore_key": t.get("restore_key", ""),
                      "app_name": t.get("app_name", ""), "t": self._now()}
                     for t in targets]
            with self._lock:
                self._ducked = saved
            # Written BEFORE the first pactl call: a crash between the write
            # and the restore is exactly what heal() exists for.
            self._save(self._merge_stale(saved))
            self._ramp_to(saved, floor, generation)
            log.info("mixer: ducked %d stream(s) to %d%%", len(saved), floor)
        else:
            log.debug("mixer: nothing local to duck")
        # The Connect device as well, not instead: pactl cannot tell the idle
        # librespot pipe (uncorked, silent) from music, so "found a local
        # stream" says nothing about where the music he hears is coming from
        # (#72, 2026-09-01).  Local first because it is 200 ms and never
        # waits on the network.
        self._remote_duck(floor, generation)

    def _restore(self, generation: int) -> None:
        self._remote_restore()
        with self._lock:
            saved = list(self._ducked)
            self._ducked = []
        if not saved:
            return
        self._ramp_back(saved, generation)
        self._save(self._stale)
        log.debug("mixer: restored %d stream(s)", len(saved))

    def _merge_stale(self, saved: list[dict]) -> list[dict]:
        keys = {e.get("restore_key") or f"index:{e.get('index')}" for e in saved}
        rest = [e for e in self._stale
                if (e.get("restore_key") or f"index:{e.get('index')}") not in keys]
        return rest + saved

    def _ramp_to(self, saved: list[dict], floor: int, generation: int) -> None:
        step_s = (self.ramp_ms / 1000.0) / max(1, RAMP_STEPS)
        for i in range(RAMP_STEPS):
            for entry in saved:
                level = ramp(int(entry["volume_pct"]), floor, RAMP_STEPS)[i]
                self._pactl(volume_argv(entry["index"], level))
            if self._changed(generation):
                return                     # he stopped talking mid-slide
            if step_s:
                self._sleep(step_s)

    def _ramp_back(self, saved: list[dict], generation: int) -> None:
        floor = self.floor_pct
        step_s = (self.ramp_ms / 1000.0) / max(1, RAMP_STEPS)
        for i in range(RAMP_STEPS):
            for entry in saved:
                level = ramp(floor, int(entry["volume_pct"]), RAMP_STEPS)[i]
                self._pactl(volume_argv(entry["index"], level))
            if i < RAMP_STEPS - 1:
                if self._changed(generation):
                    # Interrupted mid-lift: put them straight back where
                    # they were before handing over. A stream stranded at
                    # 60 % of itself is worse than an abrupt restore.
                    for entry in saved:
                        self._pactl(volume_argv(entry["index"],
                                                int(entry["volume_pct"])))
                    return
                if step_s:
                    self._sleep(step_s)

    def _changed(self, generation: int) -> bool:
        with self._lock:
            return self._generation != generation or self._stop.is_set()

    def pump(self) -> None:
        """One reconciliation pass (the worker's body; tests call it)."""
        # Before anything is ducked afresh: a device a crashed run left down
        # must be read back at HIS volume, not at the 30 % a new duck would
        # mistake for it.
        self._remote_heal()
        # Before deciding anything: a hold nobody is going to lift is not a
        # reason to keep his music down.  The worker wakes at least once a
        # second, so this is checked at ~1 Hz for free.
        self._expire_holds()
        with self._lock:
            want = bool(self._holds)
            generation = self._generation
            ducked = bool(self._ducked)
        # A remote-only duck moves nothing local, so self._ducked stays
        # empty; without this the restore edge never fires and his Spotify
        # volume stays at 30% of where he left it.
        ducked = ducked or self._remote_ducked
        if not self.enabled:
            if ducked:
                self._restore(generation)
            return
        if want and not ducked:
            self._duck(generation)
        elif not want and ducked:
            self._restore(generation)

    # ------------------------------------------------------------ thread
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                self.pump()
            except Exception:
                log.exception("mixer pump failed")

    def start(self) -> None:
        # Alive-guard + clear, like deadlines.py: a stopped instance can be
        # started again (tests, a config reload).
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            self.heal()
        except Exception:
            log.exception("mixer heal failed")
        if not self._subscribed:
            bus.subscribe(SpeakingState, self.on_speaking)
            bus.subscribe(RecordingStarted, self.on_recording_started)
            bus.subscribe(RecordingStopped, self.on_recording_stopped)
            self._subscribed = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="mixer")
        self._thread.start()

    def stop(self) -> None:
        if self._subscribed:
            bus.unsubscribe(SpeakingState, self.on_speaking)
            bus.unsubscribe(RecordingStarted, self.on_recording_started)
            bus.unsubscribe(RecordingStopped, self.on_recording_stopped)
            self._subscribed = False
        with self._lock:
            self._holds.clear()
            self._hold_at.clear()
            self._generation += 1
            generation = self._generation
        # Restore BEFORE the stop flag is set: quitting with the room at
        # 30 % is the one outcome nobody would forgive.  The remote goes
        # first and FORCED past the retry backoff -- this is the last chance
        # anything in this process has to put his Connect volume back.
        try:
            self._remote_restore(force=True)
        except Exception:
            log.exception("mixer remote restore on stop failed")
        try:
            self._restore(generation)
        except Exception:
            log.exception("mixer restore on stop failed")
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        # A remote duck that was in flight on the worker during the restore
        # above landed after it; _remote_duck lifts it itself when it sees the
        # stop flag, but the flag can be set between its check and ours, so
        # the worker having exited is the one moment both sides agree.
        try:
            self._remote_restore(force=True)
        except Exception:
            log.exception("mixer remote restore on stop failed")
