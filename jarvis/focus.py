"""Focus / study sessions: "study session biosensors", "start a fifty-minute
focus session", "how long left", "end the session".

One block of `block_min` (default 25, "fifty-minute" sets 50) then a break of
`break_min` (5), repeated until "end the session" or `max_blocks` blocks are
done; halfway through each block of ten minutes or more he says "Halfway,
sir". The timing is the timekeeper's: every block, halfway mark and break is
a SILENT timer (label prefix timekeeper.SILENT_PREFIX) in its sqlite store,
so a session survives a restart, and the timekeeper publishes
ReminderFired(item_id=..., silent=True) instead of speaking the generic
"your timer is up" line. This class owns the words.

Music (assistant.json `focus.music`): "pause" (default) pauses Spotify for
the block and resumes it for the break; "playlist" plays `focus.playlist`
for the block and pauses it for the break; "off" leaves Spotify alone.
Every Spotify call runs on its own thread (the bus delivers on the Tk
thread) and every SpotifyError is swallowed: a session with no Connect
device or an unlinked account is a session without music, never an
apology mid-study. A setup/auth failure switches music off for the rest
of the session so he does not retry it every block.

State lives in `focus_session.json` under PATHS.MEMORY_DIR (the app passes
the path): phase, block number, the item ids, what was done to the music.
`reconcile()` at boot -- after Timekeeper.start(), whose catch-up may have
fired a block that came due while the app was down -- closes a session
whose current item is no longer pending and says how many blocks were done.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from jarvis.events import ReminderFired, bus
from jarvis.logs import get_logger
from jarvis.tools.timekeeper import NO_TIMEKEEPER_LINE, count_words

log = get_logger("focus")

DEFAULT_BLOCK_MIN = 25
DEFAULT_BREAK_MIN = 5
DEFAULT_MAX_BLOCKS = 4
MIN_HALFWAY_BLOCK_MIN = 10          # a halfway call in a 5-minute block is noise
MUSIC_MODES = ("pause", "playlist", "off")

# Persona lines (fixed ones are prewarmed by app._canned_phrases).
START_LINE = "{n} minutes on {label}, sir; I'll call the halfway mark and the break."
START_PLAIN_LINE = "{n} minutes of focus, sir; I'll call the halfway mark and the break."
START_SHORT_LINE = "{n} minutes on {label}, sir; I'll call the break."
START_SHORT_PLAIN_LINE = "{n} minutes of focus, sir; I'll call the break."
HALFWAY_LINE = "Halfway, sir."
BREAK_LINE = "Time for a break, sir; that's block {n} done. {m} minutes off."
RESUME_LINE = "Break's over, sir. Block {n}."
END_LINE = "Session over, sir: {blocks} of {n} minutes."
END_NONE_LINE = "Session over, sir; no full blocks this time."
COMPLETE_LINE = "That's {blocks} done, sir; session complete."
LAPSED_LINE = "Your study session lapsed while I was down, sir; {blocks} done."
LEFT_BLOCK_LINE = "{left} left in block {n}, sir."
LEFT_BREAK_LINE = "{left} of break left, sir."
NO_SESSION_LINE = "There's no session running, sir."
ALREADY_LINE = "You're already in a session, sir; {left} left in block {n}."
ALREADY_BREAK_LINE = "You're on a break, sir; {left} left."

PERSONA_LINES = [HALFWAY_LINE, END_NONE_LINE, NO_SESSION_LINE]


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    try:
        if callable(getter):
            value = getter(key, default)
            return default if value is None else value
    except Exception:                      # noqa: BLE001 - config boundary
        log.debug("focus config read failed for %s", key, exc_info=True)
    return default


def _int(value, default: int, lo: int = 1, hi: int = 600) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def left_words(seconds: float) -> str:
    """'twelve minutes', 'a minute', 'under a minute' (rounded up: a block
    with 30 s to run is 'a minute left', not 'zero minutes')."""
    s = max(0.0, float(seconds))
    if s < 45:
        return "under a minute"
    m = int((s + 59) // 60)
    if m <= 1:
        return "a minute"
    return f"{count_words(m) if m <= 12 else m} minutes"


def blocks_words(n: int) -> str:
    n = int(n)
    return f"{count_words(n)} block{'' if n == 1 else 's'}"


class FocusSession:
    def __init__(self, services, state_path: Optional[Path] = None,
                 now: Callable[[], float] = time.time, bg: Optional[Callable] = None):
        self.services = services
        self._state_path = Path(state_path) if state_path else None
        self._now = now
        self._bg = bg or self._thread
        self._lock = threading.RLock()
        self.state: dict = self._load()
        bus.subscribe(ReminderFired, self._on_reminder)

    # ------------------------------------------------------------ plumbing
    @staticmethod
    def _thread(fn: Callable[[], None]) -> None:
        threading.Thread(target=fn, name="focus-music", daemon=True).start()

    def _svc(self, name: str):
        return getattr(self.services, name, None)

    def _cfg(self, key: str, default=None):
        return _cfg_get(self._svc("assistant"), f"focus.{key}", default)

    def _speak(self, line: str) -> None:
        say = self._svc("speak")
        if not callable(say) or not line:
            return
        try:
            try:
                # kind reaches the quiet digest ("two messages", not "two
                # warnings" -- the services speak lambda defaults to the
                # watchdog's kind).
                say(line, kind="message")
            except TypeError:
                say(line)                  # a bare test seam takes text only
        except Exception:                  # noqa: BLE001 - speech boundary
            log.exception("focus: speak failed")

    def stop(self) -> None:
        """App shutdown: nothing runs here, but drop the bus subscription so
        a rebuilt app (tests) does not hear through a dead session."""
        bus.unsubscribe(ReminderFired, self._on_reminder)
        self._save()

    # --------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict) and data.get("phase") in ("block", "break"):
                    return data
        except (OSError, ValueError):
            log.debug("focus state unreadable", exc_info=True)
        return {}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self.state))
            os.replace(tmp, self._state_path)   # atomic: never half a session
        except OSError:
            log.debug("focus state save failed", exc_info=True)

    @property
    def active(self) -> bool:
        return self.state.get("phase") in ("block", "break")

    @property
    def phase(self) -> str:
        return str(self.state.get("phase") or "")

    @property
    def blocks_done(self) -> int:
        return int(self.state.get("blocks_done") or 0)

    @property
    def label(self) -> str:
        return str(self.state.get("label") or "")

    # ------------------------------------------------------------- timing
    def _item(self, key: str):
        tk = self._svc("timekeeper")
        item_id = self.state.get(key)
        if tk is None or not item_id:
            return None
        try:
            return tk.get(item_id)
        except Exception:                  # noqa: BLE001 - store boundary
            log.exception("focus: timekeeper.get failed")
            return None

    def _seconds_left(self, key: str) -> Optional[float]:
        it = self._item(key)
        if it is None or it.state not in ("pending", "snoozed", "ringing"):
            return None
        return max(0.0, float(it.effective_due) - float(self._now()))

    def _cancel_items(self) -> None:
        tk = self._svc("timekeeper")
        if tk is None:
            return
        for key in ("block_id", "half_id", "break_id"):
            it = self._item(key)
            if it is not None and it.state in ("pending", "snoozed", "ringing"):
                try:
                    tk.cancel(it.id, "all")
                except Exception:          # noqa: BLE001 - store boundary
                    log.exception("focus: cancel %s failed", key)
            self.state[key] = ""

    def _file_block(self, tk) -> None:
        n = self.blocks_done + 1
        block_s = int(self.state["block_min"]) * 60
        self.state["block_id"] = tk.add_silent_timer(block_s, f"block {n}").id
        self.state["half_id"] = ""
        if self.state.get("halfway") and self.state["block_min"] >= MIN_HALFWAY_BLOCK_MIN:
            self.state["half_id"] = tk.add_silent_timer(block_s / 2.0, f"halfway {n}").id
        self.state["phase"] = "block"
        self.state["block_started"] = float(self._now())

    def _file_break(self, tk) -> None:
        break_s = int(self.state["break_min"]) * 60
        self.state["break_id"] = tk.add_silent_timer(break_s, f"break {self.blocks_done}").id
        self.state["phase"] = "break"

    # -------------------------------------------------------------- music
    def _music(self, moment: str) -> None:
        """moment: block | break | end.  Runs the Spotify call off-thread."""
        mode = str(self.state.get("music") or "off").lower()
        if mode not in MUSIC_MODES or mode == "off" or self.state.get("music_off"):
            return
        spotify = self._svc("spotify")
        if spotify is None:
            return
        did = str(self.state.get("music_did") or "")
        playlist = str(self.state.get("playlist") or "").strip()
        # (call, argument, what to remember on success)
        plan = None
        if mode == "pause":
            if moment == "block":
                plan = ("control", "pause", "paused")
            elif moment in ("break", "end") and did == "paused":
                plan = ("control", "resume", "")
        elif mode == "playlist":
            if moment == "block" and playlist:
                plan = ("play", playlist, "playing")
            elif moment in ("break", "end") and did == "playing":
                plan = ("control", "pause", "")
        if plan is None:
            return
        method, arg, remember = plan
        # Bound the SESSION the result belongs to at closure creation:
        # start() rebinds self.state to a fresh dict, and a Spotify call
        # still in flight from the previous session (typically end()'s
        # resume) used to land its write-back in the new session's dict.
        st = self.state

        def run():
            try:
                if method == "play":
                    spotify.play(arg, "playlist")
                else:
                    spotify.control(arg)
                with self._lock:
                    if self.state is st:
                        self.state["music_did"] = remember
                        self._save()
                log.info("focus: spotify %s %r ok", method, arg)
            except Exception as exc:       # noqa: BLE001 - SpotifyError or worse
                kind = getattr(exc, "kind", "")
                log.info("focus: spotify %s %r skipped: %s", method, arg,
                         getattr(exc, "text", exc))
                if kind in ("setup", "auth", "premium"):
                    with self._lock:
                        if self.state is st:
                            self.state["music_off"] = True   # not worth retrying this session
                            self._save()

        self._bg(run)

    # ------------------------------------------------------------ commands
    def start(self, label: str = "", block_min=None, break_min=None) -> str:
        """Begin a session; returns the spoken reply."""
        tk = self._svc("timekeeper")
        if tk is None:
            return NO_TIMEKEEPER_LINE
        with self._lock:
            if self.active:
                return self._already_line()
            block = _int(block_min, _int(self._cfg("block_min", DEFAULT_BLOCK_MIN),
                                         DEFAULT_BLOCK_MIN))
            brk = _int(break_min, _int(self._cfg("break_min", DEFAULT_BREAK_MIN),
                                       DEFAULT_BREAK_MIN), lo=1, hi=120)
            mode = str(self._cfg("music", "pause") or "pause").lower()
            self.state = {
                "label": " ".join(str(label or "").split())[:60],
                "block_min": block, "break_min": brk,
                "halfway": bool(self._cfg("halfway", True)),
                "max_blocks": _int(self._cfg("max_blocks", DEFAULT_MAX_BLOCKS),
                                   DEFAULT_MAX_BLOCKS, lo=0, hi=100),
                "music": mode if mode in MUSIC_MODES else "pause",
                "playlist": str(self._cfg("playlist", "") or ""),
                "music_did": "", "music_off": False,
                "blocks_done": 0, "started": float(self._now()),
                "block_id": "", "half_id": "", "break_id": "",
            }
            # A stale silent item from a crashed session must not fire into
            # this one.
            try:
                tk.cancel("focus:", "timer")
            except Exception:              # noqa: BLE001 - store boundary
                log.exception("focus: stale cancel failed")
            self._file_block(tk)
            self._save()
            halfway = bool(self.state["half_id"])
            n = self.state["block_min"]
            lab = self.state["label"]
        log.info("focus: session started (%d/%d min, label=%r, music=%s)",
                 block, brk, lab, self.state["music"])
        self._music("block")
        if lab:
            return (START_LINE if halfway else START_SHORT_LINE).format(n=n, label=lab)
        return (START_PLAIN_LINE if halfway else START_SHORT_PLAIN_LINE).format(n=n)

    def _already_line(self) -> str:
        if self.phase == "break":
            left = self._seconds_left("break_id")
            return ALREADY_BREAK_LINE.format(left=left_words(left or 0))
        left = self._seconds_left("block_id")
        return ALREADY_LINE.format(left=left_words(left or 0), n=self.blocks_done + 1)

    def time_left(self) -> str:
        with self._lock:
            if not self.active:
                return NO_SESSION_LINE
            if self.phase == "break":
                left = self._seconds_left("break_id")
                if left is None:
                    return NO_SESSION_LINE
                return LEFT_BREAK_LINE.format(left=left_words(left).capitalize())
            left = self._seconds_left("block_id")
            if left is None:
                return NO_SESSION_LINE
            return LEFT_BLOCK_LINE.format(left=left_words(left).capitalize(),
                                          n=self.blocks_done + 1)

    def end(self, reason: str = "user") -> str:
        with self._lock:
            if not self.active:
                return NO_SESSION_LINE
            blocks, n = self.blocks_done, int(self.state.get("block_min") or 0)
            self._cancel_items()
            self._music("end")
            self.state["phase"] = ""
            self._save()
        log.info("focus: session ended (%s) after %d block(s)", reason, blocks)
        if reason == "lapsed":
            return LAPSED_LINE.format(blocks=blocks_words(blocks))
        if reason == "complete":
            return COMPLETE_LINE.format(blocks=blocks_words(blocks))
        if blocks == 0:
            return END_NONE_LINE
        return END_LINE.format(blocks=blocks_words(blocks), n=n)

    # --------------------------------------------------------------- boot
    def reconcile(self) -> Optional[str]:
        """After Timekeeper.start(): a session whose current item the
        catch-up fired (late) has already moved on through _on_reminder; one
        whose item was missed (> 1 h) or lost is closed with the count."""
        with self._lock:
            if not self.active:
                return None
            key = "break_id" if self.phase == "break" else "block_id"
            if self._seconds_left(key) is not None:
                log.info("focus: session resumed (%s, %d block(s) done)",
                         self.phase, self.blocks_done)
                return None
        line = self.end("lapsed")
        self._speak(line)
        return line

    # ------------------------------------------------------------- events
    def _on_reminder(self, ev: ReminderFired) -> None:
        if not getattr(ev, "silent", False) or not getattr(ev, "item_id", ""):
            return
        tk = self._svc("timekeeper")
        line, finished = None, False
        with self._lock:
            if not self.active or tk is None:
                return
            item_id = ev.item_id
            if item_id == self.state.get("half_id"):
                self.state["half_id"] = ""
                # From the boot catch-up (the app was down at the halfway
                # mark) it is old news, and the block itself usually fires
                # in the same pass: "Halfway, sir. Time for a break, sir."
                line = None if getattr(ev, "late", False) else HALFWAY_LINE
            elif item_id == self.state.get("block_id") and self.phase == "block":
                self.state["blocks_done"] = self.blocks_done + 1
                self.state["block_id"] = ""
                # The halfway timer of a block that came due while the app
                # was down fires in the same catch-up; it is done, not owed.
                stale_half = self._item("half_id")
                if stale_half is not None and stale_half.state in ("pending", "snoozed"):
                    tk.cancel(stale_half.id, "all")
                self.state["half_id"] = ""
                maxb = int(self.state.get("max_blocks") or 0)
                if maxb and self.blocks_done >= maxb:
                    finished = True
                else:
                    self._file_break(tk)
                    brk = int(self.state["break_min"])
                    line = BREAK_LINE.format(n=self.blocks_done,
                                             m=count_words(brk) if brk <= 12 else brk)
                    self._music("break")
            elif item_id == self.state.get("break_id") and self.phase == "break":
                self.state["break_id"] = ""
                self._file_block(tk)
                line = RESUME_LINE.format(n=self.blocks_done + 1)
                self._music("block")
            else:
                return                     # some other owner's silent item
            self._save()
        if finished:
            line = self.end("complete")
        if line:
            self._speak(line)
