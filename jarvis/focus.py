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

While a block is running the quiet policy holds proactive lines
(`focus.dnd`, default on) and reads them back as the usual catch-up digest
when the break starts -- the session's own lines are non-proactive and speak
through regardless.

State lives in `focus_session.json` under PATHS.MEMORY_DIR (the app passes
the path): phase, block number, the item ids, what was done to the music.
`reconcile()` at boot -- after Timekeeper.start(), whose catch-up may have
fired a block that came due while the app was down -- closes a session
whose current item is no longer pending and says how many blocks were done.

That file is OVERWRITTEN by the next session, so the count used to be
spoken once and lost. `focus_history.jsonl` beside it is the durable half:
one JSON line per finished session, appended by `end()` and keyed on the
session's `started` stamp so the boot-time lapsed path cannot double-count
it. `study_days()` reads that ledger AND the timekeeper's own `focus:
block N` rows (which are never pruned) and merges them, so blocks from
before this ledger existed -- or from a session the app died in -- still
count. It is what "how much did I study this week" answers from.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis import music
from jarvis.events import ReminderFired, bus
from jarvis.logs import get_logger
from jarvis.tools.timekeeper import NO_TIMEKEEPER_LINE, SILENT_PREFIX, count_words

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

# ------------------------------------------------------------- the ledger
HISTORY_NAME = "focus_history.jsonl"
# A block item is labelled "focus: block N" (timekeeper.add_silent_timer);
# "halfway" and "break" items share the prefix and must not be counted.
BLOCK_LABEL_LIKE = f"{SILENT_PREFIX} block%"
HISTORY_TAIL_BYTES = 256_000       # enough tail to dedupe against; not the whole file
NO_STUDY_LINE = "You haven't logged any study {when}, sir."
STUDY_TOTAL_LINE = "{time} {when}, sir, over {blocks} on {days}."
STREAK_LINE = "{n} days running, sir."
STREAK_ONE_LINE = "Today, sir; that's the start of one."
STREAK_NONE_LINE = "No streak at the moment, sir."

PERSONA_LINES = [HALFWAY_LINE, END_NONE_LINE, NO_SESSION_LINE, STREAK_NONE_LINE]


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


def history_path(state_path) -> Optional[Path]:
    """The ledger beside the session-state file, or None with no state
    path (a bare test session keeps no history)."""
    return Path(state_path).with_name(HISTORY_NAME) if state_path else None


def read_history(path) -> list[dict]:
    """Every well-formed ledger row in the file; [] when it is missing."""
    try:
        return read_history_text(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return []


def read_history_text(text: str) -> list[dict]:
    """Rows from ledger text. A half-written or hand-mangled line is
    SKIPPED rather than raising: this file is read to answer a question,
    and one bad line must not make the whole feature go silent. It is also
    how the tail read in _log_history tolerates starting mid-line."""
    out: list[dict] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("started") is not None:
            out.append(row)
    return out


def timekeeper_blocks(db_path) -> list[dict]:
    """[{when, minutes}] for every finished focus BLOCK the timekeeper
    still holds. Read-only, its own connection, and every failure is an
    empty list: this is a bonus source, not the ledger.

    The block's length is (due - created), which is exactly what
    _file_block asked for -- no guess at block_min is needed."""
    import sqlite3
    out: list[dict] = []
    try:
        if not Path(db_path).exists():
            return out
        db = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True, timeout=2.0)
    except (sqlite3.Error, OSError, TypeError, ValueError):
        log.debug("focus: timekeeper history unreadable", exc_info=True)
        return out
    try:
        rows = db.execute(
            "SELECT fired_at, due, created FROM items WHERE state = 'done' "
            "AND kind = 'timer' AND lower(label) LIKE ?", (BLOCK_LABEL_LIKE,)).fetchall()
    except sqlite3.Error:
        log.debug("focus: timekeeper history query failed", exc_info=True)
        rows = []
    finally:
        db.close()
    for fired, due, created in rows:
        try:
            when = float(fired if fired is not None else due)
            minutes = int(round((float(due) - float(created)) / 60.0))
        except (TypeError, ValueError):
            continue
        if when > 0 and 0 < minutes <= 600:
            out.append({"when": when, "minutes": minutes})
    return out


def _day(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).date().isoformat()


def study_days(state_path=None, db_path=None, rows=None, blocks=None) -> dict:
    """{"YYYY-MM-DD": {"blocks": n, "minutes": m}} from the ledger plus any
    timekeeper block the ledger does not already cover.

    A timekeeper block that fired inside a logged session's window is the
    SAME block seen twice, so it is dropped; one outside every window is a
    session that ended before this ledger existed, or one the app died in,
    and it counts as itself. ``rows``/``blocks`` are the seams the tests
    and the backfill script use instead of the two paths."""
    if rows is None:
        path = history_path(state_path)
        rows = read_history(path) if path else []
    if blocks is None:
        blocks = timekeeper_blocks(db_path) if db_path else []
    out: dict = {}
    windows = []

    def _add(day: str, n_blocks: int, minutes: int):
        cell = out.setdefault(day, {"blocks": 0, "minutes": 0})
        cell["blocks"] += n_blocks
        cell["minutes"] += minutes

    for row in rows:
        try:
            started = float(row.get("started") or 0.0)
            n = int(row.get("blocks") or 0)
            per = int(row.get("block_min") or 0)
        except (TypeError, ValueError):
            continue
        if started <= 0 or n <= 0:
            continue
        ended = float(row.get("ended") or started)
        windows.append((started, max(ended, started)))
        _add(str(row.get("date") or _day(started)), n, n * max(0, per))
    for blk in blocks:
        when = float(blk.get("when") or 0.0)
        if when <= 0 or any(a <= when <= b for a, b in windows):
            continue
        _add(_day(when), 1, int(blk.get("minutes") or 0))
    return out


def week_start(now: Optional[date] = None) -> date:
    """Monday of the current week -- the week a student means by "this
    week", not a rolling seven days."""
    today = now or date.today()
    return today - timedelta(days=today.weekday())


def streak_days(days: dict, today: Optional[date] = None) -> int:
    """Consecutive days of study ending today or yesterday. Yesterday
    counts as still alive: at 9 am the streak he built last night is not
    broken yet, and calling it zero would be a lie he acts on."""
    today = today or date.today()
    cursor = today if days.get(today.isoformat()) else today - timedelta(days=1)
    n = 0
    while days.get(cursor.isoformat()):
        n += 1
        cursor -= timedelta(days=1)
    return n


def time_words(minutes: int) -> str:
    """'two hours and ten minutes' / 'fifty minutes' / 'an hour'."""
    minutes = max(0, int(minutes))
    hours, mins = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append("an hour" if hours == 1 else f"{count_words(hours)} hours")
    if mins or not hours:
        parts.append("a minute" if mins == 1 else f"{count_words(mins) if mins <= 12 else mins} minutes")
    return " and ".join(parts)


def summary_line(days: dict, since: date, when: str = "this week") -> str:
    """'Two hours and ten minutes this week, sir, over five blocks on
    three days.' or the honest nothing."""
    picked = {d: cell for d, cell in days.items() if d >= since.isoformat()}
    blocks = sum(int(c["blocks"]) for c in picked.values())
    minutes = sum(int(c["minutes"]) for c in picked.values())
    if not blocks:
        return NO_STUDY_LINE.format(when=when)
    n_days = len(picked)
    return STUDY_TOTAL_LINE.format(
        time=time_words(minutes).capitalize(), when=when,
        blocks=blocks_words(blocks),
        days="one day" if n_days == 1 else f"{count_words(n_days)} days")


def streak_line(days: dict, today: Optional[date] = None) -> str:
    n = streak_days(days, today)
    if n <= 0:
        return STREAK_NONE_LINE
    if n == 1:
        return STREAK_ONE_LINE
    return STREAK_LINE.format(n=count_words(n))


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
                # proactive=False is what makes these lines PIERCE the quiet
                # hold: quiet.py holds a focus BLOCK (jarvis/quiet.py
                # _focus_reason), and a session whose own "Time for a break,
                # sir" got parked in its own digest would never break.
                # kind reaches the quiet digest ("two messages", not "two
                # warnings" -- the services speak lambda defaults to the
                # watchdog's kind).
                say(line, proactive=False, kind="message")
            except TypeError:
                try:
                    say(line, kind="message")
                except TypeError:
                    say(line)              # a bare test seam takes text only
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
            # The call itself lives in jarvis/music.py so the class stager
            # can pause and resume the same way; what is session-shaped --
            # the write-back and "stop retrying" -- stays here.
            ok, kind = music.call(spotify, method, arg)
            if ok:
                with self._lock:
                    if self.state is st:
                        self.state["music_did"] = remember
                        self._save()
            elif music.permanent(kind):
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

    @property
    def history_path(self) -> Optional[Path]:
        return history_path(self._state_path)

    def _log_history(self, blocks: int, block_min: int) -> None:
        """Append this session to focus_history.jsonl. Called from end()
        under the lock, BEFORE the state file is overwritten -- that write
        is what used to lose the count.

        Deduped on the session's `started` stamp: reconcile() closes a
        lapsed session with end("lapsed") at boot, and without the key a
        crash-and-restart could file the same night twice."""
        path = self.history_path
        if path is None or blocks <= 0:
            return                          # no full block is nothing to log
        try:
            started = float(self.state.get("started") or 0.0)
        except (TypeError, ValueError):
            started = 0.0
        if started <= 0:
            return
        try:
            if path.exists():
                with path.open("rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    fh.seek(max(0, fh.tell() - HISTORY_TAIL_BYTES))
                    tail = fh.read().decode("utf-8", "replace")
                for row in read_history_text(tail):
                    if abs(float(row.get("started") or 0.0) - started) < 1.0:
                        log.info("focus: session %.0f already in the ledger", started)
                        return
        except OSError:
            log.debug("focus: ledger tail unreadable", exc_info=True)
        row = {"date": _day(started), "started": started, "ended": float(self._now()),
               "label": self.label, "blocks": int(blocks), "block_min": int(block_min)}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # One append of one complete line: an interrupted process can
            # lose the line but never leave half of it in front of the next.
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            log.warning("focus: could not append to the study ledger", exc_info=True)

    def end(self, reason: str = "user") -> str:
        with self._lock:
            if not self.active:
                return NO_SESSION_LINE
            blocks, n = self.blocks_done, int(self.state.get("block_min") or 0)
            self._cancel_items()
            self._music("end")
            self._log_history(blocks, n)
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
