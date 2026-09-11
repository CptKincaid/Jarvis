"""The Board's state layer: what mission control shows, with no Tk in it.

The console occupies 520x880 of a 3840x2160 panel; the Board is the second
borderless surface that fills the empty right flank (jarvis/ui/board.py).
THIS module is everything the Board knows and nothing about how it looks —
one pure function, `board_state()`, composing six panels out of INJECTED
providers:

    vitals      health.snapshot()      memory / GPU / load / trainers
    turns       the turn ledger        the last N waits, as a sparkline
    sessions    claude_session         live tasks + recent sessions
    deadlines   timekeeper + Canvas    what is due, soonest first
    focus       focus.FocusSession     the block or break running
    quiet       quiet.QuietPolicy      why he is holding his tongue

Why providers rather than imports: `health.snapshot()` spawns nvidia-smi
with a 5 s timeout and `canvas_due` is a REST call. A module that reached
for those itself could not be unit-tested without flaking, and the Board
polls every 5 s. Every provider is a zero-argument callable, every one is
optional, and every one is called inside a guard — a provider that raises
or is missing yields a panel marked `off`, never an exception and never a
half-built board. Panels are read with getattr/duck typing so a test can
drive the whole layer with SimpleNamespace fakes.

The Canvas half of `deadlines` is a NETWORK call and must never ride the
vitals tick: the app's provider caches it (CANVAS_TTL_S) and the board
simply asks. Canvas is reached through the tool registry — `canvas_due` is
a closure registered as a tool (jarvis/tools/canvas.py), not a module-level
function, exactly as jarvis/tools/briefing.py calls it.

`board_text()` renders the same state as plain text so `jarvis board` can
print it over SSH, which is how the whole layer was proved before a single
line of Tk existed.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvis.events import BoardUpdate, bus
from jarvis.logs import get_logger

log = get_logger("board")

# Panel order, top to bottom on the docked surface. The two that change
# fastest (vitals, turns) sit at the top where the eye lands.
PANEL_ORDER = ("vitals", "turns", "sessions", "deadlines", "focus", "quiet")
# Panels that appear only while they have something to say. The CAST slab
# (a thrown document / track / screen, jarvis/gesturecast.py) is on the
# board only for CAST_TTL_S after a throw: a permanent "CAST: nothing"
# would be clutter on a mission-control surface, and a stale one a lie.
OPTIONAL_PANELS = ("cast",)
CAST_TTL_S = 1800.0

POLL_S = 5.0              # BoardFeed cadence; the Canvas half is cached
SPARK_WIDTH = 28          # sparkline columns; ~2.5 minutes of turns
TURN_WINDOW = 40          # ledger records read back for the sparkline
SLOW_WAIT_S = 4.0         # a turn slower than this tints the panel amber
SESSION_ROWS = 4          # recent Claude sessions listed
# How old a session may be and still count as RECENT WORK. Past this it is
# history, not the state of the room, and it was pushing a live row off a
# four-row panel. HIS BUG, 2026-09-11: "random stuff still showing up in
# the bar like my schedule but its not its just dialogue" -- his panel was
# carrying sessions from six, seven and nine days ago.
SESSION_MAX_AGE_S = 36 * 3600.0
# WHAT THE VALUE COLUMN MEANS, and it is the other half of the same bug.
# A live task's value is its STATE ("RUNNING", "WAITING"). A recent
# session's value used to be its TITLE -- and a title is prose written by
# a model about a conversation, so beside a column of states it read as
# one. His two rows on 09-11 were "HUNTERP  Terminal closed" and "TEST
# Next class": the first reads as a status the console is reporting, the
# second as a diary entry. Neither is either. Every value in this panel is
# now a fact about state -- a status or an age -- and the slug already
# says WHICH project it is.
SESSION_AGE_WORDS = ((60.0, "just now"), (3600.0, "%d min ago"),
                     (86400.0, "%d h ago"))
DEADLINE_ROWS = 4
LOW_MEM_GB = 16.0         # below this the vitals panel goes amber
CRIT_MEM_GB = 8.0         # …and below this, red

# Spoken names for "focus on the <panel>" — the aliases a person actually
# says, not the dict keys. Longest match wins so "claude sessions" cannot
# be swallowed by "sessions".
PANEL_ALIASES = {
    "vitals": "vitals", "the vitals": "vitals", "suit vitals": "vitals",
    "engines": "vitals", "the engines": "vitals", "health": "vitals",
    "memory": "vitals", "gpu": "vitals",
    "turns": "turns", "the turns": "turns", "the ledger": "turns",
    "turn ledger": "turns", "latency": "turns", "timing": "turns",
    "sessions": "sessions", "the sessions": "sessions",
    "claude": "sessions", "claude sessions": "sessions",
    "the claude sessions": "sessions", "tasks": "sessions",
    "deadlines": "deadlines", "the deadlines": "deadlines",
    "due": "deadlines", "what's due": "deadlines", "schedule": "deadlines",
    "focus": "focus", "the focus": "focus", "the block": "focus",
    "focus block": "focus", "study": "focus",
    "quiet": "quiet", "the quiet": "quiet", "quiet hours": "quiet",
    "presence": "quiet", "do not disturb": "quiet",
    "cast": "cast", "the cast": "cast", "the throw": "cast",
    "last throw": "cast", "the last throw": "cast", "what i threw": "cast",
}


# ------------------------------------------------------------- structures
@dataclass
class Panel:
    """One slab on the Board. `rows` are (label, value) pairs; `tone` is a
    theme state key the renderer maps to a colour; `line` is the ONE
    sentence Jarvis speaks when asked to focus on this panel."""
    key: str
    title: str
    rows: list = field(default_factory=list)
    tone: str = "idle"                 # idle | ok | warn | error | off
    spark: tuple = ()                  # 0..1 floats, oldest first
    line: str = ""

    @property
    def empty(self) -> bool:
        return not self.rows


@dataclass
class BoardState:
    panels: list = field(default_factory=list)
    at: float = 0.0

    def get(self, key: str) -> Optional[Panel]:
        for p in self.panels:
            if p.key == key:
                return p
        return None

    @property
    def keys(self) -> tuple:
        return tuple(p.key for p in self.panels)


# ---------------------------------------------------------- pure helpers
def _safe(fn: Callable, what: str, default=None):
    """Call a provider; a missing or raising one is a dark panel, never an
    exception. The Board polls every 5 s, so this logs at debug only."""
    if fn is None:
        return default
    try:
        return fn()
    except Exception:                       # noqa: BLE001 - provider boundary
        log.debug("board: %s provider failed", what, exc_info=True)
        return default


def sparkline(values, width: int = SPARK_WIDTH) -> tuple:
    """Resample `values` (oldest first) to `width` columns normalised 0..1
    against the series maximum. Fewer values than columns pad on the LEFT
    with zeros, so a fresh ledger draws from the right like a strip chart.

    Pure; the renderer turns the floats into y coordinates."""
    nums = [float(v) for v in (values or [])
            if isinstance(v, (int, float)) and v == v and v >= 0]
    width = max(1, int(width))
    if not nums:
        return ()
    if len(nums) > width:
        # bucket-average down; a bucket MAX would turn one slow turn into a
        # plateau and make the chart read as a sustained problem
        step = len(nums) / width
        buckets = []
        for i in range(width):
            lo = int(i * step)
            hi = max(lo + 1, int((i + 1) * step))
            chunk = nums[lo:hi]
            buckets.append(sum(chunk) / len(chunk))
        nums = buckets
    peak = max(nums)
    if peak <= 0:
        return tuple(0.0 for _ in nums)
    scaled = [min(1.0, v / peak) for v in nums]
    pad = width - len(scaled)
    return tuple([0.0] * pad + scaled) if pad > 0 else tuple(scaled)


def wait_series(turns, limit: int = TURN_WINDOW) -> list:
    """`wait` seconds from ledger records, oldest first. "abort" and
    "superseded" are not turns anybody waited through (logtriage draws the
    same line) and a record without a wait never happened yet."""
    out = []
    for rec in (turns or []):
        if not isinstance(rec, dict):
            continue
        if rec.get("outcome") in ("abort", "superseded"):
            continue
        wait = rec.get("wait")
        if isinstance(wait, (int, float)) and wait == wait and wait >= 0:
            out.append(float(wait))
    return out[-max(1, int(limit)):]


def fmt_secs(value) -> str:
    """'840ms' / '2.4s' / '1:05' — the Board's one duration format."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    if v < 0:
        return "--"
    if v < 1:
        return f"{v * 1000:.0f}ms"
    if v < 60:
        return f"{v:.1f}s"
    m, s = divmod(int(v), 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}"


def fmt_gb(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{v:.0f}G" if v >= 10 else f"{v:.1f}G"


def fmt_when(due, now: float) -> str:
    """'in 12m' / 'in 3h' / 'now' / 'past' — relative, never a clock time:
    the Board is glanced at, and a clock time asks the reader to subtract."""
    try:
        delta = float(due) - float(now)
    except (TypeError, ValueError):
        return "--"
    if delta <= -60:
        return "past"
    if delta < 60:
        return "now"
    mins = int(delta // 60)
    if mins < 60:
        return f"in {mins}m"
    hours = mins / 60.0
    if hours < 24:
        return f"in {hours:.0f}h"
    return f"in {hours / 24:.0f}d"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


# ------------------------------------------------------------ panel makers
def vitals_panel(snap) -> Panel:
    """Memory, GPU, load and any trainer, from a health.Snapshot. Duck
    typed: the fields are read with getattr so a fake is a SimpleNamespace,
    and an unreadable snapshot is honest rather than zeroed."""
    p = Panel("vitals", "VITALS")
    if snap is None:
        p.tone = "off"
        p.line = "I can't read the machine's vitals just now, sir."
        return p
    total = getattr(snap, "mem_total_gb", None)
    avail = getattr(snap, "mem_avail_gb", None)
    if avail is not None and total:
        p.rows.append(("MEMORY", f"{fmt_gb(avail)} free of {fmt_gb(total)}"))
        if avail < CRIT_MEM_GB:
            p.tone = "error"
        elif avail < LOW_MEM_GB:
            p.tone = "warn"
        else:
            p.tone = "ok"
    gpu = getattr(snap, "gpu", None) or {}
    if gpu:
        bits = []
        if gpu.get("util_pct") is not None:
            bits.append(f"{gpu['util_pct']:.0f}%")
        if gpu.get("temp_c") is not None:
            bits.append(f"{gpu['temp_c']:.0f}°")
        if gpu.get("power_w") is not None:
            bits.append(f"{gpu['power_w']:.0f}W")
        p.rows.append(("GPU", " · ".join(bits) or "--"))
    load = getattr(snap, "load1", None)
    if load is not None:
        p.rows.append(("LOAD", f"{float(load):.1f}"))
    trainers = list(getattr(snap, "trainers", None) or [])
    if trainers:
        first = trainers[0]
        p.rows.append(("TRAINER", f"{getattr(first, 'name', '?')} "
                                  f"{fmt_gb(getattr(first, 'rss_gb', 0))}"))
        p.tone = "warn" if p.tone in ("ok", "idle") else p.tone
    if not p.rows:
        p.tone = "off"
        p.line = "I can't read the machine's vitals just now, sir."
        return p
    words = []
    if avail is not None and total:
        words.append(f"{fmt_gb(avail)} of {fmt_gb(total)} free")
    if gpu.get("util_pct") is not None:
        words.append(f"the GPU {gpu['util_pct']:.0f} per cent busy")
    if trainers:
        words.append(_count(len(trainers), "trainer") + " running")
    p.line = (("Vitals: " + ", ".join(words) + ", sir.") if words
              else "Vitals are nominal, sir.")
    return p


def turns_panel(turns) -> Panel:
    """The turn ledger as a strip chart: how long he has been making the
    user wait, last N turns. The one number that matters is the median —
    a mean is dragged around by a single cold model load."""
    p = Panel("turns", "TURNS")
    waits = wait_series(turns)
    if not waits:
        p.tone = "off"
        p.line = "No turns on the ledger yet, sir."
        return p
    p.spark = sparkline(waits)
    ordered = sorted(waits)
    median = ordered[len(ordered) // 2]
    worst = ordered[-1]
    slow = sum(1 for w in waits if w > SLOW_WAIT_S)
    p.rows = [("LAST", fmt_secs(waits[-1])),
              ("MEDIAN", fmt_secs(median)),
              ("WORST", fmt_secs(worst)),
              ("TURNS", str(len(waits)))]
    p.tone = "warn" if slow else "ok"
    tail = f", {_count(slow, 'slow one')}" if slow else ""
    p.line = (f"Last turn {fmt_secs(waits[-1])}, median {fmt_secs(median)} "
              f"over {_count(len(waits), 'turn')}{tail}, sir.")
    return p


def _session_age_s(info, now: float):
    """Seconds since that session was last touched, or None when it does
    not say. A row that cannot date itself is not recent work."""
    mtime = getattr(info, "mtime", None)
    if not isinstance(mtime, (int, float)) or isinstance(mtime, bool):
        return None
    return max(0.0, float(now) - float(mtime))


def session_age_words(age_s: float) -> str:
    """"just now" / "12 min ago" / "5 h ago" -- upper-cased by the panel.

    A FACT ABOUT STATE, which is what every other value in this panel is.
    See SESSION_AGE_WORDS for why a title is not."""
    for bound, form in SESSION_AGE_WORDS:
        if age_s < bound:
            if "%d" not in form:
                return form
            unit = 60.0 if bound <= 3600.0 else 3600.0
            return form % max(1, int(age_s // unit))
    return "%d h ago" % max(1, int(age_s // 3600.0))


def sessions_panel(sessions, tasks=None) -> Panel:
    """Live Claude work first (ClaudeTaskState the window already sees),
    then the most recent sessions on disk. A running task outranks its own
    session row, so a project never appears twice."""
    p = Panel("sessions", "CLAUDE")
    live = dict(tasks or {})
    for slug, state in sorted(live.items()):
        p.rows.append((slug.upper()[:14], str(state or "").upper()))
    running = [s for s in live.values() if s in ("running", "queued")]
    waiting = [s for s in live.values() if s == "waiting"]
    seen = set(live)
    now = time.time()
    for info in list(sessions or []):
        slug = str(getattr(info, "slug", "") or "")
        if not slug or slug in seen:
            continue
        age = _session_age_s(info, now)
        if age is None or age > SESSION_MAX_AGE_S:
            continue                       # history, not the state of the room
        seen.add(slug)
        p.rows.append((slug.upper()[:14], session_age_words(age)))
        if len(p.rows) >= SESSION_ROWS + len(live):
            break
    if not p.rows:
        p.tone = "off"
        p.line = "No Claude sessions, sir."
        return p
    if waiting:
        p.tone = "warn"
        p.line = (f"{_count(len(waiting), 'session')} waiting on a "
                  "permission answer, sir.")
    elif running:
        p.tone = "ok"
        p.line = f"{_count(len(running), 'task')} running, sir."
    else:
        p.tone = "idle"
        p.line = (f"Nothing running; {_count(len(p.rows), 'recent session')} "
                  "on the board, sir.")
    return p


def deadline_rows(items, canvas_lines=None, now: Optional[float] = None) -> list:
    """(label, value) rows from timekeeper Items and Canvas due lines,
    soonest first. Timekeeper items carry a real due timestamp so they
    sort; the Canvas lines are already ordered by the fact sheet and
    follow. Pure — `items` is any sequence of objects with `.kind`,
    `.label` and `.effective_due`."""
    now = time.time() if now is None else float(now)
    dated = []
    for it in (items or []):
        due = getattr(it, "effective_due", None)
        if due is None:
            continue
        label = str(getattr(it, "label", "") or getattr(it, "kind", "") or "?")
        dated.append((float(due), label))
    dated.sort()
    rows = [(label.upper()[:14] or "ITEM", fmt_when(due, now))
            for due, label in dated]
    for line in (canvas_lines or []):
        text = " ".join(str(line).split())
        if not text:
            continue
        head, _, tail = text.partition(" - ")
        rows.append((head.upper()[:14], (tail or text)[:26]))
    return rows


def deadlines_panel(items, canvas_lines=None, now=None) -> Panel:
    p = Panel("deadlines", "DUE")
    rows = deadline_rows(items, canvas_lines, now)
    if not rows:
        p.tone = "idle"
        p.line = "Nothing due, sir."
        return p
    p.rows = rows[:DEADLINE_ROWS]
    p.tone = "ok"
    first = p.rows[0]
    p.line = (f"{_count(len(rows), 'thing')} due; next is "
              f"{first[0].title()} {first[1]}, sir.")
    return p


def focus_panel(focus) -> Panel:
    """The running block or break. `focus` is a FocusSession (or a fake):
    .active, .phase, .label, .blocks_done, and time_left() for the words —
    all plain attribute reads, no timekeeper round trip of our own."""
    p = Panel("focus", "FOCUS")
    if focus is None:
        p.tone = "off"
        p.line = "No focus session, sir."
        return p
    active = bool(getattr(focus, "active", False))
    if not active:
        p.tone = "idle"
        p.rows = [("SESSION", "IDLE")]
        p.line = "No focus session running, sir."
        return p
    phase = str(getattr(focus, "phase", "") or "block").upper()
    p.rows = [("PHASE", phase),
              ("BLOCKS", str(getattr(focus, "blocks_done", 0)))]
    label = str(getattr(focus, "label", "") or "")
    if label:
        p.rows.append(("ON", label[:26]))
    p.tone = "ok"
    left = ""
    fn = getattr(focus, "time_left", None)
    if callable(fn):
        try:
            left = str(fn() or "")
        except Exception:                   # noqa: BLE001 - provider boundary
            log.debug("board: focus.time_left failed", exc_info=True)
    p.line = left or f"A {phase.lower()} is running, sir."
    return p


def quiet_panel(reason: str, presence_state: str = "",
                held: int = 0, configured: bool = True) -> Panel:
    """Quiet state and presence in one slab. presence.is_home() answers
    True when presence is UNCONFIGURED (home is None -> not False), so the
    row is shown ONLY when the sentinel is configured — a confident false
    "HOME" on a box with no phone_ip is worse than no row."""
    p = Panel("quiet", "QUIET")
    reason = str(reason or "").strip()
    if reason:
        p.rows.append(("HOLDING", reason[:26]))
        p.tone = "warn"
    else:
        p.rows.append(("HOLDING", "NO"))
        p.tone = "ok"
    if configured and presence_state:
        p.rows.append(("PRESENCE", str(presence_state).upper()))
    if held:
        p.rows.append(("HELD", str(held)))
    if reason:
        p.line = (f"I'm holding proactive lines — {reason}"
                  + (f", with {_count(held, 'line')} waiting" if held else "")
                  + ", sir.")
    else:
        p.line = "Nothing is holding me quiet, sir."
    return p


def cast_panel(recent, now: float) -> Optional[Panel]:
    """The last thing he threw, and where it ended up -- or None when
    nothing was thrown inside CAST_TTL_S. ``recent`` is
    gesturecast.GestureCast.recent(): status / spoken / target / kind / at,
    strings and numbers only. A HELD cast is drawn as a warning with the
    target it could not reach, because the board is where the payload fell
    back to and he should be able to see why it is here."""
    if not isinstance(recent, dict) or not recent:
        return None
    try:
        at = float(recent.get("at") or 0.0)
    except (TypeError, ValueError):
        return None
    if at <= 0.0 or now - at > CAST_TTL_S:
        return None
    status = str(recent.get("status") or "").lower()
    what = str(recent.get("spoken") or "").strip()
    target = str(recent.get("target") or "").strip()
    p = Panel("cast", "CAST")
    word = {"landed": "LANDED", "held": "HELD", "proposed": "ASKED",
            "refused": "REFUSED", "vetoed": "REFUSED",
            "empty": "NOTHING"}.get(status, status.upper() or "—")
    p.rows.append(("STATUS", word))
    if what:
        p.rows.append(("WHAT", what[:26]))
    if target:
        p.rows.append(("TO", target[:26].upper()))
    if recent.get("url"):
        p.rows.append(("FETCH", str(recent["url"])[:26]))
    p.tone = "ok" if status in ("landed", "proposed") else \
        ("warn" if status in ("held", "refused", "vetoed") else "idle")
    if status == "landed":
        p.line = f"{what or 'It'} landed on {target or 'the board'}, sir." \
            if what else "The last throw landed, sir."
    elif status == "held":
        p.line = (f"I'm holding {what or 'the last throw'}, sir; "
                  f"{target or 'the target'} wasn't answering.")
    elif status == "proposed":
        p.line = f"{what or 'It'} is waiting on your yes, sir."
    else:
        p.line = "Nothing landed, sir."
    return p


# ------------------------------------------------------------- assembly
def board_state(*, health: Optional[Callable] = None,
                sessions: Optional[Callable] = None,
                turns: Optional[Callable] = None,
                focus: Optional[Callable] = None,
                quiet: Optional[Callable] = None,
                presence: Optional[Callable] = None,
                canvas: Optional[Callable] = None,
                schedule: Optional[Callable] = None,
                tasks: Optional[Callable] = None,
                cast: Optional[Callable] = None,
                now: Optional[float] = None) -> BoardState:
    """Compose the whole Board from injected providers (each a zero-arg
    callable, each optional, none of them allowed to sink the board).

      health    -> health.Snapshot            sessions -> [SessionInfo]
      turns     -> [ledger record dict]       focus    -> FocusSession
      quiet     -> QuietPolicy                presence -> PresenceSentinel
      canvas    -> [str] Canvas due lines     schedule -> [timekeeper Item]
      tasks     -> {project slug: state}      cast     -> gesturecast.recent()

    Nothing here does I/O of its own: that is the caller's job, and it is
    why the whole layer unit-tests with fakes and no network.
    """
    at = time.time() if now is None else float(now)
    q = _safe(quiet, "quiet")
    pres = _safe(presence, "presence")
    reason = ""
    held = 0
    if q is not None:
        try:
            reason = str(q.reason() or "")
            held = len(list(getattr(q, "held", ()) or ()))
        except Exception:                   # noqa: BLE001 - provider boundary
            log.debug("board: quiet read failed", exc_info=True)
    state_word, configured = "", False
    if pres is not None:
        state_word = str(getattr(pres, "state", "") or "")
        configured = bool(getattr(pres, "configured", False))
    panels = [
        vitals_panel(_safe(health, "health")),
        turns_panel(_safe(turns, "turns") or []),
        sessions_panel(_safe(sessions, "sessions") or [],
                       _safe(tasks, "tasks") or {}),
        deadlines_panel(_safe(schedule, "schedule") or [],
                        _safe(canvas, "canvas") or [], at),
        focus_panel(_safe(focus, "focus")),
        quiet_panel(reason, state_word, held, configured),
    ]
    thrown = cast_panel(_safe(cast, "cast"), at)
    if thrown is not None:
        panels.append(thrown)
    order = {k: i for i, k in enumerate(PANEL_ORDER + OPTIONAL_PANELS)}
    panels.sort(key=lambda p: order.get(p.key, len(order)))
    return BoardState(panels=panels, at=at)


def resolve_panel(name: str) -> str:
    """A spoken panel name -> a panel key ("" when it names none). Longest
    alias first so "claude sessions" is not eaten by "sessions"."""
    text = " ".join(str(name or "").lower().split())
    text = text.strip(" .?!,")
    if not text:
        return ""
    if text in PANEL_ALIASES:
        return PANEL_ALIASES[text]
    for alias in sorted(PANEL_ALIASES, key=len, reverse=True):
        if alias in text:
            return PANEL_ALIASES[alias]
    return ""


def panel_line(state: BoardState, name: str) -> str:
    """The one-line spoken read for "focus on the sessions" — the whole
    point of the verb (a highlight animation says nothing out loud)."""
    key = resolve_panel(name)
    if not key:
        return ""
    panel = state.get(key)
    return panel.line if panel is not None else ""


def board_text(state: BoardState) -> str:
    """The Board as plain text, for `jarvis board` over SSH. Same state,
    same order, no Tk — this is what proved the layer before the window
    existed."""
    out = []
    for p in state.panels:
        head = p.title if p.tone != "off" else f"{p.title} (unavailable)"
        out.append(head)
        for label, value in p.rows:
            out.append(f"  {label:<10} {value}")
        if not p.rows:
            out.append("  --")
    return "\n".join(out)


# ------------------------------------------------------------- the feed
class BoardFeed:
    """Poll thread: calls `state_fn()` every `interval` seconds and
    publishes a BoardUpdate. Joinable, start/stop the same shape as
    jarvis/deadlines.py — no state file, because a board is a view and a
    stale view on restart would be a lie.

    `state_fn` runs OFF the Tk thread on purpose: it spawns nvidia-smi
    (health.snapshot has a 5 s timeout) and reads the ledger off disk."""

    def __init__(self, state_fn: Callable[[], BoardState],
                 interval: float = POLL_S,
                 publish: Callable = bus.publish):
        self._state_fn = state_fn
        self.interval = max(0.5, float(interval))
        self._publish = publish
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tick(self) -> Optional[BoardState]:
        """One poll: compute and publish. Never raises — a wedged provider
        must not kill the thread and leave a frozen board."""
        try:
            state = self._state_fn()
        except Exception:                   # noqa: BLE001 - provider boundary
            log.debug("board feed: state provider failed", exc_info=True)
            return None
        if state is None:
            return None
        try:
            self._publish(BoardUpdate(state=state))
        except Exception:                   # noqa: BLE001 - bus boundary
            log.debug("board feed: publish failed", exc_info=True)
        return state

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="board-feed",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self.interval)
