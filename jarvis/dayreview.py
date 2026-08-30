"""Nightly self-review: what yesterday looked like from Jarvis's own log.

Every day the app writes a few thousand lines to jarvis.log (761 WARNINGs on
2026-08-29 alone) and one record per voice turn to turns.jsonl. Nobody reads
them. This module reads them for him and reduces a day to a dozen counts --
turns, median and worst wait, aborts and "Was that for me?"s, speaker-gate
rejections, turn-watchdog releases, tool-handler exceptions, TTS fallbacks,
model reloads -- so that "how did yesterday go" gets an honest two-sentence
answer, the first-wake briefing opens with it, and the full table can be
posted to Discord for when he is away.

Everything here is pure and offline: `summarize_day(log_path, turns_path,
day)` reads files and returns a dict. Two quirks of the inputs shape it:

* The log has NO date -- lines are stamped HH:MM:SS.mmm only. The file is
  split into days where the clock runs backwards (midnight), and each slice
  is dated by matching its "turn:" lines against the epoch `at` stamps in
  turns.jsonl; a slice with no turn to anchor it is counted back from the
  next dated slice (or the file's mtime for the last one).
* Other processes write the same file: before the conftest firewall, a
  test run left a fake "cuda gone" traceback in the live log (line 4407 on
  2026-08-29). Within a day, only the lines after the app's own boot marker
  ("tools registered:", logged once per JarvisApp) are counted whenever a
  boot happened that day; a day the app ran across from the night before
  is kept whole.

`DayReviewer` is the nightly timer: a small thread that files the digest
of any day that has ended to MEMORY_DIR/reviews/<date>.json (the log dir is
tmpfs and is wiped at boot; the review must not be) and hands the table to
the Alerts hub, which posts it to Discord only when that channel is
configured. Started in start_assistant, stopped in stop_assistant.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

from jarvis.logs import get_logger

log = get_logger("dayreview")

# Logged once per JarvisApp construction (app._register_tools) and by no
# other process, so it marks a boot of the real app.
BOOT_MARKER = "tools registered:"
_LINE_RX = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) (\S+) (\S+) (.*)$")
# A clock that goes back more than this between consecutive lines is a new
# day; smaller regressions happen when two threads log out of order.
_DAY_BREAK_S = 3600.0
REVIEW_KEEP_DAYS = 60
TICK_S = 900.0                # nightly timer resolution: a quarter hour
FIRST_TICK_S = 30.0

# What gets counted, by the exact log lines the modules write. Each is a
# (key, regex) pair on the message part of the line; the module names are
# not matched so a ported line keeps counting after a refactor.
COUNTERS: tuple[tuple[str, re.Pattern], ...] = (
    # app._process_audio -> speaker.filter_segments: the transcript gate
    # dropped a whole clip (no window matched), or the fail-shut path did.
    # NOT the raw "speaker verify: ... REJECT" line: the hotword's wake check
    # and the recorder's 1 Hz poll log that too (53 on 2026-08-29 against a
    # handful of dropped clips), so it is a separate, table-only count.
    ("speaker_rejections", re.compile(r"^segment filter: 0/\d+ windows matched|"
                                      r"^speaker verify FAILED SHUT")),
    ("verify_rejects", re.compile(r"^speaker verify: score=.* REJECT$")),
    # hotword._speaker_ok: the wake word itself was refused (a TV, a guest)
    ("wake_suppressed", re.compile(r"^wake suppressed: speaker score")),
    ("watchdog_releases", re.compile(r"^turn watchdog fired after")),
    # registry.call (a tool raised) and commander (a Tier-1 handler raised)
    # (command names carry spaces: "handler last mail failed")
    ("tool_exceptions", re.compile(r"^tool \S+ failed$|^handler .+ failed$")),
    # tts: Fish credentials / f5 sidecar / XTTS load / a chunk falling back
    # to edge, and Fish being retired for the session
    ("tts_fallbacks", re.compile(r"falling back to|^fish retired")),
    ("uncertain", re.compile(r"^Uncertain intent \(conf=")),
    ("ignored", re.compile(r"^Ignored \(background chat")),
    ("boots", re.compile(re.escape(BOOT_MARKER))),
)
_RESIDENT_RX = re.compile(r"^ollama: \S+ resident \(load ([\d.]+) s\)")
# A load this long after a boot is the model having been evicted (another
# process took the unified memory) and pulled back by the residency thread.
_RELOAD_GRACE_S = 180.0
_TURN_LINE_RX = re.compile(r"^turn(?:\[[^\]]*\])?: ")


# ------------------------------------------------------------ log reading
def _parse(line: str):
    m = _LINE_RX.match(line)
    if not m:
        return None
    hh, mm, ss, ms, name, level, msg = m.groups()
    secs = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
    return secs, name, level, msg


def read_log_lines(log_path) -> list[str]:
    """The rotated files oldest first, then the live file: one stream in
    write order."""
    log_path = Path(log_path)
    out: list[str] = []
    for p in (log_path.with_name(log_path.name + ".2"),
              log_path.with_name(log_path.name + ".1"), log_path):
        try:
            out.extend(p.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
    return out


def split_days(lines: Iterable[str]) -> list[list[str]]:
    """Slice the stream where the clock runs backwards (midnight). Lines
    without a timestamp (tracebacks) stay with the slice they follow."""
    days: list[list[str]] = []
    cur: list[str] = []
    last: Optional[float] = None
    for line in lines:
        parsed = _parse(line)
        if parsed is not None:
            secs = parsed[0]
            if last is not None and last - secs > _DAY_BREAK_S:
                days.append(cur)
                cur = []
            last = secs
        cur.append(line)
    if cur:
        days.append(cur)
    return days


def read_turn_records(turns_path) -> list[dict]:
    recs = []
    try:
        text = Path(turns_path).read_text(encoding="utf-8")
    except OSError:
        return recs
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("at"), (int, float)):
            recs.append(rec)
    return recs


def _turn_anchors(records: Iterable[dict]) -> dict[int, date]:
    """Local wall-clock second -> date, for every ledger record."""
    anchors: dict[int, date] = {}
    for rec in records:
        dt = datetime.fromtimestamp(rec["at"])
        anchors[dt.hour * 3600 + dt.minute * 60 + dt.second] = dt.date()
    return anchors


def date_segments(segments: list[list[str]], records: Iterable[dict],
                  last_day: date) -> list[tuple[date, list[str]]]:
    """Attach a date to each day-slice: from a ledger anchor when the slice
    holds a "turn:" line the ledger also recorded (same second, +-1 s);
    otherwise counted back one day per slice from the next dated slice,
    the last slice defaulting to `last_day` (the log file's mtime)."""
    anchors = _turn_anchors(records)
    dated: list[Optional[date]] = [None] * len(segments)
    for i, seg in enumerate(segments):
        for line in seg:
            parsed = _parse(line)
            if parsed is None or not _TURN_LINE_RX.match(parsed[3]):
                continue
            sec = int(parsed[0])
            for cand in (sec, sec - 1, sec + 1):
                if cand in anchors:
                    dated[i] = anchors[cand]
                    break
            if dated[i] is not None:
                break
    if segments and dated[-1] is None:
        dated[-1] = last_day
    for i in range(len(segments) - 2, -1, -1):
        if dated[i] is None:
            dated[i] = dated[i + 1] - timedelta(days=1)
    return [(d, seg) for d, seg in zip(dated, segments) if d is not None]


def boot_cut(lines: list[str]) -> list[str]:
    """Drop everything before the first app boot of the day, when there was
    one: the lines before it came from other processes writing the same
    file (a pre-firewall test run, a voice_check), not from Jarvis."""
    for i, line in enumerate(lines):
        if BOOT_MARKER in line:
            return lines[i:]
    return lines


# ---------------------------------------------------------------- counting
def count_events(lines: list[str]) -> dict:
    counts = {key: 0 for key, _ in COUNTERS}
    counts.update({"residency_reloads": 0, "errors": 0, "warnings": 0,
                   "log_lines": 0})
    notes: dict[str, int] = {}
    last_boot: Optional[float] = None
    for line in lines:
        parsed = _parse(line)
        if parsed is None:
            continue
        secs, _name, level, msg = parsed
        counts["log_lines"] += 1
        if level == "ERROR":
            counts["errors"] += 1
            notes[msg[:90]] = notes.get(msg[:90], 0) + 1
        elif level == "WARNING":
            counts["warnings"] += 1
        for key, rx in COUNTERS:
            if rx.search(msg):
                counts[key] += 1
                if key == "boots":
                    last_boot = secs
                elif key in ("tool_exceptions", "watchdog_releases",
                             "tts_fallbacks"):
                    notes[msg[:90]] = notes.get(msg[:90], 0) + 1
        m = _RESIDENT_RX.match(msg)
        if m:
            try:
                load_s = float(m.group(1))
            except ValueError:
                load_s = 0.0
            fresh_boot = last_boot is not None and 0 <= secs - last_boot < _RELOAD_GRACE_S
            if load_s >= 0.5 and not fresh_boot:
                counts["residency_reloads"] += 1
    counts["notes"] = [f"{n}x {m}" if n > 1 else m
                       for m, n in sorted(notes.items(), key=lambda kv: -kv[1])[:6]]
    return counts


def turn_stats(records: Iterable[dict], day: date) -> dict:
    """Mirror of diagnostics_text(): a turn is any ledger record that is not
    an abort (an abort is the follow-up window closing on silence)."""
    waits, turns, aborts, rejected, timeouts, uncertain = [], 0, 0, 0, 0, 0
    kept = []
    for rec in records:
        if datetime.fromtimestamp(rec["at"]).date() != day:
            continue
        kept.append(rec)
        outcome = str(rec.get("outcome", ""))
        if outcome == "abort":
            aborts += 1
            continue
        turns += 1
        if outcome.startswith("rejected"):
            rejected += 1
        elif outcome == "timeout":
            timeouts += 1
        elif outcome == "uncertain":
            uncertain += 1
        wait = rec.get("wait")
        if isinstance(wait, (int, float)):
            waits.append(float(wait))
    return {"turns": turns, "aborts": aborts, "rejected": rejected,
            "timeouts": timeouts, "uncertain_turns": uncertain,
            "median_wait_s": round(statistics.median(waits), 2) if waits else None,
            "worst_wait_s": round(max(waits), 2) if waits else None,
            "answered": len(waits), "turn_records": kept}


def summarize_day(log_path, turns_path, day: date, now: Optional[Callable] = None) -> dict:
    """The digest for `day`: ledger stats + log counts, or a digest that says
    so when neither file has anything for that day."""
    log_path, turns_path = Path(log_path), Path(turns_path)
    records = read_turn_records(turns_path)
    try:
        last_day = datetime.fromtimestamp(log_path.stat().st_mtime).date()
    except OSError:
        last_day = (now or datetime.now)().date()
    lines = read_log_lines(log_path)
    day_lines: list[str] = []
    for d, seg in date_segments(split_days(lines), records, last_day):
        if d == day:
            day_lines.extend(seg)
    day_lines = boot_cut(day_lines)
    digest = {"day": day.isoformat(),
              "generated_at": (now or datetime.now)().isoformat(timespec="seconds"),
              "has_log": bool(day_lines)}
    digest.update(turn_stats(records, day))
    digest.update(count_events(day_lines))
    digest["has_data"] = digest["has_log"] or bool(digest["turn_records"])
    return digest


# ---------------------------------------------------------------- wording
def _n(count: int, noun: str, plural: Optional[str] = None) -> str:
    return f"{count} {noun if count == 1 else (plural or noun + 's')}"


def _times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def _secs(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.1f}"


def spoken_line(digest: dict, label: str = "Yesterday", name: str = "sir") -> str:
    """Two sentences in character: the numbers, then what went wrong."""
    if not digest.get("has_data"):
        return ""
    turns = int(digest.get("turns") or 0)
    if turns:
        first = f"{label}: {_n(turns, 'turn')}"
        med, worst = digest.get("median_wait_s"), digest.get("worst_wait_s")
        if med is not None:
            first += f", median wait {_secs(med)} seconds"
            if worst is not None and worst > med + 0.05:
                first += f", worst {_secs(worst)}"
        first += "."
    else:
        first = f"{label}: no voice turns."
    problems = []
    if digest.get("speaker_rejections"):
        problems.append(f"I dropped {_n(digest['speaker_rejections'], 'clip')} of "
                        f"yours at the speaker gate")
    if digest.get("uncertain"):
        problems.append(f"asked whether you meant me {_times(digest['uncertain'])}")
    if digest.get("watchdog_releases"):
        problems.append(f"the turn watchdog let go {_times(digest['watchdog_releases'])}")
    if digest.get("tool_exceptions"):
        problems.append(f"{_n(digest['tool_exceptions'], 'tool call')} failed")
    if digest.get("tts_fallbacks"):
        problems.append(f"the voice fell back {_times(digest['tts_fallbacks'])}")
    if digest.get("residency_reloads"):
        problems.append(f"the model had to be reloaded {_times(digest['residency_reloads'])}")
    if digest.get("timeouts"):
        problems.append(f"{_n(digest['timeouts'], 'answer')} never came")
    if not problems:
        second = f"Nothing went wrong that I could see, {name}."
    elif len(problems) == 1:
        second = f"{problems[0][0].upper()}{problems[0][1:]}, {name}."
    else:
        second = (f"{problems[0][0].upper()}{problems[0][1:]}, "
                  + ", ".join(problems[1:-1]) + (", " if len(problems) > 2 else "")
                  + f"and {problems[-1]}, {name}.")
    return f"{first} {second}"


def table(digest: dict) -> str:
    """The full digest as fixed-width text (Discord renders it in a code
    block, the transcript shows it as is)."""
    rows = [("turns", digest.get("turns")),
            ("answered", digest.get("answered")),
            ("median wait", _secs(digest.get("median_wait_s")) + " s" if digest.get("median_wait_s") is not None else "-"),
            ("worst wait", _secs(digest.get("worst_wait_s")) + " s" if digest.get("worst_wait_s") is not None else "-"),
            ("aborts (silent follow-ups)", digest.get("aborts")),
            ("uncertain (\"was that for me?\")", digest.get("uncertain")),
            ("ignored as background", digest.get("ignored")),
            ("rejected clips (ledger)", digest.get("rejected")),
            ("speaker-gate rejections", digest.get("speaker_rejections")),
            ("verify REJECT scores (wake checks, polls)", digest.get("verify_rejects")),
            ("wake words refused", digest.get("wake_suppressed")),
            ("turn timeouts", digest.get("timeouts")),
            ("watchdog releases", digest.get("watchdog_releases")),
            ("tool-handler exceptions", digest.get("tool_exceptions")),
            ("TTS fallbacks", digest.get("tts_fallbacks")),
            ("model reloads", digest.get("residency_reloads")),
            ("app boots", digest.get("boots")),
            ("log errors / warnings", f"{digest.get('errors', 0)} / {digest.get('warnings', 0)}")]
    width = max(len(k) for k, _ in rows)
    lines = [f"Jarvis day review {digest.get('day', '')}"]
    lines += [f"{k.ljust(width)}  {v if v is not None else '-'}" for k, v in rows]
    notes = digest.get("notes") or []
    if notes:
        lines.append("notes:")
        lines += [f"  - {n}" for n in notes]
    return "```\n" + "\n".join(lines) + "\n```"


# ------------------------------------------------------------- persistence
def review_path(reviews_dir, day: date) -> Path:
    return Path(reviews_dir) / f"{day.isoformat()}.json"


def file_review(reviews_dir, digest: dict) -> Optional[Path]:
    """Atomic write; the reviews outlive /tmp because MEMORY_DIR is not
    tmpfs. Returns the path, or None when the write failed."""
    try:
        path = review_path(reviews_dir, date.fromisoformat(digest["day"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(digest, indent=1))
        os.replace(tmp, path)
        return path
    except (OSError, ValueError, KeyError):
        log.debug("day review save failed", exc_info=True)
        return None


def load_review(reviews_dir, day: date) -> Optional[dict]:
    try:
        data = json.loads(review_path(reviews_dir, day).read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def prune_reviews(reviews_dir, keep_days: int = REVIEW_KEEP_DAYS,
                  today: Optional[date] = None) -> int:
    today = today or date.today()
    removed = 0
    try:
        for p in Path(reviews_dir).glob("*.json"):
            try:
                d = date.fromisoformat(p.stem)
            except ValueError:
                continue
            if (today - d).days > keep_days:
                p.unlink(missing_ok=True)
                removed += 1
    except OSError:
        pass
    return removed


# --------------------------------------------------------------- reviewer
class DayReviewer:
    """The nightly timer. `tick()` files the digest of every day in the last
    `lookback` days that has ended and has no review yet, and hands each new
    one to `on_filed(day, digest)` (the app posts the table through Alerts).
    A day with no data at all is filed too -- as an empty digest -- so the
    check is not repeated every quarter hour for a day Jarvis was off."""

    def __init__(self, log_path, turns_path, reviews_dir,
                 on_filed: Optional[Callable[[date, dict], None]] = None,
                 now: Optional[Callable[[], datetime]] = None, lookback: int = 3):
        self.log_path, self.turns_path = Path(log_path), Path(turns_path)
        self.reviews_dir = Path(reviews_dir)
        self.on_filed = on_filed
        self._now = now or datetime.now
        self.lookback = max(1, int(lookback))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def review(self, day: date, refresh: bool = False) -> dict:
        """The digest for `day`: the filed one when it exists (a finished day
        does not change), else computed now and filed if the day is over."""
        if not refresh:
            filed = load_review(self.reviews_dir, day)
            if filed is not None:
                return filed
        digest = summarize_day(self.log_path, self.turns_path, day, now=self._now)
        if day < self._now().date():
            file_review(self.reviews_dir, digest)
        return digest

    def tick(self) -> list[date]:
        today = self._now().date()
        filed: list[date] = []
        for back in range(1, self.lookback + 1):
            day = today - timedelta(days=back)
            if load_review(self.reviews_dir, day) is not None:
                continue
            digest = summarize_day(self.log_path, self.turns_path, day, now=self._now)
            if file_review(self.reviews_dir, digest) is None:
                continue
            filed.append(day)
            log.info("day review filed for %s: %d turns, %d errors", day,
                     digest.get("turns", 0), digest.get("errors", 0))
            if digest.get("has_data") and self.on_filed is not None:
                try:
                    self.on_filed(day, digest)
                except Exception:
                    log.exception("day review on_filed failed for %s", day)
        prune_reviews(self.reviews_dir, today=today)
        return filed

    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again (tests, a config reload), and a dead thread must
        # not block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="dayreview")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # a tick in flight (Discord post, Canvas fetch) must not outlive
            # stop_assistant into the teardown
            t.join(timeout=2.0)

    def _run(self) -> None:
        if self._stop.wait(FIRST_TICK_S):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("day review tick failed")
            if self._stop.wait(TICK_S):
                return


__all__ = ["summarize_day", "spoken_line", "table", "DayReviewer", "file_review",
           "load_review", "review_path", "split_days", "date_segments", "boot_cut",
           "count_events", "turn_stats", "read_log_lines", "BOOT_MARKER"]
