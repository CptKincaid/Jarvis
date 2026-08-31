"""The Aside: one volunteered sentence after an answer he was asked for.

    "Alarm for 7:00 am, sir."
    "Incidentally, the Lab 3 report for BIOSENSORS is due at 11:59 pm that night."

Everything Jarvis says today is request-response. This is the one door
that decides to VOLUNTEER something -- and, more importantly, the budget
that keeps it charming rather than a tic. Two a day, forty-five minutes
apart, only ever hung off a resolved anchor, and one phrase kills it for
the day.

Five rules the design turns on, each learned from a way this feature can
go wrong:

1. **Snapshot data only.** ``consider()`` runs on the reply path of a
   spoken turn. ``canvas.fetch_due`` is a live REST round-trip and
   ``deadlines._canvas_items`` calls it, so neither is ever touched here:
   the engine reads the last result ``DeadlineHeadsUp`` stashed on its own
   thread (``DeadlineHeadsUp.snapshot()``) plus ``cal.events()``, which is
   the calendar's cache and never fetches. A five-minute-old snapshot is
   indistinguishable from a fresh one in a spoken sentence; a Canvas
   outage behind a spoken turn is not.

2. **Never proactive.** ``app._say(proactive=True)`` HOLDS the line during
   quiet hours and drains it into a catch-up digest later. "Incidentally,
   the lab report is due at 11:59 that night", arriving three hours later
   with no question in front of it, is the exact opposite of the effect.
   So the quiet gate is checked HERE (``quiet.should_hold()``), a held
   aside is dropped silently and costs no budget, and what survives is
   spoken down the plain non-proactive path, behind the answer it follows.

3. **One ledger, two doors.** ``DeadlineHeadsUp`` already files a spoken
   timekeeper reminder ``lead_hours`` before each Canvas item. Without a
   cross-check the same lab report earns a reminder at 09:00 and an aside
   at 09:20 -- the tic arriving by a second door. Anything whose key is
   already in the deadline / meeting state files is skipped, and the keys
   said here are written in the same shape (str -> ISO, atomic
   ``os.replace``) as jarvis/headsup.py and jarvis/deadlines.py.

4. **A resolved anchor or nothing.** "Set for seven" is only an aside if
   "seven" is a real datetime, so ``consider()`` takes the STRUCTURED
   result of the action (the timekeeper ``Item`` with its epoch), never
   the reply text. No anchor -> no aside; guessing from the words is how
   an aside lands on the wrong day.

5. **A budget, and a kill phrase.** Two a day with a 45-minute gap, both
   config (``aside.per_day`` / ``aside.gap_min``). "No more asides" zeros
   the day's bucket and is COUNTED (``aside: silenced``, read by
   jarvis/dayreview.py): if he is being silenced most days the feature is
   wrong, and the log is how that is found out.

Off by default (``aside.enabled``). The engine is pure enough to replay
offline: scripts/aside_dryrun.py prints a transcript of what it would
have said, turn by turn, which is the only way to judge the templates.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.location import clock_words

log = get_logger("aside")

DEFAULT_PER_DAY = 2
DEFAULT_GAP_MIN = 45
# How far past the anchor an item may sit and still be "that night". A
# whole week is too far: an alarm for 7 am has nothing to do with a
# deadline two evenings later, and saying so is the tic.
DEFAULT_HORIZON_HOURS = 18
# Anchors that are not plans. The dry-run transcript's first wince was
# "set a timer for ten minutes" earning "Incidentally, you have BIOSENSORS
# lecture at 9:10 that morning": a countdown timer is a kitchen device,
# not a point in the day he has just pointed at, so the lecture an hour
# later is a non-sequitur hung off it. Alarms and reminders name a future
# moment he is planning around; timers do not.
SKIP_KINDS = ("timer",)
# The kill phrases. Deliberately NOT "stop that": that is already barge-in
# and read-aloud steering (both shipped), and an ambiguous kill is worse
# than no kill at all.
KILL_PHRASES = ("no more asides", "no more of that", "thats enough of that",
                "stop the asides", "enough asides", "no more remarks")
KILL_LINE = "Very good, sir. No more asides today."


# ------------------------------------------------------------- templates
# Hand-written, and iterated against scripts/aside_dryrun.py rather than
# against a test. The opener rotates by how many asides the day has already
# had, so the second of a day never opens like the first -- the fastest way
# to sound like a tic is to say "Incidentally" twice before lunch.
OPENERS = ("Incidentally, {clause}.",
           "Worth mentioning, sir: {clause}.",
           "{clause_cap}, sir.")


def _daypart(dt: datetime) -> str:
    if dt.hour >= 21 or dt.hour < 4:
        return "night"
    if dt.hour >= 17:
        return "evening"
    if dt.hour >= 12:
        return "afternoon"
    return "morning"


def _when_words(anchor: datetime, when: datetime) -> str:
    """"at 11:59 pm that night" / "at 9:10 am the following morning" -- the
    candidate said RELATIVE to the thing he has just set, which is why an
    aside reads as one sentence rather than as two unrelated facts."""
    clock = clock_words(when)
    days = (when.date() - anchor.date()).days
    part = _daypart(when)
    if days <= 0:
        return f"at {clock} that {part}"
    if days == 1:
        return f"at {clock} the following {part}"
    return f"at {clock} on {when.strftime('%A')}"


class Candidate:
    """One thing worth volunteering, already resolved to a local datetime."""

    __slots__ = ("key", "keys", "what", "when", "kind")

    def __init__(self, key: str, keys: tuple, what: str, when: datetime, kind: str):
        self.key = key            # our own said-ledger key
        self.keys = keys          # keys it may already carry in the OTHER ledgers
        self.what = what
        self.when = when
        self.kind = kind          # "deadline" | "exam" | "event"

    def clause(self, anchor: datetime) -> str:
        # No article: Canvas titles are proper-ish names ("Problem set 6",
        # "Lab 3 report"), and "the Problem set 6 for THERMO" was the second
        # wince in the dry run. deadlines.py says it the same way.
        when = _when_words(anchor, self.when)
        if self.kind == "exam":
            return f"{self.what} is {when}"
        return f"{self.what} is due {when}"


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


class AsideEngine:
    """``consider(turn_text, reply_text, action) -> str | None``."""

    def __init__(self, cfg=None, get_calendar: Callable = None,
                 get_deadlines: Callable = None, quiet=None,
                 state_path: Optional[Path] = None,
                 filed_paths: tuple = (), now: Callable = None):
        self._cfg = cfg
        self._get_calendar = get_calendar
        self._get_deadlines = get_deadlines
        self._quiet = quiet
        self._state_path = Path(state_path) if state_path else None
        self._filed_paths = tuple(Path(p) for p in filed_paths or ())
        self._now = now or (lambda: datetime.now().astimezone())
        self._lock = threading.Lock()
        self._state = self._load()

    # ------------------------------------------------------------ config
    def _opt(self, key: str, default):
        cfg = self._cfg
        if cfg is None:
            return default
        try:
            val = cfg.get(key, default)
        except Exception:  # noqa: BLE001 - a config read must never break a turn
            log.debug("aside: config read failed for %s", key, exc_info=True)
            return default
        return default if val is None else val

    @property
    def enabled(self) -> bool:
        return bool(self._opt("aside.enabled", False))

    def _per_day(self) -> int:
        try:
            return max(0, int(self._opt("aside.per_day", DEFAULT_PER_DAY)))
        except (TypeError, ValueError):
            return DEFAULT_PER_DAY

    def _gap(self) -> timedelta:
        try:
            return timedelta(minutes=max(0.0, float(self._opt("aside.gap_min",
                                                              DEFAULT_GAP_MIN))))
        except (TypeError, ValueError):
            return timedelta(minutes=DEFAULT_GAP_MIN)

    def _horizon(self) -> timedelta:
        try:
            return timedelta(hours=max(1.0, float(self._opt("aside.horizon_hours",
                                                            DEFAULT_HORIZON_HOURS))))
        except (TypeError, ValueError):
            return timedelta(hours=DEFAULT_HORIZON_HOURS)

    # ------------------------------------------------------------- state
    # Same shape and the same scars as jarvis/headsup.py: the said-keys are
    # str -> ISO-string only, because the values are compared as strings when
    # pruning and one hand-edited null used to raise on every pass before the
    # save could ever run.
    @staticmethod
    def _blank() -> dict:
        return {"said": {}, "spoken_at": [], "silenced_on": "", "silences": []}

    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if not isinstance(data, dict):
                    return self._blank()
                said = data.get("said")
                said = {k: v for k, v in said.items()
                        if isinstance(k, str) and isinstance(v, str)} \
                    if isinstance(said, dict) else {}
                raw_spoken = data.get("spoken_at")
                spoken = [v for v in raw_spoken if isinstance(v, str)] \
                    if isinstance(raw_spoken, list) else []
                raw_sil = data.get("silences")
                sil = [v for v in raw_sil if isinstance(v, str)] \
                    if isinstance(raw_sil, list) else []
                on = data.get("silenced_on")
                return {"said": said, "spoken_at": spoken,
                        "silenced_on": on if isinstance(on, str) else "",
                        "silences": sil}
        except (OSError, ValueError):
            log.debug("aside state unreadable", exc_info=True)
        return self._blank()

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._state))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("aside state save failed", exc_info=True)

    def _prune(self, now: datetime) -> None:
        cutoff = _utc_iso(now - timedelta(days=2))
        said = self._state["said"]
        for key in [k for k, v in said.items()
                    if not isinstance(v, str) or v < cutoff]:
            said.pop(key, None)
        today = now.date().isoformat()
        self._state["spoken_at"] = [v for v in self._state["spoken_at"]
                                    if v[:10] == today]
        self._state["silences"] = self._state["silences"][-30:]

    def _filed_keys(self) -> set:
        """Every key the deadline / meeting heads-up has already filed a
        SPOKEN reminder for. Re-read each turn: those threads write their
        files behind our back, and a stale copy is a double nudge."""
        keys: set = set()
        for path in self._filed_paths:
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                log.debug("aside: %s unreadable", path.name, exc_info=True)
                continue
            if isinstance(data, dict):
                keys.update(k for k in data if isinstance(k, str))
        return keys

    # ------------------------------------------------------------ budget
    def _spent_today(self, now: datetime) -> list:
        today = now.date().isoformat()
        return [v for v in self._state["spoken_at"] if v[:10] == today]

    def _budget_reason(self, now: datetime) -> str:
        """"" when he may speak, else why not (for the dry-run transcript)."""
        if self._state.get("silenced_on") == now.date().isoformat():
            return "silenced for the day"
        spent = self._spent_today(now)
        per_day = self._per_day()
        if len(spent) >= per_day:
            return f"budget spent ({len(spent)}/{per_day} today)"
        if spent:
            try:
                last = datetime.fromisoformat(max(spent))
            except ValueError:
                return ""
            gap = self._gap()
            if now - last < gap:
                left = gap - (now - last)
                return (f"within the {int(gap.total_seconds() // 60)}-minute gap "
                        f"({int(left.total_seconds() // 60) + 1} min to go)")
        return ""

    # ------------------------------------------------------------ anchor
    @staticmethod
    def anchor_of(action) -> Optional[datetime]:
        """The datetime the action RESOLVED to, or None.

        A timekeeper ``Item`` (alarm / timer / reminder) carries ``due`` as
        an epoch; a calendar ``Event`` carries an aware ``start``; a tool
        that hands back a plain dict may say ``due`` or ``when``. Anything
        else -- a reply string that merely contains the word "seven"
        included -- yields None and no aside is volunteered. Guessing the
        anchor from the words is how an aside lands on the wrong day."""
        if action is None:
            return None
        for attr in ("due", "start", "when"):
            value = action.get(attr) if isinstance(action, dict) \
                else getattr(action, attr, None)
            if isinstance(value, datetime):
                return value if value.tzinfo else value.astimezone()
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value > 0:
                return datetime.fromtimestamp(float(value)).astimezone()
        return None

    # -------------------------------------------------------- candidates
    def _deadline_items(self) -> list:
        """The last result DeadlineHeadsUp's own thread stashed. NEVER
        canvas.fetch_due: that is a live REST call and this runs inside a
        spoken turn."""
        src = self._get_deadlines() if callable(self._get_deadlines) \
            else self._get_deadlines
        if src is None:
            return []
        try:
            snap = src.snapshot() if hasattr(src, "snapshot") else src
            return list(snap or [])
        except Exception:  # noqa: BLE001 - source boundary
            log.debug("aside: deadline snapshot unavailable", exc_info=True)
            return []

    def _events(self) -> list:
        cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
        if cal is None:
            return []
        try:
            conf = getattr(cal, "configured", True)
            if callable(conf):
                conf = conf()
            if not conf:
                return []
            return list(cal.events())          # the cache; never a fetch
        except Exception:  # noqa: BLE001 - source boundary
            log.debug("aside: calendar unavailable", exc_info=True)
            return []

    def candidates(self, anchor: datetime, now: datetime) -> list:
        """Everything sitting between the anchor and the horizon, soonest
        first. Pure: both sources are already snapshots."""
        from jarvis.tools.canvas import exam_kind
        horizon = anchor + self._horizon()
        out: list = []
        for it in self._deadline_items():
            if not isinstance(it, dict):
                continue
            when = it.get("due")
            title = " ".join(str(it.get("title") or "").split())
            if not isinstance(when, datetime) or not title:
                continue
            when = when.astimezone(anchor.tzinfo) if when.tzinfo else when.astimezone()
            if not (anchor < when <= horizon) or when <= now:
                continue
            course = " ".join(str(it.get("course") or "").split())
            what = f"{title} for {course}" if course else title
            key = f"{course}|{title}|{_utc_iso(when)}"
            out.append(Candidate(
                key,
                # deadlines.py files a deadline under course|title|due and an
                # exam eve under exam|course|title|when
                (key, f"exam|{course}|{title}|{_utc_iso(when)}"),
                what, when, "exam" if exam_kind(title) else "deadline"))
        for ev in self._events():
            when = getattr(ev, "start", None)
            title = " ".join(str(getattr(ev, "title", "") or "").split())
            if not isinstance(when, datetime) or not title:
                continue
            if getattr(ev, "all_day", False):
                continue                        # nothing starts at midnight
            # Exams only. The dry run's third wince was "you have BIOSENSORS
            # lecture at 9:10 that morning" volunteered before a lecture he
            # attends every single week -- not news, and the meeting heads-up
            # is going to say it anyway ten minutes before. An exam is the one
            # calendar row worth interrupting an answer for.
            if not exam_kind(title):
                continue
            raw_start = when.isoformat()
            when = when.astimezone(anchor.tzinfo) if when.tzinfo else when.astimezone()
            if not (anchor < when <= horizon) or when <= now:
                continue
            key = f"{title}|{when.isoformat()}"
            out.append(Candidate(
                key,
                # headsup.py files a meeting under title|start (the event's own
                # isoformat), deadlines.py an exam eve under exam||title|when
                (key, f"{title}|{raw_start}", f"exam||{title}|{_utc_iso(when)}"),
                title, when, "exam"))
        out.sort(key=lambda c: c.when)
        return out

    # ------------------------------------------------------------ decide
    def explain(self, turn_text: str, reply_text: str, action) -> tuple:
        """``(line_or_None, why)`` -- the whole decision, no side effects.
        On a line, ``why`` is the said-ledger key. The dry-run harness
        prints ``why`` for every turn he stayed quiet on, which is the only
        way to read the budget as a transcript instead of as a test."""
        if not self.enabled:
            return None, "aside.enabled is off"
        anchor = self.anchor_of(action)
        if anchor is None:
            return None, "no resolved anchor in the action result"
        kind = action.get("kind") if isinstance(action, dict) \
            else getattr(action, "kind", "")
        if str(kind or "").lower() in SKIP_KINDS:
            return None, f"a {kind} is a countdown, not a plan"
        quiet = self._quiet
        if quiet is not None:
            try:
                if quiet.should_hold():
                    # Dropped, not held: a held aside is drained into a
                    # digest hours later, detached from its question.
                    reason = ""
                    try:
                        reason = quiet.reason() or ""
                    except Exception:  # noqa: BLE001
                        reason = ""
                    return None, f"quiet ({reason or 'holding'})"
            except Exception:  # noqa: BLE001 - a probe failure must not break a turn
                log.debug("aside: quiet gate failed", exc_info=True)
        now = self._now()
        with self._lock:
            self._prune(now)
            budget = self._budget_reason(now)
            said = set(self._state["said"])
            spent_today = len(self._spent_today(now))
        if budget:
            return None, budget
        filed = self._filed_keys()
        skipped = ""
        for cand in self.candidates(anchor, now):
            # `continue`, not `return`: a candidate already said (or already
            # owned by the heads-up) disqualifies ITSELF, not the turn. The
            # first dry run returned here and a second day of "wake me at
            # seven" went silent because Monday's lecture was in the ledger.
            if cand.key in said:
                skipped = f"already volunteered: {cand.what}"
                continue
            if any(k in filed for k in cand.keys):
                # The timekeeper is already going to say this one out loud.
                skipped = f"the heads-up already has it: {cand.what}"
                continue
            clause = cand.clause(anchor)
            template = OPENERS[spent_today % len(OPENERS)]
            return template.format(clause=clause,
                                   clause_cap=clause[:1].upper() + clause[1:]), cand.key
        return None, skipped or "nothing between the anchor and the horizon"

    def consider(self, turn_text: str, reply_text: str, action) -> Optional[str]:
        """One sentence to say after the answer, or None. Spends a budget
        token and files the said-key only when it returns a line."""
        try:
            line, why = self.explain(turn_text, reply_text, action)
        except Exception:  # noqa: BLE001 - an aside must never break a turn
            log.exception("aside: consider failed")
            return None
        if line is None:
            log.debug("aside: quiet this turn (%s)", why)
            return None
        now = self._now()
        with self._lock:
            self._state["said"][why] = _utc_iso(now)     # `why` is the key here
            self._state["spoken_at"].append(now.isoformat())
            self._save()
        log.info("aside: %r", line)
        return line

    # -------------------------------------------------------------- kill
    def silence(self, phrase: str = "") -> str:
        """"No more asides": zero today's bucket. Counted deliberately --
        jarvis/dayreview.py reads this line, and a feature that is silenced
        most days is a feature that is wrong."""
        now = self._now()
        with self._lock:
            self._state["silenced_on"] = now.date().isoformat()
            self._state["silences"].append(now.isoformat())
            self._prune(now)
            self._save()
            silenced = len(self._state["silences"])
        log.info("aside: silenced for the day by %r (%d on record)",
                 (phrase or "").strip() or "the user", silenced)
        return KILL_LINE

    def silenced_today(self, now: Optional[datetime] = None) -> bool:
        when = now or self._now()
        return self._state.get("silenced_on") == when.date().isoformat()


_APOSTROPHE_RX = re.compile(r"['‘’]")
_PUNCT_RX = re.compile(r"[^a-z0-9\s]+")


def kill_phrase(text: str) -> bool:
    """True when the utterance is the specific aside kill. Never "stop
    that": that already means barge-in and read-aloud steering, and the
    router cannot tell three meanings of one phrase apart.

    Apostrophes are DELETED rather than spaced out -- "that's" must become
    "thats", not "that s", or the phrase Whisper actually writes down never
    matches the phrase in KILL_PHRASES."""
    lowered = _APOSTROPHE_RX.sub("", str(text or "").lower())
    lowered = " ".join(_PUNCT_RX.sub(" ", lowered).split())
    return any(p in lowered for p in KILL_PHRASES)
