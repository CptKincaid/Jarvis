"""Reasoned dissent: "I would advise against that, sir" -- then he does it.

    "Wake me at two."
    "I would advise against a 2:00 am alarm, sir; your BIOSENSORS lecture
     is at 9:10 am and that leaves you under five hours. Shall I set it
     anyway?"
    "Set it anyway."
    "Setting it anyway, sir."

Four rules, each of which is the difference between J.A.R.V.I.S. and a
nagging appliance:

1. **A named reason from a real row, or silence.** Every source below
   returns the DATA it objected from -- a calendar event, a Canvas
   deadline, an existing timekeeper item, a quiet window -- and says it
   out loud. An objection that cannot name its row is a mood, and moods
   are insufferable. Nothing here guesses.

2. **Cache only.** These run inside a spoken handler, between "wake me at
   two" and the reply. ``CalendarSource.events()`` is the cache (its own
   docstring: never fetches) and the timekeeper is local sqlite; a live
   ``canvas.fetch_due`` would put seconds of network on the turn, so the
   Canvas side reads the snapshot ``DeadlineHeadsUp`` already stashed.

3. **Never twice for the same thing in a day.** The ledger is the same
   shape as jarvis/headsup.py and jarvis/deadlines.py (str -> ISO, atomic
   ``os.replace``): key includes the local date, so the second "wake me at
   two" today is simply obeyed.

4. **Never about anything reversible in under a minute.** A timer, a track
   change, a volume nudge -- objecting to those is the whole failure mode.
   ``for_alarm()`` is the only entry point, deliberately.

The resolution half lives in jarvis/commander.py, and it inverts the
read-back's default on purpose: a destructive read-back DROPS on anything
that is not a clear yes, because doing nothing is the safe end of a
delete. Here the user explicitly asked for the alarm, so ambiguity,
silence and a timeout all RUN it and say so. Refusing to set an alarm
because a reply was mumbled is a worse outcome than an alarm at two.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from jarvis.logs import get_logger
from jarvis.tools.location import clock_words

log = get_logger("objections")

# Under this many hours between now and a small-hours alarm, with something
# on the calendar later that day, he says so.
DEFAULT_SLEEP_FLOOR_H = 5.0
# "the small hours": an alarm here is the one worth a word.
SMALL_HOURS = range(0, 6)
# Two alarms this close together are the same alarm said twice.
DUPLICATE_MIN = 15
# A deadline this far before the alarm is worth naming; further back and
# the two have nothing to do with each other.
CLASH_HOURS = 12

RUN_ANYWAY_LINE = "Setting it anyway, sir."
DROPPED_LINE = "Very good, sir. I'll leave it."


_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten", "eleven", "twelve")


def _hours_words(hours: float) -> str:
    """"under four and a half hours" -- the TIGHTEST true statement.

    ``ceil`` to the nearest half, not round: for a claim of the form "under
    X", a smaller X is the harsher and more accurate one, so 4.5 hours must
    say "under four and a half", never the softer "under five". Spelled out
    because this is read aloud."""
    halves = math.ceil(hours * 2 - 1e-9)
    whole, half = divmod(max(1, halves), 2)
    if whole == 0:
        return "under half an hour"
    if whole == 1 and not half:
        return "under an hour"
    word = _NUMBER_WORDS[whole] if whole < len(_NUMBER_WORDS) else str(whole)
    return f"under {word}{' and a half' if half else ''} hours"


@dataclass
class Objection:
    source: str        # which predicate fired ("sleep window", "duplicate alarm", …)
    reason: str        # the spoken clause, naming the row
    row: str           # the data row itself, for the log and the journal
    key: str           # dedupe identity (the local date is added by the ledger)

    def line(self, what: str) -> str:
        """The whole spoken objection, ending in the question that makes it
        an offer rather than a refusal."""
        return f"I would advise against {what}, sir; {self.reason}. " \
               f"Shall I set it anyway?"


# ------------------------------------------------------------- predicates
def sleep_window(due: datetime, events, now: datetime,
                 floor_h: float = DEFAULT_SLEEP_FLOOR_H) -> Optional[Objection]:
    """A small-hours alarm that leaves under ``floor_h`` before it, with
    something already on the calendar later that day.

    Both halves matter. Without the hours there is no reason, only a
    preference about bedtimes; without the calendar row there is nothing to
    name, and "you'll be tired" is exactly the insufferable objection this
    module exists not to make."""
    if due.hour not in SMALL_HOURS:
        return None
    hours = (due - now).total_seconds() / 3600.0
    if hours <= 0 or hours >= floor_h:
        return None
    first = _first_event_after(events, due, same_day_as=due)
    if first is None:
        return None
    title = " ".join(str(getattr(first, "title", "") or "").split())
    start = getattr(first, "start", None)
    return Objection(
        source="sleep window",
        reason=f"your {title} is at {clock_words(start)} and that leaves you "
               f"{_hours_words(hours)}",
        row=f"calendar {title} {start.isoformat()}",
        key=f"sleep|{title}|{start.isoformat()}")


def duplicate_item(due: datetime, items, kind: str = "alarm",
                   tolerance_min: int = DUPLICATE_MIN) -> Optional[Objection]:
    """One already set within ``tolerance_min``. The commonest real mistake
    of the three, and the only one where the right answer is usually no."""
    window = timedelta(minutes=max(1, int(tolerance_min)))
    for it in items or []:
        if str(getattr(it, "kind", "") or "") != kind:
            continue
        try:
            other = datetime.fromtimestamp(float(getattr(it, "effective_due", None)
                                                 or getattr(it, "due"))).astimezone(due.tzinfo)
        except (TypeError, ValueError, OSError):
            continue
        if abs(other - due) > window:
            continue
        label = " ".join(str(getattr(it, "label", "") or "").split())
        named = f" ({label})" if label else ""
        return Objection(
            source=f"duplicate {kind}",
            reason=f"you already have {'an' if kind[0] in 'aeiou' else 'a'} "
                   f"{kind} at {clock_words(other)}{named}",
            row=f"timekeeper {kind} {other.isoformat()}",
            key=f"dup|{kind}|{other.isoformat()}")
    return None


def deadline_clash(due: datetime, items, now: datetime,
                   hours: float = CLASH_HOURS) -> Optional[Objection]:
    """The alarm is set for AFTER something is due. Getting up at eight for
    a report that was due at midnight is the failure the alarm was meant to
    prevent, and the Canvas row says so in as many words."""
    best = None
    floor = due - timedelta(hours=max(1.0, float(hours)))
    for it in items or []:
        if not isinstance(it, dict):
            continue
        when = it.get("due")
        title = " ".join(str(it.get("title") or "").split())
        if not isinstance(when, datetime) or not title:
            continue
        when = when.astimezone(due.tzinfo) if when.tzinfo else when.astimezone()
        if not (floor <= when < due) or when <= now:
            continue
        if best is None or when < best[0]:
            course = " ".join(str(it.get("course") or "").split())
            best = (when, title, course)
    if best is None:
        return None
    when, title, course = best
    what = f"{title} for {course}" if course else title
    return Objection(
        source="deadline clash",
        reason=f"{what} is due at {clock_words(when)}, before that",
        row=f"canvas {what} {when.isoformat()}",
        key=f"clash|{what}|{when.isoformat()}")


def quiet_conflict(due: datetime, quiet) -> Optional[Objection]:
    """The alarm lands inside a window he told Jarvis to keep clear -- a
    lecture, an exam, do-not-disturb. ``quiet.reason(ts)`` answers from the
    calendar CACHE and the settings file; it never fetches."""
    if quiet is None:
        return None
    try:
        reason = (quiet.reason(due.timestamp()) or "").strip()
    except Exception:  # noqa: BLE001 - a probe failure must not break the turn
        log.debug("objections: quiet probe failed", exc_info=True)
        return None
    if not reason or reason == "you're out":
        return None                # where he is at 7 am is not an objection
    return Objection(
        source="quiet conflict",
        reason=f"you're {reason} then",
        row=f"quiet {reason}",
        key=f"quiet|{reason}")


# ------------------------------------------------------------- the ledger
class ObjectionLedger:
    """What he has already objected to today. Same shape and the same scars
    as jarvis/headsup.py: str -> ISO-string only (one hand-edited null used
    to raise on every pass there), atomic ``os.replace`` on write."""

    def __init__(self, state_path: Optional[Path] = None, now=None):
        self._path = Path(state_path) if state_path else None
        self._now = now or (lambda: datetime.now().astimezone())
        self._seen: dict = self._load()

    def _load(self) -> dict:
        try:
            if self._path and self._path.exists():
                data = json.loads(self._path.read_text())
                if not isinstance(data, dict):
                    return {}
                return {k: v for k, v in data.items()
                        if isinstance(k, str) and isinstance(v, str)}
        except (OSError, ValueError):
            log.debug("objection ledger unreadable", exc_info=True)
        return {}

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(self._seen))
            os.replace(tmp, self._path)
        except OSError:
            log.debug("objection ledger save failed", exc_info=True)

    def _key(self, obj: Objection, now: datetime) -> str:
        return f"{now.date().isoformat()}|{obj.key}"

    def already_said(self, obj: Objection) -> bool:
        return self._key(obj, self._now()) in self._seen

    def record(self, obj: Objection) -> None:
        now = self._now()
        self._seen[self._key(obj, now)] = now.isoformat()
        cutoff = (now - timedelta(days=2)).date().isoformat()
        for key in [k for k, v in self._seen.items()
                    if not isinstance(v, str) or k[:10] < cutoff]:
            self._seen.pop(key, None)
        self._save()


# --------------------------------------------------------------- entry
def for_alarm(due: datetime, now: datetime, events=None, items=None,
              pending=None, quiet=None,
              sleep_floor_h: float = DEFAULT_SLEEP_FLOOR_H) -> Optional[Objection]:
    """The first objection to setting an alarm for ``due``, or None.

    Order is by how concrete the row is: an alarm he has already set beats
    a reason he has to reason about. Only ever called for an alarm -- see
    rule 4 in the module docstring."""
    for probe in (lambda: duplicate_item(due, pending, "alarm"),
                  lambda: sleep_window(due, events, now, sleep_floor_h),
                  lambda: quiet_conflict(due, quiet),
                  lambda: deadline_clash(due, items, now)):
        try:
            found = probe()
        except Exception:  # noqa: BLE001 - a bad row must not break the alarm
            log.debug("objections: a source raised", exc_info=True)
            continue
        if found is not None:
            return found
    return None


def _first_event_after(events, when: datetime, same_day_as: datetime = None):
    """The soonest timed event starting after ``when`` (optionally only on
    the same local day). All-day rows are skipped: nothing starts at
    midnight, and "Rosh Hashanah" is not a reason to sleep in."""
    best = None
    for ev in events or []:
        if getattr(ev, "all_day", False):
            continue
        start = getattr(ev, "start", None)
        title = " ".join(str(getattr(ev, "title", "") or "").split())
        if not isinstance(start, datetime) or not title:
            continue
        start = start.astimezone(when.tzinfo) if start.tzinfo else start.astimezone()
        if start <= when:
            continue
        if same_day_as is not None and start.date() != same_day_as.date():
            continue
        if best is None or start < getattr(best, "start"):
            best = ev
    return best


# ------------------------------------------------------- override wording
# The objection ENDS in "Shall I set it anyway?", so the words it invites
# are exactly the ones commander.parse_yes_no returns None for: "set it
# anyway", "anyway", "I know", "regardless". Scoped here on purpose -- the
# global _YES_PHRASES are shared with the alarm offer, the event confirm and
# the "was that for me?" card, and widening those to accept "I know" would
# make every one of them answerable by a shrug.
_OVERRIDE_PHRASES = ("anyway", "regardless", "even so", "nonetheless",
                     "i know", "i am aware", "i'm aware", "im aware",
                     "that's fine", "thats fine", "it's fine", "its fine",
                     "noted", "understood", "do it", "set it", "i don't mind",
                     "i dont mind")
_PUNCT_RX = re.compile(r"[^a-z0-9\s']+")


def is_override(text: str) -> bool:
    """"Set it anyway" and its neighbours: an explicit go-ahead that
    ``parse_yes_no`` reads as neither yes nor no."""
    lowered = " ".join(_PUNCT_RX.sub(" ", str(text or "").lower()).split())
    if not lowered:
        return False
    return any(p in lowered for p in _OVERRIDE_PHRASES)
