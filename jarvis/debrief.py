"""The debrief: "How did the midterm go, sir?" -- asked once, after it ended.

The calendar says the BIOSENSORS midterm ended forty minutes ago and the
phone is on the Wi-Fi, so he asks. Once. Whatever the answer is, it is
FILED -- to the activity journal and as a remembered fact -- and never
routed to the model as chat, so it comes back in the day recap, in the
nightly review, and months later as an aside before the next one.

This module contains no speech path at all. It decides that something
real ended and hands the app a candidate; the app owns the question, the
listening window and the filing. Knowing that something ended is the hard
half, and it is hard for four reasons:

1. **A cancelled event must never be asked about.** ``calendar._window``
   fetches from yesterday midnight, so an event that ended forty minutes
   ago is still in the cache -- but so is one that was ADDED retroactively
   this afternoon, and "how did the dentist go" about an appointment he
   never had is worse than silence. Hence the ``seen`` ledger: an event
   only becomes askable if this watch saw it while its start was still in
   the FUTURE. An event that first appears in the cache after it began is
   never a candidate.

2. **Once, ever.** The ``asked`` ledger is written the moment the question
   is put, not when it is answered, and it survives a restart. Being asked
   twice how the exam went is the whole feature failing.

3. **Never while he is away, never in quiet hours.** ``presence`` is idle
   until ``presence.phone_ip`` is configured, so the gate is "not known to
   be AWAY" rather than "known to be home" -- the other way round the
   feature would never fire on this box. Quiet hours do not cancel the
   question, they postpone it: a candidate met during a quiet window is
   parked in ``held`` and asked when the window lifts, up to
   ``hold_hours`` later. Holding to morning is right; asking at 2 am is
   not, and dropping it entirely would lose the exam that finished at
   ten past eleven.

4. **Calendar first.** Canvas exam rows carry ``due_at`` and no end
   (tools/canvas.py exam_kind / _EXAM_KINDS), so there is nothing there to
   say "it is over". This ships against the calendar, which does have
   ``Event.end``; Canvas is a later source.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("debrief")

INTERVAL_S = 300.0
# The window after an event ends in which the question still makes sense.
# Under 15 minutes he may still be walking out of the room; after three
# hours the moment has passed and the question is an interrogation.
DEFAULT_AFTER_MIN = 15
DEFAULT_WITHIN_MIN = 180
# How long a candidate held by quiet hours stays askable. Long enough to
# reach the morning after an evening exam, short enough that it never
# arrives on the wrong day.
DEFAULT_HOLD_HOURS = 14
# The only events worth asking about. Narrow on purpose: this is the one
# feature that speaks without being spoken to, and "how did your lunch go"
# is the version of it nobody wants.
DEFAULT_KEYWORDS = ("exam", "midterm", "final", "finals", "interview",
                    "viva", "defense", "defence", "quiz", "test",
                    "presentation", "audition")
_WORD_RX = re.compile(r"[a-z]+")


def matched_word(title: str, keywords) -> str:
    """The keyword that makes this event worth a question ("" when none).

    Whole words only: "contest" must not match "test", and "finalise the
    slides" must not become a final."""
    words = set(_WORD_RX.findall(str(title or "").lower()))
    for kw in keywords:
        if str(kw).lower() in words:
            return str(kw).lower()
    return ""


def question_for(title: str, word: str) -> str:
    """"How did the midterm go, sir?" -- the keyword, not the whole title.
    Reading a calendar title back verbatim ("How did BIOSENSORS Midterm 1
    - Rm 214 go") is how an assistant sounds like a form."""
    word = (word or "").strip() or "it"
    return f"How did the {word} go, sir?"


class Candidate:
    __slots__ = ("key", "title", "word", "end", "question")

    def __init__(self, key: str, title: str, word: str, end: datetime):
        self.key = key
        self.title = title
        self.word = word
        self.end = end
        self.question = question_for(title, word)


class DebriefWatch:
    """Ticks the calendar cache and yields at most one candidate a tick."""

    def __init__(self, cfg=None, get_calendar: Callable = None, quiet=None,
                 presence=None, state_path: Optional[Path] = None,
                 now: Callable = None, on_candidate: Callable = None):
        self._cfg = cfg
        self._get_calendar = get_calendar
        self._quiet = quiet
        self._presence = presence
        self._state_path = Path(state_path) if state_path else None
        self._now = now or (lambda tz=None: datetime.now(tz).astimezone()
                            if tz is None else datetime.now(tz))
        self._on_candidate = on_candidate
        self._state = self._load()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ config
    def _opt(self, key: str, default):
        cfg = self._cfg
        if cfg is None:
            return default
        try:
            val = cfg.get(key, default)
        except Exception:  # noqa: BLE001 - a config read must never kill the tick
            log.debug("debrief: config read failed for %s", key, exc_info=True)
            return default
        return default if val is None else val

    @property
    def enabled(self) -> bool:
        return bool(self._opt("debrief.enabled", True))

    def _keywords(self) -> tuple:
        raw = self._opt("debrief.keywords", DEFAULT_KEYWORDS)
        if isinstance(raw, str):
            raw = [raw]
        try:
            words = tuple(str(w).strip().lower() for w in raw if str(w).strip())
        except TypeError:
            return DEFAULT_KEYWORDS
        return words or DEFAULT_KEYWORDS

    def _minutes(self, key: str, default: int) -> int:
        try:
            return max(0, int(self._opt(key, default)))
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------- state
    # Copied from jarvis/headsup.py:41-73, scars included: only str -> ISO
    # string entries survive a load, because the values are compared as
    # strings when pruning and one hand-edited null used to raise on every
    # tick before the save could ever run.
    _LEDGERS = ("seen", "asked", "held")

    def _load(self) -> dict:
        blank = {name: {} for name in self._LEDGERS}
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if not isinstance(data, dict):
                    return blank
                out = {}
                for name in self._LEDGERS:
                    rows = data.get(name)
                    rows = rows if isinstance(rows, dict) else {}
                    clean = {k: v for k, v in rows.items()
                             if isinstance(k, str) and isinstance(v, str)}
                    if len(clean) != len(rows):
                        log.debug("debrief state: dropped %d malformed %s entries",
                                  len(rows) - len(clean), name)
                    out[name] = clean
                return out
        except (OSError, ValueError):
            log.debug("debrief state unreadable", exc_info=True)
        return blank

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._state))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("debrief state save failed", exc_info=True)

    def _prune(self, now: datetime) -> bool:
        """`seen` and `held` are working memory and expire in days; `asked`
        is the promise never to ask twice and is kept far longer."""
        cutoff = {"seen": (now - timedelta(days=3)).isoformat(),
                  "held": (now - timedelta(days=3)).isoformat(),
                  "asked": (now - timedelta(days=120)).isoformat()}
        dropped = False
        for name in self._LEDGERS:
            rows = self._state[name]
            for key in [k for k, v in rows.items()
                        if not isinstance(v, str) or v < cutoff[name]]:
                rows.pop(key, None)
                dropped = True
        return dropped

    @property
    def asked(self) -> dict:
        return dict(self._state["asked"])

    @property
    def seen(self) -> dict:
        return dict(self._state["seen"])

    # -------------------------------------------------------------- gates
    def _away(self) -> bool:
        """True only when presence has ESTABLISHED that he is out.

        Not "is_home() is False" via a truthiness test: presence is idle
        until phone_ip is configured, and a box with no presence at all
        must still debrief."""
        presence = self._presence
        if presence is None:
            return False
        try:
            configured = getattr(presence, "configured", False)
            if callable(configured):
                configured = configured()
            if not configured:
                return False
            return getattr(presence, "state", "unknown") == "away"
        except Exception:  # noqa: BLE001 - a probe failure must not mute him
            log.debug("debrief: presence probe failed", exc_info=True)
            return False

    def _quiet_reason(self) -> str:
        quiet = self._quiet
        if quiet is None:
            return ""
        try:
            if not quiet.should_hold():
                return ""
            return (quiet.reason() or "quiet").strip()
        except Exception:  # noqa: BLE001
            log.debug("debrief: quiet probe failed", exc_info=True)
            return ""

    # --------------------------------------------------------------- tick
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
            return list(cal.events())
        except Exception:  # noqa: BLE001 - source boundary
            log.debug("debrief: calendar unavailable", exc_info=True)
            return []

    def tick(self) -> Optional[Candidate]:
        """One pass: refresh `seen`, then return at most one candidate.

        One at a time deliberately -- two questions in a row about two
        different exams is an interrogation, and the next tick is five
        minutes away."""
        if not self.enabled:
            return None
        keywords = self._keywords()
        after = timedelta(minutes=self._minutes("debrief.after_min", DEFAULT_AFTER_MIN))
        within = timedelta(minutes=self._minutes("debrief.within_min", DEFAULT_WITHIN_MIN))
        try:
            hold = timedelta(hours=max(0.0, float(
                self._opt("debrief.hold_hours", DEFAULT_HOLD_HOURS))))
        except (TypeError, ValueError):
            hold = timedelta(hours=DEFAULT_HOLD_HOURS)
        now = self._now()
        dirty = self._prune(now)
        found = None
        for ev in self._events():
            start, end = getattr(ev, "start", None), getattr(ev, "end", None)
            title = " ".join(str(getattr(ev, "title", "") or "").split())
            if getattr(ev, "all_day", False) or not title:
                continue
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                continue
            word = matched_word(title, keywords)
            if not word:
                continue
            here = now.astimezone(start.tzinfo) if start.tzinfo else now.replace(tzinfo=None)
            key = f"{title}|{start.isoformat()}"
            if start > here:
                # Seen while it was still ahead of us: this is the ONLY way
                # an event becomes askable, and it is what makes a
                # retroactively-added row unaskable.
                if self._state["seen"].get(key) != start.isoformat():
                    self._state["seen"][key] = start.isoformat()
                    dirty = True
                continue
            if key in self._state["asked"] or key not in self._state["seen"]:
                continue
            if found is not None:
                continue                       # already have one this tick
            since = here - end
            held_at = self._state["held"].get(key)
            fresh = after <= since <= within
            still_held = bool(held_at) and since <= hold
            if not (fresh or still_held):
                continue
            reason = self._quiet_reason()
            if reason:
                if not held_at:
                    self._state["held"][key] = end.isoformat()
                    dirty = True
                    log.info("debrief: %r held (%s)", title, reason)
                continue
            if self._away():
                log.debug("debrief: %r skipped, he is out", title)
                continue
            found = Candidate(key, title, word, end)
        if dirty:
            self._save()
        if found is not None:
            log.info("debrief candidate: %r ended %s -> %r", found.title,
                     found.end.strftime("%H:%M"), found.question)
            cb = self._on_candidate
            if callable(cb):
                try:
                    cb(found)
                except Exception:  # noqa: BLE001 - the caller owns speech
                    log.exception("debrief: candidate callback failed")
        return found

    def mark_asked(self, key: str, now: datetime = None) -> None:
        """Written when the question is PUT, not when it is answered: an
        unanswered debrief is still a debrief he has been asked for."""
        when = now or self._now()
        self._state["asked"][key] = when.isoformat()
        self._state["held"].pop(key, None)
        self._save()

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again (tests, a config reload), and a dead thread must
        # not block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="debrief")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        # first pass a little after boot so the calendar has refreshed
        if self._stop.wait(40.0):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("debrief tick failed")
            if self._stop.wait(INTERVAL_S):
                return


# ------------------------------------------------------------- the filer
def fact_key(title: str, end: datetime) -> str:
    """How the answer is remembered. Reads back as English because
    memory.recall matches on the key as well as the value."""
    return f"how {title} went on {end.strftime('%-d %B %Y')}"


def file_answer(text: str, cand: Candidate, memory=None, context=None) -> bool:
    """File a debrief answer. NEVER routes it to the model as chat.

    That restraint is the feature: "it went badly, I ran out of time on
    the last question" is a fact about his year, not a conversational turn
    to be answered with sympathy and forgotten at the end of the window.
    Returns whether anything was stored."""
    text = " ".join(str(text or "").split())
    if not text:
        return False
    stored = False
    if context is not None:
        writer = getattr(context, "journal_debrief", None)
        if callable(writer):
            try:
                writer(cand.title, cand.word, text, cand.end)
                stored = True
            except Exception:  # noqa: BLE001 - one sink failing must not lose the other
                log.exception("debrief: journal write failed")
    if memory is not None:
        try:
            memory.remember(fact_key(cand.title, cand.end), text)
            stored = True
        except Exception:  # noqa: BLE001
            log.exception("debrief: memory write failed")
    log.info("debrief filed for %r: %.80s", cand.title, text)
    return stored
