"""Working sessions: the quiz's pending-state pattern promoted to a protocol.

Quiz mode already proved the shape -- ``Commander._pending_quiz`` holds a
pure state object, one rung in ``_handle_inner`` reads the next utterance
as an ANSWER rather than a command, and the mic re-opens without a wake
word between turns. That machinery was welded to flashcards. This module
lifts it out so anything multi-turn can borrow it:

    ask()      the line to speak that puts a question on the table
    settle(t)  read one answer -> the acknowledgement line, or None
    stop()     the closing line ("that's enough" / the natural end)
    stale()    the question is too old to be what he is answering
    finished   nothing left to ask

``settle`` returning **None** is the escape hatch and the reason a session
is safe to leave open: an utterance the session does not recognise is not
its answer, so the commander drops the session and routes the words as a
new subject. Nothing gets trapped in a dialogue.

Two constants matter and neither is borrowed from the quiz:

* ``SESSION_WINDOW_S`` (18 s) is the MIC window between turns. The
  follow-up window is ``CONFIG.followup_window`` = 4.0 s, which cannot
  hold "Tuesday at four, or push it to Wednesday?" -- the plan would die
  between turns. ``JarvisApp._capture_window`` asks the commander for the
  open session's ``window_s`` (the same hook lecture notes uses).
* ``SESSION_STALE_S`` (90 s) is when an unanswered question stops being
  the thing he is answering. ``quiz.ANSWER_WINDOW_S`` is 300 s -- right
  for a flashcard he is thinking about, far too long for a planning turn
  he walked away from.

The first tenant is ``WeekPlanner`` ("let us plan the week"): it walks the
week's Canvas deadlines and to-dos, proposes one slot each, and takes
yes / move that to Thursday / skip it / that's enough. Everything here is
pure -- no services, no clock but the one you pass, no I/O. Filing the
accepted plan is a callback the commander injects (``filer``), so the
planner is testable without a timekeeper.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Optional

from jarvis.endpoint import strip_fillers
from jarvis.logs import get_logger

log = get_logger("dialogue")

# The mic window between turns of a session (seconds). CONFIG.followup_window
# (4.0) is a beat, not a conversation; the recorder caps a capture at half
# its hard cap, so this stays well under 30.
SESSION_WINDOW_S = 18.0
# A question older than this is not what he is answering. Deliberately NOT
# quiz.ANSWER_WINDOW_S (300 s): a plan he wandered away from should be gone
# by the time he comes back and says something else.
SESSION_STALE_S = 90.0

# "That's enough" ends a session WITH its read-back. commander.quiet_kind
# also matches these words, but there they mean barge-in (cut the speech,
# say nothing); inside a session "enough" is the user wrapping the thing
# up, and losing the plan he just built would be the wrong reading. Checked
# ahead of quiet_kind by the session rung, and only while one is open.
_ENOUGH_RX = re.compile(
    r"^(?:that(?:'s|s| is| will|'ll)\s+(?:enough|plenty|it|the lot|do|does it)|"
    r"enough(?: for now| for today)?|"
    r"we(?:'re| are)?\s+done(?: for now)?|"
    r"leave it (?:there|at that))"
    r"(?:[,]?\s*(?:please|sir|jarvis|thanks|thank you))*[.!\s]*$", re.I)


def enough_kind(text: str) -> bool:
    """True for "that's enough" / "that'll do" / "enough for now"."""
    return bool(_ENOUGH_RX.match((text or "").strip()))


# ------------------------------------------------------------- protocol
class Session:
    """Base for a working session. Subclasses own ``settle`` and ``ask``;
    everything here is the timing the commander and the mic rely on.

    A session is a plain object with no services and no thread: the
    commander holds it in ``_pending_session``, the rung above the quiz
    feeds it utterances, and it is dropped the moment it says it is
    finished, goes stale, or does not recognise what it heard.
    """

    #: short label for the status strip ("plan the week")
    name: str = "session"
    #: mic window between turns; JarvisApp._capture_window reads this
    window_s: float = SESSION_WINDOW_S
    #: seconds after ask() before the open question stops counting
    stale_s: float = SESSION_STALE_S
    #: when ask() last put a question on the table (0.0 = never)
    asked_at: float = 0.0

    @property
    def finished(self) -> bool:
        return True

    # -- the three lines -------------------------------------------------
    def ask(self, now: Optional[float] = None) -> str:
        """The question on the table. Stamps the clock: ALWAYS call this
        through the commander rung, never for a preview."""
        self.touch(now)
        return ""

    def settle(self, text: str) -> Optional[str]:
        """Read one answer. Returns the acknowledgement to speak, or None
        when the words are not an answer at all -- which drops the session
        and lets the utterance route as a fresh command."""
        return None

    def stop(self) -> str:
        """The closing line: the natural end, or "that's enough"."""
        return ""

    # -- timing ----------------------------------------------------------
    def touch(self, now: Optional[float] = None) -> None:
        self.asked_at = time.time() if now is None else float(now)

    def stale(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        asked = getattr(self, "asked_at", 0.0) or 0.0
        return asked > 0 and now - asked > float(self.stale_s)

    def status_text(self) -> str:
        """Short progress token for the status strip ("2 of 5")."""
        return ""


@dataclass
class EchoSession(Session):
    """Pure test double: it repeats what it hears, finishes after
    ``limit`` turns, and returns None for anything containing "weather"
    (an unrelated subject, which must drop the session and route)."""

    name: str = "echo"
    limit: int = 3
    heard: list = field(default_factory=list)
    asked_at: float = 0.0

    @property
    def finished(self) -> bool:
        return len(self.heard) >= self.limit

    def ask(self, now: Optional[float] = None) -> str:
        self.touch(now)
        return f"Question {len(self.heard) + 1}?"

    def settle(self, text: str) -> Optional[str]:
        if "weather" in (text or "").lower():
            return None
        self.heard.append(text)
        return f"You said {text}."

    def stop(self) -> str:
        return f"{len(self.heard)} noted, sir."

    def status_text(self) -> str:
        return f"{len(self.heard)}/{self.limit}"


# ------------------------------------------------------- the week planner
PLAN_OPEN_LINE = "Right, sir: {n} to place this week."
PLAN_NOTHING_LINE = "Nothing to plan this week, sir; your slate is clear."
PLAN_EMPTY_LINE = "Nothing put down, sir."
PLAN_FILED_LINE = "That's the week, sir: {plan}. I've set a reminder for each."
PLAN_UNFILED_LINE = "That's the week, sir: {plan}. I couldn't file the reminders."
SKIP_LINE = "Leaving that one, sir."
TAKEN_LINE = "{day} at {hour}, then."
MOVED_LINE = "{day} instead"
NO_ROOM_LINE = "I've nowhere left to put that one, sir."

# When the day is proposed at. Late afternoon first (after classes), then
# the evening, then the morning -- one item per slot, so a day holds three.
SLOT_HOURS = (16, 19, 10)
PLAN_DAYS = 5              # Monday to Friday from today, the pitch's week
MAX_ITEMS = 8              # a spoken walk longer than this is a chore
_HOUR_WORDS = {0: "midnight", 1: "one", 2: "two", 3: "three", 4: "four",
               5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
               10: "ten", 11: "eleven", 12: "noon", 13: "one", 14: "two",
               15: "three", 16: "four", 17: "five", 18: "six", 19: "seven",
               20: "eight", 21: "nine", 22: "ten", 23: "eleven"}
_DAY_RX = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|today|tonight)\b", re.I)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")
_YES_RX = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|fine|good|right|please do|do it|"
    r"go on|that works|sounds good|perfect|aye|affirmative|"
    r"that's fine|thats fine|book it|put it (?:in|down))"
    r"(?:[,]?\s*(?:please|sir|jarvis))?[.!\s]*$", re.I)
_SKIP_RX = re.compile(
    r"^(?:no|nope|nah|skip(?: (?:it|that|this one))?|leave (?:it|that)|"
    r"drop (?:it|that)|not (?:that|this week)|pass)"
    r"(?:[,]?\s*(?:please|sir|jarvis))?[.!\s]*$", re.I)
_MOVE_RX = re.compile(
    r"\b(?:move|push|shift|make|put)\b.*", re.I)


def hour_words(hour: int) -> str:
    return _HOUR_WORDS.get(int(hour) % 24, str(hour))


def day_words(day: date, today: date) -> str:
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    return day.strftime("%A")


@dataclass
class PlanItem:
    """One thing to place. ``due`` is an epoch when the source knows one
    (a Canvas deadline); a to-do has none."""
    title: str
    source: str = ""                       # canvas | todo
    due: Optional[float] = None


@dataclass
class PlanSlot:
    item: PlanItem
    day: date
    hour: int

    def when_epoch(self) -> float:
        return datetime.combine(self.day, datetime.min.time()).replace(
            hour=int(self.hour)).timestamp()

    def words(self, today: date) -> str:
        return f"{day_words(self.day, today)} at {hour_words(self.hour)}"


def collect_items(deadlines: list, todos: list, now: Optional[float] = None,
                  limit: int = MAX_ITEMS) -> list[PlanItem]:
    """Canvas due items + to-dos -> the walk order: dated work first
    (soonest first), then the undated to-dos in their own order.

    ``deadlines`` are canvas.fetch_due dicts (course / title / due as a
    datetime); ``todos`` are NotesStore rows (dicts with 'text') or plain
    strings. Anything unreadable is skipped rather than raising -- a bad
    row must not cost the whole plan.
    """
    now = time.time() if now is None else float(now)
    dated: list[PlanItem] = []
    for row in deadlines or []:
        try:
            title = str(row.get("title") or "").strip()
            course = str(row.get("course") or "").strip()
            due = row.get("due")
            stamp = due.timestamp() if hasattr(due, "timestamp") else None
        except Exception:  # noqa: BLE001 - one bad row, not the whole plan
            log.debug("plan: unreadable deadline row", exc_info=True)
            continue
        if not title:
            continue
        if stamp is not None and stamp < now:
            continue                       # already past: nothing to plan
        label = f"{title} for {course}" if course else title
        dated.append(PlanItem(title=label, source="canvas", due=stamp))
    dated.sort(key=lambda i: (i.due is None, i.due or 0.0))
    plain: list[PlanItem] = []
    for row in todos or []:
        text = row.get("text") if isinstance(row, dict) else row
        text = str(text or "").strip()
        if text:
            plain.append(PlanItem(title=text, source="todo"))
    return (dated + plain)[:max(0, int(limit))]


def week_days(today: date, days: int = PLAN_DAYS) -> list[date]:
    """The next ``days`` days starting today. Deliberately calendar days,
    not weekdays: a Saturday deadline is still a Saturday deadline, and a
    planner that refuses to say "Saturday" is lying about the week."""
    return [today + timedelta(days=i) for i in range(max(1, int(days)))]


@dataclass
class WeekPlanner(Session):
    """"Let us plan the week." One item at a time: a proposed day and
    hour, then yes / move that to Thursday / skip it / that's enough.

    Pure: ``today`` is passed in, ``filer`` is the commander's callback
    that writes the accepted slots (timekeeper reminders). Nothing is
    filed until the session ends, so "that's enough" halfway through still
    files what was agreed and abandoning it files nothing.
    """

    items: list = field(default_factory=list)
    today: date = field(default_factory=date.today)
    days: int = PLAN_DAYS
    filer: Optional[Callable[[list], bool]] = None
    index: int = 0
    slots: list = field(default_factory=list)
    proposal: Optional[PlanSlot] = None
    asked_at: float = 0.0
    filed: Optional[bool] = None
    name: str = "plan the week"

    def __post_init__(self):
        self.asked_at = 0.0
        self._calendar = week_days(self.today, self.days)

    # -- state -----------------------------------------------------------
    @property
    def current(self) -> Optional[PlanItem]:
        return self.items[self.index] if 0 <= self.index < len(self.items) else None

    @property
    def finished(self) -> bool:
        return self.index >= len(self.items)

    def status_text(self) -> str:
        return f"{min(self.index + 1, len(self.items))}/{len(self.items)}"

    # -- slot choice -----------------------------------------------------
    def _taken(self) -> set:
        return {(s.day, s.hour) for s in self.slots}

    def _deadline(self, item: PlanItem) -> Optional[date]:
        if item.due is None:
            return None
        return datetime.fromtimestamp(item.due).date()

    def propose(self, item: PlanItem, after: Optional[date] = None) -> Optional[PlanSlot]:
        """The first free (day, hour) for ``item``, never past its due
        date. ``after`` restricts the search to that day onward (a "move
        that to Thursday" that has to fall back)."""
        taken = self._taken()
        limit = self._deadline(item)
        for day in self._calendar:
            if after is not None and day < after:
                continue
            if limit is not None and day > limit:
                break
            for hour in SLOT_HOURS:
                if (day, hour) not in taken:
                    return PlanSlot(item=item, day=day, hour=hour)
        return None

    def _parse_day(self, text: str) -> Optional[date]:
        """"Thursday" / "tomorrow" / "tonight" -> a day inside the week
        being planned, else None."""
        m = _DAY_RX.search(text or "")
        if not m:
            return None
        word = m.group(1).lower()
        if word in ("today", "tonight"):
            return self.today
        if word == "tomorrow":
            return self.today + timedelta(days=1)
        want = _WEEKDAYS.index(word)
        for day in self._calendar:
            if day.weekday() == want:
                return day
        return None

    # -- the protocol ----------------------------------------------------
    def ask(self, now: Optional[float] = None) -> str:
        """The next proposal. Items the week has no room for are dropped
        here with ONE apology, however many of them there are -- asking a
        question that has no possible answer is worse than saying so."""
        self.touch(now)
        no_room = False
        while True:
            item = self.current
            if item is None:
                return NO_ROOM_LINE if no_room else ""
            if self.proposal is None or self.proposal.item is not item:
                self.proposal = self.propose(item)
            if self.proposal is not None:
                break
            no_room = True
            self.index += 1
        head = NO_ROOM_LINE + " " if no_room else ""
        return (f"{head}{self.proposal.item.title} — "
                f"{day_words(self.proposal.day, self.today)} at "
                f"{hour_words(self.proposal.hour)}?")

    def settle(self, text: str) -> Optional[str]:
        # The filled pause first (jarvis.endpoint.strip_fillers, one list
        # with the recorder's filler hold): _YES_RX / _SKIP_RX are anchored
        # on the answer word, so "uh, yes" used to drop the whole walk and
        # route as a command. A reply that is only a filler strips to ""
        # and settles nothing -- the question stays on the table.
        t = strip_fillers(text or "").strip().rstrip(".!?")
        if not t or self.current is None:
            return None
        if _SKIP_RX.match(t):
            self.index += 1
            self.proposal = None
            return SKIP_LINE
        moved = self._parse_day(t) if (_MOVE_RX.match(t) or _DAY_RX.search(t)) else None
        if moved is not None:
            item = self.current
            slot = self.propose(item, after=moved)
            if slot is not None and slot.day != moved:
                slot = None                # the requested day is full: say so
            if slot is None:
                self.proposal = None
                return NO_ROOM_LINE
            self.proposal = slot
            return MOVED_LINE.format(day=day_words(slot.day, self.today)) + ","
        if _YES_RX.match(t):
            slot = self.proposal
            if slot is None:
                return None
            self.slots.append(slot)
            self.index += 1
            self.proposal = None
            return TAKEN_LINE.format(day=day_words(slot.day, self.today),
                                     hour=hour_words(slot.hour))
        return None                        # not an answer: route it as a command

    def plan_words(self) -> str:
        parts = [f"{s.words(self.today)}, {s.item.title}" for s in self.slots]
        if len(parts) > 1:
            return "; ".join(parts[:-1]) + f"; and {parts[-1]}"
        return parts[0] if parts else ""

    def stop(self) -> str:
        """File what was agreed and read it back. Called once -- the rung
        drops the session immediately after."""
        if not self.slots:
            return PLAN_EMPTY_LINE
        ok = True
        if self.filer is not None:
            try:
                ok = bool(self.filer(list(self.slots)))
            except Exception:  # noqa: BLE001 - a timekeeper failure is not a crash
                log.exception("plan: filing the week failed")
                ok = False
        self.filed = ok
        line = PLAN_FILED_LINE if ok else PLAN_UNFILED_LINE
        return line.format(plan=self.plan_words())


def open_line(n: int) -> str:
    return PLAN_OPEN_LINE.format(n=f"{n} things" if n != 1 else "one thing")
