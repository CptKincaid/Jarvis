"""Activity journal: "recap my day" / "what was I doing before lunch?"

The journal itself is written by ``jarvis.context.ContextEngine`` (one JSON
line per exchange, tool call, Claude task result and window-focus change,
one file per day under ``PATHS.MEMORY_DIR/journal``). This module reads it
back:

* ``parse_window`` turns "this morning", "before lunch", "yesterday",
  "the last two hours" into a (since, until, label) window;
* ``digest`` renders the rows for that window as compact plain text,
  bucketed by hour and bounded so it fits the tool-text budget (the brain
  caps a tool result at 4 000 chars against NUM_CTX); the model then
  speaks a short recap from it and the digest goes on a card;
* ``find_last_mention`` / ``last_mention_line`` answer "when did I last
  talk to my advisor?" -- the day files are walked NEWEST first and the
  scan stops at the first hit, so a term mentioned this morning costs one
  file read where ``journal_rows`` would have read ninety;
* ``journal_repeats`` renders the "Earlier today" continuity block for the
  per-turn background: the same tool with the same salient argument N
  times, the same question asked N times. It is the only thing here that
  runs on the hot path, so it is bounded to two clauses and reads only
  rows it is handed;
* ``ActivitySampler`` samples the focused window every minute through the
  context engine's existing xdotool probe (no new subprocess), skipping
  a locked or empty desktop, so "go back" and the recap both get window
  history that was dead before;
* ``make_tools`` registers ``recap_day``; the commander pins it with
  force_tool for the Tier 1 phrases.

Fully local: the journal is a file, the recap is the local model.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from jarvis.events import JarvisReply, bus
from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.journal")

DEFAULT_INTERVAL_S = 60.0
DEFAULT_KEEP_DAYS = 90
LUNCH_HOUR = 12
EVENING_HOUR = 17
# Under the brain's 4 000-char single-result cap with room for the
# "(what follows is ...)" allowance; ~900 tokens of the 8 192 window.
DIGEST_CHARS = 3200
SNIPPET_CHARS = 110
TITLE_CHARS = 48
WINDOWS_PER_HOUR = 5
RECAP_SENTENCES = 4

NOTHING_LINE = "I have nothing in the journal for {label}, sir."
NO_JOURNAL_LINE = "I'm afraid the journal isn't available, sir."

_HOURS_RX = re.compile(
    r"\b(?:last|past) (?:(?P<n>\d+|an?|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|twelve|couple of|few) )?(?P<unit>hours?|minutes?)\b", re.I)
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}


# ----------------------------------------------------------- windows
def _day_start(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_window(text: str, now: Optional[datetime] = None):
    """-> (since, until, label). "before lunch" is midnight to noon;
    "this morning" the same; "afternoon" noon to now (or five); "yesterday"
    the whole of yesterday; "the last two hours" is relative; anything
    else is today so far."""
    now = now or datetime.now()
    t = (text or "").lower()
    start = _day_start(now)
    m = _HOURS_RX.search(t)
    if m:
        raw_n = (m.group("n") or "1").lower()
        n = _NUM.get(raw_n) or (int(raw_n) if raw_n.isdigit() else
                                {"couple of": 2, "few": 3}.get(raw_n, 1))
        delta = timedelta(hours=n) if m.group("unit").startswith("hour") \
            else timedelta(minutes=n)
        unit = "hour" if m.group("unit").startswith("hour") else "minute"
        return now - delta, now, f"the last {n} {unit}{'s' if n != 1 else ''}"
    if "yesterday" in t:
        y = start - timedelta(days=1)
        if "morning" in t or "before lunch" in t:
            return y, y.replace(hour=LUNCH_HOUR), "yesterday morning"
        if "afternoon" in t or "after lunch" in t:
            return y.replace(hour=LUNCH_HOUR), y.replace(hour=EVENING_HOUR), \
                "yesterday afternoon"
        if "evening" in t or "tonight" in t or "last night" in t:
            return y.replace(hour=EVENING_HOUR), start, "yesterday evening"
        return y, start, "yesterday"
    if "before lunch" in t or "morning" in t:
        noon = start.replace(hour=LUNCH_HOUR)
        return start, min(noon, now) if noon <= now else now, \
            "before lunch" if "lunch" in t else "this morning"
    if "after lunch" in t or "afternoon" in t:
        noon = start.replace(hour=LUNCH_HOUR)
        five = start.replace(hour=EVENING_HOUR)
        until = now if now < five or "afternoon" not in t else five
        return noon, max(until, noon), "this afternoon"
    if "evening" in t or "tonight" in t:
        five = start.replace(hour=EVENING_HOUR)
        return five, max(now, five), "this evening"
    if "this week" in t:
        return start - timedelta(days=start.weekday()), now, "this week"
    return start, now, "today"


# ------------------------------------------------------------ digest
def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0").lower()


def _hour_label(dt: datetime) -> str:
    return dt.strftime("%I %p").lstrip("0").lower()


def _snip(text, n=SNIPPET_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _title(text) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    # "main.py - Jarvis - Visual Studio Code" -> keep the app and the file
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1] + "…"


def _render_hours(rows, snippet=SNIPPET_CHARS, windows=True) -> list[str]:
    """One block per hour: exchanges, tools, Claude results, then the
    windows seen in that hour."""
    hours: dict[datetime, list] = {}
    for r in rows:
        when = r.get("_when")
        if not isinstance(when, datetime):
            continue
        hours.setdefault(when.replace(minute=0, second=0, microsecond=0), []).append(r)
    out = []
    for hour in sorted(hours):
        lines = [f"{_hour_label(hour)}:"]
        seen_windows: list[str] = []
        for r in hours[hour]:
            kind = r.get("kind")
            when = _clock(r["_when"])
            if kind == "exchange":
                user, jarvis = r.get("user", ""), r.get("jarvis", "")
                if user:
                    line = f"  {when} you: {_snip(user, snippet)}"
                    if jarvis:
                        line += f" / Jarvis: {_snip(jarvis, snippet)}"
                else:
                    line = f"  {when} Jarvis: {_snip(jarvis, snippet)}"
                lines.append(line)
            elif kind == "tool":
                name = r.get("name", "tool")
                flag = "" if r.get("ok", True) else " (failed)"
                lines.append(f"  {when} ran {name}{flag}")
            elif kind == "debrief":
                # Never snipped by the `snippet` budget: a debrief is one
                # short line a day and it is the most valuable row in the
                # file. Snipping "it went badly, I ran out of time on the
                # last question" to "it went badly, I ran…" throws away the
                # only part worth reading back months later.
                what = _title(r.get("title", "")) or r.get("word", "") or "it"
                lines.append(f"  {when} debrief — {what}: {r.get('text', '')}")
            elif kind == "claude":
                state = r.get("state", "done")
                proj = r.get("project") or "a task"
                verb = "finished" if state == "done" else state
                lines.append(f"  {when} Claude {verb} {proj}: {_snip(r.get('text', ''), snippet)}")
            elif kind == "window" and windows:
                title = _title(r.get("title", ""))
                if title and title not in seen_windows:
                    seen_windows.append(title)
        if seen_windows:
            shown = seen_windows[:WINDOWS_PER_HOUR]
            more = len(seen_windows) - len(shown)
            lines.append("  windows: " + "; ".join(shown) +
                         (f" (+{more} more)" if more > 0 else ""))
        if len(lines) > 1:
            out.append("\n".join(lines))
    return out


def digest(rows, label="today", max_chars=DIGEST_CHARS) -> str:
    """Plain text for the model: a header with counts, then hour blocks.
    Bounded: window lines go first, then snippets shorten, then the oldest
    hours collapse into a count -- the recent part of the day survives."""
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return ""
    n_ex = sum(1 for r in rows if r.get("kind") == "exchange")
    n_tool = sum(1 for r in rows if r.get("kind") == "tool")
    n_claude = sum(1 for r in rows if r.get("kind") == "claude")
    n_win = len({r.get("title") for r in rows if r.get("kind") == "window"})
    first, last = rows[0]["_when"], rows[-1]["_when"]
    head = (f"Journal for {label}, {_clock(first)} to {_clock(last)}: "
            f"{n_ex} exchange{'s' if n_ex != 1 else ''}, {n_tool} tool call"
            f"{'s' if n_tool != 1 else ''}, {n_claude} Claude task"
            f"{'s' if n_claude != 1 else ''}, {n_win} window"
            f"{'s' if n_win != 1 else ''}. Times are local; newest last.")
    for snippet, windows in ((SNIPPET_CHARS, True), (SNIPPET_CHARS, False),
                             (60, False), (40, False)):
        blocks = _render_hours(rows, snippet=snippet, windows=windows)
        text = head + "\n" + "\n".join(blocks)
        if len(text) <= max_chars:
            return text
    # Still too long: drop the oldest hour blocks, say so. Debriefs are
    # PINNED across that trim: they arrive in the evening but they are the
    # one row a busy day cannot afford to lose, and the collapse takes the
    # oldest hours first -- which is exactly where an early-afternoon exam
    # sits. Re-rendered on their own so they survive their hour block.
    pinned = _render_hours([r for r in rows if r.get("kind") == "debrief"],
                           snippet=SNIPPET_CHARS, windows=False)
    dropped = 0
    budget = max_chars - 60 - len("\n".join(pinned))
    while blocks and len(head + "\n" + "\n".join(blocks)) > budget:
        blocks.pop(0)
        dropped += 1
    note = f"(earlier: {dropped} hour{'s' if dropped != 1 else ''} not shown)\n" \
        if dropped else ""
    # Only the debrief hours that the trim actually took: a pinned block
    # whose hour is still in `blocks` would print the line twice.
    kept_hours = {b.split(":", 1)[0] for b in blocks}
    rescued = [b for b in pinned if b.split(":", 1)[0] not in kept_hours]
    return (head + "\n" + note + "\n".join(rescued + blocks))[:max_chars]


# -------------------------------------------------- episodic recall
# "when did I last talk to my advisor?" / "how long since I worked on the
# thesis?".  The journal only sees what flowed through Jarvis -- voice and
# typed turns, tool calls, Claude results and sampled window titles -- so
# the honest claim is "the last time you mentioned it HERE", never "the
# last time you met her".  The wording below says "you said" / "you had X
# open" for exactly that reason.
MENTION_KEEP_DAYS = DEFAULT_KEEP_DAYS
_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd",
                   31: "st"}
# Leading words that are the question's grammar, not the thing looked for:
# "when did I last talk to my advisor" searches for "my advisor".
_MENTION_VERB_RX = re.compile(
    r"^(?:talk(?:ed)?|speak|spoke|spoken|chat(?:ted)?)\s+(?:to|with|about)\s+"
    r"|^(?:see|saw|seen|visit(?:ed)?)\s+"
    r"|^(?:go|gone|went|been)\s+(?:to\s+)?"
    r"|^(?:hear|heard)\s+(?:from|about)\s+"
    r"|^(?:work(?:ed|ing)?|focus(?:ed)?)\s+on\s+"
    r"|^(?:mention(?:ed)?|discuss(?:ed)?|bring|brought)\s+(?:up\s+)?"
    r"|^(?:ask(?:ed)?)\s+(?:you\s+)?(?:about|for)\s+"
    r"|^(?:email(?:ed)?|e-mail(?:ed)?|messag(?:e|ed)|writ(?:e|ten)|wrote)\s+(?:to\s+)?"
    r"|^(?:use[d]?|using|open(?:ed)?|touch(?:ed)?|read|look(?:ed)?\s+at)\s+", re.I)
_LEADING_ARTICLE_RX = re.compile(r"^(?:my|our|the|a|an)\s+", re.I)
# A tail that is nothing but the verb ("when did I last see") asks for no
# thing at all: the handler falls through rather than reporting that the
# journal has never seen the word "see".
_MENTION_STOPWORDS = frozenset({
    "see", "saw", "seen", "talk", "talked", "speak", "spoke", "spoken",
    "go", "gone", "went", "been", "work", "worked", "working", "use", "used",
    "open", "opened", "hear", "heard", "read", "look", "looked", "ask",
    "asked", "email", "emailed", "mention", "mentioned", "do", "did", "it",
    "that", "this", "one", "some"})


def mention_target(tail: str) -> str:
    """'talked to my advisor about the letter?' -> 'my advisor'.

    The verb phrase is stripped, then anything after an "about"/"regarding"
    clause: the FIRST noun phrase is what the question is about, and the
    rest only narrows it (searching the whole tail would match nothing)."""
    t = " ".join(str(tail or "").split()).strip(" ,.?!")
    t = _MENTION_VERB_RX.sub("", t, count=1).strip()
    t = re.split(r"\b(?:about|regarding|concerning)\b", t, maxsplit=1)[0]
    t = t.strip(" ,.?!")
    return "" if t.lower() in _MENTION_STOPWORDS else t


def mention_terms(target: str, memory=None) -> list[str]:
    """The strings to look for: the phrase as said, the same phrase without
    a leading my/the, and -- when the people book knows the alias -- the
    person's name and surname.  "when did I last talk to my advisor" has to
    find the journal line that says "Dr Peyrovi"."""
    out: list[str] = []

    def add(value):
        value = " ".join(str(value or "").split())
        if value and value.lower() not in {o.lower() for o in out}:
            out.append(value)

    add(target)
    add(_LEADING_ARTICLE_RX.sub("", str(target or "")))
    if memory is not None:
        try:
            person = memory.resolve_person(target)
        except Exception:                          # noqa: BLE001
            person = None
        if isinstance(person, dict):
            name = str(person.get("name") or "")
            add(name)
            if " " in name:
                add(name.split()[-1])              # "Peyrovi" on its own
            add(person.get("alias"))
        else:
            try:
                add(memory.expand_aliases(target))
            except Exception:                      # noqa: BLE001
                log.debug("alias expansion failed", exc_info=True)
    return [t for t in out if len(t) >= 2]


def _term_rx(term: str):
    """Whole-phrase, whole-word, case-insensitive; internal runs of spaces
    match any whitespace so "dr  peyrovi" still matches.

    The boundary is letters-and-digits only, NOT ``\\b``: half the journal
    is window titles, and `\\bthesis\\b` does not match
    "thesis_draft.tex - TeXstudio" because an underscore is a word
    character. "hesis" still fails against "thesis" -- the letter before
    it is a letter."""
    words = str(term or "").split()
    if not words:
        return None
    body = r"\s+".join(re.escape(w) for w in words)
    lead = r"(?<![0-9A-Za-z])" if words[0][0].isalnum() else ""
    tail = r"(?![0-9A-Za-z])" if words[-1][-1].isalnum() else ""
    return re.compile(lead + body + tail, re.I)


def row_text(row) -> str:
    """Everything in a journal row that a mention search may match."""
    kind = row.get("kind")
    if kind == "exchange":
        return f"{row.get('user', '')} {row.get('jarvis', '')}"
    if kind == "tool":
        return f"{row.get('name', '')} {row.get('args', '')} {row.get('text', '')}"
    if kind == "claude":
        return f"{row.get('project', '')} {row.get('text', '')}"
    if kind == "window":
        return str(row.get("title", ""))
    return ""


def find_last_mention(journal_dir, terms: Iterable[str], now: Optional[datetime] = None,
                      max_days: int = MENTION_KEEP_DAYS) -> Optional[dict]:
    """The newest journal row mentioning any of ``terms``, or None.

    Walks the day files newest-first and returns at the first hit -- the
    answer is nearly always today's or yesterday's file, and reading ninety
    days to sort them (journal_rows) would be work thrown away."""
    patterns = [rx for rx in (_term_rx(t) for t in terms) if rx is not None]
    if not patterns:
        return None
    now = now or datetime.now()
    d = Path(journal_dir)
    day = now.date()
    for _ in range(max(1, int(max_days)) + 1):
        path = d / f"{day:%Y-%m-%d}.jsonl"
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                log.debug("journal read failed: %s", path, exc_info=True)
                lines = []
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    when = datetime.fromisoformat(str(row.get("time", "")))
                except (ValueError, TypeError):
                    continue
                if when > now:
                    continue                       # a clock skew, not a memory
                text = row_text(row)
                if any(rx.search(text) for rx in patterns):
                    row["_when"] = when
                    return row
        day -= timedelta(days=1)
    return None


def _part_of_day(when: datetime) -> str:
    if when.hour < LUNCH_HOUR:
        return "morning"
    if when.hour < EVENING_HOUR:
        return "afternoon"
    return "evening"


def _ordinal(n: int) -> str:
    return f"{n}{_ORDINAL_SUFFIX.get(n, 'th')}"


def when_words(when: datetime, now: Optional[datetime] = None) -> str:
    """'this afternoon' / 'yesterday evening' / 'Tuesday afternoon' /
    'last Tuesday' / 'on the 3rd of August'.  Spoken words, no digits
    below the month: a date read as "2026-08-25" is unlistenable."""
    now = now or datetime.now()
    days = (now.date() - when.date()).days
    part = _part_of_day(when)
    if days <= 0:
        return f"this {part}"
    if days == 1:
        return f"yesterday {part}"
    if days < 7:
        return f"{when:%A} {part}"
    if days < 14:
        return f"last {when:%A}"
    stem = f"on the {_ordinal(when.day)} of {when:%B}"
    return stem if when.year == now.year else f"{stem} {when.year}"


def elapsed_words(when: datetime, now: Optional[datetime] = None) -> str:
    """'about two hours' / 'three days' / 'a fortnight' -- the answer to
    "how long since ...", which wants a duration, not a date."""
    now = now or datetime.now()
    secs = max(0.0, (now - when).total_seconds())
    minutes = int(secs // 60)
    if minutes < 2:
        return "barely a minute"
    if minutes < 60:
        return f"{minutes} minutes"
    hours = int(round(secs / 3600))
    if hours < 24:
        return "about an hour" if hours == 1 else f"about {hours} hours"
    days = (now.date() - when.date()).days
    if days == 1:
        return "a day"
    if days < 14:
        return f"{days} days"
    weeks = days // 7
    if weeks == 2:
        return "a fortnight"
    if days < 60:
        return f"{weeks} weeks"
    return f"{days // 30} months"


def mention_snippet(row, n: int = SNIPPET_CHARS) -> str:
    """What the row was, in the second person -- and never more than the
    journal actually holds."""
    kind = row.get("kind")
    if kind == "exchange":
        user = _snip(row.get("user", ""), n)
        if user:
            return f"you said, {user}"
        return f"I said, {_snip(row.get('jarvis', ''), n)}"
    if kind == "tool":
        return f"you had me run {row.get('name') or 'a tool'}"
    if kind == "claude":
        return f"Claude finished {row.get('project') or 'a task'}"
    if kind == "window":
        return f"you had {_title(row.get('title', ''))} open"
    return "there was activity"


def last_mention_line(row, target: str, now: Optional[datetime] = None,
                      duration: bool = False, name: str = "sir") -> str:
    """The spoken answer.  ``duration=True`` for "how long since I ...",
    which is asking for the gap rather than the date."""
    when = row.get("_when")
    if not isinstance(when, datetime):
        return ""
    words = when_words(when, now)
    snippet = mention_snippet(row)
    snippet = f"{snippet[0].upper()}{snippet[1:]}" if snippet else ""
    when_said = f"{words[0].upper()}{words[1:]}"
    if duration:
        # "how long since" asks for the gap; the date is the second half.
        return f"It's been {elapsed_words(when, now)}, {name}. {when_said}: {snippet}."
    return f"{when_said}, {name}. {snippet}."


def no_mention_line(target: str, name: str = "sir") -> str:
    what = " ".join(str(target or "").split()) or "that"
    return f"Nothing in the journal about {what}, {name}."


# ------------------------------------------------------- continuity
# "That would be the third coffee timer, sir." The journal is the ONLY
# place a tool's argument survives -- habits.json rows carry the user's
# utterance and no tool name at all, and memory.format_for_context already
# emits its own "Habit suggestion" line into the same background, so
# merging the two would put two numeric habit lines in every turn.
#
# What this adds over what the model already sees: format_for_prompt
# renders the last four exchanges, so a question repeated twice in a row is
# visible without help. The callback that is NOT visible is the one from
# earlier in the day, which is why a repeated question only counts when its
# first asking has fallen out of that four-turn window.
REPEAT_MIN = 2                 # a count of one is not a callback
MAX_CLAUSES = 2                # at most two clauses, one line
CONVO_WINDOW = 4               # what format_for_prompt already shows
REPEAT_SNIPPET = 60
# The argument that makes a repeat countable, per tool, in priority order.
# Deliberately excludes numbers ("minutes"): "the third five-minute timer"
# is a coincidence, "the third coffee timer" is a fact about his morning.
# It also excludes the defaulted selectors -- when="today", range="today",
# action="list" -- which are the tool's own boilerplate, not his subject;
# quoting them produced 'the 2nd "today" weather check' on the synthetic
# day, which is noise dressed up as a callback.
_SALIENT_ARGS = ("label", "topic", "query", "question", "text", "location",
                 "which", "project")
# The noun the count attaches to. A missing tool falls back to its own name
# with the underscores spoken out.
_TOOL_NOUNS = {
    "set_timer": "timer", "set_alarm": "alarm", "set_reminder": "reminder",
    "manage_schedule": "schedule change", "get_weather": "weather check",
    "get_calendar": "calendar check", "add_event": "diary entry",
    "get_mail": "mail check", "get_time": "clock check",
    "get_briefing": "briefing", "recap_day": "recap", "notes": "note",
    "get_location": "location check", "system_health": "health check",
    "canvas_due": "coursework check", "canvas_grades": "grades check",
    "canvas_announcements": "announcements check",
    "ask_docs": "document question", "screen_qa": "screen check",
    "docs_reindex": "document reindex",
}
CONTINUITY_HEAD = "Earlier today: "
# Correction, round 3: VOICE_RULES bans reciting file names and unperformed
# checks but says NOTHING about numbers, and the measured number
# suppression lives in a few-shot pool that is sampled once per process.
# A block that hands the model a count therefore carries its own rule.
CONTINUITY_RULE = ("Those counts are background: let them colour the reply, "
                   "and never read one aloud unless he asks how many.")


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else \
        {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _times(n: int) -> str:
    return "twice" if n == 2 else f"{n} times"


def _norm(text) -> str:
    """Lowercased, punctuation-free, wake-word-free key for one utterance."""
    t = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    t = re.sub(r"^(?:hey |ok |okay )?jarvis[,!.]?\s*", "", t)
    return re.sub(r"[^a-z0-9 ]", "", t).strip()


def _salient(args) -> str:
    """The one argument worth counting on, "" when the call had none."""
    if not isinstance(args, dict):
        return ""
    for key in _SALIENT_ARGS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return re.sub(r"\s+", " ", val).strip()[:40]
    return ""


def _tool_clause(name: str, arg: str, count: int) -> str:
    noun = _TOOL_NOUNS.get(name) or str(name).replace("_", " ")
    if arg:
        return f'{_ordinal(count)} "{arg}" {noun}'
    return f"{_ordinal(count)} {noun}"


def journal_repeats(rows, max_clauses: int = MAX_CLAUSES) -> str:
    """The "Earlier today" block for the per-turn background, or "".

    Two lines at most: the counts, then the rule that keeps them out of his
    ear. ``rows`` is whatever ``ContextEngine.journal_rows`` returned for
    today; nothing here reads a file or a clock.
    """
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return ""
    tools: dict = {}
    exchanges: list = []
    for row in rows:
        kind = row.get("kind")
        if kind == "tool":
            if not row.get("ok", True):
                continue          # a failed call is not something he did twice
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            key = (name, _salient(row.get("args")))
            tools[key] = tools.get(key, 0) + 1
        elif kind == "exchange":
            said = str(row.get("user") or "").strip()
            if said:
                exchanges.append(said)

    clauses = []
    for (name, arg), count in tools.items():
        if count >= REPEAT_MIN:
            clauses.append((count, 0, _tool_clause(name, arg, count)))

    seen: dict = {}
    for i, said in enumerate(exchanges):
        key = _norm(said)
        if key:
            seen.setdefault(key, []).append(i)
    cutoff = len(exchanges) - CONVO_WINDOW
    for key, hits in seen.items():
        if len(hits) < REPEAT_MIN or hits[0] >= cutoff:
            continue          # still inside the window the model already sees
        said = _snip(exchanges[hits[0]], REPEAT_SNIPPET)
        clauses.append((len(hits), 1, f'he asked "{said}" {_times(len(hits))}'))

    if not clauses:
        return ""
    clauses.sort(key=lambda c: (-c[0], c[1]))
    line = CONTINUITY_HEAD + "; ".join(c[2] for c in clauses[:max_clauses]) + "."
    return f"{line}\n{CONTINUITY_RULE}"


# ----------------------------------------------------------- sampler
class ActivitySampler:
    """Every ``interval`` seconds, feed the focused window's title to the
    context engine's journal (``journal_window`` dedupes and tracks it).
    The engine's own cached xdotool probe is the only source: a failed or
    empty probe -- a locked screen, no X -- journals nothing. start() /
    stop() own the daemon thread (the jarvis/headsup.py pattern)."""

    def __init__(self, context, interval: Optional[float] = None,
                 keep_days: int = DEFAULT_KEEP_DAYS):
        self._ctx = context
        self.interval = float(interval or DEFAULT_INTERVAL_S)
        self.keep_days = int(keep_days or 0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples = 0

    def tick(self) -> bool:
        """One sample; True when a new window was journaled."""
        ctx = self._ctx
        if ctx is None:
            return False
        try:
            title = ctx._get_active_window()
        except Exception:                          # noqa: BLE001
            log.debug("window sample failed", exc_info=True)
            return False
        self.samples += 1
        try:
            return bool(ctx.journal_window(title))
        except Exception:                          # noqa: BLE001
            log.debug("journal_window failed", exc_info=True)
            return False

    def prune(self) -> int:
        """Delete day files older than keep_days; returns how many."""
        if self.keep_days <= 0 or self._ctx is None:
            return 0
        try:
            d = Path(self._ctx.journal_dir())
        except Exception:                          # noqa: BLE001
            return 0
        cutoff = (datetime.now() - timedelta(days=self.keep_days)).date()
        removed = 0
        for path in d.glob("*.jsonl") if d.is_dir() else []:
            try:
                day = datetime.strptime(path.stem, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < cutoff:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    log.debug("journal prune failed: %s", path, exc_info=True)
        return removed

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="activity-sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.prune()
        except Exception:                          # noqa: BLE001
            log.exception("journal prune failed")
        # First sample after a short settle: at boot the focused window is
        # the terminal that launched Jarvis, not what Hunter is doing.
        if self._stop.wait(10.0):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:                      # noqa: BLE001
                log.exception("activity sample failed")
            if self._stop.wait(max(1.0, self.interval)):
                return


# --------------------------------------------------------------- tool
def _now() -> datetime:
    """The clock (tests pin it)."""
    return datetime.now()


def _cfg_get(cfg, dotted, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        cur = cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return default if cur is None else cur
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            val = get(dotted, default)
            return default if val is None else val
        except Exception:                          # noqa: BLE001
            return default
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None:
            return default
    return cur


def make_tools(cfg, services) -> list[ToolSpec]:
    engine = getattr(services, "context_engine", None) if services is not None else None

    # Parked, not started: the app starts it in start_assistant beside the
    # health watchdog and stops it in stop_assistant.
    if services is not None and engine is not None and \
            getattr(services, "activity_sampler", None) is None:
        try:
            interval = float(_cfg_get(cfg, "journal.window_interval_s", DEFAULT_INTERVAL_S)
                             or DEFAULT_INTERVAL_S)
            keep = int(_cfg_get(cfg, "journal.keep_days", DEFAULT_KEEP_DAYS) or 0)
            services.activity_sampler = ActivitySampler(engine, interval=interval,
                                                        keep_days=keep)
        except (AttributeError, TypeError, ValueError):
            log.debug("services does not accept activity_sampler")

    def recap_day(when: str = "", **_) -> ToolResult:
        if engine is None or not hasattr(engine, "journal_rows"):
            return ToolResult(text="journal unavailable", ok=False,
                              speak=NO_JOURNAL_LINE)
        since, until, label = parse_window(str(when or ""), _now())
        try:
            rows = engine.journal_rows(since, until)
        except Exception:                          # noqa: BLE001
            log.exception("journal read failed")
            return ToolResult(text="journal unreadable", ok=False,
                              speak=NO_JOURNAL_LINE)
        text = digest(rows, label)
        if not text:
            line = NOTHING_LINE.format(label=label)
            return ToolResult(text=line, speak=line)
        # The spoken recap is at most four sentences (the 500-char TTS
        # ceiling); the whole digest goes on a card, display only.
        try:
            bus.publish(JarvisReply(text=text, speak=False))
        except Exception:                          # noqa: BLE001
            log.debug("recap card publish failed", exc_info=True)
        return ToolResult(text=text, max_sentences=RECAP_SENTENCES)

    return [ToolSpec(
        name="recap_day",
        description=("Recap what Hunter did: today, this morning, before "
                     "lunch, afternoon, yesterday, last N hours."),
        parameters={"type": "object", "properties": {
            "when": {"type": "string",
                     "description": "the period, as said (default today)"}}},
        handler=recap_day)]
