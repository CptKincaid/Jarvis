"""The owner gate's own ledger, and the scorecard read off it.

WHY THIS EXISTS. The gate ships in SHADOW: it works out the verdict it
would have given and then answers the turn anyway. That is the right way to
turn a feature like this on, and it is worth nothing at all unless somebody
can READ the verdicts afterwards. Until now they existed only as prose in
``jarvis.log`` -- "gate: shadow -- would have refused this turn" -- which
cannot be counted, cannot be split by day, and does not say WHY it would
have refused or whether anything was even measuring at the time.

SO THIS IS A RECORD OF DECISIONS AND SCORES, AND OF NOTHING ELSE. One JSON
object per gated verdict:

    {"at": 1757000000.0, "mode": "shadow", "source": "voice",
     "admit": true, "would_refuse": false, "consumed": false,
     "how": "voice", "role": "owner", "who": "hunter",
     "why": "voice named hunter", "voice_running": true,
     "face_running": false, "score": 0.41, "rescued": false}

WHAT IS DELIBERATELY NOT IN IT: the transcript, the sentence, the
passphrase, the override code, the way anybody was addressed, any
embedding, any audio. ``who`` is the registry LABEL the gate already had in
hand, never a new identification, and ``why`` is one of a handful of
strings written in this repository -- the one place an exception message
could have reached it, the gate's own fault path, is stamped with the
exception TYPE and nothing else (see ``OwnerGate.judge``). A ledger that
leaked what was said would be strictly worse than the prose log it
replaces, because it would be machine-readable.

WHAT THE SCORECARD CAN AND CANNOT SAY. It can count what the gate decided.
It cannot know who was really speaking -- nothing in this repository does,
which is the whole reason the gate is a guess in the first place -- so
every "was that actually him" number here is an inference from a decision,
labelled as one in the output rather than buried in a docstring. The one
piece of real evidence available is that a refusal followed within the
grant window by HIM opening the floor by hand (the phrase or the typed
code) is him saying, at the time, that the refusal was wrong.

Pure over dicts apart from ``append`` and ``read``, which are the only I/O
and are both best-effort: a gate that cannot write its ledger must never be
the reason he cannot speak to his own assistant.
"""
from __future__ import annotations

import errno
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("gate")

DEFAULT_PATH = PATHS.LOG_DIR / "gate.jsonl"

# How long after a refusal his own phrase or code still counts as him
# saying "that one was me". The same 300 s as gate.GRANT_S, and for the
# same reason: it is one sitting.
OPENED_S = 300.0

DAY_S = 86400.0
# The window the scorecard defaults to. A week is long enough to have a
# trend in it and short enough that a change he made on Tuesday is not
# averaged away by the fortnight before it.
DEFAULT_WINDOW_S = 7 * DAY_S

# Kept as literals rather than imported from jarvis.gate: the reader must
# be able to parse a ledger written by a version of the gate that has since
# been edited, and an import would also drag the registry into a script
# whose whole promise is that it reads one file.
HOW_VOICE = "voice"
HOW_FACE = "face"
HOW_PHRASE = "passphrase"
HOW_NOBODY = "nobody"
HOW_GRANT = "grant"
HOW_CODE = "code"
HOW_BLIND = "blind"
HOW_OFF = "off"
HOW_FAULT = "fault"
HOW_EXEMPT = "exempt"
ADMITTED_HOWS = (HOW_VOICE, HOW_FACE, HOW_PHRASE, HOW_GRANT, HOW_CODE)
WINDOW_HOWS = (HOW_GRANT, HOW_CODE)
ROLE_OWNER = "owner"

# `why` is code-authored, but it is not worth an unbounded field.
WHY_MAX = 120


# ------------------------------------------------------------- the record
def row(decision, *, mode: str, source: str, stats=None,
        why: Optional[str] = None, legs=None, at: Optional[float] = None,
        turn: str = "") -> dict:
    """One ledger row from a ``gate.Decision``. Never raises on odd input.

    ``turn`` IS WHAT MAKES A TURN A TURN. One utterance can reach this
    function twice: the app judges a clip the speaker filter dropped inside
    its rescue, and judges it AGAIN once the rescue has cleared ``rejected``
    (jarvis/app.py, ``_gate_rescue_inner`` then ``_process_audio``). Two
    verdicts, one thing he said -- and it is the path his first Knightfall
    test takes, so without an id the headline count and the "somebody else
    got in" figure both inflate on the first thing he tries. It is an
    opaque stamp and nothing else: no words, no identity, no clip.
    """
    legs = legs or {}
    return {
        "at": float(time.time() if at is None else at),
        "turn": str(turn or ""),
        "mode": str(mode or ""),
        "source": str(source or ""),
        "admit": bool(getattr(decision, "admit", True)),
        "would_refuse": bool(getattr(decision, "would_refuse", False)),
        "consumed": bool(getattr(decision, "consumed", False)),
        "how": str(getattr(decision, "how", "") or ""),
        "role": str(getattr(decision, "role", "") or ""),
        "who": str(getattr(decision, "who", "") or ""),
        "why": str(why if why is not None
                   else getattr(decision, "why", ""))[:WHY_MAX],
        "voice_running": bool(legs.get("voice", False)),
        "face_running": bool(legs.get("face", False)),
        "score": best_score(stats),
        "rescued": bool(legs.get("rescued", False)),
    }


def best_score(stats) -> Optional[float]:
    """The speaker filter's best score for the clip, or None when it was
    not running. A NUMBER, which is the only thing about a voice that may
    be written down here."""
    if not isinstance(stats, dict):
        return None
    scores = stats.get("scores")
    try:
        vals = [float(s) for s in scores]
    except (TypeError, ValueError):
        return None
    return max(vals) if vals else None


# ------------------------------------------------------------------- I/O
def append(path, rec: dict) -> None:
    """Append one row. Best-effort: a full disk is not worth a turn."""
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except (OSError, TypeError, ValueError):
        log.debug("gate.jsonl append failed", exc_info=True)


def writer(path=None) -> Callable[[dict], None]:
    """The callable the app hands to ``OwnerGate(record=...)``."""
    target = Path(path or DEFAULT_PATH)
    return lambda rec: append(target, rec)


# WHAT THE READER SAYS WHEN IT COULD NOT READ. These are constants because
# they are the whole point of the fix: a scorecard that silently reports 0
# when it could not read something is WORSE THAN NO SCORECARD AT ALL,
# because he acts on it. Measured before they existed: a missing file, a
# file of rubbish, a file with no read permission and a --path pointing at
# a directory all printed output BYTE-IDENTICAL to a healthy, quiet, empty
# ledger -- and the advice he then read was "the usual reason is that
# nobody is enrolled ... Enrol yourself first", sending him back to redo an
# enrolment he had already done.
TROUBLE_MISSING = ("The ledger does not exist at that path. Nothing was "
                   "read, so every number below would be a zero this "
                   "reader invented.")
TROUBLE_DIR = ("That path is a directory, not a ledger file. Nothing was "
               "read.")
TROUBLE_PERM = ("The ledger is there, but this account has no permission "
                "to read it. Nothing was read.")
TROUBLE_UNREADABLE = ("The ledger could not be read (%s). Nothing was "
                      "read.")
TROUBLE_ALL_BAD = ("All %d lines in the ledger are UNREADABLE as verdicts. "
                   "The file is there and it is not empty, so this is a "
                   "corrupt or a foreign file, not a quiet week.")
TROUBLE_SOME_BAD = ("%d of %d lines in the ledger could not be read and "
                    "are missing from every number below.")
# The benign one, and it has to be told apart from the rest or the warning
# above becomes noise he learns to ignore: the app appends while this runs,
# so the very last line can be half-written. That is normal and expected.
TROUBLE_TORN_TAIL = ("The last line of the ledger was still being written "
                     "when this ran, so it is not counted. That is normal "
                     "-- Jarvis appends to this file while you read it.")

TROUBLE_ADVICE = (
    "No verdicts were read -- which is NOT the same fact as no verdicts\n"
    "having been recorded, and the difference is the whole reason this\n"
    "sentence exists. Do not read anything below as a measurement until the\n"
    "line above is fixed. Check the --path you asked for, and whether\n"
    "Jarvis has been restarted since the ledger was added.\n"
    "\n"
    "/tmp is wiped at boot on this machine, so an absent ledger the morning\n"
    "after a reboot is expected rather than broken."
)


@dataclass
class LedgerRead:
    """What one attempt at the ledger actually got, INCLUDING what it did
    not get. A reader that hands back only the good rows lets a missing
    file, a directory, a permission error and a corrupt file all arrive at
    the printout as "a quiet week" -- which is the one failure this whole
    instrument exists to not commit."""

    rows: list = field(default_factory=list)
    lines_seen: int = 0
    lines_dropped: int = 0
    trouble: str = ""
    # True only for the benign case: the sole dropped line was the last one.
    torn_tail: bool = False
    path: str = ""


def read_report(path) -> LedgerRead:
    """Every well-formed row, oldest first, AND an account of what could not
    be read. READ-ONLY. Never raises: an instrument that dies on a bad file
    is no better than one that lies about it."""
    rep = LedgerRead(path=str(path))
    p = Path(path)
    if p.is_dir():
        rep.trouble = TROUBLE_DIR
        return rep
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        rep.trouble = TROUBLE_MISSING
        return rep
    except PermissionError:
        rep.trouble = TROUBLE_PERM
        return rep
    except OSError as exc:
        # THE ERRNO NAME, NEVER THE MESSAGE. An OSError's string carries the
        # path it failed on, which is the one thing in this function that
        # could be his; the code is a fact about the kernel and nobody's
        # data. Same rule as the gate's own fault path.
        rep.trouble = TROUBLE_UNREADABLE % errno.errorcode.get(
            exc.errno, "unknown error")
        return rep
    last_bad_at = -1
    lines = [ln.strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        if not line:
            continue
        rep.lines_seen += 1
        try:
            rec = json.loads(line)
        except ValueError:
            rep.lines_dropped += 1
            last_bad_at = i
            continue
        if not (isinstance(rec, dict)
                and isinstance(rec.get("at"), (int, float))):
            rep.lines_dropped += 1
            last_bad_at = i
            continue
        rep.rows.append(rec)
    rep.rows.sort(key=lambda r: r["at"])
    if not rep.lines_dropped:
        return rep
    tail = max((i for i, ln in enumerate(lines) if ln), default=-1)
    if rep.lines_dropped == 1 and last_bad_at == tail:
        rep.torn_tail = True
        rep.trouble = TROUBLE_TORN_TAIL
    elif not rep.rows:
        rep.trouble = TROUBLE_ALL_BAD % rep.lines_seen
    else:
        rep.trouble = TROUBLE_SOME_BAD % (rep.lines_dropped, rep.lines_seen)
    return rep


def read(path) -> list:
    """Every well-formed row, oldest first. Kept for callers that only want
    the rows; anything that PRINTS must use ``read_report`` instead, so it
    can say what it could not read."""
    return read_report(path).rows


# ------------------------------------------------------------ the counting
@dataclass
class Bucket:
    start: float
    width_s: float
    judged: int = 0
    admitted: int = 0
    refused: int = 0
    no_opinion: int = 0

    def label(self) -> str:
        fmt = "%a %d %b" if self.width_s >= 20 * 3600 else "%d %b %H:%M"
        return time.strftime(fmt, time.localtime(self.start))


@dataclass
class Scorecard:
    # TURNS, not rows. One utterance can leave two verdicts in the file --
    # see ``row``'s ``turn`` -- and a headline labelled "turns" that counts
    # rows is a number he would act on.
    judged: int = 0
    # Rows that carried a verdict, before turns were folded together. Both
    # numbers are true; they answer different questions, and the printout
    # names which is which.
    verdicts: int = 0
    seen: int = 0
    # A verdict whose leg this reader does not know -- a gate that grew a
    # new leg, or a foreign file. Counted and NAMED rather than vanishing:
    # before this existed, 3 judged turns with two unknown legs printed
    # 33.3% / 0% / 0% / 0% and the missing 2 appeared nowhere at all.
    unclassified: int = 0
    unknown_hows: dict = field(default_factory=dict)
    # What the READER could not do, carried from LedgerRead so the printout
    # can say it above the fold instead of reporting a confident zero.
    trouble: str = ""
    torn_tail: bool = False
    lines_seen: int = 0
    lines_dropped: int = 0
    rows_in_file: int = 0
    oldest_in_file: Optional[float] = None
    newest_in_file: Optional[float] = None
    admitted_owner: int = 0
    admitted_other: int = 0
    window_admits: int = 0
    refused: int = 0
    refused_nobody: int = 0
    refused_scope: int = 0
    refused_then_he_opened: int = 0
    no_opinion: int = 0
    off_turns: int = 0
    fault_turns: int = 0
    phrase_turns: int = 0
    modes: dict = field(default_factory=dict)
    hows: dict = field(default_factory=dict)
    trend: list = field(default_factory=list)
    first_at: Optional[float] = None
    last_at: Optional[float] = None
    refused_scores: list = field(default_factory=list)
    admitted_scores: list = field(default_factory=list)

    def _rate(self, n) -> Optional[float]:
        return None if not self.judged else 100.0 * n / self.judged

    @property
    def refused_rate(self) -> Optional[float]:
        return self._rate(self.refused)

    @property
    def admitted_rate(self) -> Optional[float]:
        return self._rate(self.admitted_owner + self.admitted_other)

    @property
    def no_opinion_rate(self) -> Optional[float]:
        return self._rate(self.no_opinion)

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "trend"}
        out["trend"] = [{"label": b.label(), "judged": b.judged,
                         "admitted": b.admitted, "refused": b.refused,
                         "no_opinion": b.no_opinion} for b in self.trend]
        out["refused_rate"] = self.refused_rate
        out["admitted_rate"] = self.admitted_rate
        out["cannot_know"] = [CANNOT_KNOW_HEADLINE] + list(CANNOT_KNOW)
        # --json is for a plot, and a plot of zeroes he cannot read is the
        # same lie in a different shape. `trouble` is "" when the whole
        # file was read, and a sentence when it was not.
        out["trouble"] = self.trouble
        return out


def _is_refusal(r: dict) -> bool:
    return bool(r.get("would_refuse")) or not bool(r.get("admit", True))


def summarise(records: Iterable[dict], *, now: Optional[float] = None,
              window_s: float = DEFAULT_WINDOW_S,
              buckets: Optional[int] = None,
              opened_s: float = OPENED_S) -> Scorecard:
    """Count what the gate decided over the last ``window_s`` seconds."""
    now = time.time() if now is None else float(now)
    start = now - float(window_s)
    report = records if isinstance(records, LedgerRead) else None
    source = report.rows if report is not None else records
    everything = [r for r in source
                  if isinstance(r, dict)
                  and isinstance(r.get("at"), (int, float))]
    rows = sorted((r for r in everything
                   if start <= float(r["at"]) <= now),
                  key=lambda r: float(r["at"]))
    n = buckets or (7 if window_s >= 2 * DAY_S else 12)
    n = max(1, int(n))
    width = float(window_s) / n
    s = Scorecard(trend=[Bucket(start=start + i * width, width_s=width)
                         for i in range(n)])
    # WHAT THE FILE HELD, whether or not any of it fell in the window. An
    # empty window over a populated ledger is a different sentence from an
    # empty ledger, and the advice for one is wrong for the other.
    s.rows_in_file = len(everything)
    if everything:
        ats = [float(r["at"]) for r in everything]
        s.oldest_in_file, s.newest_in_file = min(ats), max(ats)
    if report is not None:
        s.trouble = report.trouble
        s.torn_tail = report.torn_tail
        s.lines_seen = report.lines_seen
        s.lines_dropped = report.lines_dropped
    if not rows:
        return s
    s.seen = len(rows)
    s.first_at, s.last_at = float(rows[0]["at"]), float(rows[-1]["at"])

    # When HE opened the floor by hand, in ledger order, so a refusal can
    # ask whether he answered it. Taken over every row, because the moment
    # is real whichever verdict of the turn recorded it.
    opened = [float(r["at"]) for r in rows
              if r.get("consumed") or str(r.get("how")) in WINDOW_HOWS]
    # ONE UTTERANCE IS ONE TURN. Rows sharing a ``turn`` stamp are the same
    # thing he said judged more than once, and THE LAST OF THEM IS THE
    # VERDICT THAT STOOD -- the earlier ones are the rescue's probe, taken
    # before ``rejected`` was cleared. A row with no stamp is its own turn:
    # a ledger written by a gate that did not stamp turns must never be
    # folded together by guesswork about timestamps.
    turns, by_turn = [], {}
    for r in rows:
        tid = str(r.get("turn", "") or "")
        if tid and tid in by_turn:
            by_turn[tid].append(r)
            continue
        group = [r]
        if tid:
            by_turn[tid] = group
        turns.append(group)
    s.verdicts = sum(1 for r in rows if str(r.get("how", ""))
                     not in (HOW_OFF, HOW_FAULT, HOW_EXEMPT))
    # Per TURN, like the headline, so the breakdowns and the top line
    # cannot disagree by the number of rescued clips in the window.
    s.hows = dict(Counter(str(g[-1].get("how", "")) for g in turns))
    s.modes = dict(Counter(str(g[-1].get("mode", "")) for g in turns))
    for group in turns:
        r = group[-1]
        how = str(r.get("how", ""))
        at = float(r["at"])
        idx = min(n - 1, max(0, int((at - start) // width)))
        b = s.trend[idx]
        if how == HOW_OFF:
            s.off_turns += 1
            continue
        if how == HOW_FAULT:
            s.fault_turns += 1
            continue
        if how == HOW_EXEMPT:
            continue
        s.judged += 1
        b.judged += 1
        if any(g.get("consumed") for g in group):
            s.phrase_turns += 1
        if how == HOW_BLIND:
            s.no_opinion += 1
            b.no_opinion += 1
            continue
        if _is_refusal(r):
            s.refused += 1
            b.refused += 1
            if how == HOW_NOBODY:
                s.refused_nobody += 1
            else:
                s.refused_scope += 1
            if isinstance(r.get("score"), (int, float)):
                s.refused_scores.append(float(r["score"]))
            if any(at < o <= at + opened_s for o in opened):
                s.refused_then_he_opened += 1
            continue
        if how in ADMITTED_HOWS:
            b.admitted += 1
            if str(r.get("role")) == ROLE_OWNER:
                s.admitted_owner += 1
            else:
                s.admitted_other += 1
            if how in WINDOW_HOWS:
                s.window_admits += 1
            if isinstance(r.get("score"), (int, float)):
                s.admitted_scores.append(float(r["score"]))
            continue
        # A VERDICT THIS READER DOES NOT UNDERSTAND. It is admitted (it is
        # not a refusal) on a leg no constant in this file names -- a gate
        # that has grown a leg since, or a foreign file. It must be COUNTED
        # AND NAMED rather than falling through into nothing, or the four
        # lines above stop adding up to the headline and the percentages
        # quietly stop meaning anything.
        s.unclassified += 1
        s.unknown_hows[how or "?"] = s.unknown_hows.get(how or "?", 0) + 1
    return s


# ----------------------------------------------------------- the printout
# Printed WITHOUT wrapping, so the one sentence that matters is never
# broken across two lines by a terminal width.
CANNOT_KNOW_HEADLINE = ("It CANNOT say who was really speaking -- "
                        "nothing in Jarvis can.")

CANNOT_KNOW = (
    "This ledger records what the GATE DECIDED. Nothing in Jarvis knows who "
    "was at the microphone, which is the whole reason the gate is a "
    "judgement and not a lookup.",
    "So no line above is a measurement of accuracy. A refusal counted here "
    "may have been you, or may have been the television. The scorecard "
    "cannot tell the two apart and does not try.",
    "The one piece of real evidence is the \"you answered it\" line: a "
    "refusal you followed, within five minutes, by opening the floor with "
    "the phrase or the code is you saying at the time that it was wrong.",
    "Scores are the speaker filter's, not the gate's. The gate applies no "
    "threshold of its own; it reuses the filter's verdict.",
    "Turns from the keyboard, the socket, the phone and the intercom are "
    "not gated and are not counted here. Only the microphone is.",
)

# THE LEDGER IS THERE AND READABLE AND HOLDS ROWS -- just none in the
# window he asked for. Telling him to go and enrol himself here is wrong
# and expensive: he has already done it, and the file in front of him
# proves it.
OUT_OF_WINDOW_HELP = (
    "No verdicts in THIS WINDOW -- but the ledger is not empty.\n"
    "\n"
    "It holds %d verdict(s), and every one of them falls OUTSIDE the window\n"
    "you asked for. The newest is %s.\n"
    "\n"
    "Ask for a wider window (--days 30), or point --hours at the stretch you\n"
    "actually meant. Nothing here says anything about enrolment."
)

EMPTY_HELP = (
    "No verdicts in this window.\n"
    "\n"
    "If the ledger is empty everywhere, the usual reason is that nobody is\n"
    "enrolled: with no ~/.local/state/jarvis/people.json the gate is OFF\n"
    "and takes no view at all. Enrol yourself first -- docs/assistant-setup.md,\n"
    "\"The owner gate: enrolling yourself, shadow, and the scorecard\" --\n"
    "then let Jarvis run for a day and come back.\n"
    "\n"
    "The other reasons: Jarvis has not been restarted since the ledger was\n"
    "added, or he simply has not heard a voice turn in this window.\n"
    "\n"
    "This is the message for a ledger that WAS read and held nothing. When\n"
    "the file cannot be read at all, the card says so at the top instead."
)

# Every text the reader may print in place of a table. Pinned by test
# against the document's real headings: a pointer to a section he cannot
# find is worse than none, and the empty case is exactly the moment he goes
# looking for the next step.
HELP_TEXTS = (EMPTY_HELP, OUT_OF_WINDOW_HELP, TROUBLE_ADVICE)


def _n(value: Optional[float]) -> str:
    return "--" if value is None else "%.1f%%" % value


def _line(label: str, count: int, rate: Optional[float] = None,
          width: int = 50) -> str:
    dots = label + " " + "." * max(2, width - len(label))
    tail = "" if rate is None else "  (%s)" % _n(rate)
    return "  %s %5d%s" % (dots, count, tail)


def render(s: Scorecard, *, now: Optional[float] = None,
           window_s: float = DEFAULT_WINDOW_S, path: str = "") -> str:
    """The whole printout: his summary first, the detail after, the
    assumptions last and never optional."""
    now = time.time() if now is None else float(now)
    span = "%s to %s" % (
        time.strftime("%a %d %b %H:%M", time.localtime(now - window_s)),
        time.strftime("%a %d %b %H:%M", time.localtime(now)))
    out = ["Owner-gate scorecard -- %s" % span,
           "ledger: %s" % (path or DEFAULT_PATH), ""]
    # ABOVE EVERYTHING, INCLUDING THE ADVICE. A scorecard that silently
    # reports 0 when it could not read something is worse than no scorecard
    # at all, because he acts on it -- so "I could not read that" is never
    # dressed up as "you are not enrolled", and never a footnote under a
    # table of zeroes he has already believed.
    out.extend(_trouble_block(s))
    if s.judged == 0:
        out.append(_empty_help(s))
        out.append("")
        out.append("What this cannot know")
        out.append("  " + CANNOT_KNOW_HEADLINE)
        out.append("")
        out.extend(_wrap(CANNOT_KNOW))
        return "\n".join(out)

    modes = ", ".join("%s (%d)" % (m, c) for m, c in
                      sorted(s.modes.items(), key=lambda kv: -kv[1]))
    out.append("Mode the verdicts were taken in: %s" % (modes or "unknown"))
    out.append("")
    out.append(_line("Turns the gate judged", s.judged))
    out.append(_line("  it would have ANSWERED you", s.admitted_owner,
                     s._rate(s.admitted_owner)))
    out.append(_line("  it would have ANSWERED someone else",
                     s.admitted_other, s._rate(s.admitted_other)))
    out.append(_line("  it would have REFUSED", s.refused, s.refused_rate))
    out.append(_line("      nobody was named", s.refused_nobody))
    out.append(_line("      named, but out of scope", s.refused_scope))
    out.append(_line("  it had NO OPINION (nothing measuring)",
                     s.no_opinion, s.no_opinion_rate))
    # NOT A FIFTH SIBLING. The four lines above are mutually exclusive and
    # add up to the headline; this one is a SUBSET of "ANSWERED you", and
    # sitting at the same indent it read as though it should add up too.
    out.append(_line("  (of the above, you opened the floor yourself)",
                     s.phrase_turns))
    if s.unclassified:
        out.append(_line("  a verdict this reader does not understand",
                         s.unclassified, s._rate(s.unclassified)))
        out.append("      legs it has no name for: %s"
                   % ", ".join(sorted(s.unknown_hows)))
    out.append("")
    out.append("Being locked out of your own house -- the failure you fear")
    if s.refused == 0:
        out.append("  Nothing was refused in this window.")
    else:
        out.append("  %d turns would have been refused. HOW MANY OF THOSE "
                   "WERE YOU" % s.refused)
        out.append("  cannot be known from this ledger -- see below. The "
                   "nearest evidence:")
        out.append("  %d of them you answered within five minutes with the "
                   "phrase or the code," % s.refused_then_he_opened)
        out.append("  which is you saying at the time that the gate had it "
                   "wrong.")
        if s.refused_scores:
            out.append("  Speaker score on the refused turns: best %.2f, "
                       "median %.2f." % (max(s.refused_scores),
                                         _median(s.refused_scores)))
    out.append("")
    out.append("Somebody else getting in -- the other failure")
    out.append("  %d turns would have been answered as somebody other than "
               "you." % s.admitted_other)
    out.append("  %d were admitted inside a window you had opened; a window "
               "admits the" % s.window_admits)
    out.append("  ROOM for five minutes, not your voice. Anyone speaking in "
               "it is taken for you.")
    out.append("")
    out.append("Trend")
    for b in s.trend:
        if b.judged == 0:
            out.append("  %-14s        --" % b.label())
            continue
        out.append("  %-14s %5d judged  %4d refused  %4d no-opinion"
                   % (b.label(), b.judged, b.refused, b.no_opinion))
    out.append("")
    out.append("Detail")
    for how, count in sorted(s.hows.items(), key=lambda kv: -kv[1]):
        out.append(_line("  leg: %s" % (how or "?"), count))
    if s.off_turns:
        out.append(_line("  gate switched off", s.off_turns))
    if s.fault_turns:
        out.append(_line("  gate faulted (admitted, by design)",
                         s.fault_turns))
    if s.verdicts != s.judged:
        out.append(_line("  verdicts recorded (a rescued clip is judged "
                         "twice)", s.verdicts))
    if s.admitted_scores:
        out.append("  Speaker score on admitted turns: median %.2f."
                   % _median(s.admitted_scores))
    out.append("")
    out.append("Switching to enforce")
    out.extend(_wrap(ENFORCE_ADVICE))
    out.append("")
    out.append("What this cannot know")
    out.append("  " + CANNOT_KNOW_HEADLINE)
    out.append("")
    out.extend(_wrap(CANNOT_KNOW))
    return "\n".join(out)


ENFORCE_ADVICE = (
    "THERE IS NO NUMBER HERE THAT MAKES ENFORCE SAFE, and this instrument "
    "will not invent one. It counts decisions, not correctness, so it "
    "cannot tell you a false-refusal rate -- only you know which of those "
    "turns were you.",
    "What it CAN do is show you two things going to zero and staying "
    "there: refusals you answered with the phrase or the code, and turns "
    "with no opinion. The first means the gate refused you and you said so "
    "at the time; the second means it was guessing blind.",
    "Enforce is reversible in one line (owner.mode back to shadow) and the "
    "phrase and the typed code both work in every mode, so the cost of "
    "being wrong is a sentence you have to repeat, not a lockout. That, "
    "and not a percentage, is the argument for trying it.",
)


def _empty_help(s: Scorecard) -> str:
    """Which of the three empty cases this is. They are three different
    facts about his box and only one of them is about enrolment."""
    if s.trouble and not s.rows_in_file:
        return TROUBLE_ADVICE
    if s.rows_in_file:
        when = ("unknown" if s.newest_in_file is None else
                time.strftime("%a %d %b %H:%M",
                              time.localtime(s.newest_in_file)))
        return OUT_OF_WINDOW_HELP % (s.rows_in_file, when)
    return EMPTY_HELP


def _trouble_block(s: Scorecard) -> list:
    """The sentence he has to see, in the place he actually reads, whenever
    the input was not a whole healthy file. Empty list when it was."""
    if not s.trouble:
        return []
    if s.torn_tail:
        # Benign, and told apart on purpose: Jarvis appends while he reads.
        # Crying wolf here every single run is how the loud version above
        # gets ignored.
        return ["Note"] + _wrap((s.trouble,)) + [""]
    # The HEADING is never wrapped -- it is the sentence that has to survive
    # a narrow terminal, exactly like CANNOT_KNOW_HEADLINE.
    return ["SOMETHING IS WRONG WITH THE LEDGER -- this is NOT a quiet week"] \
        + _wrap((s.trouble,)) + [""]


def _median(vals) -> float:
    vals = sorted(float(v) for v in vals)
    mid = len(vals) // 2
    if not vals:
        return 0.0
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _wrap(paragraphs, width: int = 74) -> list:
    out = []
    for para in paragraphs:
        line = "  "
        for word in str(para).split():
            if len(line) + len(word) + 1 > width:
                out.append(line)
                line = "  " + word
            else:
                line = (line + " " + word) if line.strip() else line + word
        if line.strip():
            out.append(line)
        out.append("")
    return out[:-1] if out else out
