"""Jarvis reading his own log: "anything wrong in your log?" and "why was
that slow?".

`cluster_warnings` groups the WARNING/ERROR tail of jarvis.log by logger and
normalised message (digits, paths and ids folded), drops the known noise,
and keeps the traceback that followed each record.  `last_turn` /
`turn_outliers` read the turn ledger (turns.jsonl, jarvis/turnclock.py) so
"why was that slow" can say where the seconds went — stt vs the answer vs
the silence before the recorder stopped.  The route stage cannot be split
into model time and tool time: the ledger carries no mark between them.

Pure over lines and dicts; the two `read_*` helpers are the only I/O, and
nothing here touches the bus, a model or the UI.  The log line format is
logs.py's "%H:%M:%S.mmm name LEVEL message": there is NO date in it, so a
"last N minutes" window cannot survive midnight — the window is the last
TAIL_LINES lines of the file instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

LINE_RX = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d{3}) (?P<name>\S+) "
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) (?P<msg>.*)$")
LEVELS = ("WARNING", "ERROR", "CRITICAL")
TAIL_LINES = 400
# (logger, substring): a message from that logger containing the substring
# is dropped.  Why each is here (live log, 2026-08-30):
#   - the tool-budget warning was 650 of the day's 774 WARNING/ERROR lines
#     (one per turn: the registry counts every module's tools);
#   - the memory migration notice repeats on every boot once the legacy
#     dir has been moved aside.
NOISE = (("jarvis.tools.registry", "tools registered"),
         ("jarvis.memory", "already exists"))
SLOW_WAIT_S = 5.0
_PATH_RX = re.compile(r"(?<![\w.])(?:~|/)[\w./@%+-]+")
_HEX_RX = re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{6,}\b")
_NUM_RX = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Record:
    time: str
    name: str
    level: str
    msg: str
    extra: list = field(default_factory=list)      # traceback / continuation lines


@dataclass
class Cluster:
    logger: str
    level: str
    message: str          # normalised
    count: int
    last_time: str
    example: str          # the last raw message
    traceback: str = ""   # the last record's continuation lines, joined

    @property
    def short_logger(self) -> str:
        return self.logger[7:] if self.logger.startswith("jarvis.") else self.logger


# ------------------------------------------------------------- parsing
def parse_records(lines: Iterable[str]) -> list[Record]:
    """Log lines -> records; a line with no timestamp prefix (a traceback,
    a multi-line message) belongs to the record before it."""
    out: list[Record] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        m = LINE_RX.match(line)
        if m:
            out.append(Record(m.group("time"), m.group("name"), m.group("level"),
                              m.group("msg")))
        elif out:
            out[-1].extra.append(line)
    return out


def normalise(msg: str) -> str:
    """Fold what varies between repeats of one message: paths, hex ids,
    numbers; collapse whitespace; cap the length."""
    text = _PATH_RX.sub("<path>", str(msg or ""))
    text = _HEX_RX.sub("<id>", text)
    text = _NUM_RX.sub("N", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:160]


def is_noise(rec: Record, noise=NOISE) -> bool:
    return any(rec.name == name and needle in rec.msg for name, needle in noise)


def cluster_warnings(lines: Iterable[str], noise=NOISE, limit: int = TAIL_LINES,
                     levels=LEVELS) -> list[Cluster]:
    """The WARNING/ERROR records in the last `limit` lines, grouped by
    (logger, normalised message), most frequent first (then most recent)."""
    lines = list(lines)
    if limit:
        lines = lines[-limit:]
    groups: dict[tuple, Cluster] = {}
    order: list[tuple] = []
    for rec in parse_records(lines):
        if rec.level not in levels or is_noise(rec, noise):
            continue
        key = (rec.name, normalise(rec.msg))
        hit = groups.get(key)
        tb = "\n".join(rec.extra)
        if hit is None:
            groups[key] = Cluster(rec.name, rec.level, key[1], 1, rec.time, rec.msg, tb)
            order.append(key)
        else:
            hit.count += 1
            hit.last_time = rec.time
            hit.example = rec.msg
            if tb:
                hit.traceback = tb
            if rec.level == "ERROR" or (rec.level == "CRITICAL"):
                hit.level = rec.level
    # count desc; on a tie an ERROR outranks a WARNING; then the most
    # recently started cluster (its position in the file)
    pos = {k: i for i, k in enumerate(order)}
    rank = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}
    return sorted(groups.values(),
                  key=lambda c: (-c.count, rank.get(c.level, 3),
                                 -pos[(c.logger, c.message)]))


# ------------------------------------------------------------- ledger
def read_tail(path, n: int = TAIL_LINES) -> list[str]:
    """The last `n` lines of a text file ([] when missing)."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def read_turns(path, n: int = 200) -> list[dict]:
    out = []
    for line in read_tail(path, n):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def last_turn(path) -> Optional[dict]:
    """The last turn that was a real turn: "abort" is a silent follow-up
    window closing and "superseded" a turn the next wake replaced.  The
    ledger's last record may be hours old, so "that" means this one, not
    the last minute."""
    for rec in reversed(read_turns(path)):
        if rec.get("outcome") not in ("abort", "superseded"):
            return rec
    return None


def turn_outliers(turns: Iterable[dict], wait_over: float = SLOW_WAIT_S) -> dict:
    """Counts over the ledger: turns, ones that waited longer than
    `wait_over`, and ones that never got an answer (uncertain / rejected)."""
    n = slow = lost = 0
    for rec in turns:
        outcome = rec.get("outcome")
        if outcome in ("abort", "superseded"):
            continue
        n += 1
        wait = rec.get("wait")
        if isinstance(wait, (int, float)) and wait > wait_over:
            slow += 1
        if outcome != "audio":
            lost += 1
    return {"turns": n, "slow": slow, "lost": lost, "wait_over": wait_over}


# ------------------------------------------------------------- wording
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine"}


def _word(count: int) -> str:
    return str(_WORDS.get(count, count))


def _n(count: int, noun: str) -> str:
    return f"{_word(count)} {noun}{'' if count == 1 else 's'}"


def _times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def _secs(v) -> str:
    return f"{float(v):.1f} seconds"


def _spoken_msg(text: str, n: int = 60) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.split(" -- ")[0].split(": Traceback")[0]
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def triage_text(clusters: list[Cluster], outliers: Optional[dict] = None,
                examined: int = 0) -> tuple[str, str]:
    """(spoken, card).  Spoken is two sentences at most: the top clusters,
    then the ledger's outliers.  The card carries every cluster and the
    tracebacks."""
    if not clusters:
        spoken = "Nothing wrong in the log, sir"
        spoken += f"; the last {examined} lines are clean." if examined else "."
    else:
        head = [f"'{_spoken_msg(c.example)}' from {c.short_logger} {_times(c.count)}"
                for c in clusters[:2]]
        more = len(clusters) - 2
        if more > 0:
            head.append(f"{_word(more)} more on the card")
        spoken = (f"{_n(len(clusters), 'thing').capitalize()} in the log, sir: "
                  + ", ".join(head[:-1]) + (" and " if len(head) > 1 else "") + head[-1]
                  + f", last at {clusters[0].last_time[:8]}.")
    if outliers and outliers.get("turns"):
        bits = []
        if outliers.get("slow"):
            bits.append(f"{_n(outliers['slow'], 'turn')} waited over "
                        f"{outliers['wait_over']:.0f} seconds")
        if outliers.get("lost"):
            bits.append(f"{_n(outliers['lost'], 'turn')} never got an answer")
        if bits:
            spoken += " On the ledger, " + " and ".join(bits) + "."
        elif clusters:
            spoken += " The ledger's turns all answered promptly."
    card_lines = [f"Log triage — last {examined} lines" if examined else "Log triage"]
    for c in clusters:
        card_lines.append(f"{c.count}× {c.short_logger} {c.level}: {c.example}  (last {c.last_time})")
        if c.traceback:
            card_lines.extend("    " + ln for ln in c.traceback.splitlines()[-12:])
    if outliers and outliers.get("turns"):
        card_lines.append(f"ledger: {outliers['turns']} turns, {outliers['slow']} over "
                          f"{outliers['wait_over']:.0f} s, {outliers['lost']} without an answer")
    return spoken, "\n".join(card_lines)


def slow_text(rec: Optional[dict]) -> str:
    """"Why was that slow?" from the last ledger record: the wait the user
    felt, split into the silence before the recorder stopped, transcription
    and working out the answer (model and tools together), naming the slow
    part."""
    if not rec:
        return "I have no turn on the ledger yet, sir."
    outcome = rec.get("outcome")
    if outcome != "audio":
        why = {"uncertain": "I wasn't sure it was for me",
               "rejected": "the voice check refused it",
               "ignored": "I took it for background chat",
               "no_speech": "I heard no speech"}.get(str(outcome), f"it ended as {outcome}")
        speech = rec.get("speech")
        tail = f" after {_secs(speech)} of speech" if isinstance(speech, (int, float)) else ""
        return f"The last turn never got an answer, sir: {why}{tail}."
    wait = rec.get("wait")
    stages = [("dead_air", "silence before the recorder stopped"),
              ("stt", "transcribing"),
              ("route", "working out the answer")]
    parts = [(label, float(rec[k])) for k, label in stages
             if isinstance(rec.get(k), (int, float))]
    if wait is None or not parts:
        return "The ledger has no timing for the last turn, sir."
    line = f"The last turn waited {_secs(wait)}, sir: " + ", ".join(
        f"{_secs(v)} {label}" for label, v in parts)
    slowest = max(parts, key=lambda p: p[1])
    culprit = {"silence before the recorder stopped": "the silence was the slow part",
               "transcribing": "transcription was the slow part",
               "working out the answer": "the answer was the slow part"}[slowest[0]]
    line += f"; {culprit}."
    if rec.get("stop") == "energy" and slowest[0].startswith("silence"):
        line += " The energy timer stopped it, not the voice detector."
    if isinstance(rec.get("filler"), (int, float)):
        line += f" A filler line went out at {_secs(rec['filler'])}."
    return line
