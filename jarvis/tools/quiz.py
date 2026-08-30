"""Flashcards over the documents index: "quiz me on chapter three".

``FlashcardStore`` keeps the cards in SQLite under
``PATHS.MEMORY_DIR/flashcards.db`` (NotesStore style; tests pass a tmp
path) with a Leitner box per card: a correct answer moves the card up one
box and pushes its due date out (``BOX_DAYS``), a miss drops it back to
box one and makes it due again straight away, so what he got wrong comes
back tomorrow and what he knows fades to a fortnight.

``QuizSession`` is the state the commander parks in ``_pending_quiz``:
the cards to ask, the one on the table, the tally. It is deliberately
pure -- it never speaks and never touches the model -- so the commander's
rung can be tested without Ollama. ``grade_by_string`` is the first
grader: a normalised token match that settles most short answers (a
number, a name, a phrase) with no model call at all; only the unclear
ones go to ``brain.grade_answer``.

Question generation is ``brain.make_quiz`` on ``docs.topic_chunks`` (see
those modules); this file owns nothing that talks to the network.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("tools.quiz")

DEFAULT_QUESTIONS = 5
DEFAULT_CHUNKS = 6
BOXES = 5
BOX_DAYS = {1: 0, 2: 1, 3: 3, 4: 7, 5: 14}      # days until a card is due again
DAY_S = 86_400.0
ANSWER_WINDOW_S = 300.0        # a question older than this is not what he is answering

NO_DOCS_LINE = "I have no documents to quiz you from yet, sir."
NO_TOPIC_LINE = "I couldn't find anything about {topic} in your documents, sir."
NO_QUESTIONS_LINE = "I couldn't come up with questions on {topic}, sir."
INDEX_DOWN_LINE = "My document index isn't answering, sir."
NOTHING_DUE_LINE = "Nothing is due for review, sir; your cards are all in hand."
NO_CARDS_LINE = "You have no flashcards yet, sir; ask me to quiz you on something first."
PREPARING_LINE = "Let me put some questions together on {topic}, sir."
QUIZ_DONE_LINE = "That's the lot, sir: {right} of {total}."
QUIZ_STOPPED_LINE = "Very good, sir; {right} of {asked} so far."
QUIZ_STOPPED_EARLY_LINE = "Very good, sir."
CORRECT_LINES = ("Quite right, sir.", "Correct, sir.", "Just so, sir.")
WRONG_LINE = "Not quite, sir; the answer I have is {answer}."
SKIP_LINE = "The answer was {answer}, sir."
UNSURE_LINE = "I couldn't judge that one, sir; the answer I have is {answer}."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT DEFAULT '',
    topic TEXT DEFAULT '',
    box INTEGER DEFAULT 1,
    due REAL NOT NULL,
    created REAL NOT NULL,
    seen INTEGER DEFAULT 0,
    correct INTEGER DEFAULT 0,
    last_seen REAL
);
CREATE INDEX IF NOT EXISTS cards_due ON cards(due);
"""

_STOP_TOKENS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "is", "are", "was", "were",
    "it", "its", "it's", "and", "or", "for", "by", "with", "that", "this",
    "i", "think", "believe", "guess", "um", "uh", "so", "well", "maybe",
    "probably", "about", "around", "roughly", "answer", "be", "would"})
_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                 "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                 "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
                 "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
                 "hundred": "100", "thousand": "1000"}


def default_db_path() -> Path:
    return PATHS.MEMORY_DIR / "flashcards.db"


# ------------------------------------------------------------- grading
def normalise(text: str) -> list[str]:
    """Lower-case content tokens: number words as digits, plurals and
    possessives trimmed, filler and articles dropped."""
    # apostrophes out first: "it's" must be one stop token, not "it" + "s"
    text = str(text or "").lower().replace("%", " percent ").replace("'", "").replace("’", "")
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)
    out = []
    for t in toks:
        t = _NUMBER_WORDS.get(t, t)
        if t in _STOP_TOKENS:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def grade_by_string(expected: str, given: str) -> Optional[bool]:
    """True/False when the strings settle it, None when a model should
    look. Correct: every content token of the expected answer appears in
    his (a longer answer that contains the right one is right). Wrong:
    nothing overlaps at all, or the numbers disagree. In between: None."""
    exp, got = normalise(expected), normalise(given)
    if not exp or not got:
        return None
    exp_nums = {t for t in exp if re.fullmatch(r"[0-9.]+", t)}
    got_nums = {t for t in got if re.fullmatch(r"[0-9.]+", t)}
    if exp_nums and got_nums and not (exp_nums & got_nums):
        return False                           # "40 percent" vs "20 percent"
    if all(t in got for t in exp):
        return True
    if exp_nums and exp_nums <= got_nums and len(exp) <= 2:
        return True                            # a bare number answered with the number
    # A shared stem ("mitochondria" / "mitochondrion", "acceleration" /
    # "accelerate") is not a miss, it is a question for the model.
    overlap = sum(1 for t in exp if t in got or
                  (len(t) >= 5 and any(g[:5] == t[:5] for g in got)))
    if overlap == 0:
        return False
    return None


# --------------------------------------------------------------- store
class FlashcardStore:
    """SQLite-backed Leitner cards. Thread-safe (one lock, one connection
    with check_same_thread=False, like NotesStore)."""

    def __init__(self, db_path=None):
        self.db_path = Path(db_path or default_db_path()).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def close(self):
        with self._lock:
            try:
                self._db.close()
            except Exception:
                log.debug("flashcards db close failed", exc_info=True)

    def add_cards(self, pairs: list[dict], source: str = "", topic: str = "",
                  now: Optional[float] = None) -> list[dict]:
        """Store new cards (a question already on file for the same source
        is reused, not duplicated). Returns the cards, stored order."""
        now = time.time() if now is None else float(now)
        out = []
        with self._lock:
            for pair in pairs:
                q = " ".join(str(pair.get("question") or "").split())
                a = " ".join(str(pair.get("answer") or "").split())
                if not q or not a:
                    continue
                row = self._db.execute(
                    "SELECT * FROM cards WHERE lower(question) = ? AND source = ?",
                    (q.lower(), source or "")).fetchone()
                if row is None:
                    cur = self._db.execute(
                        "INSERT INTO cards(question, answer, source, topic, box, due, created)"
                        " VALUES (?,?,?,?,1,?,?)", (q, a, source or "", topic or "", now, now))
                    row = self._db.execute("SELECT * FROM cards WHERE id = ?",
                                           (cur.lastrowid,)).fetchone()
                out.append(dict(row))
            self._db.commit()
        return out

    def due(self, limit: int = DEFAULT_QUESTIONS, now: Optional[float] = None,
            topic: str = "") -> list[dict]:
        """Cards due now, lowest box (the ones he keeps missing) first."""
        now = time.time() if now is None else float(now)
        like = f"%{topic.lower()}%" if topic else None
        with self._lock:
            if like:
                rows = self._db.execute(
                    "SELECT * FROM cards WHERE due <= ? AND (lower(topic) LIKE ? OR "
                    "lower(source) LIKE ?) ORDER BY box, due LIMIT ?",
                    (now, like, like, max(1, int(limit)))).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM cards WHERE due <= ? ORDER BY box, due LIMIT ?",
                    (now, max(1, int(limit)))).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM cards").fetchone()[0])

    def get(self, card_id: int) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT * FROM cards WHERE id = ?",
                                   (int(card_id),)).fetchone()
        return dict(row) if row else None

    def record(self, card_id: int, correct: bool, now: Optional[float] = None) -> Optional[dict]:
        """Leitner move: right -> one box up (capped), due in BOX_DAYS;
        wrong -> box one, due now (it comes back this session or tomorrow)."""
        now = time.time() if now is None else float(now)
        with self._lock:
            row = self._db.execute("SELECT * FROM cards WHERE id = ?",
                                   (int(card_id),)).fetchone()
            if row is None:
                return None
            box = min(BOXES, int(row["box"]) + 1) if correct else 1
            due = now + BOX_DAYS[box] * DAY_S if correct else now
            self._db.execute(
                "UPDATE cards SET box = ?, due = ?, seen = seen + 1, correct = correct + ?, "
                "last_seen = ? WHERE id = ?",
                (box, due, 1 if correct else 0, now, int(card_id)))
            self._db.commit()
            row = self._db.execute("SELECT * FROM cards WHERE id = ?",
                                   (int(card_id),)).fetchone()
        return dict(row)

    def stats(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else float(now)
        with self._lock:
            total = int(self._db.execute("SELECT COUNT(*) FROM cards").fetchone()[0])
            due = int(self._db.execute("SELECT COUNT(*) FROM cards WHERE due <= ?",
                                       (now,)).fetchone()[0])
            boxes = {int(r[0]): int(r[1]) for r in self._db.execute(
                "SELECT box, COUNT(*) FROM cards GROUP BY box")}
        return {"total": total, "due": due, "boxes": boxes}


# ------------------------------------------------------------- session
@dataclass
class QuizSession:
    """The quiz on the table. ``cards`` are dicts with at least id,
    question, answer; ``asked_at`` stamps the open question."""
    cards: list[dict]
    topic: str = ""
    index: int = 0
    right: int = 0
    asked: int = 0
    asked_at: float = 0.0
    results: list[tuple] = field(default_factory=list)

    @property
    def current(self) -> Optional[dict]:
        return self.cards[self.index] if 0 <= self.index < len(self.cards) else None

    @property
    def finished(self) -> bool:
        return self.index >= len(self.cards)

    @property
    def total(self) -> int:
        return len(self.cards)

    def ask(self, now: Optional[float] = None) -> str:
        """The spoken line for the current question."""
        card = self.current
        if card is None:
            return ""
        self.asked_at = time.time() if now is None else float(now)
        return f"Question {self.index + 1}: {card['question']}"

    def stale(self, now: Optional[float] = None, window: float = ANSWER_WINDOW_S) -> bool:
        now = time.time() if now is None else float(now)
        return self.asked_at > 0 and now - self.asked_at > window

    def settle(self, correct: Optional[bool]) -> dict:
        """Record the outcome of the open question and move on. ``None``
        means ungraded (skipped or the grader failed): no tally change."""
        card = self.current
        self.asked += 1
        if correct:
            self.right += 1
        self.results.append((card["id"] if card else None, correct))
        self.index += 1
        return card or {}

    def score_line(self) -> str:
        if self.finished:
            return QUIZ_DONE_LINE.format(right=self.right, total=self.total)
        if self.asked == 0:
            return QUIZ_STOPPED_EARLY_LINE
        return QUIZ_STOPPED_LINE.format(right=self.right, asked=self.asked)


def study_text(chunks: list[dict], cap: int = 6_000) -> str:
    """The chunks joined for the generator, file names as headings so a
    question can be traced back, capped so the prompt stays in NUM_CTX."""
    parts, used = [], 0
    for h in chunks:
        text = " ".join(str(h.get("text") or "").split())
        if not text:
            continue
        piece = f"[{h.get('name', 'document')}] {text}"
        if used + len(piece) > cap and parts:
            break
        parts.append(piece[:cap])
        used += len(piece)
    return "\n\n".join(parts)
