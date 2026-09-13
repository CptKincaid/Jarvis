"""``remember``: the model-callable way to file a fact he stated.

HIS RULING, 2026-09-04 18:15: "yes give him a remember tool, read it back".

Until now only the spoken rung ("remember that ...", commander._h_remember)
could store a fact; the model had no tool that does, so "put it in your
memory that I graduate December 10th 2026" went to the model, which said
"I have noted that, sir" and stored NOTHING (measured 09-04: 'when do I
graduate?' then called ask_docs twice and found nothing). The claim guard
caught the sentence after the fact; this is the tool the claim can be
backed by (brain.CLAIM_BACKERS["memory"]) -- and only a run that STORED
backs it: brain.ran_names lists the tools that succeeded.

HIS WORDS AND NOTHING ELSE. The registry hands every model call his
utterance (ToolSpec.derive) for exactly this: the deriver checks the
model's ``fact`` against what he said -- case, punctuation and the common
contractions forgiven, the rung's own head ("remember that ...") allowed
in front -- and a fact that is not in his words is refused, so an invented
fact, a paraphrase ("The user graduates on 10 December 2026") or a
third-person rewrite is never filed and never read back as his. The
2026-09-12 attack round measured all three walking through before this.
``verbatim`` is reserved: the model cannot vouch for itself. A forced call
(the commander's) carries no utterance and no model to doubt.

ONE DOOR, ONE GRAMMAR. The store is ``memory.store_fact_from_speech`` --
the rung's own helper: cleaned, turned to the second person, keyed on its
first six words so the same fact said twice, through either door, is one
fact. The rung's grammar is reused here (commander._MEM_FACT_RX and its
neighbours): a head is stripped so it is not filed as part of the fact; a
bare head, a to-do ("remember TO ..."), a recall ("what my dentist's name
is"), a question, a pointer ("for later") and a bare imperative are
refused; what is another tool's job is refused BY NAME (a reminder is
set_reminder's, a list item is the notes tool's, a timer is set_timer's),
so the model is told where to go rather than left to invent a store.

THE READ-BACK IS AUTHORED. ``speak`` carries "Noted, sir: <the fact as
stored>." verbatim, so the turn ends on the words that were filed and the
model cannot paraphrase them into something that was not. He ruled the
read-back, so a secret he asks to remember IS read back; it is not logged
-- the log line carries the key's first words only. "Scratch that" after
the read-back finds the undo parked on ``services.remember_undo``.
"""
from __future__ import annotations

import re
import time

from jarvis.logs import get_logger
from jarvis.memory import (clean_spoken_fact, fact_key, is_spoken_pointer,
                           is_spoken_question, parse_person_statement,
                           store_fact_from_speech)
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.remember")

NOTED_LINE = "Noted, sir: {fact}."
NO_MEMORY_TEXT = "memory unavailable: nothing was stored"
NOT_HIS_WORDS_TEXT = ("not his words: nothing was stored. Pass the fact exactly "
                      "as he said it, or say plainly that nothing was stored")
MAX_FACT_CHARS = 300        # a spoken fact; a paragraph is a document, not a fact

# A courtesy opener the model may have carried along ("please set an alarm").
_COURTESY_RX = re.compile(
    r"^(?:(?:please|can you|could you|would you|will you|i want you to|"
    r"i need you to|i[’']d like you to|i would like you to),?\s+)+", re.I)
# Shapes that are somebody else's job, matched on the cleaned fact; each
# names the tool the model should have called, in the text it reads.
_OTHERS = (
    (re.compile(r"^remind me\b|^(?:set|make|create)\s+(?:a\s+|an\s+|the\s+)?reminder\b", re.I),
     "not a fact: that is a reminder -- use set_reminder"),
    (re.compile(r"^(?:add|put)\b.*\blist\b", re.I),
     "not a fact: that is a list item -- use the notes tool"),
    (re.compile(r"^(?:set|start)\s+(?:a\s+|an\s+|the\s+)?(?:\S+\s+)?(?:timer|alarm)\b|^wake me\b", re.I),
     "not a fact: that is a timer or an alarm -- use set_timer or set_alarm"),
    (re.compile(r"^(?:make|take|write)\s+(?:a\s+)?note\b", re.I),
     "not a fact: that is a note -- use the notes tool"),
)
# The rung's own lookahead words, by refusal: a to-do, or a recall.
_TODO_WORDS = frozenset({"to"})
_RECALL_WORDS = frozenset({"when", "what", "how", "where", "who", "why", "about",
                           "whether", "if", "down"})
# A bare imperative is a command, not a fact about him.
_IMPERATIVE_RX = re.compile(
    r"^(?:call|phone|ring|text|email|message|buy|get|pick up|pick|fetch|pay|"
    r"turn|switch|play|pause|stop|start|open|close|send|tell|ask|book|order|"
    r"check|cancel|wake|set|schedule|find|search|look up|show|read|dim|"
    r"brighten|lock|unlock|restart|shut)\b", re.I)

_CONTRACTIONS = (
    (re.compile(r"\bi[’']m\b", re.I), "i am"), (re.compile(r"\bi[’']ve\b", re.I), "i have"),
    (re.compile(r"\bi[’']ll\b", re.I), "i will"), (re.compile(r"\bi[’']d\b", re.I), "i would"),
    (re.compile(r"\bdon[’']t\b", re.I), "do not"), (re.compile(r"\bdoesn[’']t\b", re.I), "does not"),
    (re.compile(r"\bcan[’']t\b", re.I), "cannot"), (re.compile(r"\bwon[’']t\b", re.I), "will not"),
    (re.compile(r"\bisn[’']t\b", re.I), "is not"), (re.compile(r"\baren[’']t\b", re.I), "are not"),
    (re.compile(r"\bit[’']s\b", re.I), "it is"), (re.compile(r"\bthat[’']s\b", re.I), "that is"),
    (re.compile(r"\bthere[’']s\b", re.I), "there is"), (re.compile(r"\bwhat[’']s\b", re.I), "what is"),
    (re.compile(r"\b(\w+)[’']s\b", re.I), r"\1 s"),      # possessives: "patel's" -> "patel s"
)
_NON_WORD_RX = re.compile(r"[^a-z0-9 ]+")


def _norm(text) -> str:
    """Case, punctuation, whitespace and the common contractions forgiven:
    what two takes of the same sentence have in common."""
    out = str(text or "").lower()
    for rx, rep in _CONTRACTIONS:
        out = rx.sub(rep, out)
    out = _NON_WORD_RX.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


_HEAD_ONLY_RX = None


def _rung_grammar():
    # The commander is the grammar's home (it is the rung's); imported at
    # call time so this module never drags 14k lines in at boot.
    global _HEAD_ONLY_RX
    from jarvis.commander import _MEM_ASK_RX, _MEM_COURTESY, _MEM_FACT_RX, _MEM_HEADS
    if _HEAD_ONLY_RX is None:
        # The head alone, so a head followed by what the rung refuses
        # ("remember TO call mom") still loses its head here and the
        # refusal below sees "to call mom", not a fact beginning "remember".
        _HEAD_ONLY_RX = re.compile(
            r"^" + _MEM_COURTESY + r"(?:" + _MEM_HEADS + r")\b"
            r"(?:\s*[,:]\s*|\s+)(?:(?:that|this)(?:\s*[,:]\s*|\s+))?", re.I)
    return _MEM_ASK_RX, _MEM_FACT_RX, _HEAD_ONLY_RX


def strip_head(fact: str) -> tuple:
    """(fact without the rung's head, bare_head): "remember that I
    graduate ..." -> ("I graduate ...", False); "remember that" -> ("",
    True). A fact with no head passes through."""
    ask_rx, fact_rx, head_rx = _rung_grammar()
    text = str(fact or "").strip()
    if ask_rx.match(text):
        return "", True
    m = fact_rx.match(text)
    if m:
        return m.group("fact").strip(), False
    return head_rx.sub("", text, count=1).strip(), False


def his_words(fact: str, utterance: str) -> bool:
    """Is the cleaned fact in what he said? A head on either side is
    allowed; the comparison is on the normalised text."""
    said = _norm(utterance)
    if not said:
        return False
    want, _bare = strip_head(fact)
    return bool(want) and _norm(want) in said


def make_tools(cfg, services) -> list[ToolSpec]:
    def derive(utterance: str, args: dict) -> dict:
        # A model call: judge the model's fact against his words. No
        # utterance (a scripted or forced caller) is no judgement.
        if not str(utterance or "").strip():
            return {}
        return {"verbatim": his_words(str(args.get("fact") or ""), utterance)}

    def remember(fact="", verbatim=True, **_) -> ToolResult:
        memory = getattr(services, "memory", None) if services is not None else None
        if memory is None or not hasattr(memory, "remember"):
            return ToolResult(text=NO_MEMORY_TEXT, ok=False)
        if not isinstance(fact, str):
            return ToolResult(text="not a fact: the fact is his sentence, as text", ok=False)
        raw = fact.strip()
        if len(raw) > MAX_FACT_CHARS:
            return ToolResult(text="not a fact: too long to be one thing he said; "
                                   "store nothing", ok=False)
        if not verbatim:
            return ToolResult(text=NOT_HIS_WORDS_TEXT, ok=False)
        if is_spoken_question(raw):
            return ToolResult(text="not a fact: that is a question; answer it "
                                   "or ask him what to remember", ok=False)
        body, bare_head = strip_head(_COURTESY_RX.sub("", raw, count=1))
        if bare_head:
            return ToolResult(text="not a fact: nothing to file; ask him what "
                                   "to remember, in his words", ok=False)
        cleaned = clean_spoken_fact(body)
        first = cleaned.split()[0].lower() if cleaned.split() else ""
        if first in _TODO_WORDS:
            return ToolResult(text="not a fact: that is a to-do -- use the notes "
                                   "tool or set_reminder", ok=False)
        if first in _RECALL_WORDS:
            return ToolResult(text="not a fact: that is a question about what is "
                                   "stored; answer it from memory", ok=False)
        for rx, text in _OTHERS:
            if rx.search(cleaned):
                return ToolResult(text=text, ok=False)
        if _IMPERATIVE_RX.match(cleaned):
            return ToolResult(text="not a fact: that is a command, not something "
                                   "about him; use the tool that does it", ok=False)
        if is_spoken_pointer(cleaned) or len(cleaned.split()) < 2:
            return ToolResult(text="not a fact: nothing to file; ask him what "
                                   "to remember, in his words", ok=False)
        try:
            stored = store_fact_from_speech(memory, cleaned)
        except Exception:                            # noqa: BLE001 - a store may not crash a turn
            log.exception("remember: store failed")
            return ToolResult(text="memory write failed: nothing was stored", ok=False)
        # "my advisor is Dr X, email …" is a fact AND a contact, as on the
        # rung. The address is his words too (the verbatim check above).
        person = parse_person_statement(cleaned)
        if person is not None and hasattr(memory, "add_person"):
            try:
                memory.add_person(person["alias"], person["name"], email=person["email"])
            except Exception:                        # noqa: BLE001 - the fact is filed either way
                log.debug("remember: people book write failed", exc_info=True)
        key = fact_key(stored)
        if services is not None and hasattr(memory, "forget"):
            def undo() -> str:
                memory.forget(key)
                return "Forgotten, sir."
            try:
                services.remember_undo = {"undo": undo, "at": time.monotonic()}
            except Exception:                        # noqa: BLE001 - no slot, no undo
                log.debug("remember: could not park the undo", exc_info=True)
        # The KEY only: he ruled the read-back, not a log of his facts.
        log.info("remember: filed a fact under %r", " ".join(key.split()[:3]))
        return ToolResult(text=f"stored: {stored}", speak=NOTED_LINE.format(fact=stored),
                          max_sentences=1)

    spec = ToolSpec(
        name="remember",
        # <= 20 words: the descriptions ride in every prompt.
        description=("Store a fact he just stated about himself, in his own "
                     "words, when he asks you to remember it."),
        parameters={"type": "object",
                    "properties": {"fact": {"type": "string",
                                            "description": "the fact, exactly as he said it"}},
                    "required": ["fact"]},
        handler=remember,
        reserved=frozenset({"verbatim"}),
        derive=derive,
        derive_takes_args=True,
    )
    return [spec]
