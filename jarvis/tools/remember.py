"""``remember``: the model-callable way to file a fact he stated.

HIS RULING, 2026-09-04 18:15: "yes give him a remember tool, read it back".

Until now only the spoken rung ("remember that ...", commander._h_remember)
could store a fact; the model had no tool that does, so "put it in your
memory that I graduate December 10th 2026" went to the model, which said
"I have noted that, sir" and stored NOTHING (measured 09-04: 'when do I
graduate?' then called ask_docs twice and found nothing). The claim guard
caught the sentence after the fact; this is the tool the claim can be
backed by (brain.CLAIM_BACKERS["memory"]).

ONE DOOR. The store is ``memory.store_fact_from_speech`` -- the rung's own
helper: cleaned, turned to the second person, keyed on its first six words
so the same fact said twice is one fact. What the rung refuses, this
refuses (a bare head, a question, a pointer like "for later"); what is
another tool's job it refuses BY NAME, so the model is told where to go
rather than left to invent a store: a reminder is set_reminder's, a list
item is the notes tool's, a timer is set_timer's.

THE READ-BACK IS AUTHORED. ``speak`` carries "Noted, sir: <the fact as
stored>." verbatim, so the turn ends on the words that were filed and the
model cannot paraphrase them into something that was not.
"""
from __future__ import annotations

import re

from jarvis.logs import get_logger
from jarvis.memory import (clean_spoken_fact, is_spoken_pointer, is_spoken_question,
                           parse_person_statement, store_fact_from_speech)
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.remember")

NOTED_LINE = "Noted, sir: {fact}."
NO_MEMORY_TEXT = "memory unavailable: nothing was stored"

# Shapes that are somebody else's job. Matched on the cleaned fact; each
# names the tool the model should have called, in the result text it reads.
_OTHERS = (
    (re.compile(r"^(?:please\s+)?remind me\b|^(?:set|make|create)\s+(?:a\s+)?reminder\b", re.I),
     "not a fact: that is a reminder -- use set_reminder"),
    (re.compile(r"^(?:add|put)\b.*\b(?:to|on)\s+(?:my|the)\s+\S*\s*list\b|"
                r"^(?:add|put)\b.*\blist\b", re.I),
     "not a fact: that is a list item -- use the notes tool"),
    (re.compile(r"^(?:set|start)\s+(?:a\s+)?(?:timer|alarm)\b|^wake me\b", re.I),
     "not a fact: that is a timer or an alarm -- use set_timer or set_alarm"),
    (re.compile(r"^(?:make|take|write)\s+(?:a\s+)?note\b", re.I),
     "not a fact: that is a note -- use the notes tool"),
)


def make_tools(cfg, services) -> list[ToolSpec]:
    def remember(fact="", **_) -> ToolResult:
        memory = getattr(services, "memory", None) if services is not None else None
        if memory is None or not hasattr(memory, "remember"):
            return ToolResult(text=NO_MEMORY_TEXT, ok=False)
        raw = str(fact or "").strip()
        if is_spoken_question(raw):
            return ToolResult(text="not a fact: that is a question; answer it "
                                   "or ask him what to remember", ok=False)
        cleaned = clean_spoken_fact(raw)
        for rx, text in _OTHERS:
            if rx.search(cleaned):
                return ToolResult(text=text, ok=False)
        if is_spoken_pointer(cleaned) or len(cleaned.split()) < 2:
            return ToolResult(text="not a fact: nothing to file; ask him what "
                                   "to remember, in his words", ok=False)
        try:
            stored = store_fact_from_speech(memory, cleaned)
        except Exception:                            # noqa: BLE001 - a store may not crash a turn
            log.exception("remember: store failed")
            return ToolResult(text="memory write failed: nothing was stored", ok=False)
        # "my advisor is Dr X <mail>" is a fact AND a contact, as on the rung.
        person = parse_person_statement(cleaned)
        if person is not None and hasattr(memory, "add_person"):
            try:
                memory.add_person(person["alias"], person["name"], email=person["email"])
            except Exception:                        # noqa: BLE001 - the fact is filed either way
                log.debug("remember: people book write failed", exc_info=True)
        log.info("remember: filed %r", stored[:60])
        line = NOTED_LINE.format(fact=stored)
        return ToolResult(text=f"stored: {stored}", speak=line, max_sentences=1)

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
    )
    return [spec]
