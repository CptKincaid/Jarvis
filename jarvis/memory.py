"""Persistent Memory — facts, habits, preferences, sessions, notes.

THE single memory store for Jarvis V3. Survives across sessions. All data
lives in ``PATHS.MEMORY_DIR`` (~/.aiws_trainer/jarvis_memory/).

Absorbs the legacy jarvis_agent.py stores: on first run, anything in
``PATHS.LEGACY_AGENT_DIR`` (~/.aiws_trainer/jarvis_data/) is migrated in —
habit entries are merged (so habit counts merge), voice notes are copied —
and the old dir is renamed ``jarvis_data.migrated``.

``log_habit`` is the ONE habit sink (the V1 double-log — agent.log_command +
memory.log_habit — is gone; all callers route here exactly once per
utterance).

Semantic facts (2026-08-30): facts.json stays the source of truth, and every
fact is also written through to a chromadb collection at
``MEMORY_DIR/facts_index`` embedded with Ollama's ``nomic-embed-text`` (the
same local embedder as the documents tool). ``recall()`` then finds "who's
my dentist?" from "remember that my dentist is Dr Patel" -- a paraphrase the
old substring test never matched. With chromadb or Ollama unavailable both
fall back to the substring store; nothing here ever raises into a turn.

``format_for_context(text)`` does NOT rank on the reply path while the whole
store fits the prompt -- it renders all of it and never wakes the embedder.
See "The one-slot rule" below: on this box a second model costs a swap of
the first, so retrieval is only worth running once the store outgrows the
window it is being retrieved into.

People book: ``people.json`` maps how Hunter refers to someone ("my
advisor", "mom") to a name, an address and a relation. It is rendered as a
People block on every turn, outside the relevance-ranked fact window, so a
contact never drops out of context; mail and calendar resolve the aliases
through ``resolve_person`` / ``expand_aliases``.

Usage:
    mem = JarvisMemory()
    mem.remember("project", "training uses batch size 16")
    results = mem.recall("training")
    mem.log_habit("check gpu")
    suggestion = mem.suggest_by_habit()
    mem.save_session("worked on the detector")   # app shutdown
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("memory")

# ----------------------------------------------------------------------
# Filing a fact in the second person
# ----------------------------------------------------------------------
# SYMPTOM, 2026-08-31. "Remember that it is Mara and I's birthday" was
# stored, and read back, as
#
#     it is mara and i's birthday on = it is mara and i's birthday on
#
# and he said: "does not context I's as me, probably should just be Mara
# and your anniversary on September 20th".
#
# facts.json is not a quote book. Every fact in it is rendered STRAIGHT into
# the model's prompt by format_for_context, under "Known facts", where the
# model is Jarvis and "I" is therefore Jarvis -- so a fact filed in the
# first person tells him the anniversary is HIS. He got away with it that
# evening only because the model guessed; the fact on disk still says the
# wrong thing, and every later recall re-rolls that guess.
#
# So a fact is turned around ON THE WAY IN, where it is written once, rather
# than on every read. "Mara and I's" is the one that has to be handled
# specially: it is not standard English, and the possessive belongs to the
# pair, so it becomes "Mara and your" -- his own wording for what he wanted.
#
# POSSESSIVES ONLY -- his words were "normalise first-person possessives",
# and the narrower rule is also the safe one. Turning every "I" and "me"
# around as well rewrites sentences that are not about him: the memory
# garden files notes in JARVIS's voice ("he told me this himself"), and
# "me" there is Jarvis. A possessive cannot mean that.
#
# Order matters: the "and I's" form has to go before the bare "I's", or the
# conjunction is left behind.
_SECOND_PERSON_RULES = [
    # "Mara and I's anniversary" -> "Mara and your anniversary"
    (re.compile(r"\b(and|&)\s+I(?:'|’)s\b", re.I), r"\1 your"),
    (re.compile(r"\bI(?:'|’)s\b", re.I), "your"),
    (re.compile(r"\bmyself\b", re.I), "yourself"),
    (re.compile(r"\bmine\b", re.I), "yours"),
    (re.compile(r"\bmy\b", re.I), "your"),
]


def to_second_person(text):
    """A fact as Jarvis should read it back: "my" -> "your", "Mara and I's"
    -> "Mara and your".

    Idempotent -- second-person text has nothing left to rewrite -- so it is
    safe on facts that arrive already turned around (the memory garden's
    own promotions, a debrief line), and safe to run on a QUERY as well:
    the store is written in the second person, so "who is my dentist" has
    to be asked in the second person too or the substring search misses the
    fact it just filed."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for rx, repl in _SECOND_PERSON_RULES:
        out = rx.sub(repl, out)
    return out


# ----------------------------------------------------------------------
# Filing what he SAID: the remember rung and the remember tool (2026-09-04)
# ----------------------------------------------------------------------
# to_second_person above turns POSSESSIVES only, on every write, because a
# garden note in Jarvis's own voice ("he told me this himself") must keep
# its "me". A fact that arrives from the REMEMBER RUNG is different: it is
# a sentence he spoke about himself -- "remember that I graduate December
# 10th 2026" -- and rendered under "Known facts", where the model is
# Jarvis, a bare "I" says that Jarvis graduates (measured 2026-09-04: the
# prompt line read "i graduate december 10th 2026 ...: i graduate ...").
# So at that one door, and only there, the subject pronouns turn as well:
# I -> you, I'm -> you're, me -> you, with the verb that has to agree
# (I am -> you are, I was -> you were). Word-boundary; the casing of the
# rest of the sentence is kept (Whisper's capitals on names and months are
# evidence), and the sentence keeps its opening capital if it had one.
#
# Everything the rung stores goes through store_fact_from_speech, and the
# remember TOOL (the model's door to the same store) is to call the same
# helper: one cleaning, one turning, one key rule.
_SPEECH_PRONOUN_RULES = [
    (re.compile(r"\bi[’']m\b", re.I), "you're"),
    (re.compile(r"\bi[’']ve\b", re.I), "you've"),
    (re.compile(r"\bi[’']ll\b", re.I), "you'll"),
    (re.compile(r"\bi[’']d\b", re.I), "you'd"),
    (re.compile(r"\bi am\b", re.I), "you are"),
    (re.compile(r"\bi was\b", re.I), "you were"),
    (re.compile(r"\bi\b(?![’'])", re.I), "you"),
    (re.compile(r"\bme\b", re.I), "you"),
]


def speech_to_second_person(text):
    """A sentence he said about himself, as Jarvis should read it back:
    "I graduate December 10th" -> "You graduate December 10th", "Heather
    emailed me the form" -> "Heather emailed you the form". Idempotent
    (second-person text has nothing left to turn), and a superset of
    to_second_person, so the store's own write-time pass is a no-op on
    what comes out of here."""
    if not isinstance(text, str) or not text:
        return text
    out = to_second_person(text)
    for rx, repl in _SPEECH_PRONOUN_RULES:
        out = rx.sub(repl, out)
    if text[0].isupper() and out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


# The head the rung already consumed can leave a stray "that," / "this:" /
# ":" at the front, and speech leaves courtesies and punctuation at the
# back ("...December 10th 2026, please."). Measured on 7539478: six of 31
# stored values carried "please", ", please", "!", "that," or "this:".
_FACT_LEAD_RX = re.compile(r"^\s*(?:(?:that|this)\s*[,:]\s*|[,:]\s*)+", re.I)
_FACT_TAIL_RX = re.compile(
    r"(?:[,\s]+(?:please|jarvis|sir|thanks|thank you|would you|will you)"
    r"[.!?,]*)+\s*$", re.I)
_FACT_PUNCT_RX = re.compile(r"[\s.!?,;:]+$")
FACT_KEY_WORDS = 6


def clean_spoken_fact(text) -> str:
    """The fact as said, minus what is not the fact: a leading "that," /
    "this:" / ":", a trailing courtesy ("please", "thanks", "would you",
    "sir", "jarvis") and trailing punctuation."""
    out = (text or "").strip()
    out = _FACT_LEAD_RX.sub("", out, count=1)
    out = _FACT_TAIL_RX.sub("", out, count=1)
    out = _FACT_PUNCT_RX.sub("", out)
    return re.sub(r"\s+", " ", out).strip()


def is_spoken_question(text) -> bool:
    """True when what he said, courtesy tail aside, ends in a question
    mark: "remember i asked?" is not a fact, "remember that I graduate
    December 10th 2026, will you?" is."""
    return _FACT_TAIL_RX.sub("", (text or "").strip(), count=1).rstrip().endswith("?")


# A value that only POINTS at something already said, or is only a
# courtesy, is not a fact. Measured on 1641fb3 (11 of 44 must-not
# sentences): "remember that for later" filed "for later", "remember that,
# thanks a lot" filed "thanks a lot", "remember everything i just said"
# filed "everything you just said", "remember that time we went to
# austin" filed "time we went to austin" -- the two-word floor was the
# only guard after the head. Whole-value match on the CLEANED value, in
# the first person (it runs before the store turns the pronouns), with
# the demonstrative the head left behind allowed in front ("it for
# later"). A fact that merely CONTAINS one of these ("thank you notes go
# out on friday") does not match.
_FACT_POINTER_RX = re.compile(
    r"^(?:(?:that|this|it)\s+)?(?:"
    r"for (?:later|now|next time|the future|future reference|reference|me|us)"
    r"|(?:for )?(?:when|whenever|if) i ask(?: you)?(?: (?:for|about) (?:it|that|this))?"
    r"|next time|as well|too|also|again"
    r"|all (?:of )?(?:that|this|it)|everything"
    r"|everything i (?:just )?(?:said|told you|mentioned)"
    r"|the (?:last |first )?(?:thing|name|number|code|date|address|word|bit) i"
    r" (?:just )?(?:said|told you|gave you|mentioned)"
    r"|i (?:just )?(?:said|told you|mentioned) (?:that|this|it)"
    r"|(?:that |this )?time (?:we|i|you)\b.*"
    r"|(?:thanks|thank you|cheers)(?: (?:a lot|so much|very much|a bunch|kindly|again))?"
    r")$", re.I)


def is_spoken_pointer(text) -> bool:
    """True when the cleaned value only points at something already said
    ("for later", "as well", "everything I just said", "the name I just
    gave you", "I said that", "that time we went to Austin") or is only a
    courtesy ("thanks a lot"): nothing to file, so the rung asks what to
    remember, exactly as it does for a bare head."""
    return _FACT_POINTER_RX.match((text or "").strip()) is not None


def fact_key(value) -> str:
    """The key a spoken fact is filed under: its first six words, CASE-
    FOLDED. A truncated copy of the value, which is why format_for_context
    prints such a fact once, not as 'key: value' (_fact_line lower-cases
    both sides). Folded because Whisper's capitals vary between takes:
    measured on 1641fb3, "I graduate December 10th 2026 ..." re-said as
    "i graduate december 10th 2026 ..." was filed under two keys and
    rendered twice in every prompt. One key, so the later take replaces
    the earlier one."""
    return " ".join(str(value or "").lower().split()[:FACT_KEY_WORDS])


def store_fact_from_speech(memory, text) -> str:
    """File a fact he SAID: clean it, turn it to the second person, key it
    on its first six words, write it through ``memory.remember``. Returns
    the value as stored. ONE door for the rung and the remember tool."""
    value = speech_to_second_person(clean_spoken_fact(text))
    memory.remember(fact_key(value), value)
    return value


def _fact_line(key, entry) -> str:
    """One 'Known facts' line. A fact filed under a name ("dentist",
    "gpu", the garden's keys) reads 'key: value'; a fact whose key is the
    head of its own value (store_fact_from_speech, and the old rung's
    value[:60]) reads as the value alone -- printing both printed the
    sentence twice, cut mid-phrase the first time."""
    value = str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)
    k = str(key or "").strip().lower()
    if k and value.strip().lower().startswith(k):
        return value
    return f"{key}: {value}"


# Recall by STEM when the substring misses: "my graduation" is not a
# substring of "you graduate december 10th 2026", and measured on
# 7539478 every "graduation" phrasing came back "I don't have anything
# stored about that, sir." with the fact sitting right there. Lexical
# evidence is free; the semantic index costs a chat-model swap (the
# one-slot rule above) and is not always there.
_RECALL_STOP = frozenset("""
    the a an my your our his her their its i you me we he she they it
    of to in on at for about with and or is are was were be been am do
    does did what when where who how why say said tell told that this
    these those have has had
""".split())
_STEM_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ies",
                  "ers", "er", "ed", "es", "s", "ates", "ate", "al", "ly")
_STEM_MIN = 4


def _stem(word: str) -> str:
    w = word.lower().strip("'’")
    for suf in _STEM_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= _STEM_MIN:
            return w[:-len(suf)]
    return w


def _content_stems(text) -> list:
    words = re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", (text or "").lower())
    return [_stem(w) for w in words if w not in _RECALL_STOP and len(w) >= 3]


def _stems_overlap(a: str, b: str) -> bool:
    if a == b:
        return True
    return (len(a) >= _STEM_MIN and len(b) >= _STEM_MIN
            and (a.startswith(b) or b.startswith(a)))


def stem_match(query, text) -> bool:
    """True when EVERY content word of ``query`` shares a stem with a word
    of ``text``. No content words (all function words) is no match."""
    q = _content_stems(query)
    if not q:
        return False
    t = _content_stems(text)
    return all(any(_stems_overlap(qs, ts) for ts in t) for qs in q)


FACTS_COLLECTION = "jarvis_memory"
# Measured 2026-08-30 against nomic-embed-text with the task prefixes: a
# matching fact scores 0.60-0.87 ("who's my dentist" vs "my dentist is Dr
# Patel" 0.75, "what batch size" vs the batch-size fact 0.87) while the
# best unrelated pair ("what time is it" vs the thesis fact) reaches 0.58.
# The per-turn injection floor sits above that noise; an explicit "recall
# X" is a request, so it looks a little further down.
SCORE_FLOOR_CONTEXT = 0.60
SCORE_FLOOR_RECALL = 0.50
CONTEXT_TOP_K = 3
RECALL_TOP_K = 5
# ----------------------------------------------------------------------
# The one-slot rule (2026-08-31)
# ----------------------------------------------------------------------
# Ollama here runs with OLLAMA_MAX_LOADED_MODELS=1 -- the guard added on
# 2026-08-28 after concurrent model load exhausted the GB10 unified pool
# and hard-locked the box. ONE model may be resident, so asking for a
# second one does not add a model, it SWAPS the chat model out.
#
# Measured against the live server (scratchpad probe.py, 2026-08-31):
#
#   chat gemma4:26b, resident            0.08 s wall, 0.00 s load_duration
#   /api/embed nomic-embed-text          2.5  s   -- and /api/ps then shows
#                                                    ONLY nomic: gemma is gone
#   chat gemma4:26b, next request        7.1  s wall, 6.9 s load_duration
#
# So one embedding on the reply path costs ~9.4 s of turn latency. The
# live log is unambiguous: 36 replies before 14:33 at 0.00 s of reported
# overhead, then 9 of 9 after it at 6.92-7.41 s, because that is when the
# embedder came back up and every turn began paying the swap. Worse, the
# reload burned the whole tool budget, so five of those nine turns
# answered with the "didn't get to putting it into words" degrade.
#
# THE RULE: the reply path may not ask Ollama for a model other than the
# chat model. Retrieval is not free here; it costs a model swap.
#
# It also buys nothing until the fact store outgrows the prompt. Measured
# on Hunter's live store (4 facts) across five real utterances: the
# semantic query cleared the 0.60 floor for NONE of them, so the model was
# handed zero facts -- at 9.57 s per turn. Below the budget the whole
# store is simply rendered, which is both free and strictly MORE memory
# than the <=3 that would have cleared the floor. Above it, ranking earns
# its keep again and the embedder runs (see relevant_facts).
CONTEXT_ALL_FACTS_MAX = 12
CONTEXT_ALL_FACTS_CHARS = 1200
# The embedder must not squat the single slot. keep_alive was -1 here to
# spare a cold 7.5 s load, which made sense only if two models could be
# resident at once; with one slot the pin does not stop the eviction (the
# probe above shows gemma displacing a pinned nomic anyway) and simply
# leaves the CHAT model out in the cold between turns. 0 = hand the slot
# straight back.
EMBED_KEEP_ALIVE = 0
EMBED_BACKOFF_S = 60.0
# Repeat questions are the common shape of an above-budget query ("what's
# on my calendar" many times a day); a cached query vector costs no swap.
QUERY_CACHE_MAX = 64
MAX_PEOPLE = 200

_TITLE_RX = r"(?:dr|doctor|mr|mrs|ms|miss|prof|professor)\.?"
# "my advisor is Dr Peyrovi, email hp@tamu.edu" / "my mom is Linda" /
# "my TA is Sam Ortiz, his email is sam@tamu.edu"
_PERSON_RX = re.compile(
    r"^(?:my|our) (?P<alias>[A-Za-z][A-Za-z' -]{0,30}?) is "
    # lazy word count: greedy swallowed "Mike Chen and his" as the name
    r"(?P<name>(?:" + _TITLE_RX + r" )?[A-Za-z][A-Za-z'.-]*(?: [A-Za-z][A-Za-z'.-]*){0,3}?)"
    r"(?:[,;]? (?:and )?(?:(?:his|her|their) )?(?:e-?mail(?: address)?(?: is|:)?|at) "
    r"(?P<email>[^\s,;]+@[^\s,;]+))?[.!]?$", re.I)
# Aliases that are plainly people even without a title, an address or a
# capitalised name in the transcript.
_RELATION_WORDS = frozenset({
    "advisor", "adviser", "supervisor", "professor", "prof", "ta", "teacher",
    "tutor", "mentor", "boss", "manager", "coworker", "colleague", "partner",
    "wife", "husband", "girlfriend", "boyfriend", "fiance", "fiancee",
    "mom", "mum", "mother", "dad", "father", "brother", "sister", "son",
    "daughter", "aunt", "uncle", "cousin", "grandma", "grandmother",
    "grandpa", "grandfather", "roommate", "landlord", "neighbor", "neighbour",
    "friend", "best friend", "dentist", "doctor", "physician", "therapist",
    "lawyer", "accountant", "barber", "trainer", "coach", "vet", "plumber",
    "electrician", "mechanic", "contractor", "realtor", "agent", "recruiter",
    "classmate", "labmate", "lab partner", "study partner", "pi",
})
_EMAIL_RX = re.compile(r"[^\s,;<>]+@[^\s,;<>]+")
_SINCE_RX = re.compile(
    r"\b(?P<phrase>(?:last|this|past) (?:week|month|year)|yesterday|today|"
    r"(?:in the )?last (?P<n>\d+|a|an|one|two|three|four|five|six|seven) "
    r"(?P<unit>hours?|days?|weeks?|months?))\b", re.I)
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
              "five": 5, "six": 6, "seven": 7}


def normalize_alias(alias: str) -> str:
    """'My Advisor' / 'the TA' -> 'advisor' / 'ta'."""
    a = re.sub(r"[^a-z0-9' ]+", " ", str(alias or "").lower())
    a = re.sub(r"\s+", " ", a).strip()
    a = re.sub(r"^(?:my|our|the) ", "", a)
    return a.strip()


def parse_person_statement(text: str) -> Optional[dict]:
    """'my advisor is Dr Peyrovi, email hp@tamu.edu' ->
    {alias, name, email}; None when the sentence is not about a person.

    The shape "my X is Y" also covers "my favourite colour is blue", so a
    match needs one positive sign of a person: an address, a title, a
    relation word for X, or a capitalised name in the transcript (Whisper
    capitalises proper names; it does not capitalise "blue")."""
    m = _PERSON_RX.match((text or "").strip())
    if not m:
        return None
    alias = normalize_alias(m.group("alias"))
    name = m.group("name").strip().rstrip(".")
    email = (m.group("email") or "").strip().rstrip(".")
    if not alias or not name:
        return None
    titled = re.match(_TITLE_RX + r"\s", name + " ", re.I) is not None
    capitalised = name[0].isupper() and not name.isupper()
    if not (email or titled or capitalised or alias in _RELATION_WORDS):
        return None
    if not email and not titled and alias not in _RELATION_WORDS:
        # A capitalised single common word after "my X is" ("my mood is
        # Fine") is not enough on its own: ask for two words or a title.
        if " " not in name:
            return None
    return {"alias": alias, "name": name, "email": email}


def parse_since(query: str, now: Optional[datetime] = None):
    """Strip a time phrase from a recall query -> (query, since datetime|None).
    'what did I say about the thesis last week' -> ('... the thesis', now-7d)."""
    now = now or datetime.now()
    m = _SINCE_RX.search(query or "")
    if not m:
        return (query or "").strip(), None
    phrase = m.group("phrase").lower()
    since = None
    if m.group("unit"):
        n = m.group("n").lower()
        n = _NUM_WORDS.get(n) or (int(n) if n.isdigit() else 1)
        unit = m.group("unit").rstrip("s")
        delta = {"hour": timedelta(hours=n), "day": timedelta(days=n),
                 "week": timedelta(weeks=n), "month": timedelta(days=30 * n)}[unit]
        since = now - delta
    elif phrase == "today":
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif phrase == "yesterday":
        since = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                  microsecond=0)
    elif phrase.endswith("week"):
        since = now - timedelta(weeks=1)
    elif phrase.endswith("month"):
        since = now - timedelta(days=30)
    elif phrase.endswith("year"):
        since = now - timedelta(days=365)
    cleaned = (query[:m.start()] + query[m.end():]).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?")
    return cleaned or (query or "").strip(), since


def _fact_meta(key: str, when: str, ts: float, source: Optional[str] = None) -> dict:
    """Chroma metadata for one fact.  ``source`` is provenance: a fact
    Jarvis promoted himself (the weekly memory garden, jarvis/garden.py)
    carries one, a fact Hunter told him carries none -- so a bad pass can
    be found and undone without touching what he said out loud.  Chroma
    rejects a None value, hence the key is omitted rather than nulled."""
    meta = {"key": str(key), "time": str(when), "ts": float(ts)}
    if source:
        meta["source"] = str(source)
    return meta


def _default_embed():
    """The documents tool's /api/embed seam, keep-alive pinned. Resolved at
    call time so a monkeypatched docs._embed is honoured."""
    from jarvis.tools import docs
    return functools.partial(docs._embed, keep_alive=EMBED_KEEP_ALIVE)


def _embedding_allowed() -> bool:
    """False while the chat model is lent to a trainer.

    Same one-slot rule as above, pointed the other way: during a lend the
    resident model is the trainer's (the nightly haymaker digest runs
    qwen2.5:32b at 04:09), and an embed would evict IT mid-run. The lend
    exists to give the GPU away cleanly; the memory must not take it back
    through a side door. Imported lazily -- memory.py is deliberately free
    of a brain dependency -- and a missing brain simply allows it."""
    try:
        from jarvis import brain
        return not brain.is_lent()
    except Exception:                               # noqa: BLE001
        return True


class SemanticFacts:
    """A chromadb collection of the facts, embedded locally.

    ``embed(texts, model, base_url, timeout) -> vectors`` is the network
    seam (jarvis.tools.docs._embed by default). Every method is failure-
    tolerant: chromadb missing, the index unwritable or Ollama down turns
    into ``available == False`` / an empty result and a log line, never an
    exception in the caller's turn.
    """

    def __init__(self, index_dir: Path, embed: Optional[Callable] = None,
                 model: Optional[str] = None, base_url: Optional[str] = None,
                 gate: Optional[Callable] = None):
        self.index_dir = Path(index_dir)
        self._embed = embed
        self._model = model
        self._base_url = base_url
        self._gate = gate if gate is not None else _embedding_allowed
        self._client = None
        self._collection = None
        self._lock = threading.Lock()
        self._broken = False
        self._synced = False
        self._embed_down_logged = False
        self._down_until = 0.0
        self._qcache: dict[str, list[float]] = {}

    # ---------------------------------------------------------- plumbing
    def _embedder(self):
        if self._embed is not None:
            return self._embed
        return _default_embed()

    def _embed_params(self):
        from jarvis.tools import docs
        model = self._model or docs.DEFAULT_EMBED_MODEL
        base_url = self._base_url or docs.DEFAULT_OLLAMA_URL
        return model, base_url

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        from jarvis.tools import docs
        if time.monotonic() < self._down_until:
            raise docs.EmbedError("embedder backing off after a failure")
        try:
            allowed = bool(self._gate())
        except Exception:                           # noqa: BLE001
            allowed = True
        if not allowed:
            # No back-off: a lend ends on its own and the next call should
            # go straight through, unlike a genuinely dead embedder.
            raise docs.EmbedError("chat model is lent out; not loading the "
                                  "embedder over the trainer")
        model, base_url = self._embed_params()
        try:
            vecs = docs._embed_retrying(self._embedder(), texts, model, base_url)
        except docs.EmbedError:
            # A dead or cold Ollama costs up to two 8 s timeouts; without a
            # back-off EVERY turn would pay that again for as long as it
            # stayed down. One minute of substring-only recall instead.
            self._down_until = time.monotonic() + EMBED_BACKOFF_S
            raise
        self._embed_down_logged = False
        return vecs

    def collection(self):
        """The collection, opened lazily; None when chromadb is unusable."""
        if self._broken:
            return None
        with self._lock:
            if self._collection is None:
                try:
                    import chromadb                 # lazy: ~0.3 s
                    from chromadb.config import Settings
                    self.index_dir.mkdir(parents=True, exist_ok=True)
                    # Fully local means no phone-home: chroma's telemetry
                    # is on by default and posts to posthog.
                    self._client = chromadb.PersistentClient(
                        path=str(self.index_dir),
                        settings=Settings(anonymized_telemetry=False))
                    self._collection = self._client.get_or_create_collection(
                        FACTS_COLLECTION, embedding_function=None,
                        metadata={"hnsw:space": "cosine"})
                except Exception as exc:            # noqa: BLE001 - optional store
                    self._broken = True
                    log.warning("semantic memory unavailable (%s: %s); "
                                "substring recall only", type(exc).__name__,
                                str(exc)[:80])
                    return None
            return self._collection

    @property
    def available(self) -> bool:
        return self.collection() is not None

    @staticmethod
    def _id(key: str) -> str:
        return hashlib.sha1(str(key).encode("utf-8")).hexdigest()

    def _query_vector(self, prefixed: str) -> list[float]:
        """The query embedding, remembered. Every miss costs a model swap
        (the one-slot rule at the top of the module), and Hunter asks the
        same handful of questions all day, so the same vector is worth
        keeping. Insertion-ordered dict = FIFO eviction; the entries are
        768 floats each, so the cap is about 400 kB."""
        with self._lock:
            hit = self._qcache.get(prefixed)
        if hit is not None:
            return hit
        vec = self._vectors([prefixed])[0]
        with self._lock:
            self._qcache[prefixed] = vec
            while len(self._qcache) > QUERY_CACHE_MAX:
                self._qcache.pop(next(iter(self._qcache)))
        return vec

    def _log_embed_down(self, exc):
        if not self._embed_down_logged:
            log.warning("embedder unavailable (%s); substring recall until "
                        "it is back", exc)
            self._embed_down_logged = True

    # ------------------------------------------------------------- write
    def upsert(self, key: str, value, when: Optional[str] = None,
               source: Optional[str] = None) -> bool:
        col = self.collection()
        if col is None:
            return False
        when = when or datetime.now().isoformat()
        try:
            ts = datetime.fromisoformat(when).timestamp()
        except ValueError:
            ts = time.time()
        text = str(value)
        from jarvis.tools import docs
        try:
            vec = self._vectors([docs.DOC_PREFIX + text])[0]
            col.upsert(ids=[self._id(key)], embeddings=[vec], documents=[text],
                       metadatas=[_fact_meta(key, when, ts, source)])
            return True
        except docs.EmbedError as exc:
            self._log_embed_down(exc)
        except Exception:                           # noqa: BLE001
            log.exception("semantic memory upsert failed")
        return False

    def delete(self, key: str) -> None:
        col = self.collection()
        if col is None:
            return
        try:
            col.delete(ids=[self._id(key)])
        except Exception:                           # noqa: BLE001
            log.debug("semantic memory delete failed", exc_info=True)

    def keys(self) -> set[str]:
        col = self.collection()
        if col is None:
            return set()
        try:
            got = col.get(include=["metadatas"])
        except Exception:                           # noqa: BLE001
            log.debug("semantic memory listing failed", exc_info=True)
            return set()
        return {str((m or {}).get("key", "")) for m in (got.get("metadatas") or [])}

    def count(self) -> int:
        col = self.collection()
        try:
            return int(col.count()) if col is not None else 0
        except Exception:                           # noqa: BLE001
            return 0

    def sync(self, facts: dict) -> int:
        """Index every fact the collection lacks (the facts.json migration
        and the catch-up after an outage). Runs at most once per process
        unless it could not embed. Returns how many were added."""
        if self._synced or not facts:
            return 0
        col = self.collection()
        if col is None:
            return 0
        have = self.keys()
        missing = [(k, e) for k, e in facts.items() if k not in have]
        if not missing:
            self._synced = True
            return 0
        from jarvis.tools import docs
        try:
            vecs = self._vectors([docs.DOC_PREFIX + str(e.get("value", ""))
                                  for _, e in missing])
        except docs.EmbedError as exc:
            self._log_embed_down(exc)
            return 0
        except Exception:                           # noqa: BLE001
            log.exception("semantic memory sync failed")
            return 0
        metas, ids, texts = [], [], []
        for (k, e), _v in zip(missing, vecs):
            when = str(e.get("time") or datetime.now().isoformat())
            try:
                ts = datetime.fromisoformat(when).timestamp()
            except ValueError:
                ts = time.time()
            ids.append(self._id(k))
            texts.append(str(e.get("value", "")))
            metas.append(_fact_meta(k, when, ts, e.get("source")))
        try:
            col.upsert(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)
        except Exception:                           # noqa: BLE001
            log.exception("semantic memory sync write failed")
            return 0
        self._synced = True
        log.info("semantic memory: indexed %d facts", len(ids))
        return len(ids)

    # -------------------------------------------------------------- read
    def query(self, text: str, k: int = RECALL_TOP_K,
              since: Optional[datetime] = None,
              floor: Optional[float] = None) -> Optional[list[dict]]:
        """Top-k facts by cosine similarity, at or above ``floor``
        (SCORE_FLOOR_RECALL unless given): [{key, value, time, score}].
        None when the index or the embedder is unavailable (the caller
        falls back); [] when nothing is close."""
        if floor is None:
            floor = SCORE_FLOOR_RECALL          # read at call time: tunable
        col = self.collection()
        if col is None:
            return None
        try:
            total = int(col.count())
        except Exception:                           # noqa: BLE001
            return None
        if total == 0 or not (text or "").strip():
            return []
        from jarvis.tools import docs
        try:
            vec = self._query_vector(docs.QUERY_PREFIX + text.strip())
        except docs.EmbedError as exc:
            self._log_embed_down(exc)
            return None
        except Exception:                           # noqa: BLE001
            log.exception("semantic memory query embed failed")
            return None
        kwargs = {}
        if since is not None:
            kwargs["where"] = {"ts": {"$gte": float(since.timestamp())}}
        try:
            got = col.query(query_embeddings=[vec], n_results=min(max(1, k), total),
                            include=["documents", "metadatas", "distances"], **kwargs)
        except Exception:                           # noqa: BLE001
            log.exception("semantic memory query failed")
            return None
        docs_ = (got.get("documents") or [[]])[0]
        metas = (got.get("metadatas") or [[]])[0]
        dists = (got.get("distances") or [[]])[0]
        out = []
        for doc, meta, dist in zip(docs_, metas, dists):
            meta = meta or {}
            score = round(1.0 - float(dist), 4)
            if score < floor:
                continue
            out.append({"key": str(meta.get("key", "")), "value": doc or "",
                        "time": str(meta.get("time", "")), "score": score})
        return out

    def warm(self, facts: Optional[dict] = None, probe: bool = True) -> bool:
        """Index any missing facts off the turn path, and with ``probe``
        load the embedder too; the app calls this once on a daemon thread
        at start.

        ``probe=False`` when the turn path will never query the index (the
        store fits the prompt): preloading the embedder would only evict
        the 26B that residency has just warmed, buying a 7 s reload on
        Hunter's first question for a model nothing is going to ask."""
        try:
            if facts:
                self.sync(facts)
            if self.collection() is None:
                return False
            if probe:
                self._vectors(["search_query: hello"])
            return True
        except Exception as exc:                    # noqa: BLE001
            log.debug("semantic memory warm-up failed: %s", exc)
            return False


class JarvisMemory:
    """Persistent memory across Jarvis sessions (single store)."""

    MAX_HABITS = 500
    MAX_SESSIONS = 20
    MAX_INTENTS = 500

    def __init__(self, memory_dir: Path | str | None = None,
                 legacy_dir: Path | str | None = None,
                 index_dir: Path | str | None = None,
                 embed: Optional[Callable] = None,
                 semantic: bool = True,
                 gate: Optional[Callable] = None):
        self._dir = Path(memory_dir) if memory_dir else PATHS.MEMORY_DIR
        self._legacy_dir = (Path(legacy_dir) if legacy_dir
                            else PATHS.LEGACY_AGENT_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._facts = self._load("facts.json", {})
        self._habits = self._load("habits.json", [])
        self._preferences = self._load("preferences.json", {})
        self._sessions = self._load("sessions.json", [])
        self._intent_log = self._load("intent_log.json", [])
        self._people = self._load("people.json", {})
        if not isinstance(self._people, dict):
            self._people = {}
        # The index is opened on first use, never here: JarvisMemory() is
        # built in JarvisApp.__init__ and a cold nomic load would sit on
        # the boot path. semantic=False keeps a test on the substring store.
        self._index = SemanticFacts(Path(index_dir) if index_dir
                                    else self._dir / "facts_index",
                                    embed=embed, gate=gate) if semantic else None
        self._corrections = self._load("corrections.json", [])
        self._migrate_legacy()

    # ------------------------------------------------------------------
    # File I/O (atomic writes)
    # ------------------------------------------------------------------
    def _load(self, filename, default):
        path = self._dir / filename
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                log.exception("failed to load %s; using default", filename)
        return default

    def _save(self, filename, data):
        path = self._dir / filename
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)
        except Exception:
            log.exception("save failed (%s)", filename)

    # ------------------------------------------------------------------
    # One-time migration of the legacy jarvis_agent store
    # ------------------------------------------------------------------
    def _migrate_legacy(self):
        """Merge ~/.aiws_trainer/jarvis_data into this store, once."""
        legacy = self._legacy_dir
        if not legacy.is_dir():
            return
        # A dir with NOTHING TO MIGRATE is not legacy data. JarvisAgent.
        # __init__ mkdirs the old path on every boot, so for weeks this
        # warned "already exists; leaving legacy dir in place" at every
        # start about a directory holding nothing (measured 2026-09-12).
        # The test is what the migration below actually reads -- habits.json
        # or a voice note -- not emptiness, so a stray dotfile or an empty
        # voice_notes/ cannot bring the warning back. Nothing is read here.
        try:
            has_data = ((legacy / "habits.json").exists()
                        or any((legacy / "voice_notes").glob("note_*.txt")))
        except OSError:
            log.debug("legacy dir %s unreadable; leaving it", legacy, exc_info=True)
            return
        if not has_data:
            return
        log.info("migrating legacy agent data from %s", legacy)
        try:
            # Merge habit entries (this merges the per-command counts too).
            legacy_habits_file = legacy / "habits.json"
            if legacy_habits_file.exists():
                try:
                    legacy_habits = json.loads(legacy_habits_file.read_text())
                except Exception:
                    log.exception("legacy habits.json unreadable; skipping")
                    legacy_habits = []
                if isinstance(legacy_habits, list) and legacy_habits:
                    merged = legacy_habits + self._habits
                    try:
                        merged.sort(key=lambda h: str(h.get("time", "")))
                    except Exception:
                        log.exception("habit merge sort failed; keeping order")
                    self._habits = merged[-self.MAX_HABITS:]
                    self._save("habits.json", self._habits)
                    log.info("merged %d legacy habit entries",
                             len(legacy_habits))

            # Copy voice notes into the single notes dir.
            notes_src = legacy / "voice_notes"
            if notes_src.is_dir():
                notes_dst = self._dir / "notes"
                notes_dst.mkdir(exist_ok=True)
                copied = 0
                for f in sorted(notes_src.glob("note_*.txt")):
                    dst = notes_dst / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
                        copied += 1
                log.info("copied %d legacy voice notes", copied)

            # Rename the old dir so migration never runs twice.
            migrated = legacy.with_name(legacy.name + ".migrated")
            if migrated.exists():
                log.warning("%s already exists; leaving legacy dir in place",
                            migrated)
            else:
                legacy.rename(migrated)
                log.info("legacy dir renamed to %s", migrated.name)
        except Exception:
            log.exception("legacy migration failed")

    # ------------------------------------------------------------------
    # Facts — key/value store with timestamps
    # ------------------------------------------------------------------
    def remember(self, key, value, source=None):
        """Store a fact persistently (facts.json, then the index).
        Overwrites if key exists. Returns the key it was filed under.

        ``source`` is provenance for a fact Jarvis promoted himself (the
        weekly memory garden passes "garden"); a fact Hunter told him has
        none. Only the tagged ones can be listed or undone wholesale.

        Key and value are filed in the SECOND person -- see
        ``to_second_person``. A fact is written down for Jarvis to read
        back, not quoted."""
        key = to_second_person(key)
        value = to_second_person(value) if isinstance(value, str) else value
        when = datetime.now().isoformat()
        entry = {"value": value, "time": when}
        if source:
            entry["source"] = str(source)
        self._facts[key] = entry
        self._save("facts.json", self._facts)
        log.info("remembered: %s = %s%s", key, str(value)[:50],
                 f" (from {source})" if source else "")
        if self._index is not None:
            try:
                self._index.sync(self._facts)      # first-open migration
                self._index.upsert(key, value, when, source=source)
            except Exception:                       # noqa: BLE001
                log.exception("semantic memory write-through failed")
        return key

    def store_fact_from_speech(self, text) -> str:
        """The method form of ``store_fact_from_speech`` (module level):
        a fact he SAID, cleaned, turned to the second person, keyed on
        its first six words. Returns the value as stored."""
        return store_fact_from_speech(self, text)

    def _substring_recall(self, query, since=None):
        # Asked in the second person, because that is the language the
        # store is written in (see to_second_person): "my dentist" has to
        # match the fact filed as "your dentist is Dr Patel", or filing a
        # fact correctly would make it unfindable.
        q = to_second_person((query or "").lower().strip())
        matches = []
        if not q:
            return matches
        for key, entry in self._facts.items():
            if q in key.lower() or q in str(entry.get("value", "")).lower():
                if since is not None and not self._newer_than(entry, since):
                    continue
                matches.append({"key": key, **entry, "score": 1.0})
        if matches:
            return matches
        # The substring missed. Stemmed word overlap next (stem_match):
        # "graduation" finds "you graduate ...", "my degree" finds "... an
        # electrical engineering degree". Scored under the substring hits
        # so an exact phrase still ranks first when both are present.
        for key, entry in self._facts.items():
            if stem_match(q, f"{key} {entry.get('value', '')}"):
                if since is not None and not self._newer_than(entry, since):
                    continue
                matches.append({"key": key, **entry, "score": 0.9})
        return matches

    @staticmethod
    def _newer_than(entry, since):
        try:
            return datetime.fromisoformat(str(entry.get("time", ""))) >= since
        except ValueError:
            return True

    def recall(self, query, k=RECALL_TOP_K, since=None, floor=None):
        """Facts about ``query``: exact substring hits first (they are the
        stronger evidence), then the nearest by meaning at or above
        ``floor``. ``since`` (datetime or timedelta) keeps only facts stored
        after that point. Never raises; without the index it is the
        substring search alone."""
        if isinstance(since, timedelta):
            since = datetime.now() - since
        out = self._substring_recall(query, since)
        seen = {m["key"] for m in out}
        if self._index is not None and (query or "").strip():
            try:
                self._index.sync(self._facts)
                hits = self._index.query(query, k=k, since=since, floor=floor)
            except Exception:                       # noqa: BLE001
                log.exception("semantic recall failed")
                hits = None
            for h in hits or []:
                key = h.get("key", "")
                if key in seen or key not in self._facts:
                    continue
                entry = self._facts[key]
                seen.add(key)
                out.append({"key": key, "value": entry.get("value", h.get("value")),
                            "time": entry.get("time", h.get("time", "")),
                            "score": h.get("score", 0.0)})
        return out[:k]

    def relevant_facts(self, text, k=CONTEXT_TOP_K, floor=None):
        """The facts worth showing the model for THIS utterance, by
        meaning only (a substring of a chat line is not evidence). [] when
        nothing is close enough or the index is unavailable.

        This EMBEDS, which costs a chat-model swap (the one-slot rule at
        the top of the module), so the caller decides whether ranking is
        worth it: format_for_context only reaches here once the store no
        longer fits the prompt."""
        if floor is None:
            floor = SCORE_FLOOR_CONTEXT
        if self._index is None or not (text or "").strip() or not self._facts:
            return []
        try:
            self._index.sync(self._facts)
            hits = self._index.query(text, k=k, floor=floor)
        except Exception:                           # noqa: BLE001
            log.exception("relevant-facts lookup failed")
            return []
        out = []
        for h in hits or []:
            entry = self._facts.get(h.get("key", ""))
            if entry is None:
                continue
            out.append({"key": h["key"], "value": entry.get("value", ""),
                        "time": entry.get("time", ""), "score": h.get("score", 0.0)})
        return out

    def forget(self, key):
        """Remove a fact."""
        if key in self._facts:
            del self._facts[key]
            self._save("facts.json", self._facts)
        if self._index is not None:
            self._index.delete(key)

    def get_all_facts(self):
        """Return all stored facts."""
        return self._facts

    def facts_from(self, source):
        """The facts a given writer promoted, newest first: [(key, entry)].

        The memory garden's report and its undo read this rather than
        trusting their own state file -- a fact Hunter has since replaced
        by voice loses the tag, and must then survive the undo."""
        want = str(source or "")
        if not want:
            return []
        rows = [(k, e) for k, e in self._facts.items()
                if isinstance(e, dict) and str(e.get("source") or "") == want]
        rows.sort(key=lambda kv: str(kv[1].get("time", "")), reverse=True)
        return rows

    @property
    def semantic_available(self) -> bool:
        return self._index is not None and self._index.available

    def facts_fit_context(self) -> bool:
        """True when every stored fact fits in the per-turn prompt, so
        ranking them buys nothing worth a model swap (the one-slot rule at
        the top of the module). Hunter's live store is 4 facts / ~220
        chars, an order of magnitude inside the budget."""
        facts = self._facts
        if not facts or len(facts) > CONTEXT_ALL_FACTS_MAX:
            return False
        total = 0
        for key, entry in facts.items():
            if not isinstance(entry, dict):
                return False
            total += len(str(key)) + len(str(entry.get("value", ""))) + 4
            if total > CONTEXT_ALL_FACTS_CHARS:
                return False
        return True

    def warm_index(self) -> bool:
        """Open the index, migrate facts.json into it and -- only when the
        turn path will actually query it -- load the embedder. For a daemon
        thread at start, never the boot path."""
        if self._index is None:
            return False
        return self._index.warm(self._facts, probe=not self.facts_fit_context())

    # ------------------------------------------------------------------
    # People book — how Hunter refers to someone -> who they are
    # ------------------------------------------------------------------
    def add_person(self, alias, name, email="", relation=""):
        """'my advisor' -> Dr Peyrovi <hp@tamu.edu>. Returns the stored
        entry; an existing alias is updated and keeps fields not given."""
        key = normalize_alias(alias)
        if not key or not str(name or "").strip():
            return None
        prev = self._people.get(key, {}) if isinstance(self._people.get(key), dict) else {}
        entry = {
            "name": str(name).strip(),
            "email": str(email or prev.get("email", "")).strip(),
            "relation": str(relation or prev.get("relation", "") or key).strip(),
            "time": datetime.now().isoformat(),
        }
        self._people[key] = entry
        if len(self._people) > MAX_PEOPLE:
            oldest = sorted(self._people, key=lambda a: self._people[a].get("time", ""))
            for a in oldest[:len(self._people) - MAX_PEOPLE]:
                del self._people[a]
        self._save("people.json", self._people)
        log.info("person: %s = %s <%s>", key, entry["name"], entry["email"])
        return dict(entry, alias=key)

    def remove_person(self, alias) -> bool:
        key = normalize_alias(alias)
        if key in self._people:
            del self._people[key]
            self._save("people.json", self._people)
            return True
        return False

    def people(self) -> dict:
        return dict(self._people)

    def resolve_person(self, text) -> Optional[dict]:
        """Who 'my advisor' / 'Peyrovi' / 'hp@tamu.edu' is: the entry plus
        its alias, or None. Alias first, then a name or address match."""
        raw = str(text or "").strip()
        if not raw or not self._people:
            return None
        key = normalize_alias(raw)
        entry = self._people.get(key)
        if isinstance(entry, dict):
            return dict(entry, alias=key)
        low = raw.lower()
        for alias, entry in self._people.items():
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).lower()
            email = str(entry.get("email", "")).lower()
            if (email and low == email) or (name and (low == name or
                                                      low == name.split()[-1])):
                return dict(entry, alias=alias)
        return None

    def expand_aliases(self, text) -> str:
        """'lunch with mom' -> 'lunch with Linda Peyrovi' (whole words,
        with or without a leading my/the). Text without an alias is
        returned untouched."""
        out = str(text or "")
        if not out or not self._people:
            return out
        for alias in sorted(self._people, key=len, reverse=True):
            entry = self._people.get(alias)
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            rx = re.compile(r"\b(?:(?:my|our|the) )?" + re.escape(alias) + r"\b", re.I)
            out = rx.sub(entry["name"], out)
        return out

    def format_people(self, limit=20) -> str:
        if not self._people:
            return ""
        lines = ["People (how Hunter refers to them):"]
        items = sorted(self._people.items(), key=lambda kv: kv[1].get("time", ""),
                       reverse=True)[:limit]
        for alias, entry in items:
            if not isinstance(entry, dict):
                continue
            line = f"  my {alias}: {entry.get('name', '')}"
            if entry.get("email"):
                line += f" <{entry['email']}>"
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Habits — command logging with time patterns (the ONE habit log)
    # ------------------------------------------------------------------
    def log_habit(self, command, context=None):
        """Log a command execution for pattern learning."""
        self._habits.append({
            "time": datetime.now().isoformat(),
            "hour": datetime.now().hour,
            "day": datetime.now().strftime("%A"),
            "command": command[:100],
            "context": context,
        })
        self._habits = self._habits[-self.MAX_HABITS:]
        self._save("habits.json", self._habits)

    def suggest_by_habit(self):
        """Suggest a command based on current time patterns."""
        if len(self._habits) < 10:
            return None
        hour = datetime.now().hour
        counts = {}
        for h in self._habits:
            if abs(h.get("hour", -1) - hour) <= 1:
                cmd = h["command"]
                counts[cmd] = counts.get(cmd, 0) + 1
        if counts:
            best = max(counts, key=counts.get)
            if counts[best] >= 3:
                return best
        return None

    def get_habit_summary(self):
        """Summary of habit patterns for context injection."""
        if not self._habits:
            return "No habits recorded yet."
        total = len(self._habits)
        counts = {}
        for h in self._habits:
            cmd = h["command"]
            counts[cmd] = counts.get(cmd, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:5]
        lines = [f"Total commands logged: {total}"]
        for cmd, count in top:
            lines.append(f"  {cmd}: {count}x")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Preferences — user settings/preferences
    # ------------------------------------------------------------------
    def set_preference(self, key, value):
        self._preferences[key] = value
        self._save("preferences.json", self._preferences)

    def get_preference(self, key, default=None):
        return self._preferences.get(key, default)

    def get_all_preferences(self):
        return self._preferences

    # ------------------------------------------------------------------
    # Session summaries — compressed conversation history
    # ------------------------------------------------------------------
    def save_session(self, summary):
        """Save a session summary for cross-session memory.

        Called from app shutdown so the next session remembers this one.
        """
        if not summary:
            return
        self._sessions.append({
            "time": datetime.now().isoformat(),
            "summary": str(summary)[:500],
        })
        self._sessions = self._sessions[-self.MAX_SESSIONS:]
        self._save("sessions.json", self._sessions)
        log.info("session saved: %s", str(summary)[:50])

    def get_recent_sessions(self, n=3):
        """Get last N session summaries."""
        return self._sessions[-n:]

    def format_sessions_for_prompt(self):
        """Format session summaries for context injection.

        The header earns its words. LIVE 2026-09-02 14:29:24: this block
        carried the PREVIOUS EVENING's answer to "Say hello to my family"
        -- cut at 100 characters, so it ended mid-clause -- and gemma4
        copied the opening straight back out at 2:29 in the afternoon
        ("Good evening, Ali and Heather; ... a lovely afternoon, sir").
        Rendered as a bare list of exchanges it reads like a draft to
        finish, so it now says what it is. The greeting itself is no
        longer left to the prompt either: brain.ground_greeting reads the
        word off the clock (jarvis/arc.py, greeting_word).
        """
        sessions = self.get_recent_sessions()
        if not sessions:
            return ""
        parts = ["Previous sessions (older conversations, for recall only; "
                 "never reuse their wording):"]
        for s in sessions:
            parts.append(f"  [{s['time'][:16]}] {s['summary'][:100]}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Intent learning — "Was this for me?" feedback
    # ------------------------------------------------------------------
    def log_intent(self, text, is_for_assistant):
        """Record user feedback on intent classification."""
        self._intent_log.append({
            "text": text[:200],
            "label": "yes" if is_for_assistant else "no",
        })
        self._intent_log = self._intent_log[-self.MAX_INTENTS:]
        self._save("intent_log.json", self._intent_log)

    def get_intent_log(self):
        return self._intent_log

    # ------------------------------------------------------------------
    # Mishearing corrections — "no, I said ..."
    # ------------------------------------------------------------------
    MAX_CORRECTIONS = 300

    def log_correction(self, heard, meant):
        """Record a (heard, meant) pair from a spoken correction. The file
        is the evidence for a later vocab / prompt tune; nothing reads it
        at runtime."""
        self._corrections.append({
            "time": datetime.now().isoformat(),
            "heard": (heard or "")[:200],
            "meant": (meant or "")[:200],
        })
        self._corrections = self._corrections[-self.MAX_CORRECTIONS:]
        self._save("corrections.json", self._corrections)

    def get_corrections(self):
        return self._corrections

    # ------------------------------------------------------------------
    # Voice notes — timestamped text memos
    # ------------------------------------------------------------------
    def save_note(self, text):
        """Save a timestamped voice note."""
        notes_dir = self._dir / "notes"
        notes_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = notes_dir / f"note_{ts}.txt"
        path.write_text(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n{text}\n")
        log.info("note saved: %s", path.name)
        return str(path)

    def get_notes(self, n=5):
        """List recent voice notes."""
        notes_dir = self._dir / "notes"
        if not notes_dir.exists():
            return []
        files = sorted(notes_dir.glob("note_*.txt"), reverse=True)[:n]
        return [{"file": f.name, "content": f.read_text().strip()[:100]}
                for f in files]

    # ------------------------------------------------------------------
    # Full memory dump for context
    # ------------------------------------------------------------------
    def format_for_context(self, text=""):
        """Memory for the model's user turn.

        A store that FITS the prompt is rendered whole and never touches
        the embedder -- that is the reply path's half of the one-slot rule
        at the top of the module, and it hands the model more facts than
        ranking did, not fewer. Only once the store outgrows the budget is
        it worth a model swap to rank it: then, with ``text``, the facts
        are the ones RELEVANT to it by meaning, and with no text (or no
        index) the last five stored.

        The People block is rendered outside that window on every turn, so
        a contact never drops out after five later "remember"s."""
        parts = []

        if self._facts:
            fits = self.facts_fit_context()
            relevant = self.relevant_facts(text) if (text and not fits) else []
            if relevant:
                parts.append(f"Facts he told you that bear on this ({len(self._facts)} stored):")
                for fact in relevant:
                    parts.append(f"  {fact['value']}")
            elif fits or not text or not self.semantic_available:
                parts.append(f"Known facts ({len(self._facts)}):")
                rows = list(self._facts.items())
                for key, entry in (rows if fits else rows[-5:]):
                    parts.append(f"  {_fact_line(key, entry)}")

        people_text = self.format_people()
        if people_text:
            parts.append(people_text)

        suggestion = self.suggest_by_habit()
        if suggestion:
            parts.append(
                f"Habit suggestion: user often runs '{suggestion}' at this time")

        sessions_text = self.format_sessions_for_prompt()
        if sessions_text:
            parts.append(sessions_text)

        return "\n".join(parts) if parts else ""
