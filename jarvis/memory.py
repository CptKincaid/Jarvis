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
old substring test never matched -- and ``format_for_context(text)`` hands
the model the facts RELEVANT to the current utterance instead of the last
five in insertion order. With chromadb or Ollama unavailable both fall back
to the substring store; nothing here ever raises into a turn.

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
# Keep the embedder loaded: docs._embed sends no keep_alive, so nomic
# unloads after Ollama's default five minutes and the first "who's my
# dentist" after a quiet spell paid the 7.5 s cold load (and could hit the
# retry). ~270 MB of unified memory is nothing beside the chat model.
EMBED_KEEP_ALIVE = -1
EMBED_BACKOFF_S = 60.0
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


def _default_embed():
    """The documents tool's /api/embed seam, keep-alive pinned. Resolved at
    call time so a monkeypatched docs._embed is honoured."""
    from jarvis.tools import docs
    return functools.partial(docs._embed, keep_alive=EMBED_KEEP_ALIVE)


class SemanticFacts:
    """A chromadb collection of the facts, embedded locally.

    ``embed(texts, model, base_url, timeout) -> vectors`` is the network
    seam (jarvis.tools.docs._embed by default). Every method is failure-
    tolerant: chromadb missing, the index unwritable or Ollama down turns
    into ``available == False`` / an empty result and a log line, never an
    exception in the caller's turn.
    """

    def __init__(self, index_dir: Path, embed: Optional[Callable] = None,
                 model: Optional[str] = None, base_url: Optional[str] = None):
        self.index_dir = Path(index_dir)
        self._embed = embed
        self._model = model
        self._base_url = base_url
        self._client = None
        self._collection = None
        self._lock = threading.Lock()
        self._broken = False
        self._synced = False
        self._embed_down_logged = False
        self._down_until = 0.0

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

    def _log_embed_down(self, exc):
        if not self._embed_down_logged:
            log.warning("embedder unavailable (%s); substring recall until "
                        "it is back", exc)
            self._embed_down_logged = True

    # ------------------------------------------------------------- write
    def upsert(self, key: str, value, when: Optional[str] = None) -> bool:
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
                       metadatas=[{"key": str(key), "time": when, "ts": float(ts)}])
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
            metas.append({"key": str(k), "time": when, "ts": float(ts)})
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
            vec = self._vectors([docs.QUERY_PREFIX + text.strip()])[0]
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

    def warm(self, facts: Optional[dict] = None) -> bool:
        """Load the embedder (and index any missing facts) off the turn
        path; the app calls this once on a daemon thread at start."""
        try:
            if facts:
                self.sync(facts)
            if self.collection() is None:
                return False
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
                 semantic: bool = True):
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
                                    embed=embed) if semantic else None
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
    def remember(self, key, value):
        """Store a fact persistently (facts.json, then the index).
        Overwrites if key exists."""
        when = datetime.now().isoformat()
        self._facts[key] = {"value": value, "time": when}
        self._save("facts.json", self._facts)
        log.info("remembered: %s = %s", key, str(value)[:50])
        if self._index is not None:
            try:
                self._index.sync(self._facts)      # first-open migration
                self._index.upsert(key, value, when)
            except Exception:                       # noqa: BLE001
                log.exception("semantic memory write-through failed")

    def _substring_recall(self, query, since=None):
        q = (query or "").lower().strip()
        matches = []
        if not q:
            return matches
        for key, entry in self._facts.items():
            if q in key.lower() or q in str(entry.get("value", "")).lower():
                if since is not None and not self._newer_than(entry, since):
                    continue
                matches.append({"key": key, **entry, "score": 1.0})
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
        nothing is close enough or the index is unavailable."""
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

    @property
    def semantic_available(self) -> bool:
        return self._index is not None and self._index.available

    def warm_index(self) -> bool:
        """Open the index, migrate facts.json into it and load the embedder
        -- for a daemon thread at start, never the boot path."""
        if self._index is None:
            return False
        return self._index.warm(self._facts)

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
        """Format session summaries for context injection."""
        sessions = self.get_recent_sessions()
        if not sessions:
            return ""
        parts = ["Previous sessions:"]
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
        """Memory for the model's user turn. With ``text`` (the current
        utterance) the facts are the ones RELEVANT to it by meaning; with
        no text, or no index, the last five stored. The People block is
        rendered outside that window on every turn, so a contact never
        drops out after five later "remember"s."""
        parts = []

        if self._facts:
            relevant = self.relevant_facts(text) if text else []
            if relevant:
                parts.append(f"Facts he told you that bear on this ({len(self._facts)} stored):")
                for fact in relevant:
                    parts.append(f"  {fact['value']}")
            elif not text or not self.semantic_available:
                parts.append(f"Known facts ({len(self._facts)}):")
                for key, entry in list(self._facts.items())[-5:]:
                    parts.append(f"  {key}: {entry['value']}")

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
