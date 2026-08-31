"""Ask your own documents, fully local (spec 6.x, "docs").

Text from the folders in ``docs.paths`` (.pdf / .txt / .md / .docx) is cut
into ~800-character chunks, embedded with Ollama's ``/api/embed``
(``nomic-embed-text``) and stored in a chromadb ``PersistentClient`` at
``docs.index_dir``. ``ask_docs`` embeds the question, pulls the top five
chunks and hands the model a fact sheet ("From syllabus.pdf: ...") so it
answers and names the file; ``docs_reindex`` rebuilds incrementally.

Incremental by (path, mtime, size): chroma's own metadata is the manifest,
so a crash between "chunks stored" and "manifest written" cannot leave a
file half-indexed and forgotten. The first use kicks a background reindex
so a fresh box answers from whatever it already has while it catches up.

All network I/O goes through the module-level ``_embed`` seam (tests pass
a deterministic fake); chromadb is imported lazily so a boot without the
tool never pays for it. PDF text comes from poppler's ``pdftotext`` CLI
when present, else ``pypdf`` when importable, else the file is skipped
with a warning — never an install at runtime. .docx is read with the
stdlib (zipfile + ElementTree), no python-docx needed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.docs")

DEFAULT_PATHS = ["~/Documents/Jarvis Docs"]
DEFAULT_INDEX_DIR = "~/.aiws_trainer/docs_index"
DEFAULT_MAX_FILES = 500
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
COLLECTION = "jarvis_docs"
SUFFIXES = (".pdf", ".txt", ".md", ".docx")
CHUNK_CHARS = 800
CHUNK_OVERLAP = 120
TOP_K = 5
MIN_TOPIC_SCORE = 0.3              # topic_chunks: below this a hit is off-topic
SHEET_CHUNK_CHARS = 700            # per-chunk cap in the fact sheet the model reads
EMBED_TIMEOUT = 8.0                # one /api/embed call
EMBED_BATCH = 32                   # chunks per /api/embed call
PDF_TIMEOUT = 8.0                  # one pdftotext run
MAX_TEXT_CHARS = 400_000           # a book-length file still indexes; a dump does not
# Measured 2026-08-30: a cold nomic-embed-text load is 7.5 s, a hair under
# the 8 s cap, so the FIRST call can time out while Ollama is still loading
# the model. One retry lands on the now-resident model; a second timeout is
# genuinely "Ollama is down".
EMBED_RETRIES = 1
# nomic-embed-text is trained with task prefixes; asymmetric retrieval
# (short question vs long passage) is markedly better with them.
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "

NO_DOCS_LINE = "I have no documents indexed yet, sir; put them in {folder}."
INDEX_DOWN_LINE = "My document index isn't answering, sir."
INDEXING_LINE = "I'm indexing your documents now, sir; ask me again in a moment."
INDEXED_LINE = "Indexed {n} documents, sir."
UNREADABLE_LINE = "I couldn't read any of the documents in {folder}, sir."
NO_QUESTION_LINE = "What would you like to know from your documents, sir?"

# Words a spoken document name carries that its file name never does.
_NAME_STOPWORDS = frozenset({
    "the", "a", "an", "my", "our", "this", "that", "of", "for", "on", "in",
    "to", "and", "from", "file", "document", "doc", "pdf", "docx", "paper",
    "handout", "notes", "reading", "please", "me", "it", "one"})
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                 "eleven": "11", "twelve": "12"}
_CHAPTER_RX = re.compile(
    r"\b(chapter|chap|section|unit|lecture|week|module|part|lab)\s*"
    r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.I)


class EmbedError(RuntimeError):
    """Ollama unreachable, timed out, or returned no vectors."""


# ------------------------------------------------------------- config
def _cfg_get(cfg, dotted: str, default=None):
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
        except Exception:
            log.debug("cfg.get(%s) failed", dotted, exc_info=True)
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def doc_paths(cfg) -> list[Path]:
    raw = _cfg_get(cfg, "docs.paths", None)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raw = DEFAULT_PATHS
    return [Path(os.path.expanduser(str(p))).resolve() for p in raw if str(p).strip()]


def index_dir(cfg) -> Path:
    # env > config > default. AssistantConfig always answers docs.index_dir
    # (DEFAULTS carries it), so config-first left JARVIS_DOCS_INDEX_DIR dead
    # in the app and in every test built on a real config.
    raw = os.environ.get("JARVIS_DOCS_INDEX_DIR") or _cfg_get(cfg, "docs.index_dir", "") \
        or DEFAULT_INDEX_DIR
    return Path(os.path.expanduser(str(raw)))


def _int_cfg(cfg, key: str, default: int) -> int:
    try:
        return max(1, int(_cfg_get(cfg, key, default) or default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------- embed
def _embed(texts: list[str], model: str = DEFAULT_EMBED_MODEL,
           base_url: str = DEFAULT_OLLAMA_URL,
           timeout: float = EMBED_TIMEOUT,
           keep_alive=None) -> list[list[float]]:
    """The ONE network seam: POST /api/embed -> one vector per text.
    Raises EmbedError on any failure (tests replace this function).
    ``keep_alive`` (jarvis.memory pins it to -1) rides in the payload only
    when given, so the documents index keeps Ollama's default unload."""
    payload = {"model": model, "input": list(texts)}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError,
            ValueError) as exc:
        raise EmbedError(f"{type(exc).__name__}: {str(exc)[:80]}") from exc
    vectors = data.get("embeddings") if isinstance(data, dict) else None
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbedError(f"bad embed payload for {len(texts)} texts")
    return vectors


Embed = Callable[..., list[list[float]]]


def _embed_retrying(embed: Embed, texts: list[str], model: str, base_url: str,
                    timeout: float = EMBED_TIMEOUT) -> list[list[float]]:
    last: Optional[Exception] = None
    for attempt in range(EMBED_RETRIES + 1):
        try:
            return embed(texts, model, base_url, timeout)
        except EmbedError as exc:
            last = exc
            log.warning("embed attempt %d failed: %s", attempt + 1, exc)
    raise last if last is not None else EmbedError("embed failed")


# ---------------------------------------------------------- extraction
def _pdf_text(path: Path) -> str:
    exe = shutil.which("pdftotext")
    if exe:
        try:
            proc = subprocess.run([exe, "-enc", "UTF-8", "-q", str(path), "-"],
                                  capture_output=True, timeout=PDF_TIMEOUT, check=False)
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("pdftotext failed on %s: %s", path.name, type(exc).__name__)
            return ""
        return proc.stdout.decode("utf-8", errors="replace")
    try:
        import pypdf                        # optional; never installed here
    except ImportError:
        log.warning("no pdftotext and no pypdf: skipping %s", path.name)
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:                # noqa: BLE001 - one bad file must not stop the index
        log.warning("pypdf failed on %s: %s", path.name, type(exc).__name__)
        return ""


def _docx_text(path: Path) -> str:
    """word/document.xml paragraphs joined with newlines (stdlib only)."""
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
        root = ET.fromstring(xml)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError) as exc:
        log.warning("docx read failed on %s: %s", path.name, type(exc).__name__)
        return ""
    paras = []
    for p in root.iter():
        if p.tag.rsplit("}", 1)[-1] != "p":
            continue
        runs = [t.text or "" for t in p.iter() if t.tag.rsplit("}", 1)[-1] == "t"]
        text = "".join(runs).strip()
        if text:
            paras.append(text)
    return "\n".join(paras)


def extract_text(path: Path) -> str:
    """Plain text for one supported file ("" when unreadable/unsupported)."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".pdf":
            text = _pdf_text(path)
        elif suffix == ".docx":
            text = _docx_text(path)
        else:
            return ""
    except OSError as exc:
        log.warning("read failed on %s: %s", path.name, type(exc).__name__)
        return ""
    return text[:MAX_TEXT_CHARS]


# ------------------------------------------------------------ chunking
def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """~size-char windows with overlap, cut at a paragraph/sentence/word
    boundary when one is near the end so a chunk rarely splits a sentence."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    overlap = min(overlap, size // 2)
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            # Prefer the latest boundary in the back third of the window.
            floor = size * 2 // 3
            cut = -1
            for pat in ("\n\n", "\n", ". ", "? ", "! ", " "):
                pos = window.rfind(pat)
                if pos >= floor:
                    cut = pos + len(pat)
                    break
            if cut > 0:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _name_tokens(text: str) -> list[str]:
    """Lower-case alphanumeric tokens of a spoken name or a file name
    ("Biosensors_Lab-Handout_v2.pdf" -> biosensors, lab, handout, v2),
    number words as digits, stopwords out."""
    stem = re.sub(r"\.(pdf|docx?|txt|md|rst|tex)$", "", str(text or "").strip(), flags=re.I)
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+|\d+", stem)]
    toks = [_NUMBER_WORDS.get(t, t) for t in toks]
    return [t for t in toks if t not in _NAME_STOPWORDS]


def match_name(query: str, names: list[str], min_score: float = 0.6) -> Optional[str]:
    """The file name a spoken name most plausibly means, or None.

    Score = the share of the query's content words found in the file
    name (whole token or prefix of one: "bio sensor" finds biosensors);
    ties go to the name with the fewest extra words, so "the lab handout"
    picks lab_handout.pdf over lab_handout_answers.pdf. The floor is above
    one half so a two-word name needs both words: "the lab report" must
    not resolve to the biosensors LAB handout."""
    q = _name_tokens(query)
    if not q or not names:
        return None
    best, best_key = None, None
    for name in names:
        n = _name_tokens(name)
        if not n:
            continue
        joined = " ".join(n)
        hit = 0
        for tok in q:
            if tok in n or (len(tok) >= 4 and any(t.startswith(tok) or tok.startswith(t)
                                                  for t in n if len(t) >= 4)):
                hit += 1
            elif tok.isdigit() and tok in joined:
                hit += 1
        score = hit / len(q)
        if score < min_score:
            continue
        key = (score, -abs(len(n) - len(q)), -len(name))
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best


def _compact(text: str, limit: int = SHEET_CHUNK_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# --------------------------------------------------------------- index
class DocsIndex:
    """Chunks of every supported file under ``paths`` in a chromadb
    collection at ``index_dir``; ``embed`` is the network seam."""

    def __init__(self, paths: list[Path], index_dir: Path, embed: Embed = _embed,
                 max_files: int = DEFAULT_MAX_FILES, model: str = DEFAULT_EMBED_MODEL,
                 base_url: str = DEFAULT_OLLAMA_URL):
        self.paths = list(paths)
        self.index_dir = Path(index_dir)
        self.embed = embed
        self.max_files = max_files
        self.model = model
        self.base_url = base_url
        self._client = None
        self._collection = None
        self._lock = threading.Lock()          # one reindex at a time
        self._open_lock = threading.Lock()     # one PersistentClient, even with a race
        self._thread: Optional[threading.Thread] = None
        self._last: Optional[dict] = None      # result of the latest reindex
        self._kicked = False

    # ------------------------------------------------------------ store
    def collection(self):
        with self._open_lock:
            if self._collection is None:
                import chromadb                # lazy: ~0.3 s and only for this tool
                from chromadb.config import Settings
                self.index_dir.mkdir(parents=True, exist_ok=True)
                # Fully local means no phone-home: chroma's anonymized
                # telemetry is on by default and posts to posthog.
                self._client = chromadb.PersistentClient(
                    path=str(self.index_dir),
                    settings=Settings(anonymized_telemetry=False))
                self._collection = self._client.get_or_create_collection(
                    COLLECTION, embedding_function=None,
                    metadata={"hnsw:space": "cosine"})
            return self._collection

    def indexed(self) -> dict[str, tuple[float, int]]:
        """path -> (mtime, size) as chroma remembers it."""
        got = self.collection().get(include=["metadatas"])
        out: dict[str, tuple[float, int]] = {}
        for meta in got.get("metadatas") or []:
            if isinstance(meta, dict) and meta.get("path"):
                out[str(meta["path"])] = (float(meta.get("mtime", 0)), int(meta.get("size", 0)))
        return out

    def document_count(self) -> int:
        try:
            return len(self.indexed())
        except Exception:                      # noqa: BLE001 - store boundary
            log.exception("docs index unreadable")
            return 0

    def names(self) -> list[str]:
        """The file names in the store (sorted, unique)."""
        return sorted({Path(p).name for p in self.indexed()})

    def chunks_of(self, name: str) -> list[dict]:
        """Every chunk of one indexed file, in document order:
        [{name, path, text, chunk}]. No embedding call is made."""
        got = self.collection().get(where={"name": name},
                                    include=["documents", "metadatas"])
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        out = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            out.append({"name": str(meta.get("name") or name),
                        "path": str(meta.get("path", "")), "text": doc or "",
                        "chunk": int(meta.get("chunk", 0))})
        out.sort(key=lambda h: h["chunk"])
        return out

    # ------------------------------------------------------------- scan
    def scan(self) -> list[Path]:
        """Supported files under every configured folder (sorted, capped)."""
        found: list[Path] = []
        for root in self.paths:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in SUFFIXES \
                        and not p.name.startswith("."):
                    found.append(p)
        if len(found) > self.max_files:
            log.warning("docs: %d files found, indexing the first %d",
                        len(found), self.max_files)
        return found[:self.max_files]

    # ---------------------------------------------------------- reindex
    def reindex(self) -> dict:
        """Add changed/new files, drop deleted ones. Never raises: the
        result dict carries ``error`` when Ollama or the store failed."""
        with self._lock:
            result = {"indexed": 0, "removed": 0, "skipped": 0, "documents": 0,
                      "errors": 0, "error": None}
            try:
                col = self.collection()
                known = self.indexed()
            except Exception as exc:           # noqa: BLE001 - store boundary
                log.exception("docs index open failed")
                result["error"] = f"store: {type(exc).__name__}"
                self._last = result
                if result.get("error"):
                    self._kicked = False      # let the next ask kick a fresh pass
                return result
            files = self.scan()
            present = {str(p) for p in files}
            for path in list(known):
                if path in present:
                    continue
                try:
                    col.delete(where={"path": path})
                except Exception:              # noqa: BLE001 - store boundary
                    log.exception("docs: could not drop %s", path)
                    result["errors"] += 1
                    continue
                result["removed"] += 1
                known.pop(path, None)
            t0 = time.monotonic()
            for path in files:
                try:
                    st = path.stat()
                except OSError:
                    continue
                key = (round(st.st_mtime, 3), st.st_size)
                if known.get(str(path)) == key:
                    result["skipped"] += 1
                    continue
                try:
                    stored = self._index_file(col, path, key)
                except EmbedError as exc:
                    log.warning("docs: embedding unavailable (%s); reindex halted", exc)
                    result["error"] = "embed"
                    break
                except Exception:              # noqa: BLE001 - one bad file
                    log.exception("docs: %s failed", path.name)
                    result["errors"] += 1
                    continue
                if stored:
                    result["indexed"] += 1
                    known[str(path)] = key
                else:
                    result["errors"] += 1
            result["documents"] = len(known)
            log.info("docs reindex: %d new, %d removed, %d unchanged, %d failed "
                     "in %.1fs (%d documents)", result["indexed"], result["removed"],
                     result["skipped"], result["errors"], time.monotonic() - t0,
                     result["documents"])
            self._last = result
            if result.get("error"):
                self._kicked = False      # let the next ask kick a fresh pass
            return result

    def _index_file(self, col, path: Path, key: tuple[float, int]) -> bool:
        text = extract_text(path)
        chunks = chunk_text(text)
        if not chunks:
            log.warning("docs: no text in %s", path.name)
            return False
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = [DOC_PREFIX + c for c in chunks[i:i + EMBED_BATCH]]
            vectors.extend(_embed_retrying(self.embed, batch, self.model, self.base_url))
        # Replace, never append: a shrunk file would otherwise keep its
        # tail chunks from the previous version.
        col.delete(where={"path": str(path)})
        col.upsert(
            ids=[f"{path}::{i}" for i in range(len(chunks))],
            embeddings=vectors, documents=chunks,
            metadatas=[{"path": str(path), "name": path.name, "chunk": i,
                        "mtime": key[0], "size": key[1]} for i in range(len(chunks))])
        return True

    def mark_used(self) -> bool:
        """True exactly once per process: the first use of the index."""
        if self._kicked:
            return False
        self._kicked = True
        return True

    def start_background(self) -> bool:
        """Kick ONE background reindex per process (the first use)."""
        if not self.mark_used():
            return False
        self._thread = threading.Thread(target=self.reindex, name="docs-reindex",
                                        daemon=True)
        self._thread.start()
        return True

    def kick(self) -> bool:
        """A fresh background pass regardless of the once-per-process latch
        -- a file was just written (lecture notes) and should be findable
        without waiting for the next ask. No-op while a pass is running."""
        if self.busy():
            return False
        self._kicked = False
        return self.start_background()

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def last_result(self) -> Optional[dict]:
        return self._last

    # ------------------------------------------------------------ query
    def query(self, question: str, k: int = TOP_K) -> list[dict]:
        """Top-k chunks: [{name, path, text, score}] (score = 1 - cosine
        distance). Raises EmbedError when Ollama is unreachable."""
        col = self.collection()
        total = col.count()
        if total == 0:
            return []
        vec = _embed_retrying(self.embed, [QUERY_PREFIX + question], self.model,
                              self.base_url)[0]
        got = col.query(query_embeddings=[vec], n_results=min(k, total),
                        include=["documents", "metadatas", "distances"])
        docs = (got.get("documents") or [[]])[0]
        metas = (got.get("metadatas") or [[]])[0]
        dists = (got.get("distances") or [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            out.append({"name": str(meta.get("name") or Path(str(meta.get("path", ""))).name),
                        "path": str(meta.get("path", "")), "text": doc or "",
                        "chunk": int(meta.get("chunk", 0)),
                        "score": round(1.0 - float(dist), 4)})
        return out


def _heading_rx(topic: str):
    """A regex matching the "chapter N" heading the topic names, or None.

    Shared by topic_chunks and course_chunks so a scoped ask ("teach me
    chapter three of biosensors") starts at the same heading an unscoped
    one would."""
    heading = _CHAPTER_RX.search(topic or "")
    if not heading:
        return None
    num = _NUMBER_WORDS.get(heading.group(2).lower(), heading.group(2))
    words = {v: k_ for k_, v in _NUMBER_WORDS.items()}.get(num, "")
    return re.compile(r"\b%s\s*(?:%s%s)\b" % (
        r"(?:chapter|chap\.?|section|unit|lecture|week|module|part|lab)",
        re.escape(num), f"|{words}" if words else ""), re.I)


def topic_chunks(index: DocsIndex, topic: str, k: int = 6) -> list[dict]:
    """Chunks to study for a spoken topic, in reading order.

    "Chapter three" is a poor semantic query (every chapter heading looks
    alike to the embedder), so the order is: (1) a file whose NAME matches
    the topic -> its chunks, starting at the "chapter N" heading when the
    topic names one; (2) a "chapter N" heading anywhere in the store ->
    that file from that chunk on; (3) the embedding query, its hits
    re-sorted by (file, chunk) so the questions follow the text. Raises
    EmbedError only on the last leg."""
    topic = " ".join(str(topic or "").split())
    if not topic:
        return []
    k = max(1, int(k))
    head_rx = _heading_rx(topic)

    def _from_heading(chunks: list[dict]) -> list[dict]:
        if head_rx is None:
            return chunks[:k]
        for i, h in enumerate(chunks):
            if head_rx.search(h["text"]):
                return chunks[i:i + k]
        return []

    name = match_name(topic, index.names())
    if name:
        chunks = index.chunks_of(name)
        picked = _from_heading(chunks)
        if picked:
            return picked
        if head_rx is None:
            return chunks[:k]
    if head_rx is not None:
        for other in index.names():
            if other == name:
                continue
            picked = _from_heading(index.chunks_of(other))
            if picked:
                return picked
    # chroma always returns k neighbours; below the floor they are not
    # about the topic (nomic scores unrelated passages ~0.3-0.45, related
    # ones 0.55+), and a quiz written from them would be about the wrong thing.
    hits = [h for h in index.query(topic, k=k) if h["score"] >= MIN_TOPIC_SCORE]
    hits.sort(key=lambda h: (h["name"], h["chunk"]))
    return hits


# "Week four of signals and systems" and "electrode transducers in
# biosensors" name a topic AND a course, and only the tail is the course.
# The tail after one of these words is tried as a course name in its own
# right, which is what makes the no-Canvas path work at all: resolve_course
# hands the spoken words straight back, so the whole utterance would have
# to slugify to a file stem, and it never does.
_COURSE_TAIL_RX = re.compile(r"\b(?:of|in|for|from|on|about)\b", re.I)
MAX_COURSE_CANDIDATES = 3          # each one may cost a (cached) roster lookup
MIN_COURSE_SLUG = 3                # a two-letter slug matches half the folder


def _course_candidates(topic: str) -> list[str]:
    """Readings of the topic that could name a course, longest first: the
    whole thing, then each tail after "of" / "in" / "for" / ..."""
    out = [topic]
    for m in _COURSE_TAIL_RX.finditer(topic):
        tail = topic[m.end():].strip()
        if tail and tail not in out:
            out.append(tail)
    return out[:MAX_COURSE_CANDIDATES]


def _slug_files(names: list[str], slug: str) -> list[str]:
    """The indexed file names whose stem carries ``slug`` as a whole token.

    Whole-token, not substring: the slug "physics" must find
    "physics-2026-08-29.md" and "physics_notes.txt" without "cs" finding
    every file in the folder."""
    if len(slug) < MIN_COURSE_SLUG:
        return []
    rx = re.compile(r"(?:^|[^a-z0-9])%s(?:[^a-z0-9]|$)" % re.escape(slug))
    out = []
    for name in names:
        stem = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower())
        if rx.search(stem):
            out.append(name)
    return out


def course_chunks(index: DocsIndex, cfg, topic: str, k: int = 6) -> list[dict]:
    """Chunks for a spoken topic, scoped to ONE course when it names one.

    topic_chunks' last leg is a global embedding query, so "quiz me on
    biosensors" can and does pull nearest-neighbour chunks out of another
    course's PDF -- questions about the wrong subject, and a flashcard
    filed under the wrong topic. Lecture notes are already written as
    ``<course-slug>-<YYYY-MM-DD>.md`` (jarvis/lecture.py), so when the
    topic resolves to a course on the Canvas roster -- or, with no token,
    just slugifies to something a file stem carries -- only that course's
    files are read, in reading order, and no embedding call is made.

    Falls back to ``topic_chunks`` when the topic names no course, when
    the course has no indexed file (a PDF saved under its original name is
    not slug-matchable: real coverage is notes-first, global-second), or
    when the named chapter is not in the scoped files."""
    topic = " ".join(str(topic or "").split())
    if not topic:
        return []
    k = max(1, int(k))
    hits = _course_only_chunks(index, cfg, topic, k)
    return hits if hits else topic_chunks(index, topic, k=k)


def _course_only_chunks(index: DocsIndex, cfg, topic: str, k: int) -> list[dict]:
    """The scoped leg of course_chunks; [] when the topic names no course
    this index has a file for. Never raises -- the roster lookup crosses
    the network and the caller must still get its fallback."""
    from jarvis.lecture import resolve_course, slugify
    try:
        names = index.names()
    except Exception:                          # noqa: BLE001 - store boundary
        log.exception("course_chunks: index names unreadable")
        return []
    if not names:
        return []
    files: list[str] = []
    for spoken in _course_candidates(topic):
        # Two readings of each candidate: the roster name Canvas gives
        # back, and the words as said (all resolve_course returns with no
        # token set). Longest candidate first, so "electrode transducers
        # in biosensors" prefers a course actually called that.
        slugs = []
        try:
            slugs.append(slugify(resolve_course(cfg, spoken)))
        except Exception:                      # noqa: BLE001 - network boundary
            log.debug("course_chunks: roster lookup failed", exc_info=True)
        slugs.append(slugify(spoken))
        for slug in slugs:
            files = _slug_files(names, slug)
            if files:
                log.info("course_chunks: %r scoped to %d file(s) by slug %r",
                         topic, len(files), slug)
                break
        if files:
            break
    if not files:
        return []
    head_rx = _heading_rx(topic)
    chunks: list[dict] = []
    for name in sorted(files):
        chunks.extend(index.chunks_of(name))
    if head_rx is None:
        return chunks[:k]
    for i, h in enumerate(chunks):
        if head_rx.search(h["text"]):
            return chunks[i:i + k]
    # The course is right but the chapter is not in its files: a scoped
    # answer about the wrong chapter is worse than the global search.
    return []


def fact_sheet(hits: list[dict]) -> str:
    """One line per chunk, file name first, so the model can cite it."""
    return "\n".join(f"From {h['name']}: {_compact(h['text'])}" for h in hits)


# ---------------------------------------------------------------- tool
def build_index(cfg, embed: Embed = _embed) -> DocsIndex:
    return DocsIndex(paths=doc_paths(cfg), index_dir=index_dir(cfg), embed=embed,
                     max_files=_int_cfg(cfg, "docs.max_files", DEFAULT_MAX_FILES),
                     model=str(_cfg_get(cfg, "docs.embed_model", "") or DEFAULT_EMBED_MODEL),
                     base_url=str(_cfg_get(cfg, "docs.ollama_url", "") or DEFAULT_OLLAMA_URL))


def make_tools(cfg, services, embed: Embed = _embed) -> list[ToolSpec]:
    index = build_index(cfg, embed)
    # Parked for the commander (lecture notes kick a reindex on "end notes"),
    # the way spotify.make_tools parks its tool on services.spotify.
    try:
        if services is not None and getattr(services, "docs_index", None) is None:
            services.docs_index = index
    except Exception:                          # noqa: BLE001 - services may be frozen
        log.debug("services has no room for the docs index", exc_info=True)
    folder = str(doc_paths(cfg)[0]) if doc_paths(cfg) else DEFAULT_PATHS[0]
    # Parked for the commander (quiz mode reads chunks straight from the
    # store), the same idiom as calendar / health: nothing is opened here.
    if services is not None and getattr(services, "docs", None) is None:
        try:
            services.docs = index
        except (AttributeError, TypeError):
            log.debug("services does not accept docs")
    no_docs = NO_DOCS_LINE.format(folder=folder)
    # Nothing is opened here: chromadb and the index dir are touched on the
    # first tool call, so a boot (or test_app_wiring) costs nothing.

    def ask_docs(question: str = "", **_) -> ToolResult:
        question = " ".join(str(question or "").split())
        if not question:
            return ToolResult(text="ask_docs needs a question", ok=False,
                              speak=NO_QUESTION_LINE)
        first = index.start_background()
        try:
            hits = index.query(question)
        except EmbedError as exc:
            log.warning("ask_docs: embed failed: %s", exc)
            return ToolResult(text="document index unreachable", ok=False,
                              speak=INDEX_DOWN_LINE)
        except Exception:                      # noqa: BLE001 - store boundary
            log.exception("ask_docs failed")
            return ToolResult(text="document index unreachable", ok=False,
                              speak=INDEX_DOWN_LINE)
        if not hits:
            if not index.scan():
                return ToolResult(text="no documents indexed", ok=False, speak=no_docs)
            # Files exist but nothing is in the store: either the first
            # background pass is still filling it, or the last pass could
            # not embed (Ollama down) or read anything. Blaming the folder
            # here would send Hunter to check a folder that is fine.
            last = index.last_result or {}
            if last.get("error"):
                # The last pass failed (Ollama down). A retry was just
                # kicked, but say what is wrong rather than "indexing now"
                # every time until it recovers.
                return ToolResult(text="document index unreachable", ok=False,
                                  speak=INDEX_DOWN_LINE)
            if first or index.busy():
                return ToolResult(text="indexing in progress", ok=False,
                                  speak=INDEXING_LINE)
            seen = int(last.get("documents", 0)) + int(last.get("errors", 0))
            if int(last.get("errors", 0)) > 0 and len(index.scan()) <= seen:
                return ToolResult(text="documents present but none readable", ok=False,
                                  speak=UNREADABLE_LINE.format(folder=folder))
            # Files the last pass never saw (dropped in since it ran): index
            # them now rather than tell the user his new syllabus is unreadable.
            index._kicked = False
            index.start_background()
            return ToolResult(text="indexing in progress", ok=False, speak=INDEXING_LINE)
        return ToolResult(text=fact_sheet(hits), max_sentences=4)

    def docs_reindex(**_) -> ToolResult:
        # On a worker thread with a bounded wait: a synchronous pass held the
        # brain's chat thread for the whole walk, and every other utterance
        # got "Still on the last one" until it finished.
        if not index.busy():
            index._kicked = False              # an explicit pass may run again
            index.start_background()
        index.wait(timeout=5.0)
        if index.busy():
            return ToolResult(text="indexing in progress", ok=False, speak=INDEXING_LINE)
        result = index.last_result or {}
        if result.get("error") == "embed":
            return ToolResult(text="embedding model unreachable", ok=False,
                              speak=INDEX_DOWN_LINE)
        if result.get("error"):
            return ToolResult(text=f"index failed: {result['error']}", ok=False,
                              speak=INDEX_DOWN_LINE)
        n = int(result.get("documents", 0))
        if n == 0:
            return ToolResult(text="no documents found", ok=False, speak=no_docs)
        return ToolResult(text=f"indexed {n} documents ({result.get('indexed', 0)} new)",
                          speak=INDEXED_LINE.format(n=n))

    return [
        ToolSpec(
            name="ask_docs",
            description="Search Hunter's own documents (PDF, notes) and quote the matching passages.",
            parameters={"type": "object", "properties": {
                "question": {"type": "string",
                             "description": "What to look up in the documents."}},
                "required": ["question"]},
            handler=ask_docs),
        ToolSpec(
            name="docs_reindex",
            description="Re-scan the documents folder and update the index.",
            parameters={"type": "object", "properties": {}},
            handler=docs_reindex),
    ]
