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

``CodeIndex`` is the same machinery pointed at his own repositories
(``code.paths``, defaulting to ``claude.allowed_dirs``): .py/.md/.sh cut at
def/class/heading boundaries with the line numbers kept, in its OWN chroma
collection under ``PATHS.MEMORY_DIR`` so quiz mode and the lecture flows --
which read chunks straight out of the documents store -- never see code.
``ask_code`` answers "where does the mic arbiter live" in ~2 s with a
file:line citation, where the same question sent to `claude -p` costs ~12 s.

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

from jarvis.config import PATHS
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

# ------------------------------------------------------ code index (15)
CODE_COLLECTION = "jarvis_code"
CODE_SUFFIXES = (".py", ".md", ".sh")
# Code lines are short: 800 chars is barely fifteen lines, not enough to hold
# a function AND its docstring, so a hit would cite a body with no signature.
CODE_CHUNK_CHARS = 1400
CODE_CHUNK_LINES = 120             # one enormous function is still split
CODE_TOP_K = 6
CODE_SHEET_CHARS = 900             # per-chunk cap in the fact sheet
CODE_MAX_FILES = 3000              # ~31k lines of Jarvis is ~90 files
CODE_MAX_BYTES = 300_000           # a generated/minified file is not source
# A repo he edits all day goes stale inside one app run; the pass is
# incremental, so re-walking an unchanged tree is only stat() calls.
CODE_REFRESH_S = 900.0
# Directories that are never HIS code. `repo/` is the load-bearing one: it
# holds 140 MB of StyleTTS2 weights in this very repository, and walking it
# would spend minutes indexing binary noise. The rest are caches, vendored
# dependencies and build output -- indexing site-packages would drown his
# own 31k lines in a million lines of other people's.
CODE_SKIP_DIRS = frozenset({
    "repo", ".git", "node_modules", "__pycache__", "site-packages",
    ".venv", "venv", "env", "envs", ".tox", ".nox", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", ".cache", "dist", "build", "target",
    "htmlcov", ".eggs", ".idea", ".vscode", "coverage", "wandb",
    "checkpoints", "weights", "runs", "outputs", "data", "datasets"})

NO_CODE_LINE = ("I've no code indexed yet, sir; the folders come from Claude's "
                "allowed directories.")
CODE_DOWN_LINE = "My code index isn't answering, sir."
CODE_INDEXING_LINE = "I'm reading through your code now, sir; ask me again in a moment."
NO_CODE_QUESTION_LINE = "What would you like me to find in your code, sir?"

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


def code_paths(cfg) -> list[Path]:
    """The repositories to index. ``code.paths`` when set, otherwise the
    folders Hunter already cleared for Claude (``claude.allowed_dirs``):
    those ARE his repos, and a second list to keep in sync would go stale."""
    raw = _cfg_get(cfg, "code.paths", None)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raw = _cfg_get(cfg, "claude.allowed_dirs", None) or []
        if isinstance(raw, str):
            raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out, seen = [], set()
    for entry in raw:
        if not str(entry).strip():
            continue
        path = Path(os.path.expanduser(str(entry))).resolve()
        if str(path) not in seen:
            seen.add(str(path))
            out.append(path)
    return out


def code_index_dir(cfg) -> Path:
    """env > config > MEMORY_DIR/code_index. MEMORY_DIR rather than beside
    the documents index because the test suite already firewalls it
    (conftest sets JARVIS_MEMORY_DIR), so no test can write into the real
    store just by building the tool."""
    raw = os.environ.get("JARVIS_CODE_INDEX_DIR") or \
        _cfg_get(cfg, "code.index_dir", "") or ""
    if raw:
        return Path(os.path.expanduser(str(raw)))
    return PATHS.MEMORY_DIR / "code_index"


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


# Suffixes whose bytes ARE the text. Source files are here rather than in a
# branch of their own because a .py needs no extraction at all -- only the
# chunker downstream treats them differently (chunk_code vs chunk_text).
PLAIN_SUFFIXES = (".txt", ".md", ".py", ".sh", ".bash")


def extract_text(path: Path) -> str:
    """Plain text for one supported file ("" when unreadable/unsupported)."""
    suffix = path.suffix.lower()
    try:
        if suffix in PLAIN_SUFFIXES:
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


# ------------------------------------------------- code chunking (15)
# A chunk should start where a reader would start: at a def, a class, a
# decorator above one, a markdown heading or a shell function. Splitting
# every N characters instead puts the signature in one chunk and the body
# in the next, and "where does X live" then cites the half without the name.
_PY_BOUNDARY_RX = re.compile(r"^(?:@\w|(?:async\s+)?def\s|class\s|if __name__)")
_MD_BOUNDARY_RX = re.compile(r"^#{1,6}\s")
_SH_BOUNDARY_RX = re.compile(r"^(?:function\s+[\w.-]+|[\w.-]+\s*\(\)\s*\{)")
_CODE_BOUNDARIES = {".py": _PY_BOUNDARY_RX, ".md": _MD_BOUNDARY_RX,
                    ".sh": _SH_BOUNDARY_RX, ".bash": _SH_BOUNDARY_RX}


def chunk_code(text: str, suffix: str = ".py", size: int = CODE_CHUNK_CHARS,
               max_lines: int = CODE_CHUNK_LINES) -> list[dict]:
    """[{text, start, end}] with 1-based, inclusive line numbers.

    Units are cut at the language's own boundaries and then PACKED up to
    ``size`` so a file of eight-line helpers is not eight chunks (eight
    embeddings, eight near-identical hits). A unit longer than
    ``max_lines`` is split on line boundaries.

    Deliberately no overlap, unlike ``chunk_text``: prose needs it because
    a sentence can straddle a cut, but a function cannot, and an overlap
    would make the same lines answerable under two different citations."""
    lines = (text or "").splitlines()
    if not any(ln.strip() for ln in lines):
        return []
    rx = _CODE_BOUNDARIES.get(str(suffix or "").lower())
    bounds = [0]
    if rx is not None:
        for i, line in enumerate(lines):
            if i and rx.match(line):
                bounds.append(i)
    bounds.append(len(lines))

    units: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        while b - a > max_lines:
            units.append((a, a + max_lines))
            a += max_lines
        if b > a:
            units.append((a, b))

    out: list[dict] = []
    cur_a = cur_b = None
    cur_chars = 0
    for a, b in units:
        n = sum(len(lines[i]) + 1 for i in range(a, b))
        if cur_a is None:
            cur_a, cur_b, cur_chars = a, b, n
        elif cur_chars + n <= size and (b - cur_a) <= max_lines:
            cur_b, cur_chars = b, cur_chars + n
        else:
            out.append((cur_a, cur_b))
            cur_a, cur_b, cur_chars = a, b, n
    if cur_a is not None:
        out.append((cur_a, cur_b))

    pieces = []
    for a, b in out:
        # Trim the blank lines a unit collects between definitions: the
        # citation is read aloud and typed into an editor, and "lines
        # 120-172" pointing at two blank lines is a citation that lies.
        while a < b and not lines[a].strip():
            a += 1
        while b > a and not lines[b - 1].strip():
            b -= 1
        if b > a:
            pieces.append({"text": "\n".join(lines[a:b]),
                           "start": a + 1, "end": b})
    return pieces


def short_path(path) -> str:
    """``~``-relative where possible: a citation is read aloud and spelled
    into a search box, and /home/hunterp/ is four wasted syllables."""
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if text.startswith(home + "/") else text


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

    # Subclasses point the SAME machinery at a different corpus: their own
    # chroma collection (never a shared one -- quiz mode and the lecture
    # flows read chunks straight out of this store by file name, and a
    # module.py landing in a flashcard round is a bug), their own suffixes
    # and their own chunker.
    collection_name = COLLECTION
    suffixes: tuple = SUFFIXES

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
        self._last_pass = 0.0                  # monotonic clock of the last finish

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
                    self.collection_name, embedding_function=None,
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
                if p.is_file() and p.suffix.lower() in self.suffixes \
                        and not p.name.startswith("."):
                    found.append(p)
        if len(found) > self.max_files:
            log.warning("%s: %d files found, indexing the first %d",
                        self.collection_name, len(found), self.max_files)
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
            self._last_pass = time.monotonic()
            if result.get("error"):
                self._kicked = False      # let the next ask kick a fresh pass
            return result

    def _pieces(self, path: Path, text: str) -> list[tuple[str, dict]]:
        """(chunk text, extra metadata) for one file. Prose carries no
        extra metadata; CodeIndex overrides this to keep line numbers."""
        return [(c, {}) for c in chunk_text(text)]

    def _index_file(self, col, path: Path, key: tuple[float, int]) -> bool:
        text = extract_text(path)
        pieces = self._pieces(path, text)
        if not pieces:
            log.warning("docs: no text in %s", path.name)
            return False
        chunks = [c for c, _ in pieces]
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
                        "mtime": key[0], "size": key[1], **extra}
                       for i, (_, extra) in enumerate(pieces)])
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

    def refresh_if_stale(self, max_age_s: float) -> bool:
        """A background pass when the last one finished over ``max_age_s``
        ago. ``start_background`` latches once per process, which is right
        for a documents folder Hunter rarely touches and wrong for a repo
        he edits all day: without this an app up for a week would answer
        code questions from Monday's tree. The pass itself is incremental
        (mtime+size), so a re-run over an unchanged repo is only stats."""
        if self._last_pass and (time.monotonic() - self._last_pass) < max_age_s:
            return False
        return self.kick()

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
            hit = {"name": str(meta.get("name") or Path(str(meta.get("path", ""))).name),
                   "path": str(meta.get("path", "")), "text": doc or "",
                   "chunk": int(meta.get("chunk", 0)),
                   "score": round(1.0 - float(dist), 4)}
            # CodeIndex stores these; a prose chunk has no line span.
            for extra in ("rel", "start", "end"):
                if meta.get(extra) is not None:
                    hit[extra] = meta[extra]
            out.append(hit)
        return out


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
    heading = _CHAPTER_RX.search(topic)
    head_rx = None
    if heading:
        num = _NUMBER_WORDS.get(heading.group(2).lower(), heading.group(2))
        words = {v: k_ for k_, v in _NUMBER_WORDS.items()}.get(num, "")
        head_rx = re.compile(r"\b%s\s*(?:%s%s)\b" % (
            r"(?:chapter|chap\.?|section|unit|lecture|week|module|part|lab)",
            re.escape(num), f"|{words}" if words else ""), re.I)

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


def fact_sheet(hits: list[dict]) -> str:
    """One line per chunk, file name first, so the model can cite it."""
    return "\n".join(f"From {h['name']}: {_compact(h['text'])}" for h in hits)


# ------------------------------------------------------- code index (15)
class CodeIndex(DocsIndex):
    """His own repositories in their own chroma collection.

    Everything that made DocsIndex work -- the mtime/size manifest living in
    chroma's metadata, the background first pass, the single ``embed`` seam,
    replace-never-append on reindex -- is inherited unchanged. What differs
    is the corpus (source files, not PDFs), the chunker (line-aware, cut at
    def/class/heading) and the collection, which MUST be separate: quiz mode
    and the lecture flows read chunks straight out of the documents store by
    file name, and recorder.py turning up in a flashcard round is a bug.
    """

    collection_name = CODE_COLLECTION
    suffixes = CODE_SUFFIXES
    skip_dirs = CODE_SKIP_DIRS
    max_bytes = CODE_MAX_BYTES

    def scan(self) -> list[Path]:
        """os.walk rather than rglob: rglob cannot PRUNE, so it would
        descend into repo/ (140 MB of model weights) and .git before
        filtering by suffix -- minutes of stat() calls per pass."""
        found: list[Path] = []
        for root in self.paths:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames
                                     if d not in self.skip_dirs
                                     and not d.startswith("."))
                for name in sorted(filenames):
                    if name.startswith(".") or \
                            Path(name).suffix.lower() not in self.suffixes:
                        continue
                    path = Path(dirpath) / name
                    try:
                        if path.stat().st_size > self.max_bytes:
                            continue          # generated, vendored or a dump
                    except OSError:
                        continue
                    found.append(path)
        found.sort()
        if len(found) > self.max_files:
            log.warning("code: %d files found, indexing the first %d",
                        len(found), self.max_files)
        return found[:self.max_files]

    def rel(self, path: Path) -> str:
        """"Jarvis/jarvis/recorder.py" -- relative to the repo's PARENT, so
        the citation names the repo as well as the file. Two repos with a
        jarvis/app.py would otherwise be indistinguishable when spoken."""
        for root in self.paths:
            try:
                return str(Path(path).relative_to(root.parent))
            except ValueError:
                continue
        return short_path(path)

    def _pieces(self, path: Path, text: str) -> list[tuple[str, dict]]:
        rel = self.rel(path)
        out = []
        for piece in chunk_code(text, Path(path).suffix.lower()):
            # The path rides INSIDE the chunk text as well as in the
            # metadata: nomic embeds the text only, so "where does the mic
            # arbiter live" has to be able to match on "recorder.py" too.
            body = f"{rel}:{piece['start']}\n{piece['text']}"
            out.append((body, {"rel": rel, "start": piece["start"],
                               "end": piece["end"]}))
        return out


def code_ref(hit: dict) -> str:
    """"Jarvis/jarvis/recorder.py:120-168" for one hit."""
    rel = str(hit.get("rel") or "") or short_path(hit.get("path", ""))
    start, end = hit.get("start"), hit.get("end")
    if start and end:
        return f"{rel}:{start}-{end}" if int(end) != int(start) else f"{rel}:{start}"
    return rel


def code_sheet(hits: list[dict]) -> str:
    """One line per chunk, path:lines first, so the model cites file AND
    line -- the whole point of the code index over `claude -p`."""
    return "\n".join(f"From {code_ref(h)}: {_compact(h['text'], CODE_SHEET_CHARS)}"
                      for h in hits)


# ---------------------------------------------------------------- tool
def build_index(cfg, embed: Embed = _embed) -> DocsIndex:
    return DocsIndex(paths=doc_paths(cfg), index_dir=index_dir(cfg), embed=embed,
                     max_files=_int_cfg(cfg, "docs.max_files", DEFAULT_MAX_FILES),
                     model=str(_cfg_get(cfg, "docs.embed_model", "") or DEFAULT_EMBED_MODEL),
                     base_url=str(_cfg_get(cfg, "docs.ollama_url", "") or DEFAULT_OLLAMA_URL))


def build_code_index(cfg, embed: Embed = _embed) -> CodeIndex:
    return CodeIndex(paths=code_paths(cfg), index_dir=code_index_dir(cfg),
                     embed=embed,
                     max_files=_int_cfg(cfg, "code.max_files", CODE_MAX_FILES),
                     model=str(_cfg_get(cfg, "docs.embed_model", "") or DEFAULT_EMBED_MODEL),
                     base_url=str(_cfg_get(cfg, "docs.ollama_url", "") or DEFAULT_OLLAMA_URL))


def make_tools(cfg, services, embed: Embed = _embed) -> list[ToolSpec]:
    index = build_index(cfg, embed)
    code = build_code_index(cfg, embed)
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
    if services is not None and getattr(services, "code_index", None) is None:
        try:
            services.code_index = code
        except (AttributeError, TypeError):
            log.debug("services does not accept the code index")
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

    def ask_code(question: str = "", **_) -> ToolResult:
        question = " ".join(str(question or "").split())
        if not question:
            return ToolResult(text="ask_code needs a question", ok=False,
                              speak=NO_CODE_QUESTION_LINE)
        if not code.paths:
            # No allowed_dirs and no code.paths: there is nothing to index,
            # and "indexing now" would be a lie he never stops telling.
            return ToolResult(text="no code folders configured", ok=False,
                              speak=NO_CODE_LINE)
        first = code.start_background()
        if not first:
            # He edits these repos all day; the once-per-process latch that
            # suits a documents folder would answer from a stale tree.
            code.refresh_if_stale(CODE_REFRESH_S)
        try:
            hits = code.query(question, k=CODE_TOP_K)
        except EmbedError as exc:
            log.warning("ask_code: embed failed: %s", exc)
            return ToolResult(text="code index unreachable", ok=False,
                              speak=CODE_DOWN_LINE)
        except Exception:                      # noqa: BLE001 - store boundary
            log.exception("ask_code failed")
            return ToolResult(text="code index unreachable", ok=False,
                              speak=CODE_DOWN_LINE)
        if not hits:
            if not code.scan():
                return ToolResult(text="no code files found", ok=False,
                                  speak=NO_CODE_LINE)
            if (code.last_result or {}).get("error"):
                return ToolResult(text="code index unreachable", ok=False,
                                  speak=CODE_DOWN_LINE)
            if first or code.busy():
                return ToolResult(text="indexing in progress", ok=False,
                                  speak=CODE_INDEXING_LINE)
            return ToolResult(text="nothing in the code matches that", ok=False)
        return ToolResult(text=code_sheet(hits), max_sentences=3)

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
        ToolSpec(
            name="ask_code",
            description="Find something in Hunter's own repositories and cite the file and lines.",
            parameters={"type": "object", "properties": {
                "question": {"type": "string",
                             "description": "What to find in his code."}},
                "required": ["question"]},
            handler=ask_code),
    ]
