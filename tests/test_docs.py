"""Documents tool (jarvis/tools/docs.py): a tmp folder with a .txt and a
.md through a deterministic fake embedder into a real chromadb store at
tmp_path -> the right file comes back first; incremental reindex skips
unchanged files and drops deleted ones; the persona lines for an empty
folder, an unreachable Ollama and an empty question; the description
word cap; the ``_embed`` seam against a fake urlopen.

Firewall: tmp index dir, tmp JARVIS_LOG_DIR (conftest), tmp
JARVIS_ASSISTANT_CONFIG; urlopen is stubbed so no test reaches Ollama.
"""
import io
import json
import math
import os
import re
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from jarvis.tools import docs as docs_mod
from jarvis.tools.docs import (DOC_PREFIX, INDEX_DOWN_LINE, INDEXED_LINE, INDEXING_LINE,
                               NO_DOCS_LINE, NO_QUESTION_LINE, QUERY_PREFIX, UNREADABLE_LINE,
                               DocsIndex, EmbedError, chunk_text, extract_text,
                               make_tools)
from jarvis.tools.registry import DESCRIPTION_WORD_CAP, ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")

    def _no_network(*a, **k):                    # no Ollama, ever
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    yield


# ------------------------------------------------------------- fakes
DIM = 32


class FakeEmbed:
    """Deterministic bag-of-words vectors: each token lights one of DIM
    slots, so a question sharing words with a chunk lands nearest it.
    ``down`` makes every call raise EmbedError (Ollama unreachable);
    ``gate`` blocks document embeds until set (a slow first index)."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.down = False
        self.fail_first = 0
        self.gate: threading.Event | None = None

    def __call__(self, texts, model=None, base_url=None, timeout=None):
        assert timeout is None or timeout <= 8.0, "embed timeout over the 8 s cap"
        if self.gate is not None and any(t.startswith(DOC_PREFIX) for t in texts):
            self.gate.wait(10)
        self.calls.append(list(texts))
        if self.down:
            raise EmbedError("URLError: connection refused")
        if self.fail_first > 0:
            self.fail_first -= 1
            raise EmbedError("timed out")
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text):
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z]+", text.lower()):
            v[sum(ord(c) * (i + 1) for i, c in enumerate(tok)) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    @property
    def doc_calls(self):
        return [c for c in self.calls if c and c[0].startswith(DOC_PREFIX)]


SYLLABUS = """# CS 101 Syllabus

## Grading policy
Homework is forty percent, the midterm exam is twenty percent and the
final exam is forty percent. Late homework loses ten percent per day.

## Schedule
The midterm exam is on October 14 and the final exam is on December 9.
"""

RECIPE = """Pancake recipe

Whisk two cups of flour with a tablespoon of sugar and a pinch of salt.
Add two eggs and a cup and a half of milk, then melt butter into the batter.
Cook the pancakes on a hot griddle until bubbles form, then flip.
"""


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    (d / "syllabus.md").write_text(SYLLABUS)
    (d / "recipe.txt").write_text(RECIPE)
    return d


@pytest.fixture(autouse=True)
def _fresh_env_index(tmp_path, monkeypatch):
    """conftest firewalls JARVIS_DOCS_INDEX_DIR for the whole session so a
    real-App test cannot touch ~/.aiws_trainer/docs_index -- but env beats
    cfg by design, so every test here would share ONE chromadb store and
    bleed state. Each test gets its own."""
    monkeypatch.setenv("JARVIS_DOCS_INDEX_DIR", str(tmp_path / "env_index"))


@pytest.fixture
def cfg(tmp_path, folder):
    return {"docs": {"paths": [str(folder)], "index_dir": str(tmp_path / "index"),
                     "max_files": 500}}


@pytest.fixture
def embed():
    return FakeEmbed()


def _tools(cfg, embed):
    specs = make_tools(cfg, None, embed=embed)
    reg = ToolRegistry()
    reg.register_many(specs)
    return reg


# ---------------------------------------------------------- contract
def test_descriptions_within_word_cap(cfg, embed):
    specs = make_tools(cfg, None, embed=embed)
    assert [s.name for s in specs] == ["ask_docs", "docs_reindex"]
    for spec in specs:
        assert spec.description_words() <= DESCRIPTION_WORD_CAP, spec.name
    assert "question" in specs[0].parameters["properties"]


def test_make_tools_is_lazy(cfg, embed, tmp_path):
    """Boot must not open chroma or create the index dir; test_app_wiring
    builds every tool module and must stay off ~/.aiws_trainer."""
    make_tools(cfg, None, embed=embed)
    assert not (tmp_path / "index").exists()
    assert embed.calls == []


# ------------------------------------------------------------ reindex
def test_reindex_counts_documents_and_speaks(cfg, embed):
    reg = _tools(cfg, embed)
    res = reg.call("docs_reindex", {})
    assert isinstance(res, ToolResult) and res.ok
    assert res.speak == INDEXED_LINE.format(n=2)
    assert "2 documents" in res.text
    # Every chunk embedded with the document prefix, none with the query one.
    assert embed.doc_calls and all(t.startswith(DOC_PREFIX)
                                   for c in embed.doc_calls for t in c)


def test_reindex_skips_unchanged_reembeds_changed_drops_deleted(cfg, embed, folder):
    index = docs_mod.build_index(cfg, embed)
    first = index.reindex()
    assert (first["indexed"], first["skipped"], first["removed"]) == (2, 0, 0)
    n_calls = len(embed.calls)

    second = index.reindex()
    assert (second["indexed"], second["skipped"]) == (0, 2)
    assert len(embed.calls) == n_calls, "unchanged files must not be re-embedded"

    # Touch one file's content: mtime AND size change -> only it is redone.
    (folder / "recipe.txt").write_text(RECIPE + "\nServe with maple syrup.\n")
    os.utime(folder / "recipe.txt", (2_000_000_000, 2_000_000_000))
    third = index.reindex()
    assert (third["indexed"], third["skipped"]) == (1, 1)
    assert all("syrup" in t for t in embed.calls[-1]) or \
        any("syrup" in t for c in embed.calls[n_calls:] for t in c)
    # The old chunks of the rewritten file are gone: one entry per path.
    assert set(index.indexed()) == {str(folder / "syllabus.md"),
                                    str(folder / "recipe.txt")}

    (folder / "syllabus.md").unlink()
    fourth = index.reindex()
    assert fourth["removed"] == 1 and fourth["documents"] == 1
    assert set(index.indexed()) == {str(folder / "recipe.txt")}


def test_reindex_survives_one_unreadable_file(cfg, embed, folder):
    (folder / "empty.txt").write_text("")          # no text -> counted as failed
    index = docs_mod.build_index(cfg, embed)
    res = index.reindex()
    assert res["indexed"] == 2 and res["errors"] == 1 and res["documents"] == 2


def test_max_files_cap(cfg, embed, folder):
    for i in range(5):
        (folder / f"note{i}.txt").write_text(f"note number {i} about topic {i}")
    cfg["docs"]["max_files"] = 3
    index = docs_mod.build_index(cfg, embed)
    assert len(index.scan()) == 3
    assert index.reindex()["documents"] == 3


# -------------------------------------------------------------- query
def test_ask_docs_returns_the_right_file_first(cfg, embed):
    reg = _tools(cfg, embed)
    reg.call("docs_reindex", {})
    res = reg.call("ask_docs", {"question": "what is the grading policy for homework"})
    assert res.ok and res.max_sentences == 4 and res.speak is None
    assert res.text.startswith("From syllabus.md: ")
    assert "forty percent" in res.text
    # The question goes out with the query prefix, once.
    assert embed.calls[-1] == [QUERY_PREFIX + "what is the grading policy for homework"]

    res = reg.call("ask_docs", {"question": "how many eggs go in the pancakes"})
    assert res.text.startswith("From recipe.txt: ")


def test_ask_docs_fact_sheet_is_one_line_per_chunk_with_file_names(cfg, embed):
    reg = _tools(cfg, embed)
    reg.call("docs_reindex", {})
    res = reg.call("ask_docs", {"question": "when is the midterm exam"})
    lines = res.text.splitlines()
    assert 1 <= len(lines) <= 5
    assert all(line.startswith("From ") for line in lines)
    assert lines[0].startswith("From syllabus.md:")


def test_ask_docs_arguments_as_json_string(cfg, embed):
    reg = _tools(cfg, embed)
    reg.call("docs_reindex", {})
    res = reg.call("ask_docs", json.dumps({"question": "midterm exam date"}))
    assert res.ok and "October 14" in res.text


def test_ask_docs_empty_question(cfg, embed):
    reg = _tools(cfg, embed)
    res = reg.call("ask_docs", {"question": "   "})
    assert not res.ok and res.speak == NO_QUESTION_LINE
    assert embed.calls == []


# ------------------------------------------------------- persona lines
def test_no_documents_lines_name_the_folder(tmp_path, embed):
    folder = tmp_path / "Jarvis Docs"
    folder.mkdir()
    cfg = {"docs": {"paths": [str(folder)], "index_dir": str(tmp_path / "index")}}
    reg = _tools(cfg, embed)
    res = reg.call("docs_reindex", {})
    assert not res.ok and res.speak == NO_DOCS_LINE.format(folder=str(folder))
    res = reg.call("ask_docs", {"question": "anything at all"})
    assert not res.ok and res.speak == NO_DOCS_LINE.format(folder=str(folder))
    assert str(folder) in res.speak


def test_missing_folder_is_no_documents_not_a_crash(tmp_path, embed):
    cfg = {"docs": {"paths": [str(tmp_path / "nowhere")],
                    "index_dir": str(tmp_path / "index")}}
    reg = _tools(cfg, embed)
    res = reg.call("docs_reindex", {})
    assert not res.ok and res.speak.startswith("I have no documents indexed yet, sir")


def test_ollama_down_lines(cfg, embed):
    embed.down = True
    reg = _tools(cfg, embed)
    res = reg.call("docs_reindex", {})
    assert not res.ok and res.speak == INDEX_DOWN_LINE
    res = reg.call("ask_docs", {"question": "grading policy"})
    assert not res.ok and res.speak == INDEX_DOWN_LINE


def test_unreadable_documents_line(tmp_path, embed):
    folder = tmp_path / "Jarvis Docs"
    folder.mkdir()
    (folder / "blank.txt").write_text("")      # present, but nothing to index
    cfg = {"docs": {"paths": [str(folder)], "index_dir": str(tmp_path / "index")}}
    reg = _tools(cfg, embed)
    reg.call("docs_reindex", {})
    res = reg.call("ask_docs", {"question": "anything"})
    assert not res.ok and res.speak == UNREADABLE_LINE.format(folder=str(folder))


def test_ollama_down_after_indexing_still_speaks_the_down_line(cfg, embed):
    reg = _tools(cfg, embed)
    reg.call("docs_reindex", {})
    embed.down = True
    res = reg.call("ask_docs", {"question": "grading policy"})
    assert not res.ok and res.speak == INDEX_DOWN_LINE


def test_first_use_kicks_a_background_reindex(cfg, embed):
    """Fresh index, files present: the first ask says it is indexing, the
    background pass fills the store, the next ask answers."""
    embed.gate = threading.Event()             # hold the document embeds
    reg = _tools(cfg, embed)
    res = reg.call("ask_docs", {"question": "grading policy"})
    assert not res.ok and res.speak == INDEXING_LINE
    embed.gate.set()
    # Wait for the background thread through the tool itself: docs_reindex
    # joins a running pass and reports it rather than starting another.
    res = reg.call("docs_reindex", {})
    assert res.ok and res.speak == INDEXED_LINE.format(n=2)
    assert len(embed.doc_calls) >= 1
    res = reg.call("ask_docs", {"question": "grading policy"})
    assert res.ok and res.text.startswith("From syllabus.md:")


def test_embed_retries_once_on_timeout(cfg, embed):
    """A cold nomic-embed-text load is ~7.5 s: the first call may time out
    while the model loads, the retry lands on the resident model."""
    embed.fail_first = 1
    index = docs_mod.build_index(cfg, embed)
    res = index.reindex()
    assert res["error"] is None and res["documents"] == 2
    embed.fail_first = 2                       # two in a row: genuinely down
    embed.calls.clear()
    (Path(cfg["docs"]["paths"][0]) / "new.txt").write_text("brand new text here")
    res = index.reindex()
    assert res["error"] == "embed"


# ------------------------------------------------------------ chunking
def test_chunk_text_windows_and_overlap():
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words)
    chunks = chunk_text(text, size=800, overlap=120)
    assert len(chunks) > 3
    assert all(len(c) <= 800 for c in chunks)
    assert all(not c.startswith(" ") for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        # Consecutive chunks share text (the overlap), at a word boundary.
        tail = a.split()[-1]
        assert tail in b.split()
    assert " ".join(chunks).count("word599") >= 1


def test_chunk_text_prefers_paragraph_breaks():
    para = ("Sentence one is here. " * 20).strip()
    text = para + "\n\n" + para + "\n\n" + para
    chunks = chunk_text(text, size=500, overlap=50)
    assert chunks[0].endswith("here.")
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("just a line") == ["just a line"]


# ---------------------------------------------------------- extraction
def test_extract_docx_with_stdlib(tmp_path):
    doc_xml = ("<?xml version='1.0' encoding='UTF-8'?>"
               "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
               "<w:body><w:p><w:r><w:t>Meeting notes for</w:t></w:r><w:r><w:t xml:space='preserve'> Tuesday</w:t></w:r></w:p>"
               "<w:p><w:r><w:t>Budget approved.</w:t></w:r></w:p></w:body></w:document>")
    path = tmp_path / "notes.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", doc_xml)
    assert extract_text(path) == "Meeting notes for Tuesday\nBudget approved."
    (tmp_path / "bad.docx").write_bytes(b"not a zip")
    assert extract_text(tmp_path / "bad.docx") == ""


MINIMAL_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 60 >> stream
BT /F1 18 Tf 20 100 Td (Syllabus grading policy) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
trailer << /Root 1 0 R >>
%%EOF
"""


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")
def test_extract_pdf_with_pdftotext(tmp_path):
    path = tmp_path / "syllabus.pdf"
    path.write_bytes(MINIMAL_PDF)
    assert "Syllabus grading policy" in extract_text(path)


def test_extract_unsupported_and_unreadable(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\x00\x01")
    assert extract_text(tmp_path / "x.bin") == ""
    assert extract_text(tmp_path / "missing.txt") == ""


def test_scan_skips_hidden_and_unsupported(cfg, embed, folder):
    (folder / ".hidden.md").write_text("secret")
    (folder / "photo.png").write_bytes(b"\x89PNG")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("deep text")
    names = sorted(p.name for p in docs_mod.build_index(cfg, embed).scan())
    assert names == ["deep.txt", "recipe.txt", "syllabus.md"]


# ------------------------------------------------------------ the seam
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_embed_seam_posts_to_api_embed(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data)
        return _Resp(json.dumps({"embeddings": [[0.1, 0.2], [0.3, 0.4]]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    vecs = docs_mod._embed(["a", "b"], "nomic-embed-text", "http://localhost:11434/")
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert seen["url"] == "http://localhost:11434/api/embed"
    assert seen["timeout"] == docs_mod.EMBED_TIMEOUT <= 8.0
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_embed_seam_maps_failures_to_embed_error(monkeypatch):
    def refused(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(EmbedError):
        docs_mod._embed(["a"])

    def short(req, timeout=None):              # one vector for two texts
        return _Resp(json.dumps({"embeddings": [[0.1]]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", short)
    with pytest.raises(EmbedError):
        docs_mod._embed(["a", "b"])


# ------------------------------------------------------------- config
def test_config_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_DOCS_INDEX_DIR", raising=False)
    assert docs_mod.doc_paths({}) == [Path("~/Documents/Jarvis Docs").expanduser().resolve()]
    assert docs_mod.index_dir({}) == Path("~/.aiws_trainer/docs_index").expanduser()
    monkeypatch.setenv("JARVIS_DOCS_INDEX_DIR", str(tmp_path / "env_index"))
    assert docs_mod.index_dir({}) == tmp_path / "env_index"
    # env > config: AssistantConfig always carries docs.index_dir (DEFAULTS),
    # so config-first had made the env override dead in the real app.
    assert docs_mod.index_dir({"docs": {"index_dir": str(tmp_path / "cfg")}}) == tmp_path / "env_index"
    from jarvis.assistant_config import AssistantConfig
    assert docs_mod.index_dir(AssistantConfig({})) == tmp_path / "env_index"
    monkeypatch.delenv("JARVIS_DOCS_INDEX_DIR")
    assert docs_mod.index_dir({"docs": {"index_dir": str(tmp_path / "cfg")}}) == tmp_path / "cfg"
    assert docs_mod.doc_paths({"docs": {"paths": "~/one"}}) == [Path("~/one").expanduser().resolve()]
    idx = docs_mod.build_index({"docs": {"max_files": "7", "embed_model": "x",
                                          "ollama_url": "http://h:1"}})
    assert (idx.max_files, idx.model, idx.base_url) == (7, "x", "http://h:1")
    assert isinstance(idx, DocsIndex)
