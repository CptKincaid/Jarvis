"""Semantic long-term memory (jarvis/memory.py SemanticFacts): facts.json
stays the source of truth, every fact is written through to a real
chromadb collection at tmp_path through a deterministic fake embedder,
recall finds a fact by meaning (shared words, no substring), the time
filter, forgetting, the migration of an existing facts.json, the
substring-only fallback with the embedder down (and its back-off), and
format_for_context(text) injecting the RELEVANT facts instead of the last
five. The docs ``_embed`` seam gains keep_alive only when asked for it.

Firewall: tmp memory/index dirs; urlopen is stubbed so no test reaches
Ollama; the FakeEmbed never touches the network.
"""
import json
import math
import re
import threading
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import jarvis.memory as memory_mod
from jarvis.memory import (CONTEXT_TOP_K, SCORE_FLOOR_CONTEXT, SCORE_FLOOR_RECALL,
                           JarvisMemory, SemanticFacts, parse_since)
from jarvis.tools.docs import DOC_PREFIX, QUERY_PREFIX, EmbedError

DIM = 64
# Function words carry no meaning for a bag-of-words fake; a real embedder
# discounts them on its own.
# "your" joined "my" on 2026-08-31, when facts began being filed in the
# second person (memory.to_second_person). It is a function word like the
# rest -- and in a 64-bucket bag-of-words fake it collides with "thesis"
# and "patel", which made the dentist fact win a query about the thesis.
_STOP = frozenset("search query document is my your the a an what did i say "
                  "about who on do does for to with of in and was were tell "
                  "me you this that it at are am be".split())


class FakeEmbed:
    """Bag-of-words vectors (as tests/test_docs.py): a query sharing words
    with a fact lands nearest it, so "dentist" finds the dentist fact even
    though the query is no substring of it."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.down = False

    def __call__(self, texts, model=None, base_url=None, timeout=None):
        self.calls.append(list(texts))
        if self.down:
            raise EmbedError("URLError: connection refused")
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text):
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z]+", text.lower()):
            if tok in _STOP:
                continue
            v[sum(ord(c) * (i + 1) for i, c in enumerate(tok)) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


@pytest.fixture
def embed(monkeypatch):
    # The floors are measured on nomic-embed-text (see memory.py); the
    # bag-of-words fake scores a one-word overlap around 0.3-0.5, so the
    # mechanism is exercised with floors that fit its geometry.
    monkeypatch.setattr(memory_mod, "SCORE_FLOOR_RECALL", 0.25)
    monkeypatch.setattr(memory_mod, "SCORE_FLOOR_CONTEXT", 0.3)
    return FakeEmbed()


def test_shipped_floors_sit_above_the_measured_noise():
    # 2026-08-30 probe (scratchpad embed_check.py): unrelated pairs peaked
    # at 0.58, matching pairs started at 0.60.
    assert 0.58 < SCORE_FLOOR_CONTEXT <= 0.60
    assert SCORE_FLOOR_RECALL < SCORE_FLOOR_CONTEXT


@pytest.fixture
def mem(tmp_path, embed):
    return JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                        embed=embed)


# ------------------------------------------------------------ write-through
def test_remember_writes_facts_json_and_the_index(mem, tmp_path, embed):
    # Filed in the second person since 2026-08-31 (memory.to_second_person):
    # a fact is rendered straight into the model's prompt, where "my" would
    # be the assistant's.
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    filed = "your dentist is Dr Patel on Elm Street"
    facts = json.loads((tmp_path / "mem" / "facts.json").read_text())
    assert facts["dentist"]["value"] == filed
    assert mem._index.count() == 1
    assert (tmp_path / "mem" / "facts_index").is_dir()
    # embedded with the document prefix, queried with the query prefix
    assert embed.calls[-1] == [DOC_PREFIX + filed]
    mem.recall("who is my dentist")
    assert embed.calls[-1][0].startswith(QUERY_PREFIX)


def test_recall_finds_a_fact_by_meaning_not_substring(mem):
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    mem.remember("thesis", "the thesis draft is due on the fifteenth")
    mem.remember("batch", "training uses batch size sixteen")
    hits = mem.recall("who is my dentist")          # not a substring of the value
    assert hits and hits[0]["key"] == "dentist"
    assert 0 < hits[0]["score"] <= 1.0
    hits = mem.recall("what did I say about the thesis")
    assert [h["key"] for h in hits][:1] == ["thesis"]
    # a substring hit still comes first, as before
    assert mem.recall("proj") == []
    assert mem.recall("batch size")[0]["key"] == "batch"
    assert mem.recall("zzz-not-there") == []


def test_recall_since_filters_by_when_the_fact_was_stored(mem, tmp_path):
    mem.remember("old", "the thesis outline was approved")
    facts = json.loads((tmp_path / "mem" / "facts.json").read_text())
    facts["old"]["time"] = (datetime.now() - timedelta(days=30)).isoformat()
    (tmp_path / "mem" / "facts.json").write_text(json.dumps(facts))
    mem2 = JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                        embed=FakeEmbed())
    mem2._index.upsert("old", facts["old"]["value"], facts["old"]["time"])
    mem2.remember("new", "the thesis draft is due on the fifteenth")
    keys = {h["key"] for h in mem2.recall("thesis")}
    assert keys == {"old", "new"}
    recent = {h["key"] for h in mem2.recall("thesis", since=timedelta(days=7))}
    assert recent == {"new"}


def test_forget_removes_from_both_stores(mem):
    mem.remember("k", "the dentist appointment is on Monday")
    mem.forget("k")
    assert mem.recall("dentist appointment") == []
    assert mem._index.count() == 0


def test_existing_facts_json_is_migrated_on_first_use(tmp_path):
    d = tmp_path / "mem"
    d.mkdir()
    (d / "facts.json").write_text(json.dumps({
        "training_config": {"value": "training uses batch size sixteen",
                            "time": "2026-04-01T10:00:00"},
        "jarvis_state": {"value": "the Bettany voice is the XTTS one",
                         "time": "2026-04-02T10:00:00"}}))
    embed = FakeEmbed()
    mem = JarvisMemory(memory_dir=d, legacy_dir=tmp_path / "legacy", embed=embed)
    assert embed.calls == []                    # nothing at construction (boot path)
    hits = mem.recall("what batch size for training")
    assert hits and hits[0]["key"] == "training_config"
    assert mem._index.count() == 2
    # the migration ran once: a second recall does not re-embed the facts
    n = len(embed.calls)
    mem.recall("bettany voice")
    assert len(embed.calls) == n + 1


# ---------------------------------------------------------------- fallback
def test_embedder_down_falls_back_to_substring_and_backs_off(mem, embed, monkeypatch):
    mem.remember("dentist", "my dentist is Dr Patel")
    embed.down = True
    mem.remember("thesis", "the thesis draft is due Friday")      # facts.json still written
    assert mem.get_all_facts()["thesis"]["value"] == "the thesis draft is due Friday"
    assert mem.recall("draft")[0]["key"] == "thesis"               # substring path
    # 2026-09-04: the stemmed fallback (memory.stem_match) finds "dentist"
    # lexically now; the point of this row is that no MEANING search runs
    # while the embedder is down -- the score says which path answered.
    assert mem.recall("who is my dentist")[0]["score"] == 0.9      # stem, not meaning
    calls = len(embed.calls)
    mem.recall("who is my dentist")
    assert len(embed.calls) == calls, "a second query inside the back-off hit the embedder"
    monkeypatch.setattr(memory_mod, "EMBED_BACKOFF_S", 0.0)
    mem._index._down_until = 0.0
    embed.down = False
    assert mem.recall("who is my dentist")[0]["key"] == "dentist"
    assert mem.get_all_facts()["thesis"]                            # caught up


def test_chromadb_unavailable_is_substring_only(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _no_chroma(name, *a, **k):
        if name.startswith("chromadb"):
            raise ImportError("no chromadb")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _no_chroma)
    mem = JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                       embed=FakeEmbed())
    mem.remember("dentist", "my dentist is Dr Patel")
    assert mem.semantic_available is False
    assert mem.recall("patel")[0]["key"] == "dentist"
    # lexical only: the substring (1.0) and, since 2026-09-04, the stemmed
    # word overlap (0.9) -- never a meaning search
    assert mem.recall("who is my dentist")[0]["score"] == 0.9
    assert mem.recall("play some jazz") == []
    assert "Known facts (1)" in mem.format_for_context("who is my dentist")


def test_semantic_off_switch(tmp_path):
    mem = JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                       semantic=False)
    mem.remember("a", "b")
    assert mem._index is None and mem.recall("b")[0]["key"] == "a"


# ----------------------------------------------------- format_for_context
def test_context_carries_the_relevant_facts_not_the_last_five(mem):
    for i in range(6):
        mem.remember(f"filler{i}", f"filler fact number {i} about nothing")
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    for i in range(6, 12):
        mem.remember(f"filler{i}", f"filler fact number {i} about nothing")
    text = mem.format_for_context("who is my dentist")
    assert "Dr Patel" in text
    assert "bear on this" in text
    assert text.count("filler fact") == 0
    # no utterance (the legacy think() path): the last five, as before
    plain = mem.format_for_context()
    assert "Known facts (13)" in plain and "filler fact number 11" in plain
    assert "Dr Patel" not in plain
    # nothing close enough: no fact block at all, rather than five random ones
    assert "fact" not in mem.format_for_context("play some jazz").lower()


def test_context_top_k_and_floor(mem):
    for i in range(5):
        mem.remember(f"d{i}", f"dentist note {i}: dentist visit dentist")
    facts = mem.relevant_facts("dentist")
    assert 0 < len(facts) <= CONTEXT_TOP_K
    assert all(f["score"] >= memory_mod.SCORE_FLOOR_CONTEXT for f in facts)
    assert mem.relevant_facts("dentist", floor=1.01) == []


# -------------------------------------------------------------- since words
@pytest.mark.parametrize("query,rest,days", [
    ("the thesis last week", "the thesis", 7),
    ("what I said about the move yesterday", "what I said about the move", None),
    ("the dentist in the last three days", "the dentist", 3),
    ("the dentist", "the dentist", 0),
])
def test_parse_since(query, rest, days):
    now = datetime(2026, 8, 30, 15, 0)
    cleaned, since = parse_since(query, now)
    assert cleaned == rest
    if days is None:
        assert since == datetime(2026, 8, 29, 0, 0)
    elif days == 0:
        assert since is None
    else:
        assert since == now - timedelta(days=days)


# ------------------------------------------------------------- embed seam
def test_docs_embed_sends_keep_alive_only_when_asked(monkeypatch):
    import io
    from jarvis.tools import docs as docs_mod
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _Resp(json.dumps({"embeddings": [[0.1, 0.2]]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    docs_mod._embed(["a"], "nomic-embed-text", "http://localhost:11434")
    assert "keep_alive" not in seen["body"]
    docs_mod._embed(["a"], "nomic-embed-text", "http://localhost:11434", 8.0,
                    keep_alive=-1)
    assert seen["body"]["keep_alive"] == -1


def test_default_embedder_pins_keep_alive(monkeypatch):
    from jarvis.tools import docs as docs_mod
    got = {}

    def fake_embed(texts, model, base_url, timeout, keep_alive=None):
        got["keep_alive"] = keep_alive
        return [[1.0] * DIM for _ in texts]
    monkeypatch.setattr(docs_mod, "_embed", fake_embed)
    facts = SemanticFacts(index_dir=Path("/nonexistent"))
    facts._broken = True                        # no store needed for the seam
    facts._vectors(["x"])
    assert got["keep_alive"] == memory_mod.EMBED_KEEP_ALIVE


def test_index_is_thread_safe_for_concurrent_writes(mem):
    errors = []

    def _w(i):
        try:
            mem.remember(f"k{i}", f"fact {i} about topic {i}")
        except Exception as exc:                # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=_w, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not errors and mem._index.count() == 8


# ------------------------------------------------- the one-slot rule (4.3)
# Ollama here runs OLLAMA_MAX_LOADED_MODELS=1, so an /api/embed on the reply
# path does not add the embedder, it EVICTS the chat model. Measured live
# 2026-08-31: the embed 2.5 s, the 26B's forced reload on the next request
# 6.9-7.2 s -- 9.6 s per turn, on every turn, for a store whose facts all
# fit in the prompt anyway. These tests hold the rule in place.

def _reply_path_embeds(mem, embed, text="what's on my calendar today"):
    """Embed calls made by ONE reply turn (the write-through calls from
    building the fixture are not the reply path's fault)."""
    before = len(embed.calls)
    rendered = mem.format_for_context(text)
    return embed.calls[before:], rendered


def test_reply_path_makes_no_embed_call_while_the_store_fits(mem, embed):
    # Hunter's live store is four one-line facts.
    mem.remember("training_config", "batch size 16, learning rate 0.001")
    mem.remember("project", "Jarvis V2 refactor in progress")
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    mem.remember("thesis", "the thesis defence is in November")
    assert mem.facts_fit_context()
    calls, rendered = _reply_path_embeds(mem, embed)
    # THE regression: any call here is a 9.6 s model swap on the turn.
    assert calls == []
    # ...and the model is handed MORE than ranking gave it, not less: all
    # four, where the 0.60 floor cleared none of them on the live store.
    for value in ("batch size 16", "Jarvis V2 refactor", "Dr Patel",
                  "thesis defence"):
        assert value in rendered


def test_a_store_too_big_to_fit_still_ranks_semantically(mem, embed):
    for i in range(memory_mod.CONTEXT_ALL_FACTS_MAX + 1):
        mem.remember(f"filler{i}", f"filler fact number {i} about nothing")
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    assert not mem.facts_fit_context()
    calls, rendered = _reply_path_embeds(mem, embed, "who is my dentist")
    assert calls == [[QUERY_PREFIX + "who is my dentist"]]
    assert "Dr Patel" in rendered and "filler fact" not in rendered


def test_long_facts_break_the_fit_even_when_few(mem):
    mem.remember("essay", "x" * (memory_mod.CONTEXT_ALL_FACTS_CHARS + 1))
    assert not mem.facts_fit_context()


def test_repeat_questions_reuse_the_query_vector(mem, embed):
    for i in range(memory_mod.CONTEXT_ALL_FACTS_MAX + 1):
        mem.remember(f"filler{i}", f"filler fact number {i} about nothing")
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    first, _ = _reply_path_embeds(mem, embed, "who is my dentist")
    second, rendered = _reply_path_embeds(mem, embed, "who is my dentist")
    assert len(first) == 1          # the cold query pays one swap
    assert second == []             # the repeat pays none
    assert "Dr Patel" in rendered   # ...and still answers


def test_query_cache_is_capped(mem, embed):
    for i in range(memory_mod.CONTEXT_ALL_FACTS_MAX + 1):
        mem.remember(f"filler{i}", f"filler fact number {i} about nothing")
    for i in range(memory_mod.QUERY_CACHE_MAX + 5):
        mem.relevant_facts(f"distinct question number {i}")
    assert len(mem._index._qcache) <= memory_mod.QUERY_CACHE_MAX


def test_no_embed_while_the_chat_model_is_lent_out(tmp_path, embed):
    # The nightly haymaker digest (qwen2.5:32b, 04:09) is given the GPU by
    # brain.release(); with one slot, an embed would evict IT mid-run.
    lent = {"now": True}
    mem = JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                       embed=embed, gate=lambda: not lent["now"])
    for i in range(memory_mod.CONTEXT_ALL_FACTS_MAX + 1):
        mem._facts[f"filler{i}"] = {"value": f"filler fact {i}", "time": "2026-08-31T00:00:00"}
    before = len(embed.calls)
    assert mem.relevant_facts("who is my dentist") == []
    assert embed.calls[before:] == []
    # A lend ends by itself: no 60 s back-off is armed, the next call goes.
    lent["now"] = False
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    assert len(embed.calls) > before


def test_warm_index_does_not_load_the_embedder_for_a_store_that_fits(mem, embed):
    mem.remember("dentist", "my dentist is Dr Patel on Elm Street")
    before = len(embed.calls)
    assert mem.warm_index() is True
    # Warming exists to preload nomic off the turn path; with the store
    # fitting, the turn path never asks for it and the preload would only
    # evict the 26B residency has just warmed.
    assert embed.calls[before:] == []


def test_warm_index_still_preloads_when_ranking_will_run(mem, embed):
    for i in range(memory_mod.CONTEXT_ALL_FACTS_MAX + 1):
        mem.remember(f"filler{i}", f"filler fact number {i} about nothing")
    before = len(embed.calls)
    assert mem.warm_index() is True
    assert embed.calls[before:] == [["search_query: hello"]]


def test_embedder_hands_the_single_slot_back(mem):
    # keep_alive -1 pinned nomic in the one slot the chat model needs.
    assert memory_mod.EMBED_KEEP_ALIVE == 0
