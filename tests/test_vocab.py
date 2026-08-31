"""jarvis/vocab.py — the per-turn Whisper prompt built from his own world.

Real modules throughout; the only stand-ins are files under tmp_path (the
same firewall test_corrections uses for VOCAB_FILE) and a seeded Canvas
module cache. No network, no model, no hardware.
"""
import json

import pytest

import jarvis.pronounce as pronounce
import jarvis.tools.canvas as canvas
from jarvis import vocab
from jarvis.config import PATHS
from jarvis.pronounce import Pronunciations
from jarvis.transcriber import DEFAULT_VOCAB, save_vocab


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    monkeypatch.setattr(PATHS, "NAMES_FILE", tmp_path / "voice_names.txt")
    monkeypatch.setattr(PATHS, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pronounce, "_default",
                        Pronunciations(path=tmp_path / "pron.json"))
    canvas.clear_cache()
    vocab.clear_cache()
    yield tmp_path
    canvas.clear_cache()
    vocab.clear_cache()


def _write_calendar(tmp_path, titles, locations=()):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    locations = list(locations) + [""] * (len(titles) - len(list(locations)))
    payload = {"version": 1, "fetched_at": 1.0,
               "sources": {"google-1": {
                   "fetched_at": 1.0,
                   "events": [{"title": t, "location": loc}
                              for t, loc in zip(titles, locations)]}}}
    (cache / "calendar_cache.json").write_text(json.dumps(payload))


# ------------------------------------------------------------- the seed
def test_default_vocab_is_the_assistants_not_the_warehouses():
    for jargon in ("AGV", "forklift", "pallet", "conveyor", "shipping dock"):
        assert jargon not in DEFAULT_VOCAB
    for term in ("Jarvis", "Canvas", "Ollama", "calendar", "timer"):
        assert term in DEFAULT_VOCAB


def test_seed_only_when_nothing_configured(env):
    prompt = vocab.build_prompt()
    assert prompt.startswith("Jarvis, ")
    assert "forklift" not in prompt
    assert "Ollama" in prompt


# ------------------------------------------------- layering and priority
def test_user_vocab_and_names_lead_the_prompt(env):
    (env / "voice_vocab.txt").write_text("Qwen\nLibrespot")
    (env / "voice_names.txt").write_text("Peyrovi\nAmbrose\n")
    prompt = vocab.build_prompt()
    # manual vocab first, then names newest-first, then the seed
    assert prompt.startswith("Qwen, Librespot, Ambrose, Peyrovi, Jarvis")


def test_names_are_newest_first_and_deduped(env):
    (env / "voice_names.txt").write_text(
        "Peyrovi\n# a comment\npeyrovi\nAmbrose\n")
    assert vocab.load_names() == ["Ambrose", "peyrovi"]


def test_pronunciation_user_keys_join_defaults_stay_out(env):
    pronounce.get().add("Nathania", "nuh-THAN-ya")
    vocab.clear_cache()
    prompt = vocab.build_prompt()
    assert "Nathania" in prompt
    # the ~100 shipped TTS defaults would eat the whole budget
    assert "engr" not in prompt.split(", ")
    assert "bldg" not in prompt.split(", ")


def test_calendar_titles_join(env):
    _write_calendar(env, ["BIOSENSORS", "Magnetic Resonance Engr",
                          "BIOSENSORS"])
    prompt = vocab.build_prompt()
    assert prompt.count("BIOSENSORS") == 1
    assert "Magnetic Resonance Engr" in prompt


def test_the_buildings_he_walks_to_are_in_the_prompt(env):
    """He says "how long to Wisenbaker" out loud to teach a walk
    (jarvis/leavetime.py), so the building has to be hearable. The two ETB
    rooms are ONE name, and neither the Zoom URL nor the blank location is
    a building."""
    _write_calendar(
        env, ["BIOSENSORS", "MEEN 361", "SENIOR DESIGN", "Advising"],
        ["College Station Wisenbaker Engineering Bldg 049",
         "College Station Emerging Technologies Building 1003",
         "College Station Emerging Technologies Building 1020",
         "https://tamu.zoom.us/j/94324046592?pwd=x"])
    terms = vocab.build_prompt().split(", ")
    assert "Wisenbaker" in terms
    assert terms.count("Emerging Technologies") == 1
    assert not [t for t in terms if "zoom.us" in t]
    assert not [t for t in terms if t.endswith(" 049") or t.endswith(" 1003")]


def test_corrupt_calendar_cache_is_just_a_miss(env):
    cache = env / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "calendar_cache.json").write_text("{not json")
    prompt = vocab.build_prompt()
    assert prompt.startswith("Jarvis")


def test_canvas_cached_courses_join_tidied(env):
    canvas._COURSES["https://canvas.tamu.edu"] = (canvas._clock(), [
        {"id": 1, "name": "BMEN 420 500 BIOSENSORS FA26"},
        {"id": 2, "name": "ECEN 749 MAGNETIC RESONANCE ENGR SP26"},
    ])
    prompt = vocab.build_prompt()
    assert "BIOSENSORS" in prompt
    assert "MAGNETIC RESONANCE ENGR" in prompt
    assert "BMEN 420" not in prompt       # tidy_course drops the catalogue code


def test_dedup_across_sources(env):
    # "Ollama" from the names file must not appear twice (seed has it too)
    (env / "voice_names.txt").write_text("Ollama\n")
    prompt = vocab.build_prompt()
    assert prompt.lower().count("ollama") == 1


def test_prompt_is_capped_in_priority_order(env):
    terms = [f"Zephyrname{i:03d}" for i in range(120)]
    (env / "voice_vocab.txt").write_text(", ".join(terms))
    prompt = vocab.build_prompt()
    assert len(prompt) <= vocab.PROMPT_CHAR_CAP
    assert "Zephyrname000" in prompt          # head survives
    assert "Zephyrname119" not in prompt      # tail falls off


# --------------------------------------------------------------- caching
def test_prompt_cached_for_ttl(env, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(vocab, "_clock", lambda: now[0])
    vocab.clear_cache()
    assert "Peyrovi" not in vocab.build_prompt()
    (env / "voice_names.txt").write_text("Peyrovi\n")
    assert "Peyrovi" not in vocab.build_prompt()      # still cached
    now[0] += vocab.PROMPT_TTL_S + 1
    assert "Peyrovi" in vocab.build_prompt()          # TTL expired


def test_clear_cache_busts_the_prompt(env):
    assert "Peyrovi" not in vocab.build_prompt()
    (env / "voice_names.txt").write_text("Peyrovi\n")
    vocab.clear_cache()
    assert "Peyrovi" in vocab.build_prompt()


def test_save_vocab_busts_the_prompt_cache(env):
    """The corrections path (commander._learn_vocab -> save_vocab) must
    reach the very next transcription, not the one after the TTL."""
    assert "Qwen" not in vocab.build_prompt()
    save_vocab("Qwen")
    assert vocab.build_prompt().startswith("Qwen")


# -------------------------------------------------------------- add_name
def test_add_name_appends_and_dedupes(env):
    assert vocab.add_name("Peyrovi") is True
    assert vocab.add_name("peyrovi") is False
    assert (env / "voice_names.txt").read_text() == "Peyrovi\n"
    assert vocab.add_name("Ambrose") is True
    assert vocab.load_names() == ["Ambrose", "Peyrovi"]


def test_add_name_rejects_empty(env):
    assert vocab.add_name("") is False
    assert vocab.add_name("   ") is False
    assert not (env / "voice_names.txt").exists()
