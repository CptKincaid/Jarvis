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
    """Recurring titles -- his courses -- join once each."""
    _write_calendar(env, ["BIOSENSORS", "Magnetic Resonance Engr",
                          "BIOSENSORS", "Magnetic Resonance Engr"])
    prompt = vocab.build_prompt()
    assert prompt.count("BIOSENSORS") == 1
    assert "Magnetic Resonance Engr" in prompt


# ------------------------------------------ one-off titles stay out
# 2026-09-04 20:58:45: a surname from a ONE-OFF appointment title, cut by
# title[:48] to end in "<surname>,", was echoed four times by Whisper on
# unclear audio and reached the "Was that for me?" card. The live cache
# held 14 distinct titles, 11 of them one-offs; only the 3 courses recur.
def test_a_one_off_appointment_title_stays_out_of_the_prompt(env):
    _write_calendar(env, ["BIOSENSORS", "BIOSENSORS",
                          "Dentist with Dr. Orbalind"])
    prompt = vocab.build_prompt()
    assert "BIOSENSORS" in prompt
    assert "Orbalind" not in prompt


def test_a_title_needs_min_title_recurrence_events_to_join(env):
    assert vocab.MIN_TITLE_RECURRENCE == 2
    _write_calendar(env, ["Quennevex Seminar"] * (vocab.MIN_TITLE_RECURRENCE - 1)
                    + ["Orbalind Lab"] * vocab.MIN_TITLE_RECURRENCE)
    terms = vocab.build_prompt().split(", ")
    assert "Orbalind Lab" in terms
    assert "Quennevex Seminar" not in terms


def _write_sources(tmp_path, **sources):
    """sources: name -> list of titles, one event each."""
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    payload = {"version": 1, "fetched_at": 1.0, "sources": {
        name: {"fetched_at": 1.0,
               "events": [{"title": t, "location": ""} for t in titles]}
        for name, titles in sources.items()}}
    (cache / "calendar_cache.json").write_text(json.dumps(payload))


def test_recurrence_ignores_case_and_spacing_and_keeps_the_first_spelling(env):
    """Keyed on the whitespace-collapsed lowercase title; the first
    spelling seen is the one that joins."""
    _write_sources(env, google=["Magnetic  Resonance Engr",
                                "Advising with Orbalind",
                                "magnetic resonance engr"])
    terms = vocab.build_prompt().split(", ")
    assert "Magnetic Resonance Engr" in terms
    assert "magnetic resonance engr" not in terms
    assert not [t for t in terms if "Orbalind" in t]


def test_a_one_off_mirrored_on_two_calendars_is_still_a_one_off(env):
    """Recurrence is counted WITHIN a source. The same appointment carried
    by a Google and an iCloud calendar is one event twice, not a course,
    and a cross-source sum would have put its surname straight back into
    the prompt (review of 69afb9f). Live cache 2026-09-04, counted: 3
    sources, 30 events, 14 titles, 0 of them in more than one source, so
    the per-source rule keeps the same 3 courses (6, 6 and 7 events)."""
    title = "Consult with Dr. Orbalind at Quennevex Clinic"
    _write_sources(env, google=[title, "BIOSENSORS", "BIOSENSORS"],
                   icloud=[title], outlook=[title])
    prompt = vocab.build_prompt()
    assert "BIOSENSORS" in prompt
    assert "Orbalind" not in prompt and "Quennevex" not in prompt


def test_a_course_on_one_calendar_recurs_even_if_another_mirrors_it_once(env):
    _write_sources(env, google=["Orbalind Lab", "Orbalind Lab"],
                   icloud=["Orbalind Lab"])
    assert "Orbalind Lab" in vocab.build_prompt().split(", ")


def test_recurring_titles_keep_first_appearance_order(env):
    _write_calendar(env, ["Orbalind Lab", "Quennevex Seminar",
                          "Quennevex Seminar", "Orbalind Lab"])
    terms = vocab.build_prompt().split(", ")
    assert terms.index("Orbalind Lab") < terms.index("Quennevex Seminar")


def test_untitled_is_still_skipped_even_when_it_recurs(env):
    _write_calendar(env, ["Untitled", "untitled", "BIOSENSORS", "BIOSENSORS"])
    terms = vocab.build_prompt().split(", ")
    assert "BIOSENSORS" in terms
    assert not [t for t in terms if t.lower() == "untitled"]


# ----------------------------------------------------- safe truncation
def _no_dangling_terms(prompt: str) -> None:
    assert ",," not in prompt
    for term in prompt.split(", "):
        assert term, "an empty term made it into the prompt"
        assert term == term.strip()
        assert term[-1] not in ",:;.-", term


def test_a_long_title_is_cut_at_a_word_boundary(env):
    # 53 chars; title[:48] would be "...Laboratory Sectio"
    title = "Magnetic Resonance Engineering Laboratory Section 502"
    _write_calendar(env, [title, title])
    terms = vocab.build_prompt().split(", ")
    assert "Magnetic Resonance Engineering Laboratory" in terms
    assert not [t for t in terms if t.endswith("Sectio")]
    _no_dangling_terms(", ".join(terms))


def test_a_title_cut_right_after_a_comma_loses_the_comma(env):
    # 61 chars; title[:48] is "...Orbalind, Q": the word cut leaves
    # "...Orbalind," and the punctuation strip takes the comma.
    title = "Lab Section Meeting with Dr. Sedrick Orbalind, Quennevex Hall"
    _write_calendar(env, [title, title])
    terms = vocab.build_prompt().split(", ")
    assert "Lab Section Meeting with Dr. Sedrick Orbalind" in terms
    assert "Orbalind, Q" not in ", ".join(terms)
    _no_dangling_terms(", ".join(terms))


def test_a_first_word_longer_than_the_cap_keeps_the_hard_cut(env):
    title = "Q" * 60
    _write_calendar(env, [title, title])
    terms = vocab.build_prompt().split(", ")
    assert "Q" * 48 in terms
    assert not [t for t in terms if len(t) > 48 and t.startswith("QQ")]


# The invented twin of the 20:58:45 title SHAPE: longer than the cap,
# one-off, and title[:48] ends exactly on "Quennevex,". Every word of it
# is invented -- the wording, the initial and the suffix are not the live
# title's.
LIVE_SHAPE = "Consultations: Remote Session with Q. Quennevex, PhD"


def test_the_live_shape_never_reaches_the_prompt(env):
    """Neither the surname nor a double comma may appear."""
    assert len(LIVE_SHAPE) == 52 and LIVE_SHAPE[:48].endswith("Quennevex,")
    _write_calendar(env, ["BIOSENSORS", "BIOSENSORS", LIVE_SHAPE])
    prompt = vocab.build_prompt()
    assert "Quennevex" not in prompt
    assert "Consultations" not in prompt
    _no_dangling_terms(prompt)


def test_the_live_shape_recurring_is_still_cut_clean(env):
    """Even if such a title DID recur, the dangling comma cannot survive."""
    _write_calendar(env, [LIVE_SHAPE, LIVE_SHAPE])
    terms = vocab.build_prompt().split(", ")
    assert "Consultations: Remote Session with Q. Quennevex" in terms
    _no_dangling_terms(", ".join(terms))


def test_buildings_truncate_at_a_word_boundary_too(env):
    # speech name is 46 chars; [:32] would be "...Memorial Int"
    loc = ("College Station Zachariah Quennevex Memorial Interdisciplinary "
           "Research Pavilion 210")
    _write_calendar(env, ["BIOSENSORS", "BIOSENSORS"], [loc, loc])
    terms = vocab.build_prompt().split(", ")
    assert "Zachariah Quennevex Memorial" in terms
    assert not [t for t in terms if t.endswith(" Int")]
    _no_dangling_terms(", ".join(terms))


def test_a_term_ending_in_a_comma_is_stripped_from_every_layer(env):
    (env / "voice_vocab.txt").write_text("Quennevex,\nLibrespot ,")
    (env / "voice_names.txt").write_text("Orbalind,\n,\n")
    prompt = vocab.build_prompt()
    terms = prompt.split(", ")
    assert "Quennevex" in terms and "Librespot" in terms
    assert "Orbalind" in terms
    _no_dangling_terms(prompt)


def test_a_term_starting_with_a_comma_cannot_form_an_empty_item(env, monkeypatch):
    """A leading comma reads as "y, ,x" -- an empty list item from the
    other side. Names, pronunciation keys and titles are not split on
    commas, so the strip must take both ends (review of 69afb9f)."""
    (env / "voice_names.txt").write_text(",Orbalind\n, ,\n")
    monkeypatch.setattr(vocab, "_pronounce_keys", lambda: [",Quennevex,"])
    monkeypatch.setattr(vocab, "_calendar_titles", lambda: [",Orbalind Lab"])
    prompt = vocab.build_prompt()
    terms = prompt.split(", ")
    assert "Orbalind" in terms and "Quennevex" in terms
    assert "Orbalind Lab" in terms
    assert ", ," not in prompt and not prompt.startswith(",")
    for term in terms:
        assert term and term[0] != ","
    _no_dangling_terms(prompt)


def test_clip_term_helper(env):
    clip = vocab.clip_term
    assert clip("Magnetic Resonance Engr", 48) == "Magnetic Resonance Engr"
    assert clip("Lab Section Meeting with Dr. Orbalind, PhD", 40) == \
        "Lab Section Meeting with Dr. Orbalind"
    assert clip("Orbalind, Quennevex", 9) == "Orbalind"
    assert clip("Orbalind, Quennevex", 10) == "Orbalind"
    assert clip("Orbalind, Quennevex", 12) == "Orbalind"
    assert clip("Supercalifragilistic Hall", 8) == "Supercal"
    assert clip("Seminar - ", 48) == "Seminar"
    assert clip("Seminar:;.-, ", 48) == "Seminar"
    assert clip(",,,", 48) == ""


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
