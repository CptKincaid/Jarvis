"""Per-course scoping for quiz and teach-me (docs.course_chunks).

topic_chunks' last leg is a global embedding query, so a topic that names
no file by name can be answered out of ANOTHER course's material. These
tests pin the bleed and the scoping that stops it, over a real chromadb
DocsIndex driven by tests.test_docs's deterministic FakeEmbed. No Ollama
and no network: the Canvas roster is monkeypatched where it is used at all.
"""
import pytest

import jarvis.tools.docs as docs_mod
from jarvis.tools.docs import course_chunks, topic_chunks
from tests.test_docs import FakeEmbed

# Lecture notes filed by jarvis/lecture.py: <course-slug>-<date>.md.
BIOSENSORS_AUG_29 = """# BIOSENSORS - 2026-08-29

- 09:02  A biosensor couples a biological recognition element to a transducer.
- 09:05  The Clark electrode measures oxygen at a platinum cathode.
"""
BIOSENSORS_AUG_30 = """# BIOSENSORS - 2026-08-30

- 09:01  Selectivity is set by the recognition element, sensitivity by the transducer.
- 09:06  Enzyme electrodes drift as the enzyme denatures.
"""
# Another course's notes. They talk about electrodes and transducers too,
# which is exactly why the global embedding query reaches them.
SIGNALS_NOTE = """# SIGNALS AND SYSTEMS - 2026-08-29

- 10:00  Week four. An electrode transducer converts a quantity into a voltage.
- 10:04  Sampling at twice the highest frequency reconstructs the signal.
"""


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    (d / "biosensors-2026-08-29.md").write_text(BIOSENSORS_AUG_29)
    (d / "biosensors-2026-08-30.md").write_text(BIOSENSORS_AUG_30)
    (d / "signals-and-systems-2026-08-29.md").write_text(SIGNALS_NOTE)
    return d


@pytest.fixture
def index(tmp_path, folder):
    embed = FakeEmbed()
    idx = docs_mod.DocsIndex([folder], tmp_path / "index", embed=embed)
    idx.reindex()
    idx.fake_embed = embed
    return idx


# No canvas.token: resolve_course hands the spoken name straight back, so
# the scoping falls through to slug-matching the file stems.
CFG = {"canvas": {"base_url": "https://canvas.example", "token": ""}}
BIO = {"biosensors-2026-08-29.md", "biosensors-2026-08-30.md"}


# ------------------------------------------------------- course_chunks
def test_a_concept_inside_a_course_stays_inside_that_course(index):
    """The bug, and the reason for the whole feature: "electrode
    transducers in signals and systems" matches no file NAME, so
    topic_chunks' last leg is a global embedding query -- and the
    biosensors notes talk about electrodes and transducers too, so the
    quiz would be written from the wrong course. course_chunks reads the
    tail as the course and never leaves it."""
    topic = "electrode transducers in signals and systems"
    bled = topic_chunks(index, topic, k=6)
    assert BIO & {h["name"] for h in bled}, "the corpus no longer shows the bleed"

    scoped = course_chunks(index, CFG, topic, k=6)
    assert scoped and {h["name"] for h in scoped} == {"signals-and-systems-2026-08-29.md"}


def test_a_chapter_of_a_course_starts_at_that_chapter(index):
    scoped = course_chunks(index, CFG, "week four of signals and systems", k=6)
    assert scoped and {h["name"] for h in scoped} == {"signals-and-systems-2026-08-29.md"}
    assert "Week four" in scoped[0]["text"]


def test_a_course_with_several_notes_reads_them_all(index):
    """topic_chunks' name match resolves ONE file; a course is all of them."""
    one = topic_chunks(index, "biosensors", k=6)
    assert len({h["name"] for h in one}) == 1

    scoped = course_chunks(index, CFG, "biosensors", k=6)
    assert {h["name"] for h in scoped} == BIO


def test_the_canvas_roster_name_is_what_gets_slugified(index, monkeypatch):
    """He says "systems"; the roster (and so the note file) says "SIGNALS
    AND SYSTEMS". Only the roster lookup bridges the two."""
    from jarvis.tools import canvas
    monkeypatch.setattr(canvas, "active_courses", lambda settings, fetch, budget: [
        {"name": "ECEN 314 SIGNALS AND SYSTEMS FA26"}])
    cfg = {"canvas": {"base_url": "https://canvas.example", "token": "tok"}}
    scoped = course_chunks(index, cfg, "systems", k=6)
    assert scoped and {h["name"] for h in scoped} == {"signals-and-systems-2026-08-29.md"}


def test_a_roster_lookup_that_raises_still_answers(index, monkeypatch):
    from jarvis.tools import canvas

    def _down(settings, fetch, budget):
        raise canvas.CanvasError("network", "unreachable")

    monkeypatch.setattr(canvas, "active_courses", _down)
    cfg = {"canvas": {"base_url": "https://canvas.example", "token": "tok"}}
    assert {h["name"] for h in course_chunks(index, cfg, "biosensors", k=6)} == BIO


def test_course_scoping_needs_no_embedding_call(index):
    n = len(index.fake_embed.calls)
    assert course_chunks(index, CFG, "biosensors", k=6)
    assert len(index.fake_embed.calls) == n


def test_scoped_chunks_come_back_in_reading_order(index):
    hits = course_chunks(index, CFG, "biosensors", k=6)
    assert hits == sorted(hits, key=lambda h: (h["name"], h["chunk"]))


def test_an_unknown_course_falls_back_to_the_global_query(index):
    hits = course_chunks(index, CFG, "the nyquist criterion", k=3)
    assert hits and all(h.get("score") is not None for h in hits)


def test_a_chapter_the_course_does_not_have_falls_back(index):
    """The course is right but week nine is not in its notes: a scoped
    answer about the wrong week is worse than the global search."""
    assert docs_mod._course_only_chunks(index, CFG, "week nine of biosensors", 3) == []
    assert course_chunks(index, CFG, "week nine of biosensors", k=3)   # fell back


def test_a_short_slug_never_scopes(index):
    names = index.names()
    assert docs_mod._slug_files(names, "io") == []
    assert docs_mod._slug_files(names, "sig") == []          # a token, not a prefix
    assert docs_mod._slug_files(names, "biosensors") == sorted(BIO)


def test_empty_topic_and_empty_index(index, tmp_path):
    assert course_chunks(index, CFG, "  ", k=3) == []
    empty = docs_mod.DocsIndex([tmp_path / "nothing"], tmp_path / "i2",
                               embed=FakeEmbed())
    assert course_chunks(empty, CFG, "biosensors", k=3) == []
