"""2026-08-31: a fact filed in the first person.

    Jarvis: "Remember that it is Mara and I's birthday on"
    facts.json: it is mara and i's birthday on = it is mara and i's
                birthday on

and he said, in his own words:

    "does not context I's as me, probably should just be Mara and your
     anniversary on September 20th"

facts.json is not a quote book. format_for_context renders every fact
STRAIGHT into the model's prompt under "Known facts", where the model IS
Jarvis -- so a fact filed in the first person tells him the anniversary is
his own. He got away with it that evening only because the model guessed
well; the fact on disk still said the wrong thing, and every later recall
re-rolls that guess.
"""
import pytest

from jarvis.memory import JarvisMemory, to_second_person


@pytest.fixture
def mem(tmp_path):
    # semantic=False for the same reason test_memory.py gives: the default
    # wires up the REAL embedder on localhost:11434, and under
    # OLLAMA_MAX_LOADED_MODELS=1 that evicts the chat model the running
    # Jarvis has pinned.
    return JarvisMemory(memory_dir=tmp_path / "jarvis_memory",
                        legacy_dir=tmp_path / "jarvis_data", semantic=False)


def test_his_anniversary_is_filed_the_way_he_asked_for_it(mem):
    """His exact utterance, and his exact wording for the fix."""
    said = "it is Mara and I's anniversary on September 20th"
    key = mem.remember(said[:60], said)

    assert "Mara and your anniversary on September 20th" in mem._facts[key]["value"]
    assert "I's" not in mem._facts[key]["value"]
    assert "I's" not in key, "the KEY is rendered into the prompt too"


def test_the_possessive_pair_becomes_and_your():
    """"Mara and I's" is not standard English and the possessive belongs to
    the pair, so it turns into "Mara and your" -- his own wording."""
    assert to_second_person("it is mara and i's birthday on") == \
        "it is mara and your birthday on"


@pytest.mark.parametrize("said,filed", [
    ("my dentist is Dr Patel", "your dentist is Dr Patel"),
    ("my thesis is due in December", "your thesis is due in December"),
    ("the batch size is mine to pick", "the batch size is yours to pick"),
    ("I signed myself up", "I signed yourself up"),
    ("mara and i's anniversary", "mara and your anniversary"),
])
def test_first_person_possessives_become_second_person(said, filed):
    assert to_second_person(said) == filed


@pytest.mark.parametrize("said", [
    # POSSESSIVES only -- his word. A bare "I" or "me" is left alone
    # because it does not always mean him: the memory garden files notes in
    # JARVIS's voice, and "he told me this himself" means Jarvis.
    "I work on VSS",
    "he told me this himself",
    "actually I want the long one",
])
def test_a_bare_pronoun_is_left_alone(said):
    assert to_second_person(said) == said


def test_a_query_is_asked_in_the_language_the_store_is_written_in(mem):
    """Filing "my dentist is Dr Patel" as "your dentist..." must not make
    it unfindable when he asks for it the way he said it."""
    mem.remember("dentist", "my dentist is Dr Patel")
    hits = mem.recall("my dentist")
    assert hits and hits[0]["value"] == "your dentist is Dr Patel"


def test_a_fact_that_is_already_turned_around_is_left_alone():
    """The memory garden and the debrief file facts in the third or second
    person already; the rewrite has to be a no-op on those."""
    for fact in ["your anniversary with Mara is on September 20th",
                 "the batch size is 16, learning rate 0.001",
                 "Hunter returns Wednesday April 16"]:
        assert to_second_person(fact) == fact


def test_it_is_idempotent(mem):
    """remember() overwrites by key, and a fact re-filed must not drift."""
    said = "it is Mara and I's anniversary on September 20th"
    once = to_second_person(said)
    assert to_second_person(once) == once


def test_remember_returns_the_key_it_actually_used(mem):
    """The garden mirrors what it filed into its own dedupe copy, so it has
    to be told when the key was rewritten."""
    key = mem.remember("my dentist", "my dentist is Dr Patel")
    assert key == "your dentist"
    assert key in mem._facts


def test_the_fact_survives_a_reload(mem, tmp_path):
    mem.remember("my dentist", "my dentist is Dr Patel")
    again = JarvisMemory(memory_dir=tmp_path / "jarvis_memory",
                         legacy_dir=tmp_path / "jarvis_data", semantic=False)
    hits = again.recall("dentist")
    assert hits and hits[0]["value"] == "your dentist is Dr Patel"


def test_a_non_string_value_is_not_mangled(mem):
    """Facts are usually text, but nothing stops a caller storing a number."""
    key = mem.remember("batch size", 16)
    assert mem._facts[key]["value"] == 16
