"""The model-callable `remember` tool (jarvis/tools/remember.py).

His ruling, 2026-09-04 18:15: "yes give him a remember tool, read it back".
The spoken rung ("remember that ...") stored and read back since 09-04;
the model had no tool that stores a fact, so "put it in your memory that
I graduate December 10th" went to the model, which said "I have noted
that, sir" and stored nothing -- the class the claim guard exists for.

ONE door: memory.store_fact_from_speech, the rung's own helper. The tool
refuses what the rung refuses (a bare head, a question, a pointer) and
what is somebody else's job (a to-do, a reminder, a list item), and its
read-back is an authored line -- spoken verbatim, ending the turn -- so
the model never paraphrases what was filed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis import brain as brain_mod
from jarvis.memory import JarvisMemory
from jarvis.tools import remember as rt
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def memory(tmp_path):
    return JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                        semantic=False)


def _tool(memory, services=None):
    (spec,) = rt.make_tools(None, services or SimpleNamespace(memory=memory))
    handler = spec.handler

    def judged(fact="", **kw):
        # Direct calls stand in for the FORCED path, which vouches itself;
        # the model's calls go through _call() and the deriver.
        kw.setdefault("verbatim", True)
        return handler(fact=fact, **kw)
    spec.handler = judged
    return spec


def test_the_spec_is_small_and_takes_one_fact(memory):
    spec = _tool(memory)
    assert spec.name == "remember"
    assert len(spec.description.split()) <= 20
    params = spec.schema()["function"]["parameters"]
    assert list(params["properties"]) == ["fact"]
    assert params["required"] == ["fact"]


def test_a_fact_is_filed_through_the_one_door_and_read_back_verbatim(memory):
    spec = _tool(memory)
    r = spec.handler(fact="I graduate December 10th 2026 with an electrical engineering degree")
    assert r.ok
    hits = memory.recall("graduate")
    assert len(hits) == 1
    stored = hits[0]["value"]
    assert stored.lower() == "you graduate december 10th 2026 with an electrical engineering degree"
    assert r.speak == f"Noted, sir: {stored}."            # the words filed, verbatim


def test_the_same_fact_said_twice_is_one_fact(memory):
    spec = _tool(memory)
    spec.handler(fact="my dentist is Dr Patel")
    spec.handler(fact="My dentist is Dr Patel.")
    assert len(memory.recall("dentist")) == 1


def test_a_person_statement_also_fills_the_people_book(memory):
    spec = _tool(memory)
    r = spec.handler(fact="my advisor is Dr Peyrovi, email hp@tamu.edu")
    assert r.ok
    assert any("peyrovi" in str(v).lower() for v in memory.people().values())


@pytest.mark.parametrize("fact, why", [
    ("", "empty"),
    ("me", "one word"),
    ("that", "a pointer"),
    ("for later", "a pointer"),
    ("everything I just said", "a pointer"),
    ("when do I graduate?", "a question"),
    ("remind me to call mom at 5", "a reminder -- set_reminder's job"),
    ("add milk to my shopping list", "a list item -- the notes tool's job"),
    ("set a timer for ten minutes", "a timer"),
])
def test_what_is_not_a_fact_is_refused_and_nothing_is_stored(memory, fact, why):
    spec = _tool(memory)
    r = spec.handler(fact=fact)
    assert not r.ok, why
    assert r.speak is None, why                      # the model answers, honestly
    assert memory._facts == {}


def test_the_refusal_names_the_right_tool_for_the_model(memory):
    spec = _tool(memory)
    assert "reminder" in spec.handler(fact="remind me to call mom at 5").text.lower()
    assert "list" in spec.handler(fact="add milk to my shopping list").text.lower()
    assert "question" in spec.handler(fact="when do I graduate?").text.lower()


def test_no_memory_service_is_an_honest_failure(tmp_path):
    (spec,) = rt.make_tools(None, SimpleNamespace())
    r = spec.handler(fact="my dentist is Dr Patel")
    assert not r.ok and "memory" in r.text.lower()


def test_the_tool_registers_and_the_claim_guard_knows_it(memory):
    reg = ToolRegistry()
    reg.register_many(rt.make_tools(None, SimpleNamespace(memory=memory)))
    assert reg.has("remember")
    assert "remember" in brain_mod.CLAIM_BACKERS["memory"]
    assert brain_mod.claim_backed("I have noted that", ran=["remember"])
    assert not brain_mod.claim_backed("I have noted that", ran=["get_time"])


def test_the_app_registers_it():
    from jarvis.app import TOOL_MODULES
    assert "jarvis.tools.remember" in TOOL_MODULES


def test_the_nudge_now_names_the_tool():
    assert "remember" in brain_mod.UNBACKED_MEMORY_NUDGE
    assert "no tool that stores facts" not in brain_mod.UNBACKED_MEMORY_NUDGE


# ------------------------------------------- the attack round, 2026-09-12
def _call(memory, fact, utterance):
    """Through the registry as the MODEL would call it: the deriver sees
    his utterance beside the model's argument."""
    reg = ToolRegistry()
    reg.register_many(rt.make_tools(None, SimpleNamespace(memory=memory)))
    return reg.call("remember", {"fact": fact}, from_model=True, utterance=utterance)


def test_a_fact_he_did_not_say_is_refused_and_not_stored(memory):
    """The registry hands every model call his utterance for exactly this:
    the model may pass HIS words and nothing else. Invented, paraphrased or
    third-person facts are refused, so nothing he did not say is filed."""
    r = _call(memory, "you are feeling sad today", "what time is it")
    assert not r.ok and "his words" in r.text.lower()
    assert memory._facts == {}
    r = _call(memory, "The user graduates on 10 December 2026 (EE degree)",
              "Keep this in your memory: I graduate December 10th 2026 with an EE degree")
    assert not r.ok
    assert memory._facts == {}


@pytest.mark.parametrize("fact, said", [
    ("I graduate December 10th 2026", "Jarvis, remember that I graduate December 10th 2026, please."),
    ("i graduate december 10th 2026", "Remember that I graduate December 10th 2026."),
    ("I'm allergic to penicillin", "put it in your memory that I am allergic to penicillin"),
    ("my dentist is Dr Patel", "Don't forget my dentist is Dr. Patel!"),
])
def test_his_words_pass_the_check_with_case_punctuation_and_contractions_forgiven(memory, fact, said):
    r = _call(memory, fact, said)
    assert r.ok, (fact, said)
    assert r.speak.startswith("Noted, sir:")


def test_a_call_nobody_judged_is_refused_and_a_forced_caller_vouches(memory):
    """verbatim=None means no deriver ran: not the registry's model path
    and not a caller that said verbatim=True. Fail closed."""
    (spec,) = rt.make_tools(None, SimpleNamespace(memory=memory))
    assert not spec.handler(fact="my locker code is 4412").ok
    assert memory._facts == {}
    assert spec.handler(fact="my locker code is 4412", verbatim=True).ok


def test_the_model_cannot_vouch_for_itself(memory):
    """`verbatim` is reserved: a model that sends verbatim=true is stripped."""
    reg = ToolRegistry()
    reg.register_many(rt.make_tools(None, SimpleNamespace(memory=memory)))
    r = reg.call("remember", {"fact": "you owe me money", "verbatim": True},
                 from_model=True, utterance="what time is it")
    assert not r.ok and memory._facts == {}


@pytest.mark.parametrize("fact", [
    "remember that I graduate December 10th 2026",
    "don't forget that I graduate December 10th 2026",
    "put it in your memory that I graduate December 10th 2026",
    "keep in mind I graduate December 10th 2026",
])
def test_the_rungs_head_is_stripped_so_both_doors_file_one_fact(memory, fact):
    """'His exact words' begin with the head; the head is not the fact.
    The same sentence through the rung and the tool is ONE fact."""
    spec = _tool(memory)
    r = spec.handler(fact=fact)
    assert r.ok
    assert r.speak.lower().startswith("noted, sir: you graduate december 10th 2026")
    assert len(memory._facts) == 1
    assert next(iter(memory._facts)) == "you graduate december 10th 2026"


@pytest.mark.parametrize("fact, why", [
    ("remember that", "a bare head"),
    ("note that down", "a bare head"),
    ("remember me", "a bare head"),
    ("remember to call mom at 5", "a to-do"),
    ("to call mom at five", "a to-do"),
    ("call mom at five", "a bare imperative"),
    ("buy milk", "a bare imperative"),
    ("turn off the lights", "a command"),
    ("what my dentist's name is", "recall"),
    ("when I graduate", "recall without its ?"),
    ("about the thesis", "recall"),
    ("please set an alarm for 6", "another tool's job, with a courtesy opener"),
    ("could you remind me to call mom", "another tool's job, with a courtesy opener"),
])
def test_what_the_rung_refuses_the_tool_refuses(memory, fact, why):
    spec = _tool(memory)
    r = spec.handler(fact=fact)
    assert not r.ok, why
    assert memory._facts == {}, why


def test_a_hallucinated_address_never_reaches_the_people_book(memory):
    """The people-book side effect is his words too: an address the model
    added is not in the utterance, so nothing is written under his alias."""
    r = _call(memory, "my advisor is Dr Peyrovi, email hp@tamu.edu",
              "remember that my advisor is Dr Peyrovi")
    assert not r.ok                                 # the whole fact is not his words
    assert memory.people() == {}


def test_the_read_back_is_the_fact_but_the_log_is_only_its_key(memory, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="jarvis.tools.remember")
    spec = _tool(memory)
    r = spec.handler(fact="my wifi password is hunter2swordfish")
    assert r.ok and "hunter2swordfish" in r.speak                 # his ruling: read it back
    assert not any("hunter2swordfish" in rec.getMessage() for rec in caplog.records), \
        [rec.name for rec in caplog.records if "hunter2swordfish" in rec.getMessage()]


def test_a_non_string_fact_is_refused(memory):
    spec = _tool(memory)
    assert not spec.handler(fact={"a": 1}).ok
    assert not spec.handler(fact=["x", "y"]).ok
    assert not spec.handler(fact="x" * 3000).ok                     # a paragraph is not a fact
    assert memory._facts == {}


def test_a_trailing_ellipsis_is_not_part_of_the_fact(memory):
    spec = _tool(memory)
    r = spec.handler(fact="I graduate December 10th, please…")
    assert r.ok and r.speak.lower() == "noted, sir: you graduate december 10th."


def test_the_tool_parks_an_undo_for_scratch_that(memory):
    """The rung returns an undo closure; a ToolResult cannot. The tool
    parks it on services the way the calendar parks last_add, and the
    commander's undo rung finds it there."""
    services = SimpleNamespace(memory=memory)
    spec = _tool(memory, services)
    spec.handler(fact="my locker code is 4412")
    parked = services.remember_undo
    assert callable(parked["undo"]) and parked["at"] > 0
    assert parked["undo"]() == "Forgotten, sir."
    assert memory._facts == {}


# ------------------------------------------ the second attack round
def test_a_question_that_asked_for_no_store_is_not_filed(memory):
    """'Do you know if my dentist is Dr Patel?' -- the words are his and
    the fact is in them, but he asked a question and asked for nothing
    to be remembered. Nothing is filed and the model answers."""
    r = _call(memory, "my dentist is Dr Patel", "Do you know if my dentist is Dr Patel?")
    assert not r.ok and "question" in r.text.lower()
    assert memory._facts == {}
    # ... while a question that carries the ask is one
    r = _call(memory, "my dentist is Dr Patel",
              "Could you remember that my dentist is Dr Patel?")
    assert r.ok


@pytest.mark.parametrize("fact", [
    "book club is on Tuesdays", "pay day is the 15th", "check-in is at 3pm",
    "start date is October 5th", "play rehearsal is on Friday",
])
def test_a_verb_shaped_noun_with_a_finite_verb_is_a_fact(memory, fact):
    spec = _tool(memory)
    assert spec.handler(fact=fact).ok, fact


@pytest.mark.parametrize("fact, said", [
    ("my pin is 12", "remember that my pin is 1234"),
    ("I am allergic to penicillin", "remember that it is not true that I am allergic to penicillin"),
    ("I like tea", "remember that I never said I like tea"),
])
def test_a_truncated_number_or_a_denied_clause_is_not_his_words(memory, fact, said):
    r = _call(memory, fact, said)
    assert not r.ok
    assert memory._facts == {}


@pytest.mark.parametrize("fact, said", [
    ("my dentist is Dr Patel", "my dentist, uh, is Dr Patel, keep that in mind"),
    ("she is allergic to nuts", "remember that she's allergic to nuts"),
    ("my meeting is at 5 pm", "remember that my meeting is at 5 p.m."),
    ("we live at 12 Oak Street", "don't forget we're living at 12 Oak Street" if False else "don't forget we live at 12 Oak Street"),
])
def test_fillers_contractions_and_dotted_abbreviations_are_forgiven(memory, fact, said):
    assert _call(memory, fact, said).ok, (fact, said)


@pytest.mark.parametrize("fact", [
    "that I graduate December 10th 2026",
    "Jarvis, remember that I graduate December 10th 2026",
    "please remember that I graduate December 10th 2026",
])
def test_a_copied_that_or_address_files_under_the_rungs_key(memory, fact):
    spec = _tool(memory)
    assert spec.handler(fact=fact).ok
    assert list(memory._facts) == ["you graduate december 10th 2026"]


def test_a_question_with_a_courtesy_tail_is_still_a_question(memory):
    spec = _tool(memory)
    assert not spec.handler(fact="do I graduate in December, jarvis?").ok
    assert not spec.handler(fact="do I graduate in December").ok
    assert memory._facts == {}


def test_a_deriver_failure_refuses_rather_than_trusts(memory, monkeypatch):
    """The registry swallows a deriver exception; the deriver must not
    let that become silent trust."""
    monkeypatch.setattr(rt, "his_words", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    r = _call(memory, "my dentist is Dr Patel", "remember that my dentist is Dr Patel")
    assert not r.ok and memory._facts == {}
