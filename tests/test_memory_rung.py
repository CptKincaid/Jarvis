"""The memory rung, settled (branch memory-rung, 2026-09-04).

Two refuters ran 7539478's widened rung over 106 sentences: 24 of the 55
must-store shapes missed (a comma or colon after the head, a leading
"please" / "can you", "okay jarvis" / "Jarvis.", a curly apostrophe, "do
not forget", "jot down" / "fyi" / "for the record" / "store this"), 19 of
the 51 must-not shapes were filed anyway ("remember that" stored the word
"that"; "remember when we went to austin?" stored the question), a fact
containing "suggested" was stolen by the suggest rung, and "commit this to
memory" ran `git add -A` through a shell. The spoken ack never stated the
fact, and the fact reached the prompt in the first person, doubled
("i graduate ...: i graduate ...").

THE FACT-VS-NOTE RULE (the thing 4fa27d7 left for this branch to settle):

    "that <clause>" introduces a FACT -> long-term memory (facts.json).
        remember that I graduate ...    note that my locker code is 4412
        jot down that I graduate ...    make a note that I graduate ...
    "of <thing>", ":" + a bare noun phrase, "jot down <thing>",
    "make a note: <thing>", "take a note ..." are NOTES -> the notes list,
    exactly as before.
        make a note of milk             note: buy milk
        jot down milk                   make a note: buy milk
    A bare head with nothing after it ("remember that", "note that down",
    "make a note of that") stores NOTHING and asks what to remember.
    A question ("remember when ...?", "do you remember my ...") is a RECALL.

Every sentence here runs through Commander.handle under the test_commander
fixtures (fake memory / brain / tts, background threads inline) -- the
desktop firewall in tests/conftest.py refuses xclip and the shell, so no
corpus row can reach his clipboard again. The rendering half uses a real
JarvisMemory(semantic=False) in tmp_path; no live store, no Ollama.
"""
from unittest.mock import ANY

import pytest

from jarvis.brain import build_user_turn
from jarvis.commander import (
    ASSISTANT_TIER1,
    REGISTRY,
    _m_quick_command,
    strip_jarvis_prefix,
)
from jarvis.memory import (
    JarvisMemory,
    fact_key,
    speech_to_second_person,
    store_fact_from_speech,
)

from tests.test_commander import cmdr, services  # noqa: F401  (fixtures)

ASK_LINE = "What shall I remember, sir?"

GRAD = "you graduate december 10th 2026"
GRAD_FULL = "you graduate december 10th 2026 with an electrical engineering degree"
LOCKER = "your locker code is 4412"
LAB = "the lab moved to room 049"
HEATHER = "heather prefers email"


def _stored(services):  # noqa: F811
    assert services.memory.remember.call_count == 1, \
        services.memory.remember.call_args_list
    key, value = services.memory.remember.call_args[0]
    return key, value


# =========================================================== MUST STORE
MUST_STORE = [
    # -- the heads, with the 09-02 sentence and its siblings
    ("jarvis remember that i graduate december 10th 2026 with an electrical engineering degree", GRAD_FULL),
    ("jarvis remember i graduate december 10th 2026", GRAD),
    ("jarvis remember, i graduate december 10th 2026", GRAD),
    ("jarvis remember: i graduate december 10th 2026", GRAD),
    ("jarvis remember this: i graduate december 10th 2026", GRAD),
    ("jarvis remember that, i graduate december 10th 2026", GRAD),
    # -- a leading courtesy
    ("jarvis please remember that i graduate december 10th 2026", GRAD),
    ("jarvis can you remember that i graduate december 10th 2026", GRAD),
    ("jarvis could you remember that i graduate december 10th 2026", GRAD),
    ("jarvis would you remember that i graduate december 10th 2026", GRAD),
    ("jarvis will you remember that i graduate december 10th 2026", GRAD),
    ("jarvis i want you to remember that i graduate december 10th 2026", GRAD),
    ("jarvis i need you to remember that i graduate december 10th 2026", GRAD),
    ("jarvis i'd like you to remember that i graduate december 10th 2026", GRAD),
    ("jarvis please, remember that i graduate december 10th 2026", GRAD),
    # -- note that <clause> is a FACT
    ("jarvis please note that my locker code is 4412", LOCKER),
    ("jarvis take note that my locker code is 4412", LOCKER),
    ("jarvis note that my locker code is 4412", LOCKER),
    ("jarvis note that, my locker code is 4412", LOCKER),
    ("jarvis make a note that i graduate december 10th 2026", GRAD),
    ("jarvis jot down that i graduate december 10th 2026", GRAD),
    ("jarvis write down that i graduate december 10th 2026", GRAD),
    # -- keep / bear in mind
    ("jarvis keep in mind that heather prefers email", HEATHER),
    ("jarvis keep in mind heather prefers email", HEATHER),
    ("jarvis keep in mind, heather prefers email", HEATHER),
    ("jarvis bear in mind that heather prefers email", HEATHER),
    ("jarvis bear in mind heather prefers email", HEATHER),
    # -- forget, both apostrophes, and the long forms
    ("jarvis don't forget that the lab moved to room 049", LAB),
    ("jarvis don’t forget that the lab moved to room 049", LAB),
    ("jarvis do not forget that the lab moved to room 049", LAB),
    ("jarvis never forget that the lab moved to room 049", LAB),
    ("jarvis don't forget the lab moved to room 049", LAB),
    # -- put / save / add / store ... memory
    ("jarvis put it in your memory that i graduate december 10th 2026", GRAD),
    ("jarvis put this in your memory: i graduate december 10th 2026", GRAD),
    ("jarvis put that in your memory, i graduate december 10th 2026", GRAD),
    ("jarvis put it in memory that i graduate december 10th 2026", GRAD),
    ("jarvis save this to your memory: i graduate december 10th 2026", GRAD),
    ("jarvis add this to your memory: i graduate december 10th 2026", GRAD),
    ("jarvis store that in your memory, i graduate december 10th 2026", GRAD),
    ("jarvis save it to memory: i graduate december 10th 2026", GRAD),
    ("jarvis store this: i graduate december 10th 2026", GRAD),
    ("jarvis save this: i graduate december 10th 2026", GRAD),
    ("jarvis save this to memory: i graduate december 10th 2026", GRAD),
    ("jarvis commit this to memory: i graduate december 10th 2026", GRAD),
    ("jarvis commit that to memory, i graduate december 10th 2026", GRAD),
    ("jarvis commit it to memory that i graduate december 10th 2026", GRAD),
    ("jarvis memorise that i graduate december 10th 2026", GRAD),
    ("jarvis memorize that i graduate december 10th 2026", GRAD),
    ("jarvis memorise this: i graduate december 10th 2026", GRAD),
    # -- for the record / fyi
    ("jarvis for the record, i graduate december 10th 2026", GRAD),
    ("jarvis for the record i graduate december 10th 2026", GRAD),
    ("jarvis for the record: i graduate december 10th 2026", GRAD),
    ("jarvis fyi i graduate december 10th 2026", GRAD),
    ("jarvis fyi, my locker code is 4412", LOCKER),
    # -- a fact that contains "suggested" is not a suggestion request
    ("jarvis remember that my advisor suggested i take the signals course",
     "your advisor suggested you take the signals course"),
    # -- courtesy tails and punctuation never become part of the fact
    ("jarvis remember that i graduate december 10th 2026 please", GRAD),
    ("jarvis note that my locker code is 4412 please", LOCKER),
    ("jarvis don't forget that i graduate december 10th 2026, please.", GRAD),
    ("jarvis remember that i graduate december 10th 2026!", GRAD),
    ("jarvis remember that i graduate december 10th 2026, thanks", GRAD),
    ("jarvis remember that i graduate december 10th 2026 thank you", GRAD),
    ("jarvis remember that i graduate december 10th 2026, would you", GRAD),
    ("jarvis remember that i graduate december 10th 2026 sir", GRAD),
    ("jarvis remember that i graduate december 10th 2026 jarvis", GRAD),
    ("jarvis remember that i graduate december 10th 2026, will you?", GRAD),
    # -- the wake-word spellings Whisper produces
    ("okay jarvis remember that i graduate december 10th 2026", GRAD),
    ("ok jarvis remember that i graduate december 10th 2026", GRAD),
    ("okay jarvis, remember that i graduate december 10th 2026", GRAD),
    ("jarvis. remember that i graduate december 10th 2026", GRAD),
    ("hey jarvis. remember that i graduate december 10th 2026", GRAD),
    ("hey jarvis, put this in your memory: i graduate december 10th 2026", GRAD),
    # -- the second person on the way in
    ("jarvis remember i'm allergic to penicillin", "you're allergic to penicillin"),
    ("jarvis remember that i've moved to the third floor", "you've moved to the third floor"),
    ("jarvis remember that heather emailed me the form", "heather emailed you the form"),
    ("jarvis keep in mind that my thesis defence is in march", "your thesis defence is in march"),
    ("jarvis remember that i am on the third floor", "you are on the third floor"),
    ("jarvis remember that i was born in 2003", "you were born in 2003"),
    ("jarvis remember i'll be in austin next week", "you'll be in austin next week"),
    # -- a comma after the head is the rung's own pause, not a clause
    #    boundary: "remember, my X is Y" must not be split into an ask plus
    #    a people-book miss by _compound_hijack
    ("jarvis remember, my locker code is 4412", LOCKER),
]


@pytest.mark.parametrize("said,fact", MUST_STORE)
def test_must_store(cmdr, services, said, fact):  # noqa: F811
    res = cmdr.handle(said)
    key, value = _stored(services)
    assert value == fact
    assert key == " ".join(fact.split()[:6])
    assert res.handled
    # the spoken ack STATES the fact, and it is spoken
    assert res.reply == f"Noted, sir: {fact}."
    assert res.speak is True
    services.brain.think.assert_not_called()
    services.memory.save_note.assert_not_called()


@pytest.mark.parametrize("said,source,fact", [
    ("remember that i graduate december 10th 2026", "typed", GRAD),
    ("put that in your memory, i graduate december 10th 2026", "voice", GRAD),
    ("put it in your memory that i graduate december 10th 2026 with an electrical engineering degree",
     "typed", GRAD_FULL),
    ("please remember that my locker code is 4412", "voice", LOCKER),
    ("fyi, my locker code is 4412", "voice", LOCKER),
])
def test_must_store_without_the_wake_word(cmdr, services, said, source, fact):  # noqa: F811
    """The hotword eats the wake word, so the sentence arrives bare; by
    voice it must bypass the intent gate (a Tier-1 match is addressed to
    Jarvis by construction) rather than be asked 'Was that for me?'."""
    res = cmdr.handle(said, source=source)
    _key, value = _stored(services)
    assert value == fact
    assert res.reply == f"Noted, sir: {fact}." and res.speak is True
    assert res.status != "Was that for me?"
    services.brain.think.assert_not_called()


def test_whispers_casing_survives_and_the_pronoun_is_turned(cmdr, services):  # noqa: F811
    res = cmdr.handle("Jarvis, remember that I graduate December 10th 2026 "
                      "with an Electrical Engineering degree.")
    _key, value = _stored(services)
    assert value == "You graduate December 10th 2026 with an Electrical Engineering degree"
    assert res.reply == f"Noted, sir: {value}."


def test_a_contact_in_a_remember_is_still_a_contact(cmdr, services):  # noqa: F811
    cmdr.handle("jarvis remember that my advisor is Dr Peyrovi, email hp@tamu.edu")
    _key, value = _stored(services)
    assert value == "your advisor is Dr Peyrovi, email hp@tamu.edu"
    services.memory.add_person.assert_called_once_with(
        "advisor", "Dr Peyrovi", email="hp@tamu.edu")


def test_scratch_that_forgets_the_fact_just_stored(cmdr, services):  # noqa: F811
    cmdr.handle("jarvis remember that i graduate december 10th 2026")
    key, _value = _stored(services)
    res = cmdr.handle("jarvis scratch that")
    services.memory.forget.assert_called_once_with(key)
    assert res.handled and "forgot" in (res.reply or "").lower()


# ======================================================= MUST NOT STORE
# (a) a bare head: nothing to file, so he is asked -- never "Remembered:
#     that" and never a note called "down".
ASK_SHAPES = [
    "jarvis remember that",
    "jarvis remember this",
    "jarvis remember it",
    "jarvis remember",
    "jarvis remember that please",
    "jarvis remember that, would you",
    "jarvis please remember this",
    "jarvis keep in mind that",
    "jarvis keep in mind",
    "jarvis bear in mind that",
    "jarvis don't forget that",
    "jarvis put that in your memory please",
    "jarvis put this in your memory",
    "jarvis make a note of that",
    "jarvis make a note of this",
    "jarvis make a note of it",
    "jarvis note that",
    "jarvis note that down",
    "jarvis note that down please",
    "jarvis jot that down",
    "jarvis write that down",
    "jarvis memorise this",
    "jarvis commit this to memory",
    "jarvis for the record",
    "jarvis fyi",
    "jarvis store this",
]


@pytest.mark.parametrize("said", ASK_SHAPES)
def test_a_bare_head_asks_and_stores_nothing(cmdr, services, said):  # noqa: F811
    res = cmdr.handle(said)
    services.memory.remember.assert_not_called()
    services.memory.save_note.assert_not_called()
    services.brain.think.assert_not_called()
    assert res.handled and res.reply == ASK_LINE and res.speak is True


# (b) a question is a RECALL. The third column is the query the recall
#     rung must ask the store for (None: any query; "LIST": no query at
#     all -- the latest facts are read back).
RECALL_SHAPES = [
    ("jarvis remember when we went to austin?", "we went to austin"),
    ("jarvis remember when we fixed the radar?", "we fixed the radar"),
    ("jarvis remember what i said about the thesis", "the thesis"),
    ("jarvis remember what i told you about heather?", "heather"),
    ("jarvis remember how to get to wisenbaker?", None),
    ("jarvis remember where i parked?", None),
    ("jarvis remember who my advisor is?", None),
    ("jarvis remember why we moved the lab?", None),
    ("jarvis remember about my graduation", "my graduation"),
    ("jarvis do you remember my locker code", "my locker code"),
    ("jarvis do you remember my locker code?", "my locker code"),
    ("jarvis do you remember what i said about the thesis", "the thesis"),
    ("jarvis do you remember when we went to austin", "we went to austin"),
    ("jarvis do you remember that i graduate in december", "i graduate in december"),
    ("jarvis what did i tell you to remember about my graduation", "my graduation"),
    ("jarvis what did i ask you to remember about my graduation", "my graduation"),
    ("jarvis what did i tell you to remember", "LIST"),
    ("jarvis what did i say about my graduation", "my graduation"),
    ("jarvis what did i tell you about my degree", "my degree"),
    ("jarvis recall my graduation", "my graduation"),
    ("jarvis what did i say about my X", "my x"),
]


@pytest.mark.parametrize("said,query", RECALL_SHAPES)
def test_a_question_is_a_recall_not_a_fact(cmdr, services, said, query):  # noqa: F811
    services.memory.recall.return_value = []
    services.memory.get_all_facts.return_value = {}
    res = cmdr.handle(said)
    services.memory.remember.assert_not_called()
    services.memory.save_note.assert_not_called()
    services.brain.think.assert_not_called()
    assert res.handled
    if query == "LIST":
        services.memory.recall.assert_not_called()
        assert "nothing" in (res.reply or "").lower()
    elif query is None:
        services.memory.recall.assert_called_once()
    else:
        services.memory.recall.assert_called_once_with(query)


def test_what_did_i_tell_you_to_remember_reads_the_latest_facts_back(cmdr, services):  # noqa: F811
    services.memory.get_all_facts.return_value = {
        "your locker code is 4412": {"value": "your locker code is 4412", "time": "t1"},
        "you graduate december 10th 2026": {"value": "you graduate december 10th 2026", "time": "t2"},
    }
    res = cmdr.handle("jarvis what did i tell you to remember")
    assert res.handled and res.speak is True
    assert "you graduate december 10th 2026" in res.reply
    assert "your locker code is 4412" in res.reply


def test_the_recall_rung_finds_the_graduation_fact_by_its_natural_name(cmdr, services,  # noqa: F811
                                                                        tmp_path):
    """Through the rung, against a REAL substring store: 'graduation' is
    not a substring of 'you graduate ...', so this needs the stemmed
    fallback in memory.recall."""
    services.memory = JarvisMemory(memory_dir=tmp_path / "mem",
                                   legacy_dir=tmp_path / "legacy", semantic=False)
    cmdr.handle("jarvis put it in your memory that i graduate december 10th 2026 "
                "with an electrical engineering degree")
    for said in ("jarvis what did i tell you to remember about my graduation",
                 "jarvis what did i say about my graduation",
                 "jarvis what did i tell you about my degree",
                 "jarvis do you remember when i graduate",
                 "jarvis recall graduation"):
        res = cmdr.handle(said)
        assert res.handled and "december 10th 2026" in (res.reply or ""), said


# (c) the notes list keeps what it had (the 4fa27d7 pins and their kin):
#     "of <thing>", a bare colon, "jot down <thing>", "take a note ...".
NOTE_SHAPES = [
    ("jarvis make a note of milk", "milk"),
    ("jarvis make a note of milk and eggs", "milk and eggs"),
    ("jarvis make a note of the dentist appointment", "the dentist appointment"),
    ("jarvis note: buy milk", "buy milk"),
    ("jarvis make a note: buy milk", "buy milk"),
    ("jarvis jot down milk", "milk"),
    ("jarvis note down milk", "milk"),
    ("jarvis note buy milk", "buy milk"),
    ("jarvis write this down: milk", "milk"),
    ("jarvis take a note that the boiler is broken", "the boiler is broken"),
    ("jarvis take a note: the boiler is broken", "the boiler is broken"),
    ("jarvis note down that milk", "milk"),
]


@pytest.mark.parametrize("said,note", NOTE_SHAPES)
def test_a_note_is_still_a_note(cmdr, services, said, note):  # noqa: F811
    res = cmdr.handle(said)
    services.memory.remember.assert_not_called()
    services.brain.think.assert_not_called()
    assert res.handled
    services.memory.save_note.assert_called_once_with(note)


# (d) a to-do or a stray "remember" goes where it went before: the model.
MODEL_SHAPES = [
    "jarvis remember to call mum",
    "jarvis don't forget to call mum",
    "jarvis remember to buy milk",
    "jarvis keep in mind to lock the door",
    "jarvis remember i asked?",
    "jarvis i remember that day fondly",
    "jarvis remembering is hard",
    "jarvis remember me",
    "jarvis what should i remember",
]


@pytest.mark.parametrize("said", MODEL_SHAPES)
def test_not_a_fact_reaches_the_model_as_before(cmdr, services, said):  # noqa: F811
    res = cmdr.handle(said)
    services.memory.remember.assert_not_called()
    services.memory.save_note.assert_not_called()
    assert res.handled
    services.brain.think.assert_called_once()


# (e) the one-off heads-up IS stored today (a stated remaining item: it
#     wants a TTL'd store, not facts.json). Pinned so the day it changes,
#     it changes on purpose.
def test_a_heads_up_is_filed_as_a_fact_for_now(cmdr, services):  # noqa: F811
    cmdr.handle("jarvis keep in mind i'm on a call")
    _key, value = _stored(services)
    assert value == "you're on a call"


# ============================================== registry order and steals
def test_the_memory_family_sits_above_its_thieves():
    names = [c.name for c in REGISTRY]
    remember, recall = names.index("remember"), names.index("recall")
    assert names.index("person") < remember < recall
    for thief in ("suggest", "workflow", "quick command", "take note",
                  "run shell"):
        assert remember < names.index(thief), thief
        assert recall < names.index(thief), thief
    tier1 = [c.name for c in ASSISTANT_TIER1]
    assert "remember" in tier1 and "recall" in tier1


def test_suggest_still_answers_its_own_sentences(cmdr, services):  # noqa: F811
    services.memory.suggest_by_habit.return_value = "check gpu"
    res = cmdr.handle("jarvis any suggestions")
    assert res.handled and "check gpu" in res.reply
    services.memory.remember.assert_not_called()


@pytest.mark.parametrize("t,fires", [
    ("commit", True),
    ("please commit", True),
    ("commit now", True),
    ("commit, please", True),
    ("check gpu", True),
    ("commit this to memory: i graduate december 10th 2026", False),
    ("commit to memory that i graduate december 10th 2026", False),
    ("remember that the commit failed", False),
    ("what does check gpu do", False),
    ("check gpu and the disk", False),
])
def test_quick_command_triggers_are_whole_utterances(t, fires):
    """A QUICK_COMMANDS phrase is a shell string. At 7539478 the trigger
    was a bare substring test, so 'commit' inside a memory sentence ran
    `git add -A` (measured 17:21). Anchored now: the phrase IS the
    utterance, give or take a please."""
    assert bool(_m_quick_command(t)) is fires


def test_commit_this_to_memory_never_reaches_the_shell(cmdr, services,  # noqa: F811
                                                       desktop_attempts):
    res = cmdr.handle("jarvis commit this to memory: i graduate december 10th 2026")
    _key, value = _stored(services)
    assert value == GRAD and res.reply == f"Noted, sir: {GRAD}."
    assert desktop_attempts == [], desktop_attempts
    services.workflows.get.assert_not_called()


# ======================================================== the wake word
@pytest.mark.parametrize("text,rest", [
    ("Jarvis, commit.", "commit"),
    ("hey jarvis check gpu", "check gpu"),
    ("okay jarvis remember that x", "remember that x"),
    ("ok jarvis, check gpu", "check gpu"),
    ("Okay Jarvis, what time is it", "what time is it"),
    ("Jarvis. Remember that I graduate", "remember that i graduate"),
    ("Hey Jarvis. What time is it?", "what time is it?"),
    ("commit now", None),
    ("jarvis", None),
    ("Jarvis.", None),
    ("okay jarvis", None),
    ("jarvisrocks", None),
])
def test_strip_jarvis_prefix_knows_the_spellings_whisper_produces(text, rest):
    assert strip_jarvis_prefix(text) == rest


# ================================================== the store, for real
@pytest.fixture
def mem(tmp_path):
    return JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                        semantic=False)


@pytest.mark.parametrize("said,filed", [
    ("i graduate december 10th 2026", "you graduate december 10th 2026"),
    ("I graduate December 10th 2026", "You graduate December 10th 2026"),
    ("i'm allergic to penicillin", "you're allergic to penicillin"),
    ("I’ve moved to the third floor", "You've moved to the third floor"),
    ("i'll be late", "you'll be late"),
    ("i'd rather email", "you'd rather email"),
    ("i am on a call", "you are on a call"),
    ("i was born in 2003", "you were born in 2003"),
    ("heather emailed me the form", "heather emailed you the form"),
    ("Heather said I should email her", "Heather said you should email her"),
    ("my locker code is 4412", "your locker code is 4412"),
    ("My locker code is 4412", "Your locker code is 4412"),
    ("it is mara and i's anniversary on september 20th",
     "it is mara and your anniversary on september 20th"),
    ("the meeting is at nine", "the meeting is at nine"),
    ("Dr Peyrovi is my advisor", "Dr Peyrovi is your advisor"),
    ("the iphone is mine", "the iphone is yours"),
])
def test_speech_to_second_person(said, filed):
    assert speech_to_second_person(said) == filed
    assert speech_to_second_person(filed) == filed, "must be idempotent"


def test_store_fact_from_speech_files_the_second_person_under_a_six_word_key(mem):
    stored = store_fact_from_speech(
        mem, "I graduate December 10th 2026 with an electrical engineering degree")
    assert stored == "You graduate December 10th 2026 with an electrical engineering degree"
    # the key is the case-folded head of the value (see (h) below)
    assert mem.get_all_facts() == {
        "you graduate december 10th 2026 with": {"value": stored, "time": ANY}}
    # the method form is the same helper (the remember TOOL will call it)
    assert mem.store_fact_from_speech("my locker code is 4412") == "your locker code is 4412"
    assert "your locker code is 4412" in mem.get_all_facts()


def test_store_fact_from_speech_cleans_the_value_first(mem):
    assert store_fact_from_speech(mem, "that, i graduate december 10th 2026 please.") == \
        "you graduate december 10th 2026"
    assert store_fact_from_speech(mem, ": my locker code is 4412, thanks!") == \
        "your locker code is 4412"


def test_the_rendered_line_for_when_do_i_graduate(mem):
    """MEASURED here, against the real store and the real renderer: the
    fact must read as a fact about HIM under 'Known facts', once, not
    'key: value' with the key a truncated copy of the value."""
    store_fact_from_speech(
        mem, "i graduate december 10th 2026 with an electrical engineering degree")
    line = mem.format_for_context("When do I graduate?")
    assert line == ("Known facts (1):\n"
                    "  you graduate december 10th 2026 with an electrical engineering degree")
    assert build_user_turn("", line, "When do I graduate?") == (
        "Background:\n"
        "Known facts (1):\n"
        "  you graduate december 10th 2026 with an electrical engineering degree\n"
        "\n"
        "Hunter: When do I graduate?")


def test_a_keyed_fact_still_renders_key_and_value(mem):
    """The garden, the debrief and the tool file under a short key; those
    keep the 'key: value' line -- the key carries meaning there."""
    mem.remember("gpu", "GB10 unified memory")
    # ...and a fact the OLD rung filed (key = the first 60 chars of the
    # value) stops doubling too: the key is a truncation, not a name.
    old = "i graduate december 10th 2026 with an electrical engineering degree"
    mem.remember(old[:60], old)
    text = mem.format_for_context()
    assert "  gpu: GB10 unified memory" in text
    assert text.count("december 10th 2026") == 1
    assert "  i graduate december 10th 2026 with an electrical engineering degree" in text


@pytest.mark.parametrize("query", [
    "my graduation", "graduation", "graduating", "my degree",
    "when do i graduate", "your graduation",
])
def test_recall_finds_the_graduation_fact_by_stem(mem, query):
    store_fact_from_speech(
        mem, "i graduate december 10th 2026 with an electrical engineering degree")
    hits = mem.recall(query)
    assert hits and hits[0]["value"].startswith("you graduate december 10th 2026"), query


def test_the_stem_fallback_does_not_invent_hits(mem):
    store_fact_from_speech(
        mem, "i graduate december 10th 2026 with an electrical engineering degree")
    for query in ("play some jazz", "my dentist", "the", "what did i say"):
        assert mem.recall(query) == [], query


def test_the_stem_fallback_respects_since(mem):
    from datetime import datetime, timedelta
    store_fact_from_speech(mem, "i graduate december 10th 2026")
    assert mem.recall("graduation", since=timedelta(hours=1))
    assert mem.recall("graduation", since=datetime.now() + timedelta(hours=1)) == []


# ======================================= the three holes measured on 1641fb3
# Reviewer + verdict, 2026-09-05, through Commander.handle under the firewall
# fixtures: (f) 11 of 44 must-not sentences stored a non-fact; (g) a
# comma-less "X and Y" fact stored only X; (h) the same fact re-said in
# Whisper's other casing was stored and rendered twice.

# (f) a value that only POINTS at something already said ("for later",
#     "as well", "everything i just said", "the name i just gave you",
#     "i said that", "that time we went to austin") or is only a courtesy
#     ("thanks a lot") is not a fact: the head is bare and he is asked,
#     exactly as "remember that" asks. Measured: each of the first eleven
#     stored an entry and spoke "Noted, sir: for later." / "... as well." /
#     "... everything you just said." -- the two-word floor was the only
#     guard after the head.
POINTER_SHAPES = [
    # the reviewer's eleven, verbatim
    "jarvis remember that for later",
    "jarvis remember that for next time",
    "jarvis remember all of that",
    "jarvis remember everything i just said",
    "jarvis remember the last thing i said",
    "jarvis remember that as well",
    "jarvis remember that, thanks a lot",
    "jarvis remember that, thank you very much",
    "jarvis remember the name i just gave you",
    "jarvis remember that time we went to austin",
    "jarvis remember that i said that",
    # ...and the smallest generalisation of them
    "jarvis remember it for later",
    "jarvis remember this for next time",
    "jarvis remember that too",
    "jarvis remember that as well, please",
    "jarvis remember that for when i ask",
    "jarvis remember that when i ask you",
    "jarvis remember that for future reference",
    "jarvis remember all of this, please",
    "jarvis remember that, thanks",
    "jarvis remember that, thanks so much",
    "jarvis remember everything i told you",
    "jarvis remember the number i just gave you",
    "jarvis remember the last thing i told you",
    "jarvis remember that i just said that",
    "jarvis keep in mind everything i said",
    "jarvis don't forget that as well",
    "jarvis put that in your memory for later",
    "jarvis note that for later",
]


@pytest.mark.parametrize("said", POINTER_SHAPES)
def test_a_pointer_at_something_already_said_asks_and_stores_nothing(cmdr, services, said):  # noqa: F811
    res = cmdr.handle(said)
    services.memory.remember.assert_not_called()
    services.memory.save_note.assert_not_called()
    services.brain.think.assert_not_called()
    assert res.handled and res.reply == ASK_LINE and res.speak is True


# ...whole-value only: a fact that CONTAINS one of those phrases is a fact.
@pytest.mark.parametrize("said,fact", [
    ("jarvis remember that thank you notes go out on friday",
     "thank you notes go out on friday"),
    ("jarvis remember that i said that i'd be late on friday",
     "you said that you'd be late on friday"),
    ("jarvis remember that the time we agreed on is 3pm",
     "the time we agreed on is 3pm"),
    ("jarvis remember that everything i said in the meeting is confidential",
     "everything you said in the meeting is confidential"),
    ("jarvis remember that as well as the thesis i have a lab report due",
     "as well as the thesis you have a lab report due"),
])
def test_a_fact_that_contains_a_pointer_phrase_is_still_a_fact(cmdr, services, said, fact):  # noqa: F811
    res = cmdr.handle(said)
    _key, value = _stored(services)
    assert value == fact
    assert res.reply == f"Noted, sir: {fact}."


# (g) a fact runs to the END of the utterance, courtesy tail aside: a
#     coordinating "and" inside it is part of the fact, never a clause
#     boundary. Measured: "remember that my locker code is 4412 and my
#     parking spot is b14" stored ONLY "your locker code is 4412" -- the
#     tail "my parking spot is b14" matched the person rung's shape, so
#     _compound_hijack split the breath, the person rung declined the tail,
#     and it vanished: no store, no note, no model turn. With a comma before
#     the "and" the fact was already kept whole.
TWO_CLAUSE_SHAPES = [
    ("jarvis remember that my locker code is 4412 and my parking spot is b14",
     "your locker code is 4412 and your parking spot is b14"),
    ("jarvis remember my locker code is 4412 and my parking spot is b14",
     "your locker code is 4412 and your parking spot is b14"),
    ("jarvis remember that i graduate december 10th 2026 and my thesis defence is in march",
     "you graduate december 10th 2026 and your thesis defence is in march"),
    ("jarvis keep in mind that heather prefers email and i prefer the phone",
     "heather prefers email and you prefer the phone"),
    ("jarvis don't forget that the lab moved to room 049 and the printer is on the third floor",
     "the lab moved to room 049 and the printer is on the third floor"),
    ("jarvis remember that my dentist is dr patel and my doctor is dr lee",
     "your dentist is dr patel and your doctor is dr lee"),
    # the courtesy tail still comes off the end of the whole breath
    ("jarvis remember that my locker code is 4412 and my parking spot is b14, please",
     "your locker code is 4412 and your parking spot is b14"),
    # the two controls the reviewer measured whole already
    ("jarvis remember that my locker code is 4412, and my parking spot is b14",
     "your locker code is 4412, and your parking spot is b14"),
    ("jarvis remember that i like jazz and heather likes blues",
     "you like jazz and heather likes blues"),
]


@pytest.mark.parametrize("said,fact", TWO_CLAUSE_SHAPES)
def test_a_two_clause_fact_is_stored_whole_in_the_second_person(cmdr, services, said, fact):  # noqa: F811
    res = cmdr.handle(said)
    key, value = _stored(services)
    assert value == fact
    assert key == " ".join(fact.split()[:6])
    assert res.reply == f"Noted, sir: {fact}." and res.speak is True
    services.memory.save_note.assert_not_called()
    services.brain.think.assert_not_called()
    services.memory.add_person.assert_not_called()


def test_a_remember_and_a_command_in_one_breath_is_one_fact(cmdr, services):  # noqa: F811
    """The stated trade-off of (g): a remembered fact is one breath, the
    rule the code already states for 'remind me to buy milk and eggs', so
    a command after the 'and' is filed inside the fact rather than run.
    Pinned so the day it changes, it changes on purpose."""
    res = cmdr.handle("jarvis remember that my locker code is 4412 and set a timer for five minutes")
    _key, value = _stored(services)
    assert value == "your locker code is 4412 and set a timer for five minutes"
    assert res.reply == f"Noted, sir: {value}."


# (h) the key is built from the CASE-FOLDED value: Whisper's capitals vary
#     between takes, and the same fact re-said was stored under two keys
#     and rendered twice in every prompt (measured: 'Known facts (2):'
#     with both casings). Replace, never double; the value keeps the
#     casing of the latest take and the ack is unchanged.
def test_fact_key_is_case_folded():
    assert fact_key("You graduate December 10th 2026 with an EE degree") == \
        "you graduate december 10th 2026 with"
    assert fact_key("you graduate december 10th 2026") == "you graduate december 10th 2026"
    assert fact_key("Your Locker Code IS 4412") == "your locker code is 4412"


def test_store_fact_from_speech_in_two_casings_is_one_entry(mem):
    store_fact_from_speech(
        mem, "I graduate December 10th 2026 with an electrical engineering degree")
    store_fact_from_speech(
        mem, "i graduate december 10th 2026 with an electrical engineering degree")
    facts = mem.get_all_facts()
    assert list(facts) == ["you graduate december 10th 2026 with"]
    assert facts["you graduate december 10th 2026 with"]["value"] == \
        "you graduate december 10th 2026 with an electrical engineering degree"


def test_the_same_fact_in_two_casings_renders_once_through_the_rung(cmdr, services, tmp_path):  # noqa: F811
    """MEASURED against the real store and renderer, through the rung."""
    services.memory = JarvisMemory(memory_dir=tmp_path / "mem",
                                   legacy_dir=tmp_path / "legacy", semantic=False)
    first = cmdr.handle("Jarvis, remember that I graduate December 10th 2026 "
                        "with an electrical engineering degree.")
    assert first.reply == ("Noted, sir: You graduate December 10th 2026 "
                           "with an electrical engineering degree.")
    second = cmdr.handle("jarvis remember that i graduate december 10th 2026 "
                         "with an electrical engineering degree")
    assert second.reply == ("Noted, sir: you graduate december 10th 2026 "
                            "with an electrical engineering degree.")
    assert list(services.memory.get_all_facts()) == ["you graduate december 10th 2026 with"]
    assert services.memory.format_for_context("When do I graduate?") == (
        "Known facts (1):\n"
        "  you graduate december 10th 2026 with an electrical engineering degree")
    # ...and "scratch that" still finds the entry under the folded key
    res = cmdr.handle("jarvis scratch that")
    assert res.handled and "forgot" in (res.reply or "").lower()
    assert services.memory.get_all_facts() == {}
