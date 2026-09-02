r"""Regression tests: the local model's reasoning scaffolding was READ ALOUD.

LIVE 2026-09-02, /tmp/vss_voice/jarvis.log lines 6724 and 6729 -- Hunter
asked "say hi to my family and then give me my daily briefing":

    14:43:59.323 jarvis.tts   INFO speaking (f5): thought <channel, >Good
                 afternoon, Ali and Heather; I do hope you're both having
                 a lovely day.
    14:44:00.864 jarvis.brain INFO chat reply (3.94s wall, ...): thought
                 <channel, >Good afternoon, Ali and Heather; ...

That is a control token spoken to the room. The mangled shape is this
repo's own doing and it is worth reading twice, because it is why a later
scrub cannot be the only defence: ``strip_markdown`` turns a markdown
table's cell pipes into clause breaks --

    clean = re.sub(r'[ \t]*\|[ \t]*', ', ', clean)

-- so the model's ``thought\n<channel|>`` reaches the log as
``thought <channel, >``. Reproduced byte-for-byte on a941dfa:

    _finish_spoken("thought\n<channel|>Good afternoon, ...", "", "", 2)
        -> "thought <channel, >Good afternoon, ..."

Two layers, and the order matters:

1. THE REQUEST. ``brain._chat_payload`` already sends ``think: false`` on
   every /api/chat call (chat, the tool loop, the stream, classify,
   summarize, the warm-up) -- the same field ``jarvis/tools/screen.py``
   needed for gemma4 on the vision path. That is the only request-level
   lever there is, so the test below pins it: a future edit must not drop
   it and leave the scrub carrying the whole load on its own. A ``stop``
   string cannot help here -- the scaffolding arrives BEFORE the words, so
   stopping on it would discard the reply, not the rubbish.

2. THE SCRUB. think:false demonstrably did not hold on 2026-09-02, and
   this is a class of bug whose failure mode is, by definition, garbage
   read aloud; a defensive pass costs one regex per reply. It runs FIRST
   inside ``clean_ollama_reply`` -- i.e. on the RAW text, before
   ``strip_markdown`` mangles the pipes -- which is the only place the
   token is still recognisable.

The scrub must not eat words. "I thought so, sir" is the standing
counter-example and is asserted below.
"""
import pytest

from jarvis.brain import (_chat_payload, _finish_spoken, clean_ollama_reply,
                          strip_model_scaffolding)


@pytest.fixture(autouse=True)
def _afternoon(monkeypatch):
    """The live line was spoken at 14:43 and the assertions are on it
    verbatim. brain.ground_greeting reads the wall clock (see
    tests/test_found_stale_greeting.py), so the hour is pinned here rather
    than left to whenever the suite runs."""
    from jarvis import arc as arc_mod
    monkeypatch.setattr(arc_mod, "greeting_word", lambda now=None: "afternoon")


# ---------------------------------------------------------------- layer 1
def test_every_chat_request_suppresses_thinking():
    """think:false on the payload every /api/chat path builds."""
    payload = _chat_payload([{"role": "user", "content": "hi"}])
    assert payload["think"] is False


# ---------------------------------------------------------------- layer 2
def test_the_live_line_is_not_spoken():
    """The exact 14:43:59 reply, from the raw text that produced it."""
    spoken = _finish_spoken(
        "thought\n<channel|>Good afternoon, Ali and Heather; I do hope "
        "you're both having a lovely day.", "", "", 2)
    assert spoken == ("Good afternoon, Ali and Heather; I do hope you're "
                      "both having a lovely day.")


@pytest.mark.parametrize("raw", [
    "<|channel|>analysis<|message|>He wants a greeting.<|end|>Good afternoon, sir.",
    "<|start|>assistant<|channel|>final<|message|>Good afternoon, sir.",
    "<think>He wants a greeting.</think>Good afternoon, sir.",
    "<thinking>\nHe wants a greeting.\n</thinking>\nGood afternoon, sir.",
    "<start_of_turn>model\nGood afternoon, sir.<end_of_turn>",
    "thought\n<channel|>Good afternoon, sir.",
    "<|im_start|>assistant\nGood afternoon, sir.<|im_end|>",
])
def test_scaffolding_variants_never_reach_tts(raw):
    """Not only the one string in the log: every shape of the same leak."""
    spoken = _finish_spoken(raw, "", "", 2)
    assert spoken == "Good afternoon, sir."


@pytest.mark.parametrize("line", [
    "I thought so, sir.",
    "I thought as much, sir; the branch was never pushed.",
    "Your final thought of the day, sir.",
    "The channel is quiet, sir.",
    "That analysis is a day old, sir.",
    "Nothing to think about, sir.",
])
def test_ordinary_words_survive_intact(line):
    """The guard is for control tokens, never for vocabulary."""
    assert strip_model_scaffolding(line) == line
    assert _finish_spoken(line, "", "", 2) == line


def test_a_reply_that_is_only_scaffolding_becomes_empty():
    """Nothing left to say is better than saying the tokens: chat() turns
    an empty reply into MODEL_EMPTY_LINE, which is an honest sentence."""
    assert strip_model_scaffolding("<|channel|>analysis<|message|>") == ""


def test_scrub_runs_before_the_markdown_pass():
    """clean_ollama_reply sees raw text (spoken_from_ollama's order), so
    the token is still a token. Pinned because the pipe rule in
    strip_markdown destroys the evidence."""
    assert "<" not in clean_ollama_reply("thought\n<channel|>Good afternoon.")


def test_a_real_markdown_table_still_becomes_clause_breaks():
    """The scrub must not cost the table rendering it sits in front of
    (test_found_guard_ordering pins the same behaviour)."""
    spoken = _finish_spoken("Here you are, sir:\n| Wed | 85 |", "", "", 2)
    assert "|" not in spoken
    assert "Wed, 85" in spoken


def test_a_streamed_sentence_is_scrubbed_before_it_is_spoken(monkeypatch):
    """The 14:43:59 line reached TTS 1.5 s before the reply was complete,
    so the whole-reply guard was never going to catch it in time."""
    from tests.test_streaming_replies import _brain, _chunks

    b = _brain(monkeypatch,
               _chunks("thought\n<channel|>Good afternoon, sir. All is quiet."))
    spoken = []
    b._chat_sync("say hi", on_sentence=spoken.append)
    assert spoken == ["Good afternoon, sir.", "All is quiet."]
