"""THE MEASURED COST of consuming the phrase on every turn (Knightfall,
2026-09-04). Numbers, not a feeling: how many key derivations 100 ordinary
recognised turns pay, before and after, and what one costs on this box.

BEFORE (jarvis-v3 at 2a9170c): ``_try_phrase`` ran only on HOW_NOBODY, so
a recognised owner paid ZERO derivations however long his sentences were.
AFTER: every phrase-shaped sentence (>= 12 letters and digits once
punctuation is dropped, the unchanged pre-filter) pays one scrypt per owner
with a phrase set; anything shorter still pays nothing. The count is
asserted exactly so a future "cheap bound" can be seen to change it, and
the wall time is printed (``-s``) and held under a loose ceiling so a
regression to a heavier scrypt cannot slip in unseen.

The corpus is a hundred of the things he actually says to it, taken from
the "what you can say" lists in docs/assistant-setup.md.
"""
import time

from jarvis import passphrase as pp
from tests.test_owner_gate import MATCHED, _gate

CORPUS = [
    "what time is it", "what's the date today", "what day is it",
    "what's the weather", "will it rain tomorrow", "how cold is it outside",
    "yes", "no", "pause", "stop", "skip", "louder", "quieter", "mute",
    "play some jazz", "next track", "what's playing", "resume the music",
    "set a timer for ten minutes", "wake me at seven", "cancel the alarm",
    "remind me to call mom at six", "what's on my calendar",
    "what's on tomorrow", "when's my next exam", "any new grades",
    "read me my mail", "anything from my advisor", "email the lab report to heather",
    "add milk to the shopping list", "what's on the shopping list",
    "remember that my dentist is doctor lee", "who is my advisor",
    "note that the lecture moved to room two hundred", "recap my day",
    "how much did I study this week", "start a study session",
    "quiz me on chapter four", "teach me eigenvalues", "explain this document",
    "read it to me", "go on", "back", "skip ahead", "stop reading",
    "how are you", "run diagnostics", "formal mode", "good night",
    "good morning", "how did yesterday go", "what did I do today",
    "how's the haymaker bot", "is the server up", "what about monday sync",
    "check the logs", "bring up the board", "hide the board",
    "switch to classic visuals", "use the holographic look",
    "offline mode", "come back online", "camera off", "is anyone home",
    "where's my phone", "send that to hpcomputer", "grab the file",
    "open claude's terminal", "commit this to memory", "scratch that",
    "no I said tuesday", "that was for you", "say that again",
    "what's forty two times seven", "convert five miles to kilometres",
    "how long until my next class", "when do I need to leave",
    "dim the room", "lights up", "set the scene to focus", "volume of my music down",
    "what's the gpu doing", "anything wrong in your log", "why was that slow",
    "how's the spark", "restart yourself", "start at login",
    "every forty five minutes remind me to stretch", "do not disturb until three",
    "quiet hours on", "what's the news", "look up the capital of peru",
    "who won the game last night", "tell me a joke", "thank you",
    "never mind", "that's all", "hello jarvis", "are you there",
    "what's the temperature in college station",
]
assert len(CORPUS) == 100, len(CORPUS)


def _run(g, now0=0.0):
    for i, said in enumerate(CORPUS):
        d = g.judge("voice", said, stats=MATCHED, now=now0 + float(i))
        assert d.admit is True and d.consumed is False, said


def test_kdf_calls_per_100_ordinary_recognised_turns(tmp_path):
    before = 0        # jarvis-v3: _try_phrase only on HOW_NOBODY
    shaped = sum(1 for s in CORPUS if pp.phrase_shaped(s))
    g = _gate(tmp_path, mode="shadow", phrase=True)
    _run(g)
    after = g.kdf_calls
    print("\nkdf_calls per 100 ordinary recognised turns: before=%d after=%d "
          "(phrase-shaped sentences in the corpus: %d)" % (before, after, shaped))
    assert after == shaped
    assert 80 <= after <= 100, "the corpus should be mostly phrase-shaped"
    # and none of them was an ATTEMPT
    assert g.phrase_attempts._hits == []


def test_a_turn_with_no_phrase_set_pays_nothing_at_all(tmp_path):
    g = _gate(tmp_path, mode="shadow", phrase=False)
    _run(g)
    assert g.kdf_calls == 0


def test_the_wall_cost_of_one_derivation_on_this_box(tmp_path):
    """Printed for the report; the ceiling only catches a heavier scrypt."""
    g = _gate(tmp_path, mode="shadow", phrase=True)
    t0 = time.perf_counter()
    _run(g)
    per_turn = (time.perf_counter() - t0) / len(CORPUS)
    stored = pp.hash_secret("a measured sentence of ordinary length")
    t1 = time.perf_counter()
    for _ in range(10):
        pp.check_secret("another sentence of about the same size", stored)
    per_kdf = (time.perf_counter() - t1) / 10
    print("\nmean per ordinary turn through judge(): %.1f ms; "
          "one scrypt n=2**%d: %.1f ms" % (per_turn * 1e3,
                                           pp.SCRYPT_N.bit_length() - 1,
                                           per_kdf * 1e3))
    assert per_kdf < 0.2, "scrypt got heavier than the measured 20 ms class"
