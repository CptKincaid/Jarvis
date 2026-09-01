"""2026-08-31: "Jarvis gets monotone with long sentences or lists."

Not the voice: the CHUNKER. ``_MAX_CHUNK_CHARS`` (160) and ``_CHUNK_GROWTH``
(2.5) were fitted to XTTS at RTF ~0.29, and the second pass they drive cuts
a sentence apart at its COMMAS. Each piece is then a separate utterance to
the engine, so the pitch resets in the middle of a clause.

Measured against the live F5 sidecar, 2026-08-31, on his own calendar line
(autocorrelation F0):

    chunk 2 alone   head 128 Hz  tail 126 Hz  F0 IQR 17.3   flat
    chunk 3 alone   head 171 Hz  tail 120 Hz  F0 IQR 35.8   restarts +45 Hz
    whole sentence  head 125 Hz  tail 100 Hz  F0 IQR 22.7   declines

Splitting throws away the sentence's declination, flattens the first half,
and jumps mid-clause into the second.

And F5 never needed the split. Measured on the resident sidecar the same
day: 68 chars -> 0.42 s wall / 4.09 s audio (RTF 0.103); 162 chars -> 0.61 s
/ 9.74 s (RTF 0.063). Its break-even growth is ~11x, not XTTS's 3.44x, and a
whole 162-character sentence renders in 0.61 s.
"""
import re

import pytest

from jarvis.tts import TTS

# Verbatim from his 2026-08-31 log (20:01, 21:01, 22:14).
HIS_MONDAY = ("You have four items on Monday, sir. There is BIOSENSORS at "
              "9:10 am, MAGNETIC RESONANCE ENGR at 12:40 pm, an ELECTRICAL "
              "DESIGN LAB II presentation at 4:10 pm, and a BMEN 427 lab "
              "due Saturday.")
HIS_AGENDA = ("You have BIOSENSORS at 9:10 am, MAGNETIC RESONANCE ENGR at "
              "12:40 pm, an ELECTRICAL DESIGN LAB II presentation at 4:10 "
              "pm, and a BMEN 427 lab due at midnight, sir.")

_ENDS_A_SENTENCE = re.compile(r'[.!?;:]["\')\]]*$')


def cut_mid_sentence(chunks):
    """Chunks that do not end on sentence punctuation -- i.e. a sentence
    handed to the engine in pieces. The last one is exempt: it ends where
    the reply ends."""
    return [c for c in chunks[:-1] if not _ENDS_A_SENTENCE.search(c.strip())]


def split(engine, text):
    """The chunks the room would actually render for ``text``.

    render_chunks, not _split_sentences: the pronunciation pass runs first
    and it makes his calendar lines LONGER ("12:40 pm" -> "twelve forty pee
    em"), so measuring the raw text understates the splitting. cache=False
    -- a text test must never touch the cache the live Jarvis plays from."""
    tts = TTS(engine=engine, cache=False)
    return tts.render_chunks(text)


@pytest.mark.parametrize("line", [HIS_MONDAY, HIS_AGENDA])
def test_f5_never_cuts_one_of_his_list_sentences_at_a_comma(line):
    chunks = split("f5", line)
    assert not cut_mid_sentence(chunks), (
        "his calendar list is still handed to F5 in mid-clause pieces, and "
        "the pitch resets at each cut -- that is the monotone he heard: "
        f"{chunks}")


def test_the_sentence_boundary_is_still_a_chunk_boundary():
    """Splitting is not switched off -- pipelining is what keeps the first
    reply quick. Sentences still start new chunks; clauses no longer do."""
    chunks = split("f5", HIS_MONDAY)
    assert len(chunks) == 2, chunks
    assert chunks[0] == "You have four items on Monday, sir."


def test_the_old_limits_did_cut_his_line_apart():
    """The regression this is fixing, kept as evidence: the numbers XTTS was
    measured for are the ones that chopped the list."""
    assert cut_mid_sentence(split("xtts", HIS_MONDAY)), (
        "if XTTS no longer splits this line the F5 fix is being credited to "
        "the wrong change")


@pytest.mark.parametrize("engine", ["xtts", "edge", "fish"])
def test_the_measured_engines_keep_their_own_measured_numbers(engine):
    """Only F5 was re-measured. Nothing else moves on its say-so."""
    tts = TTS(engine="edge", cache=False)
    assert tts._chunk_limits(engine) == (TTS._MAX_CHUNK_CHARS,
                                         TTS._CHUNK_GROWTH)


def test_f5_still_has_a_ceiling():
    """A runaway paragraph must still be broken up, or the producer falls
    behind the player and playback runs dry."""
    max_chars, growth = TTS(engine="f5", cache=False)._chunk_limits("f5")
    assert max_chars < TTS.MAX_SPEAK_LENGTH
    assert growth > TTS._CHUNK_GROWTH
    runaway = "word, " * 200
    assert len(split("f5", runaway)) > 1
