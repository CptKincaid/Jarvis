"""Letters said one at a time — the rule, and the reassembly.

WHY THIS EXISTS (Hunter, 2026-09-05 12:25): "i did the email test where i
spelled out the name and he made it weirdish". He spelled an address out
character by character. The capture was chopped into FOUR separate turns
inside one intended sentence, the first three fragments were thrown away,
and the fourth — a headless tail — was all that ever reached the
commander, so the uncertain-intent card fired and from his side Jarvis
simply went strange.

The endpointer stops a capture ``CONFIG.endpoint_silence`` (0.8 s) after
the VAD last heard speech. A man spelling leaves a real pause between
every character, and the filler hold cannot save him because he is saying
letters, not "um". MEASURED on a scripted VAD at the shipped 0.8 s with
0.99 s between characters (tests/test_spelling_hold.py): the capture ends
after the FIRST character, 0.83 s into the first gap, and the remaining
six become other captures. Spelling is structurally incompatible with a
0.8 s endpoint.

Raising ``endpoint_silence`` is the wrong fix and was rejected before any
code was written: it lengthens EVERY turn, and the turn wait is the number
he fought from 10.7 s down to 1.3 s. So the hold is conditional, and this
module is the condition.

THE RULE, in one sentence
-------------------------
Text is **mid-spelling** when its tail is a SPELLING RUN — two or more
single-character tokens in a row, however whisper punctuates them
("q-z-v", "q. z. v", "Q.Z.V", "q z v", and "q-z-v, k-b-w" across the comma
it puts between groups) — or a single bare LETTER left dangling after a
closed-class word that cannot end an English sentence ("send an email
to q").

WHY IT CANNOT HOLD AN ORDINARY SENTENCE OPEN
--------------------------------------------
Three separate reasons, each of which alone would do:

* **A run needs two LETTERS, not two characters.** This is the sentence
  that was WRONG IN WRITING until 09-05. It used to read "only 'a' and
  'I' are one-letter English words, so no ordinary sentence ends on two
  single characters in a row" — and that argument is false, because
  DIGITS were counted as run characters. "gate 4 b", "row 2 a",
  "channel 5 c", "unit 2 d" and "he got a c" are all runs of two under
  it, and the verdict MEASURED them being held: 11 of 20 false holds on
  a 67-phrase corpus, each costing +1792 ms once on that turn. So a run
  now needs at least TWO of its single characters to be letters other
  than "a" and "i". A digit may still ride INSIDE a run — k-b-w-7 is a
  spelled string, not a label — the floor is on the run, not on every
  character in it.
* **The dangling case needs a LETTER.** A digit never dangles, so "set a
  timer for 5" and "volume to 5" are ordinary turns; and "a" and "I" are
  excluded by name, so "that is a" is too.
* **The dangling case needs a function word in front of it.** The list
  below is closed-class words that cannot be the last word of a sentence.
  "plan", "option", "track" and "vitamin" are nouns, so "plan B",
  "option A", "track B" and "vitamin C" are never spelling — the letter
  labels the noun, it does not continue a name.

The cost of being wrong in each direction is the same asymmetry the
filler hold was built on: a wrong hold costs one ``spell_hold_s`` once,
on one turn; a missed one chops his address into four pieces.

WHAT THE NARROWING DELIBERATELY DOES NOT REMOVE (Hunter, 09-05)
---------------------------------------------------------------
After the digit repair, 9 of the 20 measured false holds remain, and
every one of them is the DANGLING-LETTER case: a phrase that ends on one
bare letter after a function word ("switch to b", "the answer is c").
**He ruled to keep them.** He accepts about 1.8 s occasionally on such a
phrase, because that same rule is what buys him the FIRST character when
he starts spelling an address, and being chopped off mid-address is worse
than a pause. That is a DECISION, not a defect: it is not to be removed
or watered down without asking him again.

THE REASSEMBLY
--------------
Holding the mic open is only half of it — the letters then have to become
what he meant. :func:`fold` collapses a spelling run into one word
("q-z-v at example dot com" -> "qzv at example dot com") so
``jarvis.outbox``'s address parser, which already reads a SAID address,
reads a SPELLED one with no new grammar. :func:`fold_spans` returns the
same string with a character-by-character map back into the raw text, so
``outbox.address_span`` keeps returning a span in the text AS GIVEN and
the commander can still cut an address out of a sentence it never folded.

Folding is not cosmetic. Before it, the parser read "q z v at example dot
com" as the local part **"v"** — a different, possibly real address, and
the one class of mistake a read-back is least likely to catch, because it
sounds almost right.

Nothing in this module opens a device, and nothing in it is allowed to
keep the characters: the recorder stores only HOW MANY there were.
"""
from __future__ import annotations

import re

# A run needs two. One single character is a stray letter in an ordinary
# sentence ("what is plan B") far more often than it is a man spelling,
# and his pin says so explicitly.
MIN_RUN = 2

# ...and two of the run's characters must be LETTERS that are not "a" or
# "i". MEASURED (the 09-05 verdict, 67 realistic non-spelling phrases):
# counting digits as run characters held "gate 4 b", "row 2 a",
# "channel 5 c", "unit 2 d" and "he got a c" -- 11 of 20 false holds, at
# +1792 ms each, once per turn. In every one of them the letter LABELS
# the number; none of them is a man spelling. The floor is on the run as
# a whole, so a digit inside a spelled string ("k-b-w-7") still rides.
MIN_RUN_LETTERS = 2

# Closed-class words that cannot be the last word of an English sentence.
# A single bare letter behind one of these is a name being spelled, not a
# label: "send an email to q" is his own first fragment. Deliberately
# short — every word added here is a turn that may wait spell_hold_s for
# nothing. Nouns are never in it, which is what keeps "plan B" out.
DANGLING_WORDS = frozenset({
    "to", "at", "for", "with", "from", "of", "in", "on", "by", "into",
    "onto", "and", "or", "the", "a", "an", "is", "are", "was", "were",
    "be", "as", "called", "named", "spell", "spells", "spelled",
    "spelling", "letter", "letters",
})

# The two one-letter English words. They may sit inside a run and be
# folded with it (a spelled name is full of them), but neither COUNTS
# toward MIN_RUN_LETTERS and neither may be the dangling letter that
# starts a run: "that is a" and "the answer is I" are finished sentences,
# and "he got a c" is a grade.
_NOT_A_DANGLING_LETTER = frozenset({"a", "i"})

# What separates two spoken characters in a transcript. Whisper writes the
# same run four ways in one session — hyphens, full stops, commas, or bare
# spaces — and an em/en dash whenever it feels literary.
_SEP = r"(?:[ \t]*[-–—.,][ \t]*|[ \t]+)"
_CHAR = r"[A-Za-z0-9]"
# "@" is in both boundary classes on purpose: a TYPED address must never be
# folded. "a.b@example.com" would otherwise read as a run and be quietly
# rewritten to "ab@example.com" — a different mailbox, from a fold that was
# only ever meant to help a SPOKEN one.
_EDGE = r"[A-Za-z0-9_'@]"
# The trailing ",?" is whisper's own punctuation, not his: it writes a comma
# between two spelled groups AND after the last one ("q-z-v, k-b-w-7, at
# example dot com" -- the exact shape of the fourth fragment he was left
# with on 09-05). The comma before "at" broke the address parser, which
# wants the local part and "at" separated by nothing but space. Only the
# alphanumerics inside a match are emitted, so including the comma in the
# match is what drops it.
_RUN_RX = re.compile(
    r"(?<!" + _EDGE + r")" + _CHAR + r"(?:" + _SEP + _CHAR + r")+(?!" + _EDGE + r"),?")

# Tokenising for the RULE (not the fold): whitespace and the punctuation
# whisper hangs on letters both split. "example.com" splits into two
# multi-character tokens, which is right — neither is a spelled character.
_TOKENS_RX = re.compile(r"[\s\-–—.,;:!?()\[\]{}\"]+")


def _tokens(text: str) -> list:
    return [t for t in _TOKENS_RX.split(text) if t]


def spelling_run(text) -> int:
    """How many characters the text's trailing SPELLING RUN is, or 0.

    3 for "send it to q-z-v", 4 for "k-b-w-7" (a digit rides inside a
    run), 1 for "send an email to q" (the dangling case), 0 for "what is
    plan B", "set a timer for 5", "gate 4 b" (one letter labelling a
    number is not a run), "q-z-v at example dot com" (the run ended when
    he started saying the domain) and every ordinary sentence.

    A count, never the characters: this is what the recorder is given, and
    a spelled local part is half an address. The recorder logs the number.
    """
    raw = str(text or "")
    if not raw.strip():
        return 0
    toks = _tokens(raw)
    run: list = []
    for tok in reversed(toks):
        if len(tok) == 1 and tok.isalnum():
            run.append(tok)
        else:
            break
    n = len(run)
    if n >= MIN_RUN:
        # Two real letters, or it is a label on a number, not spelling.
        letters = sum(1 for c in run
                      if c.isalpha() and c.lower() not in _NOT_A_DANGLING_LETTER)
        if letters >= MIN_RUN_LETTERS:
            return n
        return 0
    if n != 1:
        return 0
    tail = toks[-1]
    if not tail.isalpha() or tail.lower() in _NOT_A_DANGLING_LETTER:
        return 0                       # a digit, "a" or "I": an ordinary turn
    prev = toks[-2].lower() if len(toks) >= 2 else ""
    return 1 if prev in DANGLING_WORDS else 0


def fold_spans(text) -> tuple:
    """``(folded, index_map)`` — the text with every spelling run collapsed
    into one word, and the raw index of each folded character.

    ``index_map[i]`` is where ``folded[i]`` came from in the raw text, so a
    span found in the folded text maps back exactly:
    ``raw[index_map[start] : index_map[end - 1] + 1]``. That is what lets
    :func:`jarvis.outbox.address_span` go on returning a span in the text
    AS GIVEN — its callers cut the address out of the sentence with it.

    With no run in the text the result is the text itself and an identity
    map, so every existing caller is bit-for-bit unchanged.
    """
    raw = str(text or "")
    out: list = []
    imap: list = []
    at = 0
    for m in _RUN_RX.finditer(raw):
        for i in range(at, m.start()):
            out.append(raw[i])
            imap.append(i)
        for i in range(m.start(), m.end()):
            if raw[i].isalnum():
                out.append(raw[i])
                imap.append(i)
        at = m.end()
    for i in range(at, len(raw)):
        out.append(raw[i])
        imap.append(i)
    return "".join(out), imap


def fold(text) -> str:
    """The folded text alone: "q-z-v at example dot com" -> "qzv at example
    dot com"; an ordinary sentence unchanged."""
    return fold_spans(text)[0]
