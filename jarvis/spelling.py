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

THE DANGLING CASE CONTINUES (measured on this branch, 09-05)
------------------------------------------------------------
The two-letter floor above, applied on its own, has a hole the verdict's
eight spelled cases could not show: they are all q-z-v / k-b-w-7 shapes,
and MEASURED on 24 invented first names, 17 have "a" or "i" as their
SECOND letter. "send it to d a" (dana), "m i" (mike), "k a" (kate) is a
run with only ONE counting letter, so the capture would close after the
second character -- the dangling rule buying him the first character and
the floor throwing it away.

So a run that OPENS on a real letter directly after one of DANGLING_WORDS
is held as well: it is not a new rule, it is the dangling case one
character later. It re-admits none of the five phrases the verdict named,
because every one of them opens its run on a digit ("gate 4 b") or on "a"
("he got a c"), and neither can open a spelling. What it does cost is a
grid or seat reference whose letter follows a function word ("it's in
b 4") -- ~1.8 s once, the same trade and the same size as ruling (A), and
pinned in the tests rather than hidden.

WHAT THE NARROWING DELIBERATELY DOES NOT REMOVE (Hunter, 09-05)
---------------------------------------------------------------
After the digit repair, 9 of the 20 measured false holds remain, and
every one of them is the DANGLING-LETTER case: a phrase that ends on one
bare letter after a function word ("switch to b", "the answer is c").
**He ruled to keep them.** He accepts about 1.8 s occasionally on such a
phrase, because that same rule is the one that can hold the FIRST
character when he starts spelling an address, and being chopped off
mid-address is worse than a pause. That is a DECISION, not a defect: it
is not to be removed or watered down without asking him again. What the
rule was SOLD as on 09-05 -- "buys the first spelled character" -- was
not what the numbers said: the rule only names the text, and the decode
that carries the text has to land before the stop. With the greedy
preview as the only reporter it landed in time 4 phases in 9 (MEASURED,
tests/test_spelling_survival.py); the recorder side that makes it land
is described under THE [1, 6] SPLIT below. And with the speculative pass
reporting, those six phrases are held ~every time rather than by the
accident of preview timing -- the same 1.8 s, paid consistently.

THE [1, 6] SPLIT -- what it was, what it is now (09-06)
---------------------------------------------------------
The hold reads the newest DECODE, and a decode has to land before the
stop is due. When the decode of his first letter has not landed by then,
that first character closes a capture of its own and the remaining six
open the next one: "send it to q" / "z-v-k-b-w-7 at example.com". The
second capture then drafts a PLAUSIBLE address missing its first letter
-- zvkbw7@example.com -- where the baseline's four-way chop produced
obvious nonsense. The 09-05 verdict called that "a new failure shape";
the 09-06 adversary showed it was the COMMON path, twice over:

* THE SPECULATIVE PASS PARKED THE PREVIEW THAT FED THE HOLD. Only the
  greedy preview reported to the recorder, and the app's speculative
  full decode runs from 0.3 s into EVERY pause and pushes the greedy
  back 0.9 s. MEASURED on the recorder's real poll cadence with his own
  log's speculative decode times (n=238, p50 0.24 s, p90 0.46 s; the
  greedy latency swept 0.2-0.6 s because it is not logged;
  tests/test_spelling_survival.py): after "send an email to" the first
  letter survived 4 of 9 preview phases (3 of 9 at 0.6 s) and the whole
  seven-character address at his 0.99 s pace 0 of 9. The 09-05 harness
  had no speculative pass, which is why it showed all seven landing.
  Now the speculative pass REPORTS a decode that ends on a run
  (Recorder.note_speculative, additive only), and the recorder WAITS,
  bounded, for a decode that is still running and whose snapshot covers
  the pause (Recorder.note_decoding, DECODE_WAIT_MAX_S) instead of
  stopping past it -- the measured rates are in that test and in
  Recorder._hold_extra.
* A BARE SPELLED ANSWER WAS NEVER HELD, BY THE RULE. spelling_run("q") is
  0 -- no function word in front of it -- so answering "What is it?" by
  spelling straight away ALWAYS split [1, 6]. Now the recorder is told
  when an address is OWED (the question just asked is the function word:
  ``address_owed`` below) and a bare answer is held from its first
  character.

What has not changed: this module cannot see across captures and does
not try to, and the spoken READ-BACK stays the last line. It is
unconditional by construction: every draft in the send lane is armed by
ONE function (jarvis/commander.py _send_file_finish), which returns
outbox.read_back for it, and the ONE call to outbox.send sits behind the
yes to that sentence. tests/test_send_file.py section 27 drives the split
end to end through the fake transport and pins both counts: zero sends
before the read-back, one after the yes, and the To: on the wire is the
address the read-back spoke. A wrong address is heard before it can go;
nothing reassembled is ever sent unread.

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
# The trailing comma OR full stop is whisper's own punctuation, not his: it
# writes a comma between two spelled groups AND after the last one ("q-z-v,
# k-b-w-7, at example dot com" -- the exact shape of the fourth fragment he
# was left with on 09-05), and just as often a full stop after the LAST
# spelled character ("q. z. v. k. b. w. 7. at example.com", "q-z-v,
# k-b-w-7. at example.com"). Either one before "at" broke the address
# parser, which wants the local part and "at" separated by nothing but
# space, and the 09-06 adversary showed the full stop left him with NO
# re-ask at all (parse "" and unresolved_address "" alike). The stop is
# consumed only when whitespace or the end follows it: a stop glued to a
# word ("q-z-v.txt") is part of that word, and so is a joiner glued to one
# ("j-r.smith", "d.a.n.smith") -- the `(?![-–—.]\w)` guard refuses a run
# that continues into a longer token, INCLUDING by backtracking to a
# shorter run, so "j.r.smith@example.com" (typed) is never touched (the
# 09-06 adversary caught it folding to "jr.smith@").
_RUN_RX = re.compile(
    r"(?<!" + _EDGE + r")" + _CHAR + r"(?:" + _SEP + _CHAR + r")+(?!" + _EDGE + r")"
    r"(?![-–—.]\w)(?:,|\.(?=\s|$))?")

# Tokenising for the RULE (not the fold): whitespace and the punctuation
# whisper hangs on letters both split. "example.com" splits into two
# multi-character tokens, which is right — neither is a spelled character.
_TOKENS_RX = re.compile(r"[\s\-–—.,;:!?()\[\]{}\"]+")


def _tokens(text: str) -> list:
    return [t for t in _TOKENS_RX.split(text) if t]


_JOINER_WORDS = frozenset({"dot", "period", "dash", "hyphen", "underscore"})


def _unfinished_domain_at(toks: list):
    """The index of the LAST "at" whose tail is an unfinished domain --
    nothing, one label, or anything ending on a joiner word -- or None
    when the text does not end that way. "example.com" tokenises to two
    labels, which is a finished domain, exactly as the parser reads it.

    A tail of ONE single character ("dana at g") is not a label: it is
    the first letter of a domain being SPELLED, and the ordinary rules
    below hold it through the dangling case ("at" is in DANGLING_WORDS).
    Cutting there returned 0 for it -- the first attempt at this round
    did, and a spelled domain lost its first letter."""
    for i in range(len(toks) - 1, -1, -1):
        if toks[i].lower() == "at":
            tail = toks[i + 1:]
            if not tail or tail[-1].lower() in _JOINER_WORDS:
                return i
            if len(tail) == 1 and len(tail[0]) > 1:
                return i
            return None
    return None


def spelling_run(text, address_owed: bool = False) -> int:
    """How many characters the text's trailing SPELLING RUN is, or 0.

    3 for "send it to q-z-v", 4 for "k-b-w-7" (a digit rides inside a
    run), 1 for "send an email to q" (the dangling case), 7 for
    "q-z-v-k-b-w-7 at" (a bare "at" after a run: the domain is coming),
    0 for "what is plan B", "set a timer for 5", "gate 4 b" (one letter
    labelling a number is not a run), "q-z-v at example dot com" (the run
    ended when he started saying the domain) and every ordinary sentence.

    ``address_owed`` is the recorder's word that the commander has asked
    for an address and is waiting on it ("I've no address for Dana, sir.
    What is it?", or a read-back he may be about to correct). Then the
    question that was just asked IS the function word: a trailing run of
    ANY length holds, from its first character, whatever stands in front
    of it and whatever letter it opens on -- "q", "d a", "a l", "no, q",
    "it's a", "7". Without it the bare answer "q ... z-v-k-b-w-7 at
    example.com" was split [1, 6] EVERY time, because spelling_run("q") is
    0 by the ordinary rule, and the draft that followed was a plausible
    address missing its first letter (the 09-06 adversary's finding B).

    A count, never the characters: this is what the recorder is given, and
    a spelled local part is half an address. The recorder logs the number.
    """
    raw = str(text or "")
    if not raw.strip():
        return 0
    toks = _tokens(raw)
    if not toks:
        return 0
    # An UNFINISHED DOMAIN after a run is still mid-address: "q-z-v-k-b-w-7
    # at" (drawing breath for the domain), "... at example" (no dot yet),
    # "... at example dot" (the top level is coming). The run in front of
    # the "at" is judged by the rules below exactly as if the domain had
    # not been started. A finished domain ("at example dot com", "at
    # example.com") ends the run as before. "at" cannot end an English
    # sentence (it is in DANGLING_WORDS for that reason) and a one-label
    # domain is not a domain, so an ordinary turn is not held by this;
    # what it costs is the same ~1.8 s once on the rare "...at" or "...at
    # <word>" that trails a real run.
    cut = _unfinished_domain_at(toks)
    if cut is not None:
        return spelling_run(" ".join(toks[:cut]), address_owed) if cut else 0
    run: list = []
    for tok in reversed(toks):
        if len(tok) == 1 and tok.isalnum():
            run.append(tok)
        else:
            break
    n = len(run)
    if address_owed and n:
        # The question that was just asked is the function word. No letter
        # floor and no a/i exclusion either: "d a" (dana), "a l" (alice)
        # and a first letter that IS "a" are all his address one character
        # in. One bare DIGIT holds only when it is the whole answer ("7"
        # for a local part that opens on one); after a word it is a number
        # ("set a timer for 5" while a read-back waits) and not held.
        if n >= MIN_RUN or run[0].isalpha() or len(toks) == 1:
            return n
        return 0
    if n >= MIN_RUN:
        # Two real letters, or it is a label on a number, not spelling.
        letters = sum(1 for c in run
                      if c.isalpha() and c.lower() not in _NOT_A_DANGLING_LETTER)
        if letters >= MIN_RUN_LETTERS:
            return n
        # ...or the DANGLING CASE CONTINUING: a run that opens on a real
        # letter directly after a word that cannot end a sentence is the
        # same utterance one character later. Without this, the narrowing
        # buys him the first character and throws it away on the second.
        first = run[-1]                       # run was collected in reverse
        before = toks[-n - 1].lower() if len(toks) > n else ""
        if (first.isalpha() and first.lower() not in _NOT_A_DANGLING_LETTER
                and before in DANGLING_WORDS):
            return n
        return 0
    if n != 1:
        return 0
    # THE DANGLING CASE -- and the trade it is, written where it lives.
    # One bare letter after a function word ("send an email to q") holds.
    #   WHAT IT COSTS. MEASURED with the real poll loop, 09-05: a turn
    #   like "switch to b" or "the answer is c" ends after 2592 ms of
    #   quiet instead of 800 -- +1.8 s, ONCE, on that one turn. (Not
    #   0.8 + 2.0 = 2.8 s: his 2.5 s energy timer gets there first, so
    #   the wait is 2.5 s not 2.8.) Nine of the verdict's 67 realistic
    #   phrases have this shape; an ordinary sentence with no dangling
    #   letter is not delayed by a millisecond.
    #   WHAT IT BUYS. The RULE for the first spelled character: without
    #   it nothing can hold "send an email to q", the capture closes
    #   0.8 s after the "q", and the address he then spells has no first
    #   letter -- the 09-05 chop, and the reason this module exists. The
    #   rule alone did NOT buy the character, and the 09-05 text here
    #   said it did: a decode has to land before the stop is due, and
    #   MEASURED on the real poll cadence with the speculative pass
    #   running (tests/test_spelling_survival.py), the first letter
    #   survived 4 of 9 preview phases (3 of 9 at a 0.6 s greedy decode)
    #   with the greedy preview as the only reporter. What makes the rule
    #   worth its cost is the recorder side -- the speculative pass
    #   reporting and the bounded wait for a decode in flight -- and the
    #   rate it buys NOW is the number that test pins, not this comment.
    #   HE CHOSE IT. Asked on 09-05 with the cost in front of him whether
    #   to keep the behaviour: "yeah keep it". A decision, not a defect;
    #   not to be removed or narrowed without asking him again.
    tail = toks[-1]
    if not tail.isalpha() or tail.lower() in _NOT_A_DANGLING_LETTER:
        return 0                       # a digit, "a" or "I": an ordinary turn
    prev = toks[-2].lower() if len(toks) >= 2 else ""
    return 1 if prev in DANGLING_WORDS else 0


def _tight_dot(raw: str, i: int, start: int, end: int) -> bool:
    """Is raw[i] a full stop with an alphanumeric glued to BOTH sides of it
    inside the run [start, end)? "d.a.n" -> yes for both dots; "q. z. v"
    -> no (a space follows); the stop _RUN_RX consumed at the end of the
    run -> no (nothing alphanumeric follows inside the match)."""
    return (raw[i] == "." and start < i < end - 1
            and raw[i - 1].isalnum() and raw[i + 1].isalnum())


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

    A TIGHT DOT IS KEPT (default taken for him, 09-06). "d.a.n" and
    "j.r.smith" -- a run joined by full stops with no space on either
    side, which is how whisper wrote his spoken "dot" on 09-05 -- fold to
    themselves, dots and all, and the parser reads the dotted local part
    whole (d.a.n@example.com, read back "d dot a dot n at example dot
    com"). Flattening it to "dan" was a silently different mailbox when he
    HAD said "dot"; keeping the dots when he had NOT is a read-back with
    "dot" in it, which his ear catches. Whichever he meant, he hears
    exactly what will be sent. "q. z. v" (dot AND space) and "q-z-v" fold
    as before: those are whisper's punctuation of letters, not a dot he
    said. The cost, named: whisper also writes a spelled run
    initialism-style ("Q.Z.V.K.B.W.7"), and that shape is now read back
    with six "dot"s he never said -- a no and a re-spell, never a wrong
    mailbox.
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
            if raw[i].isalnum() or _tight_dot(raw, i, m.start(), m.end()):
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
