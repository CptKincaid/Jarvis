"""Forms of address — how often Jarvis says "sir" in one spoken BURST.

This module exists because of one measurement and it does exactly one thing.

WHAT WAS MEASURED (2026-08-31, against the real corpus and the live log)
-----------------------------------------------------------------------
* All ~938 authored spoken lines in ``jarvis/`` that contain "sir" contain
  **exactly one**.  Not one of them carries two.
* 24/24 live gemma4 replies on the 6-sentence briefing path carried exactly
  one, always in sentence 1.  Of 263 complete spoken lines in the live log,
  259 carry one and **zero** carry more than one.
* Sirs stack in one place only: where several already-finished lines are
  JOINED into a single spoken burst.

      quiet.digest catch-up          6 sentences,  4 sirs
      quiet.free() reply             7 sentences,  5 sirs
      arrival welcome + catch-up     7 sentences,  5 sirs
      first-wake morning briefing    7 TTS segments over 40 s, 4 sirs
      speak_queue.Watcher.poll_once  up to 10 sentences, 10 sirs

* Of the 702 trailing vocatives in the rendered corpus, 66% end their
  fragment, 31% end a CLAUSE the same sentence runs on from (", sir; " /
  ", sir: " / ", sir — "), and 2.7% have a whole new sentence behind them.
  The middle band is what soundbar, health, focus, runwatch, timekeeper,
  spotify and notes speak in, which is to say it is most of the catch-up
  digest; exempting it (round 3) cost two thirds of the cut, 46.8% against
  70.9%, and cut NOTHING at all out of the warning digest below.  Rule B
  admits it and refuses the last band: 69.0% over 20,000 random bursts.

So there is nothing to thin on a per-line basis — a per-line rule measures a
0.0% cut on every real corpus — and this module deliberately does not offer
one.  The cut belongs **at the join**.

WHY IT RUNS AT THE JOIN AND NOT ON A FINISHED STRING
----------------------------------------------------
The first attempt at this ran on a fully-rendered spoken line, and on a
rendered line Jarvis's own words and an interpolated mail subject are
indistinguishable.  Almost every third-party interpolation in this repo is
UNQUOTED — ``tools/notes.py`` note bodies, ``tools/spotify.py`` track and
artist, ``tools/calendar.py`` event titles, mail subjects, and every word
gemma4 generates — so a quote shield alone protects almost nothing (it is
kept below for the one case it does protect, reported speech) and a
``\\bsir\\b`` match eventually eats somebody else's word:

    "Sir Isaac Newton wrote the Principia" -> "Isaac Newton wrote..."
    "Now playing Yes Sir, I Can Boogie"    -> "Now playing Yes, I Can Boogie"
    "That is sir's coffee"                 -> "That is 's coffee"

At a join site the fragments are still SEPARATE, and each one is a whole
authored Jarvis line whose template is known.  That is where this runs.

A fragment is still not a clean slate, though: a rendered fragment is an
authored TEMPLATE with third-party slots already filled in, so the rules
below are all about telling one from the other with nothing but the shape of
the fragment.

THE RULES
---------
**B. A trailing vocative is droppable where the CLAUSE it signs off ends.**
A vocative preceded by an attaching comma and followed by nothing but
sentence punctuation and the end of the fragment — ", sir." / ", sir?" /
", sir!" / trailing ", sir" — is a sign-off, and the sign-off is the thing
that stacks.  So is one followed by a clause separator that cannot end a
sentence — ", sir; " / ", sir: " / ", sir — " — because the sentence runs on
behind it and a run-on sentence is one speaker, not two:

    "I'm coming out of the monitor, sir; the soundbar has dropped."
    "Memory is getting tight, sir: 3 gigabytes free."
    "Your Liked Songs on shuffle, sir — 500 of them."

That register is 31% of the trailing vocatives in the corpus and it is what
``soundbar``, ``health``, ``focus``, ``runwatch``, ``timekeeper``, ``spotify``
and ``notes`` speak — which is to say it is most of what reaches
``quiet.hold`` and therefore most of the catch-up digest, the burst this
module exists for.  Exempting it (the round-3 rule was fragment-final only)
cost two thirds of the cut on exactly the burst that needed it.

What is NOT droppable is a vocative behind which the fragment starts a NEW
SENTENCE — ", sir. " / ", sir! " / ", sir? " with words after it — because
that is the shape of somebody else's sentence embedded in one of Jarvis's,
and on a rendered fragment there is no way to tell it from his own:

    'He said, "Thank you, sir." and left.'         quoted third-party speech
    'New mail from Bob: Thank you, sir. Shall I read it?'   a mail subject

Nor is one inside a QUOTATION, wherever it falls: an open ``"`` or ``“``
in front of the vocative means the words around it are being reported, not
spoken (``inside_a_quotation``).  That shield is what holds the first case
above; the sentence rule and it are deliberately redundant there.

The other three documented corruptions need neither, because they are caught
by rule C — each carries TWO addresses, which no authored line does:

    'One note mentions milk, sir: buy milk, sir.'  a note body (notes.py:550)
    'Saved Yes, Sir! to your Liked Songs, sir.'    a track title
    'New mail from Bob, sir: Thank you, sir.'      a mail subject
    'Mail from Bob, sir — re: Thank you, sir.'     mailwatch.py:139

The ASCII hyphen is left out of the clause separators on purpose.  The corpus
does not use ", sir - " at all (0 of 702), and " - " is how a music service
renders a subtitle, so "Yes, Sir - Remastered" would be the one third-party
shape the rule could still reach.

A join site that BUILT a fragment out of its own words with no slot in it may
certify it as ``address.authored`` and have the sentence rule waived as well;
see ``Authored``.  Nothing waives the quotation shield.

A SENTENCE-INITIAL "Sir, ..." is a SUMMONS.  It is doing real work in
``timekeeper.REMINDER_LINE`` ("Sir, this is your reminder. {text}") and
``timekeeper.ALARM_LINE`` ("Sir, it's {time}. {label}"); an alarm that has
lost its summons is a clock radio, not a butler.  A summons is recognised
only at index 0 of the fragment and only in front of a LOWER-CASE word, for
the same reason the honorific veto exists: both of those templates put
third-party text at a sentence start further in, and

    "Sir, this is your reminder. Sir, With Love starts at eight."

is a film, not a second summons.  Both authored summonses sit at index 0 in
front of "this", "your" or "it's", so nothing real is lost.

A MEDIAL ", sir," is not droppable either, precisely because a listing that
renders a title as "Yes, Sir, I Can Boogie" would otherwise lose a word.  Two
further vetoes are stated explicitly even though the punctuation anchor
already implies them, because they are the two documented corruptions and
they should be readable as rules rather than as a side effect: never a
possessive ("sir's"), and never an honorific in front of a capitalised name.

**C. One authored line, one address.**  That is the measurement (938 of 938),
so a fragment that holds TWO addresses is a fragment whose slots were filled
with somebody else's "sir" — the note body and the mail subject above — and
it is left exactly as it is.  It still counts as having addressed him.

**D. Keep the first, drop the repeats.**  In a burst the first sign-off
survives and later fragment-final ones go, up to ``per_burst``.  A summons is
not a sign-off and is not governed by that budget — it is dropped only when
the SAME summons has already been spoken in the same burst, because five
held reminders in one catch-up is where "Sir, this is your reminder" becomes
a chant.  The first announces the list; the rest are thinned to "This is your
reminder."  A summons still spends the sign-off budget: it has addressed him.

**THE INVARIANT** (see ``tests/test_address.py``, proved by fuzz):

  1. the first fragment of a burst is never rewritten;
  2. a fragment holding two addresses is never rewritten;
  3. an address is only ever dropped when an earlier fragment of the same
     burst KEPT one — a trailing vocative behind a kept address, a summons
     behind the identical kept summons — so a burst that went in with at
     least one address always comes out with at least one;
  4. and the pass never raises.  Every caller may join what it gets back.

SWAPPING ONE FORM OF ADDRESS FOR ANOTHER (added for the people section)
-----------------------------------------------------------------------
``swap_addresses(text, honorific)`` is the SECOND thing this module does,
and it reuses every rule above.  Jarvis says "sir" throughout because the
lines are authored that way; Mara and Heather are "ma'am"
(``jarvis/identity.Person.honorific``, typed by the owner, never inferred),
and the swap happens at the door -- ``app._say`` and ``commander._speak`` --
rather than by editing a thousand literals.

Three things about it are load-bearing:

* ``honorific="sir"`` RETURNS THE INPUT OBJECT.  Not a copy: the same
  object.  So the owner's line cannot move by one character and nothing
  about his tuned voice can regress through this door.
* The swap sees TWO shapes the remover refuses, and each for the same
  reason the remover refuses it.  A MEDIAL vocative (", sir, but ...") in
  front of a LOWER-CASE word: dropping one would eat a word out of "Yes,
  Sir, I Can Boogie", but swapping one is safe under the Title-Case
  discriminator already used for a summons, and a third of the authored
  corpus addresses him there.  And a LONE "Sir?" (``app.NUDGE_LINE``).
* NOTHING INSIDE A QUOTATION IS SWAPPED OR CUT.  ``vocative_spans`` counts
  a quoted vocative (it is not droppable, but it has addressed him);
  ``swap_spans`` refuses it outright, because putting "ma'am" into a
  report of what somebody else said is the same defect as deleting a word
  from it.  ``honorific=""`` therefore goes through ``drop_swappable``
  rather than ``drop_addresses``.

THE ACCEPTED COST, stated rather than discovered: a rendered line carrying
an interpolated mail subject or note body of the shape ", sir." can have
that "sir" swapped too.  It is bounded -- ``gate._HIS`` refuses mail,
notes and messages to a KNOWN person, so those lines are never spoken to a
ma'am person in the first place -- and it never deletes anything.

Only "sir" is governed.  "Hunter" is the rare form (0 uses in 24 live replies)
and is also an ordinary word that turns up in text he asked to have read back;
it is neither counted nor touched.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# What counts as an address, and what may be dropped
# ----------------------------------------------------------------------
# \b keeps "sirloin" and "Sirius" out.  The plural "sirs" is a different word
# and is not matched.
_SIR_RX = re.compile(r"\bsir\b", re.I)

# The comma that ATTACHES a trailing vocative to the clause in front of it.
# Required, and it is the load-bearing guard: "Sir Isaac Newton" and
# "Yes Sir, I Can Boogie" have no comma in front and can never match.
_ATTACHING_COMMA_RX = re.compile(r",[ \t]*$")

# What may follow a droppable vocative: sentence punctuation, a dash, or the
# end of the fragment.  A COMMA is absent on purpose -- see the docstring.
_TERMINAL_RX = re.compile(r"^[ \t]*(?:[.!?;:]|—|–|-{1,2}(?=[ \t]|$))")

# ...and NOTHING may follow that punctuation (rule B).  A closing quote is
# allowed through only as the fragment's last character, never as the seam of
# a quotation that carries on: 'He said, "Thank you, sir." and left.'
_FRAGMENT_FINAL_RX = re.compile(
    r"^[ \t]*(?:[.!?;:]|—|–|-{1,2})?[ \t]*[\"'”’)\]]{0,2}[ \t]*$")

# ...or the fragment runs on behind the vocative, but only over a separator
# that cannot END a sentence.  ";" ":" and the dashes leave one sentence with
# one speaker in it; ".", "!" and "?" start a second sentence, and a second
# sentence inside one fragment is how an interpolated subject or a quoted
# line gets in.  The ASCII hyphen is out: 0 of 702 corpus vocatives use it and
# " - " is how a track subtitle is rendered ("Yes, Sir - Remastered").
_CLAUSE_FINAL_RX = re.compile(r"^[ \t]*[;:—–]")

# The quotation shield.  Straight quotes are counted (an odd number in front
# means the vocative is inside one); typographic quotes are nested.  The
# straight APOSTROPHE is deliberately not a quote character here -- "I'm
# coming out of the monitor, sir" is every second line in the corpus.
_OPEN_QUOTES = "“‘«"
_CLOSE_QUOTES = "”’»"

# The two explicit vetoes (finding 1 and finding 2).
_POSSESSIVE_RX = re.compile(r"^['’]")          # "sir's coffee"
_BEFORE_A_NAME_RX = re.compile(r"^[ \t]+[A-Z]")     # "sir Isaac"

# A summons is a vocative at index 0 of the fragment with a comma behind it
# and a lower-case word after that: "Sir, this is your reminder."  It is
# counted (it has already addressed him).  "Sir, With Love starts at eight."
# is a title arriving through timekeeper's {text} slot, not a summons.
_SUMMONS_AFTER_RX = re.compile(r"^[ \t]*,[ \t]*")
_SUMMONS_TITLE_RX = re.compile(r"^[ \t]*,[ \t]*[A-Z]")

# The sentence a summons opens, used to spot the second, third and fourth
# copy of it in one burst.
_SUMMONS_SENTENCE_RX = re.compile(r"^[^.!?]*[.!?]?")
_NOT_WORDS_RX = re.compile(r"[^a-z0-9]+")

DEFAULT_PER_BURST = 1


class Authored(str):
    """A fragment the join site CERTIFIES is Jarvis's own words end to end.

    Rule B is a rule about ignorance: on a rendered fragment there is no way
    to tell "That's done, sir. Anything else?" from "New mail from Bob:
    Thank you, sir. Shall I read it?"  A join site that BUILT the fragment
    out of nothing but its own template and its own counts does know, and
    says so by wrapping it -- ``quiet.digest_fragments`` is the one caller.

    It buys only the shape rule B refuses on its own, a NEW SENTENCE behind
    the vocative; the clause separators need no certificate, which is why
    "While you were busy, sir: two reminders and one warning." is thinned
    with or without one now.  The quotation shield is not waived: a
    certificate says the words are Jarvis's, not that he is not quoting.

    Wrap NOTHING that carries a slot.  A str subclass, so every caller in
    between (join, format, the TTS) treats it as the string it is, and the
    marker is simply lost the moment anyone rebuilds it -- ``str.format`` on
    a wrapped TEMPLATE returns a plain ``str``, so wrapping one by mistake
    fails safe.  An unmarked fragment is the protected one."""
    __slots__ = ()


def authored(text):
    """``Authored(text)`` -- see the class. Idempotent, and never None."""
    return text if isinstance(text, Authored) else Authored(text or "")

TRAILING = "trailing"       # droppable when fragment-final and over budget
SUMMONS = "summons"         # droppable only as a repeat of itself


# ----------------------------------------------------------------------
# Configuration (persona.address_thinning / persona.address_per_burst)
# ----------------------------------------------------------------------
# Module state, installed once at boot the way ``earcons.set_config`` is,
# because the join sites are module functions (``quiet.digest``,
# ``speak_queue.Watcher.poll_once``) with no config of their own and they must
# not need one in order to be tested.
_ENABLED = True
_PER_BURST = DEFAULT_PER_BURST


def set_config(cfg) -> None:
    """Install ``persona.address_*`` from an AssistantConfig (or None to go
    back to the defaults).  A config edit needs a restart:
    ``AssistantConfig.reload_if_changed()`` has no callers."""
    global _ENABLED, _PER_BURST
    if cfg is None:
        _ENABLED, _PER_BURST = True, DEFAULT_PER_BURST
        return
    try:
        _ENABLED = bool(cfg.get("persona.address_thinning", True))
        _PER_BURST = max(1, int(cfg.get("persona.address_per_burst",
                                        DEFAULT_PER_BURST)))
    except Exception:                       # noqa: BLE001 - boot boundary
        _ENABLED, _PER_BURST = True, DEFAULT_PER_BURST


def configure(*, enabled: bool = True,
              per_burst: int = DEFAULT_PER_BURST) -> None:
    """Set the knobs directly (tests, and the measurement harness)."""
    global _ENABLED, _PER_BURST
    _ENABLED = bool(enabled)
    # Clamped at 1: the config cannot be edited into a mute Jarvis.  To turn
    # the pass off entirely, set persona.address_thinning false.
    _PER_BURST = max(1, int(per_burst or DEFAULT_PER_BURST))


def enabled() -> bool:
    return _ENABLED


def per_burst() -> int:
    return _PER_BURST


# ----------------------------------------------------------------------
# Finding the addresses in one fragment
# ----------------------------------------------------------------------
def _is_possessive(rest: str) -> bool:
    """Finding 2: "That is sir's coffee" is not a form of address."""
    return bool(_POSSESSIVE_RX.match(rest))


def _is_before_a_name(rest: str) -> bool:
    """Finding 1: "Sir Isaac Newton" is an honorific, not a form of address."""
    return bool(_BEFORE_A_NAME_RX.match(rest))


def _at_fragment_start(text: str, pos: int) -> bool:
    """Index 0 of the fragment, give or take leading whitespace.

    NOT "after any full stop": ``REMINDER_LINE`` and ``ALARM_LINE`` both put
    third-party text at exactly a sentence start, so a summons found after a
    full stop is somebody else's title (defect D2)."""
    return text[:pos].strip() == ""


def is_fragment_final(text: str, end: int) -> bool:
    """True when nothing but sentence punctuation follows ``text[:end]``."""
    return bool(_FRAGMENT_FINAL_RX.match(text[end:]))


def is_clause_final(text: str, end: int) -> bool:
    """True when the fragment runs on behind the vocative over a separator
    that cannot end a sentence — ", sir; " / ", sir: " / ", sir — ".

    One sentence has one speaker in it, so the words behind such a separator
    are the same voice that said the vocative.  A ".", "!" or "?" with words
    after it is the opposite signal and is not accepted here."""
    return bool(_CLAUSE_FINAL_RX.match(text[end:]))


def inside_a_quotation(text: str, pos: int) -> bool:
    """True when ``text[pos]`` sits inside a quotation opened in the fragment.

    The shield for 'He said, "Thank you, sir." and left.' -- reported speech
    carries somebody else's vocative, and it is not Jarvis's to take at any
    setting, certified fragment or not."""
    head = text[:pos]
    if head.count('"') % 2:
        return True
    depth = 0
    for ch in head:
        if ch in _OPEN_QUOTES:
            depth += 1
        elif ch in _CLOSE_QUOTES and depth:
            depth -= 1
    return depth > 0


def _droppable_tail(text, start: int, end: int) -> bool:
    """Rule B in one place: may this trailing vocative be cut out?

    ``start`` is the attaching comma and ``end`` the last letter of the word.
    Read in order -- the quotation shield first because nothing waives it,
    then the join site's certificate, then the two shapes."""
    if inside_a_quotation(text, start):
        return False
    if isinstance(text, Authored):
        return True
    return is_fragment_final(text, end) or is_clause_final(text, end)


def vocative_spans(text: str) -> list[tuple]:
    """``[(start, end, kind), ...]`` for every form of address in ``text``.

    ``kind`` is ``TRAILING`` or ``SUMMONS``.  Both COUNT as having addressed
    him; whether either may be dropped is decided by ``thin_fragments``,
    which is the only thing that knows about the burst around the fragment.
    A "sir" that is neither -- an honorific before a name, a possessive, a
    word inside third-party prose -- is not an address and is not returned at
    all, so it can never spend the burst's budget either.

    ``start`` for a trailing vocative is the index of the ATTACHING COMMA, not
    of the word: the comma is part of what gets removed.
    """
    text = text or ""
    out = []
    for m in _SIR_RX.finditer(text):
        rest = text[m.end():]
        if _is_possessive(rest) or _is_before_a_name(rest):
            continue
        lead = _ATTACHING_COMMA_RX.search(text[:m.start()])
        if lead is not None:
            if rest.strip() == "" or _TERMINAL_RX.match(rest):
                out.append((lead.start(), m.end(), TRAILING))
            continue
        if _SUMMONS_AFTER_RX.match(rest) and not _SUMMONS_TITLE_RX.match(rest) \
                and _at_fragment_start(text, m.start()):
            out.append((m.start(), m.end(), SUMMONS))
    return out


def count_sirs(text: str) -> int:
    """Every ``sir`` token in ``text`` -- the raw before/after number the
    report and the tests share with the measurement harness.  Deliberately
    naive: it counts what a listener hears, not what this module considers an
    address."""
    return len(_SIR_RX.findall(text or "")) if isinstance(text, str) else 0


def summons_key(text: str) -> str:
    """The summons's own sentence, normalised, so the second and third copy
    of it in one burst can be recognised: "Sir, this is your reminder." from
    five held reminders is one announcement and four repeats (defect D4)."""
    head = _SUMMONS_SENTENCE_RX.match(text or "")
    return _NOT_WORDS_RX.sub(" ", (head.group(0) if head else "").lower()).strip()


def is_speakable(fragment) -> bool:
    """False for a fragment with nothing in it a listener would hear.

    A queued line of exactly ", sir." thins to "." and the sink must not be
    handed "... Feed the cat. . Sir," (defect D5).  ``str.isalnum`` rather
    than ``[A-Za-z0-9]``: a note read back in another script is speech."""
    return isinstance(fragment, str) and any(c.isalnum() for c in fragment)


# ----------------------------------------------------------------------
# Removal
# ----------------------------------------------------------------------
def _tidy(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _drop_trailing(text: str, start: int, end: int) -> str:
    """Cut a trailing vocative out with its attaching comma, leaving whatever
    punctuation was behind it alone:

        "The build passed, sir."      -> "The build passed."
        "Very good, sir"              -> "Very good"
        "Memory is tight, sir: 3 free."  -> "Memory is tight: 3 free."
    """
    return _tidy(text[:start] + text[end:])


def _drop_summons(text: str, start: int, end: int) -> str:
    """Cut a repeated summons out and re-capitalise the sentence it opened:

        "Sir, this is your reminder. Stand up."
            -> "This is your reminder. Stand up."
    """
    m = _SUMMONS_AFTER_RX.match(text[end:])
    tail = text[end + (m.end() if m else 0):]
    if tail[:1].isalpha() and tail[:1].islower():
        tail = tail[0].upper() + tail[1:]
    return _tidy(text[:start] + tail)


# Of 263 complete spoken lines in the live log, zero carried more than one
# form of address (module doc); this only bounds the re-read loop.
_MAX_ADDRESSES_PER_LINE = 8


def drop_addresses(text: str) -> str:
    """Every form of address cut out of ONE line, all of them.

    This is NOT the per-line thinning the module doc refuses, and the
    difference is who is being spoken to. Thinning asks "has he been
    called sir too often in this burst?" -- a question about frequency,
    which a single line cannot answer, which is why there is no per-line
    thinner. This asks "is the man being addressed even in the room?" and
    the caller already knows the answer: brain.strip_relay_address uses it
    when the line is aimed at somebody else entirely (LIVE 2026-09-02
    14:29:24, "Good evening, Ali and Heather; ... , sir.").

    It is the removal only. Every rule about WHAT counts as an address
    still comes from vocative_spans, so "Sir Isaac Newton", "sir's coffee"
    and a sir inside a quotation are untouched here as everywhere else.
    One span at a time, re-read after each cut: the removers run _tidy(),
    which strips and collapses whitespace, so indices taken before a cut
    cannot be trusted after it.
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for _ in range(_MAX_ADDRESSES_PER_LINE):
        spans = vocative_spans(out)
        if not spans:
            return out
        start, end, kind = spans[0]
        out = (_drop_trailing(out, start, end) if kind == TRAILING
               else _drop_summons(out, start, end))
    return out


# ----------------------------------------------------------------------
# Swapping one form of address for another
# ----------------------------------------------------------------------
# THE THREE VALUES, and there is no fourth. Kept as literals here rather
# than imported from jarvis/identity.py because this module is pure and has
# no dependency on the registry -- and a test pins that the two tuples say
# the same thing.
SIR = "sir"
MAAM = "ma'am"
NO_ADDRESS = ""
SWAPPABLE = (SIR, MAAM, NO_ADDRESS)

# A MEDIAL vocative -- ", sir, but I've no way in yet" -- in front of a
# LOWER-CASE word.
#
# WHY SWAPPING SEES A SHAPE THAT DROPPING REFUSES, which is the one place
# these two passes differ and it is worth being explicit about. Dropping a
# medial vocative would eat a word out of "Yes, Sir, I Can Boogie", so
# ``vocative_spans`` refuses the whole shape and that refusal stays exactly
# as it was. But a medial vocative is where a third of the authored corpus
# addresses him -- "I have the result, sir, but the model didn't get to
# putting it into words." -- and refusing it here would mean 30-odd real
# lines say "sir" to a woman. The discriminator is the one this module
# already trusts for a summons (``_SUMMONS_TITLE_RX``): a title is Title
# Case, so a LOWER-CASE word behind the comma is Jarvis's own clause
# carrying on. "Yes, Sir, I Can Boogie" has a capital there and is
# untouched, at any setting.
_MEDIAL_RX = re.compile(r"^[ \t]*,[ \t]*[a-z]")

# A vocative that IS the whole line: "Sir?" (``app.NUDGE_LINE``). No comma
# in front of it and no clause behind it, so neither of the other two
# shapes sees it.
_LONE_RX = re.compile(r"^[ \t]*[?.!…]*[ \t]*$")



def _swap_records(text: str) -> list[tuple]:
    """``[(cut_start, word_start, word_end, kind), ...]``.

    ``cut_start`` is where a REMOVAL begins -- the attaching comma when
    there is one, otherwise the word itself -- and ``word_start`` is where
    a SWAP begins. They differ because dropping a vocative takes its comma
    with it ("Very good, sir" -> "Very good") while swapping one must not.
    """
    text = text or ""
    out = []
    for m in _SIR_RX.finditer(text):
        rest = text[m.end():]
        if _is_possessive(rest) or _is_before_a_name(rest):
            continue
        if inside_a_quotation(text, m.start()):
            # NOTHING INSIDE A QUOTATION IS EVER REWRITTEN OR REMOVED HERE.
            # Reported speech carries somebody else's vocative:
            # 'He said, "Thank you, sir." and left.'
            continue
        lead = _ATTACHING_COMMA_RX.search(text[:m.start()])
        if lead is not None:
            if rest.strip() == "" or _TERMINAL_RX.match(rest) \
                    or _MEDIAL_RX.match(rest):
                out.append((lead.start(), m.start(), m.end(), TRAILING))
            continue
        if not _at_fragment_start(text, m.start()):
            # A COMMA-LESS trailing vocative -- "Good morning sir." -- is
            # deliberately NOT an address here. The only rule that would
            # catch it is loose enough to rewrite "Now playing Yes Sir."
            # and "Thank You Sir.", so the fix belongs in the one authored
            # line that had one (workflows.DEFAULT_WORKFLOWS, given its
            # comma) rather than in this pass.
            continue
        if _SUMMONS_AFTER_RX.match(rest) and not _SUMMONS_TITLE_RX.match(rest):
            out.append((m.start(), m.start(), m.end(), SUMMONS))
        elif _LONE_RX.match(rest):
            out.append((m.start(), m.start(), m.end(), TRAILING))
    return out


def swap_spans(text: str) -> list[tuple]:
    """``[(start, end), ...]`` -- every "sir" in ``text`` that is a form of
    address AND may be rewritten as a different one.

    Every veto ``vocative_spans`` applies applies here too, and one more:
    NOTHING INSIDE A QUOTATION IS EVER REWRITTEN. That shield is only
    advisory in ``vocative_spans`` (a quoted vocative is still counted, it
    is merely not droppable), and this pass needs it as a hard rule --
    'He said, "Thank you, sir." and left.' is reported speech and putting
    "ma'am" in somebody else's mouth is the same defect as deleting a word
    from it.

    A test pins that every span ``vocative_spans`` finds outside a
    quotation is also found here, so the two passes cannot drift into
    disagreeing about what an address is.
    """
    return [(ws, we) for _cut, ws, we, _kind in _swap_records(text)]


def drop_swappable(text: str) -> str:
    """Every address this pass recognises, CUT OUT, comma and all.

    What ``honorific=""`` means: a person who asked not to be addressed.

    It is NOT ``drop_addresses``. That one removes every span
    ``vocative_spans`` finds INCLUDING one inside a quotation, which is
    right for its caller (``brain.strip_relay_address``, aiming a whole
    line at somebody else) and wrong here: taking "sir" out of 'He said,
    "Thank you, sir."' edits a report of what somebody said.
    """
    out = text
    for _ in range(_MAX_ADDRESSES_PER_LINE):
        recs = _swap_records(out)
        if not recs:
            return out
        cut, start, end, kind = recs[0]
        out = (_drop_summons(out, start, end) if kind == SUMMONS
               else _tidy(out[:cut] + out[end:]))
    return out


def _cased_like(word: str, sample: str) -> str:
    """``word`` wearing ``sample``'s capitalisation.

    Only the first letter is consulted, because that is the only thing that
    varies in the corpus: a SUMMONS is "Sir, ..." at index 0 and a trailing
    vocative is ", sir." -- and both templates are authored, so an ALL-CAPS
    address does not occur and is not invented here.
    """
    if sample[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def swap_addresses(text, honorific: str = SIR):
    """The same line, addressed to somebody who is not "sir".

    THE ONE PLACE THE HONORIFIC CHANGES, and it changes nothing else. The
    1,000-odd "sir" literals in this tree are not edited, ever: they are
    authored as written and rewritten HERE, at the door, for whoever is
    actually being spoken to (``jarvis/app.py:_say``,
    ``jarvis/commander.py:_speak``).

    THE OWNER PATH IS BYTE-IDENTICAL AND RETURNS THE INPUT OBJECT. That is
    not an optimisation, it is the safety argument: with ``honorific="sir"``
    this function cannot change a single character of anything Hunter hears,
    so the test files asserting exact spoken strings stay green and his
    tuned voice cannot regress through this door.

    ``honorific=""`` is a real choice -- somebody who asked not to be
    addressed at all -- and delegates to ``drop_addresses``.

    Every "is this an address?" question is delegated to ``swap_spans``,
    which is why "Sir Isaac Newton", "Yes Sir, I Can Boogie", "sir's
    coffee" and anything inside a quotation are untouched here as
    everywhere else. A bare ``str.replace("sir", ...)`` anywhere in this
    tree is a defect.

    Never raises: a failure here must speak the line as written rather than
    cost him the sentence.
    """
    if not isinstance(text, str) or not text:
        return text
    if honorific == SIR:
        return text                     # the owner. Not one byte moves.
    try:
        if honorific == NO_ADDRESS:
            return drop_swappable(text)
        if honorific not in SWAPPABLE:
            # A value nobody typed. Speak the line as authored rather than
            # rewrite it with something that was never a choice.
            log.error("address: %r is not a form of address; speaking as "
                      "written", honorific)
            return text
        spans = swap_spans(text)[:_MAX_ADDRESSES_PER_LINE]
        if not spans:
            return text
        # Right to left, so an earlier span's indices are still good after
        # a later one has changed the length of the string.
        out = text
        for start, end in reversed(spans):
            out = (out[:start] + _cased_like(honorific, out[start:end])
                   + out[end:])
        return out
    except Exception:                       # noqa: BLE001 - never lose a line
        log.exception("address: the swap failed; speaking as written")
        return text


# ----------------------------------------------------------------------
# The join
# ----------------------------------------------------------------------
def _thin(frags: list, budget: int) -> list:
    kept = 0
    seen_summons: set = set()
    out = []
    for frag in frags:
        if not frag or not isinstance(frag, str):
            out.append(frag)
            continue
        spans = vocative_spans(frag)
        if len(spans) != 1:
            # Rule C: no addresses, or two in one fragment -- which no
            # authored line has, so one of them came out of a slot.  Left as
            # written, and still counted: he has been addressed.
            kept += len(spans)
            out.append(frag)
            continue
        start, end, kind = spans[0]
        if kind == TRAILING:
            if kept >= budget and _droppable_tail(frag, start, end):
                out.append(_drop_trailing(frag, start, end))
                continue
        else:
            key = summons_key(frag)
            if key in seen_summons:
                out.append(_drop_summons(frag, start, end))
                continue
            seen_summons.add(key)
        kept += 1
        out.append(frag)
    return out


def thin_fragments(fragments, *, per_burst_n: int | None = None,
                   on: bool | None = None) -> list[str]:
    """The fragments as they should be SPOKEN when joined into one burst.

    Each fragment must be a whole authored Jarvis line (that is the invariant
    that makes this safe -- see the module docstring).  One fragment on its
    own is returned untouched, always, and so is the first fragment of any
    burst.

    This never raises.  Three of the four join sites call it without a guard
    of their own, and a courtesy said twice is a far smaller bug than a
    catch-up digest that died on the way to the speaker (defect D7)."""
    try:
        frags = list(fragments or [])
    except Exception:                       # noqa: BLE001 - not even iterable
        log.exception("address: fragments are not a list; speaking as written")
        return []
    try:
        if on is None:
            on = _ENABLED
        if not on or len(frags) < 2:
            return list(frags)
        budget = _PER_BURST if per_burst_n is None else max(1, int(per_burst_n))
        out = _thin(frags, budget)
        # Invariant 3, enforced and not merely argued: a burst that went in
        # with an address comes out with one.  Cheap, and it is the last line
        # of defence for the thing the whole feature must never do.
        if any(count_sirs(f) for f in frags) and not any(count_sirs(f) for f in out):
            log.error("address: thinning would have removed every address; "
                      "speaking as written")
            return list(frags)
        return out
    except Exception:                       # noqa: BLE001 - never mute a burst
        log.exception("address: thinning failed; speaking as written")
        return list(frags)


def join_thinned(fragments, sep: str = " ") -> str:
    """Join fragments that have ALREADY been thinned, dropping any that are
    no longer speech (defect D5).  ``thin_fragments`` stays 1:1 with its
    input -- the arrival cue indexes into it -- so the drop happens here."""
    out = []
    for f in list(fragments or []):
        if is_speakable(f):
            out.append(f)
        elif f and not isinstance(f, str):
            out.append(str(f))
    return sep.join(out)


def join_fragments(fragments, sep: str = " ", **kw) -> str:
    """``thin_fragments`` then join -- the one-liner the join sites that build
    a string want.  Never raises, for the same reason."""
    return join_thinned(thin_fragments(fragments, **kw), sep)
