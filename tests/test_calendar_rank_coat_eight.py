"""ROUND EIGHT of the 2026-09-05 calendar-date bug: the closed list itself.

Round seven turned the tail rule the right way round.  ``_ord_tail_ok``
replaced the ~200-word ``_TAIL_WORDS`` whitelist, so a bare ordinal that
reaches the calendar is a DATE unless a PROVED RANK SHAPE follows it, and
the round-seven adversary's independent grid -- 1051 rows x 2 variants x
9 injected instants x both doors = 18,918 cells, ``calendar.now_local``
RAISING -- scored SILENT-TODAY 0 on the tip against 3,222 per door on the
base 61f0945 and 12,618 per door on mainline.  His complaint is fixed.

IT BLOCKED ON THE LIST.  158 phrasings right on 61f0945 were wrong on
dccb261.  One shape is the trade the branch declares out loud; the other
three were never declared and are mechanical.  This file pins all three,
and it pins the list's own method so a fourth cannot happen quietly.

(1) THE LIST IS SINGULAR-ONLY WHILE THE MATCHER COMPARES SURFACE FORMS.
    ``_TAIL_WORD_RX`` captures ``[a-z][\\w'.]*`` whole, so every plural and
    possessive of a word ALREADY ON THE LIST fell through into a date.
    MEASURED 2026-09-06 on the round-8 grid, all today on 61f0945 and all
    a date on dccb261: "the 3rd classes", "the 2nd meetings", "the 2nd
    meeting's time", "the 3rd labs", "the 4th items", "the 2nd lecture's
    slides", "the 3rd rows", "the 2nd exams", "the 5th pages", "the 2nd
    persons", "the 3rd sessions".  THE GUARD-ONE-HALF PATTERN AGAIN -- a
    guard written for one form of a pair and never applied to its twin.
    This is the SIXTH occurrence on this project; the fifth was merged an
    hour before this file was written.

(2) THE SIX NOUNS THAT ROUTE THE SENTENCE TO THE CALENDAR WERE ABSENT
    FROM ``_RANK_NOUNS``: appointment, event, plan, calendar, schedule,
    agenda.  Every one is in ``commander._CAL_READ_RX``, so a sentence
    carrying it is GUARANTEED to reach the calendar.  Measured end to end
    through both doors: "when is the 2nd appointment on my calendar" ->
    2026-10-02, spoken "Friday the 2nd of October".  This is the
    highest-traffic rank phrase a calendar assistant hears.

(3) THE ELIDED RANK WITH NO PRONOUN AFTER IT.  dccb261's ``_elided_rank``
    fired only on a pronoun, so the same shape in predicate position, or
    in front of one of the rank's own frames behind a preposition that is
    not "in", still became a date: "what's the first on my calendar" ->
    2026-10-01, "which one is the 2nd", "that's the 2nd", "the first on
    the list", "the 3rd from the top", "the first up", "the 1st after
    that".

WHERE THE READING COMES FROM.  The elided rank is read from what stands
IN FRONT of the ordinal -- a wh-word or a demonstrative plus a copula, or
a bare "the" with no preposition -- and NEVER from the mere fact that
something follows it, because "the 12th" standing alone must stay a date.
Behind the bare "the" the rank's own frames then decide.  A prepositional
frame ("on the 12th", "for the 12th") beats the elided rank outright.

THE TRADE STANDS, and it is the only thing allowed to be wrong: a rank
noun this list does not know answers about a DATE.  "the 3rd rehearsal on
my calendar" gives the 3rd, which is a wrong day HE CAN HEAR AND CORRECT;
the silent today it replaced was a wrong day he could not.

THE LIST'S OWN METHOD, finished here.  ``_RANK_NOUNS`` says of itself
that it is grown "only from a measured phrase, with the phrase written
beside the word".  On dccb261 two entries (lecturer, weekend) appeared
nowhere in the justifying comment and ten more were covered by a phrase
for their group rather than one of their own.  The source now carries a
RANK PHRASES block, one word one phrase, and ``test_every_rank_noun_...``
below reads that block and fails if it and the frozenset ever disagree --
so an entry with no measured phrase beside it cannot be added again.

THE CLOCK: every test injects ``now`` and replaces calendar.now_local
with a function that RAISES, so nothing here can read the wall clock.
Every event, title and utterance is INVENTED.
"""
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.calendar as calendar
from jarvis.commander import _CAL_READ_RX, calendar_range
from jarvis.tools.calendar import as_date, is_ask, sentence_date

CHI = ZoneInfo("America/Chicago")
NZ = ZoneInfo("Pacific/Auckland")

# The nine instants the round-seven adversary measured at: both 2026 DST
# edges for America/Chicago, New Year's Eve, two 31sts, a leap day, a +12
# zone and the day he reported it.
INSTANTS = [
    ("his-day", datetime(2026, 9, 6, 8, 15, tzinfo=CHI)),
    ("dst-spring", datetime(2026, 3, 8, 1, 30, tzinfo=CHI)),
    ("dst-fall", datetime(2026, 11, 1, 1, 30, fold=0, tzinfo=CHI)),
    ("nye", datetime(2026, 12, 31, 23, 30, tzinfo=CHI)),
    ("jan31", datetime(2026, 1, 31, 9, 0, tzinfo=CHI)),
    ("leap", datetime(2028, 2, 29, 12, 0, tzinfo=CHI)),
    ("feb-end", datetime(2026, 2, 28, 18, 0, tzinfo=CHI)),
    ("aug31", datetime(2026, 8, 31, 23, 59, tzinfo=CHI)),
    ("utc+12", datetime(2026, 6, 20, 10, 0, tzinfo=NZ)),
]


def _boom(tz=None):
    raise AssertionError("the wall clock was read")


@pytest.fixture(autouse=True)
def no_clock(monkeypatch):
    """Every test in this file, without having to ask for it."""
    monkeypatch.setattr(calendar, "now_local", _boom)


def _heard(said, now):
    """The model door: what the sentence itself says the day is."""
    return sentence_date(said, now.date())


def _forced(said, now):
    """The forced door: what commander hands get_calendar."""
    return calendar_range(said, now)


def _is_rank(said, now, door):
    """A rank answers about NO day: neither a date nor a question."""
    got = door(said, now)
    assert as_date(got) is None, (
        f"{said!r} at {now.date()} -> {got!r}: a rank was read as a day")
    assert not is_ask(got), f"{said!r} -> {got!r}: a rank should not ask"


def _not_a_day(said, now, door):
    """The grid's own judgement: a rank must not be read as a DAY.  An ASK
    is not a silent today -- he hears it and answers it -- so the shapes
    that already ask ("the 1st or 2nd week of october") are held to this
    weaker bar, and only to it."""
    got = door(said, now)
    assert as_date(got) is None, (
        f"{said!r} at {now.date()} -> {got!r}: a rank was read as a day")


def _is_day(said, now, door, day):
    """A date answers about the day it names."""
    got = door(said, now)
    seen = as_date(got)
    assert seen is not None, (
        f"{said!r} at {now.date()} -> {got!r}: the day it names was lost")
    assert seen.day == day, f"{said!r} at {now.date()} -> {seen}"


DOORS = [("model", _heard), ("forced", _forced)]
DOOR_IDS = [n for n, _ in DOORS]


# --------------------------------------------------------------------
# (1) the plural and the possessive of a word already on the list
#
# Measured on the round-8 grid: every one of these was TODAY on 61f0945
# and a DATE on dccb261, because _TAIL_WORD_RX captures the surface form
# whole and "meetings" is not the string "meeting".
PLURALS = [
    "the 3rd classes",            # 'classes' -> class, and the stem is the rank
    "the 2nd meetings",
    "the 2nd meeting's time",
    "the 3rd labs",
    "the 4th items",
    "the 2nd lecture's slides",
    "the 3rd rows",
    "the 2nd exams",
    "the 5th pages",
    "the 2nd persons",
    "the 3rd sessions",
    "the 2nd chapters",
    "the 3rd flights",
    "the 2nd options",
    "the 3rd sessions'",          # the plural possessive, apostrophe last
]


@pytest.mark.parametrize("said", PLURALS)
@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_a_plural_of_a_rank_noun_is_still_a_rank(said, iname, now, dname, door):
    _is_rank(said, now, door)
    _is_rank(f"{said} on my calendar", now, door)


# The other half of the stem rule, and the half that is easy to lose: a
# word whose STEM IS NOT ON THE LIST must stay a DATE.  "buses" is not
# "bus" is not a rank; "glasses" is not "glass"; and the plural must not
# be allowed to invent a rank the singular never had.
NOT_RANKS = [
    ("the 12th buses", 12),
    ("the 12th glasses", 12),
    ("the 12th tickets", 12),
    ("the 12th rehearsals", 12),      # the declared trade, in the plural
    ("the 12th slots", 12),
    ("the 12th is", 12),              # 'is' must not stem to 'i'
    ("the 12th was", 12),
    ("the 12th plus", 12),
]


@pytest.mark.parametrize("said,day", NOT_RANKS)
@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_a_stem_that_is_not_on_the_list_stays_a_date(said, day, iname, now,
                                                     dname, door):
    _is_day(said, now, door, day)


# --------------------------------------------------------------------
# (2) the six nouns that route the sentence to the calendar
CAL_WORDS = ["appointment", "event", "plan", "calendar", "schedule", "agenda"]


@pytest.mark.parametrize("word", CAL_WORDS)
def test_each_calendar_word_really_does_reach_the_calendar(word):
    """The reason these six matter more than the other fifty-nine: a
    sentence carrying one is GUARANTEED to be routed to the calendar by
    commander._CAL_READ_RX, so a rank reading it gets wrong is a rank the
    assistant WILL get wrong."""
    assert _CAL_READ_RX.search(f"when is the 2nd {word} on my calendar")


@pytest.mark.parametrize("word", CAL_WORDS)
@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_a_calendar_word_after_an_ordinal_is_a_rank(word, iname, now,
                                                    dname, door):
    for said in (f"the 2nd {word}",
                 f"when is the 2nd {word} on my calendar",
                 f"what's the 3rd {word}",
                 f"the 3rd {word}s"):           # and its plural
        _is_rank(said, now, door)


@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_the_calendar_words_do_not_eat_the_day_beside_them(iname, now,
                                                           dname, door):
    """"on my calendar" is not a rank frame -- it is where the question
    lives.  The six words become ranks only DIRECTLY behind the ordinal."""
    _is_day("the 12th on my calendar", now, door, 12)
    _is_day("what's on my calendar the 12th", now, door, 12)
    _is_day("on the 12th, check my schedule", now, door, 12)


# --------------------------------------------------------------------
# (3) the elided rank with no pronoun after it
#
# Read from the FRONT: a wh-word or demonstrative plus a copula, or a
# bare "the" with no preposition and one of the rank's own frames.
# "what's the first on my calendar" is NOT here, and its absence is a
# ruling, not an oversight -- see test_a_bare_what_leaves_the_ordinal_a_day
# below.  It answers about the 1st: the declared trade, out loud.
ELIDED = [
    "which meeting is the first",
    "which one is the 2nd",
    "which is the 4th",
    "which class is the 2nd",
    "which lab is the 3rd",
    "which of these is the 5th",
    "what number is the 4th",
    "that's the 2nd",
    "is that the 3rd",
    "mine was the 2nd",
    "the first on the list",
    "the 2nd on the list",
    "the 3rd from the top",
    "the first up",
    "the 1st after that",
]


@pytest.mark.parametrize("said", ELIDED)
@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_an_elided_rank_is_not_a_day(said, iname, now, dname, door):
    _is_rank(said, now, door)
    if "calendar" not in said:
        _is_rank(f"{said} on my calendar", now, door)


# THE PIN IN THE OTHER DIRECTION, which dccb261 never got.  Every widening
# of the elided rank is a new way for a DAY HE NAMED to become a silent
# today, so the shapes that must stay days are pinned in the same file.
STILL_DAYS = [
    ("the 12th", 12),
    ("on the 12th", 12),
    ("the 12th of october", 12),
    ("what's the 12th like", 12),            # round seven's own measured row
    ("what's the 12th like on my calendar", 12),
    ("on the 12th throughout the day", 12),
    ("on the 12th other than the lab", 12),
    ("the 12th free", 12),
    ("is the 12th free", 12),
    ("what's on my calendar the 12th", 12),
    ("on the 12th i have a thing", 12),      # the prepositional frame wins
    ("for the 12th", 12),
    ("the 12th from 2 to 4", 12),            # round five's evidence
    ("the 12th in london", 12),
    ("what's the 12th", 12),
    ("what's the 12th on my calendar", 12),
    ("what is the 12th on my calendar", 12),
    ("the 12th calendar-wise", 12),          # the compound is not the noun
    ("on the 12th, calendar-wise", 12),
    ("the 12th after lunch", 12),            # only "after THAT" is the frame
    ("on the 12th after that", 12),          # the preposition still wins
    ("the 12th up in dallas", 12),           # "up" must END the clause
]


@pytest.mark.parametrize("said,day", STILL_DAYS)
@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_the_shapes_that_must_stay_days_stay_days(said, day, iname, now,
                                                  dname, door):
    _is_day(said, now, door, day)


@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_a_bare_what_leaves_the_ordinal_a_day(iname, now, dname, door):
    """THE ONE PLACE ROUND EIGHT REFUSED THE ADVERSARY, pinned so nobody
    quietly grants it later.

    "what's the first on my calendar" (a rank) and "what's the 12th on my
    calendar" (a day) are the SAME sentence: "what" is not a subject the
    ordinal can be the complement of -- the ordinal is the subject.  Every
    rule that made the first a rank made the second one too, MEASURED
    2026-09-06: "what's the 12th on my calendar" -> today, silently, which
    is his original complaint wearing a ninth coat.

    So the elided rank requires a subject DISTINCT from the ordinal, and
    "what's the first on my calendar" answers about the 1st of the month.
    That is the declared trade doing exactly what it promises: a wrong day
    HE CAN HEAR AND CORRECT, chosen over a silent one he cannot."""
    _is_day("what's the 12th on my calendar", now, door, 12)
    _is_day("what's the 12th", now, door, 12)
    _is_day("what's the first on my calendar", now, door, 1)
    # ...while a subject of its own still makes it a rank
    _is_rank("which one is the first on my calendar", now, door)
    _is_rank("what number is the first on my calendar", now, door)


@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_a_preposition_beats_the_elided_rank(iname, now, dname, door):
    """The one thing that separates "the 2nd he walked in" from "on the
    12th i have a thing": the frame.  Same tail, opposite reading."""
    for tail in ("i have a thing", "you said", "we agreed", "it's busy",
                 "he's visiting", "they're coming"):
        _is_day(f"on the 12th {tail}", now, door, 12)
        _is_rank(f"the 12th {tail}", now, door)


@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("dname,door", DOORS, ids=DOOR_IDS)
def test_the_standing_pins_are_untouched(iname, now, dname, door):
    """The four rulings the branch already carries, re-read here because
    every round has broken one of them by accident."""
    _is_rank("the 2nd he walked in", now, door)
    _is_rank("the 1st you mentioned", now, door)
    _is_rank("the 2nd of three parts", now, door)
    _not_a_day("the 1st or 2nd week of october", now, door)
    _is_rank("the 3rd and final lecture", now, door)
    _is_day("2026/09/12", now, door, 12)


# --------------------------------------------------------------------
# (4) the list's own method, made self-enforcing
#
# _RANK_NOUNS says of itself that it is grown "only from a measured
# phrase, with the phrase written beside the word".  Until this test that
# was a promise in prose: dccb261 carried lecturer and weekend with no
# phrase anywhere, and ten more covered by a phrase for their group.
_SRC = Path(calendar.__file__).read_text(encoding="utf-8")
_PHRASE_RX = re.compile(
    r"^#\s{2,}(?P<word>[a-z]+)\s+\"(?P<phrase>[^\"]+)\"\s*$", re.M)


def _phrase_block():
    start = _SRC.index("RANK PHRASES BEGIN")
    end = _SRC.index("RANK PHRASES END")
    return _SRC[start:end]


def rank_phrases():
    return {m.group("word"): m.group("phrase")
            for m in _PHRASE_RX.finditer(_phrase_block())}


def test_the_phrase_block_and_the_frozenset_are_the_same_list():
    """A word with no phrase beside it is not on the list, and a phrase
    with no word on the list is a leftover.  Either way the suite says so
    rather than a ninth adversary."""
    listed = set(calendar._RANK_NOUNS)
    written = set(rank_phrases())
    assert written - listed == set(), (
        f"phrases for words not on the list: {sorted(written - listed)}")
    assert listed - written == set(), (
        f"on the list with no measured phrase beside it: "
        f"{sorted(listed - written)}")


@pytest.mark.parametrize("iname,now", INSTANTS[:3])
def test_every_rank_noun_earns_its_place_from_its_own_phrase(iname, now):
    """The phrase written beside each word is RUN, not admired: it must
    actually read as a rank at three instants through the model door."""
    bad = []
    for word, phrase in sorted(rank_phrases().items()):
        got = sentence_date(phrase, now.date())
        if as_date(got) is not None or is_ask(got):
            bad.append((word, phrase, got))
    assert not bad, "phrases that do not read as a rank: " + repr(bad)


def test_the_phrase_block_names_a_word_only_once():
    words = _PHRASE_RX.findall(_phrase_block())
    seen = [w for w, _ in words]
    assert len(seen) == len(set(seen)), "a word appears twice in the block"
