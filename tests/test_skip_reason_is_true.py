"""ROUND 5 SHIPPED A STATUS LINE THAT IS UNTRUE WHEN IT IS SAID.

Round 5 added three silences to status.txt, and one of the three lies.
SKIP_SUFFIXES has seven members and SKIP_PREFIXES three, and status_text
hardcoded exactly two reasons: ends ".tmp", or "starts with a dot".
MEASURED on the round-5 tip, 3 of 3 wrong:

    movie.crdownload -> "(e.g. movie.crdownload -- name ends .tmp)"
    draft.swp        -> "(e.g. draft.swp -- name ends .tmp)"
    ~$report.docx    -> "(e.g. ~$report.docx -- name starts with a dot)"

The count is right and the filename is right; the REASON is invented.  He
reads that line to decide whether to rename the file, so a wrong reason
sends him to rename the wrong part of the name.  It is a line this round
introduced, so it is this round's to fix.

Ten rules can fire.  All ten are tested here.

NOTHING HERE OPENS A SOCKET, TOUCHES ~/Desktop, OR SPEAKS TO HPCOMPUTER.
"""
# ruff: noqa: F811 -- `home` is a pytest FIXTURE imported from
# tests/test_foldersync.py rather than copied.
import pytest

from jarvis import foldersync as fs

from tests.test_foldersync import (FakeTransport, home,  # noqa: F401
                                   syncer)


# name -> the clause he must be shown.  One per member of SKIP_SUFFIXES and
# SKIP_PREFIXES, so adding an eighth suffix without a word for it fails.
CASES = [
    ("book.part",          'name ends ".part"'),
    ("movie.partial",      'name ends ".partial"'),
    ("film.crdownload",    'name ends ".crdownload"'),
    ("iso.download",       'name ends ".download"'),
    ("sheet.tmp",          'name ends ".tmp"'),
    ("draft.swp",          'name ends ".swp"'),
    (".hidden",            'name starts with "."'),
    ("~$report.docx",      'name starts with "~$"'),
    (".jarvis-part-x.txt", 'name starts with ".jarvis-part-"'),
]


@pytest.mark.parametrize("name,clause", CASES)
def test_every_skip_rule_says_the_rule_that_actually_fired(name, clause):
    assert fs.skip_reason(name) == clause, (
        f"{name} is skipped by a different rule than the sentence claims")


def test_the_note_suffix_is_the_tenth_rule_and_has_its_own_words():
    """NOTE_SUFFIX is in SKIP_SUFFIXES too, and it is the one skip that is
    not his file at all -- it is a note THIS LANE wrote.  It is filtered
    out of the count before the sentence is built, so he is never told
    about it; but the rule still has to have a true word for it."""
    assert fs.skip_reason("report.pdf" + fs.NOTE_SUFFIX) == (
        'it is a note I wrote, not a file of yours')


def test_a_name_no_rule_skips_has_no_reason():
    assert fs.skip_reason("quarterly.xlsx") == ""


def test_every_member_of_both_tuples_has_a_word():
    """The pin that survives an eighth suffix being added.  A rule with no
    sentence must fail HERE, not in his status file."""
    for suffix in fs.SKIP_SUFFIXES:
        why = fs.skip_reason("example" + suffix)
        assert why, f"SKIP_SUFFIXES member {suffix!r} has no sentence"
        assert suffix in why or suffix == fs.NOTE_SUFFIX, (
            f"{suffix!r} is skipped but the sentence says {why!r}")
    for prefix in fs.SKIP_PREFIXES:
        why = fs.skip_reason(prefix + "example.txt")
        assert why, f"SKIP_PREFIXES member {prefix!r} has no sentence"
        assert prefix in why, (
            f"{prefix!r} is skipped but the sentence says {why!r}")


def test_the_status_file_shows_the_true_reason_for_the_file_it_names(home):
    """End to end, through the real Outbox scan and the real status text.
    The three the adversary measured wrong, plus one that was right."""
    s = syncer(home, FakeTransport())
    for name in ("film.crdownload", "draft.swp", "~$report.docx",
                 "sheet.tmp"):
        (s.paths.outbox / name).write_bytes(b"half a file")

    s._candidates()                       # fills _skipped_outbox
    named = s._skipped_outbox[0]
    text = s.status_text()

    assert f"e.g. {named}" in text, text
    assert fs.skip_reason(named) in text, (
        f"status.txt names {named} and gives a reason that is not the rule "
        f"that fired.  Expected {fs.skip_reason(named)!r}.\n{text}")
    assert "4 file(s) in your Outbox I am not sending" in text, text
