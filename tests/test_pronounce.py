"""Tests for jarvis.pronounce — the TTS pronunciation dictionary."""
import json

import pytest

from jarvis.pronounce import DEFAULT_PRONUNCIATIONS, Pronunciations


@pytest.fixture
def table(tmp_path):
    return Pronunciations(path=tmp_path / "tts_pronunciations.json")


def test_defaults_rewrite_workshop_jargon(table):
    out = table.apply("VSS runs on the GB10 with CUDA and Ollama.")
    assert out == "V S S runs on the G B ten with kooda and oh-llama."


def test_whole_token_only(table):
    # "GPU" inside "GPUs" is its own entry; "TTS" glued into a word is not.
    assert table.apply("two GPUs") == "two G P Us"
    assert table.apply("xTTSy") == "xTTSy"
    assert table.apply("the TTS engine") == "the T T S engine"


def test_case_rules(table):
    # Upper-case keys are exact; lower-case keys match any casing.
    assert table.apply("gpu") == "gpu"
    assert table.apply("Nvidia NVIDIA nvidia") == "en-vidia en-vidia en-vidia"


def test_symbols(table):
    assert table.apply("load 87% & rising") == "load 87 percent and rising"


def test_user_file_overrides_and_persists(table, tmp_path):
    table.add("Peyrovi", "pay-ROH-vee")
    table.add("CUDA", "coo-dah")               # override a default
    assert table.apply("Mr Peyrovi likes CUDA") == "Mr pay-ROH-vee likes coo-dah"
    data = json.loads((tmp_path / "tts_pronunciations.json").read_text())
    assert data == {"Peyrovi": "pay-ROH-vee", "CUDA": "coo-dah"}
    # A fresh instance reads the same file.
    again = Pronunciations(path=tmp_path / "tts_pronunciations.json")
    assert again.apply("CUDA") == "coo-dah"


def test_empty_spoken_disables_a_default(table):
    table.add("VSS", "")
    assert table.apply("VSS") == "VSS"


def test_remove(table):
    table.add("Foo", "fooo")
    assert table.remove("foo") is True
    assert table.apply("Foo") == "Foo"
    assert table.remove("foo") is False


def test_reloads_when_file_changes(table, tmp_path):
    path = tmp_path / "tts_pronunciations.json"
    path.write_text(json.dumps({"Spark": "the spark"}))
    import os
    os.utime(path, (1, 1))                     # force a distinct mtime
    assert table.apply("Spark") == "the spark"


def test_bad_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "tts_pronunciations.json"
    path.write_text("not json")
    t = Pronunciations(path=path)
    assert t.apply("VSS") == "V S S"
    assert t.user_items() == {}


def test_add_rejects_empty_word(table):
    with pytest.raises(ValueError):
        table.add("  ", "x")


def test_defaults_have_no_plain_english():
    # Guard against someone adding a common word: every default key is
    # jargon-shaped (has a digit, is not lower-case, or is a tool name).
    for key in DEFAULT_PRONUNCIATIONS:
        assert key != key.lower() or any(ch.isdigit() for ch in key) or key in {
            "nvidia", "ollama", "qwen", "pytorch", "pytest", "xdotool", "xclip",
            "nmcli", "ffmpeg", "tkinter", "sudo", "ok", "aarch64", "llama3.2",
            # calendar shorthand (2026-08-28). None is an English word, so
            # matching any casing is safe. "rm" and "sem" were considered and
            # rejected: "rm" would rewrite the shell command as "Room".
            "engr", "bldg", "dept", "lect", "appt", "tamu"}


# --------------------------------------------------- spoken abbreviations
#
# 2026-08-28: Jarvis read a calendar title aloud as "ENGR" rather than
# "Engineering". Course titles arrive from Navigate360 already abbreviated
# (MAGNETIC RESONANCE ENGR, Bldg, ETB), so the text is never written out for
# him -- the pronunciation layer is the only place to expand it.
@pytest.mark.parametrize("raw,spoken", [
    ("MAGNETIC RESONANCE ENGR", "Engineering"),
    ("Wisenbaker Engineering Bldg 049", "Building"),
    ("ECEN 442 Lab", "Lab"),
])
def test_course_abbreviations_are_spoken_in_full(raw, spoken):
    from jarvis.pronounce import Pronunciations
    said = Pronunciations(path=None).apply(raw)
    assert spoken.lower() in said.lower(), f"{raw!r} -> {said!r}"


def test_expansion_is_whole_token_only():
    """'Bldg' must expand; a word merely containing those letters must not."""
    from jarvis.pronounce import Pronunciations
    p = Pronunciations(path=None)
    assert "Building" in p.apply("Bldg 049")
    assert p.apply("Engrave the plate") == "Engrave the plate"


# ------------------------------------------------- number-compound hyphens
#
# Round 10, text 09: every F5 arm read "twenty-five minutes" as
# "twenty ... five minutes". The hyphen is entry 13 of F5's character
# vocabulary and the base weights render it as a prosodic break -- it is not
# a chunk boundary (F5's chunk_text splits on ";:,.!?" + whitespace only), so
# no amount of chunking config would have helped. The character has to not
# reach the engine, and the transcript keeps the English spelling.
@pytest.mark.parametrize("raw,spoken", [
    ("twenty-five minutes", "twenty five minutes"),
    ("seventy-eight", "seventy eight"),
    ("forty-two", "forty two"),
    ("ninety-nine", "ninety nine"),
    ("sixty-one degrees", "sixty one degrees"),
    ("thirty-three and eighty-seven", "thirty three and eighty seven"),
    # case is not the number's business: only the hyphen goes.
    ("Twenty-Five", "Twenty Five"),
    # ...even when unshout has already title-cased a shouted compound.
    ("TWENTY-FIVE MINUTES", "Twenty Five Minutes"),
])
def test_number_hyphens_become_spaces(table, raw, spoken):
    assert table.apply(raw) == spoken


@pytest.mark.parametrize("raw", [
    # Every other hyphen Jarvis speaks earns its pause, or is load-bearing.
    "re-enrol before Friday",
    "a well-known problem",
    "check your e-mail",
    "ETB-1035",
    "self-hosted",
    "twenty-something people",     # "something" is not a ones word
    "twenty-fivers",               # glued to more letters: not a compound
    "five-twenty",                 # ones-tens is not the compound shape
    # "and" is not a number word, so an and-chain is deliberately untouched:
    # the shape has never been measured and the narrow rule cannot surprise.
    "a hundred-and-five",
])
def test_non_number_hyphens_are_left_alone(table, raw):
    assert table.apply(raw) == raw


def test_a_hyphen_the_table_rewrites_around_still_survives(table):
    """The table spells TTS out; the hyphen between them is not ours."""
    assert table.apply("F5-TTS is the engine") == "F5-T T S is the engine"


def test_minutes_in_words_emits_a_space_not_a_hyphen():
    from jarvis.pronounce import _minutes_in_words
    assert _minutes_in_words(22) == "twenty two"
    assert _minutes_in_words(45) == "forty five"
    assert _minutes_in_words(5) == "oh five"       # unchanged
    assert _minutes_in_words(15) == "fifteen"      # unchanged
    assert _minutes_in_words(30) == "thirty"       # no trailing space


def test_clock_minutes_reach_the_engine_without_a_hyphen(table):
    assert table.apply("Your timer ends at 6:22 pm") == \
        "Your timer ends at six twenty two pee em"


# ---------------------------------------------- the sentence-final full stop
#
# Round 10: _TIME_RX ended "([ap])\.?\s?m\.?" and swallowed the trailing dot
# unconditionally, so "at 6:00 pm. Then we leave" reached F5 as
# "six pee em Then we leave" -- the sentence boundary gone, rendered as a
# run-on. The dot is the layer's to eat only when it is the abbreviation's
# own AND the sentence carries on.
@pytest.mark.parametrize("raw,spoken", [
    # bare "pm" + a full stop: the dot is punctuation, never ours.
    ("Tomorrow at 6:00 pm. Then we leave.",
     "Tomorrow at six pee em. Then we leave."),
    ("It is 9:10 am. The inbox is quiet.",
     "It is nine ten ay em. The inbox is quiet."),
    ("Ready at 7:30 pm.", "Ready at seven thirty pee em."),
    # "p.m." mid-sentence: that dot IS the abbreviation's, and goes.
    ("Booked for 4:10 p.m. and again later.",
     "Booked for four ten pee em and again later."),
    # "p.m." doing double duty as the full stop: the sentence keeps it.
    ("Tomorrow at 6:00 p.m. Then we leave.",
     "Tomorrow at six pee em. Then we leave."),
    ("The last one is at 4:10 p.m.", "The last one is at four ten pee em."),
    # no dot at all, and a comma is not a dot.
    ("Ready at 7:30 pm", "Ready at seven thirty pee em"),
    ("At 9:10 am, then lunch.", "At nine ten ay em, then lunch."),
    # the round-10 briefing line, which had to be reordered to dodge this.
    ("at 4:10 pm. The inbox is quiet, for once.",
     "at four ten pee em. The inbox is quiet, for once."),
])
def test_time_marker_keeps_the_sentence_boundary(table, raw, spoken):
    assert table.apply(raw) == spoken


# ------------------------------------------------ three-letter shouted words
#
# Round 10 defect 2: _SHOUT_RX wanted 4+ characters and the vowel rule wants
# 2 vowels, so the real calendar title "BIOSENSORS LAB II" became
# "Biosensors LAB II" and the engine spelled L-A-B. Three letters cannot be
# decided by shape -- "LAB" and "CPU" are the same shape, "GYM" has one vowel
# like "ETB" -- so it is decided by name.
@pytest.mark.parametrize("raw,out", [
    ("BIOSENSORS LAB II", "Biosensors Lab II"),
    ("GYM at four", "Gym at four"),
    ("ART HISTORY", "Art History"),
    ("BIO", "Bio"),
    ("SEM", "Sem"),
    ("REC", "Rec"),
    ("MED", "Med"),
    # initialisms stay for the table or the engine to spell.
    ("ENG 101", "ENG 101"),            # an abbreviation, not a word
    ("ETB 1035", "ETB 1035"),
    ("CPU", "CPU"),
    ("HDMI", "HDMI"),
    ("PHYS 208", "PHYS 208"),
    # roman numerals are left as they are: "II" is not a word and "Ii" would
    # be worse. Revisit only with a measurement.
    ("II", "II"),
    ("III", "III"),
    ("LAB III", "Lab III"),
])
def test_shouted_three_letter_words(raw, out):
    from jarvis.pronounce import unshout
    assert unshout(raw) == out


def test_shouted_calendar_title_end_to_end(table):
    assert table.apply("9:10 am BIOSENSORS LAB II") == \
        "nine ten ay em Biosensors Lab II"


# ------------------------------------------------------- bare clock times
#
# 2026-09-02 08:55, heard: "Your 9:10 is Biosensors, Wisenbaker 049" was
# rattled off. Nothing rewrote either number, so F5 was handed a 40-byte
# chunk and F5 buys time by the BYTE (scripts/f5_server.py: floor
# 0.45 + 0.04988*bytes below 41.3 bytes, native K = 0.06079 s/byte above it,
# and the floor ignores speed). 2.44 s allocated against 3.00 s of speech.
# Expanding the numbers is the whole fix: it is what the hyphen fix was, one
# layer along -- the transcript and the card keep the digits and only the
# engine sees the words.
@pytest.mark.parametrize("raw,spoken", [
    ("Your 9:10 is Biosensors.", "Your nine ten is Biosensors."),
    ("Your 12:45 is Biosensors.", "Your twelve forty five is Biosensors."),
    ("Tomorrow's 4:30.", "Tomorrow's four thirty."),
    ("At 9:05.", "At nine oh five."),
    ("At 10:15.", "At ten fifteen."),
    ("At 8:22.", "At eight twenty two."),        # and no hyphen, per round 10
    # On the hour: "Your four is Biosensors" is not English.
    ("Your 4:00 is Biosensors.", "Your four o'clock is Biosensors."),
    ("Your 12:00 is Biosensors.", "Your twelve o'clock is Biosensors."),
    # a 24-hour hour still reads as the clock it is
    ("At 09:10.", "At nine ten."),
])
def test_bare_clock_times_are_spoken(table, raw, spoken):
    assert table.apply(raw) == spoken


@pytest.mark.parametrize("raw", [
    # Ratios and scores: the reason bare times were left alone until now. A
    # two-digit minute and an hour of 1-12 is what rules every one of these
    # out, and it is exactly the shape jarvis/dossier.py clock() emits.
    "16:9",
    "4:3",
    "21:9",
    "a 2:1 margin",
    # a log timestamp is not a reading
    "08:56:15",
    "at 12:45:30 exactly",
    # not a clock at all: glued to a word, or mid-number
    "9:10ish",
    "112:45",
    "9:1",
    "9:60",
])
def test_non_clock_colon_numbers_are_left_alone(table, raw):
    assert table.apply(raw) == raw


def test_bare_pass_never_invents_a_meridiem(table):
    """He did not say am or pm, so neither does Jarvis."""
    out = table.apply("Your 9:10 is Biosensors.")
    assert "ay em" not in out and "pee em" not in out
    assert "colon" not in out


def test_marked_times_still_win_over_the_bare_pass(table):
    """The bare pass runs second; a marked time keeps its marker."""
    assert table.apply("It is 9:10 am. The inbox is quiet.") == \
        "It is nine ten ay em. The inbox is quiet."
    assert table.apply("Booked for 4:10 p.m. and again later.") == \
        "Booked for four ten pee em and again later."
    assert table.apply("Ready at 7:30 pm.") == "Ready at seven thirty pee em."


# ---------------------------------------------------------- room numbers
#
# THE RULE: a 2-4 digit run with a LEADING ZERO is an identifier and is said
# digit by digit. Nothing else. A leading zero is the one written form that
# cannot be a quantity, so the rule has no ambiguous side; the quantities
# below are pinned untouched. "ETB 1035" / "Ecen 404" / "731A" are real and
# still wrong, and are left alone on purpose -- see the module comment.
@pytest.mark.parametrize("raw,spoken", [
    ("Wisenbaker 049", "Wisenbaker zero four nine"),
    ("Wisenbaker 049.", "Wisenbaker zero four nine."),
    ("in 049, sir", "in zero four nine, sir"),
    ("room 07", "room zero seven"),
    ("0800", "zero eight zero zero"),
])
def test_leading_zero_runs_are_spoken_digit_by_digit(table, raw, spoken):
    assert table.apply(raw) == spoken


@pytest.mark.parametrize("raw", [
    # QUANTITIES. "10 minutes" must never become "one zero minutes".
    "10 minutes",
    "30 minutes",
    "about 10 minutes",
    "due in 78 days",
    "a high of 95 degrees",
    "500 of them",
    "Volume 100, sir.",
    "Yesterday: 179 turns",
    "the 2025 championship",
    # IDENTIFIERS WITHOUT A LEADING ZERO: out of scope, deliberately.
    "ETB 1035",
    "Ecen 404",
    "PHYS 208",
    "Jack E. Brown 731A",
    # a leading zero that belongs to something else: dates, versions, decimals
    "2026-09-02",
    "1.049",
    "0.5 seconds",
    "v0.99",
])
def test_number_runs_that_are_not_room_numbers_are_left_alone(table, raw):
    assert table.apply(raw) == raw


def test_the_incident_line_end_to_end(table):
    """The exact string F5 was handed at 08:55:50 on 2026-09-02.

    The log prints the POST-pronunciation text (jarvis/tts.py _speak_sync),
    which is how we know "BIOSENSORS" had already been un-shouted and the two
    numbers had not been touched at all.
    """
    heard = ("While you were out, sir: one message. "
             "Your 9:10 is Biosensors, Wisenbaker 049.")
    assert table.apply(heard) == (
        "While you were out, sir: one message. "
        "Your nine ten is Biosensors, Wisenbaker zero four nine.")


def test_the_incident_sentence_clears_the_f5_duration_floor(table):
    """Why the expansion is the fix and not a rate setting.

    scripts/f5_server.py pins a chunk under ~41 bytes to
    0.45 + 0.04988*bytes and IGNORES speed there, so the only lever on how
    long Jarvis takes over this sentence is how many bytes it is. At 40 bytes
    it was floored at 2.44 s for 3.00 s of speech; expanded it is over the
    threshold and takes F5's own (still byte-proportional) allotment.
    """
    said = table.apply("Your 9:10 is Biosensors, Wisenbaker 049.")
    assert len("Your 9:10 is Biosensors, Wisenbaker 049.".encode()) < 41
    assert len(said.encode()) > 41


def test_the_card_keeps_the_digits():
    """dossier.py composes the digits on purpose; only the engine sees words.

    clock() and room_words() are what the card and the transcript show, so
    they must be untouched by any of this.
    """
    from datetime import datetime
    from jarvis import dossier
    assert dossier.clock(datetime(2026, 9, 2, 9, 10)) == "9:10"
    assert dossier.room_words(
        "College Station Wisenbaker Engineering Bldg 049") == "Wisenbaker 049"


def test_fish_gets_neither_rewrite(table):
    """s2.1-pro normalises numbers itself; the rewrites ride the same flag
    the meridiem rewrite already rides (jarvis/tts.py per-engine table)."""
    raw = "Your 9:10 is Biosensors, Wisenbaker 049."
    assert table.apply(raw, rewrite_times=False) == raw
