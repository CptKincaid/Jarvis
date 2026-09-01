"""HIS OWN REQUEST, 2026-08-31, after an evening of 155 features by voice:

    "Set up a way to catch what the transcript says vs what he actually
     says and compare the two and do a bunch of runs on that."

Two of that evening's defects are in the log verbatim -- the timer said
"timer timer" (20:42:40) and the shopping list said "milk, milk, eggs, and
and bread" (20:56:05) -- and his note reports a third, the diagnostics
turn, that the log can neither confirm nor deny. That is the point: the
only record of a spoken line was ``speaking (f5): %.60s``, sixty characters
of the text BEFORE the pronunciation pass rewrote it, and no record at all
of what the card said. There was nothing to compare.

This covers the comparators (``spoken_form``, ``speech_divergence``,
``stutters``) and the harness that runs them over a corpus of his own
phrasings, scripts/speech_diff.py.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from jarvis import tts as tts_mod
from jarvis.tts import TTS, speech_divergence, speech_words, stutters

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location(
        "speech_diff", REPO / "scripts" / "speech_diff.py")
    mod = importlib.util.module_from_spec(spec)
    # In sys.modules BEFORE exec: @dataclass resolves its annotations
    # through sys.modules[cls.__module__], and a module loaded by path is
    # not there unless it is put there.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tts():
    # cache=False: a text test must never touch the cache the live Jarvis
    # is playing out of.
    return TTS(engine="f5", cache=False)


# --------------------------------------------------------------- stutters
def test_the_timer_said_timer_timer():
    """Verbatim from his log, 20:42:40 and 21:33:00."""
    said = stutters("Sir, your 30-second 30 seconds timer timer is up.")
    assert "timer timer" in said
    assert "30 second 30 seconds" in said, (
        "the same phrase in two spellings is still the same phrase twice")


def test_the_shopping_list_said_milk_milk_and_and_bread():
    """Verbatim from his log, 20:56:05 -- and note the card said it too, so
    only a check on the LINE catches this, never a transcript diff."""
    said = stutters(
        "Four on your shopping list, sir: milk, milk, eggs, and and bread.")
    assert said == ["milk milk", "and and"]


@pytest.mark.parametrize("line", [
    "It is 87 degrees and partly cloudy in College Station, sir.",
    "I had had enough of that lab, sir.",              # legitimate double
    "One note, sir: Bettany voice is the XTTS-1.",     # spelled out "X T T S"
    "You have BIOSENSORS at 9:10 am and a lab due Saturday, sir.",
])
def test_his_ordinary_replies_are_not_flagged(line):
    """A warning that cries wolf is a warning he learns to ignore."""
    tts = TTS(engine="f5", cache=False)
    assert not stutters(tts.spoken_form(line))


# ----------------------------------------------------------- what is lost
def test_a_substitution_is_not_a_divergence():
    """Rewriting "18%" as "18 percent" is the point of the pronunciation
    pass, and must not be reported as content going missing."""
    notes = speech_divergence("It is 18% of 74, sir.",
                              "It is 18 percent of 74, sir.")
    assert not [n for n in notes if "not spoken" in n]


def test_words_the_room_never_hears_are_reported():
    """_clean_for_speech DELETES inline code rather than reading it."""
    notes = speech_divergence("See `config.py` for that, sir.",
                              "See  for that, sir.")
    assert any("not spoken" in n and "config" in n for n in notes)


def test_a_long_reply_is_cut_and_nothing_says_so(tts, harness):
    """MAX_SPEAK_LENGTH truncates the speech; the card still shows it all."""
    reply = ("Your briefing, sir. " + "There is another item on the list. "
             * 20)
    assert len(reply) > tts.MAX_SPEAK_LENGTH
    result = harness.check_line(tts, reply)
    kinds = {f.kind for f in result.findings}
    assert "truncated" in kinds, result.findings
    assert result.spoken != reply


def test_speech_words_ignores_punctuation_and_case():
    assert speech_words("Noted, sir.") == ["noted", "sir"]


def test_spoken_form_is_what_the_engine_is_handed(tts):
    """The two steps the room takes, in the room's order."""
    line = "Your ELECTRICAL DESIGN LAB II presentation is at 4:10 pm, sir."
    assert tts.spoken_form(line) == tts._pronounce(tts._clean_for_speech(line))


# ------------------------------------------------------------- the harness
def test_the_harness_flags_his_shopping_list_as_blocking(tts, harness):
    result = harness.check_line(
        tts, "Four on your shopping list, sir: milk, milk, eggs, and and bread.")
    assert any(f.blocking for f in result.findings)


def test_the_harness_passes_an_ordinary_reply(tts, harness):
    result = harness.check_line(tts, "Noted, sir: your mom is Heather.")
    assert result.findings == []


def test_the_harness_runs_his_whole_corpus(tts, harness):
    """"do a bunch of runs on that" -- the corpus is his own lines, pasted
    in because /tmp is wiped at boot and jarvis.log goes with it."""
    assert len(harness.SPOKEN) >= 50
    assert len(harness.ASKED) >= 30
    results = [harness.check_line(tts, line) for line in harness.SPOKEN]
    assert len(results) == len(harness.SPOKEN)
    # The two known-open defects from other lanes are still in there, and
    # the harness must keep finding them until they are fixed.
    stutter_lines = {r.shown for r in results
                     for f in r.findings if f.kind == "stutter"}
    assert any("timer timer" in ln for ln in stutter_lines)


def test_the_harness_refuses_to_change_his_data_in_live_mode(harness):
    """--live points at the assistant he actually uses, with his real
    lists, timers and calendar in it."""
    for utterance in harness.ASKED:
        assert not harness._MUTATING.search(utterance), (
            f"{utterance!r} would change something on his box")
    for order in ["Set a timer for 30 seconds",
                  "Add milk to the shopping list",
                  "Remember that it is Mara and I's anniversary on "
                  "September 20th",
                  "jarvis ok delete my last email",
                  "Scratch, Buy Milk off the list",
                  "Forget what I just said.",
                  "Clear the shopping list"]:
        assert harness._MUTATING.search(order), order
    # ...and it must not refuse a plain question that happens to contain
    # one of those words.
    assert not harness._MUTATING.search("What do you remember?")


def test_the_harness_skips_log_truncated_lines(tmp_path, harness):
    """The log itself truncates the spoken text, and a line cut by the LOG
    must not be reported as a line cut by the speech path."""
    log = tmp_path / "jarvis.log"
    exactly_60 = "x" * 60
    log.write_text(
        "20:42:40.000 jarvis.tts INFO speaking (f5): Noted, sir.\n"
        f"20:42:41.000 jarvis.tts INFO speaking (f5): {exactly_60}\n")
    lines, skipped = harness.load_log(log)
    assert lines == ["Noted, sir."]
    assert skipped == 1


# ------------------------------------------------------ the live log line
def test_the_log_prints_what_is_actually_spoken(tts, caplog, monkeypatch):
    """It used to print the text BEFORE the pronunciation pass, which is
    why a whole evening of divergences left no trace."""
    monkeypatch.setattr(tts, "load", lambda: True)
    monkeypatch.setattr(tts, "_speak_pipelined", lambda text, engine: None)
    with caplog.at_level("INFO", logger="jarvis.tts"):
        tts._speak_sync("It is at 4:10 pm today, sir.")
    spoken = tts.spoken_form("It is at 4:10 pm today, sir.")
    assert any(spoken[:40] in rec.getMessage()
               for rec in caplog.records if "speaking" in rec.getMessage())


def test_a_lossy_rewrite_is_warned_about(tts, caplog, monkeypatch):
    monkeypatch.setattr(tts, "load", lambda: True)
    monkeypatch.setattr(tts, "_speak_pipelined", lambda text, engine: None)
    with caplog.at_level("WARNING", logger="jarvis.tts"):
        tts.speak("The answer is in `results.json`, sir.")
    assert any("diverges from the transcript" in rec.getMessage()
               for rec in caplog.records), (
        "content dropped between the card and the speech must be visible "
        "in the log -- that is the whole ask")


def test_a_stutter_is_warned_about(tts, caplog, monkeypatch):
    monkeypatch.setattr(tts, "load", lambda: True)
    monkeypatch.setattr(tts, "_speak_pipelined", lambda text, engine: None)
    with caplog.at_level("WARNING", logger="jarvis.tts"):
        tts._speak_sync("Sir, your 30-second 30 seconds timer timer is up.")
    assert any("said twice in a row" in rec.getMessage()
               for rec in caplog.records)


def test_the_module_and_the_harness_share_one_comparator(harness):
    """If they ever diverge, the harness stops describing the room."""
    assert harness.tts_mod is tts_mod
