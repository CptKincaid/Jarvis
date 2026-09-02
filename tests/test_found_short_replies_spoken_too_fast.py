"""Short replies were "spoken too fast", and a 30-second timer said nothing.

From Hunter's 155-feature voice session, 2026-08-31. Four features he
otherwise PASSED:

    #27 take a note   "But when i asked for note he said it but said it
                       really fast"
    #28 to-dos        "said buy milk really fast"
    #30 search notes  "said note fast"
    #15 timer         "Reminder: 30 seconds timer on transcript activated
                       but no noise or speech. lets add that"

WHAT THE MEASUREMENT SAID (live F5 sidecar, 11 lines from 9 to 108 bytes,
plus the exact cached wavs from ~/.aiws_trainer/tts_cache that he actually
heard). The first three are NOT a rate defect:

    bytes  voiced   ms/syllable   line
       11   0.70 s      233       "Noted, sir."
       27   1.42 s      203       "One to-do, sir: a buy milk."
       43   2.18 s      218       "One note, sir: Bettany voice is the ..."
       64   3.31 s      195       "Very good, sir; I'll remind you ..."
      108   5.86 s      195       "The first is the quarterly review ..."

Articulation is flat at ~19 bytes of text per voiced second across the whole
range, and the short lines are at the SLOW end of it. Nothing renders them
faster.

What IS wrong is that a short reply has no rate control at all, and is over
before it lands. scripts/f5_server.py's duration_floor pins a short
utterance to 0.45 + 0.04988*bytes with NO speed term, and fix_duration
overrides infer()'s own speed handling -- so "Noted, sir." renders to
exactly 0.99 s at speed 0.85, 0.80, 0.75 AND 0.70, identical audio. Below
~41 bytes the blind-validated speed=0.85 is inert and the only thing that
decides how long Jarvis takes is the byte count. Hence: the fix is words.

Padding with the address is the fix; padding with whitespace is not. Both
buy bytes, only one buys speech:

    "Buy milk."         9 B   1.53 s   0.58 s voiced   62% dead   floor OFF
    "Buy milk.   "     12 B   1.03 s   0.60 s voiced   42% dead   floor on
    "Buy milk, sir."   14 B   1.14 s   0.77 s voiced   33% dead   floor on

#15 is a different bug entirely and the log has it outright
(/tmp/vss_voice/jarvis.log, 20:42:40):

    timekeeper: timer fired '30 seconds timer'
    quiet: held (timer): Sir, your 30-second 30 seconds timer timer is up.
    status[info]: Held (MAGNETIC RESONANCE ENGR until 8:50 pm): ...

The line was never dropped, it was HELD -- timers went out proactive=True,
so app._say handed the announcement to the quiet gate, which was inside a
calendar course window. Only ALARMS ring, alarms.sound is "" and
sound.earcons is off, so the hold left exactly what he reported: a
transcript line, no noise, no speech.
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest

import jarvis.tools.timekeeper as tk_mod
from jarvis import tts as tts_mod
from jarvis.tools.notes import NotesStore
from jarvis.tools.timekeeper import Timekeeper


def _bytes(s: str) -> int:
    return len(s.encode("utf-8"))


# --------------------------------------------------------------- the floor
def test_the_pad_target_clears_the_floors_own_cliff():
    """Below 10 bytes F5 sets local_speed=0.3, duration_floor returns None
    and the line renders on the unvalidated upstream path (measured: 62%
    dead air, 0.93 s of leading silence). The pad target must sit above it."""
    assert tts_mod.F5_FLOOR_MIN_BYTES == 10
    assert tts_mod.SHORT_LINE_BYTES > tts_mod.F5_FLOOR_MIN_BYTES


@pytest.mark.parametrize("bare,padded", [
    ("Buy milk.", "Buy milk, sir."),
    ("buy milk", "buy milk, sir."),
    ("Noted.", "Noted, sir."),
    ("Which one?", "Which one, sir?"),   # a question stays a question
    ("Done", "Done, sir."),
])
def test_a_terse_answer_is_padded_into_a_small_sentence(bare, padded):
    assert tts_mod.pad_short_line(bare) == padded


def test_the_pad_lifts_a_two_word_answer_over_the_cliff():
    """The whole point: "buy milk" alone cannot reach the floored regime."""
    assert _bytes("Buy milk.") < tts_mod.F5_FLOOR_MIN_BYTES
    out = tts_mod.pad_short_line("Buy milk.")
    assert _bytes(out) >= tts_mod.F5_FLOOR_MIN_BYTES


def test_an_already_addressed_line_is_left_alone():
    """Otherwise "Noted, sir." becomes "Noted, sir, sir." -- a stammer."""
    for line in ("Noted, sir.", "Yes, sir?", "SIR."):
        assert tts_mod.pad_short_line(line) == line


@pytest.mark.parametrize("line", [
    "One to-do, sir: a buy milk.",
    "Very good, sir; I'll remind you to call the dentist in a minute.",
    "The first is the quarterly review at quarter past six in the evening.",
])
def test_anything_long_enough_is_returned_byte_for_byte(line):
    """A no-op above the threshold, so every long-form render -- and every
    speech-cache entry keyed on it -- is exactly what it was before."""
    assert tts_mod.pad_short_line(line) == line


def test_empty_and_punctuation_only_text_survives_the_pad():
    for line in ("", "   ", "...", "?"):
        tts_mod.pad_short_line(line)          # must not raise
    assert tts_mod.pad_short_line("") == ""
    assert tts_mod.pad_short_line("...") == "..."


# ------------------------------------------------- wired into the TTS door
def _tts(engine):
    return tts_mod.TTS(engine=engine, cache=False, pronunciation=False)


def test_the_f5_path_pads_a_terse_line_on_its_way_to_synthesis():
    """_clean_for_speech is what speak(), prewarm() and the phone renderer
    all run, so the pad has to land there or the cache keys diverge."""
    t = _tts("f5")
    assert t._clean_for_speech("Buy milk.") == "Buy milk, sir."


def test_only_f5_is_padded():
    """Edge and fish normalise their own duration from the text and XTTS
    derives it from the mel decoder; none of them has the byte-count cliff,
    and rewriting their text would be a change nobody measured."""
    assert tts_mod._ENGINE_NEEDS_SHORT_PAD["f5"] is True
    for engine in ("edge", "xtts", "fish", "breeze"):
        assert tts_mod._ENGINE_NEEDS_SHORT_PAD[engine] is False
        assert _tts(engine)._clean_for_speech("Buy milk.") == "Buy milk."


def test_a_long_reply_through_the_f5_door_is_unchanged():
    line = "One to-do, sir: a buy milk, and the shopping list is still open."
    assert _tts("f5")._clean_for_speech(line) == line


# ------------------------------------------------------- the read-backs
@pytest.fixture
def store(tmp_path):
    s = NotesStore(tmp_path / "notes.db")
    yield s
    s.close()


def test_the_todo_read_back_gives_the_content_room(store):
    """#28, "said buy milk really fast". Was "One to-do, sir: a buy milk."
    -- 1.78 s end to end, of which the part he asked for was the last 0.6 s.
    Measured after: 34 bytes, 2.13 s, +26% air for the same content."""
    store.add("todo", "buy milk")
    line = store.list_text("todo")
    assert line == "You have one to-do, sir: buy milk."
    assert _bytes(line) > _bytes("One to-do, sir: buy milk.")


def test_the_note_read_back_gives_the_content_room(store):
    """#27/#30: the same line answered both "read me the note" and the
    note search."""
    store.add("note", "Bettany voice is the XTTS-1")
    line = store.list_text("note")
    assert line.startswith("You have one note, sir: ")
    assert line.endswith("Bettany voice is the XTTS-1.")


def test_the_note_search_read_back_keeps_the_address_out_of_the_subject(store):
    """"You have one note, sir about bettany" -- the address has to follow
    the whole subject, not sit inside it."""
    store.add("note", "Bettany voice is the XTTS-1")
    line = store.search_text("note", "bettany")
    assert line == ("You have one note about bettany, sir: "
                    "Bettany voice is the XTTS-1.")
    assert ", sir about" not in line


def test_a_named_list_read_back_is_widened_too(store):
    store.make_list("shopping")
    store.add("list:shopping", "milk")
    assert store.list_text("list:shopping") == \
        "You have one item on your shopping list, sir: milk."


def test_plural_read_backs_are_left_alone(store):
    """They already carry enough words; widening them would only pad."""
    store.add("todo", "buy milk")
    store.add("todo", "call the dentist")
    assert store.list_text("todo") == \
        "Two to-dos, sir: buy milk and call the dentist."


def test_every_note_reply_clears_the_floor_cliff(store):
    """Nothing this tool says may land on the sub-10-byte branch."""
    store.add("note", "x")
    store.add("todo", "y")
    lines = [store.list_text("note"), store.list_text("todo"),
             store.search_text("note", "x"), store.lists_text(),
             store.clear_line("todo", 2)]
    for line in lines:
        assert _bytes(line) >= tts_mod.F5_FLOOR_MIN_BYTES, line


# ------------------------------------------------------- #15 the timer
class _Clock:
    def __init__(self, dt):
        self.t = dt.timestamp()

    def now(self):
        return self.t

    def tick(self, seconds):
        self.t += seconds


@pytest.fixture
def spoken(tmp_path, monkeypatch):
    """A Timekeeper whose say() records the proactive flag app._say reads."""
    monkeypatch.setattr(tk_mod, "_run", lambda argv, **kw: None)
    calls = []
    clock = _Clock(datetime(2026, 8, 31, 20, 42))

    def say(text, proactive=False, kind=""):
        calls.append((text, proactive, kind))

    t = Timekeeper(tmp_path / "tk.db", say=say, cfg={}, now=clock.now,
                   tick_s=0.01, ring=False, cache_dir=tmp_path / "cache",
                   notify=False)
    t.calls, t.clock = calls, clock
    try:
        assert t._say_takes_flag
        yield t
    finally:
        t.close()


def test_a_finished_timer_is_not_proactive_so_quiet_cannot_hold_it(spoken):
    """#15. proactive=True is what sent "Sir, your ... timer is up." into
    quiet.hold() instead of the speaker. He set it thirty seconds earlier
    and asked to be told: that is an answer arriving late, not something
    Jarvis decided to bring up, and a countdown read back out of a digest
    after the window closes is worthless."""
    spoken.add_timer(30, "")
    spoken.clock.tick(31)
    spoken.tick()
    timers = [c for c in spoken.calls if c[2] == "timer"]
    assert timers, "the timer fired without speaking at all"
    text, proactive, _ = timers[0]
    assert proactive is False, (
        "a fired timer went out proactive -- app._say will hand it to the "
        "quiet gate and Hunter gets a transcript line and silence")
    assert "timer is up" in text


def test_a_reminder_is_still_proactive(spoken):
    """The digest exists for these: "call the dentist in an hour" landing
    mid-lecture is exactly what quiet hours are for."""
    spoken.add_reminder(spoken.clock.now() + 30, "call the dentist")
    spoken.clock.tick(31)
    spoken.tick()
    reminders = [c for c in spoken.calls if c[2] == "reminder"]
    assert reminders and reminders[0][1] is True


def test_an_auto_derived_timer_label_is_not_stuttered_back(spoken):
    """"Set a timer for 30 seconds" arrives labelled "30 seconds timer",
    and the labelled template read it back as "Sir, your 30-second 30
    seconds timer timer is up." (jarvis.log 20:42:40). It never reached his
    ear only because the line was being held; now that it speaks, it must
    not say that."""
    spoken.add_timer(30, "30 seconds timer")
    spoken.clock.tick(31)
    spoken.tick()
    text = [c[0] for c in spoken.calls if c[2] == "timer"][0]
    assert "timer timer" not in text
    assert text.lower().count("30") == 1, text


def test_a_real_timer_label_is_still_spoken(spoken):
    """The stutter guard must only catch labels that restate the duration."""
    spoken.add_timer(600, "pasta")
    spoken.clock.tick(601)
    spoken.tick()
    text = [c[0] for c in spoken.calls if c[2] == "timer"][0]
    assert "pasta" in text


def test_firewall():
    from jarvis import logs
    assert str(logs.LOG_DIR) != "/tmp/vss_voice"
    assert not str(os.environ.get("JARVIS_ASSISTANT_CONFIG", "")).startswith(
        os.path.expanduser("~/.config/jarvis"))
