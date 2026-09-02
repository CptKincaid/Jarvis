"""Regression tests for the F5 respelling reaching Breeze-TTS-2.

DEFECT (jarvis/pronounce.py + jarvis/tts.py):

``pronounce.apply`` took two booleans, and one of them -- ``rewrite_times`` --
gated THREE unrelated rewrites: the meridiem rewrite ("6:00 pm" -> "six pee
em"), the bare-clock rewrite ("9:10" -> "nine ten") and the leading-zero
rewrite ("049" -> "zero four nine"). ``jarvis/tts.py`` set that one flag True
for every engine but fish. So a new engine got all three or none, and there
was no way to say "reads a colon correctly, but has no duration floor".

Breeze-TTS-2 is exactly that engine, and Hunter heard it. His round-11 note
on the Breeze clip of text 12 -- which the round-11 texts had already written
in the F5 spelling, "at nine ten ay em ... at twelve forty pee em" -- was:

    "sounded pretty good but he said pm as pey or pay he did not say p em"

MEASURED 2026-09-02 on the pinned Q4 checkpoint (int4 all MLPs, group-32
depth, bf16 attention and text encoder, template ref_edit_tata, cfg_scale
4.0, temperature 0.9, repetition_penalty 1.1, ref jarvis_voice_ref_f5.wav),
each probe rendered at 10 seeds (6 for the am probes) and read back by
wav2vec2-lv-60-espeak-cv-ft, an IPA CTC head with no language model. The
round-11 gate ASR (whisper-large-v3-turbo) cannot see this defect at all --
it transcribed EVERY arm as "4pm", because its decoder is a language model
that repairs the abbreviation it expects, and the letter-level
wav2vec2-base-960h cannot either, because the name of the letter P *is*
/piː/. Clean /p iː ɛ m/, out of 10 renders each:

    "The quarterly review is at four P M, sir."     10   (not shipped, below)
    "... at four PM, sir."                           7
    "... at four p.m., sir."                         7
    "... at 4:00 pm, sir."                           6   <- untouched text
    "... at four pee em, sir."                       3   <- what F5 emits

and the am side, clean /eɪ ɛ m/ out of 6:

    "Biosensors is at 9:10 am, sir."                 6   <- untouched text
    "... at nine ten AM, sir."                       6
    "... at nine ten ay em, sir."                    3   <- what F5 emits

The three failures in that last row came back as /aɪ ɪ m/ and /aɪ ə m/ --
"I'm" -- which is the fish defect of 2026-08-28 word for word.

So the F5 respelling is the WORST of the five spellings measured, and it is
the one he heard. Pooling pm and am it is 6 of 16 against 42 of 52 for every
other spelling (Fisher two-tailed p=0.0033); the am row alone is p=0.0245.
Against the untouched text ALONE it is 6/16 vs 12/16, p=0.073 -- suggestive,
not proven -- and what carries the decision there is the SHAPE of the
failures rather than their count: the untouched form's only failure is a lax
/p ɪ ɛ m/, still audibly "p em", while the respelling collapses a vowel
("pee-um", "I'm"), which is the complaint he actually made.

The other two rules are unnecessary for a different reason: Breeze read "Your
9:10 lecture" as "NINE TEN" and "Wisenbaker 049" as "ZERO FOUR NINE"
unprompted on every seed, and it has no duration floor for the byte-buying to
climb over.

NOT DONE, deliberately: "four P M" scored 10 of 10, beating even the
untouched text. It is a real lead but two-tailed p=0.087 at n=10 over a
tense-vs-lax vowel nobody has HEARD, and this project has already spent a
round finding that respelling "reply" as "ree ply" won on ASR and lost by
ear. Swapping one unheard respelling for another is not the fix; if it is
ever wanted it is one row in ENGINE_RULES and a blind round, in that order.
"""
import pytest

from jarvis import pronounce


BRIEFING = ("On the calendar you have Biosensors at 9:10 am, "
            "Magnetic Resonance Engineering at 12:40 pm, and at "
            "4:10 pm your Electrical Design Lab presentation.")


# --------------------------------------------------------- the defect
def test_breeze_is_never_handed_the_f5_meridiem_spelling():
    """The whole of the reported defect, in one assertion."""
    said = pronounce.apply(BRIEFING, engine="breeze")
    assert "pee em" not in said
    assert "ay em" not in said
    assert said == BRIEFING, "Breeze reads the written form correctly"


@pytest.mark.parametrize("raw", [
    "The quarterly review is at 4:00 pm, sir.",
    "Biosensors is at 9:10 am, sir.",
    "Your meeting is at 12:00 p.m.",
    "Tomorrow at 7:01 AM.",
])
def test_a_marked_time_reaches_breeze_as_written(raw):
    assert pronounce.apply(raw, engine="breeze") == raw


@pytest.mark.parametrize("raw", [
    # Measured over 10 seeds: Breeze reads this as "NINE TEN" without being
    # told to -- never as a colon, never as "nine one zero".
    "Your 9:10 lecture is in Wisenbaker, sir.",
    # 10 seeds: three digit words every time, never "forty nine". The rule
    # that would rewrite it exists only to buy bytes against F5's
    # 0.45 + 0.04988*bytes floor, which Breeze does not have.
    "Biosensors is in Wisenbaker 049, sir.",
    # 10 seeds: "THREE THOUSAND AND NINETY", 10 of 10. The thousands comma
    # is safe -- and the leading-zero rule already spares it (the comma is
    # in its lookbehind), so this pins the pair rather than one of them.
    "You walked 3,090 steps today, sir.",
])
def test_breeze_reads_its_own_numbers(raw):
    assert pronounce.apply(raw, engine="breeze") == raw


def test_breeze_still_gets_the_vocabulary_and_the_symbols():
    """Only the ENGINE compensations are off. The jargon table is about what
    the words ARE, not about how one engine reads bytes, and no engine knows
    that "engr" is Engineering."""
    said = pronounce.apply("VSS is at 95%, engr bldg 330.", engine="breeze")
    assert "V S S" in said
    assert "percent" in said
    assert "Engineering" in said and "Building" in said
    assert "330" in said, "not a leading-zero identifier; left alone"


def test_breeze_keeps_unshout():
    """Measured both ways -- Breeze read "BIOSENSORS LAB II" as words either
    way -- so the rule is a no-op here rather than a fix. It is not an F5
    rule and not a duration rule, it is the acronym rule, and fish keeps it
    too; a measured no-op is a better default than an unmeasured change."""
    assert pronounce.apply("BIOSENSORS LAB II", engine="breeze") == \
        "Biosensors Lab II"


def test_breeze_keeps_the_number_hyphen_spacing():
    """Left unconditional, on evidence rather than by omission. The F5 pause
    at the hyphen measured 190 ms; the same measurement on Breeze -- CTC
    frame alignment across TWENTY->FIVE, 20 ms frames, 10 seeds a side --
    put BOTH arms at a mean 0.078 s, the same number to three decimals. The
    hyphen buys Breeze no pause, so there is nothing here to make
    engine-aware and the rule stays as it is for everyone."""
    assert pronounce.apply("Your twenty-five minute timer is up.",
                           engine="breeze") == \
        "Your twenty five minute timer is up."


# -------------------------------------------------- the split itself
def test_the_three_time_rules_are_separately_switchable():
    """The point of the change. One boolean could not express "reads a colon
    fine but has no duration floor", which is what Breeze turned out to be.
    """
    line = "Your 9:10 is Biosensors at 6:00 pm, Wisenbaker 049."
    only_marked = pronounce.RuleSet(bare_times=False, id_digits=False)
    only_bare = pronounce.RuleSet(marked_times=False, id_digits=False)
    only_ids = pronounce.RuleSet(marked_times=False, bare_times=False)
    table = pronounce.get()
    assert "six pee em" in _with(table, line, only_marked)
    assert "9:10" in _with(table, line, only_marked)
    assert "nine ten" in _with(table, line, only_bare)
    assert "6:00 pm" in _with(table, line, only_bare)
    assert "zero four nine" in _with(table, line, only_ids)
    assert "049" not in _with(table, line, only_ids)


def _with(table, line, rules, engine="probe"):
    """Apply ``line`` under one hand-built rule set."""
    saved = pronounce.ENGINE_RULES.get(engine)
    pronounce.ENGINE_RULES[engine] = rules
    try:
        return table.apply(line, engine=engine)
    finally:
        if saved is None:
            pronounce.ENGINE_RULES.pop(engine, None)
        else:
            pronounce.ENGINE_RULES[engine] = saved


def test_an_unknown_engine_gets_every_rule():
    """Defaulting ON is the safe direction: an unmeasured engine is assumed
    to be as weak as XTTS was. The opposite default would ship raw "6:00 pm"
    to a new engine and nobody would hear about it until Hunter did."""
    rules = pronounce.rules_for("some-engine-nobody-has-measured")
    assert (rules.marked_times, rules.bare_times, rules.id_digits,
            rules.unshout) == (True, True, True, True)
    assert pronounce.rules_for(None) == pronounce.RuleSet()
    assert pronounce.rules_for("") == pronounce.RuleSet()


def test_the_old_flags_still_gate_what_they_always_gated():
    """``rewrite_times`` meant all three clock/number rules, and callers that
    pass it still get that -- the flag is an override on top of the engine's
    row, not a fifth rule."""
    line = "Your 9:10 is at 6:00 pm, Wisenbaker 049."
    assert pronounce.apply(line, rewrite_times=False) == line
    assert pronounce.apply(line, engine="xtts", rewrite_times=False) == line
    # ...and it can force the rules back ON for an engine whose row is off.
    forced = pronounce.apply(line, engine="breeze", rewrite_times=True)
    assert "six pee em" in forced and "nine ten" in forced
    assert "zero four nine" in forced


# ------------------------------------------- only the engine sees it
def test_the_engine_decides_and_the_wiring_carries_the_engine(tmp_path):
    """``TTS._pronounce`` must ask for its OWN engine's row.

    ``_engine`` is set directly rather than through the constructor because
    "breeze" is not in ``_ENGINES`` on this branch -- registering it belongs
    with the synth and the sidecar, and registering it here without them
    would route breeze to the XTTS default and file the result under the
    breeze cache key. The pronunciation seam is independent of that, and
    this pins it so the two land already agreeing.
    """
    from jarvis.tts import TTS
    line = "Your 9:10 is Biosensors at 6:00 pm, Wisenbaker 049."
    voice = TTS(engine="edge", cache=False, cache_dir=tmp_path)
    assert voice._pronounce(line) == \
        "Your nine ten is Biosensors at six pee em, Wisenbaker zero four nine."
    voice._engine = "breeze"
    assert voice._pronounce(line) == line


def test_only_the_engine_sees_the_rewrite(tmp_path, monkeypatch):
    """The invariant the module was written for: the card and the transcript
    keep what Hunter wrote, and the rewrite exists only between _speak_sync
    and the synth. ``JarvisReply`` and ``TTS.speak`` are both handed the
    original by ``JarvisApp._say``, so proving it here means proving the
    queue carries the original and the SYNTH is the first thing to see the
    rewrite."""
    from jarvis.tts import TTS
    line = "Your 9:10 is Biosensors, Wisenbaker 049."
    voice = TTS(engine="edge", cache=False, cache_dir=tmp_path)

    queued = []
    monkeypatch.setattr(voice, "_speak_sync", queued.append)
    voice.speak(line, block=True)
    assert queued == [line], "the queue carries what the card shows"

    monkeypatch.undo()
    synthesised = []
    monkeypatch.setattr(voice, "_synth_edge",
                        lambda text, path: synthesised.append(text))
    monkeypatch.setattr(voice, "_start_amp_feeder", lambda path: None)
    monkeypatch.setattr(voice, "_play", lambda path: None)
    voice.speak(line, block=True)
    assert synthesised == [
        "Your nine ten is Biosensors, Wisenbaker zero four nine."]
    assert line == "Your 9:10 is Biosensors, Wisenbaker 049."


# ------------------------------------------------------ no F5 change
def test_f5_is_untouched():
    """The shipped voice. Every rule it has ever had, byte for byte."""
    assert pronounce.rules_for("f5") == pronounce.RuleSet()
    assert pronounce.apply(BRIEFING, engine="f5") == (
        "On the calendar you have Biosensors at nine ten ay em, "
        "Magnetic Resonance Engineering at twelve forty pee em, and at "
        "four ten pee em your Electrical Design Lab presentation.")
    assert pronounce.apply("Your 9:10 is Biosensors, Wisenbaker 049.",
                           engine="f5") == \
        "Your nine ten is Biosensors, Wisenbaker zero four nine."


def test_no_engine_but_f5_gained_or_lost_a_rule():
    assert pronounce.rules_for("edge") == pronounce.RuleSet()
    assert pronounce.rules_for("xtts") == pronounce.RuleSet()
    fish = pronounce.rules_for("fish")
    assert (fish.marked_times, fish.bare_times, fish.id_digits) == \
        (False, False, False)
    assert fish.unshout is True, "fish has always kept unshout"
