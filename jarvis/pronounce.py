"""TTS pronunciation dictionary for Jarvis.

Both engines mangle the workshop's jargon: XTTS spells "VSS" as a word,
Edge reads "GB10" as "gee-bee-ten-ish" and "Ollama" with a hard O. The
dictionary rewrites those tokens into how Jarvis should *say* them, right
before synthesis (``TTS._speak_sync``), so the transcript still shows the
real spelling while the voice says the right thing.

Two layers:
- ``DEFAULT_PRONUNCIATIONS`` — shipped jargon (whole-token, case-sensitive).
- the user file ``~/.aiws_trainer/tts_pronunciations.json`` — overrides and
  additions, edited by "Jarvis, pronounce X as Y" (``add()``) or by hand.
  Reloaded automatically when its mtime changes.

Matching is whole-token: a key matches only when it is not glued to other
word characters ("GPU" matches in "the GPU," but not in "GPUs" unless the
plural is its own entry). Keys are case-sensitive unless they are written
in lower case, in which case they match any casing ("nvidia" also matches
"Nvidia" and "NVIDIA").

Usage:
    from jarvis import pronounce
    pronounce.apply("VSS runs on the GB10")   # -> "V S S runs on the G B ten"
    pronounce.get().add("Peyrovi", "pay-ROH-vee")
"""
from __future__ import annotations

import dataclasses
import json
import re
import threading
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("pronounce")

USER_FILE = PATHS.AIWS / "tts_pronunciations.json"

# spelled -> spoken. Keep entries to things both engines actually get wrong;
# ordinary English needs no help. Lower-case keys match any casing.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    # calendar shorthand. Course and room titles arrive from Navigate360 and
    # Canvas already abbreviated ("MAGNETIC RESONANCE ENGR", "Wisenbaker
    # Engineering Bldg 049"), so the text is never written out for him and
    # this is the only place it can be expanded. Lower-case keys match any
    # casing; matching is whole-token, so "Engrave" is untouched.
    "engr": "Engineering",
    "bldg": "Building",
    "dept": "Department",
    "lect": "Lecture",
    "appt": "appointment",
    "tamu": "Texas A and M",
    # the workshop
    "VSS": "V S S",
    "GB10": "G B ten",
    "DGX": "D G X",
    "nvidia": "en-vidia",
    "GPU": "G P U",
    "GPUs": "G P Us",
    "CPU": "C P U",
    "CUDA": "kooda",
    "aarch64": "arm sixty-four",
    "HDMI": "H D M I",
    "X11": "X eleven",
    "GNOME": "nome",
    "ollama": "oh-llama",
    "llama3.2": "llama three point two",
    "qwen": "kwen",
    "XTTS": "X T T S",
    "TTS": "T T S",
    "STT": "S T T",
    "ASR": "A S R",
    "ChromaDB": "chroma D B",
    "SAM3": "sam three",
    "ONNX": "onyx",
    "YOLO": "yolo",
    "ReID": "re I D",
    "RTSP": "R T S P",
    "MJPEG": "M J peg",
    "KPI": "K P I",
    "KPIs": "K P Is",
    "AGV": "A G V",
    "AGVs": "A G Vs",
    "VLM": "V L M",
    "LLM": "L L M",
    "MoE": "M O E",
    "JARVIS": "Jarvis",
    # tools
    "pytorch": "pie torch",
    "pytest": "pie test",
    "xdotool": "X do tool",
    "xclip": "X clip",
    "nmcli": "N M C L I",
    "ffmpeg": "F F M peg",
    "tkinter": "T K inter",
    "Tk": "tee kay",
    "sudo": "soo-doo",
    "JSON": "jason",
    "YAML": "yammel",
    "SSH": "S S H",
    "API": "A P I",
    "URL": "U R L",
    "ETA": "E T A",
    "CLI": "C L I",
    "GUI": "gooey",
    "OK": "okay",
    "ok": "okay",
    "PID": "P I D",
}

# Symbols that have no word boundary; applied after the token pass.
DEFAULT_SYMBOLS: dict[str, str] = {
    "%": " percent",
    "&": " and ",
    "°C": " degrees Celsius",
    "°F": " degrees Fahrenheit",
    "~/": "home slash ",
}


# ------------------------------------------------------- clock times
#
# The vocabulary table below is token-based and has no notion of a clock, so
# an "h:mm" token used to reach the engine verbatim and XTTS read the colon
# form digit by digit -- "6:00 pm" came out as "six zero pm" (heard
# 2026-08-28 answering a calendar question). The calendar, alarm, reminder
# and briefing tools all emit times in exactly this shape.
#
# BARE times are rewritten too, since 2026-09-02 -- see _BARE_TIME_RX below,
# which is where the reasoning for that (and the shape it is narrowed to)
# lives. It still runs after the meridiem pass, but no longer DEPENDS on
# that: the two became separately switchable per engine (ENGINE_RULES), so
# the bare pattern carries its own "not a meridiem" lookahead rather than
# relying on the marked pass having eaten those digits first.
_ONES = ("twelve", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven")
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}

# Group 4 is the abbreviation's inner dot ("p.m."), group 5 the trailing one.
# The trailing dot is NOT always ours to eat -- see _spoken_time.
_TIME_RX = re.compile(
    r"\b(1[0-2]|0?[1-9]|[01]\d|2[0-3]):([0-5]\d)\s*"
    r"([ap])(\.?)\s?m(\.?)(?![a-z])",
    re.IGNORECASE)

# What follows the marker when its dot is doing double duty as the full stop:
# whitespace then a capital (the next sentence), or nothing at all.
_SENTENCE_END_RX = re.compile(r"\s+[A-Z]|\s*$")


def _minutes_in_words(m: int) -> str:
    """1-59 as English minutes past the hour ("oh five", "twenty two").

    A SPACE between the tens and the ones, not the hyphen English spelling
    wants: F5 hears "twenty-five" as "twenty ... five" (round 10, every F5
    arm on text 09). ``space_number_hyphens`` below would catch this anyway
    -- emitting the space here means the clock path never depends on it,
    and the mechanism is written up there.
    """
    if m < 10:
        return f"oh {_ONES[m]}"          # "oh five", never bare "five"
    if m < 20:
        return ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen")[m - 10]
    tens, ones = divmod(m, 10)
    word = _TENS[tens]
    return word if not ones else f"{word} {_ONES[ones]}"


# "am"/"pm" left as letters came out of XTTS as a spelled "A M" (heard
# 2026-08-28). Writing them the way they are SAID keeps the marker without
# handing the engine an abbreviation to spell.
_HALF = {"a": "ay em", "p": "pee em"}


def _spoken_time(match: "re.Match") -> str:
    hour, minute, half = int(match.group(1)), int(match.group(2)), match.group(3)
    hour = hour % 12                      # 12:xx and 00:xx both read "twelve"
    spoken = _ONES[hour]
    if minute:
        spoken = f"{spoken} {_minutes_in_words(minute)}"
    spoken = f"{spoken} {_HALF[half.lower()]}"
    # The trailing dot. The old pattern swallowed it unconditionally, so
    # "at 6:00 pm. Then we leave" reached F5 as "six pee em Then we leave"
    # -- the sentence boundary gone, rendered as a run-on (round 10). The
    # dot is ours to eat only when it is the abbreviation's own ("p.m.")
    # AND the sentence carries on; a dot after the bare "pm" is punctuation,
    # and an abbreviation dot that also ends the sentence stays a full stop.
    if match.group(5):
        inner_dot = bool(match.group(4))
        rest = match.string[match.end():]
        if not inner_dot or _SENTENCE_END_RX.match(rest):
            spoken += "."
    return spoken


# --------------------------------------------------- bare clock times
#
# 2026-09-02 08:55, heard: "Your 9:10 is Biosensors, Wisenbaker 049" was
# rattled off. _TIME_RX above needs an am/pm marker, so a bare "9:10" reached
# F5 as four literal characters -- and F5 buys time BY THE BYTE, not by the
# word. scripts/f5_server.py allocates
#
#     native = ref_frames / ref_text_bytes * gen_bytes / speed   (bytes)
#     floor  = 0.45 + 0.04988 * gen_bytes                        (bytes)
#
# so at the shipped reference (528 frames / 109 bytes, speed 0.85, i.e.
# K = 0.06079 s/byte, floor binding below 41.3 bytes) that sentence -- 40
# bytes, one chunk -- was given 2.44 s. What it has to SAY is "Your nine ten
# is Biosensors, Wisenbaker zero four nine": 16 syllables, 3.12-3.73 s at the
# 195-233 ms/syllable this voice measures, or 3.00 s by this speaker's own
# affine law (sec = 0.2540 + 0.04988*bytes, n=400, r=0.960). A shortfall of
# 0.55-1.29 s, and the model compresses to fit. It is not a rate setting: the
# floor ignores speed entirely (jarvis/tts.py, SHORT_LINE_BYTES).
#
# Measured over every distinct chunk Jarvis spoke this boot (n=42): all 10
# chunks with a negative margin contain a digit, and all 25 chunks without a
# digit have a positive one. Digits are the whole of the effect. So the fix is
# spending bytes on the digits -- the same "the character has to not reach the
# engine" move as the hyphen above -- and once expanded the sentence is 55
# bytes, clears the 41.3-byte floor into the native regime, and gets 3.34 s
# against a 3.00 s need: a +0.35 s margin, in line with the +0.19 s and
# +0.20 s the healthy lines have. No engine-side change is needed for it.
#
# The shape is narrow ON PURPOSE. The old note here was right that a bare
# "16:9" is a ratio, not a clock, so: hours 1-12 only (which is exactly what
# every composer emits -- jarvis/dossier.py clock() is "{h%12 or 12}:{mm:02d}")
# and a two-digit minute, which rules out "16:9", "21:9" and "4:3"; and the
# lookarounds keep it off a timestamp ("08:56:15") and out of the middle of a
# longer number. EVIDENCE that this is safe on his text: of the 18 distinct
# "h:mm" strings spoken across this boot and the previous one, all 18 are
# clock readings and none is a ratio or a score.
#
# On the hour gets "o'clock", not a bare hour word: "Your four is Biosensors"
# is not English, and "o'clock" is also 8 bytes of the time this line was
# short of. The marked path keeps its own shape ("six pee em"), where the
# marker already carries the sentence.
#
# The trailing "not a meridiem" lookahead is what makes this rule STANDALONE.
# Ordering used to be the only thing keeping it honest: speak_times ran the
# marked pass first, so by the time this pattern ran there was no "6:00 pm"
# left to find. Once the two became separately switchable (ENGINE_RULES) an
# engine could ask for the bare rule alone, and this pattern would then eat
# the digits out from under the marker -- "at 6:00 pm" became "at six
# o'clock pm", inventing an o'clock and stranding an unspoken abbreviation.
# On every engine that runs both passes the lookahead is a no-op, because
# the marked pass has already consumed those digits.
_BARE_TIME_RX = re.compile(
    r"(?<![\w:])(0?[1-9]|1[0-2]):([0-5]\d)(?![\w:])(?!\s*[apAP]\.?\s?[mM])")


def _spoken_bare_time(match: "re.Match") -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    spoken = _ONES[hour % 12]
    if not minute:
        return f"{spoken} o'clock"
    return f"{spoken} {_minutes_in_words(minute)}"


def speak_bare_times(text: str) -> str:
    """Rewrite a bare "9:10" as "nine ten" (no meridiem is invented)."""
    return _BARE_TIME_RX.sub(_spoken_bare_time, text)


def speak_times(text: str, *, marked: bool = True, bare: bool = True) -> str:
    """Rewrite "6:00 pm" as "six pm" so the engine does not spell the colon.

    Marked times first, then bare ones: the marked pattern has to see its
    "am"/"pm" still attached to the digits it belongs to. The two are
    separable because they answer different questions -- whether the engine
    spells a colon, and whether it needs bytes bought against F5's duration
    floor -- and Breeze-TTS-2 needs neither for different measured reasons.
    """
    if marked:
        text = _TIME_RX.sub(_spoken_time, text)
    return speak_bare_times(text) if bare else text


# ---------------------------------------------------- room numbers
#
# The other half of the 2026-09-02 line. "049" is three bytes carrying four
# syllables ("zero four nine"), and by the arithmetic above every byte is
# 60 ms of the sentence's air, so the room number alone was starving it of
# ~0.67 s. jarvis/dossier.py room_words() deliberately keeps the digits --
# the card shows the room the way the calendar wrote it -- so this is the
# only place it can be said out loud.
#
# THE RULE, and it is deliberately one rule and not a clever one: a run of
# 2-4 digits WITH A LEADING ZERO is an identifier, and is said digit by
# digit. Nothing else is touched. A leading zero is nearly the one thing in
# written English that cannot be a quantity -- nobody writes "049 minutes" --
# and the tests pin the quantities ("10 minutes", "30 minutes", "78 days",
# "95 degrees") unchanged.
#
# THERE IS EXACTLY ONE EXCEPTION, and the first cut of this rule shipped
# with it: the three-digit group AFTER A THOUSANDS COMMA is a quantity and
# it can begin with a zero. The lookbehind was (?<![\w:./-]), which has no
# comma in it, so "1,000 songs" came out "1,zero zero zero songs" and
# "$1,050" came out "$1,zero five zero" -- strictly worse than the digits
# F5 was given before, and silent (speech_divergence sees a substitution,
# not a loss). Every comma-grouped number Jarvis QUOTES rather than formats
# reaches this: an LLM answer, a headline, a note or mail read back.
# jarvis/mathspeak.py strips thousands commas itself, which is why the
# calculator path never showed it. The comma is in the lookbehind now, and
# NOT in the trailing lookahead -- "Wisenbaker 049, sir" must still expand.
#
# WHAT IS DELIBERATELY NOT DONE, having tried it: the other identifiers in
# his calendar -- "ETB 1035", "Ecen 404", "Jack E. Brown 731A" -- have no
# leading zero, and every shape rule that reaches them also reaches a real
# quantity he has actually been told. "a capitalised word then 3-4 digits"
# eats "Volume 100, sir." (spoken 2026-09-01); "digits then a capital letter"
# eats the model sizes "70B" and "32B". It is the LAB-versus-CPU problem from
# _SHOUTED_WORDS again: shape cannot separate them. "Ecen 404" (-0.95 s) IS
# still wrong; it wants either a building/course list drawn from his own
# calendar or the expansion done at the composer, where the string is known
# to be a room.
#
# AND THE RESIDUAL IS NOT THAT ONE COURSE CODE. Replaying every
# `speaking (f5):` line of the 2026-09-02 boot through _split_sentences,
# scripts/f5_server.duration_floor (ref 528 frames / 109 bytes at speed
# 0.85) and this speaker's affine law with each digit run expanded to the
# words it must say: 10 chunks negative before this commit, 9 after -- this
# rule flips exactly ONE, his (-0.50 -> positive). The WORST margin in the
# boot is not a course code at all but
#     "Yesterday: 179 turns, median wait 2.1 seconds, worst 24.6."
# at -1.92 s, twice as starved as "Ecen 404", plain cardinals and decimals,
# and a line jarvis/dayreview.py speaks EVERY DAY. Also still negative:
# "45 minutes" / "your 45-minute timer is up" (-0.15), "a high of 95
# degrees" (-0.04), "Chiro in 30 minutes" (-0.00).
#
# So the real remaining work is the CARDINAL class, not the identifier one:
# "30" -> "thirty" is +4 bytes = +0.20 s, "2.1" -> "two point one" is +11.
# Unlike an identifier a cardinal has no ambiguous side at all -- a cardinal
# is a cardinal -- so it is a SMALLER decision than the course-code list
# above, not a larger one. It is left out of this commit only for scope.
#
# The lookarounds keep it off an ISO date ("2026-09-02"), a version ("1.049"),
# a decimal, and a clock (which the pass above has already consumed anyway).
_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine")
_ID_DIGITS_RX = re.compile(r"(?<![\w:.,/-])0\d{1,3}(?![\w:/-])(?!\.\d)")


def _spoken_digits(match: "re.Match") -> str:
    return " ".join(_DIGIT_WORDS[int(c)] for c in match.group(0))


def speak_room_numbers(text: str) -> str:
    """Say a leading-zero identifier ("049") digit by digit."""
    return _ID_DIGITS_RX.sub(_spoken_digits, text)


# ------------------------------------------------------- shouted words
#
# Calendar titles arrive from iCal as Hunter typed them, and course names are
# shouted: "9:10 am BIOSENSORS at ...". An engine treats an all-caps token as
# an acronym, so XTTS did not say "biosensors" -- it produced what was heard
# as "bio censors" (2026-08-28). There is no list to enumerate here; the
# titles change every semester.
#
# So: an all-caps run with the vowels to carry syllables is a WORD that
# happens to be shouted, and gets title case. One without them ("HDMI",
# "PHYS", "ETB") is an initialism and is left alone for the table or the
# engine to spell. Known jargon is substituted before this runs, so entries
# like VSS -> "V S S" are already gone by the time we get here.
#
# The two rules above cannot see a three-letter word at all. The length gate
# wanted 4+ characters and the vowel gate wants 2 vowels, so the real calendar
# title "BIOSENSORS LAB II" came out as "Biosensors LAB II" and the engine
# spelled L-A-B (round 10 write-up, defect 2). Shape cannot separate these:
# "LAB" and "CPU" are the same shape, "GYM" has one vowel like "ETB", and
# "USA" has two like "AIR". So three letters is decided BY NAME -- the short
# list of real words that have actually turned up shouted in one of his course
# titles. Keep it to words. "ENG" is an abbreviation of Engineering, not a
# word, and belongs with ETB/CPU/HDMI/PHYS where the engine spells it out.
# Roman numerals after such a word ("LAB II") are left as they are: "II" is
# not a word, title-casing it would produce "Ii", and nothing measured yet
# says what F5 does with it.
_SHOUTED_WORDS = frozenset({"LAB", "GYM", "ART", "BIO", "SEM", "REC", "MED"})

_SHOUT_RX = re.compile(r"\b[A-Z][A-Z'&-]{2,}\b")
_VOWELS = set("AEIOUY")


def _unshout(match: "re.Match") -> str:
    word = match.group(0)
    if word in _SHOUTED_WORDS:
        return word.title()
    if len(word) < 4:
        return word                       # only the list above rescues these
    letters = [c for c in word if c.isalpha()]
    if sum(1 for c in letters if c in _VOWELS) < 2:
        return word                       # an initialism, not a word
    return word.title()


def unshout(text: str) -> str:
    """Title-case shouted WORDS, leaving initialisms for the engine."""
    return _SHOUT_RX.sub(_unshout, text)


# --------------------------------------------------- number compounds
#
# Round 10, text 09: EVERY F5 arm read "twenty-five minutes" as
# "twenty ... five minutes" -- a beat of silence where the hyphen is.
#
# The hyphen is not a chunk boundary anywhere in the path. scripts/f5_server.py
# hands req["text"] to F5Api.infer() verbatim, and F5's own splitter
# (f5_tts/infer/utils_infer.py chunk_text) splits on ";:,.!?" followed by
# whitespace and on CJK punctuation -- never on "-". With the shipped
# reference clip (5.805 s, 108 B) max_chars comes out at 256, so a one-line
# answer is a single batch regardless. What the hyphen IS: entry 13 of the
# model's character vocabulary (f5_tts/infer/examples/vocab.txt), a token the
# base weights learned as a prosodic break. The pause is F5 reading the
# character, so the fix is the character not reaching it.
#
# It lives here rather than in the tools that emit the text so the card still
# shows the English spelling and only the engine sees the space.
#
# Number words ONLY, and only a tens word joined to a ones word. A hyphen
# earns its pause everywhere else Jarvis speaks -- "re-enrol", "well-known",
# "e-mail", "F5-TTS", "ETB-1035" -- and the tests pin those untouched.
# "a hundred-and-five" is deliberately out of scope: "and" is not a number
# word, the shape has never been measured, and a narrow rule is the one that
# cannot surprise him.
_TENS_WORDS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy",
               "eighty", "ninety")
_ONES_WORDS = ("one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine")
_NUMBER_HYPHEN_RX = re.compile(
    r"(?<!\w)(%s)-(%s)(?!\w)" % ("|".join(_TENS_WORDS), "|".join(_ONES_WORDS)),
    re.IGNORECASE)


def space_number_hyphens(text: str) -> str:
    """Space the hyphen in "twenty-five" so F5 does not pause mid-number."""
    return _NUMBER_HYPHEN_RX.sub(r"\1 \2", text)


# ------------------------------------------------ per-engine rule sets
#
# Every rewrite above is a COMPENSATION for one engine's defect, never an
# improvement in itself, and ``TTS._pronounce`` has always passed the engine's
# needs in -- it just passed them as two booleans, and ``rewrite_times`` gated
# three unrelated rules at once. They are unrelated: the meridiem rewrite is
# "the engine spells the colon" (XTTS said "six zero pm", 2026-08-28), while
# the bare-clock and leading-zero rules are F5 BYTE-FLOOR arithmetic -- they
# buy bytes against 0.45 + 0.04988*bytes, which is a property of
# scripts/f5_server.py and of no other engine. One flag could not say "reads a
# colon fine but has a duration floor", and Breeze-TTS-2 needed the split.
#
# BREEZE, measured 2026-09-02 on the pinned Q4 checkpoint (int4 all MLPs,
# group-32 depth, bf16 attention/text encoder, ref_edit_tata, cfg 4.0, temp
# 0.9, ref jarvis_voice_ref_f5.wav), every probe rendered at 3-10 seeds and
# read back by three judges: whisper-large-v3-turbo (the round-11 gate),
# wav2vec2-base-960h greedy CTC (no language model), and
# wav2vec2-lv-60-espeak-cv-ft (IPA). Whisper is USELESS on this question --
# it wrote "4pm" for every arm including the broken ones, because its decoder
# is a language model that repairs the abbreviation it expects. The IPA head
# is what settled it:
#
#   marked_times  OFF. Hunter's round-11 note on the Breeze clip of text 12
#     was "he said pm as pey or pay he did not say p em", and the respelling
#     this rule emits is the CAUSE, not the victim. Clean meridiem out of 10
#     renders each: "four P M" 10, "four PM" 7, "four p.m." 7, the untouched
#     "4:00 pm" 6, and the respelling this rule emits, "four pee em", 3 --
#     last of five. Pooled with the am probes it is 6 of 16 against 42 of 52
#     for every other spelling (Fisher two-tailed p=0.0033); against the
#     untouched form ALONE it is 6/16 vs 12/16, p=0.073, i.e. suggestive
#     rather than proven, and the qualitative split is what carries it: the
#     untouched form's only failure is a lax /p ɪ ɛ m/, still audibly "p em",
#     while the respelling fails by COLLAPSING a vowel -- /p iː ə m/,
#     "pee-um", in 6 of 10 -- which is the complaint he actually made. The am
#     side is the fish defect verbatim: "nine ten ay em" came out /aɪ ɪ m/ or
#     /aɪ ə m/ -- "I'm" -- in 3 of 6 seeds, while "9:10 am" and "nine ten AM"
#     were clean in 12 of 12 (p=0.0245). And Breeze reads the written form
#     correctly on its own: text 12's "9:10 am ... 12:40 pm ... 4:10 pm"
#     transcribed identically to the round-11 respelled arm on all 5 seeds.
#   bare_times    OFF. "Your 9:10 lecture is in Wisenbaker" came back as
#     "NINE TEN" from the prior-free CTC head on every seed, so there is
#     nothing to fix; and the rule's only justification is F5's byte floor,
#     which Breeze -- a 12.5 Hz multi-codebook LM whose duration is generated,
#     not allocated -- does not have.
#   id_digits     OFF. Same two reasons. "Wisenbaker 049" came back as
#     "ZERO FOUR NINE" unprompted on every seed.
#   unshout       ON. Not an F5 rule and not a duration rule: it is the
#     acronym rule, and it stays on for every engine including fish. Breeze
#     read "BIOSENSORS LAB II" as words rather than letters, so the rule is a
#     no-op here rather than a fix -- but a no-op that has been measured both
#     ways is a better default than an unmeasured change.
#
# NOT TAKEN, deliberately, and this is the interesting one: "four P M"
# (spaced capitals) rendered a clean /p iː ɛ m/ in 10 of 10 seeds, beating the
# untouched "4:00 pm" at 6 of 10. That is a real lead, but it is two-tailed
# p=0.087 at n=10, the difference is a tense-vs-lax vowel in the letter P that
# nobody has HEARD, and this project has already paid for exactly that
# mistake: respelling "reply" as "ree ply" won on ASR confidence for F5 and
# LOST by ear. Swapping one unheard respelling for another is not what the
# complaint asked for. If it is ever wanted it is a one-line row here and a
# blind round, in that order.
#
# EDGE and XTTS keep all four, which is what shipped. Nothing measured says
# they need the two byte-floor rules either -- neither has a duration floor --
# but nobody has listened to those two engines without them, and the split is
# here to make that a separate, testable question rather than a side effect.
@dataclasses.dataclass(frozen=True)
class RuleSet:
    """Which of the per-engine rewrites one engine actually needs."""

    marked_times: bool = True     # "6:00 pm" -> "six pee em"
    bare_times: bool = True       # "9:10" -> "nine ten"
    id_digits: bool = True        # "049" -> "zero four nine"
    unshout: bool = True          # "BIOSENSORS" -> "Biosensors"


# Fish's s2.1-pro and Breeze-TTS-2 both normalise their own text; the two
# arrived at the same row by separate measurement, not by copying.
_SELF_NORMALISING = RuleSet(marked_times=False, bare_times=False,
                            id_digits=False)

ENGINE_RULES: dict[str, RuleSet] = {
    "edge": RuleSet(),
    "xtts": RuleSet(),
    "f5": RuleSet(),
    "fish": _SELF_NORMALISING,
    "breeze": _SELF_NORMALISING,
}


def rules_for(engine: Optional[str]) -> RuleSet:
    """The rule set for ``engine``; everything on for an unknown name.

    Defaulting ON is the safe direction: an engine nobody has measured is
    assumed to be as weak as XTTS was, so it gets the help. The opposite
    default would silently ship raw "6:00 pm" to a new engine.
    """
    return ENGINE_RULES.get(engine or "", RuleSet())


class Pronunciations:
    """A pronunciation table: shipped defaults + a user JSON file."""

    def __init__(self, path: Optional[Path] = None,
                 defaults: Optional[dict] = None,
                 symbols: Optional[dict] = None):
        self.path = Path(path) if path else USER_FILE
        self._defaults = dict(DEFAULT_PRONUNCIATIONS if defaults is None
                              else defaults)
        self._symbols = dict(DEFAULT_SYMBOLS if symbols is None else symbols)
        self._user: dict[str, str] = {}
        self._mtime: float = -1.0
        self._rx: Optional[re.Pattern] = None
        self._table: dict[str, str] = {}
        self._lower: dict[str, str] = {}
        self._lock = threading.Lock()
        self.load()

    # ------------------------------------------------------------ file
    def load(self) -> None:
        """(Re)load the user file. Missing file = defaults only."""
        with self._lock:
            user: dict[str, str] = {}
            mtime = -1.0
            try:
                if self.path.exists():
                    mtime = self.path.stat().st_mtime
                    data = json.loads(self.path.read_text())
                    if isinstance(data, dict):
                        user = {str(k).strip(): str(v) for k, v in data.items()
                                if str(k).strip()}
                    else:
                        log.warning("pronunciation file is not an object: %s",
                                    self.path)
            except Exception:
                log.exception("pronunciation file unreadable: %s", self.path)
            self._user = user
            self._mtime = mtime
            self._rebuild()

    def save(self) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._user, indent=2,
                                          ensure_ascii=False, sort_keys=True))
                tmp.replace(self.path)
                self._mtime = self.path.stat().st_mtime
            except Exception:
                log.exception("pronunciation save failed: %s", self.path)

    def _maybe_reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else -1.0
        except OSError:
            return
        if mtime != self._mtime:
            self.load()

    # ----------------------------------------------------------- table
    def _rebuild(self) -> None:
        table = dict(self._defaults)
        # A user entry whose key differs only by case replaces the default.
        for key, spoken in self._user.items():
            for existing in [k for k in table if k.lower() == key.lower()]:
                del table[existing]
            table[key] = spoken
        # An empty spoken value means "say it as written" (removes a default).
        table = {k: v for k, v in table.items() if v.strip()}
        self._table = table
        self._lower = {k.lower(): v for k, v in table.items() if k == k.lower()}
        if not table:
            self._rx = None
            return
        alts = sorted((re.escape(k) for k in table), key=len, reverse=True)
        # Whole-token: not glued to letters/digits/underscore on either side.
        self._rx = re.compile(r"(?<![\w])(?:%s)(?![\w])" % "|".join(alts),
                              re.IGNORECASE)

    def items(self) -> dict[str, str]:
        self._maybe_reload()
        return dict(self._table)

    def user_items(self) -> dict[str, str]:
        self._maybe_reload()
        return dict(self._user)

    def add(self, word: str, spoken: str) -> None:
        """Add/override one entry and persist it."""
        word = (word or "").strip()
        spoken = (spoken or "").strip()
        if not word:
            raise ValueError("word must not be empty")
        with self._lock:
            self._user[word] = spoken
            self._rebuild()
        self.save()
        log.info("pronunciation: %r -> %r", word, spoken)

    def remove(self, word: str) -> bool:
        with self._lock:
            hit = [k for k in self._user if k.lower() == word.lower()]
            for k in hit:
                del self._user[k]
            self._rebuild()
        if hit:
            self.save()
        return bool(hit)

    # ----------------------------------------------------------- apply
    def _lookup(self, token: str) -> Optional[str]:
        if token in self._table:
            return self._table[token]
        return self._lower.get(token.lower())

    def apply(self, text: str, *, engine: Optional[str] = None,
              rewrite_times: Optional[bool] = None,
              unshout_words: Optional[bool] = None) -> str:
        """Rewrite tokens/symbols in ``text`` into their spoken forms.

        The rewrites are OPTIONAL because they are compensations for a weak
        engine, not universal improvements. XTTS spelled "6:00 pm" as "six
        zero pm" and read ALL-CAPS as an acronym, so both were added. Fish's
        s2.1-pro normalises text itself, and there the rewrites make things
        WORSE -- "ay em" comes out as "I'm" (heard 2026-08-28); Breeze-TTS-2
        does the same, measured (see ENGINE_RULES). So ``engine`` picks the
        rule set and the engine's own row decides.

        ``rewrite_times`` and ``unshout_words`` remain as overrides for
        callers that know better than the table -- passing rewrite_times
        False switches off all three clock/number rules together, which is
        what that flag always meant.
        """
        if not text:
            return text
        rules = rules_for(engine)
        if rewrite_times is not None:
            rules = dataclasses.replace(rules, marked_times=rewrite_times,
                                        bare_times=rewrite_times,
                                        id_digits=rewrite_times)
        if unshout_words is not None:
            rules = dataclasses.replace(rules, unshout=unshout_words)
        self._maybe_reload()
        # Times first: the vocabulary pass is token-based and would happily
        # leave "6:00" intact for the engine to spell out.
        if rules.marked_times or rules.bare_times:
            text = speak_times(text, marked=rules.marked_times,
                               bare=rules.bare_times)
        if rules.id_digits:
            # A separate rule from the clock ones despite riding the same
            # flag for a year: this one is pure F5 byte arithmetic.
            text = speak_room_numbers(text)
        rx = self._rx
        if rx is not None:
            def _sub(m):
                spoken = self._lookup(m.group(0))
                return spoken if spoken is not None else m.group(0)
            text = rx.sub(_sub, text)
        for sym, spoken in self._symbols.items():
            if sym in text:
                text = text.replace(sym, spoken)
        # Last: known jargon has already been substituted, so whatever is
        # still shouting is Hunter's own text rather than something the
        # table wanted spelled.
        if rules.unshout:
            text = unshout(text)
        # Last, on whatever every other pass produced (unshout title-cases a
        # shouted "TWENTY-FIVE" but leaves its hyphen). Unconditional, unlike
        # the two flags above: a number compound written with a space reads
        # the same to every engine, so there is nothing here for a
        # self-normalising engine to get wrong.
        text = space_number_hyphens(text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()


_default: Optional[Pronunciations] = None
_default_lock = threading.Lock()


def get() -> Pronunciations:
    """The process-wide table (user file at ``USER_FILE``)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Pronunciations()
        return _default


def apply(text: str, *, engine: Optional[str] = None,
          rewrite_times: Optional[bool] = None,
          unshout_words: Optional[bool] = None) -> str:
    return get().apply(text, engine=engine, rewrite_times=rewrite_times,
                       unshout_words=unshout_words)
