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
# Only times carrying an am/pm marker are rewritten. A bare "16:9" or "3:15"
# is far more likely to be a ratio or a score than a clock reading, and a
# wrong rewrite there is worse than the digits.
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


def speak_times(text: str) -> str:
    """Rewrite "6:00 pm" as "six pm" so the engine does not spell the colon."""
    return _TIME_RX.sub(_spoken_time, text)


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

    def apply(self, text: str, *, rewrite_times: bool = True,
              unshout_words: bool = True) -> str:
        """Rewrite tokens/symbols in ``text`` into their spoken forms.

        The two rewrites are OPTIONAL because they are compensations for a
        weak engine, not universal improvements. XTTS spelled "6:00 pm" as
        "six zero pm" and read ALL-CAPS as an acronym, so both were added.
        Fish's s2.1-pro normalises text itself, and there the rewrites make
        things WORSE -- "ay em" comes out as "I'm" (heard 2026-08-28). The
        caller passes what its engine actually needs.
        """
        if not text:
            return text
        self._maybe_reload()
        # Times first: the vocabulary pass is token-based and would happily
        # leave "6:00" intact for the engine to spell out.
        if rewrite_times:
            text = speak_times(text)
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
        if unshout_words:
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


def apply(text: str, *, rewrite_times: bool = True,
          unshout_words: bool = True) -> str:
    return get().apply(text, rewrite_times=rewrite_times,
                       unshout_words=unshout_words)
