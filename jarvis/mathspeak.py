"""Spoken arithmetic and unit conversion, answered without the model.

"What's eighteen percent of seventy-four", "how many ounces in 300 grams",
"what's 43 times 17" used to reach the local model: `router._QUESTION_RX`
routes `calculate|compute|convert \\d` to the brain, which takes seconds and
occasionally gets the sum wrong. A few hundred lines of stdlib gets them
right in under a millisecond, so this sits in the commander's Tier-1 ladder
ahead of the router (jarvis/commander.py -- math_kind / _h_math).

The one rule that shapes everything here: **claim only what it can actually
answer.** ``solve()`` does the whole parse and returns the finished sentence
or None; there is no "looks like maths" gate that hands the handler a
half-parse. Anything unrecognised falls through to the model exactly as
before, so a new phrasing is a missed optimisation, never a wrong answer.

What it answers:

* ``18 percent of 74`` / ``15 percent off 80`` (a discount, so 68)
* ``43 times 17``, ``plus``, ``minus``, ``divided by``, ``over``
* ``square root of 144``
* ``300 grams in ounces``, ``how many ounces in 300 grams``,
  ``convert 5 miles to kilometres``, ``100 fahrenheit in celsius``
* mass (g, kg, mg, oz, lb, stone, tonne), length (mm, cm, m, km, in, ft,
  yd, mile), temperature (C, F, K) and data (byte .. TB, plus the binary
  KiB..TiB)

What it deliberately refuses, out loud, rather than guessing:

* mixing dimensions -- "how many ounces in five miles"
* dividing by zero
* currency, which needs a rate it has no way to know offline

Deliberate omissions: volume (a US pint is 473 ml and a UK pint is 568 --
"a pint in millilitres" has no honest single answer, and fluid ounces
differ too), and chained expressions ("2 plus 3 times 4"), whose spoken
precedence is genuinely ambiguous. Both fall through to the model.

Byte units follow the standards: kB/MB/GB/TB are powers of 1000 and
KiB/MiB/GiB/TiB are powers of 1024. That is the only reading that is
defensible without asking which one he meant.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from jarvis.logs import get_logger

log = get_logger("mathspeak")

SIR = "sir"

# ----------------------------------------------------------------- units
MASS, LENGTH, TEMP, DATA = "mass", "length", "temperature", "data"
DIMENSION_NAMES = {MASS: "a mass", LENGTH: "a length",
                   TEMP: "a temperature", DATA: "an amount of data"}


@dataclass(frozen=True)
class Unit:
    key: str
    one: str            # spoken singular
    many: str           # spoken plural
    dim: str
    factor: float       # to the dimension's base (gram / metre / byte)


def _u(key, one, many, dim, factor) -> Unit:
    return Unit(key, one, many, dim, factor)


UNITS: dict[str, Unit] = {u.key: u for u in (
    # mass -> grams
    _u("mg", "milligram", "milligrams", MASS, 0.001),
    _u("g", "gram", "grams", MASS, 1.0),
    _u("kg", "kilogram", "kilograms", MASS, 1000.0),
    _u("t", "tonne", "tonnes", MASS, 1_000_000.0),
    _u("oz", "ounce", "ounces", MASS, 28.349523125),
    _u("lb", "pound", "pounds", MASS, 453.59237),
    _u("st", "stone", "stone", MASS, 6350.29318),
    # length -> metres
    _u("mm", "millimetre", "millimetres", LENGTH, 0.001),
    _u("cm", "centimetre", "centimetres", LENGTH, 0.01),
    _u("m", "metre", "metres", LENGTH, 1.0),
    _u("km", "kilometre", "kilometres", LENGTH, 1000.0),
    _u("in", "inch", "inches", LENGTH, 0.0254),
    _u("ft", "foot", "feet", LENGTH, 0.3048),
    _u("yd", "yard", "yards", LENGTH, 0.9144),
    _u("mi", "mile", "miles", LENGTH, 1609.344),
    # temperature: factor is unused, _to_base/_from_base do the offsets
    _u("c", "degree Celsius", "degrees Celsius", TEMP, 1.0),
    _u("f", "degree Fahrenheit", "degrees Fahrenheit", TEMP, 1.0),
    _u("k", "kelvin", "kelvin", TEMP, 1.0),
    # data -> bytes
    _u("byte", "byte", "bytes", DATA, 1.0),
    _u("kb", "kilobyte", "kilobytes", DATA, 1e3),
    _u("mb", "megabyte", "megabytes", DATA, 1e6),
    _u("gb", "gigabyte", "gigabytes", DATA, 1e9),
    _u("tb", "terabyte", "terabytes", DATA, 1e12),
    _u("kib", "kibibyte", "kibibytes", DATA, 1024.0),
    _u("mib", "mebibyte", "mebibytes", DATA, 1024.0 ** 2),
    _u("gib", "gibibyte", "gibibytes", DATA, 1024.0 ** 3),
    _u("tib", "tebibyte", "tebibytes", DATA, 1024.0 ** 4),
)}

# Spoken and written spellings -> unit key. Longest match wins (the pattern
# is built longest-first), so "kilometres" never matches as "k".
# Deliberately ABSENT: a bare "in" (it is the preposition in "how many
# ounces in 300 grams" far more often than it is inches) and a bare "b"
# (bytes vs. a stray letter). Inches must be spelled out.
_ALIASES: dict[str, str] = {}


def _alias(key: str, *words: str) -> None:
    for w in words:
        _ALIASES[w] = key


_alias("mg", "mg", "milligram", "milligrams", "milligramme", "milligrammes")
_alias("g", "g", "gram", "grams", "gramme", "grammes")
_alias("kg", "kg", "kgs", "kilo", "kilos", "kilogram", "kilograms",
       "kilogramme", "kilogrammes")
_alias("t", "tonne", "tonnes", "metric ton", "metric tons")
_alias("oz", "oz", "ounce", "ounces")
_alias("lb", "lb", "lbs", "pound", "pounds")
_alias("st", "stone", "stones")
_alias("mm", "mm", "millimetre", "millimetres", "millimeter", "millimeters")
_alias("cm", "cm", "centimetre", "centimetres", "centimeter", "centimeters")
_alias("m", "metre", "metres", "meter", "meters")
_alias("km", "km", "kms", "kilometre", "kilometres", "kilometer", "kilometers",
       "klick", "klicks")
_alias("in", "inch", "inches")
_alias("ft", "ft", "foot", "feet")
_alias("yd", "yd", "yard", "yards")
_alias("mi", "mile", "miles")
_alias("c", "c", "celsius", "centigrade", "degrees celsius", "degree celsius",
       "degrees c", "degrees centigrade")
_alias("f", "f", "fahrenheit", "degrees fahrenheit", "degree fahrenheit",
       "degrees f")
_alias("k", "kelvin", "kelvins", "degrees kelvin")
_alias("byte", "byte", "bytes")
_alias("kb", "kb", "kilobyte", "kilobytes")
_alias("mb", "mb", "megabyte", "megabytes", "meg", "megs")
_alias("gb", "gb", "gigabyte", "gigabytes", "gig", "gigs")
_alias("tb", "tb", "terabyte", "terabytes")
_alias("kib", "kib", "kibibyte", "kibibytes")
_alias("mib", "mib", "mebibyte", "mebibytes")
_alias("gib", "gib", "gibibyte", "gibibytes")
_alias("tib", "tib", "tebibyte", "tebibytes")

_UNIT_PAT = "|".join(re.escape(w) for w in
                     sorted(_ALIASES, key=lambda w: (-len(w), w)))
# Currency is a rate, not a conversion; saying so is better than a number.
# "pound" is in here AND in the mass table on purpose: the mass conversion
# is tried first, so "five pounds in kilos" is a weight and "five pounds in
# dollars" is the refusal.
_CUR_PAT = (r"(?:[$£€¥]|dollars?|usd|bucks?|euros?|eur|gbp|quid|"
            r"pounds?(?:\s+sterling)?|yen|jpy|rupees?|pesos?|"
            r"rands?|francs?|krona|kroner)")
_CUR_CONVERT_RX = re.compile(
    rf"^(?P<n>.*?)\s*{_CUR_PAT}\s+(?:in|into|to|as)\s+{_CUR_PAT}$", re.I)
_CUR_HOWMANY_RX = re.compile(
    rf"^how\s+many\s+{_CUR_PAT}\s+(?:are\s+|is\s+)?(?:in|to)\s+"
    rf"(?P<n>.*?)\s*{_CUR_PAT}$", re.I)

# ---------------------------------------------------------- number words
_ONES = {
    "zero": 0, "oh": 0, "nought": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000,
           "billion": 1_000_000_000}
_NUMBER_TOKENS = set(_ONES) | set(_TENS) | set(_SCALES) | {"and", "point",
                                                           "a", "an", "half"}
_NUM_WORD_PAT = "|".join(sorted(_NUMBER_TOKENS - {"and", "a", "an"},
                                key=lambda w: (-len(w), w)))
# A number is digits ("74", "13.5", "1,250") or words ("seventy four", "a
# hundred and twenty"). "and" and "a" only count INSIDE a word number, so
# "a mile in km" reads as one mile but "and" alone is never a number.
_NUMBER_RX = re.compile(
    rf"(?:\d[\d,]*(?:\.\d+)?|\b(?:a|an)\s+(?:{_NUM_WORD_PAT})\b"
    rf"|\b(?:{_NUM_WORD_PAT})(?:[\s-]+(?:and\s+)?(?:{_NUM_WORD_PAT}))*\b)")


def parse_number(text: str) -> Optional[float]:
    """'74', '13.5', '1,250', 'seventy-four', 'a hundred and twenty',
    'three point five' -> a float. None when it is not a number."""
    s = re.sub(r"[\s-]+", " ", str(text or "").strip().lower())
    if not s:
        return None
    sign = 1.0
    m = re.match(r"^(?:minus|negative)\s+(.+)$", s)
    if m:
        sign, s = -1.0, m.group(1)
    plain = s.replace(",", "")
    try:
        return sign * float(plain)
    except ValueError:
        pass
    toks = [t for t in plain.split() if t]
    if toks and toks[0] in ("a", "an"):
        toks = toks[1:] or ["one"]
        if toks == ["one"]:
            return sign
    if not toks or any(t not in _NUMBER_TOKENS for t in toks):
        return None
    # "three point five": the tail is read digit by digit.
    if "point" in toks:
        i = toks.index("point")
        head, tail = toks[:i], toks[i + 1:]
        whole = _words_to_int(head) if head else 0
        if whole is None or not tail or any(t not in _ONES for t in tail):
            return None
        return sign * float(f"{whole}.{''.join(str(_ONES[t]) for t in tail)}")
    if toks[-1] == "half":                      # "one and a half"
        base = _words_to_int([t for t in toks[:-1] if t not in ("and", "a", "an")])
        return None if base is None else sign * (base + 0.5)
    n = _words_to_int(toks)
    return None if n is None else sign * float(n)


def _words_to_int(tokens) -> Optional[int]:
    total = current = 0
    seen = False
    for t in tokens:
        if t in ("and", "a", "an"):
            continue
        if t in _ONES:
            current += _ONES[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t == "hundred":
            current = (current or 1) * 100
        elif t in _SCALES:
            total += (current or 1) * _SCALES[t]
            current = 0
        else:
            return None
        seen = True
    return (total + current) if seen else None


# --------------------------------------------------------------- wording
def fmt_number(value: float) -> str:
    """A number a voice can read: no thousands commas (the TTS spells them
    out), integers bare, everything else to a sensible few places."""
    if value != value or value in (float("inf"), float("-inf")):
        return "undefined"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    av = abs(value)
    # Two decimals for anything you would say aloud; below 0.01, two
    # SIGNIFICANT figures instead, so 0.001234 is "0.0012" and not "0".
    places = 2 if av >= 0.01 else min(8, 1 - int(math.floor(math.log10(av))))
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _unit_words(unit: Unit, value: float) -> str:
    return unit.one if abs(value - 1.0) < 1e-9 else unit.many


def _amount(value: float, unit: Unit) -> str:
    return f"{fmt_number(value)} {_unit_words(unit, value)}"


# ------------------------------------------------------------ conversion
def _to_base(value: float, unit: Unit) -> float:
    if unit.dim != TEMP:
        return value * unit.factor
    if unit.key == "c":
        return value
    if unit.key == "f":
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15                      # kelvin


def _from_base(celsius_or_base: float, unit: Unit) -> float:
    if unit.dim != TEMP:
        return celsius_or_base / unit.factor
    if unit.key == "c":
        return celsius_or_base
    if unit.key == "f":
        return celsius_or_base * 9.0 / 5.0 + 32.0
    return celsius_or_base + 273.15


def convert(value: float, src: Unit, dst: Unit) -> Optional[float]:
    """None when the dimensions do not match -- the caller says so aloud."""
    if src.dim != dst.dim:
        return None
    return _from_base(_to_base(value, src), dst)


# ------------------------------------------------------------- solutions
@dataclass(frozen=True)
class Answer:
    """What to say. ``ok`` False is an honest refusal, not a failure to
    parse -- the phrasing WAS understood, the question just has no number
    for an answer."""
    text: str
    ok: bool = True


# Politeness and framing stripped before parsing. Order matters: the longer
# openers must go first or "what is" leaves "is" behind.
_LEAD_RX = re.compile(
    r"^(?:jarvis[,\s]+)?(?:"
    r"(?:hey|ok|okay)\s+jarvis[,\s]+|"
    r"(?:could|can|would)\s+you\s+(?:please\s+)?(?:tell\s+me\s+)?|"
    r"please\s+|tell\s+me\s+|"
    r"i\s+need\s+to\s+know\s+|"
    r"work\s+out\s+|figure\s+out\s+|"
    r"what(?:'|’)?s\s+|what\s+is\s+|whats\s+|"
    r"how\s+much\s+is\s+|how\s+many\s+is\s+|"
    r"give\s+me\s+|"
    r"calculate\s+|compute\s+|convert\s+"
    r")+", re.I)
_TRAIL_RX = re.compile(r"\s*(?:,?\s*(?:please|jarvis|sir))*\s*[?.!]*$", re.I)

_OPS = {
    "plus": "+", "add": "+", "added to": "+", "and then": "+",
    "minus": "-", "less": "-", "take away": "-", "subtract": "-",
    "times": "*", "multiplied by": "*", "multiplied": "*", "x": "*",
    "divided by": "/", "divide by": "/", "over": "/",
}
_OP_PAT = "|".join(re.escape(k) for k in sorted(_OPS, key=lambda k: (-len(k), k)))
_ARITH_RX = re.compile(rf"^(?P<a>.+?)\s+(?P<op>{_OP_PAT})\s+(?P<b>.+)$", re.I)
_OP_WORDS = {"plus": "plus", "+": "plus", "-": "minus", "*": "times",
             "/": "divided by"}

_PERCENT_RX = re.compile(
    r"^(?P<p>.+?)\s*(?:percent|per\s*cent|%)\s+(?P<mode>of|off)\s+(?P<n>.+)$", re.I)
_SQRT_RX = re.compile(r"^(?:the\s+)?square\s+root\s+of\s+(?P<n>.+)$", re.I)

# "300 grams in ounces" / "5 miles to km"
_CONVERT_RX = re.compile(
    rf"^(?P<n>.+?)\s*(?P<src>{_UNIT_PAT})\s+(?:in|into|to|as|in\s+terms\s+of)"
    rf"\s+(?P<dst>{_UNIT_PAT})$", re.I)
# "how many ounces in 300 grams" / "how many ounces are in 300 grams"
_HOWMANY_RX = re.compile(
    rf"^how\s+many\s+(?P<dst>{_UNIT_PAT})\s+(?:are\s+|is\s+)?"
    rf"(?:in|to|make(?:s)?(?:\s+up)?)\s+(?P<n>.+?)\s*(?P<src>{_UNIT_PAT})$", re.I)

# One cheap search decides whether the full parse is worth running at all.
_HINT_RX = re.compile(
    rf"\b(?:percent|per\s*cent|square\s+root|{_OP_PAT}|how\s+many|convert|"
    rf"calculate|compute|{_UNIT_PAT}|{_CUR_PAT})\b|%", re.I)

DIM_MISMATCH_LINE = "Those don't convert, {sir}: {a} is {da} and {b} is {db}."
DIVIDE_BY_ZERO_LINE = "You can't divide by zero, {sir}."
CURRENCY_LINE = "I can't do currency, {sir}; I've no exchange rate down here."
NEGATIVE_ROOT_LINE = "There's no real square root of a negative, {sir}."


def _clean(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("’", "'")
    s = s.replace("×", " times ").replace("÷", " divided by ")
    s = s.replace("°c", " celsius ").replace("°f", " fahrenheit ").replace("°", " degrees ")
    s = re.sub(r"(\d)\s*%", r"\1 percent", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _TRAIL_RX.sub("", s)
    s = _LEAD_RX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _unit_of(word: str) -> Optional[Unit]:
    key = _ALIASES.get(re.sub(r"\s+", " ", (word or "").strip().lower()))
    return UNITS.get(key) if key else None


def _number_only(text: str) -> Optional[float]:
    """The text is a number and nothing else (a stray word means no)."""
    return parse_number(text)


def solve(text: str) -> Optional[Answer]:
    """The finished spoken line, or None when this is not arithmetic he can
    do -- in which case the caller carries on down the ladder."""
    raw = str(text or "")
    if not _HINT_RX.search(raw):
        return None
    s = _clean(raw)
    if not s:
        return None
    try:
        return (_solve_convert(s) or _solve_currency(s) or _solve_percent(s)
                or _solve_sqrt(s) or _solve_arith(s))
    except Exception:                    # noqa: BLE001 - never break the ladder
        log.exception("mathspeak: %r", raw)
        return None


def _solve_currency(s: str) -> Optional[Answer]:
    """Only a two-sided currency CONVERSION is refused. "How many dollars
    did I spend" is somebody else's question, not a wrong answer here."""
    if _CUR_CONVERT_RX.match(s) or _CUR_HOWMANY_RX.match(s):
        return Answer(CURRENCY_LINE.format(sir=SIR), ok=False)
    return None


def _convert_answer(value: float, src: Unit, dst: Unit) -> Answer:
    out = convert(value, src, dst)
    if out is None:
        return Answer(DIM_MISMATCH_LINE.format(
            sir=SIR, a=src.many, da=DIMENSION_NAMES[src.dim],
            b=dst.many, db=DIMENSION_NAMES[dst.dim]), ok=False)
    return Answer(f"{_amount(value, src)} is {_amount(out, dst)}, {SIR}.")


def _solve_convert(s: str) -> Optional[Answer]:
    for rx in (_HOWMANY_RX, _CONVERT_RX):
        m = rx.match(s)
        if not m:
            continue
        src, dst = _unit_of(m.group("src")), _unit_of(m.group("dst"))
        value = _number_only(m.group("n"))
        if src is None or dst is None:
            continue
        if value is None:
            continue          # "how many ounces in that" -- not ours
        return _convert_answer(value, src, dst)
    return None


def _solve_percent(s: str) -> Optional[Answer]:
    m = _PERCENT_RX.match(s)
    if not m:
        return None
    pct, base = _number_only(m.group("p")), _number_only(m.group("n"))
    if pct is None or base is None:
        return None
    part = base * pct / 100.0
    if m.group("mode").lower() == "off":
        return Answer(f"{fmt_number(pct)} percent off {fmt_number(base)} "
                      f"is {fmt_number(base - part)}, {SIR}.")
    return Answer(f"{fmt_number(pct)} percent of {fmt_number(base)} "
                  f"is {fmt_number(part)}, {SIR}.")


def _solve_sqrt(s: str) -> Optional[Answer]:
    m = _SQRT_RX.match(s)
    if not m:
        return None
    n = _number_only(m.group("n"))
    if n is None:
        return None
    if n < 0:
        return Answer(NEGATIVE_ROOT_LINE.format(sir=SIR), ok=False)
    return Answer(f"The square root of {fmt_number(n)} is "
                  f"{fmt_number(math.sqrt(n))}, {SIR}.")


def _solve_arith(s: str) -> Optional[Answer]:
    m = _ARITH_RX.match(s)
    if not m:
        return None
    a, b = _number_only(m.group("a")), _number_only(m.group("b"))
    if a is None or b is None:
        return None
    op = _OPS[m.group("op").lower()]
    # A second operator means a chain, whose spoken precedence ("two plus
    # three times four") is genuinely ambiguous. Leave it to the model.
    if _ARITH_RX.match(m.group("b")):
        return None
    if op == "/" and b == 0:
        return Answer(DIVIDE_BY_ZERO_LINE.format(sir=SIR), ok=False)
    result = {"+": a + b, "-": a - b, "*": a * b,
              "/": (a / b if b else 0.0)}[op]
    return Answer(f"{fmt_number(a)} {_OP_WORDS[op]} {fmt_number(b)} "
                  f"is {fmt_number(result)}, {SIR}.")


# Fixed lines the app prewarms into the speech cache.
PERSONA_LINES = [DIVIDE_BY_ZERO_LINE.format(sir=SIR),
                 CURRENCY_LINE.format(sir=SIR),
                 NEGATIVE_ROOT_LINE.format(sir=SIR)]
