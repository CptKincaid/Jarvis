"""Saying your name: a NARROWING input, and never a credential.

WHAT HE ASKED FOR, and what it can honestly be
    "lets have there names be there 'sign in' so for them to 'login' they
    say there first and last name."

    Saying a name is not proof of being that person. Anybody who has heard
    Mara introduce herself can say "Mara Whatever", and a name said out loud
    in a kitchen is overheard by everybody in it. So a spoken name is
    treated here as what it actually is: a HINT ABOUT WHO TO LOOK FOR. It
    narrows the question; a leg -- voice or face -- answers it.

    That is why this module returns a LABEL TO CONFIRM and nothing else. It
    has no verdict, no boolean, no score and no way to admit anybody. The
    admitting is ``jarvis/recognise.py``'s, on legs that measured something,
    and ``Verdict.claimed`` carries this label only so the refusal can be a
    kinder sentence than "I don't recognise you".

PURE, TOTAL, DETERMINISTIC. No clock, no file, no model, no config, no
logger, no registry object -- it takes rows and text and returns a string.
Same reason ``recognise.py`` is: the whole decision has to be testable with
no microphone and no lens.

FIRST-NAME-ONLY IS ANSWERED ONLY WHEN IT IS UNAMBIGUOUS
    "It's Mara" narrows to ``mara`` while there is one Mara on file. The day
    a second one is enrolled the same sentence narrows to NOBODY rather than
    to whichever row happens to sort first, because guessing between two
    real people is exactly the failure this feature must not have. A test
    pins it.
"""
from __future__ import annotations

import re
from typing import Iterable, List

# Words that arrive in front of a name and are not part of it. Whisper
# punctuates and capitalises as it pleases, so this is deliberately
# generous: it only ever removes text BEFORE the name, and removing too
# much can at worst fail to narrow, which is the safe direction.
_LEAD_RX = re.compile(
    r"^(?:\s|,|\.|!|\?|-|jarvis\b|hey\b|hi\b|hello\b|it'?s\b|this\s+is\b|"
    r"i'?m\b|my\s+name\s+is\b|name'?s\b|the\s+name'?s\b|sign(?:ing)?\s+in\b|"
    r"log(?:ging)?\s+in\b|log\s*in\b|as\b|please\b)+", re.I)

_NOT_WORDS_RX = re.compile(r"[^a-z0-9']+")


def normalise(text) -> str:
    """The words of ``text``, lower-cased, with the lead-in stripped.

    Never raises: a transcript that cannot be read narrows to nobody, which
    is the same thing an unrelated sentence does.
    """
    try:
        raw = str(text or "")
    except Exception:  # noqa: BLE001 - a transcript that cannot be read
        return ""
    raw = _LEAD_RX.sub(" ", raw.lower())
    return _NOT_WORDS_RX.sub(" ", raw).strip()


def _tokens(text) -> List[str]:
    return [t for t in normalise(text).split() if t]


def candidate_from_name(text, rows: Iterable) -> str:
    """The ONE label ``text`` could be claiming, or "".

    ``rows`` is anything with ``.label``, ``.first`` and ``.last`` -- a list
    of ``identity.Person``, or fakes in a test.

    THREE OUTCOMES AND NO FOURTH:

    * a first AND last name that matches exactly one row -> that label;
    * a first name (or a bare label) ENDING the sentence that matches
      exactly one row -> that label;
    * anything else, INCLUDING AN AMBIGUOUS MATCH -> "".

    An ambiguous match returning "" is not a shortcut. Two people called
    Mara is precisely where a guess lands on the wrong one, and the cost of
    saying nothing is one extra sentence while the cost of guessing is
    telling a real person she is somebody else.
    """
    words = _tokens(text)
    if not words:
        return ""
    try:
        people = list(rows or [])
    except Exception:  # noqa: BLE001 - rows that cannot be read are no rows
        return ""

    joined = " ".join(words)
    full: List[str] = []
    first: List[str] = []
    for p in people:
        label = str(getattr(p, "label", "") or "").lower()
        f = str(getattr(p, "first", "") or "").lower().strip()
        last = str(getattr(p, "last", "") or "").lower().strip()
        if not label:
            continue
        if f and last and ("%s %s" % (f, last)) in joined:
            full.append(label)
            continue
        # A BARE FIRST NAME COUNTS ONLY WHEN IT ENDS THE SENTENCE.
        # "this is Mara" is Mara offering her name; "Mara Osei" is
        # somebody whose surname nobody on file carries, and narrowing
        # that to Mara Quinn would be answering a question that was not
        # asked. The label itself is accepted the same way, so "jarvis,
        # mara" works from the keyboard.
        if words[-1] in ([f] if f else []) or words[-1] == label:
            first.append(label)
    # A full name beats a first name, and only an unambiguous one wins.
    for pool in (full, first):
        unique = sorted(set(pool))
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            # Ambiguous at the more specific level: do NOT fall through to
            # the looser one, which would be a guess dressed as a match.
            return ""
    return ""


def sounds_like_a_sign_in(text) -> bool:
    """True when the words look like somebody offering a name.

    Used ONLY to decide whether to spend a sentence on the sign-in wording
    rather than the generic refusal. It admits nobody, it names nobody, and
    a false positive costs one friendlier refusal.
    """
    try:
        raw = str(text or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return bool(re.search(r"\b(?:sign(?:ing)?\s+in|log(?:ging)?\s+in|login|"
                          r"my name is|this is|it'?s me|i'?m)\b", raw))
