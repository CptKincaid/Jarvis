"""The two fallbacks, and the two rate limiters that must never interfere.

THEY ARE DIFFERENT IN KIND, ON PURPOSE
    THE SPOKEN PASSPHRASE is the voice path's way back in for when face and
    voice recognition both fail but he can still talk -- ill, in the dark,
    turned away, or with the camera off under the night curfew. It can be
    overheard. He was told that and accepted it.

    THE OVERRIDE CODE is KEYBOARD/SSH ONLY and is never accepted over the
    microphone, ever, so it cannot be overheard or replayed. It is
    break-glass for when even the passphrase fails. This module only hashes
    and checks; ``jarvis/gate.py`` is where the "never over the microphone"
    rule is enforced, by never calling the check on the voice path at all.

WHAT THIS MODULE DELIBERATELY DOES NOT HAVE
    A logger. No ``get_logger``, no ``logging`` import, no ``print``. The
    cheapest way to be certain a plaintext is never logged is for there to
    be nothing here that could log it -- and a test pins that, because a
    future edit would not know to keep the promise. Nothing here has a
    ``__repr__`` that could hold a secret either.

NO STRENGTH POLICY, AND THAT IS A DECISION RATHER THAN AN OVERSIGHT
    Anyone who can already type at the Spark or SSH in can edit
    ``people.json`` or delete it. The code is therefore HIS RECOVERY PATH,
    not a defence against them, and imposing a character-class rule would
    dress it up as something it is not while making it harder for him to
    remember the one thing it exists for. Only a length floor, and the
    floor is stated out loud: 12 characters for the spoken phrase (it has
    to survive Whisper and be unlikely in ordinary speech) and 6 for the
    typed code.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from hashlib import scrypt
from typing import List, Optional, Tuple

# MEASURED on this box 2026-09-03 (hashlib.scrypt, mean of 3 runs):
#   n=2**12  5.0 ms   n=2**13  9.7 ms   n=2**14  19.7 ms   n=2**15  48.3 ms
# 2**14 is the pick. It is off the turn budget for every ordinary sentence
# because jarvis/gate.py runs it ONLY when recognition has already failed
# AND the words were phrase-shaped -- and the rate limiter doubles as the
# ceiling on how often that can happen.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
DKLEN = 32
SALT_BYTES = 16
MAXMEM = 64 * 1024 * 1024

MIN_PHRASE_LEN = 12
MIN_CODE_LEN = 6

# Cool-offs, not locks, and never shared: burning the passphrase attempts
# must not lock out the break-glass, or the fallback of last resort fails
# exactly when it is needed. Two instances, two windows, nothing in common.
PHRASE_LIMIT = 5
PHRASE_WINDOW = 300.0
CODE_LIMIT = 10
CODE_WINDOW = 300.0

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalise_spoken(text) -> str:
    """Lower-case, strip punctuation, collapse whitespace.

    Whisper punctuates as it pleases and capitalises the first word, so the
    stored hash is of the NORMALISED form and the offering is normalised the
    same way. This forgives the transcriber, not the phrase: the words
    themselves still have to be right.
    """
    try:
        return _PUNCT.sub(" ", str(text or "").lower()).strip()
    except Exception:  # noqa: BLE001 - text that cannot be read is not a phrase
        return ""


def phrase_shaped(text) -> bool:
    """The cheap pre-filter that keeps a 20 ms key derivation off every
    refused turn. "Yes" and "no" and silence can never be the phrase, and
    asking scrypt about them would spend the budget to learn that."""
    return len(normalise_spoken(text)) >= MIN_PHRASE_LEN


# The wake word Whisper leaves at the FRONT of the sentence. The recorder
# keeps the audio from before the hotword fired, so "Jarvis, <phrase>" comes
# back with the vocative in the transcript -- and until 2026-09-05 that was
# not the phrase, so the commonest way of saying it was dispatched as an
# ordinary command instead of being consumed. Same list as
# commander.JARVIS_PREFIXES, in normalised form.
VOCATIVES = ("hey jarvis", "ok jarvis", "okay jarvis", "jarvis")


def spoken_candidates(text) -> Tuple[str, ...]:
    """What may be compared against a stored phrase: the normalised
    utterance, and -- only when it opens with the wake word -- the same
    sentence with that word taken off.

    AT MOST TWO, deliberately. Each candidate costs one scrypt PER OWNER
    (~18 ms measured), and this runs on every phrase-shaped turn, so a
    sweep over every span of the sentence (which would catch a phrase
    buried mid-sentence) would put ~180 ms on ordinary turns. Anything
    shorter than MIN_PHRASE_LEN is dropped for free: no stored phrase can
    be shorter, and the length rule is a public constant, not a fact about
    his phrase.
    """
    first = normalise_spoken(text)
    out = [first] if len(first) >= MIN_PHRASE_LEN else []
    for word in VOCATIVES:
        if first.startswith(word + " "):
            rest = first[len(word) + 1:].strip()
            if len(rest) >= MIN_PHRASE_LEN and rest not in out:
                out.append(rest)
            break
    return tuple(out)


def phrase_ok(plain) -> Tuple[bool, str]:
    """Is this acceptable as a NEW spoken phrase? Length only."""
    if len(normalise_spoken(plain)) < MIN_PHRASE_LEN:
        return False, ("a spoken passphrase needs at least %d letters and "
                       "digits once punctuation is dropped" % MIN_PHRASE_LEN)
    return True, ""


# A ROTATED code (Knightfall, 2026-09-04) is generated here rather than
# chosen: eight characters from an alphabet he can read back off a phone
# screen without guessing -- no 0/o, no 1/l -- and it is his only until he
# next types it, when the next one is mailed. secrets, never random.
CODE_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"
NEW_CODE_LEN = 8


def new_code(length: int = NEW_CODE_LEN) -> str:
    """A fresh typed code. The caller mails it, hashes it, and deletes it;
    nothing here keeps it."""
    return "".join(secrets.choice(CODE_ALPHABET)
                   for _ in range(max(MIN_CODE_LEN, int(length))))


def code_ok(plain) -> Tuple[bool, str]:
    """Is this acceptable as a NEW override code? Length only -- letters
    and/or digits, his own string, no character-class rule."""
    text = str(plain or "")
    if len(text.strip()) < MIN_CODE_LEN:
        return False, ("an override code needs at least %d characters; "
                       "letters, digits or both, whatever you like"
                       % MIN_CODE_LEN)
    return True, ""


def _plain(secret) -> str:
    """What is actually hashed: the secret with the surrounding whitespace
    dropped. ONE definition, used by hash_secret and check_secret alike --
    the trap this closes was the two of them disagreeing."""
    try:
        return str(secret or "").strip()
    except Exception:  # noqa: BLE001 - a value that cannot be read is not one
        return ""


def hash_secret(plain, *, salt: Optional[bytes] = None) -> str:
    """``scrypt$n$r$p$<salt_b64>$<hash_b64>`` -- a new random salt each
    time, so two owners sharing a phrase do not share a stored value.

    THE SURROUNDING WHITESPACE IS DROPPED, at BOTH ends of the pair (see
    :func:`check_secret`), and that is a fix rather than a nicety. Until
    2026-09-05 this hashed the raw ``getpass`` string while
    ``gate.check_override_code`` stripped before checking, so a code set
    with one stray space could be typed neither WITH the space nor
    without it -- and ``scripts/jarvis_people.py`` answered "set." either
    way. One keystroke killed the break-glass silently. A secret whose
    meaning depends on an invisible character is not a secret he can type.
    """
    if salt is None:
        salt = os.urandom(SALT_BYTES)
    raw = _plain(plain).encode("utf-8")
    digest = scrypt(raw, salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                    dklen=DKLEN, maxmem=MAXMEM)
    return "scrypt$%d$%d$%d$%s$%s" % (
        SCRYPT_N, SCRYPT_R, SCRYPT_P,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"))


def check_secret(plain, stored) -> bool:
    """Constant-time compare against a stored value. ANY malformed or empty
    stored value is a plain False -- never an exception, because a
    corrupted hash must not be able to raise its way past the caller, and
    never a different code path, because a different path is a signal."""
    text = _plain(plain)
    if not text:
        return False
    parts = str(stored or "").split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = base64.b64decode(parts[4], validate=True)
        want = base64.b64decode(parts[5], validate=True)
    except (ValueError, binascii.Error):
        return False
    if not salt or not want or n <= 1 or r < 1 or p < 1:
        return False
    try:
        got = scrypt(text.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                     dklen=len(want), maxmem=MAXMEM)
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(got, want)


@dataclass
class Attempts:
    """A sliding window over ``time.monotonic()``. IN MEMORY ONLY.

    A rate limiter that persists is itself a lockout: a restart is the one
    thing he can always do, and it has to clear this. And monotonic rather
    than wall clock, because an NTP step or a daylight-saving change must
    not be able to extend a cool-off by an hour.

    This is a COOL-OFF, NOT A LOCK. There is no permanent state and no
    escalation -- the window simply has to pass.
    """

    limit: int = PHRASE_LIMIT
    window_s: float = PHRASE_WINDOW
    _hits: List[float] = field(default_factory=list, repr=False)

    def _now(self, now: Optional[float]) -> float:
        return time.monotonic() if now is None else float(now)

    def _prune(self, now: float) -> None:
        cutoff = now - float(self.window_s)
        self._hits = [t for t in self._hits if t > cutoff]

    def allow(self, now: Optional[float] = None) -> Tuple[bool, float]:
        """``(ok, seconds to wait)``."""
        t = self._now(now)
        self._prune(t)
        if len(self._hits) < int(self.limit):
            return True, 0.0
        oldest = min(self._hits)
        return False, max(0.0, float(self.window_s) - (t - oldest))

    def record(self, now: Optional[float] = None) -> None:
        t = self._now(now)
        self._prune(t)
        self._hits.append(t)

    def reset(self) -> None:
        self._hits = []
