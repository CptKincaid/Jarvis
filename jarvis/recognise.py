"""The verdict: who is speaking, and which leg said so.

PURE, TOTAL, DETERMINISTIC. No clock, no file, no config, no model, no
threshold, no logger. That is not tidiness -- it is the only way the whole
decision can be tested with no microphone and no lens, which is the only way
it can be tested at all on this box.

THREE-VALUED LEGS, AND THIS IS THE WHOLE SAFETY ARGUMENT
    Each leg either NAMES SOMEONE, names SOMEBODY ELSE, or HAS NO OPINION.
    No camera, no gallery, the night curfew, his back turned, a clip too
    short to score, a model that failed to load, an enrolment stored at one
    embedding width and read at another -- every one of those is NO OPINION,
    and none of them is evidence against him. It is the same three-valued
    contract ``jarvis/roomsensor.py`` writes for the mmWave leg and
    ``eye.Attention`` writes for the camera.

PRECEDENCE, IN THREE RULES
    1. A positive on one leg is NEVER cancelled by a negative on another.
       He chose "either voice OR face is enough", so his voice being hoarse
       cannot un-recognise his face and a stranger in frame cannot
       un-recognise his voice.
    2. The highest role among the positives wins.
    3. The passphrase only ever resolves to an OWNER, and never lowers a
       verdict. It is a way back in, not a demotion.

WHY THE PASSPHRASE LEG IS A NAME AND NOT A BOOLEAN
    The brief specified ``passphrase-ok``. A boolean cannot say WHOM it
    admitted, and the model is required not to assume a single owner
    forever. So the leg is a label, or "". The gate does the hash check and
    resolves the boolean-to-label step; this function only ranks.

A CLAIMED NAME IS NOT A LEG, AND THAT IS THE WHOLE OF IT
    ``Legs.name_says`` carries whom the WORDS claimed to be
    (``jarvis/signin.py``). It is not in ``_ORDER``, it is never ranked,
    it can never win, and ``how`` can never be "name" -- because a leg
    that can win the rank is a leg that can admit, and saying a name is
    not proof of being that person. It rides through to
    ``Verdict.claimed`` so the gate can offer a way forward instead of a
    dead end, and a verdict carrying nothing but a claim is identical --
    ``who``, ``role``, ``how``, ``why`` -- to the one silence produces.

WHAT IS DELIBERATELY NOT HERE
    ``exempt`` (a fact about a socket), ``off`` and ``blind`` (facts about
    configuration) and the dead-man's stand-down (a fact about a clock).
    Those are gate-layer facts, not recognition facts, and keeping them out
    is what keeps this function pure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from jarvis.identity import RANK, ROLE_OWNER, ROLE_UNKNOWN

HOW_VOICE = "voice"
HOW_FACE = "face"
HOW_PHRASE = "passphrase"
HOW_NOBODY = "nobody"


@dataclass(frozen=True)
class Legs:
    """What each leg said. ``*_running`` distinguishes "it looked and had
    nothing to say" from "it was never switched on" -- which the verdict
    treats identically, on purpose, but the STARTUP LINE does not: he is
    owed a different sentence for a dark camera and an empty gallery."""

    voice_says: str = ""
    voice_running: bool = False
    face_says: str = ""
    face_running: bool = False
    phrase_says: str = ""
    # WHOM THE WORDS CLAIMED TO BE, from jarvis/signin.py. IT IS NOT A LEG.
    # It is never ranked, it can never win, and it can never appear in
    # ``how`` -- see the module docstring and _ORDER, which does not
    # contain it. Saying a name is not proof of being that person; this
    # only ever narrows who a REAL leg is asked to confirm.
    name_says: str = ""


@dataclass(frozen=True)
class Verdict:
    who: str = ""
    role: str = ROLE_UNKNOWN
    how: str = HOW_NOBODY
    why: str = ""
    # The name the WORDS claimed, confirmed or not. Read only to choose a
    # kinder refusal; it is never an identity and ``who`` is never set from
    # it. A verdict carrying ``claimed`` and nothing else has admitted
    # NOBODY, and its ``who``/``role``/``how``/``why`` are byte-identical to
    # the verdict an unrelated sentence produces.
    claimed: str = ""


# Voice first, then face, then the passphrase. Voice leads because it is the
# leg that is always available: the camera is dark every night under the
# curfew and the gallery is about to be invalidated by the ArcFace swap, so
# naming the camera as the decider when both agree would make the log read
# as if the feature depended on it.
_ORDER = (HOW_VOICE, HOW_FACE, HOW_PHRASE)

# There is deliberately no HOW_NAME. A leg that can win the rank is a leg
# that can admit, and "he said he was Mara" is not evidence that he is.
# tests/test_signin.py pins that recognise() never returns how == "name".
HOW_NAME_IS_NOT_A_LEG = "name"


def recognise(legs: Legs, roles: Mapping[str, str], owner: str) -> Verdict:
    """``(who, role, how)`` -- and ``how`` is the point.

    The pane and the log have to be able to say WHICH leg decided. That is
    this codebase's habit and it is what makes a wrong answer debuggable at
    all: "he was refused" is not a bug report, "the face leg named Heather
    while the voice leg abstained" is.
    """
    try:
        table = {str(k): str(v) for k, v in dict(roles or {}).items()}
    except Exception:  # noqa: BLE001 - a mapping that cannot be read is empty
        table = {}

    offered = {
        HOW_VOICE: str(getattr(legs, "voice_says", "") or ""),
        HOW_FACE: str(getattr(legs, "face_says", "") or ""),
        HOW_PHRASE: str(getattr(legs, "phrase_says", "") or ""),
    }

    best = None
    for how in _ORDER:
        name = offered[how]
        if not name:
            continue
        role = table.get(name, ROLE_UNKNOWN)
        if not role:
            # A leg named somebody the registry has never heard of. That is
            # not an identity, and it must not become one by being the only
            # thing anybody said.
            continue
        if how == HOW_PHRASE and role != ROLE_OWNER:
            # Rule 3. A KNOWN row carrying a phrase hash must not promote
            # its owner; the phrase is the OWNER's way back in.
            continue
        rank = RANK.get(role, 0)
        if best is None or rank > best[0]:
            best = (rank, how, name, role)

    claimed = str(getattr(legs, "name_says", "") or "")
    if claimed and claimed not in table:
        # A name nobody on file answers to narrows to nothing, exactly as
        # an unrelated sentence does.
        claimed = ""

    if best is None:
        # Rows 11 and 12 of the table return THE SAME OBJECT. A near miss
        # on the passphrase must not be distinguishable from an unrelated
        # sentence -- not by a field, not by a reason, not by anything that
        # could be read or timed.
        #
        # A CLAIMED NAME CHANGES NONE OF THAT. ``who`` is still "", ``role``
        # is still unknown, ``how`` is still "nobody" and ``why`` is the
        # same sentence: this verdict has admitted nobody, and every caller
        # that asks "was anyone recognised?" gets the same answer it got
        # before ``claimed`` existed. It rides along only so the refusal can
        # offer a way forward instead of being a dead end.
        return Verdict(who="", role=ROLE_UNKNOWN, how=HOW_NOBODY,
                       why="no leg named anyone enrolled", claimed=claimed)

    _rank, how, name, role = best
    return Verdict(who=name, role=role, how=how,
                   why="%s named %s" % (how, name), claimed=claimed)
