"""Somebody else's agreement: ONE rule, one set of words, two takers.

WHY THIS FILE EXISTS. ``scripts/face_enrol.py`` has taken consent since the
gallery was built, and ``scripts/jarvis_people.py`` imports that function
rather than writing a second rule that could drift. The users tab
(``jarvis/ui/users_page.py``) cannot do the same thing, and the reason is
mechanical rather than stylistic: ``face_enrol.consent`` refuses unless
stdin AND stdout are both terminals, and a Tk window has neither. So the
import would succeed and the call would always answer "no".

The answer is NOT a second dialog with its own paragraph. It is this file:
the WORDS and the ONE RULE live here, and the two takers differ only in
their precondition and in what they record afterwards.

    take_at_terminal   somebody typed it (stdin a tty) and could read what
                       they were agreeing to (stdout a tty). That check is
                       NOT weakened or deleted -- it is this taker's
                       precondition and it stays exactly where it was.
    take_at_console    a real mapped window on the console, never a
                       synthetic event.

AND THEY RECORD DIFFERENT THINGS. ``Person.consent`` and the gallery's
provenance string are the only durable record of HOW an agreement was
taken. A console consent that wrote "typed" would be a false attestation on
disk -- worse than no record, because somebody reading it later would
believe a terminal ceremony happened. The console taker returns "console".
It costs nothing and keeps the record true.

THREE TEXTS, BECAUSE THEY STORE DIFFERENT THINGS. This is a finding rather
than a preference. ``CONSENT_LINES`` (the face text, moved here verbatim
from face_enrol so its sentences are unchanged) describes storing 128
numbers of somebody's face. ``jarvis_people.py add`` stores NO face data:
it writes a registry row holding a name, a role and a face LABEL. Using the
face words there over-claims, so ``WHAT_ROW`` has its own text saying what
a row actually is and pointing at the separate ceremony the lens needs.
``WHAT_VOICE`` arrived the same way and for the same reason: it moved
here verbatim from ``scripts/voice_enrol.py`` the moment a SECOND caller
existed (the in-app run, jarvis/voicerun.py), before it could drift. It
describes 192 numbers per take and says the thing the face text cannot:
NO RECORDING IS KEPT -- the audio becomes those numbers and is thrown
away, and the numbers cannot be turned back into speech. There is still
one place and one rule; no text lies.

NOTHING HERE OPENS A DEVICE and nothing here writes a file. It formats
sentences and compares a typed string with a label.
"""
from __future__ import annotations

import sys
from typing import Callable, Optional, Sequence, Tuple

# What is being agreed to. Not a boolean: a third thing to store would
# otherwise arrive as "the other one".
WHAT_FACE = "face"      # 128 numbers per take, in the gallery
WHAT_ROW = "row"        # a registry row: a name, a role, a face LABEL
WHAT_VOICE = "voice"    # 192 numbers per take, in the voice gallery

# The provenance strings. NEVER interchangeable -- see the docstring.
HOW_TERMINAL = "typed"
HOW_CONSOLE = "console"
HOW_OWNER = "owner"

# Moved from scripts/face_enrol.py unchanged: tests/test_faceenrol_notes.py
# pins these sentences, and this file must not be the reason they change.
CONSENT_LINES = (
    "CONSENT -- this is %(who)s's data, not yours.",
    "",
    "Enrolling %(who)s stores a measurement of %(who)s's FACE: 128 numbers",
    "per take, in %(root)s, at 0600 in a 0700",
    "directory. No photograph, no video and no crop is stored, nothing is",
    "displayed, and nothing leaves this machine. It cannot be re-issued if",
    "it leaks, which is why it is treated like the voiceprint and not like",
    "a setting.",
    "",
    "What it is FOR: Jarvis can say %(who)s is here, and can keep something",
    "private when somebody else is in the room. What it never does: a",
    "recognised face cannot command Jarvis, cannot unlock anything and",
    "cannot pass the voice gate. Recognising a face may only ever make",
    "Jarvis do LESS, never more.",
    "",
    "Deleting it, at any time, and it takes about a second:",
    "    %(python)s %(script)s --delete --label %(who)s",
    "which destroys every generation that holds %(who)s -- including the",
    "older ones -- and leaves everybody else's alone.",
    "",
    "%(who)s must type their own name below. Nobody may type it for them,",
    "and --yes cannot do it either: it is his flag, and this is not his",
    "consent to give.",
)

# The registry row. Deliberately shorter, because it stores less.
ROW_LINES = (
    "CONSENT -- this is about %(who)s, and it is %(who)s's to give.",
    "",
    "Being added stores a NAME for %(who)s, a role, and the label of a face",
    "already in the gallery. It stores no measurement of %(who)s: no",
    "photograph, no video, no numbers off a lens, no recording of a voice.",
    "Nothing leaves this machine.",
    "",
    "What it is FOR: Jarvis can say %(who)s is here by name, and can keep",
    "something private while %(who)s is in the room. What it never does: a",
    "recognised person cannot command Jarvis, cannot unlock anything and",
    "cannot pass the voice gate. Being recognised may only ever make Jarvis",
    "do LESS, never more.",
    "",
    "Removing it takes one press here and nothing survives it but a face",
    "measurement, if %(who)s ever gave one -- and that is a SEPARATE",
    "agreement, taken at the camera, at a terminal. Nothing here captures",
    "anything.",
    "",
    "%(who)s must type their own name below. Nobody may type it for them.",
)

VOICE_LINES = (
    "CONSENT -- this is %(who)s's data, not yours.",
    "",
    "Enrolling %(who)s stores a measurement of %(who)s's VOICE: 192 numbers",
    "per take, in %(root)s, at 0600 in a 0700",
    "directory. NO RECORDING IS KEPT. The audio is turned into those numbers",
    "and thrown away; there is no file anywhere that can be played back. The",
    "numbers cannot be turned back into speech, and nothing leaves this",
    "machine -- no cloud, no API, no upload, ever.",
    "",
    "What it is FOR: Jarvis can tell %(who)s apart from the other people he",
    "knows, greet %(who)s by name, and keep something private when somebody",
    "else is in the room.",
    "",
    "WHAT IT IS NOT: it is not a password and not a lock. A recording of",
    "%(who)s's voice would defeat it, and it is not meant to withstand",
    "somebody determined. Being recognised here does NOT let %(who)s change",
    "settings, read the owner's mail or calendar, or send anything off this",
    "machine.",
    "",
    "Deleting it, at any time, and it takes about a second:",
    "    %(python)s %(script)s --delete --label %(who)s",
    "which destroys every generation that holds %(who)s -- including the old",
    "ones -- and leaves everybody else's alone.",
    "",
    "%(who)s must type their own name below. Nobody may type it for them,",
    "and --yes cannot do it either: that is the owner's flag, and this is",
    "not the owner's consent to give.",
)

_TEXTS = {WHAT_FACE: CONSENT_LINES, WHAT_ROW: ROW_LINES,
          WHAT_VOICE: VOICE_LINES}

PROMPT = 'Type "%s" to agree: '
PIPE_REFUSAL = ("consent for %r cannot be taken through a pipe: the person "
                "being enrolled has to read what is stored and type their "
                "own name, at a terminal. Run this without redirecting "
                "stdin or stdout.")
WINDOW_REFUSAL = ("consent for %r needs a window they can actually read: "
                  "this one is not on screen, so nothing was asked and "
                  "nothing was written")
NO_REFUSAL = ("consent was not given for %r -- nothing was captured and "
              "nothing was written")


def lines_for(what: str, fields: dict) -> Tuple[str, ...]:
    """The words for what is actually being stored, already filled in.

    A missing field is left as-is rather than raising: a consent that
    cannot be RENDERED must not become a consent that cannot be REFUSED.
    """
    text = _TEXTS.get(str(what), ROW_LINES)
    out = []
    for line in text:
        try:
            out.append(line % (fields or {}))
        except Exception:  # noqa: BLE001 - a field nobody supplied
            out.append(line)
    return tuple(out)


def agreed(label, answer) -> bool:
    """THE RULE: they type their OWN label, exactly. Nothing else counts.

    Not "yes", not "y", not a click. Trimmed and case-folded, because the
    terminal ceremony has always accepted "Pemberton" for ``pemberton`` and
    the console must not be stricter than the thing it stands in for.
    """
    want = str(label or "").strip().lower()
    if not want:
        return False
    try:
        got = str(answer or "").strip().lower()
    except Exception:  # noqa: BLE001 - an answer that cannot be read is none
        return False
    return got == want


def prompt(label) -> str:
    return PROMPT % str(label or "")


def _isatty(stream) -> bool:
    """A stream that cannot answer is NOT a terminal -- the unknown case
    fails closed, the rule scripts/face_enrol.py already wrote down."""
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _show(say: Optional[Callable], lines: Sequence[str]) -> None:
    if not callable(say):
        return
    for line in lines:
        say(line)


def take_at_terminal(label, *, what: str = WHAT_FACE, fields=None,
                     say: Optional[Callable] = None,
                     ask: Optional[Callable] = None,
                     isatty: Optional[Callable] = None,
                     owner: str = "") -> Tuple[bool, str]:
    """``(ok, how)``. BOTH ENDS must be terminals, as they always were.

    ``how`` on success is ``HOW_TERMINAL``; on refusal it is the sentence
    to print. ``ask`` and ``isatty`` are seams so the ceremony can be
    exercised without a tty -- never so the rule can be skipped: the
    default asks the real ``input`` about the real streams.
    """
    who = str(label or "")
    if owner and who == str(owner):
        return True, HOW_OWNER
    tty = isatty or _isatty
    if not (tty(sys.stdin) and tty(sys.stdout)):
        return False, PIPE_REFUSAL % who
    _show(say, lines_for(what, dict(fields or {}, who=who)))
    reader = ask or input
    try:
        answer = reader(prompt(who))
    except EOFError:
        answer = ""
    if not agreed(who, answer):
        return False, NO_REFUSAL % who
    return True, HOW_TERMINAL


def take_at_console(label, *, what: str = WHAT_ROW, fields=None,
                    show: Optional[Callable] = None,
                    ask: Optional[Callable] = None,
                    mapped: Optional[Callable] = None,
                    owner: str = "") -> Tuple[bool, str]:
    """``(ok, how)`` on the console. Same words, same rule, own provenance.

    ``mapped`` is this taker's precondition and the analogue of the tty
    check: a real window, on screen, that the person could read. It is
    asked BEFORE the words are shown, so a consent nobody could see is
    never half-taken.
    """
    who = str(label or "")
    if owner and who == str(owner):
        return True, HOW_OWNER
    on_screen = mapped or (lambda: False)
    try:
        visible = bool(on_screen())
    except Exception:  # noqa: BLE001 - a window that cannot answer is not one
        visible = False
    if not visible:
        return False, WINDOW_REFUSAL % who
    _show(show, lines_for(what, dict(fields or {}, who=who)))
    reader = ask or (lambda: "")
    try:
        answer = reader()
    except Exception:  # noqa: BLE001 - an entry that cannot be read said no
        answer = ""
    if not agreed(who, answer):
        return False, NO_REFUSAL % who
    return True, HOW_CONSOLE
