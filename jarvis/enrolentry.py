"""The way into face enrolment from inside Jarvis, and what he can ask about
the gallery without a camera opening at all.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT. He asked to "add the
enrollment option into Jarvis". Enrolment itself is a guided minute in front
of the lens -- five stations, an Enter between each, a numbers-only report he
pastes -- and three things make running that INSIDE the live window the wrong
build today:

* the capture needs the camera device, and the running Jarvis is the thing
  that owns it (jarvis/sensing.py). Two openers of /dev/video0 is one opener
  and one confusing failure;
* it needs a keyboard turn-by-turn ("press Enter when you are in position"),
  which is a modal dialog and a capture surface in jarvis/ui -- and three
  other branches are editing those files right now;
* the report is the product. It is 30-odd lines of numbers he PASTES, and a
  spoken assistant is the worst possible reader for it.

SO THIS IS AN ENTRY POINT, NOT A CAPTURE UI, AND IT SAYS SO OUT LOUD. Jarvis
answers by handing over the exact command -- with his named poses and the
right label already in it -- and putting it on the clipboard. What it DOES
answer in full, because these need no lens at all, are the questions the
notes were added for: who is enrolled, how many takes each person has, which
pose coheres worst, and which side of his face the gallery has never seen.

NOTHING HERE OPENS A DEVICE, and there is no import in this file that could.
No cv2, no jarvis.camera, no frame, no crop. It reads the gallery -- which is
128 floats and a string per take -- and formats sentences.

DELETING IS HANDED OVER TOO, AND THAT IS NOT TIMIDITY. "Forget Heather's
face" arrives as a speech-recognition result. Destroying biometric data on a
word that might have been misheard is not a risk worth taking for the sake of
saving him a paste, so the voice command produces the command and the typed
confirmation still happens in a terminal.

AND THE RULE THIS FILE CANNOT BREAK, said here because this is the file that
adds names: identity may REMOVE capability or ADD a name; it must never GRANT
capability the existing gates do not already grant. Nothing in here consults
who is in the room, and nothing in here is reachable from an identity result
-- it is reachable from a spoken command, which is gated by the wake word and
the speaker verifier exactly as every other command is.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from jarvis import faceenrol as fe
from jarvis.facegallery import clean_note, label_ok
from jarvis.logs import get_logger

log = get_logger("enrolentry")

SCRIPT = "scripts/face_enrol.py"
# The clipboard write is best-effort and must never be the thing that fails a
# command: if it does not land, the command text goes into the reply instead.
CLIP_TIMEOUT_S = 2.0


def script_path() -> str:
    """Where scripts/face_enrol.py is, from where this module is.

    Derived rather than configured: a hard-coded ~/Jarvis would be wrong in
    every worktree, and a worktree is where this was written."""
    return str((Path(__file__).resolve().parent.parent / SCRIPT))


def command_line(label: str, owner: str = "hunter",
                 poses: Sequence[str] = (), delete: bool = False,
                 append: bool = False,
                 python: Optional[str] = None,
                 script: Optional[str] = None) -> str:
    """The exact shell command, quoted so a pose with spaces survives it.

    ``shlex.quote`` on every part, including the paths: a worktree path with
    a space in it would otherwise produce a command that runs the wrong
    thing, and this string is going straight onto his clipboard.

    ``append`` IS NOT COSMETIC WHEN HE NAMED POSES. "Add more ways for me to
    be recognised" is additive by intent, and a plain run REPLACES the pool
    -- so a two-pose command without it captures six takes over an existing
    thirteen, fails the eight-sample floor, saves nothing, and costs him the
    minute for a refusal he could not have predicted."""
    out = [shlex.quote(python or sys.executable),
           shlex.quote(script or script_path())]
    if delete:
        out.append("--delete")
    if label and label != owner:
        out += ["--label", shlex.quote(label)]
    if append and not delete:
        out.append("--append")
    for pose in poses:
        note = clean_note(pose)
        if note:
            out += ["--pose", shlex.quote(note)]
    return " ".join(out)


def to_clipboard(text: str, run: Optional[Callable] = None) -> bool:
    """Put ``text`` on the X clipboard. False if that could not happen.

    xclip is what jarvis/reader.py already reads the clipboard WITH, so this
    adds no dependency. Every failure -- no xclip, no display, a timeout --
    is False and a debug line, never an exception: a clipboard that did not
    work must cost the convenience and not the answer."""
    runner = run or subprocess.run
    try:
        proc = runner(["xclip", "-selection", "clipboard"],
                      input=str(text).encode("utf-8"),
                      timeout=CLIP_TIMEOUT_S, check=False)
    except Exception:  # noqa: BLE001 - absence is not a crash
        log.debug("enrolentry: xclip could not be run", exc_info=True)
        return False
    return getattr(proc, "returncode", 1) == 0


def _clip(text: str, clipboard: Optional[Callable[[str], bool]]) -> bool:
    fn = to_clipboard if clipboard is None else clipboard
    try:
        return bool(fn(text))
    except Exception:  # noqa: BLE001
        log.debug("enrolentry: the clipboard hook raised", exc_info=True)
        return False


def _hand_over(command: str, spoken: str,
               clipboard: Optional[Callable[[str], bool]]) -> dict:
    """One shape for "here is the command": speak the short sentence when the
    clipboard took it, and put the command IN the sentence when it did not.

    Speaking a file path is a bad minute of text-to-speech, so the path is
    only ever spoken when there is no other way for him to get it."""
    clipped = _clip(command, clipboard)
    if clipped:
        reply = "%s I've put the command on your clipboard." % spoken
    else:
        reply = "%s I couldn't reach the clipboard, so here it is:\n%s" \
            % (spoken, command)
    return {"reply": reply, "command": command, "clipped": clipped}


def enrol_answer(gallery, label: str, owner: str = "hunter",
                 poses: Sequence[str] = (),
                 clipboard: Optional[Callable[[str], bool]] = None) -> dict:
    """"Enrol my face" / "add Heather's face" -- what Jarvis says and hands
    over.

    Reads the gallery to say what the run will ASK FOR, which is the part
    worth knowing before he starts: with recorded coverage it is the gap, and
    with none it is the five stations. No device is opened to work that out.
    """
    label = str(label or owner).strip().lower()
    if not label_ok(label):
        return {"reply": "That isn't a name I can store, sir - lowercase "
                         "letters, digits, dashes and underscores only.",
                "status": "Enrolment: bad name", "command": "",
                "clipped": False}
    takes = []
    try:
        gallery.load()
        takes = gallery.takes(label)
    except Exception:  # noqa: BLE001 - a missing gallery is a first enrolment
        log.debug("enrolentry: the gallery could not be read", exc_info=True)
    # THE PLAN IS CHOSEN AGAINST AN EMPTY POOL BECAUSE THE COMMAND IS A PLAIN
    # RUN. A plain run replaces this label's pool, so the stored coverage is
    # not carried forward and "the station you are missing" would be the
    # wrong number to say out loud -- it would describe a run he is not
    # about to do. The stored coverage is still read, for the sentence
    # underneath.
    plan, why = fe.choose_plan([], poses=poses)
    add = bool(poses) and bool(takes)
    command = command_line(label, owner=owner, poses=poses, append=add)
    if add:
        why = "%s, added to the %d already stored" % (why, len(takes))

    who = "you" if label == owner else label.capitalize()
    wanted = sum(s.samples for s in plan)
    spoken = ("Enrolment runs in a terminal, sir, not in this window - it "
              "needs the camera and a key press between takes. %d station%s, "
              "%d takes: %s."
              % (len(plan), "" if len(plan) == 1 else "s", wanted, why))
    if label != owner:
        # THE CONSENT STEP IS THE HEADLINE, not a footnote after the
        # instructions: it is the part he has to arrange before he starts,
        # and the part that is not his to arrange alone.
        spoken = ("Enrolling %s stores a measurement of %s's face, sir, so "
                  "%s has to read what's kept and type their own name before "
                  "anything is captured. That's the consent step, and it "
                  "isn't yours or mine to give. It runs in a terminal: %d "
                  "station%s, %d takes."
                  % (who, who, who, len(plan),
                     "" if len(plan) == 1 else "s", wanted))
    cov = fe.coverage(takes)
    if cov["recorded"] and not cov["negative"]:
        spoken += (" Worth knowing first: %s have no takes turned the other "
                   "way at all, so the third station is the one that "
                   "matters." % ("you" if label == owner else who))
    elif cov["unrecorded"] and not cov["recorded"]:
        spoken += (" %s stored take%s carry no pose record, so this run "
                   "starts the record rather than adding to it."
                   % (cov["unrecorded"],
                      "" if cov["unrecorded"] == 1 else "s"))
    out = _hand_over(command, spoken, clipboard)
    out["status"] = "Enrolment: %s" % (who if label != owner else "you")
    out["label"] = label
    out["stations"] = len(plan)
    return out


def forget_answer(gallery, label: str, owner: str = "hunter",
                  clipboard: Optional[Callable[[str], bool]] = None) -> dict:
    """"Forget Heather's face" -- and it deletes NOTHING by voice.

    A misheard word may not destroy biometric data. What this does is name
    what would go, and hand over the command that asks for it in writing."""
    label = str(label or "").strip().lower()
    if not label_ok(label):
        return {"reply": "That isn't a name I can look up, sir.",
                "status": "Face gallery", "command": "", "clipped": False}
    count = 0
    try:
        gallery.load()
        count = gallery.count(label)
    except Exception:  # noqa: BLE001
        log.debug("enrolentry: the gallery could not be read", exc_info=True)
    if not count:
        return {"reply": "There's nothing enrolled under %s, sir."
                         % (label.capitalize() if label != owner else "you"),
                "status": "Face gallery", "command": "", "clipped": False}
    command = command_line(label, owner=owner, delete=True)
    spoken = ("That would destroy %d take%s of %s face, in every generation "
              "on the disk - so I'll leave the word to you rather than to my "
              "hearing. It asks you to type the name."
              % (count, "" if count == 1 else "s",
                 "your" if label == owner else "%s's" % label.capitalize()))
    out = _hand_over(command, spoken, clipboard)
    out["status"] = "Face gallery: delete %s" % label
    out["label"] = label
    out["count"] = count
    return out


def gallery_answer(gallery, owner: str = "hunter") -> dict:
    """"Who do you recognise?" / "which pose is weakest?" -- answered in
    full, because none of it needs a lens.

    This is the half of the feature that DOES belong in the window: the
    notes were added so a bad match has an answer, and the answer is a
    sentence, not a report."""
    try:
        loaded = bool(gallery.load())
    except Exception:  # noqa: BLE001
        log.debug("enrolentry: the gallery could not be read", exc_info=True)
        loaded = False
    if not loaded and gallery.generations():
        # There IS something there and it will not parse. That is a
        # different sentence from "nothing is enrolled", and saying the
        # wrong one would send him to enrol again over a store that needs a
        # --rollback.
        return {"reply": "There's a face gallery on the disk, sir, but I "
                         "can't read any generation of it. Run the "
                         "enrolment script with --status.",
                "status": "Face gallery: unreadable", "labels": ()}
    if not loaded or not gallery.labels():
        return {"reply": "Nothing is enrolled, sir - there are no faces in "
                         "the gallery at all.",
                "status": "Face gallery: empty", "labels": ()}
    parts, extra = [], []
    for label in gallery.labels():
        who = "you" if label == owner else label.capitalize()
        n = gallery.count(label)
        parts.append("%s, %d take%s" % (who, n, "" if n == 1 else "s"))
        takes = gallery.takes(label)
        cov = fe.coverage(takes)
        rows = fe.note_rows(gallery.embeddings(label), takes)
        weakest = fe.weakest_note(rows)
        if weakest:
            extra.append("%s weakest takes are the %s ones."
                         % ("Your" if label == owner
                            else "%s's" % who, weakest))
        if cov["recorded"] and not cov["negative"]:
            extra.append("%s no takes turned the other way at all - every "
                         "recorded angle is frontal or toward the screen."
                         % ("You have" if label == owner
                            else "%s has" % who))
        if cov["unrecorded"] and not cov["recorded"]:
            extra.append("%s %d take%s carry no pose record, so I can't say "
                         "which poses they covered."
                         % ("Your" if label == owner else "%s's" % who,
                            cov["unrecorded"],
                            "" if cov["unrecorded"] == 1 else "s"))
    reply = "I know %d face%s, sir: %s." % (
        len(gallery.labels()), "" if len(gallery.labels()) == 1 else "s",
        "; ".join(parts))
    if extra:
        reply = reply + " " + " ".join(extra)
    return {"reply": reply,
            "status": "Face gallery: %d" % len(gallery.labels()),
            "labels": tuple(gallery.labels())}


# "my face" is his. These are the words that name NOBODY -- and they have to
# be refused rather than stored, because "add the face" would otherwise enrol
# somebody under the label "the", which is a perfectly valid gallery key and
# a permanent piece of junk in a store whose whole point is knowing who is in
# it.
_MINE = ("my", "me", "mine", "myself", "your", "yours")
_NOBODY = ("a new", "another", "someone", "somebody", "a", "an", "the",
           "this", "that", "his", "her", "their", "a new person",
           "another person", "a person", "some", "new")


def spoken_label(text: str, owner: str = "hunter") -> str:
    """The name in "enrol Heather's face" / "enrol my face".

    "my", "me" and "mine" are him; a word that names nobody in particular is
    "" and the caller asks whose face. Anything else is squeezed through the
    same rule the gallery stores under (``label_ok``), so a name it cannot
    hold is refused before the minute in front of the camera rather than
    after it.

    Two words become one hyphenated label -- "Mary Jane" is ``mary-jane``,
    not ``mary`` -- because taking the first token silently enrols the wrong
    name under a label that looks like it worked."""
    word = str(text or "").strip().lower()
    word = word.replace("'s", "").replace("\u2019s", "").strip()
    if not word or word in _MINE:
        return owner
    if word in _NOBODY:
        return ""
    parts = ["".join(c for c in w if c.isalnum() or c in "-_")
             for w in word.split()]
    keep = "-".join(w for w in parts if w)
    return keep if label_ok(keep) else ""
