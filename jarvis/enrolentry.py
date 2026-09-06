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

AND SINCE 2026-09-05 THAT CLAIM IS TRUE AT IMPORT TIME AS WELL, which it was
not. ``jarvis.faceenrol`` and ``jarvis.facegallery`` were imported at module
scope, and faceenrol imports ``jarvis.facedetect`` and ``jarvis.visionrig`` --
so merely importing THIS file pulled the whole vision stack in. That was found
by the USERS page (jarvis/ui/users_page.py), which asks this module for one
STRING -- the command that deletes somebody's face measurements -- and had the
photo rig's lens blocker refuse the import outright. A page for managing names
has no business loading a face detector, so the two heavy imports moved inside
the four functions that actually need a gallery. ``command_line`` and
``to_clipboard`` -- the pure string builder and the clipboard seam -- now need
neither, which is what the users tab uses.

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
                 append: bool = False, plan: str = "",
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
    minute for a refusal he could not have predicted.

    ``plan`` is the script's --plan when the named takes need the five
    stations alongside them ("full"); "" leaves the script's default. It
    goes BEFORE the poses, because the pose is the last word on the line
    and a test pins that."""
    out = [shlex.quote(python or sys.executable),
           shlex.quote(script or script_path())]
    if delete:
        out.append("--delete")
    # A delete ALWAYS names its person. Without --label the script's
    # --delete is "everything": every generation of every face, and
    # camera.identity switched off. "Forget my face" used to hand that
    # over for the owner (whose label is elided everywhere else), so a
    # sentence about his 6 takes would have erased Heather's as well and
    # asked for a different confirmation word than the one he was told
    # (F32, reproduced 2026-09-03). do_delete_label is the one-person path.
    if label and (label != owner or delete):
        out += ["--label", shlex.quote(label)]
    if append and not delete:
        out.append("--append")
    if plan and not delete:
        out += ["--plan", shlex.quote(str(plan))]
    for pose in poses:
        from jarvis.facegallery import clean_note   # noqa: PLC0415 - lazy
        note = clean_note(pose)
        if note:
            out += ["--pose", shlex.quote(note)]
    return " ".join(out)


def _read_back(runner: Callable) -> Optional[str]:
    """What the clipboard holds right now, or None if it could not be asked.

    None and "" are DIFFERENT answers and the caller needs both: "" is an
    empty clipboard (the write did not land), None is "xclip could not tell
    me", and neither may be reported as a successful write."""
    try:
        got = runner(["xclip", "-selection", "clipboard", "-o"],
                     capture_output=True, timeout=CLIP_TIMEOUT_S, check=False)
    except Exception:  # noqa: BLE001 - absence is not a crash
        log.debug("enrolentry: the clipboard could not be read back",
                  exc_info=True)
        return None
    if getattr(got, "returncode", 1) != 0:
        return None
    out = getattr(got, "stdout", None)
    if out is None:
        return None
    if isinstance(out, bytes):
        try:
            return out.decode("utf-8")
        except Exception:  # noqa: BLE001 - not our text, so not our write
            return None
    return str(out)


def to_clipboard(text: str, run: Optional[Callable] = None) -> bool:
    """Put ``text`` on the X clipboard AND read it back. False either way it
    can fail.

    WHY THE READ-BACK EXISTS, AND WHAT IT DOES NOT PROVE. xclip exiting 0
    only proves xclip STARTED. An X11 CLIPBOARD selection has no storage at
    all: a live process owns the selection and serves the bytes when
    somebody pastes. So the old ``returncode == 0`` was never evidence that
    the clipboard holds this text, and it is certainly not evidence that it
    will still hold it when he pastes -- 2026-09-03, Jarvis said "I've put
    the command on your clipboard" and the clipboard did not have it.

    Reading it back closes exactly one of those two holes: IT NEVER LANDED.
    It cannot close the other. The clipboard is shared global state with a
    single owner, and any other process on his desktop may take it in the
    microseconds after this returns -- measured that day, a sentinel written
    by an unrelated process was what he found instead. True here means "it
    was there when I looked", never "it will be there when you paste", and
    the sentence built on this flag has to say so.

    A MISMATCH IS RETRIED ONCE, because xclip backgrounds a child to own the
    selection and the parent can exit a hair before that child has taken
    ownership; the second spawn is its own few milliseconds of delay. The
    retry costs nothing on the path that worked.

    Cost, measured 2026-09-03 on this box: the read-back process is 1.4 ms
    median (min 1.07, max 2.28 over 25 runs) excluding the X round trip,
    against a 1.3 s turn. It also only runs on the two hand-over commands,
    not on every reply.

    xclip is what jarvis/reader.py already reads the clipboard WITH, so this
    adds no dependency. Every failure -- no xclip, no display, a timeout --
    is False and a debug line, never an exception: a clipboard that did not
    work must cost the convenience and not the answer."""
    runner = run or subprocess.run
    want = str(text)
    try:
        proc = runner(["xclip", "-selection", "clipboard"],
                      input=want.encode("utf-8"),
                      timeout=CLIP_TIMEOUT_S, check=False)
    except Exception:  # noqa: BLE001 - absence is not a crash
        log.debug("enrolentry: xclip could not be run", exc_info=True)
        return False
    if getattr(proc, "returncode", 1) != 0:
        return False
    for _attempt in (1, 2):
        back = _read_back(runner)
        if back is None:
            log.debug("enrolentry: xclip exited 0 but the clipboard could "
                      "not be read back")
            return False
        # A copy that appends a newline is still the command he needs; a
        # copy that holds anything else is not this write.
        if back == want or back.rstrip("\n") == want:
            return True
    log.debug("enrolentry: xclip exited 0 but the clipboard holds %d other "
              "characters", len(back))
    return False


def _clip(text: str, clipboard: Optional[Callable[[str], bool]]) -> bool:
    fn = to_clipboard if clipboard is None else clipboard
    try:
        return bool(fn(text))
    except Exception:  # noqa: BLE001
        log.debug("enrolentry: the clipboard hook raised", exc_info=True)
        return False


# WHAT HE IS TOLD, AND WHY IT IS WORDED LIKE THIS. Both lines are true
# whichever way the clipboard went, and neither promises a durability the X
# clipboard does not have (see to_clipboard). "It's in the transcript" is the
# only part of this that is a guarantee, so it leads in both branches -- the
# clipboard is named second, as the convenience it is.
#
# AND IT IS SHORT ON PURPOSE. This line is the tail of every hand-over, and
# jarvis/tts.py cuts speech at MAX_SPEAK_LENGTH (500) on the last full stop
# before the limit. The F16 reply -- his live 13-take, no-angle gallery plus
# one named pose -- measured 585 characters with the earlier 192-character
# version of this line, and the TTS spoke 405 of them: the sentence saying
# WHY the run was stopped and this line saying WHERE the command is were
# the part that went. tests/test_faceenrol_notes.py pins that reply under
# the cut with margin; this file does not import the engine to format text.
CLIP_OK_LINE = ("The command is in the transcript, sir, and I've copied it "
                "to your clipboard too - anything else that copies will "
                "take it from me.")
CLIP_FAILED_LINE = ("The command is in the transcript, sir - the clipboard "
                    "wouldn't take it, so copy it from there.")


def _hand_over(command: str, spoken: str,
               clipboard: Optional[Callable[[str], bool]]) -> dict:
    """One shape for "here is the command": say what is true, and put the
    command itself where he can always reach it.

    THE COMMAND IS NO LONGER DELIVERED BY THE CLIPBOARD. It travels in
    ``display_only``, which the console shows and the TTS never reads
    (jarvis/commander.py CommandResult.display_only, routed in
    JarvisApp._emit_result). That is the fix for the failure this file
    caused: a clipboard write that silently did not survive to his paste
    left him with a claim and nothing else. Now the clipboard can fail
    completely and he still has the command in front of him.

    Speaking a file path is still a bad minute of text-to-speech, so the
    command is SHOWN and never SPOKEN -- which is the whole reason
    ``display_only`` had to exist rather than being appended to ``reply``."""
    clipped = _clip(command, clipboard)
    reply = "%s %s" % (spoken, CLIP_OK_LINE if clipped else CLIP_FAILED_LINE)
    return {"reply": reply, "command": command, "clipped": clipped,
            "display_only": command}


def enrol_answer(gallery, label: str, owner: str = "hunter",
                 poses: Sequence[str] = (),
                 clipboard: Optional[Callable[[str], bool]] = None) -> dict:
    """"Enrol my face" / "add Heather's face" -- what Jarvis says and hands
    over.

    Reads the gallery to say what the run will ASK FOR, which is the part
    worth knowing before he starts: with recorded coverage it is the gap, and
    with none it is the five stations. No device is opened to work that out.
    """
    from jarvis import faceenrol as fe               # noqa: PLC0415 - lazy
    from jarvis.facegallery import label_ok          # noqa: PLC0415 - lazy
    label = str(label or owner).strip().lower()
    if not label_ok(label):
        return {"reply": "That isn't a name I can store, sir - lowercase "
                         "letters, digits, dashes and underscores only.",
                "status": "Enrolment: bad name", "command": "",
                "display_only": "", "clipped": False}
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
    # THE COMMAND HANDED OVER HAS TO BE ONE THAT CAN PASS. A named take is
    # one station of three, and ``judge_gallery`` wants eight in the pool
    # and two angles in each band: on a box with no gallery that is a
    # guaranteed [FAIL] samples after the consent step and the minute
    # (F34), and on his live generation -- 13 takes with no recorded angle
    # -- an --append of one pose can never reach the spread (F16). Both are
    # counting problems, ``plan_shortfalls`` counts them, and when the
    # named take alone falls short the five stations go with it
    # (--plan full --pose ...). With real coverage stored, one more way to
    # be recognised is still one station.
    kept = takes if add else []
    full = bool(poses) and bool(fe.plan_shortfalls(kept, plan))
    if full:
        plan, why = fe.choose_plan([], poses=poses, mode="full")
    command = command_line(label, owner=owner, poses=poses, append=add,
                           plan="full" if full else "")
    if add:
        why = "%s, added to the %d already stored" % (why, len(takes))
    cov = fe.coverage(takes)
    if full and not takes:
        why += (" - a first enrolment has to clear the %d-take floor on "
                "its own, so the five stations come with it"
                % fe.MIN_SAMPLES)
    elif full and not cov["recorded"]:
        # F16 exactly: every stored take is angle-less. ONE sentence says
        # so and says what follows from it; the coverage tail below is
        # skipped for this case because it would say the same thing again,
        # and the repeat is what pushed the spoken reply past the TTS cut.
        why += (" - none of those carry a pose record, so the five stations "
                "come with it")
    elif full:
        why += (" - what's stored doesn't give the pose check enough to go "
                "on, so the five stations come with it")

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
    if cov["recorded"] and not cov["negative"]:
        spoken += (" Worth knowing first: %s have no takes turned the other "
                   "way at all, so the third station is the one that "
                   "matters." % ("you" if label == owner else who))
    elif cov["unrecorded"] and not cov["recorded"] and not full:
        # Not under ``full``: that branch has already said the stored takes
        # carry no angle, in the sentence that explains the plan.
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
    from jarvis.facegallery import label_ok          # noqa: PLC0415 - lazy
    label = str(label or "").strip().lower()
    if not label_ok(label):
        return {"reply": "That isn't a name I can look up, sir.",
                "status": "Face gallery", "command": "",
                "display_only": "", "clipped": False}
    count = 0
    try:
        gallery.load()
        count = gallery.count(label)
    except Exception:  # noqa: BLE001
        log.debug("enrolentry: the gallery could not be read", exc_info=True)
    if not count:
        return {"reply": "There's nothing enrolled under %s, sir."
                         % (label.capitalize() if label != owner else "you"),
                "status": "Face gallery", "command": "",
                "display_only": "", "clipped": False}
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
    from jarvis import faceenrol as fe               # noqa: PLC0415 - lazy
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
# A FUNCTION WORD IS NOT PART OF A NAME. The commander's grammar already
# refuses to let one INTO the who group -- that is where the "add a reminder
# to wash my face" class was closed -- but this function is the one any
# caller reaches for, so it refuses them here as well rather than trusting a
# regex two modules away. Only the whole-phrase list above catches "the";
# this catches "that photo", where each token is fine and the phrase is not.
_FILLER = ("for", "to", "from", "of", "about", "on", "in", "at", "by", "with",
           "and", "but", "or", "it", "is", "was", "what", "which", "when",
           "while", "if", "so", "there", "here", "all", "any", "some",
           # determiners and possessives, which _NOBODY only catches when
           # they are the WHOLE phrase: "that photo" is two good tokens and
           # still nobody.
           "a", "an", "the", "that", "this", "these", "those", "my", "your",
           "his", "her", "its", "their", "our")


def spoken_label(text: str, owner: str = "hunter") -> str:
    """The name in "enrol Heather's face" / "enrol my face".

    "my", "me" and "mine" are him; a word that names nobody in particular is
    "" and the caller asks whose face. Anything else is squeezed through the
    same rule the gallery stores under (``label_ok``), so a name it cannot
    hold is refused before the minute in front of the camera rather than
    after it.

    Two words become one hyphenated label -- "Mary Jane" is ``mary-jane``,
    not ``mary`` -- because taking the first token silently enrols the wrong
    name under a label that looks like it worked.

    A phrase carrying a FUNCTION WORD names nobody either, and returns "" the
    same way "the" does. "That photo" and "hair from" are each two perfectly
    valid tokens and neither is a person; storing one produces a real,
    gallery-shaped, permanent label in the one store whose entire point is
    knowing whose face it holds."""
    from jarvis.facegallery import label_ok          # noqa: PLC0415 - lazy
    word = str(text or "").strip().lower()
    word = word.replace("'s", "").replace("\u2019s", "").strip()
    if not word or word in _MINE:
        return owner
    if word in _NOBODY:
        return ""
    parts = word.split()
    if any(w in _FILLER for w in parts):
        return ""
    parts = ["".join(c for c in w if c.isalnum() or c in "-_") for w in parts]
    keep = "-".join(w for w in parts if w)
    return keep if label_ok(keep) else ""


def spoken_pose(text: str) -> str:
    """The pose in "enrol my face looking at my phone" -- "looking at my
    phone" -- squeezed through the same cleaner the gallery stores notes with.

    Says out loud what the note is FOR: it is written beside the embedding so
    that six months later "which of my takes is letting me down" has an
    answer better than an index. ``clean_note`` caps it and strips the
    non-printables, and ``command_line`` shlex-quotes it before it goes
    anywhere near his clipboard, so a spoken pose cannot become shell."""
    from jarvis.facegallery import clean_note        # noqa: PLC0415 - lazy
    return clean_note(str(text or "").strip())


# ---------------------------------------------------------- what a purge did
# THE ONE THING A SURFACE MAY NOT DO WITH A PURGE IS ROUND IT UP. Both
# galleries' ``purge_label`` return NUMBERS rather than a boolean, precisely so
# a caller cannot claim a delete happened because a call did not raise: the
# invariant is that no file is destroyed until the embeddings it held, minus
# theirs, have been read back off disk from the new generation by name AND by
# sample count, and anything that fails any step is counted, named and left
# alone. Three separate bugs moved through that room. So this builds the
# sentence FROM those numbers, and when ``complete`` is False it says they are
# still there and names the command that finishes the job.
PURGE_FOLLOW_UP = ("Run the enrolment script with --status to see what is "
                   "left, then --delete with no label to destroy everything.")


def purge_line(report, *, kind: str = "face") -> str:
    """One honest sentence about what a purge actually did.

    ``kind`` is "face" or "voice", and the line always names the OTHER one --
    a face purge that quietly left a voice pool behind would be the same
    half-truth as a forget that left a gallery behind, which is the defect
    this whole surface is correcting.
    """
    rep = dict(report or {})
    who = str(rep.get("label") or "somebody")
    other = "voice" if kind == "face" else "face"
    held = list(rep.get("generations_with") or ())
    removed = int(rep.get("removed") or 0)
    complete = bool(rep.get("complete"))
    tail = "This does not touch %s's %s." % (who, other)

    if not held and complete:
        return ("Nothing to remove: no %s generation held %s. %s"
                % (kind, who, tail))

    if complete:
        return ("Removed. %d %s generation(s) held %s; %d file(s) were "
                "overwritten, and everybody else was read back from the new "
                "generation at full count. %s"
                % (len(held), kind, who, removed, tail))

    # NOT COMPLETE. Say which generations still hold them and WHY each one
    # could not be finished, because "some of it worked" is not something
    # anybody can act on.
    bits = []
    still = list(rep.get("still_holding") or ())
    if still:
        bits.append("generation(s) %s still hold %s"
                    % (", ".join(str(g) for g in still), who))
    unreadable = list(rep.get("unreadable") or ())
    if unreadable:
        bits.append("%s could not be read, so %s was left alone"
                    % (_gens(unreadable), _they(unreadable)))
    foreign = list(rep.get("foreign") or ())
    if foreign:
        models = ", ".join(str(m) for m in (rep.get("foreign_models") or ())
                           if m)
        bits.append("%s %s written by another model (%s) this build cannot "
                    "rewrite" % (_gens(foreign), _were(foreign),
                                 models or "unknown"))
    not_carried = list(rep.get("not_carried") or ())
    if not_carried:
        bits.append("%s hold somebody else who could not be carried forward, "
                    "so destroying %s would cost them their enrolment"
                    % (_gens(not_carried), _they(not_carried)))
    why = "; ".join(bits) or "the read-back could not be completed"
    return ("Not finished, and nothing here may claim otherwise: %s. %d "
            "file(s) were overwritten. %s %s"
            % (why, removed, PURGE_FOLLOW_UP, tail))


def _gens(gens) -> str:
    return "generation%s %s" % ("" if len(gens) == 1 else "s",
                                ", ".join(str(g) for g in gens))


def _were(gens) -> str:
    return "was" if len(gens) == 1 else "were"


def _they(gens) -> str:
    return "it" if len(gens) == 1 else "they"


VOICE_SCRIPT = "scripts/voice_enrol.py"


def voice_script_path() -> str:
    """Where scripts/voice_enrol.py is, derived the same way and for the same
    reason: a hard-coded ~/Jarvis is wrong in every worktree."""
    return str((Path(__file__).resolve().parent.parent / VOICE_SCRIPT))


def voice_command_line(label: str, name: str = "", delete: bool = False,
                       python: Optional[str] = None,
                       script: Optional[str] = None) -> str:
    """The hand-over for somebody ELSE's voice, quoted.

    THE VOICE HALF OF THE SAME REFUSAL. A window cannot take a third party's
    consent -- ``consent.take_at_terminal``'s precondition is two real ttys
    and its whole point is that nobody may type it for them, which a dialog
    driven by whoever is already logged in cannot reproduce. So a guest's
    voice stays at the terminal and this is the command, ready to run.

    IMPORT-LIGHT, like ``command_line`` beside it: no gallery, no model, no
    recorder is reached to build a string. That property is tested, because
    importing this module used to pull the whole vision stack in.
    """
    out = [shlex.quote(python or sys.executable),
           shlex.quote(script or voice_script_path())]
    if delete:
        out.append("--delete")
    out += ["--label", shlex.quote(str(label or ""))]
    if name:
        out += ["--name", shlex.quote(str(name))]
    return " ".join(out)
