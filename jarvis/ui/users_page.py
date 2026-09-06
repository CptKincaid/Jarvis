"""The USERS surface: who Jarvis knows, and changing it without a terminal.

Hunter, 2026-09-05, verbatim: "a better process for processing users,
prefereably through a users tab". Until now the ONLY way to manage the
people Jarvis knows was ``scripts/jarvis_people.py`` at a keyboard.

WHAT THIS PAGE IS HONEST ABOUT, said here because every sentence it puts on
screen has to be written by somebody who knows it: THIS IS RECOGNITION, NOT
A LOCK. Anybody already at that keyboard can edit or delete ``people.json``
and turn the gate off. The override code below is HIS WAY BACK IN, not a
barrier to somebody sitting at his desk, and the page must never look like
a security boundary, because it is not one.

FOUR THINGS THIS MODULE DOES NOT DO, each for a measured reason:

* IT DOES NOT OPEN A VOICE ADMISSION WINDOW. ``app.knightfall_code`` calls
  the gate's window-opener because Knightfall's whole job is to let the
  microphone answer him for five minutes. If this page copied that, pressing
  Unlock would make Jarvis answer whoever is standing in the room -- a
  capability the terminal tool never grants for an administrative action.
  The unlock is this page's OWN state, it admits no turns, and a test greps
  this file to keep it that way.
* IT DOES NOT BUILD ITS OWN ATTEMPT COUNTER. The terminal tool constructs a
  fresh ``passphrase.Attempts`` because it is a separate PROCESS. This page
  is inside the app, so a second in-process bucket would silently turn ten
  tries per five minutes into twenty. It passes ``gate.code_attempts``.
* IT DOES NOT SET A NEW PASSPHRASE OR A NEW CODE. It says whether each one
  is set and nothing else. The terminal path reads a new secret twice
  through ``getpass``, compares, hashes and throws the plaintext away; a
  long-lived secret typed into an on-screen box on ``:1`` is a worse deal
  than walking to a terminal for it.
* IT DOES NOT CAPTURE A FACE OR A VOICE. The running Jarvis owns the camera
  and the microphone. The page hands over the exact command with the right
  label already in it, through ``jarvis/enrolentry.py``.

AND ONE IT REFUSES TO DO. If ``people.json`` is there and unreadable, the
page shows the fault and the path and offers NO create button. The terminal
tool used to happily write a fresh one-row registry over a file whose rows
failed to parse (closed in the same commit as this page); a GUI button is
far easier to press by accident than a typed subcommand.

THE SPLIT. Everything above ``UsersPage`` is Tk-free -- the lock, the
control, the rows, the two confirmations -- so it is exercised without a
display, the ``RestartArm`` / ``KnightfallControl`` pattern. The widget
half is a thin drawing of those answers.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from jarvis import consent as cs
from jarvis.identity import LABEL_RX, ROLE_KNOWN, ROLE_OWNER
from jarvis.logs import get_logger

log = get_logger("ui.users_page")

# ------------------------------------------------------------------ words
UNLOCK_FAILED = "That did not work, sir; see the log."
UNLOCK_EMPTY = "Type the override code first, sir."
UNLOCK_NOT_WIRED = "The people book is not wired to this console."
UNLOCK_PLACEHOLDER = "override code"

# The unlock window. ONE unlock then a short dwell, re-armed by each
# successful action, rather than a prompt per click: the terminal tool
# authorises ONCE per invocation and then acts, so a window is the closer
# analogue -- and asking on every press trains him to type a break-glass
# code constantly, which is one more exposure on his display each time.
UNLOCK_S = 120.0

# ------------------------------------------------------------- the voice chip
# IT SAYS WHO HAS A POOL, and it used to assert that nobody but the owner
# could. The chip it replaces -- VOICE_GUEST, "cannot name a guest", drawn on
# EVERY non-owner row on the grounds that only one pool existed and it was his
# -- stopped being true when jarvis/voicegallery.py merged: gate._voice_leg
# names guests out of it, through speaker.py. Same defect class as the stale
# foot note, one chip over, so this is derived from the snapshot for the same
# reason. The old sentence is not quoted here, because a test greps this file
# for it.
VOICE_NONE = "voice: none enrolled"
VOICE_POOL = "voice: enrolled"


def voice_chip(label, role="", voices=()) -> str:
    """What this person's voice chip says, from the pools that EXIST.

    ``voices`` is the voice gallery's label list, carried on the snapshot
    beside the face gallery's. A row whose label is in it has a pool Jarvis
    can match against; a row that is not simply has none yet -- which is a
    fact about today, not a rule about who is allowed one.
    """
    who = str(label or "")
    have = {str(v) for v in (voices or ())}
    return VOICE_POOL if who and who in have else VOICE_NONE


# The one amber sentence on a person block. It NAMES the pointer, because
# the whole failure is that a face label was renamed or deleted in the
# gallery and the registry row kept pointing at the old one -- which is
# how the face leg stops naming somebody without ever saying so.
FACE_DANGLING = ('face: "%s" is not in the face gallery, so the face leg '
                 "cannot name them; re-enrol them or point the row at the "
                 "label the gallery actually holds")

# ------------------------------------------------------ the standing note
# IT IS DERIVED, NOT WRITTEN, and that is the fix for the defect he actually
# caught on 2026-09-05. ``CANNOT_DO`` used to be a hand-edited tuple of
# sentences sitting a hundred lines from the code whose truth it asserts, and
# it went stale in three places at once: it said enrolling a face needs a
# terminal after jarvis/enrolrun.py made that false for him, it said setting a
# code needs a terminal while the drawer one tab away took a typed code in a
# masked box, and the voice chip below said a guest cannot be named after the
# voice gallery made that false too. Fixing the words would have fixed it
# once. Deriving them from the seams that are actually wired stops the class:
# a capability that lands turns its own sentence off, a capability that is
# removed turns one back on, and tests/test_users_footnote.py fails when they
# disagree.

# The two that are true whatever ships. Neither is a capability, so neither
# may ever be derived away.
NOTE_NOT_A_LOCK = ("This is recognition, not a lock: anyone already at this "
                   "keyboard can edit the people file directly.")
NOTE_NO_UNDO = ("Nothing here can be undone: the people file has no history "
                "and no backup.")

# The passphrase. It may be typed here BECAUSE it is said out loud in normal
# use -- scripts/jarvis_people.py's own text says "it can be overheard; that
# is accepted" -- so a masked box on his own console makes it no worse.
NOTE_PHRASE_NO = ("Setting a new spoken passphrase still needs a terminal — "
                  "this page only says whether one is set.")

# The code. It may NEVER be typed here, and the sentence says what the tab
# does instead. Its whole value is that it never travels a microphone and is
# never chosen: a mailed code is generated from an alphabet with no 0/o and
# no 1/l and rotates on every use.
NOTE_CODE_NO = ("Setting a new override code still needs a terminal — this "
                "page only says whether one is set.")
NOTE_CODE_MAIL = ("Setting an override code by hand still needs a terminal — "
                  "`scripts/jarvis_people.py set-code`. From here you can "
                  "only ask for a new one to be emailed to you, and the old "
                  "one keeps working until the new one has actually been "
                  "sent.")

NOTE_FACE_NO = ("Enrolling a face still needs a terminal: the running Jarvis "
                "owns the camera. The command is handed over ready to run.")
NOTE_VOICE_NO = ("Enrolling a voice still needs a terminal — "
                 "`scripts/voice_enrol.py`. Nothing here records you.")

# ALWAYS TRUE, whatever else ships. A dialog is driven by whoever is already
# logged in, so the console cannot reproduce "and nobody may type it for
# them" for a MEASUREMENT of somebody else.
NOTE_OTHERS = ("Enrolling somebody else's face or voice still needs a "
               "terminal: they have to read what is stored and type their "
               "own name themselves, and nobody may type it for them. The "
               "command is handed over ready to run.")

# THE PURGE SENTENCES CARRY THE ORDERING, which the first draft left out and
# which is the half of defect 2 that no wording change alone would have fixed:
# the gallery buttons live ON THE ROW, and Forget removes the row. So "forget
# them, then press the two buttons" is an instruction that cannot be followed,
# and the panel used to imply exactly that.
NOTE_PURGE_NO = ("Forgetting somebody removes their entry here only. Their "
                 "face measurements stay in the gallery and any voice pool "
                 "stays with them, until that command is run.")
NOTE_PURGE_FACE_ONLY = ("Forgetting somebody removes their entry here only. "
                        "Their face measurements have their own button on "
                        "their row — press it BEFORE you forget them, "
                        "because forgetting takes the row away. A voice pool "
                        "needs a terminal.")
NOTE_PURGE_VOICE_ONLY = ("Forgetting somebody removes their entry here only. "
                         "Their voice pool has its own button on their row — "
                         "press it BEFORE you forget them, because forgetting "
                         "takes the row away. Face measurements need a "
                         "terminal.")
NOTE_PURGE_BOTH = ("Forgetting somebody removes their entry here only. Their "
                   "face measurements and their voice pool each have their "
                   "own button on their row — press those BEFORE you forget "
                   "them, because forgetting takes the row away.")

# The asymmetry this work creates, said rather than smoothed over. Every
# button below is under GUARDED and asks for the override code. The SPOKEN
# door to the identical write does not: "enrol my face" out loud is gated by
# the wake word, the speaker verifier and one typed word, and never asks
# admin_gate. That is the right direction -- the stricter door is the new one
# -- but he should be told rather than left to find it.
ENROL_ASYMMETRY = (
    "Asking me out loud to enrol your face does not ask for the override "
    "code; this button does.",
    "That is deliberate. The spoken path already needs the wake word, your "
    "voice, and a word typed at this keyboard. It is not loosened to match.",
)


# The seam behind each thing this console can do. ONE mapping, so the note and
# the buttons cannot come to different conclusions about the same capability.
SEAM_FOR = {"phrase": "people_set_phrase",
            "code": "people_new_code",
            "face": "face_enrol_start",
            "voice": "voice_enrol_start",
            "purge_face": "people_purge_face",
            "purge_voice": "people_purge_voice"}

# What the owner's row draws for a capability it HAS, and what it draws
# instead when it has not. The hand-over is not a fallback: NOTE_FACE_NO and
# NOTE_VOICE_NO both promise "the command is handed over ready to run", so a
# console without the seam owes him that button rather than a bare gap.
OWNER_BUTTON = {"phrase": "Change phrase",
                "code": "Send me a new code",
                "face": "Enrol my face",
                "voice": "Enrol my voice"}
HANDOVER_BUTTON = {"face": "Copy the face command",
                   "voice": "Copy the voice command"}
STOP_BUTTON = "Stop"


def console_can(services=None) -> Dict[str, bool]:
    """WHAT THIS CONSOLE CAN ACTUALLY DO, asked once and answered once.

    THE FOOT NOTE AND THE BUTTONS READ THE SAME DICT, and that is the whole
    of defect 3. Deriving the NOTE from the seams -- which is what the first
    pass did -- only fixed one side of the sentence: the buttons were still
    drawn from nothing at all, so a console where ``face_enrol_start`` never
    landed said "Enrolling a face still needs a terminal" in the foot and
    drew a button labelled "Enrol my face" two inches above it. He caught the
    ORIGINAL version of that contradiction himself on 2026-09-05, which is
    why this lane exists, so a second stale foot note is the one outcome that
    would make this work worse than not doing it.

    CALLABLE, not merely present. The half-wired case that actually happens
    is ``build_ui_services`` dropping a name the dataclass has not declared
    yet, or a field landing as None -- both of which read as "there" to a
    bare getattr and as a broken button to him.
    """
    out = {name: bool(services is not None
                      and callable(getattr(services, seam, None)))
           for name, seam in SEAM_FOR.items()}
    # ONE Stop for both runs, so it survives either seam landing alone.
    out["stop"] = bool(services is not None and (
        callable(getattr(services, "face_enrol_stop", None))
        or callable(getattr(services, "voice_enrol_stop", None))))
    return out


def owner_buttons(can: Dict[str, bool]) -> Tuple[str, ...]:
    """The labels the OWNER'S row will draw, in order, for a given set of
    capabilities. Read by ``_build_actions`` and by the tests, so the
    agreement between the note and the buttons is structural rather than
    remembered."""
    out = []
    for cap in ("phrase", "code"):
        if can.get(cap):
            out.append(OWNER_BUTTON[cap])
    for cap in ("face", "voice"):
        out.append(OWNER_BUTTON[cap] if can.get(cap) else HANDOVER_BUTTON[cap])
    if can.get("stop"):
        out.append(STOP_BUTTON)
    return tuple(out)


def cannot_do(services=None) -> Tuple[str, ...]:
    """The pinned foot note, BUILT FROM THE SEAMS THAT ARE WIRED.

    Every sentence is a fact about this build, asked of the services
    namespace the app actually handed the console -- so a half-wired UI (the
    app and the window merged in either order) describes itself correctly
    instead of promising a button that is not there.
    """
    can = console_can(services)
    out = [NOTE_NOT_A_LOCK]
    if not can["phrase"]:
        out.append(NOTE_PHRASE_NO)
    out.append(NOTE_CODE_MAIL if can["code"] else NOTE_CODE_NO)
    if not can["face"]:
        out.append(NOTE_FACE_NO)
    if not can["voice"]:
        out.append(NOTE_VOICE_NO)
    out.append(NOTE_OTHERS)
    face, voice = can["purge_face"], can["purge_voice"]
    out.append(NOTE_PURGE_BOTH if (face and voice) else
               NOTE_PURGE_FACE_ONLY if face else
               NOTE_PURGE_VOICE_ONLY if voice else NOTE_PURGE_NO)
    out.append(NOTE_NO_UNDO)
    return tuple(out)


def foot_lines(services=None) -> Tuple[str, ...]:
    """EVERY line of the pinned foot, including the asymmetry pair.

    Those two say "this button does ask for the override code", which is a
    lie on a console that draws no such button -- so they are derived from
    the same dict as the button, not appended by the painter.
    """
    out = list(cannot_do(services))
    if console_can(services)["face"]:
        out += list(ENROL_ASYMMETRY)
    return tuple(out)


# EVERY DOOR INTO THE PEOPLE BOOK OR A GALLERY IS IN HERE. "set_face" was in
# this tuple and was DEAD: the string appeared in its own definition and
# nowhere else in the tree -- no action, no button, no seam, and nothing ever
# called may("set_face"). It is gone, and the six real doors this work opens
# are in its place. Starting a camera or a microphone is not a registry write,
# but each writes to something at least as irreversible as the people book, so
# none of them gets an easier question than `forget` does.
GUARDED = ("add", "set_role", "forget", "set_phrase", "new_code",
           "face_enrol", "voice_enrol", "purge_face", "purge_voice")

LOCKED_LINE = ("Unlock with your override code before changing anything, "
               "sir.")
BROKEN_LINE = ("The people file could not be read as a registry, so nothing "
               "may be changed from here. Repair or move the file first.")


# ============================================================== the unlock
class Lock:
    """WHAT THE PAGE DRAWS, and it is a MIRROR rather than a guard.

    The authority is ``app._people_write``, which re-reads the registry and
    re-decides at the moment of the write. This object used to be the only
    thing enforcing the override code anywhere in the system, which is not
    a guard at all: it lived in the UI, it was consulted against a snapshot
    taken when the tab was OPENED, and nothing below it re-decided. It is
    now seeded from the app on every tick (``mirror``) so the countdown on
    screen is the app's countdown and not a second one beside it.

    It admits no turn, opens no microphone and grants no role. It is a
    dwell after a successful code, re-armed by each successful action so a
    run of edits is one code rather than five, and dropped the moment the
    tab is left.
    """

    def __init__(self, window_s: float = UNLOCK_S,
                 clock: Callable[[], float] = time.monotonic):
        self.window_s = float(window_s)
        self.clock = clock
        self._until = 0.0

    def unlock(self) -> None:
        self._until = self.clock() + self.window_s

    def touch(self) -> None:
        """A successful action re-arms the dwell."""
        if not self.locked():
            self._until = self.clock() + self.window_s

    def lock(self) -> None:
        self._until = 0.0

    def mirror(self, seconds) -> None:
        """Take the APP's remaining dwell as the truth. Called once a
        second from the tick, so the two can differ by at most a tick and
        the app is always the one that wins."""
        try:
            left = float(seconds)
        except (TypeError, ValueError):
            left = 0.0
        self._until = self.clock() + max(0.0, left)

    def locked(self) -> bool:
        return self.clock() >= self._until

    def remaining(self) -> float:
        return max(0.0, self._until - self.clock())


def check_code(gate, code, *, check: Optional[Callable] = None) -> Tuple[str, str]:
    """``(who, why)`` from the gate's own break-glass check.

    The counter is ``gate.code_attempts``, the object the gate already
    holds. A fresh one here would double the effective budget, and the
    doubling would be invisible.
    """
    if check is None:
        from jarvis import gate as gate_mod
        check = gate_mod.check_override_code
    attempts = getattr(gate, "code_attempts", None)
    return check(getattr(gate, "registry", None), code, attempts=attempts)


def may(action: str, state: str, lock: Lock) -> Tuple[bool, str]:
    """Should the page bother asking? ``(ok, why)``.

    THIS IS NOT THE GUARD, and saying so here is the point of the sentence.
    It used to be: the code requirement was enforced in this one function,
    from a snapshot the page read when the tab was opened, and the app
    seams below it re-read the file only to refuse the CORRUPT case. So a
    code set at a terminal while the tab sat open was never noticed and
    every administrative action went through ungated.

    What it is now is a courtesy: a toast that names the remedy instead of
    a press that travels to the app to be refused there. ``state`` is
    ``gate.admin_gate``'s answer, kept live by the page's tick, and
    ``app._people_write`` re-decides the same question from the registry as
    it is at the instant of the write.
    """
    from jarvis import gate as gate_mod
    if state == gate_mod.ADMIN_REFUSE:
        return False, BROKEN_LINE
    if str(action) not in GUARDED:
        return True, ""
    if state != gate_mod.ADMIN_CODE:
        # No owner yet (the bootstrap), or an owner who has set no code.
        # Demanding a code he never chose would lock him out of the flow
        # that sets one.
        return True, ""
    if lock is not None and lock.locked():
        return False, LOCKED_LINE
    return True, ""


class UsersUnlockControl:
    """The unlock row's logic, WITHOUT Tk -- ``KnightfallControl``'s split.

    THE ENTRY IS CLEARED BEFORE THE SERVICE IS CALLED, not after it
    returns. ``check_override_code`` is a scrypt KDF at N=2^14 looped over
    every owner holding a code, so it is real work: a typed code must not
    sit on screen while it runs, and it must not still be there if the tab
    is left mid-call.

    Five seams, all injected: ``read``/``clear`` are the entry, ``toast``
    shows a line, ``later`` marshals back onto the Tk thread and ``spawn``
    runs the call off it. The thread is named "users-unlock" -- never
    anything derived from what was typed.
    """

    def __init__(self, services, *, read: Callable[[], str],
                 clear: Callable[[], None], toast: Callable,
                 later: Callable[[Callable], None],
                 spawn: Optional[Callable] = None,
                 on_unlocked: Optional[Callable] = None):
        self.services = services
        self.read = read
        self.clear = clear
        self.toast = toast
        self.later = later
        self.spawn = spawn or self._thread
        self.on_unlocked = on_unlocked

    @staticmethod
    def _thread(fn, *args):
        threading.Thread(target=fn, args=args, daemon=True,
                         name="users-unlock").start()

    def _service(self, name):
        return getattr(self.services, name, None) if self.services else None

    def unlock_pressed(self) -> str:
        """Read the code, clear the entry AT ONCE, hand it over off this
        thread. "started" / "empty" / "not wired"."""
        code = self.read()
        self.clear()
        if not str(code or "").strip():
            del code
            self.toast(UNLOCK_EMPTY, "warn")
            return "empty"
        fn = self._service("people_unlock")
        if fn is None:
            del code
            log.warning("users: no people service is wired to this console")
            self.toast(UNLOCK_NOT_WIRED, "warn")
            return "not wired"
        self.spawn(self._run, fn, code)
        del code
        return "started"

    def _run(self, fn, code):
        try:
            ok, line = fn(code)
        except Exception as exc:  # noqa: BLE001 - the service's own boundary
            # NOT log.exception, and this is MEASURED rather than careful:
            # a traceback carries the exception's own str(), and the
            # argument that call was handed is a CODE. A service raising
            # RuntimeError("the code %s was rejected") puts the plaintext
            # into the log file at ERROR, where it stays. Only the
            # exception's TYPE NAME is logged; the type of a failure is
            # what a log is for here, and it can quote nothing.
            log.error("users: the unlock service failed (%s)",
                      type(exc).__name__)
            ok, line = False, UNLOCK_FAILED
        finally:
            del code
        line = str(line or UNLOCK_FAILED)
        self.later(lambda: self._landed(bool(ok), line))

    def _landed(self, ok: bool, line: str) -> None:
        if ok and callable(self.on_unlocked):
            try:
                self.on_unlocked()
            except Exception:  # noqa: BLE001 - a callback
                log.exception("users: the unlock callback failed")
        self.toast(line, "ok" if ok else "warn")



# =============================================== the spoken passphrase
PHRASE_MISMATCH = "Those did not match, sir; nothing was changed."
PHRASE_FAILED = "That did not work, sir; see the log."
PHRASE_NOT_WIRED = "Setting a passphrase is not wired to this console."

# WHY THIS ONE MAY BE TYPED ON SCREEN. The page's own header used to argue
# the opposite for both secrets at once -- "a long-lived secret typed into an
# on-screen box on :1 is a worse deal than walking to a terminal for it" --
# and that argument is still exactly right about the OVERRIDE CODE, which is
# why there is no box for one. It does not transfer to the spoken passphrase,
# and the difference is in the threat model rather than in the convenience:
# the passphrase is SAID OUT LOUD in normal use, and scripts/jarvis_people.py
# says so in its own text -- it can be overheard, and that is accepted. A
# secret already accepted as overhearable is not made materially worse by a
# masked box on his own console. The code exists precisely because it never
# touches a microphone.
PHRASE_PANEL = (
    "The spoken way back in, for when the camera is off and your voice will "
    "not match — ill, in the dark, or turned away.",
    "It is said out loud, so it can be overheard. That is accepted.",
    "At least %d letters and digits once punctuation is dropped." %
    __import__("jarvis.passphrase", fromlist=["x"]).MIN_PHRASE_LEN,
    "Nothing here can read it back, and it is never spoken aloud: a "
    "passphrase read back by the assistant is a passphrase in the room.",
)


def phrase_panel_lines() -> Tuple[str, ...]:
    return PHRASE_PANEL


class UsersSecretControl:
    """The passphrase panel's logic, WITHOUT Tk -- ``UsersUnlockControl``'s
    split, and its rules, for the same measured reasons.

    BOTH BOXES ARE EMPTIED BEFORE ANYTHING ELSE HAPPENS, and before the hash
    is computed rather than after the service returns. ``hash_secret`` is a
    scrypt KDF at N=2^14 -- real work -- and a typed secret must not sit on
    screen while it runs, nor still be there if the tab is left mid-call.

    THE HASHING HAPPENS IN THIS FRAME and the plaintext dies here.
    ``identity.Registry`` is documented as never seeing a plaintext, and this
    is the caller that keeps that true from the console: the service is handed
    a hash, so a service that raised with its argument in the message could
    not put a passphrase in a log file even if it tried.

    THE THREAD IS NAMED FOR THE FIELD, never for anything derived from the
    value, and every ``except`` logs the exception's TYPE only -- a traceback
    carries the exception's own str(), which is how a plaintext reaches a
    logfile at ERROR and stays there.
    """

    def __init__(self, services, *, label: str,
                 read: Callable[[], Tuple[str, str]],
                 clear: Callable[[], None], toast: Callable,
                 later: Callable[[Callable], None],
                 spawn: Optional[Callable] = None,
                 on_done: Optional[Callable] = None):
        self.services = services
        self.label = str(label or "")
        self.read = read
        self.clear = clear
        self.toast = toast
        self.later = later
        self.spawn = spawn or self._thread
        self.on_done = on_done

    @staticmethod
    def _thread(fn, *args):
        threading.Thread(target=fn, args=args, daemon=True,
                         name="users-phrase").start()

    def pressed(self) -> str:
        """Read both boxes, EMPTY THEM AT ONCE, then decide.

        "started" / "mismatch" / "too short" / "empty" / "not wired".
        """
        from jarvis import passphrase as pp
        try:
            first, second = self.read()
        except Exception:                 # noqa: BLE001 - torn down
            self.clear()
            log.error("users: the phrase entries could not be read")
            self.toast(PHRASE_FAILED, "warn")
            return "not wired"
        self.clear()
        if not str(first or "").strip() and not str(second or "").strip():
            del first, second
            self.toast("Type the new passphrase twice, sir.", "warn")
            return "empty"
        if first != second:
            del first, second
            self.toast(PHRASE_MISMATCH, "warn")
            return "mismatch"
        ok, why = pp.phrase_ok(first)
        if not ok:
            del first, second
            self.toast(why, "warn")
            return "too short"
        fn = getattr(self.services, "people_set_phrase", None) \
            if self.services else None
        if not callable(fn):
            del first, second
            log.warning("users: no passphrase service is wired to this "
                        "console")
            self.toast(PHRASE_NOT_WIRED, "warn")
            return "not wired"
        # HASHED HERE, so the plaintext never crosses the seam and never
        # leaves this frame.
        hashed = pp.hash_secret(first)
        del first, second
        self.spawn(self._run, fn, hashed)
        del hashed
        return "started"

    def _run(self, fn, hashed):
        try:
            ok, line = fn(self.label, hashed)
        except Exception as exc:          # noqa: BLE001 - the service boundary
            # NOT log.exception, and it is the same measured reason the
            # unlock control gives: a traceback carries the exception's own
            # str(). Only the TYPE is recorded, and a type can quote nothing.
            log.error("users: the passphrase service failed (%s)",
                      type(exc).__name__)
            ok, line = False, PHRASE_FAILED
        finally:
            del hashed
        line = str(line or PHRASE_FAILED)
        self.later(lambda: self._landed(bool(ok), line))

    def _landed(self, ok: bool, line: str) -> None:
        if ok and callable(self.on_done):
            try:
                self.on_done()
            except Exception:             # noqa: BLE001 - a callback
                log.exception("users: the passphrase callback failed")
        self.toast(line, "ok" if ok else "warn")


# ============================================== asking for a new override code
@dataclass(frozen=True)
class CodePlan:
    """Whether the button may be pressed, and the caption under it."""
    enabled: bool
    line: str


CODE_NO_MAILBOX = ("No mail account is set up, so there is nowhere to send a "
                   "code and this button would do nothing. %s")
CODE_TO = ("Typed only, never spoken. The next code goes to %s, and the one "
           "you have now keeps working until that mail has actually gone.")
CODE_BAD_TO = ("Your configured notice address is not an address, so nothing "
               "can be sent until that is fixed.")

# THE TRAP, SAID BEFORE THE PRESS. gate._mode_unsafe downgrades enforce to
# SHADOW while no owner row carries a code, because a wrong verdict would
# otherwise have no way back in. The moment a code exists that downgrade
# lifts -- so on a box configured owner.mode=enforce but running in shadow for
# want of a code, ONE PRESS OF THIS BUTTON starts refusing turns. It is not
# closed by refusing, because a box with no code is the state that most needs
# one; it is stated here and logged when it happens.
CODE_TURNS_GATE_ON = (
    "This will also switch the owner gate from shadow to enforce. It is "
    "running in shadow only because no override code is set, and setting one "
    "lifts that. Turns will start being refused when I am not sure it is you.")

CODE_NO_FREE_TEXT = (
    "There is no box to choose one. A code you type is one you chose, will "
    "reuse and will type again; a mailed one is generated from an alphabet "
    "with no 0/o and no 1/l, is yours only until you next use it, and rotates "
    "every time. Choosing one by hand is `scripts/jarvis_people.py set-code`.")


def code_plan(status=None, *, mode: str = "", has_code: bool = True) -> CodePlan:
    """May "Send me a new code" be pressed, and what does the caption say?

    THE CAPTION IS THE DRAWER'S OWN, reused rather than written a second
    time: ``app.knightfall_status`` already answers a masked destination, a
    problem sentence and a setup line, and the drawer's Knightfall row is
    drawn from exactly those. A promise made here that the drawer would not
    make is a promise one of them is going to break.
    """
    st = dict(status or {})
    problem = str(st.get("problem") or "")
    if problem:
        return CodePlan(False, CODE_BAD_TO)
    to = str(st.get("to") or "")
    if not to:
        return CodePlan(False, CODE_NO_MAILBOX % (st.get("setup") or ""))
    return CodePlan(True, CODE_TO % to)


def code_panel_lines(status=None, *, mode: str = "",
                     has_code: bool = True) -> Tuple[str, ...]:
    """Everything he should read BEFORE pressing, in order."""
    out = [code_plan(status, mode=mode, has_code=has_code).line]
    if str(mode) == "enforce" and not has_code:
        out.append(CODE_TURNS_GATE_ON)
    out.append(CODE_NO_FREE_TEXT)
    return tuple(out)


# ========================================================== the bootstrap
@dataclass(frozen=True)
class BootstrapPlan:
    """What the page may offer when the registry is not usable."""
    may_create: bool = False
    first_owner_only: bool = False
    line: str = ""


BOOTSTRAP_MISSING = (
    "Nobody is enrolled yet, so the gate is OFF and anyone at this keyboard "
    "can create the first owner. That is deliberate: otherwise a fresh "
    "install could never be set up.")
BOOTSTRAP_OWNERLESS = (
    "This people file has rows but names no owner, so there is nobody who "
    "could enrol anyone. Making one owner turns the gate back on; the rows "
    "already in the file are kept.")
BOOTSTRAP_BROKEN = (
    "%s exists and could not be read as a registry. Its rows did NOT load, "
    "so creating somebody here would write a new one-person file over "
    "whatever it holds — and there is no history and no backup. Repair or "
    "move the file, then reopen this tab.")


def bootstrap_plan(fault_kind: str, path: str = "") -> BootstrapPlan:
    """What to offer for each reason a registry is unusable.

    The distinction is the whole point: for a MISSING file, creating the
    first owner is the only way out of a brick. For a file that failed to
    parse it is a data shredder.
    """
    kind = str(fault_kind or "")
    if not kind:
        return BootstrapPlan()
    if kind == "missing":
        return BootstrapPlan(True, True, BOOTSTRAP_MISSING)
    if kind == "ownerless":
        return BootstrapPlan(True, True, BOOTSTRAP_OWNERLESS)
    return BootstrapPlan(False, False, BOOTSTRAP_BROKEN % (path or "the "
                                                           "people file"))


# =============================================================== the rows
@dataclass(frozen=True)
class Row:
    """One person, ready to draw. Built from ``Person.redacted()`` and
    NEVER from a ``Person``: redacted() replaces both hashes with a bare
    yes/no, so a scrypt hash cannot reach a widget even by accident."""

    label: str
    name: str
    role: str
    display: str
    voice_text: str
    face: str
    face_dim: int
    face_known: bool
    face_tone: str
    has_phrase: bool
    has_code: bool
    phrase_text: str
    code_text: str
    consent: str
    enrolled_at: str
    detail: str
    can_forget: bool
    forget_why: str
    can_change_role: bool
    role_why: str
    # WHY the face chip is amber, in a line of its own. A tint cannot say
    # what is wrong, and tinting the whole chips list said it about five
    # things that were fine. "" whenever there is nothing to explain.
    face_why: str = ""


SOLE_OWNER_FORGET = ("%s is the only owner; forgetting them would leave "
                     "nobody who can enrol anyone")
SOLE_OWNER_ROLE = ("%s is the only owner; demoting them would leave nobody "
                   "who can enrol anyone")


def rows_from(snapshot) -> Tuple[Row, ...]:
    """Every person in the snapshot, as drawable answers.

    A snapshot that cannot be read makes NO rows rather than raising: an
    empty page says "nobody is enrolled", which is true and harmless, and
    a traceback out of a repaint takes the console with it.
    """
    try:
        people = list((snapshot or {}).get("people") or ())
        gallery = set((snapshot or {}).get("gallery") or ())
        # The VOICE gallery's labels, beside the face gallery's. Without them
        # the tab cannot say who has a voice pool, which is what made the
        # stale voice chip invisible from inside the app.
        voices = set((snapshot or {}).get("voices") or ())
    except Exception:  # noqa: BLE001 - a snapshot that cannot answer
        log.exception("users: the snapshot could not be read")
        return ()
    owners = [p for p in people if _get(p, "role") == ROLE_OWNER]
    sole = owners[0].get("label") if len(owners) == 1 else None
    out = []
    for person in people:
        if not isinstance(person, dict):
            continue
        out.append(_row(person, gallery, sole, voices))
    return tuple(out)


def _get(person, key, default=""):
    try:
        return person.get(key, default)
    except Exception:  # noqa: BLE001
        return default


def _row(person: dict, gallery, sole, voices=()) -> Row:
    label = str(_get(person, "label"))
    name = str(_get(person, "name"))
    role = str(_get(person, "role")) or ROLE_KNOWN
    face = str(_get(person, "face"))
    try:
        dim = int(_get(person, "face_dim", 0) or 0)
    except (TypeError, ValueError):
        dim = 0
    known = bool(face) and face in gallery
    face_why = ""
    if not face:
        # no pointer at all is a CHOICE, not a dangling one
        tone = "muted"
    elif known:
        tone = "ok"
    else:
        # A pointer at a gallery label that is not there is exactly how the
        # face leg silently stops naming anyone -- so say so, rather than
        # tinting a list of chips that are all perfectly correct.
        tone = "warn"
        face_why = (FACE_DANGLING % face)
    voice_text = voice_chip(label, role=role, voices=voices)
    has_phrase = bool(_get(person, "has_phrase", False))
    has_code = bool(_get(person, "has_code", False))
    phrase_text = "phrase: set" if has_phrase else "phrase: not set"
    code_text = "code: set" if has_code else "code: not set"
    face_text = ("face: %s%s" % (face, " (%d-D)" % dim if dim else "")
                 if face else "face: none")
    consent = str(_get(person, "consent"))
    enrolled = str(_get(person, "enrolled_at"))
    bits = [voice_text, face_text, phrase_text, code_text]
    if consent:
        bits.append("consent: %s" % consent)
    if enrolled:
        bits.append("since %s" % enrolled.split("T")[0])
    is_sole = sole is not None and label == sole
    return Row(label=label, name=name, role=role,
               display=name or label.capitalize(), voice_text=voice_text,
               face=face, face_dim=dim, face_known=known, face_tone=tone,
               has_phrase=has_phrase, has_code=has_code,
               phrase_text=phrase_text, code_text=code_text,
               consent=consent, enrolled_at=enrolled,
               detail="  ·  ".join(bits),
               face_why=face_why,
               can_forget=not is_sole,
               forget_why=SOLE_OWNER_FORGET % label if is_sole else "",
               can_change_role=not is_sole,
               role_why=SOLE_OWNER_ROLE % label if is_sole else "")


# ========================================================== forgetting
FORGET_ARM_S = 20.0


class ForgetArm:
    """Two gestures, not one. A press ARMS and shows what is destroyed;
    the confirmation is the person's own label, TYPED.

    Not a yes/no, and the reason is the same one consent uses: typing the
    label names WHO is about to be destroyed in a way "Are you sure?" does
    not, and a mis-click on a list row cannot produce it.
    """

    def __init__(self, window_s: float = FORGET_ARM_S,
                 clock: Callable[[], float] = time.monotonic):
        self.window_s = float(window_s)
        self.clock = clock
        self._label = ""
        self._at = 0.0

    @property
    def armed_for(self) -> str:
        if self._label and (self.clock() - self._at) < self.window_s:
            return self._label
        return ""

    def press(self, label: str) -> str:
        self._label = str(label or "")
        self._at = self.clock()
        return "armed"

    def disarm(self) -> None:
        self._label = ""

    def confirm(self, label: str, typed) -> str:
        """"forget" / "refused" / "expired". A WRONG ANSWER DOES NOT
        DISARM: he mistyped, and making him press again would teach him to
        press twice quickly, which is the habit this guard exists to
        break."""
        if self.armed_for != str(label or ""):
            return "expired"
        if not cs.agreed(label, typed):
            return "refused"
        self.disarm()
        return "forget"


def forget_warning(label: str, voices=(), gallery=(),
                   can: Optional[Dict[str, bool]] = None) -> Tuple[str, ...]:
    """What is destroyed, what SURVIVES, HOW TO REMOVE IT, and in what order.

    MEASURED in ``identity.Registry.forget``: it removes the ROW and nothing
    else. Saying "removed" and leaving the rest implied is how somebody comes
    to believe a gallery was scrubbed when it was not.

    IT USED TO SAY SOMETHING FALSE, and in the most damaging place to say it.
    The line it carried told him there was no voice recording of that person
    to remove, on the grounds that only the owner had a pool. That stopped
    being true when the voice gallery merged, and it was printed on a
    DESTRUCTIVE panel -- so he could read "there is nothing of hers to remove"
    while her pool sat on the disk. The sentence is now built from what the
    two galleries actually hold. The old wording is not quoted here: a test
    greps this file for it.

    THEN IT PROMISED BUTTONS IT DID NOT DRAW -- defect 2, and two separate
    faults in one sentence. It said "each is a separate button below"
    whatever the console had actually been wired with, so a half-wired page
    sent him looking for a control that was not on it; and it never said that
    those buttons live ON THIS ROW and that Forget removes the row, so the
    only order the panel implied -- forget them, then press the two buttons --
    is one that cannot be followed.

    SO THE PANEL DOES NOT PURGE, AND THAT IS A DECISION RATHER THAN AN
    OMISSION. Three reasons, and the third is the one that settles it:

    * ONE TYPED CONFIRMATION MEANS ONE THING. He is asked to type a label to
      remove a ROW. Making that same keystroke destroy two biometric stores
      changes what he is agreeing to without changing what he was asked.
    * ``purge_label`` CAN HONESTLY SUCCEED IN PART. Its own contract is that a
      generation it cannot read, that another model wrote, or that holds a
      bystander it could not carry is LEFT ALONE and reported. A combined
      action would have to report that in the same breath as a registry write
      that either happened or did not -- and rounding a partial purge up to
      "done" is precisely the class of defect this lane exists to fix.
    * THERE IS NOTHING TO ROLL BACK TO. The people file has no history and no
      backup (this panel says so). If the purge failed after the row was
      gone, he would have lost the row, kept the measurements, and lost the
      button that removes them. Ordering it the other way -- purge first,
      forget after -- is the safe sequence, so the panel says that instead of
      hiding it inside one press.

    The commands are printed WHATEVER is wired, because after the row is gone
    the button is gone with it and the command is the only thing that still
    works.
    """
    who = str(label or "")
    have_face = who in {str(g) for g in (gallery or ())}
    have_voice = who in {str(v) for v in (voices or ())}
    can = dict(can or {})
    btn_face = have_face and bool(can.get("purge_face"))
    btn_voice = have_voice and bool(can.get("purge_voice"))
    if have_face and have_voice:
        survives = ("%s's face measurements AND %s's voice pool stay on the "
                    "disk." % (who, who))
    elif have_face:
        survives = ("%s's face measurements stay in the gallery. There is no "
                    "voice pool under that name." % who)
    elif have_voice:
        survives = ("%s's voice pool stays in the gallery. There are no face "
                    "measurements under that name." % who)
    else:
        survives = ("There are no face measurements and no voice pool under "
                    "that name, so this row is all there is of %s." % who)
    out = [
        "Forget %s?" % who,
        "",
        "This removes %s's entry: their name, their role, the face label "
        "they point at, their consent record and the date they were "
        "added." % who,
        "",
        "WHAT SURVIVES. " + survives,
    ]
    # THE ORDER, said only when there is actually a button to lose.
    if btn_face or btn_voice:
        which = ("the buttons Remove face measurements and Remove voice pool "
                 "are" if (btn_face and btn_voice) else
                 "the button Remove face measurements is" if btn_face else
                 "the button Remove voice pool is")
        out += ["",
                "IN THAT ORDER. On this row %s, and forgetting %s takes the "
                "row away with them -- so press them BEFORE you confirm here, "
                "not after." % (which, who)]
    if have_face or have_voice:
        out += ["", "Once the row is gone these are the only way left:"]
        if have_face:
            out.append(forget_face_command(who))
        if have_voice:
            out.append(forget_voice_command(who))
    out += [
        "",
        "THIS CANNOT BE UNDONE. The people file is rewritten in place; "
        "there is no history and no backup.",
        "",
        "Type %s to confirm." % who,
    ]
    return tuple(out)


PURGE_WARN_FACE = (
    "Remove %(who)s's FACE measurements?",
    "",
    "This destroys every generation on the disk that holds %(who)s -- "
    "including the older ones -- overwritten and then unlinked. Everybody "
    "else is carried forward into a new generation first, and nothing is "
    "destroyed until they have been READ BACK off the disk by name and by "
    "sample count.",
    "",
    "If any generation cannot be read, was written by another model, or "
    "holds somebody who could not be carried, it is LEFT ALONE and you are "
    "told -- %(who)s is then still in it and nothing here will claim "
    "otherwise.",
    "",
    "This does not touch %(who)s's voice pool. That is the button beside it.",
    "",
    "THIS CANNOT BE UNDONE. There is no history and no backup.",
    "",
    "Type %(who)s to confirm.",
)

PURGE_WARN_VOICE = (
    "Remove %(who)s's VOICE pool?",
    "",
    "This destroys every generation on the disk that holds %(who)s's voice "
    "embeddings -- 192 numbers per take -- overwritten and then unlinked. "
    "Everybody else is carried into a new generation and read back off the "
    "disk first; anything that cannot be finished honestly is left alone and "
    "named.",
    "",
    "No recording is destroyed because none was ever kept: the audio became "
    "those numbers and was thrown away at the microphone.",
    "",
    "This does not touch %(who)s's face measurements. That is the button "
    "beside it.",
    "",
    "THIS CANNOT BE UNDONE. There is no history and no backup.",
    "",
    "Type %(who)s to confirm.",
)


def purge_warning(label: str, kind: str = "face") -> Tuple[str, ...]:
    """What a gallery purge destroys, what it does NOT, and that it may
    refuse. The confirmation is the person's own label, typed."""
    who = str(label or "")
    text = PURGE_WARN_VOICE if kind == "voice" else PURGE_WARN_FACE
    return tuple(line % {"who": who} for line in text)


def enrol_command(label: str, kind: str = "face", name: str = "") -> str:
    """The hand-over for somebody ELSE, through the seams that already build
    it -- never a second string to drift.

    IMPORT-LIGHT BY CONSTRUCTION. ``enrolentry.command_line`` and its voice
    twin reach no gallery, no model and no device; importing this module used
    to pull the whole vision stack in and the Users page is what found that,
    so the property is tested rather than remembered.
    """
    from jarvis import enrolentry
    who = str(label or "")
    if kind == "voice":
        return enrolentry.voice_command_line(who, name=name)
    return enrolentry.command_line(who)


def forget_face_command(label: str, *, python: Optional[str] = None,
                        script: Optional[str] = None) -> str:
    """The exact command that removes their face measurements, quoted, via
    the seam that already builds it -- not a second string to drift."""
    from jarvis import enrolentry
    return enrolentry.command_line(str(label or ""), delete=True,
                                   python=python, script=script)


def forget_voice_command(label: str, *, python: Optional[str] = None,
                         script: Optional[str] = None) -> str:
    """The exact command that removes their VOICE pool, quoted.

    THE FACE HALF ALREADY HAD ONE AND THE VOICE HALF DID NOT, which is how a
    destructive panel came to list what survives and hand over a way to
    remove only some of it. Built by the same seam ``scripts/voice_enrol.py``
    documents -- there is no second way to delete somebody.
    """
    from jarvis import enrolentry
    return enrolentry.voice_command_line(str(label or ""), delete=True,
                                         python=python, script=script)


# ============================================================ adding a row
def take_consent(label: str, *, owner: str = "", show: Optional[Callable] = None,
                 ask: Optional[Callable] = None,
                 mapped: Optional[Callable] = None) -> Tuple[bool, str]:
    """Somebody else's agreement, on the console, in the SHARED words.

    The provenance it records is ``console``, never ``typed``: the terminal
    ceremony did not happen and the record must not say it did.
    """
    return cs.take_at_console(label, what=cs.WHAT_ROW,
                              fields={"who": str(label or "")},
                              show=show, ask=ask, mapped=mapped, owner=owner)


def owner_confirmed(role: str, existing_owners: Sequence[str],
                    typed) -> Tuple[bool, str]:
    """Making a SECOND owner has to name the first, typed.

    ``identity.add_person``'s ``confirm_existing_owner`` exists precisely so
    a slip of the hand cannot mint one; a pre-filled dropdown would defeat
    it entirely, so the page asks for the label.
    """
    owners = [str(o) for o in (existing_owners or ())]
    if str(role) != ROLE_OWNER or not owners:
        return True, ""
    got = str(typed or "").strip().lower()
    if got in [o.lower() for o in owners]:
        return True, ""
    return False, ("there is already an owner (%s); making a second one has "
                   "to name the first" % ", ".join(owners))


def label_fault(label) -> str:
    """"" when the gallery AND the registry could both store it.

    ``identity.LABEL_RX`` is the gallery's own pattern: a label one store
    would refuse must not reach the other, or the two disagree about who
    exists and the face leg silently stops naming anyone.
    """
    text = str(label or "")
    if not text:
        return "a person needs a label"
    if not LABEL_RX.match(text):
        return ("%r is not a label a gallery or a registry can store: "
                "lower-case letters, digits, - and _, starting with a "
                "letter or a digit, at most 31 characters" % text)
    return ""


# ============================================================ the Tk surface
import tkinter as tk                                   # noqa: E402
from tkinter import font as tkfont                     # noqa: E402


# A Tk Label's requested width is its text plus its own border and
# highlight ring even when padx and bd are 0 -- MEASURED at 14 px for this
# label at 2.0 scale (font.measure said 510, winfo_reqwidth said 524). Fit
# to the text budget, not to the allocation, or the label asks for slightly
# more than it was given and Tk cuts the difference off the anchored end.
_INK_SLACK = 16

from jarvis.ui import theme                            # noqa: E402
from jarvis.ui.widgets import RoundButton, px, ui_display, ui_font  # noqa: E402

# Design units, at the 96-dpi baseline; px() scales them by S. Every COLOUR
# below is read inside a method, never bound here: tests/test_theme_look.py
# globs jarvis/ui/*.py and a token captured at def time freezes the
# import-time look.
BLOCK_GAP = 10
ROLE_WORDS = {ROLE_OWNER: "OWNER", ROLE_KNOWN: "KNOWN"}
LOCK_TICK_MS = 1000


class UsersPage(tk.Frame):
    """The USERS surface, placed over the console's stage.

    WHERE THE TAB IS: the strip under the wordmark (jarvis/ui/tab_strip.py),
    added with one line in ``main_window._fill_tabs``. This page draws no
    tabs of its own -- two rows of tabs on one screen is a question about
    which of them is in charge.

    WHY IT COVERS THE REACTOR TOO, the same reason SensorsPage does: it is
    a list that grows, and over the transcript alone the foot falls off the
    bottom. ``cover`` is the widgets whose union it should span.

    THE SCROLLING BODY AND THE PINNED FOOT ARE THERE FROM THE FIRST BUILD.
    That is not caution: on 2026-09-05 Hunter photographed the sensors page
    with its last band row and its SAVE button off the bottom of the stage,
    and this page's height is a function of how many people he enrols. The
    "Add a person" button and the standing note never scroll; MEASURED at
    both 1040x1760 and 920x1440 in both looks, with twelve people on the
    page (tests/test_users_page_display.py).

    IT POLLS ONE THING, and only since the lock stopped being a memory.
    ``services.people_admin_state`` re-reads the registry and answers three
    small values; the tick that was already redrawing the countdown asks
    for them. The reason is his own sequence: make the first owner here
    (no code yet, so the tab is legitimately open), read the foot note
    saying a new code still needs a terminal, go and set one, come back to
    the SAME open tab. ``show()`` was the only thing that re-read, so
    switching tabs and back was the only way to notice. The full snapshot
    is too heavy for a one-second tick -- it rebuilds every row, the
    startup line and the gallery listing -- so the page asks for the cheap
    answer and does a full ``refresh`` only when it CHANGED.

    NONE OF THAT IS THE GUARD. ``app._people_write`` re-reads and
    re-decides at the write; this only keeps what is on screen true.
    """

    def __init__(self, host, services=None, cover=(),
                 on_close: Optional[Callable] = None, toast=None,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(host, bg=theme.TV_BG)
        self.host = host
        self.cover = tuple(w for w in (cover or ()) if w is not None)
        self.services = services
        self._on_close = on_close
        self._toast_bar = toast
        self._toasts: list = []           # every line this page showed
        self._open = False
        self._snapshot: dict = {}
        self._rows: Tuple[Row, ...] = ()
        self._row_widgets: dict = {}
        # Every label built into the scrolling body, so a re-wrap reaches
        # the ones a repaint made rather than only the three fixed ones.
        self._wrapped: list = []
        self._lock = Lock(clock=clock)
        self._forget = ForgetArm(clock=clock)
        # A SECOND ARM, not a shared one. Forgetting a ROW and destroying a
        # GALLERY are different destructions with different warnings, and one
        # arm would let a press on either arm the other.
        self._purge = ForgetArm(clock=clock)
        self._purge_kind = ""
        # ("phrase" | "code" | "", label) -- which non-destructive panel is
        # open, and on whose row.
        self._panel = ("", "")
        self._role_arm = ""               # the label whose promotion is armed
        self._adding = False
        self._lock_tick = None
        self._canvas = None
        self._body = None
        self._body_win = None
        self._thumb = None
        self._head = None
        self._foot = None
        self._add_btn = None
        self._code_entry = None
        self._build()

    # ----------------------------------------------------------- services
    def _service(self, name):
        return getattr(self.services, name, None) if self.services else None

    def toast(self, line, kind="info") -> None:
        """One place every line this page shows goes through, so a test can
        read them back and a hash could only appear here on purpose."""
        self._toasts.append((str(line), kind))
        bar = self._toast_bar
        if bar is not None:
            try:
                bar.show(str(line), kind=kind)
            except Exception:             # noqa: BLE001 - a toast bar
                log.debug("users page: the toast failed", exc_info=True)

    # -------------------------------------------------------------- build
    def _build(self) -> None:
        """ONE COLUMN: a pinned head, a scrolling body, a pinned foot.

        The head carries the gate's own sentence and the lock, and it is
        pinned for the same reason the foot is: an unlock box he has to
        scroll to find is an unlock box he types his code into twice.
        """
        bg = theme.TV_BG
        self._head = tk.Frame(self, bg=bg)
        self._head.pack(side="top", fill="x", padx=theme.PAD,
                        pady=(theme.PAD_S, 0))
        self._gate_lbl = tk.Label(
            self._head, text="", font=ui_display(theme.SIZE_CAPTION),
            fg=theme.MUTED, bg=bg, anchor="w", justify="left", bd=0,
            padx=0, pady=0)
        self._gate_lbl.pack(fill="x")
        self._admin_lbl = tk.Label(
            self._head, text="", font=ui_display(theme.SIZE_CAPTION),
            fg=theme.FAINT, bg=bg, anchor="w", justify="left", bd=0,
            padx=0, pady=0)
        self._admin_lbl.pack(fill="x", pady=(px(2), 0))
        self._lock_row = tk.Frame(self._head, bg=bg)
        self._lock_row.pack(fill="x", pady=(px(6), 0))
        self._code_entry = tk.Entry(
            self._lock_row, show="•", width=14, bd=0, relief="flat",
            bg=theme.BG if theme.LOOK == "holo" else theme.SURFACE,
            fg=theme.INK, insertbackground=theme.CYAN,
            font=ui_font(theme.SIZE_LABEL), highlightthickness=0)
        self._code_entry.pack(side="left", ipady=px(3))
        self._unlock_btn = RoundButton(self._lock_row, text="Unlock",
                                       kind="ghost", bg=bg, pad_x=8, pad_y=4,
                                       command=self._unlock_pressed)
        self._unlock_btn.pack(side="left", padx=(theme.PAD_S, 0))
        self._lock_btn = RoundButton(self._lock_row, text="Lock", kind="ghost",
                                     bg=bg, pad_x=8, pad_y=4,
                                     command=self._lock_pressed)
        self._lock_btn.pack(side="left", padx=(px(4), 0))
        # BELOW the row, not beside it. Beside the entry and two buttons it
        # had ~340 px of his 1040-px window and "locked -- the override code
        # opens this tab and nothing else" was cut at "ope" (measured on the
        # 1040x1760 render, 2026-09-05). The sensors page learned this same
        # lesson this morning about its own caption.
        self._lock_lbl = tk.Label(
            self._head, text="", font=ui_display(theme.SIZE_CAPTION),
            fg=theme.FAINT, bg=bg, anchor="w", justify="left", bd=0,
            padx=0, pady=0)
        self._lock_lbl.pack(fill="x", pady=(px(3), 0))

        # ---- the pinned foot, built BEFORE the view so it packs under it
        self._foot = tk.Frame(self, bg=bg)
        act = tk.Frame(self._foot, bg=bg)
        act.pack(fill="x")
        # THE PRIMARY ACTION IS ALWAYS IN THE PINNED FOOT. The consent
        # paragraph is twenty lines, so a Create button at the bottom of
        # the scrolling body sits below the fold the moment there are
        # people on the page -- the same defect as a SAVE button that
        # scrolls away, which is the one he photographed on 2026-09-05.
        # Add is swapped FOR Create while the form is open rather than
        # sitting beside it: two primary buttons is a question about which
        # one finishes the job.
        self._add_btn = RoundButton(act, text="Add a person", kind="accent",
                                    bg=bg, pad_x=12, pad_y=5,
                                    command=self._add_pressed)
        self._add_btn.pack(side="left")
        self._create_btn = RoundButton(act, text="Create", kind="accent",
                                       bg=bg, pad_x=12, pad_y=5,
                                       command=self._create_pressed)
        self._cancel_btn = RoundButton(act, text="Cancel", kind="ghost",
                                       bg=bg, pad_x=10, pad_y=5,
                                       command=self._cancel)
        # THE PATH IS THE LEAST IMPORTANT THING IN THIS ROW, so it gives up
        # its width first. It is NOT packed here: the packer allocates in
        # pack-call order, and packing it at build time -- before Create and
        # Cancel are packed at paint time -- gave it the full 380 px of the
        # people.json path and cut the right-hand end off "Cancel".
        # MEASURED 2026-09-05 at 920x1440, on the shipped merge: the foot
        # read "Create  Cance|/home/example/.local/state/jarvis/people.json".
        # This is the SECOND time this exact defect has been fixed in this
        # file -- see the destructive panel's row, which solved it the same
        # way -- so the rule is written down rather than fixed again:
        # BUTTONS TAKE THEIR WIDTH FIRST, the path takes what is left, and
        # what is left is measured, not guessed.
        self._path_lbl = tk.Label(
            act, text="", font=ui_display(theme.SIZE_CAPTION),
            fg=theme.FAINT, bg=bg, anchor="e", bd=0, padx=0, pady=0)
        self._path_full = ""
        self._act_row = act
        # AND AGAIN WHENEVER THE ROW IS RESIZED. _fit_path is called from
        # _paint_foot, but at that moment the geometry manager has not run
        # yet and winfo_width() is still 1 or the PREVIOUS width -- so the
        # first fit measures a row that does not exist yet and leaves the
        # full path in a label that is then drawn right-anchored in a
        # narrower slot, cutting its LEFT end with no ellipsis to say so
        # ("ome/example/.local/state/jarvis/people.json", photographed
        # 2026-09-05 on the first version of this fix). The Configure
        # binding is the one that runs against real numbers.
        # THE LABEL MEASURES ITSELF, and nothing else. An earlier version
        # of this fix worked out the leftover from the row's width minus
        # each sibling's requested width and padding, and it was wrong in
        # both directions depending on WHEN it ran: during a repaint the
        # row is still 1 px wide, and between repaints the foot holds a
        # different set of buttons ("Add a person" is 272 px where Create
        # and Cancel are 337), so a fit taken in one state was applied in
        # the other and left the full path in a slot 21 px too small --
        # drawn right-anchored, so its HEAD was cut, with no ellipsis to
        # say so. The label's own allocated width needs no arithmetic and
        # is never a guess.
        self._path_lbl.bind("<Configure>", self._fit_path, add=True)
        self._note_lbl = tk.Label(
            self._foot, text="",
            font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT, bg=bg,
            anchor="w", justify="left", bd=0, padx=0, pady=0)
        self._note_lbl.pack(fill="x", pady=(px(6), 0))

        # ---- the scrolling view, packed FIRST and sized to its content
        view = tk.Frame(self, bg=bg)
        view.pack(side="top", fill="x")
        self._foot.pack(side="top", fill="x", padx=theme.PAD,
                        pady=(theme.PAD_S, px(12)))
        self._canvas = tk.Canvas(view, bg=bg, highlightthickness=0, bd=0,
                                 height=px(40))
        self._canvas.pack(side="left", fill="x", expand=True)
        # A 2px strip on the canvas' right edge, shown ONLY when there is
        # something below the fold: a page that scrolls with no mark saying
        # so is a page whose bottom rows he has no reason to look for.
        self._thumb = tk.Frame(self._canvas, bg=theme.CYAN_DIM,
                               width=max(2, px(2)))
        self._body = tk.Frame(self._canvas, bg=bg)
        self._body_win = self._canvas.create_window(0, 0, anchor="nw",
                                                    window=self._body)
        self._canvas.bind("<Configure>", lambda e: self._sync_view(), add=True)
        self._body.bind("<Configure>", lambda e: self._sync_view(), add=True)
        self._canvas.bind("<Enter>", self._grab_wheel, add=True)
        self._canvas.bind("<Leave>", self._drop_wheel, add=True)
        self.bind("<Configure>", self._wrap, add=True)
        self._paint()

    def _wrap_px(self) -> int:
        """The width a line of this page may take before it wraps.

        winfo_width() is 1 until the geometry manager has run, and a
        wraplength of 1 would put one character on each line -- so the
        page's own requested width stands in until it is placed.
        """
        try:
            width = max(int(self.winfo_width()), int(self.winfo_reqwidth()))
        except Exception:                 # noqa: BLE001 - torn down
            width = 0
        return max(px(160), width - 2 * theme.PAD)

    @staticmethod
    def _wrap_to_own_slot(event=None) -> None:
        """Wrap ONE label to the width it was actually given.

        For a label that shares a row with a button, the page width is not
        the room it has. Re-entry guarded: setting wraplength changes the
        requested height, which can bring another <Configure> straight
        back round.
        """
        try:
            want = max(px(160), int(event.width))
            if int(event.widget.cget("wraplength")) != want:
                event.widget.configure(wraplength=want)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: slot wrap failed", exc_info=True)

    def _wrap(self, event=None) -> None:
        """Re-wrap EVERY label the page draws, not only the three fixed
        ones. The chips line under a person is the longest text on the
        page, and unwrapped it was cut mid-word at his window edge
        (measured 1395 px of a 1040-px window, 2026-09-05)."""
        width = (max(px(160), int(event.width) - 2 * theme.PAD)
                 if event is not None else self._wrap_px())
        for label in ([self._gate_lbl, self._admin_lbl, self._note_lbl,
                       self._lock_lbl] + list(self._wrapped)):
            try:
                label.configure(wraplength=width)
            except Exception:             # noqa: BLE001 - torn down
                log.debug("users page: wrap failed", exc_info=True)

    # --------------------------------------------------------- open/close
    def place_box(self) -> dict:
        """The place() kwargs that cover ``cover``, or the whole host --
        measured off the LIVE geometry, so a hidden footer or a packed
        camera pane cannot put the page out of step."""
        if not self.cover:
            return dict(x=0, y=0, relwidth=1.0, relheight=1.0)
        try:
            self.host.update_idletasks()
            tops = [w.winfo_y() for w in self.cover]
            bottoms = [w.winfo_y() + w.winfo_height() for w in self.cover]
            top, height = min(tops), max(bottoms) - min(tops)
        except Exception:                 # noqa: BLE001 - an unmapped widget
            log.debug("users page: cover geometry unreadable", exc_info=True)
            return dict(x=0, y=0, relwidth=1.0, relheight=1.0)
        if height < 1:
            return dict(x=0, y=0, relwidth=1.0, relheight=1.0)
        return dict(x=0, y=top, relwidth=1.0, height=height)

    @property
    def is_open(self) -> bool:
        return self._open

    def toggle(self) -> None:
        self.hide() if self._open else self.show()

    def show(self) -> None:
        if self._open:
            return
        self._open = True
        self.place(in_=self.host, **self.place_box())
        self.lift()
        # Read the file NOW rather than on a timer: it is a file, and it
        # changes when he changes it.
        self.refresh()
        self._tick_lock()

    def hide(self) -> None:
        if not self._open:
            return
        self._open = False
        # LEAVING THE TAB RELOCKS AT ONCE. The dwell is for a run of edits
        # in one sitting, not for a console left open on his desk -- and it
        # is the APP's dwell that has to go, because that is the one the
        # writes consult. A page that dropped only its own looked locked
        # and was not.
        self._relock()
        self._forget.disarm()
        self._purge.disarm()
        self._panel = ("", "")
        self._role_arm = ""
        self._adding = False
        self._untick()
        # bind_all is GLOBAL: a wheel binding left behind would scroll a
        # hidden page from anywhere in the console.
        self._drop_wheel()
        try:
            self.place_forget()
        except Exception:                 # noqa: BLE001 - a dead widget
            log.debug("users page: unplace failed", exc_info=True)
        if self._on_close:
            try:
                self._on_close()
            except Exception:             # noqa: BLE001 - a callback
                log.exception("users page: on_close failed")

    # ------------------------------------------------------------ reading
    def refresh(self) -> None:
        fn = self._service("people_snapshot")
        snap = {}
        if fn is not None:
            try:
                snap = fn() or {}
            except Exception:             # noqa: BLE001 - the app boundary
                log.exception("users page: the snapshot failed")
                snap = {}
        self._snapshot = snap if isinstance(snap, dict) else {}
        self._rows = rows_from(self._snapshot)
        self._paint()

    @property
    def admin_state(self) -> str:
        from jarvis import gate as gate_mod
        return str(self._snapshot.get("admin") or gate_mod.ADMIN_REFUSE)

    def _plan(self) -> BootstrapPlan:
        return bootstrap_plan(self._snapshot.get("fault_kind") or "",
                              self._snapshot.get("path") or "")

    # ------------------------------------------------------------- paint
    def _paint(self) -> None:
        """Rebuild the body from the snapshot. Cheap: a handful of labels
        per person, and it happens on open and after a write, never on a
        clock."""
        if self._body is None:
            return
        bg = theme.TV_BG
        self._gate_lbl.configure(
            text=self._snapshot.get("gate_line")
            or "owner-gate: nothing has been read yet", fg=theme.MUTED)
        plan = self._plan()
        self._admin_lbl.configure(
            text=plan.line or self._snapshot.get("admin_line") or "",
            fg=theme.WARN if (plan.line and not plan.may_create)
            else theme.FAINT)
        self._path_full = self._snapshot.get("path") or ""
        for child in list(self._body.winfo_children()):
            child.destroy()
        self._row_widgets = {}
        self._wrapped = []
        if self._adding:
            self._build_add_form(self._body, bg)
        if not self._rows and not self._adding:
            tk.Label(self._body,
                     text=("Nobody is enrolled." if not plan.line
                           else "No rows could be read from the people file."),
                     font=ui_display(theme.SIZE_LABEL), fg=theme.FAINT,
                     bg=bg, anchor="w", justify="left").pack(
                fill="x", padx=theme.PAD, pady=(theme.PAD_S, 0))
        for row in self._rows:
            self._build_block(self._body, bg, row)
        self._paint_foot()
        self._paint_lock()
        self._wrap()
        self._sync_view()

    def _paint_foot(self) -> None:
        """Add, or Create and Cancel -- never both."""
        self._add_btn.set_enabled(self._may_add())
        try:
            if self._adding:
                self._add_btn.pack_forget()
                if not self._create_btn.winfo_ismapped():
                    self._create_btn.pack(side="left")
                    self._cancel_btn.pack(side="left",
                                          padx=(theme.PAD_S, 0))
            else:
                self._create_btn.pack_forget()
                self._cancel_btn.pack_forget()
                if not self._add_btn.winfo_ismapped():
                    self._add_btn.pack(side="left")
            # LAST, always: re-packing it here is what puts it behind the
            # buttons in the packer's allocation order, whichever buttons
            # this state has.
            # EMPTY FIRST, THEN EXPAND INTO THE LEFTOVER. With no text the
            # label asks for nothing, so the buttons take their natural
            # width; expand=True then gives the label whatever is left, and
            # its own <Configure> fills it in against that real number. It
            # is repacked here rather than at build time so that it is
            # behind the buttons in the packer's allocation order.
            self._path_lbl.configure(text="")
            self._path_lbl.pack_forget()
            self._path_lbl.pack(side="right", fill="x", expand=True,
                                padx=(theme.PAD_S, 0))
            # THE NOTE IS BUILT HERE, from the seams this console was
            # actually handed, rather than being a tuple somebody edits by
            # hand a hundred lines from the code it describes. That is the
            # whole fix for the defect he caught on 2026-09-05: a capability
            # that lands turns its own sentence off, and a half-wired UI
            # describes itself correctly instead of promising a button it
            # has not got.
            lines = list(foot_lines(self.services))
            self._note_lbl.configure(
                text="\n".join("· " + line for line in lines))
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: the foot could not be repacked",
                      exc_info=True)

    def _fit_path(self, event=None) -> None:
        """Trim the people-file path from the LEFT until it fits the room
        the buttons left it, with a leading ellipsis.

        The TAIL is the part that identifies the file, so the head is what
        goes. A path that will not fit at all becomes the bare file name,
        and a file name that still will not fit becomes nothing -- an empty
        label is honest; half a word beside a half-drawn button is not.
        """
        full = getattr(self, "_path_full", "") or ""
        try:
            lbl = self._path_lbl
            room = int(lbl.winfo_width()) - _INK_SLACK
            if int(lbl.winfo_width()) <= 1:
                # Not laid out yet. Show NOTHING rather than guess: the
                # Configure that gives it a width fits it a moment later
                # against a real number, and an empty label for one frame
                # is better than a path with its head cut off.
                lbl.configure(text="")
                return
            font = tkfont.Font(font=lbl.cget("font"))
            if room <= 0 or not full:
                lbl.configure(text="")
                return
            if font.measure(full) <= room:
                lbl.configure(text=full)
                return
            # Give up leading path segments first; then characters.
            parts = full.split("/")
            while len(parts) > 1:
                parts.pop(0)
                shown = "\u2026/" + "/".join(parts)
                if font.measure(shown) <= room:
                    lbl.configure(text=shown)
                    return
            name = parts[-1] if parts else ""
            while name and font.measure("\u2026" + name) > room:
                name = name[1:]
            lbl.configure(text=("\u2026" + name) if name else "")
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: the path could not be fitted",
                      exc_info=True)

    def _may_add(self) -> bool:
        """Whether a write is POSSIBLE AT ALL -- not whether it is unlocked.

        A bootstrap offers exactly one action; a broken file offers none,
        and the button is the thing that says which. The LOCK is a
        different kind of no: it is temporary and he can act on it, so it
        is a toast naming the remedy rather than a dead grey button with
        no explanation, which is the same treatment Forget gets.
        """
        from jarvis import gate as gate_mod
        plan = self._plan()
        if plan.line:
            return bool(plan.may_create)
        return self.admin_state != gate_mod.ADMIN_REFUSE

    def _paint_lock(self) -> None:
        """Show or hide the unlock row for the state the registry is IN.

        IT HAS TO PACK AS WELL AS FORGET. The page paints once at BUILD
        time, before any snapshot has been read, so the state is still
        "refuse" and this row is packed away; nothing put it back when the
        snapshot then said a code IS required. Found by looking at the
        1040x1760 render on 2026-09-05: the tab said "locked" and offered
        nowhere to type a code, which makes every administrative action
        unreachable rather than guarded.
        """
        from jarvis import gate as gate_mod
        state = self.admin_state
        widgets = ((self._code_entry, dict(side="left", ipady=px(3))),
                   (self._unlock_btn, dict(side="left",
                                           padx=(theme.PAD_S, 0))),
                   (self._lock_btn, dict(side="left", padx=(px(4), 0))))
        # The row itself goes away with its contents: an empty 62-px band
        # over the first person is a control he looks for and cannot find.
        if state != gate_mod.ADMIN_CODE:
            for w, _kw in widgets:
                try:
                    w.pack_forget()
                except Exception:         # noqa: BLE001 - torn down
                    log.debug("users page: lock row hide failed",
                              exc_info=True)
            try:
                self._lock_row.pack_forget()
            except Exception:             # noqa: BLE001 - torn down
                log.debug("users page: lock row hide failed", exc_info=True)
            self._lock_lbl.configure(
                text=self._snapshot.get("admin_line") or "", fg=theme.FAINT)
            return
        try:
            if not self._lock_row.winfo_ismapped():
                self._lock_row.pack(before=self._lock_lbl, fill="x",
                                    pady=(px(6), 0))
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: lock row show failed", exc_info=True)
        for w, kw in widgets:
            try:
                if not w.winfo_ismapped():
                    w.pack(**kw)
            except Exception:             # noqa: BLE001 - torn down
                log.debug("users page: lock row show failed", exc_info=True)
        if self._lock.locked():
            self._lock_lbl.configure(text="locked — the override code opens "
                                          "this tab and nothing else",
                                     fg=theme.FAINT)
        else:
            self._lock_lbl.configure(
                text="unlocked for %d s" % int(self._lock.remaining()),
                fg=theme.CYAN)

    def _poll_admin(self) -> bool:
        """Ask the app what the FILE says now. True if it changed.

        Cheap by construction: one small JSON read and three values, which
        is why it may sit on a one-second tick where ``people_snapshot``
        may not. An app half without the seam simply never changes
        anything -- the page keeps working against the snapshot it has,
        which is what it did before this existed.
        """
        fn = self._service("people_admin_state")
        if fn is None:
            return False
        try:
            live = fn() or {}
        except Exception:                 # noqa: BLE001 - the app boundary
            log.exception("users page: the admin state could not be read")
            return False
        if not isinstance(live, dict):
            return False
        # The APP owns the dwell. Mirroring it every tick is what keeps the
        # countdown on screen from being a second, drifting clock.
        self._lock.mirror(live.get("unlocked_s") or 0.0)
        was = self._snapshot.get("admin")
        state = live.get("admin")
        if not state or state == was:
            return False
        self._snapshot["admin"] = state
        self._snapshot["admin_line"] = live.get("admin_line") or ""
        log.info("users page: the people file now says %s (was %s)",
                 state, was)
        return True

    def _tick_lock(self) -> None:
        """Repaint the countdown once a second, and notice a code set
        somewhere else. The clock is what relocks the page, not this timer:
        a press that lands after the window passed is refused by the APP
        whether or not Tk ran the tick."""
        self._lock_tick = None
        if not self._open:
            return
        if self._poll_admin():
            # A different answer changes the buttons and the foot as well
            # as the lock row, so it is a full repaint rather than a
            # _paint_lock: _may_add() and the bootstrap plan both read it.
            self._paint()
        else:
            self._paint_lock()
        try:
            self._lock_tick = self.after(LOCK_TICK_MS, self._tick_lock)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: the lock tick failed", exc_info=True)

    def _untick(self) -> None:
        if self._lock_tick is None:
            return
        try:
            self.after_cancel(self._lock_tick)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: tick cancel failed", exc_info=True)
        self._lock_tick = None

    # ------------------------------------------------------- one person
    def _build_block(self, parent, bg, row: Row) -> None:
        block = tk.Frame(parent, bg=bg)
        block.pack(fill="x", padx=theme.PAD, pady=(0, px(BLOCK_GAP)))
        head = tk.Frame(block, bg=bg)
        head.pack(fill="x")
        tk.Label(head, text=row.display,
                 font=ui_display(theme.SIZE_BODY, "semibold"), fg=theme.INK,
                 bg=bg, anchor="w", bd=0, padx=0, pady=0).pack(side="left")
        tk.Label(head, text=row.label, font=ui_display(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=bg, anchor="w", bd=0, padx=0,
                 pady=0).pack(side="left", padx=(theme.PAD_S, 0))
        tk.Label(head, text=ROLE_WORDS.get(row.role, row.role.upper()),
                 font=ui_display(theme.SIZE_CAPTION, "semibold"),
                 fg=theme.CYAN if row.role == ROLE_OWNER else theme.MUTED,
                 bg=bg, anchor="w", bd=0, padx=0,
                 pady=0).pack(side="left", padx=(theme.PAD_S, 0))
        forget = RoundButton(head, text="Forget", kind="ghost", bg=bg,
                             pad_x=8, pad_y=4,
                             command=lambda who=row.label: self._forget_pressed(who))
        forget.pack(side="right")
        role = RoundButton(
            head, text=("Make known" if row.role == ROLE_OWNER
                        else "Make owner"),
            kind="ghost", bg=bg, pad_x=8, pad_y=4,
            command=lambda who=row.label, r=row.role: self._role_pressed(who, r))
        role.pack(side="right", padx=(0, px(4)))
        forget.set_enabled(row.can_forget)
        role.set_enabled(row.can_change_role)
        self._row_widgets[row.label] = {"forget": forget, "role": role,
                                        "block": block}
        detail = tk.Label(block, text=row.detail,
                          font=ui_display(theme.SIZE_CAPTION),
                          fg=theme.MUTED, bg=bg, anchor="w",
                          justify="left", bd=0, padx=0, pady=0)
        detail.pack(fill="x", pady=(px(2), 0))
        self._wrapped.append(detail)
        if row.face_why:
            # AMBER MEANS A FAULT AND NOTHING ELSE -- the rule this tree
            # already keeps (tests/test_ui_classic_frozen.py). Colouring
            # the chips line put five correct chips in the fault colour;
            # this puts the colour on the one thing that IS at fault, and
            # lets it say what to do about it.
            face_note = tk.Label(block, text=row.face_why,
                                 font=ui_display(theme.SIZE_CAPTION),
                                 fg=theme.WARN, bg=bg, anchor="w",
                                 justify="left", bd=0, padx=0, pady=0)
            face_note.pack(fill="x")
            self._wrapped.append(face_note)
        why = row.forget_why or row.role_why
        if why:
            note = tk.Label(block, text=why,
                            font=ui_display(theme.SIZE_CAPTION),
                            fg=theme.FAINT, bg=bg, anchor="w",
                            justify="left", bd=0, padx=0, pady=0)
            note.pack(fill="x")
            self._wrapped.append(note)
        self._build_actions(block, bg, row)
        if self._forget.armed_for == row.label:
            self._build_forget_panel(block, bg, row.label)
        if self._purge.armed_for == row.label:
            self._build_purge_panel(block, bg, row.label, self._purge_kind)
        if self._panel == ("phrase", row.label):
            self._build_phrase_panel(block, bg, row.label)
        if self._panel == ("code", row.label):
            self._build_code_panel(block, bg, row.label)
        if self._role_arm == row.label:
            self._build_role_panel(block, bg, row.label)

    def _build_forget_panel(self, parent, bg, label) -> None:
        warn = tk.Label(parent, text="\n".join(forget_warning(
                            label,
                            voices=self._snapshot.get("voices") or (),
                            gallery=self._snapshot.get("gallery") or (),
                            can=console_can(self.services))),
                        font=ui_display(theme.SIZE_CAPTION), fg=theme.WARN,
                        bg=bg, anchor="w", justify="left", bd=0, padx=0,
                        pady=0)
        warn.pack(fill="x", pady=(px(4), 0))
        self._wrapped.append(warn)
        cmd = tk.Frame(parent, bg=bg)
        cmd.pack(fill="x", pady=(px(2), 0))
        command = forget_face_command(label)
        RoundButton(cmd, text="Copy", kind="ghost", bg=bg, pad_x=8, pad_y=4,
                    command=lambda c=command: self._copy(c)).pack(side="right")
        cmd_lbl = tk.Label(cmd, text=command, font=ui_font(theme.SIZE_CAPTION),
                           fg=theme.FAINT, bg=bg, anchor="w", justify="left",
                           bd=0, padx=0, pady=0)
        # PACKED AFTER Copy, deliberately: pack() hands out parcels in
        # call order, so a label asking for the whole command on one line
        # (MEASURED 1395 px) would take the cavity and leave the button
        # squeezed. The button goes first and keeps its natural width.
        cmd_lbl.pack(side="left", fill="x", expand=True)
        # DELIBERATELY NOT in self._wrapped. That list wraps to the PAGE
        # width, and this label shares its row with Copy -- so the page
        # width overstates its room by the whole button, and every line
        # that lands in the difference is drawn past the label's own
        # window and clipped mid-word (MEASURED 2026-09-05: wraplength 856
        # inside a 790 px slot at 920x1440). It follows its OWN allocation
        # instead, which stays right at any window size and whatever the
        # button beside it happens to measure.
        cmd_lbl.bind("<Configure>", self._wrap_to_own_slot, add=True)
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(px(4), 0))
        # ASKS FOR LITTLE AND GROWS. pack() never shrinks a widget below
        # its requested width -- fill and expand only ever ADD space -- so
        # a fixed 18-character entry is a hard floor that pushed the
        # buttons off the edge at 920x1440. Asking for 8 and expanding
        # into whatever is left fits every window and still gives him a
        # wide field at his own.
        entry = tk.Entry(row, width=8, bd=0, relief="flat",
                         bg=theme.BG if theme.LOOK == "holo" else theme.SURFACE,
                         fg=theme.INK, insertbackground=theme.CYAN,
                         font=ui_font(theme.SIZE_LABEL), highlightthickness=0)
        # THE BUTTONS TAKE THEIR WIDTH FIRST, anchored right, and the entry
        # gives up whatever is left. MEASURED 2026-09-05 at 920x1440: with
        # a fixed 18-character entry and both buttons packed left this row
        # asked for 977 px of a 920 px body, and because the body scrolls
        # only VERTICALLY the missing 57 px were not scrolled to -- they
        # were cut, and what was cut was the right-hand end of "Cancel".
        # A confirm entry 57 px narrower still takes a typed label; half a
        # Cancel button on the destructive panel is not acceptable.
        RoundButton(row, text="Cancel", kind="ghost", bg=bg, pad_x=8,
                    pad_y=4, command=self._cancel).pack(side="right",
                                                        padx=(px(4), 0))
        RoundButton(row, text="Forget %s" % label, kind="ghost", bg=bg,
                    pad_x=10, pad_y=4,
                    command=lambda who=label, e=entry:
                    self._forget_confirm(who, e.get())).pack(
            side="right", padx=(theme.PAD_S, 0))
        entry.pack(side="left", fill="x", expand=True, ipady=px(3))
        self._row_widgets.setdefault(label, {})["confirm"] = entry

    def _build_role_panel(self, parent, bg, label) -> None:
        owners = [r.label for r in self._rows if r.role == ROLE_OWNER]
        ask = tk.Label(parent,
                       text=("Making a second owner has to name the first. "
                             "Type %s to confirm." % ", ".join(owners)),
                       font=ui_display(theme.SIZE_CAPTION), fg=theme.WARN,
                       bg=bg, anchor="w", justify="left", bd=0, padx=0,
                       pady=0)
        ask.pack(fill="x", pady=(px(4), 0))
        self._wrapped.append(ask)
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(px(2), 0))
        entry = tk.Entry(row, width=18, bd=0, relief="flat",
                         bg=theme.BG if theme.LOOK == "holo" else theme.SURFACE,
                         fg=theme.INK, insertbackground=theme.CYAN,
                         font=ui_font(theme.SIZE_LABEL), highlightthickness=0)
        entry.pack(side="left", ipady=px(3))
        RoundButton(row, text="Make %s an owner" % label, kind="ghost",
                    bg=bg, pad_x=10, pad_y=4,
                    command=lambda who=label, e=entry:
                    self._role_confirm(who, e.get())).pack(
            side="left", padx=(theme.PAD_S, 0))
        RoundButton(row, text="Cancel", kind="ghost", bg=bg, pad_x=8,
                    pad_y=4, command=self._cancel).pack(side="left",
                                                        padx=(px(4), 0))

    # ------------------------------------------------------- the add form
    def _build_add_form(self, parent, bg) -> None:
        plan = self._plan()
        box = tk.Frame(parent, bg=bg)
        box.pack(fill="x", padx=theme.PAD, pady=(0, px(BLOCK_GAP)))
        first_only = plan.first_owner_only
        tk.Label(box, text=("Create the first owner" if first_only
                            else "Add a person"),
                 font=ui_display(theme.SIZE_BODY, "semibold"), fg=theme.INK,
                 bg=bg, anchor="w", bd=0, padx=0, pady=0).pack(fill="x")
        self._add_fields = {}
        # THE CAPTION SITS ABOVE ITS BOX, not beside it. A 26-CHARACTER
        # caption column is ~410 px of his 1040-px window at S=2, which
        # left "existing owner's label (owners only)" running into its own
        # entry and a hand's width of nothing between "role" and its
        # button (measured on the 1040x1760 render, 2026-09-05).
        for key, caption in (("name", "name, as he should say it"),
                             ("label", "label (lower-case, no spaces)"),
                             ("face", "face gallery label (optional)")):
            self._add_fields[key] = self._form_field(box, bg, caption)
        # THE CONSENT NAMES THE PERSON, and follows what he types. A
        # paragraph about "this person" is a paragraph nobody agreed to.
        self._add_fields["label"].bind("<KeyRelease>", self._retitle_consent,
                                       add=True)
        # THE ROLE. The first row in an empty or ownerless registry must be
        # an OWNER or the registry stays unusable, so there is no choice to
        # offer there -- and a general "add person" with a role dropdown is
        # exactly how somebody makes a KNOWN row and wonders why the gate
        # is still off.
        self._add_owner = bool(first_only)
        if not first_only:
            row = tk.Frame(box, bg=bg)
            row.pack(fill="x", pady=(px(6), 0))
            cap = tk.Label(row, text="role", font=ui_display(theme.SIZE_CAPTION),
                           fg=theme.FAINT, bg=bg, anchor="w", bd=0,
                           padx=0, pady=0)
            cap.pack(side="left", padx=(0, theme.PAD_S))
            # NOT wrapped: it is packed to its natural width beside the
            # buttons, so the page width is not its room either (856 px
            # claimed inside a 43 px slot). One word needs no wrapping.
            self._role_btn = RoundButton(
                row, text="known", kind="ghost", bg=bg, pad_x=10, pad_y=4,
                command=self._toggle_new_role)
            self._role_btn.pack(side="left")
            self._add_fields["confirm_owner"] = self._form_field(
                box, bg, "the EXISTING owner's label — needed only to make "
                         "a second owner")
            # THEIR AGREEMENT, in the shared words, shown before it is asked
            # for -- a consent nobody could read is not a consent.
            self._consent_lbl = tk.Label(
                box, text=self._consent_text(""),
                font=ui_display(theme.SIZE_CAPTION), fg=theme.MUTED,
                bg=bg, anchor="w", justify="left", bd=0, padx=0, pady=0)
            self._consent_lbl.pack(fill="x", pady=(px(6), 0))
            self._wrapped.append(self._consent_lbl)
            self._add_fields["consent"] = self._form_field(
                box, bg, "they type their own label here to agree")
        # Create and Cancel are in the PINNED FOOT (_paint_foot), not here.
        tail = tk.Label(box, text=("A face and a voice are still enrolled "
                                   "at a terminal; this only says who "
                                   "Jarvis knows."),
                        font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT,
                        bg=bg, anchor="w", justify="left", bd=0, padx=0,
                        pady=0)
        tail.pack(fill="x", pady=(px(4), 0))
        self._wrapped.append(tail)

    def _form_field(self, parent, bg, caption: str):
        """One labelled box: the caption on its own line, the entry under
        it filling the width. Stacked so a long caption wraps instead of
        squeezing its own entry off the right edge."""
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(px(6), 0))
        cap = tk.Label(row, text=caption, font=ui_display(theme.SIZE_CAPTION),
                       fg=theme.FAINT, bg=bg, anchor="w", justify="left",
                       bd=0, padx=0, pady=0)
        cap.pack(fill="x")
        self._wrapped.append(cap)
        entry = tk.Entry(
            row, bd=0, relief="flat",
            bg=theme.BG if theme.LOOK == "holo" else theme.SURFACE,
            fg=theme.INK, insertbackground=theme.CYAN,
            font=ui_font(theme.SIZE_LABEL), highlightthickness=0)
        entry.pack(fill="x", ipady=px(3), pady=(px(2), 0))
        return entry

    @staticmethod
    def _consent_text(label: str) -> str:
        """The shared words, named for whoever is being added."""
        return "\n".join(cs.lines_for(cs.WHAT_ROW,
                                      {"who": str(label or "").strip().lower()
                                       or "this person"}))

    def _retitle_consent(self, _event=None) -> None:
        lbl = getattr(self, "_consent_lbl", None)
        fields = getattr(self, "_add_fields", {})
        if lbl is None or "label" not in fields:
            return
        try:
            lbl.configure(text=self._consent_text(fields["label"].get()))
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: the consent text could not be retitled",
                      exc_info=True)

    def _toggle_new_role(self) -> None:
        self._add_owner = not self._add_owner
        try:
            self._role_btn.set_text("owner" if self._add_owner else "known")
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: the role button is gone", exc_info=True)

    # ------------------------------------------------------------ presses
    def _guard(self, action: str) -> bool:
        ok, why = may(action, self.admin_state, self._lock)
        if not ok:
            self.toast(why, "warn")
        return ok

    @staticmethod
    def _code_owed() -> str:
        """The app's refusal, imported from the gate rather than written
        twice: a copy here would drift and the page would silently stop
        raising its unlock row."""
        from jarvis import gate as gate_mod
        return gate_mod.ADMIN_CODE_OWED

    def _focus_code(self) -> None:
        """Put the cursor in the unlock box.

        THE PENDING GEOMETRY HAS TO SETTLE FIRST. ``refresh`` has just
        repacked the row, and Tk applies a pack at the next idle: focusing
        a window that is not yet viewable is dropped on the floor, so the
        cursor stayed wherever it was and he was told to type a code into a
        box that had not taken it. Measured on the 1040x1760 page.
        """
        if self._code_entry is None:
            return
        try:
            self.update_idletasks()
            self._code_entry.focus_set()
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: the code entry could not take focus",
                      exc_info=True)

    def _later(self, fn) -> None:
        """Marshal a control's answer back onto the Tk thread, SAFELY.

        Both secret controls run their service call on a daemon thread and
        land the line through here. If the page (or the console) has been torn
        down in between, ``after`` raises "main thread is not in main loop"
        INSIDE that thread, where nothing catches it -- it surfaces as an
        unhandled-thread-exception warning and the line is lost silently. A
        landed line that has nowhere to land is not an error; it is a page
        that has gone.
        """
        try:
            self.after(0, fn)
        except Exception:                 # noqa: BLE001 - the page has gone
            log.debug("users page: the answer had nowhere to land",
                      exc_info=True)

    def _unlock_pressed(self) -> str:
        return UsersUnlockControl(
            self.services, read=lambda: self._code_entry.get(),
            clear=lambda: self._code_entry.delete(0, "end"),
            toast=self.toast, later=self._later,
            on_unlocked=self._unlocked).unlock_pressed()

    def _unlocked(self) -> None:
        self._lock.unlock()
        self._paint_lock()

    def _relock(self) -> None:
        """Shut both: the app's dwell, which is what the writes check, and
        the page's mirror, which is what he sees."""
        self._lock.lock()
        fn = self._service("people_relock")
        if fn is None:
            return
        try:
            fn()
        except Exception:                 # noqa: BLE001 - the app boundary
            log.exception("users page: the relock seam failed")

    def _lock_pressed(self) -> None:
        self._relock()
        self._cancel()

    def _cancel(self) -> None:
        self._forget.disarm()
        self._purge.disarm()
        self._panel = ("", "")
        self._role_arm = ""
        self._adding = False
        self._paint()
        self._to_top()

    def _to_top(self) -> None:
        if self._canvas is None:
            return
        try:
            self.update_idletasks()
            self._canvas.yview_moveto(0.0)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: could not scroll to the top",
                      exc_info=True)
            return
        self._sync_thumb()

    def _copy(self, text: str) -> None:
        """The clipboard, through the seam that already owns it. Never a
        direct write: his selections are his, and a page that clobbers one
        while he is pasting is the bug this codebase already has a rule
        about."""
        from jarvis import enrolentry
        try:
            landed = enrolentry.to_clipboard(text)
        except Exception:                 # noqa: BLE001 - the clipboard
            log.debug("users page: the clipboard hook raised", exc_info=True)
            landed = False
        self.toast("The command is on your clipboard, sir." if landed
                   else "The clipboard would not take it; it is on screen.",
                   "ok" if landed else "warn")

    def _forget_pressed(self, label: str) -> None:
        if not self._guard("forget"):
            return
        self._role_arm = ""
        self._forget.press(label)
        self._paint()
        self._reveal(self._row_widgets.get(label, {}).get("confirm"))

    def _forget_confirm(self, label: str, typed) -> None:
        if not self._guard("forget"):
            return
        verdict = self._forget.confirm(label, typed)
        if verdict == "expired":
            self.toast("That confirmation has gone cold, sir; press Forget "
                       "again.", "warn")
            self._paint()
            return
        if verdict == "refused":
            self.toast("Type %s exactly to confirm, sir." % label, "warn")
            return
        self._write("people_forget", label)

    def _role_pressed(self, label: str, role: str) -> None:
        if not self._guard("set_role"):
            return
        self._forget.disarm()
        if role == ROLE_OWNER:
            # A demotion needs no second owner named; the registry refuses
            # to leave nobody in charge on its own.
            self._write("people_set_role", label, ROLE_KNOWN)
            return
        self._role_arm = label
        self._paint()
        self._reveal(self._row_widgets.get(label, {}).get("block"))

    def _role_confirm(self, label: str, typed) -> None:
        if not self._guard("set_role"):
            return
        owners = [r.label for r in self._rows if r.role == ROLE_OWNER]
        ok, why = owner_confirmed(ROLE_OWNER, owners, typed)
        if not ok:
            self.toast(why, "warn")
            return
        self._role_arm = ""
        self._write("people_set_role", label, ROLE_OWNER,
                    confirm_existing_owner=str(typed or "").strip().lower())

    def _add_pressed(self) -> None:
        if not self._guard("add"):
            return
        self._adding = not self._adding
        self._forget.disarm()
        self._role_arm = ""
        self._paint()
        # THE FORM OPENS AT ITS TOP. The body kept whatever scroll offset
        # the last open panel left, so the name, label and face rows were
        # above the fold the moment the form appeared -- visible in the
        # 31-users-add render, which opened on "role".
        self._to_top()

    def _create_pressed(self) -> None:
        if not self._guard("add"):
            return
        fields = getattr(self, "_add_fields", {})
        label = str(fields["label"].get() if "label" in fields else "")
        label = label.strip().lower()
        fault = label_fault(label)
        if fault:
            self.toast(fault, "warn")
            return
        face = str(fields["face"].get() if "face" in fields
                   else "").strip().lower()
        if face:
            # The face POINTER is why the face leg can name anyone. A label
            # the gallery would refuse must not reach the registry, or the
            # two stores disagree about who exists.
            fault = label_fault(face)
            if fault:
                self.toast("the face label: %s" % fault, "warn")
                return
        role = ROLE_OWNER if self._add_owner else ROLE_KNOWN
        owners = [r.label for r in self._rows if r.role == ROLE_OWNER]
        confirm = str(fields["confirm_owner"].get()
                      if "confirm_owner" in fields else "").strip().lower()
        ok, why = owner_confirmed(role, owners, confirm)
        if not ok:
            self.toast(why, "warn")
            return
        how = "owner"
        if role != ROLE_OWNER:
            typed = fields["consent"].get() if "consent" in fields else ""
            ok, how = take_consent(label, owner="", show=None,
                                   ask=lambda: typed,
                                   mapped=self.winfo_ismapped)
            if not ok:
                self.toast(how, "warn")
                return
        if face and face not in set(self._snapshot.get("gallery") or ()):
            # NOT a refusal: enrolling the face is a separate minute at a
            # terminal and he may well add the row first. But an
            # unresolved pointer is exactly how the face leg goes quiet,
            # so it is said at the moment it is created rather than found
            # later as an amber chip.
            self.toast("%s is not in the face gallery yet, so the face leg "
                       "cannot name %s until it is enrolled."
                       % (face, label), "warn")
        self._write("people_add", label=label,
                    name=str(fields["name"].get() if "name" in fields
                             else "").strip(),
                    role=role, face=face, consent=how,
                    confirm_existing_owner=confirm or None)

    def _write(self, service: str, *args, **kw) -> None:
        """Call one app seam and land its single line. Named ``service``
        and not ``name``: people_add takes a keyword called ``name``, and
        the collision was a TypeError at the one press it guards."""
        fn = self._service(service)
        if fn is None:
            self.toast(UNLOCK_NOT_WIRED, "warn")
            return
        try:
            ok, line = fn(*args, **kw)
        except Exception:                 # noqa: BLE001 - the app boundary
            log.exception("users page: %s failed", service)
            ok, line = False, "That did not work, sir; see the log."
        if ok:
            # Each successful action re-arms the dwell, so a run of edits
            # is one code rather than five. The app re-armed its own; this
            # is the mirror, corrected by the next tick either way.
            self._lock.touch()
            self._adding = False
        self.toast(str(line), "ok" if ok else "warn")
        # THE SEAM IS THE GUARD, so a press may be refused here even when
        # the page thought it was allowed -- a code set at a terminal, or a
        # dwell that ran out between the press and the call. Refresh reads
        # the file again, which is what makes _paint_lock draw the unlock
        # row; then put the cursor in it, because the row he needs may have
        # appeared in the head while he was looking at a person block.
        owed = not ok and str(line) == self._code_owed()
        self.refresh()
        if owed:
            self._focus_code()

    # -------------------------------------------------- the per-row actions
    def _btn_row(self, parent, bg):
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(px(4), 0))
        return row

    def _build_actions(self, parent, bg, row: Row) -> None:
        """The buttons this row offers, and NOT the ones it must not.

        ONE BUTTON PER ROW-FULL AT MOST TWO, and that is measured rather than
        tidy: the head row already carries Make owner and Forget anchored
        right, and a fifth control in it asked for 977 px of his 920 px body
        the last time this file grew (the cut Cancel, 2026-09-05). Buttons
        take their width from their own text, so a row that does not fit does
        not truncate -- it pushes its last button off the edge, where the
        body's vertical-only scroll cannot reach it.

        THE OWNER'S ROW IS THE ONLY ONE THAT ENROLS. jarvis/enrolrun.py forces
        the owner label by construction and jarvis/voicerun.py refuses any
        other, for the reason a window cannot get round: enrolling somebody
        else stores a measurement of THEM, so they must read what is kept and
        type their own name, at a terminal, where nobody can type it for
        them. A guest row gets the hand-over instead.
        """
        widgets = self._row_widgets.setdefault(row.label, {})
        is_owner = row.role == ROLE_OWNER
        # THE SAME DICT THE FOOT NOTE READS. A capability this console has not
        # got may not be drawn as a button, and one it HAS may not be denied
        # in the note -- ``owner_buttons`` is the list this branch must
        # produce, and tests/test_users_note_matches_buttons.py walks every
        # wiring to prove the two agree.
        can = console_can(self.services)
        if is_owner:
            secrets = None
            for cap, press in (("phrase", self._phrase_pressed),
                               ("code", self._code_pressed)):
                if not can[cap]:
                    continue
                if secrets is None:
                    secrets = self._btn_row(parent, bg)
                widgets[cap] = self._action(
                    secrets, bg, OWNER_BUTTON[cap],
                    lambda who=row.label, fn=press: fn(who))
            # THE HAND-OVERS GO ONE PER ROW and the short buttons share one.
            # MEASURED at 2.0 scale: "Copy the face command" is 441 px and
            # "Copy the voice command" 458, which side by side with their
            # padding is 911 px of the 856 px a 920x1440 window leaves inside
            # the block -- and the body scrolls only downwards, so the excess
            # is CUT rather than reached. "Enrol my face" and "Enrol my voice"
            # are short enough to sit together and are measured as such.
            enrol = None
            for cap, press in (("face", self._face_pressed),
                               ("voice", self._voice_pressed)):
                if can[cap]:
                    if enrol is None:
                        enrol = self._btn_row(parent, bg)
                    widgets[cap] = self._action(
                        enrol, bg, OWNER_BUTTON[cap],
                        lambda who=row.label, fn=press: fn(who))
                else:
                    # NOTE_FACE_NO and NOTE_VOICE_NO both promise "the command
                    # is handed over ready to run", so a console without the
                    # seam owes him that rather than a gap where a button was.
                    widgets["%s_handover" % cap] = self._action(
                        self._btn_row(parent, bg), bg, HANDOVER_BUTTON[cap],
                        lambda who=row.label, k=cap, n=row.name:
                        self._copy(enrol_command(who, k, n)))
            # STOP IS ALWAYS THERE while this page is open AND a stop seam
            # exists, not only once a run is known to be live. The page has no
            # live view of the run -- it is on its own thread inside the app
            # -- and a Stop that appears only when the page happens to know is
            # a Stop that is missing at exactly the moment station four has
            # put the keyboard out of reach. It answers "Nothing is running,
            # sir." when there is nothing to stop, which costs him one line
            # and never a lens.
            if can["stop"]:
                if enrol is None:
                    enrol = self._btn_row(parent, bg)
                widgets["stop"] = self._action(enrol, bg, STOP_BUTTON,
                                               self._enrol_stop)
        else:
            # ONE PER ROW, and this is measured rather than cautious. At 2.0
            # scale "Copy the face command" is 441 px and "Copy the voice
            # command" is 458; side by side with their padding that is 911 px
            # of the 856 px a 920x1440 window leaves inside the block, and
            # the body scrolls only downwards -- so the 55 px would not be
            # scrolled to, they would be CUT, off the right-hand end of the
            # second button. That is the same defect as the half-drawn Cancel
            # photographed on 2026-09-05, and it is why these two are stacked.
            self._action(self._btn_row(parent, bg), bg,
                         "Copy the face command",
                         lambda who=row.label: self._copy(
                             enrol_command(who, "face", row.name)))
            self._action(self._btn_row(parent, bg), bg,
                         "Copy the voice command",
                         lambda who=row.label: self._copy(
                             enrol_command(who, "voice", row.name)))
        # THE GALLERY BUTTONS ARE OFFERED ONLY WHERE THERE IS SOMETHING TO
        # REMOVE. A "remove their face measurements" on a row with none is a
        # button whose only possible answer is "there was nothing", which
        # reads as a failure.
        if row.face_known and can["purge_face"]:
            widgets["purge_face"] = self._action(
                self._btn_row(parent, bg), bg, "Remove face measurements",
                lambda who=row.label: self._purge_pressed(who, "face"))
        if row.label in set(self._snapshot.get("voices") or ()) \
                and can["purge_voice"]:
            widgets["purge_voice"] = self._action(
                self._btn_row(parent, bg), bg, "Remove voice pool",
                lambda who=row.label: self._purge_pressed(who, "voice"))

    def _action(self, parent, bg, text, command):
        btn = RoundButton(parent, text=text, kind="ghost", bg=bg, pad_x=8,
                          pad_y=4, command=command)
        btn.pack(side="left", padx=(0, px(6)))
        return btn

    # ------------------------------------------------------ the phrase panel
    def _build_phrase_panel(self, parent, bg, label) -> None:
        """Two masked boxes and what he is agreeing to, in his register.

        THE BOXES ARE MASKED and neither survives the press: the control
        empties both BEFORE it hashes, because ``hash_secret`` is a scrypt KDF
        at N=2^14 and a typed secret must not sit on screen while it runs, nor
        still be there if the tab is left mid-call.
        """
        text = tk.Label(parent, text="\n".join(phrase_panel_lines()),
                        font=ui_display(theme.SIZE_CAPTION), fg=theme.MUTED,
                        bg=bg, anchor="w", justify="left", bd=0, padx=0,
                        pady=0)
        text.pack(fill="x", pady=(px(4), 0))
        self._wrapped.append(text)
        boxes = []
        for caption in ("new passphrase", "again, to be sure"):
            row = tk.Frame(parent, bg=bg)
            row.pack(fill="x", pady=(px(4), 0))
            cap = tk.Label(row, text=caption,
                           font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT,
                           bg=bg, anchor="w", bd=0, padx=0, pady=0)
            cap.pack(fill="x")
            self._wrapped.append(cap)
            entry = tk.Entry(
                row, show="•", bd=0, relief="flat",
                bg=theme.BG if theme.LOOK == "holo" else theme.SURFACE,
                fg=theme.INK, insertbackground=theme.CYAN,
                font=ui_font(theme.SIZE_LABEL), highlightthickness=0)
            entry.pack(fill="x", ipady=px(3), pady=(px(2), 0))
            boxes.append(entry)
        self._row_widgets.setdefault(label, {})["phrase_boxes"] = tuple(boxes)
        act = self._btn_row(parent, bg)
        self._action(act, bg, "Set", lambda who=label: self._phrase_confirm(who))
        self._action(act, bg, "Cancel", self._cancel)

    # -------------------------------------------------------- the code panel
    def _build_code_panel(self, parent, bg, label) -> None:
        """No box to choose one, and the caption says why BEFORE he presses.

        The destination and the "no mailbox" refusal come from
        ``services.knightfall_status`` -- the drawer's own caption, reused
        rather than written a second time, because a promise made here that
        the drawer would not make is a promise one of them will break.
        """
        status = self._knightfall_status()
        row = [r for r in self._rows if r.label == label]
        has_code = bool(row[0].has_code) if row else True
        plan = code_plan(status, mode=self._gate_mode(), has_code=has_code)
        text = tk.Label(
            parent,
            text="\n".join(code_panel_lines(status, mode=self._gate_mode(),
                                            has_code=has_code)),
            font=ui_display(theme.SIZE_CAPTION),
            fg=theme.MUTED if plan.enabled else theme.WARN, bg=bg, anchor="w",
            justify="left", bd=0, padx=0, pady=0)
        text.pack(fill="x", pady=(px(4), 0))
        self._wrapped.append(text)
        act = self._btn_row(parent, bg)
        send = self._action(act, bg, "Send it",
                            lambda who=label: self._code_confirm(who))
        send.set_enabled(plan.enabled)
        self._action(act, bg, "Cancel", self._cancel)

    def _knightfall_status(self) -> dict:
        fn = self._service("knightfall_status")
        if fn is None:
            return {}
        try:
            out = fn() or {}
        except Exception:                 # noqa: BLE001 - the app boundary
            log.exception("users page: the knightfall status failed")
            return {}
        return out if isinstance(out, dict) else {}

    def _gate_mode(self) -> str:
        """The gate's live mode, read off the startup line the snapshot
        already carries -- no second call, and "" when it cannot be read."""
        line = str(self._snapshot.get("gate_line") or "").lower()
        for mode in ("enforce", "shadow", "off"):
            if "owner-gate: %s" % mode in line:
                return mode
        return ""

    # ------------------------------------------------------- the purge panel
    def _build_purge_panel(self, parent, bg, label, kind) -> None:
        warn = tk.Label(parent, text="\n".join(purge_warning(label, kind)),
                        font=ui_display(theme.SIZE_CAPTION), fg=theme.WARN,
                        bg=bg, anchor="w", justify="left", bd=0, padx=0,
                        pady=0)
        warn.pack(fill="x", pady=(px(4), 0))
        self._wrapped.append(warn)
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(px(4), 0))
        entry = tk.Entry(row, width=8, bd=0, relief="flat",
                         bg=theme.BG if theme.LOOK == "holo" else theme.SURFACE,
                         fg=theme.INK, insertbackground=theme.CYAN,
                         font=ui_font(theme.SIZE_LABEL), highlightthickness=0)
        # THE BUTTONS TAKE THEIR WIDTH FIRST and the entry gives up what is
        # left -- the rule this file already learned twice, most recently
        # when a fixed 18-character entry pushed "Cancel" off the right edge
        # at 920x1440 on the destructive panel. A confirm box 57 px narrower
        # still takes a typed label; half a Cancel button does not.
        RoundButton(row, text="Cancel", kind="ghost", bg=bg, pad_x=8, pad_y=4,
                    command=self._cancel).pack(side="right", padx=(px(4), 0))
        RoundButton(row, text="Remove %s" % label, kind="ghost", bg=bg,
                    pad_x=10, pad_y=4,
                    command=lambda who=label, k=kind, e=entry:
                    self._purge_confirm(who, k, e.get())).pack(
            side="right", padx=(theme.PAD_S, 0))
        entry.pack(side="left", fill="x", expand=True, ipady=px(3))
        self._row_widgets.setdefault(label, {})["purge_confirm"] = entry

    # ------------------------------------------------------- the new presses
    def _open_panel(self, kind: str, label: str, action: str) -> None:
        if not self._guard(action):
            return
        self._forget.disarm()
        self._purge.disarm()
        self._role_arm = ""
        self._panel = (kind, str(label))
        self._paint()
        self._reveal(self._row_widgets.get(label, {}).get("block"))

    def _phrase_pressed(self, label: str) -> None:
        self._open_panel("phrase", label, "set_phrase")

    def _code_pressed(self, label: str) -> None:
        self._open_panel("code", label, "new_code")

    def _phrase_confirm(self, label: str) -> None:
        """Hand the two boxes to the control, which empties them first."""
        if not self._guard("set_phrase"):
            return
        boxes = (self._row_widgets.get(label, {}).get("phrase_boxes")
                 or ())
        if len(boxes) != 2:
            self.toast(PHRASE_NOT_WIRED, "warn")
            return
        first, second = boxes

        def _clear():
            for box in (first, second):
                try:
                    box.delete(0, "end")
                except Exception:         # noqa: BLE001 - torn down
                    log.debug("users page: a phrase box is gone",
                              exc_info=True)

        UsersSecretControl(
            self.services, label=label,
            read=lambda: (first.get(), second.get()), clear=_clear,
            toast=self.toast, later=self._later,
            on_done=self._after_secret).pressed()

    def _after_secret(self) -> None:
        self._lock.touch()
        self._panel = ("", "")
        self.refresh()

    def _code_confirm(self, label: str) -> None:
        if not self._guard("new_code"):
            return
        self._panel = ("", "")
        self._write("people_new_code")

    def _face_pressed(self, label: str) -> None:
        if not self._guard("face_enrol"):
            return
        self._write("face_enrol_start")

    def _voice_pressed(self, label: str) -> None:
        if not self._guard("voice_enrol"):
            return
        self._write("voice_enrol_start")

    def _enrol_stop(self) -> None:
        """STOP IS NEVER GATED, and it stops both.

        A code owed must never be the reason a lens or a microphone stays
        open -- the rule ``Commander._enrol_control`` already follows for the
        spoken stop, which takes the widest door for the same reason. One
        button rather than two: he is being asked to stop the thing that is
        running, and at station four he cannot see which button is which.
        """
        lines = []
        for service in ("face_enrol_stop", "voice_enrol_stop"):
            fn = self._service(service)
            if fn is None:
                continue
            try:
                ok, line = fn()
            except Exception:             # noqa: BLE001 - the app boundary
                log.exception("users page: %s failed", service)
                ok, line = False, "That did not stop, sir; see the log."
            if ok:
                lines.append(str(line))
        self.toast(lines[0] if lines else "Nothing is running, sir.",
                   "ok" if lines else "info")

    def _purge_pressed(self, label: str, kind: str) -> None:
        if not self._guard("purge_%s" % kind):
            return
        self._forget.disarm()
        self._role_arm = ""
        self._panel = ("", "")
        self._purge_kind = str(kind)
        self._purge.press(label)
        self._paint()
        self._reveal(self._row_widgets.get(label, {}).get("purge_confirm"))

    def _purge_confirm(self, label: str, kind: str, typed) -> None:
        if not self._guard("purge_%s" % kind):
            return
        verdict = self._purge.confirm(label, typed)
        if verdict == "expired":
            self.toast("That confirmation has gone cold, sir; press Remove "
                       "again.", "warn")
            self._paint()
            return
        if verdict == "refused":
            self.toast("Type %s exactly to confirm, sir." % label, "warn")
            return
        self._write("people_purge_%s" % kind, label)

    # -------------------------------------------------------- the scroll
    def _sync_view(self) -> None:
        if self._canvas is None:
            return
        try:
            width = self._canvas.winfo_width()
            self._canvas.itemconfigure(self._body_win, width=width)
            want = min(self._body.winfo_reqheight(), self._avail_px())
            if want > 0 and want != self._canvas.winfo_reqheight():
                self._canvas.configure(height=want)
            self._canvas.configure(
                scrollregion=self._canvas.bbox("all") or (0, 0, 0, 0))
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: scroll region unreadable", exc_info=True)
            return
        self._sync_thumb()

    def _avail_px(self) -> int:
        """How much height the scrolling body may take: the page, less what
        the pinned head and the pinned foot need. THIS is what keeps the
        Add button on screen at his window with a dozen people on it."""
        if self._foot is None or self._canvas is None or self._head is None:
            return 0
        try:
            height = max(int(self.winfo_height()), 1)
            if height <= 1:
                return 0
            head = self._head.winfo_reqheight() + theme.PAD_S
            foot = self._foot.winfo_reqheight() + theme.PAD_S + px(12)
            return max(px(40), height - head - foot)
        except Exception:                 # noqa: BLE001 - torn down
            return 0

    def _sync_thumb(self) -> None:
        if self._canvas is None:
            return
        view_h = self._canvas.winfo_height()
        over = self.overflow_px()
        if over <= 0 or view_h <= 1:
            try:
                self._thumb.place_forget()
                self._canvas.yview_moveto(0.0)
            except Exception:             # noqa: BLE001 - torn down
                log.debug("users page: thumb hide failed", exc_info=True)
            return
        content = view_h + over
        top = self._canvas.canvasy(0)
        frac = max(0.08, view_h / float(content))
        try:
            self._thumb.place(relx=1.0, x=-px(3), anchor="nw",
                              y=int(top / content * view_h),
                              height=max(px(12), int(frac * view_h)))
            self._thumb.lift()
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: thumb place failed", exc_info=True)

    def overflow_px(self) -> int:
        """How much taller the body is than the viewport. 0 = it all fits."""
        if self._canvas is None:
            return 0
        try:
            room = self._avail_px() or self._canvas.winfo_height()
            return max(0, self._body.winfo_reqheight() - room)
        except Exception:                 # noqa: BLE001 - torn down
            return 0

    def _reveal(self, widget) -> None:
        """Scroll the body until ``widget`` is in the viewport.

        FOUND BY LOOKING at the 1040x1760 render: the forget warning is
        ten lines, so arming one on anybody but the first person pushed
        the confirmation entry and its button below the fold. He would
        press Forget, read a wall of amber and find no way to confirm --
        the same shape of defect as a SAVE button that scrolls away, which
        is the one he photographed this morning.
        """
        if self._canvas is None or widget is None:
            return
        try:
            self.update_idletasks()
            body_h = self._body.winfo_reqheight()
            view = self._canvas.winfo_height()
            if body_h <= view or view <= 1:
                return
            y = widget.winfo_rooty() - self._body.winfo_rooty()
            bottom = y + widget.winfo_height()
            top = self._canvas.canvasy(0)
            if top <= y and bottom <= top + view:
                return                    # already in sight
            # Put its BOTTOM at the fold rather than its top at the ceiling:
            # what he needs to see is the entry and the button under it,
            # and the warning above them is what he has just read.
            want = min(max(0.0, bottom - view + px(8)), float(body_h - view))
            self._canvas.yview_moveto(want / float(body_h))
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: reveal failed", exc_info=True)
            return
        self._sync_thumb()

    def _scroll(self, units: int) -> None:
        if self.overflow_px() <= 0:
            return
        try:
            self._canvas.yview_scroll(units, "units")
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: scroll failed", exc_info=True)
            return
        self._sync_thumb()

    def _grab_wheel(self, _e=None) -> None:
        if self._canvas is None:
            return
        self._canvas.bind_all("<Button-4>", lambda e: self._scroll(-2))
        self._canvas.bind_all("<Button-5>", lambda e: self._scroll(2))

    def _drop_wheel(self, _e=None) -> None:
        if self._canvas is None:
            return
        for seq in ("<Button-4>", "<Button-5>"):
            try:
                self._canvas.unbind_all(seq)
            except Exception:             # noqa: BLE001 - torn down
                log.debug("users page: wheel unbind failed", exc_info=True)
