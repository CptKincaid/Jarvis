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
from typing import Callable, Optional, Sequence, Tuple

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

# What the voice chip may say. MEASURED in gate._voice_leg: there is ONE
# voiceprint pool (PATHS.VOICEPRINT) and the leg answers with the OWNER's
# label on a match, whoever actually spoke. So a guest's "voice" is inert
# and a tickbox for it would let him build a row that lies.
VOICE_OWNER = "voice: owner only (one voiceprint)"
VOICE_OWNER_NONE = "voice: no voiceprint enrolled"
VOICE_GUEST = ("voice: cannot name a guest — there is one voiceprint and "
               "it is the owner's")

# The standing note in the pinned foot. Plain sentences, and the first one
# is the one this whole feature rests on.
CANNOT_DO = (
    "This is recognition, not a lock: anyone already at this keyboard can "
    "edit the people file directly.",
    "Setting a new spoken passphrase or a new override code still needs a "
    "terminal — this page only says whether each is set.",
    "Enrolling a face or a voice still needs a terminal: the running Jarvis "
    "owns the camera and the microphone. The command is handed over ready "
    "to run.",
    "Forgetting somebody removes their entry here only. Their face "
    "measurements stay in the gallery until that command is run.",
    "Nothing here can be undone: the people file has no history and no "
    "backup.",
)

# The actions that demand the code. READING THE LIST IS NOT ONE: the
# terminal tool runs `list` before it authorises, for exactly this reason,
# and the list holds no secret -- every row is built from redacted().
GUARDED = ("add", "set_role", "forget", "set_face")

LOCKED_LINE = ("Unlock with your override code before changing anything, "
               "sir.")
BROKEN_LINE = ("The people file could not be read as a registry, so nothing "
               "may be changed from here. Repair or move the file first.")


# ============================================================== the unlock
class Lock:
    """This page's own unlock state, and NOTHING else's.

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
    """May this action happen right now? ``(ok, why)``.

    ``state`` is ``gate.admin_gate``'s answer -- the SAME decision the
    terminal tool asks, not a second one written beside it.
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
    except Exception:  # noqa: BLE001 - a snapshot that cannot answer
        log.exception("users: the snapshot could not be read")
        return ()
    owners = [p for p in people if _get(p, "role") == ROLE_OWNER]
    sole = owners[0].get("label") if len(owners) == 1 else None
    out = []
    for person in people:
        if not isinstance(person, dict):
            continue
        out.append(_row(person, gallery, sole))
    return tuple(out)


def _get(person, key, default=""):
    try:
        return person.get(key, default)
    except Exception:  # noqa: BLE001
        return default


def _row(person: dict, gallery, sole) -> Row:
    label = str(_get(person, "label"))
    name = str(_get(person, "name"))
    role = str(_get(person, "role")) or ROLE_KNOWN
    face = str(_get(person, "face"))
    try:
        dim = int(_get(person, "face_dim", 0) or 0)
    except (TypeError, ValueError):
        dim = 0
    known = bool(face) and face in gallery
    if not face:
        tone = "muted"
    elif known:
        tone = "ok"
    else:
        # A pointer at a gallery label that is not there is exactly how the
        # face leg silently stops naming anyone.
        tone = "warn"
    voice = bool(_get(person, "voice", False))
    if role == ROLE_OWNER:
        voice_text = VOICE_OWNER if voice else VOICE_OWNER_NONE
    else:
        voice_text = VOICE_GUEST
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


def forget_warning(label: str) -> Tuple[str, ...]:
    """What is destroyed, what SURVIVES, and that none of it comes back.

    MEASURED in ``identity.Registry.forget``: it removes the ROW and
    nothing else. Saying "removed" and leaving the rest implied is how
    somebody comes to believe a face gallery was scrubbed when it was not.
    """
    who = str(label or "")
    return (
        "Forget %s?" % who,
        "",
        "This removes %s's entry: their name, their role, the face label "
        "they point at, their consent record and the date they were "
        "added." % who,
        "",
        "WHAT SURVIVES. %s's face measurements stay in the gallery — this "
        "page does not touch them. The command that removes those is below "
        "and it is a separate step." % who,
        "There is no voice recording of %s to remove: Jarvis holds one "
        "voiceprint and it is the owner's." % who,
        "",
        "THIS CANNOT BE UNDONE. The people file is rewritten in place; "
        "there is no history and no backup.",
        "",
        "Type %s to confirm." % who,
    )


def forget_face_command(label: str, *, python: Optional[str] = None,
                        script: Optional[str] = None) -> str:
    """The exact command that removes their face measurements, quoted, via
    the seam that already builds it -- not a second string to drift."""
    from jarvis import enrolentry
    return enrolentry.command_line(str(label or ""), delete=True,
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
