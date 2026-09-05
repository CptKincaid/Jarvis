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


# ============================================================ the Tk surface
import tkinter as tk                                   # noqa: E402

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

    NOTHING HERE POLLS. Unlike SENSORS there is nothing to poll: the
    registry is a file. It is read on ``show()`` and re-read immediately
    before every write, by the app seam itself.
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
        self._path_lbl = tk.Label(
            act, text="", font=ui_display(theme.SIZE_CAPTION),
            fg=theme.FAINT, bg=bg, anchor="e", bd=0, padx=0, pady=0)
        self._path_lbl.pack(side="right", padx=(theme.PAD_S, 0))
        self._note_lbl = tk.Label(
            self._foot, text="\n".join("· " + line for line in CANNOT_DO),
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
        # in one sitting, not for a console left open on his desk.
        self._lock.lock()
        self._forget.disarm()
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
        self._path_lbl.configure(text=self._snapshot.get("path") or "")
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
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("users page: the foot could not be repacked",
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

    def _tick_lock(self) -> None:
        """Repaint the countdown once a second. The clock is what relocks
        the page, not this timer: a press that lands after the window
        passed is refused by ``Lock`` whether or not Tk ran the tick."""
        self._lock_tick = None
        if not self._open:
            return
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
                          fg=theme.WARN if row.face_tone == "warn"
                          else theme.MUTED, bg=bg, anchor="w",
                          justify="left", bd=0, padx=0, pady=0)
        detail.pack(fill="x", pady=(px(2), 0))
        self._wrapped.append(detail)
        why = row.forget_why or row.role_why
        if why:
            note = tk.Label(block, text=why,
                            font=ui_display(theme.SIZE_CAPTION),
                            fg=theme.FAINT, bg=bg, anchor="w",
                            justify="left", bd=0, padx=0, pady=0)
            note.pack(fill="x")
            self._wrapped.append(note)
        if self._forget.armed_for == row.label:
            self._build_forget_panel(block, bg, row.label)
        if self._role_arm == row.label:
            self._build_role_panel(block, bg, row.label)

    def _build_forget_panel(self, parent, bg, label) -> None:
        warn = tk.Label(parent, text="\n".join(forget_warning(label)),
                        font=ui_display(theme.SIZE_CAPTION), fg=theme.WARN,
                        bg=bg, anchor="w", justify="left", bd=0, padx=0,
                        pady=0)
        warn.pack(fill="x", pady=(px(4), 0))
        self._wrapped.append(warn)
        cmd = tk.Frame(parent, bg=bg)
        cmd.pack(fill="x", pady=(px(2), 0))
        command = forget_face_command(label)
        cmd_lbl = tk.Label(cmd, text=command, font=ui_font(theme.SIZE_CAPTION),
                           fg=theme.FAINT, bg=bg, anchor="w", justify="left",
                           bd=0, padx=0, pady=0)
        cmd_lbl.pack(side="left", fill="x", expand=True)
        self._wrapped.append(cmd_lbl)
        RoundButton(cmd, text="Copy", kind="ghost", bg=bg, pad_x=8, pad_y=4,
                    command=lambda c=command: self._copy(c)).pack(side="right")
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(px(4), 0))
        entry = tk.Entry(row, width=18, bd=0, relief="flat",
                         bg=theme.BG if theme.LOOK == "holo" else theme.SURFACE,
                         fg=theme.INK, insertbackground=theme.CYAN,
                         font=ui_font(theme.SIZE_LABEL), highlightthickness=0)
        entry.pack(side="left", ipady=px(3))
        RoundButton(row, text="Forget %s" % label, kind="ghost", bg=bg,
                    pad_x=10, pad_y=4,
                    command=lambda who=label, e=entry:
                    self._forget_confirm(who, e.get())).pack(
            side="left", padx=(theme.PAD_S, 0))
        RoundButton(row, text="Cancel", kind="ghost", bg=bg, pad_x=8,
                    pad_y=4, command=self._cancel).pack(side="left",
                                                        padx=(px(4), 0))
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
            self._wrapped.append(cap)
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

    def _unlock_pressed(self) -> str:
        return UsersUnlockControl(
            self.services, read=lambda: self._code_entry.get(),
            clear=lambda: self._code_entry.delete(0, "end"),
            toast=self.toast, later=lambda fn: self.after(0, fn),
            on_unlocked=self._unlocked).unlock_pressed()

    def _unlocked(self) -> None:
        self._lock.unlock()
        self._paint_lock()

    def _lock_pressed(self) -> None:
        self._lock.lock()
        self._cancel()

    def _cancel(self) -> None:
        self._forget.disarm()
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
            # is one code rather than five.
            self._lock.touch()
            self._adding = False
        self.toast(str(line), "ok" if ok else "warn")
        self.refresh()

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
