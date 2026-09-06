"""Who Jarvis knows, and what each of them may do.

THIS IS RECOGNITION, NOT A LOCK. Jarvis answers the person he recognises.
A photograph defeats the face check and a recording defeats the voice check,
and the user was told that and accepted it. Nothing in this module, in
``jarvis/gate.py``, or in any line either of them says out loud may pretend
otherwise: the whole feature is aimed at accidents and confusion, not at
somebody determined to get in.

WHAT LIVES HERE
    One model of a person -- a label, a name, a role, whether a voice
    enrolment and a face label exist, and the salted hashes of the two
    fallbacks -- and one derivation of the owner's label.

WHY THE HASHES ARE NOT IN assistant.json
    ``AssistantConfig`` is deep-copied into reports, its ``__repr__`` prints
    ``redacted()``, and its whole file is read back by anything holding the
    path. Teaching ``redacted()`` two more keys would work until somebody
    added a third code path. Keeping the hashes out of that object entirely
    is the stronger version of the same promise, and it leaves SECRET_KEYS
    untouched. They live in ``PATHS.OWNER_REGISTRY`` (``people.json``),
    written 0600 through the same atomic dance ``assistant_config`` uses.

THE ONE RULE THIS FILE ENFORCES IN CODE RATHER THAN IN A COMMENT
    Only an OWNER may enrol somebody, change a role, or change a setting.
    There is one owner today and he is Hunter. The model does not assume
    there is only ever one -- but a second one cannot be created by
    accident: ``add_person`` and ``set_role`` refuse to mint an owner unless
    the caller names the existing owner it means to stand beside.

HOW SOMEBODY IS ADDRESSED IS STORED, TYPED, AND NEVER INFERRED
    ``Person.honorific`` is one of exactly ``"sir"``, ``"ma'am"`` or ``""``
    and ``scripts/jarvis_people.py add`` REQUIRES it with no default. There
    is no name list in this repo, no gender table and no code path that
    reads a name and produces one: Mara and Heather are addressed as
    "ma'am" because Hunter typed ``--honorific maam``, and for no other
    reason. An unknown stored value reads back as "" -- no form of address
    -- never as a guess, and a FORMAT-1 row (which has no such key) lands
    in the same place rather than silently acquiring "sir".

    The line itself is swapped at the door, in ``address.swap_addresses``,
    called from ``app._say`` and ``commander._speak``. None of the ~1,000
    "sir" literals in this tree is edited.

AND THE ONE THAT KEEPS IT FROM BECOMING A BRICK
    ``Registry.load`` NEVER RAISES. Missing, unparsable, wrong-shaped,
    unreadable, or holding no owner at all -- every one of those comes back
    ``usable=False`` with a plain reason, and ``jarvis/gate.py`` turns the
    gate off when it sees that. A gate that refuses its owner because its
    own storage broke is worse than no gate. That is NOT the same as failing
    open on a negative verdict: somebody the registry does not know is still
    refused.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("identity")

# --------------------------------------------------------------- roles
ROLE_OWNER = "owner"
ROLE_KNOWN = "known"
# UNKNOWN IS THE ABSENCE OF A ROW, never a stored role. Storing it would
# create a fourth state ("enrolled as nobody") that every caller would then
# have to know about.
ROLE_UNKNOWN = ""
ROLES = (ROLE_OWNER, ROLE_KNOWN)
# Only ever compared, never arithmetic: "the highest role among the
# positives wins" (jarvis/recognise.py).
RANK = {ROLE_UNKNOWN: 0, ROLE_KNOWN: 1, ROLE_OWNER: 2}

# jarvis/facegallery.py's own pattern. A label that the gallery would refuse
# must not reach the registry either, or the two stores disagree about who
# exists and the face leg silently stops naming anyone.
LABEL_RX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")

# FORMAT 2 added ``first``, ``last`` and ``honorific``. A FORMAT-1 row
# carries none of those keys, so ``Person.from_json`` reads it as
# honorific="" -- NO form of address rather than a guessed one. That is the
# whole reason the bump is here: an old row must not silently acquire "sir".
FORMAT = 2
DEFAULT_OWNER = "hunter"

# THE ONLY THREE FORMS OF ADDRESS, and the empty one is a real choice.
# There is no fourth, there is no default, and there is NO CODE PATH
# ANYWHERE that reads a name and produces one of these. Hunter is "sir"
# because he typed it; Mara and Heather are "ma'am" because he typed that.
# A name list, a gender table or a "guess and let them correct it" is the
# one thing this feature must never do -- it would be wrong in front of a
# real person standing in his kitchen.
HON_SIR = "sir"
HON_MAAM = "ma'am"
HON_NONE = ""
HONORIFICS = (HON_SIR, HON_MAAM, HON_NONE)

# What the shell is allowed to type, mapped to what is stored. "maam" and
# "none" exist so an apostrophe never has to survive a quoting mistake.
HONORIFIC_WORDS = {"sir": HON_SIR, "maam": HON_MAAM, "ma'am": HON_MAAM,
                   "ma\u2019am": HON_MAAM, "none": HON_NONE, "": HON_NONE}


def honorific_from_word(word) -> Tuple[str, str]:
    """``(value, why)`` for a typed choice. NEVER guesses from a name."""
    try:
        key = str(word or "").strip().lower()
    except Exception:  # noqa: BLE001 - a choice that cannot be read is none
        return "", "that is not a form of address I know"
    if key not in HONORIFIC_WORDS:
        return "", ("%r is not a form of address; choose sir, maam, or none"
                    % (word,))
    return HONORIFIC_WORDS[key], ""


def slug(name) -> str:
    """A label from a display name. MAY COME BACK UNSTORABLE, ON PURPOSE.

    Lower-case, keep what ``isalnum()`` likes plus ``-`` and ``_``. That
    keeps NON-ASCII -- a ``user.name`` of "José" comes back "josé", which
    the gallery's own ``^[a-z0-9][a-z0-9_-]{0,30}$`` refuses -- and it is
    meant to. ``scripts/face_enrol.target_label`` checks the result against
    that pattern and STOPS BEFORE THE CAMERA with a sentence naming
    ``user.name``, so he is told what to fix. Repairing it quietly here
    would enrol him under a mangled name he never chose and never sees:
    pinned by tests/test_faceenrol_notes.py::
    test_a_username_the_gallery_cannot_store_stops_before_the_camera.

    So validity is the CALLER's question, asked with ``LABEL_RX``.
    """
    try:
        text = str(name or "").strip().lower()
    except Exception:  # noqa: BLE001 - a name that cannot be read is no name
        return ""
    return "".join(c if (c.isalnum() or c in "-_") else "" for c in text)


def owner_label(cfg=None) -> str:
    """HIS label -- the ONE derivation of "the owner" anywhere downstream.

    There used to be three of these: ``jarvis/cast.py``'s bare literal,
    ``commander._face_owner`` and ``scripts/face_enrol.owner_label``, each
    written out separately and each free to drift. All three call this now.

    Read from HIS CONFIG and never from the gallery or the registry, which
    is where the standing ruling is anchored: the set of ENROLLED names may
    grow without the set of PRIVILEGED names growing by one, because "owner"
    is a config value and a recognised face cannot write the config.
    """
    name = ""
    if cfg is not None:
        try:
            # Either shape: an AssistantConfig (``.get``) or the app's own
            # ``get_option`` callable. Both are handed round this codebase
            # and demanding one of them would have made a fourth derivation
            # of this label the path of least resistance.
            getter = getattr(cfg, "get", None)
            if callable(getter):
                name = getter("user.name", "")
            elif callable(cfg):
                name = cfg("user.name", "")
        except Exception:  # noqa: BLE001 - a config that cannot say is not one
            log.debug("identity: could not read user.name", exc_info=True)
            name = ""
    return slug(name) or DEFAULT_OWNER


# -------------------------------------------------------------- a person
@dataclass
class Person:
    """One row. ``voice`` and ``face`` say which legs could ever name them.

    ``face_dim`` is the embedding width the face label was enrolled at, and
    it is stored for one reason: the face model is being swapped from SFace
    (128-D) to ArcFace (512-D) on another branch, and the vectors are not
    comparable. A row enrolled at one width and read by a recogniser of
    another is NO OPINION -- never a negative -- and the startup line can
    then say WHY the face leg is dark instead of leaving him guessing.
    """

    label: str
    name: str = ""
    # The two halves of the spoken sign-in. They are NOT a credential --
    # see jarvis/signin.py -- they only narrow who Jarvis thinks he is
    # hearing so a confirming leg has something to confirm.
    first: str = ""
    last: str = ""
    # "sir", "ma'am" or "" -- typed by the owner, never inferred.
    honorific: str = ""
    role: str = ROLE_KNOWN
    voice: bool = False
    face: str = ""
    face_dim: int = 0
    # Salted hashes. Never plaintext, never printed, never logged.
    phrase_hash: str = ""
    code_hash: str = ""
    # THE PENDING OVERRIDE CODE (Knightfall weekly, 2026-09-06). The weekly
    # issuer hands a fresh code to Oracle BEFORE it knows whether Sunday's
    # backup email will carry it, so until Oracle's receipt comes back
    # BOTH hashes open the door -- gate.check_override_code tries this one
    # after code_hash. ``pending_code_id`` is eight hex characters that
    # travel in the spool, the receipt and the log; it names the push and
    # says nothing about the code. ``pending_code_since`` is the UTC stamp
    # of the push, from which the drop deadline is computed. The three
    # move together, through set_pending_code / promote_pending /
    # drop_pending and nothing else. An OLDER build's to_json drops them.
    pending_code_hash: str = ""
    pending_code_id: str = ""
    pending_code_since: str = ""
    enrolled_at: str = ""
    # "owner" (his own row) or "typed" (they typed their own name at a
    # terminal -- scripts/face_enrol.consent's existing rule, not a new one).
    consent: str = ""

    def display(self) -> str:
        full = ("%s %s" % (self.first, self.last)).strip()
        return self.name or full or self.label.capitalize()

    def full_name(self) -> str:
        """First and last, or "" when they were never typed."""
        return ("%s %s" % (self.first, self.last)).strip()

    def redacted(self) -> dict:
        """Everything about this person EXCEPT the two hashes, which are
        replaced by a bare yes/no. Safe for a log, a report, or the pane."""
        return {"label": self.label, "name": self.name,
                "first": self.first, "last": self.last,
                "honorific": self.honorific, "role": self.role,
                "voice": bool(self.voice), "face": self.face,
                "face_dim": int(self.face_dim),
                "has_phrase": bool(self.phrase_hash),
                "has_code": bool(self.code_hash),
                "has_pending": bool(self.pending_code_hash),
                "enrolled_at": self.enrolled_at, "consent": self.consent}

    # A dataclass __repr__ would print both hashes into any log line that
    # ever interpolated a Person. These two are the reason it does not.
    def __repr__(self) -> str:      # noqa: D105 - see above
        return "Person(%s, role=%s, voice=%s, face=%s)" % (
            self.label, self.role, bool(self.voice), self.face or "-")

    __str__ = __repr__

    def to_json(self) -> dict:
        out = self.redacted()
        out.pop("has_phrase", None)
        out.pop("has_code", None)
        out.pop("has_pending", None)
        out["phrase_hash"] = self.phrase_hash
        out["code_hash"] = self.code_hash
        out["pending_code_hash"] = self.pending_code_hash
        out["pending_code_id"] = self.pending_code_id
        out["pending_code_since"] = self.pending_code_since
        return out

    @classmethod
    def from_json(cls, row) -> Optional["Person"]:
        """A row, or None. A malformed row costs itself and nothing else:
        one bad entry must not take the whole registry -- and the whole
        registry going down is what turns the gate off."""
        if not isinstance(row, dict):
            return None
        label = str(row.get("label", "") or "")
        if not LABEL_RX.match(label):
            return None
        role = str(row.get("role", "") or "")
        if role not in ROLES:
            return None
        try:
            dim = int(row.get("face_dim", 0) or 0)
        except (TypeError, ValueError):
            dim = 0
        # A value this build does not know is NOT a guess and NOT a
        # crash: it is no form of address at all. A FORMAT-1 row has no
        # key here and lands in exactly the same place.
        hon = str(row.get("honorific", "") or "")
        if hon not in HONORIFICS:
            log.warning("registry: %r is not a form of address I store; "
                        "%s will be addressed by name only", hon, label)
            hon = HON_NONE
        return cls(label=label, name=str(row.get("name", "") or ""),
                   first=str(row.get("first", "") or ""),
                   last=str(row.get("last", "") or ""), honorific=hon,
                   role=role, voice=bool(row.get("voice", False)),
                   face=str(row.get("face", "") or ""), face_dim=dim,
                   phrase_hash=str(row.get("phrase_hash", "") or ""),
                   code_hash=str(row.get("code_hash", "") or ""),
                   pending_code_hash=str(row.get("pending_code_hash", "") or ""),
                   pending_code_id=str(row.get("pending_code_id", "") or ""),
                   pending_code_since=str(row.get("pending_code_since", "") or ""),
                   enrolled_at=str(row.get("enrolled_at", "") or ""),
                   consent=str(row.get("consent", "") or ""))


# ------------------------------------------------------------- the store
def _write_private(path: Path, text: str) -> None:
    """Atomic 0600 write -- the pattern assistant_config._write_private and
    facegallery.save already use. Atomic because a half-written registry is
    a corrupt registry, and a corrupt registry turns the gate off: a save
    that failed halfway must not be able to disarm the feature."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".people-", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


# WHY a registry is unusable, as a TOKEN. ``fault`` is an English
# sentence for a human; a caller that has to BRANCH on the reason cannot
# match on prose. The one branch that matters is FAULT_MALFORMED /
# FAULT_UNREADABLE: for those two ``people`` comes back EMPTY even though
# the file holds rows, so an add-then-save writes a one-row registry over
# the top of them, and there is no history and no backup.
FAULT_NONE = ""
FAULT_MISSING = "missing"          # no file: a fresh install, enrol away
FAULT_UNREADABLE = "unreadable"    # the file exists and would not open
FAULT_MALFORMED = "malformed"      # it opened and is not a registry
FAULT_OWNERLESS = "ownerless"      # it parsed, rows are loaded, no owner


@dataclass
class Registry:
    """The people Jarvis knows. ``usable`` is the gate's off switch."""

    path: Optional[Path] = None
    people: List[Person] = field(default_factory=list)
    usable: bool = False
    fault: str = ""
    # The same fault, as a token a caller can branch on (see above).
    fault_kind: str = FAULT_NONE
    # A fingerprint of the file's bytes AS LOADED ("" for no file), so a
    # writer that loaded minutes ago can tell, under the lock, whether
    # somebody else wrote in between (save_checked).
    digest: str = ""

    # ------------------------------------------------------------ read
    @classmethod
    def load(cls, path=None) -> "Registry":
        """NEVER RAISES, for any state of the file on disk.

        Every failure resolves to ``usable=False`` and one plain sentence,
        because the caller's only sane response to any of them is the same:
        turn the gate off and say so at startup.
        """
        p = Path(path) if path is not None else PATHS.OWNER_REGISTRY
        reg = cls(path=p)
        try:
            raw = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            reg.fault = ("nobody is enrolled yet: %s does not exist" % p)
            reg.fault_kind = FAULT_MISSING
            return reg
        except OSError as exc:
            reg.fault = ("%s could not be read (%s)"
                         % (p, type(exc).__name__))
            reg.fault_kind = FAULT_UNREADABLE
            return reg
        reg.digest = _digest(raw)
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 - any parse failure is one fault
            reg.fault = "%s is not readable JSON" % p
            reg.fault_kind = FAULT_MALFORMED
            return reg
        if not isinstance(data, dict) or not isinstance(data.get("people"), list):
            reg.fault = "%s is not shaped like a registry" % p
            reg.fault_kind = FAULT_MALFORMED
            return reg
        for row in data["people"]:
            person = Person.from_json(row)
            if person is None:
                log.warning("registry: skipping a row that is not a person")
                continue
            if any(x.label == person.label for x in reg.people):
                log.warning("registry: %r appears twice; keeping the first",
                            person.label)
                continue
            reg.people.append(person)
        if not any(p_.role == ROLE_OWNER for p_ in reg.people):
            reg.fault = ("%s names no owner, so there is nobody who could "
                         "enrol one" % p)
            # The rows PARSED and are loaded, so adding an owner here adds
            # to them rather than replacing them: this fault is safe to
            # write over and the two above are not.
            reg.fault_kind = FAULT_OWNERLESS
            return reg
        reg.usable = True
        return reg

    # ----------------------------------------------------------- lookup
    def person(self, label) -> Optional[Person]:
        want = str(label or "")
        for p in self.people:
            if p.label == want:
                return p
        return None

    def role_of(self, label) -> str:
        p = self.person(label)
        return p.role if p is not None else ROLE_UNKNOWN

    def roles(self) -> Dict[str, str]:
        return {p.label: p.role for p in self.people}

    def labels(self) -> Tuple[str, ...]:
        return tuple(p.label for p in self.people)

    def owners(self) -> List[Person]:
        return [p for p in self.people if p.role == ROLE_OWNER]

    def may_administer(self, label) -> bool:
        """Enrol, change a role, change a setting. Owners only -- and this
        is the one predicate every admin path asks, so there is one answer
        rather than four."""
        return self.role_of(label) == ROLE_OWNER

    def face_labels(self) -> Dict[str, str]:
        """face label -> person label, for the legs that come back with a
        gallery name rather than a registry one."""
        return {p.face: p.label for p in self.people if p.face}

    # ------------------------------------------------------------ write
    def add_person(self, person: Person, *,
                   confirm_existing_owner: Optional[str] = None) -> Tuple[bool, str]:
        """Add a row. Returns ``(ok, why)`` -- never raises, never saves.

        A SECOND OWNER CANNOT BE MADE BY ACCIDENT. The model does not assume
        one owner forever, so this is not a hard ceiling; it is a
        confirmation. The caller has to name the owner it already knows
        about, which a slip of the hand cannot do.
        """
        if not isinstance(person, Person) or not LABEL_RX.match(person.label):
            return False, "that is not a label a gallery or a registry can store"
        if person.role not in ROLES:
            return False, "%r is not a role; it is %s or %s" % (
                person.role, ROLE_OWNER, ROLE_KNOWN)
        if person.honorific not in HONORIFICS:
            # Not coerced. A row created with a value nobody typed is a row
            # whose form of address was invented, which is the one thing
            # this feature refuses to do.
            return False, ("%r is not a form of address; it is sir, ma'am, "
                           "or none, and it is never inferred from a name"
                           % (person.honorific,))
        if person.voice and person.role != ROLE_OWNER:
            # THE INVARIANT THAT STOPPED THE VOICE LEG NAMING THE WRONG
            # PERSON, AND THE PREMISE UNDER IT HAS CHANGED. It was written
            # because jarvis/speaker.py held ONE voiceprint and ONE centroid
            # with no notion of a label, so "matched" could only ever mean
            # the owner and a second voice-enrolled row would make every
            # match come back as them.
            #
            # The voice-multispeaker lane ended that: speaker.py now carries
            # a LABELLED gallery and gate._voice_leg names whoever it named.
            # The refusal is KEPT ANYWAY at the integration merge (09-05),
            # deliberately and not by oversight. Nothing reads
            # ``Person.voice`` as an input to a decision -- it is written by
            # scripts/jarvis_people.py, rendered in to_json and __repr__,
            # and read nowhere else -- so the guard costs nothing, and
            # loosening an enrolment invariant inside a merge is how a hole
            # gets opened by somebody who was only trying to reconcile two
            # branches. Widening it is Hunter's call.
            return False, ("only the owner can be voice-enrolled: there is "
                           "one voiceprint on this machine, and marking a "
                           "second person voice-enrolled would make the "
                           "voice leg name the wrong person")
        if self.person(person.label) is not None:
            return False, "%s is already enrolled" % person.label
        existing = self.owners()
        if person.role == ROLE_OWNER and existing:
            names = ", ".join(p.label for p in existing)
            if confirm_existing_owner not in [p.label for p in existing]:
                return False, ("there is already an owner (%s); making a "
                               "second one has to name the first" % names)
        if not person.enrolled_at:
            person.enrolled_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.people.append(person)
        self.usable = bool(self.owners())
        if self.usable:
            self.fault = ""
            self.fault_kind = FAULT_NONE
        return True, ""

    def set_role(self, label, role, *,
                 confirm_existing_owner: Optional[str] = None) -> Tuple[bool, str]:
        p = self.person(label)
        if p is None:
            return False, "%s is not enrolled" % label
        if role not in ROLES:
            return False, "%r is not a role" % (role,)
        if role == ROLE_OWNER and p.role != ROLE_OWNER:
            others = [o.label for o in self.owners()]
            if others and confirm_existing_owner not in others:
                return False, ("there is already an owner (%s); making a "
                               "second one has to name the first"
                               % ", ".join(others))
        if p.role == ROLE_OWNER and role != ROLE_OWNER and \
                len(self.owners()) == 1:
            # Removing the last owner is the same brick as a corrupt file,
            # except it would look deliberate. There would be nobody left
            # who could enrol one.
            return False, ("%s is the only owner; demoting them would leave "
                           "nobody who can enrol anyone" % label)
        p.role = role
        return True, ""

    def set_honorific(self, label, value, *, by: str = "") -> Tuple[bool, str]:
        """How this person is addressed. OWNER-ONLY, on the same ``(ok, why)``
        contract as every other writer here.

        ``by`` is the label of whoever is asking. Empty means the caller has
        already authorised itself (``scripts/jarvis_people.py._authorise``,
        which is the keyboard/SSH path and the only one there is); a NAMED
        caller must be an owner. There is deliberately no spoken path to
        this and no socket command for it.
        """
        if by and not self.may_administer(by):
            return False, ("only the owner can change how somebody is "
                           "addressed")
        if value not in HONORIFICS:
            return False, ("%r is not a form of address; it is sir, ma'am, "
                           "or none. Ask the person which they want."
                           % (value,))
        p = self.person(label)
        if p is None:
            return False, "%s is not enrolled" % label
        p.honorific = value
        return True, ""

    def forget(self, label) -> Tuple[bool, str]:
        p = self.person(label)
        if p is None:
            return False, "%s is not enrolled" % label
        if p.role == ROLE_OWNER and len(self.owners()) == 1:
            return False, ("%s is the only owner; forgetting them would "
                           "leave nobody who can enrol anyone" % label)
        self.people = [x for x in self.people if x.label != p.label]
        return True, ""

    def set_secret(self, label, field_name, hashed) -> Tuple[bool, str]:
        """Store an ALREADY-HASHED fallback. This function never sees a
        plaintext, which is why it cannot leak one: hashing is
        ``jarvis/passphrase.hash_secret``'s job and the plaintext dies in
        the caller's frame."""
        if field_name not in ("phrase_hash", "code_hash"):
            return False, "%r is not a stored fallback" % (field_name,)
        p = self.person(label)
        if p is None:
            return False, "%s is not enrolled" % label
        if p.role != ROLE_OWNER:
            return False, ("only an owner has a way back in to set; %s is %s"
                           % (label, p.role or "not enrolled"))
        setattr(p, field_name, str(hashed or ""))
        return True, ""

    # ---------------------------------------- the pending code (weekly)
    def _owner_for_pending(self, label) -> Tuple[Optional[Person], str]:
        p = self.person(label)
        if p is None:
            return None, "%s is not enrolled" % label
        if p.role != ROLE_OWNER:
            return None, ("only an owner has a way back in to set; %s is %s"
                          % (label, p.role or "not enrolled"))
        return p, ""

    def set_pending_code(self, label, hashed, code_id, since) -> Tuple[bool, str]:
        """Store an ALREADY-HASHED weekly code BESIDE the current one.
        Never sees a plaintext, like set_secret. Replaces any pending
        already there (a forced re-push); never touches code_hash."""
        p, why = self._owner_for_pending(label)
        if p is None:
            return False, why
        if not PENDING_ID_RX.match(str(code_id or "")):
            return False, "a pending code needs an id of 8 hex characters"
        if not str(hashed or ""):
            return False, "a pending code needs a hash"
        p.pending_code_hash = str(hashed)
        p.pending_code_id = str(code_id)
        p.pending_code_since = str(since or "")
        return True, ""

    def promote_pending(self, label, code_id) -> Tuple[bool, str]:
        """The receipt came back (or he typed it): the pending hash becomes
        THE code and the previous one is gone. The id must match, so a
        receipt for a replaced push cannot promote the wrong hash."""
        p, why = self._owner_for_pending(label)
        if p is None:
            return False, why
        if not p.pending_code_hash:
            return False, "there is no pending code to promote"
        if p.pending_code_id != str(code_id or ""):
            return False, ("the pending code's id does not match (%s)"
                           % p.pending_code_id)
        p.code_hash = p.pending_code_hash
        p.pending_code_hash = ""
        p.pending_code_id = ""
        p.pending_code_since = ""
        return True, ""

    def drop_pending(self, label, code_id) -> Tuple[bool, str]:
        """No receipt, a failed send, a stale spool: the pending hash is
        forgotten and the current code stands untouched."""
        p, why = self._owner_for_pending(label)
        if p is None:
            return False, why
        if not p.pending_code_hash:
            return False, "there is no pending code to drop"
        if p.pending_code_id != str(code_id or ""):
            return False, ("the pending code's id does not match (%s)"
                           % p.pending_code_id)
        p.pending_code_hash = ""
        p.pending_code_id = ""
        p.pending_code_since = ""
        return True, ""

    def save(self) -> bool:
        if self.path is None:
            return False
        payload = {"format": FORMAT,
                   "people": [p.to_json() for p in self.people]}
        raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        try:
            _write_private(Path(self.path), raw)
        except OSError:
            log.exception("registry save failed: %s", self.path)
            return False
        # THE FINGERPRINT FOLLOWS THE WRITE. ``digest`` means "the bytes
        # this object last knew the file to hold", and save_checked refuses
        # when the file no longer matches it. Without this line a registry
        # that has saved once still carried the digest it LOADED with (or
        # "", for one built in memory), so its next save_checked compared
        # against stale bytes and refused a write nobody had raced --
        # measured 2026-09-06: scripts/jarvis_people.py add/set-role/forget
        # all returned "REFUSED: the people book changed since this command
        # started" against a registry that was simply built and saved.
        self.digest = _digest(raw)
        return True

    def save_checked(self, timeout_s: float = 10.0) -> Tuple[bool, str]:
        """Save ONLY if the file still holds the bytes this object was
        loaded from. ``(ok, why)``.

        For the writer that cannot hold the lock across its work: the
        terminal tool loads, asks questions at a prompt, and saves minutes
        later. Under the lock the file is read again and its fingerprint
        compared with ``digest``; a mismatch means another writer (the app,
        the weekly issuer) got there first, and writing this object over
        the top would silently undo what they wrote. It refuses and says
        so; the caller runs the command again against the new file.
        """
        if self.path is None:
            return False, "this registry has no file"
        path = Path(self.path)
        try:
            with registry_lock(path, timeout_s=timeout_s):
                try:
                    now = _digest(path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    now = ""
                except OSError as exc:
                    return False, ("%s could not be read (%s)"
                                   % (path, type(exc).__name__))
                if now != self.digest:
                    return False, ("the people book changed since this "
                                   "command started (another writer); "
                                   "nothing was written -- run it again")
                if not self.save():
                    return False, "the registry could not be written"
                self.digest = _digest(path.read_text(encoding="utf-8"))
                return True, ""
        except RegistryBusy as exc:
            return False, str(exc)


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ------------------------------------------------- the cross-process lock
# THREE PROCESSES WRITE people.json: the app (the drawer's rotate and the
# users tab), scripts/jarvis_people.py at a terminal, and -- since the
# weekly lane -- the issuer under a systemd timer. Until 2026-09-06 nothing
# locked between them; the app's own docstrings said so and shrank the
# window by re-reading immediately before each write. The window was
# milliseconds and the worst case one dead code, never a lockout, because
# every path leaves a working hash. This closes it anyway: an flock on
# <people.json>.lock, taken around load -> change -> save.
#
# Re-entrant ON ONE THREAD, and that is not a nicety: flock() is per open
# file description, so a second open() + flock() from the same process
# BLOCKS against the first. The app nests these (a users-tab write that
# then rotates), so a thread that already holds it just passes through.
PENDING_ID_RX = re.compile(r"^[0-9a-f]{8}$")
LOCK_POLL_S = 0.05
_lock_depth = threading.local()


class RegistryBusy(RuntimeError):
    """Another writer held the people book for the whole wait."""


@contextlib.contextmanager
def registry_lock(path, timeout_s: float = 10.0):
    """Hold <path>.lock (flock, exclusive) for the block. Raises
    RegistryBusy after ``timeout_s`` rather than waiting forever: a timer
    job or a drawer thread stuck behind a lock is worse than a refusal
    that says so."""
    depth = int(getattr(_lock_depth, "n", 0) or 0)
    if depth:
        _lock_depth.n = depth + 1
        try:
            yield
        finally:
            _lock_depth.n = depth
        return
    lock_path = Path(str(path) + ".lock")
    # NO IN-PROCESS MUTEX AROUND THIS. There was one, an RLock held for the
    # whole critical section, and it made ``timeout_s`` a lie for threads:
    # a second THREAD blocked on the mutex forever (only another PROCESS
    # ever reached the polling loop below), so a wedged drawer write would
    # hang the users tab instead of refusing it -- measured 2026-09-06, a
    # 5 s hold made a 0.3 s timeout wait the full five and then succeed.
    # flock() is per open file description, and two open() calls from ONE
    # process are separate descriptions that conflict with each other just
    # as two processes do, so the loop below serialises threads too. What
    # the thread-local depth above handles is the case flock cannot: the
    # SAME thread nesting these (a users-tab write that then rotates),
    # which would otherwise deadlock against itself.
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RegistryBusy(
                        "another writer holds the people book (%s); "
                        "nothing was changed" % lock_path.name)
                time.sleep(LOCK_POLL_S)
        _lock_depth.n = 1
        try:
            yield
        finally:
            _lock_depth.n = 0
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


def locked_update(path, change: Callable[["Registry"], Tuple[bool, str]],
                  *, timeout_s: float = 10.0) -> Tuple[bool, str]:
    """load -> change(registry) -> save, under the lock. ``(ok, why)``.

    NEVER RAISES. Refuses a registry that is not usable, for the reason
    admin_gate refuses it: a file that failed to parse loads as EMPTY, and
    saving that over it destroys whatever it held. The change callable
    answers ``(ok, why)`` like every Registry writer, and its refusal is
    handed back untouched.
    """
    p = Path(path)
    try:
        with registry_lock(p, timeout_s=timeout_s):
            reg = Registry.load(p)
            if not reg.usable:
                return False, reg.fault or "the registry could not be read"
            try:
                ok, why = change(reg)
            except Exception as exc:  # noqa: BLE001 - a boundary, never a raise
                log.exception("registry: a locked change failed")
                return False, "the change failed (%s)" % type(exc).__name__
            if not ok:
                return False, why
            if not reg.save():
                return False, "the people file could not be written; see the log"
            return True, ""
    except RegistryBusy as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 - a boundary, never a raise
        log.exception("registry: the locked update failed")
        return False, "the locked update failed (%s)" % type(exc).__name__


# -------------------------------------------------------- the one line
def startup_line(registry: Registry, mode: str, *, voice_ok: bool = False,
                 face_ok: bool = False, face_why: str = "") -> str:
    """One line at startup, ALWAYS, saying plainly which mode is live.

    A gate whose mode you have to infer from behaviour is a gate you find
    out about by being refused.
    """
    reg = registry
    if not getattr(reg, "usable", False) or str(mode) == "off":
        why = getattr(reg, "fault", "") or "there is nothing to recognise with"
        return ("owner-gate: OFF -- %s. Everyone is being answered." % why)
    owners = ", ".join(p.label for p in reg.owners()) or "nobody"
    legs = []
    legs.append("voice leg live" if voice_ok else "voice leg unavailable")
    if face_ok:
        legs.append("face leg live")
    else:
        legs.append("face leg unavailable"
                    + (" (%s)" % face_why if face_why else ""))
    tail = ("Nothing is being refused." if str(mode) == "shadow"
            else "An unrecognised speaker is refused.")
    return ("owner-gate: %s -- %d owner (%s), %s. %s"
            % (str(mode).upper(), len(reg.owners()), owners,
               ", ".join(legs), tail))
