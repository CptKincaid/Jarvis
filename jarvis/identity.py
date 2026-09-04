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

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        out["phrase_hash"] = self.phrase_hash
        out["code_hash"] = self.code_hash
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


@dataclass
class Registry:
    """The people Jarvis knows. ``usable`` is the gate's off switch."""

    path: Optional[Path] = None
    people: List[Person] = field(default_factory=list)
    usable: bool = False
    fault: str = ""

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
            return reg
        except OSError as exc:
            reg.fault = ("%s could not be read (%s)"
                         % (p, type(exc).__name__))
            return reg
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 - any parse failure is one fault
            reg.fault = "%s is not readable JSON" % p
            return reg
        if not isinstance(data, dict) or not isinstance(data.get("people"), list):
            reg.fault = "%s is not shaped like a registry" % p
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
            # THE INVARIANT THAT STOPS THE VOICE LEG NAMING THE WRONG
            # PERSON. jarvis/speaker.py holds ONE voiceprint and ONE
            # centroid and has no notion of a label at all, so "matched"
            # can only ever mean the owner. A second row marked
            # voice-enrolled would make every match come back as them.
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

    def save(self) -> bool:
        if self.path is None:
            return False
        payload = {"format": FORMAT,
                   "people": [p.to_json() for p in self.people]}
        try:
            _write_private(Path(self.path),
                           json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("registry save failed: %s", self.path)
            return False
        return True


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
