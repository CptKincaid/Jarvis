#!/usr/bin/env python3
"""Who Jarvis knows: list, enrol, change a role, set the two fallbacks.

KEYBOARD/SSH ONLY. There is no socket command and no spoken command for any
of this, which is the whole point of the override code: it can never be
overheard or replayed because it never touches a microphone.

THIS IS RECOGNITION, NOT A LOCK. Jarvis answers the person he recognises. A
photograph defeats the face check and a recording defeats the voice check.
Anybody who can already type at this machine can edit ``people.json`` or
delete it, and deleting it turns the gate off -- so the override code below
is A WAY BACK IN FOR THE OWNER, not a defence against them, and this file
will not pretend otherwise.

CONSENT IS NOT REINVENTED HERE. Enrolling somebody else takes THEIR
agreement, and this calls ``jarvis/consent.py`` -- the one file holding the
words and the "type your own label, exactly" rule -- rather than writing a
second one that could drift. Its terminal taker still requires stdin AND
stdout to be terminals, and the text it shows is the one that matches what
this command stores: a name, a role and a face LABEL, and no measurement of
anybody. The camera's own ceremony stays in ``scripts/face_enrol.py``.

THE FORM OF ADDRESS IS TYPED, NEVER INFERRED. ``add`` REQUIRES
``--honorific`` and has no default. There is no name list in this repo, no
gender table, and no code path that reads a name and produces "sir" or
"ma'am": Mara and Heather are addressed as "ma'am" because Hunter typed
``--honorific maam``, and for no other reason. "guess and let them correct
it" is not on offer -- it is wrong in front of a real person.

    jarvis_people.py list
    jarvis_people.py add heather --first Heather --last Vance --role known \
        --honorific maam --face heather --face-dim 512
    jarvis_people.py set-honorific heather --honorific maam
    jarvis_people.py set-role heather --role known
    jarvis_people.py set-phrase          # the spoken way back in
    jarvis_people.py set-code            # the typed break-glass
    jarvis_people.py forget heather
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import consent                                # noqa: E402
from jarvis import gate as gate_mod                       # noqa: E402
from jarvis import identity as identity_mod               # noqa: E402
from jarvis import knightfall_weekly as weekly_mod        # noqa: E402
from jarvis import passphrase as pp                       # noqa: E402
from jarvis.assistant_config import AssistantConfig       # noqa: E402
from jarvis.config import PATHS                           # noqa: E402
from jarvis.identity import (ROLE_KNOWN, ROLE_OWNER, LABEL_RX,  # noqa: E402
                             Person, Registry, honorific_from_word,
                             owner_label, startup_line)


def say(text=""):
    print(text)


def _isatty(stream) -> bool:
    """A stream that cannot answer is NOT a terminal. This gates both the
    consent prompt and the two secret prompts, so the unknown case fails
    closed -- the same rule scripts/face_enrol.py already writes down."""
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _keyboard_only(what: str) -> str:
    if not (_isatty(sys.stdin) and _isatty(sys.stdout)):
        return ("%s can only be done at a terminal: it has to be typed and "
                "read, not piped." % what)
    return ""


def _save(reg: Registry) -> bool:
    """Every write here lands through ``Registry.save_checked``: this tool
    loads the file, asks questions at a prompt, and saves minutes later,
    and since 2026-09-06 the app and the weekly Knightfall issuer may
    have written in between. A save that finds the file changed refuses
    and says so rather than writing this process's memory over theirs;
    the command is simply run again."""
    ok, why = reg.save_checked()
    if not ok:
        say("REFUSED: %s" % why)
    return ok


# ------------------------------------------------------- who is asking
def _authorise(reg: Registry) -> tuple:
    """``(ok, why)`` -- may whoever is at this keyboard administer the gate?

    THE DECISION IS NOT MADE HERE. ``gate.admin_gate`` makes it, because
    the users tab (jarvis/ui/users_page.py) asks the same question and two
    implementations of "may this keyboard administer the gate" is exactly
    how the two drift apart. This function still does its own ASKING --
    getpass, a tty check -- and ``check_override_code`` is still the only
    thing that checks a code.

    FOUR CASES, and the first two are the ones that keep this from becoming
    a brick:

    * NO USABLE REGISTRY, or no owner in it. Anybody at the keyboard may
      create the FIRST owner, because otherwise a fresh install is a brick
      and enrolment itself is unreachable.
    * AN OWNER EXISTS BUT HAS SET NO OVERRIDE CODE. Allowed, and said out
      loud. Demanding a code he never chose would lock him out of the very
      flow that sets one.
    * AN OWNER HAS SET A CODE. It is asked for, never echoed, and rate
      limited on its own counter -- burning the spoken passphrase's attempts
      must not close the break-glass.
    * THE FILE IS THERE AND BROKEN. REFUSED, and this one is new (2026-09-05).
      It used to fall into the first case, so `add` on a registry that failed
      to parse wrote a fresh one-row file over the top of everyone in it.

    A TYPED WEEKLY CODE IS PROMOTED HERE TOO (2026-09-06). It asks
    ``check_override_code_leg``, the same function the drawer and the users
    tab ask, so that a code out of Sunday's backup email is not left
    pending after it has been used -- he can only have that code from the
    email, so typing it is the receipt in person. Until this it used the
    two-tuple ``check_override_code``, which admits on the pending hash and
    says nothing, and the pending sat there until Wednesday dropped it.
    """
    state, why = gate_mod.admin_gate(reg)
    if state == gate_mod.ADMIN_REFUSE:
        return False, why
    if state == gate_mod.ADMIN_FIRST:
        return True, "first"
    if state == gate_mod.ADMIN_NOCODE:
        return True, "nocode"
    why = _keyboard_only("the override code")
    if why:
        return False, why
    code = getpass.getpass("Override code: ")
    who, reason, leg = gate_mod.check_override_code_leg(
        reg, code, attempts=pp.Attempts(limit=pp.CODE_LIMIT,
                                        window_s=pp.CODE_WINDOW))
    del code
    if not who:
        return False, reason
    if leg == gate_mod.LEG_PENDING:
        _promote_weekly(reg, who)
    return True, who


def _promote_weekly(reg: Registry, who: str) -> None:
    """He typed THIS WEEK'S code: promote it, under the file lock, and note
    it in the weekly lane's state so the pull knows the receipt is already
    handled.

    IT DOES NOT ROTATE AND MAILS NOTHING -- his decision #3, 2026-09-06.
    The DRAWER burns a typed weekly code (it mails the next one, as it does
    for any accepted code); the users tab's unlock and this terminal tool
    do not, because an administrative unlock that posted a fresh code every
    time he ran `forget` would be a mailbox full of live break-glass codes.

    THE REGISTRY THE CALLER HOLDS IS REFRESHED. This writes through
    ``locked_update``, which loads its own copy; leaving ``reg`` as it was
    would leave it carrying the digest of a file that has just changed, and
    ``save_checked`` would then refuse the very command the code authorised
    with "the people book changed since this command started".

    Every failure here is a line and nothing more: both codes simply stay
    honoured until the weekly pull resolves them.
    """
    person = reg.person(who)
    code_id = getattr(person, "pending_code_id", "") if person else ""
    if not code_id:
        return
    ok, why = identity_mod.locked_update(
        reg.path, lambda r: r.promote_pending(who, code_id))
    if not ok:
        say("note: that was this week's code from the backup email, but it "
            "could not be promoted (%s). Both codes still work." % why)
        return
    fresh = Registry.load(reg.path)
    if fresh.usable:
        reg.people = fresh.people
        reg.digest = fresh.digest
    try:
        weekly_mod.note_promoted_by_use(PATHS.KNIGHTFALL_WEEKLY, code_id)
    except Exception:  # noqa: BLE001 - narration, never the command
        pass
    say("note: that was this week's code from the backup email, so it is now "
        "your code and the previous one no longer works. Nothing was mailed: "
        "only the Knightfall drawer rotates a code you type.")


def _read_twice(prompt: str, check, transform=None) -> tuple:
    """A secret, typed twice, never echoed and never printed back.

    ``(hashed, why)``. THE ONLY THING THAT LEAVES THIS FUNCTION ON SUCCESS
    IS A SALTED HASH: the plaintext is never returned, never logged, and
    never taken from argv, where ``ps`` would show it to the whole machine.

    ``transform`` is the spoken phrase's normalisation. The phrase comes
    back through Whisper, which punctuates and capitalises as it pleases, so
    what is STORED has to be the normalised form or the phrase could never
    match itself.
    """
    first = getpass.getpass(prompt)
    ok, why = check(first)
    if not ok:
        del first
        return "", why
    again = getpass.getpass("Again, to be sure: ")
    if first != again:
        del first, again
        return "", "those did not match; nothing was changed"
    hashed = pp.hash_secret(transform(first) if transform else first)
    del first, again
    return hashed, ""


# ------------------------------------------------------------ the modes
def do_list(reg: Registry, cfg) -> int:
    say(startup_line(reg, "shadow", voice_ok=True, face_ok=False,
                     face_why="not asked from here"))
    say("registry   %s" % (reg.path or PATHS.OWNER_REGISTRY))
    if not reg.people:
        say("           nobody is enrolled")
        return 0
    for p in reg.people:
        row = p.redacted()
        say("  %-16s %-6s %-6s voice=%-5s face=%-12s phrase=%-5s code=%-5s %s"
            % (row["label"], row["role"], row["honorific"] or "none",
               row["voice"], row["face"] or "-",
               row["has_phrase"], row["has_code"], row["consent"] or "-"))
        full = p.full_name()
        if full:
            say("  %-16s %s" % ("", full))
    return 0


def do_add(reg: Registry, cfg, args) -> int:
    label = str(args.label or "").strip().lower()
    if not LABEL_RX.match(label):
        say("REFUSED: %r is not a label a gallery or a registry can store"
            % label)
        return 2
    # THE FORM OF ADDRESS IS NEVER INFERRED. argparse already makes the
    # flag required with no default; this is the second half of the same
    # rule, refusing anything that is not one of the three typed choices.
    hon, why = honorific_from_word(args.honorific)
    if why:
        say("REFUSED: add: %s. It is not inferred from a name." % why)
        return 2
    role = ROLE_OWNER if args.role == ROLE_OWNER else ROLE_KNOWN
    if args.voice and role != ROLE_OWNER:
        say("REFUSED: add: --voice can only be set on the owner. There is "
            "one voiceprint on this machine and marking a second person "
            "voice-enrolled would make the voice leg name the wrong person.")
        return 2
    owner = owner_label(cfg)
    if role == ROLE_KNOWN:
        # Somebody else's data. Their agreement, taken by the ONE rule
        # (jarvis/consent.py) rather than a second one -- and with the text
        # that matches what `add` actually stores. It writes a NAME, a ROLE
        # and a face LABEL; it captures nothing, so the face paragraph
        # ("128 numbers per take") over-claimed here. The lens has its own
        # ceremony in scripts/face_enrol.py and that is unchanged.
        ok, how = consent.take_at_terminal(
            label, what=consent.WHAT_ROW, owner=owner, say=say,
            fields={"who": label, "root": str(PATHS.FACE_GALLERY)})
        if not ok:
            say("REFUSED: %s" % how)
            return 2
    else:
        how = consent.HOW_OWNER
    person = Person(label=label, name=str(args.name or "").strip(),
                    first=str(args.first or "").strip(),
                    last=str(args.last or "").strip(), honorific=hon,
                    role=role, voice=bool(args.voice),
                    face=str(args.face or "").strip().lower(),
                    face_dim=int(args.face_dim or 0), consent=how)
    ok, why = reg.add_person(person,
                             confirm_existing_owner=args.confirm_owner)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not _save(reg):
        return 1
    say("enrolled %s as %s (addressed as %s, consent: %s)"
        % (label, role, hon or "no form of address", how))
    if person.face and not _gallery_has(person.face):
        # Said out loud rather than discovered later: the row exists but
        # the face leg has nothing to name them with until they stand in
        # front of the camera and type their own name.
        say("note: the face gallery has no label %r yet. The row is stored, "
            "but the face leg will say nothing about %s until they are "
            "enrolled at the camera: scripts/face_enrol.py --label %s"
            % (person.face, label, person.face))
    return 0


def _gallery_has(face_label: str) -> bool:
    """Is there a face under that label? A gallery that cannot be read
    answers True, because a warning nobody can verify is just noise.

    NOTHING HERE OPENS A CAMERA OR READS A FRAME. It reads the enrolled
    LABELS -- names and counts -- which is all the warning needs.
    """
    try:
        from jarvis import facegallery
        gal = facegallery.default_gallery()
        gal.load()
        names = gal.labels()
        return not names or str(face_label) in set(names)
    except Exception:  # noqa: BLE001 - a gallery that cannot say says yes
        return True


def do_set_honorific(reg: Registry, args) -> int:
    hon, why = honorific_from_word(args.honorific)
    if why:
        say("REFUSED: set-honorific: %s. Ask the person which they want."
            % why)
        return 2
    ok, why = reg.set_honorific(args.label, hon)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not _save(reg):
        return 1
    say("%s is addressed as %s" % (args.label, hon or "no form of address"))
    return 0


def do_set_role(reg: Registry, args) -> int:
    ok, why = reg.set_role(args.label, args.role,
                           confirm_existing_owner=args.confirm_owner)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not _save(reg):
        return 1
    say("%s is now %s" % (args.label, args.role))
    return 0


def do_forget(reg: Registry, args) -> int:
    ok, why = reg.forget(args.label)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not _save(reg):
        return 1
    say("%s is forgotten. Their face stays in the gallery; "
        "scripts/face_enrol.py --forget removes that." % args.label)
    return 0


def do_secret(reg: Registry, cfg, args, which: str) -> int:
    label = str(args.label or "").strip().lower() or owner_label(cfg)
    person = reg.person(label)
    if person is None:
        say("REFUSED: %s is not enrolled" % label)
        return 2
    if person.role != ROLE_OWNER:
        say("REFUSED: only an owner has a way back in to set")
        return 2
    why = _keyboard_only("setting a way back in")
    if why:
        say("REFUSED: %s" % why)
        return 2
    if which == "phrase":
        say("The SPOKEN passphrase -- the way back in when the camera is off "
            "and your voice will not match: ill, in the dark, or turned "
            "away. It is said out loud, so it can be overheard; that is "
            "accepted. At least %d letters and digits once punctuation is "
            "dropped." % pp.MIN_PHRASE_LEN)
        field = "phrase_hash"
        hashed, why = _read_twice("Passphrase: ", pp.phrase_ok,
                                  transform=pp.normalise_spoken)
    else:
        say("The TYPED override code -- break-glass, for when even the "
            "passphrase fails. It is NEVER accepted over the microphone, so "
            "it cannot be overheard or replayed. At least %d characters; "
            "letters, digits or both, whatever you like. No other rule, "
            "because anyone already at this keyboard can edit the registry "
            "anyway, and pretending otherwise would be a lie."
            % pp.MIN_CODE_LEN)
        field = "code_hash"
        hashed, why = _read_twice("Override code: ", pp.code_ok)
    if why:
        say("REFUSED: %s" % why)
        return 2
    ok, why = reg.set_secret(label, field, hashed)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not _save(reg):
        return 1
    # Never printed back, never logged, never echoed.
    say("set. It is stored salted-hashed; nothing here can read it back.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("label")
    a.add_argument("--name", default="")
    a.add_argument("--first", default="", help="first name, for the spoken "
                                               "sign-in")
    a.add_argument("--last", default="", help="last name, for the spoken "
                                              "sign-in")
    # REQUIRED, AND NO DEFAULT. A row cannot be created without somebody
    # typing the choice; nothing in this repo infers it from a name.
    a.add_argument("--honorific", required=True,
                   choices=["sir", "maam", "none"],
                   help="how they are addressed. Required: never inferred.")
    a.add_argument("--role", default=ROLE_KNOWN, choices=[ROLE_OWNER, ROLE_KNOWN])
    a.add_argument("--face", default="")
    a.add_argument("--face-dim", type=int, default=0, dest="face_dim")
    a.add_argument("--voice", action="store_true")
    a.add_argument("--confirm-owner", default=None, dest="confirm_owner",
                   help="the EXISTING owner's label; required to make a second")
    r = sub.add_parser("set-role")
    r.add_argument("label")
    r.add_argument("--role", required=True, choices=[ROLE_OWNER, ROLE_KNOWN])
    r.add_argument("--confirm-owner", default=None, dest="confirm_owner")
    h = sub.add_parser("set-honorific")
    h.add_argument("label")
    h.add_argument("--honorific", required=True,
                   choices=["sir", "maam", "none"])
    f = sub.add_parser("forget")
    f.add_argument("label")
    for name in ("set-phrase", "set-code"):
        sp = sub.add_parser(name)
        sp.add_argument("label", nargs="?", default="")
    args = ap.parse_args(argv)

    cfg = AssistantConfig.load()
    reg = Registry.load()
    if args.mode == "list":
        return do_list(reg, cfg)

    ok, who = _authorise(reg)
    if not ok:
        say("REFUSED: %s" % who)
        return 2
    if who == "first":
        say("note: nobody is enrolled yet, so the gate is OFF and anyone at "
            "this keyboard can enrol the first owner. That is deliberate -- "
            "otherwise a fresh install could never be set up.")
    elif who == "nocode":
        say("note: no override code has been set, so this is allowed at the "
            "keyboard. `set-code` changes that. It is a way back in for "
            "you, not a barrier to anyone already sitting here.")

    if args.mode == "add":
        return do_add(reg, cfg, args)
    if args.mode == "set-role":
        return do_set_role(reg, args)
    if args.mode == "set-honorific":
        return do_set_honorific(reg, args)
    if args.mode == "forget":
        return do_forget(reg, args)
    if args.mode == "set-phrase":
        return do_secret(reg, cfg, args, "phrase")
    if args.mode == "set-code":
        return do_secret(reg, cfg, args, "code")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
