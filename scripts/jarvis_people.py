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

    jarvis_people.py list
    jarvis_people.py add heather --name Heather --role known --face heather
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
from jarvis import passphrase as pp                       # noqa: E402
from jarvis.assistant_config import AssistantConfig       # noqa: E402
from jarvis.config import PATHS                           # noqa: E402
from jarvis.identity import (ROLE_KNOWN, ROLE_OWNER, LABEL_RX,  # noqa: E402
                             Person, Registry, owner_label, startup_line)


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
    who, reason = gate_mod.check_override_code(
        reg, code, attempts=pp.Attempts(limit=pp.CODE_LIMIT,
                                        window_s=pp.CODE_WINDOW))
    del code
    if not who:
        return False, reason
    return True, who


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
        say("  %-16s %-6s voice=%-5s face=%-12s phrase=%-5s code=%-5s %s"
            % (row["label"], row["role"], row["voice"], row["face"] or "-",
               row["has_phrase"], row["has_code"], row["consent"] or "-"))
    return 0


def do_add(reg: Registry, cfg, args) -> int:
    label = str(args.label or "").strip().lower()
    if not LABEL_RX.match(label):
        say("REFUSED: %r is not a label a gallery or a registry can store"
            % label)
        return 2
    role = ROLE_OWNER if args.role == ROLE_OWNER else ROLE_KNOWN
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
                    role=role, voice=bool(args.voice),
                    face=str(args.face or "").strip().lower(),
                    face_dim=int(args.face_dim or 0), consent=how)
    ok, why = reg.add_person(person,
                             confirm_existing_owner=args.confirm_owner)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not reg.save():
        say("REFUSED: the registry could not be written")
        return 1
    say("enrolled %s as %s (consent: %s)" % (label, role, how))
    return 0


def do_set_role(reg: Registry, args) -> int:
    ok, why = reg.set_role(args.label, args.role,
                           confirm_existing_owner=args.confirm_owner)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not reg.save():
        say("REFUSED: the registry could not be written")
        return 1
    say("%s is now %s" % (args.label, args.role))
    return 0


def do_forget(reg: Registry, args) -> int:
    ok, why = reg.forget(args.label)
    if not ok:
        say("REFUSED: %s" % why)
        return 2
    if not reg.save():
        say("REFUSED: the registry could not be written")
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
    if not reg.save():
        say("REFUSED: the registry could not be written")
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
    if args.mode == "forget":
        return do_forget(reg, args)
    if args.mode == "set-phrase":
        return do_secret(reg, cfg, args, "phrase")
    if args.mode == "set-code":
        return do_secret(reg, cfg, args, "code")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
