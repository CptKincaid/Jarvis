#!/usr/bin/env python3
"""The address book, from a terminal: list, show, add, remove.

FILE-DIRECT. This reads and writes ~/.config/jarvis/contacts.json through
jarvis.contacts and nothing else -- no socket, no AssistantConfig.load(), no
running Jarvis -- so it works over `ssh spark` with Jarvis down, and it
never opens assistant.json at all (jarvis.contacts only needs the config
file's DIRECTORY). The send lane re-reads the book by stamp on every
resolve, so an add here is live on the next "email this to Heather" with
no restart.

    jarvis_contacts.py list [--json]
    jarvis_contacts.py show "Heather"            # what Jarvis would do with it
    jarvis_contacts.py add "Heather Smith" heather@example.com \\
        [--honorific Dr] [--alias "my advisor" ...] [--note "PhD advisor"]
    jarvis_contacts.py remove "Heather Smith" [--yes]

`show` runs THE SAME resolver the send path runs and prints the verdict --
FOUND / AMBIGUOUS / UNKNOWN, exit 0 / 3 / 2 -- so a spoken name can be
tested over ssh without a microphone. That is the numbers-only instrument
for the book.

No `set`: correct a row by editing the file, or remove + add. No import
from send_file.contacts: it has no entries. Refusals print "REFUSED: <why>"
and exit 2; a file that cannot be written exits 1. Addresses are printed in
full here because this is his own terminal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import contacts as book_mod                   # noqa: E402

EXIT_OK, EXIT_WRITE, EXIT_REFUSED, EXIT_AMBIGUOUS = 0, 1, 2, 3


def say(text=""):
    print(text)


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _row_line(c: book_mod.Contact) -> str:
    parts = [f"{c.name}  <{c.email}>"]
    if c.honorific:
        parts.append(c.honorific)
    if c.aliases:
        parts.append("aliases: " + ", ".join(c.aliases))
    if c.note:
        parts.append("note: " + c.note)
    return "  ".join(parts)


# ------------------------------------------------------------- the modes
def do_list(book: book_mod.Book, as_json: bool) -> int:
    book.refresh()
    if as_json:
        say(json.dumps(book.public(), indent=2, ensure_ascii=False))
        return EXIT_OK
    for c in book.contacts:
        say(_row_line(c))
    for s in book.skipped:
        say(f"BAD (not used): row {s.index + 1} {s.name or '(no name)'}: {s.why}")
    n = len(book.contacts)
    say(f"{n} {'person' if n == 1 else 'people'} in "
        f"{book_mod.display_path(book.path)}")
    return EXIT_OK


def do_show(book: book_mod.Book, name: str) -> int:
    res = book.resolve(name)
    if res.found:
        say(f"FOUND: {res.name} <{res.addr}> (matched on {res.matched_on})")
        return EXIT_OK
    if res.ambiguous:
        line = book_mod.which_line(name, res.candidates)
        say(f"AMBIGUOUS: {', '.join(res.candidates)} - Jarvis would ask "
            f"{line!r}")
        return EXIT_AMBIGUOUS
    say(f"UNKNOWN: nothing in the book for {name!r}")
    return EXIT_REFUSED


def do_add(book: book_mod.Book, args) -> int:
    row = {"name": args.name, "email": args.email,
           "honorific": args.honorific or "", "aliases": list(args.alias or ()),
           "note": args.note or ""}
    try:
        contact, why = book.add(row)
    except OSError as exc:
        say(f"REFUSED: the book could not be written ({exc.__class__.__name__})")
        return EXIT_WRITE
    if contact is None:
        say(f"REFUSED: {why}")
        return EXIT_REFUSED
    say(f"added {contact.name} <{contact.email}>")
    return EXIT_OK


def do_remove(book: book_mod.Book, args) -> int:
    res = book.resolve(args.name)
    if res.unknown:
        say(f"REFUSED: nothing in the book for {args.name!r}")
        return EXIT_REFUSED
    if res.ambiguous:
        names = res.candidates
        listed = ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 \
            else names[0]
        say(f"REFUSED: which one - {listed}? Give the full name")
        return EXIT_REFUSED
    contact = book.by_name(res.name)
    if contact is None:
        say(f"REFUSED: nothing in the book for {args.name!r}")
        return EXIT_REFUSED
    say(_row_line(contact))
    if not args.yes:
        if not (_isatty(sys.stdin) and _isatty(sys.stdout)):
            say("REFUSED: removing needs a terminal to confirm, or --yes")
            return EXIT_REFUSED
        try:
            answer = input(f"Remove {contact.name} <{contact.email}>? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            say("nothing removed")
            return EXIT_REFUSED
    try:
        gone, why = book.remove(contact.name, contact.email)
    except OSError as exc:
        say(f"REFUSED: the book could not be written ({exc.__class__.__name__})")
        return EXIT_WRITE
    if gone is None:
        say(f"REFUSED: {why}")
        return EXIT_REFUSED
    say(f"removed {gone.name} <{gone.email}>")
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    ls = sub.add_parser("list", help="every row, in file order")
    ls.add_argument("--json", action="store_true")
    sh = sub.add_parser("show", help="what the send lane would do with a name")
    sh.add_argument("name")
    ad = sub.add_parser("add")
    ad.add_argument("name", help='"Heather Smith"')
    ad.add_argument("email")
    ad.add_argument("--honorific", default="", help="Dr, Prof, Mr ...")
    ad.add_argument("--alias", action="append", default=[],
                    help='another spoken name: --alias "my advisor"')
    ad.add_argument("--note", default="")
    rm = sub.add_parser("remove")
    rm.add_argument("name")
    rm.add_argument("--yes", action="store_true",
                    help="skip the prompt (the name must still be unambiguous)")
    args = ap.parse_args(argv)

    book = book_mod.current()
    if args.mode == "list":
        return do_list(book, args.json)
    if args.mode == "show":
        return do_show(book, args.name)
    if args.mode == "add":
        return do_add(book, args)
    if args.mode == "remove":
        return do_remove(book, args)
    return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
