#!/usr/bin/env python3
"""Rehearse an email-a-file send. Prints numbers; opens no socket.

WHY THIS EXISTS
    Until it did, the only way to prove the chain -- a spoken phrase to one
    exact path, a name to one exact address, the read-back, the yes, the
    assembled MIME -- was to send a real message to a real person, and there
    is no undo for a rehearsal that turns out to have been live. So the
    transport here is `mail.DryRunSMTP`, which logs in, accepts the message,
    counts its bytes and DISCARDS it. `socket.socket.connect` is monkey-
    patched to refuse outright as a second layer, so if the transport were
    ever wrong the script fails loudly instead of quietly emailing someone.

WHAT IT PRINTS
    the resolved absolute path, its size in bytes, the first 16 hex digits
    of its SHA-256 (so two versions of one report are visibly different),
    the recipient address, which of the configured accounts it would go out
    as and at what address, the subject, and the size of the assembled
    message on the wire. Nothing else -- no body, no password, no server
    text.

    It does NOT write jarvis_memory/sent.jsonl. That audit records what
    Jarvis actually did, and a script anyone can run a hundred times must
    not be able to fill it. A rehearsal taken through the APP instead
    (JARVIS_MAIL_DRYRUN=1, said out loud, confirmed with a yes) DOES get a
    row, marked "dry_run": true -- because the app really did take that
    turn.

USAGE
    ~/vss_env/bin/python scripts/send_rehearsal.py "the lab report" Heather
    ~/vss_env/bin/python scripts/send_rehearsal.py "~/Desktop/x.pdf" \
        h.peyrovi@tamu.edu --account school --subject "Lab three"

    --json prints one machine-readable object instead of the sheet.

THIS SCRIPT CANNOT SEND. There is no flag that makes it, and adding one
would defeat the only reason it exists. The single live end-to-end send is
Hunter's to make himself, out loud, through Jarvis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import outbox                                   # noqa: E402
from jarvis.assistant_config import AssistantConfig         # noqa: E402
from jarvis.tools import mail as mail_mod                   # noqa: E402


def _seal_the_network():
    """Refuse every outbound connection for the life of this process.

    Belt and braces over the dry-run transport: this script exists to prove
    a send WITHOUT sending, so the strongest statement it can make is that
    no socket could have been opened even if the transport were wrong.
    """
    def _refuse(sock, address):
        raise ConnectionRefusedError(
            f"send_rehearsal.py may not open a connection ({address!r}). "
            "This is a rehearsal; nothing here is allowed on the wire.")

    socket.socket.connect = _refuse


def _sha_head(path: Path, digits: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:digits]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Rehearse an email-a-file send. Sends nothing, ever.")
    ap.add_argument("file", help='the file phrase, as spoken: '
                                 '"the lab report on my desktop"')
    ap.add_argument("recipient", help="a name from your contacts, or an "
                                      "address said in full")
    ap.add_argument("--account", default="", help="personal / work / school")
    ap.add_argument("--subject", default="", help="the subject line")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object instead of the sheet")
    args = ap.parse_args(argv)

    _seal_the_network()

    # AssistantConfig.load(), never AssistantConfig(): the bare constructor
    # returns DEFAULTS, which has no accounts, and a rehearsal that reported
    # "no mailbox configured" against a perfectly good config would be a
    # false result about the one thing this script is for.
    cfg = AssistantConfig.load()
    prep = outbox.prepare(cfg, None, args.file, args.recipient,
                          account_hint=args.account, subject=args.subject)
    if prep.draft is None:
        # A question or a refusal is a RESULT, not a failure of the script:
        # it is exactly what Jarvis would have said out loud.
        out = {"ok": False, "would_say": prep.ask, "status": prep.status,
               "candidates": [str(p) for p in prep.candidates]}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print("REHEARSAL -- nothing sent, nothing assembled.")
            print(f"  Jarvis would say: {prep.ask}")
            print(f"  status:           {prep.status}")
            for path in prep.candidates:
                print(f"  candidate:        {path}")
        return 1

    draft = prep.draft
    # The MIME is assembled through the SAME builder the send uses, so the
    # byte count below is the real one and not an estimate.
    msg = mail_mod.build_message(
        draft.account["address"], draft.to_addr, draft.subject, draft.body,
        attachment=draft.path,
        from_name=str(draft.account.get("from_name") or ""))
    wire = len(bytes(msg))

    # And now the transport itself, exercised end to end with the fake.
    transport = mail_mod.DryRunSMTP
    before = len(transport.made)
    mail_mod.send_message(draft.account, draft.to_addr, draft.subject,
                          draft.body, attachment=draft.path, smtp=transport)
    rehearsed = len(transport.made) - before

    facts = {
        "ok": True,
        "rehearsal": True,
        "path": str(draft.path),
        "bytes": int(draft.size),
        "sha256_head": _sha_head(draft.path),
        "modified": draft.mtime,
        "to": draft.to_addr,
        "to_name": draft.to_name,
        "account": draft.account_label,
        "from": draft.account["address"],
        "subject": draft.subject,
        "mime_bytes": wire,
        "transports_used": rehearsed,
        "sockets_opened": 0,
        "read_back": outbox.read_back(draft),
    }
    if args.json:
        print(json.dumps(facts, indent=2))
        return 0

    print("REHEARSAL ONLY -- nothing left the machine.")
    print()
    print(_card(draft))
    print()
    print(f"sha256 head   {facts['sha256_head']}")
    print(f"MIME on wire  {wire:,} bytes "
          f"({wire / max(draft.size, 1):.2f}x the file)")
    print(f"transports    {rehearsed} rehearsed, 0 sockets opened")
    print()
    print("Jarvis would say:")
    print(f"  {facts['read_back']}")
    print(f"  ...and on a yes: {outbox.REHEARSAL_LINE}")
    return 0



def _card(draft) -> str:
    """The facts the EAR cannot check, shown rather than spoken.

    outbox.card() is deliberately not carried from the source branch --
    jarvis-v3 shows these through CommandResult.display_only instead --
    so the rehearsal builds its own from fields that still exist. The
    address is shown in full ON PURPOSE here: this script exists to let
    him check the To: header before a real send, it prints to his own
    terminal, and it writes nothing to disk.
    """
    from datetime import datetime as _dt
    try:
        stamp = _dt.fromtimestamp(float(draft.mtime)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        stamp = "unknown"
    label = mail_mod.account_label(draft.account) or "?"
    from_addr = str((draft.account or {}).get("address") or "?")
    return ("Send  %s\n"
            "      %s bytes  \u00b7  modified %s\n"
            "  to  %s\n"
            "from  %s  \u00b7  %s\n"
            "subj. %s" % (draft.path, f"{int(draft.size):,}", stamp,
                          draft.to_addr, label, from_addr, draft.subject))


if __name__ == "__main__":
    raise SystemExit(main())
