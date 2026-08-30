"""`python -m jarvis.ask "what's due this week"` -- ask the running Jarvis
from a shell and print what he answers.

Stdlib only, no jarvis imports beyond the socket path, so it works from a
bare SSH session (`ssh spark jarvis 'timer 5 minutes'` via scripts/jarvis),
from tmux, from a Claude Code hook or a VSS script. Replies go to stdout,
status lines to stderr; exit 0 when a reply came back, 2 when Jarvis is not
running, 3 when the turn ended with no reply (a timeout, an error).

    -q / --quiet      answer in text only; the soundbar stays silent
    -t / --timeout S  how long to wait for the answer (default 90)
    --json            print the raw JSON lines instead
    --status          the diagnostics line ("run diagnostics")
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_SOCK = Path(os.environ.get("JARVIS_LOG_DIR") or "/tmp/vss_voice") / "command.sock"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis", description=__doc__.split("\n\n")[0])
    ap.add_argument("text", nargs="*", help="what to ask (quoted or not)")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not speak the answer")
    ap.add_argument("-t", "--timeout", type=float, default=90.0)
    ap.add_argument("--json", action="store_true", help="raw JSON lines")
    ap.add_argument("--status", action="store_true", help="diagnostics line")
    ap.add_argument("--sock", default=os.environ.get("JARVIS_COMMAND_SOCK") or str(DEFAULT_SOCK))
    args = ap.parse_args(argv)
    text = "status" if args.status else " ".join(args.text).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        ap.print_usage(sys.stderr)
        return 1
    from jarvis.cmdsock import ask
    import json
    replied, reason = False, ""
    try:
        for msg in ask(args.sock, text, quiet=args.quiet, timeout=args.timeout):
            if args.json:
                print(json.dumps(msg), flush=True)
            kind = msg.get("kind")
            if kind == "reply":
                replied = True
                if not args.json:
                    print(msg.get("text", ""), flush=True)
            elif kind == "status" and not args.json:
                print(f"[{msg.get('level', 'info')}] {msg.get('text', '')}",
                      file=sys.stderr, flush=True)
            elif kind == "error" and not args.json:
                print(f"error: {msg.get('text', '')}", file=sys.stderr, flush=True)
            elif kind == "end":
                reason = str(msg.get("reason", ""))
    except ConnectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not replied:
        if reason and not args.json:
            print(f"(no reply: {reason})", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
