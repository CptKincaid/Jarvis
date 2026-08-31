"""`python -m jarvis.ask "what's due this week"` -- ask the running Jarvis
from a shell and print what he answers.

Stdlib only, no jarvis imports beyond the socket path, so it works from a
bare SSH session (`ssh spark jarvis 'timer 5 minutes'` via scripts/jarvis),
from tmux, from a Claude Code hook or a VSS script. Replies go to stdout,
status lines to stderr; exit 0 when a reply came back, 2 when Jarvis is not
running, 3 when the turn ended with no reply (a timeout, an error), 4 when a
clip could not be recorded or read.

    -q / --quiet      answer in text only; the soundbar stays silent
    -t / --timeout S  how long to wait for the answer (default 90)
    --json            print the raw JSON lines instead
    --status          the diagnostics line ("run diagnostics")

The intercom (jarvis/intercom.py) sends a recorded clip instead of text, so
a question needs neither the wake word nor a keyboard:

    --send-audio FILE  a wav / ogg / opus / flac clip ("-" = stdin)
    --listen SECS      record SECS seconds here, then send that
    --speak            also answer aloud on the Spark (off by default)

From the phone, with Termux + termux-api installed, both legs are one line
over the SSH session that already exists:

    termux-microphone-record -q >/dev/null 2>&1
    termux-microphone-record -d -l 8 -e opus -f $HOME/j.ogg
    sleep 9 && ssh spark 'jarvis --send-audio -' < $HOME/j.ogg

`--listen` does the same thing when this client itself has a microphone
(termux-microphone-record, else parecord, else arecord). Note that
termux-microphone-record defaults to AAC, which libsndfile cannot read:
`-e opus` (or `-e wav`) is not optional.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_SOCK = Path(os.environ.get("JARVIS_LOG_DIR") or "/tmp/vss_voice") / "command.sock"
# Enough for a push-to-talk clip; the server refuses more than 10 MB anyway.
MAX_CLIP_BYTES = 10 * 1024 * 1024
RECORD_SLACK_S = 3.0            # how long past `secs` a recorder may take


def read_clip(path: str) -> bytes:
    """The bytes of a clip file, or of stdin for "-"."""
    if path == "-":
        data = sys.stdin.buffer.read()
    else:
        data = Path(path).expanduser().read_bytes()
    if not data:
        raise ValueError("the clip was empty")
    if len(data) > MAX_CLIP_BYTES:
        raise ValueError(f"the clip is {len(data) // 1048576} MB; the limit is "
                         f"{MAX_CLIP_BYTES // 1048576} MB")
    return data


def _recorder_argv(seconds: float, out_dir: Path):
    """The first available local recorder as (argv, needs_wait, path).

    Termux first (this is the phone case it exists for), then PulseAudio /
    PipeWire, then ALSA. termux-microphone-record RETURNS at once and
    records in the background, hence needs_wait."""
    if shutil.which("termux-microphone-record"):
        # -e opus, never the AAC default: libsndfile cannot read AAC.
        path = out_dir / "jarvis-intercom.ogg"
        return (["termux-microphone-record", "-d", "-l", str(int(seconds)),
                 "-e", "opus", "-f", str(path)], True, path)
    path = out_dir / "jarvis-intercom.wav"
    if shutil.which("parecord"):
        return (["parecord", "--channels=1", "--rate=16000",
                 "--file-format=wav", str(path)], False, path)
    if shutil.which("arecord"):
        return (["arecord", "-q", "-f", "S16_LE", "-c", "1", "-r", "16000",
                 "-d", str(int(seconds)), str(path)], False, path)
    return (None, False, path)


def record_clip(seconds: float, out_dir=None) -> bytes:
    """Record `seconds` of audio here and return the file's bytes."""
    seconds = max(1.0, float(seconds))
    tmp = Path(out_dir or tempfile.gettempdir())
    tmp.mkdir(parents=True, exist_ok=True)
    argv, needs_wait, path = _recorder_argv(seconds, tmp)
    if argv is None:
        raise RuntimeError("no recorder found (termux-microphone-record, "
                           "parecord or arecord)")
    if path.exists():
        path.unlink()               # a stale clip must never be sent as new
    try:
        if needs_wait:
            subprocess.run(argv, capture_output=True, timeout=15, check=False)
            time.sleep(seconds + 0.5)
            # Stop it even though -l should have: a recorder still holding
            # the file would hand us a truncated container.
            subprocess.run([argv[0], "-q"], capture_output=True, timeout=15,
                           check=False)
        else:
            # parecord has no duration flag, so the timeout IS the duration;
            # arecord stops itself and exits well before it.
            try:
                subprocess.run(argv, capture_output=True, timeout=seconds)
            except subprocess.TimeoutExpired:
                pass
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"recording failed: {exc}") from exc
    for _ in range(int(RECORD_SLACK_S * 10)):
        if path.exists() and path.stat().st_size > 1024:
            break
        time.sleep(0.1)
    if not path.exists() or path.stat().st_size <= 44:
        raise RuntimeError("the recorder produced no audio")
    return path.read_bytes()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis", description=__doc__.split("\n\n")[0])
    ap.add_argument("text", nargs="*", help="what to ask (quoted or not)")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not speak the answer")
    ap.add_argument("-t", "--timeout", type=float, default=90.0)
    ap.add_argument("--json", action="store_true", help="raw JSON lines")
    ap.add_argument("--status", action="store_true", help="diagnostics line")
    ap.add_argument("--send-audio", metavar="FILE",
                    help="send a wav/ogg/opus/flac clip instead of text (- = stdin)")
    ap.add_argument("--listen", type=float, metavar="SECS", default=0.0,
                    help="record SECS seconds here and send that")
    ap.add_argument("--speak", action="store_true",
                    help="intercom: also answer aloud on the Spark")
    ap.add_argument("--sock", default=os.environ.get("JARVIS_COMMAND_SOCK") or str(DEFAULT_SOCK))
    args = ap.parse_args(argv)

    audio, text = None, ""
    if args.send_audio or args.listen:
        # Audio first: with --send-audio - the stdin bytes are the clip, and
        # the text branch below would swallow them as a question.
        try:
            audio = (read_clip(args.send_audio) if args.send_audio
                     else record_clip(args.listen))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
    else:
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
        for msg in ask(args.sock, text, quiet=args.quiet, timeout=args.timeout,
                       audio=audio, speak=args.speak):
            if args.json:
                print(json.dumps(msg), flush=True)
            kind = msg.get("kind")
            if kind == "reply":
                replied = True
                if not args.json:
                    print(msg.get("text", ""), flush=True)
            elif kind == "heard" and not args.json:
                # stderr, not stdout: a script piping the answer somewhere
                # must not find the question in it.
                print(f"[heard] {msg.get('text', '')}", file=sys.stderr, flush=True)
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
