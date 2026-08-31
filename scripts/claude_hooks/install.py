#!/usr/bin/env python3
"""Merge the Jarvis narration hooks into ~/.claude/settings.json.

Run it yourself, once:

    ~/vss_env/bin/python ~/Jarvis/scripts/claude_hooks/install.py
    ~/vss_env/bin/python ~/Jarvis/scripts/claude_hooks/install.py --dry-run
    ~/vss_env/bin/python ~/Jarvis/scripts/claude_hooks/install.py --uninstall

It never runs on its own (nothing in Jarvis calls it): the settings file is
the user's, and a hook that appears there unasked is exactly the kind of
surprise this repo avoids.  The merge is idempotent — our entries are
recognised by the narrate.py path in their command — and every other hook,
permission or setting in the file is left untouched.  A backup is written
beside the file before the first change.

These hooks are only the outbound half: they make your OWN Claude sessions
speak through Jarvis.  The inbound half — a session running `jarvis -q "..."`
to read his calendar, Canvas due dates, notes or memory — is automatic in panes
Jarvis launches (claude_session.SYSTEM_SUFFIX) but NOT here, because your own
sessions get no system suffix from us; docs/assistant-setup.md section 40 has
the four lines to paste into ~/.claude/CLAUDE.md if you want it.
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NARRATE = HERE / "narrate.py"
DEFAULT_SETTINGS = Path.home() / ".claude" / "settings.json"
TIMEOUT_S = 10          # the script exits in milliseconds; this only bounds a wedged disk

# PostToolUse is narrowed to Bash at the matcher so the script is not even
# started for Edit / Read / Grep calls.  The other three events have no
# meaningful matcher.  UserPromptSubmit is there ONLY to stamp the start of
# the turn for the Stop gate (narrate.py); it prints nothing.
EVENTS = ("PostToolUse", "Notification", "Stop", "UserPromptSubmit")
MATCHERS = {"PostToolUse": "Bash"}


def hook_command(python: str = sys.executable, script: Path = NARRATE) -> str:
    return f"{shlex.quote(str(python))} {shlex.quote(str(script))}"


def hooks_snippet(command: str) -> dict:
    """The `hooks` block Claude Code expects, one group per event."""
    out: dict = {}
    for event in EVENTS:
        group: dict = {}
        if event in MATCHERS:
            group["matcher"] = MATCHERS[event]
        group["hooks"] = [{"type": "command", "command": command, "timeout": TIMEOUT_S}]
        out[event] = [group]
    return out


def _is_ours(group: dict, marker: str) -> bool:
    hooks = group.get("hooks") if isinstance(group, dict) else None
    if not isinstance(hooks, list) or not hooks:
        return False
    return all(isinstance(h, dict) and marker in str(h.get("command") or "") for h in hooks)


def remove(settings: dict, marker: str) -> dict:
    """A copy of `settings` without any hook group whose commands all point
    at `marker` (the narrate.py path).  Empty event lists are dropped, and
    an emptied `hooks` block goes too."""
    out = copy.deepcopy(settings)
    hooks = out.get("hooks")
    if not isinstance(hooks, dict):
        return out
    for event in list(hooks):
        groups = hooks[event]
        if not isinstance(groups, list):
            continue
        kept = [g for g in groups if not _is_ours(g, marker)]
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if not hooks:
        del out["hooks"]
    return out


def merge(settings: dict, snippet: dict, marker: str) -> dict:
    """`settings` with our groups replaced (not duplicated) by `snippet`."""
    out = remove(settings, marker)
    hooks = out.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = out["hooks"] = {}
    for event, groups in snippet.items():
        existing = hooks.get(event)
        if not isinstance(existing, list):
            existing = []
        hooks[event] = existing + copy.deepcopy(groups)
    return out


def load_settings(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object; not touching it")
    return data


def write_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(path.name + ".bak-claude-hooks")
        if not backup.exists():
            shutil.copy2(path, backup)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS,
                    help="settings file to edit (default: ~/.claude/settings.json)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the hook line (default: this one)")
    ap.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    ap.add_argument("--uninstall", action="store_true", help="remove the Jarvis hooks")
    args = ap.parse_args(argv)

    marker = str(NARRATE)
    before = load_settings(args.settings)
    if args.uninstall:
        after = remove(before, marker)
    else:
        after = merge(before, hooks_snippet(hook_command(args.python)), marker)

    if args.dry_run:
        print(json.dumps(after.get("hooks", {}), indent=2))
        return 0
    if after == before:
        print(f"{args.settings}: already up to date")
        return 0
    write_settings(args.settings, after)
    verb = "removed from" if args.uninstall else "installed in"
    print(f"Jarvis narration hooks {verb} {args.settings}")
    print("Restart your Claude Code sessions for the change to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
