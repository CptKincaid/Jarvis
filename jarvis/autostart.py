"""GNOME autostart entry + never-sleep for Jarvis (spec section 10.2).

``install()`` writes ``~/.config/autostart/jarvis.desktop`` so Jarvis starts
15 s after login (the alarms only ring while the app runs); it is
idempotent — an identical file is left untouched.  ``disable_gnome_suspend``
sets ``sleep-inactive-ac-type`` to ``'nothing'`` only when it is not already
(it is 'nothing' on this Spark today).  Every subprocess goes through
``_run`` so tests record instead of execute.

CLI::

    python -m jarvis.autostart --install     # entry + never-sleep
    python -m jarvis.autostart --uninstall
    python -m jarvis.autostart --status
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("autostart")

REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_NAME = "jarvis.desktop"
DELAY_S = 15
GSETTINGS_SCHEMA = "org.gnome.settings-daemon.plugins.power"
GSETTINGS_KEY = "sleep-inactive-ac-type"
NEVER = "'nothing'"


def desktop_path(path: Optional[os.PathLike | str] = None) -> Path:
    """``~/.config/autostart/jarvis.desktop`` (HOME-relative so a test with
    a temp HOME never touches the real one); explicit ``path`` wins."""
    if path:
        return Path(os.path.expanduser(str(path)))
    return Path.home() / ".config" / "autostart" / DESKTOP_NAME


def default_python() -> str:
    """The venv interpreter the app runs under (``~/vss_env/bin/python``),
    else whatever is running now."""
    venv = Path.home() / "vss_env" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def default_exec() -> str:
    return f"{default_python()} -m jarvis.app"


def render_desktop(exec_cmd: Optional[str] = None,
                   working_dir: Optional[os.PathLike | str] = None) -> str:
    exec_cmd = exec_cmd or default_exec()
    working_dir = str(working_dir or REPO_ROOT)
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=Jarvis",
        "Comment=Jarvis voice assistant (starts at login)",
        f"Exec={exec_cmd}",
        f"Path={working_dir}",
        "Terminal=false",
        "StartupNotify=false",
        # Binds the window to the dock icon (tk.Tk(className="jarvis")), so a
        # login-started Jarvis shows under its own icon rather than a stray
        # "Unknown" entry — the same key the desktop/dock .desktop files carry.
        "StartupWMClass=jarvis",
        "NoDisplay=false",
        "X-GNOME-Autostart-enabled=true",
        f"X-GNOME-Autostart-Delay={DELAY_S}",
    ]) + "\n"


def install(exec_cmd: Optional[str] = None,
            path: Optional[os.PathLike | str] = None,
            working_dir: Optional[os.PathLike | str] = None) -> Path:
    """Write the autostart entry; identical content is not rewritten."""
    target = desktop_path(path)
    text = render_desktop(exec_cmd, working_dir)
    try:
        if target.exists() and target.read_text(encoding="utf-8") == text:
            log.debug("autostart entry already current: %s", target)
            return target
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    log.info("autostart entry written: %s", target)
    return target


def uninstall(path: Optional[os.PathLike | str] = None) -> bool:
    """Remove the entry; True when a file was removed."""
    target = desktop_path(path)
    try:
        target.unlink()
    except FileNotFoundError:
        return False
    log.info("autostart entry removed: %s", target)
    return True


def is_installed(path: Optional[os.PathLike | str] = None) -> bool:
    """True when the entry exists, launches ``jarvis.app`` and is enabled."""
    target = desktop_path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith(("[", "#")):
            key, _, value = line.partition("=")
            lines[key.strip()] = value.strip()
    if "jarvis.app" not in lines.get("Exec", ""):
        return False
    if lines.get("Hidden", "false").lower() == "true":
        return False
    return lines.get("X-GNOME-Autostart-enabled", "true").lower() != "false"


# ------------------------------------------------------------ gsettings
def _run(argv: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    """The one subprocess seam (tests replace it with a recorder)."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def suspend_setting(run: Callable = None) -> Optional[str]:
    """Current ``sleep-inactive-ac-type`` ('nothing', 'suspend', …) or None
    when gsettings is unavailable."""
    run = run or _run
    try:
        out = run(["gsettings", "get", GSETTINGS_SCHEMA, GSETTINGS_KEY])
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("gsettings get failed: %s", exc)
        return None
    if getattr(out, "returncode", 1) != 0:
        log.warning("gsettings get rc=%s: %s", out.returncode,
                    (getattr(out, "stderr", "") or "").strip())
        return None
    return (out.stdout or "").strip()


def disable_gnome_suspend(run: Callable = None) -> bool:
    """Make sure the desktop never suspends on AC power.  Only calls
    ``gsettings set`` when the value is not already 'nothing'.  Returns True
    when suspend is disabled after the call."""
    run = run or _run
    current = suspend_setting(run)
    if current is None:
        return False
    if current == NEVER:
        log.debug("gnome suspend already disabled (%s)", current)
        return True
    try:
        out = run(["gsettings", "set", GSETTINGS_SCHEMA, GSETTINGS_KEY, "nothing"])
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("gsettings set failed: %s", exc)
        return False
    ok = getattr(out, "returncode", 1) == 0
    log.info("gnome suspend %s -> nothing (%s)", current,
             "ok" if ok else "failed")
    return ok


# ------------------------------------------------------------------ CLI
def status(path: Optional[os.PathLike | str] = None, run: Callable = None) -> dict:
    target = desktop_path(path)
    return {"desktop": str(target), "installed": is_installed(target),
            "suspend": suspend_setting(run)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jarvis.autostart",
                                 description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--install", action="store_true",
                   help="write the autostart entry and disable GNOME suspend")
    g.add_argument("--uninstall", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--path", default=None, help="desktop file (default ~/.config/autostart/jarvis.desktop)")
    ap.add_argument("--exec", dest="exec_cmd", default=None,
                    help=f"Exec= line (default: {shlex.quote(default_exec())})")
    ap.add_argument("--keep-suspend", action="store_true",
                    help="do not touch the GNOME suspend setting")
    args = ap.parse_args(argv)

    if args.install:
        target = install(args.exec_cmd, args.path)
        print(f"autostart entry: {target}")
        if not args.keep_suspend:
            ok = disable_gnome_suspend()
            print(f"gnome suspend on AC: {'disabled' if ok else 'could not set (gsettings?)'}")
        return 0
    if args.uninstall:
        removed = uninstall(args.path)
        print("autostart entry removed" if removed else "no autostart entry to remove")
        return 0
    info = status(args.path)
    print(f"desktop file: {info['desktop']}")
    print(f"installed: {'yes' if info['installed'] else 'no'}")
    print(f"gnome sleep-inactive-ac-type: {info['suspend'] or 'unknown'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
