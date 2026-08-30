"""Alerts hub — Jarvis's away alerts (spec section 8.1).

One call, `Alerts.alert(kind, title, text)`, fans out to:

- a desktop toast: `notify-send -a Jarvis -u <urgency> -t <ms> -- <title> <text>`
  (urgency `critical` for question / blocked / alarm, `normal` otherwise;
  expiry 8000 ms, or 0 = sticky for a question), through the module-level
  `_run` seam so tests record argv instead of spawning anything;
- a Discord post `**<title>** — <text>` (a question ends with
  "Reply yes or no.") when a configured `DiscordChannel` is attached.

The caller never waits: alerts are queued to one daemon worker thread and
each delivery is capped (`timeout_s`, 5 s) — a hung notification daemon or
a slow Discord API can only delay later alerts, never the app. Failures are
logged with context and swallowed; an alert is best effort by design.

Kinds: milestone | done | blocked | question | alarm | reminder. The app
(wiring item) calls `alert` on ClaudeTaskState(done|failed),
ApprovalRequested, AlarmFired, ReminderFired and spoken milestones.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from jarvis.channels import cfg_get, redact
from jarvis.logs import get_logger

log = get_logger("channels.notify")

KINDS = ("milestone", "done", "blocked", "question", "alarm", "reminder")
CRITICAL_KINDS = frozenset({"question", "blocked", "alarm"})
EXPIRE_MS = 8000
QUESTION_SUFFIX = "Reply yes or no."
APP_NAME = "Jarvis"

_missing_warned = False


def _run(argv: list[str], timeout: float = 5.0) -> Optional[subprocess.CompletedProcess]:
    """Subprocess seam (spec 3.3). Returns the CompletedProcess, or None
    when the binary is missing / times out / fails to spawn."""
    global _missing_warned
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except FileNotFoundError:
        if not _missing_warned:
            _missing_warned = True
            log.warning("%s is not installed; desktop toasts are off", argv[0])
        return None
    except subprocess.TimeoutExpired:
        log.warning("%s timed out after %.0fs", argv[0], timeout)
        return None
    except Exception as exc:
        log.warning("%s failed: %s", argv[0], exc)
        return None


# ---------------------------------------------------------- pure helpers
def urgency_for(kind: str) -> str:
    return "critical" if kind in CRITICAL_KINDS else "normal"


def expire_ms_for(kind: str) -> int:
    return 0 if kind == "question" else EXPIRE_MS


def notify_argv(kind: str, title: str, text: str) -> list[str]:
    """The exact notify-send command for an alert. `--` keeps a body that
    starts with a dash from being read as an option."""
    return ["notify-send", "-a", APP_NAME, "-u", urgency_for(kind),
            "-t", str(expire_ms_for(kind)), "--", title, text]


def discord_text(kind: str, title: str, text: str) -> str:
    """`**<title>** — <text>`; questions end with the reply hint."""
    body = f"**{title}** — {text}".strip()
    if kind == "question" and not body.endswith(QUESTION_SUFFIX):
        body = f"{body} {QUESTION_SUFFIX}"
    return body


@dataclass
class AlertRecord:
    kind: str
    title: str
    text: str
    request_id: Optional[str] = None
    created: float = field(default_factory=time.time)
    toast_ok: Optional[bool] = None       # None = not attempted
    discord_ok: Optional[bool] = None


# ------------------------------------------------------------------ hub
class Alerts:
    """Fan-out hub. `attach(discord)` after construction; `alert(...)`
    from any thread; `flush(timeout)` in tests to wait for delivery."""

    def __init__(self, cfg: Any, run: Callable = _run, timeout_s: float = 5.0):
        self._cfg = cfg
        self._run = run
        self._timeout = float(timeout_s)
        self._discord = None
        self._queue: "queue.Queue[AlertRecord]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.recent: deque[AlertRecord] = deque(maxlen=50)
        # Optional switches (both default on): alerts.desktop / alerts.discord.
        self.desktop_enabled = _truthy(cfg_get(cfg, "alerts.desktop", True))
        self.discord_enabled = _truthy(cfg_get(cfg, "alerts.discord", True))
        self._toast_available = shutil.which("notify-send") is not None or run is not _run

    # -- wiring --------------------------------------------------------
    def attach(self, discord) -> None:
        """Attach a DiscordChannel (or None). Posts only when
        `discord.configured` is true at delivery time."""
        self._discord = discord

    @property
    def discord(self):
        return self._discord

    # -- API -----------------------------------------------------------
    def alert(self, kind: str, title: str, text: str,
              request_id: Optional[str] = None) -> AlertRecord:
        """Queue an alert and return immediately."""
        kind = (kind or "").strip().lower()
        if kind not in KINDS:
            log.warning("unknown alert kind %r; treating as milestone", kind)
            kind = "milestone"
        rec = AlertRecord(kind=kind, title=(title or APP_NAME).strip(),
                          text=(text or "").strip(), request_id=request_id)
        self.recent.append(rec)
        self._queue.put(rec)
        self._ensure_worker()
        return rec

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every queued alert has been delivered (tests)."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    # -- worker --------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker, name="alerts",
                                            daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            rec = self._queue.get()
            try:
                self._deliver(rec)
            except Exception as exc:   # never let one alert kill the worker
                log.error("alert delivery failed: %s", _safe(exc))
            finally:
                self._queue.task_done()

    def _deliver(self, rec: AlertRecord) -> None:
        if self.desktop_enabled and self._toast_available:
            rec.toast_ok = self._toast(rec)
        discord = self._discord
        if discord is not None and self.discord_enabled and \
                getattr(discord, "configured", False):
            rec.discord_ok = self._post(discord, rec)

    def _toast(self, rec: AlertRecord) -> bool:
        argv = notify_argv(rec.kind, rec.title, rec.text)
        try:
            result = self._run(argv, timeout=self._timeout)
        except Exception as exc:
            log.warning("notify-send failed: %s", _safe(exc))
            return False
        if result is None:
            return False
        rc = getattr(result, "returncode", 0)
        if rc not in (0, None):
            log.warning("notify-send rc=%s: %s", rc,
                        (getattr(result, "stderr", "") or "").strip()[:200])
            return False
        return True

    def _post(self, discord, rec: AlertRecord) -> bool:
        body = discord_text(rec.kind, rec.title, rec.text)
        try:
            return bool(discord.post(body, timeout_s=self._timeout))
        except TypeError:
            # A minimal channel without the timeout keyword.
            try:
                return bool(discord.post(body))
            except Exception as exc:
                log.warning("discord post failed: %s", _safe(exc))
                return False
        except Exception as exc:
            log.warning("discord post failed: %s", _safe(exc))
            return False


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _safe(exc: BaseException) -> str:
    """Exception text with anything token-shaped masked (belt and braces —
    the Discord channel already redacts its own messages)."""
    text = str(exc)
    for secret in _known_secrets():
        text = redact(text, secret)
    return text


_secret_sources: list[Callable[[], str]] = []


def register_secret(getter: Callable[[], str]) -> None:
    """Let a channel register a zero-arg getter for a secret so the hub's
    own log lines can mask it too."""
    _secret_sources.append(getter)


def _known_secrets() -> list[str]:
    out = []
    for getter in _secret_sources:
        try:
            value = getter()
        except Exception:
            continue
        if value:
            out.append(str(value))
    return out


__all__ = ["Alerts", "AlertRecord", "KINDS", "notify_argv", "discord_text",
           "urgency_for", "expire_ms_for", "register_secret", "_run"]
