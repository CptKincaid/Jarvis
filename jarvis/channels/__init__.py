"""Out-of-band channels for Jarvis — how he reaches Hunter when Hunter is
not at the desk, and how Hunter answers back.

- `jarvis.channels.notify`  — the Alerts hub: desktop toast (`notify-send`)
  plus a Discord post for every milestone / completion / blocked task /
  approval question / alarm / reminder. Never blocks the caller.
- `jarvis.channels.discord` — the two-way Discord channel: a bot on the
  gateway (websocket) that posts alerts and relays replies ("yes", "no",
  or any ordinary Jarvis command) back into the app.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 8.

This package deliberately imports neither submodule at package import so the
submodules can import the helpers below without a cycle, and so an
`import jarvis.channels` never pulls `websockets` in.

The helpers here are the small things both submodules share: reading a
dotted key from whatever config object the app hands over (the real
`AssistantConfig.get("discord.bot_token")`, a plain dict, or a namespace),
telling a placeholder from a real value, and redacting secrets from any
text that might reach a log or an exception message.
"""
from __future__ import annotations

import re
from typing import Any

# Values a config file may carry before the user filled a section in. The
# assistant.json template ships empty strings; these cover hand-edits too.
_PLACEHOLDER = re.compile(r"^\s*$|^<.*>$|^your[\s_-]|•|^\.\.\.$|^x{4,}$|^changeme$",
                          re.IGNORECASE)

REDACTED = "•••"


def is_placeholder(value: Any) -> bool:
    """True when a config value is empty or an obvious template marker."""
    if value is None:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    return bool(_PLACEHOLDER.search(str(value)))


def cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read `a.b.c` from an AssistantConfig (`.get(dotted, default)`), a
    nested dict, or a nested namespace. Missing anywhere -> `default`."""
    if cfg is None:
        return default
    if not isinstance(cfg, dict):
        getter = getattr(cfg, "get", None)
        if callable(getter):
            try:
                value = getter(dotted, default)
            except TypeError:
                try:
                    value = getter(dotted)
                except Exception:
                    value = None
            except Exception:
                value = None
            if value is not None:
                return value
    cur = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def redact(text: Any, *secrets: str) -> str:
    """Replace every occurrence of each secret (>= 6 chars) in `text` with
    `•••`. Safe on any input; never raises."""
    try:
        out = str(text)
    except Exception:
        return REDACTED
    for secret in secrets:
        if secret and len(secret) >= 6:
            out = out.replace(secret, REDACTED)
    return out


__all__ = ["cfg_get", "is_placeholder", "redact", "REDACTED"]
