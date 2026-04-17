"""CommandDispatcher — routes transcribed text to handlers via a regex registry.

Replaces the 350-line if/elif chain in VoiceInputGUI._check_quick_command.
Each handler is (pattern, callable). Handlers receive the re.Match and a
context dict; returning True = handled, False = fall through.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

LOG_DIR = Path("/tmp/vss_voice")


from jarvis.logging import get_logger
_log = get_logger("DISP")


@dataclass
class CommandHandler:
    pattern: str
    handler: Callable[[re.Match, dict], bool]
    flags: int = re.IGNORECASE

    def compiled(self):
        return re.compile(self.pattern, self.flags)


class CommandDispatcher:
    """Dispatches text to the first matching handler; falls through to brain."""

    def __init__(self, handlers: list[CommandHandler], brain: Any):
        self._compiled = [(h.compiled(), h.handler) for h in handlers]
        self.brain = brain

    def handle(self, text: str, ctx: dict | None = None) -> None:
        ctx = ctx if ctx is not None else {}
        for pattern, handler in self._compiled:
            m = pattern.match(text)
            if m:
                try:
                    if handler(m, ctx):
                        _log(f"Dispatched: {pattern.pattern!r} -> handled")
                        return
                except Exception as e:
                    _log(f"Handler error ({pattern.pattern!r}): {e}")
        self.brain.handle(text, ctx)
