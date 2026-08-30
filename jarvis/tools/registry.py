"""Tool registry contract for the local (Ollama /api/chat) tool loop.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 4.1.
Owned by the brain work item after this seed; the dataclass FIELD NAMES
and the ToolRegistry method names below are the contract every tool
module and the brain code against — extend, never rename.

Budget (4.1): at most MAX_TOOLS tools, descriptions of at most
DESCRIPTION_WORD_CAP words; register() logs a warning when a spec breaks
either (never refuses — a tool is better than no tool), and budget()
reports the state so the bench can print it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("tools.registry")

MAX_TOOLS = 11
DESCRIPTION_WORD_CAP = 20


@dataclass
class ToolResult:
    """What a tool hands back to the model turn."""
    text: str                          # compact plain text the model reads
    ok: bool = True                    # False -> text explains the failure
    max_sentences: int = 2             # spoken cap for the reply using it
    card: Optional[dict] = None        # optional UI payload (briefing)
    speak: Optional[str] = None        # verbatim line; skips the model turn

    def __post_init__(self):
        self.text = "" if self.text is None else str(self.text)
        try:
            self.max_sentences = max(1, int(self.max_sentences))
        except (TypeError, ValueError):
            self.max_sentences = 2


@dataclass
class ToolSpec:
    name: str
    description: str                   # <= 20 words
    parameters: dict = field(default_factory=lambda: {
        "type": "object", "properties": {}})
    handler: Callable[..., ToolResult] = None   # handler(**args) -> ToolResult

    def schema(self) -> dict:
        """Ollama /api/chat `tools` entry."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}

    def description_words(self) -> int:
        return len((self.description or "").split())


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            log.warning("tool %s re-registered", spec.name)
        words = spec.description_words()
        if words > DESCRIPTION_WORD_CAP:
            log.warning("tool %s description is %d words (cap %d)",
                        spec.name, words, DESCRIPTION_WORD_CAP)
        self._tools[spec.name] = spec
        if len(self._tools) > MAX_TOOLS:
            log.warning("%d tools registered (budget %d)",
                        len(self._tools), MAX_TOOLS)
        return spec

    def register_many(self, specs) -> None:
        for spec in specs:
            self.register(spec)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name) -> bool:
        return name in self._tools

    def budget(self) -> dict:
        """Static budget check (token count is measured by the bench)."""
        over = [t.name for t in self._tools.values()
                if t.description_words() > DESCRIPTION_WORD_CAP]
        return {"tools": len(self._tools), "max_tools": MAX_TOOLS,
                "over_word_cap": over,
                "ok": len(self._tools) <= MAX_TOOLS and not over}

    def call(self, name: str, args: Optional[dict] = None) -> ToolResult:
        """Never raises: unknown tools and handler exceptions become an
        ok=False result the model can explain. args may arrive as a JSON
        string (some models emit arguments that way)."""
        spec = self._tools.get(name)
        if spec is None or spec.handler is None:
            return ToolResult(text=f"no such tool: {name}", ok=False)
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                log.warning("tool %s: unparsable arguments %r", name, args)
                args = {}
        if not isinstance(args, dict):
            args = {}
        try:
            result = spec.handler(**args)
        except TypeError as exc:           # bad/missing arguments
            log.warning("tool %s bad args %r: %s", name, args, exc)
            return ToolResult(text=f"{name}: bad arguments ({exc})", ok=False)
        except Exception as exc:            # noqa: BLE001 - tool boundary
            log.exception("tool %s failed", name)
            return ToolResult(text=f"{name} failed: {str(exc)[:80]}", ok=False)
        if not isinstance(result, ToolResult):
            log.warning("tool %s returned %s, not a ToolResult", name,
                        type(result).__name__)
            return ToolResult(text=str(result))
        return result
