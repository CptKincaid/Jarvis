"""Tool registry contract for the local (Ollama /api/chat) tool loop.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 4.1.
Owned by the brain work item after this seed; the dataclass FIELD NAMES
and the ToolRegistry method names below are the contract every tool
module and the brain code against — extend, never rename.

Budget (4.1): at most MAX_TOOLS tools, descriptions of at most
DESCRIPTION_WORD_CAP words, and — the one that actually costs — at most
MAX_SCHEMA_TOKENS prompt tokens of schema.  register() logs a warning
when a spec breaks a per-tool rule (never refuses — a tool is better than
no tool); the whole-set cost is reported ONCE, by schemas(), because that
is the moment the registry becomes a prompt.

What the budget is FOR, measured on this box 2026-08-31 against the
resident gemma4:26b (num_ctx 8192, production static prefix):

    tools   schema chars   prompt tokens   cold prefill of the whole prefix
       0             0            1441                       661 ms
      11          4814            2660                      1111 ms
      28          9489            3667                      1405 ms

So the 28 tools the app registers today are 2226 prompt tokens — two and
a half times the 900 the spec allows — and 294 ms of prefill more than
the spec's eleven.  That 294 ms is paid ONLY on a cold prefix: with
Ollama's prefix cache intact the same round costs 0.48 s wall and the
schemas are free (they are the same bytes every turn, which is the whole
point of the static-prefix rule in jarvis/brain.py).  So the tool count is
a real tax on any turn that starts cold, and close to nothing on one that
does not — trim it, but do not expect it to buy back seconds.

WHAT MAX_SCHEMA_TOKENS IS, AND IS NOT (relabelled 2026-09-04, because it
had been read as the wrong thing all week).  It is a COLD-PREFILL LATENCY
budget: the ~294 ms above, paid once per cold prefix.  It is NOT a budget
on window SPACE.  The window is jarvis/brain.py's num_ctx, it is his to
set in assistant.json, and at its default the same 28 schemas occupy about
a seventh of it rather than a third.  Trimming schema TEXT was measured
and rejected on 2026-09-04: it buys back space that is no longer scarce,
and the risk it carries — whether a 25 B model still picks the right
argument without the descriptions it reads them from — has already cost
two live incidents.  Per-route scoping was measured and rejected harder:
editing the head of the tool array re-prefills 3455 tokens, +1.2 s on that
turn, and alternating a tools prefix with a tools-free one never goes warm
in either direction (+1.3 s per turn, forever).  Do not "optimise" either
without a tool-selection regression pass behind it.

CHARS_PER_TOKEN below is NOT a stale guess — the range under it was
measured at three points.  Leave it at 4.1.  It is the rate for schema
JSON and prose ONLY: tool RESULTS (calendar, mail) tokenize at 2.25 chars
per token, measured 2026-09-04, and jarvis/brain.py costs those at its own
TOOL_CHARS_PER_TOKEN.  Do not "fix" one number to fit the other text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("tools.registry")

MAX_TOOLS = 11
DESCRIPTION_WORD_CAP = 20
MAX_SCHEMA_TOKENS = 900         # spec 4.1, "total schema tokens <= 900"
# The schema JSON is mostly ASCII identifiers and prose, which gemma4's
# tokenizer takes at 3.95 (11 tools) to 4.26 (28 tools) characters each --
# see the table above, all three points measured through prompt_eval_count.
# 4.1 keeps the estimate inside 4% either way, which is all a budget check
# needs; the bench still has the real number.
CHARS_PER_TOKEN = 4.1


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
    # Argument names the MODEL may not set. A handler can accept a keyword
    # that only trusted code (a commander force_args) is allowed to fill:
    # spotify_liked's ``shuffle`` is the case that earned this -- the model
    # sent shuffle=true unasked for "Play my like songs." and the newest-
    # first default was lost (live log 2026-09-01 19:59). Reserved keys are
    # kept OUT of the schema (nothing to invite) and stripped by call() when
    # the args came from the model, so a model that guesses them anyway
    # still cannot override the caller's default.
    reserved: frozenset = frozenset()
    # derive(utterance) -> dict: arguments read off HIS WORDS rather than
    # off the model's guess, applied to model calls only (call() below).
    # Reserving a key takes the model's vote away, which makes the
    # commander's Tier-1 route the only thing left that can say yes -- so a
    # phrasing the matcher misses gets the default in silence. spotify_liked
    # is the case: "put my liked songs on shuffle please" missed the route,
    # the model's shuffle=true was dropped, and it played newest-first and
    # SAID so (2026-09-02 review). With a deriver the words decide either
    # way, whichever door the turn came through.
    derive: Optional[Callable[[str], dict]] = None

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
        self._logged_budget = False

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            log.warning("tool %s re-registered", spec.name)
        words = spec.description_words()
        if words > DESCRIPTION_WORD_CAP:
            log.warning("tool %s description is %d words (cap %d)",
                        spec.name, words, DESCRIPTION_WORD_CAP)
        # A reserved key that is also advertised in the schema is a drift
        # bug: the model is invited to send an argument call() will drop.
        offered = set((spec.parameters or {}).get("properties") or {})
        leaked = sorted(set(spec.reserved) & offered)
        if leaked:
            log.warning("tool %s advertises reserved argument(s) %s in its "
                        "schema", spec.name, ", ".join(leaked))
        self._tools[spec.name] = spec
        # No per-registration count warning: it used to fire once per tool
        # past the budget, so a 28-tool boot wrote a 17-line ladder of
        # WARNINGs that said the same thing 17 times and buried the tool
        # list between them. The set is reported once, by schemas().
        self._logged_budget = False
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
        """The Ollama ``tools`` block. BYTE-STABLE for a fixed tool set:
        the model turn caches on it (jarvis/brain.py, the static-prefix
        rule), so this must never vary with the utterance or the clock."""
        schemas = [t.schema() for t in self._tools.values()]
        if not self._logged_budget:
            self._logged_budget = True
            state = self.schema_budget()
            report = log.warning if not state["ok"] else log.info
            report("tools: %d registered (budget %d), schema ~%d prompt "
                   "tokens (cold-prefill budget %d, ~294 ms once per cold "
                   "prefix -- NOT a window-space budget; the window is "
                   "brain.num_ctx)%s", state["tools"], MAX_TOOLS,
                   state["schema_tokens"], MAX_SCHEMA_TOKENS,
                   "; over the word cap: " + ", ".join(state["over_word_cap"])
                   if state["over_word_cap"] else "")
        return schemas

    def schema_tokens(self) -> int:
        """Estimated prompt tokens the tool block costs on every turn.

        Built from _tools rather than schemas() so the budget report cannot
        recurse into the call that asks for it."""
        payload = json.dumps([t.schema() for t in self._tools.values()])
        return int(len(payload) / CHARS_PER_TOKEN)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name) -> bool:
        return name in self._tools

    def budget(self) -> dict:
        """Static budget check (token count is measured by the bench).

        The dict shape is a contract (tests/test_brain_tools.py compares it
        whole), so the token figures live in schema_budget() beside it."""
        over = [t.name for t in self._tools.values()
                if t.description_words() > DESCRIPTION_WORD_CAP]
        return {"tools": len(self._tools), "max_tools": MAX_TOOLS,
                "over_word_cap": over,
                "ok": len(self._tools) <= MAX_TOOLS and not over}

    def schema_budget(self) -> dict:
        """budget() plus what the schemas cost in prompt tokens.

        The count is a proxy; the tokens are the bill. A registry can sit
        inside MAX_TOOLS and still blow MAX_SCHEMA_TOKENS (the spec's own
        eleven measure 1219), and that is the number to trim against."""
        state = self.budget()
        tokens = self.schema_tokens()
        state["schema_tokens"] = tokens
        state["max_schema_tokens"] = MAX_SCHEMA_TOKENS
        state["ok"] = state["ok"] and tokens <= MAX_SCHEMA_TOKENS
        return state

    def call(self, name: str, args: Optional[dict] = None, *,
             from_model: bool = False,
             utterance: str = "") -> ToolResult:
        """Never raises: unknown tools and handler exceptions become an
        ok=False result the model can explain. args may arrive as a JSON
        string (some models emit arguments that way).

        ``from_model=True`` marks args the model wrote (the brain's tool
        loop); the spec's ``reserved`` keys are dropped from those and its
        ``derive`` reads ``utterance`` in their place. Forced calls from the
        commander leave it False and keep every key -- the commander already
        decided them from the utterance, and a second reading must not
        overrule the first."""
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
        if from_model and spec.reserved:
            dropped = {k: v for k, v in args.items() if k in spec.reserved}
            if dropped:
                # Info, not warning: the model guessing here is expected
                # and harmless now; the line is the audit trail for #66.
                log.info("tool %s: dropped model-supplied reserved args %s",
                         name, json.dumps(dropped, default=str)[:120])
                args = {k: v for k, v in args.items()
                        if k not in spec.reserved}
        if from_model and spec.derive is not None:
            # After the strip, so a deriver always wins over the model --
            # and it runs even with no utterance in hand, where it answers
            # for "he said nothing" (spotify_liked: shuffle=None, the
            # configured default).
            try:
                extra = spec.derive(utterance or "") or {}
            except Exception:               # noqa: BLE001 - tool boundary
                log.exception("tool %s: derive failed", name)
                extra = {}
            if extra:
                log.info("tool %s: %s from the utterance", name,
                         json.dumps(extra, default=str)[:120])
                args = {**args, **extra}
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
