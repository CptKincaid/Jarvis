"""FOUND 2026-08-26 (assistant-tools sweep): the live registry carries 17
tools against the spec's budget of 11, and the extra schema is paid for in
prefill on every single model turn.

jarvis/tools/registry.py sets MAX_TOOLS = 11 (spec 4.1) and register()
only logs "17 tools registered (budget 11)" — the app boots over budget
and ToolRegistry.budget()["ok"] is False.  jarvis/tools/spotify.py alone
contributes six of the seventeen.

RE-MEASURED 2026-08-31, now 28 tools, against the live Ollama (gemma4:26b
resident, num_ctx 8192, production static system prompt):

    tools   schema bytes   prompt_eval_count   cold prefill of the prefix
       0              0                 1441                      661 ms
      11           4814                 2660                     1111 ms
      28           9489                 3667                     1405 ms

So the tool block is 2226 prompt tokens against the spec's 900, and 294 ms
of prefill more than the spec's eleven would cost.

What that 294 ms is NOT: the 2026-08-26 note in this file blamed the tool
block for a 5.8-8.9 s one-tool round trip.  That was the wrong suspect.
The same round measured 0.48 s wall from a client whose prefix was cached,
and 9.6 s from the app -- the difference is that every app turn embeds the
utterance through nomic-embed-text first, OLLAMA_MAX_LOADED_MODELS=1
evicts gemma4:26b to do it, and the turn then pays 6.97 s reloading the
model plus a full 1.9 s prefill because the KV cache died with it
(ollama journal, 15:22:10: "prompt eval time = 1925.17 ms / 4569 tokens").
The tool count is a real tax on a COLD prefix and close to nothing on a
warm one, so this budget is worth keeping - but it is not where the
seconds are.

This test asserts the contract the registry documents; it fails today.
"""
import pytest

from jarvis.assistant_config import AssistantConfig
from jarvis.tools.registry import MAX_TOOLS, ToolRegistry

TOOL_MODULES = ("jarvis.tools.location", "jarvis.tools.weather",
                "jarvis.tools.calendar", "jarvis.tools.timekeeper",
                "jarvis.tools.notes", "jarvis.tools.mail",
                "jarvis.tools.briefing", "jarvis.tools.spotify")


def _live_registry():
    import importlib
    from types import SimpleNamespace
    cfg = AssistantConfig({})
    services = SimpleNamespace(assistant=cfg, tools=None, notes=None,
                               calendar=None, timekeeper=None,
                               news_cache_path=None)
    registry = ToolRegistry()
    services.tools = registry
    for name in TOOL_MODULES:
        registry.register_many(importlib.import_module(name)
                               .make_tools(cfg, services))
    return registry


@pytest.mark.xfail(reason="8 tool modules register 17 tools; MAX_TOOLS is 11",
                   strict=True)
def test_registered_tools_stay_inside_the_spec_budget():
    registry = _live_registry()
    assert len(registry) <= MAX_TOOLS, registry.names()
    assert registry.budget()["ok"], registry.budget()
