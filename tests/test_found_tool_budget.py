"""FOUND 2026-08-26 (assistant-tools sweep): the live registry carries 17
tools against the spec's budget of 11, and the extra schema is paid for in
prefill on every single model turn.

jarvis/tools/registry.py sets MAX_TOOLS = 11 (spec 4.1) and register()
only logs "17 tools registered (budget 11)" — the app boots over budget
and ToolRegistry.budget()["ok"] is False.  jarvis/tools/spotify.py alone
contributes six of the seventeen.

Measured in-process against the live Ollama (gemma4:26b resident,
load_duration 0.00 s) with the production static system prompt:

    17 tools  -> tools schema 5262 bytes, prompt_eval_count 2704-2766
    no tools  -> prompt_eval_count 1503

i.e. the tool block costs ~1200 prompt tokens per turn, and a one-tool
round trip measured 5.8-8.9 s wall against the spec bar of p50 <= 3.0 s
(spec 4.3 / section 12 check 2).  The spec's own remedy list for a missed
bar starts with "trim tool descriptions".

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
