"""Firewall: the suite must never touch the LIVE app's /tmp/vss_voice
(jarvis.log, speak_queue.txt, jarvis.pid, gui_debug.log).

jarvis.logs / jarvis.config / jarvis.jarvis_agent read JARVIS_LOG_DIR at
import, so it is set here — conftest imports before any test module — to
a throwaway directory. The session-scoped autouse fixture below then
re-asserts the redirect on the already-imported modules (belt and braces:
a test that imports jarvis.* before the env is read, or a module that
caches a Path, still lands in the throwaway dir) and fails loudly if any
of them still points at /tmp/vss_voice."""
import os
import tempfile
from pathlib import Path

import pytest

_TEST_LOG_DIR = Path(os.environ.get("JARVIS_LOG_DIR") or
                     tempfile.mkdtemp(prefix="jarvis-tests-"))
os.environ["JARVIS_LOG_DIR"] = str(_TEST_LOG_DIR)

# Personal-assistant firewall (spec 2026-08-26, section 3.2/3.3): the real
# ~/.config/jarvis/assistant.json holds the user's secrets and ~/.cache/jarvis
# the live caches; both are redirected into the throwaway dir BEFORE any
# jarvis import so AssistantConfig.load() / PATHS.CACHE_DIR never see them.
_TEST_ASSISTANT_CONFIG = Path(os.environ.get("JARVIS_ASSISTANT_CONFIG") or
                              (_TEST_LOG_DIR / "assistant.json"))
os.environ["JARVIS_ASSISTANT_CONFIG"] = str(_TEST_ASSISTANT_CONFIG)
_TEST_CACHE_DIR = Path(os.environ.get("JARVIS_CACHE_DIR") or
                       (_TEST_LOG_DIR / "cache"))
os.environ["JARVIS_CACHE_DIR"] = str(_TEST_CACHE_DIR)
# The state stores (timekeeper.db, notes.db, claude_projects.json, the typed
# history, the memory store) hang off PATHS.MEMORY_DIR — the user's real
# ~/.aiws_trainer/jarvis_memory. Redirected here so a test that constructs the
# real modules (tests/test_app_wiring.py) cannot touch it.
_TEST_MEMORY_DIR = Path(os.environ.get("JARVIS_MEMORY_DIR") or
                        (_TEST_LOG_DIR / "memory"))
os.environ["JARVIS_MEMORY_DIR"] = str(_TEST_MEMORY_DIR)
os.environ.setdefault("JARVIS_LEGACY_DIR", str(_TEST_LOG_DIR / "legacy"))
# The Spotify OAuth token sits beside the assistant config; keep the suite off
# the real one (jarvis/tools/spotify.py reads JARVIS_SPOTIFY_TOKEN first).
_TEST_SPOTIFY_TOKEN = Path(os.environ.get("JARVIS_SPOTIFY_TOKEN") or
                           (_TEST_LOG_DIR / "spotify_token.json"))
os.environ["JARVIS_SPOTIFY_TOKEN"] = str(_TEST_SPOTIFY_TOKEN)


@pytest.fixture(scope="session", autouse=True)
def _firewall_live_log_dir():
    """Every jarvis path that can write under the log dir points at the
    throwaway directory for the whole session."""
    from jarvis import config, logs
    live = Path("/tmp/vss_voice")
    assert logs.LOG_DIR != live, "jarvis.logs still targets the live app"
    assert config.PATHS.LOG_DIR != live, "PATHS.LOG_DIR still live"
    assert config.PATHS.SPEAK_QUEUE.parent != live, "speak queue still live"
    assert logs.LOG_FILE.parent == _TEST_LOG_DIR
    try:
        from jarvis import jarvis_agent
        assert jarvis_agent.LOG_DIR != live, "jarvis_agent LOG_DIR still live"
    except ImportError:
        pass
    try:
        from jarvis import speak_queue
        assert Path(speak_queue.SPEAK_QUEUE).parent != live
    except ImportError:
        pass
    # Assistant paths (section 3.2): nothing may point at the live socket /
    # task dir / MCP config under /tmp/vss_voice, nor at the user's real
    # config file or cache.
    real_cfg_dir = Path.home() / ".config" / "jarvis"
    real_cache = Path.home() / ".cache" / "jarvis"
    P = config.PATHS
    assert P.ASSISTANT_CONFIG == _TEST_ASSISTANT_CONFIG, \
        "PATHS.ASSISTANT_CONFIG ignores JARVIS_ASSISTANT_CONFIG"
    assert real_cfg_dir not in P.ASSISTANT_CONFIG.parents and \
        P.ASSISTANT_CONFIG.parent != real_cfg_dir, "assistant config still live"
    assert P.CACHE_DIR != real_cache and real_cache not in P.CACHE_DIR.parents, \
        "PATHS.CACHE_DIR still live"
    for name in ("APPROVALS_SOCK", "CLAUDE_TASK_DIR", "MCP_CONFIG"):
        path = getattr(P, name)
        assert path != live and live not in path.parents, f"PATHS.{name} still live"
        assert path.parent == _TEST_LOG_DIR or _TEST_LOG_DIR in path.parents, \
            f"PATHS.{name} not under the test log dir"
    real_memory = Path.home() / ".aiws_trainer" / "jarvis_memory"
    assert P.MEMORY_DIR == _TEST_MEMORY_DIR, "PATHS.MEMORY_DIR still live"
    assert P.LEGACY_AGENT_DIR != Path.home() / ".aiws_trainer" / "jarvis_data", \
        "PATHS.LEGACY_AGENT_DIR still live"
    for name in ("TIMEKEEPER_DB", "NOTES_DB", "CLAUDE_PROJECTS"):
        path = getattr(P, name)
        assert real_memory not in path.parents, f"PATHS.{name} still live"
    try:
        from jarvis.tools import spotify
        token = spotify.token_path()
        assert real_cfg_dir not in token.parents, "spotify token still live"
    except ImportError:
        pass
    try:
        from jarvis import assistant_config
        default = getattr(assistant_config, "DEFAULT_PATH", None)
        if default is not None:
            assert Path(default).parent != real_cfg_dir or \
                os.environ["JARVIS_ASSISTANT_CONFIG"] == str(_TEST_ASSISTANT_CONFIG)
    except ImportError:
        pass
    yield
