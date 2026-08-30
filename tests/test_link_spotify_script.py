"""The Spotify linking script must at least import.

It shipped with `from jarvis.config import AssistantConfig`, which does not
exist -- AssistantConfig lives in jarvis.assistant_config -- so the script
died on its first line for the user, with nothing in the suite to catch it.
Scripts are not otherwise unit-tested here; this is a smoke test, and it also
pins the constructor trap below.
"""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "link_spotify.py"


def _load():
    spec = importlib.util.spec_from_file_location("link_spotify", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # __name__ != "__main__", so no run
    return mod


def test_script_imports_and_exposes_main():
    assert callable(_load().main)


def test_uses_the_load_classmethod_not_the_positional_path():
    """`AssistantConfig(path)` binds the Path to `data`, not `path`, and
    _deep_merge quietly returns a defaults-only config -- which is how a
    configured mailbox was twice reported as "Gmail is not configured".
    Only AssistantConfig.load() resolves the real file."""
    src = SCRIPT.read_text()
    assert "AssistantConfig.load()" in src
    assert "AssistantConfig(PATHS" not in src, "positional path == silent defaults"


def test_token_lands_beside_the_assistant_config(tmp_path, monkeypatch):
    """token_path() reads cfg.path; load() sets it, a bare constructor does
    not. If that regresses the token is written somewhere the tool never
    reads and linking appears to succeed while playback stays broken.

    The env override is cleared first: the test suite's conftest points
    JARVIS_SPOTIFY_TOKEN and the config at the same directory, which made the
    first version of this test pass for any implementation at all."""
    from jarvis.assistant_config import AssistantConfig
    from jarvis.tools.spotify import TOKEN_FILE, token_path

    monkeypatch.delenv("JARVIS_SPOTIFY_TOKEN", raising=False)
    cfg_file = tmp_path / "elsewhere" / "assistant.json"
    cfg = AssistantConfig.load(cfg_file)
    assert cfg.path == cfg_file
    assert token_path(cfg) == cfg_file.parent / TOKEN_FILE
    # and the trap the script fell into: a bare constructor has no path
    bare = AssistantConfig()
    assert bare.path is None
    assert token_path(bare) != cfg_file.parent / TOKEN_FILE
