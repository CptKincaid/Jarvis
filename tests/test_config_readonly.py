"""Loading the assistant config is a READ. Importing jarvis is a READ.

Round 2 of brain-room put ``SETTINGS = model_settings()`` at the top of
jarvis/brain.py, which called AssistantConfig.load() the moment anything
imported the module -- and load() then SAVED ~/.config/jarvis/assistant.json
whenever DEFAULTS had gained a key. Result, 2026-09-04 15:08: an agent's
import rewrote his live, secret-bearing config (six brain.* keys appeared
in it; the app never ran). Two doors, both closed here:

  * AssistantConfig.load() never writes. What it finds (missing, corrupt,
    loose mode, keys DEFAULTS has gained) it notes in ``disk_state``; the
    write is ensure_defaults(), which ONLY jarvis.app calls, once.
  * jarvis.brain builds its settings on first use (or from the config the
    app hands it), never at import.

The load-bearing test is the first one: a fresh interpreter, HOME pointed
at a temp dir holding a config that is BOTH stale (two keys) and loose
(0644) -- every trigger armed -- imports every jarvis.* module and calls
load(); the file's bytes, mtime and mode must not move. Measured on the
round-2 sources before the fix: the same walk grew the file from 38 to
12 632 bytes and chmod'ed it, in 2.0 s.
"""
import json
import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

STALE_CONFIG = {"version": 1, "user": {"name": "T"}}


def _sub(env, code, timeout=300):
    """Run ``code`` in a fresh interpreter with ONLY ``env`` (no inherited
    JARVIS_* or HOME), cwd at the repo root."""
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONDONTWRITEBYTECODE": "1"}
    base.update(env)
    return subprocess.run([PY, "-c", textwrap.dedent(code)], cwd=str(REPO),
                          env=base, capture_output=True, text=True,
                          timeout=timeout)


def _armed_config(home: Path) -> Path:
    path = home / ".config" / "jarvis" / "assistant.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(STALE_CONFIG) + "\n", encoding="utf-8")
    os.chmod(path, 0o644)                    # loose on purpose
    return path


def _stamp(path: Path):
    st = path.stat()
    return (path.read_bytes(), st.st_mtime_ns, stat.S_IMODE(st.st_mode))


def test_importing_every_jarvis_module_and_loading_leaves_the_config_untouched(
        tmp_path):
    home = tmp_path / "home"
    path = _armed_config(home)
    before = _stamp(path)
    res = _sub({"HOME": str(home), "JARVIS_LOG_DIR": str(tmp_path / "logs")}, """
        import importlib, pkgutil, sys, os, json
        sys.path.insert(0, os.getcwd())
        import jarvis
        fails = {}
        names = []
        for m in pkgutil.walk_packages(jarvis.__path__, "jarvis."):
            names.append(m.name)
            try:
                importlib.import_module(m.name)
            except BaseException as e:      # noqa: BLE001 - report, keep going
                fails[m.name] = f"{type(e).__name__}: {str(e)[:100]}"
        from jarvis.assistant_config import AssistantConfig, config_path
        cfg = AssistantConfig.load()
        print(json.dumps({"n": len(names), "fails": fails,
                          "path": str(config_path()),
                          "state": cfg.disk_state,
                          "brain_in": "jarvis.brain" in sys.modules,
                          "brain_built": "NUM_CTX" in vars(sys.modules["jarvis.brain"])}))
    """)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["n"] >= 100, out                 # the walk really walked
    assert out["fails"] == {}, out["fails"]     # and every module imported
    assert out["path"] == str(path)             # the default path, under HOME
    assert out["brain_in"] and not out["brain_built"]   # imported, not built
    # every write trigger was armed and seen...
    assert out["state"]["new_keys"] is True
    assert out["state"]["loose_mode"] is True
    # ...and nothing moved
    assert _stamp(path) == before
    assert not path.with_name(path.name + ".bad").exists()
    assert sorted(p.name for p in path.parent.iterdir()) == ["assistant.json"]


def test_brain_import_reads_no_config_and_first_use_reads_without_writing(
        tmp_path):
    home = tmp_path / "home"
    path = _armed_config(home)
    before = _stamp(path)
    res = _sub({"HOME": str(home), "JARVIS_LOG_DIR": str(tmp_path / "logs")}, """
        import sys, os, json
        sys.path.insert(0, os.getcwd())
        from jarvis import assistant_config as ac
        calls = []
        real = ac.AssistantConfig.load.__func__
        def spy(cls, path=None):
            calls.append(str(path))
            return real(cls, path)
        ac.AssistantConfig.load = classmethod(spy)
        import jarvis.brain as brain
        at_import = list(calls)
        built_at_import = "NUM_CTX" in vars(brain)
        n = brain.NUM_CTX                      # first use builds them
        print(json.dumps({"at_import": at_import, "built_at_import": built_at_import,
                          "after": len(calls), "num_ctx": n,
                          "source": brain.settings_source(),
                          "same": brain.SETTINGS is brain.settings()}))
    """)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["at_import"] == [] and out["built_at_import"] is False
    assert out["after"] == 1 and out["num_ctx"] == 16384
    assert out["source"] == "first use" and out["same"] is True
    assert _stamp(path) == before


def test_the_app_hands_its_config_to_the_brain_and_nothing_reads_twice(tmp_path):
    """configure(config=...) builds the settings from the app's own loaded
    config (source 'app'), so the app process reads the file once."""
    home = tmp_path / "home"
    path = _armed_config(home)
    res = _sub({"HOME": str(home), "JARVIS_LOG_DIR": str(tmp_path / "logs")}, """
        import sys, os, json
        sys.path.insert(0, os.getcwd())
        from jarvis.assistant_config import AssistantConfig
        import jarvis.brain as brain
        cfg = AssistantConfig(data={"brain": {"num_ctx": 4096}})   # memory-only
        brain.configure("gemma4:26b", config=cfg)
        print(json.dumps({"num_ctx": brain.NUM_CTX, "opts": brain.CHAT_OPTIONS["num_ctx"],
                          "source": brain.settings_source()}))
    """)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out == {"num_ctx": 4096, "opts": 4096, "source": "app"}
    assert sorted(p.name for p in path.parent.iterdir()) == ["assistant.json"]


def test_only_the_app_calls_ensure_defaults():
    """Structural: the one write lives in jarvis/app.py, once, right after
    its load(); no script, tool or service module calls it."""
    callers = {}
    for root in ("jarvis", "scripts"):
        for p in (REPO / root).rglob("*"):
            if not p.is_file() or p.suffix not in ("", ".py"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            n = len(re.findall(r"\.ensure_defaults\(\)", text))
            if n:
                callers[str(p.relative_to(REPO))] = n
    assert callers == {"jarvis/app.py": 1}, callers
    app = (REPO / "jarvis" / "app.py").read_text(encoding="utf-8")
    load_at = app.index("self.assistant = AssistantConfig.load()")
    ensure_at = app.index("self.assistant.ensure_defaults()")
    assert 0 < ensure_at - load_at < 800        # right after the load


def test_the_app_passes_its_config_to_brain_configure():
    app = (REPO / "jarvis" / "app.py").read_text(encoding="utf-8")
    assert "brain_mod.configure(self.assistant.local_model,\n" \
           "                                config=self.assistant)" in app
    brain = (REPO / "jarvis" / "brain.py").read_text(encoding="utf-8")
    assert "\nSETTINGS = model_settings()" not in brain    # the round-2 line
    assert "AssistantConfig.load()" in brain               # lazily, in model_settings


# ------------------------------------------------- load() in-process
@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    from jarvis import assistant_config as ac
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = tmp_path / "cfg" / "assistant.json"
    monkeypatch.setenv(ac.ENV_VAR, str(path))
    return path


def test_load_notes_every_finding_and_touches_nothing(cfg_path):
    from jarvis.assistant_config import DEFAULTS, AssistantConfig
    # missing
    cfg = AssistantConfig.load()
    assert cfg.disk_state == {"missing": True, "corrupt": False,
                              "loose_mode": False, "new_keys": False}
    assert not cfg_path.exists() and not cfg_path.parent.exists()
    assert cfg.get("local_model") == DEFAULTS["local_model"]
    # stale + loose
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps(STALE_CONFIG))
    os.chmod(cfg_path, 0o644)
    before = _stamp(cfg_path)
    cfg = AssistantConfig.load()
    assert cfg.disk_state == {"missing": False, "corrupt": False,
                              "loose_mode": True, "new_keys": True}
    assert cfg.get("user.name") == "T" and cfg.get("alarms.snooze_min") == 10
    assert _stamp(cfg_path) == before
    # corrupt
    cfg_path.write_text("{not json")
    before = _stamp(cfg_path)
    cfg = AssistantConfig.load()
    assert cfg.disk_state["corrupt"] is True and cfg.disk_state["new_keys"] is False
    assert _stamp(cfg_path) == before
    assert not cfg_path.with_name("assistant.json.bad").exists()
    # up to date and private: nothing to do, and ensure_defaults says so
    cfg_path.write_text(json.dumps(DEFAULTS))
    os.chmod(cfg_path, 0o600)
    before = _stamp(cfg_path)
    cfg = AssistantConfig.load()
    assert cfg.disk_state == {"missing": False, "corrupt": False,
                              "loose_mode": False, "new_keys": False}
    assert cfg.ensure_defaults() is False
    assert _stamp(cfg_path) == before


def test_a_memory_only_config_has_nothing_to_ensure():
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig(data={"units": "metric"})
    assert cfg.ensure_defaults() is False
    assert cfg.get("units") == "metric"
