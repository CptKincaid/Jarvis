"""Startup regression tests.

Catches issues that only manifest when the GUI is launched as a script
(as the hotword daemon does) — the tests below would have prevented
the `jarvis/logging.py` vs stdlib `logging` shadowing bug.
"""

import importlib
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def test_stdlib_logging_has_getLogger():
    """Regression: if jarvis has a module named `logging` on sys.path it
    will shadow stdlib `logging`, breaking PIL and most every other
    library that calls `logging.getLogger(__name__)` at import time."""
    import logging
    assert hasattr(logging, "getLogger"), (
        "stdlib `logging` module is missing getLogger. "
        "A jarvis module named `logging` is probably shadowing it."
    )
    assert hasattr(logging, "INFO")
    assert callable(logging.getLogger)


def test_no_jarvis_top_level_module_named_logging():
    """Explicit: there must not be a `jarvis/logging.py` file; it must
    be renamed (e.g. `jarvis_logging.py`). This test locks that in."""
    assert not (REPO / "jarvis" / "logging.py").exists(), (
        "jarvis/logging.py shadows stdlib logging when launched as a "
        "script. Rename to jarvis/jarvis_logging.py."
    )


@pytest.mark.parametrize("module_name", [
    "jarvis.voice_input_gui",
    "jarvis.recording",
    "jarvis.transcription",
    "jarvis.dispatcher",
    "jarvis.animation",
    "jarvis.jarvis_logging",
    "jarvis.jarvis_tts",
    "jarvis.stt_engine",
    "jarvis.speaker_verification",
    "jarvis.jarvis_brain",
    "jarvis.jarvis_agent",
    "jarvis.memory",
    "jarvis.context",
    "jarvis.speak_queue_auth",
    "jarvis.jarvis_speak_queue",
])
def test_module_imports_cleanly(module_name):
    """Regression: every jarvis module must import without raising.
    Any ImportError here points to a missing dep, circular import,
    or syntax issue that would break the GUI at launch time."""
    importlib.import_module(module_name)


def test_voice_input_gui_imports_work_with_script_style_path():
    """Simulate script-mode launch: the file inserts SCRIPT_DIR (repo
    root) at sys.path position 0. If any jarvis module happens to have
    the same name as a stdlib module, stdlib imports may break."""
    repo = str(REPO)
    saved_path = sys.path[:]
    try:
        if repo not in sys.path:
            sys.path.insert(0, repo)
        # Import (or re-import) stdlib modules that PIL/numpy etc. rely on
        import logging as _logging
        import importlib as _importlib
        # Force-reload stdlib logging to defeat any earlier caching
        _importlib.reload(_logging)
        assert hasattr(_logging, "getLogger")
    finally:
        sys.path[:] = saved_path
