"""Only the app itself may write the live log.

/tmp/vss_voice/jarvis.log is what the running Jarvis writes and what every
diagnosis reads. On 2026-08-29 review agents' scratch scripts and one-off
harnesses imported jarvis.* with no JARVIS_LOG_DIR and their "Auto-stop on
silence" lines and tracebacks landed in it, twice read as live events. The
default now depends on who is running: a `python -m jarvis.*` process gets
the live directory; anything else gets /tmp/jarvis-adhoc.
"""
import os
import subprocess
import sys
from pathlib import Path

from jarvis.logs import ADHOC_LOG_DIR, LIVE_LOG_DIR, _default_log_dir


def test_the_env_override_always_wins():
    assert _default_log_dir(env="/tmp/x", main_name="jarvis.app") == "/tmp/x"
    assert _default_log_dir(env="/tmp/x", main_name="") == "/tmp/x"


def test_only_a_jarvis_module_entrypoint_gets_the_live_dir():
    assert _default_log_dir(env="", main_name="jarvis.app") == LIVE_LOG_DIR
    assert _default_log_dir(env="", main_name="jarvis.voice_check") == LIVE_LOG_DIR
    assert _default_log_dir(env="", main_name="") == ADHOC_LOG_DIR          # a script
    assert _default_log_dir(env="", main_name="__main__") == ADHOC_LOG_DIR
    assert _default_log_dir(env="", main_name="pytest") == ADHOC_LOG_DIR


def test_a_bare_import_from_a_script_does_not_touch_the_live_log():
    env = {k: v for k, v in os.environ.items() if k != "JARVIS_LOG_DIR"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    out = subprocess.run([sys.executable, "-c",
                          "import jarvis.logs as L; print(L.LOG_DIR)"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == ADHOC_LOG_DIR
