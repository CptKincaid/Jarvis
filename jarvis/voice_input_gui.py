"""Legacy entry point — the V1 monolith was replaced by Jarvis V3.

See docs/specs/2026-08-25-jarvis-v3-overhaul.md. This shim keeps the old
launch command (`python jarvis/voice_input_gui.py`) and the hotword daemon's
process detection working. The V1 implementation lives in git history
(last full version: commit 6bfe446 + one working-tree patch).
"""
from jarvis.app import main

if __name__ == "__main__":
    main()
