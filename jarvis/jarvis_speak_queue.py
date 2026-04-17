"""Jarvis speak queue — file-based communication between Claude and TTS.

Claude writes lines prefixed with "Jarvis:" to a queue file.
The voice GUI watches this file and speaks any new lines.

Usage from Claude Code:
    echo "Jarvis: Hello, how can I help?" >> /tmp/vss_voice/speak_queue.txt
"""

from pathlib import Path

SPEAK_QUEUE = Path("/tmp/vss_voice/speak_queue.txt")


def say(text):
    """Queue a message for Jarvis to speak."""
    SPEAK_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPEAK_QUEUE, "a") as f:
        f.write(text.strip() + "\n")
