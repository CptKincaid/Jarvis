"""Jarvis speak queue — file-based communication between Claude and TTS.

Claude writes HMAC-signed lines to the queue file so untrusted local
processes cannot make Jarvis speak arbitrary text.

Usage from Claude Code:
    from jarvis.jarvis_speak_queue import say
    say("Jarvis: Hello, how can I help?")
"""

from pathlib import Path

from jarvis.speak_queue_auth import sign, _ensure_queue_dir

SPEAK_QUEUE = Path("/tmp/vss_voice/speak_queue.txt")


def say(text):
    """Queue a signed message for Jarvis to speak."""
    _ensure_queue_dir()
    signed = sign(text.strip())
    with open(SPEAK_QUEUE, "a") as f:
        f.write(signed + "\n")
