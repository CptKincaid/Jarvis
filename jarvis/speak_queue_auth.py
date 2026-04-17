"""Shared HMAC authentication for the TTS speak queue.

On first use, generates a 32-byte random key at ~/.aiws_trainer/speak_queue.key
(mode 0600). Writers prepend an HMAC-SHA256 truncated to 16 hex chars; readers
verify it.
"""

import hmac
import hashlib
import os
import secrets
from pathlib import Path

KEY_FILE = Path.home() / ".aiws_trainer" / "speak_queue.key"
QUEUE_DIR = Path("/tmp/vss_voice")
HMAC_LEN = 16  # hex chars of truncated hmac


def _ensure_queue_dir():
    """Create /tmp/vss_voice with 0700 permissions, owned by caller."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(QUEUE_DIR, 0o700)
    except OSError:
        pass


def _load_or_create_key() -> bytes:
    """Return the shared key; create it on first use."""
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        key = secrets.token_bytes(32)
        KEY_FILE.write_bytes(key)
        os.chmod(KEY_FILE, 0o600)
        return key
    return KEY_FILE.read_bytes()


def sign(text: str) -> str:
    """Return 'HMAC16 text' format for writing to the queue."""
    key = _load_or_create_key()
    mac = hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()[:HMAC_LEN]
    return f"{mac} {text}"


def verify(line: str) -> str | None:
    """Given a queue line, return the text payload iff the HMAC matches; else None."""
    if len(line) < HMAC_LEN + 2:
        return None
    mac_hex, _, text = line.partition(" ")
    if len(mac_hex) != HMAC_LEN or not text:
        return None
    key = _load_or_create_key()
    expected = hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()[:HMAC_LEN]
    if hmac.compare_digest(mac_hex, expected):
        return text
    return None
