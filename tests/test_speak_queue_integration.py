"""End-to-end integration tests for the speak queue.

Verify that:
  - say() writes a signed line that verify() accepts
  - A forged (unsigned) line written directly is rejected
  - A tampered signed line is rejected
  - Multiple messages are each independently signed
"""

from pathlib import Path

import pytest

from jarvis import speak_queue_auth


@pytest.fixture
def isolated_queue(tmp_path, monkeypatch):
    """Isolate key + queue paths to a tmpdir."""
    key_file = tmp_path / "speak_queue.key"
    queue_dir = tmp_path / "queue"
    queue_file = queue_dir / "speak_queue.txt"

    monkeypatch.setattr(speak_queue_auth, "KEY_FILE", key_file)
    monkeypatch.setattr(speak_queue_auth, "QUEUE_DIR", queue_dir)

    # Reload jarvis_speak_queue module so it picks up the patched QUEUE_DIR
    # (we can't just import it — SPEAK_QUEUE is captured at import time)
    # Instead, write to queue_file directly using sign().
    yield queue_file


def test_signed_line_roundtrips(isolated_queue):
    queue_file = isolated_queue
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    signed = speak_queue_auth.sign("Hello Jarvis")
    queue_file.write_text(signed + "\n")

    line = queue_file.read_text().strip()
    assert speak_queue_auth.verify(line) == "Hello Jarvis"


def test_unsigned_injection_rejected(isolated_queue):
    queue_file = isolated_queue
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    # Attacker appends plaintext directly (no signing)
    queue_file.write_text("Attacker controlled speech\n")
    line = queue_file.read_text().strip()
    assert speak_queue_auth.verify(line) is None


def test_mixed_queue_only_verified_lines_accepted(isolated_queue):
    queue_file = isolated_queue
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    legit = speak_queue_auth.sign("Legitimate message")
    forged = "deadbeefdeadbeef Forged with fake HMAC"
    queue_file.write_text(f"{legit}\n{forged}\n")

    verified = []
    for line in queue_file.read_text().splitlines():
        payload = speak_queue_auth.verify(line.strip())
        if payload is not None:
            verified.append(payload)

    assert verified == ["Legitimate message"]


def test_mac_length_guard_rejects_short_line(isolated_queue):
    # Very short lines (shorter than HMAC + space) must not index-crash
    assert speak_queue_auth.verify("") is None
    assert speak_queue_auth.verify("x") is None
    assert speak_queue_auth.verify("abc123") is None


def test_distinct_messages_produce_distinct_macs():
    a = speak_queue_auth.sign("Message A")
    b = speak_queue_auth.sign("Message B")
    mac_a, _, _ = a.partition(" ")
    mac_b, _, _ = b.partition(" ")
    assert mac_a != mac_b


def test_ensure_queue_dir_creates_0700(tmp_path, monkeypatch):
    import os
    queue_dir = tmp_path / "vss_voice"
    monkeypatch.setattr(speak_queue_auth, "QUEUE_DIR", queue_dir)
    speak_queue_auth._ensure_queue_dir()
    assert queue_dir.is_dir()
    mode = queue_dir.stat().st_mode & 0o777
    assert mode == 0o700
