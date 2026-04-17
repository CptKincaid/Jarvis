"""Tests for HMAC-based speak queue authentication."""

import pytest

from jarvis import speak_queue_auth


@pytest.fixture(autouse=True)
def _isolated_key(tmp_path, monkeypatch):
    key_file = tmp_path / "speak_queue.key"
    monkeypatch.setattr(speak_queue_auth, "KEY_FILE", key_file)
    yield


def test_sign_verify_roundtrip():
    signed = speak_queue_auth.sign("Hello Jarvis")
    assert speak_queue_auth.verify(signed) == "Hello Jarvis"


def test_verify_rejects_unsigned_line():
    assert speak_queue_auth.verify("Hello Jarvis") is None


def test_verify_rejects_tampered_payload():
    signed = speak_queue_auth.sign("Hello Jarvis")
    mac, _, _ = signed.partition(" ")
    tampered = f"{mac} Goodbye Jarvis"
    assert speak_queue_auth.verify(tampered) is None


def test_key_file_is_mode_0600():
    speak_queue_auth._load_or_create_key()
    mode = speak_queue_auth.KEY_FILE.stat().st_mode & 0o777
    assert mode == 0o600
