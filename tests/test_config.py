"""Tests for jarvis.config — old-format load, defaults, unknown-key survival."""
import json

import pytest

import jarvis.config as config
from jarvis.config import Config


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "voice_settings.json"
    monkeypatch.setattr(config.PATHS, "SETTINGS_FILE", path)
    return path


def test_load_missing_file_gives_defaults(settings_file):
    cfg = Config.load()
    assert cfg.model == "small"
    assert cfg.talkback is True
    # 8.0 until 2026-08-27: together with the 5 s grace it put a ~13 s
    # floor under every utterance. Tuned for wake-word commands; raise
    # both again for dictation, where long pauses are normal.
    assert cfg.silence_timeout == 2.5
    assert cfg.silence_grace == 1.5
    assert cfg._extra == {}


def test_load_old_format_fills_defaults(settings_file):
    settings_file.write_text(json.dumps({
        "model": "medium",
        "gpu": 1,
        "hotword": False,
        "speaker_threshold": "0.55",     # old files may store as string
    }))
    cfg = Config.load()
    assert cfg.model == "medium"
    assert cfg.gpu == 1
    assert cfg.hotword is False
    assert cfg.speaker_threshold == pytest.approx(0.55)   # coerced to float
    # untouched keys keep defaults
    assert cfg.language == "English"
    assert cfg.tts_engine == "edge"
    assert cfg.window_geometry == ""


def test_unknown_keys_survive_round_trip(settings_file):
    settings_file.write_text(json.dumps({
        "model": "small",
        "mystery_key": 42,
        "another_unknown": {"nested": True},
    }))
    cfg = Config.load()
    assert cfg._extra == {"mystery_key": 42,
                          "another_unknown": {"nested": True}}
    cfg.model = "large"
    cfg.save()

    data = json.loads(settings_file.read_text())
    assert data["mystery_key"] == 42
    assert data["another_unknown"] == {"nested": True}
    assert data["model"] == "large"
    # every known field is written, including new V3 keys
    assert "window_geometry" in data
    assert "talkback" in data


def test_bad_value_falls_back_to_default(settings_file):
    settings_file.write_text(json.dumps({"gpu": "not-a-number"}))
    cfg = Config.load()
    assert cfg.gpu == 0


def test_corrupt_file_gives_defaults(settings_file):
    settings_file.write_text("{ this is not json")
    cfg = Config.load()
    assert cfg.model == "small"


def test_update_sets_and_saves(settings_file):
    cfg = Config.load()
    cfg.update(model="tiny", talkback=False)
    assert cfg.model == "tiny"
    data = json.loads(settings_file.read_text())
    assert data["model"] == "tiny"
    assert data["talkback"] is False


def test_save_is_atomic_no_tmp_left_behind(settings_file):
    cfg = Config.load()
    cfg.save()
    assert settings_file.exists()
    assert not settings_file.with_suffix(".tmp").exists()
