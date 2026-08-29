"""The F5-TTS engine: a sidecar, because it cannot share Jarvis's venv.

WHY IT IS A SIDECAR. Jarvis runs from ~/vss_env, and so does VSS
(vss_env/aiws_system/api.py). `uv pip install --dry-run f5-tts` against that
venv REMOVES fastapi and downgrades rich, which would break the VSS API. So
f5-tts lives in its own venv at ~/.local/share/jarvis-f5/venv and jarvis/tts.py
talks to a resident server over a unix socket.

SETTINGS, all chosen by listening on 2026-08-28:
  nfe_step=8   the latency lever. F5 is flow-matching, so it runs a FIXED
               number of denoising steps whatever the text length -- the
               default 32 costs ~1.9s per call regardless. 8 was the lowest
               step count without audible artefacts.
  speed=0.70   preferred at EVERY chunk length tested (22/43/84/122 chars).
               0.50 was rated too slow, "too long of a pause at periods" --
               lowering speed stretches the silences, not just the words.
"""
import json
import socket
import types

import pytest

from jarvis import tts as tts_mod
from jarvis.tts import TTS


def test_f5_is_a_registered_engine():
    assert "f5" in tts_mod._ENGINES


def test_the_settings_are_the_ones_chosen_by_ear():
    assert tts_mod.F5_PARAMS["nfe_step"] == 8
    assert tts_mod.F5_PARAMS["speed"] == 0.7


def test_the_reference_and_its_transcript_both_exist():
    # F5 is flow-matching infill: it needs the reference TEXT as well as audio,
    # unlike XTTS. A missing transcript is a silent quality cliff.
    assert tts_mod.F5_REF.exists(), tts_mod.F5_REF
    assert tts_mod.F5_REF_TEXT.exists(), tts_mod.F5_REF_TEXT
    assert len(tts_mod.F5_REF_TEXT.read_text().strip()) > 40


def test_f5_lives_outside_the_jarvis_venv():
    """If this ever points into vss_env, the VSS API is about to break."""
    p = str(tts_mod.F5_PYTHON)
    assert "vss_env" not in p, p
    assert "jarvis-f5" in p, p


def test_cache_key_separates_f5_from_the_other_engines(tmp_path):
    t = TTS(engine="f5", cache_dir=tmp_path / "c")
    assert t._cache_key("f5", "hello") != t._cache_key("xtts", "hello")
    assert t._cache_key("f5", "hello") != t._cache_key("edge", "hello")


def test_cache_key_changes_when_the_voice_settings_change(tmp_path, monkeypatch):
    """A settings change must never replay audio rendered at the old ones."""
    t = TTS(engine="f5", cache_dir=tmp_path / "c")
    before = t._cache_key("f5", "hello")
    monkeypatch.setitem(tts_mod.F5_PARAMS, "speed", 0.5)
    assert t._cache_key("f5", "hello") != before


def test_synthesis_sends_the_text_and_the_settings(tmp_path, monkeypatch):
    sent = {}

    def fake_request(payload, timeout=120):
        sent.update(payload)
        return {"ok": True, "seconds": 1.2}

    monkeypatch.setattr(tts_mod, "_f5_request", fake_request)
    t = TTS(engine="f5", cache_dir=tmp_path / "c")
    out = tmp_path / "out.wav"
    t._synth_f5("Good evening, sir.", str(out))
    assert sent["text"] == "Good evening, sir."
    assert sent["out"] == str(out)
    assert sent["nfe"] == 8 and sent["speed"] == 0.7


def test_a_sidecar_error_is_raised_not_swallowed(tmp_path, monkeypatch):
    """A silent failure here is silence from Jarvis, which reads as broken."""
    monkeypatch.setattr(tts_mod, "_f5_request",
                        lambda payload, timeout=120: {"ok": False,
                                                      "error": "boom"})
    t = TTS(engine="f5", cache_dir=tmp_path / "c")
    with pytest.raises(RuntimeError, match="boom"):
        t._synth_f5("hello", str(tmp_path / "o.wav"))


def test_load_falls_back_when_the_sidecar_will_not_start(tmp_path, monkeypatch):
    """Same contract as XTTS: a failed load degrades to edge, never silence."""
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda: False)
    t = TTS(engine="f5", cache_dir=tmp_path / "c")
    assert t.load() is False
    assert t.engine == "edge"
