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
from pathlib import Path
import socket

import pytest

from jarvis import tts as tts_mod
from jarvis.tts import TTS


def test_f5_is_a_registered_engine():
    assert "f5" in tts_mod._ENGINES


def test_the_settings_are_the_ones_chosen_by_ear():
    """Pinned by BLIND listening, rounds 1-5 (2026-08-29), not by preference.

    nfe_step MUST stay in F5's pruned EPSS timestep table {5,6,7,10,12,16} --
    8 is outside it. speed 0.85 replaced 0.70 once scripts/f5_server.py grew
    the affine duration floor: 0.70 existed only to stretch short utterances,
    which the floor now fixes at its source.
    """
    assert tts_mod.F5_PARAMS["nfe_step"] == 10
    assert tts_mod.F5_PARAMS["speed"] == 0.85


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
    assert sent["nfe"] == 10 and sent["speed"] == 0.85


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


# --------------------------------------------------------- duration floor
#
# F5 starves short text: utils_infer allots the generated span strictly
# PROPORTIONAL to gen-text bytes through the origin, but real speech is
# AFFINE (sec = 0.2540 + 0.04988*bytes, r=0.960 over the 400-clip corpus).
# Before the floor, "Understood." rendered as literal SILENCE and 7 of 24
# short lines failed. These tests pin the two properties that make the fix
# safe to ship: it NEVER touches normal-length text, and it always lengthens.

def _floor():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "f5_server", Path(__file__).resolve().parent.parent / "scripts" / "f5_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reference 0341 as infer() actually sees it, after preprocess_ref_audio_text
REF_FRAMES, REF_TEXT_BYTES = 528, 109


def test_floor_is_a_no_op_on_normal_length_text():
    """The whole safety argument: long-form output must be byte-identical."""
    m = _floor()
    for text in ("You have seven meetings before noon. I'd offer sympathy, "
                 "sir, but you scheduled them yourself.",
                 "The first is Thermodynamics at twenty past three in the "
                 "evening, in the Emerging Technologies Building."):
        assert m.duration_floor(REF_FRAMES, REF_TEXT_BYTES, text, 0.85) is None


def test_floor_binds_on_the_acknowledgements_that_used_to_collapse():
    m = _floor()
    for text in ("Always, sir.", "Understood.", "It is done.", "At once, sir."):
        fd = m.duration_floor(REF_FRAMES, REF_TEXT_BYTES, text, 0.85)
        assert fd is not None, f"{text!r} must be floored"
        # a floor may only ever LENGTHEN: the result must exceed the reference
        assert fd > REF_FRAMES / m.MEL_FPS


def test_floor_boundary_is_where_the_two_lines_cross():
    """C=0.45 stops binding at 41 bytes; that crossover is the design."""
    m = _floor()
    short = m.duration_floor(REF_FRAMES, REF_TEXT_BYTES, "x" * 30, 0.85)
    long_ = m.duration_floor(REF_FRAMES, REF_TEXT_BYTES, "x" * 60, 0.85)
    assert short is not None and long_ is None


def test_floor_handles_empty_text_without_dividing_by_zero():
    m = _floor()
    assert m.duration_floor(REF_FRAMES, REF_TEXT_BYTES, "", 0.85) is None
    assert m.duration_floor(REF_FRAMES, 0, "Always, sir.", 0.85) is None


# ------------------------------------------- fish retirement on no credit
#
# Fish is the default voice while credit lasts; F5 is the local backup and
# becomes PERMANENT once the balance is gone. The two failures must not be
# handled alike: a network blip should cost one chunk, but an exhausted
# balance fails every chunk forever, so retrying the API before each fallback
# would add a doomed round-trip to every sentence Jarvis ever speaks.

class _Resp:
    def __init__(self, code):
        self.status_code = code


class _HttpErr(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        if code is not None:
            self.response = _Resp(code)


def test_out_of_credit_is_detected():
    for exc in (_HttpErr("HTTP 402 Payment Required", 402),
                _HttpErr("payment required"),
                _HttpErr("insufficient balance"),
                _HttpErr("quota exceeded")):
        assert tts_mod._is_out_of_credit(exc), exc


def test_transient_failures_do_NOT_retire_fish():
    """A blip must cost one chunk, never the whole engine."""
    from fish_audio_sdk.exceptions import HttpCodeErr
    for exc in (TimeoutError("fish exceeded 10.0s"),
                ConnectionError("Connection reset by peer"),
                OSError("Name or service not known"),
                _HttpErr("HTTP 500 Internal Server Error", 500),
                _HttpErr("HTTP 429 Too Many Requests", 429),
                # the status is authoritative: a 429/5xx whose body mentions
                # credit or balance used to retire the engine permanently
                HttpCodeErr(429, "rate limited; your credit refills hourly"),
                HttpCodeErr(503, "balance service unavailable"),
                _HttpErr("billing dashboard timed out", 504)):
        assert not tts_mod._is_out_of_credit(exc), exc


def test_the_sdk_exception_shape_is_detected():
    """fish_audio_sdk raises HttpCodeErr(status, message) -- `.status`, never
    `.response.status_code`. The first detector only knew the latter."""
    from fish_audio_sdk.exceptions import HttpCodeErr
    assert tts_mod._is_out_of_credit(HttpCodeErr(402, "Payment Required"))
    assert not tts_mod._is_out_of_credit(HttpCodeErr(500, "boom"))
    # the cases the old substring detector got WRONG, so this pins the fix:
    class _StatusOnly(Exception):
        status = 402
    assert tts_mod._is_out_of_credit(_StatusOnly("nope"))            # no marker text
    assert not tts_mod._is_out_of_credit(HttpCodeErr(500, "credit balance exhausted"))


def test_retire_fish_switches_engine_and_persists(monkeypatch):
    # restore the process-wide engine setting afterwards, or every later test
    # in the session runs against f5
    monkeypatch.setattr(tts_mod.CONFIG, "tts_engine", tts_mod.CONFIG.tts_engine)
    saved = {}
    monkeypatch.setattr(tts_mod.CONFIG, "save", lambda: saved.setdefault("n", 0) or
                        saved.update(n=saved.get("n", 0) + 1))
    t = tts_mod.TTS(engine="fish")
    t.retire_fish("out of credit")
    assert t.engine == tts_mod.FISH_FALLBACK
    assert tts_mod.CONFIG.tts_engine == tts_mod.FISH_FALLBACK
    assert saved.get("n", 0) >= 1, "the choice must be persisted, not just in-memory"


def test_retire_fish_is_idempotent():
    """Called on every failing chunk of a reply; must not thrash config."""
    t = tts_mod.TTS(engine="fish")
    t._engine = tts_mod.FISH_FALLBACK
    t.retire_fish("again")          # already local -> no-op, must not raise
    assert t.engine == tts_mod.FISH_FALLBACK
