"""The 2026-08-31 volume regression: "his voice is quiet for some reason now".

The reason was the 08-30 cutover from the hosted Fish voice to the local F5
sidecar. XTTS was the ONLY engine whose render passed through
``apply_output_gain``, and F5 took its place without inheriting it -- F5
clones the reference clip's LEVEL along with its timbre, and reference clip
0341 sits at peak 0.303 / -26.8 dBFS RMS.

Measured across all 400 renders sitting in his live speech cache on
2026-08-31, i.e. the actual audio he was complaining about:

    peak   p05 0.222   p50 0.273   p95 0.329   p99 0.347   MAX 0.353
    rms                        median -27.7 dBFS

against the ~0.97 peak the XTTS path was tuned to hit. Nothing about the
voice changed; about 9 dB of level went missing at the cutover.
"""
import wave

import numpy as np
import pytest

from jarvis import tts as tts_mod

# The measured peak of a real F5 render and the loudest one in the whole
# 400-file cache. The gain is calibrated against the LOUDEST, so no chunk is
# ever pulled back by the ceiling and the level cannot pump mid-sentence.
MEASURED_F5_PEAK = 0.273
LOUDEST_F5_PEAK = 0.353


def test_f5_is_gained_like_xtts_was():
    """The regression itself: the local clone engine had no output gain."""
    assert tts_mod.output_gain_for("f5") > 1.0, (
        "F5 renders at the reference clip's own level (~ -27 dBFS) and "
        "needs the same output gain XTTS always had -- this is 'his voice "
        "is quiet for some reason now'")


def test_a_real_f5_render_reaches_a_room_level():
    """A render at his cache's median peak must come out near full scale."""
    render = np.full(1000, MEASURED_F5_PEAK, dtype=np.float32)
    out = tts_mod.apply_output_gain(render, tts_mod.output_gain_for("f5"))
    assert float(np.abs(out).max()) > 0.7, (
        "a typical F5 reply still peaks below 0.7 -- that is the quiet voice")


def test_the_loudest_measured_render_never_hits_the_ceiling():
    """Calibration check: 0.353 is the hottest of his 400 cached renders.

    If the gain ever reaches the ceiling on a real render, apply_output_gain
    pulls THAT chunk back and the level pumps between chunks of one
    sentence -- the exact thing the constant-gain rule exists to prevent."""
    assert LOUDEST_F5_PEAK * tts_mod.output_gain_for("f5") <= 0.99


def test_the_hosted_voice_is_left_at_the_level_it_was_approved_at():
    """fish and edge were never measured from this box; 1.0 means untouched."""
    assert tts_mod.output_gain_for("fish") == 1.0
    assert tts_mod.output_gain_for("edge") == 1.0


def test_gain_wav_file_raises_the_file_the_sidecar_wrote(tmp_path):
    """The F5 sidecar hands back a FILE, not samples."""
    path = tmp_path / "render.wav"
    quiet = (np.sin(np.linspace(0, 400, 8000)) * MEASURED_F5_PEAK)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((quiet * 32767).astype("<i2").tobytes())

    assert tts_mod.gain_wav_file(str(path), tts_mod.output_gain_for("f5"))

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 24000, "the sample rate must survive"
        data = np.frombuffer(w.readframes(w.getnframes()),
                             dtype="<i2").astype(np.float32) / 32768
    assert float(np.abs(data).max()) > 0.7


def test_gain_wav_file_never_raises_into_a_turn(tmp_path):
    """A level that cannot be raised is a quiet reply; an exception here
    would be silence, and silence from Jarvis reads as a broken assistant."""
    assert tts_mod.gain_wav_file(str(tmp_path / "does-not-exist.wav"),
                                 2.8) is False


def test_the_gain_is_part_of_the_cache_key():
    """Without this the 400 pre-fix renders in his cache would have gone on
    playing at the old quiet level for every line he had heard once, and
    the fix would have looked like it only half worked."""
    tts = tts_mod.TTS(engine="f5", cache=False)
    before = tts._cache_key("f5", "Noted, sir.")
    original = tts_mod.ENGINE_OUTPUT_GAIN["f5"]
    try:
        tts_mod.ENGINE_OUTPUT_GAIN["f5"] = original + 0.5
        after = tts._cache_key("f5", "Noted, sir.")
    finally:
        tts_mod.ENGINE_OUTPUT_GAIN["f5"] = original
    assert before != after, (
        "changing the level did not change the key, so a louder Jarvis "
        "would replay the quiet audio he already has on disk")


@pytest.mark.parametrize("engine", ["f5", "xtts"])
def test_the_gain_stays_constant_across_chunks(engine):
    """Per-chunk normalisation would make one sentence pump."""
    gain = tts_mod.output_gain_for(engine)
    quiet = np.full(500, 0.05, dtype=np.float32)
    loud = np.full(500, 0.20, dtype=np.float32)
    assert np.isclose(tts_mod.apply_output_gain(quiet, gain)[0] / 0.05,
                      tts_mod.apply_output_gain(loud, gain)[0] / 0.20)
