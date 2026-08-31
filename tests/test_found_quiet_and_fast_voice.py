"""The voice settings chosen by ear on 2026-08-28, pinned so they cannot drift.

Three things were settled by listening, not by measurement — and in two of
them the measurements pointed the WRONG WAY, so they are recorded here with
the reasoning:

1. REFERENCE. The film clip (jarvis_voice_ref.wav) scored best on every DSP
   metric I could compute and sounded worst. A 35.8 s reference built from
   three different Fish Audio renderings of Bettany won by a wide margin.
   Three sentences, not one repeated, so XTTS has full phonetic coverage.

2. LEVEL OF THE REFERENCE. Loudness-normalising the reference LOST, twice —
   once on the film clip, once on this one. The reference stays as recorded
   at roughly -27 LUFS. Do not "fix" it.

3. SPEED. 1.16 was clipped and robotic; 1.00 dragged. 1.05 was the one the
   user called phenomenal. repetition_penalty 2.0 (XTTS's own default) beat
   the shipped 5.0 by ear.

The remaining complaint was that the result is quiet. Measured, all four
candidates sat within 0.6 LUFS of each other (-17.7 to -18.3), so nothing had
regressed -- the calmer delivery simply reads quieter. Level is therefore
fixed where it belongs, on the OUTPUT, with a constant gain that cannot
change timbre. It must be CONSTANT rather than per-utterance normalisation:
the pipelined path synthesises chunk by chunk, and normalising each chunk
independently would make the volume pump between them mid-sentence.
"""
import numpy as np

from jarvis import tts as tts_mod


def test_the_reference_is_the_three_sample_fish_set():
    assert tts_mod.VOICE_REF.name == "jarvis_voice_ref_fish3.wav"
    assert tts_mod.VOICE_REF.exists(), "the chosen reference is missing"


def test_the_voice_parameters_are_the_ones_chosen_by_ear():
    assert tts_mod.XTTS_PARAMS["speed"] == 1.05
    assert tts_mod.XTTS_PARAMS["repetition_penalty"] == 2.0


def test_gain_is_constant_so_chunks_cannot_pump():
    """The same gain must apply regardless of how loud the chunk is."""
    quiet = np.full(1000, 0.05, dtype=np.float32)
    loud = np.full(1000, 0.30, dtype=np.float32)
    gq = tts_mod.apply_output_gain(quiet)
    gl = tts_mod.apply_output_gain(loud)
    assert np.isclose(gq[0] / quiet[0], gl[0] / loud[0]), (
        "gain varied with input level -- that is per-chunk normalisation, "
        "and it makes the volume pump between chunks of one sentence")


def test_gain_actually_raises_the_level():
    a = np.full(1000, 0.2, dtype=np.float32)
    assert np.abs(tts_mod.apply_output_gain(a)).max() > 0.2


def test_a_hot_chunk_is_never_clipped():
    """A loud outlier is pulled back rather than driven past full scale."""
    hot = np.linspace(-0.98, 0.98, 2000).astype(np.float32)
    out = tts_mod.apply_output_gain(hot)
    assert np.abs(out).max() <= 1.0, "output clipped"
    assert np.abs(out).max() > 0.9, "a hot chunk should stay loud, not duck"


def test_silence_survives_untouched():
    z = np.zeros(100, dtype=np.float32)
    assert not np.any(tts_mod.apply_output_gain(z))
