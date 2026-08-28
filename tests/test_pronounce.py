"""Tests for jarvis.pronounce — the TTS pronunciation dictionary."""
import json

import pytest

from jarvis.pronounce import DEFAULT_PRONUNCIATIONS, Pronunciations


@pytest.fixture
def table(tmp_path):
    return Pronunciations(path=tmp_path / "tts_pronunciations.json")


def test_defaults_rewrite_workshop_jargon(table):
    out = table.apply("VSS runs on the GB10 with CUDA and Ollama.")
    assert out == "V S S runs on the G B ten with kooda and oh-llama."


def test_whole_token_only(table):
    # "GPU" inside "GPUs" is its own entry; "TTS" glued into a word is not.
    assert table.apply("two GPUs") == "two G P Us"
    assert table.apply("xTTSy") == "xTTSy"
    assert table.apply("the TTS engine") == "the T T S engine"


def test_case_rules(table):
    # Upper-case keys are exact; lower-case keys match any casing.
    assert table.apply("gpu") == "gpu"
    assert table.apply("Nvidia NVIDIA nvidia") == "en-vidia en-vidia en-vidia"


def test_symbols(table):
    assert table.apply("load 87% & rising") == "load 87 percent and rising"


def test_user_file_overrides_and_persists(table, tmp_path):
    table.add("Peyrovi", "pay-ROH-vee")
    table.add("CUDA", "coo-dah")               # override a default
    assert table.apply("Mr Peyrovi likes CUDA") == "Mr pay-ROH-vee likes coo-dah"
    data = json.loads((tmp_path / "tts_pronunciations.json").read_text())
    assert data == {"Peyrovi": "pay-ROH-vee", "CUDA": "coo-dah"}
    # A fresh instance reads the same file.
    again = Pronunciations(path=tmp_path / "tts_pronunciations.json")
    assert again.apply("CUDA") == "coo-dah"


def test_empty_spoken_disables_a_default(table):
    table.add("VSS", "")
    assert table.apply("VSS") == "VSS"


def test_remove(table):
    table.add("Foo", "fooo")
    assert table.remove("foo") is True
    assert table.apply("Foo") == "Foo"
    assert table.remove("foo") is False


def test_reloads_when_file_changes(table, tmp_path):
    path = tmp_path / "tts_pronunciations.json"
    path.write_text(json.dumps({"Spark": "the spark"}))
    import os
    os.utime(path, (1, 1))                     # force a distinct mtime
    assert table.apply("Spark") == "the spark"


def test_bad_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "tts_pronunciations.json"
    path.write_text("not json")
    t = Pronunciations(path=path)
    assert t.apply("VSS") == "V S S"
    assert t.user_items() == {}


def test_add_rejects_empty_word(table):
    with pytest.raises(ValueError):
        table.add("  ", "x")


def test_defaults_have_no_plain_english():
    # Guard against someone adding a common word: every default key is
    # jargon-shaped (has a digit, is not lower-case, or is a tool name).
    for key in DEFAULT_PRONUNCIATIONS:
        assert key != key.lower() or any(ch.isdigit() for ch in key) or key in {
            "nvidia", "ollama", "qwen", "pytorch", "pytest", "xdotool", "xclip",
            "nmcli", "ffmpeg", "tkinter", "sudo", "ok", "aarch64", "llama3.2",
            # calendar shorthand (2026-08-28). None is an English word, so
            # matching any casing is safe. "rm" and "sem" were considered and
            # rejected: "rm" would rewrite the shell command as "Room".
            "engr", "bldg", "dept", "lect", "appt", "tamu"}


# --------------------------------------------------- spoken abbreviations
#
# 2026-08-28: Jarvis read a calendar title aloud as "ENGR" rather than
# "Engineering". Course titles arrive from Navigate360 already abbreviated
# (MAGNETIC RESONANCE ENGR, Bldg, ETB), so the text is never written out for
# him -- the pronunciation layer is the only place to expand it.
@pytest.mark.parametrize("raw,spoken", [
    ("MAGNETIC RESONANCE ENGR", "Engineering"),
    ("Wisenbaker Engineering Bldg 049", "Building"),
    ("ECEN 442 Lab", "Lab"),
])
def test_course_abbreviations_are_spoken_in_full(raw, spoken):
    from jarvis.pronounce import Pronunciations
    said = Pronunciations(path=None).apply(raw)
    assert spoken.lower() in said.lower(), f"{raw!r} -> {said!r}"


def test_expansion_is_whole_token_only():
    """'Bldg' must expand; a word merely containing those letters must not."""
    from jarvis.pronounce import Pronunciations
    p = Pronunciations(path=None)
    assert "Building" in p.apply("Bldg 049")
    assert p.apply("Engrave the plate") == "Engrave the plate"
