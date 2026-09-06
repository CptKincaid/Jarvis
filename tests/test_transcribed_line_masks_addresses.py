"""An address he SAID or SPELLED never reaches jarvis.log raw -- not on
the ``Transcribed:`` line either, and not on the speculative-decode line.

The round-2 verdict on fix-spelled-address (09-06): every commander line
masks the address, and the ONE line upstream of all of them --
jarvis/transcriber.py's ``Transcribed: %r`` at INFO -- wrote it whole.
So did its twin in app._log_transcript (the quiet decode, when an owner
phrase is set) and app._loggable (the speculative-decode line), neither
of which the verdict named: the guard-one-half shape again. The words
are written down by ONE function now, transcriber.shown_words, and the
census at the end of this file pins every site to it.

The spelled shapes below are the ones whisper has actually written for a
letter-by-letter address (the 09-05 log, the 09-06 verdict). Every name
and domain is invented; no recording is played and no model is loaded.
"""
from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis import outbox
from jarvis import transcriber as transcriber_mod
from jarvis.config import PATHS
from jarvis.transcriber import REDACTED_WORDS, Transcriber, shown_words
from tests.test_decode_bounds import FakeFasterWhisper, _audio

SPELLED = "send an email to q. z. v. k. b. w. 7. at example.com"
SPELLED_SHOWN = "send an email to q… at example.com"


# ---- the mask reads the spelled shape -----------------------------------
@pytest.mark.parametrize("said, want", [
    (SPELLED, SPELLED_SHOWN),
    ("send an email to q-z-v, k-b-w-7 at example.com",
     "send an email to q… at example.com"),
    ("q-z-v, k-b-w-7. at example.com", "q… at example.com"),
    ("q-z-v, k-b-w-7, at example dot com", "q… at example dot com"),
    ("d.a.n at example.com", "d… at example.com"),
    ("send an email to q z v k b w seven at example dot com",
     "send an email to q… at example dot com"),
    ("q, z, v, k, b, w, 7 at example dot com", "q… at example dot com"),
    ("Q. Z. V. K. B. W. 7 at example dot com.", "Q… at example dot com."),
    ("it's q z v at gmail. com, I think", "it's q… at gmail. com, I think"),
    ("it’s q z v at gmail dot com", "it’s q… at gmail dot com"),
    ("no, send it to d a n a at example dot com instead",
     "no, send it to d… at example dot com instead"),
    ("q. z. v. at example dot com and dana at example dot org",
     "q… at example dot com and d… at example dot org"),
])
def test_mask_addresses_masks_the_spelled_shape(said, want):
    assert outbox.mask_addresses(said) == want


@pytest.mark.parametrize("prose", [
    "grades a, b, c at noon. Come by after",
    "plan a. b at the dot",
    "vitamin b 12 at noon",
    "rooms 1, 2, 3 at the end",
    "a b c",
    "q. z. v. k. b. w. 7.",
    "I'm at home. See you at six.",
    "what is plan B",
    "is it a or b at the moment",
])
def test_the_spelled_mask_leaves_prose_alone(prose):
    """A run of letters is not an address until a domain follows it."""
    assert outbox.mask_addresses(prose) == prose


def test_a_single_spelled_character_is_the_said_masks_case():
    """One character is not a run; the said-shape rule already drops a
    one-letter local part whole, and the two rules must agree."""
    assert outbox.mask_addresses("send it to q at example dot com") == \
        "send it to … at example dot com"


# ---- the one function every site goes through ----------------------------
def test_shown_words_redacts_first_masks_second_and_leaves_empty_alone():
    assert shown_words(SPELLED, redact=True) == REDACTED_WORDS
    assert shown_words(SPELLED) == SPELLED_SHOWN
    assert shown_words("what time is it") == "what time is it"
    assert shown_words("", redact=True) == ""
    assert shown_words("") == ""
    assert shown_words(None) is None


# ---- the transcriber's own line ------------------------------------------
def test_the_transcribed_line_masks_the_spelled_address(
        caplog, tmp_path, monkeypatch):
    """The REAL Transcriber over a fake faster-whisper model, CPU branch:
    the line carries the mask and its numbers; the RESULT carries the
    decode untouched, because the mask is for the log, not the commander."""
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    tr = Transcriber()
    tr._model = FakeFasterWhisper(segments=[
        SimpleNamespace(text=SPELLED, avg_logprob=-0.30, compression_ratio=1.2)])
    tr._gpu, tr._backend = False, "CPU int8"
    with caplog.at_level(logging.INFO, logger=transcriber_mod.log.name):
        res = tr.transcribe(_audio(2.0))
    lines = [r.getMessage() for r in caplog.records if "Transcribed" in str(r.msg)]
    assert lines, "the Transcribed: line was not written"
    assert all("q… at example.com" in ln for ln in lines), lines
    assert all("k. b. w. 7" not in ln for ln in lines), lines
    assert any("avg_logprob=-0.30" in ln for ln in lines), lines
    assert "q. z. v. k. b. w. 7." in res.text


# ---- the quiet decode's twin, and the speculative line ------------------
def _bound(name, *, phrase: bool):
    a = SimpleNamespace(_owner_has_phrase=lambda: phrase)
    return getattr(app_mod.JarvisApp, name).__get__(a)


def test_log_transcript_masks_the_address_after_the_gate_cleared_it(caplog):
    with caplog.at_level(logging.INFO, logger=app_mod.log.name):
        _bound("_log_transcript", phrase=True)(
            SimpleNamespace(text="send it to dana at example dot com",
                            confidence=-0.20))
    lines = [r.getMessage() for r in caplog.records if "Transcribed" in str(r.msg)]
    assert lines == ["Transcribed: 'send it to d… at example dot com'"
                     " (avg_logprob=-0.20)"]


def test_loggable_masks_the_speculative_line_and_still_redacts():
    said = "q-z-v, k-b-w-7 at example.com"
    assert _bound("_loggable", phrase=False)(said) == repr("q… at example.com")
    assert _bound("_loggable", phrase=True)(said) == repr(gate_mod.REDACTED_TEXT)
    assert _bound("_loggable", phrase=False)("") == repr("")


# ---- the census: no site writes the words around shown_words ------------
def _first_arg(rest: str) -> str:
    """The first argument after the format string, parens balanced:
    "shown, avg_conf)" -> "shown"; "shown_words(text)," -> "shown_words(text)"."""
    depth = 0
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return rest[:i].strip()
            depth -= 1
        elif ch == "," and depth == 0:
            return rest[:i].strip()
    return rest.strip()


def test_every_site_that_writes_the_words_goes_through_shown_words():
    root = Path(app_mod.__file__).parent
    sites: dict = {}
    for path in sorted(root.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r'log\.\w+\(\s*"Transcribed:[^"]*"\s*,\s*([^\n]*)', src):
            sites.setdefault(path.name, []).append(_first_arg(m.group(1)))
    assert set(sites) == {"transcriber.py", "app.py"}, sites
    assert sites["app.py"] == ["shown_words(text)"], sites
    assert sites["transcriber.py"] and all(a == "shown" for a in sites["transcriber.py"]), sites
    assert "shown = shown_words(text, redact=redact)" in \
        (root / "transcriber.py").read_text(encoding="utf-8")
    assert "shown_words(" in inspect.getsource(app_mod.JarvisApp._loggable)
