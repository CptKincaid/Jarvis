"""A term from the Whisper prompt, echoed back as the whole transcript.

2026-09-04, /tmp/vss_voice/jarvis.log, a 4 s follow-up window after a mail
answer (he spoke; speaker verify matched 0.54/0.60/0.47):

    20:58:40.458  '<surname>, ' x 13, avg_logprob -0.21 -> rejected by the
                  ratio gate (compression_ratio 7.68 > 4.00 for 5.3 s),
                  "Say that again, sir?", follow-up mic re-opened.
    20:58:45.815  '<surname>, <surname>, <surname>, <surname>,' at
                  avg_logprob -0.50 on 3.4 s of audio.

The SECOND one passed every gate: four repeats of a 14-char unit compress
to roughly 1.7-2.0, under loop_ratio_limit(3.4) = 3.7; -0.50 clears the
-2.90 confidence floor; collapse_repeats only folds repeated SENTENCES of
three or more words split on .!?; and the intent classifier finds no signal
in a four-word phrase, so it became a "Was that for me?" card showing a
name he never said. The surname came from a one-off appointment title
that jarvis/vocab.py had cut to "<surname>," at the 48-char cap -- exactly
the shape a greedy decoder continues on unclear audio.

jarvis/vocab.py stops the source (one-off titles stay out, no term ends in
a comma). This file is the GATE for whatever the prompt still carries:
transcriber.prompt_echo() names a transcript that is nothing but prompt
words repeated, and TranscribeResult.looping fires on it the same way it
fires on the ratio gate -- no salvage, "Say that again, sir?" once.

The loop-gate comment in transcriber.py protects real insistence ("stop"
x10 at 3.06, "turn it up" x5 at 2.46); those words are not in the prompt,
and the tests here pin that they still pass. Invented names throughout.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

from jarvis.config import PATHS
from jarvis.transcriber import (
    ECHO_MIN_REPEATS,
    TranscribeResult,
    Transcriber,
    loop_ratio_limit,
    prompt_echo,
)
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)
from tests.test_decode_bounds import (
    FakeFasterWhisper,
    FakeGpuWhisper,
    _audio,
    _feed,
)


@pytest.fixture
def firewall(tmp_path, monkeypatch):
    """No vocab file, no CUDA context (same as test_decode_bounds): torch
    is a stub so the GPU branch's cuda.synchronize() cannot initialise a
    device inside the suite."""
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None)))
    return tmp_path

# The invented twin of the live prompt tail's SHAPE: a one-off title cut
# at the 48-char cap so that it ends in "<surname>,", a course, and the
# seed's own terms. Every word of the title is invented.
PROMPT = ("Jarvis, Hunter, calendar, office hours, do not disturb, timer, "
          "BIOSENSORS, Consultation: Remote Session with Z. Q. Quennevex,")

FOUR = "Quennevex, Quennevex, Quennevex, Quennevex,"
THIRTEEN_CUT = "Quennevex, " * 12 + "Quenne"       # sample_len cut it mid-word
A_THEN_B = ("Quennevex, Consultation, Consultation, Consultation, "
            "Consultation,")


# ============================================================ prompt_echo
def test_the_four_fold_echo_that_reached_the_card():
    assert prompt_echo(FOUR, PROMPT) == ("quennevex", 4)


def test_the_thirteen_fold_echo_cut_mid_word():
    assert prompt_echo(THIRTEEN_CUT, PROMPT) == ("quennevex", 12)


def test_a_partial_last_word_must_be_a_strict_prefix_of_the_unit():
    assert prompt_echo("Quennevex, Quennevex, Quennevex, Quenta", PROMPT) \
        == ("", 0)
    # a whole extra word is a fourth repeat, not a cut
    assert prompt_echo("Quennevex, Quennevex, Quennevex, Quennevex", PROMPT) \
        == ("quennevex", 4)


def test_one_prompt_word_then_another_repeated_is_an_echo():
    """The logged 'X, <word>, <word>, <word>, ...' shape: a runaway that
    runs to the END of the decode."""
    assert prompt_echo(A_THEN_B, PROMPT) == ("consultation", 4)
    # the same run cut mid-word by sample_len
    assert prompt_echo(A_THEN_B + " Consul", PROMPT) == ("consultation", 4)


def test_calling_him_three_times_then_a_command_word_is_not_an_echo():
    """Review of 69afb9f: "Jarvis, Jarvis, Jarvis, timer" is a person
    calling an assistant that is ignoring him and then giving a one-word
    Tier-1 command that happens to be a seed term. A decoder runaway
    continues to the end of the output; a real last word ends the run,
    so the two-word rule needs the repeated word LAST."""
    assert prompt_echo("Jarvis, Jarvis, Jarvis, timer", PROMPT) == ("", 0)
    assert prompt_echo("Jarvis, Jarvis, Jarvis, calendar.", PROMPT) == ("", 0)
    assert prompt_echo("Hunter, Quennevex, Quennevex, Quennevex, Hunter",
                       PROMPT) == ("", 0)
    # ...while the pure runaway of the name alone is still an echo
    assert prompt_echo("Jarvis, Jarvis, Jarvis", PROMPT) == ("jarvis", 3)


def test_a_multi_word_unit_three_times_is_an_echo():
    assert prompt_echo("Remote Session, Remote Session, Remote Session.",
                       PROMPT) == ("remote session", 3)


def test_the_smallest_period_wins():
    assert prompt_echo("Quennevex Quennevex Quennevex Quennevex "
                       "Quennevex Quennevex", PROMPT) == ("quennevex", 6)


def test_two_repeats_are_not_an_echo():
    assert prompt_echo("Quennevex, Quennevex", PROMPT) == ("", 0)
    assert prompt_echo("Remote Session, Remote Session", PROMPT) == ("", 0)


def test_echo_min_repeats_is_three():
    assert ECHO_MIN_REPEATS == 3
    assert prompt_echo("Quennevex, " * 3, PROMPT) == ("quennevex", 3)


def test_stop_ten_times_is_insistence_not_an_echo():
    """Measured 3.06 under the ratio gate and deliberately let through;
    "stop" is not a prompt word, so this gate has no say either."""
    assert prompt_echo(", ".join(["stop"] * 10), PROMPT) == ("", 0)


def test_turn_it_up_five_times_is_not_an_echo():
    text = "Turn it up, turn it up, turn it up, turn it up, turn it up."
    assert prompt_echo(text, PROMPT) == ("", 0)


def test_yes_yes_yes_is_not_an_echo():
    assert prompt_echo("yes yes yes", PROMPT) == ("", 0)
    assert prompt_echo("No. No. No. No.", PROMPT) == ("", 0)


def test_a_sentence_that_merely_repeats_a_prompt_term_is_not_an_echo():
    assert prompt_echo("Jarvis, does my calendar say Jarvis is busy", PROMPT) \
        == ("", 0)
    assert prompt_echo("Hunter, Hunter, this is Hunter speaking", PROMPT) \
        == ("", 0)


def test_a_unit_with_a_non_prompt_word_is_not_an_echo():
    assert prompt_echo("the calendar, the calendar, the calendar", PROMPT) \
        == ("", 0)


def test_a_trailing_real_word_breaks_the_echo():
    assert prompt_echo("Quennevex, Quennevex, Quennevex, hello", PROMPT) \
        == ("", 0)


def test_no_prompt_means_no_echo():
    assert prompt_echo(FOUR, None) == ("", 0)
    assert prompt_echo(FOUR, "") == ("", 0)
    assert prompt_echo(FOUR, " , , ") == ("", 0)


def test_empty_text_is_not_an_echo():
    assert prompt_echo("", PROMPT) == ("", 0)
    assert prompt_echo(None, PROMPT) == ("", 0)
    assert prompt_echo("...", PROMPT) == ("", 0)


def test_case_and_punctuation_do_not_matter():
    assert prompt_echo("QUENNEVEX! quennevex? Quennevex...", PROMPT) \
        == ("quennevex", 3)
    assert prompt_echo(FOUR, PROMPT.upper()) == ("quennevex", 4)


# ======================================================= TranscribeResult
def _echo_result(**over) -> TranscribeResult:
    fields = dict(text=FOUR, confidence=-0.50, segments=[(FOUR, -0.50)],
                  compression_ratio=1.9, audio_seconds=3.4,
                  echo_unit="quennevex", echo_repeats=4)
    fields.update(over)
    return TranscribeResult(**fields)


def test_an_echo_is_looping_and_not_accepted_under_the_ratio_gate():
    res = _echo_result()
    assert res.compression_ratio < loop_ratio_limit(res.audio_seconds)
    assert res.looping is True
    assert res.accepted is False


def test_the_result_defaults_carry_no_echo():
    res = TranscribeResult(text="hello", confidence=-0.3,
                           segments=[("hello", -0.3)])
    assert res.echo_unit == "" and res.echo_repeats == 0
    assert res.looping is False and res.accepted is True


def test_under_the_minimum_the_echo_fields_do_not_gate():
    res = _echo_result(echo_repeats=ECHO_MIN_REPEATS - 1)
    assert res.looping is False and res.accepted is True


# ============================================================ Transcriber
def _gpu(segments, provider):
    tr = Transcriber(prompt_provider=provider)
    tr._model = FakeGpuWhisper(segments=segments)
    tr._gpu, tr._backend = True, "GPU fp16"
    return tr


def _cpu(segments, provider):
    tr = Transcriber(prompt_provider=provider)
    tr._model = FakeFasterWhisper(segments=segments)
    tr._gpu, tr._backend = False, "CPU int8"
    return tr


def _gpu_segs(text, logprob=-0.50, ratio=1.9):
    return [{"text": text, "avg_logprob": logprob, "compression_ratio": ratio}]


def _cpu_segs(text, logprob=-0.50, ratio=1.9):
    return [SimpleNamespace(text=text, avg_logprob=logprob,
                            compression_ratio=ratio)]


def test_the_gpu_path_rejects_the_live_echo(firewall, caplog):
    tr = _gpu(_gpu_segs(FOUR), lambda: PROMPT)
    with caplog.at_level("INFO"):
        res = tr.transcribe(_audio(3.4))
    assert res.echo_unit == "quennevex" and res.echo_repeats == 4
    assert res.looping is True and res.accepted is False
    assert res.compression_ratio == pytest.approx(1.9)
    assert "prompt echo" in caplog.text
    assert "repetition loop" not in caplog.text


def test_the_cpu_path_rejects_the_live_echo(firewall, caplog):
    tr = _cpu(_cpu_segs(FOUR), lambda: PROMPT)
    with caplog.at_level("INFO"):
        res = tr.transcribe(_audio(3.4))
    assert res.echo_unit == "quennevex" and res.echo_repeats == 4
    assert res.looping is True and res.accepted is False
    assert "prompt echo" in caplog.text


def test_the_rejection_log_names_the_ratio_and_the_limit_it_was_under(
        firewall, caplog):
    tr = _gpu(_gpu_segs(FOUR), lambda: PROMPT)
    with caplog.at_level("INFO"):
        tr.transcribe(_audio(3.4))
    line = [r.getMessage() for r in caplog.records
            if "prompt echo" in r.getMessage()]
    assert len(line) == 1
    assert "x 4" in line[0]
    assert "1.90 under the 3.70 loop limit for 3.4s" in line[0]


def test_the_ratio_gate_still_logs_as_a_repetition_loop(firewall, caplog):
    """When BOTH fire, the ratio line is the one already understood; the
    echo line is only for the case the ratio gate let by."""
    tr = _gpu(_gpu_segs(THIRTEEN_CUT, logprob=-0.21, ratio=7.68),
              lambda: PROMPT)
    with caplog.at_level("INFO"):
        res = tr.transcribe(_audio(5.3))
    assert res.looping is True and res.echo_repeats == 12
    assert "repetition loop" in caplog.text
    assert "prompt echo" not in caplog.text


def test_the_prompt_is_fetched_once_and_handed_to_the_model(firewall):
    calls = []

    def provider():
        calls.append(1)
        return PROMPT

    tr = _gpu(_gpu_segs(FOUR), provider)
    tr.transcribe(_audio(3.4))
    assert calls == [1]
    assert tr._model.calls[0]["initial_prompt"] == PROMPT
    tr = _cpu(_cpu_segs(FOUR), provider)
    tr.transcribe(_audio(3.4))
    assert calls == [1, 1]
    assert tr._model.calls[0]["initial_prompt"] == PROMPT


def test_real_words_with_the_same_prompt_are_not_an_echo(firewall, caplog):
    tr = _gpu(_gpu_segs("what is on my calendar today"), lambda: PROMPT)
    with caplog.at_level("INFO"):
        res = tr.transcribe(_audio(2.0))
    assert res.echo_repeats == 0 and res.echo_unit == ""
    assert res.looping is False and res.accepted is True
    assert "prompt echo" not in caplog.text


def test_an_echo_of_a_sentence_unit_is_judged_before_collapse_repeats(
        firewall, caplog):
    """collapse_repeats folds three repeated >= 3-word SENTENCES to one
    before the gate ran, so the folded phrase reached the classifier as
    a real utterance (review of 69afb9f). The gate now sees the raw text
    first; the collapsed text is what is returned and logged."""
    raw = "Remote Session Quennevex. " * 3
    tr = _gpu(_gpu_segs(raw.strip()), lambda: PROMPT)
    with caplog.at_level("INFO"):
        res = tr.transcribe(_audio(3.4))
    assert res.text == "Remote Session Quennevex."
    assert res.echo_unit == "remote session quennevex"
    assert res.echo_repeats == 3
    assert res.looping is True and res.accepted is False
    assert "prompt echo" in caplog.text


def test_without_a_provider_the_vocab_file_is_the_prompt(firewall):
    """The legacy path: no provider, the file IS the prompt, and an echo of
    a word from it is still an echo."""
    (firewall / "voice_vocab.txt").write_text("Quennevex, Librespot")
    tr = _gpu(_gpu_segs(FOUR), None)
    res = tr.transcribe(_audio(3.4))
    assert res.looping is True and res.echo_repeats == 4


def test_the_warmup_decode_is_untouched(firewall):
    tr = _gpu(_gpu_segs(FOUR), lambda: PROMPT)
    assert tr.warmup() is True
    assert tr._model.calls[0]["initial_prompt"] == PROMPT
    assert tr._decoded is True


# ---------------------------------------------------------- the preview
def test_the_live_preview_blanks_an_echo(firewall):
    """The ghost card must never show a name he never said."""
    tr = _gpu(_gpu_segs(FOUR), lambda: PROMPT)
    assert tr.partial(_audio(3.4)) == ""
    tr = _cpu(_cpu_segs(FOUR), lambda: PROMPT)
    assert tr.partial(_audio(3.4)) == ""


def test_the_live_preview_still_shows_real_words(firewall):
    tr = _gpu(_gpu_segs("what is on my calendar"), lambda: PROMPT)
    assert tr.partial(_audio(2.0)) == "what is on my calendar"
    tr = _cpu(_cpu_segs("what is on my calendar"), lambda: PROMPT)
    assert tr.partial(_audio(2.0)) == "what is on my calendar"


def test_the_preview_fetches_the_prompt_once_per_pass(firewall):
    calls = []

    def provider():
        calls.append(1)
        return PROMPT

    tr = _gpu(_gpu_segs(FOUR), provider)
    tr.partial(_audio(3.4))
    assert calls == [1]
    assert tr._model.calls[0]["initial_prompt"] == PROMPT


def test_the_preview_never_raises_on_a_broken_provider(firewall):
    def boom():
        raise RuntimeError("no")

    tr = _gpu(_gpu_segs("hello"), boom)
    assert tr.partial(_audio(1.0)) == "hello"


# ================================================== what the app does
@pytest.fixture
def app(build):          # noqa: F811
    return build()


def test_the_echo_is_never_dispatched_and_is_reported_as_looping(
        app, monkeypatch):
    seen, dispatched = _feed(app, monkeypatch, _echo_result())
    assert dispatched == []
    assert seen and seen[-1].accepted is False
    assert seen[-1].reject_reason == "looping"


def test_the_echo_is_not_salvaged_into_an_open_yes_no(app, monkeypatch):
    app.commander._pending_destructive = (
        lambda: None, "Clear all three, sir?", time.monotonic())
    seen, dispatched = _feed(app, monkeypatch, _echo_result())
    assert dispatched == []
    assert seen[-1].accepted is False and seen[-1].reject_reason == "looping"


def test_the_echo_is_not_salvaged_into_an_open_uncertain_card(
        app, monkeypatch):
    """The 20:58:45 turn: the card was open on the echo itself. Even with
    one open, the next echo must not be read as its answer."""
    app.commander._pending_destructive = None
    app._pending_uncertain = {"req-1": "turn the lights down"}
    seen, dispatched = _feed(app, monkeypatch, _echo_result())
    assert dispatched == []
    assert seen[-1].accepted is False and seen[-1].reject_reason == "looping"


def test_the_echo_is_asked_about_once_then_the_mic_stays_shut(
        app, monkeypatch):
    """The existing looping path: "Say that again, sir?" and a re-opened
    follow-up mic on the first echo, "I did not catch that" with the mic
    shut on the second -- the 20:58:40 / 20:58:45 pair, ended one turn
    earlier."""
    import numpy as np

    from jarvis.app import NOT_CAUGHT_LINE, SAY_AGAIN_LINE
    from jarvis.config import CONFIG

    spoken: list[str] = []
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    monkeypatch.setattr(app, "_decode_clip",
                        lambda audio, verify=True: (audio, {}, False,
                                                    _echo_result()))
    monkeypatch.setattr(app, "_dispatch", lambda *a, **k: None)
    monkeypatch.setattr(app, "_say", lambda text, **kw: spoken.append(text))
    app._say_again_count = 0
    app._followup_after_speech = False
    app._process_audio(np.zeros(16000, dtype=np.float32))
    assert spoken == [SAY_AGAIN_LINE]
    assert app._followup_after_speech is True
    app._followup_after_speech = False
    app._process_audio(np.zeros(16000, dtype=np.float32))
    assert spoken == [SAY_AGAIN_LINE, NOT_CAUGHT_LINE]
    assert app._followup_after_speech is False
