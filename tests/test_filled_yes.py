"""A filled pause in front of an answer must not stop it being an answer.

2026-09-05, 12:24, the live log, four consecutive lines:

    first wake of the day: offering the briefing
    speaking (breeze): Shall I run your briefing, sir?
    Transcribed: 'Uh, yeah.' (avg_logprob=-0.60)
    briefing offer: 'Uh, yeah.' is a new subject

He said yes. The offer's answer parser is end-anchored on a yes WORD after
an optional "jarvis", so the "Uh," in front of it made the whole sentence
un-answer-shaped, the offer was dropped, the briefing never ran, and the
day was already marked -- so there was no second ask.

This file pins two separate things.

1. THE HINT QUESTION, settled by measurement rather than by reading.
   CONFIG.filler_prompt_hint was turned on by hand at 11:5x the same
   morning. It appends "Um, uh, hmm, er." to the PREVIEW's initial_prompt
   so the recorder's filler hold has an "um" to hold on. The filler-hold
   lane's promise is that the FINAL transcribe() never carries it. These
   tests drive the real Transcriber and the real app._decode_clip with the
   hint ON and read back the initial_prompt the decoder was actually
   handed, on both backends and with a preview pass run first.

2. THE CLASS. A leading or trailing filled pause comes off before ANY
   answer is judged, from ONE vocabulary (jarvis.endpoint.FILLER_WORDS,
   the list the filler hold already uses). The trap in both directions is
   pinned for every parser: a filler that is the WHOLE utterance ("uh...")
   must not become a yes, and a new subject that merely STARTS with one
   ("uh, what time is it") must still be a new subject.

No audio, no model, no network: synthetic float32 arrays and fake decoders.
Invented names throughout.
"""
from __future__ import annotations

import sys
import time as _t
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import jarvis.app as app_mod
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG, PATHS
from jarvis.endpoint import FILLER_WORDS
from jarvis.transcriber import FILLER_PROMPT_HINT, Transcriber
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)
from tests.test_decode_bounds import FakeFasterWhisper, FakeGpuWhisper

SAMPLE_RATE = 16000


def _audio(seconds: float = 1.5) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


@pytest.fixture
def firewall(tmp_path, monkeypatch):
    """No vocab file, no CUDA context (as tests/test_decode_bounds)."""
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None)))
    return tmp_path


VOCAB = "Jarvis, Hunter, calendar, office hours, timer, BIOSENSORS"


def _tr(model, *, gpu: bool):
    tr = Transcriber(prompt_provider=lambda: VOCAB)
    tr._model = model
    tr._gpu = gpu
    tr._backend = "GPU fp16" if gpu else "CPU int8"
    return tr


# ===================================================================
# 1. THE HINT: can it reach the FINAL decode's prompt?
# ===================================================================
class TestTheFillerPromptHint:
    """MEASURED on the live path, not on the unit test's seam."""

    def test_the_hint_does_reach_the_previews_prompt_when_it_is_on(
            self, firewall, monkeypatch):
        """The control. Without this the three tests below prove nothing:
        a hint that reaches NOTHING would pass them all."""
        monkeypatch.setattr(CONFIG, "filler_prompt_hint", True)
        model = FakeGpuWhisper()
        _tr(model, gpu=True).partial(_audio())
        assert FILLER_PROMPT_HINT in model.calls[-1]["initial_prompt"]

    @pytest.mark.parametrize("gpu", [True, False])
    def test_the_hint_never_reaches_the_final_decodes_prompt(
            self, firewall, monkeypatch, gpu):
        monkeypatch.setattr(CONFIG, "filler_prompt_hint", True)
        model = FakeGpuWhisper() if gpu else FakeFasterWhisper()
        _tr(model, gpu=gpu).transcribe(_audio())
        prompt = model.calls[-1]["initial_prompt"]
        assert prompt == VOCAB
        for word in ("um", "uh", "hmm", "er"):
            assert word not in prompt.lower().split()

    @pytest.mark.parametrize("gpu", [True, False])
    def test_a_preview_pass_leaves_no_hint_behind_for_the_final_decode(
            self, firewall, monkeypatch, gpu):
        """The only way the promise could break without an obvious bug:
        preview state carried into the next transcribe() on the SAME
        Transcriber, which is exactly how a real turn runs."""
        monkeypatch.setattr(CONFIG, "filler_prompt_hint", True)
        model = FakeGpuWhisper() if gpu else FakeFasterWhisper()
        tr = _tr(model, gpu=gpu)
        tr.partial(_audio())
        tr.partial(_audio())
        tr.transcribe(_audio())
        assert FILLER_PROMPT_HINT in model.calls[0]["initial_prompt"]
        assert model.calls[-1]["initial_prompt"] == VOCAB

    def test_the_hint_never_reaches_the_final_decode_through_the_app(
            self, build, firewall, monkeypatch):        # noqa: F811
        """app._decode_clip -> _clip_decode -> Transcriber.transcribe, the
        path the 12:24:18 line came out of."""
        monkeypatch.setattr(CONFIG, "filler_prompt_hint", True)
        monkeypatch.setattr(CONFIG, "speaker_verify", False)
        a = build()
        model = FakeGpuWhisper()
        a.transcriber = _tr(model, gpu=True)
        monkeypatch.setattr(a, "_owner_has_phrase", lambda: False)
        a.transcriber.partial(_audio())                 # a real turn previews
        _audio_out, _stats, rejected, result = a._decode_clip(_audio())
        assert rejected is False and result is not None
        assert model.calls[-1]["initial_prompt"] == VOCAB
        assert FILLER_PROMPT_HINT not in model.calls[-1]["initial_prompt"]


# ===================================================================
# 2. THE VOCABULARY: one list, not two
# ===================================================================
class TestStripFillers:
    def test_it_is_the_filler_holds_own_list(self):
        from jarvis import endpoint
        assert endpoint.strip_fillers.__module__ == "jarvis.endpoint"
        for word in FILLER_WORDS:
            assert endpoint.strip_fillers(f"{word}, yes") == "yes", word
            assert endpoint.strip_fillers(f"yes, {word}") == "yes", word

    @pytest.mark.parametrize("said,want", [
        ("Uh, yeah.", "yeah."),
        ("um yes", "yes"),
        ("uh, um, yeah", "yeah"),
        ("yeah, uh", "yeah"),
        ("Um, yes, uh.", "yes,"),
        ("uh, what time is it", "what time is it"),
        ("yes", "yes"),
        ("", ""),
    ])
    def test_the_shapes(self, said, want):
        from jarvis.endpoint import strip_fillers
        got = strip_fillers(said)
        assert got == want or got.rstrip(",") == want.rstrip(","), (said, got)

    @pytest.mark.parametrize("said", ["uh", "uh...", "um, uh", "Hmm?", "er."])
    def test_a_pure_filler_leaves_nothing(self, said):
        from jarvis.endpoint import strip_fillers
        assert strip_fillers(said) == ""

    def test_an_interior_hyphen_is_not_a_filler(self):
        """trailing_filler already draws this line: 'uh-huh' is an answer."""
        from jarvis.endpoint import strip_fillers
        assert strip_fillers("uh-huh") == "uh-huh"
        assert strip_fillers("uh-huh, send it") == "uh-huh, send it"

    def test_it_does_not_eat_the_inside_of_a_sentence(self):
        from jarvis.endpoint import strip_fillers
        assert strip_fillers("tell him um I am late") == "tell him um I am late"

    def test_the_answer_parsers_get_the_endpoint_one_not_the_undo_lanes(self):
        """The near-miss, pinned. commander.py already had a module-level
        ``strip_fillers`` for the undo phrase, so importing endpoint's under
        the same name SHADOWED it for every caller in the file and the
        answer parsers quietly ran the undo lane's rules ("eh" is not in
        that list, so "eh, send it" was still not a yes). Two names now."""
        import jarvis.commander as cmd_mod
        assert cmd_mod.strip_fillers.__module__ == "jarvis.endpoint"
        assert cmd_mod.strip_inline_fillers.__module__ == "jarvis.commander"
        # ...and the one thing they must agree on is the vocabulary
        for word in FILLER_WORDS:
            assert cmd_mod.strip_inline_fillers(f"undo {word} that") == "undo that"


# ===================================================================
# 3. THE ROW HE HIT
# ===================================================================
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    svc = SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(),
                          memory=MagicMock(), context=MagicMock(),
                          tts=MagicMock(), briefing_offer=None)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock(), chat=MagicMock())
    return Commander(svc)


def _offer(cmdr, ok=True):
    ran = []
    cmdr.services.briefing_offer = {
        "made_at": _t.time(), "deliver": lambda: (ran.append(1), ok)[1]}
    return ran


def test_the_row_he_hit_uh_yeah_runs_the_briefing(cmdr):
    """12:24:18.286, verbatim: briefing offer: 'Uh, yeah.' is a new subject."""
    ran = _offer(cmdr)
    res = cmdr.handle("Uh, yeah.", "voice")
    assert ran == [1], "he said yes and the briefing must run"
    assert res is not None and res.handled
    assert res.status.startswith("Briefing")
    assert cmdr.services.briefing_offer is None


# The six rows the brief asks of EVERY parser: a filler before a yes,
# before a no, at both ends, doubled, the whole utterance, and a new
# subject that merely starts with one.
FILLED_YES = ["uh, yes", "Uh, yeah.", "um yes", "er, yes", "hmm, yes",
              "uh, um, yeah", "yes, uh", "uh, yes, um"]
FILLED_NO = ["uh, no", "Um, no thanks.", "er, no", "uh, um, no", "no, uh"]
PURE_FILLER = ["uh", "uh...", "um", "Hmm?", "uh, um", "er."]


class TestTheBriefingOffer:
    @pytest.mark.parametrize("said", FILLED_YES)
    def test_a_filled_yes_runs_it(self, cmdr, said):
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert ran == [1], said
        assert res is not None and res.status.startswith("Briefing")

    @pytest.mark.parametrize("said", FILLED_NO)
    def test_a_filled_no_declines_it(self, cmdr, said):
        from jarvis.commander import BRIEFING_DECLINED_LINE
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert ran == [], said
        assert res is not None and res.reply == BRIEFING_DECLINED_LINE, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_filler_on_its_own_is_never_a_yes(self, cmdr, said):
        ran = _offer(cmdr)
        cmdr.handle(said, "voice")
        assert ran == [], said

    @pytest.mark.parametrize("said", [
        "uh, what time is it",
        "um, turn the lights off",
        "uh, set a timer for ten minutes",
        "er, what's the weather",
        "uh, skip this song",
    ])
    def test_a_new_subject_that_starts_with_a_filler_is_still_a_new_subject(
            self, cmdr, said):
        """THE TRAP. Get this wrong and a question becomes a yes."""
        from jarvis.commander import BRIEFING_DECLINED_LINE
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert ran == [], said
        assert res is None or not str(res.status or "").startswith("Briefing"), said
        assert res is None or res.reply != BRIEFING_DECLINED_LINE, said

    def test_the_answer_is_one_function_the_rung_calls(self):
        from jarvis.commander import briefing_answer
        assert briefing_answer("Uh, yeah.") is True
        assert briefing_answer("uh, no thanks") is False
        assert briefing_answer("uh...") is None
        assert briefing_answer("uh, what time is it") is None


# ===================================================================
# 4. parse_yes_no -- the shared word bag behind eleven rungs
# ===================================================================
class TestParseYesNo:
    @pytest.mark.parametrize("said", FILLED_YES)
    def test_a_filled_yes(self, said):
        from jarvis.commander import parse_yes_no
        assert parse_yes_no(said) is True, said

    @pytest.mark.parametrize("said", FILLED_NO)
    def test_a_filled_no(self, said):
        from jarvis.commander import parse_yes_no
        assert parse_yes_no(said) is False, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_is_neither(self, said):
        from jarvis.commander import parse_yes_no
        assert parse_yes_no(said) is None, said

    @pytest.mark.parametrize("said", ["uh, what time is it",
                                      "um, turn the lights off",
                                      "er, how long until dinner"])
    def test_a_new_subject_that_starts_with_a_filler_is_neither(self, said):
        from jarvis.commander import parse_yes_no
        assert parse_yes_no(said) is None, said

    def test_the_six_word_guard_counts_words_he_said_not_fillers(self):
        """The guard exists to refuse overheard speech. A filled pause is
        not a word of the answer, and padding the count with one used to
        turn a six-word yes into no answer at all."""
        from jarvis.commander import parse_yes_no
        assert parse_yes_no("yes go ahead and do that") is True
        assert parse_yes_no("uh, yes go ahead and do that") is True


# ===================================================================
# 5. the approval / terminal-offer grammar (_YES_RX / _NO_RX)
# ===================================================================
class TestTheApprovalGrammar:
    @pytest.mark.parametrize("said", FILLED_YES)
    def test_a_filled_yes_allows(self, cmdr, said):
        from jarvis.commander import ALLOWED_LINE
        answers = []
        cmdr.services.approvals = SimpleNamespace(
            pending=lambda: True,
            answer=lambda ok, source="": answers.append(ok))
        res = cmdr.handle(said, "voice")
        assert answers == [True], said
        assert res is not None and res.reply == ALLOWED_LINE

    @pytest.mark.parametrize("said", FILLED_NO)
    def test_a_filled_no_declines(self, cmdr, said):
        from jarvis.commander import DECLINED_LINE
        answers = []
        cmdr.services.approvals = SimpleNamespace(
            pending=lambda: True,
            answer=lambda ok, source="": answers.append(ok))
        res = cmdr.handle(said, "voice")
        assert answers == [False], said
        assert res is not None and res.reply == DECLINED_LINE

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_answers_nothing(self, cmdr, said):
        answers = []
        cmdr.services.approvals = SimpleNamespace(
            pending=lambda: True,
            answer=lambda ok, source="": answers.append(ok))
        cmdr.handle(said, "voice")
        assert answers == [], said

    def test_a_filled_yes_opens_the_offered_terminal(self, cmdr):
        opened = []
        cmdr.services.claude = SimpleNamespace(
            open_terminal=lambda slug: (opened.append(slug), True)[1])
        cmdr._pending_terminal_slug = "vss"
        cmdr._pending_terminal_made = _t.monotonic()
        res = cmdr._try_terminal_offer("Uh, yes.")
        assert opened == ["vss"] and res is not None and res.handled

    def test_a_new_subject_that_starts_with_a_filler_does_not_approve(self, cmdr):
        answers = []
        cmdr.services.approvals = SimpleNamespace(
            pending=lambda: True,
            answer=lambda ok, source="": answers.append(ok))
        cmdr.handle("uh, what time is it", "voice")
        assert answers == []


# ===================================================================
# 6. the send read-back (parse_send_answer / _send_clean)
# ===================================================================
class TestTheSendReadBack:
    @pytest.mark.parametrize("said", [
        "uh, yes", "um, send it", "er, yes send it to her",
        "ah, yes", "eh, send it", "yes, uh", "uh, um, yes"])
    def test_a_filled_yes_sends(self, said):
        from jarvis.commander import parse_send_answer
        assert parse_send_answer(said) is True, said

    @pytest.mark.parametrize("said", [
        "uh, no", "ah, no don't send it", "um, no thanks", "no, uh"])
    def test_a_filled_no_does_not(self, said):
        from jarvis.commander import parse_send_answer
        assert parse_send_answer(said) is False, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_is_not_consent(self, said):
        from jarvis.commander import parse_send_answer
        assert parse_send_answer(said) is not True, said

    @pytest.mark.parametrize("said", ["uh, what time is it",
                                      "um, turn the lights off"])
    def test_a_new_subject_is_not_consent(self, said):
        from jarvis.commander import parse_send_answer
        assert parse_send_answer(said) is not True, said

    def test_the_backchannel_yes_still_works(self):
        """"uh huh" is a yes on this lane and must not be shaved down to
        "huh" by a strip that runs too early."""
        from jarvis.commander import parse_send_answer
        assert parse_send_answer("uh huh, send it") is True
        assert parse_send_answer("uh-huh, send it") is True


# ===================================================================
# 7. the router's "shall I hand this to Claude?" answer
# ===================================================================
class TestTheRouterAnswer:
    @pytest.mark.parametrize("said", ["uh, yes", "um, yeah", "er, go ahead",
                                      "yes, uh", "uh, um, yes"])
    def test_a_filled_yes_hands_it_over(self, said):
        from jarvis.router import answer_kind
        assert answer_kind(said) == "claude", said

    @pytest.mark.parametrize("said", ["uh, no", "um, nope", "no, uh"])
    def test_a_filled_no_keeps_it_local(self, said):
        from jarvis.router import answer_kind
        assert answer_kind(said) == "local", said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_answers_nothing(self, said):
        from jarvis.router import answer_kind
        assert answer_kind(said) is None, said

    def test_a_new_subject_that_starts_with_a_filler_is_not_an_answer(self):
        from jarvis.router import answer_kind
        assert answer_kind("uh, what time is it") is None


# ===================================================================
# 8. the week-plan walk (jarvis/dialogue.py)
# ===================================================================
class TestThePlanWalk:
    def _walk(self):
        from datetime import date

        from jarvis.dialogue import PlanItem, WeekPlanner
        walk = WeekPlanner(items=[PlanItem(title="finish the lab report")],
                           today=date(2026, 9, 7))
        walk.ask()
        return walk

    @pytest.mark.parametrize("said", ["uh, yes", "um, sure", "yes, uh"])
    def test_a_filled_yes_takes_the_slot(self, said):
        walk = self._walk()
        assert walk.settle(said) is not None, said
        assert walk.slots, said

    @pytest.mark.parametrize("said", ["uh, no", "um, skip it", "no, uh"])
    def test_a_filled_no_skips_it(self, said):
        from jarvis.dialogue import SKIP_LINE
        walk = self._walk()
        assert walk.settle(said) == SKIP_LINE, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_settles_nothing(self, said):
        walk = self._walk()
        assert walk.settle(said) is None, said
        assert not walk.slots, said

    def test_a_new_subject_that_starts_with_a_filler_is_routed(self):
        walk = self._walk()
        assert walk.settle("uh, what time is it") is None
        assert not walk.slots


# ===================================================================
# 9. the Discord / approval word bag (app.yes_no)
# ===================================================================
class TestTheAppYesNo:
    @pytest.mark.parametrize("said", ["uh, yes", "um, allow it", "yes, uh",
                                      "uh, um, yes"])
    def test_a_filled_yes(self, said):
        assert app_mod.yes_no(said) is True, said

    @pytest.mark.parametrize("said", ["uh, no", "um, deny", "no, uh"])
    def test_a_filled_no(self, said):
        assert app_mod.yes_no(said) is False, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_is_neither(self, said):
        assert app_mod.yes_no(said) is None, said

    def test_the_four_word_cap_counts_words_he_said(self):
        assert app_mod.yes_no("yes go ahead please") is True
        assert app_mod.yes_no("uh, yes go ahead please") is True


# ===================================================================
# 10. the uncertain-intent card ("Was that for me?")
# ===================================================================
def test_a_filled_yes_answers_the_uncertain_card(build, monkeypatch):  # noqa: F811
    """The card rides parse_yes_no through app._ask_uncertain. The seam is
    the transcriber: no mic is ever opened."""
    from jarvis.transcriber import TranscribeResult
    a = build()
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    monkeypatch.setattr(app_mod.MACHINE, "has_mic", True)
    a.recorder = SimpleNamespace(recording=False,
                                 record_fixed=lambda s: _audio(1.0))
    a.transcriber = SimpleNamespace(transcribe=lambda audio: TranscribeResult(
        text="Uh, yeah.", confidence=-0.6, compression_ratio=1.2,
        segments=[("Uh, yeah.", -0.6)]))
    answered = []
    monkeypatch.setattr(a, "uncertain_answer",
                        lambda rid, yes, source="ui": answered.append((rid, yes)))
    with a._uncertain_lock:
        a._pending_uncertain["req-1"] = "turn the lights down"
    a._ask_uncertain("req-1")
    assert answered == [("req-1", True)]
