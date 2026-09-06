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
from jarvis.commander import Commander, CommandResult, IntentClassifier
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
        ("Um, yes, uh.", "yes."),           # the tail's "." is carried (round 3)
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
def make_cmdr(tmp_path, monkeypatch) -> Commander:
    """The slim Commander every filled-pause test drives: a factory, so a
    sibling module (test_hesitation_teeth) can build the same one without
    importing this fixture by name."""
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


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    return make_cmdr(tmp_path, monkeypatch)


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


# ===================================================================
# 11. THE COMPLETION (2026-09-05, second pass)
# ===================================================================
# The first pass stopped at eleven rungs and the verdict measured six more
# that still throw his answer away. Same class, same one-line move, same
# vocabulary -- and each one is pinned BARE vs FILLED below, because the
# only thing that must change is what a filled pause does. The bare row is
# the control: it passed before this pass and it has to keep passing.
#
#   1. pick_from_answer's ORDINAL leg   ("uh, the second one")
#   2. _PICK_CANCEL_RX                  ("uh, neither")
#   3. _LEAVE_DECLINE_RX                ("uh, no idea")
#   4. correction_kind, BOTH shapes     ("uh, I said Lisbon")
#   5. feedback_kind                    ("uh, that was for you")
#   6. _QUIZ_SKIP_RX                    ("uh, skip it")  <- marks a card WRONG
FILLED_PREFIX = ["uh, ", "Um, ", "er, ", "uh, um, ", "Hmm, "]


class TestThePickAnswer:
    """"Which one, sir?" -- the ambiguous-file question (rung 1)."""

    CANDS = ("/tmp/example/alpha.txt", "/tmp/example/bravo.txt")

    def test_bare_the_control(self):
        from jarvis.commander import pick_from_answer
        choice, near = pick_from_answer("the second one", self.CANDS)
        assert choice is not None and choice.name == "bravo.txt"
        assert near is False

    @pytest.mark.parametrize("said", [
        "uh, the second one", "Um, the second one.", "er, the second one",
        "the second one, uh", "uh, second", "uh, um, the second one",
    ])
    def test_a_filled_ordinal_picks_the_same_file(self, said):
        from jarvis.commander import pick_from_answer
        choice, near = pick_from_answer(said, self.CANDS)
        assert choice is not None and choice.name == "bravo.txt", said
        assert near is False, said

    def test_a_filled_name_picks_the_same_file(self):
        from jarvis.commander import pick_from_answer
        bare = pick_from_answer("bravo", self.CANDS)
        filled = pick_from_answer("uh, bravo", self.CANDS)
        assert bare[0] is not None and bare[0].name == "bravo.txt"
        assert filled[0] is not None and filled[0].name == bare[0].name

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_picks_nothing(self, said):
        from jarvis.commander import pick_from_answer
        assert pick_from_answer(said, self.CANDS) == (None, False), said

    @pytest.mark.parametrize("said", ["uh, what time is it",
                                      "um, turn the lights off"])
    def test_a_new_subject_that_starts_with_a_filler_picks_nothing(self, said):
        from jarvis.commander import pick_from_answer
        assert pick_from_answer(said, self.CANDS)[0] is None, said


class TestThePickCancel:
    """"uh, neither" must drop the question, not fall through (rung 2)."""

    def _arm(self, cmdr):
        import time as t
        cmdr._pending_filepick = (
            ["/tmp/example/alpha.txt", "/tmp/example/bravo.txt"],
            lambda choice: None, "voice", t.monotonic(), False)

    def test_bare_the_control(self, cmdr):
        self._arm(cmdr)
        res = cmdr._try_filepick_answer("neither", "voice")
        assert res is not None and res.status == "Dropped"
        assert cmdr._pending_filepick is None

    @pytest.mark.parametrize("said", ["uh, neither", "Um, neither.",
                                      "er, never mind", "uh, forget it",
                                      "neither, uh", "uh, um, cancel"])
    def test_a_filled_cancel_drops_the_question(self, cmdr, said):
        self._arm(cmdr)
        res = cmdr._try_filepick_answer(said, "voice")
        assert res is not None and res.status == "Dropped", said
        assert cmdr._pending_filepick is None, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_does_not_cancel(self, cmdr, said):
        self._arm(cmdr)
        res = cmdr._try_filepick_answer(said, "voice")
        assert res is None or res.status != "Dropped", said


class TestTheLeaveDecline:
    """"How long do you need to get to X, sir?" -- the decline (rung 3)."""

    @pytest.fixture
    def leave_cmdr(self, cmdr):
        import time as t
        cmdr.services.leavetime = SimpleNamespace(
            learn=lambda key, minutes: int(minutes))
        cmdr._pending_leave = ("Example Hall", "Example Hall", t.monotonic())
        return cmdr

    def test_bare_the_control(self, leave_cmdr):
        from jarvis.commander import LEAVE_DROPPED_LINE
        res = leave_cmdr._try_leave_answer("no idea")
        assert res is not None and res.reply == LEAVE_DROPPED_LINE
        assert leave_cmdr._pending_leave is None

    @pytest.mark.parametrize("said", ["uh, no idea", "Um, no idea.",
                                      "er, not sure", "uh, never mind",
                                      "no idea, uh", "uh, um, dunno"])
    def test_a_filled_decline_closes_the_question(self, leave_cmdr, said):
        from jarvis.commander import LEAVE_DROPPED_LINE
        res = leave_cmdr._try_leave_answer(said)
        assert res is not None and res.reply == LEAVE_DROPPED_LINE, said
        assert leave_cmdr._pending_leave is None, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_leaves_the_question_standing(self, leave_cmdr, said):
        assert leave_cmdr._try_leave_answer(said) is None, said
        assert leave_cmdr._pending_leave is not None, said

    def test_a_command_inside_the_window_is_still_that_command(self, leave_cmdr):
        assert leave_cmdr._try_leave_answer("uh, set a timer for five minutes") is None
        assert leave_cmdr._pending_leave is not None


class TestTheCorrection:
    """"I said Lisbon" / "not the terminal, the calendar" (rung 4).

    A man correcting a mishearing is PRECISELY when he hesitates, so this
    is the rung where the bug should have been expected first."""

    def test_bare_the_control(self):
        from jarvis.commander import correction_kind
        assert correction_kind("I said Lisbon") == "Lisbon"
        assert correction_kind("not the terminal, the calendar") == "the calendar"

    @pytest.mark.parametrize("prefix", FILLED_PREFIX)
    def test_a_filled_i_said_is_still_a_correction(self, prefix):
        from jarvis.commander import correction_kind
        assert correction_kind(prefix + "I said Lisbon") == "Lisbon", prefix

    @pytest.mark.parametrize("prefix", FILLED_PREFIX)
    def test_a_filled_not_x_y_is_still_a_correction(self, prefix):
        from jarvis.commander import correction_kind
        said = prefix + "not the terminal, the calendar"
        assert correction_kind(said) == "the calendar", prefix

    def test_a_trailing_filler_does_not_end_up_in_the_meant_text(self):
        from jarvis.commander import correction_kind
        assert correction_kind("I said Lisbon, uh") == "Lisbon"

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_is_not_a_correction(self, said):
        from jarvis.commander import correction_kind
        assert correction_kind(said) is None, said

    @pytest.mark.parametrize("said", ["uh, what time is it",
                                      "um, not now", "er, not really"])
    def test_a_new_subject_that_starts_with_a_filler_is_not_a_correction(self, said):
        from jarvis.commander import correction_kind
        assert correction_kind(said) is None, said

    def test_the_question_mark_still_survives(self):
        """09-04's sixth pass: a "?" on the meant text is the send bar."""
        from jarvis.commander import correction_kind
        assert correction_kind("uh, I said yes?") == "yes?"


class TestTheFeedback:
    """"that was for you" -- the rung that RE-RUNS a dropped command (5)."""

    def test_bare_the_control(self):
        from jarvis.commander import feedback_kind
        assert feedback_kind("that was for you") is True
        assert feedback_kind("that wasn't for you") is False

    @pytest.mark.parametrize("prefix", FILLED_PREFIX)
    def test_a_filled_yes_is_still_feedback(self, prefix):
        from jarvis.commander import feedback_kind
        assert feedback_kind(prefix + "that was for you") is True, prefix

    @pytest.mark.parametrize("prefix", FILLED_PREFIX)
    def test_a_filled_no_is_still_feedback(self, prefix):
        from jarvis.commander import feedback_kind
        assert feedback_kind(prefix + "that wasn't for you") is False, prefix

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_is_not_feedback(self, said):
        from jarvis.commander import feedback_kind
        assert feedback_kind(said) is None, said

    @pytest.mark.parametrize("said", ["uh, what time is it",
                                      "um, that was for her"])
    def test_a_new_subject_that_starts_with_a_filler_is_not_feedback(self, said):
        from jarvis.commander import feedback_kind
        assert feedback_kind(said) is None, said

    def test_the_dropped_command_is_re_run_through_the_real_rung(self, cmdr):
        """The teeth: a command the classifier dropped is run again."""
        import time as t
        from jarvis.commander import LastTurn
        cmdr._last_turn = LastTurn("what time is it", "Ignored (background)",
                                   t.monotonic())
        res = cmdr._try_feedback("uh, that was for you", None, "voice")
        assert res is not None and res.handled
        assert res.status == "Clock", res.status
        assert res.corrected == "what time is it"


class TestTheQuizSkip:
    """THE ONE WITH TEETH, driven through the real handler (rung 6).

    Measured before the fix, same three words, same open card:

        "skip it"      -> session.results == [(1, None)]   ungraded
        "uh, skip it"  -> session.results == [(1, False)]  MARKED WRONG

    A card marked wrong is not a lost turn he can repeat: it is a Leitner
    demotion written to his deck, and it will come back at him for weeks.
    And on the briefing rung -- which the first pass DID fix -- those same
    three words are correctly a decline. Identical words, opposite
    treatment, one of them silently editing his study schedule."""

    @pytest.fixture
    def quiz_cmdr(self, cmdr, tmp_path):
        from jarvis.tools.quiz import FlashcardStore, QuizSession
        store = FlashcardStore(tmp_path / "cards.db")
        cards = store.add_cards(
            [{"question": "What is impedance?", "answer": "V over I"},
             {"question": "Name a transducer", "answer": "thermistor"}],
            source="example.md", topic="example")
        cmdr.services.flashcards = store
        cmdr._pending_quiz = QuizSession(cards, topic="example")
        cmdr._pending_quiz.ask()
        yield cmdr, store, cards
        store.close()

    def test_bare_the_control(self, quiz_cmdr):
        from jarvis.tools.quiz import SKIP_LINE
        cmdr, store, cards = quiz_cmdr
        cid = cards[0]["id"]
        res = cmdr.handle("skip it", "voice")
        assert res is not None and res.reply.startswith(
            SKIP_LINE.format(answer="V over I"))
        assert cmdr._pending_quiz.results == [(cid, None)]
        assert store.get(cid)["seen"] == 0 and store.get(cid)["box"] == 1

    @pytest.mark.parametrize("said", ["uh, skip it", "Um, skip it.",
                                      "er, skip it", "uh, pass",
                                      "uh, no idea", "skip it, uh",
                                      "uh, um, I don't know"])
    def test_a_filled_skip_settles_the_card_ungraded(self, quiz_cmdr, said):
        cmdr, store, cards = quiz_cmdr
        cid = cards[0]["id"]
        cmdr.handle(said, "voice")
        assert cmdr._pending_quiz.results == [(cid, None)], said
        assert store.get(cid)["seen"] == 0, said
        assert store.get(cid)["box"] == 1, said

    def test_a_filled_skip_names_the_answer_like_a_bare_one(self, quiz_cmdr):
        from jarvis.tools.quiz import SKIP_LINE
        cmdr, store, cards = quiz_cmdr
        res = cmdr.handle("uh, skip it", "voice")
        assert res is not None
        assert res.reply.startswith(SKIP_LINE.format(answer="V over I"))

    def test_a_real_filled_answer_is_still_graded(self, quiz_cmdr):
        """The trap in the other direction: stripping the filler must not
        turn a genuine answer into a skip."""
        cmdr, store, cards = quiz_cmdr
        cid = cards[0]["id"]
        cmdr.handle("uh, v over i", "voice")
        assert cmdr._pending_quiz.results == [(cid, True)]


# ===================================================================
# 12. THE SEVENTH, and what the AST sweep found with it
# ===================================================================
# The six above were the verdict's list. An AST sweep of jarvis/ -- every
# re.compile whose pattern is start-anchored, reassembled from the tree
# rather than grepped, and every .match() site traced back to the
# assignment that feeds it -- turned up 28 spoken-answer grammars across
# 33 match sites, and FIVE more of them were still losing his answer.
# Measured bare vs filled before the fix:
#
#   _person_from_answer ordinal  "the second one" -> Bea Example
#                                "uh, the second one" -> None
#   _QUIZ_STOP_RX                True  -> False   (and the card is GRADED)
#   _TAKE_QUIZ_RX                True  -> False
#   _ENROL_CONFIRM_RX            "enrol" -> "uh, enrol" is not the word
#   _SEND_MAYBE_RX (destructive) True  -> False   (the re-ask is lost)
#
# quiz_kind / review_kind: COMMAND grammars the registry shares. The
# registry's copy stays unstripped (the 5caf86c ruling), but the quiz
# rung's ESCAPE now runs on the stripped words (round 3, 09-06): a filled
# "uh, quiz me on the recipe" over an open card drops the card ungraded
# and routes on -- it used to reach the GRADER and mark the card wrong.
# The registry may then not start the new quiz; a repeat, never a card.
# Measured in test_hesitation_teeth.TestTheCardStandsOnAHesitation.
class TestThePersonPick:
    """"Which Heather, sir?" -- the same ordinal leg as the file pick,
    which is why it was missed: two functions, one grammar."""

    NAMES = ["Ada Example", "Bea Example"]

    def test_bare_the_control(self):
        from jarvis.commander import _person_from_answer
        assert _person_from_answer("the second one", self.NAMES) == "Bea Example"

    @pytest.mark.parametrize("said", ["uh, the second one", "Um, the second one.",
                                      "er, second", "the second one, uh"])
    def test_a_filled_ordinal_names_the_same_person(self, said):
        from jarvis.commander import _person_from_answer
        assert _person_from_answer(said, self.NAMES) == "Bea Example", said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_names_nobody(self, said):
        from jarvis.commander import _person_from_answer
        assert _person_from_answer(said, self.NAMES) is None, said


class TestTheQuizStopWords:
    """"uh, stop the quiz" must stop the quiz, not be marked wrong."""

    @pytest.fixture
    def quiz_cmdr(self, cmdr, tmp_path):
        from jarvis.tools.quiz import FlashcardStore, QuizSession
        store = FlashcardStore(tmp_path / "cards.db")
        cards = store.add_cards(
            [{"question": "What is impedance?", "answer": "V over I"},
             {"question": "Name a transducer", "answer": "thermistor"}],
            source="example.md", topic="example")
        cmdr.services.flashcards = store
        cmdr._pending_quiz = QuizSession(cards, topic="example")
        cmdr._pending_quiz.ask()
        yield cmdr, store, cards
        store.close()

    def test_bare_the_control(self, quiz_cmdr):
        cmdr, store, cards = quiz_cmdr
        res = cmdr.handle("stop the quiz", "voice")
        assert res is not None and res.status == "Quiz stopped"
        assert cmdr._pending_quiz is None
        assert store.get(cards[0]["id"])["seen"] == 0

    @pytest.mark.parametrize("said", ["uh, stop the quiz", "Um, stop the quiz.",
                                      "er, no more questions"])
    def test_a_filled_stop_stops_it_without_grading_the_card(self, quiz_cmdr, said):
        cmdr, store, cards = quiz_cmdr
        res = cmdr.handle(said, "voice")
        assert res is not None and res.status == "Quiz stopped", said
        assert cmdr._pending_quiz is None, said
        assert store.get(cards[0]["id"])["seen"] == 0, said


class TestTheTeachOffer:
    """"Shall I quiz you on it, sir?" -- "uh, go on" takes the offer."""

    def _arm(self, cmdr):
        import time as t
        cmdr._pending_teach = ("example", "some body text", t.monotonic())

    def test_bare_the_control(self, cmdr):
        from jarvis.commander import _TAKE_QUIZ_RX
        assert _TAKE_QUIZ_RX.match("go on")
        self._arm(cmdr)
        res = cmdr._try_teach_offer("go on")
        assert res is not None and res.status != "No quiz"

    @pytest.mark.parametrize("said", ["uh, go on", "Um, go ahead.",
                                      "er, quiz me", "go on, uh"])
    def test_a_filled_yes_takes_the_offer(self, cmdr, said):
        self._arm(cmdr)
        res = cmdr._try_teach_offer(said)
        assert res is not None, said
        assert res.status != "No quiz", said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_takes_nothing(self, cmdr, said):
        self._arm(cmdr)
        assert cmdr._try_teach_offer(said) is None, said


class TestTheEnrolConfirm:
    """The enrolment offer is taken by ONE word ("enrol") on purpose --
    it starts the camera. A filled pause must not un-say it."""

    def _arm(self, cmdr):
        cmdr.services.enrol_offer = {"made_at": _t.time(), "name": "Ada Example"}
        cmdr.services.enrol_run = None

    def test_bare_the_control(self):
        from jarvis.commander import _ENROL_CONFIRM_RX
        assert _ENROL_CONFIRM_RX.match("enrol")

    @pytest.mark.parametrize("said", ["uh, enrol", "Um, enrol.", "enrol, uh"])
    def test_a_filled_confirm_is_still_the_word(self, cmdr, said):
        self._arm(cmdr)
        res = cmdr._try_enrol(said)
        assert res is not None, said

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_starts_no_camera(self, cmdr, said):
        self._arm(cmdr)
        assert cmdr._try_enrol(said) is None, said

    def test_a_new_subject_that_starts_with_a_filler_starts_no_camera(self, cmdr):
        self._arm(cmdr)
        assert cmdr._try_enrol("uh, what time is it") is None


class TestTheDestructiveVagueAnswer:
    """A vague answer to a destructive read-back earns ONE re-ask. Filled,
    it earned nothing: the read-back was dropped in silence."""

    def _arm(self, cmdr):
        import time as t
        pend = (lambda: CommandResult(handled=True, reply="done", status="Done"),
                "delete the example list", t.monotonic())
        cmdr._pending_destructive = pend
        cmdr._pending_destructive_meta = ("voice", True, pend)
        cmdr._strict_reasked = False
        cmdr.services.notes = None

    def test_bare_the_control(self, cmdr):
        self._arm(cmdr)
        res = cmdr._try_destructive_confirm("okay", "voice")
        assert res is not None and res.status == "Confirm?"

    @pytest.mark.parametrize("said", ["uh, okay", "Um, sure.", "er, alright",
                                      "okay, uh"])
    def test_a_filled_vague_answer_still_earns_the_re_ask(self, cmdr, said):
        self._arm(cmdr)
        res = cmdr._try_destructive_confirm(said, "voice")
        assert res is not None and res.status == "Confirm?", said

    def test_a_filled_yes_is_still_a_yes(self, cmdr):
        self._arm(cmdr)
        res = cmdr._try_destructive_confirm("uh, yes do it", "voice")
        assert res is not None and res.status == "Done"


# The AST sweep that used to sit here fed a HAND-WRITTEN set of rung names
# to a walker -- the same mistake as the two inventories before it, one
# level down. It is superseded by tests/test_answer_census.py, which
# derives the rungs from the code (any function that reads a parked
# question) and follows his words from each one into every parser.


# ===================================================================
# 13. ONE pin on CONFIG, not two (the integration collision)
# ===================================================================
def test_exactly_one_autouse_fixture_pins_the_live_config():
    """The merge brought TWO autouse fixtures pinning the same field.

    Both were written the same day for the same incident -- he turned
    CONFIG.filler_prompt_hint on by hand at 11:5x on 2026-09-05 (a correct
    change to his own box: the probe had measured that the filler hold
    does nothing without it) and four tests in test_transcriber_prompt.py
    and test_prompt_echo.py went red on a tree where nothing had been
    committed. Two autouse fixtures setting one field to one value are
    harmless and redundant, and redundant suite-wide autouse state is how
    the NEXT one gets added without anybody noticing the first.

    ``_pin_live_tuning_settings`` is the one kept, on two measurable
    counts: it restores inside try/finally, so a throw into the fixture at
    the yield cannot leave his setting stamped on CONFIG for the rest of
    the session; and the field list is a NAMED module constant with the
    instruction to extend it, rather than a tuple buried in the body.
    """
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(__file__).parent / "conftest.py").read_text())
    pinners = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        autouse = any("autouse" in ast.unparse(d) for d in fn.decorator_list)
        if autouse and "filler_prompt_hint" in ast.unparse(fn):
            pinners.append(fn.name)
    # the constant lives outside the function, so count that spelling too
    src = ast.unparse(tree)
    if "_PINNED_CONFIG_FIELDS" in src:
        for fn in ast.walk(tree):
            if (isinstance(fn, ast.FunctionDef)
                    and any("autouse" in ast.unparse(d) for d in fn.decorator_list)
                    and "_PINNED_CONFIG_FIELDS" in ast.unparse(fn)
                    and fn.name not in pinners):
                pinners.append(fn.name)
    assert pinners == ["_pin_live_tuning_settings"], pinners


def test_the_pin_is_the_shipped_default_while_a_test_runs():
    """And it pins to the DATACLASS default, so it tracks the code rather
    than freezing today's value."""
    from dataclasses import fields
    from jarvis.config import Config
    default = {f.name: f.default for f in fields(Config)}["filler_prompt_hint"]
    assert CONFIG.filler_prompt_hint == default


# ===================================================================
# 14. What the DERIVED census found (tests/answercensus.py)
# ===================================================================
# The eight-item inventory was a list, the adversary's six were a list,
# and the AST sweep in section 12 was a list too (a hand-written set of
# rung names, fed to a walker). tests/test_answer_census.py derives the
# rungs from the code instead -- a rung is any function that reads a
# parked question -- and follows his words from each one into every
# parser. These are the sites it found that the lists had not, each
# measured bare vs filled BEFORE the fix, each the same one-line move.
class TestTheSendAskAnswers:
    """"Which file, sir?" / "Which account?" / "To whom?" -- the three
    answer parsers of the send-ask lane. Each strips a LEAD ("it's the",
    "from my", "send it to") anchored at ^, so a filler in front of the
    lead left the whole lead on:

        _file_answer("it's the lab report")      -> "the lab report"
        _file_answer("uh, it's the lab report")  -> "uh, it's the lab report"
        _account_answer("the work one")          -> "work"
        _account_answer("um, the work one")      -> "um, the work"
        _recipient_answer("to dana")             -> "dana"
        _recipient_answer("um, to dana")         -> "um, to dana"
    """

    def test_bare_the_control(self):
        from jarvis.commander import _account_answer, _file_answer, _recipient_answer
        assert _file_answer("it's the lab report") == "the lab report"
        assert _account_answer("the work one") == "work"
        assert _recipient_answer("to dana") == "dana"

    @pytest.mark.parametrize("prefix", ["uh, ", "Um, ", "er ", "hmm, uh, "])
    def test_a_filled_file_answer_names_the_same_file(self, prefix):
        from jarvis.commander import _file_answer
        assert _file_answer(prefix + "it's the lab report") == "the lab report", prefix
        assert _file_answer(prefix + "the lab report") == "the lab report", prefix

    @pytest.mark.parametrize("prefix", ["uh, ", "Um, ", "er ", "hmm, uh, "])
    def test_a_filled_account_answer_names_the_same_account(self, prefix):
        from jarvis.commander import _account_answer
        assert _account_answer(prefix + "the work one") == "work", prefix
        assert _account_answer(prefix + "school") == "school", prefix

    @pytest.mark.parametrize("prefix", ["uh, ", "Um, ", "er ", "hmm, uh, "])
    def test_a_filled_recipient_answer_names_the_same_person(self, prefix):
        from jarvis.commander import _recipient_answer
        assert _recipient_answer(prefix + "to dana") == "dana", prefix
        assert _recipient_answer(prefix + "it's dana at example dot com") == \
            "dana at example dot com", prefix

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_names_nothing(self, said):
        from jarvis.commander import _account_answer, _file_answer, _recipient_answer
        assert _file_answer(said) == ""
        assert _account_answer(said) == ""
        assert _recipient_answer(said) == ""

    def test_the_trailing_filler_comes_off_too(self):
        from jarvis.commander import _file_answer
        assert _file_answer("the lab report, uh") == "the lab report"

    @pytest.mark.parametrize("said", ["it's the lab report", "uh, it's the lab report",
                                      "Um, the lab report."])
    def test_the_file_question_hands_prepare_the_same_phrase(
            self, cmdr, monkeypatch, said):
        """Through the rung: the phrase that reaches outbox.prepare is the
        file phrase, bare or filled -- never "uh, it's the lab report"."""
        import jarvis.commander as cm
        from jarvis.commander import SendAsk
        handed = []

        def fake_prepare(cfg, memory, phrase, who, **kw):
            handed.append(phrase)
            return object()

        monkeypatch.setattr(cm.outbox, "prepare", fake_prepare)
        monkeypatch.setattr(cm, "_send_file_finish", lambda *a, **k: CommandResult(
            handled=True, status="Prepared"))
        cmdr._pending_sendask = SendAsk(kind="file", said_file="", who="dana@example.com",
                                        hint="", source="voice", made_at=_t.monotonic())
        res = cmdr._try_sendask_answer(said, "voice")
        assert res is not None and res.status == "Prepared", said
        assert handed == ["the lab report"], said


class TestTheSessionEnough:
    """"That's enough" inside a working session ends it WITH its read-back
    (dialogue.enough_kind). Filled, it was not "enough" at all: the words
    went to settle() as an ANSWER. For the week planner that is a None
    from settle and the plan he just built is dropped; for the echo double
    below it is echoed back and the session carries on. Either way the
    read-back he asked for never comes."""

    @pytest.fixture
    def session_cmdr(self, cmdr):
        from jarvis.dialogue import EchoSession
        session = EchoSession()
        session.ask()
        cmdr._pending_session = session
        return cmdr, session

    def test_bare_the_control(self, session_cmdr):
        cmdr, session = session_cmdr
        res = cmdr._try_session("that's enough")
        assert res is not None and res.status == "echo ended"
        assert res.reply == "0 noted, sir."
        assert cmdr._pending_session is None
        assert session.heard == []

    @pytest.mark.parametrize("said", ["uh, that's enough", "Um, that'll do.",
                                      "er, enough for now", "that's enough, uh"])
    def test_a_filled_enough_still_ends_it_with_the_read_back(self, session_cmdr, said):
        cmdr, session = session_cmdr
        res = cmdr._try_session(said)
        assert res is not None and res.status == "echo ended", said
        assert res.reply == "0 noted, sir.", said
        assert session.heard == [], said

    @pytest.mark.parametrize("said", ["uh, quiet", "um, cancel that", "er, stop that"])
    def test_a_filled_quiet_or_cancel_still_drops_it_silently(self, session_cmdr, said):
        cmdr, session = session_cmdr
        res = cmdr._try_session(said)
        assert res is not None and res.status == "echo stopped", said
        assert res.speak is False
        assert session.heard == [], said

    def test_a_filled_answer_is_still_the_answer(self, session_cmdr):
        cmdr, session = session_cmdr
        res = cmdr._try_session("uh, Tuesday")
        assert res is not None and res.status != "echo ended"
        assert session.heard == ["uh, Tuesday"]     # settle() sees his words


class TestTheQuizCancel:
    """The same teeth as the skip: "cancel that" / "quiet" over an open
    card end the quiz silently and grade nothing. Filled, they were not
    a cancel, not a stop word, and fell through to the GRADER -- the card
    was marked wrong."""

    @pytest.fixture
    def quiz_cmdr(self, cmdr, tmp_path):
        from jarvis.tools.quiz import FlashcardStore, QuizSession
        store = FlashcardStore(tmp_path / "cards.db")
        cards = store.add_cards(
            [{"question": "What is impedance?", "answer": "V over I"},
             {"question": "Name a transducer", "answer": "thermistor"}],
            source="example.md", topic="example")
        cmdr.services.flashcards = store
        cmdr._pending_quiz = QuizSession(cards, topic="example")
        cmdr._pending_quiz.ask()
        yield cmdr, store, cards
        store.close()

    def test_bare_the_control(self, quiz_cmdr):
        cmdr, store, cards = quiz_cmdr
        cid = cards[0]["id"]
        res = cmdr.handle("cancel that", "voice")
        assert res is not None and res.status == "Quiz stopped"
        assert cmdr._pending_quiz is None
        assert store.get(cid)["seen"] == 0 and store.get(cid)["box"] == 1

    @pytest.mark.parametrize("said", ["uh, cancel that", "Um, cancel that.",
                                      "uh, quiet", "er, stop that", "cancel that, uh"])
    def test_a_filled_cancel_grades_nothing(self, quiz_cmdr, said):
        cmdr, store, cards = quiz_cmdr
        cid = cards[0]["id"]
        session = cmdr._pending_quiz
        res = cmdr.handle(said, "voice")
        assert res is not None and res.status == "Quiz stopped", said
        assert session.results == [], said
        assert store.get(cid)["seen"] == 0, said
        assert store.get(cid)["box"] == 1, said


class TestTheEnrolControls:
    """"Stop" / "ready" / "hold on" while the CAMERA is capturing. The stop
    is the safe direction and takes the widest door -- every source -- and
    a filled "uh, stop" went through none of them."""

    class FakeRun:
        running = True

        def __init__(self):
            self.aborted, self.skipped, self.paused = [], 0, 0

        def abort(self, reason=""):
            self.aborted.append(reason)

        def skip(self):
            self.skipped += 1

        def pause(self):
            self.paused += 1

    @pytest.fixture
    def run_cmdr(self, cmdr):
        run = self.FakeRun()
        cmdr.services.enrol_run = run
        cmdr.services.enrol_offer = None
        return cmdr, run

    def test_bare_the_control(self, run_cmdr):
        cmdr, run = run_cmdr
        assert cmdr._try_enrol("stop").status == "Enrolment stopped"
        assert run.aborted
        assert cmdr._try_enrol("ready").status == "Enrolment: capturing"
        assert cmdr._try_enrol("hold on").status == "Enrolment: holding"

    @pytest.mark.parametrize("said", ["uh, stop", "Um, cancel.", "er, that's enough",
                                      "stop, uh"])
    def test_a_filled_stop_still_shuts_the_lens(self, run_cmdr, said):
        cmdr, run = run_cmdr
        res = cmdr._try_enrol(said)
        assert res is not None and res.status == "Enrolment stopped", said
        assert run.aborted, said

    def test_a_filled_ready_and_a_filled_wait(self, run_cmdr):
        cmdr, run = run_cmdr
        assert cmdr._try_enrol("uh, ready").status == "Enrolment: capturing"
        assert run.skipped == 1
        assert cmdr._try_enrol("um, hold on").status == "Enrolment: holding"
        assert run.paused == 1

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_a_pure_filler_steers_nothing(self, run_cmdr, said):
        cmdr, run = run_cmdr
        assert cmdr._try_enrol(said) is None, said
        assert not run.aborted and run.skipped == 0 and run.paused == 0

    def test_a_new_subject_that_starts_with_a_filler_steers_nothing(self, run_cmdr):
        cmdr, run = run_cmdr
        assert cmdr._try_enrol("uh, what time is it") is None
        assert not run.aborted


class TestTheDayShift:
    """"And the next day?" after "what do I have on tomorrow" -- a reply
    to the last turn, the same class as a correction, and start-anchored
    on "and / what about"."""

    PREV = "what do I have on tomorrow"

    def test_bare_the_control(self):
        from jarvis.commander import day_shift_followup
        assert day_shift_followup(self.PREV, "what about the next day") == \
            "what do I have on Tuesday"

    @pytest.mark.parametrize("said", ["uh, what about the next day",
                                      "Um, and the day after that?",
                                      "er, the next day", "and the next day, uh"])
    def test_a_filled_follow_up_still_moves_the_day(self, said):
        from jarvis.commander import day_shift_followup
        assert day_shift_followup(self.PREV, said) == "what do I have on Tuesday", said

    @pytest.mark.parametrize("said", PURE_FILLER + ["uh, what's the weather"])
    def test_a_pure_filler_or_a_new_subject_moves_nothing(self, said):
        from jarvis.commander import day_shift_followup
        assert day_shift_followup(self.PREV, said) is None, said


class TestTheLeaveTimeList:
    """leavetime._ANSWER_FILLER is a second, pre-existing list. Round 2 of
    the adversary contradicted "not a live gap": its "uh"/"um" entries
    wanted "uh " with a space, Whisper writes "Uh, fifteen.", and only the
    unit-word shapes below survived because _MIN_RX is a search. The rung
    now strips canonically first and the two entries are gone
    (test_hesitation_teeth.TestTheLeaveTimeAnswer measures the rung); this
    probe stays as the record of what the list itself does with the one
    vocabulary: the unit-word shapes, and nothing else."""

    @pytest.mark.parametrize("word", sorted(FILLER_WORDS))
    def test_every_filler_word_in_front_of_a_duration(self, word):
        from jarvis.leavetime import answer_minutes
        assert answer_minutes(f"{word}, ten minutes") == 10, word
        assert answer_minutes(f"{word} ten minutes") == 10, word
        assert answer_minutes(f"ten minutes, {word}") == 10, word


class TestABareHesitationLeavesTheQuestionStanding:
    """Integration note (b). The terminal offer left a bare "Uh..." parked
    -- "not an answer, and not a new subject either" -- while the briefing
    offer SPENT the day's one offer on it. One rule now, for both: a
    hesitation is not an answer to either question. (Taken as the default
    for him; flagged in the report.)"""

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_the_briefing_offer_stands_and_the_next_yes_runs_it(self, cmdr, said):
        ran = _offer(cmdr)
        cmdr.handle(said, "voice")
        assert ran == [], said
        assert isinstance(cmdr.services.briefing_offer, dict), \
            f"{said!r} spent the day's offer"
        assert cmdr.question_open()
        res = cmdr.handle("yeah", "voice")
        assert ran == [1], said
        assert res is not None and res.status.startswith("Briefing")

    @pytest.mark.parametrize("said", PURE_FILLER)
    def test_the_terminal_offer_stands_and_the_next_yes_opens_it(self, cmdr, said):
        cmdr.services.claude = MagicMock()
        cmdr.services.claude.open_terminal.return_value = True
        cmdr._pending_terminal_slug = "example"
        cmdr._pending_terminal_made = _t.monotonic()
        cmdr.handle(said, "voice")
        assert cmdr._pending_terminal_slug == "example", said
        res = cmdr.handle("yes", "voice")
        assert res is not None and res.status == "Terminal", said
        assert cmdr._pending_terminal_slug == ""

    def test_a_new_subject_still_drops_the_briefing_offer(self, cmdr):
        """The other half of the rule is untouched: a real change of
        subject drops the offer, as it always did."""
        ran = _offer(cmdr)
        cmdr.handle("uh, what time is it", "voice")
        assert ran == []
        assert cmdr.services.briefing_offer is None


class TestTheSendReadBackMaybe:
    """The send lane's own two filler lists (_SEND_FILLER_LEAD_RX and
    _SEND_FILLER_MID_RX) are backstopped by the canonical strip after them
    -- with ONE hole, found by probing the "backstopped" claim rather than
    repeating it. _send_clean hands the ORIGINAL back whole when the clean
    leaves no letters, which is right for "okay" (a maybe, and the re-ask
    it earns) and wrong for "uh, okay": the whole of "uh, okay" came back
    with the filler on, and _SEND_MAYBE_RX -- anchored on the maybe word
    -- refused it. Bare "okay" was asked again; "uh, okay" was not.

    The lists are left alone. The maybe judgement strips at its own call
    site, the same one-line move as everywhere else."""

    @pytest.fixture
    def readback_cmdr(self, cmdr, tmp_path):
        from jarvis import outbox
        root = tmp_path / "Desktop"
        root.mkdir()
        f = root / "lab_report.pdf"
        f.write_bytes(b"%PDF-1.4 body")
        st = f.stat()
        draft = outbox.Draft(
            path=f, size=st.st_size, mtime=st.st_mtime,
            to_addr="dana@example.com", to_name="Dana",
            account={"label": "school", "address": "me@example.com",
                     "password": "example-secret"},
            subject="Lab report", roots=[str(root)], made_at=_t.monotonic(),
            asked_from="voice")
        cmdr._pending_send = draft
        cmdr.services.smtp = None            # nothing here may send
        return cmdr, draft

    def test_the_hole_in_the_backstop_is_real(self):
        """Pinned as it stands: the lane's own clean hands "uh, okay" back
        whole. The fix is downstream of it, not in it."""
        from jarvis.commander import _send_clean
        assert _send_clean("okay") == "okay"
        assert _send_clean("uh, okay") == "uh, okay"

    def test_bare_the_control(self, readback_cmdr):
        cmdr, draft = readback_cmdr
        res = cmdr._try_send_confirm("okay", "voice")
        assert res is not None and res.status == "Confirm?"
        assert cmdr._pending_send is draft and draft.reasked

    @pytest.mark.parametrize("said", ["uh, okay", "Um, sure.", "er, alright",
                                      "okay, uh", "uh, uh-huh"])
    def test_a_filled_maybe_still_earns_the_re_ask(self, readback_cmdr, said):
        cmdr, draft = readback_cmdr
        res = cmdr._try_send_confirm(said, "voice")
        assert res is not None and res.status == "Confirm?", said
        assert cmdr._pending_send is draft and draft.reasked, said

    @pytest.mark.parametrize("word", sorted(FILLER_WORDS))
    def test_every_filler_word_still_leaves_a_yes_and_a_no(self, word):
        """The backstop itself, probed word by word: a filled yes and a
        filled no on the read-back come through the lane's own lists AND
        the canonical strip behind them."""
        from jarvis.commander import parse_send_answer
        assert parse_send_answer(f"{word}, yes") is True, word
        assert parse_send_answer(f"{word}, no") is False, word


class TestTheAddressAndTheFiller:
    """What the derived census found LAST, and reading had not: three rungs
    take the address off BEFORE the filler -- ``correction_kind(
    strip_address(text))`` and its two siblings -- and strip_address is
    anchored at ^. So "jarvis, uh, I said Lisbon" was a correction and
    "uh, jarvis, I said Lisbon" was not: the address was not at the
    front, stayed on, and the grammar behind it (which does not take an
    address of its own) refused the sentence. Measured before the fix:

        correction_kind(strip_address("uh, jarvis, I said Lisbon"))  -> None
        _LEAVE_DECLINE_RX  ... "uh, jarvis, no idea"                  -> no
        day_shift_followup ... "uh, jarvis, what about the next day"  -> None

    A filler can sit on EITHER side of the address, so the words are
    stripped, then un-addressed, then stripped again -- the same one
    function, one more time, at the call site."""

    def _turn(self, cmdr, text):
        from jarvis.commander import LastTurn
        cmdr._last_turn = LastTurn(text, "ok", _t.monotonic())

    @pytest.fixture
    def redispatch(self, cmdr, monkeypatch):
        meant = []

        def fake_inner(text, source, gate=True, **kw):
            meant.append(text)
            return CommandResult(handled=True, status="Re-run")

        monkeypatch.setattr(cmdr, "_handle_inner", fake_inner)
        return meant

    # -- the correction
    def test_bare_the_control_correction(self, cmdr, redispatch):
        self._turn(cmdr, "book a table in Lisburn")
        res = cmdr._try_correction("jarvis, I said Lisbon", "voice")
        assert res is not None and res.corrected == "Lisbon"
        assert redispatch == ["Lisbon"]

    @pytest.mark.parametrize("said", ["uh, jarvis, I said Lisbon",
                                      "Um, Jarvis, I said Lisbon.",
                                      "jarvis, uh, I said Lisbon",
                                      "uh, jarvis, uh, I said Lisbon"])
    def test_a_filler_on_either_side_of_the_address_is_still_a_correction(
            self, cmdr, redispatch, said):
        self._turn(cmdr, "book a table in Lisburn")
        res = cmdr._try_correction(said, "voice")
        assert res is not None and res.corrected == "Lisbon", said
        assert redispatch == ["Lisbon"], said

    # -- the leave-time decline
    @pytest.fixture
    def leave_cmdr(self, cmdr):
        cmdr.services.leavetime = SimpleNamespace(
            learn=lambda key, minutes: int(minutes))
        cmdr._pending_leave = ("Example Hall", "Example Hall", _t.monotonic())
        return cmdr

    def test_bare_the_control_leave(self, leave_cmdr):
        from jarvis.commander import LEAVE_DROPPED_LINE
        res = leave_cmdr._try_leave_answer("jarvis, no idea")
        assert res is not None and res.reply == LEAVE_DROPPED_LINE

    @pytest.mark.parametrize("said", ["uh, jarvis, no idea", "Um, Jarvis, no idea.",
                                      "jarvis, uh, no idea"])
    def test_a_filler_on_either_side_of_the_address_still_declines(
            self, leave_cmdr, said):
        from jarvis.commander import LEAVE_DROPPED_LINE
        res = leave_cmdr._try_leave_answer(said)
        assert res is not None and res.reply == LEAVE_DROPPED_LINE, said
        assert leave_cmdr._pending_leave is None, said

    # -- the day-shift follow-up
    def test_bare_the_control_day_shift(self, cmdr, redispatch):
        self._turn(cmdr, "what do I have on tomorrow")
        res = cmdr._try_day_shift("jarvis, what about the next day", "voice")
        assert res is not None and res.corrected == "what do I have on Tuesday"

    @pytest.mark.parametrize("said", ["uh, jarvis, what about the next day",
                                      "jarvis, uh, and the day after that?"])
    def test_a_filler_on_either_side_of_the_address_still_moves_the_day(
            self, cmdr, redispatch, said):
        self._turn(cmdr, "what do I have on tomorrow")
        res = cmdr._try_day_shift(said, "voice")
        assert res is not None and res.corrected == "what do I have on Tuesday", said
        assert redispatch == ["what do I have on Tuesday"], said

    @pytest.mark.parametrize("said", PURE_FILLER + ["uh, jarvis"])
    def test_a_pure_filler_with_or_without_the_address_is_nothing(
            self, cmdr, redispatch, said):
        self._turn(cmdr, "what do I have on tomorrow")
        assert cmdr._try_correction(said, "voice") is None, said
        assert cmdr._try_day_shift(said, "voice") is None, said
        assert redispatch == []
