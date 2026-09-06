"""Round 4 of the filled pause (09-06): the '?' carry reached the STOPS.

Round 3 gave ``strip_fillers`` a bar it needed -- a terminal "?" that rode
on a dropped trailing filler is kept, so "yes, uh?" never sends, grants,
opens or runs.  The carry lives in the CANONICAL strip, so it also reached
every grammar where REFUSING is the unsafe direction, and those grammars
ended on ``[.!\\s]*$``.  Round 3's adversary measured it through the real
``Commander.handle`` with the state armed:

  * the flashcard: 13 stop words x 13 fillers x 4 rising shapes = 676
    forms, 0 stopped and 676 MARKED WRONG and written to the store, the
    quiz carrying on ("stop, uh?" -> "Quiz 2/2");
  * the quiz skip words: 78/234 marked wrong;
  * the working session: 429/429 swallowed as an ANSWER and handed to
    ``settle()``;
  * a live camera run: 0/195 aborted, one line under a docstring that
    says "uh, stop not stopping a CAMERA is the unsafe direction";
  * the undo lane, which REGRESSED against mainline: 0/104.

The single-character ellipsis was the same defect wearing another hat:
"stop, uh…" -> "stop…", refused by five more grammars that already
tolerated a "?".  Note that "stop, uh..." with three ASCII dots was fine
throughout -- ``[.!]*$`` takes those -- which is why nobody saw it.

THE FIX IS AT THE ANCHOR, NOT THE CALL SITE.  Each grammar's terminal
class is that grammar's own statement about the punctuation it tolerates,
and it is true for every caller; a ``keep_terminal=False`` flag on
``strip_fillers`` would have had to be threaded through every stop-side
call site, and a call site MISSED is silent -- which is the defect this
file exists to close.  The anchors are also enumerable from the source,
so the mirror rule below can be a pin rather than a promise.

THE DEFAULT TAKEN FOR HIM, flagged in the report: A STOP WORD WITH A "?"
OR AN "…" HUNG ON IT STOPS.  A man who says "stop?" at six in the morning
is not asking a question, and ``_try_ringing``'s own rule is that stop is
the safe direction.  The "?" bar stays exactly where acting is
irreversible -- send, approve, open a terminal, run a strict destructive,
replace the face gallery -- and ``_YES_RX`` / ``_NO_RX`` /
``_SEND_YES_BAR_RX`` / ``_OPEN_IT_RX`` / ``_ENROL_CONFIRM_RX`` are
untouched.  test_hesitation_teeth.py holds the other half: 520 rising
forms still act on nothing and 390 flat ones still act on everything.

Every utterance here is a string written in this file; every clock is the
commander's own.  Nothing opens a microphone, a camera or a mailbox.
"""
from __future__ import annotations

import time as _t
from types import SimpleNamespace

import pytest

import jarvis.commander as C
import jarvis.dialogue as dialogue_mod
from jarvis.commander import CommandResult
from jarvis.endpoint import FILLER_WORDS
from tests.test_filled_yes import make_cmdr

FILLERS = sorted(FILLER_WORDS)                     # 13 spellings, one vocabulary

#: The shapes Whisper hangs a rising tail on.  THESE ARE THE ADVERSARY'S
#: THREE, kept exactly so the counts below are the measured ones and can
#: be compared line for line with round 3's report: the "?" on the
#: filler with and without the comma, and the single-character ellipsis.
#: "..." (three ASCII dots) is in the FLAT set because ``[.!]*$`` always
#: took those, which is precisely why nobody saw the "…" for a round.
RISING = ["{b}, {f}?", "{b} {f}?", "{b}, {f}…"]
#: The fourth shape -- the "?" as its own token -- swept over the card
#: (13 x 13 x 4 = 676) and pinned on its own for the other four rungs.
SPACED = "{b}, {f} ?"
CARD_RISING = RISING + [SPACED]
FLAT = ["{b}, {f}", "{b}, {f}.", "{b}, {f}..."]


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    return make_cmdr(tmp_path, monkeypatch)


def _through(cmdr, said: str):
    """One utterance through the REAL handler, as the mic would give it."""
    cmdr._last_turn = None
    return cmdr.handle(said, "voice")


def _forms(bases, shapes):
    for b in bases:
        for f in FILLERS:
            for sh in shapes:
                yield b, sh.format(b=b, f=f)


# ===================================================================
# 1. THE FLASHCARD -- 676 forms, the biggest of the five
# ===================================================================
STOP_WORDS = ["stop", "stop it", "quiet", "be quiet", "silence", "enough",
              "that's enough", "never mind", "cancel that", "stop that",
              "shut up", "that'll do", "hush"]


@pytest.fixture
def quiz_cmdr(cmdr, tmp_path):
    from jarvis.tools.quiz import FlashcardStore, QuizSession
    store = FlashcardStore(tmp_path / "cards.db")
    cards = store.add_cards(
        [{"question": "What is impedance?", "answer": "V over I"},
         {"question": "Name a transducer", "answer": "thermistor"}],
        source="example.md", topic="example")
    cmdr.services.flashcards = store

    def fresh():
        cmdr._pending_quiz = QuizSession(cards, topic="example")
        cmdr._pending_quiz.ask()
        return cmdr._pending_quiz

    yield cmdr, store, cards, fresh
    store.close()


def _untouched(store, cid) -> bool:
    row = store.get(cid)
    return row["seen"] == 0 and row["box"] == 1


class TestARisingHesitatedStopStopsTheCard:
    """676 forms: 0 marked wrong, 676 stopped.  Was 676 marked wrong."""

    def test_676_rising_hesitated_stops_stop_the_quiz(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        cid = cards[0]["id"]
        bad, n = [], 0
        for _, said in _forms(STOP_WORDS, CARD_RISING):
            n += 1
            session = fresh()
            _through(cmdr, said)
            if not (cmdr._pending_quiz is None and session.results == []
                    and _untouched(store, cid)):
                bad.append(said)
        assert n == 676 and bad == [], f"{len(bad)}/{n} did NOT stop: {bad[:8]}"

    def test_the_flat_forms_stop_exactly_as_they_always_did(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        cid = cards[0]["id"]
        bad, n = [], 0
        for _, said in _forms(STOP_WORDS, FLAT):
            n += 1
            session = fresh()
            _through(cmdr, said)
            if not (cmdr._pending_quiz is None and session.results == []
                    and _untouched(store, cid)):
                bad.append(said)
        assert n == 507 and bad == [], f"{len(bad)}/{n} did NOT stop: {bad[:8]}"

    def test_the_card_is_still_graded_when_he_actually_answers(self, quiz_cmdr):
        """The stop must not have eaten the answer path."""
        cmdr, store, cards, fresh = quiz_cmdr
        session, cid = fresh(), cards[0]["id"]
        _through(cmdr, "uh, v over i")
        assert session.results == [(cid, True)]


# ===================================================================
# 2. THE QUIZ SKIP WORDS -- 234 forms, the "…" half
# ===================================================================
SKIP_WORDS = ["skip it", "pass", "i don't know", "no idea", "not sure",
              "tell me the answer"]


class TestARisingHesitatedSkipSkipsUngraded:
    def test_234_rising_hesitated_skips_are_never_marked_wrong(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        cid = cards[0]["id"]
        bad, n = [], 0
        for _, said in _forms(SKIP_WORDS, RISING):
            n += 1
            session = fresh()
            _through(cmdr, said)
            if not (session.results == [(cid, None)] and _untouched(store, cid)):
                bad.append(said)
        assert n == 234 and bad == [], f"{len(bad)}/{n} not skipped: {bad[:8]}"

    def test_the_flat_skips_are_unchanged(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        cid = cards[0]["id"]
        bad, n = [], 0
        for _, said in _forms(SKIP_WORDS, FLAT):
            n += 1
            session = fresh()
            _through(cmdr, said)
            if not (session.results == [(cid, None)] and _untouched(store, cid)):
                bad.append(said)
        assert n == 234 and bad == [], f"{len(bad)}/{n} not skipped: {bad[:8]}"


# ===================================================================
# 3. THE WORKING SESSION -- 429 forms handed to settle()
# ===================================================================
SESSION_STOPS = ["stop", "quiet", "enough", "that's enough", "never mind",
                 "cancel that", "that'll do", "we're done", "leave it there",
                 "silence", "be quiet"]


class _Sess(dialogue_mod.Session):
    """A plan-the-week session: it records what it was handed, so a stop
    swallowed as an ANSWER is visible rather than merely absent."""

    name = "plan the week"

    def __init__(self):
        self.settled, self.stopped = [], False
        self.asked_at = _t.time()

    @property
    def finished(self):
        return False

    def stale(self, now=None):
        return False

    def ask(self, now=None):
        self.asked_at = _t.time()
        return "Which day, sir?"

    def settle(self, text):
        self.settled.append(text)
        return "Noted."

    def status_text(self):
        return "1/3"

    def stop(self):
        self.stopped = True
        return "Stopped."


class TestARisingHesitatedStopEndsTheSession:
    def test_429_rising_hesitated_stops_end_the_session(self, cmdr):
        bad, n = [], 0
        for _, said in _forms(SESSION_STOPS, RISING):
            n += 1
            sess = _Sess()
            cmdr._pending_session = sess
            _through(cmdr, said)
            if not (cmdr._pending_session is None and sess.settled == []):
                bad.append((said, sess.settled))
        assert n == 429 and bad == [], f"{len(bad)}/{n} not ended: {bad[:6]}"

    def test_the_flat_forms_end_it_as_they_always_did(self, cmdr):
        bad, n = [], 0
        for _, said in _forms(SESSION_STOPS, FLAT):
            n += 1
            sess = _Sess()
            cmdr._pending_session = sess
            _through(cmdr, said)
            if not (cmdr._pending_session is None and sess.settled == []):
                bad.append((said, sess.settled))
        assert n == 429 and bad == [], f"{len(bad)}/{n} not ended: {bad[:6]}"

    def test_a_real_answer_still_reaches_settle(self, cmdr):
        sess = _Sess()
        cmdr._pending_session = sess
        _through(cmdr, "uh, tuesday")
        assert sess.settled == ["uh, tuesday"] or sess.settled == ["tuesday"], \
            sess.settled


# ===================================================================
# 4. THE CAMERA RUN -- 195 forms; "not stopping a CAMERA is the unsafe
#    direction" is the rung's own docstring
# ===================================================================
ENROL_STOPS = ["stop", "cancel", "abort", "never mind", "forget it"]


class _Run:
    def __init__(self):
        self.running, self.aborted = True, []
        self.skipped = self.paused = 0

    def abort(self, why):
        self.aborted.append(why)
        self.running = False

    def skip(self):
        self.skipped += 1

    def pause(self):
        self.paused += 1


class TestARisingHesitatedStopAbortsTheCameraRun:
    def test_195_rising_hesitated_stops_abort_the_run(self, cmdr):
        bad, n = [], 0
        for _, said in _forms(ENROL_STOPS, RISING):
            n += 1
            run = _Run()
            cmdr.services.enrol_run = run
            _through(cmdr, said)
            if not run.aborted:
                bad.append(said)
        assert n == 195 and bad == [], f"{len(bad)}/{n} left the lens open: {bad[:8]}"

    def test_the_flat_forms_abort_it_as_they_always_did(self, cmdr):
        bad, n = [], 0
        for _, said in _forms(ENROL_STOPS, FLAT):
            n += 1
            run = _Run()
            cmdr.services.enrol_run = run
            _through(cmdr, said)
            if not run.aborted:
                bad.append(said)
        assert n == 195 and bad == [], f"{len(bad)}/{n} left the lens open: {bad[:8]}"

    def test_the_two_other_run_controls_take_the_same_punctuation(self, cmdr):
        """``_ENROL_READY_RX``'s twin ``_ENROL_WAIT_RX`` was NOT in the
        brief.  It is the same anchor on the same rung, and shipping one
        widened and not the other is the guard-one-half pattern itself."""
        for word, attr in (("ready", "skipped"), ("next", "skipped"),
                           ("wait", "paused"), ("hold on", "paused")):
            for w in FILLERS:
                for said in (f"{word}, {w}?", f"{word}, {w}…"):
                    run = _Run()
                    cmdr.services.enrol_run = run
                    _through(cmdr, said)
                    assert getattr(run, attr) == 1, said


# ===================================================================
# 5. THE UNDO LANE -- 104 forms, and the one that REGRESSED against
#    mainline rather than merely failing to improve on it
# ===================================================================
UNDO_WORDS = ["scratch that", "undo that", "undo", "cancel that"]


class TestARisingHesitatedUndoUndoes:
    def test_104_rising_hesitated_undos_undo(self, cmdr):
        """Only the two "?" shapes: mainline 13162d2 was 80/104 here and
        683f6a4 was 104/104, so this lane is a REGRESSION fix, not a new
        default.  The "…" shapes were never lost and are covered below."""
        bad, n = [], 0
        for b in UNDO_WORDS:
            for w in FILLERS:
                for said in (f"{b}, {w}?", f"{b} {w}?"):
                    n += 1
                    done = []
                    cmdr._last_undo = (
                        lambda: (done.append(1),
                                 CommandResult(handled=True, reply="Undone.",
                                               status="Undone"))[1],
                        _t.monotonic())
                    _through(cmdr, said)
                    if not done:
                        bad.append(said)
        assert n == 104 and bad == [], f"{len(bad)}/{n} not undone: {bad[:8]}"

    def test_the_other_six_undo_shapes_are_unchanged(self, cmdr):
        bad, n = [], 0
        for b in UNDO_WORDS:
            for w in FILLERS:
                for said in (f"{b}, {w}…", f"{b}, {w}", f"{b}, {w}.",
                             f"{b} {w}", f"{w.capitalize()} - {b}",
                             f"{w.capitalize()}, {b}"):
                    n += 1
                    done = []
                    cmdr._last_undo = (
                        lambda: (done.append(1),
                                 CommandResult(handled=True, reply="Undone.",
                                               status="Undone"))[1],
                        _t.monotonic())
                    _through(cmdr, said)
                    if not done:
                        bad.append(said)
        assert n == 312 and bad == [], f"{len(bad)}/{n} not undone: {bad[:8]}"


# ===================================================================
# 6. THE RINGING ALARM -- the pin round 3 left standing, INVERTED
# ===================================================================
def _ring(cmdr):
    calls = []
    cmdr.services.timekeeper = SimpleNamespace(
        ringing=SimpleNamespace(id="a1", label="Get up", kind="alarm"),
        stop_ringing=lambda why: calls.append(("stop", why)),
        snooze=lambda n: calls.append(("snooze", int(n))))
    return calls


class TestARisingHesitatedStopSilencesTheAlarm:
    """test_hesitation_teeth's ``test_a_RISING_hesitated_stop_leaves_the
    _alarm_ringing`` is now this, and the assertion is inverted: 26/26
    SILENCED where it pinned 26/26 ringing."""

    def test_26_rising_hesitated_stops_silence_the_alarm(self, cmdr):
        lost, n = [], 0
        for w in FILLERS:
            for said in (f"{w.capitalize()}, stop?", f"stop, {w}?"):
                n += 1
                calls = _ring(cmdr)
                _through(cmdr, said)
                if calls != [("stop", "dismiss")]:
                    lost.append(said)
        assert n == 26 and lost == [], f"{len(lost)}/{n} kept ringing: {lost[:8]}"

    def test_the_ellipsis_shape_too(self, cmdr):
        lost, n = [], 0
        for w in FILLERS:
            for said in (f"stop, {w}…", f"stop {w}…"):
                n += 1
                calls = _ring(cmdr)
                _through(cmdr, said)
                if calls != [("stop", "dismiss")]:
                    lost.append(said)
        assert n == 26 and lost == [], f"{len(lost)}/{n} kept ringing: {lost[:8]}"

    def test_a_rising_hesitated_snooze_snoozes(self, cmdr):
        lost, n = [], 0
        for w in FILLERS:
            for said in (f"snooze, {w}?", f"snooze, {w}…"):
                n += 1
                calls = _ring(cmdr)
                _through(cmdr, said)
                if calls != [("snooze", 10)]:
                    lost.append((said, calls))
        assert n == 26 and lost == [], f"{len(lost)}/{n} not snoozed: {lost[:6]}"


# ===================================================================
# 7. THE LEAVE-TIME ANSWER -- the "…" that ate a duration
# ===================================================================
class TestALeaveAnswerSurvivesTheEllipsis:
    def test_a_hesitated_duration_is_learned_on_every_shape(self, cmdr):
        learned = []
        cmdr.services.leavetime = SimpleNamespace(
            learn=lambda *a, **k: (learned.append(a), 12)[1])
        bad, n = [], 0
        for b in ("fifteen", "about ten", "half an hour", "ten minutes"):
            for w in FILLERS:
                for said in (f"{b}, {w}?", f"{b}, {w}…", f"{b}, {w}"):
                    n += 1
                    learned.clear()
                    cmdr._pending_leave = ("office", "the office", _t.monotonic())
                    _through(cmdr, said)
                    if not learned:
                        bad.append(said)
        assert n == 156 and bad == [], f"{len(bad)}/{n} not learned: {bad[:8]}"


# ===================================================================
# 8. THE FOURTH RISING SHAPE -- swept over the card above (676); pinned
#    here for the other four rungs, so no rung is verified on three
#    shapes and shipped on four
# ===================================================================
class TestTheSpacedQuestionMarkShape:
    """"stop, uh ?" -- Whisper writes the "?" as its own token often
    enough that round 3's card sweep carried it. Each rung gets a FRESH
    commander state here: an open flashcard left behind by the loop
    above sits ABOVE the undo rung and eats its words, which is a bug in
    a test rather than in the app, and it cost half an hour."""

    def test_the_skip_words(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        cid = cards[0]["id"]
        bad, n = [], 0
        for b in SKIP_WORDS:
            for w in FILLERS:
                n += 1
                session = fresh()
                _through(cmdr, SPACED.format(b=b, f=w))
                if not (session.results == [(cid, None)] and _untouched(store, cid)):
                    bad.append(SPACED.format(b=b, f=w))
        assert n == 78 and bad == [], f"{len(bad)}/{n} not skipped: {bad[:8]}"

    def test_the_session(self, cmdr):
        bad, n = [], 0
        for b in SESSION_STOPS:
            for w in FILLERS:
                n += 1
                said = SPACED.format(b=b, f=w)
                sess = _Sess()
                cmdr._pending_session = sess
                _through(cmdr, said)
                if not (cmdr._pending_session is None and sess.settled == []):
                    bad.append(said)
        assert n == 143 and bad == [], f"{len(bad)}/{n} not ended: {bad[:8]}"

    def test_the_camera_run(self, cmdr):
        bad, n = [], 0
        for b in ENROL_STOPS:
            for w in FILLERS:
                n += 1
                said = SPACED.format(b=b, f=w)
                run = _Run()
                cmdr.services.enrol_run = run
                _through(cmdr, said)
                if not run.aborted:
                    bad.append(said)
        assert n == 65 and bad == [], f"{len(bad)}/{n} left the lens open: {bad[:8]}"

    def test_the_undo_lane(self, cmdr):
        bad, n = [], 0
        for b in UNDO_WORDS:
            for w in FILLERS:
                n += 1
                said = SPACED.format(b=b, f=w)
                done = []
                cmdr._last_undo = (
                    lambda: (done.append(1),
                             CommandResult(handled=True, reply="Undone.",
                                           status="Undone"))[1],
                    _t.monotonic())
                _through(cmdr, said)
                if not done:
                    bad.append(said)
        assert n == 52 and bad == [], f"{len(bad)}/{n} not undone: {bad[:8]}"

    def test_the_alarm(self, cmdr):
        bad, n = [], 0
        for w in FILLERS:
            n += 1
            said = f"stop, {w} ?"
            calls = _ring(cmdr)
            _through(cmdr, said)
            if calls != [("stop", "dismiss")]:
                bad.append(said)
        assert n == 13 and bad == [], f"{len(bad)}/{n} kept ringing: {bad}"
