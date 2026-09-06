"""Round 3 of the filled pause (09-06): the places where the FIX had teeth.

Round 2 of the adversary confirmed the strip at seventeen rungs and the
derived census, and blocked on what follows -- every number measured
through the real Commander.handle with the state armed, every utterance a
string somebody typed, nothing from his microphone or the running process:

  1. THE BAR. strip_fillers dropped a terminal "?" that rode on a trailing
     filler ("yes, uh?" -> "yes"), against its own docstring. Through
     handle(): send read-back 156/156 SENT, approval 156/156 GRANTED,
     terminal offer 104/104 OPENED, strict destructive 104/104 RAN --
     520/520 -- where the bare "yes?" was refused 0/10 on the same tree
     and mainline 13162d2 was 0/520 (its send lane left "yes, ?" behind
     and the bar caught it). The canonical strip now carries the dropped
     tail's terminal punctuation, and every rung inherits that.
  2. THE CARD. A bare filled pause over an open flashcard ("Hmm.",
     "Ah...", "Er,") was GRADED: 99/117 marked wrong and written to the
     store, 18/117 spent the card and read the answer out. A hesitated
     "Ah, quiz me on chemistry" reached the grader too, 13/13 marked
     wrong. The card now stands on a bare pause, and the new-quiz ESCAPE
     runs on the stripped words.
  3. TWO RUNGS THE CENSUS COULD NOT SEE: a ringing alarm ("Uh, stop."
     lost 1170/1170, the alarm kept ringing) and lecture notes ("Uh, end
     notes." filed as a note, 468/468).
  4. SMALLER: leavetime's second list ("Uh, fifteen." not learned,
     625/1170), the undo dash form ("Uh - scratch that", 65/585), and
     three word counts that counted fillers as words he said.

DEFAULTS TAKEN FOR HIM, each flagged in the report: a hesitant rising yes
never sends, grants, opens or runs; a flashcard stands on a bare pause
until he answers, skips or stops; a hesitated "quiz me on X" drops the
open card and routes on (the registry still sees the unstripped words --
the 5caf86c ruling -- so it may cost him a repeat, never a card).

TWO THINGS ARE LEFT STANDING and pinned in section 5 rather than fixed,
because neither was in the four items and each would be a fifth default
taken for him: the send-ask question ("Which Heather, sir?") still reads
a bare filled pause as an answer and spends its one re-ask, 117/117,
where the flashcard was given the opposite rule this round; and a RISING
hesitated stop over a ringing alarm ("Uh, stop?") leaves it ringing,
26/26, which is the "?" bar and the "stop is the safe direction" rule
pulling in opposite directions. Neither is a regression -- mainline
13162d2 lost both -- and both wait on him.
"""
from __future__ import annotations

import time as _t
from types import SimpleNamespace

import pytest

from jarvis.commander import CommandResult, SendAsk
from jarvis.endpoint import FILLER_WORDS, strip_fillers
from tests.test_filled_yes import make_cmdr

FILLERS = sorted(FILLER_WORDS)          # 13 spellings, one vocabulary


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    return make_cmdr(tmp_path, monkeypatch)


def _through(cmdr, said: str):
    """One utterance through the REAL handler, as the mic would give it."""
    cmdr._last_turn = None            # the previous form is not this one's context
    return cmdr.handle(said, "voice")


def _disarm(cmdr):
    """No other rung holds a question: what one arm leaves behind must not
    answer the next form first (measured: an approval left armed by an
    earlier sweep swallowed "yes, uh?" before the terminal rung saw it)."""
    cmdr.services.approvals = None
    cmdr.services.claude = None
    cmdr._pending_send = None
    cmdr._send_aside = None
    cmdr._pending_terminal_slug = ""
    cmdr._pending_destructive = None
    cmdr._pending_destructive_meta = None
    cmdr._pending_sendask = None


# ===================================================================
# 1. THE BAR: a "?" on a dropped trailing filler is still the "?"
# ===================================================================
def _rising(phrase: str, w: str) -> list:
    """The four shapes Whisper gives a rising tail: the "?" on the filler,
    with and without the comma, as its own token, and doubled."""
    return [f"{phrase}, {w}?", f"{phrase} {w}?", f"{phrase}, {w} ?", f"{phrase}, {w}??"]


def _flat(phrase: str, w: str) -> list:
    """A hesitated PLAIN yes: the bar must not swallow these."""
    return [f"{phrase}, {w}", f"{phrase}, {w}.", f"{phrase}, {w}!"]


class TestStripFillersKeepsTheTailsPunctuation:
    @pytest.mark.parametrize("w", FILLERS)
    def test_the_four_punctuation_shapes(self, w):
        assert strip_fillers(f"yes, {w}?") == "yes?", w
        assert strip_fillers(f"send it, {w}?") == "send it?", w
        assert strip_fillers(f"yes, {w}.") == "yes.", w
        assert strip_fillers(f"yes, {w}!") == "yes!", w

    @pytest.mark.parametrize("w", FILLERS)
    def test_the_question_mark_survives_wherever_it_landed(self, w):
        assert strip_fillers(f"yes {w} ?") == "yes?", w            # its own token
        assert strip_fillers(f"yes, {w}? {w}") == "yes?", w        # mid-run
        assert strip_fillers(f"{w}, yes, {w}?") == "yes?", w       # both ends
        assert "?" in strip_fillers(f"yes. {w}?"), w               # never lost

    @pytest.mark.parametrize("w", FILLERS)
    def test_a_plain_hesitated_yes_is_still_plain(self, w):
        assert strip_fillers(f"yes, {w}") == "yes", w
        assert strip_fillers(f"yes, {w},") == "yes", w
        assert strip_fillers(f"{w}? yes") == "yes", w              # a leading "?" is his hesitation

    @pytest.mark.parametrize("w", FILLERS)
    def test_nothing_but_fillers_is_still_nothing(self, w):
        assert strip_fillers(f"{w}?") == ""
        assert strip_fillers(f"{w}, {w}?") == ""


def _arm_send(cmdr, tmp_path):
    """A file-send read-back armed for the next utterance. The send itself
    goes through the ``_bg`` seam: counted, never run."""
    _disarm(cmdr)
    from jarvis import outbox
    root = tmp_path / "Desktop"
    root.mkdir(exist_ok=True)
    f = root / "lab_report.pdf"
    if not f.exists():
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
    cmdr.services.smtp = None
    launched = []
    cmdr._bg = lambda fn: launched.append(fn)
    return launched


def _arm_approval(cmdr):
    _disarm(cmdr)
    answers = []
    cmdr.services.approvals = SimpleNamespace(
        pending=lambda: True, answer=lambda ok, source="": answers.append(ok))
    return answers


def _arm_terminal(cmdr):
    _disarm(cmdr)
    opened = []
    cmdr.services.claude = SimpleNamespace(
        open_terminal=lambda slug: (opened.append(slug), True)[1])
    cmdr._pending_terminal_slug = "vss"
    cmdr._pending_terminal_made = _t.monotonic()
    return opened


def _arm_strict(cmdr):
    _disarm(cmdr)
    ran = []
    pend = (lambda: (ran.append(1), CommandResult(handled=True, reply="done",
                                                   status="Done"))[1],
            "move the lab report to HPCOMPUTER", _t.monotonic())
    cmdr._pending_destructive = pend
    cmdr._pending_destructive_meta = ("voice", True, pend)
    cmdr._strict_reasked = False
    cmdr.services.notes = None
    return ran


# What each rung takes as a plain yes (the send lane has no "allow it",
# the approval grammar no "send it"): 3 + 3 + 2 + 2 phrases x 4 rising
# shapes x 13 fillers = 156 + 156 + 104 + 104 = 520 forms.
SEND_YES = ["yes", "send it", "go ahead"]
APPROVE_YES = ["yes", "allow it", "go ahead"]
TERMINAL_YES = ["yes", "open it"]
STRICT_YES = ["yes", "go ahead"]


def _sweep(cmdr, arm, acted, phrases, shapes):
    """Every form through handle() against freshly armed state; the forms
    on which the rung ACTED, and n."""
    hits, n = [], 0
    for w in FILLERS:
        for p in phrases:
            for said in shapes(p, w):
                n += 1
                record = arm()
                _through(cmdr, said)
                if acted(record):
                    hits.append(said)
    return n, hits


class TestTheBarThroughTheHandler:
    """520 rising forms, 0 acted; the bare "?" forms refused; the flat
    hesitated yes still acts on every rung (the bar must not swallow the
    yes)."""

    def test_the_send_read_back_never_sends_on_a_rising_hesitated_yes(self, cmdr, tmp_path):
        n, hits = _sweep(cmdr, lambda: _arm_send(cmdr, tmp_path), bool,
                         SEND_YES, _rising)
        assert n == 156 and hits == [], f"{len(hits)}/{n} SENT: {hits[:8]}"

    def test_the_approval_never_grants_on_a_rising_hesitated_yes(self, cmdr):
        n, hits = _sweep(cmdr, lambda: _arm_approval(cmdr), lambda a: a == [True],
                         APPROVE_YES, _rising)
        assert n == 156 and hits == [], f"{len(hits)}/{n} GRANTED: {hits[:8]}"

    def test_the_terminal_offer_never_opens_on_a_rising_hesitated_yes(self, cmdr):
        n, hits = _sweep(cmdr, lambda: _arm_terminal(cmdr), bool,
                         TERMINAL_YES, _rising)
        assert n == 104 and hits == [], f"{len(hits)}/{n} OPENED: {hits[:8]}"

    def test_the_strict_read_back_never_runs_on_a_rising_hesitated_yes(self, cmdr):
        n, hits = _sweep(cmdr, lambda: _arm_strict(cmdr), bool,
                         STRICT_YES, _rising)
        assert n == 104 and hits == [], f"{len(hits)}/{n} RAN: {hits[:8]}"

    @pytest.mark.parametrize("said", ["yes?", "send it?", "go ahead?", "allow it?",
                                      "open it?"])
    def test_the_bare_rising_forms_are_refused_as_they_always_were(self, cmdr, tmp_path, said):
        launched = _arm_send(cmdr, tmp_path)
        _through(cmdr, said)
        answers = _arm_approval(cmdr)
        _through(cmdr, said)
        opened = _arm_terminal(cmdr)
        _through(cmdr, said)
        ran = _arm_strict(cmdr)
        _through(cmdr, said)
        assert launched == [] and answers != [True] and opened == [] and ran == [], said

    def test_the_flat_hesitated_yes_still_sends(self, cmdr, tmp_path):
        n, hits = _sweep(cmdr, lambda: _arm_send(cmdr, tmp_path), bool, SEND_YES, _flat)
        assert n == 117 and len(hits) == n, f"{n - len(hits)}/{n} NOT sent"

    def test_the_flat_hesitated_yes_still_grants(self, cmdr):
        n, hits = _sweep(cmdr, lambda: _arm_approval(cmdr), lambda a: a == [True],
                         APPROVE_YES, _flat)
        assert n == 117 and len(hits) == n, f"{n - len(hits)}/{n} NOT granted"

    def test_the_flat_hesitated_yes_still_opens(self, cmdr):
        n, hits = _sweep(cmdr, lambda: _arm_terminal(cmdr), bool, TERMINAL_YES, _flat)
        assert n == 78 and len(hits) == n, f"{n - len(hits)}/{n} NOT opened"

    def test_the_flat_hesitated_yes_still_runs(self, cmdr):
        n, hits = _sweep(cmdr, lambda: _arm_strict(cmdr), bool, STRICT_YES, _flat)
        assert n == 78 and len(hits) == n, f"{n - len(hits)}/{n} NOT run"

    def test_a_rising_hesitated_yes_earns_the_send_lanes_one_re_ask(self, cmdr, tmp_path):
        """Not silence: "yes, uh?" is a NEAR yes, and he is asked again by
        name, exactly as the bare "yes?" is."""
        _arm_send(cmdr, tmp_path)
        res = _through(cmdr, "yes, uh?")
        assert res is not None and res.status == "Confirm?"
        assert cmdr._pending_send is not None and cmdr._pending_send.reasked


# ===================================================================
# 2. THE CARD: a hesitation is not an answer to a flashcard
# ===================================================================
def _pauses(w: str) -> list:
    """Nine shapes of a bare filled pause: 13 x 9 = 117."""
    W = w.capitalize()
    return [w, f"{W}.", f"{W}...", f"{w},", f"{W}?", f"{w}!",
            f"{w}, {w}", f"{W} {w}.", f"{W}... {W}."]


NEW_QUIZ = ["quiz me on chemistry", "review my flashcards", "test me on biology"]


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


def _card_untouched(store, cid) -> bool:
    row = store.get(cid)
    return row["seen"] == 0 and row["box"] == 1


class TestTheCardStandsOnAHesitation:
    def test_117_bare_pauses_grade_spend_and_write_nothing(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        session, cid = fresh(), cards[0]["id"]
        bad, n = [], 0
        for w in FILLERS:
            for said in _pauses(w):
                n += 1
                _through(cmdr, said)
                if not (cmdr._pending_quiz is session and session.results == []
                        and _card_untouched(store, cid) and cmdr.question_open()):
                    bad.append(said)
        assert n == 117 and bad == [], f"{len(bad)}/{n} touched the card: {bad[:8]}"
        # ...and the card is still his to answer: graded when he does
        _through(cmdr, "v over i")
        assert session.results == [(cid, True)]

    def test_39_hesitated_new_quiz_sentences_never_reach_the_grader(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        cid = cards[0]["id"]
        bad, n = [], 0
        for w in FILLERS:
            for sentence in NEW_QUIZ:
                n += 1
                session = fresh()
                said = f"{w.capitalize()}, {sentence}"
                _through(cmdr, said)
                if not (session.results == [] and _card_untouched(store, cid)
                        and cmdr._pending_quiz is not session):
                    bad.append(said)
        assert n == 39 and bad == [], f"{len(bad)}/{n} graded or kept the card: {bad[:8]}"

    @pytest.mark.parametrize("sentence", NEW_QUIZ)
    def test_bare_the_control_drops_the_card_ungraded(self, quiz_cmdr, sentence):
        cmdr, store, cards, fresh = quiz_cmdr
        session, cid = fresh(), cards[0]["id"]
        _through(cmdr, sentence)
        assert session.results == [] and _card_untouched(store, cid)
        assert cmdr._pending_quiz is not session

    def test_a_hesitated_skip_and_stop_still_work_over_the_standing_card(self, quiz_cmdr):
        cmdr, store, cards, fresh = quiz_cmdr
        session, cid = fresh(), cards[0]["id"]
        _through(cmdr, "Hmm.")
        _through(cmdr, "uh, skip it")
        assert session.results == [(cid, None)] and _card_untouched(store, cid)
        session = fresh()
        _through(cmdr, "Er...")
        res = _through(cmdr, "um, stop the quiz")
        assert res is not None and res.status == "Quiz stopped"
        assert cmdr._pending_quiz is None and session.results == []


# ===================================================================
# 3. THE TWO RUNGS THE CENSUS COULD NOT SEE
# ===================================================================
def _ring(cmdr):
    calls = []
    cmdr.services.timekeeper = SimpleNamespace(
        ringing=SimpleNamespace(id="a1", label="Get up", kind="alarm"),
        stop_ringing=lambda why: calls.append(("stop", why)),
        snooze=lambda n: calls.append(("snooze", int(n))))
    return calls


STOP_WORDS = ["stop", "I'm up", "shut it off", "okay", "that's enough"]
SNOOZE_WORDS = ["snooze", "five more minutes"]


def _steers(p: str, w: str) -> list:
    W = w.capitalize()
    return [f"{W}, {p}.", f"{w} {p}", f"{p}, {w}", f"{W}, jarvis, {p}"]


class TestTheRingingAlarmHearsAHesitatedStop:
    def test_260_hesitated_stops_silence_it(self, cmdr):
        lost, n = [], 0
        for w in FILLERS:
            for p in STOP_WORDS:
                for said in _steers(p, w):
                    n += 1
                    calls = _ring(cmdr)
                    _through(cmdr, said)
                    if calls != [("stop", "dismiss")]:
                        lost.append(said)
        assert n == 260 and lost == [], f"{len(lost)}/{n} kept ringing: {lost[:8]}"

    def test_104_hesitated_snoozes_snooze_it(self, cmdr):
        lost, n = [], 0
        for w in FILLERS:
            for p in SNOOZE_WORDS:
                for said in _steers(p, w):
                    n += 1
                    calls = _ring(cmdr)
                    _through(cmdr, said)
                    if not (len(calls) == 1 and calls[0][0] == "snooze"):
                        lost.append(said)
        assert n == 104 and lost == [], f"{len(lost)}/{n} not snoozed: {lost[:8]}"

    @pytest.mark.parametrize("said", ["uh", "Uh...", "um, um", "Hmm?",
                                      "uh, what's the weather"])
    def test_a_bare_pause_or_a_new_subject_still_routes_on(self, cmdr, said):
        calls = _ring(cmdr)
        _through(cmdr, said)
        assert calls == [], said


class _Capture:
    """A stand-in for lecture.LectureNotes: what _handle_lecture uses."""

    def __init__(self):
        self.lines, self.closed = [], False

    def add(self, text: str) -> int:
        self.lines.append(text)
        return len(self.lines)

    def close(self) -> str:
        self.closed = True
        return "Notes closed, sir: 0 for biosensors."


def _notes(cmdr) -> _Capture:
    cap = _Capture()
    cmdr.lecture_course = "biosensors"
    cmdr._lecture = cap
    return cap


class TestLectureNotesHearAHesitatedEnd:
    @pytest.mark.parametrize("w", FILLERS)
    def test_a_hesitated_end_closes_the_notes(self, cmdr, w):
        for said in (f"{w.capitalize()}, end notes.", f"{w} end notes",
                     f"end notes, {w}", f"{w}, jarvis, end notes"):
            cap = _notes(cmdr)
            res = _through(cmdr, said)
            assert cap.closed and cap.lines == [], said
            assert cmdr.lecture_course is None, said
            assert res is not None and res.status == "Notes closed", said

    @pytest.mark.parametrize("w", FILLERS)
    def test_a_hesitated_line_is_filed_exactly_as_he_said_it(self, cmdr, w):
        """The end TEST strips; the filed LINE never does -- an "um" in a
        lecture note is dictation."""
        cap = _notes(cmdr)
        said = f"{w.capitalize()}, the mitochondria is the powerhouse of the cell."
        res = _through(cmdr, said)
        assert cap.lines == [said] and not cap.closed, said
        assert res is not None and res.status.startswith("Noting")

    def test_bare_the_control(self, cmdr):
        cap = _notes(cmdr)
        _through(cmdr, "end notes")
        assert cap.closed and cmdr.lecture_course is None

    def test_a_flag_without_a_capture_is_cleared_and_the_words_route_on(self, cmdr):
        """The ghost-mode recovery moved from _handle_lecture up into
        _handle_inner (so the lecture rung is no longer a dispatcher to
        the census): same behaviour, for every source."""
        cmdr.lecture_course = "biosensors"
        cmdr._lecture = None
        res = cmdr.handle("what time is it", "typed")
        assert cmdr.lecture_course is None
        assert res is not None and res.status != "Noting: biosensors (1)"


# ===================================================================
# 4. THE SMALLER ONES
# ===================================================================
LEAVE_ANSWERS = [("fifteen", 15), ("about ten", 10), ("twenty", 20), ("15", 15),
                 ("roughly five", 5)]


class TestTheLeaveTimeAnswer:
    """"How long do you need to get to X, sir?" -- the number, hesitated.
    The list's own "uh"/"um" entries wanted "uh " with a space; Whisper
    writes "Uh, fifteen." and only the unit-word shapes survived."""

    def _arm(self, cmdr):
        cmdr.services.leavetime = SimpleNamespace(learn=lambda key, minutes: int(minutes))
        cmdr._pending_leave = ("Example Hall", "Example Hall", _t.monotonic())

    def test_195_hesitated_durations_are_learned(self, cmdr):
        lost, n = [], 0
        for w in FILLERS:
            for answer, minutes in LEAVE_ANSWERS:
                for said in (f"{w.capitalize()}, {answer}.", f"{w} {answer}",
                             f"{answer}, {w}"):
                    n += 1
                    self._arm(cmdr)
                    res = cmdr._try_leave_answer(said)
                    if not (res is not None and res.status == f"Walk: Example Hall {minutes} min"
                            and cmdr._pending_leave is None):
                        lost.append(said)
        assert n == 195 and lost == [], f"{len(lost)}/{n} not learned: {lost[:8]}"

    def test_the_second_list_no_longer_carries_the_one_vocabulary(self):
        from jarvis import leavetime
        assert not set(leavetime._ANSWER_FILLER) & FILLER_WORDS, \
            "two lists drift; the filled pause comes off at the rung"

    def test_a_command_inside_the_window_is_still_that_command(self, cmdr):
        self._arm(cmdr)
        assert cmdr._try_leave_answer("uh, set a timer for five minutes") is None
        assert cmdr._pending_leave is not None


class TestTheUndoDashForm:
    @pytest.mark.parametrize("w", FILLERS)
    def test_a_filler_then_a_lone_dash_is_still_an_undo(self, w):
        from jarvis.commander import undo_kind
        W = w.capitalize()
        for dash in ("-", "—", "–", "..."):
            assert undo_kind(f"{W} {dash} scratch that"), (w, dash)
            assert undo_kind(f"{W} {dash} belay that last order"), (w, dash)

    def test_bare_the_control(self):
        from jarvis.commander import undo_kind
        assert undo_kind("scratch that") and not undo_kind("uh - scratch the surface")


class TestTheWordCountsCountWordsHeSaid:
    """Three guards counted fillers as words: a short attempt with a
    throat-clear in front of it looked like a sentence and was treated as
    a change of subject where the bare form earned the one re-ask."""

    def test_the_send_aside_re_ask(self, cmdr, tmp_path):
        _arm_send(cmdr, tmp_path)
        draft, cmdr._pending_send = cmdr._pending_send, None
        cmdr._send_aside = draft
        res = cmdr._send_aside_answer("uh, um, ah, that one there please", "no route")
        assert res is not None and res.status == "Confirm?"
        assert cmdr._pending_send is draft and draft.reasked

    def test_the_send_ask_person_attempt(self, cmdr):
        ask = SendAsk(kind="person", said_file="lab report", who="Heather", hint="",
                      source="voice", made_at=_t.monotonic(),
                      candidates=["Heather Smith", "Heather Jones"])
        cmdr._pending_sendask = ask
        res = cmdr._try_sendask_answer("Ah, ah, ah, Heather", "voice")
        assert res is not None and res.handled, "dropped as a new subject"
        assert cmdr._pending_sendask is ask and ask.reasked

    def test_the_strict_read_backs_vague_leg(self, cmdr):
        _arm_strict(cmdr)
        res = cmdr._try_destructive_confirm("uh, um, ah, I think so, yes", "voice")
        assert res is not None and res.status == "Confirm?"
        assert cmdr._pending_destructive is not None


# ===================================================================
# 5. WHAT ROUND 3 DID NOT CHANGE -- pinned as it stands, so the two open
#    questions are numbers in the tree and not only in a report
# ===================================================================
class TestTheTwoThingsLeftStanding:
    """Neither is a regression -- both behave exactly as mainline 13162d2
    did -- and neither is in the four items this round was asked to fix.
    They are pinned so that changing either is a DECISION somebody makes,
    not a drift. Both wait on him."""

    def test_the_send_ask_question_still_spends_its_re_ask_on_a_bare_pause(self, cmdr):
        """OPEN. "Which Heather, sir?" -- a bare "Hmm." is taken as an
        answer, spends the one re-ask, and the second hesitation drops the
        question out loud. The flashcard was given the opposite rule this
        round (the card STANDS); this rung was not, because it was not in
        the four items. Measured: 117/117 spend it, 117/117 drop it once
        spent. The adversary called it minor; it is his call, not mine."""
        spent, dropped, n = 0, 0, 0
        for w in FILLERS:
            for said in _pauses(w):
                n += 1
                ask = SendAsk(kind="person", said_file="lab report", who="Heather",
                              hint="", source="voice", made_at=_t.monotonic(),
                              candidates=["Heather Smith", "Heather Jones"])
                cmdr._pending_sendask = ask
                _through(cmdr, said)
                if getattr(ask, "reasked", False):
                    spent += 1
                ask2 = SendAsk(kind="person", said_file="lab report", who="Heather",
                               hint="", source="voice", made_at=_t.monotonic(),
                               candidates=["Heather Smith", "Heather Jones"])
                ask2.reasked = True
                cmdr._pending_sendask = ask2
                _through(cmdr, said)
                if cmdr._pending_sendask is None:
                    dropped += 1
        assert (n, spent, dropped) == (117, 117, 117)

    def test_a_RISING_hesitated_stop_leaves_the_alarm_ringing(self, cmdr):
        """OPEN, and the one place the two defaults pull apart. The "?"
        bar says a rising hesitated yes never acts; the alarm rung says
        stop is the safe direction. "Uh, stop?" now strips to "stop?",
        the ring grammars rstrip only ".!", and the alarm goes on
        ringing: 26/26. Mainline lost these forms too, so nothing was
        taken away -- but a man woken at six who says "uh, stop?" is not
        asking a question, and one character in _try_ringing's rstrip
        would change it."""
        lost, n = [], 0
        for w in FILLERS:
            for said in (f"{w.capitalize()}, stop?", f"stop, {w}?"):
                n += 1
                calls = _ring(cmdr)
                _through(cmdr, said)
                if calls == []:
                    lost.append(said)
        assert n == 26 and len(lost) == 26
        assert strip_fillers("stop, uh?") == "stop?"
