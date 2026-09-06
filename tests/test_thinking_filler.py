"""The slow-lookup filler must not land on top of -- or after -- the answer.

Second half of this file, added 2026-08-31 from Hunter's 155-feature voice
session. Turns are ~8x faster than when the filler was tuned (median wait
10.68 s -> 1.30 s), so a line scheduled at a fixed 4.5 s stopped arriving
BEFORE the answer and started arriving on top of it. From jarvis.log:

  21:36:36.125 handle 'Is Nightfall up?'
  21:36:37.322 speaking "Knightfall is up on demon-bot, sir..."   <- the answer
  21:36:41.696 speech complete
  21:36:41.696 speaking "Checking right now, sir. One moment."    <- #154
  21:37:36.126 turn watchdog fired after 60s

  21:22:36.898 speaking "I am beginning the creation of your markdown file"
  21:22:40.863 speaking "One moment, sir - I'm checking."         <- #89, +3.97 s

  21:02:13.055 handle 'add hello to 4 30 p.m. tomorrow'
  21:02:17.554 speaking "Checking right now, sir. One moment."    <- #144, +4.50 s
  21:02:18.205 chat reply ready (+5.15 s) -- queued behind the filler
  21:02:19.970 speaking "Added hello, Tuesday at 4:30 PM"         <- 1.77 s late

The Oracle turn is the one that proves _turn_busy is the wrong guard: it had
spoken its answer and still held the turn open until the 60 s watchdog.

The first half of this file is the original 2026-08-29 defect:
the slow-lookup filler must not answer a question Jarvis just asked.

"Was that for me?" leaves the turn open (done=False) so a yes/no can arrive,
which left the THINKING_DELAY_S timer running. Live at 21:08 on 2026-08-29:

    08.865 Uncertain intent -> "Was that for me?"
    12.365 speaking "Looking into it now, sir."     <- 3.5 s later, unanswered
    12.570 status: Discarded

The user heard their question answered with a promise to work on an utterance
that was then thrown away. By 12.365 _ask_uncertain is also recording the
spoken reply, and the mic arbiter is a depth counter rather than a mutex, so
Jarvis was talking into his own yes/no window.
"""
import threading

from jarvis.app import JarvisApp


class Clock:
    """A hand-cranked monotonic clock for the ledger (as in test_turnclock)."""

    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt
        return self.t


class FakeTTS:
    """Just the two properties _say_thinking reads, plus a queue that keeps
    order -- the real TTS queue is FIFO, which is the whole defect."""

    def __init__(self, speaking=False, pending=0):
        self.is_speaking = speaking
        self.pending = pending
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


def _app(pending, real_say=False):
    a = object.__new__(JarvisApp)
    a._turn_busy = threading.Event()
    a._turn_busy.set()
    a._uncertain_lock = threading.Lock()
    a._pending_uncertain = dict(pending)
    a._thinking_i = 0
    # The filler's own state (JarvisApp.__init__): a per-turn generation and
    # the "this turn has already spoken" latch the timer thread checks.
    a._filler_lock = threading.Lock()
    a._turn_seq = 1
    a._turn_answered = False
    a._turn_timer = None
    a.tts = FakeTTS()
    a.said = []
    if not real_say:
        a._say = a.said.append
    return a


def test_held_while_a_question_is_open():
    app = _app({"abc123": "what have you, and"})
    app._say_thinking()
    assert app.said == [], f"spoke while awaiting a yes/no: {app.said}"


def test_still_speaks_for_a_genuinely_slow_lookup():
    """The filler must keep working; this is a suppression, not a removal."""
    app = _app({})
    app._say_thinking()
    assert len(app.said) == 1 and app.said[0]


def test_silent_once_the_answer_has_landed():
    app = _app({})
    app._turn_busy.clear()
    app._say_thinking()
    assert app.said == []


def _answerable_app(result, monkeypatch):
    """An app mid-prompt, with the real turn machinery and a commander whose
    resolve_uncertain returns `result`."""
    from types import SimpleNamespace
    import jarvis.app as app_mod
    app = _app({"abc123": "what have you, and"})
    app._turn_timer = app._turn_watchdog = None
    app._thinking_delay_s, app._turn_timeout_s = 60.0, 120.0   # never fire here
    app._emit_result = lambda r: r
    app.commander = SimpleNamespace(resolve_uncertain=lambda text, yes, **kw: result)
    # via monkeypatch: a bare assignment here silenced the event bus for every
    # test that ran after this file (timekeeper, speak queue) -- it did.
    monkeypatch.setattr(app_mod.bus, "publish", lambda ev: None)
    return app


def test_answering_no_closes_the_turn(monkeypatch):
    """Discarded is done. Before this the turn stayed busy and every wake
    word for the next 60 s got "One moment -- still on the last one"."""
    from types import SimpleNamespace
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "talkback", True)
    app = _answerable_app(SimpleNamespace(handled=True, status="Discarded", done=True,
                                          reply=None, speak=False), monkeypatch)
    app.uncertain_answer("abc123", False, source="ui")
    assert not app._turn_busy.is_set(), "turn left open after a NO"
    assert app._turn_timer is None and app._turn_watchdog is None


def test_answering_yes_that_routes_to_the_brain_rearms_the_filler(monkeypatch):
    """The lookup the YES started deserves its own filler and watchdog; the
    brain callback closes them as for any other turn."""
    from types import SimpleNamespace
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "talkback", True)
    app = _answerable_app(SimpleNamespace(handled=True, status="Thinking", done=False,
                                          reply=None, speak=False), monkeypatch)
    try:
        app.uncertain_answer("abc123", True, source="voice")
        assert app._turn_busy.is_set()
        assert app._turn_timer is not None and app._turn_watchdog is not None
    finally:
        app._turn_cancel_timers()


def test_the_delay_clears_the_measured_tool_latency():
    """Tool-backed lookups landed right at 3.5 s, so the filler started and
    the real answer queued behind it."""
    import jarvis.app as app_mod
    assert app_mod.THINKING_DELAY_S >= 4.5


def test_a_yes_whose_lookup_closes_the_turn_synchronously_is_not_reopened(monkeypatch):
    """brain.chat's busy branch invokes _on_brain_tags -> _turn_finished
    INSIDE resolve_uncertain. Re-arming after it returned re-opened a turn
    nothing would close: every wake word refused until the 60 s watchdog."""
    from types import SimpleNamespace
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "talkback", True)
    app = _answerable_app(SimpleNamespace(handled=True, status="Thinking", done=False,
                                          reply=None, speak=False), monkeypatch)
    def resolve(text, yes, **kw):
        app._turn_finished()                      # the brain answered inline
        return SimpleNamespace(handled=True, status="Thinking", done=False,
                               reply=None, speak=False)
    app.commander = SimpleNamespace(resolve_uncertain=resolve)
    app.uncertain_answer("abc123", True, source="voice")
    assert not app._turn_busy.is_set(), "re-opened a turn the brain had closed"
    assert app._turn_timer is None and app._turn_watchdog is None


# ------------------------------------------------------------------ 2026-08-31
# The filler arriving AFTER the answer. All three of Hunter's reports are the
# same shape: something was already spoken for this turn, and the filler timer
# -- which only ever asked "is the turn still busy?" -- spoke anyway.

def test_no_filler_once_this_turn_has_already_spoken(monkeypatch):
    """#154, the Oracle box: "had the answer then said checking one moment
    sir". The Oracle route spoke its reply at 21:36:37.322 and left the turn
    open until the 60 s watchdog, so _turn_busy was STILL SET when the timer
    fired at 21:36:41 and the filler played on the next queue slot."""
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "talkback", True)
    app = _app({}, real_say=True)
    app._say("Knightfall is up on demon-bot, sir.")     # the answer, spoken
    assert app._turn_busy.is_set(), "the Oracle turn is still open -- that is the point"
    app._say_thinking()
    assert app.tts.spoken == ["Knightfall is up on demon-bot, sir."], \
        f"spoke a filler after the answer: {app.tts.spoken}"


def test_no_filler_while_a_burst_is_still_playing():
    """#89: the answer was mid-burst, so the filler could only land behind it.
    "One moment, sir" started at 21:22:40.863, the instant the reply finished."""
    app = _app({})
    app.tts.is_speaking = True
    app._say_thinking()
    assert app.said == [], f"queued a filler behind live speech: {app.said}"


def test_no_filler_while_anything_is_queued_ahead_of_it():
    """The TTS queue is FIFO: one line already waiting means this one is late."""
    app = _app({})
    app.tts.pending = 1
    app._say_thinking()
    assert app.said == [], f"queued a filler behind a pending line: {app.said}"


def test_a_timer_from_the_previous_turn_cannot_speak_into_this_one():
    """Timer.cancel() is a no-op once the callback has started, so a stale
    filler has to disqualify itself. Without the generation check it spoke
    into whatever turn happened to be open when it woke up."""
    app = _app({})
    stale = app._turn_seq
    app._turn_seq += 1                                  # a new turn opened
    app._say_thinking(stale)
    assert app.said == [], f"a stale timer spoke: {app.said}"
    app._say_thinking(app._turn_seq)                    # the live one still may
    assert len(app.said) == 1


def test_the_say_door_disarms_the_pending_filler(monkeypatch):
    """Routes used to cancel the timer one by one and the ones that forgot --
    the Oracle tool, a Claude session ack -- talked over themselves. _say is
    the single door every spoken line passes, so the disarm belongs there."""
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "talkback", True)
    app = _app({}, real_say=True)
    timer = threading.Timer(30.0, lambda: None)
    timer.daemon = True
    app._turn_timer = timer
    timer.start()
    try:
        app._say("The test project is set up, sir; ready when you are.")
        assert app._turn_timer is None, "the filler timer was left armed"
        assert not timer.is_alive() or timer.finished.is_set(), "timer not cancelled"
        assert app._turn_answered, "the answer was not latched"
    finally:
        timer.cancel()


def test_a_slow_turn_that_has_said_nothing_still_gets_its_filler():
    """This is a suppression, not a removal: the mail lookup at 14:33 took
    16.5 s and the filler at 5.59 s was the only sign of life."""
    app = _app({})
    app._say_thinking(app._turn_seq)
    assert len(app.said) == 1 and app.said[0]


# ----------------------------------------------------------- the delay scales

class _Ledger:
    def __init__(self, typical):
        self._typical = typical

    def typical_wait_s(self, min_samples=5):
        return self._typical


def _delay_app(typical):
    app = _app({})
    app._thinking_delay_s = 4.5
    app.turns = _Ledger(typical)
    return app


def test_the_delay_scales_past_a_fast_boxs_normal_turn():
    """#144: the calendar answer was READY at +5.15 s and the filler fired at
    +4.50 s, so the answer queued behind it and arrived 1.77 s late. The 105
    answered turns in turns.jsonl for 2026-08-31 have a median wait of 2.07 s
    and a p90 of 4.45 s, so the filler now arms at 6.2 s and that turn is
    answered before it ever speaks."""
    app = _delay_app(2.07)
    assert app._filler_delay_s() > 5.15, \
        "still fires before a calendar lookup that was about to answer"
    assert app._filler_delay_s() < 16.5, \
        "the 16.5 s mail lookup must still be acknowledged"


def test_the_delay_never_undercuts_the_hand_measured_floor():
    """A quiet spell of one-word replies must not make the filler MORE eager
    than the value tuned by hand on 2026-08-29."""
    import jarvis.app as app_mod
    app = _delay_app(0.20)
    assert app._filler_delay_s() == app_mod.THINKING_DELAY_S


def test_the_delay_is_capped_so_a_stuck_turn_is_not_pure_silence():
    import jarvis.app as app_mod
    app = _delay_app(30.0)
    assert app._filler_delay_s() == app_mod.FILLER_DELAY_MAX_S


def test_a_cold_ledger_falls_back_to_the_tuned_constant():
    """Fewer than a handful of answered turns is not a measurement."""
    app = _delay_app(None)
    assert app._filler_delay_s() == 4.5


def test_the_ledger_only_counts_turns_that_actually_answered():
    """A rejection or a watchdog abandon says nothing about how fast the box
    answers; letting them in would drag the median and re-arm the filler
    early on the next real turn."""
    from jarvis.turnclock import TurnLedger
    clock = Clock()
    led = TurnLedger(clock=clock, emit=lambda rec: None)
    assert led.typical_wait_s() is None
    for _ in range(6):
        led.mark("wake")
        led.mark("mic")
        clock.tick(40.0)                 # a long rejected clip: not a "wait"
        led.abandon("rejected:speaker")
    assert led.typical_wait_s() is None, "abandoned turns were counted as waits"


def test_the_ledger_reports_the_median_of_recent_waits():
    from jarvis.turnclock import TurnLedger
    clock = Clock()
    led = TurnLedger(clock=clock, emit=lambda rec: None)
    for wait in (1.2, 1.3, 1.4, 1.3, 24.6):       # one Claude session in the set
        led.mark("wake")
        led.mark("mic")
        speech_end = clock.t
        clock.tick(0.9)
        led.mark("speech_end", at=speech_end)
        led.mark("stop", stop="vad")
        clock.tick(wait - 0.9)
        led.mark("audio")
    # the median, not the mean: 6.16 s of mean would push the filler to its cap
    assert abs(led.typical_wait_s() - 1.3) < 0.01

