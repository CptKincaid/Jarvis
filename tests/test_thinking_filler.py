"""The slow-lookup filler must not answer a question Jarvis just asked.

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


def _app(pending):
    a = object.__new__(JarvisApp)
    a._turn_busy = threading.Event()
    a._turn_busy.set()
    a._uncertain_lock = threading.Lock()
    a._pending_uncertain = dict(pending)
    a._thinking_i = 0
    a.said = []
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
    app.commander = SimpleNamespace(resolve_uncertain=lambda text, yes: result)
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
    def resolve(text, yes):
        app._turn_finished()                      # the brain answered inline
        return SimpleNamespace(handled=True, status="Thinking", done=False,
                               reply=None, speak=False)
    app.commander = SimpleNamespace(resolve_uncertain=resolve)
    app.uncertain_answer("abc123", True, source="voice")
    assert not app._turn_busy.is_set(), "re-opened a turn the brain had closed"
    assert app._turn_timer is None and app._turn_watchdog is None
