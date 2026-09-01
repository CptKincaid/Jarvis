"""Two things Hunter hit with a mouse and a keyboard on 2026-08-31.

GENERAL BUG, in his words:
    "if i press cancel button on the screen when he is waiting on a
     response he will say i didnt quite get that sir"

The "cancel button" is the mic button: while a capture is open it is drawn
as a filled disc with a STOP glyph (CommandBar._draw_mic), and it was wired
to `services.stop_recording` — stop AND transcribe — whichever way the mic
had been opened. So a press meant to abandon a turn sent whatever fraction
of a second of room noise had been captured to Whisper, and Jarvis answered
the salad out loud. From /tmp/vss_voice/jarvis.log, three times:

    20:46:37.689 Stopped: 1.2s audio          -> Stopped (manual)
    20:46:41.053 Transcribed: 'Thank you so much, so that you will see.'
                 (avg_logprob=-2.35)          -> "Say that again, sir?"
    20:46:56.477 Stopped: 0.3s audio          -> Stopped (manual)
    20:46:58.293 Transcribed: 'express iesekret node meta ...' (-5.84)
    21:26:06.893 Stopped: 1.0s audio          -> Stopped (manual)
    21:26:11.375 Transcribed: 'Kamen-427.st групп.com battalion-427 ...'
                 (-4.85)                      -> "Say that again, sir?"

Every one of those captures was opened by JARVIS (a follow-up window: the
"Listening…" line has no wake word above it), not by the button. That is
the distinction the fix rests on — a press that ends a capture the user
STARTED still means "done, take it" (push-to-talk depends on it).

#135 "typed-command history — arrow keys do nothing":
jarvis/history.py exists, persists, and documents prev()/next() as "the
command bar's Up/Down navigation". Nothing was ever bound to those keys and
Services had no field to reach the store through, so the consumer half of
the feature simply did not exist.

Tk-free throughout: the two methods under test are bound onto stand-ins,
exactly like the worker probes in tests/test_ui_assistant.py.
"""
from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("JARVIS_LOG_DIR", tempfile.mkdtemp(prefix="jarvis-ui-"))
os.environ.setdefault("JARVIS_ASSISTANT_CONFIG",
                      os.path.join(tempfile.mkdtemp(prefix="jarvis-ui-cfg-"),
                                   "assistant.json"))

from dataclasses import fields  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from jarvis.history import TypedHistory  # noqa: E402
from jarvis.ui.main_window import MainWindow, Services  # noqa: E402
from jarvis.ui.views import CommandBar  # noqa: E402


# ----------------------------------------------------------- the mic button
class _Bar:
    """MainWindow's recording controls with no Tk root behind them."""

    _toggle_recording = MainWindow._toggle_recording
    _start_recording = MainWindow._start_recording
    _stop_recording = MainWindow._stop_recording
    _cancel_recording = MainWindow._cancel_recording
    _ev_rec_stop = MainWindow._ev_rec_stop

    def __init__(self, services, recording=False):
        self.services = services
        self._recording = recording
        self._mic_available = True
        self._mic_opened_here = False
        self.statuses: list[tuple] = []
        # what _ev_rec_stop touches beyond the two flags
        self.command_bar = SimpleNamespace(set_mic_state=lambda s: None)
        self.tray = SimpleNamespace(update_state=lambda b: None)

    def set_status(self, text, kind="info"):
        self.statuses.append((text, kind))

    def _refresh_pill(self):
        pass


def _services():
    calls: list[str] = []
    svc = SimpleNamespace(
        start_recording=lambda: calls.append("start"),
        stop_recording=lambda: calls.append("stop"),
        cancel_recording=lambda: calls.append("cancel"),
    )
    return svc, calls


def test_the_stop_glyph_cancels_a_capture_jarvis_opened():
    """His bug, verbatim: the mic opened by a follow-up window ("he is
    waiting on a response"), the button pressed, and NOTHING transcribed."""
    svc, calls = _services()
    bar = _Bar(svc, recording=True)     # opened by a wake word / follow-up
    bar._toggle_recording()
    assert calls == ["cancel"], \
        "the cancel press still went to stop_recording -> Whisper -> a reply"


def test_a_capture_he_started_is_still_submitted_on_the_second_press():
    """The other half: click to record, click to send. Push-to-talk's
    key-up relies on the same path, so breaking this would cost every
    typed-free capture he makes with the button."""
    svc, calls = _services()
    bar = _Bar(svc)
    bar._toggle_recording()             # first press: start
    assert calls == ["start"] and bar._mic_opened_here is True
    bar._recording = True
    bar._toggle_recording()             # second press: submit
    assert calls == ["start", "stop"]


def test_ownership_does_not_leak_into_the_next_capture():
    """A wake word arriving after a button-started turn must not inherit
    "he pressed the button" — that would transcribe the room again."""
    svc, calls = _services()
    bar = _Bar(svc)
    bar._toggle_recording()             # he started one
    bar._recording = True
    bar._ev_rec_stop(SimpleNamespace(reason="manual"))
    assert bar._mic_opened_here is False
    bar._recording = True               # ...and now Jarvis opens the mic
    bar._toggle_recording()
    assert calls == ["start", "cancel"]


def test_a_failed_start_does_not_leave_the_button_owning_the_mic():
    def boom():
        raise OSError("no mic")

    bar = _Bar(SimpleNamespace(start_recording=boom))
    bar._toggle_recording()
    assert bar._mic_opened_here is False
    assert bar.statuses == [("Recording failed to start", "error")]


def test_cancel_recording_is_a_service_the_app_must_wire():
    names = {f.name for f in fields(Services)}
    assert "cancel_recording" in names
    # the default is the warning no-op, never a silent alias for stop
    assert Services().cancel_recording() is None


# -------------------------------------------------------- #135 arrow keys
class _Entry:
    """Just enough tk.Entry for the history handlers."""

    def __init__(self):
        self.value = ""
        self.fg = None
        self.cursor = None
        self.bindings: dict = {}

    def bind(self, seq, fn, add=False):
        self.bindings[seq] = fn

    def delete(self, first, last=None):
        self.value = ""

    def insert(self, index, text):
        self.value = text

    def configure(self, **kw):
        self.fg = kw.get("fg", self.fg)

    def icursor(self, index):
        self.cursor = index

    def get(self):
        return self.value


class _CmdBar:
    _bind_history = CommandBar._bind_history
    _history_prev = CommandBar._history_prev
    _history_next = CommandBar._history_next
    _history_step = CommandBar._history_step
    set_text = CommandBar.set_text

    def __init__(self, on_history=None):
        self.on_history = on_history
        self.entry = _Entry()
        self._showing_placeholder = True


def test_up_and_down_are_actually_bound_to_the_entry():
    """The whole of #135: the keys had no binding at all."""
    bar = _CmdBar()
    bar._bind_history(bar.entry)
    assert set(bar.entry.bindings) == {"<Up>", "<Down>"}
    assert bar.entry.bindings["<Up>"] == bar._history_prev
    assert bar.entry.bindings["<Down>"] == bar._history_next


def test_up_walks_back_through_the_real_history_store(tmp_path):
    """Against jarvis.history.TypedHistory itself, not a stub: the store
    was the half that already worked and must keep working."""
    hist = TypedHistory(tmp_path / "typed_history.jsonl")
    for cmd in ("set a timer for ten minutes", "what's on my calendar",
                "play my liked songs"):
        hist.add(cmd)

    def on_history(delta):
        return hist.prev() if delta < 0 else hist.next()

    bar = _CmdBar(on_history)
    assert bar._history_prev() == "break"      # Tk's own Up must not run too
    assert bar.entry.get() == "play my liked songs"
    bar._history_prev()
    assert bar.entry.get() == "what's on my calendar"
    bar._history_prev()
    assert bar.entry.get() == "set a timer for ten minutes"
    bar._history_prev()                        # oldest: stays put
    assert bar.entry.get() == "set a timer for ten minutes"
    bar._history_next()
    assert bar.entry.get() == "what's on my calendar"
    bar._history_next()
    assert bar.entry.get() == "play my liked songs"
    bar._history_next()                        # past the newest: empty field
    assert bar.entry.get() == ""
    assert bar._showing_placeholder is False   # a live caret, not grey hint


def test_recalled_text_is_typed_text_not_the_placeholder():
    """The field starts life showing the grey placeholder; a recalled
    command must clear that flag or _submit would refuse to send it."""
    bar = _CmdBar(lambda d: "standup")
    assert bar._showing_placeholder is True
    bar._history_prev()
    assert bar._showing_placeholder is False
    assert bar.entry.get() == "standup"
    assert bar.entry.cursor == "end"


def test_no_history_provider_leaves_the_arrow_keys_to_tk():
    """build_ui_services drops fields an older UI/app half does not
    declare, so the unwired case must not swallow the keystroke."""
    bar = _CmdBar(None)
    assert bar._history_prev() is None
    assert bar.entry.get() == ""


def test_a_broken_history_provider_does_not_kill_the_keystroke():
    def boom(_delta):
        raise RuntimeError("history file vanished")

    bar = _CmdBar(boom)
    assert bar._history_prev() == "break"
    assert bar.entry.get() == ""


def test_history_services_exist_for_the_app_to_wire():
    names = {f.name for f in fields(Services)}
    assert {"history_prev", "history_next"} <= names


# ------------------------------------------ the app half of both wires
# A real JarvisApp (hardware stubbed) so the two service names above are
# proved to reach recorder.abort and jarvis.history, not just to exist.
import jarvis.app as app_mod                            # noqa: E402
import pytest                                           # noqa: E402
from jarvis.events import RecordingStopped, bus         # noqa: E402
from tests.test_app_wiring import build, paths, seams    # noqa: E402,F401


@pytest.fixture
def app(build):                     # noqa: F811 - the shared factory
    return build()


def test_the_app_hands_the_ui_a_cancel_that_aborts(app):
    aborted: list = []
    app.recorder.abort = lambda: aborted.append(True)
    fn = app.ui_service_kwargs()["cancel_recording"]
    fn()
    for _ in range(200):            # it runs on a daemon thread
        if aborted:
            break
        time.sleep(0.01)
    assert aborted, "cancel_recording did not reach recorder.abort"


def test_an_aborted_capture_says_nothing_at_all(app, monkeypatch):
    """The point of the cancel: no transcript, no "Say that again, sir?",
    no nudge earcon. Anything else and the press is not a cancel."""
    spoken: list[str] = []
    monkeypatch.setattr(app, "_say", lambda text, **kw: spoken.append(text))
    nudged: list[str] = []
    monkeypatch.setattr(app, "_nudge", lambda reason: nudged.append(reason))
    transcribed: list = []
    app.transcriber.transcribe = lambda *a, **kw: transcribed.append(a)

    app._on_recording_stopped(RecordingStopped(reason="abort"))
    bus.drain()
    assert spoken == [] and nudged == [] and transcribed == []


def test_the_app_hands_the_ui_its_typed_history(app):
    kwargs = app.ui_service_kwargs()
    app.history.add("what's on my calendar")
    app.history.add("standup")
    assert kwargs["history_prev"]() == "standup"
    assert kwargs["history_prev"]() == "what's on my calendar"
    assert kwargs["history_next"]() == "standup"
    assert kwargs["history_next"]() is None


def test_build_ui_services_keeps_the_three_new_hooks(app):
    svc = app_mod.build_ui_services(Services, app.ui_service_kwargs())
    assert svc.cancel_recording is not None
    assert svc.history_prev is not None and svc.history_next is not None


def test_the_bar_gets_no_history_hook_when_the_app_half_is_missing():
    """build_ui_services drops fields an older app does not supply. With
    neither side wired the bar must be handed None, not a callback that
    would answer "" and wipe the line he is typing."""
    win = SimpleNamespace(services=Services())
    assert MainWindow._history_available(win) is False
    win.services = Services(history_prev=lambda: "standup")
    assert MainWindow._history_available(win) is True
