"""An unanswered "Was that for me?" must END the turn, not hold it.

THE 2026-09-06 00:29 SOFT LOCK, in his words: "jarvis is soft locked ...
stuck from saying is that for me and being in standby mode and wont come
out". The reproducer (tests/test_uncertain_hang_repro.py) showed, with a
hard bound, that the decode thread RETURNED on that path and ``_audio_busy``
was clear 0.04 s after the stop. What held the floor for the next minute was
the OTHER flag: the prompt returns ``CommandResult(done=False)`` so that a
yes/no can arrive, ``_ask_uncertain`` heard a reply it could not read
("Nudge, ROS, thank you" -> None) and returned WITHOUT closing anything,
and ``_turn_busy`` stayed set until the 60 s turn watchdog. Five wake words
at 0.85-0.98 were refused in that minute, every one with a line that named
the wrong flag: "still transcribing the previous clip".

    00:29:28.323  Uncertain intent (conf=0.50)          _turn_busy SET
    00:29:34.697  uncertain follow-up heard '...' -> None   returns, releases nothing
    00:29:43.581  hotword ignored: still transcribing the previous clip   (x5)
    00:30:28.323  turn watchdog fired after 60s; releasing the wake word

Same shape at 12:25 and 21:53 (reply refused as not the enrolled speaker).
The two prompts that resolved to a yes or a no (14:56, 14:57) were fine,
because ``uncertain_answer`` closes the turn.

THE CONTRACT PINNED HERE. When the spoken window ends without an answer --
whichever way: the speaker filter refused it, the words were not a yes or a
no, nothing was captured, there is no microphone, an enrolment owns it --

  * the turn is released at once (``_turn_busy`` clear, both timers off),
  * the card STAYS UP for a click, and no UncertainResolved is published,
  * the log says the question expired and why,
  * the very next wake word opens the microphone,
  * and ``_audio_busy`` was clear well inside the 5 s follow-up window.

And while the window IS open, a wake word is refused with the true reason
("waiting on your answer"), never "still transcribing" -- that line is kept
for the audio flag alone. An expiry that lands after a newer turn has taken
the floor leaves that turn alone.

Everything is synthetic: clips are ``np.zeros``, transcripts are strings
written in the reproducer, no microphone or recording is opened, no Tk is
attached. Every wait is bounded.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import jarvis.app as app_mod
from jarvis import voicerun
from jarvis.events import Status, UncertainResolved, bus
# Imported, not copied: pytest discovers fixture objects in a module's
# globals, so `make` and `lines` are the reproducer's own, and `build` /
# `paths` / `seams` the wiring suite's.
from tests.test_app_wiring import Sink, build, paths, seams  # noqa: F401
from tests.test_uncertain_hang_repro import (  # noqa: F401
    JOIN_S, REPLY_NONE, VARIANTS, _drive, _thread_named, _wait_clear, lines,
    make)

WINDOW_S = float(app_mod.JarvisApp.UNCERTAIN_LISTEN_S)   # "within the follow-up window"
SETTLE_S = 8.0                                            # any other bounded wait
WAKE_S = 2.0                                              # a wake word opening the mic


# ------------------------------------------------------------------ helpers
def _join_ask():
    ask = _thread_named("uncertain-ask")
    if ask is not None:
        ask.join(JOIN_S)
    return ask


def _wait_until(pred, timeout=SETTLE_S) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return bool(pred())


def _wake_opens_the_mic(app, score=0.896) -> bool:
    """A wake word at the 00:29:43 score; True if the recorder was started."""
    before = app.recorder.started
    app._on_hotword(score)
    return _wait_until(lambda: app.recorder.started == before + 1, WAKE_S)


def _refusals(lines, start) -> list[str]:
    return [ln for ln in lines.lines[start:] if "hotword ignored" in ln]


# ------------------------------------------------- every way to go unanswered
# name -> (make() kwargs, extra preparation after make())
def _prep_empty(app, monkeypatch):
    # The 5 s window captured nothing at all.
    monkeypatch.setattr(app.recorder, "record_fixed",
                        lambda seconds: np.zeros(0, dtype=np.float32))


def _prep_no_mic(app, monkeypatch):
    monkeypatch.setattr(app_mod.MACHINE, "has_mic", False)


def _prep_enrolment(app, monkeypatch):
    # A voice enrolment owns the microphone: the ask must not open it.
    monkeypatch.setattr(voicerun, "runs_live", lambda services: ("voice",))


UNANSWERED = {
    "ignored": (dict(reply=REPLY_NONE, reply_match=False), None),   # 12:25, 21:53
    "none": (dict(reply=REPLY_NONE, reply_match=True), None),       # 00:29
    "empty": (dict(), _prep_empty),
    "no-mic": (dict(), _prep_no_mic),
    "enrolment-live": (dict(), _prep_enrolment),
}


@pytest.mark.parametrize("exit_", sorted(UNANSWERED))
def test_an_unanswered_prompt_ends_the_turn_and_keeps_the_card(
        make, lines, monkeypatch, exit_):
    kwargs, prep = UNANSWERED[exit_]
    app = make(**kwargs)
    if prep is not None:
        prep(app, monkeypatch)
    resolved = Sink(UncertainResolved)
    try:
        t = _drive(app, speculative=True)
        took = _wait_clear(app, t, timeout=WINDOW_S)
        assert took is not None, "_audio_busy did not clear inside the follow-up window"
        ask = _join_ask()
        assert ask is None or not ask.is_alive(), "the ask thread never returned"
        # THE FIX: the unanswered question releases the turn by itself.
        assert _wait_until(lambda: not app._turn_busy.is_set()), (
            "the unanswered prompt still holds _turn_busy -- every wake word "
            "is refused until the 60 s turn watchdog")
        assert app._turn_watchdog is None and app._turn_timer is None
        joined = "\n".join(lines.lines)
        assert "uncertain question expired" in joined, joined
        # ...and the card stays up for him to click: nothing resolved it.
        assert app._uncertain_open() == 1, "the card must stay up for a click"
        bus.drain()
        assert resolved.of(UncertainResolved) == []
        if exit_ != "no-mic":
            before = len(lines.lines)
            assert _wake_opens_the_mic(app), (
                "the wake word must open the microphone once the question expired")
            assert _refusals(lines, before) == []
        # A click on the card that stayed up still answers it, and closes
        # cleanly -- the expiry took nothing the click needs.
        (rid,) = list(app._pending_uncertain)
        app.uncertain_answer(rid, False, source="ui")
        assert app._uncertain_open() == 0
        bus.drain()
        assert [e.yes for e in resolved.of(UncertainResolved)] == [False]
        assert not app._turn_busy.is_set()
    finally:
        resolved.close()


# --------------------------------------- the three resolutions, both flags
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_resolution_frees_both_flags_inside_the_window(make, lines, variant):
    """bool / ignored speaker / None: _audio_busy clear within the 5 s
    window on all of them, and the turn released on every one that does
    not hand the utterance to the router (a YES routes it, and that turn
    then belongs to the router's own reply, not to the prompt)."""
    reply, match = VARIANTS[variant]
    app = make(reply=reply, reply_match=match)
    t = _drive(app, speculative=True)
    took = _wait_clear(app, t, timeout=WINDOW_S)
    assert took is not None, "_audio_busy held past the follow-up window"
    _join_ask()
    assert app._uncertain_open() == (0 if variant.startswith("bool") else 1)
    if variant == "bool-yes":
        return
    assert _wait_until(lambda: not app._turn_busy.is_set()), variant
    assert _wake_opens_the_mic(app), variant


# ------------------------------------------ the refusal names the real flag
def test_a_wake_during_the_window_is_refused_for_the_true_reason_then_accepted(
        make, lines):
    """Freeze the ask mid-window (record_fixed held on an Event) and knock."""
    app = make()                                    # the 00:29 reply -> None
    hold = threading.Event()
    app.recorder.block = hold
    statuses = Sink(Status)
    try:
        t = _drive(app, speculative=True)
        assert _wait_clear(app, t, timeout=WINDOW_S) is not None
        assert _wait_until(lambda: bool(app.recorder.fixed_calls)), (
            "the ask thread never reached record_fixed")
        assert app._turn_busy.is_set() and not app._audio_busy.is_set()
        before = len(lines.lines)
        app._on_hotword(0.896)                      # 00:29:43.581
        refused = _refusals(lines, before)
        assert refused, "a wake word during the open window must still be refused"
        assert all("waiting on your answer" in ln for ln in refused), refused
        assert not any("still transcribing" in ln for ln in refused), refused
        bus.drain()
        assert any("waiting on your answer" in e.text.lower()
                   for e in statuses.of(Status)), [e.text for e in statuses.of(Status)]
        assert app.recorder.started == 0
    finally:
        hold.set()
        statuses.close()
    _join_ask()
    assert _wait_until(lambda: not app._turn_busy.is_set())
    assert any("uncertain question expired" in ln for ln in lines.lines)
    before = len(lines.lines)
    assert _wake_opens_the_mic(app)
    assert _refusals(lines, before) == []


def test_a_refusal_for_the_audio_flag_keeps_its_own_line(make, lines):
    """"still transcribing the previous clip" stays -- for the flag it is
    true of. The audio watchdog and its instrument are untouched."""
    app = make()
    app._audio_started()
    try:
        before = len(lines.lines)
        app._on_hotword(0.9)
        refused = _refusals(lines, before)
        assert refused and all("still transcribing the previous clip" in ln
                               for ln in refused), refused
        assert app.recorder.started == 0
    finally:
        app._audio_finished()


# ------------------------------------------- an expiry owns only its turn
def test_an_expiry_never_closes_a_newer_turn(make, lines):
    """The window is still open when a newer turn takes the floor (its
    sequence number moves on). The stale expiry must leave it alone --
    closing a turn it did not open is the clobber _dispatch already
    guards against for the socket sources."""
    app = make()
    hold = threading.Event()
    app.recorder.block = hold
    try:
        t = _drive(app, speculative=True)
        assert _wait_clear(app, t, timeout=WINDOW_S) is not None
        assert _wait_until(lambda: bool(app.recorder.fixed_calls))
        app._turn_start()                           # a newer turn, card left up
        assert app._turn_busy.is_set()
    finally:
        hold.set()
    _join_ask()
    try:
        assert app._turn_busy.is_set(), "the stale expiry closed a turn it did not open"
        assert any("a newer turn holds the floor" in ln for ln in lines.lines), \
            [ln for ln in lines.lines if "uncertain" in ln]
    finally:
        app._turn_finished()


def test_the_expiry_helper_degrades_on_a_bare_stand_in():
    """Several suites bind _ask_uncertain on an object.__new__ stand-in
    that owns no turn state at all; the release must not raise there."""
    a = object.__new__(app_mod.JarvisApp)
    assert app_mod.JarvisApp._uncertain_unanswered(a, "rid", None, "a test") is False
