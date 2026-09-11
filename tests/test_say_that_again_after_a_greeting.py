""""Say that again" repeats what he just heard, whoever started the turn.

HIS BUG, 2026-09-11: "said welcome back and then when i asked say that
again he said that was a while ago ask me again."

THE MECHANISM. ``_h_repeat`` bounded the repeat with ``_owner_said_at``,
which ``Commander.handle`` stamps at the end of HIS OWN command turn and
nowhere else. A PROACTIVE line -- the arrival greeting, a reminder, a
soundbar line -- never touches it, while ``tts.last_text`` holds exactly
that line. So after three hours out he was greeted, asked for it again,
and was told it was a while ago by a clock that was timing something
else entirely: the two halves tracked different events.

The bound is now the age of THE LINE, which the TTS stamps when it queues
it. Every clock here is injected.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from jarvis import commander as C


class _Tts:
    """Just enough TTS: what was said, and when."""

    def __init__(self, text="", at=None):
        self.last_text = text
        self.last_text_at = at if at is not None else time.monotonic()
        self.repeats = 0

    def repeat_last(self):
        self.repeats += 1


def _cmdr(tts, owner_said_at=None):
    c = SimpleNamespace(_svc={"tts": tts}.get)
    if owner_said_at is not None:
        c._owner_said_at = owner_said_at
    return c


GREETING = "Welcome back, sir."


def test_the_greeting_he_just_heard_is_repeated():
    """The bug, exactly: a PROACTIVE line seconds ago, and his last
    answered turn hours back."""
    tts = _Tts(GREETING, at=time.monotonic() - 2.0)
    res = C._h_repeat(_cmdr(tts, owner_said_at=time.monotonic() - 10_000.0),
                      "say that again", None)
    assert res.reply == GREETING
    assert res.status == "Repeating"


@pytest.mark.parametrize("age", [0.0, 1.0, 60.0, C.REPEAT_MAX_AGE_S - 1.0])
def test_a_line_inside_the_window_is_repeated(age):
    tts = _Tts(GREETING, at=time.monotonic() - age)
    res = C._h_repeat(_cmdr(tts, owner_said_at=time.monotonic() - 10_000.0),
                      "say that again", None)
    assert res.reply == GREETING, age


@pytest.mark.parametrize("age", [C.REPEAT_MAX_AGE_S + 1.0, 10_000.0])
def test_a_genuinely_old_line_is_still_refused(age):
    """The bound is not removed, only measured against the right thing:
    a line from an hour ago is still 'ask me again'."""
    tts = _Tts(GREETING, at=time.monotonic() - age)
    res = C._h_repeat(_cmdr(tts, owner_said_at=time.monotonic()), "say that again", None)
    assert res.reply == C.REPEAT_STALE_LINE, age


def test_nothing_said_yet_is_unchanged():
    res = C._h_repeat(_cmdr(_Tts("", at=0.0)), "say that again", None)
    assert res.reply == "I haven't said anything yet, sir."


def test_a_tts_that_does_not_stamp_falls_back_to_the_old_bound():
    """A stand-in TTS without last_text_at (tests, an older object) keeps
    the previous behaviour rather than becoming unbounded."""
    tts = SimpleNamespace(last_text=GREETING, repeat_last=lambda: None)
    fresh = C._h_repeat(_cmdr(tts, owner_said_at=time.monotonic()), "say that again", None)
    assert fresh.reply == GREETING
    stale = C._h_repeat(_cmdr(tts, owner_said_at=time.monotonic() - 10_000.0),
                        "say that again", None)
    assert stale.reply == C.REPEAT_STALE_LINE


def test_the_repeat_does_not_refresh_its_own_age():
    """The rule the original comment states, and it must survive: asking
    twice must not keep a line alive for ever."""
    at = time.monotonic() - (C.REPEAT_MAX_AGE_S - 2.0)
    tts = _Tts(GREETING, at=at)
    C._h_repeat(_cmdr(tts), "say that again", None)
    assert tts.last_text_at == at
