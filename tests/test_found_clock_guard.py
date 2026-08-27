"""Regression tests for a defect in jarvis.brain.guard_clock_claims().

DEFECT (jarvis/brain.py, guard_clock_claims + _CLOCK_RX):

The guard is meant to drop clock readings the model cannot know. It does
so by extracting every ``hh:mm`` token from the reply and requiring each
one to appear *verbatim* in the context/tool text. That token-equality
test has two consequences the guard did not intend:

1.  A reply that is CORRECT and fully grounded in a tool result is
    destroyed whenever the model reformats or derives the time — a 24-hour
    rendering of the very value get_time returned ("19:14" for "7:14 pm"),
    or honest arithmetic on it ("two hours from now it will be 9:14 pm").
    Neither token appears in the tool text, so every sentence is dropped
    and the whole reply is replaced by NO_CLOCK_LINE — a statement that
    Jarvis has no clock, made immediately after get_time returned one.
    That is a false statement produced by the guard, not by the model.

2.  The guard disables itself entirely when the context contains the
    string "Current time" — which ContextEngine.format_for_prompt always
    emits. So in production the guard is a no-op and an invented reading
    passes through untouched; it only ever fires in the artificial
    no-clock contexts used by the eval harness.

Observed through the real production path (JarvisBrain._chat_sync with the
reference tool registry), 3/3 and 4/4 respectively:

    "Wie spät ist es und wie ist das Wetter?"
      model raw : 'Es ist 19:14 Uhr, sir, und es ist leicht bewölkt bei 72 Grad.'
      spoken    : "I'm afraid I haven't a clock in front of me just now, sir."

    "If I start a two hour build right now, what time will it finish?"
      spoken    : "I'm afraid I haven't a clock in front of me just now, sir."
      (get_time was called and returned successfully in every sample)

FIXED 2026-08-26 (brain work item, finding H11):

  * the "Current time" short-circuit is gone, so the guard runs on the
    live path (that is what test 3 below pins);
  * readings are compared as TIMES, not as strings (_clock_minutes), so
    "19:14" matches a tool's "7:14 pm" and a bare "10:30" matches either
    half of the day;
  * only a sentence that asserts the present time (_NOW_CLAIM_RX) has to
    match a known reading. A derived time ("your build finishes at 9:14
    pm") survives whenever a real clock reading is available at all;
  * with no clock available anywhere, any hh:mm is still an invention and
    still goes, and NO_CLOCK_LINE is then true. When a clock IS available
    and the model contradicts it, the reply becomes UNSURE_CLOCK_LINE
    instead — claiming to have no clock would be a second false statement.

These tests were the strict-xfail record of the defect; they now pass.
"""
from jarvis.brain import (NO_CLOCK_LINE, UNSURE_CLOCK_LINE,
                          guard_clock_claims)

# What get_time actually handed back, as it appears in the guard context.
TOOL_TEXT = "It's 7:14 pm on Wednesday, August 26, in Chicago."


def test_reformatted_tool_time_survives_the_guard():
    reply = "Es ist 19:14 Uhr, sir."          # 19:14 == the tool's 7:14 pm
    assert guard_clock_claims(reply, TOOL_TEXT, "") == reply


def test_derived_time_is_not_replaced_by_the_no_clock_line():
    reply = "Your two hour build would finish at 9:14 pm, sir."
    assert guard_clock_claims(reply, TOOL_TEXT, "") != NO_CLOCK_LINE
    assert guard_clock_claims(reply, TOOL_TEXT, "") == reply


def test_invented_time_is_still_caught_when_context_has_a_clock_line():
    context = "Current time: 4:32 PM, Wednesday August 26 2026"
    reply = "It is 3:00 am, sir."             # nowhere in the context
    assert guard_clock_claims(reply, context, "") != reply
    # ...and what replaces it is not the (now false) "I haven't a clock"
    assert guard_clock_claims(reply, context, "") == UNSURE_CLOCK_LINE
    # the same context, the right reading: untouched
    assert guard_clock_claims("It is 4:32 pm, sir.", context, "") == \
        "It is 4:32 pm, sir."


def test_the_guard_runs_on_the_production_context_shape():
    """The bug was that ContextEngine.format_for_prompt always emits a
    "Current time:" line, which used to switch the guard off entirely."""
    from datetime import datetime

    from jarvis.context import ContextEngine

    ctx = ContextEngine(memory=None)
    text = ctx.format_for_prompt(ctx.get_context("standard"), spoken=True)
    assert "Current time" in text, text
    now = datetime.now()
    wrong = f"It's {(now.hour + 5) % 12 + 1}:{(now.minute + 7) % 60:02d}, sir."
    assert guard_clock_claims(wrong, text, "") != wrong
