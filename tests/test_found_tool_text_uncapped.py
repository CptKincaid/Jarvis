"""FOUND 2026-08-26 (adversarial re-check of the assistant-tools sweep):
nothing capped the length of a ToolResult.text between the tool and the
model, and the model runs at num_ctx 8192 (jarvis/brain.py NUM_CTX).

jarvis/brain.py:1040 appended the tool text verbatim:

    messages.append({"role": "tool", "content": result.text, ...})

Two producers are unbounded:

* jarvis/tools/calendar.py format_events() joins EVERY event in the range;
* jarvis/tools/mail.py fact_sheet() writes each Subject: header in full
  (the snippet is capped, the subject is not).

Measured in-process against the live Ollama (gemma4:26b, quiet box), a
subscribed busy calendar of 60 events a day:

    5  events/day  ->  3 017 chars  -> 2nd-pass prompt 3 516 tok, correct answer
    20 events/day  -> 11 907 chars  -> 2nd-pass prompt 5 896 tok, correct answer
    60 events/day  -> 35 889 chars  -> 2nd-pass prompt 2 687 tok (!), and the
                                       answer to "what's on my calendar this
                                       week?" came back
                                       "Good evening, sir; it is rather late
                                       for a Wednesday."

i.e. once the tool message no longer fits in num_ctx, Ollama drops it and
the model answers something confident and entirely unrelated instead of
failing.  The same happens with five spam mails carrying 12 000-character
subjects (60 085 chars -> "Good morning, sir; the forecast for today ...").

FIXED 2026-08-26 (brain work item, finding H2). The cap lives in
jarvis/brain.py, the one place EVERY tool result passes through on its way
into the prompt, so it holds for any producer — calendar, mail, a web
fetch, a tool written next month:

  * cap_tool_text() trims one result to MAX_TOOL_TEXT_CHARS and appends a
    marker naming what was lost;
  * _chat_sync spends a MAX_TOOL_TEXT_TOTAL_CHARS budget across the whole
    turn, so ten results cannot add up past the window either;
  * the marker tells the model, in the tool message itself, to say it is
    only seeing part — and if the reply does not say so, the brain appends
    PARTIAL_RESULT_LINE. A half-seen result can no longer produce a
    confident whole-looking answer.

format_events() itself is still unbounded (it belongs to the calendar
work item); this test therefore uses it as the real-world producer and
asserts on what reaches the model.
"""
from datetime import datetime, timedelta

from jarvis import brain
from jarvis.tools.calendar import Event, format_events, system_tz
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec

# a tool result has to leave room for the ~2 700-token system+tools prefix
# inside NUM_CTX = 8192; ~8 000 characters is already generous.
TOOL_TEXT_BUDGET_CHARS = 8000


def _busy_week(per_day=60, days=7):
    tz = system_tz()
    base = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    events = []
    for day in range(days):
        for i in range(per_day):
            start = base + timedelta(days=day, hours=7, minutes=7 * i)
            events.append(Event(start=start, end=start + timedelta(minutes=5),
                                all_day=False,
                                title=f"Team sync {i} on project Aurora "
                                      "with the platform group",
                                calendar="Busy", location="Conference room B"))
    return events


def _loop(monkeypatch, tool_text, reply):
    """Run one real tool round; return (tool messages seen by the model,
    spoken line)."""
    payloads = []
    reg = ToolRegistry()
    reg.register(ToolSpec(name="get_calendar", description="what's on",
                          handler=lambda **kw: ToolResult(text=tool_text)))

    def http(path, payload=None, timeout=None):
        payloads.append(payload)
        if len(payloads) == 1:
            return {"message": {"content": "",
                                "tool_calls": [{"function": {
                                    "name": "get_calendar",
                                    "arguments": {}}}]}}
        return {"message": {"content": reply}}

    monkeypatch.setattr(brain, "_http", http)
    b = brain.JarvisBrain(context=None, memory=None, registry=reg)
    tags = b._chat_sync("what's on my calendar this week?")
    spoken = [t for tag, t in tags if tag == "SPEAK"][0]
    tool_msgs = [m for m in payloads[-1]["messages"] if m["role"] == "tool"]
    return tool_msgs, spoken


def test_a_huge_tool_result_is_capped_before_the_model_sees_it(monkeypatch):
    text = format_events(_busy_week(), "week", now=datetime.now(system_tz()))
    assert len(text) > 30000, len(text)          # the real producer, unbounded
    tool_msgs, _ = _loop(monkeypatch, text, "Rather a lot on, sir.")
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    assert len(content) <= TOOL_TEXT_BUDGET_CHARS, len(content)
    # not silently dropped: the model is told what it is missing
    assert "truncated" in content
    assert str(len(text)) in content
    # and what it does get is the head of the real result, not a stub
    assert content.startswith(text[:200])


def test_the_spoken_answer_admits_the_result_was_truncated(monkeypatch):
    """The worst failure mode is a confident answer about something else.
    Whatever the model says, the reply has to say it is partial."""
    text = format_events(_busy_week(), "week", now=datetime.now(system_tz()))
    _, spoken = _loop(monkeypatch, text, "Rather a lot on, sir.")
    assert spoken.startswith("Rather a lot on, sir.")
    assert brain.PARTIAL_RESULT_LINE in spoken
    assert brain._PARTIAL_RX.search(spoken)


def test_a_model_that_says_so_itself_is_not_second_guessed(monkeypatch):
    _, spoken = _loop(monkeypatch, "x" * 50000,
                      "That's only part of the list, sir.")
    assert spoken == "That's only part of the list, sir."


def test_a_result_that_fits_is_untouched_and_unremarked(monkeypatch):
    small = format_events(_busy_week(per_day=1, days=1), "week",
                          now=datetime.now(system_tz()))
    tool_msgs, spoken = _loop(monkeypatch, small, "One thing, sir.")
    assert tool_msgs[0]["content"] == small
    assert spoken == "One thing, sir."


def test_many_results_cannot_add_up_past_the_turn_budget(monkeypatch):
    """One capped result fits; ten of them still have to fit."""
    payloads = []
    reg = ToolRegistry()
    reg.register(ToolSpec(name="get_calendar", description="what's on",
                          handler=lambda **kw: ToolResult(text="y" * 20000)))

    def http(path, payload=None, timeout=None):
        payloads.append(payload)
        if len(payloads) == 1:
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "get_calendar", "arguments": {}}}
                for _ in range(6)]}}
        return {"message": {"content": "Busy, sir."}}

    monkeypatch.setattr(brain, "_http", http)
    brain.JarvisBrain(None, None, registry=reg)._chat_sync("everything")
    tool_msgs = [m for m in payloads[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 6
    total = sum(len(m["content"]) for m in tool_msgs)
    assert total <= brain.MAX_TOOL_TEXT_TOTAL_CHARS + \
        6 * len(brain.TOOL_TRUNCATED_MARKER), total


def test_cap_tool_text_is_exact():
    text = "line one\nline two\n" + "z" * 100
    assert brain.cap_tool_text(text, 4000) == (text, False)
    out, cut = brain.cap_tool_text(text, 20)
    assert cut and out.startswith("line one\nline two")
    assert "truncated" in out and str(len(text)) in out
    assert brain.cap_tool_text("", 10) == ("", False)
    assert brain.cap_tool_text(None, 10) == ("", False)
