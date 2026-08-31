"""Regression tests: a raised sentence cap the model was never told about.

DEFECT (jarvis/brain.py tool_message + jarvis/tools/calendar.py):

A ToolResult can raise max_sentences -- mail asks for 4, briefing for 6 -- but
that number only ever reached the POST-trim (limit_sentences). The message the
model actually reads carried the tool text and nothing else, while the static
system prompt kept telling it "answer in one or two sentences" and "[SPEAK]
text — read aloud (max 2 sentences)". So the model wrote two sentences, the
raised cap trimmed nothing, and the allowance was inert.

get_calendar did not even ask: it returned ToolResult(text=text) with the
default 2. Heard on 2026-08-28 -- asked "What's on the calendar for Monday?"
against a result holding four events, Jarvis said:

    "You have four items on Monday, sir, starting with BIOSENSORS at 9:10 am."

One sentence, one event named, three dropped. The count was right and the
content was not there.
"""


from tests.test_brain_tools import (FakeOllama, setup, brain, text_reply,  # noqa: F401
                                    tool_reply)


def _tool_messages(fake):
    """Every tool-role message the model was shown, across all rounds."""
    seen = []
    for payload in fake.chat_payloads():
        seen += [m for m in payload["messages"] if m.get("role") == "tool"]
    return seen


def test_a_raised_cap_is_stated_in_the_message_the_model_reads(brain, setup):
    b, fake, _ = setup
    fake.replies = [tool_reply(("get_briefing", {})),
                    text_reply("Weather is fair, sir.")]
    b._chat_sync("Give me the briefing.")

    briefing_msgs = [m for m in _tool_messages(fake)
                     if m.get("tool_name") == "get_briefing"]
    assert briefing_msgs, "the briefing result never reached the model"
    content = briefing_msgs[0]["content"]
    assert "6" in content and "sentence" in content.lower(), (
        "the model is told 'max 2 sentences' by the system prompt and was "
        f"never told this result allows 6: {content!r}")


def test_an_ordinary_result_says_nothing_extra(brain, setup):
    b, fake, _ = setup
    fake.replies = [tool_reply(("get_weather", {"when": "now"})),
                    text_reply("Seventy-two and partly cloudy, sir.")]
    b._chat_sync("What's the weather?")

    weather = [m for m in _tool_messages(fake)
               if m.get("tool_name") == "get_weather"]
    assert weather
    assert "sentence" not in weather[0]["content"].lower(), (
        "a default 2-sentence result must not gain an allowance line")


def test_the_calendar_asks_for_room_to_name_the_events():
    """The reported case: four events, one named."""
    from jarvis.tools import calendar as cal
    assert cal.CALENDAR_MAX_SENTENCES >= 4, (
        "a day with four events cannot be read out in two sentences")
