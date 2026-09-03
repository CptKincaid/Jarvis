"""The brain's reply-coverage check: a turn that FETCHED an answer may not
end without one.

LIVE 2026-09-02 23:20:41. One compound question:

    "What's on my calendar tomorrow and then can you add milk to my
     shopping list?"

Both tools ran and both succeeded, in the same round:

    23:20:42.640  tool get_calendar {"range": "tomorrow"} -> ok=True
                  "Tomorrow: 11:15 am Hunter Peyrovi and ValerieAnne
                   Staffeldt for 30 minutes at ..."
    23:20:42.659  tool notes {"action":"add","kind":"todo",
                              "list":"shopping","text":"milk"} -> ok=True
    23:20:42.660  chat reply: "Added to your shopping list, sir."

One millisecond between the second tool and the reply, and 0.00 s of
Ollama overhead: NO model round ran after the tools at all. The reply is
the notes tool's own authored ``speak=`` line, and the loop's

    if result.speak:
        speak = result.speak
        break

ended the turn on it. The calendar answer was fetched, was correct, and
was thrown away. He concluded Jarvis had ignored half his request.

WHAT COVERAGE MEANS HERE. Not keyword overlap: "You have a meeting at
11:15" and "Tomorrow: 11:15 am Hunter Peyrovi and ValerieAnne Staffeldt"
share almost nothing, and the live log's covered turns ("You have four
items on Monday, sir.") share less still. Coverage is decided by
PROVENANCE, which the loop already knows: a result is answered only if a
model round ran with that result in the transcript and that round's prose
is what goes out. An authored ``speak=`` line comes from ONE tool and
ends the turn, so every earlier result that had no authored line of its
own -- data the model still owed him a sentence for -- is dropped by
construction.

WHAT HAPPENS INSTEAD. The authored line is held, not spoken, and the
turn spends its reserved render round: one model round with the tools
STRIPPED, which writes the reply from every result in hand. The model,
not a rule, decides which results deserve a sentence -- so a calendar
read used only to place an appointment still gets no forced recital.

WHY THIS CANNOT BECOME THE TIMER BUG. The unbacked-action guard must not
arm on questions because its retry is offered the tools and executes
(2026-09-02: "did you set my timer?" made the retry SET A TIMER). This
repair round is offered NO tools and any tool_calls it returns are
dropped, so it cannot write -- which is what lets it run on a question,
and it must, because the live turn above IS a question.

Real JarvisBrain over the FakeOllama seam (tests/test_brain_tools.py).
No Ollama, no network, no audio, no display.
"""
import pytest

import jarvis.brain as brain_mod
from jarvis.router import is_question
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec
from tests.test_brain_tools import (FakeContext, FakeMemory,  # noqa: F401
                                    FakeOllama, brain, text_reply, tool_reply)

# ------------------------------------------------------- the live strings
ASKED = ("What's on my calendar tomorrow and then can you add milk to my "
         "shopping list?")
# get_calendar's result, verbatim from the 23:20:42.640 log line (the tail
# past 80 chars is the meeting's video link, which the log cuts).
CALENDAR_TEXT = ("Tomorrow: 11:15 am Hunter Peyrovi and ValerieAnne "
                 "Staffeldt for 30 minutes at https://meet.example/x; "
                 "12:45 pm BIOSENSORS for about 2 hours; 4:00 pm Chiro.")
# notes.py: ToolResult(text=f"{k} added: {text}", speak=line)
NOTES_TEXT = "todo added: milk"
NOTES_LINE = "Added to your shopping list, sir."
# what the render round writes once it is allowed to run
COVERED = ("You have a meeting with ValerieAnne at 11:15 tomorrow, sir, "
           "then Biosensors and the chiropractor. Milk is on the shopping "
           "list.")


def make_registry(record, calendar=CALENDAR_TEXT, calendar_ok=True):
    """The four tools this turn can reach, shaped like the real ones: the
    read hands back TEXT for the model to phrase, the writes hand back an
    authored ``speak`` line of their own."""
    reg = ToolRegistry()

    def get_calendar(range="today", **_):
        record.append(("get_calendar", range))
        return ToolResult(text=calendar, ok=calendar_ok, max_sentences=4)

    def notes(action="list", text="", **_):
        record.append(("notes", action, text))
        return ToolResult(text=NOTES_TEXT, speak=NOTES_LINE)

    def add_event(title="", when="", **_):
        record.append(("add_event", title, when))
        line = f"Added {title}, {when}, to your calendar, sir."
        return ToolResult(text=line, speak=line)

    def set_reminder(when="", text="", **_):
        record.append(("set_reminder", when, text))
        line = f"Timer set for {when}, sir."
        return ToolResult(text=line, speak=line)

    def get_weather(when="now", **_):
        record.append(("get_weather", when))
        return ToolResult(text="Tomorrow: high 96, low 77, heavy showers.")

    def system_health(**_):
        record.append(("system_health",))
        return ToolResult(text="")            # a probe with nothing to say

    obj = {"type": "object", "properties": {}}
    reg.register_many([
        ToolSpec("get_calendar", "What is on the calendar.", obj, get_calendar),
        ToolSpec("notes", "Notes, to-dos and lists.", obj, notes),
        ToolSpec("add_event", "Put an event on the calendar.", obj, add_event),
        ToolSpec("set_reminder", "Set a timer or reminder.", obj, set_reminder),
        ToolSpec("get_weather", "Weather now or a forecast.", obj, get_weather),
        ToolSpec("system_health", "How the machine is doing.", obj, system_health),
    ])
    return reg


@pytest.fixture
def setup(brain, monkeypatch):  # noqa: F811
    """(build, record): build(**kw) -> (JarvisBrain, FakeOllama)."""
    brain.reset_static_prompt()
    record = []

    def build(**kw):
        reg = make_registry(record, **kw)
        monkeypatch.setattr(brain, "_REGISTRY", reg)
        fake = FakeOllama()
        monkeypatch.setattr(brain, "_http", fake)
        return brain.JarvisBrain(context=FakeContext(), memory=FakeMemory()), fake

    return build, record


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelname == "WARNING" and "answer" in r.getMessage()]


# ------------------------------------------------------- the pure helper
@pytest.mark.parametrize("result, owed", [
    # data with no authored line: only a model round can turn it into words
    (ToolResult(text=CALENDAR_TEXT), True),
    (ToolResult(text="Nothing on tomorrow, sir."), True),
    # an authored line has already said its piece
    (ToolResult(text=NOTES_TEXT, speak=NOTES_LINE), False),
    # a failure is not an answer he is owed; the model is not held for it
    (ToolResult(text="calendar unreachable", ok=False), False),
    # a probe that found nothing to report forces no sentence
    (ToolResult(text=""), False),
    (ToolResult(text="   "), False),
])
def test_only_unphrased_data_is_an_answer_owed(result, owed):
    assert brain_mod.answer_owed(result) is owed


def test_the_owed_names_are_listed_in_the_order_they_ran():
    ran = [("get_calendar", ToolResult(text=CALENDAR_TEXT)),
           ("system_health", ToolResult(text="")),
           ("get_weather", ToolResult(text="high 96")),
           ("notes", ToolResult(text=NOTES_TEXT, speak=NOTES_LINE))]
    assert brain_mod.answers_owed(ran) == ["get_calendar", "get_weather"]
    assert brain_mod.answers_owed([]) == []


# --------------------------------------------------------- the live turn
def test_the_23_20_turn_reaches_him(setup, caplog):
    """The exact turn replayed: both tools succeed in one round and the
    notes line tries to end the turn. The calendar answer gets said."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "kind": "todo",
                                          "list": "shopping", "text": "milk"})),
                    text_reply(COVERED)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(ASKED)
    # both tools still ran, exactly as they did live
    assert record == [("get_calendar", "tomorrow"), ("notes", "add", "milk")]
    spoken = dict(tags)["SPEAK"]
    # the half he was told about
    assert "shopping list" in spoken or "Milk" in spoken
    # ...and the half that was fetched and dropped
    assert "11:15" in spoken
    assert spoken != NOTES_LINE, "the authored line still ate the turn"
    assert _warnings(caplog) == [
        "chat: the reply would drop the answer from get_calendar; "
        "spending the render round on it"]


def test_the_repair_round_is_offered_no_tools(setup):
    """The round that fixes the reply is the RENDER round: tools stripped,
    told in-message that no more will run, and the static prefix
    untouched (Ollama's cache lives on those bytes)."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    text_reply(COVERED)]
    b._chat_sync(ASKED)
    p1, p2 = fake.chat_payloads()
    assert p1["tools"] and "tools" not in p2, "the repair round was offered tools"
    assert p2["messages"][0] == p1["messages"][0]
    assert p2["messages"][-1] == {"role": "user",
                                  "content": brain_mod.RENDER_NOW_LINE}
    # the repair round is written from the results, both of them
    tools = [m for m in p2["messages"] if m["role"] == "tool"]
    assert [m["tool_name"] for m in tools] == ["get_calendar", "notes"]
    assert CALENDAR_TEXT in tools[0]["content"]
    assert tools[1]["content"] == NOTES_TEXT


# ------------------------------------------- it cannot become the timer bug
def test_a_question_is_repaired_but_the_repair_cannot_write(setup):
    """The 2026-09-02 timer bug in one test. The live turn IS a question,
    so a guard gated on is_question would leave it broken -- and the
    unbacked guard's retry, which IS offered tools, is exactly why that
    gate exists. This repair round is offered none: a model that asks for
    a write during it gets no write, and no second chance to ask."""
    assert is_question(ASKED), "the live turn is a question"
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    # the repair round tries to set a timer instead of writing
                    tool_reply(("set_reminder", {"when": "ten minutes",
                                                 "text": "milk"}))]
    tags = b._chat_sync(ASKED)
    assert [name for name, *_ in record] == ["get_calendar", "notes"]
    assert "set_reminder" not in [name for name, *_ in record]
    assert len(fake.chat_payloads()) == 2, "no round after the render round"
    # nothing was written and he is told plainly what is in hand
    spoken = dict(tags)["SPEAK"]
    assert spoken.startswith(NOTES_LINE)
    assert "your calendar" in spoken


def test_the_degrade_keeps_the_authored_line_and_names_what_was_dropped(setup):
    """The render round writing nothing is the one case where he loses the
    calendar answer -- so he keeps the confirmation he earned and hears
    which source went unspoken, in Jarvis's own vocabulary rather than the
    result's words (TOOL_SOURCE_NAMES)."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    text_reply("   ")]
    tags = b._chat_sync(ASKED)
    spoken = dict(tags)["SPEAK"]
    assert spoken.startswith(NOTES_LINE)
    assert "your calendar" in spoken
    assert "Staffeldt" not in spoken and "11:15" not in spoken


def test_a_dead_ollama_during_the_repair_still_speaks_the_confirmation(setup):
    """The write happened; a model that dies phrasing the rest must not
    cost him the news of it."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    TimeoutError("ollama took too long")]
    tags = b._chat_sync(ASKED)
    spoken = dict(tags)["SPEAK"]
    assert spoken.startswith(NOTES_LINE)
    assert "your calendar" in spoken


# ------------------------------------------------------------- negatives
# Every shape below is from the real log (jarvis.log, jarvis.log.1: 187
# tool-running turns). The check must be invisible on all of them.
def test_a_lone_authored_line_still_answers_instantly(setup, caplog):
    """LIVE, many times: "Play my like songs." / "Stop the music!" /
    "cross unicorn off the list" -- one tool, its own line, no model round
    at all. That is the latency win the short-circuit exists for."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("notes", {"action": "add", "text": "milk"}))]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("add milk to my shopping list")
    assert tags == [("SPEAK", NOTES_LINE)]
    assert len(fake.chat_payloads()) == 1, "a round was spent on nothing"
    assert _warnings(caplog) == []


def test_two_reads_are_rendered_as_they_always_were(setup, caplog):
    """LIVE 2026-08-31 18:48:55 "what's the weather and what's on my
    calendar?" -- two results with no authored line between them. The
    model's own round covers both; nothing is held and nothing is added."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_weather", {"when": "tomorrow"}),
                               ("get_calendar", {"range": "tomorrow"})),
                    text_reply("Heavy showers tomorrow, sir, and a meeting "
                               "at 11:15.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("what's the weather and what's on my calendar?")
    assert tags == [("SPEAK", "Heavy showers tomorrow, sir, and a meeting "
                              "at 11:15.")]
    assert len(fake.chat_payloads()) == 2
    assert _warnings(caplog) == []


def test_a_lookup_used_to_decide_is_not_forced_into_a_sentence(setup):
    """LIVE 2026-08-31 21:5x "add hello to 4 30 p.m. tomorrow on my
    calendar": a model may read the calendar to place the event. He asked
    ONE thing, so the answer is the confirmation -- the check hands the
    results to the model and the MODEL decides what deserves a sentence;
    it never appends a recital of its own."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("add_event", {"title": "hello",
                                              "when": "Tuesday at 4:30 PM"})),
                    text_reply("Added hello, Tuesday at 4:30 PM, to your "
                               "calendar, sir.")]
    tags = b._chat_sync("add hello to 4 30 p.m. tomorrow on my calendar")
    assert tags == [("SPEAK", "Added hello, Tuesday at 4:30 PM, to your "
                              "calendar, sir.")]
    assert "11:15" not in dict(tags)["SPEAK"]


def test_a_failed_read_does_not_hold_the_authored_line(setup, caplog):
    """LIVE "check my emails" -> get_mail ok=False. A tool that failed
    owes him no answer: the model is not held up to phrase a failure, and
    the turn keeps the instant confirmation."""
    b, fake = setup[0](calendar="calendar unreachable", calendar_ok=False)
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"}))]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(ASKED)
    assert tags == [("SPEAK", NOTES_LINE)]
    assert len(fake.chat_payloads()) == 1
    assert _warnings(caplog) == []


def test_a_silent_probe_does_not_hold_the_authored_line(setup):
    """A status lookup that produced no text is not an answer: nothing to
    drop, so nothing to repair."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("system_health", {}),
                               ("notes", {"action": "add", "text": "milk"}))]
    tags = b._chat_sync("check the box and add milk to the list")
    assert tags == [("SPEAK", NOTES_LINE)]
    assert len(fake.chat_payloads()) == 1


def test_the_authored_line_first_is_untouched(setup):
    """The write comes back BEFORE the read in the round: the loop stops
    at the authored line, so the calendar never ran and nothing is owed.
    (The skipped call is answered in the transcript as it always was.)"""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("notes", {"action": "add", "text": "milk"}),
                               ("get_calendar", {"range": "tomorrow"}))]
    record = setup[1]
    tags = b._chat_sync(ASKED)
    assert record == [("notes", "add", "milk")]
    assert tags == [("SPEAK", NOTES_LINE)]


# --------------------------------------------------------- the streamed path
def test_the_held_line_is_never_spoken_twice(brain, monkeypatch):  # noqa: F811
    """Streaming: a tool round speaks nothing, so the held authored line
    was never streamed. The repaired reply is what he hears, once."""
    brain.reset_static_prompt()
    record = []
    monkeypatch.setattr(brain, "_REGISTRY", make_registry(record))
    rounds = [
        [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_calendar", "arguments": {"range": "tomorrow"}}},
            {"function": {"name": "notes", "arguments": {"action": "add",
                                                         "text": "milk"}}}]},
          "done": True, "load_duration": 0}],
        [{"message": {"role": "assistant", "content": COVERED},
          "done": True, "load_duration": 0}],
    ]
    sent = []

    def stream(path, payload, timeout=None):
        sent.append(payload)
        yield from rounds[len(sent) - 1]

    monkeypatch.setattr(brain, "_http_stream", stream)
    b = brain.JarvisBrain(context=FakeContext(), memory=FakeMemory())
    spoken = []
    tags = b._chat_sync(ASKED, on_sentence=spoken.append)
    assert NOTES_LINE not in spoken
    assert spoken and "11:15" in " ".join(spoken)
    assert len(sent) == 2 and "tools" not in sent[1]
    assert dict(tags)["STREAMED"] == str(len(spoken))
