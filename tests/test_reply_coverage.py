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
is what goes out. An authored ``speak=`` line comes from ONE tool, so
every result beside it that had no authored line of its own -- data the
model still owed him a sentence for -- was dropped by construction.

THE ORDER MUST NOT DECIDE THE ANSWER. That ``break`` also ended the ROUND,
so the same request phrased write-first ("Add milk to my shopping list and
what's on my calendar tomorrow?") never ran get_calendar at all, and two
writes in one round ran only the first. gemma4 emits its calls in the
order of his clauses -- all eight multi-tool turns across jarvis.log and
jarvis.log.1 do, including the reversed pair 18:48:55 "what's the weather
and what's on my calendar?" -> get_weather, get_calendar. So the round now
finishes the calls the model asked for (the fan-out cap and the work
budget still bound it) and the decision is made once, with every result of
the round in hand.

WHAT HAPPENS ON A MISS. The authored lines are held, not spoken, and the
turn spends its reserved render round: one model round with the tools
STRIPPED, which writes the reply from every result in hand. The model, not
a rule, decides which results deserve a sentence. But the confirmation is
NOT the model's to drop: the held lines are appended to whatever that
round writes, past the spoken-sentence cap, unless the reply already
carries them verbatim. A write that happened is reported.

WHAT IT DOES NOT PROMISE. The check fires on any read beside an authored
line, whatever the intent behind it, and RENDER_NOW_LINE pushes the model
toward answering: it never COMPOSES a recital, but nothing here stops the
model writing one. And the convention it reads -- reads hand back text,
writes author their own line -- has exceptions already: this header used
to name one, screen_qa (jarvis/tools/screen.py, because the vision call
can take 25 s), and a census of jarvis/tools on 2026-09-03 found ten
(AUTHORED_BY_READS at the bottom of this file, pinned by a test that
parses the tools rather than trusting this prose). Nothing at register
time enforces the convention.

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
# notes.py:834, `act == "list"`: speak= is s.list_text(k), i.e. the items
# HE dictated, in his words. Markdown and an emoji here because that is
# what a shopping list picks up (2026-09-03, F27).
LIST_LINE = "Three on your shopping list, sir: **milk**, eggs 🥚 and bread."
LIST_LINE_SPOKEN = "Three on your shopping list, sir: milk, eggs and bread."
# timekeeper.py list_text: three sentences, and the tool says so
# (max_sentences=3).
SCHEDULE_LINE = ("Two timers running, sir. The kitchen one has four minutes "
                 "left. The laundry one has half an hour.")
# what the render round writes once it is allowed to run
COVERED = ("You have a meeting with ValerieAnne at 11:15 tomorrow, sir, "
           "then Biosensors and the chiropractor. Milk is on the shopping "
           "list.")
# screen.py's answer: a READ that authors its own success line, because
# the vision call can take 25 s (jarvis/tools/screen.py, MAX_SENTENCES).
SCREEN_LINE = "A pull request, sir."


def make_registry(record, calendar=CALENDAR_TEXT, calendar_ok=True,
                  screen=None):
    """The four tools this turn can reach, shaped like the real ones: the
    read hands back TEXT for the model to phrase, the writes hand back an
    authored ``speak`` line of their own."""
    reg = ToolRegistry()

    def get_calendar(range="today", **_):
        record.append(("get_calendar", range))
        return ToolResult(text=calendar, ok=calendar_ok, max_sentences=4)

    def notes(action="list", text="", **_):
        record.append(("notes", action, text))
        if action == "list":
            # notes.py:834 -- the READ half of the same tool authors its
            # line too, and the line is HUNTER'S OWN text: whatever he put
            # on the list, markdown, emoji and all.
            return ToolResult(text=LIST_LINE, speak=LIST_LINE)
        return ToolResult(text=NOTES_TEXT, speak=NOTES_LINE)

    def add_event(title="", when="", **_):
        record.append(("add_event", title, when))
        line = f"Added {title}, {when}, to your calendar, sir."
        return ToolResult(text=line, speak=line)

    def set_reminder(when="", text="", **_):
        record.append(("set_reminder", when, text))
        line = f"Timer set for {when}, sir."
        return ToolResult(text=line, speak=line)

    def manage_schedule(action="list", **_):
        # timekeeper.py `a == "list"`: a READ that authors its line, and
        # the line is several sentences (max_sentences=3, as the real one
        # declares). Held beside an owed read it is three sentences of
        # authored text in the degrade, not one.
        record.append(("manage_schedule", action))
        return ToolResult(text=SCHEDULE_LINE, speak=SCHEDULE_LINE,
                          max_sentences=3)

    def get_weather(when="now", **_):
        record.append(("get_weather", when))
        return ToolResult(text="Tomorrow: high 96, low 77, heavy showers.")

    def system_health(**_):
        record.append(("system_health",))
        return ToolResult(text="")            # a probe with nothing to say

    def screen_qa(question="", **_):
        record.append(("screen_qa", question))
        # `screen` overrides the result TEXT only: a vision answer long
        # enough to be cut against NUM_CTX, with the authored line intact.
        return ToolResult(text=screen or f"Active window: Chrome. {SCREEN_LINE}",
                          max_sentences=3, speak=SCREEN_LINE)

    obj = {"type": "object", "properties": {}}
    reg.register_many([
        ToolSpec("get_calendar", "What is on the calendar.", obj, get_calendar),
        ToolSpec("notes", "Notes, to-dos and lists.", obj, notes),
        ToolSpec("add_event", "Put an event on the calendar.", obj, add_event),
        ToolSpec("set_reminder", "Set a timer or reminder.", obj, set_reminder),
        ToolSpec("manage_schedule", "List, cancel or adjust timers.", obj,
                 manage_schedule),
        ToolSpec("get_weather", "Weather now or a forecast.", obj, get_weather),
        ToolSpec("system_health", "How the machine is doing.", obj, system_health),
        ToolSpec("screen_qa", "Answer a question about the screen.", obj,
                 screen_qa),
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
@pytest.mark.parametrize("result, owed, in_hand", [
    # data with no authored line: only a model round can turn it into words
    (ToolResult(text=CALENDAR_TEXT), True, True),
    (ToolResult(text="Nothing on tomorrow, sir."), True, True),
    # an authored line has already said its piece
    (ToolResult(text=NOTES_TEXT, speak=NOTES_LINE), False, False),
    # A FAILURE IS OWED A SENTENCE TOO -- "I couldn't reach your calendar,
    # sir" is exactly what the short-circuit was eating -- but it is NOT in
    # hand, so the degrade may never say Jarvis HAS it.
    (ToolResult(text="calendar unreachable", ok=False), True, False),
    # ...unless the tool wrote its own excuse, which has already said it
    (ToolResult(text="calendar unreachable", ok=False,
                speak="I couldn't reach your calendar, sir."), False, False),
    # a probe that found nothing to report forces no sentence
    (ToolResult(text=""), False, False),
    (ToolResult(text="   "), False, False),
])
def test_only_unphrased_results_are_owed(result, owed, in_hand):
    assert brain_mod.answer_owed(result) is owed
    assert brain_mod.answer_in_hand(result) is in_hand


def test_the_owed_names_are_listed_in_the_order_they_ran():
    ran = [("get_calendar", ToolResult(text=CALENDAR_TEXT)),
           ("system_health", ToolResult(text="")),
           ("get_mail", ToolResult(text="mail unreachable", ok=False)),
           ("get_weather", ToolResult(text="high 96")),
           ("notes", ToolResult(text=NOTES_TEXT, speak=NOTES_LINE))]
    assert brain_mod.answers_owed(ran) == ["get_calendar", "get_mail",
                                           "get_weather"]
    # the degrade's list: the failed read is owed a sentence but is not
    # something to claim to have
    assert brain_mod.answers_owed(ran, in_hand=True) == ["get_calendar",
                                                         "get_weather"]
    assert brain_mod.answers_owed([]) == []


@pytest.mark.parametrize("spoken, missing", [
    # the model wrote the confirmation itself, verbatim (the common case:
    # app.py #144 was this exact duplicate) -- nothing to append
    (f"You have a meeting at 11:15, sir. {NOTES_LINE}", []),
    # ...whitespace and case are not the test
    (f"{NOTES_LINE.upper()}   and a meeting at 11:15", []),
    # it wrote the FACT in its own words: not a match, and a repeat is the
    # cheap failure -- silence about a write is the expensive one
    ("Milk is on the shopping list, sir.", [NOTES_LINE]),
    ("You have a meeting at 11:15 tomorrow, sir.", [NOTES_LINE]),
])
def test_a_held_line_is_only_skipped_when_the_reply_says_it_verbatim(
        spoken, missing):
    assert brain_mod.held_lines_missing(spoken, [NOTES_LINE]) == missing


def test_a_held_line_is_appended_past_the_char_budget_by_shortening_the_head():
    head = "You have a meeting at 11:15 tomorrow, sir. " * 20
    out = brain_mod.append_spoken_lines(head, [NOTES_LINE])
    assert out.endswith(NOTES_LINE)
    assert len(out) <= brain_mod.HARD_SPOKEN_CHARS
    assert brain_mod.append_spoken_lines("", [NOTES_LINE]) == NOTES_LINE
    assert brain_mod.append_spoken_lines("word.", []) == "word."


def test_the_degrade_never_claims_a_source_that_failed():
    assert brain_mod.coverage_degrade([NOTES_LINE], ["get_calendar"]) == \
        f"{NOTES_LINE} {brain_mod.tool_only_line(['get_calendar'])}"
    # nothing in hand: the confirmation alone, with no invented claim
    assert brain_mod.coverage_degrade([NOTES_LINE], []) == NOTES_LINE
    assert brain_mod.coverage_degrade(NOTES_LINE, []) == NOTES_LINE


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
    # the half he was told about -- and the CONFIRMATION itself, which the
    # repair may not trade away for the answer
    assert NOTES_LINE in spoken
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
    told = p2["messages"][-1]
    assert told["role"] == "user"
    assert told["content"].startswith(brain_mod.RENDER_NOW_LINE)
    # ...and it is TOLD the write already happened, so its reply is not
    # free to be about the calendar alone
    assert NOTES_LINE in told["content"]
    assert "do not repeat" in told["content"].lower()
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


# ------------------------------- the held line is not the model's to drop
def test_a_repair_that_answers_only_the_read_still_reports_the_write(setup):
    """The render round is given the results and its own judgement, and
    "You have a meeting with ValerieAnne at 11:15 tomorrow, sir." is a
    legal render of this turn. Before this, that answer DISCARDED the held
    confirmation -- milk went on the list and nobody told him. The
    confirmation is appended by the code, not asked of the model."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    text_reply("You have a meeting with ValerieAnne at "
                               "11:15 tomorrow, sir.")]
    spoken = dict(b._chat_sync(ASKED))["SPEAK"]
    assert "11:15" in spoken
    assert spoken.endswith(NOTES_LINE), spoken


def test_the_spoken_cap_cannot_trim_the_confirmation_off_the_end(setup):
    """get_calendar carries max_sentences=4, so the turn's cap is 4 and a
    five-sentence render is trimmed to its first four -- which is exactly
    where a trailing confirmation sits. The held line is appended AFTER
    the cap: it reports an action that really happened, and the cap is a
    rule about how much prose he wants, not a licence to drop that."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    text_reply("You have a meeting with ValerieAnne at "
                               "11:15 tomorrow, sir. Then Biosensors at "
                               "12:45. The chiropractor is at four. And "
                               "the gym at six. Milk is on the list.")]
    spoken = dict(b._chat_sync(ASKED))["SPEAK"]
    assert spoken.endswith(NOTES_LINE), spoken
    assert "11:15" in spoken


def test_a_repair_that_writes_the_confirmation_itself_says_it_once(setup):
    """...and when the model DOES write the authored sentence -- copying a
    tool's line verbatim is common enough that app.py has a guard for it
    (#144) -- it is not said a second time."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    text_reply(f"You have a meeting at 11:15, sir. "
                               f"{NOTES_LINE}")]
    spoken = dict(b._chat_sync(ASKED))["SPEAK"]
    assert spoken.count("Added to your shopping list") == 1, spoken


@pytest.mark.parametrize("boom", [
    brain_mod.OllamaDown("refused"),
    brain_mod.MalformedReply("not /api/chat shape"),
    ValueError("no JSON in body"),
    TimeoutError("ollama took too long"),
])
def test_ollama_dying_during_the_repair_never_costs_him_the_write(setup, boom):
    """All four failure branches around the repair round. Three of them
    returned early and threw the held line away, so an Ollama hiccup
    turned a note that WAS added into "I'm afraid my local model is down,
    sir." The write happened; he hears about it whatever the model does."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    boom]
    spoken = dict(b._chat_sync(ASKED))["SPEAK"]
    assert record == [("get_calendar", "tomorrow"), ("notes", "add", "milk")]
    assert spoken.startswith(NOTES_LINE), spoken
    assert "your calendar" in spoken           # ...and what went unspoken
    assert brain_mod.MODEL_DOWN_LINE not in spoken
    assert brain_mod.MODEL_EMPTY_LINE not in spoken


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


def test_the_check_never_composes_a_recital_of_its_own(setup):
    """A model may read the calendar in order to place an appointment. He
    asked ONE thing, so the confirmation is the whole answer.

    STATED HONESTLY, because the first cut of this claimed more than it
    could: the check never COMPOSES a recital -- it hands the results to
    the model and the model decides -- but nothing here prevents the model
    writing one, and RENDER_NOW_LINE ("answer my question now, from those
    results") pushes that way. What the note added beside it does is give
    the model the out: the write is already confirmed, so "a word of
    acknowledgement is enough".

    The attribution the first cut carried was wrong too. The real
    2026-08-31 21:02:13 turn ("add hello to 4 30 p.m. tomorrow on my
    calendar", jarvis.log.1) ran add_event ALONE -- no get_calendar, no
    second tool, 0.00 s of Ollama overhead -- so it is not this shape at
    all. This is a scripted model, and it is pinned as a REGRESSION guard
    on the composing rule, not as an observed turn."""
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
    # the model wrote the authored line itself, so it is not appended twice
    assert dict(tags)["SPEAK"].count("Added hello") == 1


def test_a_failed_read_beside_a_write_is_still_a_sentence_he_is_owed(setup):
    """The read half FAILS and the write half succeeds. The first cut of
    this check exempted failures -- "its excuse is the model's to phrase"
    -- but the short-circuit is exactly what stops any model round from
    ever seeing the failure, so the whole turn came out as "Added to your
    shopping list, sir." and he never learnt the calendar had not been
    read. The failure text reaches a round, and the round gets to say so."""
    b, fake = setup[0](calendar="calendar unreachable", calendar_ok=False)
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    text_reply("I couldn't reach your calendar, sir.")]
    tags = b._chat_sync(ASKED)
    spoken = dict(tags)["SPEAK"]
    assert "couldn't reach your calendar" in spoken
    assert NOTES_LINE in spoken, "the write went unreported"
    # the failure reached the round that had to phrase it
    told = fake.chat_payloads()[1]["messages"]
    assert any("calendar unreachable" in m.get("content", "")
               for m in told if m["role"] == "tool")


def test_a_failed_read_is_never_claimed_as_something_jarvis_has(setup):
    """...and if that round writes nothing, the degrade names only what is
    IN HAND. "I have your calendar, sir" off "calendar unreachable" would
    be the degrade lying all by itself."""
    b, fake = setup[0](calendar="calendar unreachable", calendar_ok=False)
    fake.replies = [tool_reply(("get_calendar", {"range": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"})),
                    text_reply("   ")]
    spoken = dict(b._chat_sync(ASKED))["SPEAK"]
    assert spoken == NOTES_LINE
    assert "your calendar" not in spoken


def test_a_silent_probe_does_not_hold_the_authored_line(setup):
    """A status lookup that produced no text is not an answer: nothing to
    drop, so nothing to repair."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("system_health", {}),
                               ("notes", {"action": "add", "text": "milk"}))]
    tags = b._chat_sync("check the box and add milk to the list")
    assert tags == [("SPEAK", NOTES_LINE)]
    assert len(fake.chat_payloads()) == 1


def test_the_same_request_phrased_write_first_is_the_same_incident(setup):
    """THE ORDER MUST NOT DECIDE THE ANSWER. "Add milk to my shopping list
    and what's on my calendar tomorrow?" is 23:20:41 with the clauses
    swapped, and gemma4 emits its calls in the order of his clauses -- all
    eight multi-tool turns in the live logs do. Stopping the round at the
    first authored line made this shape WORSE than the incident: the
    calendar never ran at all, and he was told about the milk as if that
    were the whole request. The round finishes, and both halves land."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("notes", {"action": "add", "text": "milk"}),
                               ("get_calendar", {"range": "tomorrow"})),
                    text_reply("You have a meeting at 11:15 tomorrow, sir.")]
    spoken = dict(b._chat_sync(
        "Add milk to my shopping list and what's on my calendar tomorrow?"
    ))["SPEAK"]
    assert record == [("notes", "add", "milk"),
                      ("get_calendar", "tomorrow")], "the read was skipped"
    assert "11:15" in spoken and NOTES_LINE in spoken


def test_a_second_write_in_the_round_is_never_silently_dropped(setup):
    """The same break also dropped WRITES. notes + set_reminder ran the
    note, skipped the timer and reported only the note -- a thing he asked
    for that did not happen and that he was not told about. Both run now,
    and both authored lines are spoken; no model round is needed for
    either, so the latency win survives."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("notes", {"action": "add", "text": "milk"}),
                               ("set_reminder", {"when": "ten minutes"}))]
    tags = b._chat_sync("add milk to my list and set a timer for ten minutes")
    assert [name for name, *_ in record] == ["notes", "set_reminder"]
    spoken = dict(tags)["SPEAK"]
    assert NOTES_LINE in spoken and "Timer set for ten minutes, sir." in spoken
    assert len(fake.chat_payloads()) == 1, "a round was spent on nothing"


def test_the_same_write_asked_for_twice_in_a_round_happens_once(setup):
    """The safety half of finishing the round. The old break hid a second,
    IDENTICAL call behind the first authored line; running the round to
    its end would EXECUTE it, and a second "notes add milk" is a second
    item on his list off one sentence of his. Same tool, same arguments,
    same round is the model repeating itself, never him asking twice."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("notes", {"action": "add", "text": "milk"}),
                               ("notes", {"action": "add", "text": "milk"}))]
    tags = b._chat_sync("add milk to my shopping list")
    assert record == [("notes", "add", "milk")], "the write ran twice"
    assert tags == [("SPEAK", NOTES_LINE)]
    # every tool_call still gets an answer, or the next round re-asks it:
    # two calls, two tool messages, one of them saying it already ran
    msgs = [m for m in fake.chat_payloads()[0]["messages"]
            if m["role"] == "tool"]
    assert [m["content"] for m in msgs] == [NOTES_TEXT,
                                            brain_mod.TOOL_REPEAT_TEXT]


def test_two_different_writes_in_a_round_both_happen(setup):
    """...and the guard is on identical ARGUMENTS, not on the tool: "add
    milk and bread to my list" is two writes and stays two."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("notes", {"action": "add", "text": "milk"}),
                               ("notes", {"action": "add", "text": "bread"}))]
    tags = b._chat_sync("add milk and bread to my shopping list")
    assert record == [("notes", "add", "milk"), ("notes", "add", "bread")]
    # one authored line, said once -- both adds report the same sentence
    assert tags == [("SPEAK", NOTES_LINE)]


def test_a_read_that_authors_its_own_line_still_lets_the_rest_run(setup):
    """screen_qa is the one READ that authors its success line
    (jarvis/tools/screen.py: the vision call can take 25 s, so a second
    model turn to phrase it would be refused). "What's on my screen and
    what's on my calendar?" put it FIRST, and the round stopped there:
    half the request never ran and the check saw nothing to repair."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("screen_qa", {"question": "what is this"}),
                               ("get_calendar", {"range": "tomorrow"})),
                    text_reply("You have a meeting at 11:15 tomorrow, sir.")]
    spoken = dict(b._chat_sync(
        "what's on my screen and what's on my calendar?"))["SPEAK"]
    assert [name for name, *_ in record] == ["screen_qa", "get_calendar"]
    assert "11:15" in spoken and SCREEN_LINE in spoken


# --------------------------------------------------------- the streamed path
def _streamer(brain, monkeypatch, rounds, record=None):  # noqa: F811
    """A JarvisBrain whose /api/chat streams the scripted `rounds` (each a
    list of NDJSON chunks). Returns (brain, sent payloads)."""
    brain.reset_static_prompt()
    monkeypatch.setattr(brain, "_REGISTRY",
                        make_registry(record if record is not None else []))
    sent = []

    def stream(path, payload, timeout=None):
        sent.append(payload)
        yield from rounds[len(sent) - 1]

    monkeypatch.setattr(brain, "_http_stream", stream)
    return brain.JarvisBrain(context=FakeContext(), memory=FakeMemory()), sent


def _tool_chunk(*calls):
    return [{"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": n, "arguments": a}} for n, a in calls]},
        "done": True, "load_duration": 0}]


def test_the_held_line_is_spoken_once_and_last(brain, monkeypatch):  # noqa: F811
    """Streaming: a tool round speaks nothing, so the held authored line
    was never streamed -- and SPEAK is not spoken again once the STREAMED
    tag is on the batch (app.py). So the confirmation has to go to TTS
    itself, after the repaired answer, exactly once."""
    b, sent = _streamer(brain, monkeypatch, [
        _tool_chunk(("get_calendar", {"range": "tomorrow"}),
                    ("notes", {"action": "add", "text": "milk"})),
        [{"message": {"role": "assistant", "content": COVERED},
          "done": True, "load_duration": 0}]])
    spoken = []
    tags = b._chat_sync(ASKED, on_sentence=spoken.append)
    assert "11:15" in " ".join(spoken)
    assert spoken.count(NOTES_LINE) == 1, spoken
    assert spoken[-1] == NOTES_LINE, "the write was written but never said"
    assert len(sent) == 2 and "tools" not in sent[1]
    assert dict(tags)["STREAMED"] == str(len(spoken))
    assert dict(tags)["SPEAK"].endswith(NOTES_LINE)


def test_prose_before_a_tool_call_is_not_answered_a_second_time(brain,  # noqa: F811
                                                                monkeypatch):
    """_stream_round only silences a round AFTER its first tool_call
    chunk, so a round that talks and then asks for a write has already
    spoken the read's answer. Repairing it would say the same thing twice.
    Provenance settles it: that prose came out of a round with the
    calendar result in its transcript, so the calendar IS answered."""
    record = []
    b, sent = _streamer(brain, monkeypatch, [
        _tool_chunk(("get_calendar", {"range": "tomorrow"})),
        # round 2: two sentences of prose, THEN a tool call
        [{"message": {"role": "assistant",
                      "content": "You have a meeting with ValerieAnne at "
                                 "11:15 tomorrow, sir. "}},
         {"message": {"role": "assistant", "content": "Then Biosensors. "}},
         {"message": {"role": "assistant", "content": "", "tool_calls": [
             {"function": {"name": "notes",
                           "arguments": {"action": "add", "text": "milk"}}}]},
          "done": True, "load_duration": 0}],
    ], record=record)
    spoken = []
    tags = b._chat_sync(ASKED, on_sentence=spoken.append)
    assert [name for name, *_ in record] == ["get_calendar", "notes"]
    assert len(sent) == 2, "a repair round was spent on an answer he heard"
    assert len(spoken) == 1 and "11:15" in spoken[0]
    # the authored line ends the turn as it always did, and is spoken by
    # the SPEAK tag (no STREAMED marker, because nothing held it back)
    assert dict(tags) == {"SPEAK": NOTES_LINE}
    assert NOTES_LINE not in spoken


def test_a_result_from_the_talking_round_itself_is_still_owed(brain,  # noqa: F811
                                                              monkeypatch):
    """...and the stand-down is exactly as wide as the provenance claim:
    prose covers what was in the transcript when it was written, not the
    results of the very round it introduced. Here the calendar runs in the
    SAME round as the prose, so nothing had answered it."""
    record = []
    b, sent = _streamer(brain, monkeypatch, [
        [{"message": {"role": "assistant", "content": "Right away, sir. "}},
         {"message": {"role": "assistant", "content": "", "tool_calls": [
             {"function": {"name": "get_calendar",
                           "arguments": {"range": "tomorrow"}}},
             {"function": {"name": "notes",
                           "arguments": {"action": "add", "text": "milk"}}}]},
          "done": True, "load_duration": 0}],
        [{"message": {"role": "assistant", "content": COVERED},
          "done": True, "load_duration": 0}],
    ], record=record)
    spoken = []
    b._chat_sync(ASKED, on_sentence=spoken.append)
    assert [name for name, *_ in record] == ["get_calendar", "notes"]
    assert len(sent) == 2, "the calendar answer was never phrased"
    assert "11:15" in " ".join(spoken)
    assert spoken[-1] == NOTES_LINE


# ============== JARVIS'S OWN WORDS ARE NOT THE MODEL'S TO BE CAPPED (F26)
# The two spoken caps -- MAX_SPOKEN_SENTENCES (2) and MAX_SPOKEN_CHARS
# (250) -- are a rule about how much PROSE he wants back from a model.
# They were being applied to lines the CODE wrote as well: the
# confirmation a write authored, and the degrade's honest notice about
# what went unspoken beside it. append_spoken_lines has exempted a HELD
# line from exactly this since it was written ("the cap is a rule about
# how much prose he wants, not a licence to drop the news of a write");
# the speak branch and the degrade never got the same exemption.
def test_the_degrade_still_says_what_went_unspoken_after_two_confirmations(
        setup, caplog):
    """"What's the weather tomorrow, add milk to my list and what timers
    do I have": one read owed, a write done, a three-sentence authored
    read-out held beside it, and a render round that writes nothing at
    all. The reply is the degrade -- both authored lines plus the notice
    naming the weather, FIVE sentences -- and it went through the model's
    sentence cap (4 since 09-04, 2 the night this was found), so
    limit_sentences dropped the notice, which is the last sentence. He
    heard about the milk and his timers and nothing whatever about the
    weather he asked for, while the WARNING claimed the sources had been
    named instead."""
    b, fake = setup[0]()
    record = setup[1]
    fake.replies = [tool_reply(("get_weather", {"when": "tomorrow"}),
                               ("notes", {"action": "add", "text": "milk"}),
                               ("manage_schedule", {"action": "list"})),
                    text_reply("   ")]           # the render round says nothing
    with caplog.at_level("WARNING", logger="jarvis.brain"):
        spoken = dict(b._chat_sync(
            "what's the weather tomorrow, add milk to my list and what "
            "timers do I have"))["SPEAK"]
    assert [n for n, *_ in record] == ["get_weather", "notes",
                                       "manage_schedule"]
    assert NOTES_LINE in spoken and SCHEDULE_LINE in spoken
    assert len(brain_mod.split_sentences(spoken)) > brain_mod.MAX_SPOKEN_SENTENCES, \
        "the case only bites past the model's cap; this reply is under it"
    assert brain_mod.tool_only_line(["get_weather"]) in spoken, \
        "the log says the sources were named; the cap had eaten the notice"


def test_two_confirmations_in_one_round_both_survive_the_char_cap(setup):
    """Two writes, nothing owed, so the authored lines end the turn:
    speak = " ".join(authored). The sentence cap was raised for them and
    the CHARACTER cap was not -- MAX_SPOKEN_CHARS on a reply of that many
    sentences -- so a long calendar confirmation ate the timer's. Both
    things happened; he is told about both. The title is sized at run
    time so the pair lands between the prose cap and the TTS hard limit,
    whatever the caps are set to."""
    b, fake = setup[0]()
    timer = "Timer set for ten minutes, sir."
    title = "Dinner with the Staffeldts"
    while len(f"Added {title}, Tuesday at 4:30 PM, to your calendar, sir.") \
            + 1 + len(timer) <= brain_mod.MAX_SPOKEN_CHARS:
        title += " at the place by the river"
    event = f"Added {title}, Tuesday at 4:30 PM, to your calendar, sir."
    assert brain_mod.MAX_SPOKEN_CHARS < len(event) + 1 + len(timer) \
        <= brain_mod.HARD_SPOKEN_CHARS, "the case needs room under the TTS limit"
    fake.replies = [tool_reply(("add_event", {"title": title,
                                              "when": "Tuesday at 4:30 PM"}),
                               ("set_reminder", {"when": "ten minutes"}))]
    spoken = dict(b._chat_sync("put dinner in my calendar and set a timer "
                               "for ten minutes"))["SPEAK"]
    assert event in spoken
    assert timer in spoken, "the second write's confirmation was trimmed off"


def test_a_long_authored_read_out_keeps_its_own_allowance_and_no_more(setup):
    """The per-line rule, both halves. A ten-sentence note read-out is
    still not read whole (2026-08-26, tests/test_found_speak_bypass.py):
    the cap is applied to the LINE, at the larger of the model cap and
    the tool's own allowance. And it is applied to the line only: the
    timer's confirmation beside it is not what pays for the trim."""
    b, fake = setup[0]()
    ten = " ".join(f"Item {n} is on it." for n in range(1, 11))
    long_line = f"Ten on your list, sir. {ten}"
    reg = brain_mod._REGISTRY
    reg.register(ToolSpec("notes_long", "A long read-out.",
                          {"type": "object", "properties": {}},
                          lambda **_: ToolResult(text=long_line,
                                                 speak=long_line)))
    fake.replies = [tool_reply(("notes_long", {}),
                               ("set_reminder", {"when": "ten minutes"}))]
    spoken = dict(b._chat_sync("read me my list and set a timer for ten "
                               "minutes"))["SPEAK"]
    sentences = brain_mod.split_sentences(spoken)
    assert sentences[-1] == "Timer set for ten minutes, sir."
    assert len(sentences) == brain_mod.MAX_SPOKEN_SENTENCES + 1, sentences


def test_the_half_a_result_notice_never_costs_him_a_confirmation(setup):
    """A tool text too long for the context window is cut, and the reply
    then owes him PARTIAL_RESULT_LINE. The model's answer gives up a
    sentence to make room for it -- that is the trade, and it is the
    model's prose to give. Two authored confirmations are not: the notice
    joins them or it is left out."""
    b, fake = setup[0](screen="Active window: Chrome. " + "detail. " * 700)
    fake.replies = [tool_reply(("screen_qa", {"question": "what is this"}),
                               ("set_reminder", {"when": "ten minutes"}))]
    spoken = dict(b._chat_sync("what's on my screen and set a timer for "
                               "ten minutes"))["SPEAK"]
    assert SCREEN_LINE in spoken
    assert "Timer set for ten minutes, sir." in spoken, \
        "the partial-result notice was paid for with a write he was owed"


# ======= A HELD LINE GETS THE GUARDS EVERY OTHER AUTHORED LINE GETS (F27)
# notes list / notes search / timekeeper list put HUNTER'S OWN text on the
# speak= path. Spoken alone it goes through _finish_spoken, which strips
# the markdown and the emoji. Held beside an owed read and appended by the
# code, it went to TTS exactly as he dictated it -- and the same raw line
# was what held_lines_missing compared against the GUARDED reply, so a
# model that copied it verbatim never matched and he heard it twice.
def test_a_held_list_is_guarded_exactly_as_it_is_when_it_is_spoken_alone(setup):
    b, fake = setup[0]()
    fake.replies = [tool_reply(("notes", {"action": "list"}))]
    alone = dict(b._chat_sync("read me my shopping list"))["SPEAK"]
    assert alone == LIST_LINE_SPOKEN
    fake.replies = [tool_reply(("notes", {"action": "list"}),
                               ("get_weather", {"when": "tomorrow"})),
                    text_reply("Heavy showers tomorrow, sir.")]
    held = dict(b._chat_sync("read me my list and what's the weather"))["SPEAK"]
    assert held.endswith(LIST_LINE_SPOKEN), held
    assert "**" not in held and "🥚" not in held


def test_a_held_list_reaches_tts_guarded_on_the_streamed_path_too(brain,  # noqa: F811
                                                                  monkeypatch):
    """The streamed path hands the held line to on_sentence itself, which
    is the one place TTS gets it: raw there is raw in the room."""
    b, sent = _streamer(brain, monkeypatch, [
        _tool_chunk(("notes", {"action": "list"}),
                    ("get_weather", {"when": "tomorrow"})),
        [{"message": {"role": "assistant",
                      "content": "Heavy showers tomorrow, sir."},
          "done": True, "load_duration": 0}]])
    spoken = []
    b._chat_sync("read me my list and what's the weather",
                 on_sentence=spoken.append)
    assert spoken[-1] == LIST_LINE_SPOKEN, spoken


def test_a_reply_that_copies_the_guarded_line_is_not_made_to_say_it_twice(setup):
    """The model is TOLD the held line in HELD_LINE_NOTE and copying it is
    the common case. It copies what it was told, so what it was told has
    to be the line that will actually be spoken -- otherwise the match
    fails on the markdown alone and the code appends a second copy."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("notes", {"action": "list"}),
                               ("get_weather", {"when": "tomorrow"})),
                    text_reply(f"Heavy showers tomorrow, sir. "
                               f"{LIST_LINE_SPOKEN}")]
    spoken = dict(b._chat_sync("read me my list and what's the weather"))["SPEAK"]
    assert spoken.count("shopping list") == 1, spoken
    told = [m["content"] for m in fake.chat_payloads()[-1]["messages"]
            if m["role"] == "user"][-1]
    assert LIST_LINE_SPOKEN in told and "**" not in told


def test_a_model_that_dies_mid_repair_still_says_the_line_guarded(setup):
    """The two early returns that keep a write's news when Ollama dies
    hand back degrade() untouched -- no guards at all on that path."""
    b, fake = setup[0]()
    fake.replies = [tool_reply(("notes", {"action": "list"}),
                               ("get_weather", {"when": "tomorrow"})),
                    brain_mod.OllamaDown("connection refused")]
    spoken = dict(b._chat_sync("read me my list and what's the weather"))["SPEAK"]
    assert LIST_LINE_SPOKEN in spoken
    assert "**" not in spoken and "🥚" not in spoken


# ===== THE CONVENTION THE CHECK TRUSTS, COUNTED RATHER THAN ASSUMED (F28)
# answer_owed reads one convention: reads hand back text and leave the
# phrasing to a round, writes and controls author their own confirmation.
# The comment above it named screen_qa as the ONE deliberate exception.
# That was wrong, and the census below is what it should have said.
def _authored_success_lines():
    """Every ToolResult in jarvis/tools that carries speak= on a path that
    is not a plain failure, as (module, function). Static parse: nothing
    is imported, called, or reached over the network."""
    import ast
    import pathlib

    found = set()
    root = pathlib.Path(brain_mod.__file__).resolve().parent / "tools"
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        holder = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                holder[child] = node
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "ToolResult"):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            ok = kw.get("ok")
            if "speak" not in kw or (isinstance(ok, ast.Constant)
                                     and ok.value is False):
                continue
            up = node
            while up in holder:
                up = holder[up]
                if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.add((path.name, up.name))
                    break
    return found


# Writes and controls: authoring the line is the CONVENTION for these, and
# the coverage check is right to treat them as already answered.
AUTHORED_BY_WRITES = {
    ("calendar.py", "add_event"), ("docs.py", "docs_reindex"),
    ("notes.py", "notes"), ("spotify.py", "play"), ("spotify.py", "control"),
    ("spotify.py", "queue"), ("spotify.py", "radio"), ("spotify.py", "liked"),
    ("spotify.py", "_like"), ("timekeeper.py", "set_reminder"),
    ("timekeeper.py", "set_timer"), ("timekeeper.py", "set_alarm"),
    ("timekeeper.py", "manage_schedule"),
}
# READS that author a success line -- every one of them invisible to
# answer_owed, exactly the way screen_qa's own answer is. Measured
# 2026-09-03; the comment in brain.py claimed there was one.
AUTHORED_BY_READS = {
    ("screen.py", "screen_qa"),            # the documented one
    ("notes.py", "notes"),                 # list / search
    ("timekeeper.py", "manage_schedule"),  # list
    ("spotify.py", "now_playing"),
    ("oracle.py", "oracle_status"),
    ("canvas.py", "canvas_due"),           # "nothing due"
    ("canvas.py", "canvas_grades"),        # "no grades posted"
    ("canvas.py", "canvas_announcements"),
    ("journal.py", "recap_day"),           # "nothing in your journal"
    ("mail.py", "get_mail"),               # "nothing new"
}


def test_the_reads_that_author_their_own_success_line_are_counted():
    """A new name in this failure is a decision, not a nuisance: if what
    authored the line is a READ, the coverage check has just gone blind to
    it and brain.py's convention comment needs it by name."""
    assert _authored_success_lines() == AUTHORED_BY_WRITES | AUTHORED_BY_READS


def test_a_read_that_authors_its_line_is_invisible_to_the_check():
    """Which is the whole reason the list above has to be maintained: the
    check cannot tell such a read from a write, and never could."""
    for authored in (SCREEN_LINE, LIST_LINE):
        assert brain_mod.answer_owed(
            ToolResult(text="data he can hear", speak=authored)) is False
