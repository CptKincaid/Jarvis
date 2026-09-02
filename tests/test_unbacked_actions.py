"""The brain's unbacked-action guard: a reply that claims an action in a
turn where NO tool ran is not spoken as it stands.

LIVE 2026-09-01 20:56:42: "Say hello to my family and then add milk to my
shopping list and then start playing my Spotify." reached gemma4 whole and
it answered "Good evening, Ali and Heather ... I've added milk to your
shopping list, sir, and I'm starting your music now." with ZERO tool calls
-- no 'brain INFO tool' line in the log. Nothing was added, nothing
played, and the narration was spoken with full confidence.

Now: WARNING "brain: unbacked action claim ..." is logged, the model gets
ONE retry with UNBACKED_NUDGE appended to the per-turn messages, and

  * a retry that calls a tool is the normal path from there;
  * a retry that still runs no tool has the claiming sentences of the
    FIRST reply replaced with UNBACKED_LINE, the greeting kept;
  * a reply with no action claim costs nothing: no retry, no log line.

Only when zero tools ran in the turn: a turn that ran a tool is trusted.

Real JarvisBrain over the FakeOllama seam (tests/test_brain_tools.py),
both the plain and the streamed paths. No Ollama, no network, no audio.
"""
import pytest

import jarvis.brain as brain_mod
from tests.test_brain_tools import (FakeContext, FakeMemory, FakeOllama,  # noqa: F401
                                    brain, make_registry, text_reply, tool_reply)

GREETING = "Good evening, Ali and Heather — lovely to have you both."
CLAIM = ("I've added milk to your shopping list, sir, and I'm starting "
         "your music now.")
REPLY = f"{GREETING} {CLAIM}"


@pytest.fixture
def setup(brain, monkeypatch):  # noqa: F811
    brain.reset_static_prompt()
    record = []
    reg = make_registry(record)
    monkeypatch.setattr(brain, "_REGISTRY", reg)
    fake = FakeOllama()
    monkeypatch.setattr(brain, "_http", fake)
    b = brain.JarvisBrain(context=FakeContext(), memory=FakeMemory())
    return b, fake, record


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelname == "WARNING" and "unbacked" in r.getMessage()]


# ------------------------------------------------------------ the table
@pytest.mark.parametrize("line, claim", [
    (REPLY, "I've added"),
    ("I'm starting your music now, sir.", "I'm starting"),
    ("I have set a timer for ten minutes.", "I have set"),
    ("Milk is added to your shopping list.", "added to your"),
    ("Your timer is set, sir.", "Your timer is set"),
    ("The music is on, sir.", "The music is on"),
    ("Playing now, sir.", "Playing now"),
    ("I've just cancelled the alarm.", "I've just cancelled"),
    ("I am adding it to the list.", "I am adding"),
])
def test_action_claims_are_recognised(line, claim):
    assert brain_mod.unbacked_claim(line) == claim


@pytest.mark.parametrize("line", [
    "Quite well, sir.",
    "Good evening, Ali and Heather.",
    "Very good, sir.",
    "Right away, sir.",
    "Shall I add milk to the list, sir?",           # an offer, not a claim
    "I could start the music if you like.",          # conditional
    "It's ten past nine, sir.",
    "Resumed, sir.",                                 # a tool's own line
    "I've been meaning to ask about the thesis.",    # 've + not an action verb
    "I'm afraid I can't reach Spotify, sir.",
])
def test_ordinary_replies_are_not_claims(line):
    assert brain_mod.unbacked_claim(line) is None


def test_the_claiming_sentences_are_replaced_and_the_rest_kept():
    assert brain_mod.strip_unbacked_claims(REPLY) == \
        f"{GREETING} {brain_mod.UNBACKED_LINE}"
    # two claiming sentences: ONE apology, in the first one's place
    text = f"{GREETING} I've added milk. I'm starting your music now. Enjoy your evening."
    assert brain_mod.strip_unbacked_claims(text) == \
        f"{GREETING} {brain_mod.UNBACKED_LINE} Enjoy your evening."
    # the apology survives the spoken cap, at the cost of a trailing sentence
    text = "One. Two. I've added milk. Four."
    assert brain_mod.strip_unbacked_claims(text, 2) == f"One. {brain_mod.UNBACKED_LINE}"
    assert brain_mod.strip_unbacked_claims("One. I've added milk. Three.", 2) == \
        f"One. {brain_mod.UNBACKED_LINE}"
    # nothing claimed: untouched, byte for byte
    assert brain_mod.strip_unbacked_claims("Quite well, sir.") == "Quite well, sir."


# ------------------------------------------------------- the tool loop
def test_the_20_56_reply_is_retried_and_the_claims_replaced(setup, caplog):
    """The live turn replayed: the model narrates twice, no tool ever
    runs. He hears the greeting and the honest line, never the claims."""
    b, fake, record = setup
    fake.replies = [text_reply(REPLY), text_reply(REPLY)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("Say hello to my family and then add milk to my "
                            "shopping list and then start playing my Spotify.")
    assert tags == [("SPEAK", f"{GREETING} {brain_mod.UNBACKED_LINE}")]
    spoken = dict(tags)["SPEAK"]
    assert "starting your music" not in spoken and "added milk" not in spoken
    assert record == []                                   # nothing ran
    assert _warnings(caplog) == [
        "brain: unbacked action claim \"I've added\" (no tool ran)",
        "brain: unbacked action claim stands after the retry (no tool ran); "
        "replacing it"]
    # exactly ONE retry, carrying the first reply and the nudge, WITH the
    # tools still offered (the point is to make it use them)
    p1, p2 = fake.chat_payloads()
    assert [m["role"] for m in p2["messages"]] == ["system", "user", "assistant", "user"]
    assert p2["messages"][2]["content"] == REPLY
    assert p2["messages"][3]["content"] == brain_mod.UNBACKED_NUDGE
    assert p2["tools"] == p1["tools"] and p2["tools"]
    # the static prefix is untouched: the nudge is a per-turn message
    assert p2["messages"][0] == p1["messages"][0]


def test_a_retry_that_calls_a_tool_is_the_normal_path(setup, caplog):
    b, fake, record = setup
    fake.replies = [text_reply(REPLY),
                    tool_reply(("notes", {"action": "add", "text": "milk"}))]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("say hello to my family and add milk to the list")
    assert record == [("notes", "add", "milk")]
    assert tags == [("SPEAK", "Noted, sir.")]                # the tool's line
    assert _warnings(caplog) == [
        "brain: unbacked action claim \"I've added\" (no tool ran)"]
    assert len(fake.chat_payloads()) == 2


def test_a_retry_tool_without_a_speak_line_gets_its_render_round(setup):
    b, fake, record = setup
    fake.replies = [text_reply("Hello there. I've set the weather check going."),
                    tool_reply(("get_weather", {"when": "now"})),
                    text_reply("Seventy-two and partly cloudy, sir.")]
    tags = b._chat_sync("say hi and check the weather")
    assert record == [("get_weather", "now", None)]
    assert tags == [("SPEAK", "Seventy-two and partly cloudy, sir.")]
    assert len(fake.chat_payloads()) == 3


def test_a_reply_with_no_claim_costs_nothing(setup, caplog):
    b, fake, record = setup
    fake.replies = [text_reply("Good evening, Ali and Heather; a pleasure.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("say hello to my family")
    assert tags == [("SPEAK", "Good evening, Ali and Heather; a pleasure.")]
    assert len(fake.chat_payloads()) == 1
    assert _warnings(caplog) == []


def test_a_turn_that_ran_a_tool_is_trusted(setup, caplog):
    """"I've added milk" AFTER the notes tool ran is a report, not a
    claim: the guard looks only at zero-tool turns."""
    b, fake, record = setup
    fake.replies = [tool_reply(("get_weather", {"when": "now"})),
                    text_reply("I've set that aside, sir; it's seventy-two out.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("weather, and note it")
    assert tags == [("SPEAK", "I've set that aside, sir; it's seventy-two out.")]
    assert _warnings(caplog) == []
    assert len(fake.chat_payloads()) == 2


def test_the_forced_path_is_not_second_guessed(setup, caplog):
    """A commander-forced tool ran BEFORE the model spoke: the turn is a
    tool turn, whatever the render round says."""
    b, fake, record = setup
    fake.replies = [text_reply("I've set the weather for you, sir: seventy-two.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("weather", force_tool="get_weather",
                            force_args={"when": "now"})
    assert record == [("get_weather", "now", None)]
    assert tags == [("SPEAK", "I've set the weather for you, sir: seventy-two.")]
    assert _warnings(caplog) == []


def test_with_no_tools_on_offer_the_guard_stays_out_of_it(brain, monkeypatch, caplog):  # noqa: F811
    """No registry -> nothing a retry could use: the reply stands and no
    second round is spent (the old tests in test_streaming_replies build
    exactly this brain)."""
    # registry=None falls back to the module registry, which another test
    # module may have left installed: pin it to none for real
    monkeypatch.setattr(brain, "_REGISTRY", None)
    b = brain.JarvisBrain(None, None, registry=None)
    monkeypatch.setattr(b, "_dynamic_context", lambda text="": ("", ""))
    fake = FakeOllama([text_reply("I've added milk to your list, sir.")])
    monkeypatch.setattr(brain, "_http", fake)
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("add milk")
    assert tags == [("SPEAK", "I've added milk to your list, sir.")]
    assert len(fake.chat_payloads()) == 1 and _warnings(caplog) == []


# ---------------------------------------------------------- streaming
def _chunks(text):
    out = [{"message": {"role": "assistant", "content": text[i:i + 7]}, "done": False}
           for i in range(0, len(text), 7)]
    out.append({"message": {"role": "assistant", "content": ""}, "done": True,
                "load_duration": 0})
    return out


@pytest.fixture
def streamed(setup, monkeypatch):
    """The setup brain with a scripted _http_stream beside the FakeOllama:
    streamed rounds pop from `streams`, plain rounds from fake.replies."""
    b, fake, record = setup
    streams = []

    def stream(path, payload, timeout=None):
        assert streams, "no scripted stream left"
        for chunk in streams.pop(0):
            yield chunk
    monkeypatch.setattr(brain_mod, "_http_stream", stream)
    return b, fake, record, streams


def test_streamed_claims_are_withheld_and_the_honest_line_spoken(streamed, caplog):
    """Live, replies stream (CONFIG.stream_replies): the greeting is spoken
    as it lands, the claim sentence is HELD BACK rather than spoken before
    the round is judged, the retry runs off the stream (its words are tool
    calls or nothing), and UNBACKED_LINE is spoken once in the claim's
    place. STREAMED counts it so the caller does not speak it again."""
    b, fake, record, streams = streamed
    streams.append(_chunks(REPLY))
    fake.replies = [text_reply(REPLY)]                    # the retry, plain
    spoken = []
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("say hello to my family, add milk and start my music",
                            on_sentence=spoken.append)
    assert spoken == [GREETING, brain_mod.UNBACKED_LINE]
    assert tags == [("STREAMED", "2"),
                    ("SPEAK", f"{GREETING} {brain_mod.UNBACKED_LINE}")]
    assert record == []
    assert len(_warnings(caplog)) == 2
    # the retry was the one plain (non-streamed) round, and carried the nudge
    (p,) = fake.chat_payloads()
    assert p["stream"] is False
    assert p["messages"][-1]["content"] == brain_mod.UNBACKED_NUDGE


def test_streamed_retry_that_calls_a_tool_speaks_the_tools_line(streamed):
    b, fake, record, streams = streamed
    streams.append(_chunks(REPLY))
    fake.replies = [tool_reply(("notes", {"action": "add", "text": "milk"}))]
    spoken = []
    tags = b._chat_sync("say hello and add milk", on_sentence=spoken.append)
    assert spoken == [GREETING]                           # the claim never aired
    assert record == [("notes", "add", "milk")]
    assert tags == [("SPEAK", "Noted, sir.")]


def test_streamed_retry_tool_then_render_round_streams_again(streamed):
    b, fake, record, streams = streamed
    streams.append(_chunks("Hello there. I've set the weather check going."))
    fake.replies = [tool_reply(("get_weather", {"when": "now"}))]
    streams.append(_chunks("Seventy-two and partly cloudy, sir."))
    spoken = []
    tags = b._chat_sync("say hi and check the weather", on_sentence=spoken.append)
    assert spoken == ["Hello there.", "Seventy-two and partly cloudy, sir."]
    assert record == [("get_weather", "now", None)]
    assert tags == [("STREAMED", "2"), ("SPEAK", "Seventy-two and partly cloudy, sir.")]


def test_a_streamed_reply_with_no_claim_is_untouched(streamed, caplog):
    b, fake, record, streams = streamed
    streams.append(_chunks("It is ten past nine, sir. The evening is clear."))
    spoken = []
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("what time is it", on_sentence=spoken.append)
    assert spoken == ["It is ten past nine, sir.", "The evening is clear."]
    assert tags == [("STREAMED", "2"),
                    ("SPEAK", "It is ten past nine, sir. The evening is clear.")]
    assert fake.chat_payloads() == [] and _warnings(caplog) == []
