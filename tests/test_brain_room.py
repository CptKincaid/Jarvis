"""How much room the local model gets to think in, and who decides.

This covers the 2026-09-04 change: the four values the brain gives the
model (window, spoken cap, temperature, thinking) plus the guard that
keeps HIS QUESTION in the window, all moved out of module constants and
into ``~/.config/jarvis/assistant.json`` under ``brain``.

The load-bearing test here is test_num_ctx_is_one_value_everywhere and
its neighbour test_config_edited_after_import_changes_nothing. Ollama
keys its loaded runner on num_ctx: a request that asks for a different
one makes the 25 B model RELOAD (measured 8.838 s) and throws the prefix
cache away with it. So "configurable" must never become "per request" --
it is read ONCE at import and a change needs a restart. If someone later
makes num_ctx a per-call argument, these two fail.

No network: brain._http is replaced, and tests/conftest.py blocks port
11434 outright anyway.
"""
import json

import pytest

from jarvis.assistant_config import DEFAULTS, AssistantConfig
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec


@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("brain_room")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JARVIS_LOG_DIR", str(tmp))
        import jarvis.logs as logs
        mp.setattr(logs, "LOG_DIR", tmp)
        mp.setattr(logs, "LOG_FILE", tmp / "jarvis.log")
        import jarvis.config as config
        mp.setattr(config.PATHS, "LOG_DIR", tmp)
        mp.setattr(config.PATHS, "SPEAK_QUEUE", tmp / "speak_queue.txt")
        import jarvis.brain as brain_mod
        assert "vss_voice" not in str(logs.LOG_FILE)
        yield brain_mod


class FakeContext:
    def get_context(self, level):
        return {"level": level}

    def format_for_prompt(self, ctx, spoken=False):
        return "Current time: 4:32 PM, Wednesday"

    def add_exchange(self, user, jarvis):
        pass


class FakeMemory:
    def format_for_context(self, text=""):
        return "Known facts (1):\n  editor: vim"

    def log_habit(self, text):
        pass


class FakeOllama:
    def __init__(self):
        self.replies = []
        self.calls = []

    def __call__(self, path, payload=None, timeout=None):
        self.calls.append((path, payload, timeout))
        if path == "/api/ps":
            return {"models": []}
        if path == "/api/generate":
            return {"done": True}
        if path == "/api/chat":
            if not self.replies:
                raise AssertionError("no scripted reply left")
            return self.replies.pop(0)
        raise AssertionError(f"unexpected path {path}")

    def chat_payloads(self):
        return [p for path, p, _ in self.calls if path == "/api/chat"]


def text_reply(content, **extra):
    d = {"model": "fake",
         "message": {"role": "assistant", "content": content},
         "done": True, "load_duration": 1_000_000,
         "prompt_eval_count": 1200, "eval_count": 14}
    d.update(extra)
    return d


def tool_reply(name, args):
    return text_reply("", message={
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": args}}]})


def make_registry(long_text="x"):
    reg = ToolRegistry()
    reg.register(ToolSpec("get_calendar", "Today's events.",
                          {"type": "object", "properties": {}},
                          lambda **_: ToolResult(text=long_text)))
    return reg


@pytest.fixture
def setup(brain, monkeypatch):
    brain.reset_static_prompt()
    reg = make_registry()
    monkeypatch.setattr(brain, "_REGISTRY", reg)
    fake = FakeOllama()
    monkeypatch.setattr(brain, "_http", fake)
    b = brain.JarvisBrain(context=FakeContext(), memory=FakeMemory())
    return b, fake


class FakeConfig:
    """Just the dotted read model_settings uses."""

    def __init__(self, values=None):
        self._values = dict(values or {})

    def get(self, dotted, default=None):
        return self._values.get(dotted, default)


# ------------------------------------------------------------- defaults
def test_shipped_defaults_are_the_planned_values(brain):
    """The plan's numbers, in the file he actually edits.

    16384 is the largest context measured loading on this box (+0.19 GB of
    KV cache, +0.011 s a turn); 160 is the deliberate brevity cap on the
    SPOKEN answer, which has never once bound; think stays off because at
    160 it returned an empty reply 6 times out of 6.
    """
    assert DEFAULTS["brain"] == {
        "num_ctx": 16384,
        "num_predict": 160,
        "temperature": 0.7,
        "think": False,
        "answer_reserve_tokens": 128,
        "protect_question": True,
    }
    # ...and the code's own fallback matches the file's, so a config that
    # cannot be read at all still starts him on the same numbers.
    d = brain.ModelSettings()
    assert (d.num_ctx, d.num_predict, d.temperature, d.think,
            d.answer_reserve_tokens, d.protect_question) == \
        (16384, 160, 0.7, False, 128, True)


def test_defaults_are_in_force_in_this_process(brain):
    assert brain.NUM_CTX == brain.SETTINGS.num_ctx == 16384
    assert brain.CHAT_OPTIONS == {"num_ctx": 16384, "temperature": 0.7,
                                  "num_predict": 160,
                                  "stop": ["\nUser:", "\nHunter:"]}
    assert brain.OLLAMA_OPTIONS is brain.CHAT_OPTIONS


def test_a_real_config_file_drives_the_settings(brain, tmp_path):
    """End to end through AssistantConfig, not just the fake."""
    path = tmp_path / "assistant.json"
    path.write_text(json.dumps({"brain": {"num_ctx": 32768,
                                          "num_predict": 320,
                                          "temperature": 0.2,
                                          "protect_question": False}}),
                    encoding="utf-8")
    got = brain.model_settings(AssistantConfig.load(path))
    assert got.num_ctx == 32768 and got.num_predict == 320
    assert got.temperature == 0.2 and got.protect_question is False
    assert got.think is False           # untouched keys keep the default


def test_config_values_are_read(brain):
    got = brain.model_settings(FakeConfig({
        "brain.num_ctx": 8192, "brain.num_predict": 400,
        "brain.temperature": 0.1, "brain.think": True,
        "brain.answer_reserve_tokens": 64,
        "brain.protect_question": False}))
    assert got.num_ctx == 8192 and got.num_predict == 400
    assert got.temperature == 0.1 and got.think is True
    assert got.answer_reserve_tokens == 64 and got.protect_question is False


@pytest.mark.parametrize("values", [
    {"brain.num_ctx": "banana"},        # not a number
    {"brain.num_ctx": 512},             # below the rail
    {"brain.num_ctx": 999_999_999},     # past the model's own 262144
    {"brain.num_ctx": None},
])
def test_a_bad_num_ctx_falls_back_rather_than_wedging_the_box(brain, values):
    """A typo in his config must not ask Ollama for a context this box
    cannot hold: on 2026-08-28 an exhausted unified pool took a hard
    power-off. Outside the rails the default is used and the reason
    logged; the brain still starts."""
    assert brain.model_settings(FakeConfig(values)).num_ctx == 16384


@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False), ("true", True), ("off", False),
    ("maybe", False), (None, False), (3, False),
])
def test_think_only_turns_on_for_a_real_yes(brain, raw, expected):
    assert brain.model_settings(
        FakeConfig({"brain.think": raw})).think is expected


def test_bad_numbers_elsewhere_fall_back(brain):
    got = brain.model_settings(FakeConfig({
        "brain.num_predict": 0, "brain.temperature": 9.0,
        "brain.answer_reserve_tokens": -5}))
    assert (got.num_predict, got.temperature,
            got.answer_reserve_tokens) == (160, 0.7, 128)


# --------------------------------------------------- THE ONE-VALUE RULE
def test_num_ctx_is_one_value_everywhere(brain, setup):
    """Every request this module makes sends the SAME num_ctx.

    A different one per request would make Ollama reload the 25 B model --
    ~7-9 s, and it drops the prefix cache too. This walks every entry
    point that talks to /api/chat.
    """
    b, fake = setup
    fake.replies = [text_reply("Quite well, sir."),          # chat
                    text_reply('{"route": "local", "confidence": 0.9}'),
                    text_reply("A short summary."),          # summarize
                    text_reply("Right away, sir."),          # local_line
                    text_reply("")]                          # warm-up
    b._chat_sync("How are you?")
    b.classify_route("what's the weather")
    b.summarize("a wall of text " * 40)
    b.local_line("acknowledge", "the router work", fallback="Right away.")
    brain._RESIDENCY["unloaded_once"] = False
    brain.ensure_resident(first=True)

    payloads = fake.chat_payloads()
    assert len(payloads) == 5
    assert {p["options"]["num_ctx"] for p in payloads} == {brain.NUM_CTX}
    # num_predict, by contrast, IS per request on purpose: a route
    # classification and a spoken reply want different lengths, and
    # changing it does not reload anything.
    assert len({p["options"]["num_predict"] for p in payloads}) > 1


def test_an_override_cannot_change_num_ctx(brain):
    """_options() re-asserts it last, so no caller can reach past it."""
    assert brain._options(num_ctx=99, num_predict=40)["num_ctx"] == \
        brain.NUM_CTX
    assert brain._options(num_predict=40)["num_predict"] == 40


def test_config_edited_after_import_changes_nothing(brain, setup, tmp_path,
                                                    monkeypatch):
    """A change needs a RESTART, and that is the design.

    Rebuilding SETTINGS while the process runs would send a num_ctx the
    loaded runner does not match, which is the reload this whole invariant
    exists to prevent -- and on the live server three of four attempts to
    do it hung indefinitely.
    """
    b, fake = setup
    before = brain.NUM_CTX
    path = tmp_path / "assistant.json"
    path.write_text(json.dumps({"brain": {"num_ctx": 4096}}),
                    encoding="utf-8")
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(path))
    fake.replies = [text_reply("Quite well, sir.")]
    b._chat_sync("How are you?")
    assert brain.NUM_CTX == before
    assert fake.chat_payloads()[0]["options"]["num_ctx"] == before


def test_think_is_config_driven_and_off_by_default(brain, setup):
    b, fake = setup
    fake.replies = [text_reply("Quite well, sir.")]
    b._chat_sync("How are you?")
    assert fake.chat_payloads()[0]["think"] is False
    assert fake.chat_payloads()[0]["think"] is brain.SETTINGS.think


# ------------------------------------------------- the question guard
def test_prompt_ceiling_leaves_room_for_the_answer(brain):
    s = brain.ModelSettings(num_ctx=16384, num_predict=160,
                            answer_reserve_tokens=128)
    assert s.prompt_ceiling == 16096
    # A window smaller than the answer it is asked for still leaves a
    # usable floor rather than a negative ceiling.
    assert brain.ModelSettings(num_ctx=2048, num_predict=8192,
                               answer_reserve_tokens=128).prompt_ceiling == 1024


def test_guard_drops_the_oldest_tool_result_not_his_question(brain):
    """The measured failure, in miniature.

    LIVE 2026-09-04: a 9 000-char calendar result took prompt_eval_count
    from 8253 DOWN to 7754 -- exactly the 499 tokens of his question,
    background and memory. Ollama deletes whole messages oldest-first
    after the system prompt, so what it takes out is HIM. The guard drops
    a tool result instead, and says so in the log.
    """
    question = "what's on my calendar and what did that email say?"
    messages = [
        {"role": "system", "content": "PERSONA " * 100},
        {"role": "user", "content": question},
        {"role": "tool", "content": "old calendar " * 400,
         "tool_name": "get_calendar"},
        {"role": "tool", "content": "newer mail " * 400,
         "tool_name": "get_mail"},
    ]
    ceiling = brain.estimate_prompt_tokens(messages) - 200
    dropped, estimate = brain.fit_prompt(messages, ceiling=ceiling)

    assert dropped == 1
    assert messages[2]["content"] == brain.TOOL_DROPPED_TEXT   # the OLDEST
    assert messages[3]["content"].startswith("newer mail")     # the newest
    assert messages[1]["content"] == question                  # HIS QUESTION
    assert messages[0]["content"].startswith("PERSONA")
    assert estimate <= ceiling


def test_guard_drops_more_than_one_when_it_has_to(brain):
    messages = [{"role": "system", "content": "P" * 100},
                {"role": "user", "content": "his question"}]
    for i in range(4):
        messages.append({"role": "tool", "content": f"result {i} " * 300,
                         "tool_name": f"tool{i}"})
    dropped, _ = brain.fit_prompt(messages, ceiling=600)
    assert dropped >= 2
    assert messages[2]["content"] == brain.TOOL_DROPPED_TEXT
    assert messages[1]["content"] == "his question"


def test_guard_leaves_a_prompt_that_fits_completely_alone(brain):
    messages = [{"role": "system", "content": "P"},
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "72F and sunny",
                 "tool_name": "get_weather"}]
    dropped, estimate = brain.fit_prompt(messages)
    assert dropped == 0
    assert messages[2]["content"] == "72F and sunny"
    assert estimate < brain.SETTINGS.prompt_ceiling


def test_guard_can_be_switched_off(brain, monkeypatch):
    """brain.protect_question false = the old behaviour, where Ollama
    makes room its own way. Off is a choice he can make; it is not the
    default, because its failure mode is silent."""
    monkeypatch.setattr(brain, "SETTINGS",
                        brain.ModelSettings(protect_question=False))
    messages = [{"role": "system", "content": "P"},
                {"role": "user", "content": "his question"},
                {"role": "tool", "content": "z" * 200_000,
                 "tool_name": "get_calendar"}]
    dropped, _ = brain.fit_prompt(messages)
    assert dropped == 0
    assert len(messages[2]["content"]) == 200_000


def test_guard_gives_up_loudly_when_nothing_is_left_to_drop(brain, caplog):
    """The system prompt, the schemas and his turn are all that remain --
    and his turn is the thing the guard exists to keep. It warns rather
    than deleting him."""
    messages = [{"role": "system", "content": "P" * 40_000},
                {"role": "user", "content": "his question"}]
    with caplog.at_level("WARNING"):
        dropped, _ = brain.fit_prompt(messages, ceiling=100)
    assert dropped == 0
    assert messages[1]["content"] == "his question"
    assert any("still" in r.message and "Raise brain.num_ctx" in r.message
               for r in caplog.records)


def test_the_guard_runs_on_every_round_of_a_real_turn(brain, setup,
                                                      monkeypatch):
    """Not just reachable -- wired into the loop, before each request."""
    seen = []
    real = brain.fit_prompt

    def spy(messages, tools=None, ceiling=None):
        seen.append(len(messages))
        return real(messages, tools, ceiling)

    monkeypatch.setattr(brain, "fit_prompt", spy)
    b, fake = setup
    fake.replies = [tool_reply("get_calendar", {}),
                    text_reply("Nothing today, sir.")]
    b._chat_sync("what's on my calendar?")
    assert len(seen) == 2               # one per round
    assert seen[1] > seen[0]            # and the transcript had grown


# --------------------------------------------------- making it visible
def test_startup_line_names_the_settings_and_the_room_left(brain, caplog):
    brain._SETTINGS_LOGGED.clear()
    reg = make_registry()
    with caplog.at_level("INFO"):
        brain.log_settings(reg)
    lines = [r.message for r in caplog.records if "window" in r.message]
    assert len(lines) == 1
    line = lines[0]
    assert f"window {brain.SETTINGS.num_ctx} tokens" in line
    assert f"capped at {brain.SETTINGS.num_predict}" in line
    assert "thinking off" in line
    assert "left for his question" in line
    assert "needs a restart" in line
    # and only once per process, like the tool-budget report
    caplog.clear()
    with caplog.at_level("INFO"):
        brain.log_settings(reg)
    assert not [r for r in caplog.records if "window" in r.message]


def test_the_startup_line_is_emitted_when_the_prompt_is_first_built(brain,
                                                                    caplog):
    brain._SETTINGS_LOGGED.clear()
    with caplog.at_level("INFO"):
        brain._registry_schemas(make_registry())
    assert any("window" in r.message for r in caplog.records)


def test_each_round_logs_what_the_prompt_actually_cost(brain, caplog):
    """prompt_eval_count is in every /api/chat body and appeared in no log
    at all before 2026-09-04 -- so the one number that says whether his
    question survived was thrown away every turn."""
    with caplog.at_level("INFO"):
        brain._log_round_tokens(text_reply("hi"), estimated=1180)
    line = [r.message for r in caplog.records if r.message.startswith("ctx:")]
    assert len(line) == 1
    assert f"prompt 1200/{brain.NUM_CTX}" in line[0]
    assert "estimated 1180" in line[0]


def test_a_nearly_full_window_is_a_warning_not_a_note(brain, caplog):
    nearly = int(brain.NUM_CTX * 0.97)
    with caplog.at_level("INFO"):
        brain._log_round_tokens(text_reply("hi", prompt_eval_count=nearly))
    hits = [r for r in caplog.records if r.message.startswith("ctx:")]
    assert len(hits) == 1 and hits[0].levelname == "WARNING"
    assert "his question" in hits[0].message


def test_round_token_logging_never_raises_on_a_junk_body(brain):
    for body in ({}, {"prompt_eval_count": None},
                 {"prompt_eval_count": "lots"}, None, []):
        brain._log_round_tokens(body)
