"""How much room the local model gets to remember in, and who decides.

This covers the 2026-09-04 change: the four values the brain gives the
model (window, per-round generation budget, temperature, thinking) plus the guard that
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
    KV cache, +0.011 s a turn); 160 is the per-round generation budget
    (reply text plus tool-call JSON), which has never once bound; think
    stays off because at 160 it returned an empty reply 6 times out of 6.
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
    # 16384 - 160 (generation) - 128 (reserve) - 655 (4% estimate margin).
    # Round 2 kept 288 tokens of margin against an estimate whose measured
    # error on the static prefix alone was ~930 tokens at that ceiling.
    assert s.prompt_ceiling == 16384 - 160 - 128 - int(16384 * 0.04) == 15441
    # A window smaller than the answer it is asked for still leaves a
    # usable floor rather than a negative ceiling.
    assert brain.ModelSettings(num_ctx=2048, num_predict=8192,
                               answer_reserve_tokens=128).prompt_ceiling == 1024


def test_guard_cuts_the_largest_tool_result_not_his_question(brain):
    """The measured failure, in miniature.

    LIVE 2026-09-04: a 9 000-char calendar result took prompt_eval_count
    from 8253 DOWN to 7754 -- exactly the 499 tokens of his question,
    background and memory. Ollama deletes whole messages oldest-first
    after the system prompt, so what it takes out is HIM. The guard cuts
    the TAIL of the largest tool result instead (round 3: it used to drop
    the whole result, which for the only result of a turn meant the
    model saw nothing of the calendar it was asked about; the round-3
    fix: largest, not oldest, and cut again rather than drop), keeps the
    other whole, and says so in the log.
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
    ceiling = brain.calibrated(brain.estimate_prompt_tokens(messages)) - 200
    dropped, estimate = brain.fit_prompt(messages, ceiling=ceiling)

    assert dropped == 0                                        # cut, not dropped
    assert messages[2]["content"].startswith("old calendar")   # the LARGEST, its head
    assert messages[2]["content"].endswith(brain.TOOL_CUT_TEXT)
    assert len(messages[2]["content"]) < len("old calendar " * 400)
    assert messages[3]["content"] == "newer mail " * 400       # the newest, whole
    assert messages[1]["content"] == question                  # HIS QUESTION
    assert messages[0]["content"].startswith("PERSONA")
    assert estimate <= ceiling


def test_a_small_result_survives_when_a_large_one_can_absorb_the_overflow(brain):
    """Round 3 as shipped went oldest-first: a 300-char weather result
    that could not absorb a 300-token overflow was DROPPED whole, and only
    then was the 5 600-char calendar behind it cut. The round-3 review's
    point, in miniature: a result is never dropped while a large one has
    trimmable tokens. Largest first: the calendar absorbs it all, the
    weather is untouched."""
    messages = [
        {"role": "system", "content": "PERSONA " * 100},
        {"role": "user", "content": "his question"},
        {"role": "tool", "content": "tiny " * 60,           # 300 chars
         "tool_name": "get_weather"},
        {"role": "tool", "content": "long calendar " * 400,  # 5 600 chars
         "tool_name": "get_calendar"},
    ]
    ceiling = brain.calibrated(brain.estimate_prompt_tokens(messages)) - 300
    dropped, estimate = brain.fit_prompt(messages, ceiling=ceiling)
    assert dropped == 0
    assert messages[2]["content"] == "tiny " * 60             # whole
    assert messages[3]["content"].startswith("long calendar")
    assert messages[3]["content"].endswith(brain.TOOL_CUT_TEXT)
    assert messages[1]["content"] == "his question"
    assert estimate <= ceiling


def test_a_cut_keeps_the_sentence_allowance_that_rides_at_the_end(brain):
    """tool_message() appends SENTENCE_ALLOWANCE to a list-shaped result,
    at the END. A tail cut that ate it would leave a partial calendar the
    model may name only two items of; the cut lifts it off, cuts the data
    in front of it, and puts it back after the marker."""
    allowance = brain.SENTENCE_ALLOWANCE.format(n=4)
    content = "09:00 lecture\n" * 400 + allowance
    messages = [{"role": "user", "content": "his question"},
                {"role": "tool", "content": content,
                 "tool_name": "get_calendar"}]
    ceiling = brain.calibrated(brain.estimate_prompt_tokens(messages)) - 200
    dropped, est = brain.fit_prompt(messages, ceiling=ceiling)
    got = messages[1]["content"]
    assert dropped == 0 and est <= ceiling
    assert got.startswith("09:00 lecture\n")
    assert got.endswith(brain.TOOL_CUT_TEXT + allowance)
    assert got.count(allowance) == 1 and got.count(brain.TOOL_CUT_TEXT) == 1
    assert messages[0]["content"] == "his question"
    # a second pass finds it fits and leaves the cut result alone
    assert brain.fit_prompt(messages, ceiling=ceiling)[0] == 0
    assert messages[1]["content"] == got


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

    def spy(messages, tools=None, ceiling=None, **kw):
        seen.append(len(messages))
        return real(messages, tools, ceiling, **kw)

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


# ------------------------------------------------------------------------
# 2026-09-04 review, five blocking findings. Each test below fails when its
# fix is reverted; the docstrings say which.
# ------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LIVE_STATIC_PREFIX_TOKENS = 3761   # measured: prompt_eval_count, tools-only
LIVE_TURN_TOKENS = 499             # measured: question + background + memory
LIVE_RESULT_CHARS = 9000           # the calendar result that did the damage


def _tiny_window(brain, monkeypatch, num_ctx=3072):
    """A window the persona alone nearly fills, so any real material must
    be trimmed. Only SETTINGS is replaced: NUM_CTX (the value on the wire)
    is left alone, which the one-value tests further up depend on. 3072,
    not 2048: with the round-3 margin and calibration a 2048 window leaves
    the ~1 500-token persona no room for even a one-line question."""
    s = brain.ModelSettings(num_ctx=num_ctx)
    monkeypatch.setattr(brain, "SETTINGS", s)
    return s


def _ctx_lines(caplog, label):
    return [r for r in caplog.records
            if r.message.startswith("ctx:") and f"[{label}]" in r.message]


# ------------------------------------- finding 2: two rates, not one
def test_dense_tool_text_is_costed_at_its_measured_rate(brain):
    """Finding 2. LIVE 2026-09-04: a 9 000-char calendar result cost 3 993
    prompt tokens (8253 minus the 4260 of prefix and turn), i.e. 2.25 chars
    per token. The registry's 4.1 is right for prose and schemas and wrong
    by 1.8x for this, in the direction that lets the window overflow."""
    dense = brain.estimate_prompt_tokens(
        [{"role": "tool", "content": "x" * LIVE_RESULT_CHARS}])
    assert 3800 <= dense <= 4200, dense
    prose = brain.estimate_prompt_tokens(
        [{"role": "user", "content": "x" * LIVE_RESULT_CHARS}])
    assert 2100 <= prose <= 2300, prose
    assert brain.TOOL_CHARS_PER_TOKEN == 2.25
    assert brain.PROSE_CHARS_PER_TOKEN == 4.1


def test_the_guard_fires_on_the_live_turn_it_was_built_for(brain, monkeypatch):
    """Finding 2, the consequence. The measured turn in numbers: an 8192
    window (ceiling 7577 = 8192 - 160 - 128 - 327), the 3761-token static
    prefix, the 499-token turn, and the 9 000-char result. Ollama measured
    it at 8253 and deleted his question. Costed at 4.1 the estimate came
    to ~6500 and the guard SLEPT through the very turn it exists for; at
    2.25 it fires -- and since round 3 it CUTS the calendar's tail rather
    than dropping the whole of it, so the morning survives."""
    monkeypatch.setattr(brain, "SETTINGS", brain.ModelSettings(num_ctx=8192))
    assert brain.SETTINGS.prompt_ceiling == 8192 - 160 - 128 - 327 == 7577
    line = "09:00 BMEN 427 lecture, room 1.14; 10:30 office hours\n"
    result = (line * (LIVE_RESULT_CHARS // len(line) + 1))[:LIVE_RESULT_CHARS]
    msgs = [{"role": "system",
             "content": "p" * int(LIVE_STATIC_PREFIX_TOKENS * 4.1)},
            {"role": "user", "content": "q" * int(LIVE_TURN_TOKENS * 4.1)},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_calendar", "arguments": {}}}]},
            {"role": "tool", "tool_name": "get_calendar", "content": result}]
    before = brain.estimate_prompt_tokens(msgs)
    assert before > 7577, before                 # it really is over
    assert abs(before - 8253) < 300, before      # and close to the real count
    dropped, after = brain.fit_prompt(msgs)
    assert dropped == 0                          # cut, not dropped
    assert after <= 7577
    assert msgs[1]["content"].startswith("q")    # his question, untouched
    assert msgs[3]["content"].startswith(line)   # the morning, kept
    assert msgs[3]["content"].endswith(brain.TOOL_CUT_TEXT)
    assert 5000 < len(msgs[3]["content"]) < LIVE_RESULT_CHARS


# ---------------------------- finding 1: every path, not the smallest one
def test_summarize_goes_through_the_guard_and_logs_the_real_count(
        brain, setup, monkeypatch, caplog):
    """Finding 1. summarize() carries a Claude result or a mail digest in
    ONE user message; nothing in it is a tool result, so the guard trims
    the material's tail and keeps the instruction in front of it."""
    b, fake = setup
    _tiny_window(brain, monkeypatch)
    digest = ("From: registrar@tamu.edu  Subject: BMEN 427 grade posted\n"
              * 400)                               # 20 000 dense chars
    fake.replies = [text_reply("Your grade is posted, sir.",
                               prompt_eval_count=1750)]
    with caplog.at_level("INFO"):
        out = brain.summarize(digest, max_sentences=2)
    assert out == "Your grade is posted, sir."
    sent = fake.chat_payloads()[-1]["messages"][-1]["content"]
    assert sent.startswith("Background:")          # the instruction survived
    assert sent.endswith(brain.MATERIAL_CUT_TEXT)  # the material was cut
    assert len(sent) < len(digest)
    assert brain.estimate_prompt_tokens(
        fake.chat_payloads()[-1]["messages"],
        fake.chat_payloads()[-1].get("tools")) <= brain.SETTINGS.ceiling_for(120)
    lines = _ctx_lines(caplog, "persona")
    assert len(lines) == 1 and "prompt 1750/" in lines[0].message


def test_local_line_material_that_fits_is_sent_untouched(brain, setup, caplog):
    b, fake = setup
    fake.replies = [text_reply("Right away, sir.")]
    with caplog.at_level("INFO"):
        assert brain.local_line("Acknowledge.", "open the calendar") == \
            "Right away, sir."
    sent = fake.chat_payloads()[-1]["messages"][-1]["content"]
    assert "open the calendar" in sent
    assert brain.MATERIAL_CUT_TEXT not in sent
    assert len(_ctx_lines(caplog, "persona")) == 1


def test_classify_route_goes_through_the_guard(brain, setup, monkeypatch,
                                               caplog):
    """Finding 1. The router's tie-breaker: a normal utterance is untouched,
    and its prompt_eval_count is logged like every other request's."""
    b, fake = setup
    _tiny_window(brain, monkeypatch)
    fake.replies = [text_reply(json.dumps({"route": "local",
                                           "confidence": 0.9}),
                               prompt_eval_count=1600)]
    with caplog.at_level("INFO"):
        assert brain.classify_route("what's the weather like") == \
            ("local", 0.9)
    sent = fake.chat_payloads()[-1]["messages"][-1]["content"]
    assert sent.endswith("what's the weather like")
    assert len(_ctx_lines(caplog, "route")) == 1
    # an absurd transcript (a pasted page) is cut rather than overflowing
    fake.replies = [text_reply(json.dumps({"route": "local",
                                           "confidence": 0.5}))]
    brain.classify_route("word " * 8000)
    sent = fake.chat_payloads()[-1]["messages"][-1]["content"]
    assert sent.endswith(brain.MATERIAL_CUT_TEXT)
    assert brain.ROUTE_INSTRUCTION[:40] in sent


def test_explain_text_goes_through_the_guard(brain, setup, monkeypatch,
                                             caplog):
    """Finding 1. The JSON helpers write their material LAST in the
    instruction (explain_text, make_quiz, read_syllabus, extract_facts,
    grade_answer all do), so the tail-trim cuts the document, never the
    instruction that says what to do with it."""
    b, fake = setup
    # 4096, not 2048: explain_text asks for 420 tokens of answer, and at
    # 2048 the persona alone leaves nothing for the document.
    _tiny_window(brain, monkeypatch, num_ctx=4096)
    doc = "The lab measures impedance across a range of frequencies. " * 200
    fake.replies = [text_reply(json.dumps({"lead": "A lab handout, sir.",
                                           "summary": "It measures things."}),
                               prompt_eval_count=1500)]
    with caplog.at_level("INFO"):
        lead, summary = brain.explain_text(doc, name="lab.pdf")
    assert lead == "A lab handout, sir."
    sent = fake.chat_payloads()[-1]["messages"][-1]["content"]
    assert sent.startswith("Instruction for Jarvis")
    assert "Document:\n" in sent
    assert sent.endswith(brain.MATERIAL_CUT_TEXT)
    assert len(sent) < len(doc)
    assert fake.chat_payloads()[-1].get("tools") in (None, [])
    assert len(_ctx_lines(caplog, "json")) == 1


def test_both_warm_ups_log_the_real_static_prefix_cost(brain, setup,
                                                      monkeypatch, caplog):
    """Finding 1 and 2. The warm-ups send the static prefix and nothing
    else, so their prompt_eval_count IS the prefix's real cost -- the one
    number the startup estimate can be checked against. It was thrown
    away on both paths."""
    b, fake = setup
    monkeypatch.setitem(brain._RESIDENCY, "lent", False)
    monkeypatch.setitem(brain._RESIDENCY, "unloaded_once", True)
    fake.replies = [text_reply("", prompt_eval_count=3556),
                    text_reply("", prompt_eval_count=3556)]
    with caplog.at_level("INFO"):
        assert brain.warm_static() is True
        assert brain.ensure_resident(first=False) is True
    assert len(_ctx_lines(caplog, "rewarm")) == 1
    assert len(_ctx_lines(caplog, "warm")) == 1
    assert all("prompt 3556/" in r.message
               for r in _ctx_lines(caplog, "warm") + _ctx_lines(caplog, "rewarm"))
    for payload in fake.chat_payloads()[-2:]:
        assert payload["options"]["num_predict"] == 1
        assert payload["options"]["num_ctx"] == brain.NUM_CTX


def test_every_chat_request_in_the_module_goes_through_a_guard(brain):
    """Finding 1, structurally. Only the tool loop (fit_prompt before,
    _log_round_tokens after, streamed or not) and _chat_once may POST to
    /api/chat. A new caller that builds its own payload and posts it has
    skipped the guard, and this counts it."""
    src = Path(brain.__file__).read_text(encoding="utf-8")
    posts = [ln.strip() for ln in src.splitlines()
             if '"/api/chat"' in ln
             and ("_http(" in ln or "_http_stream(" in ln)]
    # _chat_once's own post, the tool loop's plain round, its streamed round
    assert len(posts) == 3, posts
    helper = src.split("def _chat_once(", 1)[1].split("\ndef ", 1)[0]
    assert '_http("/api/chat"' in helper
    assert "fit_material(" in helper and "_log_round_tokens(" in helper
    # and the one-shot callers all name it
    for fn in ("warm_static", "ensure_resident", "_persona_request",
               "classify_route", "_json_request"):
        body = src.split(f"\ndef {fn}(", 1)[1].split("\ndef ", 1)[0]
        assert "_chat_once(" in body, fn


# ------------------------------ finding 3: what num_predict really is
def test_num_predict_is_no_longer_called_the_spoken_cap(brain):
    """Finding 3. num_predict caps what the model GENERATES in a round;
    what is SPOKEN is clamped by MAX_SPOKEN_SENTENCES / MAX_SPOKEN_CHARS.
    The dataclass, the config defaults and the setup doc all said the
    former was the latter."""
    import jarvis.assistant_config as ac
    sources = {
        "brain.py": Path(brain.__file__).read_text(encoding="utf-8"),
        "assistant_config.py": Path(ac.__file__).read_text(encoding="utf-8"),
        "assistant-setup.md": (REPO / "docs" / "assistant-setup.md")
        .read_text(encoding="utf-8"),
        # Round 2's review found the old wording surviving in the TESTS
        # (this file's own docstring, a comment in test_brain_tools.py),
        # which this grep did not scan. Now it does.
        "test_brain_room.py": Path(__file__).read_text(encoding="utf-8"),
        "test_brain_tools.py": (REPO / "tests" / "test_brain_tools.py")
        .read_text(encoding="utf-8"),
    }
    # spelt in two pieces, so this file never contains its own banned text
    wrong = tuple(a + b for a, b in (
        ("cap on the ", "SPOKEN answer"), ("spoken answer ", "capped"),
        ("how long he is ", "allowed to speak"),
        ("cap on the ", "spoken answer"), ("window, ", "spoken cap"),
        ("the window and the ", "spoken cap"),
        ("local model gets to ", "think in")))
    for name, text in sources.items():
        for phrase in wrong:
            assert phrase not in text, (name, phrase)
    assert "MAX_SPOKEN_SENTENCES" in sources["assistant_config.py"]
    assert "MAX_SPOKEN" in sources["assistant-setup.md"]
    # the spoken clamp really is elsewhere, and smaller than the budget
    assert brain.MAX_SPOKEN_SENTENCES == 4
    assert brain.MAX_SPOKEN_CHARS < brain.SETTINGS.num_predict * 4.1


def test_startup_line_says_what_the_budget_is(brain, caplog):
    brain._SETTINGS_LOGGED.clear()
    with caplog.at_level("INFO"):
        brain.log_settings(make_registry())
    line = [r.message for r in caplog.records if "window" in r.message][0]
    assert "generation per round capped at" in line
    assert "speech is clamped separately" in line
    assert "spoken answer" not in line


# ------------------------ finding 4: what the estimate itself costs
def test_schema_cost_is_computed_once_per_tool_set(brain, monkeypatch):
    """Finding 4. The per-round json.dumps of messages+tools was MEASURED
    at 0.055 ms for a 30 KB round on this box (2026-09-04) -- nothing
    against a 1.3 s turn. The schema half is byte-stable per tool set, so
    it is dumped once and cached; the messages are walked, not dumped."""
    import types
    brain._SCHEMA_TOKENS_CACHE.clear()
    dumped = []
    real = json.dumps

    def counting(obj, *a, **k):
        dumped.append(obj)
        return real(obj, *a, **k)

    monkeypatch.setattr(brain, "json",
                        types.SimpleNamespace(dumps=counting, loads=json.loads))
    tools_a = make_registry().schemas()
    tools_b = make_registry().schemas()          # equal, not identical
    msgs = [{"role": "system", "content": "s" * 4000},
            {"role": "tool", "content": "t" * 9000}]
    first = brain.estimate_prompt_tokens(msgs, tools_a)
    second = brain.estimate_prompt_tokens(msgs, tools_b)
    assert first == second > 0
    assert len(brain._SCHEMA_TOKENS_CACHE) == 1
    assert sum(1 for d in dumped if isinstance(d, list)) == 1   # tools once
    assert not any(d is msgs for d in dumped)                   # never the transcript


# -------------------- finding 5: room to REMEMBER, not room to think
def test_the_docs_sell_the_window_as_memory_not_thought():
    """Finding 5. Doubling num_ctx buys room to REMEMBER (a long calendar
    and a long inbox in one turn, more history); thinking is `think`, and
    it is off. The premise correction stays with it: 1 truncation in
    5,328 prompts over a week."""
    doc = (REPO / "docs" / "assistant-setup.md").read_text(encoding="utf-8")
    assert "how much room he gets to remember in" in doc
    assert "how much room he gets to think in" not in doc
    assert "5,328" in doc
    import jarvis.assistant_config as ac
    cfg_src = Path(ac.__file__).read_text(encoding="utf-8")
    assert "ROOM THE LOCAL MODEL GETS TO REMEMBER IN" in cfg_src


# ------------------------------------------ the tail-trim, on its own
def test_fit_material_cuts_the_tail_and_keeps_the_head(brain, monkeypatch):
    monkeypatch.setattr(brain, "SETTINGS", brain.ModelSettings(num_ctx=2048))
    head = "Instruction for Jarvis: read this.\n\nDocument:\n"
    msgs = [{"role": "system", "content": "s" * 2000},
            {"role": "user", "content": head + "d" * 30000}]
    cut, est = brain.fit_material(msgs, None, num_predict=100)
    assert cut > 0
    assert est <= brain.SETTINGS.ceiling_for(100)
    assert msgs[1]["content"].startswith(head)
    assert msgs[1]["content"].endswith(brain.MATERIAL_CUT_TEXT)
    # a second pass finds it fits and changes nothing
    again = msgs[1]["content"]
    assert brain.fit_material(msgs, None, num_predict=100)[0] == 0
    assert msgs[1]["content"] == again


def test_fit_material_respects_the_switch_and_a_missing_user_turn(
        brain, monkeypatch, caplog):
    monkeypatch.setattr(brain, "SETTINGS",
                        brain.ModelSettings(num_ctx=2048,
                                            protect_question=False))
    msgs = [{"role": "user", "content": "x" * 40000}]
    assert brain.fit_material(msgs)[0] == 0
    assert len(msgs[0]["content"]) == 40000
    monkeypatch.setattr(brain, "SETTINGS", brain.ModelSettings(num_ctx=2048))
    only_system = [{"role": "system", "content": "s" * 40000}]
    with caplog.at_level("WARNING"):
        assert brain.fit_material(only_system, label="json")[0] == 0
    assert any("no material to trim" in r.message for r in caplog.records)


def test_the_ceiling_follows_the_request_s_own_num_predict(brain):
    s = brain.ModelSettings(num_ctx=16384, num_predict=160,
                            answer_reserve_tokens=128)
    assert s.ceiling_for(None) == s.prompt_ceiling == 15441
    assert s.ceiling_for(1) == 15600          # a warm-up
    assert s.ceiling_for(420) == 15181        # explain_text
    assert s.ceiling_for("junk") == 15441     # falls back, never raises


# ------------------------------------------------------------------------
# 2026-09-04 round 3: the guard's margin, its calibration, cut-before-drop
# and the record of what the model saw. (Import/config hygiene is in
# tests/test_config_readonly.py.)
# ------------------------------------------------------------------------
def test_the_margin_arithmetic_at_the_shipped_settings(brain):
    """Round 2's ceiling was 16384 - 160 - 128 = 16096: 288 tokens of
    margin against an estimate whose own measured error on the static
    prefix was 5.8% (estimated 3556, counted 3761), i.e. ~930 tokens at
    that ceiling. Round 3: the estimate is multiplied by a calibration
    factor that starts at the measured 1.06, and 4% of the window (655)
    comes off the ceiling on top of the reserve."""
    s = brain.ModelSettings()
    assert brain.ESTIMATE_MARGIN == 0.04
    assert brain.CALIBRATION_INITIAL == 1.06
    assert s.prompt_ceiling == 15441
    raw_limit = int(s.prompt_ceiling / brain.CALIBRATION_INITIAL)
    assert raw_limit == 14566
    # a real cost 10% above the raw estimate (worse than anything
    # measured) plus the whole 160-token reply still fits the window
    assert int(raw_limit * 1.10) + s.num_predict < s.num_ctx
    # and the old ceiling would not have survived the measured 5.8%
    assert int(16096 * 1.058) + 160 > 16384


def test_the_guard_compares_the_calibrated_estimate(brain, monkeypatch):
    """A prompt whose RAW estimate is just under the ceiling but whose
    calibrated one is over gets trimmed; with the factor at 1.0 it would
    have been sent as-is (and, at the measured 5.8% error, truncated)."""
    msgs = [{"role": "system", "content": "P" * 4100},           # ~1004
            {"role": "user", "content": "his question"},
            {"role": "tool", "content": "r" * 9000,               # ~4004
             "tool_name": "get_calendar"}]
    raw = brain.estimate_prompt_tokens(msgs)
    ceiling = raw + 50                                           # raw fits
    assert brain.calibrated(raw) > ceiling                       # calibrated does not
    copy_ = [dict(m) for m in msgs]
    monkeypatch.setitem(brain._CALIBRATION, "factor", 1.0)
    assert brain.fit_prompt(copy_, ceiling=ceiling) == (0, raw)  # round 2 behaviour
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    dropped, est = brain.fit_prompt(msgs, ceiling=ceiling)
    assert dropped == 0 and est <= ceiling
    assert msgs[2]["content"].endswith(brain.TOOL_CUT_TEXT)


def test_calibration_learns_from_ollama_s_count_and_only_upward(brain, caplog,
                                                                 monkeypatch):
    """Each ctx: line feeds the measured/estimated ratio into the factor
    the NEXT round's guard uses (EMA, half weight), logged with both
    numbers. It never goes below the measured baseline, and a count under
    half the estimate (not a whole-prompt count) is ignored."""
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    f0 = brain.calibration_factor()
    assert f0 == brain.CALIBRATION_INITIAL == 1.06
    raw = 1000
    with caplog.at_level("INFO"):
        brain._log_round_tokens(text_reply("hi", prompt_eval_count=1200),
                                estimated=brain.calibrated(raw))
    f1 = brain.calibration_factor()
    assert f1 == round(f0 + 0.5 * (1.2 - f0), 4) == 1.13     # measured 1.20, half way
    lines = [r.message for r in caplog.records
             if r.message.startswith("ctx-calibration:")]
    assert lines == ["ctx-calibration: 1.060 -> 1.130 (Ollama counted 1200 "
                     "against an estimate of 1000 [chat])"]
    ctx = [r.message for r in caplog.records if r.message.startswith("ctx:")]
    assert len(ctx) == 1 and "(estimated 1060, raw 1000 x1.060)" in ctx[0]
    # Ollama counting LESS than the estimate pulls it back down, never
    # below the baseline
    for _ in range(6):
        brain._log_round_tokens(text_reply("hi", prompt_eval_count=900),
                                estimated=brain.calibrated(raw))
    assert brain.calibration_factor() == brain.CALIBRATION_INITIAL
    # a count under half the estimate is not a whole-prompt count
    brain._log_round_tokens(text_reply("hi", prompt_eval_count=300),
                            estimated=brain.calibrated(raw))
    assert brain.calibration_factor() == brain.CALIBRATION_INITIAL
    # and it is capped
    for _ in range(20):
        brain._log_round_tokens(text_reply("hi", prompt_eval_count=90000),
                                estimated=brain.calibrated(raw))
    assert brain.calibration_factor() == brain.CALIBRATION_MAX
    # the next guard uses it
    msgs = [{"role": "user", "content": "q"},
            {"role": "tool", "content": "r" * 2250, "tool_name": "t"}]
    assert brain.fit_prompt(msgs, ceiling=10_000)[1] == \
        brain.calibrated(brain.estimate_prompt_tokens(msgs))


def test_the_startup_line_names_the_margin_and_the_factor(brain, caplog):
    brain._SETTINGS_LOGGED.clear()
    with caplog.at_level("INFO"):
        brain.log_settings(make_registry())
    line = [r.message for r in caplog.records if "window" in r.message][0]
    assert "4% of the window kept as estimate margin" in line
    assert "estimate x1.060 from the last measured round" in line
    assert f"above ~{brain.SETTINGS.prompt_ceiling} calibrated tokens" in line


def test_seen_after_guard_records_what_the_model_now_sees(brain):
    """The loop's tool_texts / ran_results are what the coverage and
    provenance checks judge a reply against. After a cut they hold the
    kept head (marker and all); after a drop, the dropped marker -- never
    the words the model no longer has."""
    cal = ToolResult(text="09:00 lecture\n" * 100, max_sentences=4)
    mail = ToolResult(text="mail " * 200)
    ran = [("get_calendar", cal), ("get_mail", mail)]
    texts = [cal.text, mail.text]
    m_cal = {"role": "tool", "content": cal.text, "tool_name": "get_calendar"}
    m_mail = {"role": "tool", "content": mail.text, "tool_name": "get_mail"}
    index = {id(m_cal): 0, id(m_mail): 1}
    edits = []
    ceiling = brain.calibrated(brain.estimate_prompt_tokens([m_cal, m_mail])) - 150
    brain.fit_prompt([m_cal, m_mail], ceiling=ceiling, changed=edits)
    assert [k for _m, k, _b in edits] == ["cut"]
    assert brain.seen_after_guard(edits, index, ran, texts) == [0]
    assert texts[0] == m_cal["content"] and texts[0].endswith(brain.TOOL_CUT_TEXT)
    assert ran[0][1].text == m_cal["content"]
    assert ran[0][1].max_sentences == 4 and ran[0][0] == "get_calendar"
    assert texts[1] == mail.text and ran[1][1] is mail   # untouched
    # a drop
    m_cal["content"] = brain.TOOL_DROPPED_TEXT
    brain.seen_after_guard([(m_cal, "dropped", cal.text)], index, ran, texts)
    assert texts[0] == ran[0][1].text == brain.TOOL_DROPPED_TEXT
    # a message the loop never indexed is left alone
    assert brain.seen_after_guard([({"role": "tool", "content": "x"}, "cut",
                                    "xx")], index, ran, texts) == []


def test_a_real_turn_cuts_the_only_result_and_keeps_the_morning(
        brain, setup, monkeypatch, caplog):
    """The round-2 review's probe: a small window, ONE calendar result the
    model needs. Round 2 replaced it whole with TOOL_DROPPED_TEXT and the
    turn ended as 'an earlier result was dropped'. Round 3 sends the
    calendar's head with the cut marked, and records exactly that."""
    b, fake = setup
    line = "09:00 BMEN 427 lecture, room 1.14; 10:30 office hours\n"
    cal = line * 80                                        # 4 400 chars
    monkeypatch.setattr(brain, "_REGISTRY", make_registry(long_text=cal))
    _tiny_window(brain, monkeypatch, num_ctx=3072)
    edits_seen = []
    real = brain.fit_prompt

    def spy(messages, tools=None, ceiling=None, **kw):
        out = real(messages, tools, ceiling, **kw)
        edits_seen.extend(kw.get("changed") or [])
        return out
    monkeypatch.setattr(brain, "fit_prompt", spy)
    fake.replies = [tool_reply("get_calendar", {}),
                    text_reply("A lecture at nine and office hours at "
                               "half ten, sir.", prompt_eval_count=2600)]
    with caplog.at_level("WARNING"):
        b._chat_sync("what's on my calendar?")
    sent = [m for m in fake.chat_payloads()[-1]["messages"]
            if m["role"] == "tool"]
    assert len(sent) == 1
    assert sent[0]["content"].startswith(line)             # the morning
    assert sent[0]["content"].endswith(brain.TOOL_CUT_TEXT)
    assert sent[0]["content"] != brain.TOOL_DROPPED_TEXT
    assert [k for _m, k, _b in edits_seen] == ["cut"]
    assert edits_seen[0][0] is sent[0]                     # the same message
    assert any("cut the last" in r.message and "get_calendar" in r.message
               for r in caplog.records)


def test_the_screen_tool_goes_through_the_guard_and_logs_its_count(
        brain, monkeypatch, caplog):
    """The fourth /api/chat path in the package (jarvis/tools/screen.py)
    now takes the same guard and writes the same ctx: line, tagged
    [screen]; the image is costed as a fixed allowance, never as its
    base64, so a screenshot cannot trim the question. (The round-3 review
    measured the allowance at ZERO on this path; the tests further down
    hold the number and the factor.)"""
    from jarvis.tools import screen as scr
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    seen = []

    def vision(payload, timeout=None):
        seen.append(payload)
        return {"model": payload["model"], "done": True,
                "prompt_eval_count": 1900, "eval_count": 30,
                "message": {"role": "assistant", "content": "A terminal."}}
    monkeypatch.setattr(scr, "_ask_vision", vision)
    b64 = "A" * 200_000                                   # a real screenshot's size
    with caplog.at_level("INFO"):
        assert scr.ask_screen("what's on my screen?", brain.OLLAMA_MODEL,
                              b64, "hunter@spark", caps=frozenset()) == \
            "A terminal."
    lines = [r.message for r in caplog.records
             if r.message.startswith("ctx:") and "[screen]" in r.message]
    assert len(lines) == 1 and "prompt 1900/" in lines[0]
    assert "(1 image at %d)" % brain.IMAGE_TOKENS_ALLOWANCE in lines[0]
    assert seen[-1]["messages"][-1]["images"] == [b64]     # image intact
    assert brain.MATERIAL_CUT_TEXT not in seen[-1]["messages"][-1]["content"]
    est = brain.estimate_prompt_tokens(seen[-1]["messages"])
    assert est < 2500, est                                 # not 48 000 for the b64
    # a pasted page as the question is cut, the image and the title kept
    monkeypatch.setattr(brain, "SETTINGS", brain.ModelSettings(num_ctx=2048))
    scr.ask_screen("word " * 8000, brain.OLLAMA_MODEL, b64, "hunter@spark",
                   caps=frozenset())
    user = seen[-1]["messages"][-1]
    assert user["content"].startswith("Active window: hunter@spark\nQuestion:")
    assert user["content"].endswith(brain.MATERIAL_CUT_TEXT)
    assert user["images"] == [b64]


def test_the_docs_no_longer_say_an_agent_may_write_his_config():
    doc = (REPO / "docs" / "assistant-setup.md").read_text(encoding="utf-8")
    sec = doc.split("## 84.", 1)[1].split("\n## ", 1)[0]
    assert "ensure_defaults" in sec
    assert "never writes" in sec
    assert "[screen]" in sec
    assert "calibration" in sec.lower()
    assert "15441" in sec


def test_the_docs_describe_the_round_3_review_fixes():
    """Section 84 says what the guard does now: the LARGEST result, cut
    AGAIN on a later round, a drop only when the floors would not fit;
    and an image round is costed at the allowance but never calibrated
    on. The words the two holes were reported against are gone."""
    doc = (REPO / "docs" / "assistant-setup.md").read_text(encoding="utf-8")
    sec = doc.split("## 84.", 1)[1].split("\n## ", 1)[0]
    assert "tail of the largest tool" in sec
    assert "cut again" in sec
    assert "400 characters" in sec
    assert "never feeds the calibration" in sec
    assert "tail of the oldest tool" not in sec


# ---------------------------------------------------------------------
# 2026-09-04, the round-3 review's two measured holes
# ---------------------------------------------------------------------
# (1) The screen tool's image was costed at ZERO on the guard path it
# uses: fit_material summed the other messages and the last message's
# TEXT and never looked at its images (measured: the walk put the real
# vision payload at 1122 tokens, the guard at 116). And because that
# [screen] count fed the shared calibration EMA, ONE screenshot question
# pinned the process-wide factor at the 2.0 clamp and halved the tool
# loop's raw trim threshold (14566 -> 7720) for the next seven rounds.
# (2) The once-only cut rule: a result already carrying TOOL_CUT_TEXT was
# skipped, so on the round after a cut the guard dropped the NEWEST result
# whole and stayed over, with hundreds of trimmable tokens left in the cut
# one -- and logged "every tool result dropped", which was false.

def _vision_messages(brain, question="what's on my screen?", images=1):
    from jarvis.tools import screen as scr
    payload = scr.vision_payload(brain.OLLAMA_MODEL, "A" * 200_000,
                                 question, "hunter@spark")
    msgs = [dict(m) for m in payload["messages"]]
    msgs[-1]["images"] = ["A" * 200_000] * images
    return msgs


def test_fit_material_costs_the_image_the_screen_question_carries(brain,
                                                                   monkeypatch):
    """Measured pre-fix on the real vision payload: estimate_prompt_tokens
    1122, fit_material 116 (raw 109). The guard now costs the last
    message's images at IMAGE_TOKENS_ALLOWANCE each, the same figure the
    walk uses, so the two estimators agree to within the text-rate
    difference on the 58-char question."""
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    msgs = _vision_messages(brain)
    walk = brain.estimate_prompt_tokens(msgs)
    assert walk > brain.IMAGE_TOKENS_ALLOWANCE                # the walk always did
    _, est = brain.fit_material([dict(m) for m in msgs], label="screen")
    raw = int(est / brain.calibration_factor() + 0.5)
    assert raw >= brain.IMAGE_TOKENS_ALLOWANCE + 100, raw    # not 109
    assert abs(raw - walk) <= 20, (raw, walk)
    # two images, two allowances
    _, est2 = brain.fit_material(_vision_messages(brain, images=2),
                                 label="screen")
    raw2 = int(est2 / brain.calibration_factor() + 0.5)
    assert raw2 - raw == brain.IMAGE_TOKENS_ALLOWANCE
    # and the image's allowance counts toward the cut of a pasted page
    monkeypatch.setattr(brain, "SETTINGS", brain.ModelSettings(num_ctx=4096))
    ceiling = brain.SETTINGS.prompt_ceiling
    with_image = _vision_messages(brain, question="word " * 3000)
    without = _vision_messages(brain, question="word " * 3000)
    without[-1].pop("images")
    cut_with, est_with = brain.fit_material(with_image, label="screen")
    cut_without, est_without = brain.fit_material(without, label="screen")
    assert est_with <= ceiling and est_without <= ceiling
    assert cut_with - cut_without >= int(
        brain.IMAGE_TOKENS_ALLOWANCE * brain.TOOL_CHARS_PER_TOKEN) - 20


def test_an_image_round_is_logged_but_never_calibrated_on(brain, caplog,
                                                          monkeypatch):
    """_log_round_tokens(images=n) writes the ctx: line with the image
    count and leaves the factor alone, whatever Ollama counted; the same
    count with images=0 moves it. The exclusion is per round, not a
    switch."""
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    raw = 1134
    with caplog.at_level("INFO"):
        brain._log_round_tokens(text_reply("A terminal.", prompt_eval_count=4000),
                                estimated=brain.calibrated(raw),
                                label="screen", images=1)
    assert brain.calibration_factor() == brain.CALIBRATION_INITIAL
    ctx = [r.message for r in caplog.records if r.message.startswith("ctx:")]
    assert len(ctx) == 1 and "(1 image at 1024) [screen]" in ctx[0]
    cal = [r.message for r in caplog.records
           if r.message.startswith("ctx-calibration:")]
    assert len(cal) == 1 and cal[0].startswith("ctx-calibration: unchanged at 1.060")
    assert "1 image" in cal[0] and "[screen]" in cal[0]
    # the same count without an image is a real measurement
    brain._log_round_tokens(text_reply("hi", prompt_eval_count=4000),
                            estimated=brain.calibrated(raw))
    assert brain.calibration_factor() == brain.CALIBRATION_MAX


@pytest.mark.parametrize("count", [371, 1900, 4000])
def test_one_screen_question_leaves_the_calibration_factor_where_it_was(
        brain, monkeypatch, caplog, count):
    """Measured pre-fix through ask_screen with a fake vision reply: a
    prompt_eval_count of 371 (the 115 text tokens plus the 256 the model
    card gives an image), 1900 or 4000 each took the factor from 1.060 to
    the 2.000 clamp in ONE question, and the tool loop's raw trim
    threshold at the shipped settings from 14566 to 7720. Now the factor
    and the threshold do not move, and a chat round afterwards still
    calibrates."""
    from jarvis.tools import screen as scr
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    monkeypatch.setattr(brain, "SETTINGS", brain.ModelSettings())
    ceiling = brain.SETTINGS.prompt_ceiling
    assert ceiling == 15441
    threshold_before = int(ceiling / brain.calibration_factor())
    assert threshold_before == 14566

    def vision(payload, timeout=None):
        return {"model": payload["model"], "done": True,
                "prompt_eval_count": count, "eval_count": 30,
                "message": {"role": "assistant", "content": "A terminal."}}
    monkeypatch.setattr(scr, "_ask_vision", vision)
    with caplog.at_level("INFO"):
        assert scr.ask_screen("what's on my screen?", brain.OLLAMA_MODEL,
                              "A" * 200_000, "hunter@spark",
                              caps=frozenset()) == "A terminal."
    assert brain.calibration_factor() == brain.CALIBRATION_INITIAL
    assert int(ceiling / brain.calibration_factor()) == threshold_before
    ctx = [r.message for r in caplog.records
           if r.message.startswith("ctx:") and "[screen]" in r.message]
    assert len(ctx) == 1 and f"prompt {count}/" in ctx[0]
    assert "(1 image at 1024)" in ctx[0]
    raw = int(ctx[0].split("raw ")[1].split(" ")[0])
    assert raw >= brain.IMAGE_TOKENS_ALLOWANCE + 100, raw   # the image, costed
    assert not [r for r in caplog.records
                if r.message.startswith("ctx-calibration:")
                and "->" in r.message]                       # nothing moved
    # the tool loop's guard is untouched by the screenshot
    msgs = [{"role": "user", "content": "q"},
            {"role": "tool", "content": "r" * 2250, "tool_name": "t"}]
    assert brain.fit_prompt(msgs, ceiling=10_000)[1] == \
        brain.calibrated(brain.estimate_prompt_tokens(msgs))
    assert brain.calibrated(1000) == 1060
    # a chat round afterwards is still a measurement
    brain._log_round_tokens(text_reply("hi", prompt_eval_count=1200),
                            estimated=brain.calibrated(1000))
    assert brain.calibration_factor() == 1.13


def test_the_round_after_a_cut_cuts_the_largest_again_and_keeps_the_newest(
        brain, monkeypatch):
    """The review's isolated replay (ceiling 1700). Measured pre-fix:
    round 1 cut a 4 400-char calendar to 1261 chars (1671/1700); round 2
    added a 550-char mail result and the guard DROPPED it whole, skipped
    the cut calendar ("once is enough"), and was still over at 1743.
    Now the calendar is cut again (one marker, not two) and the mail is
    sent whole."""
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    line = "09:00 BMEN 427 lecture, room 1.14; 10:30 office hours\n"
    msgs = [{"role": "system", "content": "P" * 4100},
            {"role": "user", "content": "calendar and mail?"},
            {"role": "tool", "content": line * 80, "tool_name": "get_calendar"}]
    dropped, est = brain.fit_prompt(msgs, ceiling=1700)
    assert dropped == 0 and est <= 1700
    after_round_1 = msgs[2]["content"]
    assert after_round_1.endswith(brain.TOOL_CUT_TEXT)
    mail = "mail line\n" * 55                                  # 550 chars
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "get_mail", "arguments": {}}}]})
    msgs.append({"role": "tool", "content": mail, "tool_name": "get_mail"})
    assert brain.calibrated(brain.estimate_prompt_tokens(msgs)) > 1700
    edits = []
    dropped, est = brain.fit_prompt(msgs, ceiling=1700, changed=edits)
    assert dropped == 0
    assert est <= 1700                                         # not 1743
    assert msgs[4]["content"] == mail                          # the newest, whole
    cal = msgs[2]["content"]
    assert cal.startswith(line)                                # the morning
    assert len(cal) < len(after_round_1)                       # cut again
    assert cal.count(brain.TOOL_CUT_TEXT) == 1                 # marked once
    assert cal.endswith(brain.TOOL_CUT_TEXT)
    assert [(k, b) for _m, k, b in edits] == [("cut", after_round_1)]
    assert msgs[1]["content"] == "calendar and mail?"


def test_the_largest_result_is_cut_even_when_it_is_the_newest(brain, monkeypatch):
    """Oldest-first dropped a 720-char weather result whole (it could not
    absorb 200 tokens above the floor) before touching the 5 200-char
    calendar behind it. Largest first cuts the calendar and keeps the
    weather whole, whichever came first."""
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    msgs = [{"role": "system", "content": "PERSONA " * 100},
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "old weather " * 60,
             "tool_name": "get_weather"},
            {"role": "tool", "content": "new calendar " * 400,
             "tool_name": "get_calendar"}]
    ceiling = brain.calibrated(brain.estimate_prompt_tokens(msgs)) - 200
    dropped, est = brain.fit_prompt(msgs, ceiling=ceiling)
    assert dropped == 0 and est <= ceiling
    assert msgs[2]["content"] == "old weather " * 60
    assert msgs[3]["content"].startswith("new calendar")
    assert msgs[3]["content"].endswith(brain.TOOL_CUT_TEXT)


def test_when_the_largest_cannot_absorb_it_all_every_result_keeps_its_head(
        brain, monkeypatch, caplog):
    """A 3 000-char calendar (oldest) and a 600-char mail, 1 200 tokens
    over: the calendar cannot absorb that above the 400-char floor, but
    both results AT the floor would fit -- so the calendar goes to its
    floor and the mail takes the rest, and nothing is dropped. Pre-fix:
    the calendar dropped whole, the mail untouched."""
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    cal = ("09:00 lecture\n" * 300)[:3000]
    mail = "mail line\n" * 60
    msgs = [{"role": "system", "content": "P" * 4100},
            {"role": "user", "content": "calendar and mail?"},
            {"role": "tool", "content": cal, "tool_name": "get_calendar"},
            {"role": "tool", "content": mail, "tool_name": "get_mail"}]
    ceiling = brain.calibrated(brain.estimate_prompt_tokens(msgs)) - 1200
    with caplog.at_level("WARNING"):
        dropped, est = brain.fit_prompt(msgs, ceiling=ceiling)
    assert dropped == 0 and est <= ceiling
    c, m = msgs[2]["content"], msgs[3]["content"]
    assert c.startswith("09:00 lecture\n") and c.endswith(brain.TOOL_CUT_TEXT)
    assert len(c) - len(brain.TOOL_CUT_TEXT) >= brain.MIN_TOOL_KEEP_CHARS
    assert m.startswith("mail line\n") and m.endswith(brain.TOOL_CUT_TEXT)
    assert len(m) - len(brain.TOOL_CUT_TEXT) >= brain.MIN_TOOL_KEEP_CHARS
    assert brain.TOOL_DROPPED_TEXT not in (c, m)
    assert not any("Ollama may truncate" in r.message for r in caplog.records)


def test_a_drop_is_taken_first_when_even_the_floors_would_not_fit(
        brain, monkeypatch, caplog):
    """Same two results, 1 260 tokens over: even both at the floor would
    not fit, so a drop is unavoidable -- and it is taken FIRST, oldest,
    so the mail (the newest, the one the model just asked for) is sent
    WHOLE rather than cut to its floor and then orphaned. The log says
    why."""
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    cal = ("09:00 lecture\n" * 300)[:3000]
    mail = "mail line\n" * 60
    msgs = [{"role": "system", "content": "P" * 4100},
            {"role": "user", "content": "calendar and mail?"},
            {"role": "tool", "content": cal, "tool_name": "get_calendar"},
            {"role": "tool", "content": mail, "tool_name": "get_mail"}]
    ceiling = brain.calibrated(brain.estimate_prompt_tokens(msgs)) - 1260
    with caplog.at_level("WARNING"):
        dropped, est = brain.fit_prompt(msgs, ceiling=ceiling)
    assert dropped == 1 and est <= ceiling
    assert msgs[2]["content"] == brain.TOOL_DROPPED_TEXT
    assert msgs[3]["content"] == mail                          # whole
    why = [r.message for r in caplog.records if "dropped the oldest" in r.message]
    assert len(why) == 1
    assert "would still be over with every result cut to its 400-char floor" in why[0]
    assert "get_calendar, 3000 chars" in why[0]


@pytest.mark.parametrize("num_ctx", [4096, 3072])
def test_a_real_turn_keeps_the_newest_result_on_the_round_after_a_cut(
        brain, setup, monkeypatch, caplog, num_ctx):
    """The review's replay through JarvisBrain._chat_sync: two tools, the
    calendar round cuts, the mail round follows. Measured pre-fix at a
    3072 window: the 600-char mail result was dropped whole (85-char
    marker), the 937-char cut calendar skipped, the estimate still 2709
    against a 2662 ceiling, and the log said "every tool result dropped".

    At 4096 the calendar is cut again and BOTH results reach the model,
    the mail whole. At 3072 two floors cannot fit (measured: 2672 with
    both at 400 chars), so the calendar is dropped first and the mail is
    still sent whole, with the estimate under the ceiling (2574) and no
    "may truncate" warning -- the guard never gives up while something
    can still be cut, and never says it did."""
    b, fake = setup
    monkeypatch.setitem(brain._CALIBRATION, "factor", brain.CALIBRATION_INITIAL)
    line = "09:00 BMEN 427 lecture, room 1.14; 10:30 office hours\n"
    mail = "mail line\n" * 60                                  # 600 chars
    reg = ToolRegistry()
    reg.register(ToolSpec("get_calendar", "Today's events.",
                          {"type": "object", "properties": {}},
                          lambda **_: ToolResult(text=line * 80)))
    reg.register(ToolSpec("get_mail", "Unread mail.",
                          {"type": "object", "properties": {}},
                          lambda **_: ToolResult(text=mail)))
    monkeypatch.setattr(brain, "_REGISTRY", reg)
    _tiny_window(brain, monkeypatch, num_ctx=num_ctx)
    ceiling = brain.SETTINGS.prompt_ceiling
    fake.replies = [tool_reply("get_calendar", {}),
                    tool_reply("get_mail", {}),
                    text_reply("Lecture at nine; one mail, sir.",
                               prompt_eval_count=2600)]
    with caplog.at_level("WARNING"):
        b._chat_sync("what's on my calendar and in my mail?")
    sent = fake.chat_payloads()[-1]
    tools = {m["tool_name"]: m["content"]
             for m in sent["messages"] if m["role"] == "tool"}
    assert tools["get_mail"] == mail                           # the newest, WHOLE
    assert brain.calibrated(brain.estimate_prompt_tokens(
        sent["messages"], sent.get("tools"))) <= ceiling
    warnings = [r.message for r in caplog.records
                if r.message.startswith("chat:")]
    assert not any("Ollama may truncate" in w for w in warnings)
    assert not any("every tool result dropped" in w for w in warnings)
    cal = tools["get_calendar"]
    if num_ctx == 4096:
        assert cal.startswith(line) and cal.endswith(brain.TOOL_CUT_TEXT)
        assert cal.count(brain.TOOL_CUT_TEXT) == 1
        assert brain.TOOL_DROPPED_TEXT not in tools.values()
        assert [w for w in warnings if "cut the last" in w
                and "get_calendar, again" in w]                # cut TWICE
    else:
        assert cal == brain.TOOL_DROPPED_TEXT                  # unavoidable
        assert [w for w in warnings if "dropped the oldest result (get_calendar"
                in w and "every result cut to its 400-char floor" in w]
