"""Persona tests for jarvis.brain — the film-JARVIS voice.

Pure tests check the rendered prompts (few-shot pool and per-call sampling,
negative rules, shared voice across Tier 2 / Tier 3 / autonomous), the
guards between Ollama and TTS (label/emoji/file-extension cleaning, the
invented-clock guard, the two-sentence cap, the never-mid-word trim) and
the exact Tier 2 request via a mocked Ollama (gemma4 /api/chat: a STATIC
system message plus a dynamic user turn). The live test runs only with
JARVIS_LIVE_OLLAMA=1 and asks the real gemma4 one question.

The tool loop itself (rounds, force_tool, budgets, residency) is tested in
tests/test_brain_tools.py; this file owns the voice.

Firewall: nothing here may touch the live app's /tmp/vss_voice. The `brain`
fixture patches jarvis.logs / jarvis.config.PATHS to a tmp dir BEFORE
importing jarvis.brain (tests/conftest.py also sets JARVIS_LOG_DIR).
"""
import io
import json
import os
import random
import re

import pytest

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
_MARKDOWN = re.compile(
    r"(^|\n)\s*([-*•]\s+|\d+[.)]\s+|#{1,6}\s+)|\*\*|`|__", re.M)
_ABBR = re.compile(r"\b(e\.g|i\.e|etc|Mr|Mrs|Dr|vs|a\.m|p\.m)\.", re.I)
BANNED = ["as an ai", "i'd be happy", "happy to help", "happy to assist",
          "i'm here to help", "here to assist", "certainly!",
          "great question", "language model", "how may i assist",
          "initiate", "topic of your choice", "at your earliest"]
FILM_LINES = ["For you, sir, always", "As you wish", "I'd advise against it",
              "Working on it, sir", "At your service"]
# Claims of having acted that a Tier 2 example must never model.
FAKE_ACTION = re.compile(r"\bI(?:'ve| have) (?:run|checked|noticed|found|"
                         r"scanned)\b|\bI noticed\b|\bI found\b|"
                         r"\bI(?:'ll| can) keep\b", re.I)


def count_sentences(text):
    text = text.strip()
    if not text:
        return 0
    text = text.replace("...", "…")
    text = _ABBR.sub(lambda m: m.group(1).replace(".", ""), text)
    return len([p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()])


# ------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    """Import jarvis.brain with every /tmp/vss_voice path redirected."""
    tmp = tmp_path_factory.mktemp("persona")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JARVIS_LOG_DIR", str(tmp))
        import jarvis.logs as logs
        mp.setattr(logs, "LOG_DIR", tmp)
        mp.setattr(logs, "LOG_FILE", tmp / "jarvis.log")
        import jarvis.config as config
        mp.setattr(config.PATHS, "LOG_DIR", tmp)
        mp.setattr(config.PATHS, "SPEAK_QUEUE", tmp / "speak_queue.txt")
        import jarvis.brain as brain_mod
        yield brain_mod


class FakeContext:
    def __init__(self):
        self.calls = []

    def get_context(self, level):
        return {"level": level}

    def format_for_prompt(self, ctx, spoken=False):
        self.calls.append(spoken)
        return "Active window: Terminal — ~/Jarvis"


class FakeMemory:
    def format_for_context(self):
        return "Known facts (1):\n  editor: vim"


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ollama_returning(monkeypatch, response):
    """One mocked /api/chat turn in the gemma4 shape (no tool calls)."""
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["payload"] = json.loads(req.data)
        sent["url"] = req.full_url
        return _FakeResp(json.dumps(
            {"message": {"role": "assistant", "content": response}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return sent


# ----------------------------------------------------- prompt structure
def test_static_system_holds_the_facts_and_the_background_rides_in_the_turn(
        brain):
    """gemma4 shape (spec 4.2/4.4): the system prompt is static so Ollama's
    prefix cache hits; everything that changes between calls is in the user
    turn."""
    assert "{examples}" in brain.JARVIS_SYSTEM
    assert "{context}" not in brain.JARVIS_SYSTEM   # was the llama shape
    rendered = brain.build_ollama_system()
    assert "DGX Spark" in rendered and "GB10" in rendered
    assert "Hunter" in rendered and "text-to-speech" in rendered
    # a context passed the old way must not leak into the static prompt
    assert "CTX-MARKER" not in brain.build_ollama_system("CTX-MARKER",
                                                         "MEM-MARKER")
    turn = brain.build_user_turn("CTX-MARKER", "MEM-MARKER", "how are you")
    assert turn.startswith("Background:\nCTX-MARKER")
    assert turn.index("CTX-MARKER") < turn.index("MEM-MARKER")
    assert turn.endswith("\n\nHunter: how are you")
    # no context at all is said honestly rather than left blank
    assert brain.build_user_turn("", "", "hi") == \
        "Background:\n(none)\n\nHunter: hi"
    # the background is labelled as reference, not as material to recite
    assert "The background in his message is there so you can answer " \
        "questions about it accurately; never recite it unprompted" in rendered


def test_tool_rules_are_in_the_static_prompt(brain):
    """4.4: the model must reach for a tool for live data and answer only
    from the result."""
    for rule in ("Use a tool whenever the answer depends on live data",
                 "never guess those",
                 "Call the tool first, without commentary",
                 "using only the numbers, names and times in the result",
                 "never invent a figure the result does not contain",
                 "If a tool says something is not set up, say so in one "
                 "sentence and name the thing",
                 "If a tool reports a failure, say what could not be "
                 "reached in one sentence",
                 "Do not call a tool for a greeting, thanks, a joke, an "
                 "opinion or general knowledge"):
        assert rule in brain.JARVIS_SYSTEM, rule
    # the capability answer is the new truth, not the old "no weather feed"
    assert "you keep his calendar, weather, mail, reminders and alarms, " \
        "run the desktop, and hand the real coding to Claude" in \
        brain.JARVIS_SYSTEM


def test_few_shot_pool_is_in_voice(brain):
    shots = brain.FEW_SHOTS
    # gemma4 does not learn the manner from an example, it lifts the
    # example: the eight-family pool of the port copied 6 of the 14 eval
    # prompts verbatim, so only the three shape-teaching families survive
    # (scratchpad persona_gemma4.md, w5/ab_parrot.json)
    assert 8 <= len(shots) <= 12
    assert len(brain.FEW_SHOT_PINNED) == 3
    assert 2 <= len(brain.FEW_SHOT_POOL) <= 4       # + pinned = per call
    assert all(1 <= len(v) <= 2 for _, v in brain.FEW_SHOT_POOL)
    # no example carries a number, a clock or a file name the model could
    # copy into an answer regardless of the real background or tool result
    for _, reply in shots:
        assert not re.search(r"\d|\b\w+\.py\b|hours? ago|days? ago", reply), \
            reply
    sir = hunter = offer = one_sentence = 0
    for user, reply in shots:
        assert count_sentences(reply) <= 2, reply
        assert not _MARKDOWN.search(reply), reply
        assert not _EMOJI.search(reply), reply
        assert not any(b in reply.lower() for b in BANNED), reply
        assert not FAKE_ACTION.search(reply), reply
        assert not re.search(r"\b\d{1,2}:\d{2}\b", reply), reply
        sir += " sir" in reply.lower()
        hunter += "hunter" in reply.lower()
        offer += "shall i" in reply.lower()
        one_sentence += count_sentences(reply) == 1
    assert sir >= 0.7 * len(shots)          # "sir" most of the time
    assert 1 <= hunter <= 3                 # "Hunter" occasionally
    assert offer <= 1                       # at most ONE offer in a prompt
    assert one_sentence >= 0.6 * len(shots)  # the length prior is short
    families = dict(brain.FEW_SHOT_POOL)
    for name in ("greeting", "joke", "advice"):
        assert name in families, name
    # every family whose rule is already stated in prose is gone: an
    # example only teaches gemma4 which sentence to copy
    for gone in ("no data", "status", "wit", "small talk", "cannot act",
                 "capability", "bad idea", "mistake"):
        assert gone not in families, gone
    # ... and the prose that replaced them is still there
    assert "Admit limits gracefully (\"I'm afraid...\")" in brain.VOICE_RULES
    assert "you cannot buy, book, browse, call, text, order" in \
        brain.JARVIS_SYSTEM
    assert "When he owns up to a mistake, a light word of reassurance " \
        "comes before the fix" in brain.VOICE_RULES
    prompts = [u.lower() for u, _ in shots]
    assert not any("time" in p for p in prompts)     # the clock is a tool now
    assert any("you there" in p for p in prompts)    # presence -> one line
    assert any("night" in p for p in prompts)        # good night stays short
    assert any("joke" in p for p in prompts)
    assert any("luck" in p for p in prompts)         # advice w/o inspection
    assert "what can you do" not in prompts          # eval prompt held out
    # the honesty examples model the manner, never a fabricated inspection
    luck = [r for u, r in shots if "luck" in u.lower()][0]
    assert "haven't looked" in luck and "check" in luck.lower()


def test_pool_keeps_the_fixes_the_judges_asked_for(brain):
    """Round-2/3 judge findings that survived the port: the model
    paraphrases gags out of existence and parrots whatever example matches
    the situation."""
    pool = dict(brain.FEW_SHOT_POOL)
    # jokes are dry remarks about his situation, never a set-up and
    # punchline and never a list-count gag it will 'correct'
    for _, reply in pool["joke"]:
        assert not re.search(r"\bwhy did\b|knock|cross the road|\?\s*$",
                             reply, re.I), reply
        assert "cache invalidation" not in reply.lower()
        assert count_sentences(reply) == 1 and " sir" in reply, reply
    # no exemplar models the textbook-reason vocabulary the rules ban
    for _, reply in brain.FEW_SHOTS:
        assert not re.search(r"complexity|unnecessary|busy system|properly "
                             r"configured|ensure|potentially|can cause",
                             reply, re.I), reply
    # the two variants of a family never share an opener (round 2: one
    # opener copied onto every bad idea)
    for family, variants in brain.FEW_SHOT_POOL:
        openers = {re.split(r"[,;.:]", r)[0] for _, r in variants}
        assert len(openers) == len(variants), (family, openers)
    # the capability answer is the post-tools truth, and it is prose now,
    # not an example to copy
    assert "you keep his calendar, weather, mail, reminders and alarms" in \
        brain.JARVIS_SYSTEM
    assert "weather feed" not in brain.JARVIS_SYSTEM
    # the thank-you is pinned: paraphrases ('appreciate it') fall
    # through Tier 1 and the model answered one with a greeting
    assert ("Cheers, Jarvis.", "Not at all, sir.") in brain.FEW_SHOT_PINNED


def test_few_shots_are_sampled_one_per_family_and_vary(brain):
    picked = brain.select_few_shots(random.Random(0))
    assert len(picked) == len(brain.FEW_SHOT_POOL) + len(brain.FEW_SHOT_PINNED)
    sampled = [picked[0]] + picked[3:-1]         # minus the three pinned
    for (_, variants), shot in zip(brain.FEW_SHOT_POOL, sampled):
        assert shot in variants
    # "you there" and the thank-you are pinned right after the greeting,
    # good night last
    assert picked[1] == brain.FEW_SHOT_PINNED[0]
    assert picked[2] == brain.FEW_SHOT_PINNED[1]
    assert picked[-1] == brain.FEW_SHOT_PINNED[2]
    assert picked[1][0] == "Jarvis, you there?"
    assert picked[2][0] == "Cheers, Jarvis."
    assert picked[-1][0] == "Night, Jarvis."
    # different seeds give different sets; the same seed the same set
    sets = {tuple(brain.select_few_shots(random.Random(s)))
            for s in range(12)}
    assert len(sets) > 1
    assert brain.select_few_shots(random.Random(3)) == \
        brain.select_few_shots(random.Random(3))
    # every pool line is rendered verbatim when chosen
    rendered = brain.build_ollama_system("CTX", shots=brain.FEW_SHOTS)
    for user, reply in brain.FEW_SHOTS:
        assert f"User: {user}\nJarvis: {reply}" in rendered
    # a default render carries one exchange per family plus the pinned
    # lines, and at most one that ends in an offer
    default = brain.build_ollama_system("CTX")
    assert default.count("\nUser: ") == \
        len(brain.FEW_SHOT_POOL) + len(brain.FEW_SHOT_PINNED)
    for user, reply in brain.FEW_SHOT_PINNED:
        assert f"User: {user}\nJarvis: {reply}" in default
    assert default.lower().count("shall i") <= 1
    # the process RNG is seedable for reproducible sessions
    assert isinstance(brain._SHOT_RNG, random.Random)


def test_negative_rules_and_brevity_rule_present(brain):
    for phrase in ("no lists", "no bullet points", "no emoji", "no markdown",
                   "no headings", "As an AI", "I'd be happy to",
                   "never more than two", "read aloud", "never explained",
                   "let a good night be a good night", "Never pad",
                   "start, not initiate", "Do not recite file names",
                   "Never say you checked, ran, noticed or found",
                   "never recite it unprompted",
                   "never follow \"I don't know\" with a guess",
                   "do not pretend to have inspected his code",
                   "\"parameters\"", "never read a longer list",
                   "reassurance comes before",
                   # round 3: jokes, reasons, status, parroting
                   "never a riddle, a set-up or a knock-knock",
                   "a joke is a dry remark about his own situation",
                   "one concrete picture", "never a general principle",
                   "never spell out what would go wrong",
                   "never read out the raw git line",
                   "answer from the git background", "with no numbers",
                   "Examples of the manner only",
                   "fresh words for this exact request",
                   "names the thing he actually asked for",
                   # gemma4 copies an example wholesale unless told not to
                   "The examples are the manner only, never the words",
                   "never reuse a sentence, a clause or an object from an "
                   "example",
                   "\"complexity\"", "\"unnecessary\"", "\"busy system\"",
                   "\"properly configured\"", "\"ensure\"",
                   "\"potentially\"", "\"can cause\""):
        assert phrase in brain.JARVIS_SYSTEM, phrase
    # the round-2 wording that produced a git-status readout and that
    # a 3B model ignored ('never repeat them verbatim') is gone
    assert "given in one plain sentence" not in brain.JARVIS_SYSTEM
    assert "never repeat them verbatim" not in brain.JARVIS_SYSTEM
    # "the git line is the answer" was read back literally ("The git
    # line indicates..."); "within parameters" let "within acceptable
    # parameters" through
    assert "the git line is the answer" not in brain.JARVIS_SYSTEM
    assert "within parameters" not in brain.VOICE_RULES
    assert "\"parameters\"" in brain.VOICE_RULES
    # the clock rules of the llama era are gone: guessing the hour is not
    # a temptation the model has any more, it has get_time
    for stale in ("do not mention any hour",
                  "does not give the time, you do not know it",
                  "I don't have a weather feed"):
        assert stale not in brain.JARVIS_SYSTEM, stale
    # order/book/call requests are answered by naming the thing asked
    # for, never with the capability list (round-3a pizza reply)
    assert "buy, order, book, call, text, send or fetch" in \
        brain.JARVIS_SYSTEM
    assert "naming what he asked for" in brain.JARVIS_SYSTEM
    assert "never answer with what you can do instead" in \
        brain.JARVIS_SYSTEM


def test_rules_quote_nothing_the_model_could_copy(brain):
    """A full sentence quoted inside a rule is copied like an example,
    only into every situation: round 3a's quoted reason sample came back
    as the greeting, the joke and the answer to 'I'm bored'. Rules may
    quote phrases ("I'm afraid..."), never a sentence."""
    for name in ("VOICE_RULES", "JARVIS_SYSTEM", "CLAUDE_SYSTEM",
                 "AUTONOMOUS_PROMPT"):
        text = getattr(brain, name)
        # pair quotes left to right, then filter: a min-length regex would
        # skip a short phrase and pair its closer with the next opener
        quotes = re.findall(r'"([^"]*)"', text)
        long_quotes = [q for q in quotes if len(q) >= 45]
        assert not long_quotes, (name, long_quotes)
        assert len(quotes) >= 10, name          # the phrases are still there
        for frag in ("two in the morning is no time",
                     "Rather a lot changed on main",
                     "a handful of the Jarvis modules",
                     "Why did the"):
            assert frag not in text, (name, frag)
    # the reason and joke rules live in VOICE_RULES, so Tier 3 and the
    # autonomous loop get them too
    for rule in ("never a riddle", "one concrete picture", "can cause"):
        assert rule in brain.VOICE_RULES, rule
    # the closing brevity instruction is the last thing the model reads
    # (recency wins) and survives rendering
    tail = ("Now answer Hunter as Jarvis, in your own words, keeping the "
            "manner of the examples, and call him sir. One short sentence is "
            "the norm; add a second only if it says something new that he "
            "asked for, and never describe his screen, files or machine "
            "unless he asked. If he asks for a joke, it is one dry remark "
            "about his situation, never a question and its answer. The "
            "examples are the manner only, never the words: never reuse a "
            "sentence, a clause or an object from an example — if an example "
            "speaks of a phone and he asks about dinner, the reply is about "
            "dinner. Then stop.")
    rendered = brain.build_ollama_system()
    assert rendered.rstrip().endswith(tail)
    assert rendered.index("Examples of the manner only") < rendered.index(tail)
    # the old wording that primed a two-sentence quota / context recital
    assert "one or two short sentences" not in brain.JARVIS_SYSTEM
    assert "Only state facts that appear" not in brain.JARVIS_SYSTEM


def test_prompt_practices_what_it_preaches(brain):
    rendered = brain.build_ollama_system(shots=brain.FEW_SHOTS)
    for line in rendered.splitlines():
        assert not re.match(r"\s*([-*•]\s|\d+[.)]\s|#)", line), line
    assert not _EMOJI.search(rendered)
    assert "**" not in rendered and "`" not in rendered
    assert "{" not in rendered and "}" not in rendered   # .format() safe


def test_no_verbatim_film_lines(brain):
    for line in FILM_LINES:
        assert line.lower() not in brain.JARVIS_SYSTEM.lower(), line
        for _, reply in brain.FEW_SHOTS:
            assert line.lower() not in reply.lower(), (line, reply)


def test_claude_and_autonomous_prompts_share_the_voice(brain):
    assert brain.VOICE_RULES in brain.JARVIS_SYSTEM
    assert brain.VOICE_RULES in brain.CLAUDE_SYSTEM
    assert brain.VOICE_RULES in brain.AUTONOMOUS_PROMPT
    # tag protocol untouched
    for tag_line in ("[SPEAK] text — read aloud (max 2 sentences)",
                     "[RUN] command — execute shell command",
                     "[TYPE] text — type into active window",
                     "[WINDOW] name — switch to window",
                     "[SILENT] text — show in GUI only",
                     "[DONE] text — task complete, speak this"):
        assert tag_line in brain.CLAUDE_SYSTEM, tag_line
    for tag in ("[RUN]", "[SPEAK]", "[DONE]"):
        assert tag in brain.AUTONOMOUS_PROMPT
    assert "nothing you did not do" in brain.CLAUDE_SYSTEM
    assert "report only what the previous results show" in \
        brain.AUTONOMOUS_PROMPT
    rendered = brain.CLAUDE_SYSTEM.format(context="CTX", input="hello")
    assert "CTX" in rendered and "User (Hunter) said: hello" in rendered
    auto = brain.AUTONOMOUS_PROMPT.format(
        task="ship it", step=2, max_steps=10,
        results=json.dumps([{"rc": 0}]), context="CTX")
    assert "Task: ship it" in auto and "Step 2/10" in auto
    assert '"rc": 0' in auto and "CTX" in auto


# ------------------------------------------------------- reply guards
def test_clean_ollama_reply_strips_small_model_artefacts(brain):
    clean = brain.clean_ollama_reply
    assert clean("Jarvis: Quite well, sir.") == "Quite well, sir."
    assert clean("JARVIS:  Yes, sir.\nUser: thanks\nJarvis: any time") == \
        "Yes, sir."
    assert clean("Nothing on fire, sir. 😊🔥") == "Nothing on fire, sir."
    assert clean("Of course, sir. *chuckles* Carry on.") == \
        "Of course, sir. Carry on."
    assert clean("Options:\n- reboot\n- pray\n1. neither") == \
        "Options:\nreboot\npray\nneither"
    assert clean("") == "" and clean(None) == ""
    # file extensions are not read aloud as "dot pee why"
    assert clean("edited brain.py and tts.py") == "edited brain and tts"
    assert clean("see notes.md, config.json and run.sh") == \
        "see notes, config and run"
    assert clean("It's 2 p.m., sir.") == "It's 2 p.m., sir."   # untouched


def test_split_sentences_is_abbreviation_and_ellipsis_aware(brain):
    split = brain.split_sentences
    assert split("Not at all, sir. It's rather what I'm for.") == \
        ["Not at all, sir.", "It's rather what I'm for."]
    assert split("Rebuilding at 2 a.m. is unwise, sir.") == \
        ["Rebuilding at 2 a.m. is unwise, sir."]
    assert split("It appears to be... 23:47.") == \
        ["It appears to be... 23:47."]
    assert split("Yes, sir! Shall I? Now.") == ["Yes, sir!", "Shall I?",
                                                 "Now."]
    assert split("") == []


def test_two_sentence_cap_is_enforced_in_code(brain, monkeypatch):
    three = ("I'm afraid I don't know, sir. The clock is elsewhere. "
             "It was last updated yesterday.")
    assert brain.limit_sentences(three) == \
        "I'm afraid I don't know, sir. The clock is elsewhere."
    _ollama_returning(monkeypatch, three)
    b = brain.JarvisBrain(context=FakeContext(), memory=None)
    monkeypatch.setattr(b, "_query_claude",
                        lambda text: pytest.fail("fell back to Claude"))
    actions = b._chat_sync("Where did I leave off?")
    spoken = " ".join(d for t, d in actions if t == "SPEAK")
    assert count_sentences(spoken) == 2
    assert spoken == "I'm afraid I don't know, sir. The clock is elsewhere."


def test_invented_clock_reading_is_stripped_without_a_clock_line(brain,
                                                                 monkeypatch):
    guard = brain.guard_clock_claims
    ctx = "Active window: Terminal\nGit: on branch main, nothing uncommitted"
    # the round-1 failure: "I don't know" followed by a guessed time
    assert guard("I'm afraid I don't know the current time, sir. The clock "
                 "shows the last update, which appears to be... 23:47.",
                 ctx) == "I'm afraid I don't know the current time, sir."
    # nothing honest left -> the honest line
    assert guard("It's 23:47, sir.", ctx) == brain.NO_CLOCK_LINE
    # a real clock line in the context makes the number legitimate
    assert guard("It's 10:30, sir.", "Current time: 10:30 AM") == \
        "It's 10:30, sir."
    # a number Hunter himself said is not an invention
    assert guard("Ten thirty it is, sir; 10:30 sharp.", ctx,
                 "remind me at 10:30") == "Ten thirty it is, sir; 10:30 sharp."
    assert guard("No numbers here, sir.", ctx) == "No numbers here, sir."
    # end to end through the mocked Tier 2 call
    _ollama_returning(monkeypatch, "I'm afraid I don't know, sir. It "
                                   "appears to be... 23:47.")
    b = brain.JarvisBrain(context=FakeContext(), memory=None)
    monkeypatch.setattr(b, "_query_claude",
                        lambda text: pytest.fail("fell back to Claude"))
    assert b._chat_sync("What time is it?") == \
        [("SPEAK", "I'm afraid I don't know, sir.")]


def test_trim_spoken_never_ends_mid_word(brain):
    trim = brain.trim_spoken
    b = brain.JarvisBrain(context=None, memory=None)
    # whole sentences that fit the 250 target are kept, the rest dropped:
    # 2 x 26 chars + 7 x 27 chars = 241 fit; the eighth "Third" would not
    many = ("First sentence here, sir. " * 2 +
            "Third one is dropped, sir. " * 8).strip()
    out = trim(many)
    assert out == many[:240]
    assert out.count("Third") == 7 and out.endswith(", sir.")
    assert len(out) == 240 and len(out) + 27 > 250
    # a 300-char single sentence: spoken whole (fits the TTS limit), ends on
    # a word boundary with terminal punctuation
    s300 = ("word " * 59 + "finish.").strip()
    assert 290 <= len(s300) <= 310
    out = trim(s300)
    assert out == s300 and out.endswith("finish.")
    # the round-1 'what can you do' sample (357 chars, 265-char first
    # sentence) is no longer chopped to "...delegating t"
    sample = ("I can manage the NVIDIA DGX Spark desktop, launch "
              "applications, switch between windows, type and dictate text, "
              "take screenshots, keep notes and reminders, set timers, run "
              "shell commands, search the web, and assist with "
              "problem-solving by delegating tasks to Claude. Currently, I "
              "am in a terminal window on the spark desktop, with a Git "
              "repository behind it.")
    assert len(sample) == 357
    spoken = b._parse_response(sample)
    assert spoken == [("SPEAK", sample.split(". Currently")[0] + ".")]
    assert spoken[0][1].endswith("to Claude.")
    # beyond the TTS limit a single sentence is cut at a clause boundary
    # beyond index 100, never inside a word, and given a period
    s600 = ("alpha beta, " * 50).strip()
    out = trim(s600)
    assert len(out) <= brain.HARD_SPOKEN_CHARS
    assert out.endswith("beta.") and " ," not in out
    # short text is untouched
    assert trim("Always, sir.") == "Always, sir."
    assert brain.HARD_SPOKEN_CHARS == 500       # one number, shared w/ TTS


def test_chat_sends_the_static_prompt_and_the_dynamic_turn(brain,
                                                          monkeypatch):
    sent = _ollama_returning(monkeypatch, "Jarvis: Quite well, sir. 😊")
    ctx, mem = FakeContext(), FakeMemory()
    b = brain.JarvisBrain(context=ctx, memory=mem)
    monkeypatch.setattr(b, "_query_claude",
                        lambda text: pytest.fail("fell back to Claude"))

    actions = b._chat_sync("How are you?")

    assert actions == [("SPEAK", "Quite well, sir.")]
    assert ctx.calls == [True]          # the spoken (count-only) rendering
    assert sent["url"].endswith("/api/chat")
    p = sent["payload"]
    assert p["model"] == brain.OLLAMA_MODEL
    assert p["stream"] is False and p["think"] is False
    assert p["keep_alive"] == -1
    assert p["options"]["num_ctx"] == 8192
    assert p["options"]["stop"] == ["\nUser:", "\nHunter:"]
    system, user = p["messages"]
    assert system == {"role": "system", "content": brain.static_system()}
    assert user["role"] == "user"
    assert user["content"] == brain.build_user_turn(
        ctx.format_for_prompt({}, spoken=True), mem.format_for_context(),
        "How are you?")
    # the dynamic half carries the background; the static half never does
    assert "Active window: Terminal" in user["content"]
    assert "editor: vim" in user["content"]
    assert "Active window: Terminal" not in system["content"]
    assert "editor: vim" not in system["content"]


def test_fallback_lines_are_in_voice(brain, monkeypatch):
    monkeypatch.setattr(brain.MACHINE, "claude_bin", "")
    b = brain.JarvisBrain(context=None, memory=None)
    actions = b._query_claude("do something hard")
    assert actions == [("SPEAK",
                        "I'm afraid the Claude CLI isn't available, sir.")]


# ------------------------------------------------------------- live
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("JARVIS_LIVE_OLLAMA") != "1",
                    reason="set JARVIS_LIVE_OLLAMA=1 to talk to Ollama")
def test_live_ollama_reply_is_short_spoken_prose(brain, monkeypatch):
    b = brain.JarvisBrain(context=None, memory=None)
    monkeypatch.setattr(b, "_query_claude",
                        lambda text: pytest.fail("Ollama unavailable"))
    actions = b._chat_sync("How are you today, Jarvis?")
    spoken = " ".join(d for t, d in actions if t == "SPEAK")
    assert spoken
    assert count_sentences(spoken) <= 2, spoken
    assert not _MARKDOWN.search(spoken), spoken
    assert not _EMOJI.search(spoken), spoken
    assert not any(bad in spoken.lower() for bad in BANNED), spoken
    assert not FAKE_ACTION.search(spoken), spoken
    assert not re.search(r"\b\d{1,2}:\d{2}\b", spoken), spoken
