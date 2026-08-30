"""User-defined phrase -> tool call, straight from assistant.json.

Checked BEFORE the intent classifier and the local model: a personal
shortcut should not depend on an LLM parsing it, should not cost a model
round-trip, and should behave identically every time.
"""
from types import SimpleNamespace

from jarvis.commander import Commander


class _Tools:
    def __init__(self, ok=True, boom=False):
        self.calls = []
        self._ok, self._boom = ok, boom

    def call(self, name, args=None):
        if self._boom:
            raise RuntimeError("tool exploded")
        self.calls.append((name, args))
        return SimpleNamespace(ok=self._ok, text="played", speak=None)


class _Cfg:
    def __init__(self, phrases):
        self._p = phrases

    def get(self, key, default=None):
        return self._p if key == "phrases" else default


def _commander(phrases, tools=None):
    c = object.__new__(Commander)
    c.services = SimpleNamespace(assistant=_Cfg(phrases),
                                 tools=tools or _Tools())
    return c


PHRASE = [{"say": ["drop my needle", "drop the needle"],
           "tool": "spotify_play",
           "args": {"query": "Jingle Bells"},
           "reply": "Dropping the needle, sir."}]


def test_phrase_fires_and_calls_the_tool():
    tools = _Tools()
    c = _commander(PHRASE, tools)
    res = c._try_custom_phrase("jarvis drop my needle")
    assert res is not None and res.handled
    assert tools.calls == [("spotify_play", {"query": "Jingle Bells"})]
    assert res.reply == "Dropping the needle, sir." and res.speak


def test_matching_ignores_case_punctuation_and_extra_words():
    for said in ("Drop My Needle!", "jarvis, drop the needle please",
                 "  drop   my  needle  "):
        assert _commander(PHRASE)._try_custom_phrase(said) is not None, said


def test_a_near_miss_does_not_fire():
    """It must not swallow unrelated speech -- everything downstream of
    this stage (classifier, local model) never sees a consumed utterance."""
    for said in ("drop a needle", "needle", "drop my pen", "what time is it"):
        assert _commander(PHRASE)._try_custom_phrase(said) is None, said


def test_longest_phrase_wins():
    """A more specific shortcut must beat one that is a prefix of it."""
    phrases = [{"say": "play music", "tool": "generic"},
               {"say": "play music loudly", "tool": "specific"}]
    tools = _Tools()
    _commander(phrases, tools)._try_custom_phrase("play music loudly")
    assert tools.calls[0][0] == "specific"


def test_no_phrases_configured_is_a_clean_pass_through():
    for phrases in ([], None, "not a list", [{"say": "x"}]):   # no tool key
        assert _commander(phrases)._try_custom_phrase("drop my needle") is None


def test_a_failing_tool_apologises_rather_than_going_silent():
    """The user said something they expect to work; silence reads as broken."""
    c = _commander(PHRASE, _Tools(boom=True))
    res = c._try_custom_phrase("drop my needle")
    assert res is not None and res.handled and res.speak
    assert "failed" in res.reply.lower()


def test_fires_without_the_wake_word_still_in_the_transcript():
    """The regression that made "Jarvis, drop my needle" do nothing.

    The hotword consumes the wake word, so the transcript reaching handle()
    is bare "drop my needle" -- no "jarvis" prefix left to strip. The phrase
    check used to sit inside the `if cmd_text is not None` block, so it was
    skipped entirely and the utterance fell through to the intent classifier,
    which calls short phrases background chat and drops them silently:

        handle 'drop my needle' source=voice
        Ignored (background chat, conf=0.80): 'drop my needle'

    A phrase the user wrote into assistant.json by hand is addressed to
    Jarvis by construction; it must never be subject to that guess.
    """
    tools = _Tools()
    c = _commander(PHRASE, tools)
    c.dictation = False
    for stub in ("_try_ringing", "_try_approval", "_try_terminal_offer",
                 "_try_event_confirm", "_try_router_answer"):
        setattr(c, stub, lambda *a, **k: None)
    # Anything reaching the classifier is already a failure; make that loud
    # rather than letting a NO verdict quietly look like a pass.
    def _boom(*a, **k):
        raise AssertionError("reached the intent classifier")
    c.intent = SimpleNamespace(classify=_boom)

    res = c.handle("drop my needle", source="voice")

    assert res is not None and res.handled
    assert tools.calls == [("spotify_play", {"query": "Jingle Bells"})]
    assert res.reply == "Dropping the needle, sir."
