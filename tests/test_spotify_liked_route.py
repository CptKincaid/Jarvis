"""Liked Songs play newest-added first, in order, and shuffle only when
Hunter says so -- decided from his words, never by the model.

LIVE 2026-09-01 19:59:26, by voice: "Play my like songs." went
local:music -> brain -> gemma4, which picked spotify_liked AND filled
shuffle=true on its own; the log read "tool spotify_liked -> ok=True Your
Liked Songs on shuffle, sir -- 500 of them, on HPCOMPUTER." for a request
that never said the word, and it could not say whose shuffle that was.

Three layers, each pinned here on the REAL object:

1. the commander (real Commander + real Router, mocked brain): every
   liked-songs phrasing is a Tier-1 route that forces spotify_liked with
   ``shuffle`` from wants_shuffle(<utterance>, default=spotify.liked_shuffle)
   -- so the model round trip is gone and so is its vote;
2. the brain's tool loop (real JarvisBrain + FakeOllama + real SpotifyTool
   over a FakeSpotify client): a model that sends {"shuffle": true} anyway
   ends up playing newest-first and SAYING newest-first, because the knob
   is off the schema and reserved (registry drops it on model calls);
3. the app (real JarvisApp, hardware stubbed): the spoken command reaches
   the forced call through the real services wiring.

No Ollama, no network, no audio: the brain's one HTTP seam is a fake and
the Spotify client is a fake.
"""
import json
import types
from unittest.mock import MagicMock

import pytest

from jarvis import brain as brain_mod
from jarvis.commander import liked_songs_kind
from jarvis.tools import spotify as sp
from jarvis.tools.registry import ToolRegistry
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)
from tests.test_brain_tools import (FakeContext, FakeMemory, FakeOllama,  # noqa: F401
                                    brain, text_reply, tool_reply)
from tests.test_route_shortcut import Cfg, rich  # noqa: F401  (fixture)
from tests.test_spotify import CFG, FakeSpotify, make_tool, start_calls

ORDER_LINE = sp.LIKED_ORDER_LINE.format(n=30, device="HPCOMPUTER")
SHUFFLE_LINE = sp.LIKED_URIS_LINE.format(n=30, device="HPCOMPUTER")


def _only_forced(svc):
    assert svc.brain.chat.call_count == 1, svc.brain.chat.call_args_list
    call = svc.brain.chat.call_args
    assert call.kwargs.get("force_tool") == "spotify_liked", call
    return call.kwargs["force_args"]


# ------------------------------------------------- 1. the commander route
@pytest.mark.parametrize("said,shuffle", [
    # the plain asks: in order, newest first
    ("play my liked songs", False),
    ("Play my like songs.", False),          # Whisper drops the d (19:59:26)
    ("play my likes", False),
    ("my liked tracks", False),
    ("songs I liked", False),
    ("play the songs I've liked", False),
    ("saved songs", False),
    ("play my saved songs", False),
    ("put on my liked songs", False),
    ("play all of my liked songs", False),
    ("play my liked songs on Spotify", False),
    ("could you play my liked songs please", False),
    ("play my like songs playlist in order", False),   # his 21:14:37 words
    # he asked for order
    ("play my liked songs in order", False),
    ("play my liked songs, newest first", False),
    ("play my liked songs in the order I added them", False),
    ("play my liked songs and don't shuffle them", False),
    # he asked for shuffle -- and only then
    ("play liked songs on shuffle", True),
    ("shuffle my liked songs", True),
    ("play my liked songs shuffled", True),
    ("play my liked songs randomly", True),
    ("play my liked songs in shuffle mode", True),
    # 2026-09-02 review: these ASK for shuffle and missed the matcher, and
    # a miss is not neutral any more -- the model's shuffle is reserved, so
    # the fall-through played them in order and said "newest first". The
    # verb list and the tail vocabulary now cover them.
    ("put my liked songs on shuffle please", True),
    ("shuffle play my liked songs", True),
    ("play my liked songs but shuffle them", True),
    ("play my liked songs in a random order", True),
    ("play my liked songs in random order", True),
    ("queue my liked songs", False),
    ("stick my liked songs on", False),
    ("throw on my liked songs", False),
])
@pytest.mark.parametrize("source", ["voice", "typed"])
def test_liked_songs_phrasings_force_the_tool_with_shuffle_from_the_words(
        rich, said, shuffle, source):  # noqa: F811
    c, svc = rich
    res = c.handle(said, source=source)
    assert res.handled and res.done is False, res
    assert _only_forced(svc) == {"shuffle": shuffle}
    # the route is the matcher's, not the router's: no classify, no model
    assert svc.brain.chat.call_args.args == (said.lower().rstrip(".!?"),)


def test_the_prefixed_form_takes_the_registry_pass(rich):  # noqa: F811
    """"Jarvis, play my liked songs" (typed, or Whisper keeping the address)
    is the REGISTRY entry; the bare voice form is the ASSISTANT_TIER1 view.
    Same Command, both doors."""
    c, svc = rich
    res = c.handle("Jarvis, play my liked songs", source="typed")
    assert res.handled
    assert _only_forced(svc) == {"shuffle": False}
    assert svc.brain.chat.call_args.args == ("play my liked songs",)


def test_the_bare_voice_form_bypasses_the_intent_gate(rich, caplog):  # noqa: F811
    """The hotword eats the wake word, so the words arrive bare; the
    classifier calls short phrases background chat and would drop them in
    silence. A Tier-1 match is addressed to Jarvis by definition."""
    c, svc = rich
    with caplog.at_level("INFO", logger="jarvis.commander"):
        res = c.handle("play my liked songs", source="voice")
    assert res.handled and res.status != "Ignored (background chat)"
    assert any("tier-1 match 'liked songs' bypasses the intent gate" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]
    assert _only_forced(svc) == {"shuffle": False}


def test_the_configured_default_is_used_when_he_did_not_say(rich):  # noqa: F811
    """spotify.liked_shuffle: true restores the old habit -- but "in order"
    still wins over it, the same rule as wants_shuffle everywhere else."""
    c, svc = rich
    svc.assistant.data["spotify.liked_shuffle"] = True
    c.handle("play my liked songs", source="voice")
    assert _only_forced(svc) == {"shuffle": True}
    svc.brain.chat.reset_mock()
    c.handle("play my liked songs in order", source="voice")
    assert _only_forced(svc) == {"shuffle": False}


def test_the_wired_tool_is_the_source_of_the_default(rich, tmp_path):  # noqa: F811
    """With services.spotify present the route reads the SAME
    liked_shuffle the tool applies, so the two can never disagree."""
    c, svc = rich
    cfg = {"spotify": {**CFG["spotify"], "liked_shuffle": True}}
    svc.spotify = make_tool(tmp_path, FakeSpotify(n_saved=30), cfg=cfg)
    assert svc.spotify.liked_shuffle is True
    c.handle("play my liked songs", source="voice")
    assert _only_forced(svc) == {"shuffle": True}


def test_a_connect_device_rides_along(rich):  # noqa: F811
    c, svc = rich
    c.handle("play my liked songs on my phone", source="voice")
    assert _only_forced(svc) == {"shuffle": False, "device": "phone"}
    svc.brain.chat.reset_mock()
    c.handle("shuffle my liked songs on HPCOMPUTER", source="typed")
    assert _only_forced(svc) == {"shuffle": True, "device": "hpcomputer"}


@pytest.mark.parametrize("said", [
    "play my liked songs by Drake",              # a search, not the library
    "what's in my liked songs",                  # a question
    "add this to my liked songs",                # a write
    "remove this from my liked songs",
    "how many liked songs do I have",
    "like this song",
    "play my liked songs on repeat",             # a mode the route can't set
    "play my liked songs and what's on my calendar",   # a compound
    "I like songs with a beat",
])
def test_anything_else_falls_through_to_the_router(rich, said):  # noqa: F811
    """The matcher is anchored and the tail must be words the route
    understands: everything else keeps its old path (the model), which can
    no longer set shuffle either."""
    c, svc = rich
    assert not liked_songs_kind(said.lower())
    c.handle(said, source="typed")
    for call in svc.brain.chat.call_args_list:
        assert call.kwargs.get("force_tool") != "spotify_liked", call


def test_the_route_needs_a_brain_that_chats(rich):  # noqa: F811
    c, svc = rich
    svc.brain = types.SimpleNamespace(think=MagicMock())
    res = c.handle("play my liked songs", source="typed")
    # legacy wiring: Tier 2 as before, nothing forced, nothing crashed
    assert res.handled
    svc.brain.think.assert_called_once()


# --------------------------------------------- 2. the brain's tool loop
@pytest.fixture
def liked_brain(brain, monkeypatch, tmp_path):  # noqa: F811
    """A real JarvisBrain whose registry holds the REAL Spotify tools over a
    FakeSpotify client, and a FakeOllama as its one HTTP seam.
    Returns (b, ollama, spotify_fake, registry)."""
    brain.reset_static_prompt()
    fake_sp = FakeSpotify(n_saved=30)
    tool = make_tool(tmp_path, fake_sp)
    reg = ToolRegistry()
    reg.register_many(tool.tools())
    monkeypatch.setattr(brain, "_REGISTRY", reg)
    ollama = FakeOllama()
    monkeypatch.setattr(brain, "_http", ollama)
    b = brain.JarvisBrain(context=FakeContext(), memory=FakeMemory())
    return b, ollama, fake_sp, reg


def test_a_model_that_sends_shuffle_true_still_plays_newest_first(liked_brain, caplog):
    """The 19:59:26 turn, replayed with the fix: the model picks the tool
    and invents shuffle=true; the player is put IN ORDER, the spoken line
    says newest first, and the log shows both what the model sent and what
    was done with it."""
    b, ollama, fake_sp, reg = liked_brain
    ollama.replies = [tool_reply(("spotify_liked", {"shuffle": True}))]
    with caplog.at_level("INFO"):
        tags = b._chat_sync("Play my like songs.")
    # what he hears
    assert tags == [("SPEAK", ORDER_LINE)]
    assert "shuffle" not in ORDER_LINE.lower()
    # what the player did: sticky flag cleared BEFORE the ordered start
    assert fake_sp.named("shuffle") == [((False,), {"device_id": "dev-hp"})]
    order = [m for m, _, _ in fake_sp.calls if m in ("shuffle", "start_playback")]
    assert order == ["shuffle", "start_playback"]
    assert start_calls(fake_sp)[0]["uris"] == [f"spotify:track:liked-{i}" for i in range(30)]
    # the speak line skipped the render turn: ONE model round, not two
    assert len(ollama.chat_payloads()) == 1
    # what the model was offered: no shuffle knob on spotify_liked
    liked_schema = next(t for t in ollama.chat_payloads()[0]["tools"]
                        if t["function"]["name"] == "spotify_liked")
    assert "shuffle" not in json.dumps(liked_schema)
    # the audit trail
    said = [r.getMessage() for r in caplog.records]
    assert 'tool spotify_liked: dropped model-supplied reserved args {"shuffle": true}' in said
    assert f'tool spotify_liked {{"shuffle": true}} -> ok=True {ORDER_LINE[:80]}' in said


@pytest.mark.parametrize("said,shuffled", [
    # phrasings the Tier-1 matcher does not claim (a question, a compound,
    # a shape nobody thought of): the model picks the tool, and the WORDS
    # still decide the mode -- the model's own guess never does.
    ("put on the songs I like, shuffled, and tell me the weather", True),
    ("I'd love to hear my liked songs at random", True),
    ("what's in my liked songs? play them, newest first", False),
    ("I'd love to hear my liked songs", False),
])
def test_the_model_path_reads_shuffle_off_the_utterance(liked_brain, said, shuffled):
    """2026-09-02 review: reserving ``shuffle`` closed the model's vote on
    it, so a phrasing the Tier-1 route missed played newest-first however
    plainly he asked for shuffle. The spec's deriver reads his words on the
    model path too, so a missed match can no longer invert the mode."""
    b, ollama, fake_sp, reg = liked_brain
    from jarvis.commander import liked_songs_kind as kind
    assert not kind(said.lower()), "this phrasing must MISS the Tier-1 route"
    ollama.replies = [tool_reply(("spotify_liked", {"shuffle": not shuffled}))]
    tags = b._chat_sync(said)
    assert tags == [("SPEAK", SHUFFLE_LINE if shuffled else ORDER_LINE)]
    # in-order play clears the player's sticky shuffle first; shuffled play
    # is client-side (the URIs are shuffled) and leaves the flag alone
    assert fake_sp.named("shuffle") == ([] if shuffled else
                                        [((False,), {"device_id": "dev-hp"})])


def test_the_configured_default_still_answers_when_he_says_neither(liked_brain, tmp_path):
    """No shuffle word either way on the model path: the deriver says
    nothing and spotify.liked_shuffle decides, as it always did."""
    b, ollama, fake_sp, reg = liked_brain
    ollama.replies = [tool_reply(("spotify_liked", {}))]
    assert b._chat_sync("I'd love to hear my liked songs") == [("SPEAK", ORDER_LINE)]
    # the same turn with the old habit configured back on
    reg2 = ToolRegistry()
    reg2.register_many(make_tool(
        tmp_path, fake_sp,
        cfg={"spotify": {**CFG["spotify"], "liked_shuffle": True}}).tools())
    fake_sp.calls.clear()
    ollama.replies = [tool_reply(("spotify_liked", {}))]
    assert reg2.call("spotify_liked", {}, from_model=True,
                     utterance="I'd love to hear my liked songs").speak == SHUFFLE_LINE


def test_the_forced_call_keeps_the_commanders_shuffle(liked_brain):
    """force_args are the utterance's, so they are trusted whole: "shuffle
    my liked songs" shuffles, "play my liked songs" does not, and neither
    costs a model call at all (the tool speaks for itself)."""
    b, ollama, fake_sp, reg = liked_brain
    tags = b._chat_sync("shuffle my liked songs", force_tool="spotify_liked",
                        force_args={"shuffle": True})
    assert tags == [("SPEAK", SHUFFLE_LINE)]
    assert fake_sp.named("shuffle") == []            # client-side shuffle only
    assert ollama.chat_payloads() == []
    fake_sp.calls.clear()
    tags = b._chat_sync("play my liked songs", force_tool="spotify_liked",
                        force_args={"shuffle": False})
    assert tags == [("SPEAK", ORDER_LINE)]
    assert fake_sp.named("shuffle") == [((False,), {"device_id": "dev-hp"})]
    assert ollama.chat_payloads() == []


def test_the_tool_log_line_carries_the_args_truncated(liked_brain, caplog):
    """brain's "tool X -> ok" line gains the arguments, capped, so the next
    "on shuffle" in the log can be traced to whoever asked for it."""
    b, ollama, fake_sp, reg = liked_brain
    long_name = "x" * 400
    with caplog.at_level("INFO", logger="jarvis.brain"):
        b._chat_sync("play my liked songs", force_tool="spotify_liked",
                     force_args={"shuffle": False, "device": long_name})
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("tool spotify_liked")]
    assert len(lines) == 1, lines
    line = lines[0]
    head, _, rest = line.partition(" -> ")
    args_part = head[len("tool spotify_liked "):]
    assert args_part.startswith('{"device": "xxx')
    assert args_part.endswith("…") and len(args_part) == brain_mod.TOOL_ARGS_LOG_CHARS
    assert rest.startswith("ok=False")             # no device called xxx...
    # the formatter on its own: nothing for no args, str() for odd values
    assert brain_mod._args_for_log({}) == "" and brain_mod._args_for_log(None) == ""
    assert brain_mod._args_for_log({"when": sp.Device("d", "n", "computer", True)}) \
        .startswith(' {"when": "')


# --------------------------------------------------------- 3. the app
def test_the_spoken_command_reaches_the_forced_call_through_the_app(build):  # noqa: F811
    app = build()
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None, force_args=None, **kw: \
        seen.append((text, force_tool, force_args))
    res = app.dispatch_text("play my liked songs")
    assert res.handled and res.done is False
    assert seen == [("play my liked songs", "spotify_liked", {"shuffle": False})]
    seen.clear()
    app.dispatch_text("shuffle my liked songs")
    assert seen == [("shuffle my liked songs", "spotify_liked", {"shuffle": True})]
    # the registered tool is the guarded one: no knob for the model,
    # reserved for the commander
    spec = app.tools.get("spotify_liked")
    assert spec is not None and spec.reserved == frozenset({"shuffle"})
    assert "shuffle" not in spec.parameters["properties"]
    assert "shuffle" not in spec.description.lower()
    # ...and the model path that the route does not claim reads his words
    # itself: the registered spec carries the deriver, not just the ban
    assert spec.derive is not None
    assert spec.derive("play my liked songs shuffled") == {"shuffle": True}
    assert spec.derive("play my liked songs newest first") == {"shuffle": False}
    assert spec.derive("play my liked songs") == {}
