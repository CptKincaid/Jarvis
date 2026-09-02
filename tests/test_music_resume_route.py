""""Start playing my Spotify" is a Tier-1 route onto spotify_control resume.

LIVE 2026-09-01 20:56:42, by voice: "Say hello to my family and then add
milk to my shopping list and then start playing my Spotify." went to the
model whole -- the greeting clause is not Tier-1, so _try_multi's
all-or-nothing rule could not split it -- and gemma4 answered "I'm
starting your music now" having called NO tool. The brain's unbacked-
action guard (tests/test_unbacked_actions.py) catches the narration; this
route removes the model from the request: a bare "start / resume / play my
music" has one meaning and one tool action, so the commander forces
spotify_control resume (force_tool/force_args) and the tool's own
"Resumed, sir." is spoken.

Real Commander + real Router with a mocked brain (the rich fixture), and
the real JarvisApp with hardware stubbed for the wiring. No Ollama, no
network, no audio.
"""
import types
from unittest.mock import MagicMock

import pytest

from jarvis.commander import liked_songs_kind, music_resume_kind
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)
from tests.test_route_shortcut import Cfg, rich  # noqa: F401  (fixture)


def _only_forced(svc, tool):
    assert svc.brain.chat.call_count == 1, svc.brain.chat.call_args_list
    call = svc.brain.chat.call_args
    assert call.kwargs.get("force_tool") == tool, call
    return call.kwargs["force_args"]


# --------------------------------------------------------- the route
@pytest.mark.parametrize("said", [
    "start playing my Spotify",            # his 20:56:42 clause
    "start my Spotify",
    "start my music",
    "start the music",
    "start playing my music",
    "resume my music",
    "resume the music",
    "resume playback",
    "resume spotify",
    "play my music",
    "play music",
    "play some music",
    "play me some music",
    "play spotify",
    "put my music on",
    "put some music on",
    "put on some music please",
    "turn the music back on",
    "unpause the music",
    "could you start my music please",
    "play my tunes",
])
@pytest.mark.parametrize("source", ["voice", "typed"])
def test_a_bare_music_request_forces_spotify_control_resume(rich, said, source):  # noqa: F811
    c, svc = rich
    res = c.handle(said, source=source)
    assert res.handled and res.done is False, res
    assert _only_forced(svc, "spotify_control") == {"action": "resume"}
    # the route is the matcher's, not the router's: no classify, no model
    assert svc.brain.chat.call_args.args == (said.lower(),)


def test_the_prefixed_form_takes_the_registry_pass(rich):  # noqa: F811
    c, svc = rich
    res = c.handle("Jarvis, start playing my Spotify", source="typed")
    assert res.handled
    assert _only_forced(svc, "spotify_control") == {"action": "resume"}
    assert svc.brain.chat.call_args.args == ("start playing my spotify",)


def test_a_connect_device_rides_along(rich):  # noqa: F811
    c, svc = rich
    c.handle("play my music on my phone", source="voice")
    assert _only_forced(svc, "spotify_control") == {"action": "resume", "device": "phone"}


@pytest.mark.parametrize("said", [
    "play Drake",                         # an artist: the model's search
    "play some jazz",                     # a genre
    "play my playlist",                   # a playlist with no name
    "play my liked songs",                # the liked-songs route (asserted below)
    "play my music on shuffle",           # a mode the route cannot set
    "start spotify",                      # launching the app, not resuming
    "open spotify",
    "play",                               # the desktop media key
    "resume",                             # the reader's word when reading
    "pause my music",                     # not a resume
    "play music by Drake",
    "play my music from the top",
    "what music is playing",
])
def test_anything_else_keeps_its_old_path(rich, said):  # noqa: F811
    c, svc = rich
    assert not music_resume_kind(said.lower())
    c.handle(said, source="typed")
    for call in svc.brain.chat.call_args_list:
        assert call.kwargs.get("force_tool") != "spotify_control", call


def test_liked_songs_wins_over_the_bare_resume(rich):  # noqa: F811
    """"play my liked songs" is the liked-songs route, never a resume --
    the matcher refuses it on its own, and the REGISTRY order agrees."""
    c, svc = rich
    assert liked_songs_kind("play my liked songs") and not music_resume_kind("play my liked songs")
    c.handle("play my liked songs", source="voice")
    assert _only_forced(svc, "spotify_liked") == {"shuffle": False}


def test_the_compound_now_splits_when_every_clause_is_tier_1(rich):  # noqa: F811
    """"add milk to my shopping list and then start playing my Spotify":
    with the resume clause a Tier-1 name, _try_multi runs both halves
    itself instead of handing the pair to the model."""
    c, svc = rich
    res = c.handle("set a ten minute timer and then start playing my spotify",
                   source="voice")
    assert res.handled and res.status == "Timer set: 10 minutes + Music…"
    svc.timekeeper.add_timer.assert_called_once_with(600, "10 minutes timer")
    assert _only_forced(svc, "spotify_control") == {"action": "resume"}
    assert svc.brain.chat.call_args.args == ("start playing my spotify",)


def test_the_route_needs_a_brain_that_chats(rich):  # noqa: F811
    c, svc = rich
    svc.brain = types.SimpleNamespace(think=MagicMock())
    res = c.handle("start my music", source="typed")
    assert res.handled
    svc.brain.think.assert_called_once()


# ------------------------------------------------------------ the app
def test_the_spoken_request_reaches_the_forced_call_through_the_app(build):  # noqa: F811
    """The real wiring: dispatch_text -> commander -> services.brain.chat
    -> the forced spotify_control call, with the tool registered."""
    app = build()
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None, force_args=None, **kw: \
        seen.append((text, force_tool, force_args))
    res = app.dispatch_text("start playing my Spotify")
    assert res.handled and res.done is False
    assert seen == [("start playing my spotify", "spotify_control", {"action": "resume"})]
    assert app.tools.get("spotify_control") is not None
