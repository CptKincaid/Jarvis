"""Unit tests for jarvis.tools.spotify — every path against a FakeSpotify
client (no network): auth/setup excuses, token cache 0600 + refresh, device
resolution, search & play with type inference (own playlists first), Liked
Songs (both strategies and their fallbacks), transport/volume/now-playing/
like, queue, radio without /recommendations, error translation (no device,
no results, expired token, Free-account 403) and the persona wording rules.

Firewall: tests/conftest.py points JARVIS_ASSISTANT_CONFIG at a tmp file,
so the default token path lands in tmp too; every tool here also passes an
explicit tmp ``token_path``.  The live check runs only with JARVIS_LIVE=1
and is read-only (devices + now playing).
"""
import copy
import inspect
import json
import os
import random
import re
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.tools import spotify as sp
from jarvis.tools.registry import ToolRegistry, ToolResult

FIX = Path(__file__).parent / "fixtures"
CFG = {"spotify": {"client_id": "abc123clientid", "client_secret": "abc123clientsecret",
                   "default_device": "HPCOMPUTER"}}
UNSET = {"spotify": {"client_id": "", "client_secret": "<paste here>",
                     "default_device": "HPCOMPUTER"}}
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
_MARKDOWN = re.compile(r"(^|\n)\s*([-*•]\s+|\d+[.)]\s+|#{1,6}\s+)|\*\*|`|__", re.M)


def load(name):
    return json.loads((FIX / f"spotify_{name}.json").read_text(encoding="utf-8"))


def count_sentences(text):
    text = text.strip().replace("...", "…")
    return len([p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()])


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


# ------------------------------------------------------------ the fake
class FakeSpotify:
    """Enough of spotipy.Spotify for the tool, driven by the fixtures.
    Records every call as (method, args, kwargs)."""

    def __init__(self, devices=None, premium=True, max_uris=100, collection_ok=True,
                 expire_first=False, n_saved=None, playlists=None, search=None,
                 now_playing="playing", fail=None, top_tracks=None):
        self.calls = []
        self._devices = copy.deepcopy(devices if devices is not None
                                      else load("devices")["devices"])
        self.premium, self.max_uris, self.collection_ok = premium, max_uris, collection_ok
        self.expire_first, self._expired_once = expire_first, False
        self.n_saved = n_saved
        self._playlists = playlists if playlists is not None else load("playlists")
        self._search = search if search is not None else load("search")
        self._top = top_tracks if top_tracks is not None else load("top_tracks")
        self.now_playing = now_playing
        self.saved_ids = set()
        self.fail = fail

    # helpers
    def named(self, method):
        return [(a, kw) for m, a, kw in self.calls if m == method]

    def _rec(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if self.fail is not None:
            raise self.fail
        if self.expire_first and not self._expired_once:
            self._expired_once = True
            raise sp.SpotifyException(401, -1, "The access token expired")

    def _premium(self):
        if not self.premium:
            raise sp.SpotifyException(403, -1, "Player command failed: Premium required",
                                      reason="PREMIUM_REQUIRED")

    def _device(self, device_id):
        if device_id:
            if not any(d["id"] == device_id for d in self._devices):
                raise sp.SpotifyException(404, -1, "Device not found", reason="NO_ACTIVE_DEVICE")
            return
        if not any(d["is_active"] for d in self._devices):
            raise sp.SpotifyException(404, -1, "Player command failed: No active device found",
                                      reason="NO_ACTIVE_DEVICE")

    # identity / devices / state
    def me(self):
        self._rec("me")
        return {"id": "hunter", "display_name": "Hunter"}

    def devices(self):
        self._rec("devices")
        return {"devices": copy.deepcopy(self._devices)}

    def current_playback(self, market=None, additional_types=None):
        self._rec("current_playback", market=market, additional_types=additional_types)
        if self.now_playing is None:
            return None
        pb = load("now_playing")
        pb["is_playing"] = self.now_playing == "playing"
        return pb

    # catalogue
    @staticmethod
    def _matches(item, nq):
        name = _norm(item.get("name"))
        if nq in name or (name and name in nq):
            return True
        return any(nq in _norm(a.get("name")) for a in item.get("artists", []) or [])

    def search(self, q, limit=10, offset=0, type="track", market=None):
        self._rec("search", q=q, limit=limit, type=type, market=market)
        nq = _norm(q)
        out = {}
        for t in type.split(","):
            key = t + "s"
            items = (self._search.get(key) or {}).get("items", [])
            hits = [it for it in items if it is None or self._matches(it, nq)]
            out[key] = {"items": hits[:limit]}
        return out

    def current_user_playlists(self, limit=50, offset=0):
        self._rec("current_user_playlists", limit=limit, offset=offset)
        return copy.deepcopy(self._playlists) if offset == 0 else {"items": [], "next": None}

    def current_user_saved_tracks(self, limit=20, offset=0, market=None):
        self._rec("current_user_saved_tracks", limit=limit, offset=offset, market=market)
        page = load("saved_tracks")
        if self.n_saved is None:
            return page if offset == 0 else {"items": [], "next": None}
        tmpl = page["items"][0]
        items = []
        for i in range(offset, min(offset + limit, self.n_saved)):
            it = copy.deepcopy(tmpl)
            it["track"]["id"] = f"liked-{i}"
            it["track"]["uri"] = f"spotify:track:liked-{i}"
            items.append(it)
        nxt = "more" if offset + limit < self.n_saved else None
        return {"items": items, "next": nxt, "total": self.n_saved}

    def artist_top_tracks(self, artist_id, country="US"):
        self._rec("artist_top_tracks", artist_id, country=country)
        return copy.deepcopy(self._top) if artist_id == "art-the-weeknd" else {"tracks": []}

    # player
    def transfer_playback(self, device_id, force_play=True):
        self._rec("transfer_playback", device_id, force_play=force_play)
        self._premium()
        self._device(device_id)
        for d in self._devices:
            d["is_active"] = d["id"] == device_id

    def start_playback(self, device_id=None, context_uri=None, uris=None, offset=None,
                       position_ms=None):
        self._rec("start_playback", device_id=device_id, context_uri=context_uri,
                  uris=uris, offset=offset, position_ms=position_ms)
        self._premium()
        self._device(device_id)
        if uris is not None and len(uris) > self.max_uris:
            raise sp.SpotifyException(400, -1, "Too many uris")
        if context_uri and context_uri.endswith(":collection") and not self.collection_ok:
            raise sp.SpotifyException(403, -1, "Player command failed: Restriction violated",
                                      reason="UNKNOWN")
        self.now_playing = "playing"

    def _simple(self, method, device_id, *args, **kwargs):
        self._rec(method, *args, device_id=device_id, **kwargs)
        self._premium()
        self._device(device_id)

    def pause_playback(self, device_id=None):
        self._simple("pause_playback", device_id)
        self.now_playing = "paused"

    def next_track(self, device_id=None):
        self._simple("next_track", device_id)

    def previous_track(self, device_id=None):
        self._simple("previous_track", device_id)

    def seek_track(self, position_ms, device_id=None):
        self._simple("seek_track", device_id, position_ms)

    def repeat(self, state, device_id=None):
        self._simple("repeat", device_id, state)

    def volume(self, volume_percent, device_id=None):
        self._simple("volume", device_id, volume_percent)

    def shuffle(self, state, device_id=None):
        self._simple("shuffle", device_id, state)

    def add_to_queue(self, uri, device_id=None):
        self._simple("add_to_queue", device_id, uri)

    # library
    def current_user_saved_tracks_add(self, tracks=None):
        self._rec("current_user_saved_tracks_add", tracks)
        self.saved_ids.update(tracks or [])

    def current_user_saved_tracks_contains(self, tracks=None):
        self._rec("current_user_saved_tracks_contains", tracks)
        return [t in self.saved_ids for t in (tracks or [])]


class FakeAuth:
    """spotipy.SpotifyOAuth stand-in: refresh + the interactive login."""

    def __init__(self, client_id, client_secret, cache, open_browser):
        self.client_id, self.cache, self.open_browser = client_id, cache, open_browser
        self.refreshed = []
        self.opened = None

    def refresh_access_token(self, refresh_token):
        self.refreshed.append(refresh_token)
        tok = {"access_token": "new-access", "refresh_token": refresh_token,
               "expires_at": 9_999_999_999, "expires_in": 3600, "scope": sp.SCOPES,
               "token_type": "Bearer"}
        self.cache.save_token_to_cache(tok)
        return tok

    def get_authorize_url(self, state=None):
        return "https://accounts.spotify.com/authorize?client_id=" + self.client_id

    def get_auth_response(self, open_browser=None):
        self.opened = open_browser
        return "AUTH-CODE"

    def get_access_token(self, code=None, as_dict=True, check_cache=True):
        assert code == "AUTH-CODE" and not check_cache
        return {"access_token": "first-access", "refresh_token": "first-refresh",
                "expires_at": 9_999_999_999, "expires_in": 3600, "scope": sp.SCOPES,
                "token_type": "Bearer"}


# ------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def _firewall(monkeypatch):
    real = Path.home() / ".config" / "jarvis"
    monkeypatch.delenv(sp.TOKEN_ENV, raising=False)
    tp = sp.token_path()
    assert tp.parent != real and real not in tp.parents, "token path still live"
    yield


def make_tool(tmp_path, client=None, cfg=CFG, **kw):
    kw.setdefault("spawn", lambda fn: fn())
    kw.setdefault("rng", random.Random(0))
    kw.setdefault("settle", lambda s: None)
    return sp.SpotifyTool(cfg, client=client, token_path=tmp_path / "spotify_token.json", **kw)


def registry(tool):
    reg = ToolRegistry()
    reg.register_many(tool.tools())
    return reg


@pytest.fixture
def fake():
    return FakeSpotify()


@pytest.fixture
def reg(tmp_path, fake):
    return registry(make_tool(tmp_path, fake))


def start_calls(fake):
    return [kw for _, kw in fake.named("start_playback")]


# ------------------------------------------------------- pure helpers
@pytest.mark.parametrize("query, kind, cleaned", [
    ("Blinding Lights", "auto", "Blinding Lights"),
    ("play Blinding Lights", "auto", "Blinding Lights"),
    ("blinding lights by the weeknd", "track", "blinding lights by the weeknd"),
    ("songs by the weeknd", "artist", "the weeknd"),
    ("some music by Daft Punk", "artist", "daft punk"),
    ("something by the weeknd", "artist", "the weeknd"),
    ("the artist Daft Punk", "artist", "daft punk"),
    ("some daft punk", "artist", "daft punk"),
    ("the album After Hours", "album", "after hours"),
    ("After Hours album", "album", "after hours"),
    ("my gym mix", "playlist", "gym mix"),
    ("my gym mix playlist", "playlist", "gym mix"),
    ("the chill hits playlist", "playlist", "chill hits"),
    ("playlist called Focus", "playlist", "focus"),
    ("Discover Weekly", "playlist", "Discover Weekly"),
    ("release radar", "playlist", "release radar"),
    ("my liked songs", "liked", "my liked songs"),
    ("liked songs", "liked", "liked songs"),
    ("my library", "liked", "my library"),
    ("Blinding Lights on spotify", "auto", "Blinding Lights"),
    ("", "auto", ""),
])
def test_infer_kind_table(query, kind, cleaned):
    assert sp.infer_kind(query) == (kind, cleaned)


@pytest.mark.parametrize("value, kind", [
    ("Song", "track"), ("TRACK", "track"), ("band", "artist"), ("Album", "album"),
    ("playlist", "playlist"), ("liked", "liked"), ("", "auto"), (None, "auto"),
    ("whatever", "auto")])
def test_coerce_kind(value, kind):
    assert sp.coerce_kind(value) == kind


@pytest.mark.parametrize("query, rest, device", [
    ("Blinding Lights on my phone", "Blinding Lights", "phone"),
    ("Blinding Lights on the tv", "Blinding Lights", "tv"),
    ("Blinding Lights on HPCOMPUTER", "Blinding Lights", "HPCOMPUTER"),
    ("Blinding Lights on the laptop", "Blinding Lights", "laptop"),
    ("Blinding Lights", "Blinding Lights", None),
    ("Dancing on My Own", "Dancing on My Own", None),     # not a device word
    ("Living on a Prayer", "Living on a Prayer", None),
])
def test_split_device(query, rest, device):
    assert sp.split_device(query) == (rest, device)


@pytest.mark.parametrize("value, out", [
    ("my phone", "phone"), ("on the tv", "tv"), ("HPCOMPUTER", "HPCOMPUTER"),
    ("", None), (None, None), ("the phone.", "phone")])
def test_clean_device_name(value, out):
    assert sp.clean_device_name(value) == out


@pytest.mark.parametrize("value, current, out", [
    ("40", None, 40), ("40%", None, 40), (40, None, 40), (40.4, None, 40),
    ("140", None, 100), ("-5", None, 5), ("0.5", None, 50), ("up", 40, 50),
    ("louder", 95, 100), ("down", 40, 30), ("quieter", 5, 0), ("up 20", 40, 60),
    ("max", 10, 100), ("mute", 10, 0), ("half", 10, 50), ("", None, None),
    (None, None, None), ("purple", None, None), (True, None, None)])
def test_parse_volume(value, current, out):
    assert sp.parse_volume(value, current) == out


@pytest.mark.parametrize("value, out", [
    ("1:30", 90_000), ("90", 90_000), ("90 seconds", 90_000), ("2 minutes", 120_000),
    ("1.5 min", 90_000), (45, 45_000), ("start", 0), ("beginning", 0), ("", None),
    (None, None), ("later", None)])
def test_parse_seek(value, out):
    assert sp.parse_seek(value) == out


def test_parse_onoff_and_repeat():
    assert sp.parse_onoff("on") is True and sp.parse_onoff("off") is False
    assert sp.parse_onoff("") is True and sp.parse_onoff(None, default=False) is False
    assert sp.parse_onoff("maybe") is None
    assert sp.parse_repeat("") == "context" and sp.parse_repeat("all") == "context"
    assert sp.parse_repeat("one") == "track" and sp.parse_repeat("this song") == "track"
    assert sp.parse_repeat("off") == "off" and sp.parse_repeat("sideways") is None
    assert sp.fmt_ms(90_000) == "1:30" and sp.fmt_ms(0) == "0:00" and sp.fmt_ms(-5) == "0:00"


def test_items_drops_none_entries():
    res = {"playlists": {"items": [None, {"name": "a"}, None]}}
    assert sp._items(res, "playlists") == [{"name": "a"}]
    assert sp._items({"items": [None]}) == []
    assert sp._items(None, "tracks") == []


def test_cfg_get_reads_dicts_objects_and_none():
    assert sp._cfg_get({"a": {"b": 1}}, "a.b") == 1
    assert sp._cfg_get({"a": {"b": 1}}, "a.c", 7) == 7
    assert sp._cfg_get(None, "a.b", "d") == "d"
    obj = SimpleNamespace(get=lambda dotted, default=None: {"a.b": 2}.get(dotted, default))
    assert sp._cfg_get(obj, "a.b") == 2 and sp._cfg_get(obj, "x", 3) == 3


# -------------------------------------------------------- token cache
def test_token_cache_writes_0600_and_round_trips(tmp_path):
    cache = sp.TokenCache(tmp_path / "deep" / "spotify_token.json")
    assert cache.get_cached_token() is None and not cache.linked()
    cache.save_token_to_cache({"access_token": "a", "refresh_token": "r"})
    mode = stat.S_IMODE(cache.path.stat().st_mode)
    assert mode == 0o600, oct(mode)
    assert stat.S_IMODE(cache.path.parent.stat().st_mode) == 0o700
    assert cache.get_cached_token() == {"access_token": "a", "refresh_token": "r"}
    assert cache.linked()
    assert not list(cache.path.parent.glob(".spotify-*.tmp"))
    cache.path.write_text("{not json")
    assert cache.get_cached_token() is None
    cache.clear()
    assert not cache.path.exists()
    cache.clear()                                   # idempotent


def test_token_path_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "cfg" / "assistant.json"))
    monkeypatch.delenv(sp.TOKEN_ENV, raising=False)
    from jarvis.config import PATHS
    monkeypatch.setattr(PATHS, "ASSISTANT_CONFIG", tmp_path / "cfg" / "assistant.json")
    assert sp.token_path() == tmp_path / "cfg" / sp.TOKEN_FILE
    cfg = SimpleNamespace(path=tmp_path / "other" / "assistant.json")
    assert sp.token_path(cfg) == tmp_path / "other" / sp.TOKEN_FILE
    monkeypatch.setenv(sp.TOKEN_ENV, str(tmp_path / "env.json"))
    assert sp.token_path(cfg) == tmp_path / "env.json"
    assert sp.token_path(cfg, tmp_path / "explicit.json") == tmp_path / "explicit.json"


def test_token_sits_next_to_the_assistant_config(tmp_path):
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig(path=tmp_path / "assistant.json")
    tool = sp.SpotifyTool(cfg)
    assert tool.cache.path == tmp_path / sp.TOKEN_FILE


# ------------------------------------------------------------- setup
def test_unconfigured_speaks_setup_line_for_every_tool(tmp_path):
    reg = registry(make_tool(tmp_path, FakeSpotify(), cfg=UNSET))
    for name in reg.names():
        res = reg.call(name, {"query": "x", "action": "pause"})
        assert res.ok is False and res.speak == sp.SETUP_LINE, name
        assert res.text == sp.SETUP_LINE


def test_unconfigured_uses_assistant_config_setup_line(tmp_path):
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig(path=tmp_path / "assistant.json")
    reg = registry(sp.SpotifyTool(cfg, client=FakeSpotify()))
    res = reg.call("spotify_now_playing", {})
    assert res.ok is False and res.speak == cfg.setup_line("spotify")
    assert "set up, sir" in res.speak and "docs/assistant-setup.md" in res.speak


def test_configured_but_not_linked(tmp_path):
    tool = make_tool(tmp_path)                       # real spotipy, no token
    assert tool.configured() and not tool.linked()
    res = registry(tool).call("spotify_play", {"query": "Blinding Lights"})
    assert res.ok is False and res.speak == sp.NOT_LINKED_LINE
    assert "--spotify-login" in res.speak


def test_missing_spotipy_is_a_spoken_excuse(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "spotipy", None)
    res = registry(make_tool(tmp_path)).call("spotify_now_playing", {})
    assert res.ok is False and res.speak == sp.MISSING_LIB_LINE


def test_assistant_config_with_real_keys_is_configured(tmp_path):
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig(path=tmp_path / "assistant.json")
    cfg.update({"spotify.client_id": "abc123clientid",
                "spotify.client_secret": "abc123clientsecret",
                "spotify.default_device": "HPCOMPUTER"})
    tool = sp.SpotifyTool(cfg, client=FakeSpotify())
    assert tool.configured() and tool.default_device == "HPCOMPUTER"
    assert tool.now_playing().speak == "Blinding Lights by The Weeknd, sir — on HPCOMPUTER."


# ---------------------------------------------------------- registry
def test_make_tools_contract(tmp_path):
    services = SimpleNamespace()
    specs = sp.make_tools(CFG, services)
    names = [s.name for s in specs]
    assert names == ["spotify_play", "spotify_liked", "spotify_control",
                     "spotify_now_playing", "spotify_queue", "spotify_radio"]
    assert isinstance(services.spotify, sp.SpotifyTool)
    for s in specs:
        assert len(s.description.split()) <= 20, s.name
        assert s.parameters["type"] == "object"
        json.dumps(s.schema())
    by = {s.name: s for s in specs}
    assert by["spotify_play"].parameters["properties"]["kind"]["enum"] == list(sp.KINDS)
    assert by["spotify_control"].parameters["properties"]["action"]["enum"] == list(sp.ACTIONS)
    assert by["spotify_play"].parameters["required"] == ["query"]
    assert sp.make_tools(CFG, None)                  # services may be absent
    assert len(sp.make_tools(None, SimpleNamespace(spotify="taken"))) == 6


def test_handlers_tolerate_extra_kwargs_and_loose_values(reg, fake):
    res = reg.call("spotify_now_playing", {"bogus": 1, "device": "x"})
    assert res.ok and res.speak
    res = reg.call("spotify_control", {"action": "Volume", "value": "40", "extra": True})
    assert res.speak == "Volume 40, sir."
    res = reg.call("spotify_play", {"query": "Blinding Lights", "kind": "SONG"})
    assert res.speak.startswith("Blinding Lights by The Weeknd")


# ----------------------------------------------------------- devices
def test_play_prefers_default_device_over_active_and_transfers(reg, fake):
    res = reg.call("spotify_play", {"query": "Blinding Lights"})
    assert res.ok and res.speak == "Blinding Lights by The Weeknd, sir — on HPCOMPUTER."
    assert fake.named("transfer_playback") == [(("dev-hp",), {"force_play": False})]
    assert start_calls(fake) == [{"device_id": "dev-hp", "context_uri": None,
                                  "uris": ["spotify:track:t-blinding"], "offset": None,
                                  "position_ms": None}]


def test_named_device_wins_and_no_transfer_when_active(reg, fake):
    res = reg.call("spotify_play", {"query": "Blinding Lights", "device": "my phone"})
    assert res.speak == "Blinding Lights by The Weeknd, sir — on Hunter's iPhone."
    assert fake.named("transfer_playback") == []
    assert start_calls(fake)[0]["device_id"] == "dev-phone"


def test_device_in_the_query_is_peeled_off(reg, fake):
    res = reg.call("spotify_play", {"query": "Blinding Lights on my phone"})
    assert res.speak.endswith("on Hunter's iPhone.")
    assert fake.named("search")[0][1]["q"] == "Blinding Lights"


def test_device_matching_by_type_and_substring(tmp_path):
    tool = make_tool(tmp_path, FakeSpotify())
    assert tool.resolve_device("iphone").id == "dev-phone"
    assert tool.resolve_device("hp").id == "dev-hp"
    assert tool.resolve_device("the computer").id == "dev-hp"
    assert tool.resolve_device("hpcomputer").id == "dev-hp"


def test_unknown_named_device_lists_what_is_seen(reg):
    res = reg.call("spotify_play", {"query": "Blinding Lights", "device": "toaster"})
    assert res.ok is False
    assert res.speak == "I can't see toaster on Spotify, sir; I can see HPCOMPUTER, Hunter's iPhone."


def test_no_devices_at_all_is_the_nothing_listening_line(tmp_path):
    fake = FakeSpotify(devices=[])
    reg = registry(make_tool(tmp_path, fake))
    res = reg.call("spotify_play", {"query": "Blinding Lights"})
    assert res.ok is False
    assert res.speak == "Nothing's listening, sir — open Spotify on HPCOMPUTER or your phone."
    res = reg.call("spotify_play", {"query": "Blinding Lights", "device": "phone"})
    assert res.speak == "I can't see phone on Spotify, sir; nothing's listening."
    assert start_calls(fake) == []


def test_only_an_inactive_non_default_device_is_still_nothing_listening(tmp_path):
    devs = load("devices")["devices"]
    phone_only = [d | {"is_active": False} for d in devs if d["id"] == "dev-phone"]
    reg = registry(make_tool(tmp_path, FakeSpotify(devices=phone_only)))
    res = reg.call("spotify_play", {"query": "Blinding Lights"})
    assert res.speak.startswith("Nothing's listening, sir")
    res = reg.call("spotify_play", {"query": "Blinding Lights", "device": "phone"})
    assert res.ok and res.speak.endswith("on Hunter's iPhone.")


def test_falls_back_to_active_device_when_default_absent(tmp_path):
    devs = [d for d in load("devices")["devices"] if d["id"] == "dev-phone"]
    fake = FakeSpotify(devices=devs)
    res = registry(make_tool(tmp_path, fake)).call("spotify_play", {"query": "Blinding Lights"})
    assert res.speak.endswith("on Hunter's iPhone.")
    assert fake.named("transfer_playback") == []


def test_default_device_name_comes_from_config(tmp_path):
    cfg = {"spotify": {**CFG["spotify"], "default_device": "Hunter's iPhone"}}
    devs = [d | {"is_active": False} for d in load("devices")["devices"]]
    fake = FakeSpotify(devices=devs)
    res = registry(make_tool(tmp_path, fake, cfg=cfg)).call(
        "spotify_play", {"query": "Blinding Lights"})
    assert res.speak.endswith("on Hunter's iPhone.")
    assert fake.named("transfer_playback")[0][0] == ("dev-phone",)
    tool = make_tool(tmp_path, FakeSpotify(devices=[]), cfg=cfg)
    with pytest.raises(sp.SpotifyError) as exc:
        tool.resolve_device()
    assert "open Spotify on Hunter's iPhone or your phone" in exc.value.line


# -------------------------------------------------------------- play
def test_play_artist_uses_top_tracks(reg, fake):
    res = reg.call("spotify_play", {"query": "The Weeknd"})
    assert res.speak == "The Weeknd's top tracks, sir — on HPCOMPUTER."
    assert fake.named("artist_top_tracks")[0][0] == ("art-the-weeknd",)
    uris = start_calls(fake)[0]["uris"]
    assert len(uris) == 5 and uris[0] == "spotify:track:top-1"


def test_play_songs_by_artist_phrase(reg, fake):
    res = reg.call("spotify_play", {"query": "songs by the weeknd"})
    assert res.speak == "The Weeknd's top tracks, sir — on HPCOMPUTER."


def test_play_artist_without_top_tracks_or_this_is(reg, fake):
    res = reg.call("spotify_play", {"query": "Daft Punk", "kind": "artist"})
    assert res.ok is False and res.speak == "I can't find anything of Daft Punk's to play, sir."
    assert start_calls(fake) == []


def test_play_album_by_exact_name_and_by_kind(reg, fake):
    res = reg.call("spotify_play", {"query": "After Hours"})
    assert res.speak == "After Hours by The Weeknd, sir — the whole album on HPCOMPUTER."
    assert start_calls(fake)[-1]["context_uri"] == "spotify:album:alb-after-hours"
    res = reg.call("spotify_play", {"query": "the album Discovery"})
    assert res.speak == "Discovery by Daft Punk, sir — the whole album on HPCOMPUTER."
    assert start_calls(fake)[-1]["context_uri"] == "spotify:album:alb-discovery"


def test_play_my_playlist_searches_own_playlists_first(reg, fake):
    res = reg.call("spotify_play", {"query": "my gym mix"})
    assert res.speak == "Gym Mix, sir — on HPCOMPUTER."
    assert start_calls(fake)[0]["context_uri"] == "spotify:playlist:pl-gym"
    assert fake.named("search") == []                 # never hit the catalogue
    res = reg.call("spotify_play", {"query": "gym mix"})   # auto: own exact match too
    assert res.speak == "Gym Mix, sir — on HPCOMPUTER."
    assert fake.named("search") == []


def test_play_discover_weekly_is_an_own_playlist(reg, fake):
    res = reg.call("spotify_play", {"query": "Discover Weekly"})
    assert res.speak == "Discover Weekly, sir — on HPCOMPUTER."
    assert start_calls(fake)[0]["context_uri"] == "spotify:playlist:pl-discover"


def test_own_playlists_are_cached_for_five_minutes(tmp_path, fake):
    clock = [1000.0]
    tool = make_tool(tmp_path, fake, now=lambda: clock[0])
    tool.play("my gym mix")
    tool.play("my gym mix")
    assert len(fake.named("current_user_playlists")) == 1
    clock[0] += sp.PLAYLIST_CACHE_S + 1
    tool.play("my gym mix")
    assert len(fake.named("current_user_playlists")) == 2


def test_play_public_playlist_prefers_spotify_owned_and_skips_none_items(reg, fake):
    res = reg.call("spotify_play", {"query": "This Is The Weeknd", "kind": "playlist"})
    assert res.speak == "This Is The Weeknd, sir — on HPCOMPUTER."
    assert start_calls(fake)[0]["context_uri"] == "spotify:playlist:pl-this-is-weeknd"
    res = reg.call("spotify_play", {"query": "chill hits playlist"})
    assert res.speak == "Chill Hits, sir — on HPCOMPUTER."


def test_play_track_by_phrase_and_no_results(reg, fake):
    res = reg.call("spotify_play", {"query": "one more time by daft punk"})
    assert res.speak == "One More Time by Daft Punk, sir — on HPCOMPUTER."
    res = reg.call("spotify_play", {"query": "xyzzy plugh"})
    assert res.ok is False and res.speak == "I couldn't find xyzzy plugh on Spotify, sir."


def test_play_with_empty_query_resumes(reg, fake):
    res = reg.call("spotify_play", {"query": ""})
    assert res.speak == "Resumed, sir."
    assert start_calls(fake) == [{"device_id": "dev-phone", "context_uri": None,
                                  "uris": None, "offset": None, "position_ms": None}]


def test_play_liked_songs_phrase_routes_to_liked(reg, fake):
    res = reg.call("spotify_play", {"query": "my liked songs"})
    assert res.speak == "Your Liked Songs on shuffle, sir — 4 of them, on HPCOMPUTER."


def test_track_with_two_artists_names_both():
    item = {"artists": [{"name": "The Weeknd"}, {"name": "Daft Punk"}, {"name": "X"}]}
    assert sp._artist_names(item) == "The Weeknd and Daft Punk"
    assert sp._artist_names({}) == "an unknown artist"


# ------------------------------------------------------- liked songs
def test_liked_uris_strategy_pages_shuffles_chunks_and_queues(tmp_path):
    fake = FakeSpotify(n_saved=230)
    tool = make_tool(tmp_path, fake)
    res = tool.liked()
    assert res.speak == "Your Liked Songs on shuffle, sir — 230 of them, on HPCOMPUTER."
    pages = fake.named("current_user_saved_tracks")
    assert len(pages) == 5 and all(kw["limit"] == 50 for _, kw in pages)
    starts = start_calls(fake)
    assert len(starts) == 1 and len(starts[0]["uris"]) == 100
    assert starts[0]["uris"] != [f"spotify:track:liked-{i}" for i in range(100)]  # shuffled
    assert len(set(starts[0]["uris"])) == 100
    queued = fake.named("add_to_queue")
    assert len(queued) == sp.LIKED_QUEUE_AHEAD
    assert all(kw["device_id"] == "dev-hp" for _, kw in queued)
    assert not set(a[0] for a, _ in queued) & set(starts[0]["uris"])
    assert fake.named("shuffle") == []                 # client-side shuffle only


def test_liked_caps_at_500_and_skips_local_files(tmp_path):
    fake = FakeSpotify(n_saved=900)
    tool = make_tool(tmp_path, fake)
    assert len(tool._liked_uris()) == 500
    assert len(fake.named("current_user_saved_tracks")) == 10
    fake2 = FakeSpotify()                              # fixture page: 5 items, 1 local
    assert make_tool(tmp_path, fake2)._liked_uris() == [
        "spotify:track:liked-1", "spotify:track:liked-2", "spotify:track:liked-3",
        "spotify:track:liked-5"]


def test_liked_collection_strategy(tmp_path):
    fake = FakeSpotify(n_saved=230)
    cfg = {"spotify": {**CFG["spotify"], "liked_strategy": "collection"}}
    res = make_tool(tmp_path, fake, cfg=cfg).liked()
    assert res.speak == "Your Liked Songs on shuffle, sir — on HPCOMPUTER."
    assert fake.named("shuffle") == [((True,), {"device_id": "dev-hp"})]
    assert start_calls(fake) == [{"device_id": "dev-hp",
                                  "context_uri": "spotify:user:hunter:collection",
                                  "uris": None, "offset": None, "position_ms": None}]
    assert fake.named("add_to_queue") == []


def test_liked_collection_falls_back_to_uris(tmp_path):
    fake = FakeSpotify(n_saved=120, collection_ok=False)
    cfg = {"spotify": {**CFG["spotify"], "liked_strategy": "collection"}}
    res = make_tool(tmp_path, fake, cfg=cfg).liked()
    assert res.speak == "Your Liked Songs on shuffle, sir — 120 of them, on HPCOMPUTER."
    starts = start_calls(fake)
    assert starts[0]["context_uri"] == "spotify:user:hunter:collection"
    assert len(starts) == 2 and len(starts[1]["uris"]) == 100


def test_liked_uris_falls_back_to_collection(tmp_path):
    fake = FakeSpotify(n_saved=120, max_uris=10)        # the 100-chunk is rejected
    res = make_tool(tmp_path, fake).liked()
    assert res.speak == "Your Liked Songs on shuffle, sir — on HPCOMPUTER."
    starts = start_calls(fake)
    assert len(starts) == 2 and starts[1]["context_uri"] == "spotify:user:hunter:collection"


def test_liked_chunk_and_queue_ahead_are_configurable(tmp_path):
    fake = FakeSpotify(n_saved=60)
    cfg = {"spotify": {**CFG["spotify"], "liked_chunk": 25, "liked_queue_ahead": 5,
                       "liked_cap": 40}}
    res = make_tool(tmp_path, fake, cfg=cfg).liked()
    assert res.speak.endswith("40 of them, on HPCOMPUTER.")
    assert len(start_calls(fake)[0]["uris"]) == 25
    assert len(fake.named("add_to_queue")) == 5


def test_liked_queue_runs_on_a_worker_by_default(tmp_path):
    fake = FakeSpotify(n_saved=130)
    tool = sp.SpotifyTool(CFG, client=fake, token_path=tmp_path / "t.json",
                          rng=random.Random(0), settle=lambda s: None)
    import threading
    started = []
    orig = threading.Thread.start

    def start(self):
        started.append(self.name)
        orig(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(threading.Thread, "start", start)
        tool.liked()
    assert started == ["spotify-queue"]
    for t in threading.enumerate():
        if t.name == "spotify-queue":
            t.join(timeout=2)
    assert len(fake.named("add_to_queue")) == sp.LIKED_QUEUE_AHEAD


def test_liked_queue_stops_quietly_on_error(tmp_path):
    fake = FakeSpotify(n_saved=130)
    tool = make_tool(tmp_path, fake)
    dev = tool.resolve_device()
    fake.fail = sp.SpotifyException(500, -1, "boom")
    assert tool._queue_rest(dev, ["spotify:track:a", "spotify:track:b"]) == 0


def test_liked_empty_and_premium_is_fatal(tmp_path):
    fake = FakeSpotify(n_saved=0)
    res = registry(make_tool(tmp_path, fake)).call("spotify_liked", {})
    assert res.ok is False and res.speak == sp.LIKED_EMPTY_LINE
    fake = FakeSpotify(n_saved=20, premium=False)
    res = registry(make_tool(tmp_path, fake)).call("spotify_liked", {})
    assert res.speak == sp.PREMIUM_LINE
    assert len(fake.named("transfer_playback")) == 1     # no second strategy tried
    assert start_calls(fake) == []


def test_liked_on_a_named_device(tmp_path):
    fake = FakeSpotify(n_saved=3)
    res = registry(make_tool(tmp_path, fake)).call("spotify_liked", {"device": "phone"})
    assert res.speak == "Your Liked Songs on shuffle, sir — 3 of them, on Hunter's iPhone."


# --------------------------------------------------------- transport
def test_transport_targets_the_active_device_not_the_default(reg, fake):
    assert reg.call("spotify_control", {"action": "pause"}).speak == "Paused, sir."
    assert fake.named("pause_playback") == [((), {"device_id": "dev-phone"})]
    assert fake.named("transfer_playback") == []
    assert reg.call("spotify_control", {"action": "next"}).speak == "Skipped, sir."
    assert reg.call("spotify_control", {"action": "previous"}).speak == "Back one, sir."
    assert reg.call("spotify_control", {"action": "skip"}).speak == "Skipped, sir."
    assert reg.call("spotify_control", {"action": "stop"}).speak == "Paused, sir."
    assert fake.named("next_track")[0][1] == {"device_id": "dev-phone"}


def test_resume_with_nothing_active_wakes_the_default_device(tmp_path):
    devs = [d | {"is_active": False} for d in load("devices")["devices"]]
    fake = FakeSpotify(devices=devs)
    res = registry(make_tool(tmp_path, fake)).call("spotify_control", {"action": "resume"})
    assert res.speak == "Resumed, sir."
    assert fake.named("transfer_playback") == [(("dev-hp",), {"force_play": False})]
    assert start_calls(fake)[0] == {"device_id": "dev-hp", "context_uri": None, "uris": None,
                                    "offset": None, "position_ms": None}


def test_transport_with_no_devices_is_nothing_listening(tmp_path):
    reg = registry(make_tool(tmp_path, FakeSpotify(devices=[])))
    for action in ("pause", "next", "volume", "resume"):
        res = reg.call("spotify_control", {"action": action, "value": "40"})
        assert res.ok is False and res.speak.startswith("Nothing's listening, sir"), action


def test_volume_absolute_and_relative(reg, fake):
    assert reg.call("spotify_control", {"action": "volume", "value": "40"}).speak == "Volume 40, sir."
    assert fake.named("volume")[-1] == ((40,), {"device_id": "dev-phone"})
    assert reg.call("spotify_control", {"action": "volume", "value": "up"}).speak == "Volume 50, sir."
    assert reg.call("spotify_control", {"action": "volume", "value": "down"}).speak == "Volume 30, sir."
    assert reg.call("spotify_control", {"action": "louder"}).speak == "Volume 50, sir."
    assert reg.call("spotify_control", {"action": "mute"}).speak == "Volume 0, sir."
    res = reg.call("spotify_control", {"action": "volume"})
    assert res.ok is False and res.speak == "I'll need a volume level for that, sir."


def test_shuffle_repeat_seek(reg, fake):
    assert reg.call("spotify_control", {"action": "shuffle"}).speak == "Shuffle on, sir."
    assert reg.call("spotify_control", {"action": "shuffle", "value": "off"}).speak == "Shuffle off, sir."
    assert fake.named("shuffle") == [((True,), {"device_id": "dev-phone"}),
                                     ((False,), {"device_id": "dev-phone"})]
    assert reg.call("spotify_control", {"action": "repeat", "value": "one"}).speak == \
        "Repeat on this track, sir."
    assert reg.call("spotify_control", {"action": "repeat"}).speak == "Repeat on, sir."
    assert reg.call("spotify_control", {"action": "repeat", "value": "off"}).speak == "Repeat off, sir."
    assert [a for a, _ in fake.named("repeat")] == [("track",), ("context",), ("off",)]
    assert reg.call("spotify_control", {"action": "seek", "value": "1:30"}).speak == \
        "Jumped to 1:30, sir."
    assert fake.named("seek_track") == [((90_000,), {"device_id": "dev-phone"})]
    res = reg.call("spotify_control", {"action": "seek", "value": "later"})
    assert res.ok is False and res.speak == "I'll need a position for that, sir."


def test_like_saves_once(reg, fake):
    res = reg.call("spotify_control", {"action": "like"})
    assert res.speak == "Saved Blinding Lights to your Liked Songs, sir."
    assert fake.named("current_user_saved_tracks_add") == [((["t-blinding"],), {})]
    res = reg.call("spotify_control", {"action": "like"})
    assert res.speak == "Blinding Lights is already in your Liked Songs, sir."
    assert len(fake.named("current_user_saved_tracks_add")) == 1
    fake.now_playing = None
    res = reg.call("spotify_control", {"action": "like"})
    assert res.ok is False and res.speak == "Nothing's playing, sir."


def test_transfer_moves_playback_with_force_play(reg, fake):
    res = reg.call("spotify_control", {"action": "transfer", "value": "HPCOMPUTER"})
    assert res.speak == "Moved to HPCOMPUTER, sir."
    assert fake.named("transfer_playback") == [(("dev-hp",), {"force_play": True})]
    res = reg.call("spotify_control", {"action": "transfer", "value": "my phone"})
    assert res.speak == "Moved to Hunter's iPhone, sir."
    res = reg.call("spotify_control", {"action": "transfer"})
    assert res.ok is False and res.speak == "I'll need a device name for that, sir."
    res = reg.call("spotify_control", {"action": "transfer", "value": "toaster"})
    assert res.speak.startswith("I can't see toaster on Spotify, sir")


def test_unknown_action(reg):
    res = reg.call("spotify_control", {"action": "levitate"})
    assert res.ok is False and res.speak == sp.BAD_ACTION_LINE


def test_now_playing_playing_paused_nothing(tmp_path):
    fake = FakeSpotify()
    reg = registry(make_tool(tmp_path, fake))
    res = reg.call("spotify_now_playing", {})
    assert res.speak == "Blinding Lights by The Weeknd, sir — on HPCOMPUTER."
    assert res.text == ("Blinding Lights by The Weeknd, sir — on HPCOMPUTER. "
                        "1:01 of 3:20, album After Hours.")
    fake.now_playing = "paused"
    assert reg.call("spotify_now_playing", {}).speak == \
        "Blinding Lights by The Weeknd, sir, paused on HPCOMPUTER."
    fake.now_playing = None
    res = reg.call("spotify_now_playing", {})
    assert res.ok and res.speak == "Nothing's playing, sir."


# ------------------------------------------------------------- queue
def test_queue_adds_next_on_the_active_device(reg, fake):
    res = reg.call("spotify_queue", {"query": "One More Time"})
    assert res.speak == "One More Time by Daft Punk is up next, sir."
    assert fake.named("add_to_queue") == [(("spotify:track:t-otss",), {"device_id": "dev-phone"})]
    res = reg.call("spotify_queue", {"query": "queue up save your tears by the weeknd"})
    assert res.speak == "Save Your Tears by The Weeknd is up next, sir."


def test_queue_no_results_no_device_no_query(tmp_path):
    reg = registry(make_tool(tmp_path, FakeSpotify()))
    res = reg.call("spotify_queue", {"query": "xyzzy"})
    assert res.ok is False and res.speak == "I couldn't find xyzzy on Spotify, sir."
    res = reg.call("spotify_queue", {"query": ""})
    assert res.ok is False and res.speak == "I'll need a song name for that, sir."
    reg = registry(make_tool(tmp_path, FakeSpotify(devices=[])))
    res = reg.call("spotify_queue", {"query": "One More Time"})
    assert res.speak.startswith("Nothing's listening, sir")


# ------------------------------------------------------------- radio
def test_radio_from_current_track_plays_the_this_is_playlist(reg, fake):
    res = reg.call("spotify_radio", {})
    assert res.speak == "This Is The Weeknd, sir — on HPCOMPUTER."
    assert start_calls(fake)[0]["context_uri"] == "spotify:playlist:pl-this-is-weeknd"
    assert [kw["q"] for _, kw in fake.named("search")] == ["This Is The Weeknd",
                                                          "The Weeknd Radio"]


def test_radio_prefers_spotify_owned_over_fan_copies(tmp_path):
    search = load("search")
    items = search["playlists"]["items"]
    search["playlists"]["items"] = [items[3], items[1]]        # fan copy first
    fake = FakeSpotify(search=search)
    res = registry(make_tool(tmp_path, fake)).call("spotify_radio", {})
    assert res.speak == "This Is The Weeknd, sir — on HPCOMPUTER."
    assert start_calls(fake)[0]["context_uri"] == "spotify:playlist:pl-this-is-weeknd"


def test_radio_falls_back_to_top_tracks_and_says_so(tmp_path):
    search = load("search")
    search["playlists"]["items"] = [None]                       # restricted for new apps
    fake = FakeSpotify(search=search)
    res = registry(make_tool(tmp_path, fake)).call("spotify_radio", {})
    assert res.speak == "No radio for The Weeknd, sir, so it's their top tracks on HPCOMPUTER."
    uris = start_calls(fake)[0]["uris"]
    assert len(uris) == 5 and "spotify:track:t-blinding" not in uris
    assert fake.named("artist_top_tracks")[0][0] == ("art-the-weeknd",)


def test_radio_seeded_by_name_and_with_nothing_to_go_on(tmp_path):
    fake = FakeSpotify(now_playing=None)
    reg = registry(make_tool(tmp_path, fake))
    res = reg.call("spotify_radio", {})
    assert res.ok is False and res.speak == sp.NO_SEED_LINE
    res = reg.call("spotify_radio", {"seed": "the weeknd"})
    assert res.speak == "This Is The Weeknd, sir — on HPCOMPUTER."
    res = reg.call("spotify_radio", {"seed": "Daft Punk"})
    assert res.ok is False and res.speak == "I've nothing to go on for Daft Punk, sir."
    res = reg.call("spotify_radio", {"seed": "one more time"})   # a track seeds its artist
    assert res.speak == "I've nothing to go on for Daft Punk, sir."


# ------------------------------------------------------------ errors
def test_expired_token_is_refreshed_and_the_call_retried(tmp_path):
    fake = FakeSpotify(expire_first=True)
    tool = make_tool(tmp_path, fake, auth_factory=FakeAuth)
    tool.cache.save_token_to_cache({"access_token": "old", "refresh_token": "ref-1",
                                    "expires_at": 1})
    res = registry(tool).call("spotify_now_playing", {})
    assert res.ok and res.speak == "Blinding Lights by The Weeknd, sir — on HPCOMPUTER."
    assert tool._auth.refreshed == ["ref-1"]
    assert len(fake.named("current_playback")) == 2
    saved = json.loads(tool.cache.path.read_text())
    assert saved["access_token"] == "new-access" and saved["refresh_token"] == "ref-1"
    assert stat.S_IMODE(tool.cache.path.stat().st_mode) == 0o600


def test_expired_token_without_refresh_token_has_lapsed(tmp_path):
    fake = FakeSpotify(expire_first=True)
    tool = make_tool(tmp_path, fake, auth_factory=FakeAuth)
    res = registry(tool).call("spotify_now_playing", {})
    assert res.ok is False and res.speak == sp.LAPSED_LINE
    assert len(fake.named("current_playback")) == 1


def test_refresh_failure_is_lapsed_not_a_crash(tmp_path):
    class BrokenAuth(FakeAuth):
        def refresh_access_token(self, refresh_token):
            raise sp.SpotifyOauthError("nope")

    fake = FakeSpotify(expire_first=True)
    tool = make_tool(tmp_path, fake, auth_factory=BrokenAuth)
    tool.cache.save_token_to_cache({"access_token": "old", "refresh_token": "ref-1"})
    res = registry(tool).call("spotify_now_playing", {})
    assert res.ok is False and res.speak == sp.LAPSED_LINE


def test_free_account_403_is_the_premium_line(tmp_path):
    fake = FakeSpotify(premium=False)
    reg = registry(make_tool(tmp_path, fake))
    res = reg.call("spotify_play", {"query": "Blinding Lights"})
    assert res.ok is False and res.speak == "I'm afraid that needs Spotify Premium, sir."
    assert res.text == "spotify: premium required"
    assert reg.call("spotify_control", {"action": "pause"}).speak == sp.PREMIUM_LINE


def test_api_failures_translate_to_persona_lines(tmp_path):
    cases = [
        (sp.SpotifyException(500, -1, "Server error"), sp.API_DOWN_LINE),
        (sp.SpotifyException(429, -1, "Too many requests"), sp.RATE_LIMIT_LINE),
        (sp.SpotifyException(403, -1, "Player command failed: Restriction violated",
                             reason="UNKNOWN"), sp.RESTRICTED_LINE),
        (sp.SpotifyException(404, -1, "Player command failed: No active device found",
                             reason="NO_ACTIVE_DEVICE"),
         "Nothing's listening, sir — open Spotify on HPCOMPUTER or your phone."),
        (ConnectionError("refused"), sp.UNREACHABLE_LINE),
        (TimeoutError("slow"), sp.API_DOWN_LINE),
        (sp.SpotifyOauthError("bad"), sp.LAPSED_LINE),
    ]
    for exc, line in cases:
        reg = registry(make_tool(tmp_path, FakeSpotify(fail=exc)))
        res = reg.call("spotify_now_playing", {})
        assert res.ok is False and res.speak == line, exc
        assert res.text and "sir" not in res.text or res.text == line


def test_unexpected_exception_is_contained(tmp_path):
    class Odd(FakeSpotify):
        def devices(self):
            raise RuntimeError("wat")
    res = registry(make_tool(tmp_path, Odd())).call("spotify_play", {"query": "Blinding Lights"})
    assert res.ok is False and res.speak == sp.API_DOWN_LINE
    assert res.text.startswith("spotify error RuntimeError")


def test_registry_never_raises_on_bad_args(reg):
    res = reg.call("spotify_play", {})                # required query missing -> resume
    assert isinstance(res, ToolResult)
    res = reg.call("spotify_control", {})
    assert res.ok is False and res.speak == sp.BAD_ACTION_LINE


# ------------------------------------------------------------- login
def test_login_flow_saves_token_0600_and_confirms(tmp_path, capsys):
    fake = FakeSpotify()
    tool = make_tool(tmp_path, fake, auth_factory=FakeAuth)
    assert not tool.linked() or tool._injected
    line = tool.login(open_browser=True)
    assert line == "Spotify's linked, sir."
    assert tool._auth.opened is True
    tok = json.loads(tool.cache.path.read_text())
    assert tok["refresh_token"] == "first-refresh"
    assert stat.S_IMODE(tool.cache.path.stat().st_mode) == 0o600
    assert fake.named("me") == [((), {})]
    assert "accounts.spotify.com/authorize" in capsys.readouterr().err
    assert sp.login(CFG, token_path=tmp_path / "t2.json", auth_factory=FakeAuth,
                    open_browser=False) == sp.LINKED_LINE


def test_login_refuses_when_unconfigured(tmp_path):
    with pytest.raises(sp.SpotifyError) as exc:
        make_tool(tmp_path, cfg=UNSET, auth_factory=FakeAuth).login()
    assert exc.value.line == sp.SETUP_LINE


def test_login_failure_is_a_spoken_excuse(tmp_path):
    class Broken(FakeAuth):
        def get_auth_response(self, open_browser=None):
            raise sp.SpotifyOauthError("Server listening on localhost has not been accessed")

    with pytest.raises(sp.SpotifyError) as exc:
        make_tool(tmp_path, auth_factory=Broken).login()
    assert exc.value.kind == "auth" and "sir" in exc.value.line
    assert not (tmp_path / "spotify_token.json").exists()


def test_login_cli_status_and_login(tmp_path, monkeypatch, capsys):
    from jarvis import assistant_config as ac
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    monkeypatch.setenv(sp.TOKEN_ENV, str(tmp_path / "tok.json"))
    cfg = ac.AssistantConfig.load(tmp_path / "assistant.json")
    cfg.update({"spotify.client_id": "abc123clientid",
                "spotify.client_secret": "abc123clientsecret"})
    assert sp.login_cli(["--status"]) == 0
    out = capsys.readouterr().out
    assert "configured: True" in out and "linked: False" in out
    announced = []
    monkeypatch.setattr(sp, "_announce", announced.append)
    monkeypatch.setattr(sp.SpotifyTool, "login", lambda self, open_browser=True: sp.LINKED_LINE)
    assert sp.login_cli(["--login"]) == 0
    assert announced == [sp.LINKED_LINE]
    assert "Spotify's linked, sir." in capsys.readouterr().out

    def failing(self, open_browser=True):
        raise sp.SpotifyError(sp.SETUP_LINE, "setup")
    monkeypatch.setattr(sp.SpotifyTool, "login", failing)
    assert sp.login_cli(["--login"]) == 1
    assert sp.SETUP_LINE in capsys.readouterr().err


def test_announce_only_when_the_app_is_running(tmp_path, monkeypatch):
    from jarvis.config import PATHS
    monkeypatch.setattr(PATHS, "LOG_DIR", tmp_path)
    said = []
    from jarvis import speak_queue
    monkeypatch.setattr(speak_queue, "say", said.append)
    sp._announce("hello")
    assert said == []
    (tmp_path / "jarvis.pid").write_text("1")
    sp._announce("hello")
    assert said == ["hello"]


# ----------------------------------------------------------- persona
def _every_spoken_line(tmp_path):
    lines = list(sp.PERSONA_LINES)
    fake = FakeSpotify(n_saved=7)
    reg = registry(make_tool(tmp_path, fake))
    calls = [("spotify_play", {"query": "Blinding Lights"}),
             ("spotify_play", {"query": "The Weeknd"}),
             ("spotify_play", {"query": "After Hours"}),
             ("spotify_play", {"query": "my gym mix"}),
             ("spotify_play", {"query": "xyzzy"}),
             ("spotify_play", {"query": "Blinding Lights", "device": "toaster"}),
             ("spotify_liked", {}), ("spotify_now_playing", {}),
             ("spotify_control", {"action": "pause"}), ("spotify_control", {"action": "resume"}),
             ("spotify_control", {"action": "volume", "value": "40"}),
             ("spotify_control", {"action": "shuffle"}), ("spotify_control", {"action": "repeat"}),
             ("spotify_control", {"action": "seek", "value": "1:30"}),
             ("spotify_control", {"action": "like"}), ("spotify_control", {"action": "like"}),
             ("spotify_control", {"action": "transfer", "value": "phone"}),
             ("spotify_control", {"action": "volume"}),
             ("spotify_queue", {"query": "One More Time"}), ("spotify_radio", {}),
             ("spotify_radio", {"seed": "Daft Punk"})]
    for name, args in calls:
        res = reg.call(name, args)
        assert res.speak, (name, args)
        lines.append(res.speak)
    fake.now_playing = None
    lines.append(reg.call("spotify_now_playing", {}).speak)
    lines.append(reg.call("spotify_radio", {}).speak)
    lines.append(registry(make_tool(tmp_path, FakeSpotify(devices=[]))).call(
        "spotify_play", {"query": "x"}).speak)
    lines.append(registry(make_tool(tmp_path, FakeSpotify(premium=False))).call(
        "spotify_play", {"query": "Blinding Lights"}).speak)
    lines.append(registry(make_tool(tmp_path, cfg=UNSET)).call("spotify_now_playing", {}).speak)
    lines.append(registry(make_tool(tmp_path)).call("spotify_now_playing", {}).speak)
    return lines


def test_every_spoken_line_is_in_persona(tmp_path):
    lines = _every_spoken_line(tmp_path)
    assert len(lines) >= 40
    for line in lines:
        assert count_sentences(line) <= 2, line
        assert "sir" in line, line
        assert line[0].isupper() and line.endswith("."), line
        assert not _MARKDOWN.search(line) and not _EMOJI.search(line), line
        assert "\n" not in line and "http" not in line, line
        assert not re.search(r"\b(I'd be happy|certainly!|great question)\b", line, re.I), line


def test_confirmations_name_what_actually_started(reg, fake):
    fake._search["tracks"]["items"][0]["name"] = "Blinding Lights - Remastered"
    res = reg.call("spotify_play", {"query": "Blinding Lights"})
    assert res.speak == "Blinding Lights - Remastered by The Weeknd, sir — on HPCOMPUTER."


def test_persona_lines_are_prewarmable_constants():
    assert sp.LINKED_LINE == "Spotify's linked, sir."
    assert sp.NO_DEVICE_LINE.format(default="HPCOMPUTER") in sp.PERSONA_LINES
    assert all(isinstance(x, str) and x for x in sp.PERSONA_LINES)


# ------------------------------------------------- market & podcasts
def test_the_fake_matches_the_real_spotipy_signatures():
    """The fake is only worth anything if it is call-compatible with the
    library the tool actually talks to."""
    spotipy = pytest.importorskip("spotipy")
    for name in ("me", "devices", "current_playback", "search",
                 "current_user_playlists", "current_user_saved_tracks",
                 "artist_top_tracks", "transfer_playback", "start_playback",
                 "pause_playback", "next_track", "previous_track", "seek_track",
                 "repeat", "volume", "shuffle", "add_to_queue",
                 "current_user_saved_tracks_add", "current_user_saved_tracks_contains"):
        real = inspect.signature(getattr(spotipy.Spotify, name))
        mine = inspect.signature(getattr(FakeSpotify, name))
        assert [p.name for p in real.parameters.values()][1:] == \
            [p.name for p in mine.parameters.values()][1:], name


def test_market_defaults_and_is_configurable(tmp_path):
    tool = make_tool(tmp_path, FakeSpotify())
    assert tool.market == "from_token" and tool.country == "US"
    cfg = {"spotify": dict(CFG["spotify"], market="gb")}
    tool = make_tool(tmp_path, FakeSpotify(), cfg=cfg)
    assert tool.market == "gb" and tool.country == "GB"


def test_market_is_sent_and_the_country_reaches_top_tracks(tmp_path):
    cfg = {"spotify": dict(CFG["spotify"], market="GB")}
    fake = FakeSpotify()
    reg = registry(make_tool(tmp_path, fake, cfg=cfg))
    reg.call("spotify_play", {"query": "some the weeknd"})
    assert fake.named("search")[0][1]["market"] == "GB"
    assert fake.named("artist_top_tracks")[0][1] == {"country": "GB"}


def test_a_400_on_the_market_retries_without_it_and_remembers(tmp_path):
    class Picky(FakeSpotify):
        def search(self, q, limit=10, offset=0, type="track", market=None):
            if market:
                self.calls.append(("search", (), {"q": q, "market": market}))
                raise sp.SpotifyException(400, -1, "Invalid market code")
            return super().search(q, limit=limit, offset=offset, type=type)

    fake = Picky()
    reg = registry(make_tool(tmp_path, fake))
    res = reg.call("spotify_play", {"query": "Blinding Lights"})
    assert res.speak == "Blinding Lights by The Weeknd, sir — on HPCOMPUTER."
    markets = [kw.get("market") for _, kw in fake.named("search")]
    assert markets[0] == "from_token" and markets[1] is None
    fake.calls.clear()
    reg.call("spotify_play", {"query": "Blinding Lights"})       # market not tried again
    assert [kw.get("market") for _, kw in fake.named("search")] == [None]


def test_a_non_400_on_a_market_call_still_speaks_its_line(tmp_path):
    fake = FakeSpotify(fail=sp.SpotifyException(429, -1, "Too many requests"))
    res = registry(make_tool(tmp_path, fake)).call("spotify_play", {"query": "x"})
    assert res.ok is False and res.speak == sp.RATE_LIMIT_LINE


def test_like_refuses_a_podcast_episode(tmp_path):
    class Podcast(FakeSpotify):
        def current_playback(self, market=None, additional_types=None):
            pb = super().current_playback(market, additional_types)
            pb["item"] = {"id": "ep-1", "name": "Some Episode", "type": "episode",
                          "show": {"name": "A Show"}, "duration_ms": 60_000}
            return pb

    fake = Podcast()
    reg = registry(make_tool(tmp_path, fake))
    res = reg.call("spotify_control", {"action": "like"})
    assert res.ok is False and res.speak == sp.LIKE_EPISODE_LINE
    assert count_sentences(res.speak) <= 2 and "sir" in res.speak
    assert not fake.named("current_user_saved_tracks_add")


def test_now_playing_names_the_show_for_an_episode(tmp_path):
    class Podcast(FakeSpotify):
        def current_playback(self, market=None, additional_types=None):
            pb = super().current_playback(market, additional_types)
            pb["item"] = {"id": "ep-1", "name": "Some Episode", "type": "episode",
                          "show": {"name": "A Show"}, "duration_ms": 60_000}
            return pb

    res = registry(make_tool(tmp_path, Podcast())).call("spotify_now_playing", {})
    assert res.speak == "Some Episode by A Show, sir — on HPCOMPUTER."


# -------------------------------------------------------------- live
@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"), reason="set JARVIS_LIVE=1")
def test_live_read_only_devices_and_now_playing():
    """Read-only against the real account: needs the user's real
    assistant.json (JARVIS_LIVE_SPOTIFY_CONFIG or ~/.config/jarvis) linked
    via --spotify-login.  Never starts, stops or changes playback."""
    from jarvis.assistant_config import AssistantConfig
    path = os.environ.get("JARVIS_LIVE_SPOTIFY_CONFIG") or \
        str(Path.home() / ".config" / "jarvis" / "assistant.json")
    if not Path(path).exists():
        pytest.skip("no real assistant.json")
    cfg = AssistantConfig.load(path)
    tool = sp.SpotifyTool(cfg, token_path=Path(path).parent / sp.TOKEN_FILE)
    if not (tool.configured() and tool.linked()):
        pytest.skip("spotify not configured/linked")
    devs = tool.devices()
    assert isinstance(devs, list)
    res = tool.now_playing()
    assert res.speak and "sir" in res.speak
    if devs:
        assert tool.resolve_device(prefer_active=True).name
