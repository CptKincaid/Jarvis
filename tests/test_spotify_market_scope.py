"""market=from_token needs a scope the token may not carry.

Spotify resolves `market=from_token` from the user, so it requires
`user-read-private`. Without it every search returns 403 "Insufficient client
scope" -- and the translated persona line was "Spotify isn't answering, sir",
which points at the network for what is really a permissions problem. Live on
2026-08-29 that made "Jarvis, drop my needle" fail after the account had
linked successfully.

Widening SCOPES is NOT the fix: spotipy rejects a cached token whose scopes do
not cover the request, so adding the scope silently unlinks a working account
until the user re-authorises. Dropping the market costs nothing, so a
scope-denied 403 joins the 400 path.
"""
import pytest

from jarvis.tools.spotify import SpotifyError, SpotifyTool


class _Tool(SpotifyTool):
    """Only _api is replaced; _api_market -- the code under test -- is real."""

    def __init__(self, fail_with):
        self.calls = []
        self._market_bad = False
        self._fail_with = fail_with

    def _api(self, method, *args, **kwargs):
        self.calls.append(kwargs.get("market") or kwargs.get("country"))
        if self._fail_with is not None and "market" in kwargs:
            raise self._fail_with
        return {"ok": True}


class _Translating(SpotifyTool):
    default_device = "HPCOMPUTER"        # shadows the cfg-backed property


def _scope_403():
    """A REAL spotipy exception through the REAL _translate.

    The first version of this test hand-wrote the error text with "scope" in
    it and passed while the production path was dead: spotipy's message
    begins with the request URL, so on a real search "scope" sits at index
    157 of 162 and _translate's 80-char cut removed it before the check.
    """
    from spotipy.exceptions import SpotifyException
    url = ("https://api.spotify.com/v1/search?q=Jingle+Bells+Bombay+Dub+Orchestra"
           "+Remix+Joe+Williams&limit=5&offset=0&type=track&market=from_token")
    raw = SpotifyException(403, -1, "%s:\n %s" % (url, "Insufficient client scope"),
                           reason=None)
    assert raw.msg.find("scope") > 80, "fixture no longer reproduces the truncation"
    t = object.__new__(_Translating)
    return t._translate(raw)


def test_a_real_scope_403_is_classified_before_truncation():
    err = _scope_403()
    assert err.kind == "scope" and err.status == 403
    assert not err.fatal, "a scope error is routable; it must not short-circuit"


def test_scope_denied_403_retries_without_the_market():
    t = _Tool(_scope_403())
    assert t._api_market("search", q="x", market="from_token") == {"ok": True}
    assert t.calls == ["from_token", None], t.calls


def test_the_market_stays_off_for_the_rest_of_the_session():
    """One 403 per call would double every search for no benefit."""
    t = _Tool(_scope_403())
    t._api_market("search", q="x", market="from_token")
    t.calls.clear()
    t._api_market("search", q="y", market="from_token")
    assert t.calls == [None], "re-sent a market already known to be refused"


def test_a_403_that_is_not_about_scope_still_raises():
    """403 also means restricted/premium-required; those must not be
    mistaken for a market problem and silently retried into a wrong error."""
    other = SpotifyError("Restricted, sir.", "restricted",
                         "http status: 403, Player command failed", 403)
    t = _Tool(other)
    with pytest.raises(SpotifyError) as ei:
        t._api_market("search", q="x", market="from_token")
    assert ei.value.status == 403 and ei.value.kind == "restricted"


def test_a_400_still_works_as_before():
    t = _Tool(SpotifyError("bad market", "api", "http status: 400", 400))
    assert t._api_market("search", q="x", market="ZZ") == {"ok": True}
    assert t.calls == ["ZZ", None]


def test_a_call_carrying_no_market_is_untouched():
    """Nothing to drop: retrying would just repeat the same failure."""
    t = _Tool(None)
    boom = _scope_403()

    def _api(method, *a, **kw):
        raise boom
    t._api = _api
    with pytest.raises(SpotifyError):
        t._api_market("devices")
