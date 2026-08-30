#!/usr/bin/env python
"""One-time Spotify authorisation.

Client credentials alone cannot control playback -- the playback scopes are
per-user, so Spotify needs you to approve them once in a browser. This writes
the resulting token to the same path the tool reads (`spotify_token.json`
beside assistant.json), after which "Jarvis, drop my needle" works offline
from the cached refresh token.

Before running, add this EXACT redirect URI to your app at
https://developer.spotify.com/dashboard -> your app -> Settings:

    http://127.0.0.1:8888/callback

Usage:  ~/vss_env/bin/python scripts/link_spotify.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.assistant_config import AssistantConfig                 # noqa: E402
from jarvis.tools.spotify import (REDIRECT_URI, SCOPES, TokenCache,  # noqa: E402
                                  token_path)


def main() -> int:
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError:
        print("spotipy is not installed in this interpreter.")
        return 1

    cfg = AssistantConfig.load()
    cid = cfg.get("spotify.client_id")
    secret = cfg.get("spotify.client_secret")
    if not cid or not secret:
        print("No spotify.client_id / client_secret in", cfg.path)
        return 1

    out = token_path(cfg)
    cache = TokenCache(out)
    if cache.linked():
        print("Already linked; token at", out)
        print("Delete that file and re-run to link a different account.")
        return 0

    print("Redirect URI in use:", REDIRECT_URI)
    print("It must be listed in your Spotify app settings, character for")
    print("character, or Spotify returns INVALID_CLIENT.\n")

    auth = SpotifyOAuth(
        client_id=cid, client_secret=secret, redirect_uri=REDIRECT_URI,
        scope=SCOPES, cache_handler=cache, open_browser=False,
        show_dialog=True)

    print("1. Open this URL and approve:\n")
    print("   " + auth.get_authorize_url() + "\n")
    print("2. The browser will fail to load a page at 127.0.0.1:8888 -- that")
    print("   is expected, nothing is listening. Copy the FULL address bar")
    print("   URL (it carries ?code=...) and paste it here.\n")

    try:
        redirected = input("Pasted URL: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 1
    if not redirected:
        print("Nothing pasted.")
        return 1

    code = auth.parse_response_code(redirected)
    if not code or code == redirected:
        print("No ?code= found in that URL.")
        return 1

    auth.get_access_token(code, as_dict=False, check_cache=False)
    if not cache.linked():
        print("Spotify returned a token but it did not reach", out)
        return 1

    me = spotipy.Spotify(auth_manager=auth).me()
    print("\nLinked as %s. Token saved to %s"
          % (me.get("display_name") or me.get("id"), out))
    print('Try: "Jarvis, drop my needle"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
