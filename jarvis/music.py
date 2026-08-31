"""The two-line Spotify seam: pause it, put it back.

Two features want the music down and then up again -- a focus block
(jarvis/focus.py) and the start of a class (jarvis/classflow.py) -- and
neither wants to learn what a SpotifyError is. ``focus._music`` had the
only copy, welded to focus-session state (``self.state`` music / music_did
/ playlist, MUSIC_MODES, the per-session rebind), so the stager could not
call it without dragging a session in.

Everything here is synchronous and never raises: the caller decides which
thread it runs on (the bus delivers on the Tk thread, so both callers hand
it to a worker) and reads the ``kind`` back to decide whether a failure is
worth retrying. ``setup`` / ``auth`` / ``premium`` are not: no linked
account will link itself between two blocks of a study session.
"""
from __future__ import annotations

from typing import Optional

from jarvis.logs import get_logger

log = get_logger("music")

# Failure kinds that will not fix themselves; the caller should stop asking.
PERMANENT_KINDS = ("setup", "auth", "premium")


def call(spotify, method: str, arg: str = "") -> tuple:
    """Run one Spotify call: ``control`` with an action ("pause",
    "resume") or ``play`` with a playlist name. -> (ok, kind).

    ``kind`` is the SpotifyError kind on failure ("device", "auth", …),
    "none" when there was no Spotify handle at all, "" on success."""
    if spotify is None:
        return False, "none"
    try:
        if method == "play":
            spotify.play(arg, "playlist")
        else:
            spotify.control(arg)
        log.info("music: spotify %s %r ok", method, arg)
        return True, ""
    except Exception as exc:               # noqa: BLE001 - SpotifyError or worse
        kind = str(getattr(exc, "kind", "") or "")
        log.info("music: spotify %s %r skipped: %s", method, arg,
                 getattr(exc, "text", exc))
        return False, kind


def pause(spotify) -> tuple:
    """Stop the music. -> (ok, kind)."""
    return call(spotify, "control", "pause")


def resume(spotify) -> tuple:
    """Put it back. -> (ok, kind)."""
    return call(spotify, "control", "resume")


def playing(spotify) -> Optional[bool]:
    """True / False / None ("nobody could tell me").

    A caller that will owe a resume must know whether anything was playing
    in the first place: pausing an idle player and then "restoring" it at
    the end of the hour starts music he never had on."""
    if spotify is None:
        return None
    state = getattr(spotify, "playback_state", None)
    if not callable(state):
        return None
    try:
        return str(state()) == "playing"
    except Exception as exc:               # noqa: BLE001 - SpotifyError or worse
        log.info("music: playback state unavailable: %s", getattr(exc, "text", exc))
        return None


def permanent(kind: Optional[str]) -> bool:
    """True when retrying this failure is pointless."""
    return str(kind or "") in PERMANENT_KINDS
