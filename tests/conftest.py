"""Firewall: the suite must never touch the LIVE app's /tmp/vss_voice
(jarvis.log, speak_queue.txt, jarvis.pid, gui_debug.log).

jarvis.logs / jarvis.config / jarvis.jarvis_agent read JARVIS_LOG_DIR at
import, so it is set here — conftest imports before any test module — to
a throwaway directory. The session-scoped autouse fixture below then
re-asserts the redirect on the already-imported modules (belt and braces:
a test that imports jarvis.* before the env is read, or a module that
caches a Path, still lands in the throwaway dir) and fails loudly if any
of them still points at /tmp/vss_voice.

The same firewall covers the three pieces of shared state that have no
throwaway equivalent at all -- the user's display, his speakers and the
single local Ollama server -- by blocking the call rather than redirecting
it. See _firewall_live_log_dir."""
import os
import socket
import tempfile
from pathlib import Path

import pytest

# A pre-set JARVIS_LOG_DIR is honoured ONLY if it is not the live app's
# directory: a shell that exported it to /tmp/vss_voice to read the live
# log (2026-08-29: review agents did) would otherwise make the suite write
# its "throwaway" log straight into the file the running Jarvis is writing.
_LIVE_LOG_DIR = "/tmp/vss_voice"
_preset = os.environ.get("JARVIS_LOG_DIR") or ""
if not _preset or Path(_preset).resolve() == Path(_LIVE_LOG_DIR) or \
        str(Path(_preset).resolve()).startswith(_LIVE_LOG_DIR + "/"):
    _preset = ""
_TEST_LOG_DIR = Path(_preset or tempfile.mkdtemp(prefix="jarvis-tests-"))
os.environ["JARVIS_LOG_DIR"] = str(_TEST_LOG_DIR)

# Personal-assistant firewall (spec 2026-08-26, section 3.2/3.3): the real
# ~/.config/jarvis/assistant.json holds the user's secrets and ~/.cache/jarvis
# the live caches; both are redirected into the throwaway dir BEFORE any
# jarvis import so AssistantConfig.load() / PATHS.CACHE_DIR never see them.
_TEST_ASSISTANT_CONFIG = Path(os.environ.get("JARVIS_ASSISTANT_CONFIG") or
                              (_TEST_LOG_DIR / "assistant.json"))
os.environ["JARVIS_ASSISTANT_CONFIG"] = str(_TEST_ASSISTANT_CONFIG)
_TEST_CACHE_DIR = Path(os.environ.get("JARVIS_CACHE_DIR") or
                       (_TEST_LOG_DIR / "cache"))
os.environ["JARVIS_CACHE_DIR"] = str(_TEST_CACHE_DIR)
# The state stores (timekeeper.db, notes.db, claude_projects.json, the typed
# history, the memory store) hang off PATHS.MEMORY_DIR — the user's real
# ~/.aiws_trainer/jarvis_memory. Redirected here so a test that constructs the
# real modules (tests/test_app_wiring.py) cannot touch it.
_TEST_MEMORY_DIR = Path(os.environ.get("JARVIS_MEMORY_DIR") or
                        (_TEST_LOG_DIR / "memory"))
os.environ["JARVIS_MEMORY_DIR"] = str(_TEST_MEMORY_DIR)
os.environ.setdefault("JARVIS_LEGACY_DIR", str(_TEST_LOG_DIR / "legacy"))
# The Spotify OAuth token sits beside the assistant config; keep the suite off
# the real one (jarvis/tools/spotify.py reads JARVIS_SPOTIFY_TOKEN first).
_TEST_SPOTIFY_TOKEN = Path(os.environ.get("JARVIS_SPOTIFY_TOKEN") or
                           (_TEST_LOG_DIR / "spotify_token.json"))
os.environ["JARVIS_SPOTIFY_TOKEN"] = str(_TEST_SPOTIFY_TOKEN)
# The intent classifier's feedback log: tests that resolve "Was that for
# me?" call log_feedback, which used to write the user's real
# ~/.aiws_trainer/intent_log.json (found 2026-08-30). Forced, not
# setdefault: a preset pointing at the live file would defeat the firewall.
os.environ["JARVIS_INTENT_LOG"] = str(_TEST_LOG_DIR / "intent_log.json")
# The enrolled VOICEPRINT (jarvis/speaker.py reads PATHS.VOICEPRINT at import
# and save() writes that global). On 2026-09-02 a test that built a real
# SpeakerVerifier and called enroll_from_audio overwrote his 6-sample pool with
# two copies of its fixture's constant vector; his own enrolment clips then
# scored 0.055-0.125 against a 0.30 threshold, i.e. the transcript gate --
# which fails SHUT -- would have dropped every command he spoke after the next
# restart. Forced, not setdefault: a shell pointing at the live file must not
# defeat this.
os.environ["JARVIS_VOICEPRINT"] = str(_TEST_LOG_DIR / "voiceprint.npz")
# The enrolled FACE gallery (jarvis/facegallery.py, PATHS.FACE_GALLERY). Set
# before the same class of accident can happen a second time: the voiceprint
# was destroyed by a test that built the real object and saved, and a face
# embedding is the same kind of irreplaceable measurement of one person.
# Forced, not setdefault, for the same reason as the line above.
os.environ["JARVIS_FACE_GALLERY"] = str(_TEST_LOG_DIR / "face_gallery")
# The downloaded YuNet/SFace weights (jarvis/facemodels.py). Forced at a
# throwaway directory so the suite is identical on a box that has them and a
# box that does not: a test that quietly passed only because 38 MB of SFace
# happened to be on this machine would be worse than no test. Anything
# wanting the real weights is a script he runs, not a test.
os.environ["JARVIS_FACE_MODEL_DIR"] = str(_TEST_LOG_DIR / "face_models")
# The docs embedding index (jarvis/tools/docs.py: env > config > default):
# without this a test building the real App indexes into the user's
# ~/.aiws_trainer/docs_index.
os.environ["JARVIS_DOCS_INDEX_DIR"] = str(_TEST_LOG_DIR / "docs_index")
# The room controls (jarvis/room.py, jarvis/mixer.py) shell out to xrandr,
# gsettings and pactl, which act on the USER'S LIVE SESSION -- there is no
# per-process display or sound server to redirect. The suite builds the real
# app (tests/test_app_wiring.py) and publishes real events, so a
# SpeakingState in a test would duck whatever he is actually listening to
# and a scene test would dim the screen he is reading. Forced, not
# setdefault: this firewall is not one a shell may switch off.
os.environ["JARVIS_ROOM_CONTROL"] = "0"
# The desk-presence probe (jarvis/deskpresence.py) reads GNOME's idle monitor
# over the session bus -- the one piece of state that cannot be redirected to
# a throwaway path, because it is the DEVELOPER'S live desktop. A test that
# builds the real App would otherwise learn that the chair has been empty all
# afternoon and start holding proactive speech in unrelated tests (found
# 2026-08-30: test_app_wiring's timer went into the quiet digest). Forced, not
# setdefault: a shell that exported it on must not defeat the firewall.
# tests/test_deskpresence.py clears it for its own cases.
os.environ["JARVIS_DESK_PRESENCE"] = "0"


# Ollama's port. The live server is a single shared process (see the
# firewall fixture below for why that matters); JARVIS_TEST_ALLOW_OLLAMA=1
# lets a deliberate live smoke test through.
_OLLAMA_PORT = 11434
_ollama_blocked: list = []

# The SMTP SUBMISSION ports. jarvis/outbox.py sends mail with an attachment,
# and that is the one thing this app does that reaches a stranger's inbox
# and cannot be recalled -- so no test may open one of these, ever, and
# there is no environment variable to let one through. The suite drives
# mail.send_message with a fake transport (smtp=), exactly as the read side
# is driven with a fake IMAP class; a socket on 465 means a test lost its
# fake, and the right outcome is a loud failure rather than a real email
# from his account. Same argument as the room controls and the speakers
# above: some shared state has no throwaway copy, and his correspondents
# are the least redirectable of all.
_SMTP_PORTS = (25, 465, 587, 2525)
_smtp_blocked: list = []


def _blocked_player(argv) -> bool:
    """Stands in for earcons._spawn: the tone is "played" and the caller
    sees the same True, but nothing reaches the sound server."""
    _blocked_player.calls.append(list(argv))
    return True


_blocked_player.calls = []


@pytest.fixture(scope="session", autouse=True)
def _firewall_live_log_dir():
    """Every jarvis path that can write under the log dir points at the
    throwaway directory for the whole session."""
    from jarvis import config, logs
    live = Path("/tmp/vss_voice")
    assert logs.LOG_DIR != live, "jarvis.logs still targets the live app"
    assert config.PATHS.LOG_DIR != live, "PATHS.LOG_DIR still live"
    assert config.PATHS.SPEAK_QUEUE.parent != live, "speak queue still live"
    assert logs.LOG_FILE.parent == _TEST_LOG_DIR
    from jarvis.commander import IntentClassifier
    assert IntentClassifier.INTENT_LOG.parent == _TEST_LOG_DIR, \
        "IntentClassifier.INTENT_LOG still points at the user's real log"
    from jarvis import mixer as _mixer
    assert _mixer.blocked(), \
        "the room controls could reach the user's live display and audio"
    from jarvis import speaker as _speaker
    real_voiceprint = Path.home() / ".aiws_trainer" / "voiceprint.npz"
    assert _speaker.VOICEPRINT_FILE != real_voiceprint, \
        "speaker.VOICEPRINT_FILE still targets the user's enrolled voice"
    assert config.PATHS.VOICEPRINT != real_voiceprint, "PATHS.VOICEPRINT still live"
    try:
        from jarvis import jarvis_agent
        assert jarvis_agent.LOG_DIR != live, "jarvis_agent LOG_DIR still live"
    except ImportError:
        pass
    try:
        from jarvis import speak_queue
        assert Path(speak_queue.SPEAK_QUEUE).parent != live
    except ImportError:
        pass
    # Assistant paths (section 3.2): nothing may point at the live socket /
    # task dir / MCP config under /tmp/vss_voice, nor at the user's real
    # config file or cache.
    real_cfg_dir = Path.home() / ".config" / "jarvis"
    real_cache = Path.home() / ".cache" / "jarvis"
    P = config.PATHS
    assert P.ASSISTANT_CONFIG == _TEST_ASSISTANT_CONFIG, \
        "PATHS.ASSISTANT_CONFIG ignores JARVIS_ASSISTANT_CONFIG"
    assert real_cfg_dir not in P.ASSISTANT_CONFIG.parents and \
        P.ASSISTANT_CONFIG.parent != real_cfg_dir, "assistant config still live"
    assert P.CACHE_DIR != real_cache and real_cache not in P.CACHE_DIR.parents, \
        "PATHS.CACHE_DIR still live"
    for name in ("APPROVALS_SOCK", "COMMAND_SOCK", "CLAUDE_TASK_DIR", "MCP_CONFIG"):
        path = getattr(P, name)
        assert path != live and live not in path.parents, f"PATHS.{name} still live"
        assert path.parent == _TEST_LOG_DIR or _TEST_LOG_DIR in path.parents, \
            f"PATHS.{name} not under the test log dir"
    real_memory = Path.home() / ".aiws_trainer" / "jarvis_memory"
    assert P.MEMORY_DIR == _TEST_MEMORY_DIR, "PATHS.MEMORY_DIR still live"
    assert P.LEGACY_AGENT_DIR != Path.home() / ".aiws_trainer" / "jarvis_data", \
        "PATHS.LEGACY_AGENT_DIR still live"
    for name in ("TIMEKEEPER_DB", "NOTES_DB", "CLAUDE_PROJECTS"):
        path = getattr(P, name)
        assert real_memory not in path.parents, f"PATHS.{name} still live"
    try:
        from jarvis.tools import spotify
        token = spotify.token_path()
        assert real_cfg_dir not in token.parents, "spotify token still live"
    except ImportError:
        pass
    try:
        from jarvis import assistant_config
        default = getattr(assistant_config, "DEFAULT_PATH", None)
        if default is not None:
            assert Path(default).parent != real_cfg_dir or \
                os.environ["JARVIS_ASSISTANT_CONFIG"] == str(_TEST_ASSISTANT_CONFIG)
    except ImportError:
        pass
    # The room's AUDIO OUT. jarvis/earcons.py synthesizes a tone and hands it
    # to a real `paplay`, which reaches the USER'S SPEAKERS -- there is no
    # per-process sound server to redirect, exactly as with the display above.
    # On 2026-08-31 a full-suite run played two "heard-you" beeps and two
    # "arrival" trills into the room while he sat at the desk, twice: four
    # tests (test_app_wiring) call earcons.play() without injecting a runner.
    # Tests that pass their own `run=` recorder are unaffected.
    from jarvis import earcons
    real_spawn = earcons._spawn
    earcons._spawn = _blocked_player
    # The LOCAL MODEL SERVER. Ollama is one shared process on this box and,
    # exactly like the display and the sound server above, there is no
    # per-process instance to point a test at. It also runs under
    # OLLAMA_MAX_LOADED_MODELS=1 -- the guard added after the 2026-08-28
    # unified-memory lock-up -- so only ONE model may be resident: a test
    # that reaches for nomic-embed-text does not ADD a model, it EVICTS the
    # chat model the running Jarvis pinned with keep_alive -1, and the next
    # real turn pays a ~7 s reload (brain.RESIDENCY_INTERVAL_S is 300 s, so
    # the loop may not notice for five minutes).
    #
    # Found 2026-08-31, while the user was timing his turns: a full-suite
    # run fired 24 of these from 11 tests (test_memory, test_webapp,
    # test_app_wiring), because JarvisMemory defaults to semantic=True with
    # embed=None -- i.e. the REAL embedder on localhost:11434. He measured
    # `chat reply (10.01s wall, 7.07s ollama overhead)` on "what's the
    # weather" against a suite running in another terminal.
    #
    # Refused rather than asserted: "Ollama is down" is a state every one of
    # these call sites already handles (they fall back to the substring
    # store), so the suite stays green AND hermetic, instead of green and
    # quietly coupled to whichever model happens to be loaded.
    real_connect = socket.socket.connect
    allow_ollama = os.environ.get("JARVIS_TEST_ALLOW_OLLAMA") == "1"

    def _refuse(sock, address):
        port = (address[1] if isinstance(address, tuple) and len(address) > 1
                else None)
        if port in _SMTP_PORTS:
            _smtp_blocked.append(address)
            raise ConnectionRefusedError(
                f"the suite must not open an SMTP connection ({address!r}): "
                "a real email cannot be recalled. Pass a fake transport -- "
                "mail.send_message(..., smtp=FakeSMTP) -- the way the read "
                "side takes imap=. There is no override for this one.")
        if port == _OLLAMA_PORT and not allow_ollama:
            _ollama_blocked.append(address)
            raise ConnectionRefusedError(
                f"the suite must not reach the live Ollama at {address!r}: "
                "pass semantic=False or an embed= stub (see "
                "tests/conftest.py). Set JARVIS_TEST_ALLOW_OLLAMA=1 for a "
                "deliberate live test.")
        return real_connect(sock, address)

    # Installed unconditionally now: the Ollama leg still honours its
    # environment escape inside _refuse, but the SMTP leg has none.
    socket.socket.connect = _refuse
    try:
        yield
    finally:
        earcons._spawn = real_spawn
        socket.socket.connect = real_connect
