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
import sys
import socket
import subprocess as _real_subprocess
import sys as _sys
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
# The zone transition log (jarvis/zones.py, PATHS.STATE_DIR): the real
# ~/.local/state/jarvis/zones.jsonl is a record of which room he was in and
# when, so the suite must never append to it -- nor read it. Forced, not
# setdefault, for the same reason as the lines above.
os.environ["JARVIS_STATE_DIR"] = str(_TEST_LOG_DIR / "state")
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


class SmtpFirewallRefused(BaseException):
    """The SMTP leg's refusal. A BaseException on purpose (F24, 09-03):
    as a ConnectionRefusedError it was caught by mail.send_message's
    ``except Exception``, re-raised as MailSendFailed, and spoken by the
    commander as the ordinary "I couldn't send that, sir." -- so a test
    that lost its fake passed unless it happened to assert on the sent
    list, and the "loud failure" the docstring above promises never
    happened. Nothing in jarvis/ catches BaseException, so this one comes
    out of the test that caused it. pytest_sessionfinish below is the
    second belt: any refusal nobody owned up to fails the run."""


def _blocked_player(argv) -> bool:
    """Stands in for earcons._spawn: the tone is "played" and the caller
    sees the same True, but nothing reaches the sound server."""
    _blocked_player.calls.append(list(argv))
    return True


_blocked_player.calls = []


@pytest.fixture(autouse=True)
def _reset_brain_calibration():
    """brain._CALIBRATION is process state fed by every fake Ollama reply's
    prompt_eval_count; a test must not inherit the factor another test's
    replies taught. Only touches jarvis.brain if a test already imported
    it (never imports it: an import is what tests/test_config_readonly.py
    keeps honest)."""
    brain = sys.modules.get("jarvis.brain")
    if brain is not None and "_CALIBRATION" in vars(brain):
        brain._CALIBRATION.update(factor=brain.CALIBRATION_INITIAL, samples=0)
    yield
    if brain is not None and "_CALIBRATION" in vars(brain):
        brain._CALIBRATION.update(factor=brain.CALIBRATION_INITIAL, samples=0)


# ---------------------------------------------------------------------------
# THE DESKTOP FIREWALL (2026-09-04, 17:21)
# ---------------------------------------------------------------------------
# A probe ran a corpus of "remember ..." sentences through Commander.handle
# as a plain script. One of them, "jarvis remember heather's face", is a face
# ENROLMENT, and that rung hands its command line to xclip on :1 -- so the
# probe overwrote his clipboard while he was working at the desk. Another,
# "jarvis commit this to memory: ...", matched the bare-substring
# QUICK_COMMANDS trigger "commit" and ran `git add -A` in ~/vss_env through
# a shell. Neither was a test; but nothing in this file would have stopped a
# test doing the same, because the seams below were never blocked here --
# only the sound server, the room controls and Ollama were.
#
# So: inside every jarvis module a Commander.handle call can reach (and the
# few beside them that fork onto the same shared state), the name
# ``subprocess`` is replaced for the whole session by a proxy that REFUSES
# ``run`` and ``Popen``, records the attempt, and raises an OSError naming
# the module and the program. OSError on purpose: every one of those seams
# already handles a missing binary that way ("the clipboard wouldn't take
# it", "Could not read clipboard", an error Status, a debug line), so the
# suite stays green AND hermetic -- the same reasoning as the Ollama leg
# above. Everything else on the proxy (PIPE, DEVNULL, CompletedProcess,
# TimeoutExpired, SubprocessError) is the real module's.
#
# Three ways a test still gets a process, all of them deliberate:
#   1. the seam the module already has (run= / popen= / _run), which is how
#      every test of these modules is written;
#   2. a monkeypatch of the GLOBAL subprocess.run / subprocess.Popen: the
#      proxy forwards to whatever is installed there when it is not the
#      original (tests/test_oracle.py, test_remote_files_and_shell.py,
#      test_tts_voice_io.py all do this);
#   3. ``@pytest.mark.real_subprocess("jarvis.context", ...)`` for a test
#      that must fork the genuine program (git in a tmp_path repo), which
#      puts the module back for that one test and nothing else.
#
# Import-time defaults are covered too: ``reader._xclip(selection,
# run=subprocess.run)`` bound the REAL run at import, so the proxy is also
# written into every default argument in the module that held it.
_REAL_RUN = _real_subprocess.run
_REAL_POPEN = _real_subprocess.Popen


class DesktopFirewallRefused(OSError):
    """A jarvis module asked for a subprocess under pytest. OSError so the
    module's own missing-binary handling absorbs it (see above); the message
    names the module, the program and the three ways through."""


# The session ledger: (test nodeid, module, kind, program) for every refusal.
_desktop_ledger: list = []
# The running test's own sink, handed out by the `desktop_attempts` fixture.
_desktop_current = {"nodeid": "", "sink": None}

# What is wrapped, and why. Read this list before adding a subprocess call
# to any of these modules: under pytest it will be refused.
_DESKTOP_MODULES = {
    "jarvis.commander": "xclip write (transform case), xclip read, xdotool "
                        "type/key, xdg-open, app launch, QUICK_COMMANDS shell",
    "jarvis.enrolentry": "xclip write: the 17:21 clipboard overwrite",
    "jarvis.workflows": "shell steps, xdotool key, notify-send",
    "jarvis.desktop": "every xdotool/xclip primitive",
    "jarvis.reader": "xclip -o: the contents of his clipboard/selection",
    "jarvis.context": "xdotool window titles, git, ps, nvidia-smi, find",
    "jarvis.brain": "the claude CLI (a paid, acting agent) and the "
                    "autonomous RUN shell",
    "jarvis.app": "xdotool windowactivate, notify-send",
    "jarvis.claude_session": "tmux send-keys into his live Claude sessions, "
                             "gnome-terminal, xdotool, git init",
    "jarvis.tts": "the audio player (his speakers) and the F5/Breeze sidecars",
    "jarvis.classflow": "xdg-open of a document on his screen",
    "jarvis.jarvis_agent": "the V1 agent: xclip write, xdotool, shell",
    "jarvis.ask": "the phone client's recorder: it opens a MICROPHONE",
    "jarvis.roomtone": "paplay: his speakers",
    "jarvis.voice_check": "pactl sink probes",
    "jarvis.winddown": "xdotool / gsettings on his session",
    "jarvis.soundbar": "bluetoothctl on his soundbar",
    "jarvis.tools.screen": "`import -window root`: a SCREENSHOT of his desktop",
    "jarvis.tools.timekeeper": "the ringer (his speakers), notify-send",
    "jarvis.tools.health": "nvidia-smi (hermeticity only)",
    "jarvis.tools.remote": "ssh into HPCOMPUTER",
    "jarvis.tools.oracle": "ssh into the Oracle box",
    "jarvis.channels.notify": "notify-send banners on his desktop",
    "jarvis.ui.board": "xdotool on the board window",
    "jarvis.ui.main_window": "xdotool/xprop/xrdb/nvidia-smi from the window",
}
# Left alone, and why: jarvis.room / jarvis.mixer (JARVIS_ROOM_CONTROL=0
# above turns their seam into a no-op before any subprocess); jarvis.presence
# / jarvis.deskpresence (JARVIS_DESK_PRESENCE=0, their own `_run` seam, and
# tests/test_presence.py pins that seam's default runner by identity);
# jarvis.earcons (`_spawn`, above); jarvis.autostart (`_run` seam, and
# tests/test_autostart.py exercises it against the real systemctl on
# purpose); jarvis.config (`arecord -l`, a read-only device listing);
# jarvis.tools.docs (pdftotext over a file the test itself supplies);
# jarvis.ui.avatar_bake (a build script, never on the reply path).
# jarvis.cast.focused_window_title binds subprocess INSIDE the function, so
# it is replaced as a function (below) rather than through the module name.
_wrapped: list = []          # (module, real subprocess module) to restore
_rebound: list = []          # (function, old __defaults__, old __kwdefaults__)


def _program_of(argv) -> str:
    if isinstance(argv, (list, tuple)):
        return str(argv[0]) if argv else ""
    if isinstance(argv, (str, bytes)):
        text = argv.decode("utf-8", "replace") if isinstance(argv, bytes) else argv
        return text.split()[0] if text.split() else ""
    return repr(argv)


# pytest's own scratch root (tmp_path_factory.getbasetemp()), set by the
# session fixture. Two things under it are not his desktop and are let
# through: a PROGRAM that lives there (tests/test_web_route.py writes a
# stand-in `claude` script into tmp_path and runs it), and `git` run WITH
# `cwd` there (tests/test_context.py and test_standup.py probe repos they
# built in tmp_path). A shell string is never a list, so `cd ~/vss_env &&
# git add -A` cannot use either door.
_TMP_ROOT: list = []


def _under_tmp(path) -> bool:
    if not _TMP_ROOT or path is None:
        return False
    try:
        resolved = Path(str(path)).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    root = _TMP_ROOT[0]
    return resolved == root or root in resolved.parents


def _sandboxed(argv, kwargs) -> bool:
    if not isinstance(argv, (list, tuple)) or not argv:
        return False
    program = str(argv[0])
    if program == "git":
        return _under_tmp(kwargs.get("cwd"))
    if program == "find":                     # context._fetch_recent_files
        return len(argv) > 1 and _under_tmp(argv[1])
    return "/" in program and _under_tmp(program)


def _refuse(owner: str, kind: str, argv):
    program = _program_of(argv)
    _desktop_ledger.append((_desktop_current["nodeid"], owner, kind, program))
    sink = _desktop_current["sink"]
    if sink is not None:
        sink.append((owner, kind, program, argv))
    raise DesktopFirewallRefused(
        f"{owner} asked subprocess.{kind} for {program!r} under pytest: the "
        "suite must not fork onto the user's desktop, shell, tmux, ssh, mic "
        "or speakers (tests/conftest.py, the desktop firewall). Pass the "
        "module's own seam (run= / popen= / _run), monkeypatch subprocess.run "
        "or subprocess.Popen on the real module, or mark the test "
        f"@pytest.mark.real_subprocess({owner!r}) to say you mean it.")


class _SubprocessProxy:
    """Stands in for the ``subprocess`` module inside ONE jarvis module."""

    def __init__(self, owner: str):
        self._owner = owner
        real = _real_subprocess

        class Popen(_REAL_POPEN):
            """A subclass so ``isinstance``, ``except`` and annotations keep
            working; ``__new__`` never builds one."""
            def __new__(cls, *args, **kwargs):
                argv = args[0] if args else kwargs.get("args")
                if real.Popen is not _REAL_POPEN:        # a test's global fake
                    return real.Popen(*args, **kwargs)
                if _sandboxed(argv, kwargs):             # a script in tmp_path
                    return _REAL_POPEN(*args, **kwargs)
                _refuse(owner, "Popen", argv)

        Popen.__qualname__ = f"{owner}.subprocess.Popen"
        self.Popen = Popen

    def run(self, *args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        if _real_subprocess.run is not _REAL_RUN:          # a test's global fake
            return _real_subprocess.run(*args, **kwargs)
        if _sandboxed(argv, kwargs):                       # git in a tmp_path repo
            return _REAL_RUN(*args, **kwargs)
        _refuse(self._owner, "run", argv)

    # The convenience wrappers build a REAL Popen inside the real module, so
    # forwarding them would slip past the proxy: refused outright.
    def call(self, *args, **kwargs):
        _refuse(self._owner, "call", args[0] if args else kwargs.get("args"))

    def check_call(self, *args, **kwargs):
        _refuse(self._owner, "check_call", args[0] if args else kwargs.get("args"))

    def check_output(self, *args, **kwargs):
        _refuse(self._owner, "check_output", args[0] if args else kwargs.get("args"))

    def getoutput(self, cmd, *args, **kwargs):
        _refuse(self._owner, "getoutput", cmd)

    def getstatusoutput(self, cmd, *args, **kwargs):
        _refuse(self._owner, "getstatusoutput", cmd)

    def __getattr__(self, name):
        return getattr(_real_subprocess, name)

    def __repr__(self):
        return f"<desktop-firewalled subprocess for {self._owner}>"


def _rebind_defaults(fn, proxy) -> None:
    """``def f(run=subprocess.run)`` captured the REAL runner at import; the
    proxy is written into that default so the seam is still the seam."""
    swap = {id(_REAL_RUN): proxy.run, id(_REAL_POPEN): proxy.Popen}
    defaults = getattr(fn, "__defaults__", None)
    kwdefaults = getattr(fn, "__kwdefaults__", None)
    hit = (defaults and any(id(v) in swap for v in defaults)) or \
          (kwdefaults and any(id(v) in swap for v in kwdefaults.values()))
    if not hit:
        return
    _rebound.append((fn, defaults, dict(kwdefaults) if kwdefaults else None))
    if defaults:
        fn.__defaults__ = tuple(swap.get(id(v), v) for v in defaults)
    if kwdefaults:
        fn.__kwdefaults__ = {k: swap.get(id(v), v) for k, v in kwdefaults.items()}


def _wrap_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # noqa: BLE001 - a module this box cannot import is not on the path
        return False
    mod = _sys.modules[name]
    if getattr(mod, "subprocess", None) is not _real_subprocess:
        return False
    proxy = _SubprocessProxy(name)
    _wrapped.append((mod, _real_subprocess))
    mod.subprocess = proxy
    import inspect as _inspect
    for obj in list(vars(mod).values()):
        if _inspect.isfunction(obj) and obj.__module__ == name:
            _rebind_defaults(obj, proxy)
        elif _inspect.isclass(obj) and obj.__module__ == name:
            for member in list(vars(obj).values()):
                if _inspect.isfunction(member):
                    _rebind_defaults(member, proxy)
    return True


def _blocked_focused_window_title(timeout_s: float = 0.5):
    """cast.focused_window_title imports subprocess inside the function;
    the real one returns None when xdotool is not there, and so does this."""
    if _real_subprocess.run is not _REAL_RUN:
        return _real_focused_window_title(timeout_s)
    try:
        _refuse("jarvis.cast", "run", ["xdotool", "getactivewindow", "getwindowname"])
    except DesktopFirewallRefused:
        return None


_real_focused_window_title = None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_subprocess(*modules): put the REAL subprocess module back inside "
        "the named jarvis modules for this test (the desktop firewall in "
        "tests/conftest.py refuses run/Popen there otherwise)")


@pytest.fixture(autouse=True)
def _desktop_firewall_turn(request, monkeypatch):
    """Per test: a fresh sink for `desktop_attempts`, the ledger's nodeid,
    and the `real_subprocess` marker's opt-in."""
    _desktop_current["nodeid"] = request.node.nodeid
    _desktop_current["sink"] = []
    marker = request.node.get_closest_marker("real_subprocess")
    for name in (marker.args if marker else ()):
        mod = _sys.modules.get(name)
        if mod is None:
            raise pytest.UsageError(f"real_subprocess: {name} is not imported")
        monkeypatch.setattr(mod, "subprocess", _real_subprocess)
    yield
    _desktop_current["sink"] = None
    _desktop_current["nodeid"] = ""


@pytest.fixture
def desktop_attempts(request):
    """The subprocess calls the desktop firewall refused DURING THIS TEST:
    a list of (module, kind, program, argv), in order. Asking for it says
    the test owns them (they leave the session-end ledger)."""
    _owned_nodeids.add(request.node.nodeid)
    return _desktop_current["sink"]


@pytest.fixture(scope="session", autouse=True)
def _firewall_live_log_dir(tmp_path_factory):
    """Every jarvis path that can write under the log dir points at the
    throwaway directory for the whole session."""
    _TMP_ROOT[:] = [Path(str(tmp_path_factory.getbasetemp())).resolve()]
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
    # The FACE GALLERY, asserted here and not only in tests/test_faceenrol.py.
    # The env var above is forced, but a forced env var is one belt; the
    # voiceprint was given two after it was destroyed, and a face embedding is
    # the same kind of irreplaceable measurement. Asserted in the
    # SESSION-SCOPED fixture so a broken redirect fails before the first test
    # runs, rather than after one has already written -- and the gallery does
    # not exist on this box yet, so a leak now would silently CREATE a fixture
    # gallery at the real path, which is harder to notice than corrupting one.
    real_face = Path.home() / ".aiws_trainer" / "face_gallery"
    assert config.PATHS.FACE_GALLERY != real_face, \
        "PATHS.FACE_GALLERY still targets the user's enrolled face"
    assert real_face not in config.PATHS.FACE_GALLERY.parents, \
        "PATHS.FACE_GALLERY is inside the user's real face gallery"
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
    # THE DESKTOP FIREWALL (see the block above _SubprocessProxy): every
    # subprocess seam a Commander.handle call can reach is refused for the
    # session. The four the 17:21 incident went through are asserted, not
    # merely attempted: a box on which jarvis.commander did not import has
    # no suite to run anyway.
    global _real_focused_window_title
    wrapped = [name for name in _DESKTOP_MODULES if _wrap_module(name)]
    for name in ("jarvis.commander", "jarvis.enrolentry", "jarvis.workflows",
                 "jarvis.desktop"):
        assert name in wrapped, f"the desktop firewall did not wrap {name}"
    from jarvis import cast as _cast
    _real_focused_window_title = _cast.focused_window_title
    _cast.focused_window_title = _blocked_focused_window_title
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
            raise SmtpFirewallRefused(
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
        _cast.focused_window_title = _real_focused_window_title
        for fn, defaults, kwdefaults in _rebound:
            fn.__defaults__ = defaults
            if kwdefaults is not None:
                fn.__kwdefaults__ = kwdefaults
        for mod, real in _wrapped:
            mod.subprocess = real


def pytest_sessionfinish(session, exitstatus):
    """The SMTP firewall's second belt (F24, 09-03). SmtpFirewallRefused
    already comes out of the test that lost its fake; this catches the one
    it cannot reach -- a refusal on a worker thread, which dies with a
    traceback on stderr and nothing else. A test that trips the firewall ON
    PURPOSE (test_send_file.py asserts it works) takes its own entry back
    off the list; anything left here is a test that reached for a real
    mail server without knowing it, and the run fails."""
    if _smtp_blocked:
        session.exitstatus = 1
        rep = session.config.pluginmanager.get_plugin("terminalreporter")
        if rep is not None:
            rep.write_sep("=", "SMTP FIREWALL: unowned connection attempts",
                          red=True, bold=True)
            for address in _smtp_blocked:
                rep.write_line(f"  refused {address!r}", red=True)
            rep.write_line("A test lost its FakeSMTP. Pass smtp= to "
                           "mail.send_message or put one on services.smtp.",
                           red=True)
    # The desktop firewall's ledger: every refusal, by test. Reported, not
    # failed -- each refusal was already absorbed by the module's own
    # missing-binary path, so the suite is hermetic either way; this line
    # is how a test that reaches for the desktop without knowing it gets
    # noticed and given a seam. A test that owns its refusals (it asked
    # for `desktop_attempts`) is not listed.
    unowned = [row for row in _desktop_ledger
               if row[0] and row[0] not in _owned_nodeids]
    if unowned:
        rep = session.config.pluginmanager.get_plugin("terminalreporter")
        if rep is not None:
            rep.write_sep("-", "DESKTOP FIREWALL: refused subprocess calls "
                          "no test owned", yellow=True)
            # One line per test FILE: a hundred lines for one unseamed boot
            # path is noise, and the file is where the seam goes.
            by_file: dict = {}
            for nodeid, owner, kind, program in unowned:
                path = nodeid.split("::", 1)[0]
                calls, tests = by_file.setdefault(path, (set(), set()))
                calls.add(f"{owner}.{kind}({program})")
                tests.add(nodeid)
            for path, (calls, tests) in sorted(by_file.items()):
                rep.write_line(f"  {path} ({len(tests)} tests): "
                               f"{', '.join(sorted(calls))}", yellow=True)
            rep.write_line("Give each a seam (run=/popen=/_run), a global "
                           "subprocess monkeypatch, `desktop_attempts` or "
                           "@pytest.mark.real_subprocess.", yellow=True)


_owned_nodeids: set = set()
