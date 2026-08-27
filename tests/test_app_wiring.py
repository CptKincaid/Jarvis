"""Integration tests for the wiring item (spec section 11).

Everything here builds the REAL modules — assistant config, tool registry,
brain, router, timekeeper, notes, approvals, Claude session manager, alerts,
Discord channel, commander — on tmp paths, with only the hardware-bound
classes stubbed (TTS, whisper, the mic chain, the retained V1 agent) and
without a single Tk call. Nothing here touches Ollama, the network, X, or
audio: the brain's model calls are replaced per test, and the two subprocess
seams (`jarvis.channels.notify._run`, `jarvis.tools.timekeeper._run`) are
recorders.

One test is deliberately heavier: the approval round trip runs the real
`jarvis.mcp_permissions` CLI as a subprocess talking to this process's
`ApprovalBroker` over the real UNIX socket, and answers it the way Discord
does. That is the one path where a wiring mistake is invisible in unit tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.channels.notify as notify_mod
import jarvis.tools.timekeeper as tk_mod
from jarvis.config import CONFIG, PATHS
from jarvis.events import (AlarmFired, ApprovalRequested, ApprovalResolved,
                           BriefingReady, ClaudeProgress, ClaudeTaskState,
                           JarvisReply, ReminderFired, bus)

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LIVE = Path("/tmp/vss_voice")


# ------------------------------------------------------------------ stubs
class _Stub:
    """Anything hardware-bound: every attribute is a no-op callable."""

    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, name):
        def _noop(*a, **kw):
            return None
        return _noop


class FakeTTS(_Stub):
    """Records what Jarvis says (app._say is the only speech door)."""

    def __init__(self, *a, **kw):
        self.spoken: list[str] = []

    def speak(self, text):
        self.spoken.append(text)

    def interrupt(self):
        return False

    def prewarm(self, phrases):
        self.prewarmed = list(phrases)


class Sink:
    def __init__(self, *types):
        self.events: list = []
        self._types = types
        for t in types:
            bus.subscribe(t, self.events.append)

    def close(self):
        for t in self._types:
            bus.unsubscribe(t, self.events.append)

    def of(self, etype):
        return [e for e in self.events if isinstance(e, etype)]

    def wait(self, etype, timeout=8.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            bus.drain()
            hits = self.of(etype)
            if hits:
                return hits[-1]
            time.sleep(0.05)
        return None


def _wait(predicate, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        bus.drain()
        if predicate():
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------- fixtures
@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Every path the app writes to, under tmp (the conftest firewall already
    redirects the log dir / config / cache / memory dir for the session)."""
    cfg = tmp_path / "assistant.json"
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(cfg))
    monkeypatch.setattr(PATHS, "ASSISTANT_CONFIG", cfg)
    monkeypatch.setattr(PATHS, "LOG_DIR", tmp_path)
    monkeypatch.setattr(PATHS, "MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(PATHS, "TIMEKEEPER_DB", tmp_path / "memory" / "timekeeper.db")
    monkeypatch.setattr(PATHS, "NOTES_DB", tmp_path / "memory" / "notes.db")
    monkeypatch.setattr(PATHS, "CLAUDE_PROJECTS", tmp_path / "memory" / "projects.json")
    monkeypatch.setattr(PATHS, "REMINDERS", tmp_path / "memory" / "reminders.json")
    monkeypatch.setattr(PATHS, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(PATHS, "APPROVALS_SOCK", tmp_path / "approvals.sock")
    monkeypatch.setattr(PATHS, "CLAUDE_TASK_DIR", tmp_path / "claude")
    monkeypatch.setattr(PATHS, "MCP_CONFIG", tmp_path / "mcp_jarvis.json")
    monkeypatch.setattr(PATHS, "AUTOSTART_DESKTOP", tmp_path / "autostart" /
                        "jarvis.desktop")
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def seams(monkeypatch):
    """No hardware, no toasts, no ringer, no whisper/XTTS/CUDA."""
    for name in ("Transcriber", "Recorder", "MicArbiter", "SpeakerVerifier",
                 "Hotword", "JarvisAgent"):
        monkeypatch.setattr(app_mod, name, _Stub)
    monkeypatch.setattr(app_mod, "TTS", FakeTTS)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(CONFIG, "hotword", False)
    runs: list[tuple[str, list]] = []
    monkeypatch.setattr(notify_mod, "_run",
                        lambda argv, timeout=5.0: runs.append(("notify", argv)))
    monkeypatch.setattr(tk_mod, "_run",
                        lambda argv, **kw: runs.append(("ring", argv)))
    return runs


@pytest.fixture
def build(paths, seams, monkeypatch):
    """Factory: build a real JarvisApp and clean up after it (bus
    subscriptions included — a leaked subscriber would fire in later
    tests against a dead app)."""
    built: list = []

    def _make(**patches):
        for name, value in patches.items():
            monkeypatch.setattr(app_mod, name, value)
        before = {k: list(v) for k, v in bus._subs.items()}
        a = app_mod.JarvisApp()
        built.append((a, before))
        # No Ollama, ever: the brain's three model doors are recorders unless
        # a test replaces them.
        a.brain.classify_route = lambda text, timeout=None: ("local", 1.0)
        a.brain.chat = lambda text, callback=None, force_tool=None: None
        a.brain.local_line = lambda *args, **kw: kw.get("fallback", "")
        a.brain.summarize = lambda text, n=2: ""
        return a

    yield _make
    from jarvis import speak_queue
    for a, before in built:
        try:
            a.stop_assistant()
        finally:
            with bus._lock:
                bus._subs.clear()
                bus._subs.update({k: list(v) for k, v in before.items()})
    speak_queue.stop_watcher()
    bus.drain()


@pytest.fixture
def app(build):
    return build()


# ------------------------------------------------- 1. services namespace
SERVICE_MEMBERS = ("assistant", "tools", "brain", "router", "timekeeper",
                   "notes", "claude", "approvals", "alerts",
                   "desktop", "context", "memory", "workflows", "tts",
                   "reader", "history")


def test_services_namespace_has_every_member_of_the_spec_table(app):
    s = app.services
    for name in SERVICE_MEMBERS:
        assert getattr(s, name, None) is not None, f"services.{name} missing"
    assert type(s.assistant).__name__ == "AssistantConfig"
    assert type(s.tools).__name__ == "ToolRegistry"
    assert type(s.router).__name__ == "Router"
    assert type(s.timekeeper).__name__ == "Timekeeper"
    assert type(s.notes).__name__ == "NotesStore"
    assert type(s.claude).__name__ == "ClaudeSessionManager"
    assert type(s.approvals).__name__ == "ApprovalBroker"
    assert type(s.alerts).__name__ == "Alerts"
    for fn in ("think", "chat", "classify_route", "summarize", "local_line",
               "execute_autonomous"):
        assert callable(getattr(s.brain, fn)), f"brain.{fn} missing"
    # tool modules park their own state here
    assert hasattr(s, "calendar") and hasattr(s, "news_cache_path")
    assert hasattr(s, "spotify"), "spotify.make_tools never parked its tool"
    assert type(s.discord).__name__ == "DiscordChannel"


def test_every_tool_module_is_registered(app):
    names = app.tools.names()
    for expected in ("get_time", "get_location", "get_weather", "get_calendar",
                     "set_reminder", "set_timer", "set_alarm", "manage_schedule",
                     "notes", "get_mail", "get_briefing", "spotify_play",
                     "spotify_control", "spotify_now_playing"):
        assert expected in names, f"{expected} not registered"
    assert len(names) == len(set(names)), "duplicate tool names"
    # the brain sees the same registry
    from jarvis import brain as brain_mod
    assert brain_mod._REGISTRY is app.tools
    # spec 4.1 budget: the 11 spec tools plus the (later) Spotify six
    assert len(names) == 17


def test_a_tool_module_that_fails_to_import_does_not_abort_boot(build,
                                                                monkeypatch):
    import importlib
    real = importlib.import_module

    def picky(name, *a, **kw):
        if name in ("jarvis.tools.mail", "jarvis.router"):
            raise ImportError("boom")
        return real(name, *a, **kw)

    monkeypatch.setattr(app_mod.importlib, "import_module", picky)
    a = build()
    assert a.router is None and getattr(a.services, "router", "x") is None
    assert "get_mail" not in a.tools.names()
    assert "get_weather" in a.tools.names()        # the rest still landed
    # a missing router degrades to the brain's legacy think(), it does not raise
    seen = []
    a.brain.think = lambda text, callback=None: seen.append(text)
    result = a.dispatch_text("what's the weather like")
    assert seen == ["what's the weather like"]
    assert result.handled


# --------------------------------------------------------- 2. the routes
def test_local_utterance_reaches_the_brain(app):
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None: seen.append(text)
    result = app.dispatch_text("what's the weather like tomorrow")
    assert seen == ["what's the weather like tomorrow"]
    assert result.handled and result.done is False


def test_claude_utterance_reaches_the_session_manager(app):
    calls = []
    app.claude.submit = lambda prompt, project=None, parallel=False, model=None: \
        calls.append((prompt, project, parallel, model))
    app.brain.local_line = lambda *a, **kw: "Right away, sir."
    result = app.dispatch_text("fix the failing tests in jarvis/router.py")
    assert calls and calls[0][0] == "fix the failing tests in jarvis/router.py"
    assert result.reply == "Right away, sir." and result.speak
    assert app.tts.spoken == ["Right away, sir."]


def test_action_utterance_reaches_the_manager_method(app):
    calls = []
    app.claude.cancel = lambda *a, **kw: calls.append(("cancel", a, kw)) or "Stopped, sir."
    result = app.dispatch_text("stop that")
    assert calls, "claude.cancel was never called"
    assert result.reply == "Stopped, sir."


def test_outside_dir_refusal_offers_the_terminal_and_yes_opens_it(app):
    """claude.submit refuses with OUTSIDE_LINE; the next 'yes' opens the
    pop-out terminal instead of starting a new task (hooks_W7 §3)."""
    from jarvis.claude_session import OUTSIDE_LINE
    opened = []
    app.claude.submit = lambda prompt, project=None, parallel=False, model=None: OUTSIDE_LINE
    app.claude.open_terminal = lambda slug=None: opened.append(slug) or True
    first = app.dispatch_text("write a config file in /etc for me")
    assert first.reply == OUTSIDE_LINE
    second = app.dispatch_text("yes please")
    assert opened, "the terminal was never opened"
    assert second.reply == "Up on screen, sir."
    # the offer is one-shot
    app.brain.chat = lambda text, callback=None, force_tool=None: None
    third = app.dispatch_text("yes please")
    assert third.reply != "Up on screen, sir."


def test_ambiguous_utterance_asks_exactly_one_question(app):
    from jarvis.router import ROUTER_QUESTION
    app.brain.classify_route = lambda text, timeout=None: ("local", 0.0)
    result = app.dispatch_text("sort out the thing we talked about")
    assert result.reply == ROUTER_QUESTION and result.speak
    assert app.tts.spoken == [ROUTER_QUESTION]


def test_music_utterance_stays_local(app):
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None: seen.append(text)
    app.dispatch_text("play some miles davis")
    assert seen == ["play some miles davis"]


# ------------------------------------------------------ 3. the timekeeper
def test_a_timer_fires_through_the_bus_and_the_alerts_hub(app, seams):
    sink = Sink(ReminderFired, AlarmFired)
    try:
        app.start_assistant(residency=False)
        app.timekeeper.add_timer(1.0, "")
        ev = sink.wait(ReminderFired, timeout=10)
        assert ev is not None, "the timer never reached the bus"
        assert "1-second timer" in ev.text
        assert app.tts.spoken == ["Sir, your 1-second timer is up."]
        app.alerts.flush(3)
        kinds = [(r.kind, r.title) for r in app.alerts.recent]
        assert ("reminder", "Reminder") in kinds
    finally:
        sink.close()


def test_the_timekeeper_does_not_toast_twice(app):
    # The alerts hub owns notify-send once it is attached (spec 8.1).
    assert app.alerts is not None
    assert app.timekeeper.notify_enabled is False


def test_alarm_action_dismisses_and_snoozes(app):
    calls = []
    app.timekeeper.stop_ringing = lambda action="dismiss": calls.append(("stop", action)) or True
    app.timekeeper.snooze = lambda minutes: calls.append(("snooze", minutes)) or True
    assert app.alarm_action("a1", "dismiss") is True
    assert app.alarm_action("a1", "snooze", 10) is True
    assert calls == [("stop", "dismiss"), ("snooze", 10)]
    app.timekeeper = None                       # the tolerant path (spec 2.2)
    assert app.alarm_action("a1", "dismiss") is False


# ----------------------------------------------------- 4. Claude events
def test_claude_done_speaks_a_summary_and_alerts(app):
    app.brain.summarize = lambda text, n=2: "The tests pass now, sir."
    app.claude.task = lambda task_id: SimpleNamespace(
        result_text="Fixed the parser. Ran the suite. All 1066 tests pass. "
                    "Nothing else needed doing.")
    sink = Sink(JarvisReply)
    try:
        bus.publish(ClaudeTaskState(project="jarvis", task_id="t1", state="done"))
        assert _wait(lambda: sink.of(JarvisReply), 8), "no reply for a finished task"
        assert sink.of(JarvisReply)[-1].text == "The tests pass now, sir."
        assert app.tts.spoken[-1] == "The tests pass now, sir."
        app.alerts.flush(3)
        assert ("done", "Claude · jarvis") in [(r.kind, r.title)
                                               for r in app.alerts.recent]
    finally:
        sink.close()


def test_claude_failed_alerts_blocked(app):
    sink = Sink(JarvisReply)
    try:
        bus.publish(ClaudeTaskState(project="jarvis", task_id="t2", state="failed",
                                    text="The build broke, sir."))
        bus.drain()
        assert sink.of(JarvisReply)[-1].text == "The build broke, sir."
        app.alerts.flush(3)
        assert ("blocked", "Claude · jarvis") in [(r.kind, r.title)
                                                  for r in app.alerts.recent]
    finally:
        sink.close()


def test_a_milestone_is_spoken_once_and_alerted(app):
    bus.publish(ClaudeProgress(project="jarvis", task_id="t3",
                               line="Tests: 1066 passed", milestone=True))
    bus.publish(ClaudeProgress(project="jarvis", task_id="t3",
                               line="Edit jarvis/router.py", milestone=False))
    bus.drain()
    assert app.tts.spoken == ["Tests: 1066 passed"]
    app.alerts.flush(3)
    assert [r.kind for r in app.alerts.recent] == ["milestone"]


# -------------------------------------------------------- 5. the briefing
def test_brain_tags_briefing_becomes_one_card_and_no_reply(app):
    sections = {"weather": "Today: high 100, low 76.",
                "news": ["AWS acquires DuckLabs (Hacker News)"]}
    sink = Sink(BriefingReady, JarvisReply)
    try:
        app._on_brain_tags([("BRIEFING", json.dumps(sections)),
                            ("SPEAK", "Briefing for Wednesday, sir.")])
        bus.drain()
        cards = sink.of(BriefingReady)
        assert len(cards) == 1
        assert cards[0].sections == sections
        assert cards[0].spoken == "Briefing for Wednesday, sir."
        assert sink.of(JarvisReply) == [], "the briefing turn must not add a reply card"
        assert app.tts.spoken == ["Briefing for Wednesday, sir."]
    finally:
        sink.close()


def test_brain_tags_speak_without_a_briefing_is_an_ordinary_reply(app):
    sink = Sink(BriefingReady, JarvisReply)
    try:
        app._on_brain_tags([("SPEAK", "It's 7:42 pm, sir.")])
        bus.drain()
        assert sink.of(BriefingReady) == []
        assert [e.text for e in sink.of(JarvisReply)] == ["It's 7:42 pm, sir."]
    finally:
        sink.close()


# ------------------------------------------------------- 6. approvals
def test_approval_question_is_spoken_and_alerted(app):
    bus.publish(ApprovalRequested(request_id="r1", question="Allow it, sir?",
                                  tool_name="Write", detail="/etc/hosts",
                                  project="jarvis"))
    bus.drain()
    assert app.tts.spoken == ["Allow it, sir?"]
    app.alerts.flush(3)
    rec = app.alerts.recent[-1]
    assert (rec.kind, rec.request_id) == ("question", "r1")


def test_approval_resolved_elsewhere_is_acknowledged(app):
    bus.publish(ApprovalResolved(request_id="r1", allowed=True, source="discord"))
    bus.publish(ApprovalResolved(request_id="r2", allowed=False, source="timeout"))
    bus.publish(ApprovalResolved(request_id="r3", allowed=True, source="typed"))
    bus.drain()
    # typed / voice answers are acknowledged by the commander, not here
    assert app.tts.spoken == [app_mod.ALLOWED_LINE, app_mod.APPROVAL_TIMEOUT_LINE]


def test_approval_round_trip_through_the_real_socket(app, tmp_path):
    """The whole permission path, app included: the MCP CLI runs as a
    subprocess, reaches this process's broker over the UNIX socket, the app
    speaks the question and toasts it, a Discord 'yes' allows it, and the
    subprocess gets `allow` back."""
    sink = Sink(ApprovalRequested, ApprovalResolved)
    try:
        app.start_assistant(residency=False)
        assert app.approvals.running

        def answer_when_asked():
            for _ in range(200):
                time.sleep(0.05)
                bus.drain()
                if app.approvals.pending():
                    app._on_discord("yes please", "42")
                    return

        threading.Thread(target=answer_when_asked, daemon=True).start()
        env = dict(os.environ, JARVIS_APPROVAL_SOCK=str(PATHS.APPROVALS_SOCK),
                   JARVIS_PROJECT="jarvis", PYTHONPATH=str(REPO_ROOT))
        target = str(tmp_path / "outside" / "notes.txt")
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "approve",
                          "arguments": {"tool_name": "Write",
                                        "input": {"file_path": target,
                                                  "content": "hi"}}}}
        proc = subprocess.run([PY, "-m", "jarvis.mcp_permissions"],
                              input=json.dumps(msg).encode() + b"\n",
                              capture_output=True, cwd=str(tmp_path), env=env,
                              timeout=60)
        assert proc.returncode == 0, proc.stderr.decode()
        body = json.loads(json.loads(proc.stdout.splitlines()[0])
                          ["result"]["content"][0]["text"])
        assert body["behavior"] == "allow"
        bus.drain()
        asked = sink.of(ApprovalRequested)[-1]
        assert asked.tool_name == "Write" and target in asked.detail
        assert asked.project == "jarvis"
        resolved = sink.of(ApprovalResolved)[-1]
        assert (resolved.allowed, resolved.source) == (True, "discord")
        assert asked.question in app.tts.spoken
        assert app_mod.ALLOWED_LINE in app.tts.spoken
        app.alerts.flush(3)
        assert "question" in [r.kind for r in app.alerts.recent]
    finally:
        sink.close()


def test_ui_approval_answer_uses_the_broker(app):
    calls = []
    app.approvals.answer = lambda allowed, request_id=None, source="": \
        calls.append((allowed, request_id, source)) or True
    assert app.approval_answer("r9", True) is True
    assert calls == [(True, "r9", "ui")]
    app.approvals = None
    assert app.approval_answer("r9", True) is False


# --------------------------------------------------------- 7. Discord
@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("yes please", True), ("allow it", True), ("go ahead", True),
    ("no", False), ("deny", False), ("no thanks", False),
    ("what's the weather", None), ("", None),
    ("yes and no", None), ("play some miles davis on the kitchen speaker", None),
])
def test_yes_no_words(text, expected):
    assert app_mod.yes_no(text) is expected


def test_discord_text_becomes_an_ordinary_command(app):
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None: seen.append(text)
    t = app._on_discord("what's the weather like", "42")
    if t is not None:
        t.join(5)
    assert seen == ["what's the weather like"]


def test_discord_yes_answers_a_pending_approval_instead(app):
    calls = []
    app.approvals.pending = lambda: [SimpleNamespace(request_id="r1")]
    app.approvals.answer = lambda allowed, request_id=None, source="": \
        calls.append((allowed, source)) or True
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None: seen.append(text)
    assert app._on_discord("yes", "42") is None
    assert calls == [(True, "discord")]
    assert seen == []


def test_replies_are_posted_back_while_a_discord_exchange_is_active(app):
    posts = []
    app.discord.post = lambda text, **kw: posts.append(text)
    app._discord_active_until = time.time() + 60
    bus.publish(JarvisReply(text="It's 7:42 pm, sir.", speak=True))
    bus.drain()
    assert _wait(lambda: posts, 5)
    assert posts == ["It's 7:42 pm, sir."]
    # the same text again inside 30 s is not re-posted
    bus.publish(JarvisReply(text="It's 7:42 pm, sir.", speak=True))
    bus.drain()
    time.sleep(0.2)
    assert posts == ["It's 7:42 pm, sir."]


def test_replies_are_not_posted_when_no_exchange_is_active(app):
    posts = []
    app.discord.post = lambda text, **kw: posts.append(text)
    app._discord_active_until = 0.0
    bus.publish(JarvisReply(text="Quietly, sir.", speak=False))
    bus.drain()
    time.sleep(0.2)
    assert posts == []


# ---------------------------------------------------------- 8. options
def test_get_and_set_option_go_through_the_assistant_config(app):
    assert app.get_option("briefing.enabled") is False
    assert app.set_option("briefing.enabled", True)
    assert app.assistant.get("briefing.enabled") is True
    assert json.loads(PATHS.ASSISTANT_CONFIG.read_text())["briefing"]["enabled"] is True


def test_autostart_option_installs_and_removes_the_entry(app):
    assert app.set_option("autostart.enabled", True)
    assert PATHS.AUTOSTART_DESKTOP.exists()
    text = PATHS.AUTOSTART_DESKTOP.read_text()
    assert "Exec=" in text and "-m jarvis.app" in text
    assert app.set_option("autostart.enabled", False)
    assert not PATHS.AUTOSTART_DESKTOP.exists()


def test_open_terminal_is_tolerant(app):
    calls = []
    app.claude.open_terminal = lambda slug=None: calls.append(slug) or True
    assert app.open_terminal("jarvis") is True
    assert calls == ["jarvis"]
    app.claude = None
    assert app.open_terminal() is False


def test_ui_service_kwargs_carry_the_assistant_hooks(app):
    kwargs = app.ui_service_kwargs()
    for name in ("open_terminal", "alarm_action", "approval_answer",
                 "get_option", "set_option", "dispatch_text"):
        assert callable(kwargs[name]), name


def test_build_ui_services_drops_unknown_fields():
    import dataclasses

    @dataclasses.dataclass
    class OldServices:
        dispatch_text: object = None

    built = app_mod.build_ui_services(OldServices, {"dispatch_text": print,
                                                    "set_option": print})
    assert built.dispatch_text is print


# ------------------------------------------------------- 9. prewarm lines
def test_canned_phrases_include_the_fixed_persona_lines(app):
    phrases = app._canned_phrases()
    assert len(phrases) == len(set(phrases)), "duplicate prewarm phrase"
    from jarvis.assistant_config import AssistantConfig
    for line in AssistantConfig.setup_lines():
        assert line in phrases
    from jarvis import approvals as ap_mod
    from jarvis import claude_session as cs_mod
    from jarvis.router import ROUTER_QUESTION
    from jarvis.tools.briefing import BRIEFING_OFF_LINE
    from jarvis.commander import TERMINAL_OPEN_LINE
    from jarvis.tools.spotify import LINKED_LINE, NOT_LINKED_LINE
    for line in (TERMINAL_OPEN_LINE,
                 ROUTER_QUESTION, BRIEFING_OFF_LINE, cs_mod.CANCELLED_LINE,
                 cs_mod.OUTSIDE_LINE, ap_mod.TIMEOUT_LINE, ap_mod.DECLINED_LINE,
                 NOT_LINKED_LINE, LINKED_LINE, app_mod.APPROVAL_TIMEOUT_LINE):
        assert line in phrases, line
    # never prewarm a template
    assert not any("{" in p for p in phrases)


# ------------------------------------------------- 10. launcher / CLI
def test_focus_running_instance_writes_the_pid_when_none_is_alive(paths,
                                                                  monkeypatch):
    ran = []
    monkeypatch.setattr(app_mod, "_run", lambda argv, timeout=5: ran.append(argv))
    assert app_mod._focus_running_instance() is False
    assert (PATHS.LOG_DIR / "jarvis.pid").read_text() == str(os.getpid())
    assert ran == [], "nothing to raise when we are the first instance"


def test_focus_running_instance_raises_an_existing_window(paths, monkeypatch):
    (PATHS.LOG_DIR / "jarvis.pid").write_text(str(os.getpid()))   # "alive"
    ran = []

    def fake_run(argv, timeout=5):
        ran.append(argv)
        if argv[:2] == ["xdotool", "search"]:
            return SimpleNamespace(stdout="4194305\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(app_mod, "_run", fake_run)
    monkeypatch.setattr(app_mod, "_sleep", lambda s: None)
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    assert app_mod._focus_running_instance() is True
    assert ran[0] == ["xdotool", "search", "--classname", "jarvis"]
    assert [a[1] for a in ran[1:]] == ["windowmap", "windowactivate", "windowraise"]
    assert all(a[2] == "4194305" for a in ran[1:])


def test_focus_running_instance_notifies_when_the_window_never_appears(
        paths, monkeypatch):
    (PATHS.LOG_DIR / "jarvis.pid").write_text(str(os.getpid()))
    ran = []
    monkeypatch.setattr(app_mod, "_run", lambda argv, timeout=5: (
        ran.append(argv), SimpleNamespace(stdout="", returncode=1))[1])
    monkeypatch.setattr(app_mod, "_sleep", lambda s: None)
    monkeypatch.setattr(app_mod, "FOCUS_WAIT_S", 0.05)
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    assert app_mod._focus_running_instance() is True
    assert ran[-1][0] == "notify-send"
    assert "Starting up, sir…" in ran[-1]


def test_install_autostart_flag_never_touches_the_ui(paths, monkeypatch):
    calls = []
    monkeypatch.setattr(app_mod, "install_autostart_cli",
                        lambda: calls.append("autostart") or 0)
    monkeypatch.setattr(sys, "argv", ["jarvis", "--install-autostart"])
    with pytest.raises(SystemExit) as exc:
        app_mod.main()
    assert exc.value.code == 0 and calls == ["autostart"]


def test_spotify_login_flag_never_touches_the_ui(paths, monkeypatch):
    calls = []
    import jarvis.tools.spotify as sp
    monkeypatch.setattr(sp, "login_cli", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["jarvis", "--spotify-login", "--no-browser"])
    with pytest.raises(SystemExit) as exc:
        app_mod.main()
    assert exc.value.code == 0
    assert calls == [["--login", "--no-browser"]]


def test_install_autostart_cli_writes_the_entry_and_records_the_choice(
        paths, monkeypatch):
    from jarvis import autostart
    monkeypatch.setattr(autostart, "disable_gnome_suspend", lambda **kw: True)
    monkeypatch.setattr(autostart, "PATH", PATHS.AUTOSTART_DESKTOP, raising=False)
    assert app_mod.install_autostart_cli() == 0
    assert PATHS.AUTOSTART_DESKTOP.exists()
    from jarvis.assistant_config import AssistantConfig
    assert AssistantConfig.load().get("autostart.enabled") is True


def test_preload_runs_off_the_main_thread_and_gates_the_model_load(monkeypatch):
    order = []
    monkeypatch.setattr(app_mod, "_preload_heavy_imports",
                        lambda: (time.sleep(0.05), order.append("preload")))
    t = app_mod.start_preload()
    assert not app_mod._PRELOAD_DONE.is_set()
    t.join(10)
    assert app_mod._PRELOAD_DONE.is_set() and order == ["preload"]


# ----------------------------------------------------------- 11. paths
def test_paths_match_spec_3_2():
    assert PATHS.TIMEKEEPER_DB.name == "timekeeper.db"
    assert PATHS.NOTES_DB.name == "notes.db"
    assert PATHS.CLAUDE_PROJECTS.name == "claude_projects.json"
    assert PATHS.APPROVALS_SOCK.name == "approvals.sock"
    assert PATHS.MCP_CONFIG.name == "mcp_jarvis.json"
    assert PATHS.CLAUDE_TASK_DIR.name == "claude"
    assert PATHS.AUTOSTART_DESKTOP.name == "jarvis.desktop"
    assert PATHS.ASSISTANT_CONFIG.name == "assistant.json"


def test_the_suite_never_points_at_the_live_app():
    for name in ("LOG_DIR", "APPROVALS_SOCK", "CLAUDE_TASK_DIR", "MCP_CONFIG"):
        path = getattr(PATHS, name)
        assert path != LIVE and LIVE not in path.parents


# ------------------------------------------------------- 12. shutdown
def test_stop_assistant_stops_every_started_thread(app):
    app.start_assistant(residency=False)
    assert app.approvals.running
    assert app.timekeeper.running if hasattr(app.timekeeper, "running") else True
    app.stop_assistant()
    assert not app.approvals.running


def test_start_assistant_is_idempotent(app, monkeypatch):
    starts = []
    app.timekeeper.start = lambda *a, **kw: starts.append("tk")
    app.approvals.start = lambda *a, **kw: starts.append("ap")
    app.discord.start = lambda *a, **kw: starts.append("dc")
    app.start_assistant(residency=False)
    app.start_assistant(residency=False)
    assert starts == ["tk", "ap", "dc"]
