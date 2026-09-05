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
import jarvis.desktop as desktop_mod
import jarvis.tools.timekeeper as tk_mod
import jarvis.tts as tts_mod
from jarvis.config import CONFIG, PATHS
from jarvis.events import (AlarmFired, ApprovalRequested, ApprovalResolved,
                           UncertainResolved, UncertainUtterance,
                           BriefingReady, ClaudeProgress, ClaudeTaskState,
                           JarvisReply, ReminderFired, Status, bus)

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


class FakeRendition:
    """What ``TTS.render`` hands back, without an engine behind it.

    Shaped exactly like ``jarvis.tts.Rendition`` -- ``cached`` / ``plan`` /
    ``body()`` / ``stream()`` -- because the phone client branches on all
    four, and a fake that only had the one the happy path reads would let
    the cached branch rot untested. The audio is a fifth of a second of
    square wave, so it is real wav that a decoder accepts, and NOTHING here
    is synthesized: a test must never put a sentence on the GPU.
    """

    RATE = 24000
    PCM = b"".join(int(6000 * ((i // 60) % 2 * 2 - 1)).to_bytes(
        2, "little", signed=True) for i in range(RATE // 5))

    def __init__(self, text, cached=True, chunks=None):
        self.text = text
        # `chunks=[]` is meaningful (a reply that cleans down to nothing
        # sayable), so it cannot go through an `or [text]`.
        self.chunks = [text] if chunks is None else list(chunks)
        self.cached = bool(cached) and bool(self.chunks)
        self.plan = [(c, "cached.wav" if self.cached else None)
                     for c in self.chunks]
        self.gate = None            # an Event a test can hold a chunk on

    def __bool__(self):
        return bool(self.chunks)

    def __len__(self):
        return len(self.chunks)

    def stream(self):
        yield tts_mod.wav_header(self.RATE)
        for i, _ in enumerate(self.chunks):
            if i and self.gate is not None:
                self.gate.wait(timeout=10)
            yield self.PCM

    def body(self):
        pcm = self.PCM * len(self.chunks)
        return tts_mod.wav_header(self.RATE, data_bytes=len(pcm)) + pcm


class FakeTTS(_Stub):
    """Records what Jarvis says (app._say is the only speech door), and
    what was rendered for somebody who is NOT in the room."""

    def __init__(self, *a, **kw):
        self.spoken: list[str] = []
        self.rendered: list[str] = []
        self.rendition = FakeRendition          # a test may swap this

    def speak(self, text, block=False):
        self.spoken.append(text)

    def interrupt(self):
        return False

    def prewarm(self, phrases):
        self.prewarmed = list(phrases)

    def render(self, text):
        """The phone client's seam (jarvis/tts.py TTS.render). It records
        separately from ``spoken`` on purpose: the two lists are how a test
        proves a phone turn reached the phone and not the soundbar."""
        self.rendered.append(text)
        return self.rendition(text)


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
    monkeypatch.setattr(PATHS, "COMMAND_SOCK", tmp_path / "command.sock")
    monkeypatch.setattr(PATHS, "REVIEWS_DIR", tmp_path / "memory" / "reviews")
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
    # the brain fakes below take (text, callback, force_tool): no streaming
    monkeypatch.setattr(CONFIG, "stream_replies", False)
    monkeypatch.setattr(app_mod, "TTS", FakeTTS)
    # the running-commit stamp shells out to git, which the desktop
    # firewall refuses under pytest; stub it so the ledger stays clean
    monkeypatch.setattr(app_mod, "running_commit", lambda **kw: "")
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(CONFIG, "hotword", False)
    runs: list[tuple[str, list]] = []
    monkeypatch.setattr(notify_mod, "_run",
                        lambda argv, timeout=5.0: runs.append(("notify", argv)))
    monkeypatch.setattr(tk_mod, "_run",
                        lambda argv, **kw: runs.append(("ring", argv)))
    # JarvisApp.__init__ -> desktop.restore_target() -> list_windows(), i.e.
    # `xdotool search --onlyvisible --name ""` against HIS display: every
    # test that built the app read his window titles until the desktop
    # firewall (tests/conftest.py, 2026-09-04) refused it. An empty desktop
    # is what a headless box has, so the app boots the same either way.
    monkeypatch.setattr(desktop_mod, "list_windows", lambda *a, **k: [])
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
                     "spotify_control", "spotify_now_playing",
                     "canvas_due", "canvas_grades", "canvas_announcements",
                     "ask_docs", "docs_reindex", "ask_code", "screen_qa",
                     "system_health", "recap_day"):
        assert expected in names, f"{expected} not registered"
    assert len(names) == len(set(names)), "duplicate tool names"
    # the brain sees the same registry
    from jarvis import brain as brain_mod
    assert brain_mod._REGISTRY is app.tools
    # spec 4.1 budget: the 11 spec tools plus the (later) Spotify six
    # 18 since 2026-08-28: calendar gained add_event alongside get_calendar
    # 25 since 2026-08-30: canvas (3), docs (2), screen (1), health (1). All
    # of them ride every turn: the tools are part of the cached static prefix.
    # 26 since 2026-08-30: the activity journal's recap_day.
    # 27 since 2026-08-30: ask_code, the local index over his own repos --
    # one more schema in the static prefix, bought back on every "where
    # does X live" that no longer costs a claude -p.
    assert len(names) == 27


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
    app.brain.chat = lambda text, callback=None, force_tool=None, **kw: seen.append(text)
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
    app.brain.chat = lambda text, callback=None, force_tool=None, **kw: seen.append(text)
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
    app.brain.chat = lambda text, callback=None, force_tool=None, **kw: seen.append(text)
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
    app.brain.chat = lambda text, callback=None, force_tool=None, **kw: seen.append(text)
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


def test_unset_option_removes_the_key_from_the_file_and_not_just_the_value(app):
    """A superseded key that is merely IGNORED still sits in his file
    looking live -- plausible numbers, no way to tell it from one that
    drives something. The SENSORS page retires presence.desk_band_m
    through this once its bands are in zones.rooms."""
    assert app.set_option("presence.desk_band_m", [2.25, 3.75])
    assert app.unset_option("presence.desk_band_m")
    assert app.get_option("presence.desk_band_m") is None
    written = json.loads(PATHS.ASSISTANT_CONFIG.read_text())
    assert "desk_band_m" not in written["presence"]
    # a key that is not there is not an error and is not a write
    assert app.unset_option("presence.desk_band_m") is False
    assert app.unset_option("nothing.at.all") is False


def test_autostart_option_installs_and_removes_the_entry(app):
    from jarvis import autostart
    assert app.set_option("autostart.enabled", True)
    assert PATHS.AUTOSTART_DESKTOP.exists()
    text = PATHS.AUTOSTART_DESKTOP.read_text()
    # Either shape counts as "Jarvis starts at login": the bare command, or
    # scripts/jarvis-autostart, which is that command behind a wait for the
    # Breeze sidecar. THIS write is why the wrapper preference has to live in
    # jarvis.autostart -- app.py re-runs install() at every start once the
    # option is on, so anything the module does not know to render it erases.
    assert "Exec=" in text
    assert "-m jarvis.app" in text or "jarvis-autostart" in text
    assert autostart.is_installed(path=PATHS.AUTOSTART_DESKTOP)
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
    from jarvis.commander import TERMINAL_OPEN_LINE
    from jarvis.tools.spotify import LINKED_LINE, NOT_LINKED_LINE
    for line in (TERMINAL_OPEN_LINE,
                 ROUTER_QUESTION, cs_mod.CANCELLED_LINE,
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


# ------------------------------------ 2b. "Was that for me?" is answerable
#
# The commander has always had `on_uncertain` and `resolve_uncertain`, but the
# app never set the hook, so an uncertain utterance produced a bare warn Status
# -- a 4 s toast, no way to reply -- and the utterance was dropped. Nothing in
# the shipping app ever reached resolve_uncertain, so the classifier feedback
# it feeds was dead code too. These tests hold that wiring in place.
def _make_uncertain(app, monkeypatch):
    from jarvis.commander import IntentClassifier
    monkeypatch.setattr(app.commander.intent, "classify",
                        lambda text: (IntentClassifier.UNCERTAIN, 0.5))
    # the spoken follow-up window needs a mic; covered by test_uncertain_reply
    monkeypatch.setattr(app, "_ask_uncertain", lambda rid: None)


def test_uncertain_utterance_asks_something_that_can_be_answered(app, monkeypatch):
    _make_uncertain(app, monkeypatch)
    sink = Sink(UncertainUtterance)
    try:
        app._dispatch("sort out the thing we talked about", "voice")
        ev = sink.wait(UncertainUtterance)
        assert ev is not None, "on_uncertain was never wired to the app"
        assert ev.request_id
        assert "Was that for me?" in ev.question
        assert app._pending_uncertain[ev.request_id] == \
            "sort out the thing we talked about"
    finally:
        sink.close()


def test_answering_yes_actually_runs_the_utterance(app, monkeypatch):
    _make_uncertain(app, monkeypatch)
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None, **kw: seen.append(text)
    monkeypatch.setattr(app.brain, "classify_route",
                        lambda text, timeout=None: ("local", 1.0))
    sink = Sink(UncertainUtterance, UncertainResolved)
    try:
        app._dispatch("play some miles davis", "voice")
        ev = sink.wait(UncertainUtterance)
        assert seen == [], "must not run before it is answered"

        app.uncertain_answer(ev.request_id, True)

        assert seen == ["play some miles davis"], "YES must dispatch the utterance"
        done = sink.wait(UncertainResolved)
        assert done is not None and done.yes is True
    finally:
        sink.close()


def test_answering_no_discards_it_and_teaches_the_classifier(app, monkeypatch):
    _make_uncertain(app, monkeypatch)
    taught = []
    monkeypatch.setattr(app.commander.intent, "log_feedback",
                        lambda text, is_for_assistant: taught.append(
                            (text, is_for_assistant)))
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None, **kw: seen.append(text)
    sink = Sink(UncertainUtterance)
    try:
        app._dispatch("she said no way lol haha dude", "voice")
        ev = sink.wait(UncertainUtterance)
        app.uncertain_answer(ev.request_id, False)

        assert seen == [], "NO must not run anything"
        assert taught == [("she said no way lol haha dude", False)], \
            "the answer must train the intent classifier"
    finally:
        sink.close()


def test_only_the_first_answer_counts(app, monkeypatch):
    """The card and the spoken window race; resolving twice would route the
    same utterance twice."""
    _make_uncertain(app, monkeypatch)
    seen = []
    app.brain.chat = lambda text, callback=None, force_tool=None, **kw: seen.append(text)
    monkeypatch.setattr(app.brain, "classify_route",
                        lambda text, timeout=None: ("local", 1.0))
    sink = Sink(UncertainUtterance)
    try:
        app._dispatch("play some miles davis", "voice")
        ev = sink.wait(UncertainUtterance)
        app.uncertain_answer(ev.request_id, True)
        app.uncertain_answer(ev.request_id, True)
        assert seen == ["play some miles davis"], f"ran {len(seen)} times"
    finally:
        sink.close()


def test_a_newer_question_supersedes_the_old_one(app, monkeypatch):
    _make_uncertain(app, monkeypatch)
    sink = Sink(UncertainUtterance, UncertainResolved)
    try:
        app._dispatch("first ambiguous thing", "voice")
        first = sink.wait(UncertainUtterance)
        app._dispatch("second ambiguous thing", "voice")
        bus.drain()
        assert first.request_id not in app._pending_uncertain
        superseded = [e for e in sink.of(UncertainResolved)
                      if e.source == "superseded"]
        assert superseded, "the stale card must be closed, not left hanging"
    finally:
        sink.close()


# ---------------------------------- 2c. one utterance at a time
#
# _on_hotword guarded only on `recorder.recording`, which is already False for
# the whole transcription pass -- ~20 s of Whisper on the 2026-08-27 clip. A
# second wake word inside that window opened a competing recording while the
# first transcript was still in flight. Releasing the mic after finalising
# (recorder.stop) closes ~200 ms of it; this closes the rest.
def test_a_wake_word_during_transcription_is_refused(app, monkeypatch):
    started = []
    # the stubbed recorder reports truthy .recording, which would satisfy the
    # OLD guard and make this pass for the wrong reason
    app.recorder.recording = False
    monkeypatch.setattr(app.recorder, "start", lambda: started.append(1))
    app._audio_busy.set()                    # pretend a transcript is in flight

    app._on_hotword(0.9)
    time.sleep(0.35)                         # the start is on a 0.2 s timer

    assert started == [], "a second capture opened mid-transcription"


def test_the_next_wake_word_works_once_processing_finishes(app, monkeypatch):
    started = []
    # speaker filtering is exercised elsewhere; this is about the busy flag
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    app.recorder.recording = False
    monkeypatch.setattr(app.recorder, "start", lambda: started.append(1))
    monkeypatch.setattr(app, "_dispatch", lambda text, source: None)
    monkeypatch.setattr(app.transcriber, "transcribe",
                        lambda audio: SimpleNamespace(
                            text="", confidence=0.0, accepted=True))

    app._process_audio(object())             # runs and clears the flag
    assert not app._audio_busy.is_set()

    app._on_hotword(0.9)
    time.sleep(0.35)
    assert started == [1]


def test_a_failure_mid_transcription_does_not_leave_jarvis_deaf(app, monkeypatch):
    """If the flag leaked on an exception, no wake word would ever work again."""
    def boom(audio):
        raise RuntimeError("cuda gone")

    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    monkeypatch.setattr(app.transcriber, "transcribe", boom)
    app._process_audio(object())
    assert not app._audio_busy.is_set()


# ------------------------------ 2d. "one moment" while he goes and looks
#
# A calendar question on 2026-08-27 spent 2.2 s in the local model with no
# sign anything was happening, on top of 12.9 s of listening. A short cached
# acknowledgement fills that gap -- cached because an uncached XTTS line took
# 12.6 s to render, which would make the reassurance slower than the answer.
def test_a_fast_reply_says_nothing_extra(app, monkeypatch):
    monkeypatch.setattr(app, "_thinking_delay_s", 5.0)   # far longer than the reply
    monkeypatch.setattr(app.commander, "handle",
                        lambda text, source: SimpleNamespace(
                            reply="Right away, sir.", speak=True, status=""))

    app._dispatch("what time is it", "voice")
    time.sleep(0.3)

    assert app.tts.spoken == ["Right away, sir."], app.tts.spoken


def test_a_slow_lookup_gets_an_acknowledgement_first(app, monkeypatch):
    monkeypatch.setattr(app, "_thinking_delay_s", 0.15)

    def slow(text, source):
        time.sleep(0.6)
        return SimpleNamespace(reply="Nothing on today, sir.", speak=True, status="")

    monkeypatch.setattr(app.commander, "handle", slow)

    app._dispatch("what's on my calendar", "voice")
    time.sleep(0.3)

    assert app.tts.spoken, "nothing was said while he was looking"
    assert app.tts.spoken[0] in app_mod.THINKING_LINES
    assert app.tts.spoken[-1] == "Nothing on today, sir."


def test_typed_input_never_gets_the_spoken_filler(app, monkeypatch):
    """You can see a typed answer arriving; being told to wait is noise."""
    monkeypatch.setattr(app, "_thinking_delay_s", 0.15)

    def slow(text, source):
        time.sleep(0.5)
        return SimpleNamespace(reply="Done, sir.", speak=False, status="")

    monkeypatch.setattr(app.commander, "handle", slow)

    app._dispatch("what's on my calendar", "typed")

    assert all(line not in app_mod.THINKING_LINES for line in app.tts.spoken)


def test_the_acknowledgement_is_prewarmed(app):
    """Uncached it would take longer to say than the answer it covers."""
    phrases = app._canned_phrases()
    for line in app_mod.THINKING_LINES:
        assert line in phrases, f"{line!r} would be rendered live"


# --------------------------- 2e. a turn is not over when handle() returns
#
# brain.chat runs on a worker thread (brain.py:902) and the commander returns
# CommandResult(done=False) meaning "the answer is still coming". Two things
# built on 2026-08-27/28 wrongly treated handle() returning as the end of the
# turn, and the 2026-08-28 01:03 log shows both failing: the "one moment"
# filler never spoke despite a 2.3 s wait, and a wake word at 01:03:17 opened
# a competing recording while the reply was still being generated.
def _async_result():
    return SimpleNamespace(reply="", speak=False, status="Thinking…", done=False)


def test_an_async_turn_stays_busy_until_the_reply_arrives(app, monkeypatch):
    monkeypatch.setattr(app.commander, "handle", lambda t, s: _async_result())

    app._dispatch("what's on my agenda for monday", "voice")
    assert app._turn_busy.is_set(), "handle() returning is not the end of the turn"

    app._on_brain_tags([("SPEAK", "You have Biosensors at nine ten, sir.")])
    assert not app._turn_busy.is_set(), "the reply should close the turn"


def test_a_wake_word_during_reply_generation_is_refused(app, monkeypatch):
    monkeypatch.setattr(app.commander, "handle", lambda t, s: _async_result())
    app.recorder.recording = False
    started = []
    monkeypatch.setattr(app.recorder, "start", lambda: started.append(1))

    app._dispatch("what's on my agenda", "voice")
    app._on_hotword(0.9)
    time.sleep(0.35)

    assert started == [], "opened a second capture mid-reply"
    app._on_brain_tags([("SPEAK", "done")])


def test_the_filler_speaks_for_a_slow_async_reply(app, monkeypatch):
    """The case it was built for, and the case it silently missed."""
    monkeypatch.setattr(app, "_thinking_delay_s", 0.15)
    monkeypatch.setattr(app.commander, "handle", lambda t, s: _async_result())

    app._dispatch("what's on my agenda for monday", "voice")
    time.sleep(0.45)

    assert app.tts.spoken and app.tts.spoken[0] in app_mod.THINKING_LINES
    app._on_brain_tags([("SPEAK", "done")])


def test_a_reply_that_never_comes_cannot_leave_him_deaf(app, monkeypatch):
    """A stuck turn flag would make every future wake word a no-op."""
    monkeypatch.setattr(app, "_turn_timeout_s", 0.2)
    monkeypatch.setattr(app.commander, "handle", lambda t, s: _async_result())

    app._dispatch("something that never answers", "voice")
    assert app._turn_busy.is_set()
    time.sleep(0.5)
    assert not app._turn_busy.is_set(), "the watchdog must release the turn"


def test_a_synchronous_answer_closes_the_turn_immediately(app, monkeypatch):
    monkeypatch.setattr(app.commander, "handle", lambda t, s: SimpleNamespace(
        reply="Half past nine, sir.", speak=True, status="", done=True))

    app._dispatch("what time is it", "voice")
    assert not app._turn_busy.is_set()


# ------------------------------- 2f. the screen may be verbose; the voice not
#
# 2026-08-28 12:54: the reply appeared on screen instantly and then took 23.0 s
# to speak. Not slow synthesis -- _on_brain_tags spoke the model's SPEAK text
# verbatim, and for "what's on my calendar Monday?" that was four classes with
# building names. Every other spoken path caps at two sentences; this one did
# not.
LONG_REPLY = (
    "You have four academic commitments on Monday, sir, starting with "
    "Biosensors at 9:10 am in Wisenbaker 049. Then Magnetic Resonance "
    "Engineering at 12:40 pm in the Emerging Technologies Building 1003. "
    "After that Electrical Design Lab Two presentation at 4:10 pm in ETB "
    "1020. Finally Magnetic Resonance Engineering again at 6:00 pm in "
    "Zachry 330 for about three hours.")


def test_a_long_reply_is_spoken_in_full(app):
    """Capping this at two sentences was tried and reverted the same day: the
    user wants him to sound human, and stopping mid-list does not. The delay
    that prompted the cap came from chunking, not from length."""
    app._on_brain_tags([("SPEAK", LONG_REPLY)])

    assert app.tts.spoken, "nothing was spoken"
    assert app.tts.spoken[-1] == LONG_REPLY


def test_the_full_text_still_reaches_the_screen(app):
    sink = Sink(JarvisReply)
    try:
        app._on_brain_tags([("SPEAK", LONG_REPLY)])
        ev = sink.wait(JarvisReply)
        assert ev is not None and ev.text == LONG_REPLY, \
            "the screen should keep the detail the voice drops"
    finally:
        sink.close()


def test_a_short_reply_is_untouched(app):
    app._on_brain_tags([("SPEAK", "You have nothing on today, sir.")])
    assert app.tts.spoken[-1] == "You have nothing on today, sir."


def test_the_filler_does_not_speak_once_the_answer_has_landed(app, monkeypatch):
    """It fired 31 ms before the reply on 2026-08-28 and the answer then
    queued behind it, adding ~2.9 s for no benefit."""
    app._turn_busy.clear()
    app._say_thinking()
    assert all(line not in app_mod.THINKING_LINES for line in app.tts.spoken)


def test_the_filler_threshold_clears_a_typical_reply(app):
    """Local replies measured 2.0-2.3 s; firing at 2.0 s guarantees it always
    speaks, which is the opposite of 'only when he has to really look'."""
    assert app_mod.THINKING_DELAY_S >= 3.0


def test_services_brain_chat_forwards_every_keyword_the_brain_takes(build, monkeypatch):
    """The commander never sees JarvisBrain.chat, only the wrapper in
    _build_services. A keyword the wrapper drops raises TypeError inside the
    handler and the user hears "Command failed: <name>" -- which is exactly
    what "what was my last email about?" said until force_args was forwarded.
    The routing tests stub the brain and so could not see this."""
    import inspect
    app = build()
    seen = {}
    monkeypatch.setattr(app.brain, "chat", lambda text, **kw: seen.update(kw) or None)
    app.services.brain.chat("my last email", force_tool="get_mail",
                            force_args={"unread_only": False, "limit": 1})
    assert seen["force_tool"] == "get_mail"
    assert seen["force_args"] == {"unread_only": False, "limit": 1}
    # and structurally: every keyword the real method accepts (bar callback,
    # which the wrapper supplies) must be accepted by the wrapper
    from jarvis.brain import JarvisBrain
    # callback and on_sentence are supplied by the wrapper itself
    real = set(inspect.signature(JarvisBrain.chat).parameters) - {"self", "callback", "max_rounds", "on_sentence"}
    wrapper = set(inspect.signature(app.services.brain.chat).parameters)
    assert real <= wrapper, f"wrapper drops {real - wrapper}"


def test_turn_ledger_reports_one_line_per_voice_turn(build, monkeypatch):
    """The "turn:" line is assembled from five events on four threads; this
    drives them through the real bus wiring and checks the arithmetic lands."""
    from jarvis.events import (RecordingStarted, RecordingStopped, SpeakingState,
                               Transcribed, bus as _bus)
    app = build()
    got = []
    monkeypatch.setattr(app.turns, "_emit", got.append)
    # the stub recorder's fake audio must not run STT; clear the flag the handler set
    monkeypatch.setattr(app, "_process_audio", lambda audio: app._audio_busy.clear())
    app.recorder.recording = False
    app._on_hotword(0.9)                              # "wake" is marked here, not via the bus
    _bus.publish(RecordingStarted())
    _bus.publish(RecordingStopped(reason="silence", endpoint="vad", dead_air_s=0.8))
    _bus.publish(Transcribed(text="what time is it", accepted=True))
    app.turns.mark("handle")
    _bus.publish(SpeakingState(active=True, amplitude=0.3))
    _bus.publish(SpeakingState(active=True, amplitude=0.5))     # amplitude ticks
    assert len(got) == 1, got
    rec = got[0]
    assert rec["outcome"] == "audio" and rec["stop"] == "vad"
    assert abs(rec["dead_air"] - 0.8) < 0.05
    assert rec["wait"] is not None and rec["wait"] >= rec["dead_air"]
    # a rejected clip closes the turn without a reply
    app._on_hotword(0.9)
    _bus.publish(RecordingStarted())
    _bus.publish(RecordingStopped(reason="silence", endpoint="energy", dead_air_s=2.5))
    _bus.publish(Transcribed(text="", accepted=False, reject_reason="speaker"))
    assert len(got) == 2 and got[1]["outcome"] == "rejected:speaker"
    _bus.publish(SpeakingState(active=True))                     # TTS from elsewhere
    assert len(got) == 2


def test_endpointer_installs_without_a_voiceprint(build, monkeypatch):
    """The first version constructed it inside `if self.speaker.enrolled:`,
    so a box with no voiceprint silently kept the 2.5 s energy timer."""
    import jarvis.endpoint as ep_mod
    from jarvis.config import CONFIG

    class FakeEP:
        def warm(self):
            return True
    monkeypatch.setattr(ep_mod, "VoiceEndpointer", FakeEP)
    monkeypatch.setattr(CONFIG, "endpoint_vad", True)
    app = build()
    monkeypatch.setattr(type(app.speaker), "enrolled", property(lambda self: False), raising=False)
    app.recorder.endpointer = None
    app._install_endpointer()
    assert isinstance(app.recorder.endpointer, FakeEP)
    monkeypatch.setattr(CONFIG, "endpoint_vad", False)
    app.recorder.endpointer = None
    app._install_endpointer()
    assert app.recorder.endpointer is None


def test_a_refused_wake_word_does_not_supersede_the_turn_being_answered(build, monkeypatch):
    """The mark is taken in _on_hotword itself: through the bus it arrived
    after the recorder had opened and read as refused ("wake→mic —")."""
    app = build()
    got = []
    monkeypatch.setattr(app.turns, "_emit", got.append)
    app.recorder.recording = False
    app._on_hotword(0.9)
    assert app.turns.open
    app._turn_busy.set()                              # "still on the last one"
    app._on_hotword(0.9)
    assert got == [], "a refused wake superseded the open turn"
    app._turn_busy.clear()


def test_a_filler_line_does_not_close_the_ledger(build, monkeypatch):
    from jarvis.events import (RecordingStarted, RecordingStopped, SpeakingState,
                               Transcribed, bus as _bus)
    app = build()
    got = []
    monkeypatch.setattr(app.turns, "_emit", got.append)
    app.recorder.recording = False
    monkeypatch.setattr(app, "_process_audio", lambda audio: app._audio_busy.clear())
    app.recorder.recording = False
    app._on_hotword(0.9)
    _bus.publish(RecordingStarted())
    _bus.publish(RecordingStopped(reason="silence", endpoint="vad", dead_air_s=0.8))
    _bus.publish(Transcribed(text="what's the weather", accepted=True))
    app.turns.mark("handle")
    app._turn_filler_pending = True                   # _say_thinking sets this
    _bus.publish(SpeakingState(active=True))          # "Looking into it now, sir."
    assert got == [] and app.turns.open
    _bus.publish(SpeakingState(active=False))
    _bus.publish(SpeakingState(active=True))          # the answer
    assert len(got) == 1 and got[0]["outcome"] == "audio" and got[0]["filler"] is not None


def test_services_brain_has_web_answer_and_an_ack_is_a_filler(build, monkeypatch):
    """The commander only sees the services wrapper (the force_args lesson),
    and an acknowledgement must not close the turn ledger as the answer."""
    from jarvis.commander import CommandResult
    app = build()
    seen = {}
    monkeypatch.setattr(app.brain, "web_answer",
                        lambda q, callback=None, model="haiku": seen.update(q=q, cb=callback, model=model) or "t")
    assert app.services.brain.web_answer("who won", model="sonnet") == "t"
    assert seen["q"] == "who won" and seen["model"] == "sonnet" and seen["cb"] == app._on_brain_tags
    app._turn_filler_pending = False
    monkeypatch.setattr(app, "_say", lambda text: None)
    app._emit_result(CommandResult(handled=True, reply="Looking that up, sir.", speak=True, ack=True))
    assert app._turn_filler_pending is True


def test_an_ack_disarms_the_thinking_filler_but_not_the_watchdog(build, monkeypatch):
    """Live: "Looking that up, sir." was followed 4.5 s later by "Checking
    right now, sir. One moment." -- the ack IS the filler."""
    import threading
    from jarvis.commander import CommandResult
    app = build()
    monkeypatch.setattr(app, "_say", lambda text: None)
    app._turn_timer = threading.Timer(60, lambda: None)
    app._turn_timer.start()
    app._turn_watchdog = threading.Timer(60, lambda: None)
    app._turn_watchdog.start()
    try:
        app._emit_result(CommandResult(handled=True, reply="Looking that up, sir.", speak=True, ack=True))
        assert app._turn_timer is None
        assert app._turn_watchdog is not None and app._turn_watchdog.is_alive()
    finally:
        app._turn_cancel_timers()


def test_the_app_report_is_built_from_real_sources(build, tmp_path, monkeypatch):  # noqa: E501
    """Real sources, fake host: the report reads /proc/meminfo and asks
    nvidia-smi, and both are stubbed here so the test neither depends on
    this box's memory nor spawns a GPU probe under the test runner."""
    import builtins
    import io
    from jarvis.config import PATHS
    from jarvis.tools import health as health_mod
    app = build()
    monkeypatch.setattr(PATHS, "LOG_DIR", tmp_path)
    meminfo = "MemTotal:       127000000 kB\nMemAvailable:    43000000 kB\n"
    real_open = builtins.open

    def fake_open(path, *args, **kw):
        if str(path) == "/proc/meminfo":
            return io.StringIO(meminfo)
        return real_open(path, *args, **kw)
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(health_mod, "read_meminfo",
                        lambda: {"MemTotal": 127000000, "MemAvailable": 43000000})
    monkeypatch.setattr(health_mod, "run_nvidia_smi", lambda: "41, 2418 MHz, 12.4 W, 2 %")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "41\n", ""))
    (tmp_path / "turns.jsonl").write_text(
        '{"outcome": "audio", "wait": 1.3, "at": %f}\n{"outcome": "audio", "wait": 2.1, "at": %f}\n'
        % (__import__("time").time(), __import__("time").time()))
    state = app.self_state()
    assert state["turns_today"] == 2 and round(state["median_wait"], 2) == 1.7
    assert state["gpu_temp_c"] == 41 and state["gpu_mhz"] == 2418
    assert state["gpu_util_pct"] == 2

    text = app.diagnostics_text()
    # "All systems nominal, sir." is gone: brain.VOICE_RULES bans the
    # phrase "all systems", and app.py hardcoded it anyway.
    assert "all systems" not in text.lower()
    assert text.startswith("Everything's where I left it, sir.")
    assert "2 turns today, median wait 1.7 seconds" in text
    assert "Memory 41 of 121 gigabytes free" in text
    # the clock is the wedge tell, not the draw: 611 MHz vs 2400 MHz is a
    # 4x hit while idle power reads ~15 W in both states
    assert "GPU at 41 degrees, 2418 megahertz, 2 percent busy" in text
    assert "power" not in text.lower() and "watt" not in text.lower()


def test_the_app_shows_but_does_not_respeak_a_streamed_reply(build, monkeypatch):
    from jarvis.events import JarvisReply, bus as _bus
    app = build()
    said, shown = [], []
    monkeypatch.setattr(app, "_say", said.append)
    unsub = _bus.subscribe(JarvisReply, lambda ev: shown.append((ev.text, ev.speak)))
    try:
        app._last_source = "voice"
        app._on_stream_sentence("It is ten, sir.")
        app._on_brain_tags([("STREAMED", "1"), ("SPEAK", "It is ten, sir.")])
        assert said == ["It is ten, sir."], said
        assert shown and shown[-1] == ("It is ten, sir.", False)
        assert app._followup_after_speech
    finally:
        try:
            unsub()
        except TypeError:
            pass


def test_services_brain_chat_forwards_the_stream_hook_only_when_enabled(build, monkeypatch):
    """The seams fixture pins stream_replies=False for every wiring test, so
    the wrapper branch that hands _on_stream_sentence to the brain never ran
    under test and the structural signature check subtracts on_sentence."""
    app = build()
    seen = {}
    monkeypatch.setattr(app.brain, "chat", lambda text, **kw: seen.update(kw) or None)
    monkeypatch.setattr(CONFIG, "stream_replies", True)
    app.services.brain.chat("what time is it")
    assert seen.get("on_sentence") == app._on_stream_sentence
    assert seen.get("callback") == app._on_brain_tags
    seen.clear()
    monkeypatch.setattr(CONFIG, "stream_replies", False)
    app.services.brain.chat("what time is it")
    assert "on_sentence" not in seen and seen.get("callback") == app._on_brain_tags


# ------------------------------------------------ deadline heads-up thread
def test_deadline_heads_up_starts_beside_the_meeting_one_and_stops(app, paths, monkeypatch):
    """Same shape as headsup: built in start_assistant, state under
    MEMORY_DIR (never AIWS), stopped by stop_assistant. Without a token the
    thread is silent -- no Canvas call, no state file."""
    import jarvis.tools.canvas as cv
    monkeypatch.setattr(cv, "fetch_due",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Canvas asked")))
    app.start_assistant(residency=False)
    d = app.deadlines
    assert d is not None and d._thread is not None and d._thread.is_alive()
    assert d._state_path == paths / "memory" / "deadlines_state.json"
    assert d.lead_hours == 3.0                                   # canvas.heads_up_hours
    assert d._tk is app.timekeeper
    assert d._get_calendar() is app.services.calendar
    assert d.tick() == 0 and not d._state_path.exists()
    app.stop_assistant()
    d._thread.join(timeout=5)
    assert not d._thread.is_alive()


# ------------------------------------------------------- 11b. watchers
def test_the_three_watchers_start_dark_and_stop(app, paths):
    """grades / mailwatch / keyword watch (jarvis/watchers.py): started beside
    the deadline thread, state under MEMORY_DIR, and on a box with no token,
    no mailbox and no keywords every tick is a no-op that writes nothing."""
    app.start_assistant(residency=False)
    watchers = [(app.gradewatch, "grades_state.json"),
                (app.mailwatch, "mailwatch_state.json"),
                (app.keyword_watch, "keyword_watch_state.json")]
    for w, name in watchers:
        assert w is not None and w._thread is not None and w._thread.is_alive()
        assert w._state_path == paths / "memory" / name
        assert w.tick() == 0 and not w._state_path.exists()
        assert w._announce == app._announce
    assert app.mailwatch._contacts() == [] or app.memory.people()
    app.stop_assistant()
    for w, _name in watchers:
        w._thread.join(timeout=5)
        assert not w._thread.is_alive()


def test_announce_speaks_proactively_and_files_an_alert(app):
    """A watcher line must take BOTH doors: the proactive speech path (so
    quiet hours hold it for the digest) and the alerts hub (so it reaches
    Discord when he is out)."""
    said, alerted = [], []
    app._say = lambda text, proactive=False, kind="message": \
        said.append((text, proactive, kind))
    app._alert = lambda kind, title, text, request_id=None: \
        alerted.append((kind, title, text))
    app._announce("Canvas grade", "A grade posted for BIOSENSORS, sir.")
    assert said == [("A grade posted for BIOSENSORS, sir.", True, "message")]
    assert alerted == [("milestone", "Canvas grade",
                        "A grade posted for BIOSENSORS, sir.")]


# ---------------------------------------------- 12. study sessions & notes
def test_focus_session_is_wired_through_the_real_app(app):
    """"study session biosensors" typed into the real app: the session is on
    services, its timers are SILENT items in the real timekeeper, the spoken
    line comes from jarvis/focus.py (not the timer line), and a silent
    ReminderFired never reaches the alerts hub."""
    assert app.services.focus is app.focus and app.focus is not None
    assert callable(getattr(app.services.docs_index, "kick", None)), \
        "docs.make_tools never parked its index"
    result = app.dispatch_text("study session biosensors")
    assert result.handled and result.speak
    assert app.tts.spoken[-1].startswith("25 minutes on biosensors, sir")
    labels = sorted(i.label for i in app.timekeeper.list("timer"))
    assert labels == ["focus: block 1", "focus: halfway 1"]
    assert app.focus.active and app.focus.label == "biosensors"
    # the alerts hub is not toasted for the session's own items
    alerts = []
    app.alerts.alert = lambda *a, **k: alerts.append(a)
    bus.publish(ReminderFired(text="focus block 1", item_id="x", silent=True))
    bus.publish(ReminderFired(text="call mum", item_id="y"))
    assert [a[2] for a in alerts] == ["call mum"]
    result = app.dispatch_text("how long left")
    assert result.reply.startswith("25 minutes left in block 1") or \
        result.reply.startswith("24 minutes left in block 1")
    result = app.dispatch_text("end the session")
    assert result.reply == "Session over, sir; no full blocks this time."
    assert app.timekeeper.list("timer") == [] and not app.focus.active
    # the state file lives under the (firewalled) memory dir
    assert (PATHS.MEMORY_DIR / "focus_session.json").exists()


def test_lecture_notes_are_wired_through_the_real_app(app, paths, monkeypatch):
    folder = paths / "Jarvis Docs"
    app.assistant.set("docs.paths", [str(folder)])
    result = app.dispatch_text("notes for biosensors")
    assert result.speak and app.commander.lecture_course == "biosensors"
    # source="voice": the mode listens to the microphone only, so that a
    # `jarvis "..."` from a shell is answered rather than filed
    result = app.dispatch_text("impedance is the ratio of voltage to current",
                               source="voice")
    assert result.status.startswith("Noting: biosensors")
    kicks = []
    app.services.docs_index.kick = lambda: kicks.append(1) or True
    result = app.dispatch_text("end notes")
    assert result.reply == "Notes closed, sir: one line for biosensors."
    assert kicks == [1] and app.commander.lecture_course is None
    files = list((folder / "notes").glob("biosensors-*.md"))
    assert len(files) == 1 and "impedance is the ratio" in files[0].read_text()
    assert [n["text"] for n in app.notes.list("note")] == [
        "impedance is the ratio of voltage to current"]


# ------------------------------------------------- #72 the remote duck
def test_the_mixer_can_reach_the_spotify_tool_for_the_remote_duck(app):
    """The Connect duck (#72) was built, tested against a fake remote and
    never wired: RoomMixer.set_remote had no caller, so on the live box the
    mixer still found nothing local and stood down. The mixer is built
    before any tool exists, so the handle has to be passed AFTER
    _register_tools parks the tool on services -- and it must be the same
    object, not a second SpotifyTool with its own token cache."""
    assert app.mixer is not None, "no mixer built; the test proves nothing"
    spotify = getattr(app.services, "spotify", None)
    assert spotify is not None, "spotify.make_tools never parked its tool"
    assert app.mixer._remote is spotify
    assert callable(getattr(app.mixer._remote, "duck", None))
    assert callable(getattr(app.mixer._remote, "unduck", None))


def test_the_wake_gate_learns_about_music_from_the_same_spotify_tool(build):
    """The music-aware wake bar (2026-09-01) is three hand-offs: SpotifyTool
    caches what it knows about playback, the mixer reads that cache, and the
    Hotword is handed a callable that reaches the mixer.  Each piece has a
    unit test against a fake; this one flips the cache on the REAL tool the
    app parked on services and watches the answer arrive at the callable
    the app actually gave Hotword -- because #72 was a duck built, tested,
    and never wired."""
    class RecordingHotword(_Stub):
        made: list = []

        def __init__(self, *a, **kw):
            RecordingHotword.made.append(kw)

    app = build(Hotword=RecordingHotword)
    assert RecordingHotword.made, "the app never built a Hotword"
    kw = RecordingHotword.made[-1]
    music = kw.get("music_playing")
    assert callable(music), "Hotword was not handed a music_playing callable"
    assert kw.get("on_guest") == app._on_guest
    # The stub above swallows anything; the REAL class has to take exactly
    # what the app hands it, or every keyword here is a keyword into the void.
    # (tests/test_hotword_gating.py then drives the real constructor's
    # music_playing through _listen_loop.)
    import inspect as _inspect

    import jarvis.hotword as hw_mod
    takes = set(_inspect.signature(hw_mod.Hotword.__init__).parameters)
    assert set(kw) <= takes, f"Hotword does not accept {set(kw) - takes}"
    spotify = getattr(app.services, "spotify", None)
    assert spotify is not None and app.mixer._remote is spotify
    assert music() is False                      # a fresh box: nothing known
    spotify._note_music(True)                    # what a play command records
    assert music() is True
    assert app._music_playing() is True
    spotify._note_music(False)                   # ...and a pause
    assert music() is False
    # The play above also started the bounded playback poller -- the one
    # piece of this that could reach the network -- and quitting must stop
    # it: a poller outliving the app is a request against a room nobody is
    # in (and, in this suite, a stray thread in a later test).
    t = spotify._poll_thread
    assert t is not None and t.name == "spotify-poll"
    # stop_assistant resolves stop() before close(); the tool must keep
    # having only close(), or a transport stop() would pause his music on
    # every quit.
    assert getattr(spotify, "stop", None) is None
    app.stop_assistant()
    assert not t.is_alive() and spotify._poll_enabled is False


# ------------------------------------------------- the Board and the room
def test_the_board_is_a_real_service_and_composes_from_real_providers(app):
    """No fakes: the app's own providers answer, and the panels that need
    something this box does not have (Canvas, a live session) come back
    honestly dark rather than missing."""
    from jarvis import board as board_mod
    state = app.board_state()
    assert state.keys == board_mod.PANEL_ORDER
    assert app.board_text()                        # renders over SSH
    assert callable(app.services.board.show)


def test_bringing_the_board_up_starts_one_feed_and_publishes_for_the_window(app):
    from jarvis.events import BoardCommand
    seen = []
    bus.subscribe(BoardCommand, seen.append)
    try:
        assert app.services.board.show() is False   # it was not already up
        feed = app._board_feed
        assert feed is not None and feed.running
        assert app.services.board.show() is True    # …and now it is
        assert app._board_feed is feed              # one feed, not two
        app.services.board.hide()
        assert app._board_feed is None and not feed.running
    finally:
        bus.unsubscribe(BoardCommand, seen.append)
    assert [e.action for e in seen] == ["show", "show", "hide"]


def test_quitting_stops_the_board_feed(app):
    app.services.board.show()
    feed = app._board_feed
    app.stop_assistant()
    assert not feed.running


def test_focus_on_a_panel_answers_with_that_panels_own_sentence(app):
    line = app.services.board.read("the vitals")
    assert line.endswith("sir.")
    assert app.services.board.read("the fridge") == ""


def test_live_claude_states_reach_the_boards_sessions_panel(app):
    bus.publish(ClaudeTaskState(project="jarvis", task_id="t1",
                                state="running"))
    assert app._board_tasks == {"jarvis": "running"}
    panel = app.board_state().get("sessions")
    assert ("JARVIS", "RUNNING") in panel.rows
    bus.publish(ClaudeTaskState(project="jarvis", task_id="t1", state="done"))
    assert app._board_tasks == {}


def test_the_canvas_half_is_cached_rather_than_polled_every_five_seconds(
        app, monkeypatch):
    calls = []

    def fake_due(registry):
        calls.append(1)
        return ["BIOSEN - Lab 3 report, tonight"]
    monkeypatch.setattr("jarvis.tools.briefing._due_lines", fake_due)
    assert app._board_canvas_lines() == ["BIOSEN - Lab 3 report, tonight"]
    app._board_canvas_lines()
    app._board_canvas_lines()
    assert calls == [1]                    # one REST call, not three


def test_the_room_state_never_claims_home_when_presence_is_unconfigured(app):
    """presence.is_home() answers True with no phone_ip set, so the slab
    must show nothing rather than a confident, false HOME."""
    room = app.room_state()
    assert room["presence"] == ""
    assert set(room) >= {"playing", "next", "due", "temp", "arc", "quiet",
                         "gpu"}


def test_the_room_state_shows_where_he_is_once_presence_is_configured(app):
    """The complement of the test above: with a phone_ip set and a real
    answer, the slab SAYS it. "unknown" (the state before the first poll
    answers) is still suppressed -- a WHERE row reading "unknown" is worth
    no more than a false HOME."""
    from types import SimpleNamespace
    for state, shown in (("home", "home"), ("away", "away"), ("unknown", "")):
        app.presence = SimpleNamespace(configured=True, state=state)
        assert app.room_state()["presence"] == shown, state


def test_the_next_row_names_the_day_it_is_talking_about(app):
    """Found from the console 2026-08-31: standby read "BIOSENSORS 12:45 pm"
    at eight in the evening, which says nothing about WHICH 12:45. Today
    keeps no day word on purpose -- the standby face is a clock."""
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    now = datetime.now().astimezone()

    def cal(at):
        ev = SimpleNamespace(title="BIOSENSORS", start=at, all_day=False)
        return SimpleNamespace(events=lambda: [ev])

    app.services.calendar = cal(now + timedelta(hours=2))
    assert " today " not in app._room_next_event()

    app.services.calendar = cal((now + timedelta(days=1)).replace(hour=12, minute=45))
    row = app._room_next_event()
    assert "tomorrow" in row and "12:45 pm" in row and row.startswith("BIOSENSORS")

    later = now + timedelta(days=3)
    app.services.calendar = cal(later.replace(hour=12, minute=45))
    assert later.strftime("%A") in app._room_next_event()


def test_the_slab_reads_in_upper_case(app):
    """His request, 2026-08-31: "i want all of the things to be
    capatalized, like HOME". The labels always were; the values were not,
    so the panel read as a caption with a sentence after it."""
    from jarvis.ui.ambient import room_rows, standby_rows
    room = {"presence": "home", "next": "BIOSENSORS tomorrow 12:45 pm",
            "due": "Lab 3 report", "temp": "84\u00b0F", "playing": "Blinding Lights",
            "arc": "evening"}
    for rows in (room_rows(room), standby_rows(room)):
        assert rows, "no rows rendered"
        for label, value in rows:
            assert value == value.upper(), f"{label} value not upper: {value!r}"
    assert ("WHERE", "HOME") in standby_rows(room)
    assert ("WHERE", "AWAY") in standby_rows({"presence": "away"})
    # an absent value still disappears rather than becoming ""
    assert [label for label, _ in standby_rows({"presence": ""})] == []


def test_standby_shows_where_he_is(app):
    """STANDBY_KEYS had NEXT/DUE/OUTSIDE but not WHERE, so the panel shown
    when nobody is at the desk was the one panel that could not answer
    "is he home"."""
    from jarvis.ui.ambient import AMBIENT_KEYS, STANDBY_KEYS
    assert ("presence", "WHERE") in STANDBY_KEYS
    assert ("presence", "WHERE") in AMBIENT_KEYS
    # standby still drops the two it means to drop
    keys = {k for k, _ in STANDBY_KEYS}
    assert "playing" not in keys and "arc" not in keys


def test_the_room_state_backs_off_spotify_instead_of_a_heartbeat(app):
    calls = []

    class FakeSpotify:
        def now_playing(self):
            calls.append(1)
            return SimpleNamespace(speak="Kind of Blue, sir.", text="")
    app.services.spotify = FakeSpotify()
    assert app._room_playing() == "Kind of Blue, sir."
    assert app._room_playing() == "Kind of Blue, sir."
    assert calls == [1]                    # the second read came from cache


def test_the_power_up_sweep_fires_once_a_day_off_the_briefing_latch(app):
    from jarvis.events import PowerUp
    seen = []
    bus.subscribe(PowerUp, seen.append)
    try:
        assert app._maybe_power_up("hotword") is True
        assert app._maybe_power_up("presence") is False
    finally:
        bus.unsubscribe(PowerUp, seen.append)
    assert len(seen) == 1 and seen[0].reason == "hotword"
    state = json.loads(app._briefing_state_path().read_text())
    assert state["boot_sweep"] == time.strftime("%Y-%m-%d")


def test_the_sweep_latch_does_not_eat_the_briefings_own_key(app):
    app._mark_briefing_delivered()
    app._maybe_power_up("hotword")
    state = json.loads(app._briefing_state_path().read_text())
    assert "delivered" in state and "boot_sweep" in state


def test_the_sweep_can_be_switched_off_in_the_config(app):
    app.assistant.set("console.powerup", False)
    assert app._maybe_power_up("hotword") is False


# ------------------------------- 20. desk presence, the walks, the watch
#
# The three modules have their own unit tests (test_deskpresence,
# test_leavetime, test_calwatch). What only the real app can prove is the
# WIRING: that the two away probes meet in one signal and one greeting,
# that the board's dimming is nothing but this app's own state, and that
# the leave question is never asked into a moment where nobody could
# answer it.
def _quiet_open(app):
    """Silence the clock: quiet hours would otherwise defer the greeting
    depending on what time the suite is run."""
    app.quiet.reason = lambda *a, **kw: ""
    app.quiet.release = lambda *a, **kw: ""


def test_the_two_away_probes_meet_in_the_one_quiet_seam(app):
    from jarvis.deskpresence import DeskSentinel
    from jarvis.presence import PresenceSentinel
    assert isinstance(app.desk, DeskSentinel)
    assert isinstance(app.presence, PresenceSentinel)
    # ONE suppression path: quiet.py's existing hold_when_away gate.
    assert app.quiet._is_home == app._is_home


def test_an_unconfigured_phone_and_a_session_without_mutter_read_as_home(app):
    """Both probes fail OPEN. The phone has no address on this box and the
    test suite has no idle monitor, so the app must behave exactly as it
    did before either existed -- never "he's out"."""
    assert app.desk.at_desk is None and not app.presence.configured
    assert app._is_home() is True


def test_an_established_empty_chair_is_the_away_signal(app):
    app.desk.at_desk = False
    assert app._is_home() is False
    app.desk.at_desk = True
    assert app._is_home() is True


def test_one_walk_through_the_door_is_one_greeting(app):
    """The phone crosses its threshold minutes after the keyboard does.
    That is ONE return: release() drains atomically so the digest could not
    double, but the greeting line would."""
    from jarvis.events import DeskState, Presence
    from jarvis.presence import WELCOME_LINE
    _quiet_open(app)
    bus.publish(DeskState(at_desk=True, idle_s=1.0, returned=True))
    bus.publish(Presence(home=True, returned=True))
    bus.drain()
    assert app.tts.spoken.count(WELCOME_LINE) == 1


def test_the_desk_probe_defers_to_a_configured_phone(app):
    """Leaving the building is the return worth marking; sitting back down
    after a coffee is not. With the phone configured it owns the line."""
    from jarvis.events import DeskState
    from jarvis.presence import WELCOME_LINE
    _quiet_open(app)
    app.assistant.set("presence.phone_ip", "192.168.1.42")
    assert app.presence.configured
    bus.publish(DeskState(at_desk=True, idle_s=1.0, returned=True))
    bus.drain()
    assert WELCOME_LINE not in app.tts.spoken


def test_walking_away_is_never_announced(app):
    from jarvis.events import DeskState
    _quiet_open(app)
    bus.publish(DeskState(at_desk=False, idle_s=1800.0))
    bus.drain()
    assert app.tts.spoken == []


def test_services_hand_out_a_live_desk_reading_not_a_stale_number(app):
    assert callable(app.services.desk_idle_s)
    assert app.services.desk_idle_s() is None       # no reading in the suite
    app.desk.last_idle = 42.0
    assert app.services.desk_idle_s() == 42.0


def test_the_walk_question_waits_for_a_moment_he_could_answer_it(app):
    """A held question with an armed pending answer is a trap: he never
    hears it and the next duration he says is filed as a walk."""
    _quiet_open(app)
    app.recorder.recording = False        # the stub answers callables, not bools
    app.quiet.should_hold = lambda *a, **kw: True
    assert app._ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker") is False
    assert app.commander._pending_leave is None
    app.quiet.should_hold = lambda *a, **kw: False
    app._turn_busy.set()
    try:
        assert app._ask_leave_time("Wisenbaker Engineering Bldg",
                                   "Wisenbaker") is False
    finally:
        app._turn_busy.clear()
    assert app.commander._pending_leave is None


def test_the_walk_question_arms_the_answer_when_it_is_actually_asked(app):
    from jarvis import leavetime as lt_mod
    _quiet_open(app)
    app.quiet.should_hold = lambda *a, **kw: False
    app.recorder.recording = False        # the stub answers callables, not bools
    assert app._ask_leave_time("Wisenbaker Engineering Bldg", "Wisenbaker") is True
    assert app.tts.spoken[-1] == lt_mod.ASK_LINE.format(place="Wisenbaker")
    assert app.commander._pending_leave[0] == "Wisenbaker Engineering Bldg"


def test_the_two_calendar_watches_are_started_and_joined(app):
    """Both file their state under the (firewalled) memory dir and both
    are on the stop list, so their threads are joined at quit."""
    app.start_assistant(residency=False)
    assert app.calwatch._state_path == PATHS.MEMORY_DIR / "calwatch_state.json"
    assert app.leavetime._state_path == PATHS.MEMORY_DIR / "leavetime_state.json"
    assert app.calwatch._thread is not None and app.calwatch._thread.is_alive()
    app.stop_assistant()
    assert not app.calwatch._thread.is_alive()
    assert app.leavetime._thread is None or not app.leavetime._thread.is_alive()
    assert not app.desk.running


# ------------------- 21. the greeter, the desk seam, and the 5 s probes
#
# Regression sites, all four found by review after the arrival/desk merge:
# the shared greeter was deleted while a caller still named it, the
# desk-idle seam was read off the wrong object in two places, and both 5 s
# providers paid for a network / nvidia-smi round trip they already had.
def test_a_desk_return_is_actually_greeted(app):
    """`_on_desk` called `self._greet_return`, which the arrival rework had
    deleted: every desk return raised AttributeError inside the bus
    subscriber (swallowed there), so the room stayed silent."""
    from jarvis.events import DeskState
    from jarvis.presence import WELCOME_LINE
    _quiet_open(app)
    bus.publish(DeskState(at_desk=True, idle_s=1.0, returned=True))
    bus.drain()
    assert app.tts.spoken.count(WELCOME_LINE) == 1


def test_the_phone_path_honours_the_greeting_damper_too(app):
    """The damper is shared or it is nothing: `_on_presence` greeted
    without consulting `_last_greeted`, so two returned events inside
    GREET_DAMPER_S said "Welcome back, sir" twice."""
    from jarvis.events import Presence
    from jarvis.presence import WELCOME_LINE
    _quiet_open(app)
    bus.publish(Presence(home=True, returned=True))
    bus.drain()
    bus.publish(Presence(home=True, returned=True))
    bus.drain()
    assert app.tts.spoken.count(WELCOME_LINE) == 1
    assert app._last_greeted > 0.0          # the damper was actually stamped


def test_a_return_after_the_damper_is_greeted_again(app):
    """The damper suppresses the SECOND probe on one walk through the
    door, not the next time he comes home."""
    from jarvis.events import DeskState
    from jarvis.presence import WELCOME_LINE
    _quiet_open(app)
    bus.publish(DeskState(at_desk=True, idle_s=1.0, returned=True))
    bus.drain()
    app._last_greeted = time.monotonic() - app_mod.GREET_DAMPER_S - 1.0
    bus.publish(DeskState(at_desk=True, idle_s=1.0, returned=True))
    bus.drain()
    assert app.tts.spoken.count(WELCOME_LINE) == 2


def test_the_power_up_sweep_applies_the_overnight_gap_gate(app):
    """`_maybe_power_up` probed `getattr(self, "desk_idle_s")` — a name the
    app has never had — so `idle` was always None and the "left alone for
    gap_h hours" gate was skipped on every box."""
    from jarvis.events import PowerUp
    seen = []
    bus.subscribe(PowerUp, seen.append)
    try:
        app.desk.last_idle = 600.0             # ten minutes: not a night
        assert app._maybe_power_up("hotword") is False
        assert seen == []
        app.desk.last_idle = 8 * 3600.0        # a night
        assert app._maybe_power_up("hotword") is True
    finally:
        bus.unsubscribe(PowerUp, seen.append)
    assert len(seen) == 1 and round(seen[0].gap_h) == 8


def test_the_console_is_handed_the_live_desk_probe(app, monkeypatch):
    """`ui_service_kwargs` passed `getattr(self, "desk_idle_s", None)` —
    always None — so console_mode.resolve_idle_fn fell through to its
    XScreenSaver probe and the console's idle clock diverged from the
    app's."""
    monkeypatch.delenv("JARVIS_DESK_PRESENCE", raising=False)
    assert app.desk.enabled
    fn = app.ui_service_kwargs()["desk_idle_s"]
    assert callable(fn)
    app.desk.last_idle = 42.0
    assert fn() == 42.0


def test_a_switched_off_sentinel_leaves_the_console_its_own_probe(app):
    """A disabled sentinel answers None forever and resolve_idle_fn takes
    ANY callable: handing it over would kill the console's fallback, so the
    seam is None (conftest forces JARVIS_DESK_PRESENCE=0 for the suite)."""
    assert not app.desk.enabled
    assert app.ui_service_kwargs()["desk_idle_s"] is None


def test_an_empty_canvas_answer_is_cached_like_any_other(app, monkeypatch):
    """The TTL tested the payload, not the timestamp, so "nothing due" —
    which has already paid for the REST call — re-fetched on every 5 s
    Board tick, ~720 round trips an hour."""
    calls = []

    def fake_due(registry):
        calls.append(1)
        return []
    monkeypatch.setattr("jarvis.tools.briefing._due_lines", fake_due)
    assert app._board_canvas_lines() == []
    app._board_canvas_lines()
    app._board_canvas_lines()
    assert calls == [1]


def test_the_slab_takes_the_gpu_reading_the_probe_already_has(app, monkeypatch):
    """room_state() ran health.snapshot(gpu=True) — a second nvidia-smi
    plus two /proc walks — on the very pass that had just forked
    nvidia-smi for the temps row."""
    from jarvis.tools import health
    snaps, smis = [], []
    monkeypatch.setattr(health, "snapshot",
                        lambda *a, **kw: snaps.append(1))
    monkeypatch.setattr(health, "run_nvidia_smi",
                        lambda *a, **kw: (smis.append(1), "40, 2400, 90, 55")[1])
    assert app.room_state(gpu_pct=42)["gpu"] == pytest.approx(0.42)
    assert snaps == [] and smis == []        # the caller's number, no fork
    assert app.room_state()["gpu"] == pytest.approx(0.55)
    app.room_state()
    assert snaps == [] and smis == [1]       # cached, and never the snapshot


def test_a_quiet_hours_panel_does_not_arm_the_greeting_damper(app):
    """A deferred arrival is panel-only (arrival.arrival_plan): nothing was
    said, so the damper must not swallow the greeting the policy lifts into
    minutes later."""
    from jarvis.events import Presence
    from jarvis.presence import WELCOME_LINE
    app.quiet.reason = lambda *a, **kw: "quiet hours until 7:00 am"
    app.quiet.release = lambda *a, **kw: ""
    bus.publish(Presence(home=True, returned=True))
    bus.drain()
    assert WELCOME_LINE not in app.tts.spoken
    assert app._last_greeted == 0.0
    _quiet_open(app)
    bus.publish(Presence(home=True, returned=True))
    bus.drain()
    assert app.tts.spoken.count(WELCOME_LINE) == 1


# ------------------- 22. the cross-lane stitches (review round 3)
#
# Every one of these is a seam where two waves' features met: a predicate
# that grew on one side of the wall and a caller that never learned about
# it, a heal that overrides a guard made one screenful earlier, a Timer
# that outlives the room. They are integration bugs by construction, so
# they are tested against the real app, not a slice of it.
def test_the_mic_window_asks_the_commander_what_is_open(app):
    """_question_open grew its own half-list of pending questions while the
    commander grew the authoritative one, so the study offer and the
    objection -- both spoken yes/no questions -- got the 4 s follow-up
    window instead of the 15 s answer window."""
    assert app._question_open(app.commander) is False
    app.services.study_offer = {"made_at": time.time(), "n": 10}
    try:
        assert app.commander.question_open() is True
        assert app._question_open(app.commander) is True
    finally:
        app.services.study_offer = None
    assert app._question_open(app.commander) is False


def test_the_walk_question_is_not_put_over_another_open_question(app):
    """leavetime._maybe_ask burns its once-ever ask the instant this
    returns True, and _try_leave_answer is the LAST pending rung -- so a
    duration said while another question is open is eaten by that rung and
    the walk is never learned. Say nothing; the watch retries."""
    _quiet_open(app)
    app.quiet.should_hold = lambda *a, **kw: False
    app.recorder.recording = False        # the stub answers callables, not bools
    app.services.study_offer = {"made_at": time.time(), "n": 10}
    said = len(app.tts.spoken)
    try:
        assert app._ask_leave_time("Wisenbaker Engineering Bldg",
                                   "Wisenbaker") is False
    finally:
        app.services.study_offer = None
    assert app.commander._pending_leave is None
    assert len(app.tts.spoken) == said, "a refused arm must not speak"


def _ringing(app, monkeypatch):
    """Make the real timekeeper answer "an alarm is ringing"."""
    monkeypatch.setattr(type(app.timekeeper), "ringing",
                        property(lambda self: SimpleNamespace(id=1)))


def test_the_debrief_is_not_asked_over_a_floor_someone_else_owns(app,
                                                                 monkeypatch):
    """The debrief is the ONE pending stage hoisted above commander.handle,
    so arming it over a ringing alarm or a live flashcard files THEIR
    answer into the episodic record. The withheld question must also not
    spend the once-ever ask."""
    from jarvis.debrief import Candidate
    from datetime import datetime, timedelta
    _quiet_open(app)
    app.recorder.recording = False
    # FakeTTS answers every unknown attribute with a (truthy) callable, so
    # the busy-speech gate above would otherwise short-circuit this test.
    app.tts.is_speaking, app.tts.pending = False, 0
    marked = []
    app.debrief = SimpleNamespace(mark_asked=marked.append)
    cand = Candidate(key="k", title="the midterm", word="exam",
                     end=datetime.now() - timedelta(minutes=30))
    _ringing(app, monkeypatch)
    app._ask_debrief(cand)
    assert app._pending_debrief is None
    assert marked == [], "a question that was never put must not be marked asked"


def test_an_open_debrief_stands_down_for_a_holder_that_appears_after_it(
        app, monkeypatch):
    """A bare "stop" to a ringing alarm inside the 120 s window was filed as
    how the midterm went: commander._try_ringing sits BELOW this filter and
    "stop" is in no ASSISTANT_TIER1 matcher. The question must survive the
    stand-down -- it is still answerable once the floor is free."""
    from jarvis.debrief import Candidate
    from datetime import datetime, timedelta
    cand = Candidate(key="k", title="the midterm", word="exam",
                     end=datetime.now() - timedelta(minutes=30))
    app._pending_debrief = {"key": "k", "cand": cand, "at": time.monotonic()}
    _ringing(app, monkeypatch)
    assert app._debrief_reply("stop", "voice") is None
    assert app._pending_debrief is not None, "the question is not spent"


def test_the_room_light_asks_the_wind_down_before_an_automatic_heal(app):
    """winddown.py and room.py drive the SAME xrandr output through two
    separate state files. The boot and quit heals used to override a guard
    winddown.restore(expired_only=True) had just honoured."""
    probe = app.room_light._held_by
    assert callable(probe)
    assert probe() is False and app.room_light.held_elsewhere() is False
    app.winddown = SimpleNamespace(holding=True)
    assert probe() is True and app.room_light.held_elsewhere() is True


def test_the_boot_and_quit_heals_are_both_marked_healing(app):
    """Both automatic calls, and only those: "lights up" passes healing
    False and always wins. WindDown.stop() does not brighten the room
    either, so a quit at midnight that did would be the same floodlight."""
    calls = []
    app.room_light = SimpleNamespace(
        changed=True,
        restore=lambda healing=False: (calls.append(healing), (True, ""))[1])
    app.start_assistant(residency=False)
    app.stop_assistant()
    assert calls == [True, True]


def test_a_dissent_timer_does_not_outlive_the_room(app):
    """The 60 s objection Timer RUNS the deferred action and speaks it, so
    it must be cancelled beside the departure Timer stop_assistant already
    cancels -- and the slot cleared, for a Timer already past cancel()."""
    from jarvis.objections import Objection
    obj = Objection(source="sleep window", reason="that is in four hours",
                    row="alarm 7am", key="alarm-7am")
    app.commander.stash_objection(lambda: None, "Shall I set it anyway?", obj)
    timer = app.commander._objection_timer
    assert timer is not None and timer.is_alive()
    app.stop_assistant()
    assert app.commander._objection_timer is None
    assert app.commander._pending_objection is None
    assert app.commander.objection_timeout() is None
    assert not timer.is_alive()


def test_ui_service_kwargs_carry_the_board_wm_close_hook(app):
    """MainWindow._board_closed falls back to services.board.hide and,
    failing that, only logs -- so without this kwarg the Board's WM close
    left its 5 s feed polling a window nobody can see."""
    assert app.ui_service_kwargs()["board_closed"] == app._board_hide


def test_ui_service_kwargs_hand_over_a_live_desk_reading(app):
    """services.desk_idle_s, not a name that never existed on self."""
    fn = app.ui_service_kwargs()["desk_idle_s"]
    assert callable(fn) is app.desk.enabled


def test_the_open_debrief_gets_the_long_answer_window(app):
    """"How did the midterm go, sir?" is answered in a sentence, so the 4 s
    follow-up window cut him off mid-answer. Checked in _capture_window and
    NOT in _question_open: debrief.floor_holder calls that predicate, so the
    debrief would name itself as the holder and stand down from its own
    answer for the whole 120 s."""
    from jarvis.debrief import Candidate
    from datetime import datetime, timedelta
    assert app._capture_window() is None
    cand = Candidate(key="k", title="the midterm", word="exam",
                     end=datetime.now() - timedelta(minutes=30))
    app._pending_debrief = {"key": "k", "cand": cand, "at": time.monotonic()}
    assert app._capture_window() == 15.0
    # ...and the question is still answerable: it never holds its own floor.
    assert app._debrief_reply("it went fine", "voice") is not None
    app._pending_debrief = {"key": "k", "cand": cand,
                            "at": time.monotonic() - app.DEBRIEF_TTL_S - 1}
    assert app._capture_window() is None


# ------------------------------------------ the config's ONE write
def test_startup_fills_new_default_keys_once_and_hands_the_config_to_the_brain(
        build, paths, monkeypatch):
    """AssistantConfig.load() is a READ since 2026-09-04 (an agent's import
    of jarvis.brain had rewritten his live, secret-bearing file through
    it -- tests/test_config_readonly.py). The app is the process that owns
    the file, so JarvisApp() alone fills in keys DEFAULTS has gained and
    tightens a loose mode, once, right after its load -- and hands that
    same loaded config to brain.configure, so the brain reads no config
    of its own."""
    import stat
    from jarvis.assistant_config import DEFAULTS
    cfg = paths / "assistant.json"
    cfg.write_text(json.dumps({"version": 1, "user": {"name": "T"}}) + "\n")
    os.chmod(cfg, 0o644)
    seen = []
    real = app_mod.brain_mod.configure

    def spy(model=None, config=None):
        seen.append(config)
        return real(model, config=config)
    monkeypatch.setattr(app_mod.brain_mod, "configure", spy)
    a = build()
    on_disk = json.loads(cfg.read_text())
    assert on_disk["user"]["name"] == "T"                   # his value kept
    assert on_disk["brain"] == DEFAULTS["brain"]            # new keys written
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600        # and tightened
    assert seen == [a.assistant]                            # the same object, once
    assert a.assistant.disk_state == {"missing": False, "corrupt": False,
                                      "loose_mode": False, "new_keys": False}


# ------------------------------------------------ the Restart button
def test_the_running_commit_is_stamped_once_at_construction(build, monkeypatch):
    """Read at startup, never at import: the drawer compares it with what
    is on disk NOW, so the stamp must be the tree the process actually
    started from."""
    monkeypatch.setattr(app_mod, "running_commit", lambda **kw: "abc1234")
    a = build()
    assert a.running_commit == "abc1234"
    monkeypatch.setattr(app_mod, "running_commit", lambda **kw: "changed")
    assert a.running_commit == "abc1234"


def test_code_status_compares_the_stamp_with_the_repo_now(app, monkeypatch):
    seen = []

    def probe(running, repo, git=None):
        seen.append((running, Path(repo)))
        return {"running": running, "disk": "d15c000", "behind": 1,
                "dirty": False}
    monkeypatch.setattr(app_mod, "probe_code_status", probe)
    app.running_commit = "ru77777"
    assert app.code_status()["disk"] == "d15c000"
    assert seen == [("ru77777", Path(app_mod.relaunch.REPO_ROOT))]


def test_ui_service_kwargs_carry_restart_and_code_status(app):
    kwargs = app.ui_service_kwargs()
    assert kwargs["restart"] == app.restart
    assert kwargs["code_status"] == app.code_status


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_restart_says_the_line_spawns_the_helper_with_this_pid_then_quits(
        app, paths):
    """Order is the whole feature: helper first (with OUR pid to wait on),
    quit second. A quit that ran first would leave nobody to relaunch."""
    order = []
    spawned = []

    def spawn(old_pid, cmd, cwd, env, log_path, grace_s=20.0):
        spawned.append(dict(old_pid=old_pid, cmd=cmd, cwd=cwd, env=env,
                            log_path=Path(log_path)))
        order.append("spawn")
        return 999
    app.close_window = lambda: order.append("close")
    clock = _Clock()
    helper = app.restart(spawn=spawn, sleep=clock.sleep, clock=clock)
    assert helper == 999
    assert order == ["spawn", "close"]
    assert app.tts.spoken[-1] == "Back in a moment, sir."
    s = spawned[0]
    assert s["old_pid"] == os.getpid()
    assert s["cmd"] == [sys.executable, "-m", "jarvis.app"]
    assert Path(s["cwd"]) == REPO_ROOT
    assert s["log_path"] == PATHS.LOG_DIR / "relaunch.log"
    assert s["env"].get("PATH") == os.environ.get("PATH")


def test_restart_waits_for_the_spoken_line_but_never_past_the_ceiling(
        app, paths):
    clock = _Clock()
    spawn_at = []

    def spawn(*a, **kw):
        spawn_at.append(clock.t)
        return 1
    app.close_window = lambda: None
    # still talking for 2 s (the same TTS.is_speaking the thinking filler
    # reads through _tts_busy): the helper starts once the line is out
    app.tts.is_speaking = True

    def sleep(s):
        clock.sleep(s)
        if clock.t >= 2.0:
            app.tts.is_speaking = False
    app.restart(spawn=spawn, sleep=sleep, clock=clock)
    assert 2.0 <= spawn_at[0] < 2.5
    # a line still QUEUED counts too, and a TTS that never reports idle is
    # bounded at RESTART_SAY_WAIT_S
    app._restarting = False
    app.tts.pending = 1
    clock.t = 0.0
    app.restart(spawn=spawn, sleep=clock.sleep, clock=clock)
    assert app_mod.RESTART_SAY_WAIT_S <= spawn_at[1] < \
        app_mod.RESTART_SAY_WAIT_S + 0.5


def test_restart_that_cannot_start_the_helper_does_not_quit(app, paths):
    """No helper means nobody to bring him back: stay up, say so."""
    closed = []
    app.close_window = lambda: closed.append(1)
    quits = []
    app.quit = lambda: quits.append(1)
    sink = Sink(Status)
    try:
        def spawn(*a, **kw):
            raise OSError("no interpreter")
        clock = _Clock()
        assert app.restart(spawn=spawn, sleep=clock.sleep, clock=clock) is None
        assert closed == [] and quits == []
        st = sink.wait(Status, timeout=2.0)
        assert st is not None and "estart" in st.text and st.kind == "error"
        # ...and a later press is allowed to try again
        assert app._restarting is False
    finally:
        sink.close()


def test_restart_falls_back_to_quit_when_no_window_is_attached(app, paths):
    quits = []
    app.quit = lambda: quits.append(1)
    clock = _Clock()
    app.restart(spawn=lambda *a, **kw: 5, sleep=clock.sleep, clock=clock)
    assert quits == [1]


def test_restart_never_spawns_twice_while_one_is_in_flight(app, paths):
    spawns = []
    app.close_window = lambda: None
    clock = _Clock()
    app.restart(spawn=lambda *a, **kw: spawns.append(1) or 7,
                sleep=clock.sleep, clock=clock)
    app.restart(spawn=lambda *a, **kw: spawns.append(2) or 8,
                sleep=clock.sleep, clock=clock)
    assert spawns == [1]


def test_attach_window_routes_the_close_through_tk_after(app):
    """The drawer's press runs restart() on a worker thread; the window's
    _on_close (geometry save, tray, services.quit, root.destroy) must run
    on the Tk thread, so the hook is an after(0, ...) marshal."""
    afters = []

    class _Root:
        def after(self, ms, fn):
            afters.append((ms, fn))

    class _Window:
        root = _Root()

        def _on_close(self):
            pass
    w = _Window()
    app_mod.attach_window(app, w)
    app.close_window()
    assert afters == [(0, w._on_close)]


def test_main_attaches_the_window_it_creates():
    import inspect
    src = inspect.getsource(app_mod.main)
    assert "attach_window(app, window)" in src
    assert src.index("window = create(") < src.index("attach_window(app, window)")


def test_restart_still_restarts_when_the_spoken_line_raises(app, paths):
    """Review 2026-09-04: a TTS error inside restart() left _restarting set
    forever, so every later press said 'already in flight' and the button
    was dead until a manual restart. The line is a courtesy; the restart
    he pressed for still happens."""
    order = []

    def boom(*a, **k):
        raise RuntimeError("tts down")
    app._say = boom
    app.close_window = lambda: order.append("close")
    clock = _Clock()
    helper = app.restart(spawn=lambda *a, **k: (order.append("spawn"), 7)[1],
                         sleep=clock.sleep, clock=clock)
    assert helper == 7
    assert order == ["spawn", "close"]
