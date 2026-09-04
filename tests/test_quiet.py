"""Quiet hours / do-not-disturb / the catch-up digest (jarvis/quiet.py) and
its wiring: the app's proactive flag on _say, the timekeeper's alarm-vs-
reminder tag, the desktop-banner gate, the commander's Tier-1 phrases and
the first-wake briefing waiting for the window to close.

Firewall: a fake config (no file), fake clocks, no threads except the one
start/stop test (which joins), no TTS -- every spoken line lands in a list.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis.channels import notify as notify_mod
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.events import Presence, Status, bus
from jarvis.quiet import (AWAY_PREFIX, BUSY_PREFIX, DND_ALREADY_FREE_LINE,
                          DND_SET_LINE, FOCUS_REASON, FREE_LINE, NOTHING_HELD_LINE,
                          QUIET_HOURS_OFF_LINE, QUIET_HOURS_SET_LINE,
                          QUIET_STATUS_FREE_LINE, QuietPolicy, digest,
                          parse_clock)
from jarvis.tools.timekeeper import Timekeeper
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)


# ------------------------------------------------------------------ fakes
class FakeCfg:
    """assistant_config's get/set on a dict; `saved` counts writes."""

    def __init__(self, data=None):
        self.data = dict(data or {})
        self.saved = 0

    def get(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, key, value):
        parts = key.split(".")
        node = self.data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
        self.saved += 1
        return True


class Clock:
    def __init__(self, dt: datetime):
        self.dt = dt

    def now(self) -> float:
        return self.dt.timestamp()

    def tick(self, **kw):
        self.dt += timedelta(**kw)


def _ev(title, start: datetime, hours=1.0, all_day=False):
    # The offset in force AT `start`, not today's: Clock.now() resolves the
    # naive NOON with August's own offset (CDT), so stamping the event with a
    # November tzinfo (CST) put it an hour off and reddened four tests here for
    # the whole of standard time. astimezone() on a naive value keeps the wall
    # clock and picks the right offset for that instant, which is what
    # self.dt.timestamp() above already does.
    s = start.astimezone() if start.tzinfo is None else start
    return SimpleNamespace(title=title, start=s, end=s + timedelta(hours=hours),
                           all_day=all_day)


class FakeCal:
    configured = True

    def __init__(self, *events):
        self._events = list(events)

    def events(self):
        return list(self._events)


NOON = datetime(2026, 8, 31, 12, 0)          # a Monday


def _policy(cfg=None, clock=None, **kw):
    clock = clock or Clock(NOON)
    p = QuietPolicy(cfg if cfg is not None else FakeCfg(), now=clock.now, **kw)
    p.clock = clock
    return p


# ------------------------------------------------------------ parse_clock
@pytest.mark.parametrize("text,default,expected", [
    ("eleven", "pm", (23, 0)),
    ("seven", "am", (7, 0)),
    ("11 pm", "", (23, 0)),
    ("7 a.m.", "", (7, 0)),
    ("seven thirty", "pm", (19, 30)),
    ("7:30 am", "pm", (7, 30)),
    ("half past ten", "pm", (22, 30)),
    ("midnight", "", (0, 0)),
    ("noon", "", (12, 0)),
    ("twelve", "pm", (0, 0)),             # "from twelve" = midnight
    ("twelve", "am", (0, 0)),
    ("23:00", "", (23, 0)),
    ("07:00", "pm", (7, 0)),              # a leading zero is 24 h, not "7 pm"
    ("ten in the morning", "pm", (10, 0)),
    ("nine at night", "am", (21, 0)),
    ("twenty five past", "", None),
    ("an hour", "", None),
    ("", "", None),
])
def test_parse_clock(text, default, expected):
    assert parse_clock(text, default=default) == expected


# ------------------------------------------------------------- reasons
def test_not_quiet_by_default():
    p = _policy()
    assert p.reason() == "" and not p.is_quiet() and not p.should_hold()


def test_dnd_holds_until_the_timestamp_and_persists_in_the_config():
    cfg = FakeCfg()
    p = _policy(cfg)
    until = p.set_dnd(3600)
    assert cfg.get("quiet.dnd_until") == until and cfg.saved >= 1
    assert p.reason() == "do not disturb until 1 pm"
    p.clock.tick(minutes=59)
    assert p.is_quiet()
    p.clock.tick(minutes=2)
    assert not p.is_quiet()


def test_dnd_minimum_is_a_minute():
    p = _policy()
    until = p.set_dnd(5)
    assert until - p.clock.now() == pytest.approx(60)


def test_quiet_hours_wrap_midnight():
    cfg = FakeCfg({"quiet": {"hours": {"start": "23:00", "end": "07:00"}}})
    p = _policy(cfg, Clock(datetime(2026, 8, 31, 22, 59)))
    assert not p.is_quiet()
    p.clock.tick(minutes=1)
    assert p.reason() == "quiet hours until 7 am"
    p.clock.tick(hours=7)                   # 06:00
    assert p.is_quiet()
    p.clock.tick(hours=1)                   # 07:00 exactly: the window is [start, end)
    assert not p.is_quiet()


def test_quiet_hours_same_day_window_and_malformed_config():
    p = _policy(FakeCfg({"quiet": {"hours": {"start": "13:00", "end": "14:00"}}}),
                Clock(datetime(2026, 8, 31, 13, 30)))
    assert p.is_quiet()
    p2 = _policy(FakeCfg({"quiet": {"hours": {"start": "nope", "end": "07:00"}}}))
    assert p2.quiet_hours() is None and not p2.is_quiet()
    p3 = _policy(FakeCfg({"quiet": {"hours": "23-7"}}))
    assert p3.quiet_hours() is None


def test_a_running_class_on_the_calendar_is_quiet_but_lunch_is_not():
    cal = FakeCal(_ev("Lunch with Sam", NOON), _ev("BIOSENSORS class", NOON - timedelta(minutes=30)))
    p = _policy(get_calendar=lambda: cal)
    assert p.reason() == "BIOSENSORS class until 12:30 pm"
    cal2 = FakeCal(_ev("Lunch with Sam", NOON))
    assert not _policy(get_calendar=lambda: cal2).is_quiet()


def test_calendar_keywords_match_whole_words_and_skip_all_day_and_future():
    cal = FakeCal(_ev("Classic rock hour", NOON - timedelta(minutes=5)),
                  _ev("Exam week", NOON, all_day=True),
                  _ev("Meeting with Dr K", NOON + timedelta(hours=2)))
    assert not _policy(get_calendar=lambda: cal).is_quiet()
    cal2 = FakeCal(_ev("Busy: thesis block", NOON - timedelta(minutes=5)))
    assert _policy(get_calendar=lambda: cal2).is_quiet()


def test_a_recurring_course_is_quiet_even_though_it_is_called_no_such_thing():
    """The shipped keyword list -- class / exam / meeting / busy -- matches
    none of his course titles, so the calendar leg had never once fired for
    him. A title at the same weekday and clock time on two weeks IS a class
    (jarvis/courses.py)."""
    running = _ev("BIOSENSORS", NOON - timedelta(minutes=30))
    same_slot_next_week = _ev("BIOSENSORS", NOON - timedelta(minutes=30) +
                              timedelta(days=7))
    cal = FakeCal(running, same_slot_next_week)
    assert _policy(get_calendar=lambda: cal).reason() == "BIOSENSORS until 12:30 pm"


def test_a_one_off_appointment_is_still_not_a_class():
    cal = FakeCal(_ev("Chiro", NOON - timedelta(minutes=30)))
    assert not _policy(get_calendar=lambda: cal).is_quiet()


def test_the_course_leg_can_be_switched_off_on_its_own():
    cal = FakeCal(_ev("BIOSENSORS", NOON - timedelta(minutes=30)),
                  _ev("BIOSENSORS", NOON - timedelta(minutes=30) + timedelta(days=7)))
    off = _policy(FakeCfg({"quiet": {"calendar_courses": False}}),
                  get_calendar=lambda: cal)
    assert not off.is_quiet()


def test_reason_without_the_calendar_leg_ignores_the_running_event():
    """What the class stager asks: it is acting ON the running event, so a
    gate the event itself trips would never let it fire."""
    cal = FakeCal(_ev("BIOSENSORS class", NOON - timedelta(minutes=30)))
    p = _policy(get_calendar=lambda: cal)
    assert p.reason() and p.reason(calendar=False) == ""
    dnd = _policy(FakeCfg({"quiet": {"dnd_until": NOON.timestamp() + 600}}),
                  get_calendar=lambda: cal)
    assert dnd.reason(calendar=False).startswith("do not disturb")


def test_calendar_block_can_be_switched_off_and_a_broken_calendar_is_ignored():
    cal = FakeCal(_ev("Exam", NOON - timedelta(minutes=5)))
    off = _policy(FakeCfg({"quiet": {"calendar": False}}), get_calendar=lambda: cal)
    assert not off.is_quiet()

    class Broken:
        configured = True

        def events(self):
            raise RuntimeError("cache unreadable")
    assert not _policy(get_calendar=lambda: Broken()).is_quiet()
    unconf = FakeCal(_ev("Exam", NOON - timedelta(minutes=5)))
    unconf.configured = False
    assert not _policy(get_calendar=lambda: unconf).is_quiet()


def test_away_holds_speech_unless_disabled():
    p = _policy(is_home=lambda: False)
    assert p.reason() == "you're out"
    p2 = _policy(FakeCfg({"quiet": {"hold_when_away": False}}), is_home=lambda: False)
    assert not p2.is_quiet()
    p3 = _policy(is_home=lambda: True)
    assert not p3.is_quiet()


# --------------------------------------------------------- hold / digest
def test_digest_counts_by_kind_and_reads_every_line():
    items = [(0, "Sir, this is your reminder. Call mum", "reminder"),
             (0, "Sir, this is your reminder. Submit the lab", "reminder"),
             (0, "Memory is getting tight, sir: 12 GB free.", "warning")]
    text = digest(items)
    assert text.startswith(BUSY_PREFIX + ": two reminders and one warning. ")
    # One summons announces the list; the second copy of it is the chant the
    # address pass thins (jarvis/address.py, defect D4).
    assert "Call mum. This is your reminder. Submit the lab. Memory" in text
    assert digest([]) == ""


def test_hold_caps_the_backlog_at_hold_max():
    p = _policy(hold_max=3)
    for i in range(5):
        p.hold(f"warning {i}", "warning")
    held = [t for _, t, _ in p.held]
    assert held == ["warning 2", "warning 3", "warning 4"]
    assert "three warnings" in p.release()
    assert p.release() == ""                 # drained


def test_release_prefix_follows_the_reason_that_held_it():
    p = _policy(is_home=lambda: False)
    assert p.should_hold()
    p.hold("Sir, your ten minute timer is up.", "timer")
    assert p.release().startswith(AWAY_PREFIX)


def test_free_ends_the_window_early_and_reads_the_digest():
    cfg = FakeCfg({"quiet": {"hours": {"start": "23:00", "end": "07:00"}}})
    p = _policy(cfg, Clock(datetime(2026, 8, 31, 23, 30)))
    assert p.should_hold()
    p.hold("Memory is getting tight, sir.", "warning")
    line = p.free()
    # "Very good, sir." has already addressed him, so the digest behind it
    # opens "While you were busy:" -- the join thins across the fragments
    # (jarvis/address.py, tests/test_address.py).
    assert line.startswith(FREE_LINE + " While you were busy: one warning.")
    assert not p.is_quiet()
    # ...for the rest of THIS window only
    assert cfg.get("quiet.free_until") == datetime(2026, 9, 1, 7, 0).timestamp()
    p.clock.tick(hours=24)                  # 23:30 the next night
    assert p.is_quiet()


def test_free_clears_dnd_and_says_so_when_nothing_was_quiet():
    p = _policy()
    p.set_dnd(3600)
    assert p.free() == FREE_LINE and not p.is_quiet()
    assert p.free() == DND_ALREADY_FREE_LINE


def test_free_overrides_a_wrong_away_reading():
    """He just spoke: he is in the room whatever his sleeping phone says."""
    p = _policy(is_home=lambda: False)
    assert p.is_quiet()
    p.free()
    # No window end to compute for "away", so free_until is not set --
    # but DND is cleared and the next probe decides. Speak now regardless:
    assert p.free() in (FREE_LINE, DND_ALREADY_FREE_LINE)


def test_config_without_set_keeps_state_in_memory():
    p = _policy(SimpleNamespace(get=lambda k, d=None: d))
    p.set_dnd(600)
    assert p.is_quiet()


# --------------------------------------------------------------- tick
def test_tick_speaks_the_digest_when_the_window_closes_and_only_once():
    said = []
    p = _policy(say=said.append)
    p.set_dnd(600)
    assert p.tick() == "" and p.should_hold()
    p.hold("Sir, this is your reminder. Stand up", "reminder")
    p.hold("Memory is getting tight, sir", "warning")
    assert p.tick() == ""                   # still quiet
    p.clock.tick(minutes=11)
    text = p.tick()
    assert text.startswith(BUSY_PREFIX + ": one reminder and one warning.")
    assert said == [text]
    assert p.tick() == "" and said == [text]


def test_tick_with_nothing_held_says_nothing():
    said = []
    p = _policy(say=said.append)
    p.set_dnd(600)
    p.tick()
    p.clock.tick(minutes=11)
    assert p.tick() == "" and said == []


def test_start_and_stop_join_the_thread():
    p = _policy(tick_s=0.01)
    p.start()
    assert p.running
    p.start()                               # idempotent
    p.stop()
    assert not p.running


# ------------------------------------------------------- banner gate
def test_quiet_gate_silences_desktop_banners_in_process(monkeypatch):
    notify_mod.reset_desktop_banner_cache()
    monkeypatch.setattr(notify_mod, "cfg_get", lambda cfg, key, default=None: True)
    quiet = {"on": False}
    notify_mod.set_quiet_gate(lambda: quiet["on"])
    try:
        assert notify_mod.desktop_banners_enabled() is True
        quiet["on"] = True
        assert notify_mod.desktop_banners_enabled() is False   # no cache in the way
        notify_mod.set_quiet_gate(lambda: 1 / 0)
        assert notify_mod.desktop_banners_enabled() is True    # a broken gate = on
    finally:
        notify_mod.set_quiet_gate(None)
        notify_mod.reset_desktop_banner_cache()


# ------------------------------------------------------- timekeeper tag
def test_timekeeper_tags_reminders_proactive_and_alarms_not(tmp_path, monkeypatch):
    import jarvis.tools.timekeeper as tk_mod
    monkeypatch.setattr(tk_mod, "_run", lambda argv, **kw: None)
    calls = []

    def say(text, proactive=False, kind=""):
        calls.append((text, proactive, kind))
    clock = Clock(NOON)
    t = Timekeeper(tmp_path / "tk.db", say=say, cfg={}, now=clock.now, tick_s=0.01,
                   ring=False, cache_dir=tmp_path / "cache", notify=False)
    try:
        t.add_reminder(clock.now() + 60, "call mum")
        t.add_alarm(clock.now() + 60, "up")
        clock.tick(minutes=2)
        t.tick()
        flags = {kind: proactive for _, proactive, kind in calls}
        assert flags.get("reminder") is True
        assert flags.get("alarm") is False
        assert t._say_takes_flag
    finally:
        t.close()


def test_timekeeper_keeps_the_one_argument_say_contract(tmp_path, monkeypatch):
    import jarvis.tools.timekeeper as tk_mod
    monkeypatch.setattr(tk_mod, "_run", lambda argv, **kw: None)
    said = []
    clock = Clock(NOON)
    t = Timekeeper(tmp_path / "tk.db", say=said.append, cfg={}, now=clock.now,
                   tick_s=0.01, ring=False, cache_dir=tmp_path / "cache", notify=False)
    try:
        assert not t._say_takes_flag
        t.add_reminder(clock.now() + 60, "call mum")
        clock.tick(minutes=2)
        t.tick()
        assert said and "call mum" in said[0].lower()
    finally:
        t.close()


def test_timekeeper_toast_obeys_the_banner_gate(tmp_path, monkeypatch):
    import jarvis.tools.timekeeper as tk_mod
    runs = []
    monkeypatch.setattr(tk_mod, "_run", lambda argv, **kw: runs.append(argv))
    monkeypatch.setattr(tk_mod, "_banners_ok", lambda: False)
    t = Timekeeper(tmp_path / "tk.db", say=lambda s: None, cfg={}, tick_s=0.01,
                   ring=False, cache_dir=tmp_path / "cache", notify=True)
    try:
        t._toast("Jarvis reminder", "x")
        assert runs == []
        monkeypatch.setattr(tk_mod, "_banners_ok", lambda: True)
        t._toast("Jarvis reminder", "x")
        assert runs and runs[0][0] == "notify-send"
    finally:
        t.close()


# ------------------------------------------------------------ the app
def _app(monkeypatch, quiet=None):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: {"briefing.on_first_wake": True,
                                                         "briefing.after": "06:00"}.get(k, d),
                                  user_name="Hunter")
    a._init_assistant_state()
    a.tts = SimpleNamespace(spoken=[])
    a.tts.speak = a.tts.spoken.append
    a.quiet = quiet
    a._briefing_state_path = lambda: SimpleNamespace(read_text=lambda: "{}")
    monkeypatch.setattr(CONFIG, "talkback", True)
    return a


def test_say_holds_only_proactive_lines_while_quiet(monkeypatch):
    p = _policy()
    p.set_dnd(3600)
    a = _app(monkeypatch, quiet=p)
    statuses = []
    bus.subscribe(Status, statuses.append)
    try:
        a._say("Memory is getting tight, sir.", proactive=True, kind="warning")
        a._say("It's 12 pm, sir.")                          # an answer
        a._say("Sir, it's 12 pm. Time to get up.", proactive=False, kind="alarm")
        assert a.tts.spoken == ["It's 12 pm, sir.", "Sir, it's 12 pm. Time to get up."]
        assert [t for _, t, _ in p.held] == ["Memory is getting tight, sir."]
        assert any(s.text.startswith("Held (do not disturb") for s in statuses)
    finally:
        bus.unsubscribe(Status, statuses.append)


def test_say_speaks_proactive_lines_when_not_quiet_or_without_a_policy(monkeypatch):
    a = _app(monkeypatch, quiet=_policy())
    a._say("Sir, your timer is up.", proactive=True, kind="timer")
    assert a.tts.spoken == ["Sir, your timer is up."]
    b = _app(monkeypatch, quiet=None)
    b._say("Sir, your timer is up.", proactive=True)
    assert b.tts.spoken == ["Sir, your timer is up."]


def test_say_speaks_when_the_policy_itself_breaks(monkeypatch):
    broken = SimpleNamespace(should_hold=lambda: 1 / 0)
    a = _app(monkeypatch, quiet=broken)
    a._say("line", proactive=True)
    assert a.tts.spoken == ["line"]


def test_services_speak_is_the_watchdogs_proactive_door(monkeypatch):
    p = _policy()
    p.set_dnd(3600)
    a = _app(monkeypatch, quiet=p)
    # the lambda _build_services installs, verbatim
    def speak(text, proactive=True, kind="warning"):
        a._say(text, proactive=proactive, kind=kind)
    speak("Memory is getting tight, sir.")
    assert a.tts.spoken == [] and p.held[0][2] == "warning"


def test_first_wake_briefing_waits_for_the_quiet_window(monkeypatch):
    p = _policy()
    a = _app(monkeypatch, quiet=p)
    at = datetime(2026, 8, 31, 9, 0)
    assert a._briefing_due(now=at)
    p.set_dnd(3600)
    assert not a._briefing_due(now=at)
    p.free()
    assert a._briefing_due(now=at)


def test_welcome_back_once_per_return_with_the_held_lines(monkeypatch):
    # is_home must FLIP with the return: _on_presence now consults
    # quiet.should_hold(), and a policy stuck on "you're out" would
    # (correctly) keep holding the welcome forever.
    home = [False]
    p = _policy(is_home=lambda: home[0])
    a = _app(monkeypatch, quiet=p)
    a._say("Sir, this is your reminder. Water the plants", proactive=True, kind="reminder")
    assert a.tts.spoken == []
    a._on_presence(Presence(home=False, since=1.0, returned=False))
    assert a.tts.spoken == []                          # leaving is silent
    home[0] = True
    a._on_presence(Presence(home=True, since=2.0, returned=True))
    assert a.tts.spoken[0] == "Welcome back, sir."
    # One burst over two _say calls: the welcome keeps the address and the
    # catch-up behind it drops its own (jarvis/address.py).
    assert a.tts.spoken[1].startswith("While you were out: one reminder.")
    a._on_presence(Presence(home=True, since=3.0, returned=False))   # boot-time "home"
    assert len(a.tts.spoken) == 2


# -------------------------------------------------------- the commander
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    p = _policy(clock=Clock(datetime(2026, 8, 31, 14, 0)))
    svc = SimpleNamespace(quiet=p, tts=None, reader=None, claude=None, timekeeper=None,
                          desktop=None)
    c = Commander(svc)
    c.policy = p
    return c


def test_do_not_disturb_for_an_hour(cmdr):
    res = cmdr.handle("Jarvis, do not disturb for an hour", source="voice")
    assert res.handled and res.speak
    assert res.reply == DND_SET_LINE.format(until="3 pm")
    assert cmdr.policy.reason() == "do not disturb until 3 pm"


@pytest.mark.parametrize("text,until", [
    ("don't disturb me for ninety minutes", "3:30 pm"),
    ("do not disturb until seven", "7 pm"),
    ("do not disturb until 9:15 pm", "9:15 pm"),
    ("hold my notifications for two hours", "4 pm"),
    ("I'm busy for the rest of the day", "11:59 pm"),
    ("do not disturb", "3 pm"),
    ("quiet for twenty minutes", "2:20 pm"),
])
def test_dnd_phrasings(cmdr, text, until):
    res = cmdr.handle(text, source="typed")
    assert res.handled and res.reply == DND_SET_LINE.format(until=until), (text, res.reply)


def test_dnd_with_an_unreadable_time_asks_again(cmdr):
    res = cmdr.handle("do not disturb until the cows come home", source="typed")
    assert res.handled and res.status == "Do not disturb: when?"
    assert not cmdr.policy.is_quiet()


def test_bare_quiet_is_still_the_barge_in(cmdr):
    res = cmdr.handle("Jarvis, quiet", source="voice")
    assert res.status == "Quiet" and not cmdr.policy.is_quiet()


def test_quiet_hours_from_eleven_to_seven(cmdr):
    res = cmdr.handle("quiet hours from eleven to seven", source="typed")
    assert res.reply == QUIET_HOURS_SET_LINE.format(start="11 pm", end="7 am")
    assert cmdr.policy.quiet_hours() == ((23, 0), (7, 0))
    res = cmdr.handle("what are my quiet hours", source="typed")
    assert res.reply == QUIET_HOURS_SET_LINE.format(start="11 pm", end="7 am")
    res = cmdr.handle("turn off quiet hours", source="typed")
    assert res.reply == QUIET_HOURS_OFF_LINE and cmdr.policy.quiet_hours() is None


def test_quiet_hours_with_explicit_am_pm(cmdr):
    cmdr.handle("set quiet hours to 10 pm until 6:30 am", source="typed")
    assert cmdr.policy.quiet_hours() == ((22, 0), (6, 30))


def test_i_am_free_reads_what_was_held(cmdr):
    cmdr.handle("do not disturb for an hour", source="typed")
    cmdr.policy.hold("Sir, this is your reminder. Call mum", "reminder")
    res = cmdr.handle("I am free", source="typed")
    assert res.reply.startswith(FREE_LINE + " While you were busy: one reminder.")
    assert not cmdr.policy.is_quiet()
    res = cmdr.handle("what did I miss", source="typed")
    assert res.reply == NOTHING_HELD_LINE


def test_quiet_status(cmdr):
    res = cmdr.handle("are you on do not disturb", source="typed")
    assert res.reply == QUIET_STATUS_FREE_LINE
    cmdr.handle("do not disturb for an hour", source="typed")
    res = cmdr.handle("are you on do not disturb", source="typed")
    assert "do not disturb until 3 pm" in res.reply


def test_commands_step_aside_without_the_policy(cmdr):
    cmdr.services.quiet = None
    res = cmdr.handle("do not disturb for an hour", source="typed")
    assert res.status != "Do not disturb until 3 pm"


# ------------------------------------------------ the real app, wired
def test_real_app_wires_quiet_and_presence(build, monkeypatch):  # noqa: F811
    """Build the real JarvisApp (test_app_wiring's fixture): the policy and
    sentinel exist, sit on services, start and stop with the assistant,
    the speak-queue sink is proactive and the config gained the keys."""
    a = build()
    assert a.quiet is not None and a.presence is not None
    assert a.services.quiet is a.quiet and a.services.presence is a.presence
    assert a.assistant.get("quiet.calendar_keywords") == ["class", "exam", "meeting", "busy"]
    assert a.assistant.get("presence.away_after_min") == 12
    a.quiet.set_dnd(3600)
    from jarvis import speak_queue
    speak_queue.say("Tests: 12 passed")               # the hooks narrator
    assert a.tts.spoken == [] and [t for _, t, _ in a.quiet.held] == ["Tests: 12 passed"]
    a.services.speak("Memory is getting tight, sir.")  # the watchdog's door
    assert len(a.quiet.held) == 2
    assert notify_mod._quiet_gate.__self__ is a.quiet   # banners ask this policy
    a.start_assistant(residency=False)
    assert a.quiet.running
    assert not a.presence.running                      # no phone configured: idle
    a.stop_assistant()
    assert not a.quiet.running
    assert notify_mod._quiet_gate is None              # and let go of it on stop



# ----------------------------------------------------- can_speak deferral
def test_the_digest_waits_for_a_clear_moment():
    """The catch-up digest used to speak the instant the window closed --
    including over an open capture or a running turn. With can_speak False
    the backlog is kept and the next tick retries."""
    said, clear = [], [False]
    p = _policy(say=said.append, can_speak=lambda: clear[0])
    p.set_dnd(600)
    p.hold("Sir, this is your reminder. Stand up", "reminder")
    p.clock.tick(minutes=11)
    assert p.tick() == "" and said == []            # mid-turn: held, not lost
    assert p.tick() == "" and said == []            # still busy next tick
    clear[0] = True
    text = p.tick()
    assert "one reminder" in text and said == [text]


def test_a_broken_can_speak_probe_never_mutes_the_digest():
    said = []

    def boom():
        raise RuntimeError("probe died")
    p = _policy(say=said.append, can_speak=boom)
    p.set_dnd(600)
    p.hold("Sir, this is your reminder.", "reminder")
    p.clock.tick(minutes=11)
    assert p.tick() != "" and len(said) == 1


def test_a_held_alarm_notice_is_digested_as_an_alarm():
    p = _policy()
    p.set_dnd(600)
    p.hold("You missed your alarm at 7:00 am, sir.", "alarm")
    p.clock.tick(minutes=11)
    assert "one alarm" in p.tick(), "kind='alarm' must not read as 'message'"


# --------------------------------------------------- focus-aware DND (n=2)
class FakeFocus:
    """The `phase` slice of jarvis.focus.FocusSession."""

    def __init__(self, phase=""):
        self.phase = phase


def test_a_focus_block_holds_proactive_lines_and_the_break_reads_them_back():
    """assistant-setup.md said outright there was no do-not-disturb for a
    study block. Now the block holds and the existing tick() digest lands
    in the break, not mid-pomodoro."""
    said = []
    focus = FakeFocus("block")
    p = _policy(say=said.append, get_focus=lambda: focus)
    assert p.reason() == FOCUS_REASON and p.should_hold()
    p.hold("Sir, your Canvas deadline is in three hours.", "reminder")
    p.hold("Memory is getting tight, sir", "warning")
    assert p.tick() == "" and said == []            # mid-block: nothing spoken
    focus.phase = "break"                            # the break IS the moment
    text = p.tick()
    assert text.startswith(BUSY_PREFIX + ": one reminder and one warning.")
    assert said == [text]


def test_a_focus_break_and_a_finished_session_are_not_quiet():
    assert _policy(get_focus=lambda: FakeFocus("break")).reason() == ""
    assert _policy(get_focus=lambda: FakeFocus("")).reason() == ""
    assert _policy(get_focus=lambda: None).reason() == ""
    assert _policy().reason() == ""                   # no probe wired at all


def test_focus_dnd_can_be_switched_off():
    cfg = FakeCfg({"focus": {"dnd": False}})
    p = _policy(cfg, get_focus=lambda: FakeFocus("block"))
    assert p.reason() == "" and not p.should_hold()


def test_a_broken_focus_probe_never_mutes_him():
    def boom():
        raise RuntimeError("focus died")
    assert _policy(get_focus=boom).reason() == ""


def test_dnd_outranks_the_focus_block_in_the_spoken_reason():
    """Both hold; the explicit one is what he names when asked why."""
    p = _policy(get_focus=lambda: FakeFocus("block"))
    p.set_dnd(3600)
    assert p.reason().startswith("do not disturb until")


def test_i_am_free_pierces_a_focus_block():
    p = _policy(get_focus=lambda: FakeFocus("block"))
    p._set("quiet.free_until", p.now() + 600)
    assert p.reason() == ""


# -------------------------------------------------- nudges expire (n=24)
def test_a_nudge_expires_instead_of_joining_the_digest():
    """"Stand up, sir" read back after a two-hour meeting is noise, and
    five of them is worse -- so a held nudge is dropped, not queued."""
    p = _policy()
    p.set_dnd(600)
    assert p.hold("Sir, this is your reminder. Drink water.", "nudge") is False
    assert p.hold("Sir, this is your reminder. Drink water.", "nudge") is False
    assert p.held == []
    assert p.hold("Sir, your lab report is due in three hours.", "reminder") is True
    p.clock.tick(minutes=11)
    text = p.tick()
    assert "one reminder" in text and "drink water" not in text.lower()


def test_a_nudge_never_eats_a_slot_in_the_capped_backlog():
    p = _policy(hold_max=3)
    p.set_dnd(600)
    for i in range(10):
        p.hold(f"nudge {i}", "nudge")
    for i in range(3):
        p.hold(f"warning {i}", "warning")
    assert [t for _, t, _ in p.held] == ["warning 0", "warning 1", "warning 2"]


def test_an_unheld_nudge_is_spoken_normally(monkeypatch):
    """Expiry is a QUIET-hours rule, not a mute: with the window open the
    nudge goes straight to the TTS like anything else."""
    a = _app(monkeypatch, quiet=_policy())
    a._say("Sir, this is your reminder. Drink water.", proactive=True, kind="nudge")
    assert a.tts.spoken == ["Sir, this is your reminder. Drink water."]


def test_the_status_line_says_expired_not_held(monkeypatch):
    p = _policy()
    p.set_dnd(600)
    a = _app(monkeypatch, quiet=p)
    seen = []
    unsub = bus.subscribe(Status, seen.append)
    try:
        a._say("Drink water.", proactive=True, kind="nudge")
        a._say("Memory is tight.", proactive=True, kind="warning")
    finally:
        bus.unsubscribe(Status, unsub)
    kinds = [s.text.split(" ", 1)[0] for s in seen]
    assert kinds == ["Expired", "Held"] and a.tts.spoken == []


def test_timekeeper_tags_an_interval_reminder_as_a_nudge(tmp_path, monkeypatch):
    """The whole chain: 'every 45 minutes' -> repeat '45m' -> kind 'nudge'
    -> the quiet policy expires it."""
    import jarvis.tools.timekeeper as tk_mod
    monkeypatch.setattr(tk_mod, "_run", lambda argv, **kw: None)
    calls = []

    def say(text, proactive=False, kind=""):
        calls.append((text, proactive, kind))
    clock = Clock(NOON)
    t = Timekeeper(tmp_path / "tk.db", say=say, cfg={}, now=clock.now, tick_s=0.01,
                   ring=False, cache_dir=tmp_path / "cache", notify=False)
    try:
        t.add_reminder(clock.now() + 60, "drink water", repeat="45m")
        t.add_reminder(clock.now() + 60, "call mum")
        clock.tick(minutes=2)
        t.tick()
        flags = {kind: proactive for _, proactive, kind in calls}
        assert flags == {"nudge": True, "reminder": True}
    finally:
        t.close()


def test_real_app_asks_its_own_focus_session(build, monkeypatch):  # noqa: F811
    """The probe must be LATE-bound: app._make_quiet runs before
    _make_focus, so a policy holding the object itself would hold None."""
    a = build()
    assert a.focus is not None and a.assistant.get("focus.dnd") is True
    assert a.quiet.reason() == ""
    a.focus.state["phase"] = "block"
    assert a.quiet.reason() == FOCUS_REASON
    a.services.speak("Memory is getting tight, sir.")     # the watchdog's door
    assert a.tts.spoken == [] and len(a.quiet.held) == 1
    a.focus._speak("Time for a break, sir.")              # the session's own
    assert a.tts.spoken == ["Time for a break, sir."]


# =====================================================================
# take_fragments: a drain you can undo
# =====================================================================
# 2026-09-03. The arrival catch-up drained the backlog on the pump thread,
# went off to read a mailbox on a worker, and then DROPPED the digest as
# stale -- taking the drained lines with it. They are the things he missed
# while he was out and there is no second copy of them anywhere. Any caller
# that can still decide not to speak takes the reversible form.
def test_a_take_that_is_put_back_leaves_the_backlog_exactly_as_it_was():
    p = _policy()
    p.set_dnd(600)
    p.hold("The build passed, sir.", "message")
    p.hold("Canvas: Lab 3 graded.", "message")
    before = p.held
    frags, put_back = p.take_fragments()
    assert frags and p.held == [], "take_fragments did not drain"
    assert put_back() == 2
    assert p.held == before, "the held lines did not come back as they were"


def test_putting_back_twice_puts_them_back_once():
    p = _policy()
    p.set_dnd(600)
    p.hold("The build passed, sir.", "message")
    _, put_back = p.take_fragments()
    assert put_back() == 1 and put_back() == 0
    assert len(p.held) == 1


def test_a_put_back_goes_in_FRONT_of_anything_held_meanwhile():
    """The lines he missed first are still read first."""
    p = _policy()
    p.set_dnd(600)
    p.hold("First.", "message")
    _, put_back = p.take_fragments()
    p.hold("Second.", "message")
    put_back()
    assert [text for _, text, _ in p.held] == ["First.", "Second."]


def test_a_put_back_over_the_cap_drops_the_OLDEST_as_hold_would_have():
    """The backlog is a bounded deque; putting lines back must not make it
    unbounded, and must drop the same line hold() would have."""
    from jarvis.quiet import HOLD_MAX
    p = _policy()
    p.set_dnd(600)
    p.hold("oldest.", "message")
    _, put_back = p.take_fragments()
    for i in range(HOLD_MAX):
        p.hold(f"later {i}.", "message")
    put_back()
    texts = [text for _, text, _ in p.held]
    assert len(texts) == HOLD_MAX and "oldest." not in texts
    assert texts[-1] == f"later {HOLD_MAX - 1}."


def test_a_put_back_is_still_SPOKEN_by_the_policys_own_next_tick():
    """"Still held" is worth nothing if nothing ever says it. The digest
    goes out on the FALLING edge of the quiet window and never again, and
    the take consumed the tick that edge belonged to -- so the put-back
    re-arms it."""
    said = []
    p = _policy(say=said.append)
    p.set_dnd(600)
    p.hold("Sir, this is your reminder. Stand up", "reminder")
    p.clock.tick(minutes=11)                 # the window is over
    frags, put_back = p.take_fragments()     # ...and the arrival cue took it
    assert frags
    assert p.tick() == "" and said == []     # nothing left to say this tick
    put_back()                               # ...and then did not speak it
    text = p.tick()
    assert "one reminder" in text and said == [text]


def test_a_take_of_an_empty_backlog_still_answers_the_contract():
    p = _policy()
    frags, put_back = p.take_fragments()
    assert frags == [] and put_back() == 0 and p.held == []


def test_release_fragments_is_still_the_one_way_form():
    """The one-way primitive is unchanged for the callers that speak what
    they took there and then ("I am free", the policy's own tick)."""
    p = _policy()
    p.set_dnd(600)
    p.hold("The build passed, sir.", "message")
    frags = p.release_fragments()
    assert frags and p.held == []
    assert frags[0].startswith(BUSY_PREFIX)


def test_a_digest_the_TTS_could_not_speak_is_KEPT_and_retried():
    """The same class one level up in this file: tick() drained one-way and
    then swallowed a TTS failure, so a digest that could not be spoken took
    the whole backlog with it. A failed speak keeps the lines and the next
    tick tries again."""
    tries = []

    def say(text):
        tries.append(text)
        if len(tries) == 1:
            raise RuntimeError("the speaker is unplugged")
    p = _policy(say=say)
    p.set_dnd(600)
    p.hold("Sir, this is your reminder. Stand up", "reminder")
    p.clock.tick(minutes=11)
    assert p.tick() == "", "a digest that was not spoken was reported as said"
    assert len(p.held) == 1, "the TTS failure destroyed the backlog"
    text = p.tick()
    assert "one reminder" in text and len(tries) == 2 and p.held == []


def test_a_digest_that_JOINS_TO_NOTHING_is_not_counted_as_spent():
    """The other way out of tick() without words: nothing speakable came of
    the fragments. They are not spent either."""
    said = []
    p = _policy(say=said.append)
    p.set_dnd(600)
    p.hold("The build passed, sir.", "message")
    p.clock.tick(minutes=11)
    import jarvis.quiet as quiet_mod
    real = quiet_mod.address.join_fragments
    quiet_mod.address.join_fragments = lambda *a, **kw: ""
    try:
        assert p.tick() == ""
    finally:
        quiet_mod.address.join_fragments = real
    assert len(p.held) == 1 and said == []
    assert "The build passed" in p.tick()
