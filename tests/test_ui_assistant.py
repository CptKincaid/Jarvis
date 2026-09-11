"""Tk-free helpers behind the assistant UI (spec 2026-08-26 section 9):
pill precedence with WORKING / WAITING, the PROJECT chip formatter and
strip fitting plan, progress-card collapsing, briefing rows, the
terminal-button state from ClaudeTaskState sequences, the command-bar
width budget and the Services contract. No display is needed: only pure
functions and dataclasses are imported (tkinter is imported by the
modules but no root is created)."""
import os
import tempfile

os.environ.setdefault("JARVIS_LOG_DIR", tempfile.mkdtemp(prefix="jarvis-ui-"))
os.environ.setdefault("JARVIS_ASSISTANT_CONFIG",
                      os.path.join(tempfile.mkdtemp(prefix="jarvis-ui-cfg-"),
                                   "assistant.json"))

from dataclasses import fields  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from jarvis.events import ClaudeTaskState  # noqa: E402
from jarvis.ui import theme  # noqa: E402
from jarvis.ui.main_window import (STATE_WORDS, ClaudeTaskTracker,  # noqa: E402
                                   MainWindow, Services, alarm_modal_text,
                                   resolve_state, session_exists,
                                   terminal_attached, terminal_button_state,
                                   terminal_tooltip)
from jarvis.ui.views import (PROGRESS_MAX, CommandBar, briefing_rows,  # noqa: E402
                             command_bar_field_px, fmt_project_chip,
                             plan_strip, progress_card_lines)


# ------------------------------------------------------------- pill
def test_resolve_state_precedence_with_claude_states():
    # speaking > listening > thinking > waiting > working > error > idle
    assert resolve_state(True, True, True, True, True, True) == "speaking"
    assert resolve_state(False, True, True, True, True, True) == "listening"
    assert resolve_state(False, False, True, True, True, True) == "thinking"
    assert resolve_state(False, False, False, True, True, True) == "waiting"
    assert resolve_state(False, False, False, True, False, True) == "working"
    assert resolve_state(False, False, False, True, False, False) == "error"
    assert resolve_state(False, False, False, False) == "idle"
    # the old four-argument call keeps working (defaults False)
    assert resolve_state(False, False, False, False, waiting=True) == "waiting"


def test_state_words_and_colours():
    assert STATE_WORDS["working"] == "WORKING"
    assert STATE_WORDS["waiting"] == "WAITING"
    # <= 9 and no ellipsis since 2026-09-03: the header keeps 312 px for
    # this chip and the sensing badge together at his 920-px window and
    # the wordmark is not allowed to yield; the participles stay, the
    # padding and the '…' went (tests/test_header_fit.py).
    for word in STATE_WORDS.values():
        assert word.isupper() and len(word) <= 9 and "…" not in word
    assert theme.STATE_COLORS["working"] == theme.CYAN_DIM
    assert theme.STATE_COLORS["waiting"] == theme.WARN
    # the word stays FOCAL for idle / working / waiting (dot carries colour)
    assert set(theme.FOCAL_WORD_STATES) == {"idle", "working", "waiting"}


# --------------------------------------------------------- tracker
def _run(tracker, events):
    for project, task_id, state in events:
        tracker.apply(task_id, state, project)
    return tracker.terminal_state()


def test_tracker_running_then_done():
    t = ClaudeTaskTracker()
    assert t.terminal_state() == "idle"
    assert _run(t, [("jarvis", "t1", "queued")]) == "idle"
    assert _run(t, [("jarvis", "t1", "running")]) == "working"
    assert t.working and not t.waiting and t.last_project == "jarvis"
    assert _run(t, [("jarvis", "t1", "done")]) == "idle"
    assert not t.working and t.live == 0


def test_tracker_waiting_beats_working_and_clears():
    t = ClaudeTaskTracker()
    events = [("jarvis", "a", "running"), ("hay", "b", "running"),
              ("jarvis", "a", "waiting")]
    assert _run(t, events) == "waiting"
    assert t.waiting and t.working
    assert _run(t, [("jarvis", "a", "running")]) == "working"
    assert _run(t, [("jarvis", "a", "cancelled")]) == "working"   # b still runs
    assert _run(t, [("hay", "b", "failed")]) == "idle"


def test_tracker_ignores_unknown_and_empty():
    t = ClaudeTaskTracker()
    t.apply("", "running", "jarvis")
    t.apply("x", "bogus", "jarvis")
    assert t.terminal_state() == "idle" and t.live == 0


def test_tracker_from_real_events():
    t = ClaudeTaskTracker()
    seq = [ClaudeTaskState(project="jarvis", task_id="t9", state="queued"),
           ClaudeTaskState(project="jarvis", task_id="t9", state="running"),
           ClaudeTaskState(project="jarvis", task_id="t9", state="waiting"),
           ClaudeTaskState(project="jarvis", task_id="t9", state="running"),
           ClaudeTaskState(project="jarvis", task_id="t9", state="done")]
    seen = []
    for ev in seq:
        t.apply(ev.task_id, ev.state, ev.project)
        seen.append(t.terminal_state())
    assert seen == ["idle", "working", "waiting", "working", "idle"]
    # pill precedence uses the same booleans
    t.apply("t9", "waiting", "jarvis")
    assert resolve_state(False, False, False, False, t.waiting,
                         t.working) == "waiting"


def test_session_exists_reads_tmux_session_names():
    class R:
        def __init__(self, out):
            self.stdout = out
    seen = []

    def run(cmd, out=""):
        seen.append(cmd)
        return R(out)

    assert session_exists(lambda c: run(c, "jarvis-jarvis\nwork\n")) is True
    assert seen[0] == ["tmux", "ls", "-F", "#{session_name}"]
    assert session_exists(lambda c: run(c, "work\nmisc\n")) is False
    assert session_exists(lambda c: run(c, "")) is False          # none alive
    # tmux missing / hung: never raises, never claims a session
    def boom(_c):
        raise OSError("no tmux")
    assert session_exists(boom) is False


def test_terminal_attached_sees_only_jarvis_clients():
    """The window's half of `terminal_open()`: is anyone watching Claude?"""
    class R:
        def __init__(self, out, rc=0):
            self.stdout, self.returncode = out, rc
    seen = []

    def run(cmd, out="", rc=0):
        seen.append(cmd)
        return R(out, rc)

    assert terminal_attached(lambda c: run(c, "jarvis-jarvis\n")) is True
    # bare list-clients, every client on the server: tmux 3.4 has no `-a`
    # flag for this command and fails the whole call if one is passed
    assert seen[0] == ["tmux", "list-clients", "-F", "#{session_name}"]
    assert terminal_attached(lambda c: run(c, "")) is False       # detached
    assert terminal_attached(lambda c: run(c, "work\nmisc\n")) is False
    assert terminal_attached(lambda c: run(c, "", 1)) is False    # no server

    def boom(_c):
        raise OSError("no tmux")
    assert terminal_attached(boom) is False


def test_terminal_button_state_dims_when_there_is_nothing_to_attach_to():
    # boot: no project, no tmux session → faint (but still clickable)
    assert terminal_button_state("idle", "", False) == "no_project"
    # a live session from an earlier run, or a project this session → normal
    assert terminal_button_state("idle", "", True) == "idle"
    assert terminal_button_state("idle", "jarvis", False) == "idle"
    # a running / blocked task always wins, project known or not
    assert terminal_button_state("working", "", False) == "working"
    assert terminal_button_state("waiting", "", False) == "waiting"
    # and it is exactly what the tracker + window feed it
    t = ClaudeTaskTracker()
    assert terminal_button_state(t.terminal_state(), "", False) == "no_project"
    t.apply("t1", "running", "jarvis")
    assert terminal_button_state(t.terminal_state(), t.last_project,
                                 False) == "working"
    t.apply("t1", "done", "jarvis")
    # the project sticks after the task ends, so the button stays lit
    assert terminal_button_state(t.terminal_state(), t.last_project,
                                 False) == "idle"
    assert set(CommandBar.TERMINAL_STATES) >= {"no_project", "idle", "open",
                                               "working", "waiting"}


def test_terminal_button_state_says_when_a_terminal_is_already_open():
    """The button reflects whether the pop-out is up: `open` instead of
    `idle` while a client is attached, but a live task still wins."""
    assert terminal_button_state("idle", "jarvis", True, True) == "open"
    assert terminal_button_state("idle", "", False, True) == "open"
    assert terminal_button_state("idle", "jarvis", True, False) == "idle"
    assert terminal_button_state("working", "jarvis", True, True) == "working"
    assert terminal_button_state("waiting", "jarvis", True, True) == "waiting"
    # the ring closes for every "somebody can see this" state
    assert "open" in CommandBar.TERMINAL_STATES


def test_terminal_tooltip():
    assert terminal_tooltip("jarvis") == "Open Claude's terminal — jarvis"
    assert terminal_tooltip("") == CommandBar.TIP_NO_PROJECT == "No project yet"
    assert terminal_tooltip("jarvis", True) == "Claude's terminal is open — jarvis"
    assert terminal_tooltip("", True) == CommandBar.TIP_OPEN == \
        "Claude's terminal is open"


# ------------------------------------------------------ command bar
def test_command_bar_fits_at_minimum_width():
    # [PAD] field [PAD_S] terminal [PAD_S] mic [PAD] at 460 design px
    assert command_bar_field_px(460) == 460 - 16 - 8 - 44 - 8 - 44 - 16 == 324
    assert command_bar_field_px(460) >= 300
    assert command_bar_field_px(460, buttons=1) == 376   # the old bar
    assert CommandBar.TERMINAL_STATES == ("idle", "open", "working",
                                          "waiting", "no_project", "disabled")


# ----------------------------------------------------- project chip
def test_fmt_project_chip():
    assert fmt_project_chip("jarvis", 10) == "JARVIS"
    assert fmt_project_chip("haymaker-digest", 15) == "HAYMAKER-DIGEST"
    assert fmt_project_chip("haymaker-digest", 8) == "HAYMAKE…"
    assert len(fmt_project_chip("haymaker-digest", 8)) == 8
    # the budget never drops below six characters (ellipsis included)
    assert fmt_project_chip("haymaker-digest", 3) == "HAYMA…"
    assert fmt_project_chip("", 10) == ""
    assert fmt_project_chip("jarvis", 0) == ""
    assert fmt_project_chip("  vss  ", 2) == "VSS"     # short slug, tiny budget


def test_plan_strip_value_budget_shrinks_first():
    segs = [("CPU", 223), ("GPU", 209), ("MEMORY", 258)]
    # no project → nothing hidden, no chars
    assert plan_strip(1040, 237, 130, 14, 0, segs) == (0, [])
    # roomy: the whole 15-char slug fits, everything shown
    assert plan_strip(1600, 237, 130, 14, 15, segs) == (15, [])
    # tight: the value shrinks toward six characters before anything yields
    chars, hidden = plan_strip(1240, 237, 130, 14, 15, segs)
    assert hidden == [] and 6 <= chars < 15          # 183 px → 13 chars
    assert chars == 13
    # too narrow even for six: MEMORY yields, chip stays (>= 6 chars)
    chars, hidden = plan_strip(1040, 237, 130, 14, 15, segs)
    assert hidden == ["MEMORY"] and chars >= 6
    # slug shorter than six needs only its own length
    assert plan_strip(1040, 237, 130, 14, 3, segs)[0] == 3
    # absurdly narrow: the chip itself yields -- and takes its yield with
    # it. This read (0, ["MEMORY"]) until 2026-09-03: a plan with no chip
    # that still hid the segment the chip had displaced is the footer of
    # the 09-03 shot 23 ('CPU 52°C · 7% | GPU 44°C · 2%', no memory figure,
    # no PROJECT chip); tests/test_ui_chrome.py has the measured widths
    # (ui-polish U14)
    assert plan_strip(500, 237, 130, 14, 15, segs) == (0, [])


def test_plan_strip_at_460_design_px_keeps_the_chip():
    # measured at S=2 (dev px): wake segment 237, PROJECT fixed 130,
    # mono char 14, CPU 223 / GPU 209 / MEMORY 258 — the 460 minimum
    # window (920 dev px) still shows PROJECT with six characters.
    segs = [("CPU", 223), ("GPU", 209), ("MEMORY", 258)]
    chars, hidden = plan_strip(920, 237, 130, 14, 6, segs)
    assert chars == 6
    assert fmt_project_chip("jarvis", chars) == "JARVIS"


# ----------------------------------------------------- progress card
def test_progress_card_lines_collapse():
    lines = [f"step {i}" for i in range(1, 6)]
    assert progress_card_lines(lines, 12) == lines
    many = [f"step {i}" for i in range(1, 21)]
    out = progress_card_lines(many, 12)
    assert len(out) == 12
    assert out[0] == "… 9 earlier steps"
    assert out[1:] == many[-11:]
    assert progress_card_lines([], 12) == []
    assert progress_card_lines(["", " ", "x"], 12) == ["x"]
    assert PROGRESS_MAX == 12


# ------------------------------------------------------ briefing card
def test_briefing_rows():
    sections = {
        "weather": "72°F and partly cloudy; high 85, low 64.",
        "calendar": ["10:00 am dentist", "2:30 pm standup"],
        "news": [{"title": "Gemma 4 lands", "source": "Hacker News"},
                 {"title": "Verge story", "source": "The Verge"},
                 {"title": "Ars story", "source": "Ars Technica"},
                 {"title": "fourth item never shown", "source": "x"}],
        "sports": [], "stocks": [],
    }
    rows = briefing_rows(sections)
    assert rows == [
        ("WEATHER", "72°F and partly cloudy; high 85, low 64."),
        ("CALENDAR", "10:00 am dentist"), ("", "2:30 pm standup"),
        ("NEWS", "Gemma 4 lands — Hacker News"),
        ("", "Verge story — The Verge"), ("", "Ars story — Ars Technica")]
    labels = [lab for lab, _t in rows if lab]
    assert labels == ["WEATHER", "CALENDAR", "NEWS"]     # each label once


def test_briefing_rows_optional_sections_and_strings():
    rows = briefing_rows({"weather": "", "calendar": "Nothing on today, sir.",
                          "news": ["plain title"], "sports": "Cubs won",
                          "stocks": ["NVDA 130.2 +1.1%"]})
    assert rows == [("CALENDAR", "Nothing on today, sir."),
                    ("NEWS", "plain title"), ("SPORTS", "Cubs won"),
                    ("STOCKS", "NVDA 130.2 +1.1%")]
    assert briefing_rows({}) == []
    assert briefing_rows(None) == []


def test_briefing_rows_coursework_sits_between_calendar_and_news():
    rows = briefing_rows({"weather": "", "calendar": ["10:00 am dentist"],
                          "due": ["BIOSENSORS - Lab 3 report, today 11:59 pm",
                                  "CIRCUITS - Quiz 2, tomorrow 5:00 pm"],
                          "exam": "Midterm 1 for BIOSENSORS, in 6 days, Tuesday at 9:00 am",
                          "news": [{"title": "T", "source": "S"}]})
    assert rows == [("CALENDAR", "10:00 am dentist"),
                    ("DUE", "BIOSENSORS - Lab 3 report, today 11:59 pm"),
                    ("", "CIRCUITS - Quiz 2, tomorrow 5:00 pm"),
                    ("EXAM", "Midterm 1 for BIOSENSORS, in 6 days, Tuesday at 9:00 am"),
                    ("NEWS", "T — S")]
    # unconfigured Canvas: the keys are there but empty, and no row appears
    assert briefing_rows({"due": [], "exam": "", "news": []}) == []


# ------------------------------------------------------- alarm modal
def test_alarm_modal_text():
    assert alarm_modal_text("Time to get up.", "alarm", "7:00 am") == \
        ("Time to get up.", "7:00 AM")
    assert alarm_modal_text("", "alarm", "") == ("ALARM", "")
    assert alarm_modal_text("", "timer", "in 1 minute") == ("TIMER",
                                                            "IN 1 MINUTE")
    assert alarm_modal_text(None, None, None) == ("ALARM", "")


# ---------------------------------------------------------- services
def test_services_gains_assistant_callables_with_noop_defaults():
    names = {f.name for f in fields(Services)}
    for name in ("open_terminal", "alarm_action", "approval_answer",
                 "get_option", "set_option"):
        assert name in names
    svc = Services()
    # every default is callable and swallows arguments (returns None)
    assert svc.open_terminal() is None
    assert svc.alarm_action("a1", "dismiss", 10) is None
    assert svc.approval_answer("r1", True) is None
    assert svc.get_option("briefing.enabled") is None
    assert svc.set_option("briefing.enabled", True) is None


# ---------------------------------- the 5 s worker's one nvidia-smi
# The temps row and the ambient slab ran on the SAME 5 s pass and each
# forked its own nvidia-smi -- ~720 extra spawns an hour (plus two /proc
# walks inside health.snapshot) for a number the pass had just parsed and
# thrown away. The pass now hands its reading over.
class _Pass:
    """MainWindow's two worker-thread probes, bound with no Tk root."""

    _read_temps = MainWindow._read_temps
    _probe_room = MainWindow._probe_room

    _room_state_kwargs = MainWindow._room_state_kwargs

    def __init__(self, services):
        self.services = services
        self._room_data: dict = {}
        self._gpu_util_pct = None
        self._room_state_gpu_kw = None

    def _cpu_percent(self):
        return None


class _Smi:
    def __init__(self, stdout):
        self.stdout = stdout


def test_the_temps_pass_hands_its_gpu_reading_to_the_room_probe(monkeypatch):
    import subprocess as sp
    seen = {}
    monkeypatch.setattr(sp, "run",
                        lambda *a, **kw: _Smi("55, 42\n"))
    def room_state(gpu_pct=None):
        seen["gpu_pct"] = gpu_pct
        return {"gpu": 0.42}

    p = _Pass(SimpleNamespace(room_state=room_state))
    assert "gpu 55° 42%" in p._read_temps()
    assert p._gpu_util_pct == 42
    p._probe_room()
    assert seen["gpu_pct"] == 42
    assert p._room_data == {"gpu": 0.42}


def test_a_failed_gpu_probe_leaves_no_stale_number_on_the_slab(monkeypatch):
    import subprocess as sp
    p = _Pass(SimpleNamespace(room_state=lambda gpu_pct=None: {"gpu": None}))
    monkeypatch.setattr(sp, "run", lambda *a, **kw: _Smi("55, 42\n"))
    p._read_temps()
    assert p._gpu_util_pct == 42

    def boom(*a, **kw):
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(sp, "run", boom)
    p._read_temps()
    assert p._gpu_util_pct is None


def test_the_room_probe_still_works_against_a_provider_with_no_gpu_arg(
        monkeypatch):
    """build_ui_services can hand over an older room_state; losing the slab
    entirely would be a worse trade than the extra fork."""
    p = _Pass(SimpleNamespace(room_state=lambda: {"gpu": None, "temp": "72"}))
    p._gpu_util_pct = 42
    p._probe_room()
    assert p._room_data["temp"] == "72"


# =====================================================================
# 2026-09-03 ui-polish: pill precedence (U05/U07), the alarm parts (U10),
# the standby alpha wiring (U08). The 09-03 panel's measurements are in
# the docstrings of the functions under test.
# =====================================================================
import pytest  # noqa: E402

from jarvis.events import ApprovalResolved, ModelInfo  # noqa: E402
from jarvis.ui.console_mode import ACTIVE, STANDBY  # noqa: E402
from jarvis.ui.main_window import alarm_modal_parts, pill_look  # noqa: E402


@pytest.fixture(autouse=True)
def _holo():
    theme.select_look("holo")
    yield
    theme.select_look(theme.DEFAULT_LOOK)


# ------------------------------------ U05 / U07 the pill's precedence
def test_resolve_state_places_alarm_under_the_turn_and_loading_over_idle():
    assert resolve_state(False, False, False, False, alarm=True) == "alarm"
    assert resolve_state(False, False, True, False, alarm=True) == "thinking"
    assert resolve_state(False, True, False, False, alarm=True) == "listening"
    assert resolve_state(False, False, False, False, waiting=True,
                         alarm=True) == "alarm"
    assert resolve_state(False, False, False, True, loading=True) == "error"
    assert resolve_state(False, False, False, False, working=True,
                         loading=True) == "working"
    assert resolve_state(False, False, False, False, loading=True) == "loading"
    assert STATE_WORDS["loading"] == "LOADING" and STATE_WORDS["alarm"] == "ALARM"
    assert theme.STATE_COLORS["loading"] == theme.FAINT
    assert theme.STATE_COLORS["alarm"] == theme.WARN


def test_pill_look_holo_error_outranks_a_warn_hold():
    # measured in 07-error: dot (255,180,84) = WARN with the word in ERR
    assert pill_look("error", True, "holo") == (theme.ERR, theme.ERR, "disc")
    assert pill_look("error", False, "holo") == (theme.ERR, theme.ERR, "disc")
    # every other state still takes the amber dot while the hold lives
    assert pill_look("idle", True, "holo") == (theme.WARN, theme.FOCAL, "disc")
    assert pill_look("listening", True, "holo")[0] == theme.WARN


def test_pill_look_holo_gives_the_claude_states_a_ring():
    # measured in 23: WORKING's dot (24,153,189) == READY's
    assert pill_look("idle", False, "holo") == (theme.CYAN_DIM, theme.FOCAL, "disc")
    assert pill_look("working", False, "holo") == (theme.CYAN, theme.FOCAL, "ring")
    assert pill_look("waiting", False, "holo") == (theme.WARN, theme.FOCAL, "ring")
    assert pill_look("working", False, "holo")[:2] != pill_look("idle", False, "holo")[:2]
    assert pill_look("loading", False, "holo") == (theme.FAINT, theme.MUTED, "disc")
    assert pill_look("alarm", False, "holo") == (theme.WARN, theme.WARN, "disc")


def test_pill_look_classic_is_todays_table():
    theme.select_look("classic")
    for state in ("idle", "working", "waiting", "error", "thinking"):
        color = theme.STATE_COLORS[state]
        word = theme.FOCAL if state in theme.FOCAL_WORD_STATES else color
        assert pill_look(state, False) == (color, word, "disc")
        assert pill_look(state, True) == (theme.WARN, word, "disc")   # warts included


class _Pill:
    def __init__(self):
        self.states = []

    def set_state(self, word, dot, word_color, shape="disc"):
        self.states.append((word, dot, word_color, shape))

    @property
    def last(self):
        return self.states[-1]


class _Win:
    """The pill's inputs on a bare window: the shipping methods, unbound."""

    _app_state = MainWindow._app_state
    _refresh_pill = MainWindow._refresh_pill
    set_status = MainWindow.set_status
    _ev_model = MainWindow._ev_model
    _question_opened = MainWindow._question_opened
    _question_closed = MainWindow._question_closed
    _ev_approval_done = MainWindow._ev_approval_done
    _hide_alarm = MainWindow._hide_alarm
    _ev_alarm_stopped = MainWindow._ev_alarm_stopped

    def __init__(self):
        self.pill = _Pill()
        self._speaking = self._recording = self._thinking = False
        self._error_until = self._warn_until = 0.0
        self._tasks = ClaudeTaskTracker()
        self._alarm = None
        self._alarm_scrim = None
        self._booting = True
        self._pending = set()
        self._bar_state_color = theme.CYAN_DIM
        self._rule = None
        self.toast = SimpleNamespace(show=lambda *a, **k: None)
        self.root = SimpleNamespace(after=lambda ms, fn: None)
        self.transcript = SimpleNamespace(resolve_approval=lambda *a, **k: None)
        self.reactor = SimpleNamespace(delete=lambda tag: None)


def test_the_pill_reads_loading_until_the_model_lands():
    win = _Win()
    win._refresh_pill()
    assert win.pill.last == ("LOADING", theme.FAINT, theme.MUTED, "disc")
    # the boot's own busy Status changes nothing
    win.set_status("Loading speech model…", "busy")
    assert win.pill.last[0] == "LOADING"
    # ModelInfo is the model coming up: READY
    win._ev_model(ModelInfo(text="small · GPU fp16"))
    assert win.pill.last == ("READY", theme.CYAN_DIM, theme.FOCAL, "disc")
    assert not win._booting


def test_any_non_busy_status_ends_the_boot_and_classic_never_starts_it():
    win = _Win()
    win.set_status("Speech model failed to load", "error")
    assert not win._booting and win.pill.last[0] == "ERROR"
    win = _Win()
    win.set_status("Mic: Snowball", "info")
    assert not win._booting and win.pill.last[0] == "READY"
    theme.select_look("classic")
    win = _Win()
    win._refresh_pill()
    assert win.pill.last[0] == "READY"           # the 08-31 pill, untouched


def test_an_answer_owed_reads_waiting_and_a_ringing_alarm_reads_alarm():
    win = _Win()
    win._booting = False
    win._question_opened("u-1")
    assert win.pill.last == ("WAITING", theme.WARN, theme.FOCAL, "ring")
    win._question_opened("a-1")
    win._ev_approval_done(ApprovalResolved(request_id="u-1", allowed=True,
                                           source="ui"))
    assert win.pill.last[0] == "WAITING"         # a-1 is still open
    win._question_closed("a-1")
    assert win.pill.last[0] == "READY"
    win._question_closed("never-opened")          # harmless
    win._alarm = ("al-1", None, ())
    win._refresh_pill()
    assert win.pill.last == ("ALARM", theme.WARN, theme.WARN, "disc")
    # the scrim and the card go with the alarm; the header returns
    win._alarm = ("al-1", SimpleNamespace(place_forget=lambda: None,
                                          destroy=lambda: None), ())
    win._alarm_scrim = 7
    win._ev_alarm_stopped(SimpleNamespace(alarm_id="al-1"))
    assert win._alarm is None and win._alarm_scrim is None
    assert win.pill.last[0] == "READY"
    # classic sees neither input
    theme.select_look("classic")
    win = _Win()
    win._booting = False
    win._pending.add("u-9")
    win._alarm = ("al-9", None, ())
    win._refresh_pill()
    assert win.pill.last[0] == "READY"


# ---------------------------------------------- U10 the alarm's words
def test_alarm_modal_parts_always_names_the_kind():
    assert alarm_modal_parts("Biosensors lecture", "alarm", "9:45 am") == \
        ("ALARM", "Biosensors lecture", "9:45 AM")
    assert alarm_modal_parts("", "timer", "in 1 minute") == ("TIMER", "", "IN 1 MINUTE")
    assert alarm_modal_parts(None, None, None) == ("ALARM", "", "")
    assert alarm_modal_parts("  ", "reminder", " 7:00 pm ") == ("REMINDER", "", "7:00 PM")
    # the classic modal keeps its own text (the label OR the kind word)
    assert alarm_modal_text("Biosensors lecture", "alarm", "9:45 am")[0] == \
        "Biosensors lecture"
    assert MainWindow.ALARM_MARGIN == 16


# ------------------------------------------- U08 standby dims the stage
class _ModeWin:
    """_on_console_mode with every seam it touches stubbed and the standby
    alpha recorded."""

    _on_console_mode = MainWindow._on_console_mode

    def __init__(self):
        self.room = SimpleNamespace(set_mode=lambda mode: None)
        self.reactor = SimpleNamespace(set_speed_scale=lambda s: None)
        self.transcript = SimpleNamespace(pause_atmo=lambda p: None)
        self._standby_origin = None
        self._standby_drift = (0, 0)
        self.root = SimpleNamespace(winfo_x=lambda: 0, winfo_y=lambda: 0)
        self.alphas = []

    def _set_footer_hidden(self, hidden):
        pass

    def _set_tabs_hidden(self, hidden):
        pass

    def _preview_apply(self, mode=None, enabled=None):
        pass

    def _move_to(self, x, y):
        pass

    def _apply_standby(self, at_desk):
        self.alphas.append(at_desk)


def test_keyboard_idle_standby_dims_the_whole_window_in_holo():
    win = _ModeWin()
    win._on_console_mode(STANDBY)
    assert win.alphas == [False]
    win._on_console_mode(ACTIVE)
    assert win.alphas == [False, True]
    theme.select_look("classic")
    win = _ModeWin()
    win._on_console_mode(STANDBY)
    assert win.alphas == []                       # classic: the slab alone


def test_a_fake_without_the_standby_seam_still_changes_mode():
    win = _ModeWin()
    del _ModeWin._apply_standby
    try:
        win._on_console_mode(STANDBY)             # no AttributeError
    finally:
        _ModeWin._apply_standby = lambda self, at_desk: self.alphas.append(at_desk)


# ==================================================================
# THE 5 s WORKER: one bad pass must not be permanent (his bug #7)
# ==================================================================
# Two defects, and together they are the only thing that can freeze BOTH
# standby dates at once, which is what he actually reported.
#
#  * _temps_worker's loop body had no try/except at all. _read_temps,
#    _read_mem and _probe_llm are unguarded, so one escape ends the
#    thread silently and for ever -- no watchdog, no log line -- while
#    the slab goes on repainting the last dict it was handed at 1 Hz.
#    (_probe_room and _probe_sensing guard themselves. Three of five did
#    not: guard-one-half, again.)
#  * _probe_room keeps the previous dict on failure and says so at DEBUG
#    only, so a provider that is down stays invisible and its last answer
#    is served for ever. The same shape as the desk cache that answered
#    "5 seconds ago" through 400 dead polls.
class _Loop(_Pass):
    """The worker loop bound with no Tk root and no sleeping."""

    _temps_worker = MainWindow._temps_worker
    _temps_pass = MainWindow._temps_pass
    _pass_failed = MainWindow._pass_failed

    def __init__(self, services, passes=3):
        super().__init__(services)
        self._left = passes
        self._closing = False
        self.seen = 0

    def _probe_devices(self):
        pass


def test_a_bad_pass_does_not_kill_the_five_second_thread(monkeypatch):
    import jarvis.ui.main_window as mw
    monkeypatch.setattr(mw.time, "sleep", lambda s: None)
    loop = _Loop(SimpleNamespace(room_state=lambda gpu_pct=None: {}))

    def boom(beat):
        loop.seen += 1
        loop._left -= 1
        if loop._left <= 0:
            loop._closing = True
        raise RuntimeError("nvidia-smi went away")

    loop._temps_pass = boom
    loop._temps_worker()
    assert loop.seen == 3, "the thread died on the first bad pass"


def test_a_room_probe_down_for_a_minute_stops_repainting_a_stale_slab(
        monkeypatch):
    import jarvis.ui.main_window as mw
    clock = [1_000.0]
    monkeypatch.setattr(mw.time, "time", lambda: clock[0])
    good = {"due": "BIO - Lab 3, today 11:59 pm", "next": "SEMINAR 2:00 pm"}
    box = {"fn": lambda gpu_pct=None: dict(good)}
    p = _Pass(SimpleNamespace(room_state=lambda gpu_pct=None: box["fn"]()))
    p._probe_room()
    assert p._room_data == good

    def boom():
        raise RuntimeError("the provider is down")
    box["fn"] = boom

    clock[0] += 30.0
    p._probe_room()
    assert p._room_data == good, "thirty seconds is not stale yet"

    clock[0] += 40.0
    p._probe_room()
    assert p._room_data == {}, "a dead provider's last answer is painted for ever"


def test_after_tolerates_a_window_on_its_way_IN_as_well_as_OUT():
    """MEASURED on his box, 2026-09-11 17:33:15, on the first pass after a
    restart -- and it is the cause of his bug #7.

    _after guarded tk.TclError ("a window on its way out") and nothing
    else. The SAME call raises RuntimeError("main thread is not in main
    loop") on a window on its way IN: the 5 s worker starts before Tk's
    mainloop does, and _probe_sensing's _after(0, ...) is the first thing
    to reach it. Guard-one-half -- one direction of one lifecycle.

    What it cost: that RuntimeError escaped _temps_pass and killed the
    worker thread on pass 0 of EVERY boot. _probe_room runs just before
    _probe_sensing, so _room_data was filled once at startup and then
    never again -- which is exactly "the due and whats next dates in the
    standby screen do not update after the date passes". The loop's new
    net (this commit's parent) makes the thread survive; this stops the
    traceback and lets pass 0 finish.
    """
    class Root:
        def after(self, ms, fn):
            raise RuntimeError("main thread is not in main loop")

    w = object.__new__(MainWindow)
    w.root = Root()
    w._after(0, lambda: None)          # must not raise
