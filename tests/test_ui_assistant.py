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

from jarvis.events import ClaudeTaskState  # noqa: E402
from jarvis.ui import theme  # noqa: E402
from jarvis.ui.main_window import (STATE_WORDS, ClaudeTaskTracker,  # noqa: E402
                                   Services, alarm_modal_text, resolve_state,
                                   session_exists, terminal_attached,
                                   terminal_button_state, terminal_tooltip)
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
    for word in STATE_WORDS.values():
        assert word.isupper() and len(word) <= 10
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
    # absurdly narrow: the chip itself yields, MEMORY stays hidden
    assert plan_strip(500, 237, 130, 14, 15, segs) == (0, ["MEMORY"])


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
