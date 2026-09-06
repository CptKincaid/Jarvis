"""A GUEST'S SENTENCE LEAVES NOTHING IN HIS RECORD -- the asynchronous half.

ROUND-3 BLOCKER, MEASURED (09-05). ``app._after_dispatch`` was gated on the
turn's reading and the merge report called the class closed: "his record is
his: a turn that was not his leaves no trace in it". It was closed on the
SYNCHRONOUS half only. ``_handle_known`` step 5 -- the door EVERY guest
question that is not the clock or arithmetic goes through -- hands the words
to ``brain.chat`` on a worker thread, and ``JarvisBrain._remember`` took no
addressee and read no scope at all. Measured on the lane's own tip:

  * ``b.chat("mara asked about the spare key", addressee=("Mara","ma'am"))``
    put her words in HIS conversation ring, and his NEXT turn's outgoing
    payload contained "spare key" -- so it is not write-only: the ring
    ``ContextEngine.add_exchange`` appends to is the ring ``_dynamic_context``
    renders into his prompt;
  * the journal on disk recorded
    ``('exchange', {'user': "what's the weather like",
                    'jarvis': 'Ten degrees, ma'am.'})``;
  * ``memory.log_habit`` learned her sentence as one of his habits.

THREE PLACES, PINNED SEPARATELY, because they fail independently: the
in-memory ring, the file on disk, and the payload of his next turn. His own
turn must still land in all three -- a gate that files nothing is not a fix.

The fix is the seam the lane already built (7e0966a): the turn CARRIES whose
it is. ``chat`` / ``think`` / ``web_answer`` take ONE reading, on the
caller's thread and therefore before the worker's wait, and hand that VALUE
to ``_remember``. ``_remember`` is told; it does not look anything up.

Fakes only: the mocked Ollama and registry from tests/test_brain_tools.py, a
REAL ContextEngine writing to tmp_path. No mic, no camera, no model, no live
app, no real journal directory.
"""
import json
import pathlib

import pytest

from jarvis import brain as brain_mod
from jarvis import context as context_mod
from jarvis import scope as scope_mod
from tests.test_brain_tools import (FakeMemory, FakeOllama, make_registry,
                                    text_reply)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARA = ("Mara", "ma'am")
GUEST_WORDS = "mara asked about the spare key"
HIS_WORDS = "what did i leave running"


@pytest.fixture
def brain(tmp_path, monkeypatch):
    """A brain with a REAL ContextEngine -- the ring, the journal and the
    prompt rendering are the code under test, not a stand-in that only
    counts calls."""
    brain_mod.reset_static_prompt()
    record = []
    monkeypatch.setattr(brain_mod, "_REGISTRY", make_registry(record))
    fake = FakeOllama()
    monkeypatch.setattr(brain_mod, "_http", fake)
    ctx = context_mod.ContextEngine(project_dir=tmp_path,
                                    journal_dir=tmp_path / "journal")
    # The four ambient probes only: git, the focused window, the file list
    # and the session log all shell out, and none of them is what is under
    # test here. The ring, format_for_prompt, add_exchange and the journal
    # writer stay REAL -- they are the three sinks being measured.
    monkeypatch.setattr(ctx, "_get_git_state", lambda: {})
    monkeypatch.setattr(ctx, "_get_active_window", lambda: "")
    monkeypatch.setattr(ctx, "_get_recent_files", lambda: [])
    monkeypatch.setattr(ctx, "_get_sessions", lambda: [])
    mem = FakeMemory()
    b = brain_mod.JarvisBrain(context=ctx, memory=mem)
    return b, fake, ctx, mem


def _journal_rows(ctx):
    d = ctx.journal_dir()
    rows = []
    for path in sorted(pathlib.Path(d).glob("*.jsonl")) if d.exists() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _exchanges(ctx):
    return [(e["user"], e["jarvis"]) for e in ctx._conversation]


def _say(fake, line):
    fake.replies.append(text_reply(line))


def _turn(b, fake, text, addressee, line="Ten degrees, ma'am."):
    _say(fake, line)
    t = b.chat(text, addressee=addressee)
    assert t is not None, "the brain refused the job"
    t.join(timeout=10)
    assert not t.is_alive()


# ---------------------------------------------------------------- the ring
def test_a_guests_chat_leaves_nothing_in_the_conversation_ring(brain):
    b, fake, ctx, _mem = brain
    _turn(b, fake, GUEST_WORDS, MARA)
    assert _exchanges(ctx) == [], _exchanges(ctx)


def test_a_guests_chat_writes_no_journal_row(brain):
    b, fake, ctx, _mem = brain
    _turn(b, fake, GUEST_WORDS, MARA)
    assert _journal_rows(ctx) == [], _journal_rows(ctx)


def test_a_guests_chat_teaches_him_no_habit(brain):
    b, fake, _ctx, mem = brain
    _turn(b, fake, GUEST_WORDS, MARA)
    assert mem.habits == [], mem.habits


def test_a_guests_words_are_absent_from_his_next_payload(brain):
    """THE ONE THAT MAKES IT NOT WRITE-ONLY. add_exchange appends to the
    ring _dynamic_context renders, so her question came back as background
    on HIS next turn."""
    b, fake, _ctx, _mem = brain
    _turn(b, fake, GUEST_WORDS, MARA)
    _turn(b, fake, HIS_WORDS, scope_mod.OWNER, line="Two builds, sir.")
    his = json.dumps(fake.chat_payloads()[-1])
    assert "spare key" not in his, his[:400]
    assert "Mara" not in his and "mara" not in his, his[:400]


# ----------------------------------------------------------------- his own
def test_his_own_chat_still_lands_in_all_three(brain):
    b, fake, ctx, mem = brain
    _turn(b, fake, HIS_WORDS, scope_mod.OWNER, line="Two builds, sir.")
    assert _exchanges(ctx) == [(HIS_WORDS, "Two builds, sir.")]
    rows = _journal_rows(ctx)
    assert [(r["kind"], r["user"]) for r in rows] == [("exchange", HIS_WORDS)]
    assert mem.habits == [HIS_WORDS[:50]]


def test_his_own_words_do_reach_his_next_payload(brain):
    """The control for the payload test: the gate is on WHOSE turn it was,
    not on remembering at all."""
    b, fake, _ctx, _mem = brain
    _turn(b, fake, "the fan is loud", scope_mod.OWNER, line="Noted, sir.")
    _turn(b, fake, HIS_WORDS, scope_mod.OWNER, line="Two builds, sir.")
    assert "the fan is loud" in json.dumps(fake.chat_payloads()[-1])


def test_a_caller_that_passes_nothing_is_read_before_the_worker_waits(brain):
    """``chat(addressee=None)`` is an older call site: it takes the ambient
    reading, but ON THE CALLER'S THREAD. Round 3 measured what reading it
    after a wait costs; a gate flip while the worker runs must not decide
    whose record this is."""
    b, fake, ctx, mem = brain
    _say(fake, "Two builds, sir.")
    scope_mod.clear_addressee()
    t = b.chat(HIS_WORDS)                       # his turn, ambient owner
    scope_mod.set_addressee(*MARA)              # the gate names her mid-turn
    t.join(timeout=10)
    assert _exchanges(ctx) == [(HIS_WORDS, "Two builds, sir.")]
    assert mem.habits == [HIS_WORDS[:50]]


# ------------------------------------------------- the other two async doors
def test_think_does_not_file_a_guests_turn(brain, monkeypatch):
    """``_handle_known`` step 5 falls back to ``brain.think`` when the brain
    has no ``chat``; ``think`` remembers through the same ``_remember``."""
    b, _fake, ctx, mem = brain
    monkeypatch.setattr(b, "_query_ollama",
                        lambda text: [("SPEAK", "Ten degrees, ma'am.")])
    monkeypatch.setattr(b, "_query_claude",
                        lambda text: [("SPEAK", "Ten degrees, ma'am.")])
    scope_mod.set_addressee(*MARA)
    b.think(GUEST_WORDS)
    for t in list(getattr(b, "_threads", [])):
        t.join(timeout=10)
    _drain()
    assert _exchanges(ctx) == []
    assert _journal_rows(ctx) == []
    assert mem.habits == []


def test_web_answer_does_not_file_a_guests_turn(brain, monkeypatch):
    b, _fake, ctx, mem = brain
    monkeypatch.setattr(brain_mod.MACHINE, "claude_bin", "/bin/true")
    monkeypatch.setattr(b, "_run_claude", lambda *a, **k: "Ten degrees.")
    monkeypatch.setattr(brain_mod, "clean_web_answer", lambda out: "Ten degrees.")
    scope_mod.set_addressee(*MARA)
    t = b.web_answer("what's the weather like")
    if t:
        t.join(timeout=10)
    assert _exchanges(ctx) == []
    assert _journal_rows(ctx) == []
    assert mem.habits == []


def _drain():
    import threading
    import time
    for _ in range(100):
        if not any(t.name.startswith("brain-") for t in threading.enumerate()):
            return
        time.sleep(0.05)


# ------------------------------------------------------------- the source pin
def test_remember_is_told_whose_turn_it_is_and_never_looks_it_up(brain):
    """The seam, pinned in source: ``_remember`` takes the reading as an
    ARGUMENT. A future edit that re-reads the ambient scope inside the
    worker re-opens exactly the read-after-wait round 3 measured."""
    src = (ROOT / "jarvis" / "brain.py").read_text(encoding="utf-8")
    start = src.index("    def _remember(self")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    assert "addressee" in src[start:src.index("\n", start)], \
        "_remember takes no addressee"
    assert "scope_mod.addressee()" not in body, \
        "_remember reads the ambient scope after the worker's wait"
    # every call site hands it a reading
    for line in src.splitlines():
        if "self._remember(" in line:
            assert "addressee=" in line or "turn_addr" in line, line


# =====================================================================
# THE SWEEP: every OTHER writer into his record, and why each is safe.
# =====================================================================
# Four sinks were grepped across jarvis/*.py (the sink modules context.py
# and memory.py excluded): context.add_exchange, memory.log_habit,
# context.journal_debrief and memory.remember. Nine call sites. Two of
# them were _remember, fixed above. The census below fails if a TENTH
# appears, and the tests under it pin the reason each of the other seven
# cannot be reached by a guest -- so a future edit that opens one of those
# doors trips a test here rather than being found by another review round.

WRITER_NAMES = {"add_exchange", "log_habit", "journal_debrief", "remember"}
SINK_MODULES = {"context.py", "memory.py"}     # where the sinks are DEFINED

# (module, enclosing def) -> why a guest's words cannot arrive here
KNOWN_WRITERS = {
    ("app.py", "_async_reply"):
        "a guest starts no async Tier-1 worker (KNOWN_TIER1 is clock+math, "
        "both synchronous), and a late worker answer landing after somebody "
        "else's turn is stale by _dispatch_gen and returns before the write",
    ("app.py", "_on_brain_tags"):
        "_on_brain_tags' [DONE] branch; _chat_sync emits no DONE tag, and "
        "_handle_known reaches brain.think only when the brain has no chat",
    ("app.py", "_finish_claude_task"):
        "a Claude session finishing: proactive, his own session, no turn",
    ("app.py", "_after_dispatch"):
        "GATED on the turn's carried reading (his_turn)",
    ("brain.py", "_remember"):
        "GATED on the turn's carried reading (this file's subject)",
    ("commander.py", "_route_text"):
        "the owner path only: handle() returns _handle_known before it",
    ("debrief.py", "file_answer"):
        "its caller app._debrief_reply is GATED on the carried reading",
    ("garden.py", "run_pass"):
        "a background pass over his own journal; there is no turn",
}


def test_the_census_of_writers_into_his_record_is_still_eight():
    """Parsed, not grepped: a CALL node, so a doc line that merely names
    ``log_habit(command)`` is not counted as a writer."""
    import ast
    found = set()
    for path in sorted((ROOT / "jarvis").glob("*.py")):
        if path.name in SINK_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and \
                        isinstance(sub.func, ast.Attribute) and \
                        sub.func.attr in WRITER_NAMES:
                    found.add((path.name, node.name))
    new = found - set(KNOWN_WRITERS)
    assert not new, ("a writer into his record that nobody has scoped: %s"
                     % sorted(new))
    assert set(KNOWN_WRITERS) - found == set(), \
        "a writer went away; retire its entry: %s" % sorted(
            set(KNOWN_WRITERS) - found)


def test_a_guests_turn_never_reaches_the_commanders_habit_log(monkeypatch,
                                                              tmp_path):
    """``_route_text`` logs every sentence as one of HIS habits. It reads no
    scope; it does not need to, because ``handle`` returns ``_handle_known``
    above it. Pinned as behaviour, not by eye."""
    from unittest.mock import MagicMock
    from types import SimpleNamespace
    from jarvis.commander import Commander
    from jarvis.config import CONFIG

    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = SimpleNamespace(memory=MagicMock(), brain=MagicMock(),
                          desktop=MagicMock(), workflows=MagicMock(),
                          context=MagicMock(), tts=MagicMock())
    svc.workflows.get.return_value = None
    svc.desktop.parse_action = lambda part: None
    svc.memory.suggest_by_habit.return_value = None
    c = Commander(svc)

    c.handle("what did i leave running", source="typed", addressee=MARA)
    assert svc.memory.log_habit.call_args_list == [], \
        svc.memory.log_habit.call_args_list
    # and it went to the model with HER reading carried, not to his path
    assert svc.brain.chat.call_args.kwargs.get("addressee") == MARA

    c.handle("what did i leave running", source="typed",
             addressee=scope_mod.OWNER)
    assert svc.memory.log_habit.call_count == 1     # his own turn does log


def test_a_late_worker_answer_after_a_guests_turn_is_stale_and_writes_nothing(
        tmp_path, monkeypatch):
    """``_async_reply`` files ``self._last_user_text`` -- which ``_dispatch``
    has already overwritten with the NEXT speaker's words. The generation
    check is what stops a guest's sentence being filed under his answer."""
    from unittest.mock import MagicMock
    import threading
    import jarvis.app as app_mod

    a = object.__new__(app_mod.JarvisApp)
    a.context = MagicMock()
    a._turn_busy = threading.Event()
    a._turn_finished = lambda: None
    a._say = lambda *args, **kw: None
    a._dispatch_gen = 7                      # a newer turn (hers) owns the floor
    a._async_turn = (6, "voice")             # his explain job, dispatched before
    a._last_user_text = GUEST_WORDS          # _dispatch already overwrote it
    a._last_source = "voice"
    a._active_turn_id = ""
    a._async_reply("Here is what that document says, sir.")
    assert a.context.add_exchange.call_args_list == [], \
        a.context.add_exchange.call_args_list


def test_the_done_tag_door_has_no_guest_route():
    """``_on_brain_tags``'s [DONE] branch files ``_last_user_text`` ungated.
    Two independent reasons a guest cannot arrive there, both pinned so an
    edit to either trips."""
    brain_src = (ROOT / "jarvis" / "brain.py").read_text(encoding="utf-8")
    start = brain_src.index("    def _chat_sync(")
    end = brain_src.index("\n    def ", start + 1)
    assert '("DONE"' not in brain_src[start:end], \
        "_chat_sync now emits a DONE tag; the app files it ungated"
    cmd_src = (ROOT / "jarvis" / "commander.py").read_text(encoding="utf-8")
    start = cmd_src.index("    def _handle_known(")
    end = cmd_src.index("\n    def ", start + 1)
    body = cmd_src[start:end]
    assert 'not hasattr(brain, "chat")' in body and "brain.think(" in body, \
        "the think() fallback in _handle_known changed shape"
    app_src = (ROOT / "jarvis" / "app.py").read_text(encoding="utf-8")
    assert "def chat(text, force_tool=None, force_args=None, addressee=None)" \
        in app_src, "the app's brain namespace no longer defines chat"
