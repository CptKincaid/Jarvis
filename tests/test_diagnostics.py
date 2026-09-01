"""'Run diagnostics': a status line in character, from real data."""
from types import SimpleNamespace

from jarvis import selfstate
from jarvis.commander import _DIAG_RX, _h_diagnostics
from jarvis.events import JarvisReply, bus


def test_the_phrasings():
    for said in ("run diagnostics", "diagnostics", "run a diagnostic", "system status",
                 "status report", "self test", "how are your systems"):
        assert _DIAG_RX.match(said), said
    assert not _DIAG_RX.match("what's the status of the build")


def test_the_handler_speaks_what_the_app_reports():
    c = SimpleNamespace(_svc=lambda n: (lambda: "All systems nominal, sir.") if n == "diagnostics" else None)
    res = _h_diagnostics(c, "run diagnostics", None)
    assert res.handled and res.speak and res.reply == "All systems nominal, sir."
    assert _h_diagnostics(SimpleNamespace(_svc=lambda n: None), "diagnostics", None) is None


# ----------------------------------------------------- one answer per turn
def _sheet():
    """Enough of app.self_state() for both renderings to differ."""
    return {"uptime_s": 42240, "stt_model": "turbo", "brain_model": "gemma4",
            "tts_engine": "f5", "gpu_temp_c": 42.0, "gpu_mhz": 2418.0,
            "gpu_util_pct": 0.0, "mem_free_gb": 74.0, "mem_total_gb": 122.0,
            "turns_today": 0, "claude_panes": 1, "claude_working": 1,
            "voiceprint_samples": 6, "brain_resident": True}


def _commander(register="normal"):
    sheet = _sheet()
    brain = SimpleNamespace(register=lambda: register)
    services = {"diagnostics": lambda: selfstate.diagnostics_line(sheet),
                "self_state": lambda full=True: sheet, "brain": brain}
    return SimpleNamespace(_svc=services.get), sheet


def test_one_utterance_gets_exactly_one_answer():
    """His note of 2026-08-31 -- "(didnt say this but in transcript) said
    this" -- reproduced on 2026-09-01: `jarvis --quiet "run diagnostics"`
    printed TWO whole status sentences, the plain sheet then the film line.

    The handler was publishing the plain sheet as a companion "card", but
    the bus has no card: a JarvisReply IS the answer, drawn by the UI as a
    Jarvis bubble (ui/main_window._ev_reply) and streamed by cmdsock as
    {"kind": "reply"}. So the second render arrived as a second answer, in
    a different voice, and he heard one while reading the other.
    """
    c, sheet = _commander()
    seen = []

    def grab(ev):
        seen.append(ev)

    bus.subscribe(JarvisReply, grab)
    try:
        res = _h_diagnostics(c, "run diagnostics", None)
    finally:
        bus.unsubscribe(JarvisReply, grab)

    assert res.handled and res.speak
    assert res.reply == selfstate.stark_line(sheet)
    assert seen == [], (
        "the handler published a second, differently-worded answer for the "
        f"same turn: {[e.text for e in seen]}")


def test_the_film_register_is_the_answer_and_the_plain_sheet_is_not_lost():
    """Which voice wins: the film line he actually heard from the room.
    The plain sheet is still what `jarvis status` (cmdsock), ask.py
    --status, the phone and the formal register render -- off this very
    dict -- so nothing is dropped by not saying it twice."""
    c, sheet = _commander()
    assert _h_diagnostics(c, "run diagnostics", None).reply == \
        selfstate.stark_line(sheet)
    formal, sheet = _commander(register="formal")
    assert _h_diagnostics(formal, "run diagnostics", None).reply == \
        selfstate.diagnostics_line(sheet)
