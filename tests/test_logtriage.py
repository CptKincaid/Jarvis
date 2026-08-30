"""Jarvis reading his own log (jarvis/logtriage.py): clustering the
WARNING/ERROR tail with tracebacks attached, the ledger's last turn, and
the two commands that speak them.  Pure: fixture lines and tmp files."""
import json
from types import SimpleNamespace

import jarvis.app as app_mod
import jarvis.logs as logs_mod
from jarvis import logtriage as lt
from jarvis.commander import (_LOGTRIAGE_RX, _SLOW_RX, ASSISTANT_TIER1, _h_log_triage,
                              _h_slow_turn)
from jarvis.config import PATHS
from jarvis.events import JarvisReply, bus

LOG = """\
23:58:01.100 jarvis.tools.registry WARNING 12 tools registered (budget 11)
23:58:02.200 jarvis.memory WARNING /home/hunterp/.aiws_trainer/jarvis_data.migrated already exists; leaving legacy dir in place
23:58:03.300 jarvis.app INFO handle 'what time is it' source=voice
23:59:10.400 jarvis.app ERROR audio processing failed
Traceback (most recent call last):
  File "/home/hunterp/Jarvis/jarvis/app.py", line 850, in _process_audio
    text = self.transcriber.transcribe(audio)
RuntimeError: model exploded
23:59:40.500 jarvis.tools.registry WARNING 12 tools registered (budget 11)
00:00:05.600 jarvis.app ERROR audio processing failed
Traceback (most recent call last):
  File "/home/hunterp/Jarvis/jarvis/app.py", line 850, in _process_audio
    text = self.transcriber.transcribe(audio)
RuntimeError: model exploded again
00:01:00.700 jarvis.speaker WARNING voiceprint predates silence trimming (format 2 < 3); re-enrol
00:02:00.800 jarvis.app WARNING turn watchdog fired after 45s; releasing the wake word
00:02:30.900 jarvis.app WARNING turn watchdog fired after 61s; releasing the wake word
00:03:00.000 jarvis.tools.spotify WARNING spotify: no device available
"""


def _lines():
    return LOG.splitlines()


# ------------------------------------------------------------ clustering
def test_tracebacks_attach_to_the_record_before_them():
    recs = lt.parse_records(_lines())
    errs = [r for r in recs if r.level == "ERROR"]
    assert len(errs) == 2
    assert errs[0].extra[0] == "Traceback (most recent call last):"
    assert errs[0].extra[-1] == "RuntimeError: model exploded"
    assert errs[1].extra[-1] == "RuntimeError: model exploded again"


def test_clusters_fold_numbers_and_drop_the_known_noise():
    """The tool-budget warning was 650 of one day's 774 W/E lines; without
    the noise list it would head every answer.  The two watchdog lines
    differ only in a number and cluster together; the two audio errors
    straddle midnight and still cluster (the line has no date, so the
    window is the tail of the file, never a clock range)."""
    clusters = lt.cluster_warnings(_lines())
    names = [(c.short_logger, c.count) for c in clusters]
    assert names[0] == ("app", 2) and clusters[0].example == "audio processing failed"
    assert ("app", 2) in names[1:] or names[0] == ("app", 2)
    watchdog = next(c for c in clusters if "watchdog" in c.message)
    assert watchdog.count == 2 and watchdog.message == "turn watchdog fired after Ns; releasing the wake word"
    assert watchdog.last_time == "00:02:30.900"
    assert not any("tools registered" in c.message for c in clusters)
    assert not any(c.logger == "jarvis.memory" for c in clusters)
    assert clusters[0].traceback.endswith("RuntimeError: model exploded again")
    assert clusters[0].level == "ERROR"
    assert [c.short_logger for c in clusters][-2:] == ["speaker", "tools.spotify"] or \
        {c.short_logger for c in clusters} >= {"speaker", "tools.spotify"}


def test_the_limit_is_a_line_tail_not_a_clock_window():
    clusters = lt.cluster_warnings(_lines(), limit=2)
    assert sorted(c.short_logger for c in clusters) == ["app", "tools.spotify"]
    assert all(c.count == 1 for c in clusters)       # only the second watchdog line


def test_normalise_folds_paths_ids_and_numbers():
    assert lt.normalise("wrote /tmp/x/123.txt in 4.5s id 0badf00dcafe") == \
        "wrote <path> in Ns id <id>"
    assert lt.normalise("  a   b ") == "a b"


# --------------------------------------------------------------- wording
def test_triage_text_speaks_two_sentences_and_cards_the_traceback():
    clusters = lt.cluster_warnings(_lines())
    spoken, card = lt.triage_text(clusters, {"turns": 5, "slow": 2, "lost": 1, "wait_over": 5.0},
                                  examined=len(_lines()))
    assert spoken.startswith("Four things in the log, sir: 'audio processing failed' from app twice")
    assert "two more on the card" in spoken and spoken.count(". ") == 1
    assert spoken.endswith("On the ledger, two turns waited over 5 seconds and one turn never got an answer.")
    assert "RuntimeError: model exploded again" in card
    assert "2× app ERROR: audio processing failed" in card
    assert "ledger: 5 turns, 2 over 5 s, 1 without an answer" in card


def test_a_clean_log_says_so():
    spoken, card = lt.triage_text([], {"turns": 3, "slow": 0, "lost": 0, "wait_over": 5.0}, examined=40)
    assert spoken == "Nothing wrong in the log, sir; the last 40 lines are clean."
    assert card.startswith("Log triage")


# ---------------------------------------------------------------- ledger
def _ledger(tmp_path, *recs):
    p = tmp_path / "turns.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_last_turn_skips_aborts_and_outliers_count(tmp_path):
    p = _ledger(tmp_path,
                {"outcome": "audio", "wait": 6.3, "dead_air": 2.7, "stt": 0.7, "route": 2.9, "stop": "energy"},
                {"outcome": "uncertain", "speech": 2.5, "stop": "vad"},
                {"outcome": "abort"})
    rec = lt.last_turn(p)
    assert rec["outcome"] == "uncertain"
    assert lt.turn_outliers(lt.read_turns(p)) == {"turns": 2, "slow": 1, "lost": 1, "wait_over": 5.0}
    assert lt.last_turn(tmp_path / "missing.jsonl") is None


def test_slow_text_names_the_slow_stage():
    rec = {"outcome": "audio", "wait": 6.3, "dead_air": 2.7, "stt": 0.7, "route": 2.9,
           "stop": "energy", "filler": None}
    line = lt.slow_text(rec)
    assert line.startswith("The last turn waited 6.3 seconds, sir: 2.7 seconds silence before the "
                           "recorder stopped, 0.7 seconds transcribing, 2.9 seconds working out the answer")
    assert "the answer was the slow part." in line
    assert "energy timer" not in line                # silence was not the culprit
    rec.update(route=1.0, filler=1.5)
    line = lt.slow_text(rec)
    assert "the silence was the slow part. The energy timer stopped it, not the voice detector." in line
    assert line.endswith("A filler line went out at 1.5 seconds.")
    assert lt.slow_text(None) == "I have no turn on the ledger yet, sir."
    assert lt.slow_text({"outcome": "uncertain", "speech": 2.5}) == \
        "The last turn never got an answer, sir: I wasn't sure it was for me after 2.5 seconds of speech."


# -------------------------------------------------------------- commands
def test_the_phrasings():
    for said in ("anything wrong in your log?", "is there anything wrong in your logs",
                 "check your log", "read the log for errors", "any errors in your log today",
                 "what's in your log", "log triage", "triage the logs"):
        assert _LOGTRIAGE_RX.match(said), said
    for said in ("what went wrong", "any errors", "check the build log"):
        assert not _LOGTRIAGE_RX.match(said), said
    for said in ("why was that slow?", "why did that take so long", "what took so long",
                 "where did the time go", "why so slow", "why was the last one so slow",
                 "break down that turn"):
        assert _SLOW_RX.match(said), said
    assert not _SLOW_RX.match("why is the build slow")
    names = [c.name for c in ASSISTANT_TIER1]
    assert "log triage" in names and "slow turn" in names       # unprefixed voice reaches them


def test_the_handlers_speak_the_line_and_card_the_detail():
    shown = []
    unsub = bus.subscribe(JarvisReply, lambda ev: shown.append((ev.text, ev.speak)))
    try:
        c = SimpleNamespace(_svc=lambda n: (lambda: ("Two things, sir.", "2× app ERROR: x\n    Traceback"))
                            if n == "log_triage" else None)
        res = _h_log_triage(c, "anything wrong in your log", None)
        assert res.handled and res.speak and res.reply == "Two things, sir."
        assert shown == [("2× app ERROR: x\n    Traceback", False)]
        assert _h_log_triage(SimpleNamespace(_svc=lambda n: None), "log triage", None) is None
        c = SimpleNamespace(_svc=lambda n: (lambda: "The last turn waited 6.3 seconds, sir.")
                            if n == "slow_turn" else None)
        res = _h_slow_turn(c, "why was that slow", None)
        assert res.speak and res.reply.startswith("The last turn waited")
    finally:
        bus.unsubscribe(JarvisReply, unsub) if callable(unsub) else None


# ------------------------------------------------------------------ app
def test_the_app_reads_its_own_log_and_ledger(tmp_path, monkeypatch):
    log_file = tmp_path / "jarvis.log"
    log_file.write_text(LOG)
    monkeypatch.setattr(logs_mod, "LOG_FILE", log_file)
    monkeypatch.setattr(PATHS, "LOG_DIR", tmp_path)
    _ledger(tmp_path, {"outcome": "audio", "wait": 1.3, "dead_air": 0.8, "stt": 0.3, "route": 0.2})
    app = app_mod.JarvisApp.__new__(app_mod.JarvisApp)
    spoken, card = app.log_triage_text()
    assert spoken.startswith("Four things in the log, sir: 'audio processing failed' from app twice")
    assert "The ledger's turns all answered promptly." in spoken
    assert "RuntimeError: model exploded again" in card
    assert app.slow_turn_text().startswith("The last turn waited 1.3 seconds, sir")
    # the context engine's error slice comes from the same clustering
    from jarvis.context import ContextEngine
    monkeypatch.setattr("jarvis.context.LOG_FILE", log_file)
    ctx = ContextEngine.__new__(ContextEngine)
    errs = ContextEngine._get_recent_errors(ctx)
    assert errs[0] == "app ERROR: audio processing failed (x2)"
    assert not any("tools registered" in e for e in errs)
