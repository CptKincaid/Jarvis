"""Keyword watch (jarvis/keyword_watch.py): whole-word matching over unread
mail and Canvas announcements, one line per hit, dedupe across ticks, and
total silence with an empty keyword list or a missing credential. Both
sources are callable seams; nothing here touches IMAP or Canvas.
"""
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from jarvis.keyword_watch import KeywordWatch
from jarvis.tools.canvas import CanvasError
from jarvis.tools.mail import Mail, MailNotConfigured
from jarvis.watchers import keyword_hit

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)
CFG = {"gmail": {"address": "hunter@example.com", "app_password": "abcd efgh ijkl"},
       "canvas": {"token": "7~abcDEF123secret", "base_url": "https://canvas.tamu.edu"},
       "watch": {"keywords": ["biosensors", "REU application"]}}


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)


def _mail(subject, snippet="", addr="pi@tamu.edu", name="Dr. Villalobos", minutes=5):
    return Mail(from_name=name, from_addr=addr, subject=subject,
                date=NOW - timedelta(minutes=minutes), snippet=snippet)


def _ann(title, snippet="", course="BIOSENSORS", hours=2):
    return {"course": course, "title": title, "posted": NOW - timedelta(hours=hours),
            "snippet": snippet}


def _make(tmp_path, mails=(), anns=(), cfg=CFG, state="kw.json"):
    said = []
    calls = []

    def fetch_unread(cfg_arg, since_hours=24, limit=20):
        calls.append("mail")
        if isinstance(mails, Exception):
            raise mails
        return list(mails)

    def fetch_announcements(settings, days, fetch, now):
        calls.append("canvas")
        if isinstance(anns, Exception):
            raise anns
        return list(anns)

    w = KeywordWatch(cfg, announce=lambda title, text: said.append((title, text)),
                     state_path=tmp_path / state, now=lambda: NOW,
                     fetch_unread=fetch_unread,
                     fetch_announcements=fetch_announcements)
    return w, said, calls


# ------------------------------------------------------------- matching
def test_whole_word_matching_only():
    kws = ("ai", "lab")
    assert keyword_hit("The AI seminar is Friday", kws) == "ai"
    assert keyword_hit("Say that again, please", kws) == ""
    assert keyword_hit("collaboration notes", kws) == ""
    assert keyword_hit("Lab 3 report", kws) == "lab"
    assert keyword_hit("", kws) == "" and keyword_hit("anything", ()) == ""


def test_a_phrase_matches_as_a_phrase():
    assert keyword_hit("your REU application is in", ("reu application",)) == \
        "reu application"
    assert keyword_hit("REU deadline moved", ("reu application",)) == ""


# ---------------------------------------------------------------- hits
def test_a_mail_hit_names_the_keyword_the_subject_and_the_sender(tmp_path):
    w, said, _c = _make(tmp_path, mails=[_mail("Meeting about biosensors")])
    assert w.tick() == 1
    assert said == [("Keyword watch",
                     "Mail about biosensors, sir: Meeting about biosensors, "
                     "from Dr. Villalobos.")]


def test_the_snippet_counts_not_only_the_subject(tmp_path):
    w, said, _c = _make(tmp_path, mails=[_mail("quick question",
                                               "your REU application looks good")])
    assert w.tick() == 1 and "reu application" in said[0][1]


def test_an_announcement_hit_names_the_course(tmp_path):
    w, said, _c = _make(tmp_path, anns=[_ann("Project groups posted",
                                             "biosensors teams are up")])
    assert w.tick() == 1
    assert said[0][1] == ("Canvas announcement about biosensors, sir: "
                          "BIOSENSORS — Project groups posted.")


def test_both_sources_are_read_in_one_tick(tmp_path):
    w, said, calls = _make(tmp_path, mails=[_mail("re: biosensors")],
                           anns=[_ann("biosensors lab moved")])
    assert w.tick() == 2
    assert calls == ["mail", "canvas"]


def test_a_non_matching_message_says_nothing(tmp_path):
    w, said, _c = _make(tmp_path, mails=[_mail("Parking permit renewal")],
                        anns=[_ann("Office hours changed", course="CIRCUITS")])
    assert w.tick() == 0 and said == []


# --------------------------------------------------------------- dedupe
def test_a_hit_is_filed_once_and_survives_a_restart(tmp_path):
    mails = [_mail("re: biosensors")]
    w, said, _c = _make(tmp_path, mails=mails)
    assert w.tick() == 1
    assert w.tick() == 0
    w2, said2, _c2 = _make(tmp_path, mails=mails)
    assert w2.tick() == 0 and said2 == []
    assert list(json.loads((tmp_path / "kw.json").read_text()))[0].startswith(
        "mail|pi@tamu.edu|re: biosensors|")


def test_the_cap_holds_the_tail_back_and_still_marks_it_seen(tmp_path):
    mails = [_mail(f"biosensors note {i}", minutes=i) for i in range(1, 7)]
    w, said, _c = _make(tmp_path, mails=mails)
    assert w.tick() == 3
    assert w.tick() == 0, "a held-back hit must not come round every 15 minutes"


def test_a_sink_failure_does_not_repeat_the_line(tmp_path):
    def boom(title, text):
        raise RuntimeError("discord down")

    w = KeywordWatch(CFG, announce=boom, state_path=tmp_path / "kw.json",
                     now=lambda: NOW,
                     fetch_unread=lambda *a, **k: [_mail("re: biosensors")],
                     fetch_announcements=lambda *a, **k: [])
    assert w.tick() == 0
    assert w.tick() == 0


# -------------------------------------------------------------- silence
def test_an_empty_keyword_list_reads_nothing_at_all(tmp_path):
    w, said, calls = _make(tmp_path, mails=[_mail("re: biosensors")],
                           cfg=dict(CFG, watch={"keywords": []}))
    assert w.tick() == 0 and calls == []
    # a missing watch section behaves the same way
    cfg = {k: v for k, v in CFG.items() if k != "watch"}
    w2, _s2, calls2 = _make(tmp_path, mails=[_mail("re: biosensors")], cfg=cfg)
    assert w2.tick() == 0 and calls2 == []


def test_a_bare_string_keyword_counts_as_a_one_item_list(tmp_path):
    w, said, _c = _make(tmp_path, mails=[_mail("re: biosensors")],
                        cfg=dict(CFG, watch={"keywords": "Biosensors"}))
    assert w.tick() == 1


def test_each_source_is_skipped_on_its_own_when_unconfigured(tmp_path):
    cfg = dict(CFG, canvas={"token": ""})
    w, said, calls = _make(tmp_path, mails=[_mail("re: biosensors")],
                           anns=[_ann("biosensors")], cfg=cfg)
    assert w.tick() == 1 and calls == ["mail"]
    cfg2 = dict(CFG, gmail={"address": "", "app_password": ""})
    w2, said2, calls2 = _make(tmp_path, mails=[_mail("re: biosensors")],
                              anns=[_ann("biosensors lab")], cfg=cfg2, state="k2.json")
    assert w2.tick() == 1 and calls2 == ["canvas"]


def test_an_outage_on_one_source_leaves_the_other_running(tmp_path):
    w, said, _c = _make(tmp_path, mails=OSError("connection reset"),
                        anns=[_ann("biosensors lab moved")])
    assert w.tick() == 1 and said[0][1].startswith("Canvas announcement")
    w2, said2, _c2 = _make(tmp_path, mails=[_mail("re: biosensors")],
                           anns=CanvasError("unreachable", "timeout"),
                           state="k2.json")
    assert w2.tick() == 1 and said2[0][1].startswith("Mail about")
    w3, said3, _c3 = _make(tmp_path, mails=MailNotConfigured("gone"),
                           state="k3.json")
    assert w3.tick() == 0 and said3 == []


# ---------------------------------------------------------------- thread
def test_the_thread_is_joinable_and_restartable(tmp_path):
    w, _said, _c = _make(tmp_path)
    w.start()
    t1 = w._thread
    assert t1 is not None and t1.is_alive()
    w.start()
    assert w._thread is t1, "start is idempotent"
    w.stop()
    assert not t1.is_alive(), "stop() joins the thread out"
    w.start()
    assert w._thread is not t1 and w._thread.is_alive()
    w.stop()
