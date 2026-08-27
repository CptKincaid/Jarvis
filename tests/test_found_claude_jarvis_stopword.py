"""FOUND 2026-08-26 (Claude-session sweep): "jarvis" is in the session-
matching stopword list, so the user's own ~/Jarvis repo can never be
picked by name when resuming.

jarvis/claude_session.py:_STOPWORDS (~line 134) ends with

    ... you your jarvis claude let lets keep going back

The word makes sense as an address ("Jarvis, resume ...") but _tokens()
is also what pick_session() (7.5) and _resolve_candidate() use to score a
session's *name*:

    _tokens("pick up the jarvis session") == set()

so the +0.6 name-match and the +0.2 body-match bonuses are both
unreachable for the path ~/Jarvis — the very repo the shipped
assistant.json lists first in claude.allowed_dirs.  Ranking collapses to
pure recency and the wrong session wins; the two-candidate question
cannot be answered with "the jarvis one" either (_resolve_candidate
returns None).

Measured with two synthetic sessions (~/Jarvis 3 days old, ~/haymaker-
digest 1 minute old): "pick up the jarvis session" scores jarvis 0.0669
vs haymaker-digest 0.2999 and resumes haymaker-digest, while "pick up the
haymaker digest session" correctly scores 1.0999 for the right one.

Fix: drop "jarvis" from _STOPWORDS and strip the address separately (it
is already stripped upstream by the wake word), or keep a stopword list
for the *body* match only and score names against the raw tokens.

FIXED 2026-08-26: "jarvis" is out of claude_session._STOPWORDS and the
vocative form only ("hey jarvis, ...", "jarvis: ...", "..., jarvis") is
stripped by claude_session._ADDRESS_RX inside _tokens().  A bare
mid-sentence "jarvis" is now a project name again.
"""
import os
import time

from jarvis import claude_session as cs
from jarvis.claude_session import SessionInfo


def _pair(now):
    return [SessionInfo("s1", os.path.expanduser("~/Jarvis"), now - 3 * 86400,
                        first_user="fix the hotword", turns=9),
            SessionInfo("s2", os.path.expanduser("~/haymaker-digest"), now - 60,
                        first_user="run the digest", turns=9)]


def test_naming_the_jarvis_project_picks_it():
    now = time.time()
    sessions = _pair(now)
    best, _runner, _q = cs.pick_session("pick up the jarvis session", sessions, now)
    assert best.cwd.endswith("/Jarvis"), {s.slug: s.score for s in sessions}


def test_naming_any_other_project_works():
    """Control: the same utterance shape resolves a project whose name is
    not a stopword."""
    now = time.time()
    sessions = _pair(now)
    best, _runner, _q = cs.pick_session("pick up the haymaker digest session",
                                        sessions, now)
    assert best.cwd.endswith("/haymaker-digest")


def test_answering_the_two_candidate_question_with_the_jarvis_one():
    now = time.time()
    a, b = _pair(now)
    mgr = object.__new__(cs.ClaudeSessionManager)      # no I/O; pure helper
    assert cs.ClaudeSessionManager._resolve_candidate(mgr, "the jarvis one",
                                                      (a, b)) is a


def test_the_address_form_is_not_read_as_the_project_name():
    """Control for the fix: "Jarvis," as an ADDRESS must not score the
    ~/Jarvis repo — only a bare mention of the name does."""
    now = time.time()
    for utterance in ("jarvis, pick up the haymaker digest session",
                      "hey jarvis, pick up the haymaker digest session",
                      "pick up the haymaker digest session, jarvis"):
        sessions = _pair(now)
        best, _runner, _q = cs.pick_session(utterance, sessions, now)
        assert best.cwd.endswith("/haymaker-digest"), (
            utterance, {s.slug: s.score for s in sessions})
    assert cs._tokens("jarvis, fix the parser") == {"fix", "parser"}
    assert "jarvis" in cs._tokens("the jarvis one")
