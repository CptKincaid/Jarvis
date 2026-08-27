"""FOUND 2026-08-26 (routing sweep): jarvis.router._PROJECT_RX captures a
bare determiner as a project NAME, so "in my repo" / "in this repo" is
slugified into a project called "my" / "this" and handed to
ClaudeSessionManager.submit(project=...), which does not know it.

Rule trace for "fix the parser in my repo":

    _PROJECT_RX = r"\\b(?:in|on|for|inside|under|within)\\s+(?:the\\s+)?
                   (?P<name>(?:[\\w-]+\\s+){0,2}[\\w-]+?)\\s+
                   (?:project|repo|repository|codebase)\\b"
    matches " in my repo" with name='my'  -> project = slugify('my') = 'my'
    work becomes 'fix the parser'
    => RouteDecision(kind='claude', project='my', prompt='fix the parser')

Measured end to end against the real ClaudeSessionManager built on the
user's real assistant config (projects: jarvis, haymaker-digest):

    project_for('my')      -> None
    project_for('this')    -> None
    project_for('our')     -> None
    project_for('current') -> None
    submit('fix the parser', project='my')
        -> "I don't know a project called my, sir; shall I set one up?"

So the user says "fix the parser in my repo" while a project is active
and Jarvis offers to create a project named "my" instead of doing the
work in the active project.

A second, quieter consequence of the same regex: when the project phrase
carries the ONLY code object, stripping it removes the coding cue -
"what is in my repo" strips to "what is", which is a short question and
routes LOCAL (reason 'local:question') instead of to Claude.

Defect: jarvis/router.py _PROJECT_RX has no exclusion for determiners /
possessives (my, our, this, that, the current, ...).

FIXED 2026-08-26: _PROJECT_RX now CONSUMES a run of determiners after the
preposition and refuses to CAPTURE one as <name>, so "in my repo" names no
project (the active one keeps the task) while "in my jarvis repo" still
captures "jarvis".  This test is no longer xfail.
"""
import pytest

from jarvis.router import Router

DETERMINERS = ["my", "this", "that", "our", "current"]


@pytest.mark.parametrize("det", DETERMINERS)
def test_a_determiner_is_not_a_project_name(det):
    noun = "codebase" if det == "that" else ("project" if det == "current"
                                             else "repo")
    text = f"fix the parser in {det} {noun}"
    d = Router(None, classify=None).route(text)
    assert d.kind == "claude", (text, d.kind, d.reason)
    assert d.project == "", (text, d.project)


def test_a_question_about_the_repo_still_reaches_claude():
    d = Router(None, classify=None).route("what is in my repo")
    assert d.kind == "claude", (d.kind, d.reason, d.prompt)
