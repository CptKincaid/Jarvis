"""FOUND 2026-08-26 (adversarial routing re-verification): when an utterance
that carries an explicit model choice falls through to rule 6 and Jarvis asks
"Shall I hand that to Claude, sir?", answering "yes" runs the task on the
DEFAULT model — the alias is silently discarded.

Router._tie_break() returns the ask decision with args intact, but remembers
only three fields:

    self._pending = PendingAsk(text=work, project=..., at=self._now())

PendingAsk has no args field (text / project / at), so the model, parallel and
size arguments die with the decision.  commander._try_router_answer() then
rebuilds the decision from what was kept:

    d = RouteDecision(kind=kind, reason="answer", prompt=pend.text,
                      project=pend.project)          # <-- args defaults to {}

and dispatches that, so submit() is called with model=None.

Measured with the real Router + the real ClaudeSessionManager on the user's
real assistant config (spawn seam blocked, no CLI invoked):

    'the calendar module is broken, use haiku'
        ask decision args={'model': 'haiku'}
        -> answer 'claude' -> submit(model=None) -> task model 'opus'

    ... the same for 'use sonnet' and 'use fable'; model_for(prompt, None)
    is 'opus', so the user asks for the cheap model and gets the expensive one.

Defect: jarvis/router.py PendingAsk / Router._tie_break() (the args are not
carried), with the consequence at jarvis/commander.py _try_router_answer().

FIXED 2026-08-26: PendingAsk carries an `args` dict, _tie_break() fills it,
and commander._try_router_answer() rebuilds the decision with it, so "yes"
runs the task on the model the user named.  This test is no longer xfail.
"""
import pytest

from jarvis.router import Router

ALIASES = ["haiku", "sonnet", "fable"]


@pytest.mark.parametrize("alias", ALIASES)
def test_the_remembered_question_keeps_the_model_choice(alias):
    r = Router(None, classify=None)
    d = r.route(f"the calendar module is broken, use {alias}")
    assert d.kind == "ask", (d.kind, d.reason)
    assert d.args.get("model") == alias, d.args
    pend = r.pending()
    assert pend is not None
    # What the commander can still see when the answer arrives:
    assert getattr(pend, "args", {}).get("model") == alias, (
        "PendingAsk fields: %s" % list(pend.__dataclass_fields__))


def test_the_ask_decision_itself_does_capture_the_alias():
    """Control: the alias IS captured on the way in, so the loss is in what
    PendingAsk remembers, not in the model regex."""
    r = Router(None, classify=None)
    d = r.route("the calendar module is broken, use haiku")
    assert d.kind == "ask", (d.kind, d.reason)
    assert d.args.get("model") == "haiku", d.args
