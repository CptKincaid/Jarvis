"""FOUND 2026-08-26 (routing sweep): "use sonnet for this one" silently
drops the model switch and answers locally.

jarvis.router._MODEL_RX ends with an optional tail:

    (?:\\s+(?:model|for (?:this|it|that)))?

On "use sonnet for this one" that tail greedily consumes "for this",
so the matched span is "use sonnet for this" and _strip_span leaves the
orphan word "one".  Router._action() is then asked to recognise the
remainder as a bare model switch with:

    re.fullmatch(r"(?:for (?:this|the next one|now)|please|from now on|"
                 r"on this one|for this one|for this task)?", t)

which lists "for this one" and "for this task" explicitly - the author
clearly intended both to work - but by now the text is "one", not
"for this one", so the fullmatch fails.  Routing falls through to rule 7
(a bare <= 6-word sentence with no cue) and returns:

    RouteDecision(kind='local', reason='short', args={})

Note args is EMPTY: the alias that _MODEL_RX did capture is discarded
with the rest of the decision, so nothing switches model and nothing is
said about it.

Measured with the real Router on the real AssistantConfig:

    use sonnet for this one   -> local  / short      args={}
    use opus for this one     -> local  / short      args={}
    use fable for this task   -> local  / short      args={}
    use sonnet for it         -> action / set_model  args={'alias': 'sonnet'}   (works)
    use sonnet please         -> action / set_model  args={'alias': 'sonnet'}   (works)

Defect: the optional tail in jarvis/router.py _MODEL_RX must not match a
prefix of a longer phrase that _action() is expecting whole (e.g. require
a word boundary that "for this one"/"for this task" cannot cross, or list
the longer phrases in _MODEL_RX itself).

FIXED 2026-08-26: the tail now reads
`for (?:this|it|that)(?:\\s+(?:one|task))?`, so it swallows the whole
phrase instead of a prefix of it and nothing orphaned is left behind.
This test is no longer xfail.
"""
import pytest

from jarvis.router import Router

SWALLOWED = [
    ("use sonnet for this one", "sonnet"),
    ("use opus for this one", "opus"),
    ("use fable for this task", "fable"),
    ("use haiku for this one", "haiku"),
]


@pytest.mark.parametrize("text,alias", SWALLOWED)
def test_use_model_for_this_one_switches_the_model(text, alias):
    d = Router(None, classify=None).route(text)
    assert d.kind == "action" and d.action == "set_model", (d.kind, d.reason)
    assert d.args.get("alias") == alias, d.args


def test_the_shorter_phrasings_still_work():
    """Control: the same intent one word shorter is handled correctly."""
    for text, alias in (("use sonnet for it", "sonnet"),
                        ("use opus please", "opus")):
        d = Router(None, classify=None).route(text)
        assert d.action == "set_model" and d.args.get("alias") == alias, (text, d)
