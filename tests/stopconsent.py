"""Classify every answer-grammar SITE by WHICH DIRECTION IS UNSAFE.

The census (tests/answercensus.py) derives WHERE his words meet a
grammar.  This module asks the question round 4 needed and round 3 did
not have a name for: at each of those sites, which way does a mistake
fall?

  STOP     -- NOT matching is the unsafe direction.  The alarm goes on
              ringing, the camera goes on filming, the flashcard is
              marked wrong, the working session swallows a stop as an
              answer, a send read-back stays armed for its ninety
              seconds.  A stop word with a "?" or an "…" hung on it must
              still match.
  CONSENT  -- MATCHING is the unsafe direction, and the act cannot be
              taken back: the file is sent, the permission granted, the
              terminal opened, the biometric gallery replaced.  A rising
              hesitated form must NOT match -- either the grammar refuses
              it, or a named BAR refuses it for the whole lane.
  NEITHER  -- both directions are inert.  A refusal ("no", "nope"), a
              lead-stripping extractor, a normaliser, or a reversible
              offer whose worst case is that Jarvis says something.

WHY THIS EXISTS.  strip_fillers gained a "?" carry in round 3 so that
"yes, uh?" could not send a file.  The carry lives in the CANONICAL
strip, so it also reached every STOP grammar, and those ended on
``[.!\\s]*$``.  The fixer saw the shape on ONE rung -- the ringing alarm
-- pinned it as an open question, and never enumerated its twins.  That
is the guard-one-half pattern, and it was its FIFTH occurrence.  A rule
that only lives in a report will be half-applied again; this module makes
it a derivation, and test_stop_consent_mirror.py makes it a pin.

Nothing here imports jarvis at module scope beyond the compiled patterns
the test hands it, and nothing here runs a rung.  Every witness is a
string written in this file.
"""
from __future__ import annotations

import ast
import importlib
import io
import pathlib
from dataclasses import dataclass
from typing import Optional

#: Words whose whole meaning is HALT SOMETHING THAT IS ALREADY RUNNING.
#: Deliberately free of "okay" / "go on" / "sure" / "next" / "ready",
#: which are a stop to one rung and a consent to another and so decide
#: nothing; a site that matches only those falls to the KNOWN table.
HALT_WITNESSES = (
    "stop", "stop it", "stop that", "stop talking", "cancel", "cancel that",
    "abort", "quiet", "be quiet", "hush", "silence", "enough",
    "that's enough", "never mind", "forget it", "that'll do", "skip it",
    "no idea", "not sure", "i don't know", "undo", "undo that",
    "scratch that", "belay that", "dismiss", "snooze", "i'm up",
    "turn it off", "shut up", "end notes", "stop notes", "stop the quiz",
    "we're done", "leave it there", "wait", "hold on", "not yet",
    "five more minutes", "pass",
)
#: Words that GRANT.  Only where a grant cannot be taken back does the
#: site become CONSENT; the KNOWN table says which those are.
GRANT_WITNESSES = (
    "yes", "yeah", "yep", "yup", "allow it", "approve", "open it",
    "send it", "enrol", "enroll", "affirmative", "permitted", "allowed",
)
#: A bare refusal.  Neither matching nor missing it does anything.
REFUSE_WITNESSES = ("no", "nope", "nah", "deny", "denied", "decline")

#: The punctuation a filled pause carries onto the word in front of it
#: (jarvis.endpoint.strip_fillers): "stop, uh?" -> "stop?", "stop, uh…"
#: -> "stop…".  "..." is absent on purpose -- three ASCII dots always
#: passed ``[.!]*$``, which is exactly why the "…" hid for a round.
CARRIED = ("?", "…")

STOP, CONSENT, NEITHER = "STOP", "CONSENT", "NEITHER"


@dataclass(frozen=True)
class Verdict:
    key: str
    kind: str                 # STOP | CONSENT | NEITHER
    why: str                  # "derived: ..." or "known: ..."
    witness: Optional[str]    # the word the mirror rule is checked on
    tolerates: dict           # punctuation -> bool, empty when no witness


def _regex_for(root: pathlib.Path, module: str, name: str):
    """The compiled pattern a site names, or None for an inline regex, a
    ``startswith`` or an ``in`` over a word set -- those have no terminal
    class of their own and are settled by the KNOWN table."""
    if name.startswith("re.") or "(" in name:
        return None
    mod = "jarvis." + module[: -len(".py")].replace("/", ".")
    try:
        m = importlib.import_module(mod)
    except Exception:                         # noqa: BLE001 - a module that
        return None                           # will not import is not a site
    if "." in name:                           # a regex held on a class
        cls, attr = name.split(".", 1)
        obj = getattr(getattr(m, cls, None), attr, None)
    else:
        obj = getattr(m, name, None)
    return obj if hasattr(obj, "pattern") else None


_SRC: dict = {}


def function_source(root: pathlib.Path, module: str, function: str) -> str:
    """The source of the rung a site sits in -- docstring and comments,
    which is where "stop is the safe direction" is written down."""
    path = root / module
    if path not in _SRC:
        _SRC[path] = io.open(path, encoding="utf-8").read()
    text = _SRC[path]
    want = function.split(".")[-1]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == want:
            return ast.get_source_segment(text, node) or ""
    return ""


def known_row(known: dict, key: str):
    """The KNOWN row for a site key: exact, or by PREFIX.

    A site whose grammar is not a named regex carries the grammar's own
    text in its key -- ``re.sub('^(?:my|our|the)\\s+')``,
    ``startswith(('/', '~'))``, ``in('wake','wakeup',\u2026)``. Spelling
    those out in a table means escaping a regex inside a Python string
    inside a dict key, which is how a table row silently stops matching
    the site it was written for. The table names them up to the opening
    bracket instead.
    """
    if key in known:
        return known[key]
    for k, row in known.items():
        if k.endswith("(") and key.startswith(k):
            return row
    return None


def classify(root: pathlib.Path, site, known: dict) -> Verdict:
    """One site's verdict.

    DERIVED where the grammar's own vocabulary settles it: it matches
    halt words and no grant words (STOP), or grant words and no halt
    words (CONSENT).  Everything else -- an ambiguous head, a refusal, a
    lead-stripping extractor, a grammar that is not a regex -- must be in
    ``known``, so a NEW site is never silently unclassified.
    """
    rx = _regex_for(root, site.module, site.regex)
    row = known_row(known, site.key)
    if rx is None:
        if row is None:
            return Verdict(site.key, "", "UNCLASSIFIED: not a regex and not known",
                           None, {})
        return Verdict(site.key, row[0], f"known: {row[1]}", None, {})

    halt = [w for w in HALT_WITNESSES if rx.match(w)]
    grant = [w for w in GRANT_WITNESSES if rx.match(w)]
    refuse = [w for w in REFUSE_WITNESSES if rx.match(w)]
    body = function_source(root, site.module, site.function)
    says_safe = "safe direction" in body

    if halt and not grant:
        kind, why, witness = STOP, "derived: halt words only", halt[0]
        if says_safe:
            why += "; the rung says 'safe direction'"
    elif grant and not halt:
        kind, why, witness = CONSENT, "derived: grant words only", grant[0]
    else:
        if row is None:
            what = ("both halt and grant words" if (halt and grant)
                    else "a bare refusal" if refuse
                    else "no witness of either kind")
            return Verdict(site.key, "", f"UNCLASSIFIED: {what}",
                           (halt or grant or refuse or [None])[0], {})
        kind, why = row[0], f"known: {row[1]}"
        witness = (halt or grant or refuse or [None])[0]

    # A KNOWN row overrides a derivation, and says why in its own words.
    if row is not None and not why.startswith("known"):
        kind, why = row[0], f"known (overrides {why}): {row[1]}"

    tolerates = {}
    if witness is not None:
        for p in CARRIED:
            tolerates[p] = bool(rx.match(witness + p))
    return Verdict(site.key, kind, why, witness, tolerates)


def verdicts(root: pathlib.Path, cen, known: dict) -> dict:
    """key -> Verdict, one per site key (a site reached from two rungs is
    the same grammar and gets one verdict)."""
    out = {}
    for site in cen.sites:
        if site.key not in out:
            out[site.key] = classify(root, site, known)
    return out
