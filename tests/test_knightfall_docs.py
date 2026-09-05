"""Knightfall, written down (design item 6).

HE RUNS THE TWO COMMANDS HIMSELF -- setting the spoken phrase and the first
typed code is his keyboard step, not something this repository may do for
him -- so the commands have to be in the setup document, exactly as the CLI
spells them, or the feature does not exist for him.

This test NEVER RUNS the people CLI. ``scripts/jarvis_people.py`` loads the
REAL registry and the REAL assistant config as soon as its arguments parse,
and neither belongs in a test. The subcommands are read out of its source
instead.
"""
import re
from pathlib import Path

import jarvis
from jarvis import passphrase as pp

DOC = Path(jarvis.__file__).parent.parent / "docs" / "assistant-setup.md"
CLI = Path(jarvis.__file__).parent.parent / "scripts" / "jarvis_people.py"

SET_PHRASE = "~/vss_env/bin/python scripts/jarvis_people.py set-phrase hunter"
SET_CODE = "~/vss_env/bin/python scripts/jarvis_people.py set-code hunter"


def _doc() -> str:
    return DOC.read_text()


def test_the_document_has_a_knightfall_section():
    heads = [ln for ln in _doc().splitlines()
             if ln.startswith("#") and "knightfall" in ln.lower()]
    assert heads, "no Knightfall heading in docs/assistant-setup.md"


def test_it_carries_the_two_commands_he_types():
    text = _doc()
    assert SET_PHRASE in text
    assert SET_CODE in text


def test_the_commands_are_the_ones_the_cli_actually_has():
    src = CLI.read_text()
    assert 'for name in ("set-phrase", "set-code")' in src
    assert 'sp.add_argument("label", nargs="?"' in src
    # and the secret is never taken from argv, where ps would show it
    assert "getpass" in src
    assert "--code" not in src and "--phrase" not in src


def _section() -> str:
    text = _doc()
    return text[text.index("## 84"):]


def test_the_section_says_what_stays_the_same_and_what_rotates():
    section = _section().lower()
    for phrase in ("shadow", "rotate", "inbox", "five minutes"):
        assert phrase in section, phrase


def test_his_phrase_still_passes_the_length_rule():
    """He said he will use "Knightfall protocol". Nothing here stores it --
    this is arithmetic on the PUBLIC rule, so the document cannot promise
    him a phrase the CLI would refuse."""
    ok, why = pp.phrase_ok("Knightfall protocol")
    assert ok and not why
    assert len(pp.normalise_spoken("Knightfall protocol")) >= pp.MIN_PHRASE_LEN


def test_no_secret_is_written_down():
    """A document that carried an example code long enough to paste is a
    document that trains him to type an example code. A generated one is
    eight characters of pp.CODE_ALPHABET with at least one digit in it."""
    words = re.findall(r"[A-Za-z0-9]+", _section())
    looks_like = [w for w in words
                  if len(w) == 8 and set(w) <= set(pp.CODE_ALPHABET)
                  and any(c.isdigit() for c in w)]
    assert looks_like == [], looks_like
