"""The answer grammars, DERIVED -- and the few that stay unstripped, NAMED.

2026-09-05, 12:24: "Shall I run your briefing, sir?" -- "Uh, yeah." -- and
the briefing did not run. The fix stripped the filled pause at six answer
sites and called an eight-item inventory complete. An adversary found six
more. A sweep with a hand-written set of rung names found five more. The
census in tests/answercensus.py, which derives the rungs from the code
instead of naming them, found nine more on top of those -- and every one
of the four inventories before it was wrong by exactly the rungs its
author had not read. So this file does not list the answer grammars. It
asks the census for them, and pins only two things by hand:

  KNOWN_STRIPPERS   the two pre-existing second filler lists, whose output
                    counts as stripped although the derivation cannot
                    prove it -- each with its reason and its probe;
  KNOWN_UNSTRIPPED  every site the census still reports unstripped, with
                    its KIND and the reason it stays. A new unstripped site
                    fails the first test. A row that goes stale -- the site
                    fixed, moved, or gone -- fails it too. Do not edit the
                    table to make it green; read the failure.

WHAT THE INSTRUMENT CANNOT SEE, so nobody reads its green as a proof: a
rung that reads its parked question through a name outside the code's own
convention (ANSWER_STATE_RX); a strip that runs on only one branch; a
grammar over a captured piece (``m.group(1)``); and any grammar that is
not start-anchored at all (a word bag, a ``search`` in the middle). The
floor test below keeps the derivation honest the other way: it must go on
finding the rungs and sites the adversary and the two passes named, so a
drift in the convention cannot shrink the census in silence.

Numbers only. The census reads the AST and imports nothing from jarvis.
"""
from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

from tests import answercensus
from tests.answercensus import CANONICAL, Census, anchored, census

ROOT = pathlib.Path(__file__).resolve().parent.parent / "jarvis"

# The two pre-existing second filler lists. LEFT ALONE, and named here
# because the derivation classifies both as "pass" (see the test below
# that says why): their output counts as stripped on the strength of a
# probe, not a proof.
KNOWN_STRIPPERS = {
    ("commander.py", "_send_clean"):
        "the send lane's own two lists (_SEND_FILLER_LEAD_RX, _SEND_FILLER_MID_RX), "
        "backstopped by the canonical strip AFTER them. One hole, found by probing "
        "rather than repeating the claim: its `else t` hands the ORIGINAL back whole "
        "when the clean leaves no letters, so \"uh, okay\" came back with the filler "
        "on; the maybe judgement in _try_send_confirm now strips at its own call site. "
        "Probed word by word by test_filled_yes.TestTheSendReadBackMaybe.",
}
# The OTHER pre-existing list, leavetime._ANSWER_FILLER, needs no row at
# all since round 3 (09-06): the rung hands answer_minutes the stripped
# text, so its startswith loop and the parser behind it are reached
# stripped on entry, and its "uh"/"um" entries are gone.

# Every site the census reports UNSTRIPPED, and why it stays.
#   command grammar: a grammar the dispatcher applies to every utterance,
#     consulted by a rung in ANSWER position too, whose job there is to
#     RE-DISPATCH the words as that command. Stripping at the rung alone
#     would drop the open question and hand the registry the unstripped
#     words -- half a fix that loses both. The fix belongs in the command
#     grammar and is a different decision (the 5caf86c ruling).
#   second list: a site inside, or fed only through, the function that OWNS
#     a pre-existing second vocabulary -- named, backstopped or probed.
# Rows are (kind, owner, reason); the owner is the list's function for a
# second-list row and "" for a command grammar.
#   filed: the words are FILED, not judged -- a lecture line -- and nothing
#     may be taken off them; the grammar that JUDGES the same utterance
#     (the end phrase) is a separate, stripped site. Owner: the rung.
#   not words: the mode convention (MODE_STATE) made a rung of a helper
#     whose parameter is a tool argument, never the utterance; the grammar
#     it reaches is over that argument. Owner: the rung. A false hit named
#     here rather than silenced in the derivation.
KNOWN_UNSTRIPPED = {
    "commander.py:read_control_kind:_READ_CTL_RX": (
        "command grammar", "",
        "\"skip\" while something is being READ belongs to the reader (rung 4b), "
        "and the briefing offer stands aside for it; the reader's own rung is the "
        "command grammar and is unstripped."),
    "commander.py:_send_clean:_SEND_FILLER_LEAD_RX": (
        "second list", "_send_clean",
        "the send lane's own lead list, inside _send_clean itself; backstopped by "
        "the canonical strip after it, hole and probe named on KNOWN_STRIPPERS."),
    "commander.py:strip_address:_ADDRESS_RX": (
        "filed", "Commander._handle_lecture",
        "the lecture LINE: strip_address takes the address off the line that is "
        "filed, and the line is filed as he said it -- an \"um\" in a lecture note "
        "is dictation. The end phrase is judged by _lecture_end, stripped."),
    "tools/timekeeper.py:normalize_kind:in": (
        "not words", "adjust_empty_line",
        "a tool's kind argument (\"alarm\" / \"timer\"), never the utterance: "
        "adjust_empty_line reads tk.ringing to phrase a line, the mode convention "
        "makes it a rung, and the in-tuple grammar it reaches is over the kind."),
}


@pytest.fixture(scope="module")
def cen() -> Census:
    return census(ROOT, KNOWN_STRIPPERS)


def _known_for(key: str):
    """The KNOWN row for a site key: exact, or by prefix for an inline
    ``re.<method>(<pattern>)`` whose label carries the pattern's head."""
    if key in KNOWN_UNSTRIPPED:
        return key
    for k in KNOWN_UNSTRIPPED:
        if key.startswith(k + "("):
            return k
    return None


# ===================================================================
# 1. THE ASSERTION: stripped, or named
# ===================================================================
def test_every_answer_grammar_is_reached_through_a_stripped_string_or_is_named(cen):
    unstripped = {}
    for s in cen.unstripped():
        unstripped.setdefault(_known_for(s.key) or s.key, []).append(s)
    new = {k: v for k, v in unstripped.items() if k not in KNOWN_UNSTRIPPED}
    stale = [k for k in KNOWN_UNSTRIPPED if k not in unstripped]
    lines = []
    for k, sites in new.items():
        lines.append("NEW unstripped answer grammar (a filled pause is thrown away here):")
        lines += [f"    {s}" for s in sites]
    for k in stale:
        lines.append(f"STALE row -- fixed, moved or gone; drop it from KNOWN_UNSTRIPPED: {k}")
    assert not new and not stale, "\n" + "\n".join(lines)


def _referenced_outside_rungs(module: str, fname: str, rungs: set) -> bool:
    """Is ``fname`` named anywhere in its module OUTSIDE a rung's body --
    a registry table, a handler, the dispatcher? That is what makes it a
    COMMAND grammar that a rung merely shares, rather than an answer
    grammar of the rung's own."""
    tree = ast.parse((ROOT / module).read_text())
    rung_names = {q.split(".")[-1] for m, q in rungs if m == module}
    inside = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in rung_names:
            inside.update(id(n) for n in ast.walk(fn))
    return any(isinstance(n, ast.Name) and n.id == fname and id(n) not in inside
               for n in ast.walk(tree))


def test_the_known_rows_are_still_the_kind_they_claim(cen):
    """A command-grammar row must still BE a command grammar -- something
    that is not a rung names it -- and a second-list row must still sit
    inside, or be fed only through, the function that owns its list."""
    by_key = {}
    for s in cen.unstripped():
        by_key.setdefault(_known_for(s.key) or s.key, []).append(s)
    for key, (kind, owner, _why) in KNOWN_UNSTRIPPED.items():
        sites = by_key.get(key, [])
        assert sites, key
        for s in sites:
            if kind == "command grammar":
                assert _referenced_outside_rungs(s.module, s.function, cen.rungs), \
                    f"{key}: nothing but a rung names {s.function} any more; " \
                    f"it is not a shared command grammar, strip it like the rest"
            elif kind == "second list":
                assert s.function == owner or owner in s.chain.split(" > "), \
                    f"{key}: reached outside {owner}: <{s.chain}>"
            elif kind in ("filed", "not words"):
                assert s.chain.split(" > ")[0] == owner, \
                    f"{key}: reached from a rung other than {owner}: <{s.chain}>"
            else:
                pytest.fail(f"unknown kind {kind!r} on {key}")


def test_the_named_strippers_are_named_because_the_derivation_cannot_prove_them(cen):
    """Each KNOWN_STRIPPERS row exists because the fixpoint classifies the
    function as "pass" -- _send_clean's `else t`, answer_minutes' loop.
    If one becomes derivable, the row is redundant: drop it."""
    for fn in KNOWN_STRIPPERS:
        assert cen.kinds.get(fn) == "pass", \
            f"{fn} is now derived as {cen.kinds.get(fn)!r}; the KNOWN row is redundant"


# ===================================================================
# 2. THE FLOOR: the derivation must keep finding these
# ===================================================================
# The rungs the verdict, the two passes and this census named. A rung
# missing here means the naming convention drifted and the census shrank.
RUNGS_FLOOR = {
    ("commander.py", "Commander._try_briefing_offer"),
    ("commander.py", "Commander._try_terminal_offer"),
    ("commander.py", "Commander._try_approval"),
    ("commander.py", "Commander._try_send_confirm"),
    ("commander.py", "Commander._try_destructive_confirm"),
    ("commander.py", "Commander._try_router_answer"),
    ("commander.py", "Commander._try_filepick_answer"),
    ("commander.py", "Commander._try_sendask_answer"),
    ("commander.py", "Commander._try_leave_answer"),
    ("commander.py", "Commander._try_correction"),
    ("commander.py", "Commander._try_feedback"),
    ("commander.py", "Commander._try_quiz_answer"),
    ("commander.py", "Commander._try_teach_offer"),
    ("commander.py", "Commander._try_enrol"),
    ("commander.py", "Commander._try_session"),
    ("commander.py", "Commander._try_day_shift"),
    ("commander.py", "Commander._try_event_confirm"),
    ("commander.py", "Commander._try_study_offer"),
    ("commander.py", "Commander._try_undo"),
    ("app.py", "JarvisApp.uncertain_answer"),
    # the two MODE rungs round 2 of the adversary found invisible (09-06)
    ("commander.py", "Commander._try_ringing"),
    ("commander.py", "Commander._handle_lecture"),
}
# The sites that were gaps on this branch, every one measured bare vs
# filled in tests/test_filled_yes.py. They must stay VISIBLE and stripped.
SITES_FLOOR = {
    "commander.py:briefing_answer:_BRIEFING_YES_RX",
    "commander.py:Commander._try_approval:_YES_RX",
    "commander.py:pick_from_answer:_PICK_ORDINAL_RX",
    "commander.py:_person_from_answer:_PICK_ORDINAL_RX",
    "commander.py:Commander._try_filepick_answer:_PICK_CANCEL_RX",
    "commander.py:Commander._try_leave_answer:_LEAVE_DECLINE_RX",
    "commander.py:correction_kind:_CORRECTION_RX",
    "commander.py:correction_kind:_CORRECTION_NOT_RX",
    "commander.py:feedback_kind:_FEEDBACK_YES_RX",
    "commander.py:Commander._try_quiz_answer:_QUIZ_SKIP_RX",
    "commander.py:Commander._try_quiz_answer:_QUIZ_STOP_RX",
    "commander.py:Commander._try_teach_offer:_TAKE_QUIZ_RX",
    "commander.py:Commander._try_enrol:_ENROL_CONFIRM_RX",
    "commander.py:Commander._enrol_control:_ENROL_STOP_RX",
    "commander.py:Commander._try_destructive_confirm:_SEND_MAYBE_RX",
    "commander.py:Commander._try_send_confirm:_SEND_MAYBE_RX",
    "commander.py:_file_answer:_FILE_ANSWER_LEAD_RX",
    "commander.py:_account_answer:_ACCOUNT_ANSWER_LEAD_RX",
    "commander.py:_recipient_answer:_RECIPIENT_ANSWER_LEAD_RX",
    "commander.py:day_shift_followup:_DAY_SHIFT_TAIL_RX",
    "commander.py:cancel_kind:_CANCEL_TASK_RX",
    "commander.py:quiet_kind:_QUIET_RX",
    "commander.py:strip_address:_ADDRESS_RX",
    "dialogue.py:enough_kind:_ENOUGH_RX",
    "dialogue.py:WeekPlanner.settle:_YES_RX",
    "router.py:answer_kind:_ANSWER_CLAUDE_RX",
    # round 3 (09-06): the two mode rungs, the quiz escape, the second
    # lists now reached stripped, the non-regex grammars, the undo lane
    "commander.py:Commander._try_ringing:_RING_STOP_RX",
    "commander.py:Commander._try_ringing:_SNOOZE_RX",
    "commander.py:_lecture_end:_LECTURE_END_RX",
    "commander.py:quiz_kind:_QUIZ_RX",
    "commander.py:review_kind:_REVIEW_RX",
    "commander.py:parse_yes_no:in(_YES_WORDS)",
    "commander.py:_undo_match:_UNDO_RX",
    "leavetime.py:answer_minutes:startswith(word + ' ')",
    "leavetime.py:parse_minutes:re.fullmatch",
}


def test_the_floor_of_rungs_is_still_found(cen):
    missing = RUNGS_FLOOR - cen.rungs
    assert not missing, f"the census lost these rungs: {sorted(missing)}"


def _on_floor(key: str) -> bool:
    return key in SITES_FLOOR or any(key.startswith(k + "(") for k in SITES_FLOOR)


def test_the_floor_of_sites_is_still_found_and_stripped(cen):
    keys = set(cen.keys())
    missing = [k for k in SITES_FLOOR if not any(_on_floor(x) and (x == k or x.startswith(k + "("))
                                                 for x in keys)]
    assert not missing, f"the census lost these sites: {sorted(missing)}"
    def _named_chain(s) -> bool:
        """The one chain a KNOWN row owns (the lecture line through
        strip_address): unstripped by design, and pinned by its own test."""
        k = _known_for(s.key)
        return bool(k) and s.chain.split(" > ")[0] == KNOWN_UNSTRIPPED[k][1]

    lost = [s for s in cen.sites if _on_floor(s.key) and not s.stripped and not _named_chain(s)]
    assert not lost, "a fixed site went unstripped again:\n" + "\n".join(map(str, lost))


def test_the_parsers_fixed_on_this_branch_are_derived_as_strippers_or_sinks(cen):
    """No hand list of strippers: these strip on their first line, and the
    fixpoint must read that off the tree. A sink (briefing_answer returns
    a bool; correction_kind returns a capture) carries no raw words."""
    for fn in ("_file_answer", "_account_answer", "_recipient_answer",
               "pick_from_answer", "_person_from_answer", "parse_yes_no"):
        assert cen.kinds.get(("commander.py", fn)) == "strip", (fn, cen.kinds.get(("commander.py", fn)))
    for fn in ("briefing_answer", "correction_kind", "feedback_kind"):
        assert cen.kinds.get(("commander.py", fn)) in ("strip", "sink"), fn


def test_question_open_names_no_slot_the_census_does_not_walk(cen):
    """Commander.question_open is the code's own enumeration of a parked
    question. Every answer-state token it reads must be the token that
    made at least one rung a rung -- so the convention the census leans
    on is the one question_open keeps."""
    tree = ast.parse((ROOT / "commander.py").read_text())
    qo = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "question_open")
    named = set()
    for n in ast.walk(qo):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and answercensus.ANSWER_STATE_RX.match(n.value):
            named.add(repr(n.value))
        if isinstance(n, ast.Attribute) and answercensus.ANSWER_STATE_RX.match(n.attr):
            named.add("." + n.attr)
    assert named, "question_open reads no answer state at all?"
    tokens = set().union(*cen.tokens.values())
    # a slot is walked if some rung reads it, as an attribute or by name
    unwalked = {t for t in named if t not in tokens and
                ("." + t.strip("'")) not in tokens and repr(t.lstrip(".")) not in tokens}
    assert not unwalked, f"question_open names slots no rung reads: {sorted(unwalked)}"


# ===================================================================
# 3. FAILS CLOSED: a planted gap is reported, and a planted fix clears it
# ===================================================================
ENDPOINT = "def strip_fillers(text):\n    return str(text or '').strip()\n"

PLANT = '''
import re
from fake.endpoint import strip_fillers

_JV = r"(?:jarvis[,\\s]+)?"
_YES_RX = re.compile(r"^" + _JV + r"(?:yes|yeah)[.!]*$", re.I)     # anchor via a constant
_NO_RX = re.compile(r"^(?:no|nope)[.!]*$", re.I)
_LEAD_RX = re.compile(r"^(?:it's|the)\\s+", re.I)
_TAIL_RX = re.compile(r"\\s+please$", re.I)                          # not anchored at ^


def helper_strips(text: str) -> str:
    return strip_fillers(text).strip()


def helper_passes(text: str) -> str:
    return text.strip()


def yes_kind(text: str) -> bool:
    return bool(_YES_RX.match(text))


class Commander:
    def _try_thing_offer(self, text: str):
        offer = self.services.thing_offer
        if not offer:
            return None
        if _NO_RX.match(text):                       # THE PLANTED GAP
            return "no"
        if yes_kind(helper_strips(text)):            # stripped, through a derived stripper
            return "yes"
        if yes_kind(helper_passes(text)):            # raw, through a passthrough
            return "yes2"
        m = _TAIL_RX.search(text)
        if m and _LEAD_RX.match(m.group(0)):         # a captured piece: not a site
            return "cap"
        return None

    def _try_pending(self, text: str):
        if self._pending_thing is None:
            return None
        t = strip_fillers(text)
        if re.search(r"^(?:skip)", t):                # inline anchored search, stripped
            return "skip"
        if _LEAD_RX.sub("", text):                    # anchored sub on the raw words
            return "lead"
        return None

    def _try_third(self, text: str):
        if self._last_turn is None:
            return None
        return _NO_RX.match(strip_fillers(text)) and "third"

    def not_a_rung(self, text: str):
        return bool(_NO_RX.match(text))               # no answer state read: never walked
'''


def _plant(tmp_path, body: str) -> Census:
    pkg = tmp_path / "fake"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "endpoint.py").write_text(ENDPOINT)
    (pkg / "rungs.py").write_text(textwrap.dedent(body))
    return census(pkg)


def test_a_planted_gap_is_reported_and_a_planted_fix_clears_it(tmp_path):
    c = _plant(tmp_path, PLANT)
    assert c.rungs == {("rungs.py", "Commander._try_thing_offer"),
                       ("rungs.py", "Commander._try_pending"),
                       ("rungs.py", "Commander._try_third")}
    assert c.kinds[("rungs.py", "helper_strips")] == "strip"
    assert c.kinds[("rungs.py", "helper_passes")] == "pass"
    unstripped = {s.key for s in c.unstripped()}
    assert unstripped == {
        "rungs.py:Commander._try_thing_offer:_NO_RX",       # the planted gap
        "rungs.py:yes_kind:_YES_RX",                        # raw through the passthrough
        "rungs.py:Commander._try_pending:_LEAD_RX",         # anchored sub on raw words
    }, sorted(unstripped)
    stripped = {s.key for s in c.sites if s.stripped}
    assert any(k.startswith("rungs.py:Commander._try_pending:re.search(") for k in stripped)
    assert not any("_TAIL_RX" in k for k in c.keys()), "an unanchored search is not a site"
    assert not any(s.arg.startswith("m.group") for s in c.sites), "a capture is not a site"

    fixed = (PLANT
             .replace("_NO_RX.match(text)", "_NO_RX.match(strip_fillers(text))")
             .replace("yes_kind(helper_passes(text))", "yes_kind(strip_fillers(helper_passes(text)))")
             .replace('_LEAD_RX.sub("", text)', '_LEAD_RX.sub("", t)'))
    c2 = _plant(tmp_path / "again", fixed)
    assert c2.unstripped() == [], "\n".join(map(str, c2.unstripped()))
    assert set(c2.keys()) == set(c.keys()), "the fix must not hide a site, only strip it"


def test_a_rung_hidden_behind_a_dispatcher_is_still_walked_but_the_dispatcher_is_not(tmp_path):
    body = PLANT + '''

class App:
    def _dispatch(self, text: str):
        c = Commander()
        if c._try_thing_offer(text): return 1
        if c._try_pending(text): return 2
        if c._try_third(text): return 3
        return c.not_a_rung(text)
'''
    c = _plant(tmp_path, body)
    assert ("rungs.py", "App._dispatch") in c.dispatchers
    assert ("rungs.py", "Commander.not_a_rung") not in c.rungs
    # walked from the rungs, not from the dispatcher: the dispatcher's own
    # calls add no site
    assert not any(s.chain.startswith("App._dispatch") for s in c.sites)


# ===================================================================
# 4. THE PIECES
# ===================================================================
@pytest.mark.parametrize("pattern, want", [
    ("^yes", True), ("(?i)^yes", True), ("(?:^yes|^no)", True), (r"\Ayes", True),
    ("yes$", False), (r"\byes\b", False), (None, False), ("", False),
])
def test_anchored_reads_flags_and_openers(pattern, want):
    assert anchored(pattern) is want


def test_the_canonical_strippers_are_the_one_vocabulary():
    """strip_fillers and strip_inline_fillers are the two spellings of ONE
    list (endpoint.FILLER_WORDS); test_filled_yes pins that the undo lane's
    variant is built FROM it. Nothing else launders taint by name."""
    assert CANONICAL == {"strip_fillers", "strip_inline_fillers"}


def test_the_census_imports_nothing_from_jarvis():
    src = pathlib.Path(answercensus.__file__).read_text()
    assert "import jarvis" not in src and "from jarvis" not in src


# ===================================================================
# 5. THE ADVERSARY'S EIGHT PLANTS (round 2, 09-06): caught or BLIND
# ===================================================================
# Each plant is one rung in a package of its own. Six left the census
# GREEN in round 2 (P2, P3, P4, P5, P6, P8); the census now catches P2,
# P4, P5 and P6. P3 and P8 are BLIND, and stated so in the module's
# docstring: the test below asserts the blindness too, so that a census
# that learns to see one of them fails here and the docstring is moved.
_PLANT_HEAD = '''
import re
from fake.endpoint import strip_fillers

_P_RX = re.compile(r"^(?:yes|yeah)[.!]*$", re.I)


class Commander:
'''
PLANTS = {
    # a raw ^-regex on the raw words inside a walked rung
    "P1": (_PLANT_HEAD + '''
    def _try_briefing_offer(self, text: str):
        if not self.services.briefing_offer:
            return None
        return "yes" if _P_RX.match(text) else None
''', "rungs.py:Commander._try_briefing_offer:_P_RX"),
    # the grammar is a startswith, not a regex
    "P2": (_PLANT_HEAD + '''
    def _try_briefing_offer(self, text: str):
        if not self.services.briefing_offer:
            return None
        return "yes" if text.lower().startswith(("yes", "yeah")) else None
''', "rungs.py:Commander._try_briefing_offer:startswith(('yes', 'yeah'))"),
    # a new rung on a slot OUTSIDE the naming convention
    "P3": (_PLANT_HEAD + '''
    def _try_confirm(self, text: str):
        if self._awaiting_confirm is None:
            return None
        self._awaiting_confirm = None
        return "yes" if _P_RX.match(text) else None
''', None),
    # the words arrive under a name that is not text/said, unannotated
    "P4": (_PLANT_HEAD + '''
    def _try_pending(self, utterance):
        if self._pending_thing is None:
            return None
        return "yes" if _P_RX.match(utterance) else None
''', "rungs.py:Commander._try_pending:_P_RX"),
    # the regex is held on the class
    "P5": (_PLANT_HEAD + '''
    _PLANT_RX = re.compile(r"^(?:yes|yeah)[.!]*$", re.I)

    def _try_pending(self, text: str):
        if self._pending_thing is None:
            return None
        return "yes" if self._PLANT_RX.match(text) else None
''', "rungs.py:Commander._try_pending:self._PLANT_RX"),
    # a local alias of the regex
    "P6": (_PLANT_HEAD + '''
    def _try_pending(self, text: str):
        if self._pending_thing is None:
            return None
        rx = _P_RX
        return "yes" if rx.match(text) else None
''', "rungs.py:Commander._try_pending:_P_RX"),
    # a NEW rung on a _pending_* slot with a raw grammar
    "P7": (_PLANT_HEAD + '''
    def _try_new_thing(self, text: str):
        if getattr(self, "_pending_new_thing", None) is None:
            return None
        return "yes" if _P_RX.match(text) else None
''', "rungs.py:Commander._try_new_thing:_P_RX"),
    # a strip on ONE branch only, last in source order
    "P8": (_PLANT_HEAD + '''
    def _try_pending(self, text: str):
        if self._pending_thing is None:
            return None
        t = text
        if self.flag:
            t = strip_fillers(text)
        return "yes" if _P_RX.match(t) else None
''', None),
}
BLIND = {"P3", "P8"}


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_the_adversarys_plant_is_caught_or_declared_blind(tmp_path, name):
    body, key = PLANTS[name]
    c = _plant(tmp_path, body)
    unstripped = {s.key for s in c.unstripped()}
    if name in BLIND:
        assert key is None and not unstripped, \
            f"{name} is now CAUGHT ({sorted(unstripped)}): move it out of the " \
            f"blind spots in answercensus.py's docstring and out of BLIND here"
    else:
        assert key in unstripped, f"{name} walked past the census: {sorted(unstripped)}"


def test_a_registry_handler_is_derived_from_the_table_and_is_never_a_rung(cen):
    """_h_send_file reads _pending_send while RUNNING a command the registry
    already matched; it is not judging a reply. The handlers come from the
    REGISTRY table itself, not from a list of names."""
    assert len(cen.handlers) > 100
    assert ("commander.py", "_h_send_file") in cen.handlers
    assert ("commander.py", "_h_quiz_stop") in cen.handlers
    assert not {h for h in cen.handlers} & cen.rungs


def test_the_two_mode_rungs_are_walked_and_stripped(cen):
    """Round 2 (09-06): "uh, stop" over a ringing alarm and "uh, end notes"
    over open lecture notes were lost, and the census reported both rungs
    as rung=False, 0 sites."""
    for rung in (("commander.py", "Commander._try_ringing"),
                 ("commander.py", "Commander._handle_lecture")):
        assert rung in cen.rungs, rung
        assert rung not in cen.dispatchers, f"{rung} is a dispatcher: walked for nothing"
    keys = {s.key: s for s in cen.sites}
    for key in ("commander.py:Commander._try_ringing:_RING_STOP_RX",
                "commander.py:Commander._try_ringing:_SNOOZE_RX",
                "commander.py:_lecture_end:_LECTURE_END_RX"):
        assert key in keys and keys[key].stripped, key
