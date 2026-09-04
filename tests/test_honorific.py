""""sir" never reaches a ma'am person, and "ma'am" never reaches Hunter.

THIS IS THE TEST THE WHOLE FEATURE STANDS ON, so it is not a spot check.
The corpus is rebuilt BY AST FROM SOURCE on every run: every module-level
string constant in ``jarvis/`` and ``scripts/`` that contains a "sir". A
constant added next month is covered the day it lands, without anyone
remembering to add it here.

FOUR PROPERTIES
    (a) under ma'am, no authored spoken constant still says "sir";
    (b) under sir, the swap returns THE INPUT OBJECT -- byte-identical, so
        Hunter's tuned voice cannot regress through this door and the 149
        test files that assert exact spoken strings stay green;
    (c) the measured corruptions in address.py's own docstring, plus a
        mail subject and a note body, are untouched;
    (d) jarvis/reader.py neither imports nor calls the swap -- it reads
        documents aloud and its words are not Jarvis's.

WHAT IS EXCLUDED FROM (a), AND WHY EACH ONE IS NOT A HOLE
    * REGEX SOURCES (``_ADJ_TAIL``, ``_SEND_TAIL``, ...). They are
      LISTENING patterns -- the words Jarvis strips off the end of what he
      HEARS -- and are never spoken. Rewriting one would break recognition
      of "...please, sir" and fix nothing.
    * THE THREE MODEL PROMPTS in jarvis/brain.py. They are not spoken
      either; they are instructions TO gemma4, and they are handled by
      ``brain.addressee_clause`` instead. Property (e) below pins that.
    * BARE TOKENS -- a constant whose whole value is the word. They are
      matcher vocabulary, not sentences.
"""
import ast
import pathlib
import re

import pytest

from jarvis import address, brain, honorific
from jarvis import gate as gt
from jarvis.identity import HONORIFICS, Person, Registry, ROLE_KNOWN, ROLE_OWNER

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIR_RX = re.compile(r"\bsir\b", re.I)
SLOT_RX = re.compile(r"\{[^{}]*\}")

# Names whose value is a regular expression Jarvis MATCHES against, never
# says. Recognised by shape as well, so a new one is covered.
_MATCHER_MARKS = ("(?:", "(?i", "\\b", "\\s", "[?.!", "$")
# The model prompts. Not spoken; see the module docstring and property (e).
_PROMPTS = {"VOICE_RULES", "JARVIS_SYSTEM", "CLAUDE_SYSTEM",
            "AUTONOMOUS_PROMPT", "REGISTER_CLAUSES"}


def _strings(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for e in node.elts:
            yield from _strings(e)
    elif isinstance(node, ast.Dict):
        for e in node.values:
            yield from _strings(e)
    elif isinstance(node, ast.JoinedStr):
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                yield v.value
    elif isinstance(node, ast.BinOp):
        yield from _strings(node.left)
        yield from _strings(node.right)


def _render(text: str) -> str:
    """A template as it is actually SPOKEN: slots filled in.

    ``{sir}`` is the address itself (jarvis/mathspeak.py) and renders as
    the word; every other slot renders as nothing, which is the shape that
    exercises the vocative rules -- "Epoch {epoch}, sir{loss}." becomes
    "Epoch , sir." and its trailing vocative is then visible.
    """
    return SLOT_RX.sub("", text.replace("{sir}", "sir"))


def _looks_like_a_matcher(name: str, value: str) -> bool:
    if name.endswith(("_RX", "_TAIL", "_PATTERN", "_WORDS", "_FILLER")):
        return True
    return any(mark in value for mark in _MATCHER_MARKS)


def _corpus():
    """``[(file, line, name, text), ...]`` rebuilt from source each run."""
    out = []
    for sub in ("jarvis", "scripts"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if node.value is None:
                    continue
                targets = ([node.target] if isinstance(node, ast.AnnAssign)
                           else node.targets)
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                name = names[0] if names else "?"
                for text in _strings(node.value):
                    if SIR_RX.search(text):
                        out.append((path.relative_to(ROOT).as_posix(),
                                    node.lineno, name, text))
    return out


def _spoken_corpus():
    keep = []
    for f, line, name, text in _corpus():
        if name in _PROMPTS or _looks_like_a_matcher(name, text):
            continue
        if text.strip().lower() == "sir":
            continue          # a bare token, not a sentence
        keep.append((f, line, name, _render(text)))
    return keep


CORPUS = _corpus()
SPOKEN = _spoken_corpus()


def test_the_corpus_is_not_empty_and_is_most_of_the_tree():
    """A filter that quietly emptied the corpus would make every property
    below vacuously true. This is the guard against that."""
    assert len(CORPUS) > 400, len(CORPUS)
    assert len(SPOKEN) > 400, len(SPOKEN)
    # The exclusions are a handful, not a strategy.
    assert len(CORPUS) - len(SPOKEN) < 40


# ------------------------------------------------- (a) no sir reaches her
def test_no_authored_line_still_says_sir_to_a_maam_person():
    """THE PROPERTY THE FEATURE EXISTS FOR. Every authored spoken constant
    in the tree, rendered, swapped for ma'am, must contain no "sir"."""
    leaks = [(f, line, name, text) for f, line, name, text in SPOKEN
             if address.count_sirs(address.swap_addresses(text, "ma'am"))]
    assert not leaks, "\n".join("%s:%s %s %r" % row for row in leaks[:20])


def test_the_gates_own_new_lines_are_clean():
    """Spot-checked separately because they are the lines Mara and Heather
    are most likely to hear first."""
    for line in (gt.SIGNIN_OK_LINE.format(first="Mara"),
                 gt.ENROL_AT_KEYBOARD_LINE,
                 gt.UNKNOWN_LINE, gt.UNKNOWN_PHRASE_LINE,
                 gt.SIGNIN_PENDING_LINE.format(name="Mara"),
                 gt.SIGNIN_NO_LEG_LINE.format(name="Mara"),
                 gt.NOT_YOURS_LINE.format(name="Mara"),
                 gt.KNOWN_SCOPE_LINE.format(name="Mara")):
        assert address.count_sirs(address.swap_addresses(line, "ma'am")) == 0
        assert address.count_sirs(address.swap_addresses(line, "")) == 0


def test_signed_in_reads_right_in_all_three_forms():
    """HIS words (09-04), verbatim, with the first name in them. Mara,
    Heather and a no-address person hear the SAME sentence: it carries
    no honorific, so the swap has nothing to change in any register."""
    assert gt.SIGNIN_OK_LINE == ("Voice and identity recognized, welcome "
                                 "back {first}. How may I be of assistance "
                                 "today?")
    for first in ("Mara", "Heather", "Alex"):
        line = gt.SIGNIN_OK_LINE.format(first=first)
        assert line == ("Voice and identity recognized, welcome back %s. "
                        "How may I be of assistance today?" % first)
        for hon in ("sir", "ma'am", ""):
            assert address.swap_addresses(line, hon) == line
    # A template is never prewarmed: the cache would hold "{first}".
    assert gt.SIGNIN_OK_LINE not in gt.PREWARM_LINES
    assert "{first}" not in "".join(gt.PREWARM_LINES)


def test_first_name_is_the_typed_one_then_the_display_names_first_word():
    from jarvis.identity import Person
    assert gt.first_name(Person(label="mara", first="Mara", last="Quinn")) \
        == "Mara"
    assert gt.first_name(Person(label="heather", name="Heather Vance")) \
        == "Heather"
    assert gt.first_name(Person(label="alex")) == "Alex"
    assert gt.first_name(None, "there") == "there"


# --------------------------------------- (b) nothing moves for the owner
def test_under_sir_the_swap_returns_the_input_object():
    """BYTE-IDENTICAL, and identity-identical: not a copy that happens to
    compare equal. This is why 149 test files asserting exact spoken
    strings are unaffected and his tuned voice cannot regress."""
    for f, line, name, text in SPOKEN:
        assert address.swap_addresses(text, "sir") is text, \
            "%s:%s %s" % (f, line, name)


def test_maam_never_reaches_hunter():
    """The other direction of the same promise. Whatever a line says, the
    owner's rendering of it never acquires "ma'am"."""
    for f, line, name, text in SPOKEN:
        out = address.swap_addresses(text, "sir")
        assert "ma'am" not in out.lower() or "ma'am" in text.lower(), \
            "%s:%s %s" % (f, line, name)


def test_the_resolver_gives_the_owner_sir_even_with_no_registry(tmp_path):
    """A FRESH BOX IS NEVER ADDRESSED WRONGLY. people.json does not exist
    on his machine today, and the answer for him is still "sir"."""
    reg = Registry.load(tmp_path / "nothing.json")
    assert reg.usable is False
    got = honorific.for_addressee(reg, lambda: "", lambda: 0.0,
                                  "hunter", 100.0)
    assert got == "sir"


# ------------------------------------------- (c) third-party text is safe
CORRUPTIONS = (
    "Sir Isaac Newton wrote the Principia.",
    "Now playing Yes Sir, I Can Boogie.",
    "That is sir's coffee.",
    'He said, "Thank you, sir." and left.',
    "Now playing Thank You Sir by The Somebodies.",
)


@pytest.mark.parametrize("text", CORRUPTIONS)
def test_third_party_text_is_never_rewritten(text):
    """The three measured corruptions in address.py's docstring, plus a
    quotation and a Title-Case track name."""
    assert address.swap_addresses(text, "ma'am") == text
    assert address.swap_addresses(text, "") == text


def test_the_swap_and_the_remover_agree_about_what_an_address_is():
    """Every span the REMOVER counts, outside a quotation, the SWAP also
    finds. The two passes cannot drift into disagreeing."""
    for f, line, name, text in SPOKEN:
        starts = {end for _s, end, _k in address.vocative_spans(text)
                  if not address.inside_a_quotation(text, _s)}
        swapped = {end for _s, end in address.swap_spans(text)}
        assert starts <= swapped, "%s:%s %s %r" % (f, line, name, text)


def test_a_bare_replace_is_not_used_anywhere_in_the_tree():
    """``str.replace("sir", ...)`` is the defect address.py was written
    against -- it eats "Sir Isaac Newton" -- and it must not exist in
    jarvis/ or scripts/. Read by AST, so the prose warning ABOUT it in
    address.py's own docstring is not mistaken for the thing itself."""
    bad = []
    for sub in ("jarvis", "scripts"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if not (isinstance(fn, ast.Attribute) and fn.attr == "replace"):
                    continue
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Constant) and \
                        isinstance(first.value, str) and \
                        SIR_RX.fullmatch(first.value.strip()):
                    bad.append("%s:%s" % (path.name, node.lineno))
    assert not bad, bad


# ------------------------------------------------ (d) the reader is out
def test_the_document_reader_never_swaps_an_address():
    """jarvis/reader.py reads HIS documents aloud. Those words are not
    Jarvis's and rewriting one inside somebody's file is the one thing
    this pass must never do, so it bypasses _say and is excluded here."""
    src = (ROOT / "jarvis" / "reader.py").read_text(encoding="utf-8")
    assert "swap_addresses" not in src
    assert "honorific" not in src


# -------------------------------------------- (e) the prompt half agrees
def test_the_owner_prompt_is_byte_identical():
    """gemma4's instructions for Hunter must not move by one character."""
    brain.set_addressee("", "sir")
    before = brain.build_ollama_system(shots=brain.FEW_SHOT_PINNED)
    brain.set_addressee("Mara", "ma'am")
    other = brain.build_ollama_system(shots=brain.FEW_SHOT_PINNED)
    brain.set_addressee("", "sir")
    after = brain.build_ollama_system(shots=brain.FEW_SHOT_PINNED)
    assert before == after
    assert other != before
    assert "Mara" in other and "ma'am" in other


def test_the_prompt_never_tells_the_model_to_call_her_he():
    brain.set_addressee("Mara", "ma'am")
    try:
        text = brain.build_ollama_system(shots=brain.FEW_SHOT_PINNED)
        assert 'they mean Mara for this reply' in text
        assert 'Call them "ma\'am"' in text
    finally:
        brain.set_addressee("", "sir")


def test_a_person_who_wants_no_address_is_told_so_by_name():
    brain.set_addressee("Alex", "")
    try:
        text = brain.build_ollama_system(shots=brain.FEW_SHOT_PINNED)
        assert 'Call them "Alex" now and then' in text
    finally:
        brain.set_addressee("", "sir")


# ------------------------------------------------------- the resolver
def _registry(tmp_path):
    r = Registry(path=tmp_path / "people.json")
    r.add_person(Person(label="hunter", name="Hunter", first="Hunter",
                        last="Peyrovian", role=ROLE_OWNER, voice=True,
                        honorific="sir"))
    r.add_person(Person(label="mara", first="Mara", last="Quinn",
                        role=ROLE_KNOWN, honorific="ma'am", face="mara",
                        consent="typed"))
    r.add_person(Person(label="alex", first="Alex", last="Reed",
                        role=ROLE_KNOWN, honorific="", face="alex",
                        consent="typed"))
    return r


def test_the_honorific_is_read_from_the_row_and_nowhere_else(tmp_path):
    reg = _registry(tmp_path)
    assert honorific.for_label(reg, "hunter") == "sir"
    assert honorific.for_label(reg, "mara") == "ma'am"
    assert honorific.for_label(reg, "alex") == ""
    # A person nobody enrolled gets NO form of address, never a guess.
    assert honorific.for_label(reg, "heather") == ""


def test_a_stale_attribution_falls_back_to_the_owner(tmp_path):
    """One turn in which the camera named Mara must not address tonight's
    reminder and tomorrow's briefing to her."""
    reg = _registry(tmp_path)
    fresh = honorific.for_addressee(reg, lambda: "mara", lambda: 1000.0,
                                    "hunter", 1000.0 + 5)
    stale = honorific.for_addressee(
        reg, lambda: "mara", lambda: 1000.0, "hunter",
        1000.0 + honorific.ADDRESSEE_TTL + 1)
    assert fresh == "ma'am"
    assert stale == "sir"


def test_an_app_that_cannot_say_who_is_here_answers_sir(tmp_path):
    def boom():
        raise RuntimeError("no")
    reg = _registry(tmp_path)
    assert honorific.for_addressee(reg, boom, lambda: 0.0,
                                   "hunter", 1.0) == "sir"


def test_the_two_lists_of_honorifics_agree():
    """address.py is pure and keeps its own copy of the three values.
    They must say the same thing."""
    assert set(address.SWAPPABLE) == set(HONORIFICS)


# ----------------------------------------------------- the door itself
def _door(tmp_path, who="", when=0.0):
    """A stand-in with the REAL _say, _honorific and _address_for_addressee
    bound to it. No Tk, no TTS, no mic: the sink is a list."""
    from types import SimpleNamespace

    from jarvis import app as app_mod
    from jarvis import gate as gate_mod

    reg = _registry(tmp_path)
    reg.save()
    a = SimpleNamespace()
    a.said = []
    a.assistant = SimpleNamespace(get=lambda k, d=None: "Hunter")
    a.tts = SimpleNamespace(speak=a.said.append, busy=False)
    a.quiet = None
    a._quiet_turn = False
    a._last_source = "voice"
    a._note_spoke = lambda: None
    a._gate_who, a._gate_who_ts = who, when
    a.gate = gate_mod.OwnerGate(
        registry=reg, owner="hunter",
        get_option=lambda k, d=None: {"owner.mode": "shadow"}.get(k, d))
    a._honorific = app_mod.JarvisApp._honorific.__get__(a)
    a._address_for_addressee = \
        app_mod.JarvisApp._address_for_addressee.__get__(a)
    a._say = app_mod.JarvisApp._say.__get__(a)
    return a


def test_the_door_swaps_the_address_for_a_maam_person(tmp_path, monkeypatch):
    """THE END-TO-END ONE. A line authored with "sir" reaches the TTS as
    "ma'am" when the gate's addressee is Mara -- and the literal in the
    source is not edited."""
    import time as _time

    from jarvis import app as app_mod
    monkeypatch.setattr(app_mod.CONFIG, "talkback", True, raising=False)
    a = _door(tmp_path, who="mara", when=_time.monotonic())
    a._say("Very good, sir.")
    assert a.said == ["Very good, ma'am."]


def test_the_door_moves_nothing_for_the_owner(tmp_path, monkeypatch):
    import time as _time

    from jarvis import app as app_mod
    monkeypatch.setattr(app_mod.CONFIG, "talkback", True, raising=False)
    a = _door(tmp_path, who="hunter", when=_time.monotonic())
    a._say("Very good, sir.")
    assert a.said == ["Very good, sir."]


def test_the_door_addresses_nobody_when_the_person_asked_for_none(
        tmp_path, monkeypatch):
    import time as _time

    from jarvis import app as app_mod
    monkeypatch.setattr(app_mod.CONFIG, "talkback", True, raising=False)
    a = _door(tmp_path, who="alex", when=_time.monotonic())
    a._say("Very good, sir.")
    assert a.said == ["Very good."]


def test_a_swap_that_throws_still_speaks_the_line(tmp_path, monkeypatch):
    """A failure in the courtesy must never cost him the sentence."""
    import time as _time

    from jarvis import app as app_mod
    monkeypatch.setattr(app_mod.CONFIG, "talkback", True, raising=False)
    a = _door(tmp_path, who="mara", when=_time.monotonic())

    def boom(*_a, **_kw):
        raise RuntimeError("no")
    monkeypatch.setattr(app_mod.address_mod, "swap_addresses", boom)
    a._say("Very good, sir.")
    assert a.said == ["Very good, sir."]


def test_the_commander_bypass_is_wired_too():
    """commander._speak is the one path that does not go through _say."""
    from types import SimpleNamespace

    from jarvis import commander as cmd_mod
    said = []
    c = SimpleNamespace(
        services=SimpleNamespace(tts=SimpleNamespace(speak=said.append),
                                 honorific=lambda: "ma'am"),
        _suppress_speak=None)
    c._svc = cmd_mod.Commander._svc.__get__(c)
    c._for_addressee = cmd_mod.Commander._for_addressee.__get__(c)
    c._speak = cmd_mod.Commander._speak.__get__(c)
    c._speak("Noted, sir.")
    assert said == ["Noted, ma'am."]


# ------------------------------------------ the face-dim line tells the truth
def test_the_startup_line_does_not_call_a_512_row_stale(tmp_path):
    """app.py read facegallery.EMBED_DIM -- 128, the SFace width -- while
    the live backend is arcface_mbf at 512, so an honest 512-D enrolment
    was reported as stale and he was told to re-enrol for nothing."""
    from types import SimpleNamespace

    from jarvis import app as app_mod
    from jarvis import facemodels
    from jarvis import gate as gate_mod

    width = facemodels.backend_for(None).embed_dim
    assert width == 512, "the live backend moved; this test names the width"
    r = Registry(path=tmp_path / "people.json")
    r.add_person(Person(label="hunter", role=ROLE_OWNER, honorific="sir",
                        face="hunter", face_dim=512))
    a = SimpleNamespace()
    a.gate = gate_mod.OwnerGate(registry=r, owner="hunter",
                                get_option=lambda k, d=None: d)
    a.get_option = lambda k, d=None: True
    a._face_leg_why = app_mod.JarvisApp._face_leg_why.__get__(a)
    why = a._face_leg_why()
    assert "re-enrol" not in why, why
    # ...and a genuinely stale row still says so, naming the live width.
    r.person("hunter").face_dim = 128
    assert "512-D and hunter was enrolled at 128-D" in a._face_leg_why()
