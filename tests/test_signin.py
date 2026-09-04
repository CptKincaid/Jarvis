"""Saying your name: what it narrows, what it can never do, and the two
directions the gate has to fail in.

NO MIC, NO LENS, NO MODEL. Every row here is the dicts and labels the real
pipeline produces, fed to pure functions.

THE TWO PROMISES THIS FILE HOLDS
    1. A SPOKEN NAME NEVER ADMITS ANYBODY. It narrows who a real leg is
       asked to confirm, and a claim with nothing behind it is
       indistinguishable, in the verdict, from an unrelated sentence.
    2. THE GATE FAILS OPEN ON ITS OWN FAILURE and SHUT ON A NEGATIVE. A
       fresh box with no registry answers everybody, so it is never mute;
       once a registry exists, a stranger reaching for his settings is
       refused.
"""
import pytest

from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis import signin
from jarvis.identity import (HONORIFICS, Person, Registry, ROLE_KNOWN,
                             ROLE_OWNER)
from jarvis.recognise import HOW_NOBODY, Legs, recognise

MATCHED = {"total": 3, "matched": 2, "scores": [0.38, 0.41, 0.12]}
CODE = "xxx000code"


def _registry(tmp_path, *, code=True, mara=True, second_mara=False):
    r = Registry(path=tmp_path / "people.json")
    r.add_person(Person(label="hunter", name="Hunter", first="Hunter",
                        last="Peyrovian", role=ROLE_OWNER, voice=True,
                        honorific="sir"))
    if mara:
        r.add_person(Person(label="mara", first="Mara", last="Quinn",
                            role=ROLE_KNOWN, honorific="ma'am", face="mara",
                            face_dim=512, consent="typed"))
        r.add_person(Person(label="heather", first="Heather", last="Vance",
                            role=ROLE_KNOWN, honorific="ma'am",
                            face="heather", face_dim=512, consent="typed"))
    if second_mara:
        r.add_person(Person(label="mara2", first="Mara", last="Osei",
                            role=ROLE_KNOWN, honorific="ma'am",
                            face="mara2", consent="typed"))
    if code:
        r.set_secret("hunter", "code_hash", pp.hash_secret(CODE))
    r.save()
    return r


def _gate(tmp_path, mode="enforce", **kw):
    reg = kw.pop("registry", None) or _registry(tmp_path, **kw)
    opts = {"owner.mode": mode}
    return gt.OwnerGate(registry=reg, owner="hunter",
                        get_option=lambda k, d=None: opts.get(k, d))


# ------------------------------------------------------- what a name narrows
@pytest.mark.parametrize("said,want", [
    ("Mara Quinn", "mara"),
    ("my name is Mara Quinn", "mara"),
    ("Jarvis, it's Heather Vance", "heather"),
    ("signing in, Heather Vance.", "heather"),
    ("this is Mara", "mara"),
    ("I'm heather", "heather"),
    ("what's the weather", ""),
    ("", ""),
    (None, ""),
    ("Mara Osei", ""),          # a last name nobody on file carries
    ("it's Mara, put some music on", ""),   # a name in passing, not a claim
])
def test_a_name_narrows_to_exactly_one_person_or_to_nobody(said, want,
                                                           tmp_path):
    rows = _registry(tmp_path).people
    assert signin.candidate_from_name(said, rows) == want


def test_two_people_called_mara_narrow_to_nobody(tmp_path):
    """GUESSING BETWEEN TWO REAL PEOPLE IS THE FAILURE THIS MUST NOT HAVE.
    A first name that is no longer unique narrows to nothing; the full
    name still works."""
    rows = _registry(tmp_path, second_mara=True).people
    assert signin.candidate_from_name("it's Mara", rows) == ""
    assert signin.candidate_from_name("Mara Quinn", rows) == "mara"
    assert signin.candidate_from_name("Mara Osei", rows) == "mara2"


def test_a_name_nobody_answers_to_narrows_to_nobody(tmp_path):
    rows = _registry(tmp_path).people
    assert signin.candidate_from_name("this is Charlotte Webb", rows) == ""


# ------------------------------------------- a name can never admit anybody
def test_recognise_never_returns_a_name_leg(tmp_path):
    """THE INVARIANT. For any combination of legs, ``how`` is never
    "name": a leg that can win the rank is a leg that can admit."""
    roles = _registry(tmp_path).roles()
    for claimed in ("", "mara", "hunter", "nobody-at-all"):
        for voice in ("", "hunter"):
            for face in ("", "mara", "heather"):
                v = recognise(Legs(voice_says=voice, voice_running=bool(voice),
                                   face_says=face, face_running=bool(face),
                                   name_says=claimed), roles, "hunter")
                assert v.how != "name"
                # Anybody NAMED was named by a real leg, never by the claim.
                assert not v.who or v.how in ("voice", "face", "passphrase")


def test_a_claim_with_no_leg_is_the_same_verdict_as_silence(tmp_path):
    """who / role / how / why are IDENTICAL to the verdict an unrelated
    sentence produces. Only ``claimed`` differs, and ``claimed`` admits
    nobody -- it exists so the refusal can offer a way forward."""
    roles = _registry(tmp_path).roles()
    silent = recognise(Legs(), roles, "hunter")
    claim = recognise(Legs(name_says="mara"), roles, "hunter")
    assert (claim.who, claim.role, claim.how, claim.why) == \
           (silent.who, silent.role, silent.how, silent.why)
    assert claim.how == HOW_NOBODY
    assert claim.claimed == "mara" and silent.claimed == ""


def test_saying_her_name_does_not_get_her_in(tmp_path):
    """End to end through the gate: the words are a perfect claim and
    nothing measured anything. She is not admitted."""
    g = _gate(tmp_path)
    d = g.judge("voice", "signing in, Mara Quinn", stats=MATCHED,
                rejected=True, face="", face_running=False)
    assert d.admit is False
    assert d.who == "" and d.role == ""


def test_the_claim_is_confirmed_by_a_leg_and_only_then(tmp_path):
    """The camera names her IN THE SAME TURN. The name narrowed; the FACE
    admitted -- and the welcome says so."""
    g = _gate(tmp_path)
    d = g.judge("voice", "signing in, Mara Quinn", stats=MATCHED,
                rejected=True, face="mara", face_running=True)
    assert d.admit is True and d.who == "mara" and d.how == gt.HOW_FACE
    assert d.line == gt.SIGNIN_OK_LINE


def test_the_welcome_is_said_once_not_on_every_turn(tmp_path):
    g = _gate(tmp_path)
    first = g.judge("voice", "it's Mara Quinn", stats=MATCHED, rejected=True,
                    face="mara", face_running=True, now=0.0)
    again = g.judge("voice", "it's Mara Quinn", stats=MATCHED, rejected=True,
                    face="mara", face_running=True, now=5.0)
    assert first.line == gt.SIGNIN_OK_LINE
    assert again.line == ""


def test_an_owner_is_never_named_back_to_a_stranger(tmp_path):
    """He is recognised by his voice. A stranger saying "Hunter Peyrovian"
    must not have that confirmed for them."""
    g = _gate(tmp_path)
    d = g.judge("voice", "my name is Hunter Peyrovian", stats=MATCHED,
                rejected=True)
    assert d.admit is False
    assert "hunter" not in (d.line or "").lower()


# ------------------------------------------------------ the two sentences
def test_a_running_camera_asks_her_to_look_at_it(tmp_path):
    g = _gate(tmp_path)
    d = g.judge("voice", "signing in, Mara Quinn", stats=MATCHED,
                rejected=True, face="", face_running=True)
    assert d.admit is False
    assert d.line == gt.SIGNIN_PENDING_LINE.format(name="Mara Quinn")


def test_a_dark_camera_says_so_instead_of_a_dead_end(tmp_path):
    """THE HONEST CONSEQUENCE OF ONE VOICEPRINT: at night, under the
    curfew, her sign-in cannot complete and Jarvis says so."""
    g = _gate(tmp_path)
    d = g.judge("voice", "signing in, Mara Quinn", stats=MATCHED,
                rejected=True, face="", face_running=False)
    assert d.admit is False
    assert d.line == gt.SIGNIN_NO_LEG_LINE.format(name="Mara Quinn")
    assert "camera's off" in d.line


def test_the_sign_in_sentence_is_not_a_chant(tmp_path):
    """A room where a name keeps coming up must not turn Jarvis into a
    doorman repeating himself."""
    g = _gate(tmp_path)
    said = []
    for t in (0.0, 1.0, 2.0, 3.0):
        said.append(g.judge("voice", "that was Mara Quinn on the phone",
                            stats=MATCHED, rejected=True, face="",
                            face_running=True, now=t).line)
    assert said[0] and said[1] == "" and said[2] == "" and said[3] == ""
    later = g.judge("voice", "Mara Quinn again", stats=MATCHED,
                    rejected=True, face="", face_running=True,
                    now=gt.SIGNIN_COOLDOWN_S + 1.0)
    assert later.line


def test_the_unknown_line_offers_the_sign_in_and_names_nobody(tmp_path):
    g = _gate(tmp_path)
    d = g.judge("voice", "let me in", stats=MATCHED, rejected=True)
    assert d.admit is False
    assert "first and last name" in d.line
    assert "hunter" not in d.line.lower()


# ------------------------------------- fail OPEN on its own failure...
def test_a_missing_registry_answers_everybody(tmp_path):
    """A FRESH BOX IS NEVER MUTE. people.json does not exist on his
    machine today; every turn is admitted and the gate says why."""
    reg = Registry.load(tmp_path / "not-there.json")
    assert reg.usable is False
    g = gt.OwnerGate(registry=reg, owner="hunter",
                     get_option=lambda k, d=None: {"owner.mode":
                                                   "enforce"}.get(k, d))
    d = g.judge("voice", "read me my mail and change the camera settings",
                stats={}, rejected=True)
    assert d.admit is True and d.how == gt.HOW_OFF
    assert "does not exist" in g.startup_line()
    assert "Everyone is being answered" in g.startup_line()


def test_a_corrupt_registry_answers_everybody(tmp_path):
    path = tmp_path / "people.json"
    path.write_text("{ this is not json", encoding="utf-8")
    reg = Registry.load(path)
    assert reg.usable is False
    g = gt.OwnerGate(registry=reg, owner="hunter",
                     get_option=lambda k, d=None: {"owner.mode":
                                                   "enforce"}.get(k, d))
    assert g.judge("voice", "hello", stats={}, rejected=True).admit is True


def test_a_registry_with_no_owner_answers_everybody(tmp_path):
    path = tmp_path / "people.json"
    path.write_text('{"format": 2, "people": [{"label": "mara", '
                    '"role": "known"}]}', encoding="utf-8")
    reg = Registry.load(path)
    assert reg.usable is False and "no owner" in reg.fault
    g = gt.OwnerGate(registry=reg, owner="hunter",
                     get_option=lambda k, d=None: {"owner.mode":
                                                   "enforce"}.get(k, d))
    assert g.judge("voice", "hello", stats={}, rejected=True).admit is True


# ------------------------------- ...and SHUT on a negative once it exists
def test_once_a_registry_exists_a_stranger_is_refused(tmp_path):
    g = _gate(tmp_path)
    d = g.judge("voice", "what time is it", stats=MATCHED, rejected=True)
    assert d.admit is False and d.how == HOW_NOBODY


PRIVILEGED = (
    "change the camera settings", "enrol my friend", "set the passphrase",
    "read me my mail", "what's on my calendar", "run my briefing",
    "send that to HPCOMPUTER", "what did I write in my notes",
    "remember that I like tea", "turn the curfew off", "restart yourself",
    "ssh into hpcomputer", "text my mum", "what's in my inbox",
    "give me the override code", "open a claude session",
)
ALLOWED_TO_A_KNOWN_PERSON = (
    "what time is it", "what's the weather", "pause the music",
    "skip this track", "what day is it", "how far is the moon",
    "why is the sky blue", "who wrote the Principia", "what's a quasar",
)


@pytest.mark.parametrize("said", PRIVILEGED)
def test_a_known_person_is_refused_everything_privileged(said, tmp_path):
    g = _gate(tmp_path)
    d = g.judge("voice", said, stats=MATCHED, rejected=True, face="mara",
                face_running=True)
    assert d.admit is False, said
    assert d.who == "mara" and d.role == ROLE_KNOWN


@pytest.mark.parametrize("said", ALLOWED_TO_A_KNOWN_PERSON)
def test_a_known_person_may_ask_the_open_things_and_a_question(said,
                                                               tmp_path):
    """The ONE scope change: "I'll answer what I can" means a plain
    question is answered as chat. Everything that ACTS is still refused."""
    g = _gate(tmp_path)
    d = g.judge("voice", said, stats=MATCHED, rejected=True, face="mara",
                face_running=True)
    assert d.admit is True, said


def test_the_owner_is_unaffected_by_the_scope_rule(tmp_path):
    g = _gate(tmp_path)
    for said in PRIVILEGED + ALLOWED_TO_A_KNOWN_PERSON:
        ok, line = g.allowed_for(ROLE_OWNER, said)
        assert ok is True and line == ""


def test_asking_out_loud_to_enrol_somebody_is_answered_with_the_keyboard(
        tmp_path):
    """CONSENT HAS NO SPOKEN EQUIVALENT. The answer is one sentence naming
    the keyboard, and it is the whole answer."""
    g = _gate(tmp_path)
    ok, line = g.allowed_for(ROLE_KNOWN, "enrol me please")
    assert ok is False and line == gt.ENROL_AT_KEYBOARD_LINE


def test_no_spoken_path_reaches_face_enrol():
    """Nothing on the voice path imports the enrolment script."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "jarvis"
    bad = [p.name for p in root.rglob("*.py")
           if "face_enrol" in p.read_text(encoding="utf-8")
           and "import" in p.read_text(encoding="utf-8").split("face_enrol")[0]
           .splitlines()[-1]]
    assert not bad, bad


# --------------------------------------------------- he cannot lock himself out
def test_enforce_without_an_override_code_runs_in_shadow(tmp_path, caplog):
    """LOCKOUT IS A STATE THE CODE WILL NOT ENTER. With no typed way back
    in, enforce is downgraded and the reason is logged."""
    g = _gate(tmp_path, code=False)
    with caplog.at_level("WARNING"):
        assert g.effective_mode() == gt.MODE_SHADOW
    assert "no override code" in caplog.text
    assert "Running in shadow" in caplog.text
    d = g.judge("voice", "let me in", stats=MATCHED, rejected=True)
    assert d.admit is True and d.would_refuse is True


def test_with_a_code_set_enforce_is_really_enforce(tmp_path):
    g = _gate(tmp_path, code=True)
    assert g.effective_mode() == gt.MODE_ENFORCE
    assert g.judge("voice", "let me in", stats=MATCHED,
                   rejected=True).admit is False


def test_the_break_glass_is_never_on_the_voice_path(tmp_path):
    """The typed code admits at the keyboard and nowhere else: saying it
    out loud is worth exactly as much as any other sentence."""
    g = _gate(tmp_path)
    assert gt.check_override_code(g.registry, CODE)[0] == "hunter"
    assert g.judge("voice", CODE, stats=MATCHED, rejected=True).admit is False


# ------------------------------------------------- the record, and its refusals
def test_the_honorific_is_never_inferred_and_never_defaulted(tmp_path):
    """A row cannot be created carrying a form of address nobody typed."""
    r = Registry(path=tmp_path / "people.json")
    r.add_person(Person(label="hunter", role=ROLE_OWNER, honorific="sir"))
    ok, why = r.add_person(Person(label="mara", first="Mara",
                                  role=ROLE_KNOWN, honorific="madam"))
    assert ok is False and "never inferred from a name" in why
    assert r.person("mara") is None


def test_no_module_carries_a_name_to_gender_table():
    """There is no name list in this repo, and there must never be one."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for sub in ("jarvis", "scripts"):
        for path in (root / sub).rglob("*.py"):
            src = path.read_text(encoding="utf-8").lower()
            assert "gender_by_name" not in src, path
            assert "name_gender" not in src, path


def test_only_the_owner_can_be_voice_enrolled(tmp_path):
    """ONE VOICEPRINT, ONE CENTROID. A second row marked voice-enrolled
    would make the voice leg name the wrong person -- which is the honest
    reason Mara cannot sign in at night."""
    r = Registry(path=tmp_path / "people.json")
    r.add_person(Person(label="hunter", role=ROLE_OWNER, honorific="sir",
                        voice=True))
    ok, why = r.add_person(Person(label="mara", role=ROLE_KNOWN,
                                  honorific="ma'am", voice=True))
    assert ok is False and "voice leg name the wrong person" in why


def test_set_honorific_is_owner_only(tmp_path):
    reg = _registry(tmp_path)
    ok, why = reg.set_honorific("mara", "sir", by="mara")
    assert ok is False and "only the owner" in why
    ok, why = reg.set_honorific("mara", "sir", by="hunter")
    assert ok is True and reg.person("mara").honorific == "sir"
    ok, why = reg.set_honorific("mara", "madam", by="hunter")
    assert ok is False and "Ask the person which they want" in why


def test_a_format_one_row_gets_no_form_of_address(tmp_path):
    """An OLD row must not silently acquire "sir"."""
    path = tmp_path / "people.json"
    path.write_text('{"format": 1, "people": [{"label": "hunter", '
                    '"name": "Hunter", "role": "owner", "voice": true}]}',
                    encoding="utf-8")
    reg = Registry.load(path)
    assert reg.usable is True
    assert reg.person("hunter").honorific == ""


def test_a_stored_honorific_nobody_recognises_becomes_no_address(tmp_path):
    path = tmp_path / "people.json"
    path.write_text('{"format": 2, "people": [{"label": "hunter", '
                    '"role": "owner", "honorific": "your grace"}]}',
                    encoding="utf-8")
    reg = Registry.load(path)
    assert reg.person("hunter").honorific == ""
    assert "" in HONORIFICS


# ------------------------------------------------------------- the CLI
def _cli(monkeypatch, tmp_path):
    """scripts/jarvis_people.py with its registry pointed at tmp_path.

    NOTHING HERE TOUCHES ~/.local/state/jarvis. Turning the gate on is
    HIS action; a test that created people.json on this machine would
    change how the live app answers him.
    """
    import importlib
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__)
                           .resolve().parents[1] / "scripts"))
    mod = importlib.import_module("jarvis_people")
    monkeypatch.setattr(mod.PATHS, "OWNER_REGISTRY", tmp_path / "people.json",
                        raising=False)
    said = []
    monkeypatch.setattr(mod, "say", said.append)
    return mod, said


def test_the_cli_refuses_add_without_a_typed_honorific(monkeypatch, tmp_path):
    """argparse itself makes it required with no default, so this exits
    non-zero before any row is created."""
    mod, _said = _cli(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        mod.main(["add", "mara", "--first", "Mara", "--role", "known"])
    assert exc.value.code != 0
    assert not (tmp_path / "people.json").exists()


def test_the_cli_stores_what_he_typed_and_nothing_it_guessed(monkeypatch,
                                                             tmp_path):
    mod, said = _cli(monkeypatch, tmp_path)
    rc = mod.main(["add", "hunter", "--first", "Hunter", "--last",
                   "Peyrovian", "--name", "Hunter", "--role", "owner",
                   "--honorific", "sir", "--face", "hunter",
                   "--face-dim", "512", "--voice"])
    assert rc == 0
    reg = Registry.load(tmp_path / "people.json")
    row = reg.person("hunter")
    assert (row.first, row.last, row.honorific) == ("Hunter", "Peyrovian",
                                                    "sir")
    assert row.face_dim == 512 and row.voice is True
    assert any("addressed as sir" in s for s in said)


def test_maam_is_stored_with_its_apostrophe_however_it_was_typed(monkeypatch,
                                                                tmp_path):
    """The shell types "maam"; the registry stores "ma'am". A quoting
    mistake must never be able to produce a fourth value."""
    mod, _said = _cli(monkeypatch, tmp_path)
    mod.main(["add", "hunter", "--role", "owner", "--honorific", "sir"])
    reg = Registry.load(tmp_path / "people.json")
    ok, why = reg.set_honorific("hunter", "ma'am")
    assert ok and reg.person("hunter").honorific == "ma'am"
    assert mod.honorific_from_word("maam")[0] == "ma'am"
    assert mod.honorific_from_word("none")[0] == ""
    assert mod.honorific_from_word("madam")[1]


# --------------------------------------------- reload without a restart
def test_people_reload_re_reads_the_registry_over_the_socket(tmp_path):
    """Closes the gap where every enrolment cost him a full restart."""
    from types import SimpleNamespace

    from jarvis import cmdsock

    path = tmp_path / "people.json"
    reg = Registry.load(path)
    assert reg.usable is False
    gate = gt.OwnerGate(registry=reg, owner="hunter",
                        get_option=lambda k, d=None: d)
    app = SimpleNamespace(gate=gate)
    assert "does not exist" in cmdsock._reload_people(app)
    _registry(tmp_path)          # he runs the enrolment at the keyboard
    line = cmdsock._reload_people(app)
    assert "1 owner (hunter)" in line
    assert gate.registry.person("mara") is not None


def test_a_reload_that_fails_changes_nothing(tmp_path):
    from types import SimpleNamespace

    from jarvis import cmdsock
    assert "no gate" in cmdsock._reload_people(SimpleNamespace(gate=None))

    class Broken:
        def reload(self):
            raise RuntimeError("no")
    assert "nothing changed" in cmdsock._reload_people(
        SimpleNamespace(gate=Broken()))
