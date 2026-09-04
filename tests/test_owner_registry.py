"""The one identity model, and the rule that its own breakage never costs
him his assistant.

A gate that locks him out because its storage broke is worse than no gate,
so every unreadable state of ``people.json`` resolves to ``usable=False``
and one plain reason -- never an exception, never a refusal.
"""
import json
import os
import stat

import pytest

from jarvis import identity as ident
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry


@pytest.fixture()
def reg_path(tmp_path):
    return tmp_path / "people.json"


def _owner(label="hunter"):
    return Person(label=label, name="Hunter", role=ROLE_OWNER, voice=True)


# ------------------------------------------------------- the single label
def test_owner_label_is_derived_in_exactly_one_place():
    """cast.py, commander._face_owner and face_enrol.owner_label were three
    independent derivations of the same string. They are one now."""
    class Cfg:
        def get(self, key, default=None):
            return "Hunter" if key == "user.name" else default
    assert ident.owner_label(Cfg()) == "hunter"


def test_owner_label_survives_a_config_that_cannot_answer():
    class Broken:
        def get(self, *a, **k):
            raise RuntimeError("no config")
    assert ident.owner_label(Broken()) == "hunter"
    assert ident.owner_label(None) == "hunter"


def test_a_username_the_gallery_cannot_store_is_handed_back_unrepaired():
    """NOT repaired quietly. The label becomes an npz key and a registry
    key, and scripts/face_enrol.target_label checks it against the gallery's
    own pattern and stops before the camera with a sentence naming
    `user.name`. Mangling "Jose" with an accent into "jos" here would enrol
    him under a name he never chose and never sees -- and would delete the
    one signal that tells him what to fix."""
    class Cfg:
        def __init__(self, name):
            self.name = name

        def get(self, key, default=None):
            return self.name if key == "user.name" else default
    assert ident.owner_label(Cfg("Jos\u00e9")) == "jos\u00e9"
    assert not ident.LABEL_RX.match(ident.owner_label(Cfg("Jos\u00e9")))
    # ...and an empty or unusable name still falls back rather than crashing.
    for raw in ("  ", "!!!", None):
        assert ident.owner_label(Cfg(raw)) == "hunter"


def test_an_unstorable_label_is_refused_by_the_registry_out_loud(tmp_path):
    r = Registry(path=tmp_path / "people.json")
    ok, why = r.add_person(Person(label="jos\u00e9", role=ROLE_OWNER))
    assert ok is False and "store" in why


# ------------------------------------------------- it never fails loudly
def test_a_missing_file_is_not_an_error_it_is_an_off_switch(reg_path):
    r = Registry.load(reg_path)
    assert r.usable is False and r.people == []
    assert r.fault and "people.json" in r.fault


def test_corrupt_json_turns_the_gate_off_and_says_why(reg_path):
    reg_path.write_text("{not json at all")
    r = Registry.load(reg_path)
    assert r.usable is False
    assert r.fault


def test_the_wrong_shape_turns_the_gate_off(reg_path):
    reg_path.write_text(json.dumps(["a", "list", "of", "nothing"]))
    assert Registry.load(reg_path).usable is False


def test_an_unreadable_file_turns_the_gate_off(reg_path):
    reg_path.write_text(json.dumps({"format": 1, "people": []}))
    os.chmod(reg_path, 0o000)
    try:
        r = Registry.load(reg_path)
    finally:
        os.chmod(reg_path, 0o600)
    assert r.usable is False and r.fault


def test_a_file_with_no_owner_is_not_usable(reg_path):
    reg_path.write_text(json.dumps({"format": 1, "people": [
        {"label": "heather", "role": ROLE_KNOWN}]}))
    r = Registry.load(reg_path)
    assert r.usable is False
    assert "owner" in r.fault


def test_one_bad_row_does_not_cost_the_good_ones(reg_path):
    reg_path.write_text(json.dumps({"format": 1, "people": [
        {"label": "hunter", "role": ROLE_OWNER},
        {"label": "!!!bad", "role": ROLE_KNOWN},
        "not even a dict"]}))
    r = Registry.load(reg_path)
    assert r.usable is True
    assert r.labels() == ("hunter",)


# --------------------------------------------------------------- writing
def test_a_saved_registry_is_private_and_written_atomically(reg_path):
    r = Registry(path=reg_path)
    assert r.add_person(_owner())[0] is True
    assert r.save() is True
    mode = stat.S_IMODE(reg_path.stat().st_mode)
    assert mode == 0o600, oct(mode)
    assert not list(reg_path.parent.glob(".people*"))
    again = Registry.load(reg_path)
    assert again.usable and again.role_of("hunter") == ROLE_OWNER


def test_a_second_owner_cannot_be_created_by_accident(reg_path):
    r = Registry(path=reg_path)
    r.add_person(_owner())
    ok, why = r.add_person(Person(label="heather", role=ROLE_OWNER))
    assert ok is False and "hunter" in why
    ok, why = r.add_person(Person(label="heather", role=ROLE_OWNER),
                           confirm_existing_owner="hunter")
    assert ok is True, why
    assert sorted(p.label for p in r.owners()) == ["heather", "hunter"]


def test_promoting_to_owner_needs_the_same_confirmation(reg_path):
    r = Registry(path=reg_path)
    r.add_person(_owner())
    r.add_person(Person(label="heather", role=ROLE_KNOWN))
    assert r.set_role("heather", ROLE_OWNER)[0] is False
    assert r.set_role("heather", ROLE_OWNER,
                      confirm_existing_owner="hunter")[0] is True


def test_the_last_owner_cannot_be_demoted_or_forgotten(reg_path):
    """Removing the only owner is the same brick as a corrupt file, except
    it would look deliberate. Refused with a reason."""
    r = Registry(path=reg_path)
    r.add_person(_owner())
    assert r.set_role("hunter", ROLE_KNOWN)[0] is False
    assert r.forget("hunter")[0] is False


def test_a_known_person_is_stored_with_how_consent_was_taken(reg_path):
    r = Registry(path=reg_path)
    r.add_person(_owner())
    r.add_person(Person(label="heather", name="Heather", role=ROLE_KNOWN,
                        face="heather", face_dim=128, consent="typed"))
    r.save()
    p = Registry.load(reg_path).person("heather")
    assert p.consent == "typed" and p.face_dim == 128


def test_a_role_that_is_not_a_role_is_refused(reg_path):
    r = Registry(path=reg_path)
    r.add_person(_owner())
    assert r.add_person(Person(label="x", role="admin"))[0] is False


# ------------------------------------------------------ never the secret
def test_the_registry_is_not_the_assistant_config(reg_path):
    """The hashes live in their own 0600 file on purpose: AssistantConfig
    is deep-copied everywhere and its __repr__ prints redacted(), so the
    strongest way to keep a hash out of a log is to keep it out of that
    object entirely."""
    from jarvis.assistant_config import DEFAULTS, SECRET_KEYS
    flat = json.dumps(DEFAULTS)
    assert "phrase_hash" not in flat and "code_hash" not in flat
    assert not any("phrase" in k or "code" in k for k in SECRET_KEYS)


def test_a_person_never_prints_its_own_hashes(reg_path):
    p = Person(label="hunter", role=ROLE_OWNER,
               phrase_hash="scrypt$1$2$3$AAAA$BBBB", code_hash="scrypt$x")
    printed = "%r %s %s" % (p, p, json.dumps(p.redacted()))
    assert "AAAA" not in printed and "BBBB" not in printed
    assert "scrypt" not in printed


# ------------------------------------------------------- the startup line
def test_the_startup_line_says_which_mode_is_live_and_why(reg_path):
    r = Registry(path=reg_path)
    r.add_person(_owner())
    line = ident.startup_line(r, "shadow", voice_ok=True, face_ok=False,
                              face_why="the gallery is empty")
    assert "SHADOW" in line and "hunter" in line
    assert "the gallery is empty" in line
    assert "Nothing is being refused" in line

    off = ident.startup_line(Registry.load(reg_path.with_name("gone.json")),
                             "off", voice_ok=True, face_ok=True)
    assert "OFF" in off and "answered" in off


OVERSTATEMENTS = ("authenticat", "secure ", "security", "locked out of",
                  "password protect")


def no_overstatement(mod):
    """Recognition, not a lock: a photo defeats the face check and a
    recording defeats the voice check. Nothing may overstate that, in any
    line the user or the log ever sees -- or in the prose beside it, which
    is what the next person reads before changing the wording."""
    import inspect
    for line in inspect.getsource(mod).lower().splitlines():
        for word in OVERSTATEMENTS:
            if word in line:
                raise AssertionError("%s: %r" % (mod.__name__, line))


def test_no_wording_calls_this_a_lock():
    no_overstatement(ident)
