"""The five app seams the USERS tab writes through.

The tab never touches people.json. It calls these, each of which reloads,
mutates, saves and then re-reads the gate so the change is LIVE without a
restart -- and each of which returns ONE line to toast and never a hash.

No window is built here and no registry of his is opened: the app object is
the real ``JarvisApp`` methods bound to a stand-in holding a tmp gate, which
is how the rest of the suite exercises app methods without constructing the
whole assistant.
"""
import json

from jarvis import app as app_mod
from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry

FAKE_CODE = "zzz-not-a-real-code-zzz"


class Stub:
    """Just enough JarvisApp for the people seams: a gate and nothing else."""

    def __init__(self, gate):
        self.gate = gate

    _people_write = app_mod.JarvisApp._people_write
    people_snapshot = app_mod.JarvisApp.people_snapshot
    people_unlock = app_mod.JarvisApp.people_unlock
    people_add = app_mod.JarvisApp.people_add
    people_set_role = app_mod.JarvisApp.people_set_role
    people_forget = app_mod.JarvisApp.people_forget


def _app(tmp_path, *, code=False, known=False, raw=None):
    path = tmp_path / "people.json"
    if raw is not None:
        path.write_text(raw)
    else:
        reg = Registry(path=path)
        reg.add_person(Person(label="alderman", name="Alderman",
                              role=ROLE_OWNER, voice=True, face="alderman",
                              face_dim=128))
        if known:
            reg.add_person(Person(label="pemberton", name="Pemberton",
                                  role=ROLE_KNOWN, consent="typed"))
        if code:
            reg.set_secret("alderman", "code_hash", pp.hash_secret(FAKE_CODE))
        assert reg.save()
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: d)
    return Stub(gate), path


# ==================================================== what the page reads
def test_the_snapshot_carries_no_hash_and_says_which_mode_is_live(tmp_path):
    app, path = _app(tmp_path, code=True, known=True)
    snap = app.people_snapshot()
    assert "zzz" not in json.dumps(snap)
    assert "code_hash" not in json.dumps(snap)
    assert "phrase_hash" not in json.dumps(snap)
    labels = [p["label"] for p in snap["people"]]
    assert labels == ["alderman", "pemberton"]
    assert snap["people"][0]["has_code"] is True
    assert snap["gate_line"].startswith("owner-gate:")
    assert snap["fault_kind"] == ""
    assert snap["path"] == str(path)
    assert snap["admin"] == gt.ADMIN_CODE


def test_the_snapshot_names_the_fault_when_the_file_is_broken(tmp_path):
    app, path = _app(tmp_path, raw='{"people": [ truncated')
    snap = app.people_snapshot()
    assert snap["fault_kind"] == "malformed"
    assert snap["admin"] == gt.ADMIN_REFUSE
    assert snap["people"] == []


def test_the_snapshot_lists_the_gallery_without_opening_a_lens(tmp_path,
                                                               monkeypatch):
    """The face chip needs to know whether a pointer resolves. Gallery
    LABELS are strings beside 128 floats; nothing here reads a frame and
    nothing here imports cv2."""
    app, _path = _app(tmp_path)
    monkeypatch.setattr(app_mod, "gallery_labels", lambda: ("alderman",))
    assert app.people_snapshot()["gallery"] == ["alderman"]


def test_a_snapshot_never_raises_even_with_no_gate(tmp_path):
    snap = Stub(None).people_snapshot()
    assert snap["people"] == [] and snap["admin"] == gt.ADMIN_REFUSE


# ======================================================== the unlock seam
def test_the_right_code_unlocks_and_the_wrong_one_does_not(tmp_path):
    app, _path = _app(tmp_path, code=True)
    ok, line = app.people_unlock(FAKE_CODE)
    assert ok is True
    assert FAKE_CODE not in line
    bad, why = app.people_unlock("nope")
    assert bad is False
    assert "not a code I know" in why


def test_the_unlock_opens_no_voice_window(tmp_path):
    """The users tab's unlock is administrative. Knightfall's is not, and
    that difference must survive somebody reading the two side by side."""
    app, _path = _app(tmp_path, code=True)
    opened = []
    app.gate.open_window = lambda *a, **k: opened.append(a)
    assert app.people_unlock(FAKE_CODE)[0] is True
    assert opened == []
    assert app.gate._granted(None) == ""


def test_the_unlock_spends_the_code_counter_not_the_phrase_one(tmp_path):
    app, _path = _app(tmp_path, code=True)
    before = len(app.gate.phrase_attempts._hits)
    app.people_unlock("nope")
    assert len(app.gate.code_attempts._hits) == 1
    assert len(app.gate.phrase_attempts._hits) == before


def test_the_unlock_is_refused_when_no_code_has_been_set(tmp_path):
    app, _path = _app(tmp_path)
    ok, why = app.people_unlock("anything")
    assert ok is False and "no override code" in why


# ========================================================== the writes
def test_adding_somebody_takes_effect_without_a_restart(tmp_path):
    app, path = _app(tmp_path)
    ok, line = app.people_add(label="pemberton", name="Pemberton",
                              role=ROLE_KNOWN, consent="console")
    assert ok is True, line
    # on disk...
    assert "pemberton" in json.loads(path.read_text())["people"][1]["label"]
    # ...and in the LIVE gate, with no restart
    assert app.gate.registry.role_of("pemberton") == ROLE_KNOWN


def test_a_second_owner_needs_the_first_named(tmp_path):
    app, _path = _app(tmp_path)
    ok, why = app.people_add(label="brightwell", role=ROLE_OWNER,
                             consent="console")
    assert ok is False and "alderman" in why
    ok2, _line = app.people_add(label="brightwell", role=ROLE_OWNER,
                                consent="console",
                                confirm_existing_owner="alderman")
    assert ok2 is True


def test_a_non_owner_cannot_be_added_without_a_consent_record(tmp_path):
    """The record is the whole point: a row whose consent field is empty
    is a row nobody can show was agreed to."""
    app, _path = _app(tmp_path)
    ok, why = app.people_add(label="pemberton", role=ROLE_KNOWN, consent="")
    assert ok is False and "consent" in why.lower()


def test_a_write_is_refused_outright_on_a_corrupt_registry(tmp_path):
    """THE HAZARD. reg.people is EMPTY for a file that failed to parse, so
    an add-then-save writes a one-row registry over it."""
    app, path = _app(tmp_path, raw='{"people": [ truncated')
    before = path.read_text()
    ok, why = app.people_add(label="pemberton", role=ROLE_KNOWN,
                             consent="console")
    assert ok is False
    assert "read" in why or "repair" in why.lower()
    assert path.read_text() == before, "the broken file was overwritten"


def test_the_first_owner_can_still_be_made_with_no_file_at_all(tmp_path):
    """A fresh install must not be a brick."""
    path = tmp_path / "people.json"
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: d)
    app = Stub(gate)
    assert app.people_snapshot()["fault_kind"] == "missing"
    ok, line = app.people_add(label="alderman", name="Alderman",
                              role=ROLE_OWNER, consent="owner")
    assert ok is True, line
    assert path.exists()
    assert app.gate.registry.usable is True


def test_changing_a_role_refuses_to_leave_nobody_in_charge(tmp_path):
    app, _path = _app(tmp_path, known=True)
    ok, why = app.people_set_role("alderman", ROLE_KNOWN)
    assert ok is False and "only owner" in why
    ok2, _l = app.people_set_role("pemberton", ROLE_OWNER,
                                  confirm_existing_owner="alderman")
    assert ok2 is True
    assert app.gate.registry.role_of("pemberton") == ROLE_OWNER


def test_forgetting_removes_the_row_and_says_what_it_did_not(tmp_path):
    app, path = _app(tmp_path, known=True)
    ok, line = app.people_forget("pemberton")
    assert ok is True
    assert "gallery" in line.lower()
    assert [p["label"] for p in json.loads(path.read_text())["people"]] == \
        ["alderman"]
    assert app.gate.registry.person("pemberton") is None


def test_the_sole_owner_cannot_be_forgotten(tmp_path):
    app, _path = _app(tmp_path)
    ok, why = app.people_forget("alderman")
    assert ok is False and "only owner" in why


def test_every_write_re_reads_the_file_first(tmp_path):
    """THE TWO-WRITER WINDOW, made as small as it can be. The terminal
    tool is a separate PROCESS, so nothing in-process can lock against it;
    re-reading immediately before each write shrinks the window without
    closing it, and that is stated rather than hidden."""
    app, path = _app(tmp_path)
    # somebody else edits the file behind the tab's back
    reg = Registry.load(path)
    reg.add_person(Person(label="brightwell", role=ROLE_KNOWN,
                          consent="typed"))
    assert reg.save()
    ok, _line = app.people_add(label="pemberton", role=ROLE_KNOWN,
                               consent="console")
    assert ok is True
    labels = [p["label"] for p in json.loads(path.read_text())["people"]]
    assert labels == ["alderman", "brightwell", "pemberton"], \
        "the terminal's row was overwritten instead of kept"


def test_no_seam_ever_returns_a_hash(tmp_path):
    app, _path = _app(tmp_path, code=True, known=True)
    lines = [app.people_unlock("nope")[1],
             app.people_add(label="brightwell", role=ROLE_KNOWN,
                            consent="console")[1],
             app.people_set_role("brightwell", ROLE_KNOWN)[1],
             app.people_forget("brightwell")[1]]
    hashed = app.gate.registry.owners()[0].code_hash
    assert hashed
    for line in lines:
        assert hashed not in line
        assert FAKE_CODE not in line


# ====================================================== the wiring itself
def test_the_five_seams_are_offered_to_the_ui():
    """build_ui_services drops what the UI does not declare, so passing
    them is safe in either merge order -- but they have to be passed."""
    import inspect
    src = inspect.getsource(app_mod.JarvisApp.ui_service_kwargs)
    for name in ("people_snapshot", "people_unlock", "people_add",
                 "people_set_role", "people_forget"):
        assert "%s=self.%s" % (name, name) in src


def test_the_ui_declares_all_five():
    import dataclasses
    from jarvis.ui.main_window import Services
    names = {f.name for f in dataclasses.fields(Services)}
    assert {"people_snapshot", "people_unlock", "people_add",
            "people_set_role", "people_forget"} <= names
