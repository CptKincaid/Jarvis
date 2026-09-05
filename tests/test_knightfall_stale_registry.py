"""Rotating the Knightfall code must not write a stale people book back.

WHY THIS FILE EXISTS. On 2026-09-05 an attack on the Users tab found the
defect it had just closed there still living one method away.
``JarvisApp._knightfall_rotate`` read ``self.gate.registry`` -- the copy
loaded at BOOT -- called ``set_secret`` on it and then ``save()``d that
whole object over the file. Anything written to the people book since the
process started was overwritten by the in-memory copy that never knew
about it. Measured on the tree of the day: pressing "Email me a new
Knightfall code" DELETED a person enrolled at a terminal since boot, and
REVERTED a passphrase set at a terminal to empty.

That matters more here than almost anywhere: jarvis/ui/users_page.py says
on screen that the people book has no history and no backup. The lost row
is simply gone.

The rule, the same one the Users tab now follows: a write decides from the
file as it is AT THE WRITE, never from a memory of it.
"""
from types import SimpleNamespace

from jarvis import app as app_mod
from jarvis import gate as gate_mod
from jarvis import passphrase as pp
from jarvis.identity import Person, Registry

from tests.test_notes_mail import GMAIL_CFG, FakeCfg
from tests.test_owner_gate import _registry
from tests.test_send_file import FakeSMTP


def _app(tmp_path, mode="shadow"):
    reg = _registry(tmp_path, code=True)
    opts = {"owner.mode": mode}
    a = SimpleNamespace()
    a.assistant = FakeCfg(GMAIL_CFG)
    a.get_option = lambda k, d=None: opts.get(k, d)
    a.gate = gate_mod.OwnerGate(registry=reg, owner="hunter",
                                get_option=a.get_option)
    for name in ("knightfall_code", "knightfall_new_code",
                 "_knightfall_rotate"):
        setattr(a, name, getattr(app_mod.JarvisApp, name).__get__(a))
    return a


def _enrol_at_a_terminal(path, label):
    """What scripts/jarvis_people.py does: load the FILE, add, save."""
    disk = Registry.load(path)
    ok, why = disk.add_person(Person(label=label, name=label.title(),
                                     role="known", consent="owner"))
    assert ok, why
    assert disk.save()


def test_a_person_enrolled_since_boot_survives_a_rotation(tmp_path, monkeypatch):
    a = _app(tmp_path)
    path = a.gate.registry.path
    before = sorted(p.label for p in a.gate.registry.people)
    _enrol_at_a_terminal(path, "marchbanks")
    assert sorted(p.label for p in Registry.load(path).people) == \
        sorted(before + ["marchbanks"]), "the terminal write landed"

    FakeSMTP.made = []
    a._knightfall_rotate("hunter", smtp=FakeSMTP)

    on_disk = sorted(p.label for p in Registry.load(path).people)
    assert "marchbanks" in on_disk, (
        "the rotation wrote a boot-time copy back over him: %s" % on_disk)


def test_a_phrase_set_since_boot_survives_a_rotation(tmp_path):
    a = _app(tmp_path)
    path = a.gate.registry.path
    disk = Registry.load(path)
    ok, why = disk.set_secret("hunter", "phrase_hash",
                              pp.hash_secret("xxx not a real phrase xxx"))
    assert ok, why
    assert disk.save()
    was = Registry.load(path).person("hunter").phrase_hash
    assert was, "the terminal write landed"

    FakeSMTP.made = []
    a._knightfall_rotate("hunter", smtp=FakeSMTP)

    now = Registry.load(path).person("hunter").phrase_hash
    assert now == was, "the rotation reverted the phrase set at a terminal"


def test_the_rotation_still_stores_the_new_code(tmp_path):
    """The guard must not cost the feature: a rotation still rotates."""
    a = _app(tmp_path)
    path = a.gate.registry.path
    old = Registry.load(path).person("hunter").code_hash
    FakeSMTP.made = []
    line, mailed = a._knightfall_rotate("hunter", smtp=FakeSMTP)
    assert mailed is True, line
    assert Registry.load(path).person("hunter").code_hash != old
