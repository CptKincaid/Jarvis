"""The honorific follows the PERSON across turns, through the cached prompt.

Review finding 1 (09-04): ``brain.set_addressee`` wrote module state, and
every live caller read ``static_system()`` -- ONE cached render, keyed on
nothing -- so whoever the prompt was first built for stayed its addressee
for the life of the process. Mara admitted after Hunter got his "sir"
prompt; Hunter admitted after Mara got told to call him "ma'am". The
existing tests all called ``build_ollama_system`` directly, which is not
the path a turn takes.

The fix is a cache KEYED on the addressee, not a reset: a reset would
resample the few-shots and evict his prefix from Ollama every time a guest
spoke. These tests go through ``static_system()`` -- the real path -- and
through ``_gate_admits``, the real hook, with alternating people.
"""
import re
import pathlib

from jarvis import brain
from jarvis import gate as gate_mod
from tests.test_owner_gate_wiring import MATCHED, _stand_in, _watching

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _owner(text):
    # The addressee CLAUSE, not the bare word: since jarvis-v3 ac934b0 the
    # persona itself carries a literal "say ma'am" example for him.
    return 'Call them "ma\'am"' not in text and "Mara" not in text and \
        "Heather" not in text and "not Hunter" not in text


def test_the_cached_prompt_follows_the_addressee_across_turns():
    """Sir, ma'am, sir, no-address, sir -- five turns, one process."""
    brain.reset_static_prompt()
    brain.set_addressee("", "sir")
    hunter_1 = brain.static_system()
    assert _owner(hunter_1)

    brain.set_addressee("Mara", "ma'am")
    mara = brain.static_system()
    assert "they mean Mara for this reply" in mara
    assert 'Call them "ma\'am"' in mara

    brain.set_addressee("", "sir")
    hunter_2 = brain.static_system()
    assert _owner(hunter_2), "Mara's honorific survived into his turn"
    # ...and not merely equal: the SAME object, so Ollama's prefix for
    # him was never rebuilt behind her and the few-shots never resampled.
    assert hunter_2 is hunter_1

    brain.set_addressee("Alex", "")
    alex = brain.static_system()
    assert 'Call them "Alex" now and then' in alex
    assert 'Call them "ma\'am"' not in alex and "Mara" not in alex

    brain.set_addressee("", "sir")
    assert brain.static_system() is hunter_1


def test_a_guest_prompt_is_built_from_his_few_shot_sample():
    """The few-shots are sampled ONCE per process; a guest's render must
    reuse that sample rather than draw its own."""
    brain.reset_static_prompt()
    brain.set_addressee("", "sir")
    hunter = brain.static_system()
    brain.set_addressee("Mara", "ma'am")
    mara = brain.static_system()
    brain.set_addressee("", "sir")
    shots = brain._STATIC["shots"][brain.register()]
    for user, jarvis in shots:
        assert user in hunter and user in mara
        assert jarvis in hunter and jarvis in mara


def test_every_live_system_message_reads_the_keyed_cache():
    """The seven readers the review counted. Every ``{"role": "system"}``
    message brain.py builds must come from ``static_system()`` -- the one
    function that follows the addressee -- and never from a private copy
    or a direct read of the cache."""
    src = (ROOT / "jarvis" / "brain.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[2]                 # past the module docstring
    readers = re.findall(r'\{"role": "system",\s*"content": ([^}]+)\}', body)
    assert len(readers) >= 6, readers
    # The tool loop hands static_system its OWN reading of the turn
    # (addressee_to=...); every other reader takes the current one.
    assert all(r.startswith("static_system(") for r in readers), readers
    assert "static_system(addressee_to=turn_addr)" in readers, readers
    # No other code reads the cache by hand: every touch of the "system"
    # table sits inside static_system() or reset_static_prompt().
    def span(name):
        start = src.index(f"def {name}(")
        return start, src.index("\ndef ", start + 1)
    allowed = [span("static_system"), span("reset_static_prompt")]
    touches = [m.start() for m in re.finditer(r'_STATIC\["system"\]', src)]
    assert touches
    for at in touches:
        assert any(lo < at < hi for lo, hi in allowed), src[at - 80:at + 40]


def test_the_real_hook_alternates_people_through_the_real_prompt(tmp_path):
    """``_gate_admits`` on the stand-in: Hunter by voice, Heather by face,
    Hunter by voice again. After each judged turn the prompt the tool
    loop would send (``static_system()``) is written for THAT person."""
    a = _stand_in(tmp_path, known=True)
    brain.reset_static_prompt()

    assert a._gate_admits("what time is it", MATCHED) is True
    assert a._gate_who == "hunter"
    first = brain.static_system()
    assert _owner(first)

    _watching(a, "heather")
    assert a._gate_admits("what time is it", {}) is True
    assert (a._gate_who, a._gate_how) == ("heather", gate_mod.HOW_FACE)
    hers = brain.static_system()
    assert "they mean Heather for this reply" in hers
    assert 'Call them "ma\'am"' in hers

    _watching(a, "hunter")
    assert a._gate_admits("what time is it", MATCHED) is True
    assert a._gate_who == "hunter"
    again = brain.static_system()
    assert _owner(again), "ma'am reached Hunter"
    assert again is first
