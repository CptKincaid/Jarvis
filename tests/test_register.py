"""Register control: "formal mode", "banter up" -- one cache miss, not one
per turn.

The prompt half (static-prefix stability, the clause, the dropped joke
family) is tested in tests/test_persona.py; this file owns the spoken
command: the phrasings, both stores, the canned answer, the background
re-warm, and the register's reach into the courtesies and the diagnostics.
"""
from types import SimpleNamespace

import pytest

from jarvis import brain as brain_mod
from jarvis import selfstate
from jarvis.commander import (COURTESY_BY_REGISTER, COURTESY_REPLIES,
                              REGISTER_ALREADY_LINES, REGISTER_LINES,
                              _greeting_line, _h_courtesy, _h_diagnostics,
                              _h_register, courtesy_reply, register_kind)


@pytest.fixture
def fake_brain():
    """A brain stand-in recording set_register calls."""
    return SimpleNamespace(calls=[], name="normal")


@pytest.fixture
def c(fake_brain):
    """A commander stand-in with brain, assistant config and memory."""
    cfg = SimpleNamespace(store={},
                          set=lambda k, v: cfg.store.__setitem__(k, v),
                          get=lambda k, d=None: cfg.store.get(k, d))
    mem = SimpleNamespace(prefs={},
                          set_preference=lambda k, v: mem.prefs.__setitem__(k, v))

    def set_register(name):
        fake_brain.calls.append(name)
        changed = name != fake_brain.name
        fake_brain.name = name
        return changed

    fake_brain.set_register = set_register
    fake_brain.register = lambda: fake_brain.name
    services = {"brain": fake_brain, "assistant": cfg, "memory": mem}
    return SimpleNamespace(_svc=services.get, cfg=cfg, mem=mem,
                           brain=fake_brain)


# --------------------------------------------------------------- phrasings
def test_the_phrasings():
    for said, want in (
            ("formal mode", "formal"), ("be more formal", "formal"),
            ("switch to formal", "formal"), ("be serious", "formal"),
            ("less banter", "formal"), ("cut the jokes", "formal"),
            ("banter up", "banter"), ("more banter", "banter"),
            ("be more playful", "banter"), ("loosen up", "banter"),
            ("dial up the wit", "banter"),
            ("back to normal", "normal"), ("your usual", "normal"),
            ("banter down", "normal"), ("less formal", "normal"),
            ("stop being so formal", "normal"),
            ("turn down the banter", "normal")):
        assert register_kind(said) == want, said
    # whole utterance only: these are questions and requests for the model
    for said in ("what's the formal name for it", "tell me a joke",
                 "is this a formal dinner", "loosen up the bolt", ""):
        assert register_kind(said) is None, said


# ---------------------------------------------------------------- handler
def test_the_change_reaches_both_stores_and_the_prompt(c):
    res = _h_register(c, "formal mode", "formal")
    assert res.handled and res.speak and res.reply == REGISTER_LINES["formal"]
    # assistant.json is the source of truth (read back at app start);
    # memory's preferences.json is the record of what he asked for
    assert c.cfg.store["persona.register"] == "formal"
    assert c.mem.prefs["persona.register"] == "formal"
    assert c.brain.calls == ["formal"]


def test_a_no_op_change_costs_nothing_and_says_so(c):
    _h_register(c, "formal mode", "formal")
    res = _h_register(c, "be more formal", "formal")
    assert res.reply == REGISTER_ALREADY_LINES["formal"]
    assert res.handled and res.speak


def test_the_answer_is_canned_and_never_a_model_turn(c):
    """The prefix reprocess the change just triggered is running behind
    this line; a model turn here would queue up behind its own cache miss."""
    for want in ("formal", "banter", "normal"):
        for line in (REGISTER_LINES[want], REGISTER_ALREADY_LINES[want]):
            assert "{" not in line and not any(ch.isdigit() for ch in line)
            assert "sir" in line and line.endswith(".")
            assert len(line.split()) <= 10, line     # one short sentence
    assert c._svc("router") is None            # no route, no brain.chat


def test_a_config_that_cannot_persist_declines_the_command():
    """A register that forgets itself overnight is worse than none: fall
    through to the model rather than pretend it stuck."""
    brain = SimpleNamespace(set_register=lambda n: True, register=lambda: "normal")
    c = SimpleNamespace(_svc={"brain": brain}.get)
    assert _h_register(c, "formal mode", "formal") is None
    # ... and with no brain at all there is nothing to change
    cfg = SimpleNamespace(set=lambda k, v: None)
    c2 = SimpleNamespace(_svc={"assistant": cfg}.get)
    assert _h_register(c2, "formal mode", "formal") is None


def test_a_brain_that_raises_does_not_break_the_turn(c):
    def boom(_name):
        raise RuntimeError("ollama on fire")
    c.brain.set_register = boom
    assert _h_register(c, "formal mode", "formal") is None


# -------------------------------------------------- the register's reach
def test_the_courtesies_change_register_and_stay_prewarmable(c):
    for kind in ("presence", "thanks", "goodnight", "availability"):
        formal = set(COURTESY_BY_REGISTER["formal"][kind])
        banter = set(COURTESY_BY_REGISTER["banter"][kind])
        assert formal and banter and formal != banter
        for line in formal | banter:
            assert "{" not in line              # a template cannot be prewarmed
            assert not any(ch.isdigit() for ch in line)
            assert "sir" in line.lower()
    # an unknown / normal register falls back to the shipped lines
    assert courtesy_reply("presence") in COURTESY_REPLIES["presence"]
    assert courtesy_reply("presence", register="normal") in \
        COURTESY_REPLIES["presence"]
    assert courtesy_reply("presence", register="formal") in \
        COURTESY_BY_REGISTER["formal"]["presence"]


def test_the_greeting_is_still_chosen_by_the_clock_in_every_register():
    """"Good afternoon" answered with "Good morning" is worse than no
    variation at all."""
    from datetime import datetime
    for register in (None, "formal", "banter"):
        for hour, want in ((8, "morning"), (14, "afternoon"), (21, "evening")):
            line = _greeting_line(datetime(2026, 8, 30, hour), register=register)
            assert want in line.lower(), (register, hour)


def test_a_courtesy_handler_reads_the_register_from_the_brain(c):
    c.brain.name = "banter"
    res = _h_courtesy(c, "are you there", "presence")
    assert res.reply in COURTESY_BY_REGISTER["banter"]["presence"]
    c.brain.name = "formal"
    res = _h_courtesy(c, "thanks", "thanks")
    assert res.reply in COURTESY_BY_REGISTER["formal"]["thanks"]


def test_formal_gets_the_plain_diagnostics_sheet(c):
    """The film register is an aside, and the register that bans asides
    bans this one too."""
    sheet = {"uptime_s": 60, "stt_model": "small", "brain_model": "gemma4",
             "tts_engine": "f5", "gpu_mhz": 2424.0, "turns_today": 3}
    plain = selfstate.diagnostics_line(sheet)
    services = {"diagnostics": lambda: plain,
                "self_state": lambda full=True: sheet,
                "brain": c.brain}
    cc = SimpleNamespace(_svc=services.get)
    c.brain.name = "formal"
    assert _h_diagnostics(cc, "run diagnostics", None).reply == plain
    c.brain.name = "normal"
    assert _h_diagnostics(cc, "run diagnostics", None).reply == \
        selfstate.stark_line(sheet)


# ------------------------------------------------------------- the module
def test_the_module_seam_warms_the_new_prefix_off_the_audio_path(monkeypatch):
    """Without the background re-warm the one cache miss is paid by
    whatever he asks NEXT -- the one turn he is listening to."""
    warmed = []
    monkeypatch.setattr(brain_mod, "warm_static",
                        lambda *a, **k: warmed.append(brain_mod.register()))
    b = brain_mod.JarvisBrain.__new__(brain_mod.JarvisBrain)
    was = brain_mod.register()
    try:
        assert b.set_register("banter") is True
        for _ in range(50):                    # the thread is a daemon
            if warmed:
                break
            import time
            time.sleep(0.02)
        assert warmed == ["banter"]
        assert b.register() == "banter"
        # a rejected change warms nothing
        assert b.set_register("banter") is False
        assert b.set_register("nonsense") is False
        assert warmed == ["banter"]
    finally:
        brain_mod._REGISTER["name"] = was
        brain_mod.reset_static_prompt()


def test_warm_static_never_raises_and_declines_a_lent_model(monkeypatch):
    monkeypatch.setitem(brain_mod._RESIDENCY, "lent", True)
    assert brain_mod.warm_static() is False
    monkeypatch.setitem(brain_mod._RESIDENCY, "lent", False)

    def boom(*a, **k):
        raise OSError("ollama down")
    monkeypatch.setattr(brain_mod, "_http", boom)
    assert brain_mod.warm_static() is False
