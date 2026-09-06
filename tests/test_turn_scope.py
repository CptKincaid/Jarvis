"""The scope is PER TURN: the owner is the default for every non-voice path
and every proactive call, and a voice attribution expires.

Round-2 review (09-04), finding A -- measured, with the reviewer's own
probe: ``brain.set_addressee`` was written only by the voice path
(``app._gate_admits``) and never expired, so after ONE admitted guest turn
Hunter's typed / CLI / phone / Discord turns and his own first-wake
briefing (``force_tool="get_briefing"``) all ran scoped as the guest --
offered get_time and get_weather, refused with "That one's Hunter's,
Mara." The spoken honorific had a 120 s TTL; the scope had none.

Now (jarvis/scope.py):
  * the attribution carries a stamp and expires after the SAME
    honorific.ADDRESSEE_TTL;
  * ``app._dispatch`` clears it for every source the gate does not judge;
  * the briefing clears it before it asks;
  * ``_chat_sync`` reads it EXACTLY once and hands that reading to the
    prompt, the schemas and every refusal.

Fakes only: the mocked Ollama from tests/test_brain_tools.py, the app
stand-ins from tests/test_owner_gate_wiring.py and
tests/test_briefing_offer.py. No mic, no camera, no model, no live app.
"""
import pathlib
import re
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import brain as brain_mod
from jarvis import gate as gate_mod
from jarvis import honorific as honorific_mod
from jarvis import scope as scope_mod
from tests.test_brain_tools import (FakeContext, FakeMemory, FakeOllama,
                                    make_registry, text_reply, tool_reply)
from tests.test_briefing_offer import _app as _briefing_app, _clock
from tests.test_owner_gate_wiring import MATCHED, _stand_in, _watching

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARA_LINE = "That one's Hunter's, Mara. I can give you the time and the weather."


@pytest.fixture
def loop(monkeypatch):
    brain_mod.reset_static_prompt()
    record = []
    reg = make_registry(record)
    monkeypatch.setattr(brain_mod, "_REGISTRY", reg)
    fake = FakeOllama()
    monkeypatch.setattr(brain_mod, "_http", fake)
    b = brain_mod.JarvisBrain(context=FakeContext(), memory=FakeMemory())
    return b, fake, record, reg


def _offered(payload):
    return sorted(t["function"]["name"] for t in payload.get("tools", []))


def _at(monkeypatch, seconds):
    """Freeze the scope's clock (its ONE dependency on time)."""
    monkeypatch.setattr(scope_mod, "_now",
                        lambda now: float(seconds) if now is None else float(now))


# ------------------------------------------------------------ the module
def test_the_scope_expires_with_the_spoken_honorific():
    """One TTL for both halves of the same attribution."""
    scope_mod.set_addressee("Mara", "ma'am", now=0.0)
    assert scope_mod.addressee(now=0.0) == ("Mara", "ma'am")
    assert scope_mod.addressee(now=honorific_mod.ADDRESSEE_TTL - 1) == \
        ("Mara", "ma'am")
    assert scope_mod.addressee(now=honorific_mod.ADDRESSEE_TTL + 1) == \
        scope_mod.OWNER
    assert scope_mod.is_owner(now=honorific_mod.ADDRESSEE_TTL + 1)
    assert honorific_mod.ADDRESSEE_TTL == 120.0
    # ...and it is the honorific module's number, not a second copy.
    src = (ROOT / "jarvis" / "scope.py").read_text(encoding="utf-8")
    assert "from jarvis.honorific import ADDRESSEE_TTL" in src
    assert not re.search(r"^\s*ADDRESSEE_TTL\s*=", src, re.M)


def test_clearing_is_the_owner_at_once():
    scope_mod.set_addressee("Mara", "ma'am")
    scope_mod.clear_addressee()
    assert scope_mod.addressee() == ("", "sir")
    assert scope_mod.owner_only("calendar") == ""
    scope_mod.set_addressee("Mara", "ma'am")
    assert scope_mod.owner_only("calendar") == MARA_LINE
    assert scope_mod.owner_only("calendar", who="Heather").startswith(
        "That one's Hunter's, Heather.")
    assert scope_mod.owner_only("calendar", who="") == ""


def test_brain_exposes_the_same_state_under_its_old_names():
    brain_mod.set_addressee("Mara", "ma'am", now=0.0)
    assert brain_mod.addressee(now=1.0) == ("Mara", "ma'am")
    assert brain_mod.addressee(now=1000.0) == ("", "sir")
    assert brain_mod.is_owner_addressee(now=1000.0)
    assert brain_mod.KNOWN_TOOL_LINE is scope_mod.HIS_LINE


# ---------------------------------------------------------- the tool loop
def test_ten_minutes_after_a_guest_spoke_his_briefing_is_his(loop, monkeypatch):
    """The reviewer's PART 2, with a clock: Mara at t=0, his forced
    briefing at t=600. Before: refused, handler never ran."""
    b, fake, record, reg = loop
    _at(monkeypatch, 0.0)
    brain_mod.set_addressee("Mara", "ma'am")
    _at(monkeypatch, 600.0)
    fake.replies = [text_reply("Seventy-two and sunny, sir.")]
    tags = b._chat_sync("my morning briefing", force_tool="get_briefing")
    assert record == [("get_briefing",)]
    assert _offered(fake.chat_payloads()[0]) == sorted(reg.names())
    assert tags[-1] == ("SPEAK", "Seventy-two and sunny, sir.")


def test_within_the_ttl_a_guest_turn_is_still_hers(loop, monkeypatch):
    """The other side of the same clock: the TTL is not "always his"."""
    b, fake, record, reg = loop
    _at(monkeypatch, 0.0)
    brain_mod.set_addressee("Mara", "ma'am")
    _at(monkeypatch, 60.0)
    tags = b._chat_sync("my morning briefing", force_tool="get_briefing")
    assert record == []
    assert tags == [("SPEAK", MARA_LINE)]


def test_a_proactive_caller_names_the_owner_and_is_not_scoped_by_the_room(loop):
    """``chat(addressee=scope.OWNER)``: whoever the gate last named."""
    b, fake, record, reg = loop
    brain_mod.set_addressee("Mara", "ma'am")
    fake.replies = [text_reply("Seventy-two and sunny, sir.")]
    tags = b._chat_sync("my morning briefing", force_tool="get_briefing",
                        addressee=scope_mod.OWNER)
    assert record == [("get_briefing",)]
    assert tags[-1] == ("SPEAK", "Seventy-two and sunny, sir.")
    assert "who is not Hunter" not in fake.chat_payloads()[0]["messages"][0]["content"]


def test_the_loop_reads_the_addressee_exactly_once(loop, monkeypatch):
    """Measured (a counting wrapper) and pinned (the source)."""
    b, fake, record, reg = loop
    brain_mod.set_addressee("Heather", "ma'am")
    reads = []
    real = scope_mod.addressee

    def counted(now=None):
        reads.append(now)
        return real(now)
    monkeypatch.setattr(scope_mod, "addressee", counted)
    fake.replies = [tool_reply(("notes", {"action": "list"})),
                    text_reply("spare"), text_reply("spare")]
    b._chat_sync("what did I write down?")
    assert len(reads) == 1, reads
    # The forced path too: refusal, prompt and schemas all off ONE read.
    reads.clear()
    b._chat_sync("good morning", force_tool="get_briefing")
    assert len(reads) == 1, reads

    src = (ROOT / "jarvis" / "brain.py").read_text(encoding="utf-8")
    start = src.index("    def _chat_sync(")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    readers = re.findall(r"(?:scope_mod\.)?(?:addressee|is_owner_addressee)\(\)",
                         body)
    assert len(readers) == 1, readers
    # ...and the prompt is rendered from that reading, not a second one.
    assert "static_system(addressee_to=turn_addr)" in body
    assert "self._dynamic_context(text) if owner_turn" in body


def test_a_guest_turn_carries_none_of_his_background(loop):
    """The dynamic context -- his active window, his git state, his
    recent conversation, what he told me about his life -- is not in
    her user turn. Before: it was, with every tool refused around it."""
    b, fake, record, reg = loop
    brain_mod.set_addressee("", "sir")
    fake.replies = [text_reply("spare")]
    b._chat_sync("what am I looking at?")
    his = fake.chat_payloads()[-1]["messages"][1]["content"]
    brain_mod.set_addressee("Heather", "ma'am")
    fake.replies = [text_reply("spare")]
    b._chat_sync("what am I looking at?")
    hers = fake.chat_payloads()[-1]["messages"][1]["content"]
    assert "what am I looking at?" in hers
    ctx, mem = b._dynamic_context("what am I looking at?")
    assert ctx or mem, "the fake context carries nothing; test is vacuous"
    for piece in (ctx, mem):
        if piece:
            assert piece in his
            assert piece not in hers
    # The seam keeps its one-argument shape: six tests stub it so.
    import inspect
    assert list(inspect.signature(brain_mod.JarvisBrain._dynamic_context)
                .parameters) == ["self", "text"]


# ---------------------------------------------------- the app's dispatch
def _dispatching(tmp_path):
    """The gate stand-in plus the little _dispatch needs, with a commander
    that records WHO the scope said at the moment handle() ran."""
    a = _stand_in(tmp_path, known=True)
    a.seen = []
    a.commander = SimpleNamespace(
        handle=lambda text, source, **kw: (
            a.seen.append((source, scope_mod.addressee(), a._gate_who)),
            SimpleNamespace(reply="", speak=False, status="", done=True))[1])
    # NOTE (round-3, 09-05): stubbing this out is what hid BLOCKER 3 from
    # the lane's own evidence -- _debrief_reply is the one door hoisted
    # ABOVE commander.handle, and a guest's sentence went straight into
    # his memory and his journal through it. The door itself is measured
    # in tests/test_turn_carried.py with the REAL method; here it is only
    # kept out of the way, and it takes the carried reading like the rest.
    a._debrief_reply = lambda text, source, addressee=None: None
    a._emit_result = lambda r: r
    a._after_dispatch = lambda *x, **k: None
    a._the_turn_is_his = app_mod.JarvisApp._the_turn_is_his.__get__(a)
    a._dispatch = app_mod.JarvisApp._dispatch.__get__(a)
    return a


def test_every_non_voice_source_is_his_even_a_second_after_a_guest_spoke(tmp_path):
    """Heather by face, then his keyboard, the socket, his phone and
    Discord, each within the TTL. Before: all four were Heather's."""
    a = _dispatching(tmp_path)
    brain_mod.reset_static_prompt()
    his_prompt = brain_mod.static_system()
    for source in ("typed", "cli", "intercom", "discord"):
        _watching(a, "heather")
        assert a._gate_admits("what time is it", {}) is True
        assert a._gate_who == "heather"
        assert scope_mod.addressee() == ("Heather", "ma'am")
        a._dispatch("what's on my calendar", source)
        assert a.seen[-1] == (source, ("", "sir"), ""), a.seen[-1]
        assert brain_mod.static_system() is his_prompt
    assert [s for s, _w, _g in a.seen] == ["typed", "cli", "intercom", "discord"]
    # gate.GATED_SOURCES is the allow-list this reads, so a new source is
    # his by default until somebody gates it.
    assert gate_mod.GATED_SOURCES == ("voice",)


def test_the_voice_path_still_names_the_person(tmp_path):
    """The other half of "per turn" is unchanged: an admitted voice turn
    attributes, and the next voice turn re-attributes."""
    a = _dispatching(tmp_path)
    _watching(a, "heather")
    assert a._gate_admits("what time is it", {}) is True
    assert scope_mod.addressee() == ("Heather", "ma'am")
    _watching(a, "hunter")
    assert a._gate_admits("what time is it", MATCHED) is True
    assert scope_mod.addressee() == ("", "sir")


def test_the_first_wake_briefing_is_his_whoever_spoke_last(monkeypatch, tmp_path):
    """The reviewer's exact shape: app._deliver_first_wake_briefing after
    a guest turn. The model is asked with the OWNER in force."""
    a = _briefing_app(monkeypatch, tmp_path)
    a.services = SimpleNamespace(
        brain=SimpleNamespace(
            chat=lambda t, **kw: a.chats.append((t, scope_mod.addressee(), kw))),
        briefing_offer=None)
    _clock(monkeypatch, 9)
    scope_mod.set_addressee("Mara", "ma'am")
    assert a._deliver_first_wake_briefing() is True
    text, who, kw = a.chats[-1]
    assert text == "my morning briefing"
    # Cleared AND named: the ambient attribution is the owner's, and the
    # ask carries scope.OWNER explicitly so the worker cannot read anything
    # else -- ``chat(addressee=...)`` is the API for a proactive caller.
    assert kw == {"force_tool": "get_briefing", "addressee": ("", "sir")}
    assert who == ("", "sir")
    assert scope_mod.addressee() == ("", "sir")
    assert a._gate_who == ""


def test_the_turn_is_his_clears_both_halves(tmp_path):
    a = _stand_in(tmp_path, known=True)
    a._the_turn_is_his = app_mod.JarvisApp._the_turn_is_his.__get__(a)
    _watching(a, "heather")
    assert a._gate_admits("what time is it", {}) is True
    assert a._honorific() == "ma'am"
    a._the_turn_is_his()
    assert (a._gate_who, a._gate_how) == ("", "")
    assert a._honorific() == "sir"
    assert scope_mod.addressee() == ("", "sir")


def test_dispatch_clears_before_the_commander_and_only_for_ungated_sources():
    """Source guard for the ordering: the clear sits above handle()."""
    src = (ROOT / "jarvis" / "app.py").read_text(encoding="utf-8")
    start = src.index("    def _dispatch(")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    clear = body.index("self._the_turn_is_his()")
    handle = body.index("self.commander.handle(")
    assert clear < handle
    assert "elif source not in gate_mod.GATED_SOURCES:" in body
