"""He asked for one thing above all: "he won't claim he did something".

On 2026-09-03 Jarvis said "I've put the command on your clipboard" and the
clipboard did not have it. The claim was built on ``xclip`` exiting 0, and
exit 0 only proves xclip STARTED. An X11 CLIPBOARD selection has NO STORAGE:
a live process owns the selection and serves the bytes at paste time, so any
other process can take it in between -- which is exactly what he caught,
finding a sentinel written by an unrelated process where his command should
have been.

A false success is worse than a plain failure. So three things are pinned
here, and they are three different tests because they close three different
holes:

1. THE WRITE IS VERIFIED. ``to_clipboard`` reads the clipboard back and
   compares before it returns True. That catches "it never landed" and
   nothing else -- these tests say so out loud, because a suite that implied
   the read-back was a guarantee would be making the same over-promise in a
   different file.
2. THE CLIPBOARD IS NO LONGER THE DELIVERY. The command goes into
   ``CommandResult.display_only``, which the console SHOWS and the TTS never
   READS, so he has it in the transcript whichever way the clipboard went.
   Speaking a shell command with a file path in it is a bad minute of
   text-to-speech -- that constraint stands, which is why a second field had
   to exist rather than the command being appended to ``reply``.
3. THE SENTENCE MATCHES REALITY in both branches.

NOTHING HERE TOUCHES HIS DESKTOP. Every xclip call is a fake ``run``; the
whole point of the injected seam is that the real clipboard is never the
thing under test.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import enrolentry as ee
from jarvis.commander import CommandResult
from jarvis.facegallery import FaceGallery

CMD = "/usr/bin/python /home/x/Jarvis/scripts/face_enrol.py"


# ------------------------------------------------------------- the fake xclip
class FakeClip:
    """An X clipboard with an owner, which is the part that matters.

    ``holds`` is what a reader would get back. ``steal`` lets a test act as
    the other process that took the selection between the write and the
    read -- the mechanism he actually caught."""

    def __init__(self, holds=None, accept=True, read_rc=0, steal=None):
        self.holds = holds
        self.accept = accept
        self.read_rc = read_rc
        self.steal = list(steal or [])
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[-1] == "-o":                       # a read
            if self.steal:
                self.holds = self.steal.pop(0)
            return SimpleNamespace(returncode=self.read_rc,
                                   stdout=(self.holds or "").encode("utf-8"))
        if self.accept:                            # a write
            self.holds = kw["input"].decode("utf-8")
            return SimpleNamespace(returncode=0, stdout=b"")
        return SimpleNamespace(returncode=1, stdout=b"")

    @property
    def reads(self):
        return [c for c in self.calls if c[-1] == "-o"]


# ------------------------------------------------------- 1. verify the write
def test_a_clipboard_that_took_the_text_is_the_only_true():
    clip = FakeClip()
    assert ee.to_clipboard(CMD, run=clip) is True
    assert clip.holds == CMD
    assert clip.reads, "nothing read the clipboard back"


def test_exit_zero_is_not_evidence_and_no_longer_returns_true():
    """THE BUG, reproduced. xclip exits 0, the clipboard holds something
    else, and the old code said True. This is the CANARY-RC-TEST-12345 shape
    he found: a sentinel from a different process where his command should
    have been."""
    def run(argv, **kw):
        if argv[-1] == "-o":
            return SimpleNamespace(returncode=0,
                                   stdout=b"CANARY-RC-TEST-12345")
        return SimpleNamespace(returncode=0, stdout=b"")   # xclip "started"

    assert ee.to_clipboard(CMD, run=run) is False


def test_a_clipboard_that_cannot_be_read_back_is_not_a_success():
    """"I could not check" is not "it worked". Unreadable is False."""
    assert ee.to_clipboard(CMD, run=FakeClip(read_rc=1)) is False


def test_an_empty_clipboard_after_the_write_is_a_failure():
    def run(argv, **kw):
        return SimpleNamespace(returncode=0, stdout=b"")

    assert ee.to_clipboard(CMD, run=run) is False


def test_a_write_that_failed_outright_never_bothers_to_read_back():
    clip = FakeClip(accept=False)
    assert ee.to_clipboard(CMD, run=clip) is False
    assert clip.reads == [], "read the clipboard after a failed write"


def test_the_ownership_race_gets_exactly_one_retry():
    """xclip backgrounds a child to own the selection and the parent can
    exit a hair before that child has taken it. One retry turns that false
    negative into the truth; it must not become a loop."""
    clip = FakeClip(steal=["", CMD])         # first read early, second good
    assert ee.to_clipboard(CMD, run=clip) is True
    assert len(clip.reads) == 2


def test_a_persistent_mismatch_stops_after_the_retry():
    clip = FakeClip(steal=["other", "other", "other"])
    assert ee.to_clipboard(CMD, run=clip) is False
    assert len(clip.reads) == 2, "the retry must not become a loop"


def test_a_trailing_newline_is_still_his_command():
    clip = FakeClip(steal=[CMD + "\n"])
    assert ee.to_clipboard(CMD, run=clip) is True


def test_a_text_read_back_as_str_is_handled_like_bytes():
    """reader.py runs xclip with text=True; a str stdout must not be read as
    a mismatch."""
    def run(argv, **kw):
        if argv[-1] == "-o":
            return SimpleNamespace(returncode=0, stdout=CMD)
        return SimpleNamespace(returncode=0, stdout="")

    assert ee.to_clipboard(CMD, run=run) is True


def test_the_readback_is_asked_for_on_the_same_injected_seam():
    """The whole reason the seam exists: a test must never reach his real
    clipboard, and the read must not sneak past it into subprocess."""
    clip = FakeClip()
    ee.to_clipboard(CMD, run=clip)
    assert all(c[0] == "xclip" for c in clip.calls)
    assert clip.reads == [["xclip", "-selection", "clipboard", "-o"]]


def test_the_docstring_refuses_to_call_the_check_a_guarantee():
    """The comment is load-bearing here. The read-back closes ONE hole and
    the file has to say which, or the next reader rebuilds the same
    over-promise on top of it."""
    doc = ee.to_clipboard.__doc__
    assert "never" in doc and "paste" in doc


# ------------------------------- 2. delivery no longer rides the clipboard
def test_the_command_reaches_the_transcript_even_when_the_clipboard_took_it(
        tmp_path):
    out = ee.enrol_answer(FaceGallery(root=tmp_path / "g"), "hunter",
                          clipboard=lambda text: True)
    assert out["display_only"] == out["command"]


def test_the_command_reaches_the_transcript_when_the_clipboard_failed(
        tmp_path):
    """THE POINT OF THE WHOLE FIX. The clipboard is allowed to fail
    completely and he is still not stranded."""
    out = ee.enrol_answer(FaceGallery(root=tmp_path / "g"), "hunter",
                          clipboard=lambda text: False)
    assert out["clipped"] is False
    assert "face_enrol.py" in out["display_only"]


def test_the_delete_hand_over_carries_the_command_too(tmp_path):
    from tests.test_faceenrol import base_vec, same_face

    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(808), 6, seed=17):
        g.add("heather", vec)
    g.save(reason="test")
    out = ee.forget_answer(FaceGallery(root=g.root), "heather",
                           clipboard=lambda text: False)
    assert "--delete --label heather" in out["display_only"]


@pytest.mark.parametrize("clipped", [True, False])
def test_the_command_is_shown_and_never_spoken(tmp_path, clipped):
    """The constraint that stands: speaking a file path is a bad minute of
    text-to-speech. The command may appear in display_only and nowhere
    else."""
    out = ee.enrol_answer(FaceGallery(root=tmp_path / "g"), "hunter",
                          clipboard=lambda text: clipped)
    assert "face_enrol.py" not in out["reply"]
    assert "face_enrol.py" in out["display_only"]


def test_a_refused_name_hands_over_nothing_to_show(tmp_path):
    out = ee.enrol_answer(FaceGallery(root=tmp_path / "g"), "not a name!")
    assert out["display_only"] == "" and out["command"] == ""


# ------------------------------------------ 3. the sentence matches reality
@pytest.mark.parametrize("clipped", [True, False])
def test_neither_branch_promises_the_clipboard_will_still_hold_it(clipped):
    line = ee.CLIP_OK_LINE if clipped else ee.CLIP_FAILED_LINE
    low = line.lower()
    assert "i've put the command on your clipboard" not in low
    assert "transcript" in low, "the guarantee has to lead"


def test_the_verified_branch_still_admits_the_clipboard_can_be_taken():
    """True means "it was there when I looked", and the sentence he hears has
    to carry that, because the clipboard is shared global state with a single
    owner."""
    low = ee.CLIP_OK_LINE.lower()
    assert "copied it" in low
    assert "take it from me" in low


def test_the_failed_branch_reads_aloud_without_the_command_in_it():
    assert "face_enrol" not in ee.CLIP_FAILED_LINE
    assert ee.CLIP_FAILED_LINE.endswith(".")


@pytest.mark.parametrize("clipped", [True, False])
def test_both_branches_are_one_flowing_sentence_after_the_spoken_part(
        tmp_path, clipped):
    """Both branches of the hand-over have to read aloud, so the joined text
    must not run two sentences together without a space."""
    out = ee.enrol_answer(FaceGallery(root=tmp_path / "g"), "hunter",
                          clipboard=lambda text: clipped)
    assert ". The command is in the transcript" in out["reply"]


# --------------------------------- the field itself, on every handler's path
def test_display_only_defaults_to_none_so_no_handler_changes():
    """This field sits on the path EVERY handler returns through. A default
    of anything but None would change every reply in the app."""
    assert CommandResult(handled=True).display_only is None
    assert CommandResult(handled=True, reply="x",
                         speak=True).display_only is None


def test_the_face_handlers_pass_the_command_through_to_the_display(
        monkeypatch, tmp_path):
    from jarvis import commander as cm
    from tests.test_faceenrol_notes import FakeCommander, _face_cmd

    monkeypatch.setattr(cm, "_face_gallery",
                        lambda _c: FaceGallery(root=tmp_path / "g"))
    monkeypatch.setattr(ee, "to_clipboard", lambda text, run=None: False)
    cmd = _face_cmd("face enrol")
    out = cmd.handler(FakeCommander(), "enrol my face",
                      cmd.matcher("enrol my face"))
    assert "face_enrol.py" in out.display_only
    assert "face_enrol.py" not in out.reply


# ----------------------------------------------- the routing in _emit_result
class Spy:
    """A JarvisApp stand-in for _emit_result alone: what was SHOWN, and what
    was SPOKEN. No Tk, no TTS, no bus subscription of its own."""

    def __init__(self):
        self.said = []
        self.shown = []
        self.status = []

    def _say(self, text):
        self.said.append(text)

    def _disarm_filler(self):
        pass


def _emit(result):
    from jarvis import app as app_mod
    from jarvis.events import JarvisReply, Status, bus

    spy = Spy()
    on_reply = bus.subscribe(JarvisReply, lambda e: spy.shown.append(e))
    on_status = bus.subscribe(Status, lambda e: spy.status.append(e))
    try:
        app_mod.JarvisApp._emit_result(spy, result)
    finally:
        bus.unsubscribe(JarvisReply, on_reply)
        bus.unsubscribe(Status, on_status)
    return spy


def test_the_shown_text_carries_the_extra_line_and_the_spoken_text_does_not():
    spy = _emit(CommandResult(handled=True, reply="Here you are, sir.",
                              speak=True, display_only=CMD))
    assert spy.shown[0].text == "Here you are, sir.\n" + CMD
    assert spy.said == ["Here you are, sir."], "the command was spoken"


def test_without_the_field_the_routing_is_what_it_was():
    spy = _emit(CommandResult(handled=True, reply="Twenty past four, sir.",
                              speak=True, status="Clock"))
    assert spy.shown[0].text == "Twenty past four, sir."
    assert spy.shown[0].speak is True
    assert spy.said == ["Twenty past four, sir."]
    assert spy.status[0].text == "Clock"


def test_a_silent_result_still_shows_and_still_says_nothing():
    spy = _emit(CommandResult(handled=True, reply="Copied.", speak=False,
                              display_only=CMD))
    assert spy.shown[0].text == "Copied.\n" + CMD
    assert spy.said == []


def test_display_only_alone_shows_and_speaks_nothing():
    spy = _emit(CommandResult(handled=True, display_only=CMD, speak=True))
    assert spy.shown[0].text == CMD
    assert spy.said == [], "there is no reply to speak"


def test_an_empty_result_publishes_nothing_at_all():
    spy = _emit(CommandResult(handled=True))
    assert spy.shown == [] and spy.said == []


# -------------------------------------- the other place that claimed a write
class _Proc:
    """A finished xclip write. The old code never looked at returncode at
    all, which is how "Pasted: ..." was said over a write that failed."""

    returncode = 0

    def communicate(self, input=None, timeout=None):
        return b"", b""


def _agent_with_one_item():
    from jarvis.jarvis_agent import JarvisAgent

    agent = object.__new__(JarvisAgent)
    agent._init_clipboard()
    agent._clipboard_history.append({"time": "t", "text": "the real thing"})
    return agent


def test_the_paste_no_longer_claims_a_paste_the_clipboard_refused():
    """jarvis_agent.paste_from_history returned the text as soon as the two
    subprocesses had been STARTED -- no return code checked at all. Same
    false success, different file."""
    keyed = []

    def run(argv, **kw):
        if argv[0] == "xdotool":
            keyed.append(argv)
            return SimpleNamespace(returncode=0, stdout=b"")
        return SimpleNamespace(returncode=0, stdout=b"CANARY-RC-TEST-12345")

    assert _agent_with_one_item().paste_from_history(
        0, popen=lambda *a, **k: _Proc(), run=run) is None
    assert keyed == [], "it pressed ctrl+v over a clipboard it had not set"


def test_the_paste_still_works_when_the_clipboard_really_took_it():
    def run(argv, **kw):
        if argv[0] == "xdotool":
            return SimpleNamespace(returncode=0, stdout=b"")
        return SimpleNamespace(returncode=0, stdout=b"the real thing")

    assert _agent_with_one_item().paste_from_history(
        0, popen=lambda *a, **k: _Proc(), run=run) == "the real thing"


def _paste_result(items):
    from jarvis import commander as cm

    class Ctx:
        def paste_from_history(self, idx):
            return None

        def get_clipboard_history(self, n=5):
            return items

    class C:
        def _svc(self, name):
            return Ctx()

    cmd = next(c for c in cm.REGISTRY if c.name == "paste item")
    text = "paste item 1"
    return cmd.handler(C(), text, cmd.matcher(text))


def test_a_failed_paste_is_not_reported_as_nothing_to_paste():
    """"Nothing to paste" over a clipboard that refused the write is the
    same false report in a friendlier voice."""
    out = _paste_result([{"time": "t", "text": "something"}])
    assert out.status == "Paste failed"
    assert "couldn't" in out.reply


def test_an_index_with_nothing_behind_it_is_still_nothing_to_paste():
    out = _paste_result([])
    assert out.status == "Nothing to paste" and not out.reply


# ------------------------------------------------------ nobody else claims it
def test_no_clipboard_write_in_the_tree_returns_true_on_a_return_code_alone():
    """The sweep, mechanical: any file that WRITES the X clipboard must also
    read it back. If a new one appears it has to justify itself here rather
    than shipping another "I've put it on your clipboard"."""
    root = Path(__file__).resolve().parent.parent / "jarvis"
    writers, readers = set(), set()
    for path in sorted(root.rglob("*.py")):
        body = "\n".join(ln for ln in path.read_text().splitlines()
                         if not ln.lstrip().startswith("#"))
        if '"xclip", "-selection", "clipboard"]' in body or \
                "'xclip', '-selection', 'clipboard']" in body:
            writers.add(path.name)
        if '"-o"' in body and "xclip" in body:
            readers.add(path.name)
    assert writers, "the sweep matched nothing -- it has stopped checking"
    assert writers <= readers, sorted(writers - readers)
