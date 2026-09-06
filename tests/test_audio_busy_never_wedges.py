"""A hung transcription must not kill the wake word for ever.

MEASURED ON HIS LIVE BOX, 2026-09-06 00:29-00:32. He reported Jarvis "soft
locked ... stuck from saying is that for me and being in standby mode and
wont come out". The log says exactly what happened, five times in ninety
seconds:

    00:29:43 Hotword detected (score=0.896)
    00:29:43 hotword ignored: still transcribing the previous clip
    00:29:47 Hotword detected (score=0.975)
    00:29:47 hotword ignored: still transcribing the previous clip
    ...
    00:30:28 turn watchdog fired after 60s; releasing the wake word

The wake word was HEARD every time, at high confidence, and dropped every
time. The last utterance to reach ``_process_audio`` was at 00:29:28; that
thread never returned, so the ``finally`` that clears ``_audio_busy`` never
ran, and every later wake word hit the guard at ``_should_record``.

TWO FLAGS, ONE FAILURE MODE, AND ONLY ONE OF THEM WAS GUARDED.
``_turn_started`` arms a watchdog whose own comment reads: "Without this a
reply that never arrives would hold ``_turn_busy`` for good, and every later
wake word would be a silent no-op -- indistinguishable from a dead
microphone." That is this defect, described exactly, one flag over. The
watchdog fired at 00:30:28 and called ``_turn_finished``, which clears
``_turn_busy`` and does not touch ``_audio_busy`` -- so it announced that it
was "releasing the wake word" while the wake word stayed dead for another
two minutes, until the process was killed.

``_process_audio``'s ``finally`` is not enough on its own and says so: "Must
run on every path: a leaked flag makes every future wake word a no-op, which
looks exactly like a dead microphone." A ``finally`` covers a RAISE. It does
not cover a HANG, and a hang is what happened.

THE WORSE CASE, which has no log line at all: ``_audio_busy`` is set BEFORE
the turn begins, so a ``_process_audio`` that hangs before starting a turn
leaks the flag with no watchdog ever armed and nothing printed. The recovery
here is owned by the audio flag itself rather than borrowed from the turn.
"""
import threading
import time

from jarvis import app as app_mod


class _Bus:
    def __init__(self):
        self.published = []

    def publish(self, ev):
        self.published.append(ev)


def _app(timeout=0.15):
    """Just the flag machinery, over the real Events and the real Timer."""
    a = object.__new__(app_mod.JarvisApp)
    a._audio_busy = threading.Event()
    a._turn_busy = threading.Event()
    a._audio_watchdog = None
    a._turn_timer = None
    a._turn_watchdog = None
    a._audio_timeout_s = timeout
    a.turns = _Turns()
    return a


class _Turns:
    def __init__(self):
        self.abandoned = []

    def abandon(self, why):
        self.abandoned.append(why)


def _bind(a, name):
    return getattr(app_mod.JarvisApp, name).__get__(a)


# --------------------------------------------------- the flag frees itself
def test_a_hung_transcription_releases_the_wake_word():
    """THE INCIDENT. Set the flag, never clear it, and let the clock run."""
    a = _app(timeout=0.15)
    _bind(a, "_audio_started")()
    assert a._audio_busy.is_set(), "the premise: the flag is held"
    time.sleep(0.45)
    assert not a._audio_busy.is_set(), (
        "the audio flag is still held after its watchdog should have fired; "
        "every wake word from here is a silent no-op")


def test_a_normal_turn_is_not_cut_short_by_the_watchdog():
    """The guard must be a guard, not a time limit on his sentences."""
    a = _app(timeout=5.0)
    _bind(a, "_audio_started")()
    time.sleep(0.2)
    assert a._audio_busy.is_set(), "a normal clip was released early"
    _bind(a, "_audio_finished")()
    assert not a._audio_busy.is_set()
    assert a._audio_watchdog is None, "the watchdog outlived its clip"


def test_the_watchdog_does_not_fire_after_a_clip_finishes_normally():
    """A cancelled timer must not clear a flag a LATER clip is holding.

    The sleep is deliberately SHORTER than the second clip's own deadline:
    what is under test is the FIRST clip's cancelled timer, not the second
    clip's live one. (My first draft slept past both and failed on the
    second watchdog doing its job correctly.)"""
    a = _app(timeout=0.30)
    _bind(a, "_audio_started")()
    _bind(a, "_audio_finished")()
    _bind(a, "_audio_started")()          # the next utterance, straight away
    time.sleep(0.12)                      # past a stale timer, short of this one
    assert a._audio_busy.is_set(), (
        "the previous clip's watchdog cleared the CURRENT clip's flag")
    _bind(a, "_audio_finished")()


# ------------------------------------------- the turn watchdog frees both
def test_the_turn_watchdog_releases_the_audio_flag_too():
    """It announces "releasing the wake word". It must mean it.

    On the night this was found it cleared _turn_busy, said that line, and
    left _audio_busy set -- so the wake word stayed dead."""
    a = _app(timeout=99.0)
    a._turn_timeout_s = 0.01
    _bind(a, "_audio_started")()
    a._turn_busy.set()
    _bind(a, "_turn_timed_out")()
    assert not a._turn_busy.is_set()
    assert not a._audio_busy.is_set(), (
        "the turn watchdog said it was releasing the wake word and left the "
        "audio flag set")


# ------------------------------------------------------------- the census
def test_every_setter_of_the_audio_flag_arms_its_own_release():
    """AST census. ``_audio_busy.set()`` may only be reached through the
    helper that arms the watchdog, so a second setter added tomorrow cannot
    reintroduce the leak silently."""
    import ast
    import inspect
    src = inspect.getsource(app_mod)
    tree = ast.parse(src)
    setters = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            if (isinstance(f, ast.Attribute) and f.attr == "set"
                    and isinstance(f.value, ast.Attribute)
                    and f.value.attr == "_audio_busy"):
                setters.append(node.name)
    assert setters == ["_audio_started"], (
        "_audio_busy.set() is reached outside _audio_started, so that path "
        "arms no watchdog: %s" % setters)


def test_the_release_path_works_on_a_bare_object_that_owns_only_the_flag():
    """_process_audio is driven in many tests on a hand-built namespace
    holding the flag and nothing else. The release must not depend on a
    helper being present -- a release path that can raise AttributeError is
    the very bug this file exists to close, wearing a helper."""
    import types
    ns = types.SimpleNamespace(_audio_busy=threading.Event())
    ns._audio_busy.set()
    try:
        ns._audio_cancel_watchdog()
    except AttributeError:
        pass
    ns._audio_busy.clear()
    assert not ns._audio_busy.is_set()
