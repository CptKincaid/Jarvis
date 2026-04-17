"""Tests for CommandDispatcher."""

from unittest.mock import MagicMock

from jarvis.dispatcher import CommandDispatcher, CommandHandler


def _make_handler(name, called_list):
    def _fn(match, ctx):
        called_list.append((name, match.group(0)))
        return True
    return _fn


def test_matching_command_routes_to_handler():
    called = []
    handlers = [
        CommandHandler(r"^list files$", _make_handler("list", called)),
        CommandHandler(r"^show time$", _make_handler("time", called)),
    ]
    brain = MagicMock()
    d = CommandDispatcher(handlers=handlers, brain=brain)
    d.handle("list files", ctx={})
    assert called == [("list", "list files")]
    brain.handle.assert_not_called()


def test_first_match_wins_when_multiple_patterns_match():
    called = []
    handlers = [
        CommandHandler(r"^list.*", _make_handler("broad", called)),
        CommandHandler(r"^list files$", _make_handler("specific", called)),
    ]
    brain = MagicMock()
    d = CommandDispatcher(handlers=handlers, brain=brain)
    d.handle("list files", ctx={})
    assert called == [("broad", "list files")]


def test_unmatched_text_falls_through_to_brain():
    brain = MagicMock()
    d = CommandDispatcher(handlers=[], brain=brain)
    d.handle("hello there", ctx={})
    brain.handle.assert_called_once_with("hello there", {})


def test_handler_returning_false_falls_through():
    called = []
    def _refused(match, ctx):
        called.append("refused")
        return False
    handlers = [CommandHandler(r"^list.*", _refused)]
    brain = MagicMock()
    d = CommandDispatcher(handlers=handlers, brain=brain)
    d.handle("list files", ctx={})
    assert called == ["refused"]
    brain.handle.assert_called_once()


def test_empty_text_falls_through_to_brain():
    brain = MagicMock()
    d = CommandDispatcher(handlers=[CommandHandler(r"^list", lambda m, c: True)],
                          brain=brain)
    d.handle("", ctx={})
    brain.handle.assert_called_once_with("", {})


def test_handler_receives_match_object_with_groups():
    captured = []
    def _h(match, ctx):
        captured.append(match.group(1))
        return True
    d = CommandDispatcher(
        handlers=[CommandHandler(r"^run (\w+)$", _h)],
        brain=MagicMock(),
    )
    d.handle("run tests", ctx={})
    assert captured == ["tests"]


def test_handler_receives_ctx_and_can_read_it():
    captured = []
    def _h(match, ctx):
        captured.append(ctx.get("user"))
        return True
    d = CommandDispatcher(
        handlers=[CommandHandler(r"^hello$", _h)],
        brain=MagicMock(),
    )
    d.handle("hello", ctx={"user": "Hunter"})
    assert captured == ["Hunter"]


def test_case_insensitive_by_default():
    called = []
    d = CommandDispatcher(
        handlers=[CommandHandler(r"^LIST$", lambda m, c: called.append("x") or True)],
        brain=MagicMock(),
    )
    d.handle("list", ctx={})
    assert called == ["x"]


def test_handler_exception_falls_through_to_brain():
    def _broken(match, ctx):
        raise RuntimeError("boom")
    brain = MagicMock()
    d = CommandDispatcher(
        handlers=[CommandHandler(r"^list$", _broken)],
        brain=brain,
    )
    d.handle("list", ctx={})
    # Handler raised, so dispatcher continues to brain fallback
    brain.handle.assert_called_once()
