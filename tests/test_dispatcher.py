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
