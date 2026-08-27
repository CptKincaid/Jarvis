"""FOUND 2026-08-26 (Claude-session sweep): the spoken "N tests failed"
verdict is unreachable for a real failing test run.

jarvis/claude_session.py:_parse_tool_result() (~line 559) returns early
whenever the tool_result block carries `is_error`:

    if block.get("is_error"):
        task.error_streak += 1
        out.append((f"Error: {_excerpt(...)}", False, False))   # NOT a milestone
        ...
        return                       # <- _test_outcome() never runs

The Claude CLI sets `is_error: true` on every non-zero-exit Bash result.
Measured in a live stream-json capture from this sweep
(/tmp/.../scratchpad/logdir/claude/testproj/20260826-221539-d15e0d.jsonl):

    tool_result is_error=True : Exit code 127 /bin/bash: python: not found
    tool_result is_error=True : Exit code 1 /usr/bin/python3: No module named pytest

and `pytest` exits 1 exactly when tests fail.  So the failure half of
_test_outcome() ("N tests failed, sir." / "The tests errored, sir.") can
only fire when a test command exits 0, which is the case where it would
say "Tests passed" instead.  The app only speaks milestone=True lines
(jarvis/app.py:_on_claude_progress), so a failing test run is silent: the
user hears "Running the tests, sir." and then nothing.

Fix belongs in _parse_tool_result: when the tool_use id is in
task.test_ids, run _test_outcome() on the error text before returning.

FIXED 2026-08-26 (claude_session._parse_tool_result): an is_error
tool_result whose id is in task.test_ids and whose text yields a verdict
now speaks that verdict and does NOT count towards error_streak (a failing
suite is work, not a tool fault).  An is_error with no recognisable verdict
("Exit code 127 ... python: not found") still takes the old error path.
"""
from jarvis.claude_session import Task, parse_stream_event

RUN_TESTS = {"type": "assistant", "message": {"content": [
    {"type": "tool_use", "id": "T1", "name": "Bash",
     "input": {"command": "python3 -m pytest -q"}}]}}
FAILED_RESULT = {"type": "user", "message": {"content": [
    {"type": "tool_result", "tool_use_id": "T1", "is_error": True,
     "content": "Exit code 1\n2 failed, 3 passed in 0.42s"}]}}


def test_failing_pytest_run_speaks_a_verdict():
    task = Task(task_id="t", project="p", prompt="p", model="m")
    spoken = []
    for event in (RUN_TESTS, FAILED_RESULT):
        for prog in parse_stream_event(event, task, now=1000.0):
            if prog.milestone:
                spoken.append(prog.line)
    assert "Running the tests, sir." in spoken
    assert any("failed" in line for line in spoken), spoken


def test_the_same_result_without_is_error_does_speak_it():
    """Control: the verdict logic itself is fine — only the is_error path
    loses it."""
    ok_result = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "T1", "is_error": False,
         "content": "Exit code 1\n2 failed, 3 passed in 0.42s"}]}}
    task = Task(task_id="t", project="p", prompt="p", model="m")
    spoken = []
    for event in (RUN_TESTS, ok_result):
        for prog in parse_stream_event(event, task, now=1000.0):
            if prog.milestone:
                spoken.append(prog.line)
    assert "2 tests failed, sir." in spoken


def test_a_non_test_error_still_takes_the_error_path():
    """Control: an is_error with no verdict in it is still just an error —
    it counts towards error_streak and speaks nothing on its own."""
    task = Task(task_id="t", project="p", prompt="p", model="m")
    missing = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "T1", "is_error": True,
         "content": "Exit code 127\n/bin/bash: python: not found"}]}}
    spoken = []
    for event in (RUN_TESTS, missing):
        for prog in parse_stream_event(event, task, now=1000.0):
            if prog.milestone:
                spoken.append(prog.line)
    assert spoken == ["Running the tests, sir."]
    assert task.error_streak == 1


def test_a_failed_suite_does_not_count_as_a_tool_error():
    """Three failing test runs must not trip 'Hitting errors, sir'."""
    task = Task(task_id="t", project="p", prompt="p", model="m")
    spoken = []
    for _ in range(3):
        for event in (RUN_TESTS, FAILED_RESULT):
            for prog in parse_stream_event(event, task, now=1000.0):
                if prog.milestone:
                    spoken.append(prog.line)
    assert task.error_streak == 0
    assert spoken.count("2 tests failed, sir.") == 3
    assert "Hitting errors, sir; carrying on." not in spoken
