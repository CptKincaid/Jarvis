"""Regression tests for defects found by the adversarial-verification sweep
(2026-08-26).  Both are FIXED 2026-08-26 by the brain work item (M9 and
the LOW busy-guard finding); the text below is the defect they pin.

DEFECT 1 -- jarvis/brain.py ~line 1039 (`for call in calls:` inside the tool
loop).  The number of tool_calls the model may ask for in ONE round is
unbounded, and the CHAT_WALL_BUDGET_S deadline check sits OUTSIDE that inner
loop, so a single round executes every call the model emitted before the
budget is ever consulted.  Measured: one round carrying 5000 tool_calls ran
5000 real handler invocations in 12.3 s -- 1.5x the documented 8 s budget
from a single round.  With a real tool (an HTTP fetch, a SQLite write) that
is 5000 outbound requests from one confused model turn.

DEFECT 2 -- jarvis/brain.py ~line 902 (`_acquire_busy`).  The busy guard is a
check-then-set on a plain bool with no lock, so two threads can both observe
`self._busy == False` and both proceed.  Latent: at CPython's default
sys.setswitchinterval(0.005) the race did not fire in 16,000 trials, but at a
1e-6 switch interval up to 12 of 64 threads acquired it at once.

FIXES: the tool loop now runs at most MAX_TOOL_CALLS_PER_ROUND calls from
one round and checks the CHAT_WALL_BUDGET_S deadline INSIDE the inner
loop; _acquire_busy does its check-then-set under self._busy_lock.
"""
import sys
import threading

import jarvis.brain as B
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec


def test_one_round_of_tool_calls_is_capped(monkeypatch):
    calls = {"n": 0}
    reg = ToolRegistry()
    reg.register(ToolSpec(name="fetch", description="fetch a thing",
                          handler=lambda **kw: (calls.__setitem__("n", calls["n"] + 1)
                                                or ToolResult(text="x"))))
    monkeypatch.setattr(B, "_REGISTRY", reg, raising=False)
    n = 2000
    monkeypatch.setattr(B, "_http", lambda p, payload=None, timeout=None: {
        "message": {"content": "",
                    "tool_calls": [{"function": {"name": "fetch", "arguments": {}}}
                                   for _ in range(n)]}})
    B.JarvisBrain(None, None)._chat_sync("hi", max_rounds=3)
    # A sane bound is the per-round cap, far below "whatever the model
    # emitted"; three rounds of it is the most one turn can run.
    assert calls["n"] <= 64, f"{calls['n']} tools ran in a single round"
    assert calls["n"] <= 3 * B.MAX_TOOL_CALLS_PER_ROUND, calls["n"]


def test_busy_guard_is_atomic():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)          # a loaded machine, exaggerated
    try:
        worst = 0
        for _ in range(40):
            brain = B.JarvisBrain(None, None)
            n = 64
            barrier = threading.Barrier(n)
            won, lock = [], threading.Lock()

            def go():
                barrier.wait()
                if brain._acquire_busy():
                    with lock:
                        won.append(1)

            threads = [threading.Thread(target=go) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            worst = max(worst, len(won))
        assert worst == 1, f"{worst} threads held the busy guard at once"
    finally:
        sys.setswitchinterval(old)
