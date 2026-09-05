# The clock guard

**Prove the suite does not care what time it is:**

```bash
scripts/clock_guard.sh
```

Exit 0 means every configuration agreed. Exit 1 names the configuration and
the test whose verdict moved. Takes about five minutes (five full runs).

## Why

On 2026-09-05 the same commit was **12217 passed / 0 failed at 03:50** and
**4 failed / 197 passed at 09:48**. Nothing had changed but the wall clock.

Four tests in `test_claim_guard_memory.py` hard-coded `"Good evening"` in
their expected strings, while `jarvis/brain.py:ground_greeting` correctly
rewrites a stale greeting to match the hour. The product was right — Jarvis
should not say "good evening" at breakfast — and the tests were asserting
against a clock they did not control. Measured: they passed 17:00–04:59 and
failed 05:00–16:59, which is the evening band in `jarvis/arc.py`.

This had happened before. A note from 2026-09-02 records "1 pre-existing
time-of-day failure in `test_study_briefing`" and moves on. That write-off
is the real problem: **a suite that is only green in the evening cannot be
used as a merge gate**, and every lane that hits it has to stop and
re-derive that the red is "expected".

## How it works

It does **not** freeze the clock. It shifts the pytest process's local
`TZ`, so `datetime.now()` reads the hour we asked for while the clock keeps
ticking at the normal rate and every absolute instant is unchanged.

That choice matters:

* a frozen clock hangs every test that waits on a deadline — twelve files
  in this suite do, and they hang to the timeout under `freezegun`;
* a frozen clock breaks anything that subclasses `date`/`datetime`
  (pydantic), which cost 75 spurious failures in the original sweep;
* shifting the zone needs no library, no `sudo` and no `LD_PRELOAD`.

The system clock and the machine's timezone are **never** touched. `TZ` is
set for the child process only.

## The two halves

**Runtime** — `tests/conftest.py` adds two options:

```bash
pytest --clock-at=09:00        # as if the local clock said 09:00
pytest --clock-at=23:59:30     # ... including a run that crosses midnight
pytest --clock-tz=UTC          # as if the machine were in another zone
```

Both print the simulated time in the pytest header, so a red CI run is never
mistaken for a real failure. Scope a sweep while iterating with
`PYTEST_ARGS=tests/test_foo.py scripts/clock_guard.sh`, and use
`scripts/clock_guard.sh --hours` for all 24.

**Static** — `tests/test_clock_hygiene.py` runs inside the normal suite
(~0.2 s) and parses `tests/` for the two shapes that produce a
clock-dependent test:

1. **Reading the wall clock at import time.** `TODAY = date.today()` at
   module level, asserted against later, while the product reads its own
   clock when the assertion runs. A run that crosses midnight — or 31
   August, or 31 December — goes red.
2. **Snapshotting the UTC offset, then doing wall-clock arithmetic.**
   `(datetime.now().astimezone() + timedelta(days=3)).replace(hour=13)`
   carries *today's* offset onto a day three days away. Across a DST change
   it is an hour wrong. Do the arithmetic **naive** and call `.astimezone()`
   **last**.

A line that genuinely needs one of these says why, on the line:

```python
NOW = datetime.now().astimezone()   # clock-hygiene: <reason>
```

The waiver is deliberately local: it sits where the next person to edit
that line will read it, rather than in a list nobody opens.

## What the guard does not cover

Stated plainly, because the point of this file is that numbers mean
something:

* It moves the **hour** and the **zone**. It cannot move the **date**, so
  it does not reach a named calendar date or a DST transition day. The
  five DST bugs found on 2026-09-05 were reproduced by running in
  `Pacific/Easter` and `America/Santiago`, which happened to change over
  that weekend; that trick is not available year-round. The static half is
  what defends this axis day to day.
* Anything keyed to `time.monotonic()` is untouched by a zone shift, by
  design — that is what keeps the deadline tests working.
* A test needing both a foreign zone *and* a particular hour is reachable
  (`--clock-at` and `--clock-tz` are exclusive, but a zone pins an hour too)
  yet is not in the default five.

## If the guard goes red

Fix the **test**, not the product. Jarvis is right to reground a stale
greeting against the hour, to render a timestamp in the machine's local
zone, and to read today's date. Give the test the clock through a seam the
product already has —

| seam | module |
| --- | --- |
| `ground_greeting(text, user_text, now=)` | `jarvis/brain.py` |
| `DayReviewer(..., now=)` | `jarvis/dayreview.py` |
| `now_local(tz)` ("one module-level seam. Tests freeze this") | `jarvis/tools/calendar.py` |
| `SensingPolicy(..., now=)` | `jarvis/sensing.py` |
| `Arc(..., now=)` | `jarvis/arc.py` |

— or narrow the assertion to the thing the test is actually about. A test
about the claim guard should not care what the greeting says.

If a file's fixtures are written in a particular zone (several are: the
house is in Central time), say so rather than assume it:

```python
pytestmark = pytest.mark.local_tz("America/Chicago")
```
