# The clock guard

**Prove the suite does not care what time it is:**

```bash
scripts/clock_guard.sh
```

Exit 0 means every configuration agreed. Exit 1 names the configuration and
the test whose verdict moved.

Six full runs, so budget about **half an hour** on an idle box — measured
2026-09-05, 17:04-17:32, at ~4m30s per configuration, plus a few minutes of
confirmation re-runs when there is a candidate. While iterating on one file,
scope it and it takes seconds:

```bash
PYTEST_ARGS=tests/test_foo.py scripts/clock_guard.sh
```

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

The synthesised offset is always a whole **minute**, and is rounded **up**
so the simulated local time sits at or just after the time you asked for,
within 59 s. Both halves of that matter, and the second was learned the
hard way on 2026-09-05:

* no real timezone has ever had a sub-minute offset, and simulating one is
  not harmless. An RFC 2822 date (`email.utils.format_datetime`) can only
  express ±HHMM, so odd seconds are silently truncated and a serialised
  timestamp comes back up to 59 s adrift. The first full-suite run of this
  guard reported three `tests/test_notes_mail.py` tests as clock-dependent
  at `--clock-at=09:00`, which had produced the offset `+12:08:12`. They are
  green at all 24 hours under libfaketime and green at a whole-hour offset
  on either date — the finding was the **instrument**, not the suite.
* rounding up rather than down keeps `--clock-at=09:00` inside hour 09, and
  keeps `--clock-at=23:59` before midnight, which is what makes that
  configuration cross the date during the run.

Note also that reaching a given hour may put the process on the **next or
previous local date** — `--clock-at=09:00` run at a real 15:51 lands on
tomorrow at UTC+12:07. That is harmless for date-relative tests and is why
`--clock-at=23:59` is a real midnight-crossing, but it is worth knowing when
reading a failure.

## A moved verdict is not yet a finding

The suite carries known flakes — measured 2026-09-05, `test_brain_room`'s
`[3072]` case fails about **1 run in 10 on a pristine checkout** — and a
flake landing in one configuration looks exactly like clock dependence.

So `scripts/clock_guard.sh` re-runs every candidate before reporting it:
three times in a configuration where it was green and three times in one
where it was red. Only a verdict that holds **6/6** is called
clock-dependent; anything else is printed under "NOT clock-dependent —
verdict did not hold on re-run (flaky)" and does not fail the guard.

This is not politeness. A guard that cries wolf gets ignored, and being
ignored is exactly how "1 pre-existing time-of-day failure" survived three
days. Set `CONFIRM_N` to change the number of repeats.

Node ids are matched as **fixed strings**. A parametrised id ends in
`[3072]`, which to `grep`'s default expression syntax is a character class
matching one of `3 0 7 2` — so `grep -x` silently never matched it. Measured
2026-09-05: that left the confirmation step with no configuration to re-run,
it passed an empty argument to pytest, pytest failed on it, and the guard
reported `RED in  (3/3)` — a confirmed finding, against nothing. Every
parametrised test in the suite was in that blind spot. `grep -qxF` is the
fix, and the confirmation step now refuses to run rather than confirm a
finding it cannot locate on both sides.

## The two halves

**Runtime** — `tests/conftest.py` adds two options:

```bash
pytest --clock-at=09:00        # as if the local clock said 09:00
pytest --clock-at=23:59        # ... including a run that crosses midnight
pytest --clock-tz=UTC          # as if the machine were in another zone
```

Both print the simulated time in the pytest header, so a red CI run is never
mistaken for a real failure. Scope a sweep while iterating with
`PYTEST_ARGS=tests/test_foo.py scripts/clock_guard.sh`, and use
`scripts/clock_guard.sh --hours` for all 24.

**Static** — `tests/test_clock_hygiene.py` runs inside the normal suite
(~0.2 s) and parses `tests/` for the three shapes that produce a
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
3. **Reading the time of day inside a zone-pinned scope.** This one exists
   because the runtime half has a blind spot in it, and the blind spot was
   measured rather than reasoned about. `--clock-at` simulates an hour *by
   shifting the zone*, so a `pytest.mark.local_tz(...)` scope overrides it
   and is invisible to the sweep. On 2026-09-05, at a real clock of 14:52,
   a deliberately clock-dependent test added to the module-pinned
   `tests/test_arc.py` **passed all six** guard configurations — `real`,
   `--clock-at=09:00 / 13:00 / 21:00 / 23:59:30` and `--clock-tz=UTC` —
   while being plainly red at any real hour outside the afternoon. Inside a
   pinned scope, take the instant from the file's own fixtures.

   Shape 3 is scoped narrowly on purpose, because a guard that cries wolf
   gets waived and then written off, which is the failure this guard exists
   to end. Only *time-of-day* reads count — `time.time()` is an epoch
   scalar and cannot expose a local hour, so `time.time() + 8 * 3600` in
   `test_winddown.py` is left alone — and only *pinned* scope counts: a
   module-level `pytestmark` pins the file, while a `@pytest.mark.local_tz`
   decorator pins only that test and the rest of the file is still swept.

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
* The **runtime** half cannot see inside a `local_tz` scope at all, for the
  reason given under shape 3. Shape 3 is the compensation, and it is static:
  it can see a wall-clock *read*, not a wrong *answer*. The four pinned
  files were therefore checked by hand with libfaketime, which moves the
  absolute instant and so survives the pin — 131 passed at each of the 24
  hours on 2026-09-05. A change to those files owes that check again.
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
