# Fable queue — deferred hard work

Written 2026-08-26 at the end of the multi-agent build of the personal-assistant
spec (`docs/specs/2026-08-26-jarvis-personal-assistant.md`). Everything here was
deliberately **not** done in that pass: it needs either a live Claude CLI run, a
quiet machine, the app restarted, or judgement a small model should not make.

This file is written for a **fresh agent on a stronger model with no memory of
that conversation**. Every item names the exact file and line, what to do, and
what "done" looks like. Nothing below requires reading the build transcript.

Ground rules that still apply:

- Python is `~/vss_env/bin/python`. Tests: `cd ~/Jarvis && ~/vss_env/bin/python -m pytest -q tests`.
- The local model is `gemma4:26b` (Ollama, tools, `think:false`, `keep_alive:-1`,
  `num_ctx:8192`) and must stay the **only** resident model.
- Spoken text is film-JARVIS, <= 2 sentences (briefings and list read-outs excepted).
- Never write to `/tmp/vss_voice`. Screen capture only via
  `DISPLAY=:1 ~/vss_env/bin/python <scratchpad>/shot.py PREFIX` (flash-free), never `gnome-screenshot`.
- No real secrets in the tree — placeholders only.
- The tree is ~60 files uncommitted; no `git checkout/stash/reset/clean/restore`.

Priority order: **1 (permissions) > 2 (verification passes) > 3 (open issues) > 4 (persona, likely not needed)**.

---

## 1. The `TODO(fable)` — real non-interactive permissions for the Claude CLI

> **DONE 2026-08-27** (Opus 5, live against Claude CLI **2.1.247**). All five
> checks passed unmodified — the Jarvis side needed no correction. Three live
> runs on a throwaway dir drove `claude -p` against this repo's own MCP server
> and a real `ApprovalBroker`:
>
> | run | result |
> |---|---|
> | answer **yes** | tool fired, `{"behavior":"allow","updatedInput":…}` accepted, **the file appeared**, rc 0 |
> | answer **no** | file absent, our message came back verbatim as an `is_error` tool_result (`non_execution_kind: "permission-rule"`), task ended cleanly, rc 0 |
> | **no answer** | denied with `approvals.TIMEOUT_LINE` ("No answer in two minutes, sir; I've declined it."), rc 0, no hang |
>
> Findings, now recorded at the top of `jarvis/claude_session.py`:
> 1. `--permission-prompt-tool` exists in 2.1.247, hidden from `--help`, and
>    takes `mcp__<server>__<tool>`. (An unknown flag dies with "unknown
>    option"; this one parses.)
> 2. `--mcp-config <file>` accepts the `{"mcpServers": {…}}` shape
>    `write_mcp_config()` writes — init reports the server `connected`.
> 3. `permission_result()`'s payload shape is correct as written.
> 4. `--permission-mode acceptEdits` routes **only** out-of-project actions to
>    the prompt tool; an in-project Write is auto-accepted and never asks.
>    That is the split we want — no spoken question for ordinary work.
> 5. The MCP subprocess inherits the `env` block (socket, project, PYTHONPATH).
> 6. Extra: the broad `Write(/**)` rules in `ALLOW_RULES` do **not** suppress
>    the prompt tool for out-of-project paths, so no project scoping is needed.
>
> Wired up: `claude.permission_prompt_tool` now defaults **true** (in
> `DEFAULTS` and in `~/.config/jarvis/assistant.json`); `submit()` no longer
> fails closed outside `allowed_dirs` — it runs under the prompt tool and asks,
> and `OUTSIDE_LINE` now means only "the prompt tool is off" (a real fallback);
> the `TODO(fable)` blocks are gone; the gate tests are inverted and a new test
> covers the fallback. Full suite green (1242 passed, 11 xfailed, 0 failed).
>
> Two defects found and fixed on the way, both unrelated to permissions:
> - `jarvis/app.py` built the brain namespace from **bound methods captured at
>   init**, so `app.brain.<fn> = …` was silently a no-op and calls reached the
>   real local model. Now delegates lazily.
> - `tests/test_found_calendar_partial_failure.py` hard-coded `20260826`, so it
>   stopped exercising its bug (and XPASSed strict) at the first midnight. Now
>   date-relative.

### Where it lives

| file | line | what is there |
|---|---|---|
| `jarvis/claude_session.py` | 76-113 | the `TODO(fable) 7.3` block: the five unverified assumptions, verbatim |
| `jarvis/claude_session.py` | 52 | `PERMISSION_TOOL = "mcp__jarvis__approve"` |
| `jarvis/claude_session.py` | 1151-1162 | `write_mcp_config()` — writes `{"mcpServers": {"jarvis": {command, args, env}}}` to `<task_dir>/<slug>/mcp_jarvis.json` |
| `jarvis/claude_session.py` | 1243-1275 | `build_command()` — the two flags are appended **only** when `self.permission_prompt_tool` is true |
| `jarvis/claude_session.py` | 1195-1201 | `submit()` fails closed with `OUTSIDE_LINE` when `project_allowed()` is false |
| `jarvis/claude_session.py` | 955 | `project_allowed()` |
| `jarvis/assistant_config.py` | 66-70 | `DEFAULTS["claude"]["permission_prompt_tool"] = False` + the same TODO |
| `jarvis/mcp_permissions.py` | 1-248 | the stdio MCP server; `permission_result()` at 112, `serve()` at 195, `main()` at 235 |
| `jarvis/approvals.py` | 244-451 | `ApprovalBroker`: `_handle()` 368, `auto_policy()` 414, `answer()` 399, `ask()` 473 |
| `tests/test_claude_session.py` | 403, 495-520 | the two tests that assert the flags stay **off** |
| `tests/test_mcp_permissions.py` | 14 | the note saying the CLI half is unproven |

The Jarvis half is finished and unit-tested. What is unproven is the **CLI
contract**. Installed CLI on this box is **2.1.247** (the spec block says
2.1.246 — re-read `claude --version` first; the contract may have moved).

### The five things to verify (do them in this order, on a throwaway dir)

1. **Does `--permission-prompt-tool` exist in this build, and does it take
   `mcp__<server>__<tool>`?** It is hidden from `claude --help`. Check
   `claude --help`, `claude -p --help`, and the official Claude Code docs for
   the current names (the flag, `--permission-mode`, and whether the MCP
   permission-tool contract has been renamed). Do **not** assume the shape in
   the TODO block is still current — that is the whole point of this item.
2. **Does `--mcp-config <file>` accept the file shape** `write_mcp_config()`
   writes (`{"mcpServers": {...}}`), or does this build want an inline JSON
   string / a different key?
3. **What tool-result payload does the CLI expect back?**
   `jarvis/mcp_permissions.py:112 permission_result()` returns ONE text block
   holding `json.dumps({"behavior": "allow", "updatedInput": <input>})` or
   `{"behavior": "deny", "message": "..."}`. Confirm against the docs and a live
   run; fix `permission_result()` if the shape differs.
4. **Does `--permission-mode acceptEdits` still route out-of-project edits to
   the prompt tool?** If it does not, the tool never fires and the whole flow is
   dead — find the mode that does (`default`? `plan`?) and set
   `claude.permission_mode` accordingly.
5. **Does the MCP subprocess inherit the `env` block** (`JARVIS_APPROVAL_SOCK`,
   `JARVIS_PROJECT`, `PYTHONPATH`) from the config? Prove it by logging in
   `mcp_permissions._log()` (stderr only — stdout is JSON-RPC).

Suggested probe (cheap, no Anthropic credits beyond one tiny task):

```
mkdir -p /tmp/claude-1000/.../scratchpad/fable_probe && cd it
claude -p --output-format stream-json --verbose \
  --mcp-config <path to a hand-written mcp_jarvis.json> \
  --permission-prompt-tool mcp__jarvis__approve \
  --permission-mode acceptEdits <<< 'Write the word OK into /home/hunterp/Jarvis/NOT_ALLOWED.txt'
```
and read **stderr** plus the `stream-json` events. Point
`JARVIS_APPROVAL_SOCK` at a broker you start yourself in-process
(`ApprovalBroker(sock, cfg).start()`) so nothing needs the running app.

### Then wire the ask-aloud path

The user decision of 2026-08-26 (recorded at `jarvis/claude_session.py:107-112`)
is that work is auto-approved **anywhere**: `claude.auto_approve_anywhere`
defaults to `True`, so `submit()` accepts any project and `OUTSIDE_LINE` is only
reachable with that flag off. The fable pass must make the **other** branch work:
with `auto_approve_anywhere` false (or for a tool the allow list does not cover),
an outside-allowed-dir action must make Jarvis **ask the user aloud** —
`ApprovalRequested` -> spoken question -> "yes"/"no" from voice, typed, UI or
Discord -> `ApprovalBroker.answer()` -> `{"behavior": "allow"}` back to the CLI —
instead of failing closed at `claude_session.py:1195-1201`.

Concretely, once 1-5 pass:

- flip `claude.permission_prompt_tool` to `True` in `DEFAULTS`
  (`jarvis/assistant_config.py:69`) and in `~/.config/jarvis/assistant.json`;
- replace the fail-closed return at `claude_session.py:1197-1201` with the start
  of the task under the prompt tool (keep `OUTSIDE_LINE` only for the case where
  the prompt tool is unavailable — i.e. keep a real fallback, never a silent one);
- delete the `TODO(fable)` blocks at `claude_session.py:76` and
  `assistant_config.py:66` **only** when all five checks are recorded as passed;
- update the two gate tests (`tests/test_claude_session.py:403`, `:507`) — they
  currently assert the flags are absent; they must then assert they are present
  and correctly formed, and a new test must cover the still-unavailable fallback.

### Acceptance criteria (item 1)

- `claude --version` and the flag/contract findings written into the top of
  `jarvis/claude_session.py` replacing the TODO block (facts, not assumptions),
  with the date and CLI version.
- A live run in a scratch project where a `Write` to a path outside
  `claude.allowed_dirs` produces a **spoken** Jarvis question, a typed/spoken
  "yes" allows it, and **the file actually appears**; and a second run where
  "no" denies it, the CLI reports the denial, and the task continues or ends
  cleanly (no hang, no crash).
- A two-minute no-answer denies with `approvals.TIMEOUT_LINE`.
- `claude.permission_prompt_tool` true by default, tests updated, full suite green.
- Spec 13.6 satisfied: "the outside-dir Write triggers the spoken approval
  question, a typed 'yes' allows it and the file appears".

Budget note: Claude CLI runs cost the user money. Two small tasks (one allow,
one deny) are enough. Do not loop.

---

## 2. The four adversarial verification passes that never ran

The build fleet was killed before its verification wave. Each pass below is
**adversarial**: the job is to try to break the claim, not to confirm it.

### 2.1 Regression / integration

- Run the whole suite **on a quiet box**: `cd ~/Jarvis && ~/vss_env/bin/python -m pytest -q tests`.
  Last known good: **1066 passed, 12 skipped in ~40 s**. Anything slower than
  ~2 minutes means another process is competing (during the build, ~20 concurrent
  pytest runs made it look like a hang) or a real deadlock has been reintroduced
  — one was found and fixed in `jarvis/tools/calendar.py` (`_save_cache` took a
  non-reentrant lock, then read the `fetched_at` property which takes it again).
- Then `JARVIS_LIVE=1 ~/vss_env/bin/python -m pytest -q tests` for the live-network
  tests (Open-Meteo, ipapi, Hacker News, a public Google ICS, Ollama). Expect the
  12 skips to become passes.
- `tests/test_app_wiring.py` is the integration surface: real modules on tmp
  paths, one utterance per route, an approval round trip through the real socket
  and `mcp_permissions` in a subprocess, a timekeeper timer firing through the
  bus, the `BRIEFING` tag path. Read it adversarially: does each test actually
  assert the service call, or only that nothing raised?
- Cross-item fragility to check: `tests/test_notes_mail.py` and
  `tests/test_briefing.py` reach into `jarvis/brain.py` internals (`_chat_sync`,
  the module-level `_http` seam, the `{"role": "tool", "tool_name": ...}` message
  shape). If the brain's tool-loop message shape changes, those two files break.
- **Acceptance:** full suite green twice in a row on an idle machine, `JARVIS_LIVE=1`
  green, and a one-line note of the wall time so the next agent can spot a hang.

### 2.2 Assistant tools + the latency bar, re-measured with gemma4 resident alone

Spec 4.3 / 13.2 bars: simple answer **p50 <= 1.0 s, p90 <= 2.0 s**; one-tool
round trip **p50 <= 3.0 s**; `ollama ps` shows only `gemma4:26b`;
`prompt_eval_count` on the second identical-prefix call < 300 tokens.

Measured at build time (`<scratchpad>/bench_resident.md`, `persona_gemma4.md`):

| metric | measured | bar |
|---|---|---|
| simple answer, model-side p50 / p90 | 0.55 s / 0.95 s | met |
| simple answer, **wall** p50 / p90 | 2.42 s / 3.07 s | **missed** |
| one-tool round trip, wall p50 | 4.64 s | **missed** |
| Ollama per-request overhead (resident, prompt-size independent) p50 | 1.86 s | — |
| tool schema tokens | 749 | <= 900, met |
| tool choice | 11/12 | 12/12 |

The gap is **not** in `brain.py`: Ollama 0.30.11 spends 1.3-2.5 s per request in
its own scheduler before the runner sees the prompt, reproducible with curl on a
19-token prompt with no system message and no tools
(`<scratchpad>/w5/probe_overhead.py`; one request: runner slot work 269 ms, HTTP
total 2.92 s). This needs an **ops** change — a newer Ollama, or scheduler
settings — which no build agent had sudo for.

Do, in this order:
1. `ollama ps` — confirm gemma4:26b is the only resident model, `keep_alive` forever.
2. Re-run `~/vss_env/bin/python <scratchpad>/bench_resident.py --runs 3` **with
   nothing else on the box** and record model-side vs wall separately.
3. Try the ops fix (upgrade Ollama / tune the scheduler), re-measure, and write
   the before/after into `docs/capabilities.md`'s latency column.
4. Adversarially exercise every assistant tool end to end through the registry —
   timekeeper (`set_reminder`, `set_timer`, `set_alarm`, `manage_schedule`),
   location/time, weather (now/today/tomorrow/week + a named city), calendar,
   notes/to-dos, mail, briefing — and check the **spoken** output is film-JARVIS
   and <= 2 sentences (list read-outs may reach 3; `manage_schedule(action="list")`
   reports `max_sentences=3` and the brain must honour it, not clamp to 2).
5. Re-check tool choice: "What time is it?" is currently answered from the
   background block instead of calling `get_time` (11/12). Decide whether that is
   acceptable (the commander answers clock questions at Tier 1 anyway) or fix it.
- **Acceptance:** either the bars are met and recorded, or a written, evidenced
  statement of exactly which layer misses them and what ops change is required,
  with `bench_resident.md` regenerated on an idle box.

### 2.3 Claude session, end to end (including an outside-dir denial and a two-turn resume)

Depends on item 1 for the denial half. Cover:
- a smoke task in a scratch project ("Reply with the single word OK") — the pill
  shows WORKING, progress lines appear in the transcript, a spoken summary lands
  at the end, and the summary does not invent facts;
- an **outside-dir Write** that must trigger the spoken approval question:
  once answered "yes" (file appears), once "no" (denied cleanly), once left
  unanswered for two minutes (`TIMEOUT_LINE`, denied);
- a **two-turn resume**: submit a task, let it finish, then "pick up where we
  left off" — `--resume <session id>` must reach the same session and the second
  turn must build on the first (assert on the transcript, not on the reply text);
- `cancel` stops a running task within 5 s (spec 13.6);
- the terminal button opens exactly one `gnome-terminal` attached to the right
  tmux session, and a second click **raises** it rather than opening a second;
- queueing: a second task for the same project answers with `BUSY_LINE` and runs
  after the first;
- model escalation: "use fable" / "think hard" / a large size estimate produce
  `--model fable`.
- **Acceptance:** spec 13.6 fully demonstrated, with the two-turn resume and the
  denial explicitly shown; each claim backed by a log excerpt or screenshot.

### 2.4 UI

Spec 11 item 11 and 13.8. This pass owns the **only** app restarts — no other
item may restart Jarvis (`/tmp/vss_voice/jarvis.pid`).
- Idle text census via `shot.py` unchanged from the clean spec except the
  terminal button glyph; with an active project the PROJECT chip appears exactly
  once; after a briefing exactly one card carries it.
- A typed "what's the weather" exchange renders correctly.
- A typed "timer for 1 minute" fires with speech + toast; an alarm rings via
  `paplay`, the modal shows, DISMISS stops it.
- The approval modal shows ALLOW / DENY and its answer reaches
  `ApprovalBroker.answer(..., source="ui")`.
- A Claude task shows WORKING, progress lines, and the summary card.
- Check the typeface decision landed (`jarvis/ui/theme.py` — Chakra Petch display,
  JetBrains Mono values; both families are already installed under
  `~/.local/share/fonts/jarvis/`).
- **Acceptance:** screenshots for each of the above, taken with `shot.py` only,
  and the census diff written down.

---

## 3. Open issues reported by the build items

Carried over verbatim in substance from the item reports. **The report stream
this file was written from was truncated after W6**, so items W7-W10 and the
FONT / LAUNCHER / PRIVACY / SPOTIFY side-items may have open issues not listed
here — read `<scratchpad>/hooks_*.md` (W1-W10 plus those four) before starting,
since each records what its item still needs from files it did not own.

### Blocking-ish

1. **Latency bars missed** (W5) — see 2.2. Ops-level, not code.
2. **Gmail never proven against a real mailbox** (W4) — `~/.config/jarvis/assistant.json`
   has no `gmail.address` / `app_password` (placeholders only). Every mail
   assertion rests on a fake IMAP. Watch two things on the first real run:
   Gmail's exact FETCH response framing through `_parse_fetch`, and whether Gmail
   honours the `BODY.PEEK[TEXT]<0.2000>` partial fetch.
3. **No real calendar configured** (W3) — the Google secret-iCal and iCloud CalDAV
   paths are proven only against fixtures, a fake CalDAV client, and one live read
   of Google's public US-holidays ICS. The briefing therefore reads
   "Calendar: not available" in every recorded run.
4. **`home_location` is unset** (W3) — location falls back to IP geolocation and
   resolves to College Station, Texas. If that is wrong, set `home_location` in
   `~/.config/jarvis/assistant.json`.

### Deliberate spec deviations to ratify or reverse

5. **`FEW_SHOT_POOL` is greeting/joke/advice only**, not the seven families spec
   4.4 enumerates (`jarvis/brain.py`). Evidence: with the eight-family pool, 6 of
   14 base prompts came back as verbatim few-shot copies; with the trimmed pool
   plus the closing clause "the examples are the manner only, never the words",
   0 of 14 (`<scratchpad>/w5/ab_parrot.json`, variants A vs E). Restoring the
   enumerated pool restores the parroting.
6. **`brain.CLASSIFY_TIMEOUT_S = 5.0`**, not the 3.0 s of spec 4.2 — 3 s timed out
   into a silent `("local", 0.0)` on a third of the bench's router questions.
7. **`brain.local_line()` at its spec default (2.0 s)** effectively never returns
   model text on this box; the caller always gets the fallback. The fix belongs in
   `jarvis/commander.py` (pass `timeout=5.0`); the exact snippet is in
   `<scratchpad>/hooks_W5.md` (1).
8. **W4 parses feeds with `xml.etree`**, per spec 6.7, not `feedparser` as an
   early brief said. No `feedparser` dependency is installed. Leave as is unless
   someone wants the dependency.

### Smaller

9. **Alarm with `ring=False` never times out** (W2) — `_service_ring` returns
   immediately with no `_Ring`, and `max_ring_s` is only checked there, so a
   ringing alarm waits indefinitely for a dismiss. Matches the existing test
   `test_ring_disabled_still_publishes`; it is the one asymmetry with the spec's
   "after `max_ring_s`, mark missed" rule.
10. **`parse_when('tomorrow at 6:30')`** resolves to 18:30 for reminders and 06:30
    for alarms (`set_alarm` passes `prefer="morning"`). Correct per spec for the
    alarm example; a user wanting a 6:30 **reminder** must say "am".
11. **`get_weather` for a named city** emits "In London, England: ..." — a
    fragment before the forecast clause. Reads acceptably; re-check once gemma4 is
    speaking it aloud.
12. **`CalendarSource.get()` takes unused `range` / `now` arguments** — dead
    surface area matching the spec signature.
13. **`summarize()` occasionally embellishes a noun** ("all 1066 tests pass" came
    back as "1066 persona evaluation tests"). Within the guards, but worth an eye
    if summaries of Claude results start naming things Claude did not say.
14. **`docs/capabilities.md` is append-only and was appended by several items
    concurrently.** The llama3.2-era persona row is now stale in its model,
    few-shot-pool and status columns; rows may interleave. The wiring item owns
    the file — fold and re-order it in one pass.
15. **Local data hygiene** (`<scratchpad>/hooks_PRIVACY.md`): `/tmp/vss_voice` is
    world-readable and `jarvis.log` records conversation text. Once calendar,
    email and notes are live that is personal content in a world-readable file.
    Required: log dir `0o700` / log file `0o600`, the same for the SQLite stores,
    and a `privacy.log_conversation_text` switch defaulting to false.
16. **Launcher latency** (`<scratchpad>/hooks_LAUNCHER.md`): 15 s from click to
    visible window because `main()` preloads torch/whisper/TTS before mapping the
    Tk window, and a second click is a silent no-op while the first instance
    boots. Map the window first, import after.

---

## 4. Persona quality pass on gemma4 — **not triggered**

Condition: run this only if `<scratchpad>/persona_gemma4.md` shows film fidelity
below 8. It does not. The shipped-pool column reports **100 % clean rows**:

- 0 verbatim pool parrots, 0 near-parrots (difflib ratio >= 0.75), 0 riddles,
  0 fake actions, 0 invented clock readings;
- 0 markdown, 0 emoji, 0 assistant-speak / corporate words, 0 "Jarvis:" labels;
- 100 % address "sir", 100 % <= 2 sentences, 93 % exactly one sentence;
- avg 71 chars, avg 1.07 sentences.

That is comfortably at or above the bar of spec 13.3 (zero PARROT/RIDDLE/FAKE-ACTION,
"sir" in >= 85 %, every reply <= 2 sentences). **No persona rework is queued.**
The only persona-adjacent follow-ups are the two ratification questions above
(items 5 and 13) and, optionally, a fresh 14-prompt run after any change to
`JARVIS_SYSTEM` or `FEW_SHOT_POOL` — re-run
`~/vss_env/bin/python <scratchpad>/persona_eval.py` and regenerate
`persona_gemma4.md` rather than eyeballing a handful of replies.

---

## Recovery notes

**What happened.** A fleet of ~14 agents (W1-W10 plus the FONT, LAUNCHER,
PRIVACY and SPOTIFY side-items) was implementing the spec in parallel when the
Anthropic API credit balance ran out. Every agent was killed mid-flight. No agent
got to run its verification pass; several were killed between writing a module
and writing its tests.

**What the outage interrupted.**

- The four verification passes of section 2 — none of them ran.
- The `TODO(fable)` permission work of section 1 — deliberately deferred before
  the outage, and still open.
- Several test files were left in a pre-port state (e.g. `tests/test_persona.py`
  was still the llama3.2-era file with `/api/generate` mocks and 8 failures).
- One real deadlock was left in the tree (`jarvis/tools/calendar.py`
  `_save_cache` re-entering its own non-reentrant lock), which made
  `pytest -q tests` hang forever for **every** agent on the box.

**How the tree was recovered.** A second fleet was launched with recovery
instructions: read the files you own first, diff them mentally against the spec,
**finish rather than rewrite**, preserve working code, and make the tests real
and green. Ownership stayed exclusive per spec 12; anything an item needed in a
file it did not own went as an exact snippet into `<scratchpad>/hooks_<ITEM>.md`,
and the wiring item (W10) was the only one allowed to apply those snippets. The
result: all modules import cleanly, the calendar deadlock is fixed, and the full
suite reports **1066 passed, 12 skipped in ~40 s** on an idle box.

**Traps this left behind, for the next agent.**

- Concurrent pytest runs make the suite look hung. It is not; it is contention.
  Check for other `pytest` processes before diagnosing a hang.
- Nothing is committed. ~60 files are uncommitted and there is no clean baseline
  to diff against — do not use `git checkout/stash/reset/clean/restore`.
- The scratchpad
  (`/tmp/claude-1000/-home-hunterp/7142dc22-eac5-41b8-9af7-9ba3e52cdbc2/scratchpad`)
  holds the evidence: `hooks_W1.md` … `hooks_W10.md`, `hooks_FONT.md`,
  `hooks_LAUNCHER.md`, `hooks_PRIVACY.md`, `hooks_SPOTIFY.md`,
  `persona_gemma4.md` / `.json`, `persona_eval.py`, `bench_resident.md` / `.py`,
  `w5/probe_overhead.py`, `w5/ab_parrot.json`, and the per-item demo scripts
  (`w6_demo.py`, `w7_demo.py`, `tk_live.py`). It is a `/tmp` scratchpad — copy
  anything worth keeping into the repo before it is cleaned.
