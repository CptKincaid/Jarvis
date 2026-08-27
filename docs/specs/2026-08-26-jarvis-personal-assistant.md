# Jarvis Personal Assistant — binding architecture spec

**Date:** 2026-08-26 (evening)
**Builds on:** `2026-08-25-jarvis-v3-overhaul.md` (event bus, module layout, threading, "commands are data") and the clean-HUD pass (scratchpad `clean_spec.md`: every datum once, no decorative text, <= 2 spoken sentences except briefings, film-JARVIS persona lives in `brain.py`).
**Machine:** DGX Spark GB10 (aarch64, 128 GB unified, 20 cores), Ubuntu GNOME X11 `DISPLAY=:1`, **no microphone** (typed in, spoken out), Python `~/vss_env/bin/python`, Ollama 0.30.11 at `http://localhost:11434`, Claude CLI 2.1.246 at `~/.local/bin/claude`, tmux 3.4, gnome-terminal, xdotool, paplay, notify-send. No sudo. The Jarvis app is RUNNING on the user's desktop (`/tmp/vss_voice/jarvis.pid`).

## 0. The user's plan (verbatim, binding)

> "I want Jarvis to basically act my personal assistant so i can ask him about the weather, the whats on my google calendars etc. We can upgrade from llama to whatever the best model for conversation is ... I want to be able to have him be my automated personal assistant that i can use to code and he will use you, the claude cli to code and do tasks that you would be able to do he basically is just a new coat of paint instead of this boring UI. HOWEVER, for simple tasks like checking my google calendar or apple calendar or weather or time and date and location he will use a local model."
> "a button that i can click that will show me the cli terminal he is working in/out of put the icon next to the mic at the bottom"

Answers that bind design choices: chat / simple tasks must be *smart but fast, under a second, max 2 seconds*; calendars read-only with zero OAuth now (Google secret iCal URLs; iCloud CalDAV app-specific password), Claude connectors later; Claude coding = a live session per project with auto-approval inside allowed project dirs and a pop-out terminal button; extras = ALL of reminders & timers, Gmail summaries, notes & to-dos, briefing — all SPOKEN; "he will need some sort of local time keeper and i want to be able to set alarms with him where he can wake me up". User's name Hunter; timezone = system (America/Chicago); US units. Claude session rules, assistant extras and the chosen local model are reproduced in the relevant sections below.

## 1. Hard rules (every work item, every judge)

1. **No git checkout/stash/reset/clean/commit/restore** (~50 uncommitted files). Edit in place only. No destructive shell commands.
2. **Never write to `/tmp/vss_voice` from tests** (`tests/conftest.py` firewalls it; every new module takes its paths from `PATHS`/constructor args so the firewall holds). New state files go under `PATHS.MEMORY_DIR` (`~/.aiws_trainer/jarvis_memory/`) or `PATHS.LOG_DIR`; tests pass tmp paths.
3. **Only the UI work item may restart the app** (budget 2). Launch: `cd ~/Jarvis && DISPLAY=:1 setsid nohup ~/vss_env/bin/python -m jarvis.app > /tmp/vss_voice/jarvis_launch.log 2>&1 < /dev/null & disown`. Stop: `kill $(cat /tmp/vss_voice/jarvis.pid)`, wait until `pgrep -f jarvis.app` is empty, `sleep 5`. Screen capture ONLY via `DISPLAY=:1 ~/vss_env/bin/python <scratchpad>/shot.py PREFIX`; never gnome-screenshot; no continuous captures; never loop-create Tk windows.
4. **Claude CLI calls are budgeted: <= 4 in the whole workflow** (they use the user's subscription). Section 12 allocates them. Unit tests never call Claude or the network; live tests run only with `JARVIS_LIVE=1`.
5. **Secrets:** `~/.config/jarvis/assistant.json` is created `chmod 600` with placeholders only; no real token/password is ever written by an agent, logged, echoed, or put in a test. Everything must work with placeholders missing (spoken "I'll need X set up, sir").
6. **Exclusive file ownership** (section 12). No two items share a file. `brain.py` -> brain item; `commander.py` -> router item; `jarvis/ui/*` -> UI item; `app.py`/`events.py`/`config.py`/`conftest.py`/`docs/capabilities.md` -> wiring item. `events.py` and `jarvis/tools/registry.py` were pre-seeded with the contracts in this spec (already on disk, tests green); their owners may extend, never rename.
7. **Tests stay green:** `cd ~/Jarvis && ~/vss_env/bin/python -m pytest -q tests` (272 pass, 5 skip today) plus every new test file. No new blanket `except: pass`; log with context; surface user-facing failures as Status events or spoken persona lines.
8. **The pip rule:** pure-python packages may be installed into `~/vss_env` (`icalendar`, `recurring_ical_events`, `caldav`, `feedparser`). `websockets` 15, `aiohttp`, `httpx`, `requests`, `python-dateutil`, `tzdata` are already present. Prefer stdlib `urllib`/`imaplib`/`sqlite3`/`xml.etree` where the payoff of a dependency is small.
9. **Persona everywhere:** every spoken line — model output, canned lines, tool excuses, milestone lines, approval questions, alarm wording — is film-JARVIS: dry British understatement, "sir" mostly, one sentence as the norm, two at most, no lists/markdown/emoji. Briefings may run to six sentences. The Tier-2 guards in `brain.py` (`spoken_from_ollama`) remain the gate before TTS.
10. **The clean-HUD principles stay binding for the UI:** every datum once, no decorative text, one annotation size, telemetry only in the status bar, app state only in the header pill.

## 2. Architecture

```
typed / voice / Discord text
        │
        ▼
Commander.handle(text, source)                       [jarvis/commander.py]
  1 dictation  2 ringing-alarm words  3 pending approval yes/no
  4 pending router question  5 desktop chains  6 Tier-1 registry (instant)
  7 voice intent gate  8 jarvis-mode Tier-1 (clock, courtesies, quiet, …)
  9 Router.route(text, active, terminal_open) ─┐   [jarvis/router.py]
        │ local                     │ claude                │ ask
        ▼                           ▼                       ▼
brain.chat(text)             claude.submit(...)      speak ONE question,
[jarvis/brain.py]            [jarvis/claude_session.py]  remember, resolve
gemma4:26b /api/chat            tmux jarvis-<slug>       on the next answer
+ ToolRegistry                  claude --resume … (interactive TUI)
[jarvis/tools/*]                 ├─ ~/.claude/projects/…/<sid>.jsonl tailed
        │                        ├─ milestones → ClaudeProgress / ClaudeTaskState
        │                        ├─ permission tool → mcp_permissions.py → UNIX socket
        ▼                        │     → ApprovalBroker [jarvis/approvals.py] → spoken question
JarvisReply + _say               └─ result → brain.summarize → spoken
                                                   │
                       Alerts hub [jarvis/channels/notify.py] → notify-send + Discord
                       Discord gateway [jarvis/channels/discord.py] → replies back in
Timekeeper [jarvis/tools/timekeeper.py]: SQLite, 1 s tick, reminders/timers/ALARMS,
   AlarmFired → UI modal + paplay ringer; catch-up on boot; autostart entry
```

Threads: Tk main thread = UI only. Every module below runs on worker threads and talks to the UI only through `bus.publish(Event)`. Speech happens ONLY through `JarvisApp._say` (talkback-gated); modules that need speech get a `say` callable injected, never the TTS.

### 2.1 What stays

- Tier-1 registry (desktop chains, clock, courtesies, quiet, say-again, read-aloud, dictation, launch, windows, git status, …) runs BEFORE the router and stays instant.
- `brain.think()` (the legacy `_is_local_question` split) is REPLACED by the router; the tag protocol `[SPEAK]/[RUN]/[TYPE]/[WINDOW]/[SILENT]/[DONE]` and `execute_autonomous` remain for the `deploy`/`autonomous:` phrases (unchanged behaviour, still Tier 3 via `-p --output-format text`).
- `workflows.Workflows` stays; `workflows.Reminders` is superseded by the timekeeper (the app no longer constructs it; its persisted `reminders.json` is imported once).
- TTS, speech cache, pronunciation, speak-queue, reader, history, hotword/recorder paths: untouched.

### 2.2 Services namespace (the app-level contract)

`JarvisApp._build_services()` returns a `SimpleNamespace`; commander/router/tools reach everything through it. New members (all constructed in `app.py`, wiring item; interfaces defined in the sections named):

| member | type | section |
|---|---|---|
| `assistant` | `AssistantConfig` | 10 |
| `tools` | `ToolRegistry` (filled with every `make_tools()` result) | 4.1 |
| `brain` | `SimpleNamespace(think, chat, classify_route, summarize, local_line, execute_autonomous)` | 4 |
| `router` | `Router` | 5 |
| `timekeeper` | `Timekeeper` | 6.1 |
| `notes` | `NotesStore` | 6.5 |
| `claude` | `ClaudeSessionManager` | 7 |
| `approvals` | `ApprovalBroker` | 7.3 |
| `alerts` | `Alerts` | 8.1 |
| existing | `desktop, context, memory, workflows, tts, reader, history` | unchanged |

Every consumer must tolerate a missing member (`getattr(services, name, None)`), exactly as `Commander._svc` does today, so partial wiring never crashes a handler.

## 3. Shared contracts

### 3.1 Events (already in `jarvis/events.py`)

`ClaudeTaskState(project, task_id, state, text)` with `state in {queued, running, waiting, done, failed, cancelled}`; `ClaudeProgress(project, task_id, line, milestone)`; `ActiveProject(slug, path)`; `ApprovalRequested(request_id, question, tool_name, detail, project)`; `ApprovalResolved(request_id, allowed, source)`; `AlarmFired(alarm_id, label, kind, due_text)`; `AlarmStopped(alarm_id, action, snooze_min)`; `BriefingReady(sections, spoken)`. Existing `ReminderFired(text)`, `JarvisReply`, `Status`, `BrainState` keep their meaning. Speech is NEVER triggered by the UI: the app subscribes to the same events and calls `_say`.

### 3.2 Paths

| what | path |
|---|---|
| assistant config | `~/.config/jarvis/assistant.json` (env `JARVIS_ASSISTANT_CONFIG` overrides; tests always set it) |
| timekeeper db | `PATHS.MEMORY_DIR / "timekeeper.db"` |
| notes db | `PATHS.MEMORY_DIR / "notes.db"` |
| claude project state | `PATHS.MEMORY_DIR / "claude_projects.json"` |
| calendar / location / news caches | `~/.cache/jarvis/*.json` (constructor arg; tests use tmp) |
| approvals socket | `PATHS.LOG_DIR / "approvals.sock"` |
| claude task files | `PATHS.LOG_DIR / "claude" / <slug> / <task_id>.{prompt,jsonl,rc,stderr}` |
| mcp config | `PATHS.LOG_DIR / "mcp_jarvis.json"` |
| autostart | `~/.config/autostart/jarvis.desktop` |

The wiring item adds these to `config.PATHS` (`ASSISTANT_CONFIG`, `TIMEKEEPER_DB`, `NOTES_DB`, `CLAUDE_PROJECTS`, `CACHE_DIR`, `APPROVALS_SOCK`, `CLAUDE_TASK_DIR`, `MCP_CONFIG`, `AUTOSTART_DESKTOP`); until then every module computes the same defaults locally from `PATHS.MEMORY_DIR` / `PATHS.LOG_DIR` and accepts an override argument — so no item blocks on `config.py`.

### 3.3 Test policy

- Unit tests: no network, no Ollama, no Claude, no X, no audio. Every module funnels HTTP through ONE module-level function (`_fetch(url, timeout=..., headers=None) -> bytes` or a `transport` argument) that tests monkeypatch. Every subprocess (`tmux`, `paplay`, `notify-send`, `gnome-terminal`, `xdotool`, `gsettings`) goes through a module-level `_run(argv, ...)` seam that tests replace with a recorder.
- Live tests: `@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"))`; also honour the existing `JARVIS_LIVE_OLLAMA`. Live tests are read-only against real services and never touch the live app.
- Every new test module starts with the same firewall pattern as `tests/test_persona.py` (tmp `JARVIS_LOG_DIR`, tmp `JARVIS_ASSISTANT_CONFIG`).
- Time: pure functions take `now` as an argument; scheduler tests use a fake clock, never `sleep` > 0.2 s.

### 3.4 Persona lines used by code (fixed strings, prewarmed in the speech cache)

| situation | line |
|---|---|
| section not configured | `setup_line(section)` from `AssistantConfig` (10.2) |
| router question | "Shall I hand that to Claude, sir, or is it a quick one for me?" |
| approval timeout | "No answer in two minutes, sir; I've declined it." |
| approval denied by user | "Declined, sir." / allowed: "Allowed, sir." |
| alarm ringing | "Sir, it's {time}. {label}." (label default "Time to get up.") |
| missed while down | "You missed {label} at {time} while I was down, sir." |
| late but < 1 h | "While I was down, sir: {label}, due at {time}." |
| timer up | "Sir, your {n}-minute timer is up." |
| reminder | "Sir, this is your reminder. {text}" (existing, keep) |
| claude cancelled | "Stopped, sir." |
| claude busy | "Claude's still on the last one for {project}, sir; I've queued it." |
| briefing off | (tool text, model relays) "The morning briefing is switched off, sir; the toggle is in settings under Briefing." |

## 4. Local model, tool loop and persona (brain.py + tools/registry.py)

**Chosen model (binding):** `gemma4:26b` (MoE 25.2B/4B active, Q4_K_M 18.6 GB, tools+thinking+vision, 262k ctx; `think:false` honoured). Bench: tools 12/12, persona 9.2/10, model-side p50 0.53 s / p90 0.78 s, decode 30-42 tok/s, cold prompt 1897 tok in 2.3 s, cold load 23-59 s. The wall numbers in the bench (p50 2.3 s) were inflated by Ollama swapping models between two concurrent benches (load_duration 0.45-1.4 s on resident models). Files: scratchpad `bench_installed.md`, `bench_scout.md`.

### 4.1 Registry contract (`jarvis/tools/registry.py`, seeded)

`ToolResult(text, ok=True, max_sentences=2, card=None, speak=None)`, `ToolSpec(name, description, parameters, handler)` with `.schema()` producing the Ollama `{"type":"function","function":{...}}` entry, `ToolRegistry.register/register_many/has/names/schemas/call(name, args) -> ToolResult` (never raises). Each tool module exposes exactly `make_tools(cfg: AssistantConfig, services) -> list[ToolSpec]`; handlers are keyword-only and must accept unknown extra kwargs (`**_`) and coerce loose model values ("15" -> 15, "this week" -> "week").

**Tool set (<= 11 tools, descriptions <= 20 words, total schema tokens <= 900 measured via `prompt_eval_count`):**

| tool | parameters | module |
|---|---|---|
| `get_time` | `location?` (city) | tools/location.py (uses zoneinfo; home tz when omitted) |
| `get_location` | — | tools/location.py |
| `get_weather` | `when: now\|today\|tomorrow\|week`, `location?` | tools/weather.py |
| `get_calendar` | `range: today\|tomorrow\|week\|next` | tools/calendar.py |
| `set_reminder` | `when` (natural language), `text` | tools/timekeeper.py |
| `set_timer` | `minutes` (number), `label?` | tools/timekeeper.py |
| `set_alarm` | `when`, `label?`, `repeat?: once\|daily\|weekdays` | tools/timekeeper.py |
| `manage_schedule` | `action: list\|cancel\|stop\|snooze`, `kind?: reminder\|timer\|alarm\|all`, `which?: last\|all\|<text>`, `minutes?` | tools/timekeeper.py |
| `notes` | `action: add\|list\|remove\|search\|done`, `kind: note\|todo`, `text?`, `which?` | tools/notes.py |
| `get_mail` | `limit?` | tools/mail.py |
| `get_briefing` | — | tools/briefing.py |

Enum values are exactly the strings above (the bench showed gemma4 emits clean enums when the schema is an enum; gpt-oss did not).

### 4.2 The tool loop (`JarvisBrain.chat`)

```
chat(text, callback, force_tool=None, max_rounds=3)   # worker thread, publishes BrainState
  messages = [ {"role":"system","content": STATIC_SYSTEM},          # never changes per process
               {"role":"user","content": f"Background:\n{ctx}\n\nHunter: {text}"} ]
  POST /api/chat {model, messages, tools: registry.schemas(), stream: false, think: false,
                  keep_alive: -1, options: {num_ctx: 8192, temperature: 0.7, num_predict: 160,
                  stop: ["\nUser:", "\nHunter:"]}}
  loop (<= max_rounds):
     if message.tool_calls: for each call -> registry.call(name, arguments)
         append assistant message (with tool_calls) then {"role":"tool","content": result.text, "tool_name": name}
         if any result.speak: stop the loop, speak that verbatim (card if any)
         re-POST with the same tools
     else final = message.content
  spoken = spoken_from_ollama(final, ctx, text); cap = max(result.max_sentences) or 2
  tags = [("SPEAK", spoken)] (+ [("BRIEFING", json.dumps(card))] when a briefing card was produced)
  callback(tags); memory.log_habit; context.add_exchange
```

Rules: `force_tool="get_briefing"` is passed as Ollama `tool_choice`-equivalent by putting the tool result in directly (call the tool first, then a single model turn with the result — do not rely on tool_choice support). The dynamic context (time, active window, git line, last 4 exchanges, memory facts) lives in the USER message, never the system prompt, so Ollama's prefix cache hits: the system prompt + tool schemas are byte-identical for the life of the process (few-shots sampled ONCE per process, already so). Request timeout 20 s; a tool loop that exceeds 8 s wall speaks the best available text. On connection refused: Status(warn) "Ollama isn't running" and the persona excuse "I'm afraid my local model is down, sir." — NO silent fallback to Claude for chat (Claude is for work, not weather).

`classify_route(text) -> (route, confidence)`: one `/api/chat` call with `format` = `{"type":"object","properties":{"route":{"enum":["local","claude"]},"confidence":{"type":"number"}},"required":["route","confidence"]}`, `num_predict: 40`, `think: false`, a 40-word system prompt ("local = weather/time/calendar/mail/notes/reminders/chat/general knowledge; claude = writing or changing code, files, repos, running tests, multi-step system work"), timeout 3 s; on failure return `("local", 0.0)`.

`summarize(text, max_sentences=2, timeout=6.0) -> str` and `local_line(instruction, text, max_sentences=1, timeout=2.0, fallback="") -> str`: persona-voiced rewrites through the same static system prompt (no tools), used for Claude result summaries, mail summaries and task acknowledgements. Both return `fallback`/a truncated `trim_spoken(text)` on timeout — never raise.

`think(text, callback)` keeps its signature for the `deploy`/`autonomous:` path only.

### 4.3 Residency and latency (binding)

- `configure(model: str)` at app start (from `assistant.local_model`, env `JARVIS_OLLAMA_MODEL` overrides); default constant `OLLAMA_MODEL = "gemma4:26b"`.
- `ensure_resident()` (worker thread at boot, then every 5 min): `GET /api/ps`; every loaded model that is not ours is unloaded ONCE at startup (`POST /api/generate {"model": m, "keep_alive": 0}`) — never periodically (the user runs other agents); then a warm `POST /api/chat` with the real static system prompt + tool schemas and an empty user turn, `keep_alive: -1`, which also primes the prompt cache; log `ollama: <model> resident (load 23.4 s)`. The periodic check only re-warms ours if it fell out.
- `num_ctx` 8192, not 131072 (the context_length Ollama would otherwise allocate for gemma4 is 262k).
- **Latency bar, re-measured with the model resident alone** (brain item, `scratchpad/bench_resident.py` using `brain.chat` messages exactly as production, 8 one-sentence prompts, 3 warm runs): simple answers **p50 <= 1.0 s, p90 <= 2.0 s** wall as the app sees it; one-tool round trip p50 <= 3.0 s. If the bar is missed: shorten few-shots, trim tool descriptions, drop the git line from the background — before considering any other model. Record the numbers in `docs/capabilities.md`.

### 4.4 Persona port (brain item)

`JARVIS_SYSTEM` is re-tuned for gemma4 (a 25B MoE follows rules; it does not need eleven few-shot families): keep `VOICE_RULES`; keep `FEW_SHOT_PINNED` (you there / thanks / good night); reduce `FEW_SHOT_POOL` to greeting, small talk, joke, cannot-act, bad idea, mistake, advice (one or two variants each); DELETE the "no data" family (the model now has weather/clock tools — the old "I don't have a weather feed" line would be parroted) and rewrite "capability" to the new truth ("I keep your calendar, weather, mail, reminders and alarms, sir, run the desktop, and hand the real coding to Claude"). Add the tool rules to the static prompt: use a tool whenever the answer depends on live data (time, weather, calendar, mail, schedule, notes); never guess those; after a tool result answer in one or two sentences using only numbers from the result; if a tool says something is not set up, say so in one sentence and name the thing. Keep `build_ollama_system(context_text, memory_text, shots)` importable for the eval harness (it now returns the STATIC prompt; a new `build_user_turn(context_text, memory_text, text)` renders the dynamic part). Re-run `scratchpad/persona_eval.py` against gemma4 via `/api/chat` (the harness must be updated to the new call shape; it lives in the scratchpad and is owned by the brain item), write `persona_gemma4.md`, and keep `tests/test_persona.py` green (its mocked-Ollama tests move to `/api/chat`; the FEW_SHOT invariants are updated to the new pool; the parrot/riddle/fake-action flags must be zero on the 14 base prompts).

## 5. Router and commander

### 5.1 Router (`jarvis/router.py`)

```python
@dataclass
class RouteDecision:
    kind: str            # "local" | "claude" | "ask" | "action"
    reason: str          # rule name, for the log
    prompt: str = ""     # the text handed to Claude (utterance, skill-expanded)
    project: str = ""    # slug when the utterance named one
    action: str = ""     # for kind == "action": cancel | work_on | resume | new_project | set_model | fast_mode | parallel
    args: dict = field(default_factory=dict)
    confidence: float = 1.0

class Router:
    def __init__(self, cfg: AssistantConfig, classify=None):   # classify = brain.classify_route
    def route(self, text: str, active_project: str | None) -> RouteDecision
    def resolve_answer(self, text: str) -> str | None   # "claude" | "local" | None for a pending ask
```

Rule order (pure, tested; the model is ONLY the tie-breaker):
1. Actions: `cancel|stop that|abort` -> `cancel`; `work on <name>` / `switch to the <name> project` -> `work_on`; `pick up where we left off|what we were working on (yesterday)?|continue the <name> project` -> `resume` (args: `when`, `name`); `start a new project called <name>` -> `new_project`; `use (opus|sonnet|fable|haiku)|think hard|this is a big one` -> `set_model` (fable for the last two); `fast mode (on|off)` -> `fast_mode`; `in parallel|at the same time` inside a task -> `parallel=True`.
2. Skill phrases from `cfg.claude.skill_phrases` (regex -> template, `$1` substitution; defaults in 10.1): a match is a `claude` decision with `prompt` = the expanded slash command; `run the <name> skill (on|with)? <args>` -> `/<name> <args>` passthrough.
3. Explicit Claude cues (`claude`, `have claude`, `ask claude`) -> `claude`, cue stripped from the prompt.
4. Coding/dev/system cues -> `claude`: verbs {fix, implement, refactor, debug, write, add, remove, rename, migrate, deploy, build, test, lint, review, commit, push, merge, rebase, install, upgrade, scaffold, generate, optimise/optimize, profile, document, explain this code, why is <x> failing} combined with objects {code, function, class, module, file, test(s), bug, feature, branch, repo, PR, script, error, traceback, package, dependency, config, pipeline, dataset, model training, docker, service} or a path/filename token (`\S+\.(py|js|ts|md|json|yaml|toml|sh)`), or the words "step by step"/"then … then" with a coding verb.
5. Local cues -> `local`: weather, temperature, rain, forecast, time, date, day, calendar, schedule, meeting, appointment, remind, reminder, timer, alarm, wake me, snooze, note, to-do/todo, task list, email, mail, inbox, briefing, news, where am i, location, greetings/courtesies, joke, general-knowledge question words without a coding object.
6. Neither or both -> `classify(text)`: confidence >= 0.75 -> route silently; else `ask` (speak the router question once, remember the utterance for 90 s; `resolve_answer`: yes/claude/have claude/go ahead -> claude; no/you/just you/quick/answer it -> local).
7. Length prior: a `claude` decision needs at least one cue; a bare short sentence (<= 6 words) with no cue is `local` without asking.

The router never speaks or publishes; the commander does.

### 5.2 Commander changes (`jarvis/commander.py`, router item)

`handle()` gains, in this order, before dictation-independent steps: (a) if `services.timekeeper.ringing`: `stop|dismiss|okay|ok|i'm up|shut it off|enough` -> `timekeeper.stop_ringing("dismiss")`, `snooze( \d+)?` -> `snooze(n or cfg.alarms.snooze_min)`; (b) if `services.approvals.pending()`: yes-words -> `approvals.answer(True, source=source)`, no-words -> `answer(False)`; reply "Allowed, sir." / "Declined, sir."; (c) if a router question is pending: `router.resolve_answer(text)` -> dispatch the remembered utterance. Then the existing pipeline. Sources: `"discord"` is treated like `"typed"` (no intent gate).

Registry re-points: `timer` -> `services.timekeeper.add_timer(seconds, label)`; `remind me` -> `timekeeper.add_reminder(due_epoch, text)` via `timekeeper.parse_when`; new Tier-1 entries `alarm` (`wake me (up )?at <t>|set an alarm for <t>|alarm at <t>`) -> `add_alarm`, `what reminders/timers/alarms do i have` -> `list_text()`, `cancel (the|my|all) (reminder|timer|alarm)s?` -> `cancel`; `take note`/`show notes` -> `services.notes`; `answer question` only for ip/uptime/battery (weather and time never reach `jarvis_agent.answer_question` again); `good morning|morning briefing|briefing|what's my briefing` -> `brain.chat(text, force_tool="get_briefing")` ONLY when `cfg.briefing.enabled`, otherwise "good morning" is an ordinary greeting for the local model and an explicit "briefing" request goes to the tool (which answers that it is off). The stale `morning` workflow stays reachable only as the exact word "morning" (unchanged).

The jarvis-mode branch of `_route_text` replaces `brain.think(text)` with:
```
d = router.route(text, claude.active_project)
action -> services.claude.<action>(**args) (cancel/work_on/resume/new_project/set_model/fast_mode); reply from the manager
local  -> brain.chat(text)                  ; CommandResult(status="Thinking…", done=False)
claude -> ack = brain.local_line("Acknowledge in one short sentence that you're starting this, naming the task", d.prompt, fallback="Right away, sir.")
          claude.submit(d.prompt, project=d.project or active, parallel=d.args.get("parallel"))
          CommandResult(reply=ack, speak=True, done=False)
ask    -> CommandResult(reply=ROUTER_QUESTION, speak=True)
```
`brain.think` remains for `deploy`/`autonomous:`. `tests/test_commander.py` (37 tests) is updated for the new services (`timekeeper`, `notes`, `router`, `claude`, `approvals` mocks) and gains the ordering tests: ringing words beat everything; approval words beat the registry; skill map expansion; router decisions dispatch to the right service.

## 6. Tools

Common rules: every tool returns compact plain text (no markdown), US units, 12-hour times ("7:00 am"), relative words where natural ("today", "tomorrow"); `ok=False` text explains the failure in one clause ("weather service unreachable"); a section that is not configured returns `ToolResult(text=cfg.setup_line(section), ok=False, speak=cfg.setup_line(section))` so the excuse is spoken verbatim without a model turn.

### 6.1 Timekeeper (`jarvis/tools/timekeeper.py`)

```python
class Timekeeper:
    def __init__(self, db_path, say, cfg, now=time.time, run=_run, tick_s=1.0, ring=True)
    start()/stop()                                   # scheduler thread, 1 s tick, daemon
    add_reminder(due: float, text: str) -> Item
    add_timer(seconds: float, label: str = "") -> Item
    add_alarm(due: float, label: str = "", repeat: str = "once") -> Item   # once|daily|weekdays
    list(kind="all", include_done=False) -> list[Item]     # soonest first
    list_text(kind="all", now=None) -> str                  # "Two reminders, sir: …"
    cancel(which="last"|"all"|<id>|<label substring>, kind="all") -> int
    stop_ringing(action="dismiss") -> bool ; snooze(minutes) -> bool
    ringing -> Item | None
    catch_up(now) -> list[Item]                              # boot: fire (< 1 h late) or mark missed
    import_legacy(path) -> int                               # workflows reminders.json, once
    parse_when(text, now: datetime) -> datetime | None       # module-level pure function
    describe_due(due, now) -> str                            # "in 10 minutes" / "at 7:00 am tomorrow"
```
SQLite schema `items(id TEXT PK, kind TEXT, label TEXT, due REAL, created REAL, repeat TEXT DEFAULT '', state TEXT, snooze_until REAL, fired_at REAL, note TEXT)`; `state in pending|ringing|snoozed|done|missed|cancelled`. Scheduler: each tick selects `pending` (or `snoozed` with `snooze_until <= now`) items with `due <= now`; a reminder -> `say("Sir, this is your reminder. …")`, notify-send, `bus.publish(ReminderFired(text))`, state done; a timer -> "Sir, your N-minute timer is up.", `ReminderFired`, done; an alarm -> `say("Sir, it's 7:00 am. Time to get up.")`, `bus.publish(AlarmFired(...))`, state ringing, ringer thread: `paplay --volume=<0..65536 from cfg.alarms.volume> <cfg.alarms.sound>` in a loop with 1 s gaps; after 30 s with `escalate` true, switch to full volume and a 0.3 s gap; after `max_ring_s` (300) stop, mark missed, say the missed line. Repeat alarms re-schedule on stop (daily +24 h; weekdays skip Sat/Sun). Sound: `cfg.alarms.sound` else `/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga` else a generated 880/660 Hz WAV written once to the cache dir (stdlib `wave`). Catch-up on `start()`: pending items with `due < now`: `< 3600 s` late -> fire now with the "while I was down" preamble; else -> missed + the missed line, once, joined into a single sentence when several. `parse_when` handles: `in N (seconds|minutes|hours)`, `in an hour (and a half)`, `in half an hour`, `at 3`, `at 3 pm`, `at 6:30`, `at noon|midnight`, `tomorrow (morning|afternoon|evening|night)` (8:00/14:00/18:00/21:00), `tonight`, `this evening`, `tomorrow at 8`, `on friday at 9`, `next monday`, `every day at 7` / `every weekday at 6:30` (returns the first occurrence; the repeat comes from the tool arg), bare hour disambiguation (an hour already past today rolls to tomorrow; `at 7` with no am/pm before 12:00 means morning if the next 7 is > 10 h away — pick the NEXT occurrence). Tools per 4.1; `manage_schedule(action="stop")` and `"snooze"` map to `stop_ringing`/`snooze`. Tests (`tests/test_timekeeper.py`): parse_when table (>= 25 cases, fixed `now`), scheduler with fake clock fires/reschedules/persists across a re-open, catch-up rules at 59 min and 61 min late, snooze/dismiss/timeout, legacy import, `list_text` wording, ringer `_run` calls recorded (no audio).

### 6.2 Location and time (`jarvis/tools/location.py`)

`resolve_home(cfg, cache_path, fetch=_fetch) -> Location(city, region, country, lat, lon, tz, source)`: `cfg.home_location` when lat/lon set, else `https://ipapi.co/json/` (fallback `http://ip-api.com/json/`) cached 24 h at `~/.cache/jarvis/location.json`; `geocode(name, fetch) -> Location | None` via `https://geocoding-api.open-meteo.com/v1/search?name=<q>&count=1&language=en` (cached per name). `get_location` -> "You're in Chicago, Illinois, sir (from your network address)". `get_time(location?)` -> local time via `zoneinfo` from the location's tz ("It's 4:05 pm in Tokyo."); home time falls back to `datetime.now()`. Never blocks > 4 s.

### 6.3 Weather (`jarvis/tools/weather.py`)

Open-Meteo, no key: `https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation,relative_humidity_2m&hourly=temperature_2m,precipitation_probability,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=auto&forecast_days=7`. WMO code table -> words ("clear", "partly cloudy", "light rain", "thunderstorms", …). Cache 10 min per rounded (lat, lon). `format_weather(data, when, now) -> str`: now: "72°F and partly cloudy, feels like 75, wind 8 mph; high 85, low 64, 10% chance of rain."; today: high/low/conditions/rain chance/sunset; tomorrow: same; week: one clause per day, max 7, "Monday 88 and sunny, Tuesday 79 with showers, …". `location` argument -> geocode. Tests: canned JSON -> exact strings; cache hit; failure text.

### 6.4 Calendar (`jarvis/tools/calendar.py`)

Sources: `cfg.google_ical_urls` (Google "Secret address in iCal format"), fetched every 10 min with `If-None-Match`/`If-Modified-Since`, parsed with `icalendar` + `recurring_ical_events` (pip), and iCloud via `caldav` (`DAVClient(url="https://caldav.icloud.com", username=apple_id, password=app_password)` -> principal -> calendars -> `search(start, end, event=True, expand=True)`), refreshed every 10 min; both merged into `Event(start, end, all_day, title, calendar, location)` in local time; disk cache `~/.cache/jarvis/calendar_cache.json` with `fetched_at` so a boot without network answers with "as of 9:10 am". Refresh runs on a worker thread started by `start()`; `get_calendar(range)` never fetches synchronously (it returns the cache and triggers a refresh if stale > 10 min). `format_events(events, range, now) -> str`: "Today: 10:00 am dentist for an hour, 2:30 pm standup; nothing else." / "Nothing on tomorrow, sir." / `next` -> the first event after now with `describe_due`. All-day events first ("all day: Mum's birthday"). Tests: a fixture .ics with a weekly recurrence and an all-day event -> exact strings across ranges (fixed `now`), stale-cache wording, unconfigured excuse, iCloud path with a fake client.

### 6.5 Notes and to-dos (`jarvis/tools/notes.py`)

`NotesStore(db_path)`: tables `notes(id INTEGER PK, text, created REAL, tags TEXT)` and `todos(id INTEGER PK, text, created REAL, done INTEGER DEFAULT 0, done_at REAL)`; `add(kind, text) -> id`, `list(kind, limit=10, include_done=False)`, `search(kind, query)`, `remove(kind, which)` / `complete(which)` where `which` is `last`, an index ("the second one"), or a substring; `import_legacy(memory_notes_dir)` once (`note_*.txt` -> notes). `list_text(kind)` -> "Three to-dos, sir: buy milk, call the dentist, and fix the bike." Tool `notes(action, kind, text, which)`; spoken confirmations "Noted, sir." / "Added to your list, sir." / "Done, sir; two left." Tests: CRUD, `which` resolution, wording, legacy import.

### 6.6 Mail (`jarvis/tools/mail.py`)

`fetch_unread(cfg, since_hours=24, limit=20, imap=imaplib.IMAP4_SSL) -> list[Mail(from_name, from_addr, subject, date, snippet)]`: `LOGIN` with the app password, `SELECT INBOX` read-only, `SEARCH UNSEEN SINCE <date>`, `FETCH ... (BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] BODY.PEEK[TEXT]<0.600>)` — PEEK so nothing is marked read; decode RFC 2047 headers; snippet = first 200 chars of the text part with quoted lines and signatures dropped. `get_mail(limit=5)` -> fact sheet "5 unread since yesterday: 1) Jane Doe — Invoice 4471 due Friday (2:10 pm) …" with `max_sentences=4`; zero -> `speak="Nothing new in the inbox, sir."`. Never log the password or bodies; errors -> "I can't reach your mailbox, sir" (`ok=False`). Tests: a fake IMAP class returning canned RFC 822 bytes; header decoding; snippet cleanup; unconfigured excuse.

### 6.7 Briefing (`jarvis/tools/briefing.py`)

`build_briefing(cfg, registry, fetch, now) -> (sections: dict, fact_sheet: str)`: sections = `{"weather": str, "calendar": [str], "news": [{"title","source"}], "sports": [str], "stocks": [str]}`; weather and calendar come from `registry.call("get_weather", {"when":"today"})` / `("get_calendar", {"range":"today"})`; news = Tech & AI only: Hacker News top 3 by score from `https://hacker-news.firebaseio.com/v0/topstories.json` + `item/<id>.json`, plus the first item each from `cfg.briefing.news_feeds` (defaults: The Verge `https://www.theverge.com/rss/index.xml`, Ars Technica `https://feeds.arstechnica.com/arstechnica/index`), parsed with `xml.etree` (Atom + RSS), deduplicated by title, cut to 3 items total, each a title + source; sports/stocks only when `cfg.briefing.sports_feeds` / `stock_symbols` are non-empty (stooq CSV `https://stooq.com/q/l/?s=<sym>.us&f=sd2t2ohlcv&h&e=csv`); news cache 15 min. `get_briefing()`: disabled -> `ToolResult(text=BRIEFING_OFF_LINE, ok=False, speak=BRIEFING_OFF_LINE)`; enabled -> `ToolResult(text=fact_sheet, max_sentences=6, card=sections)`. The model renders the spoken briefing (weather, then calendar, then the three news items one sentence each, then sports/stocks if present; persona; <= 6 sentences); the brain emits `("BRIEFING", json(card))` + `("SPEAK", spoken)`; the app publishes `BriefingReady(sections, spoken)` and speaks. Tests: canned HN/RSS/CSV -> sections and fact sheet; disabled path; feed failure degrades to the sections that worked.

## 7. Claude session manager (`jarvis/claude_session.py`, `jarvis/mcp_permissions.py`, `jarvis/approvals.py`)

### 7.1 Rules (user's answers, binding)

- **Models:** default `opus` (`--model opus`); escalate to `fable` (`--model fable`) when the user says "use fable" / "think hard" / "this is a big one" or when the router's size estimate is large (multi-file refactor, new feature, debugging across modules — `estimate_size(prompt) -> "small"|"large"`, rules on the same cue tables as 5.1: two or more coding objects, "refactor", "feature", "across", "all the", "migrate", "debug … and …"). Voice overrides `use opus|sonnet|fable|haiku` stick for the session until changed. `fast mode on|off` -> `--settings '{"fastMode": true}'` (unknown keys are ignored by the CLI; log once if the CLI rejects it).
- **Skills/plugins:** all user-scope plugins are available inside `claude -p` automatically; the phrase map (10.1) is config; unknown skill names pass through as `/<name> <args>`.
- **Resume:** discovery from `~/.claude/projects/<encoded-cwd>/*.jsonl` (7.5).
- **Progress:** milestones only (7.4). **Permissions:** 7.3. **Concurrency:** one task per project, queue; a second project in parallel only when asked explicitly; `cancel`/`stop that` aborts the active task.

### 7.2 Manager interface

```python
@dataclass
class Project: slug: str; path: str; session_id: str = ""; model: str = ""; last_used: float = 0
@dataclass
class Task: task_id: str; project: str; prompt: str; model: str; state: str; started: float; files_touched: set; result_text: str = ""; rc: int | None = None

class ClaudeSessionManager:
    def __init__(self, cfg, brain, approvals, state_path, task_dir, run=_run, claude_bin=MACHINE.claude_bin, python=sys.executable)
    projects() -> list[Project]; active_project -> str | None; project_for(slug_or_name) -> Project | None
    submit(prompt, project=None, parallel=False, model=None) -> Task | str      # str = a spoken refusal/queue line
    cancel(project=None) -> bool
    work_on(name) -> str                      # switches active project; unknown -> "I don't know a project called X, sir; shall I set one up?"
    new_project(name) -> str                  # 7.6
    resume(utterance) -> str                  # 7.5; may return the ONE question naming two candidates
    set_model(alias) -> str ; set_fast_mode(on: bool) -> str
    open_terminal(slug=None) -> bool          # 7.7
    ensure_project_settings(project) -> Path  # 7.3 settings.local.json
    status_text() -> str                      # "Claude's working on the Jarvis router, sir; started four minutes ago."
```
Prompt files and streams live in `task_dir/<slug>/<task_id>.*`. `submit` publishes `ClaudeTaskState(queued|running)`; a task ends with `done|failed|cancelled` and `ClaudeProgress(milestone=True)` lines; the app speaks milestones and the final summary (`brain.summarize(result_text, 2)` when longer than two sentences).

**Runner (per task, inside tmux):**
```
tmux has-session -t jarvis-<slug> || tmux new-session -d -s jarvis-<slug> -c <path> -x 220 -y 50
tmux send-keys -t jarvis-<slug> "clear; cd <path> && <claude> -p --output-format stream-json --verbose --model <model> [--resume <session_id>] [--effort <lvl>] [--settings '{\"fastMode\":true}'] --permission-mode <cfg.claude.permission_mode|acceptEdits> --mcp-config <PATHS.MCP_CONFIG> --permission-prompt-tool mcp__jarvis__approve --append-system-prompt-file <task_dir>/system_suffix.txt < <task>.prompt 2>> <task>.stderr | <python> -m jarvis.claude_session render --out <task>.jsonl; echo \${PIPESTATUS[0]} > <task>.rc" Enter
```
The `render` subcommand copies each raw stream-json line to `--out` (flush per line) and prints a readable line per event to the pane (the human-facing terminal), so the pop-out shows what Claude is doing while Jarvis tails the file (poll 0.25 s). Completion = the `result` event, or the `.rc` file, whichever first; liveness = `tmux list-panes -t jarvis-<slug> -F '#{pane_current_command}'`. First task in a project runs without `--resume`; the `init` event's `session_id` is stored in `claude_projects.json` and every later task uses `--resume <id>` (never bare `--continue`, which would hijack the user's own terminal session in that dir); a resume failure ("No conversation found" in stderr, rc != 0, no events) retries once without `--resume`. `system_suffix.txt`: "You are being driven by Jarvis, Hunter's voice assistant. Your final message is read aloud: end with one or two plain-prose sentences saying what you did and anything he must decide." `cancel`: `tmux send-keys -t jarvis-<slug> C-c`, wait up to 5 s for `.rc`, else `kill -INT` the pane's child processes (`tmux list-panes -F '#{pane_pid}'` + `pgrep -P`), then `ClaudeTaskState(cancelled)` and "Stopped, sir." If `claude` is missing (`MACHINE.claude_bin` empty) every submit returns the setup line for `claude`.

### 7.3 Permissions

Inside `cfg.claude.allowed_dirs` everything is allowed via per-project `.claude/settings.local.json` written by `ensure_project_settings` before each task (merge: keep every existing key, union the `allow` list, never remove user entries):
```json
{"permissions": {"allow": ["Bash(*)", "Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Glob", "Grep", "WebFetch", "WebSearch", "Task", "TodoWrite", "mcp__*"],
                 "deny": ["Bash(sudo *)", "Bash(rm -rf /*)", "Bash(rm -rf ~*)"]}}
```
plus `--permission-mode acceptEdits`. The global `--dangerously-skip-permissions` is used ONLY when `cfg.claude.dangerously_skip_permissions` is true (default false). Anything Claude Code still asks about (paths outside the project, tools not in the list) goes to the permission-prompt tool: `PATHS.MCP_CONFIG` = `{"mcpServers": {"jarvis": {"command": "<python>", "args": ["-m", "jarvis.mcp_permissions"], "env": {"JARVIS_APPROVAL_SOCK": "<sock>", "JARVIS_PROJECT": "<slug>"}}}}` (written per task with the slug), `--permission-prompt-tool mcp__jarvis__approve` (the flag exists in 2.1.246 but is hidden from `--help`; verify on the first real run and adapt the tool name if the CLI reports it differently).

`jarvis/mcp_permissions.py` — a dependency-free MCP stdio server (JSON-RPC 2.0, one JSON object per line on stdin/stdout, protocol version echoed from the client's `initialize`): handles `initialize` (capabilities `{"tools": {}}`, serverInfo `jarvis-permissions`), `notifications/initialized`, `ping`, `tools/list` (one tool `approve`, inputSchema `{tool_name: string, input: object, tool_use_id?: string}`), `tools/call` -> connects to the UNIX socket, sends `{"id", "tool_name", "input", "project", "cwd"}\n`, waits up to 125 s for `{"id", "behavior": "allow"|"deny", "message"}`, returns `{"content": [{"type": "text", "text": json.dumps({"behavior": "allow", "updatedInput": input})}]}` or `{"behavior": "deny", "message": "..."}`; socket missing -> deny "Jarvis is not running". Nothing else on stdout (logs to stderr). Unit test: drive `initialize`/`tools/list`/`tools/call` through pipes against a fake socket server.

`jarvis/approvals.py`:
```python
@dataclass
class ApprovalRequest: request_id; tool_name; input: dict; project; cwd; created: float; question: str; detail: str
class ApprovalBroker:
    def __init__(self, sock_path, cfg, timeout_s=120.0, on_request=None, on_resolved=None)
    start()/stop()                                 # SOCK_STREAM server thread; one thread per client
    pending() -> list[ApprovalRequest]             # oldest first
    answer(allowed: bool, request_id=None, source="typed") -> bool
    auto_policy(req) -> "allow" | None             # every absolute path in input under an allowed dir and tool in {Read,Glob,Grep,Edit,Write,MultiEdit,NotebookEdit} -> allow (source "policy"), else None
describe_request(req) -> str   # "Claude wants to run “git push origin main”, sir; shall I allow it?" / "… edit ~/.bashrc, sir, which is outside the project; allow it?" / "… fetch example.com, sir; allow it?"
```
On a request the broker publishes `ApprovalRequested` and calls `on_request` (the app speaks the question, marks the task `waiting`, fires the away alert); on `answer` it replies over the socket, publishes `ApprovalResolved`, task back to `running`; after `timeout_s` it denies with the timeout line. Answers come from typed/voice (commander 5.2), the UI buttons, or Discord.

### 7.4 Progress and milestones (`parse_stream_event(event, task) -> list[ClaudeProgress]`, pure)

Stream-json shapes (Claude Code 2.x; the parser ignores unknown types and never raises on missing keys): `{"type":"system","subtype":"init","session_id","cwd","model",…}`; `{"type":"assistant","message":{"content":[{"type":"text","text"} | {"type":"tool_use","id","name","input"}]}}`; `{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id","content","is_error"}]}}`; `{"type":"result","subtype":"success"|"error_*","is_error","result","session_id","num_turns","duration_ms"}`. Every event becomes ONE compact line (`Read jarvis/router.py`, `Bash pytest -q tests`, `Edit jarvis/brain.py`, `Search "def route"`, a 90-char text excerpt for assistant prose, `Error: …`); a line is a milestone (spoken) when: start (spoken once by the commander's acknowledgement, not here); the first Edit/Write per file, coalesced as "Editing router.py and brain.py, sir" at most once per 20 s; a Bash command matching `pytest|npm test|cargo test|go test|make test|ruff|mypy` -> "Running the tests"; its tool_result -> "Tests passed, sir" / "N tests failed" (regex on `passed|failed|error`); an `ExitPlanMode` tool_use or assistant text starting with a plan heading -> "Plan chosen: <first sentence>"; three consecutive `is_error` results -> "Hitting errors, sir; carrying on"; the `result` event -> done (summary spoken by the app) or failed. Rate limit: <= 1 spoken milestone per 20 s except start/done/blocked/question. The first real stream from the live check is saved to `tests/fixtures/claude_stream_sample.jsonl` (secrets-free) and drives the parser test.

### 7.5 Session discovery (resume)

`discover_sessions(projects_dir=~/.claude/projects, limit=40) -> list[SessionInfo(session_id, cwd, mtime, first_user, last_assistant, turns, path)]`: walk `*.jsonl`, take `cwd`/`sessionId` from any line, `first_user` = first `type=="user"` text content that is not meta (skip lines starting with `<`, lines starting with "You are Jarvis" — Jarvis's own Tier-3 prompts — and sessions with <= 2 turns), `last_assistant` = last assistant text; skip sessions whose cwd no longer exists. `pick_session(utterance, sessions, now) -> (best, runner_up, question)`: score = name-token overlap between the utterance and the cwd's last two path components (+0.6), date words (`yesterday` -> sessions modified that calendar day +0.4, `today`, `last week`), recency prior (+0.3 · e^(-age/2 days)), `first_user` keyword overlap (+0.2); ties (|Δ| < 0.15) -> one persona question naming both by project + one-line summary ("Two candidates, sir: the VSS labeler from yesterday, or the haymaker digest — which?"); the answer resolves by name. Resuming = `work_on(project of cwd)` (adding the cwd to `allowed_dirs` only if it is under `~`, after confirming when it is not already allowed) and `submit("Pick up where we left off; summarise the state in two sentences first, then continue.", model=…)` with `--resume <session_id>`. The acknowledgement is generated ("Right away, sir — picking up the VSS labeler where we left off."), never hardcoded.

### 7.6 New projects

`new_project(name)`: slug = lowercase, `[a-z0-9-]`, from the spoken name; path = `cfg.claude.projects_root / slug` (default `~/projects`), refuse anything outside the root; `mkdir -p`, `git init`, `CLAUDE.md` scaffold (project name, "Python 3 / ~/vss_env unless told otherwise", "tests under tests/, run with pytest -q", "Jarvis drives this repo; keep replies short"), `README.md`; add to `allowed_dirs` (saves `assistant.json`), `ensure_project_settings`, set active, publish `ActiveProject`, return "The X project is set up, sir; ready when you are."

### 7.7 Pop-out terminal

`open_terminal(slug=None)`: slug = active project or the first allowed dir's slug; ensure the tmux session exists (`new-session -d -c <path>`); if a window titled `Jarvis · <slug>` exists (`xdotool search --name "^Jarvis · <slug>$"`) -> `xdotool windowactivate` and return True; else `gnome-terminal --title "Jarvis · <slug>" -- tmux attach -t jarvis-<slug>` (env `DISPLAY` inherited). Tests record `_run` argv.

## 8. Channels (`jarvis/channels/notify.py`, `jarvis/channels/discord.py`)

### 8.1 Alerts hub

```python
class Alerts:
    def __init__(self, cfg, run=_run): ...; attach(discord: DiscordChannel | None)
    def alert(kind, title, text, request_id=None)   # kind: milestone | done | blocked | question | alarm | reminder
```
`alert` sends `notify-send -a Jarvis -u <normal|critical for question/blocked/alarm> -t <8000|0 for question> "<title>" "<text>"` and, when Discord is configured, posts `**<title>** — <text>` (questions end with "Reply yes or no."). The app calls it on `ClaudeTaskState(done|failed)`, `ApprovalRequested`, `AlarmFired`, `ReminderFired` and spoken milestones. Never blocks the caller (worker thread, 5 s cap).

### 8.2 Discord (two-way)

`DiscordChannel(cfg, on_message, transport=None)` with `cfg.discord.{bot_token, channel_id, user_id}`: `start()` -> gateway thread: `GET https://discord.com/api/v10/gateway/bot` (Authorization `Bot <token>`), connect `wss://…?v=10&encoding=json` with `websockets`, HELLO -> heartbeat loop, IDENTIFY with intents `GUILDS | GUILD_MESSAGES | MESSAGE_CONTENT | DIRECT_MESSAGES` (1<<0 | 1<<9 | 1<<15 | 1<<12), handle `MESSAGE_CREATE` where `channel_id == cfg.channel_id` (or a DM from `cfg.user_id`) and `author.id != bot id` -> `on_message(text, author_id)`; RESUME on disconnect with backoff 1-60 s; after 3 failed gateway attempts fall back to REST polling `GET /channels/<id>/messages?after=<last_id>&limit=20` every 5 s. `post(text)` -> `POST /channels/<id>/messages` (2000-char chunks, 429 -> wait `retry_after`). Never log the token (`redacted()` everywhere); unconfigured -> `start()` is a no-op and `configured` is False. The app's `on_message`: if `approvals.pending()` and the text is a yes/no word -> `approvals.answer(..., source="discord")`; else `dispatch_text(text, source="discord")`, and while a Discord exchange is active (10 min window) every `JarvisReply` is also posted back. Tests: a fake transport (scripted gateway frames, recorded REST calls): identify payload, heartbeat interval, message filter, reconnection, polling fallback, chunking, token never in any log record.

## 9. UI (`jarvis/ui/*`, UI item only)

All geometry in design units per the clean spec (`px()`, `theme.PAD`/`PAD_S`, one annotation size `SIZE_CAPTION`, labels display/MUTED, values mono/FOCAL).

1. **Terminal button next to the mic** (`views.CommandBar`): a second 44-px round canvas button packed `side="right"` after the mic (so it sits to the LEFT of the mic), `padx=(0, PAD_S)`; drawn like the mic (4-arc CYAN ring, RAISED disc) with a terminal glyph: a chamfered 18x14 rectangle outline, a `>` chevron and a 6-px underscore, glyph colour CYAN idle, FOCAL while working. States via `set_terminal_state(state, tooltip)`: `idle` (ring gapped), `working` (ring closed, glyph FOCAL), `waiting` (ring WARN), `disabled` (FAINT, when tmux/gnome-terminal are missing). Tooltip "Open Claude's terminal — <project>" / "No project yet". Click -> `services.open_terminal()`. The bar must still fit at 460 design px (field >= 300).
2. **Header pill:** new words `WORKING` and `WAITING` (<= 10 chars), colours `STATE_COLORS["working"] = CYAN_DIM dot / FOCAL word`, `["waiting"] = WARN dot / FOCAL word`; precedence speaking > listening > thinking > **waiting > working** > error > idle (`resolve_state` gains two booleans; `Reactor.state()` is NOT changed — the stage keeps idle/listen/think/speak). Driven by `ClaudeTaskState` (running -> working, waiting -> waiting, done/failed/cancelled -> clear; keep a set of running task ids).
3. **Status-bar project chip:** `StatusStrip` gains a `PROJECT` segment placed right after the wake-word segment (`| PROJECT JARVIS |`), shown ONLY when `set_project(slug)` is non-empty (idle bar unchanged), value = slug uppercased, ellipsized to fit: the four segments must fit at 460 design px (value budget shrinks first, min 6 chars). Driven by `ActiveProject`.
4. **Progress lines:** `TranscriptView.add_progress(line)`: a single compact card (mono `SIZE_CAPTION`, MUTED, no head row, fill SURFACE, left at PAD, width like a JARVIS card); consecutive progress lines append to the same card (max 12 visible; older lines collapse into a first line "… 14 earlier steps"); a new user/JARVIS card ends the run. Driven by `ClaudeProgress` (all lines, milestone or not).
5. **Approval card:** on `ApprovalRequested` add a JARVIS card with the question and two `RoundButton`s `ALLOW` / `DENY` (`services.approval_answer(request_id, bool)`); on `ApprovalResolved` disable the buttons and append `· allowed` / `· declined` to the head row. One card per request id.
6. **Alarm modal:** on `AlarmFired` show an overlay `Card` centred on the stage (300x140: label display `SIZE_LABEL` semibold, time mono `SIZE_BODY`, buttons `DISMISS` and `SNOOZE 10`), deiconify + raise the window (tray case), `services.alarm_action(alarm_id, "dismiss"|"snooze", 10)`; hide on `AlarmStopped`. The reactor keeps animating beneath; no other text on the stage.
7. **Briefing card:** on `BriefingReady` add ONE JARVIS card (`add_briefing(sections)`): head row `JARVIS  HH:MM`, body rows `WEATHER <text>`, `CALENDAR <lines>`, `NEWS <title — source>` ×3, optional `SPORTS`/`STOCKS`; labels display MUTED, values body font INK; the app does NOT publish a separate `JarvisReply` for that turn, so the datum appears once.
8. **Settings drawer:** section "Assistant" with a toggle "Morning briefing" (`services.get_option("briefing.enabled")` / `set_option`), a read-only line "Config: ~/.config/jarvis/assistant.json", a button "Open Claude's terminal", and a toggle "Start at login" (`services.get_option("autostart.enabled")`/`set_option` — the app installs/uninstalls the entry).
9. **Engine card THINK row** shows the configured model (`fmt_llm("gemma4:26b") == "GEMMA4:26B"`; `_probe_llm` reads `brain.OLLAMA_MODEL` after `configure`).
10. `Services` gains `open_terminal`, `alarm_action`, `approval_answer`, `get_option`, `set_option` (all optional, `_noop` default). Tk-free helpers for tests: `resolve_state` precedence, `fmt_project_chip(slug, budget)`, `progress_card_lines(lines, max_visible)`, `briefing_rows(sections)`, terminal-state mapping from `ClaudeTaskState` sequences (`tests/test_ui_assistant.py`).
11. Verification with the 2 restarts + `shot.py`: idle screenshot text census unchanged except the terminal button; a typed "what's the weather" exchange; a typed "timer for 1 minute" with the AlarmFired/ReminderFired toast; the terminal button pops a gnome-terminal attached to `jarvis-<slug>` (screenshot), a second click raises it (no second window); a Claude task (section 12 budget) shows WORKING, progress lines, and the summary card.

## 10. Config and setup (`jarvis/assistant_config.py`, `jarvis/autostart.py`, `docs/assistant-setup.md`)

### 10.1 Schema (`~/.config/jarvis/assistant.json`, created with placeholders, `chmod 600`)

```json
{
  "version": 1,
  "user": {"name": "Hunter"},
  "units": "us",
  "local_model": "gemma4:26b",
  "home_location": {"city": "", "region": "", "lat": null, "lon": null},
  "location_lookup": true,
  "google_ical_urls": [],
  "icloud": {"apple_id": "", "app_password": "", "url": "https://caldav.icloud.com"},
  "gmail": {"address": "", "app_password": "", "imap_host": "imap.gmail.com"},
  "claude": {
    "allowed_dirs": ["/home/hunterp/Jarvis", "/home/hunterp/haymaker-digest"],
    "projects_root": "/home/hunterp/projects",
    "permission_mode": "acceptEdits",
    "dangerously_skip_permissions": false,
    "model": "opus", "big_model": "fable", "fast_mode": false, "effort": "",
    "skill_phrases": {
      "^review (this|my|the) code$": "/code-review",
      "^commit (this|it|that)$": "/commit",
      "^simplify (this|it|that)$": "/simplify",
      "^security review$": "/security-review",
      "^run a ralph loop on (.+)$": "/ralph-loop $1",
      "^plan a feature (.+)$": "/feature-dev $1"
    }
  },
  "briefing": {"enabled": false, "hn_items": 3,
               "news_feeds": ["https://www.theverge.com/rss/index.xml", "https://feeds.arstechnica.com/arstechnica/index"],
               "sports_feeds": [], "stock_symbols": []},
  "alarms": {"sound": "", "volume": 0.8, "escalate": true, "max_ring_s": 300, "snooze_min": 10},
  "discord": {"bot_token": "", "channel_id": "", "user_id": ""},
  "autostart": {"enabled": false}
}
```

### 10.2 API

```python
class AssistantConfig:
    @classmethod load(path=None) -> AssistantConfig      # path or env JARVIS_ASSISTANT_CONFIG or the default; creates the file (0600) from DEFAULTS when missing; a corrupt file is renamed .bad and recreated; never raises
    get(dotted, default=None) ; set(dotted, value) -> saves atomically, 0600 ; save() ; reload_if_changed()
    is_configured(section) -> bool     # home_location | google_ical | icloud | gmail | discord | claude
    setup_line(section) -> str         # persona: "I'll need your Google calendar link set up, sir; the notes are in docs/assistant-setup.md."
    redacted() -> dict                 # secrets masked as "•••" (SECRET_KEYS = icloud.app_password, gmail.app_password, discord.bot_token)
    home_location -> dict | None ; allowed_dirs -> list[str] ; local_model -> str ; skill_phrases -> dict
    is_allowed_path(path) -> bool      # realpath under an allowed dir
    add_allowed_dir(path)
```
`jarvis/autostart.py`: `install(exec_cmd=None) -> Path` (writes `~/.config/autostart/jarvis.desktop`: `Type=Application`, `Name=Jarvis`, `Exec=/home/hunterp/vss_env/bin/python -m jarvis.app`, `Path=/home/hunterp/Jarvis`, `X-GNOME-Autostart-enabled=true`, `X-GNOME-Autostart-Delay=15`; idempotent), `uninstall()`, `is_installed()`, `disable_gnome_suspend(run=_run) -> bool` (`gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type`; set `'nothing'` only if it is not already — it is `'nothing'` today), CLI `python -m jarvis.autostart --install|--uninstall|--status`. The wiring item adds `--install-autostart` to `jarvis.app` and calls `install()` on first run when `autostart.enabled` is true.

### 10.3 `docs/assistant-setup.md`

Step-by-step, placeholders only: creating the file (or first run), home location (or leave blank for IP lookup), Google secret iCal URL (Calendar settings > Integrate calendar > "Secret address in iCal format"), iCloud app-specific password (appleid.apple.com > Sign-In and Security > App-Specific Passwords) and CalDAV, Gmail app password (Google Account > Security > 2-Step Verification > App passwords; IMAP enabled), Discord bot (Developer Portal > Bot > Reset Token; enable MESSAGE CONTENT intent; invite URL with `bot` scope + Send Messages/Read Message History; Developer Mode to copy the channel id), briefing toggle and feeds, alarms (sound file, volume, escalate), Claude (allowed dirs, projects root, models, fast mode, skill phrases, the permission model and the outside-dir approval flow), autostart + never-sleep, and a "what Jarvis says when something is missing" table.

## 11. Wiring (`jarvis/app.py`, `jarvis/config.py`, `jarvis/events.py`, `tests/conftest.py`, `docs/capabilities.md`)

`JarvisApp.__init__` constructs in this order: `assistant = AssistantConfig.load()`; `brain.configure(assistant.local_model)`; `tools = ToolRegistry()`; `timekeeper = Timekeeper(PATHS.TIMEKEEPER_DB, say=self._say, cfg=assistant)`; `notes = NotesStore(PATHS.NOTES_DB)`; `approvals = ApprovalBroker(PATHS.APPROVALS_SOCK, assistant, on_request=self._on_approval, on_resolved=self._on_approval_done)`; `claude = ClaudeSessionManager(assistant, brain, approvals, PATHS.CLAUDE_PROJECTS, PATHS.CLAUDE_TASK_DIR)`; `router = Router(assistant, classify=brain.classify_route)`; `alerts = Alerts(assistant)`; `discord = DiscordChannel(assistant, on_message=self._on_discord)`; `alerts.attach(discord)`; then `tools.register_many(m.make_tools(assistant, services))` for every tool module (import failures log and skip, never abort boot); `brain.set_registry(tools)`. `start_background` starts `timekeeper` (with `catch_up`), `approvals`, `discord`, calendar refresh, `brain.ensure_resident()`; `quit` stops them. Subscriptions: `ClaudeProgress(milestone)` -> `_say`; `ClaudeTaskState(done)` -> summary via `brain.summarize` -> `JarvisReply` + `_say` + `alerts.alert("done", …)`; `failed` -> "blocked" alert; `ApprovalRequested` -> speak + `alerts.alert("question", …)`; `AlarmFired`/`ReminderFired` -> alerts. `_on_brain_tags` handles `("BRIEFING", json)` by publishing `BriefingReady` and suppressing the `JarvisReply` for that SPEAK. `Services` gets `open_terminal=claude.open_terminal`, `alarm_action`, `approval_answer`, `get_option`, `set_option`. `main()` handles `--install-autostart`. `config.PATHS` gains the entries in 3.2; `conftest.py` sets `JARVIS_ASSISTANT_CONFIG` to a tmp path and asserts the new PATHS are not live. `docs/capabilities.md` gains one row per capability in this spec (invocation phrasings, module, status, notes incl. the measured latency numbers). Integration tests (`tests/test_app_wiring.py`): construct the services namespace with the real modules on tmp paths and mocks for TTS/Ollama/tmux; one utterance of each route ends in the expected service call; a fake approval round trip through the real socket + `mcp_permissions` in a subprocess; a timekeeper timer fires through the bus; `_on_brain_tags` BRIEFING path.

## 12. Work items (exclusive ownership, order, budgets)

`order` is the execution wave: 1 first (config, everyone imports it), 2 in parallel, 3 wiring, 4 UI last (it owns the only two restarts and the live verification).

| # | item | order | owns (exclusive) | depends on (interfaces) | Claude calls | restarts |
|---|---|---|---|---|---|---|
| W1 | Assistant config, autostart, setup doc | 1 | `jarvis/assistant_config.py`, `jarvis/autostart.py`, `docs/assistant-setup.md`, `tests/test_assistant_config.py`, `tests/test_autostart.py` | — | 0 | 0 |
| W2 | Timekeeper (reminders, timers, alarms) | 2 | `jarvis/tools/timekeeper.py`, `tests/test_timekeeper.py` | W1 (`AssistantConfig` API 10.2), registry (4.1) | 0 | 0 |
| W3 | Location, weather, calendar tools | 2 | `jarvis/tools/location.py`, `jarvis/tools/weather.py`, `jarvis/tools/calendar.py`, `tests/test_weather_location.py`, `tests/test_calendar_tool.py` | W1, registry | 0 | 0 |
| W4 | Notes, mail, briefing tools | 2 | `jarvis/tools/notes.py`, `jarvis/tools/mail.py`, `jarvis/tools/briefing.py`, `tests/test_notes_mail.py`, `tests/test_briefing.py` | W1, registry, W3 tool names (called through the registry, mocked) | 0 | 0 |
| W5 | Brain: gemma4 tool loop, residency, latency, persona port | 2 | `jarvis/brain.py`, `jarvis/tools/registry.py`, `tests/test_persona.py`, `tests/test_brain_tools.py`, scratchpad `persona_eval.py`, `bench_resident.py`, `persona_gemma4.md` | W1 (`local_model`), registry | 0 | 0 |
| W6 | Router + commander | 2 | `jarvis/router.py`, `jarvis/commander.py`, `tests/test_router.py`, `tests/test_commander.py` | W1 (skill phrases, allowed dirs), W5 (`classify_route`, `chat`, `local_line`), W7 (`ClaudeSessionManager` methods), W2 (`Timekeeper` API), W4 (`NotesStore`), W7 (`ApprovalBroker.pending/answer`) — all mocked in tests | 0 | 0 |
| W7 | Claude session manager, MCP permission server, approvals, discovery, projects, terminal pop-out | 2 | `jarvis/claude_session.py`, `jarvis/mcp_permissions.py`, `jarvis/approvals.py`, `tests/test_claude_session.py`, `tests/test_mcp_permissions.py`, `tests/fixtures/claude_stream_sample.jsonl` | W1, W5 (`summarize`, `local_line` — mocked) | **2** (one harmless in-project smoke in a temp project under `~/projects` or the scratchpad: "Reply with the single word OK"; one outside-dir Write that must trigger the approval round trip, answered by the test harness) | 0 |
| W8 | Channels: alerts hub + Discord | 2 | `jarvis/channels/__init__.py`, `jarvis/channels/notify.py`, `jarvis/channels/discord.py`, `tests/test_channels.py` | W1, W7 (`ApprovalBroker.answer` — mocked) | 0 | 0 |
| W9 | UI | 4 | `jarvis/ui/*` (all), `tests/test_ui_assistant.py`, `tests/test_hud_format.py` | events (seeded), `Services` extension (its own), W10 wiring for live checks | **1** (the live Claude task in the final verification; 1 spare stays unused) | **2** |
| W10 | Wiring, config paths, capabilities doc, integration tests | 3 | `jarvis/app.py`, `jarvis/config.py`, `jarvis/events.py`, `tests/conftest.py`, `tests/test_app_wiring.py`, `docs/capabilities.md` | everything above (real modules on tmp paths) | 0 | 0 |

Interfaces each pair must agree on are the ones written in sections 3-10; where a dependency has not landed yet an item codes against the spec and mocks it in tests. Nobody edits a file outside their row. `jarvis/workflows.py`, `jarvis/jarvis_agent.py`, `jarvis/memory.py`, `jarvis/context.py` and the voice modules are not edited by anyone.

## 13. Acceptance (judges)

1. Tests: full suite green (272 existing + new), `JARVIS_LIVE=1` tests pass against Open-Meteo/HN/Ollama when run by the judge.
2. Latency (W5, model resident alone): p50 <= 1.0 s, p90 <= 2.0 s for simple answers; tool round trip p50 <= 3.0 s; `ollama ps` shows only gemma4 after boot; `prompt_eval_count` on the second identical-prefix call is < 300 tokens (cache hit).
3. Persona: `persona_gemma4.md` — 14 base prompts, zero PARROT/RIDDLE/FAKE-ACTION, "sir" in >= 85 %, every reply <= 2 sentences; tool prompts 12/12 (the bench set).
4. Router: the 60-utterance table in `tests/test_router.py` (20 local, 20 claude, 10 actions, 10 ambiguous) routes as labelled without the model; ambiguous ones ask exactly one question.
5. Timekeeper: a 1-minute timer set by typing fires with speech + toast; an alarm set for +2 min rings via paplay, the modal shows, DISMISS stops it; an alarm left pending across an app restart fires with the "while I was down" line when < 1 h late.
6. Claude: the smoke task shows WORKING in the pill, progress lines in the transcript, a spoken summary; the outside-dir Write triggers the spoken approval question, a typed "yes" allows it and the file appears; `cancel` stops a running task within 5 s; the terminal button opens exactly one gnome-terminal attached to the right tmux session and re-clicking raises it.
7. Config: `assistant.json` created 0600 with placeholders; every feature with placeholders missing answers with its setup line; no secret string appears in `jarvis.log` (grep for the placeholder markers and for "token"/"password" values).
8. UI census (idle): the clean-spec text list plus the terminal button glyph and nothing else; with an active project the PROJECT chip appears once; after a briefing exactly one card carries it.
9. Docs: `docs/assistant-setup.md` and `docs/capabilities.md` describe what shipped, including measured numbers.
