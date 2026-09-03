# Screen control on the Spark — a design, not a build

His words: *"lets do screen control on the spark for jarvis. that would be a hard
but useful one."*

**Nothing here is implemented.** Everything marked *measured* was run on his box
on 2026-09-03 against the live desktop, read-only, without touching the running
Jarvis. One experiment changed a setting and put it back; it is flagged where it
appears.

---

## The short answer

Jarvis can already move the mouse and the keyboard. What is missing is the other
half — deciding **where** — and that is the whole difficulty, exactly as he
guessed.

Measured today, there is a much better answer to "how does it know what to
click" than pixels: **the accessibility bus is already running on this box and
hands back real widget names.** 175 named, on-screen, actionable widgets across
the whole desktop in 2.93 s. No install, no sudo, no setting change, no vision
model. `push button: 'New Tab'`, `list item: 'Bluetooth'`, `table cell:
'JARVIS-TESTING.md'` — the actual labels, with a click action attached.

And the honest other half: **Brave reports exactly 0 widgets**, and Brave is
where he spends most of his time. The browser gap is real and nothing cheap
closes it.

So the shape of the feature is: **sharp on GNOME/GTK apps, blind in the browser.**
That is still worth building — the GTK half is roughly one evening for something
genuinely useful — provided it says so out loud when the target is a browser
instead of guessing.

---

## 1. What is actually on the box — measured 2026-09-03

| | |
|---|---|
| Session | **X11**, `DISPLAY=:1`, `Type=x11`, `XDG_SESSION_TYPE=x11`, active |
| Screen | one output, HDMI-0, **3840×2160**, `Xft.dpi: 192` — i.e. a 2× HiDPI desktop, logically 1920×1080 |
| Present | `xdotool` (3.20160805.1), `xclip`, `gnome-screenshot`, `xprop`, `xwininfo`, `xrandr`, `notify-send`, `gdbus`, `busctl`, `dbus-send`, `gsettings` |
| Missing | `wmctrl`, `scrot`, `import`, `ydotool`, **`tesseract`**, `accerciser` |
| In `~/vss_env` | `python-xlib` 0.33, `pynput` 1.8.2, `evdev` 1.9.3, `opencv-python`, **`pytesseract` 0.3.13**, `PIL` 12.0.0 |
| Not in `~/vss_env` | **`gi` / PyGObject** — `pyvenv.cfg` says `include-system-site-packages = false` |

Three corrections to the earlier note, all measured:

* **`pytesseract` is installed but the `tesseract` binary is not.** The Python
  wrapper is a shell-out; without the binary it is dead weight. Local OCR
  therefore needs an `apt install tesseract-ocr` → **no sudo → blocker to name,
  not to work around.**
* **`import` is missing but capture still works.** `PIL.ImageGrab.grab(xdisplay=':1')`
  returns 3840×2160 in **0.05 s**; downscale to 1280 wide + JPEG q80 = 142 KB,
  **0.10 s total**. `jarvis/tools/screen.py` already does exactly this, and its
  own comment forbids `gnome-screenshot` (it drives the Shell's UI).
* **`ydotool` being missing does not matter.** `python-xlib` in the venv reports
  `XTEST extension present: True` against `:1`, so synthetic input is available
  in-process without any new binary. `xdotool` is already used and is fine;
  this is just a note that the missing tool is not a blocker.

---

## 2. What Jarvis can do today

`jarvis/desktop.py` (659 lines) is further along than "some helpers":

| Already there | What it does |
|---|---|
| `press_key(keys)` | any xdotool chord, `--clearmodifiers` |
| `parse_desktop_action` / `parse_command` | chained speech → actions: tabs, scroll, click, minimise/maximise/fullscreen/close, media keys, copy/paste/undo/redo/save/select-all/find, `wait N` |
| `DesktopControl.execute` | runs a chain, with the xdotool pacing sleeps |
| `list_windows()` | every visible window id + title, **batched** into one subprocess |
| `target_by_query` / `pin_target` / `restore_target` | "type into *that* window", with a 3/2/1 scoring match on the title |
| `type_text` / `type_dictation` / `live_type_partial` | typing into a pinned window, a found Claude terminal, or the active one |
| `find_claude_terminal` | window whose title starts with a spinner glyph |
| `launch_app` | `Popen`, then `gtk-launch` |
| `screenshot()` | PIL grab, keeps the last 20, hands it to the Claude terminal |

`jarvis/tools/screen.py` (597 lines) is a mature **read** path: grab → 1280 px →
JPEG → Ollama `/api/chat`, `think: false`, `keep_alive: -1` on the resident chat
model, capability probing, per-model "this build can't load it" memory, spoken
answer capped at three sentences.

**So the primitives exist. What is missing is everything between "he said a
sentence" and "I know which pixel that is":**

| Missing | Why it matters |
|---|---|
| Click **a point** | `("click","left")` clicks *wherever the pointer already is*. There is no `click_at(x, y)` anywhere. |
| **Drag** | not present at all — no `mousedown`/`mousemove`/`mouseup` sequence |
| **Move the pointer** | no `xdotool mousemove`, so nothing can aim a click |
| Type into an **arbitrary** app | `type_text` is built around the Claude terminal and a pinned target; there is no "type this into the app called X" |
| **Focus** a window by app rather than title text | `target_by_query` matches title substrings only; a window with a document-name title (`Good evening Ali and Heather.txt (~/) - Text Editor`) is hard to name out loud |
| Read the screen **to decide what to click** | `screen_qa` answers a question in prose. It returns no target, no coordinate, no widget. It is eyes with no hands attached. |
| Know whether an action **worked** | nothing observes the result of any action |
| **Stop** a chain that is already running | `execute()` is a `for` loop with sleeps and no abort check |

That last pair is the real gap. Everything else is an afternoon of `xdotool`
wrappers.

---

## 3. The hard part: knowing what to click

Four routes. They are not alternatives so much as layers, and the honest answer
is that the best one is *not* the one everybody reaches for.

### Route A — the accessibility tree (AT-SPI). **Recommended primary.**

Measured, right now, with nothing installed and nothing changed:

```
gnome-terminal-server   21 actionable widgets   0.04 s
gnome-control-center    40 actionable widgets   0.61 s
gnome-text-editor       32 actionable widgets   0.63 s
org.gnome.Nautilus      63 actionable widgets   1.25 s
gnome-shell             15 actionable widgets   0.83 s
Brave Browser            0 actionable widgets   0.00 s
--------------------------------------------------------
whole desktop          175 actionable widgets   2.93 s
```

And the content is not a guess — it is the app's own labels, with screen
extents, state flags and an invocable action:

```
push button: 'Minimize' @(1800,40,34x30)  [SHOWING/VISIBLE/SENSITIVE/ENABLED] ACTIONS=click
push button: 'New Tab'  @(72,32,36x46)    [SHOWING/VISIBLE/SENSITIVE/ENABLED] ACTIONS=click
list item:   'Bluetooth'                  [SHOWING/VISIBLE/SENSITIVE]
table cell:  'JARVIS-TESTING.md'          [SHOWING/VISIBLE/SENSITIVE]
```

Finding the *active* frame takes **0.023 s**, so scoping the walk to "the window
he is looking at" is free.

**Failure modes, all measured:**

1. **Chromium is a stub.** Brave exposes an `application` and one `frame` and
   nothing inside — 2 nodes. *Experiment:* `org.a11y.Status` reports
   `IsEnabled: false`, so I set `org.gnome.desktop.interface toolkit-accessibility`
   to `true` and polled Brave's tree once a second for 12 s. **It stayed at 2
   nodes.** The setting was restored to `false` immediately (verified). The
   remaining lever is relaunching Brave with `--force-renderer-accessibility`,
   which costs renderer memory permanently and **has not been tested here** —
   it would mean restarting his browser, which I did not do.
2. **GTK4 coordinates lie.** `gnome-control-center` reports its frame at
   `(0,0,980×606)`; `xdotool getwindowgeometry` says the window is really at
   `(10,-46)` and `2204×1456`. GTK4 hands back window-relative origins even when
   asked for `SCREEN` coordinates. **Never click a GTK4 widget by its reported
   extents.**
3. **GTK3 coordinates are correct but in the wrong unit.** `gnome-terminal`
   reports `(66,32,1854×1048)`; the window is really at `(132,64)` and
   `3708×2096` — **exactly ×2**, the HiDPI factor. AT-SPI speaks logical pixels,
   `xdotool` speaks device pixels. Any coordinate crossing between them must be
   scaled, and the factor is a property of the display, not a constant to
   hardcode.
4. **Names are not unique.** `gnome-text-editor` alone showed `'Close'` three
   times and `'Discard Changes and Reload'` three times (several frames, and
   infobars that are present but not showing). Nautilus reports every desktop
   file **twice** (a cell wrapping a cell). "Click Close" is genuinely ambiguous
   *inside one app*. This is not a corner case; it is the first thing you hit.
5. **The venv cannot import it.** `~/vss_env` was built with
   `include-system-site-packages = false`, so `import gi` fails there. Two fixes,
   both measured working, no install:
   * `sys.path.append('/usr/lib/python3/dist-packages')` — the venv is 3.12.3
     and so is the system Python, same ABI. **Works.**
   * run the walk as a `/usr/bin/python3` subprocess returning JSON —
     **0.04 s round trip**. Slower per call but the GI/DBus stack cannot take
     Jarvis down with it, and it is trivially fakeable in tests.
   The subprocess is the better default for exactly the reason the second half
   says: an accessibility bus that hangs must not hang the assistant.
6. **It is a live tree, not a snapshot.** Between reading "the button is here"
   and clicking it, a menu can close. Anything that acts on a widget must
   re-check `SHOWING`/`SENSITIVE` immediately before acting, and even then it is
   a race.

**What it buys beyond clicking:** it is the only route that gives Jarvis a
vocabulary. "What can I do in this window?" is answerable in 0.6 s with real
names, instead of a 9–25 s vision call that returns prose.

### Route B — the app's own command names (GActions). **The sleeper.**

Every GTK toplevel publishes its GAction map through the same interface. Nautilus
exposes 45 of them on its frame:

```
win.new-tab  win.go-home  win.back  win.up  win.reload  view.new-folder
view.select-all  view.properties  view.zoom-in  window.close  window.minimize
```

`gnome-text-editor` exposes `page.save`, `page.save-as`, `win.open`,
`page.begin-search`, `settings.wrap-text`, `page.print`, `window.close`.

These are **semantic, position-independent, invisible to a screenshot, immune to
the coordinate problems above, and enumerable at runtime**. "Jarvis, new folder"
→ `view.new-folder` is a fuzzy string match against a list the app itself
publishes, not a guess about pixels.

**Failure modes:** the names are internal and undocumented, and they change
between app versions — so the mapping must be discovered live and never cached
across a restart. Some are no-ops when the context is wrong (`page.save` with no
document) and report success anyway. And **invoking one is untested here** — I
deliberately did not fire an action on his live desktop. The read side is proven;
the write side is a one-line experiment he should watch happen.

### Route C — screenshot + the vision model. **Read only. Never for coordinates.**

The capture half is fast and already built: 0.05 s to grab, 0.10 s to a 142 KB
JPEG. The problem is what the model receives and what it can return.

* **Resolution.** 3840 → 1280 is a factor of **3.0**, but the desktop is 2× HiDPI,
  so the model effectively sees a 1280×720 render of a 1920×1080 desktop. Menu
  and tab labels land around 10–12 px tall — right at the edge of legibility.
  Raising `screen.max_width` fixes legibility and costs latency and tokens on
  every look.
* **No grounding model on this box.** Installed: `gemma4:26b` (the resident chat
  model, carries the clip projector), `llama3.2-vision:latest` (**pulled but
  unloadable** — ollama 0.33.1 answers 500 `unknown model architecture: 'mllama'`,
  already recorded in `screen.py`). Neither is a detection/grounding model.
  gemma4 will *describe* the screen well. Asking it for a bounding box is asking
  a describer to be a detector, and there is no second source to check it
  against.
* **Cost.** `VISION_TIMEOUT_S = 25`, a cold projector load is ~9 s, and
  `OLLAMA_MAX_LOADED_MODELS=1` means any non-resident model evicts the chat model
  and costs ~7 s on his next spoken turn.
* **Unverifiable.** A returned coordinate cannot be checked before it is used.
  A wrong one is a real click on a real thing.

**Verdict: vision answers "what does it say", never "where do I click".** It is
already built for the first job and should stay there.

### Route D — no targeting at all (keyboard and accelerators)

Half-built already: `press_key`, the shortcut tables, `type_text`. Deterministic,
instant, no tree, no pixels. `Ctrl+S`, `Ctrl+W`, `F11`, `Alt+F4`, menu
traversal by `Alt+F` then arrow keys.

**Failure mode:** it is app-specific, and it is silently wrong when the wrong
window has focus — a `Ctrl+W` meant for a browser tab that lands in an editor
closes a document. Which makes **"what has focus, and did I say so out loud"**
the safety question, not an implementation detail.

### The recommendation

Layer them, cheapest and most deterministic first:

> **D (keyboard) → B (named app action) → A (named widget) → C (vision, read-only).**

Never fall through to a coordinate click. If A cannot name a target, the correct
behaviour is to **say so**, not to reach for pixels.

### What breaks if he ever moves to Wayland

Plainly, because it is not symmetric:

* **`xdotool` dies completely.** No synthetic input to native Wayland clients, no
  input to the shell, and `xdotool search` cannot even see native Wayland
  windows. XTEST reaches only apps still running under Xwayland. Everything in
  `desktop.py` — typing, keys, clicks, scroll, window activation — stops.
* **`PIL.ImageGrab` dies.** Capture would have to go through the xdg-desktop
  portal, which prompts.
* **AT-SPI *reading* survives untouched** — it is D-Bus, not X.
* **Route B survives** — GActions are in-app D-Bus calls.
* **Route A's clicking dies** with xdotool, but `Action.do_action` and
  `grab_focus` are in-process and survive.

So the Wayland-portable subset is exactly the subset this design recommends
building first, which is a second reason to prefer it. If he moves, the loss is
"clicking arbitrary pixels" and "typing into arbitrary windows" — real losses,
but not the whole feature.

---

## 4. Are accessibility APIs actually available here? — yes, verified

Because the question was asked directly, here is the evidence rather than the
conclusion:

| Check | Result |
|---|---|
| `at-spi-bus-launcher` | running, pid 3159 |
| `at-spi2-registryd` | running, pid 3302, `--use-gnome-session` |
| a11y bus | `unix:path=/run/user/1000/at-spi/bus_1`, its own `dbus-daemon` |
| `Atspi-2.0.typelib` | present in `/usr/lib/*/girepository-1.0/` |
| PyGObject | `gi` 3.48.2, `gi.require_version('Atspi','2.0')` succeeds |
| `pyatspi` (the old wrapper) | **not installed** — use `gi.repository.Atspi` directly, which is the modern API anyway |
| Apps on the bus | **20**, including Brave, Nautilus, gnome-terminal, Text Editor, Settings, gnome-shell |
| `Atspi.generate_mouse_event` / `generate_keyboard_event` | present (but XTEST-backed, so X11-only) |
| `Atspi.Action` / `Component` / `Text` / `EditableText` / `Value` / `Selection` | all present |
| `Wnck-3.0.typelib` | **present** — window list/activate/close without `wmctrl`, which closes that gap too |
| `org.a11y.Status` | `IsEnabled: false`, `ScreenReaderEnabled: false` |
| `toolkit-accessibility` gsetting | `false` |

The last two lines are the interesting ones: **the tree is fully populated for
GTK2/3/4 apps despite both flags being false**, because `GTK_MODULES=gail:atk-bridge`
is already in the environment and GTK4 registers unconditionally. Nothing needs
enabling. Chromium is the only toolkit that gates on the flag, and flipping the
flag did not wake it (§3 A.1).

So: **recommend AT-SPI, and recommend `gi.repository.Atspi`, not `pyatspi`, not
`accerciser`, and not an apt install.**

---

## 5. Safety

Synthetic input is indistinguishable from him. There is no dry run, no undo, no
sandbox, and no permission dialog between Jarvis and anything he can reach. A
mis-aimed `Ctrl+W` closes unsaved work; a mis-aimed click sends a half-written
message; a mis-aimed `Return` on a focused dialog confirms a deletion. This has
to shape the design.

### Reuse the confirmation machinery that exists

Do not invent a second one. `jarvis/commander.py` already has:

* `c.stash_destructive(run, question)` — speak a read-back, hold the closure,
  and let the **next yes** run it; it expires after `DESTRUCTIVE_TTL_S` and a
  change of subject drops it.
* `confirm.read_back` (on) and `confirm.shaky_logprob` (−0.7) in
  `~/.config/jarvis/assistant.json`, with `c.shaky_transcript()` — **a
  low-confidence transcript forces a read-back even for things that are normally
  free.** That is precisely right here: "close the window" and "close the
  browser" are one ASR slip apart.
* `jarvis/approvals.py` already publishes `ApprovalRequested` → the app speaks the
  question, the UI shows ALLOW / DENY, voice or UI or Discord can answer, and
  **an unanswered request denies after two minutes**. That is the model for
  anything heavier than a read-back.

### Three tiers

| Tier | Contents | Gate |
|---|---|---|
| **Free** | read the tree, name the windows, say what's clickable, focus/raise a window, scroll, volume, `Ctrl+C`, zoom, next/previous tab, move the pointer *without clicking* | just do it — all reversible or inert |
| **Confirm (spoken read-back, next-yes-runs-it)** | any click on a named widget, any GAction, typing into a window that is not the pinned target, `Ctrl+S`, `Ctrl+W`, close a window, close a tab, anything in a window whose title matches a `send`/`delete`/`payment` pattern | `stash_destructive` |
| **Refused outright** | clicking a bare coordinate that no widget backs; anything in a browser (the tree is empty, so every target is a guess); `Alt+F4` on a window with unsaved-changes markers; anything while a password/authentication widget has focus; typing into a window Jarvis cannot name; chains longer than N steps; a target found only by vision | say why, name the reason |

The refusals matter more than the confirmations. **"I can see it says Brave and I
have no idea what is in it, sir"** is a correct and useful answer.

### The read-back must name the window, not just the target

The failure that actually happens is *the right button in the wrong window*. So
the sentence is `"Click 'Close' in Text Editor, Good evening Ali and Heather.txt?"`
— widget, app, and document. Not `"Click Close?"`.

### Misidentification

Borrow the rule already written down in `jarvis/tools/filepick.py`, which exists
for the same reason on the file side: **ambiguity asks, it never guesses; a miss
is a miss.** The measurement in §3 A.4 proves it is needed — `'Close'` appears
three times in one app. So:

* one match → read it back
* more than one → read back the alternatives and ask which, capped at a
  speakable number
* zero → *"I can't find anything called that in Settings, sir"* — and stop. There
  is no confidence threshold at which the wrong widget becomes acceptable.
* a match that is `SHOWING` but not `SENSITIVE` → say it's greyed out; do not
  click it to find out.

### Stopping it mid-action — the part that needs real design

This is the weakest point and deserves the honesty. **Keystrokes already sent
cannot be recalled.** So "stop" can only mean "send nothing further", and the
design has to make that boundary as tight as possible:

1. **One chokepoint.** Every synthetic event goes through a single function that
   checks an abort flag *between* steps. `DesktopControl.execute` today is a bare
   `for` loop with sleeps and no check; that is the first thing to change.
2. **Short chains, hard cap.** A chain of 3 is a command; a chain of 20 is a
   script and should be refused. The cap makes "already sent" a small number.
3. **The wake word pre-empts.** `quiet_kind` is already Tier-1 barge-in that cuts
   speech; screen control needs the same treatment for *actions*, and it must be
   checked before the turn is routed, not after.
4. **He moves the mouse, it stops.** Sample `xdotool getmouselocation` before
   each step; if the pointer is somewhere Jarvis did not put it, abort. He
   reaching for the mouse is the most natural "stop" gesture there is, and it
   needs no words.
5. **A watchdog.** Any chain still running after N seconds aborts itself and says
   so. A stuck chain must not be stopped only by him noticing.
6. **Park the pointer.** Save the position, restore it after. A pointer that
   silently moved is how he loses track of what happened.
7. **Never act while he is typing.** If the target window has keyboard focus and
   there has been recent key activity, defer and say so — Jarvis and he typing
   into the same widget is unrecoverable.

### And then: verify

After acting, Jarvis has no idea whether it worked. Re-reading the window title
and the widget's state after the action costs ~0.05 s and turns *"I clicked
Save"* into *"Saved — the title no longer shows a modified marker"*. **That is
part of the design, not a nicety** — without it every report is a claim about
something Jarvis never observed.

---

## 6. Build order

Cheapest genuinely-useful thing first, and each step is shippable on its own.

### Evening 1 — read only. **This is what he could have in one evening.**

`jarvis/screenctl.py`, no input, nothing irreversible:

* `windows()` → app, window title, geometry, which one is active (0.023 s)
* `widgets(window)` → role, name, state, extents, available actions (0.04–1.25 s)
* `actions(window)` → the GAction names the app publishes
* the walk runs in a `/usr/bin/python3` subprocess with a JSON contract (0.04 s
  round trip) so the GI/DBus stack cannot hang or crash Jarvis, and so tests
  replace one seam with a recorded tree — no display, no bus, no GPU
* one tool, `screen_targets`, registered like `screen_qa`: **"Jarvis, what can I
  click here?"** answered from real labels in under a second

That alone is worth having, and it is honest about Brave from day one ("I can see
Brave but not inside it, sir"). Every later step needs it. It is also the piece
that survives a move to Wayland.

### Evening 2 — focus and window management

`focus(app)` by application name rather than title substring (via AT-SPI's app
list, or Wnck now that we know the typelib is there), plus `click_widget(name)`
behind the **Confirm** tier, GTK3-only at first because those are the only
coordinates that are trustworthy — with the ×2 HiDPI scale derived at runtime
from `xrandr` vs the reported root extents, never hardcoded.

### Evening 3 — GActions

`do_action(window, name)` with fuzzy matching against the app's published list
and the same read-back. This is where "Jarvis, new folder" and "Jarvis, save
this" start working properly, and it sidesteps every coordinate problem.

### Evening 4 — the safety plumbing made real

The abort chokepoint, the pointer-moved abort, the watchdog, the chain cap, the
post-action verification, and the refusal list wired to actual predicates rather
than good intentions.

### Deliberately later, or never

* **The browser.** Would need Brave relaunched with
  `--force-renderer-accessibility` (permanent renderer memory cost, must go in
  the launcher, **and untested here**), or a browser extension, or CDP over a
  debug port. Each is its own project. Until then, say so out loud.
* **OCR fallback.** Blocked on `apt install tesseract-ocr`, which he cannot do.
* **Vision-driven clicking.** Not with the models on this box. Route C stays
  read-only.

---

## 7. Decisions only he can make

1. **Is browser-blind screen control worth having?** It covers Settings, Files,
   the text editor, terminals, and the shell — but not the app he lives in. That
   is the single question that decides the whole build.
2. **Is he willing to relaunch Brave with `--force-renderer-accessibility`?**
   It costs renderer memory permanently, it means editing the launcher, and even
   then it is unproven on this box. Worth one measured experiment before anyone
   commits to it.
3. **Which applications does he actually want driven by voice?** Naming three
   real ones turns this from a general capability into a testable feature — and a
   general "drive anything" surface is the version with the biggest blast radius
   and the least value.
4. **Should the confirm tier be settable?** `confirm.read_back` already exists
   and defaults on. A `screen_control.confirm_clicks` that he could turn off is
   easy to add and is exactly the switch that would be regretted; recommendation
   is not to offer it.
