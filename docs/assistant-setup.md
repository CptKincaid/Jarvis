# Jarvis personal-assistant setup

Everything personal lives in **one file**: `~/.config/jarvis/assistant.json`
(mode `600`, yours only). Jarvis creates it with placeholders the first time
he starts; fill in what you want, leave the rest blank. Anything left blank
just gets a polite excuse ("I'll need your Google calendar link set up, sir;
the notes are in docs/assistant-setup.md.") instead of an error — see the
table at the end.

**Never paste a real password or token anywhere but this file.** The file is
never logged, `redacted()` masks secrets as `•••` everywhere the app shows
config, and the tests only ever use placeholders.

Code: `jarvis/assistant_config.py` (`AssistantConfig`), `jarvis/autostart.py`.

---

## 1. Create the file

Either start Jarvis once (the app creates it), or from a terminal:

```bash
cd ~/Jarvis && ~/vss_env/bin/python -c "from jarvis.assistant_config import AssistantConfig as C; print(C.load().path)"
```

Both write `~/.config/jarvis/assistant.json` from the defaults below with
`chmod 600`. Edit it with any editor; Jarvis reloads it when the file changes
(`reload_if_changed()`), so a restart is not needed for most values.

A corrupt file (a stray comma, half a paste) is not fatal: it is moved to
`assistant.json.bad` and a fresh one is written — copy your values back from
the `.bad` file.

Override the location for tests or a second profile with
`JARVIS_ASSISTANT_CONFIG=/path/to/file.json`.

### The whole file, with placeholders

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
    "permission_prompt_tool": false,
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
               "news_feeds": ["https://www.theverge.com/rss/index.xml",
                              "https://feeds.arstechnica.com/arstechnica/index"],
               "sports_feeds": [], "stock_symbols": [],
               "sections": {"weather": true, "calendar": true, "news": true, "sports": true,
                            "stocks": true, "canvas": true, "todos": true, "alarms": true,
                            "reminders": true},
               "verbosity": "normal",
               "wake_offer": true, "early_before": "09:00", "wake_lead_min": 60},
  "alarms": {"sound": "", "volume": 0.8, "escalate": true, "max_ring_s": 300, "snooze_min": 10},
  "discord": {"bot_token": "", "channel_id": "", "user_id": ""},
  "spotify": {"client_id": "", "client_secret": "", "default_device": "HPCOMPUTER",
              "liked_strategy": "uris", "market": "from_token"},
  "autostart": {"enabled": false}
}
```

`""` and `null` are the placeholders. Values that look like
`<paste here>`, `PASTE-…`, `your-…`, `changeme` or `xxxx` are treated as
blank too, so a half-finished edit never counts as configured.

---

## 2. Home location (weather, "where am I", local time)

```json
"home_location": {"city": "Chicago", "region": "Illinois", "lat": 41.8781, "lon": -87.6298},
"location_lookup": true
```

- Fill `lat`/`lon` (decimal degrees; a map app's "copy coordinates" gives
  them) and Jarvis uses them for weather without asking anyone where you are.
- Leave `lat`/`lon` `null` and, with `location_lookup` true, he looks the
  location up from your network address once a day (city-level accuracy,
  cached in `~/.cache/jarvis/location.json`). Set `location_lookup` false as
  well and location questions get the setup line instead.
- `units`: `"us"` (Fahrenheit, mph, miles) or `"metric"`.

## 3. Google Calendar (read-only, no OAuth)

Google gives every calendar a private iCal address that needs no login.

1. Open [Google Calendar](https://calendar.google.com) on the web.
2. Left sidebar > hover the calendar > the three dots > **Settings and sharing**.
3. Scroll to **Integrate calendar** > copy **Secret address in iCal format**
   (it ends in `/basic.ics`; do not use the public address, it is empty for a
   private calendar).
4. Paste it into the list — one URL per calendar you want Jarvis to see:

```json
"google_ical_urls": [
  "https://calendar.google.com/calendar/ical/<your-address>%40gmail.com/private-<secret>/basic.ics",
  "https://calendar.google.com/calendar/ical/<family-calendar-id>/private-<secret>/basic.ics"
]
```

The secret address is a credential: anyone holding it can read that
calendar. If it leaks, **Reset** it on the same settings page and paste the
new one. Jarvis refreshes every 10 minutes and keeps a cache
(`~/.cache/jarvis/calendar_cache.json`) so a start without network still
answers "as of 9:10 am".

## 4. Apple / iCloud Calendar (CalDAV with an app-specific password)

1. Go to [appleid.apple.com](https://appleid.apple.com) > **Sign-In and Security**
   > **App-Specific Passwords** > **+** (Generate). Name it "Jarvis".
2. Copy the password (`xxxx-xxxx-xxxx-xxxx` shape).
3. Fill in:

```json
"icloud": {
  "apple_id": "<your-apple-id@icloud.com>",
  "app_password": "<app-specific-password>",
  "url": "https://caldav.icloud.com"
}
```

Read-only: Jarvis lists and searches events, never writes. Revoke the
password on the same Apple page if it ever leaks. Two-factor authentication
must be on for your Apple ID (it is, if you can see the App-Specific
Passwords section).

## 5. Gmail summaries (IMAP with an app password)

1. Google Account > **Security** > **2-Step Verification** must be on.
2. Same page, bottom: **App passwords** (or search "App passwords" in the
   account search box) > app name "Jarvis" > **Create**. Copy the 16-character
   password (spaces are fine, they are ignored).
3. Gmail > Settings (gear) > **See all settings** > **Forwarding and POP/IMAP**
   > **Enable IMAP** (already on for most accounts).

```json
"gmail": {"address": "<you@gmail.com>", "app_password": "<16-char-app-password>", "imap_host": "imap.gmail.com"}
```

Jarvis only reads: `SELECT INBOX` read-only, `BODY.PEEK`, so nothing is
marked read and nothing is sent. Message bodies are never logged. A Google
Workspace account uses the same host.

## 6. Discord (away alerts, two-way)

Jarvis posts milestones and questions ("It wants to push to origin main,
sir — shall I? Reply yes or no.") to one channel and reads your replies
there as answers or ordinary commands.

1. [Discord Developer Portal](https://discord.com/developers/applications)
   > **New Application** ("Jarvis") > **Bot**.
2. **Reset Token** > copy it once (it is shown once).
3. Under **Privileged Gateway Intents** enable **MESSAGE CONTENT INTENT**
   (Jarvis cannot read your replies without it). Save.
4. **OAuth2** > **URL Generator**: scope `bot`; permissions **Send
   Messages**, **Read Message History** (View Channel is implied). Open the
   generated URL and invite the bot to your server.
5. Discord app > **User Settings** > **Advanced** > **Developer Mode** on.
   Right-click the channel > **Copy Channel ID**; right-click your own name >
   **Copy User ID** (optional — lets you DM the bot instead of using the channel).

```json
"discord": {"bot_token": "<bot-token>", "channel_id": "<channel-id-digits>", "user_id": "<your-user-id-digits>"}
```

Only messages in that channel (or DMs from `user_id`) are read; everything
else is ignored. The token is never logged. If it leaks, **Reset Token** in
the portal and paste the new one.

`user_id` matters for safety: with it set, Jarvis obeys only you, so a
stranger's "yes" in a shared channel can never approve a `git push`; left
blank, anyone who can post in `channel_id` can command him. Two optional
switches silence a channel independently (both default on, and the key may
be absent): `"alerts": {"desktop": true, "discord": true}`.

Troubleshooting from `jarvis.log`: gateway close **4014** means the MESSAGE
CONTENT intent is still off in the portal (Jarvis falls back to REST polling
every 5 s until it is on); close **4004** means the token is wrong and he
gives up until the app restarts.

## 7. Morning briefing

Off by default. Turn it on in the settings drawer (**Assistant > Morning
briefing**) or in the file:

```json
"briefing": {
  "enabled": true,
  "hn_items": 3,
  "news_feeds": ["https://www.theverge.com/rss/index.xml", "https://feeds.arstechnica.com/arstechnica/index"],
  "sports_feeds": [],
  "stock_symbols": []
}
```

Content is weather, today's calendar, then **tech and AI news only**: the top
Hacker News stories plus the first item from each feed, three items in
total, one sentence each. Add an RSS/Atom URL to `sports_feeds` or a ticker
(`"NVDA"`, `"AAPL"`) to `stock_symbols` and those sections appear too;
they are silent while empty. While disabled, "good morning" is an ordinary
greeting and "briefing" gets: "The morning briefing is switched off, sir; the
toggle is in settings under Briefing."

### The good-night preview

With `briefing.enabled` on, "good night" answers "Good night, sir. Tomorrow,
briefly." and then reads tomorrow in one breath: the weather, the first
event and where, Canvas work due within a day (only once `canvas.token` is
set), open to-dos and the alarm that is set. If the first event starts by
`early_before` (09:00) and no alarm rings before it, he asks **"Shall I wake
you at 7:00 am, sir?"** — a plain "yes" in the follow-up window sets it
(`wake_lead_min` = 60 minutes before the event, on the quarter hour); "no",
a change of subject, or three minutes of silence drops the offer. Asked
outright — "what does tomorrow look like", "preview tomorrow", "tomorrow's
briefing" — the preview runs whatever the toggle says.

```json
"briefing": {"wake_offer": true, "early_before": "09:00", "wake_lead_min": 60}
```

### The week ahead

"How's my week looking?", "weekly forecast", "how busy is my week": the
calendar, Canvas deadlines and reminders for the next seven days, day by
day, plus the open to-dos — "Heavy Tuesday, sir: Biosensors, the lecture,
the lab and the lab report; Wednesday and Saturday are clear." One card,
one row per day. ("What does my week look like" stays a calendar question.)

### Switching sections off, and how long he talks

Say it and it sticks (written to this file under `briefing.sections`, and to
the memory's `preferences.json`):

- "no news in the morning", "I don't want the weather in my morning briefing",
  "skip the sports in my briefing", "leave out canvas in the evening preview",
  "drop the reminders from my weekly forecast"
- back on: "put the news back in the briefing", "include the to-dos in my
  briefing again"
- "be briefer" / "shorter briefings" → `briefing.verbosity: "brief"`, which
  halves the briefing, preview and forecast allowances; "the full briefing" /
  "more detail" restores it. (Ordinary answers are already two sentences at
  most, so this only changes the briefings.)

```json
"briefing": {
  "sections": {"weather": true, "calendar": true, "news": true, "sports": true,
               "stocks": true, "canvas": true, "todos": true, "alarms": true,
               "reminders": true},
  "verbosity": "normal"
}
```

A section that is off is not fetched and not mentioned; a name missing from
the map counts as on.

## 8. Alarms, timers, reminders

```json
"alarms": {"sound": "", "volume": 0.8, "escalate": true, "max_ring_s": 300, "snooze_min": 10}
```

- `sound`: path to a `.wav`/`.oga`/`.ogg` file; blank uses the desktop's
  `alarm-clock-elapsed.oga`, or a generated two-tone ring if that is missing.
- `volume`: 0.0-1.0 for `paplay`.
- `escalate`: after 30 s of ringing, full volume and a shorter gap.
- `max_ring_s`: stop after this many seconds and mark the alarm missed
  (spoken: "You missed …").
- `snooze_min`: the default for "snooze".

Say "wake me up at 6:30", "set an alarm for 7 every weekday", "timer for 10
minutes", "remind me at 3 to call the dentist", "what alarms do I have",
"cancel the alarm"; while it rings: "stop", "I'm up", "snooze", "snooze 5".

Alarms need the app running — see section 10.

## 9. Claude (coding sessions)

```json
"claude": {
  "allowed_dirs": ["/home/hunterp/Jarvis", "/home/hunterp/haymaker-digest"],
  "projects_root": "/home/hunterp/projects",
  "permission_mode": "acceptEdits",
  "dangerously_skip_permissions": false,
  "model": "opus", "big_model": "fable", "fast_mode": false, "effort": "",
  "skill_phrases": { ... }
}
```

**Allowed dirs.** Inside these, Claude may do everything (edit, run
commands, install, push): Jarvis writes each project's
`.claude/settings.local.json` allow-list before a task. "start a new project
called X" creates `projects_root/X` (git init, CLAUDE.md, README) and adds it
here automatically. "work on the VSS project" for a directory that is not
listed asks first, then adds it.

**Outside allowed dirs** — any tool call touching a path elsewhere, or a
tool not on the list — Claude cannot proceed on his own: a small MCP
permission tool relays the question to Jarvis, who speaks it ("It wants to
write to /etc/hosts, sir — shall I?"). Answer "yes"/"no" (typed, spoken, or in
Discord). No answer in two minutes counts as no. `dangerously_skip_permissions`
turns that whole gate off globally; leave it false.

**Models.** `model` is the everyday one (`opus`); `big_model` (`fable`) is
used when you say "use fable", "think hard", "this is a big one", or when the
task is plainly large (multi-file refactor, new feature). "use opus / sonnet /
haiku / fable" switches for the rest of the session. `fast_mode` maps to
Claude's fast mode where the CLI supports it ("fast mode on/off");
`effort` is passed as `--effort` when set (`low`, `medium`, `high`).

**Skill phrases.** Regex (case-insensitive, whole utterance) → the slash
command sent to Claude; `$1` is the first capture group. Add your own:

```json
"^ship it$": "/commit",
"^tidy (.+)$": "/simplify $1"
```

Unknown skills pass through: "run the vercel skill on the landing page" →
`/vercel on the landing page`. All plugins installed for your user
(superpowers, code-review, commit-commands, feature-dev, ralph-loop, …) are
available inside the sessions automatically.

**Resume.** "pick up where we left off" / "what were we doing yesterday" /
"continue the haymaker project" finds the newest matching Claude session
under `~/.claude/projects` (yours included) and resumes it in that
directory. Two equal candidates get one question naming both.

**The terminal button** (next to the mic) opens the tmux session Claude is
working in; a second click raises the same window.

**"What's Claude doing?"** — also "how's Claude getting on", "is Claude still
working", "is Claude done yet", "any word from Claude", "Claude status". One
spoken line from live state: the project, how long ago it started, how many
files it has touched, the last milestone Jarvis spoke ("the last word was:
Editing router.py"), whether Claude is sitting on an in-pane question (the
AskUserQuestion menu — answer it in the terminal; a permission prompt is
spoken separately and reads "waiting on you about"), how many tasks are
queued, and whether your OWN `cc-<dir>` tmux sessions (the `claude` wrapper
in `~/.bashrc`) are mid-turn or idle. With nothing running: "Claude's idle,
sir; the active project is jarvis." A plain "status" or "status report" is not
this question — that stays the persona's own status line.

**Clipboard to Claude.** Copy a traceback, a log excerpt or a snippet anywhere
and say "have Claude fix what I copied", "ask Claude about the clipboard",
"send what I copied to Claude and fix the error" or "give Claude the
clipboard". Highlight text in a terminal instead and say "have Claude look at
the selection and explain it" / "tell Claude to fix what I highlighted" (the X
primary selection). The clip is written to `<active project>/.jarvis/clips/`
(mode 0600, the folder git-ignores itself) and the task Claude gets is your
sentence with the clip phrase replaced by that file — "fix the text in
…/.jarvis/clips/20260830-134905.txt. That file holds what I just copied on my
screen; read it first." The clip itself is never put in the prompt, so it never
reaches the task log, the bus or Discord. It goes to the active project; switch
first ("work on the VSS project") if it belongs elsewhere. An empty clipboard
gets "The clipboard is empty, sir." A plain "have Claude fix this" is still an
ordinary task, not a clip.

### Web lookups

"Look up …", "search the web for …", "who won …", "what's the latest news about …",
"price of …" go to a one-shot `claude -p` with web search allowed — the CLI has it
built in, so nothing is added to the venv. Jarvis says "Looking that up, sir." at once
and speaks the answer when it lands (~12–15 s). `claude.web_model` picks the model
(default `haiku`: fastest, and it obeys "no links"; `sonnet` works too). This is the
only path that hands what you said to a web search; Jarvis's own reasoning otherwise stays
on the local model (the coding sessions, weather, calendars, mail and Discord each talk to
their own service).

## 10. Always on: start at login, never sleep

Alarms and reminders ring only while Jarvis runs, so let him start with the
desktop and keep the Spark awake:

```bash
cd ~/Jarvis && ~/vss_env/bin/python -m jarvis.autostart --install   # entry + never-sleep
~/vss_env/bin/python -m jarvis.autostart --status
~/vss_env/bin/python -m jarvis.autostart --uninstall
```

`--install` writes `~/.config/autostart/jarvis.desktop`
(`Exec=/home/hunterp/vss_env/bin/python -m jarvis.app`, `Path=/home/hunterp/Jarvis`,
15 s delay so the session is up first; running it twice changes nothing)
and sets `org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type`
to `nothing` if it is not already (it is on this machine). Setting
`"autostart": {"enabled": true}` in the file, or the **Start at login**
toggle in settings, does the same from inside the app; `jarvis.app
--install-autostart` is the equivalent command.

A second launch while Jarvis is already running just raises the existing
window (pid file guard), so the entry is safe alongside a manual start.

If the machine was off or Jarvis was closed when something was due: less
than an hour late fires at once ("While I was down, sir: …"), later than
that is announced as missed.

---

## 11. Spotify (music)

Jarvis plays to any Spotify Connect device: HPCOMPUTER, your phone, a speaker — or the Spark
itself, which advertises as **"Spark"** once `jarvis-spotify.service` is running (below).
`spotify.default_device` still decides where playback goes unless you name a device in the
request. Needs Spotify Premium for anything that controls playback (Free accounts get
"I'm afraid that needs Spotify Premium, sir.").

### 0. The Spark as a Connect device (optional)

There is no Spotify client for aarch64 Linux and the web player needs Widevine, which Google
does not ship for it, so the Spark runs librespot and feeds PipeWire. No sudo is needed:
librespot is built with the `pipe` backend (links no audio libraries) and `pacat` carries the
PCM to whatever the default sink is (the soundbar here).

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
~/.cargo/bin/cargo install librespot --locked --no-default-features \
    --features rustls-tls-webpki-roots,with-libmdns --root ~/.local
install -m755 scripts/jarvis-spotify-device ~/.local/bin/
install -m644 scripts/systemd/jarvis-spotify.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now jarvis-spotify.service
```

`--locked` matters (an unlocked build hits a `vergen_lib` version skew in the build script),
and `--no-default-features` alone fails on a missing TLS feature — hence the two features.
Pick "Spark" once in any Spotify app; librespot caches the credentials under
`~/.cache/librespot` and reconnects by itself across restarts and reboots (`Restart=always`).
`LIBRESPOT_NAME` renames the device.

### 1. Create the developer app (once)

1. Go to https://developer.spotify.com/dashboard and sign in with your own Spotify account.
2. **Create app**. Name and description are free text (e.g. "Jarvis").
3. **Redirect URIs**: add exactly `http://127.0.0.1:8888/callback` (not `localhost`; Spotify
   requires the loopback IP for new apps).
4. **APIs used**: tick "Web API". Save.
5. Open the app → **Settings** → copy the **Client ID** and (click "View client secret") the
   **Client secret**.

### 2. Put the keys in `~/.config/jarvis/assistant.json`

```json
"spotify": {
  "client_id": "<paste client id>",
  "client_secret": "<paste client secret>",
  "default_device": "HPCOMPUTER",
  "liked_strategy": "uris"
}
```

- `default_device` is the Spotify Connect name that plays when you do not name one — the
  name shown in Spotify's device picker (HPCOMPUTER is the Windows PC). Say "on my phone" /
  "on the tv" / "on HPCOMPUTER" to pick another; type words (phone, computer, tv, speaker)
  match by device type, anything else by name.
- `liked_strategy`: `"uris"` (default, documented API: shuffles up to 500 of your Liked Songs,
  starts 100 and queues 20 more in the background) or `"collection"` (server-side shuffle of
  the whole library via the undocumented `spotify:user:<id>:collection` context — try it; if it
  fails the tool silently falls back to the other strategy). Optional: `liked_chunk` (100),
  `liked_queue_ahead` (20), `liked_cap` (500).
- The file stays mode 600; the secret is masked in logs.

### 3. Link the account (once, on the Spark's desktop)

```bash
cd ~/Jarvis && ~/vss_env/bin/python -m jarvis.app --spotify-login
# or, without the app: ~/vss_env/bin/python -m jarvis.tools.spotify --login
```

Brave opens the Spotify consent page; approve it and the tab lands on
`127.0.0.1:8888/callback` ("You can close this tab"). The token is written to
`~/.config/jarvis/spotify_token.json` (mode 600) and refreshes itself for good; Jarvis says
"Spotify's linked, sir." Over SSH, add `--no-browser` and paste the URL into any browser, then
paste the redirected URL back. `python -m jarvis.tools.spotify --status` shows
configured/linked. Scopes requested: playback state + control, currently playing, library
read/write (for "like this song"), private/collaborative playlists, top items.

### 4. What you can say

| you say | Jarvis does |
|---|---|
| "play Blinding Lights" / "play One More Time by Daft Punk" | searches, plays the track on the default device, "Blinding Lights by The Weeknd, sir — on HPCOMPUTER." |
| "play some Daft Punk" / "play songs by the Weeknd" | the artist's top tracks |
| "play the album After Hours" | the album |
| "play my gym mix" / "play Discover Weekly" | your own playlists first, then public ones |
| "play my liked songs" / "shuffle my likes" | Liked Songs on shuffle |
| "play Blinding Lights on my phone" / "move it to HPCOMPUTER" | named device (transfers playback) |
| "pause", "resume", "next", "previous", "volume 40", "louder", "quieter", "shuffle on", "repeat this song", "jump to 1:30" | transport on whatever is playing |
| "what's playing" | "Blinding Lights by The Weeknd, sir — on HPCOMPUTER." |
| "like this song" | saves it to Liked Songs |
| "queue Save Your Tears next" | adds to the queue |
| "play something like this" / "something like Daft Punk" | artist radio: the "This Is …" / "… Radio" playlist when Spotify still returns one, else the artist's top tracks (and he says which — the recommendations API is closed to new apps) |

### 5. When something is missing

| situation | Jarvis says |
|---|---|
| keys not in assistant.json | "I'll need spotify set up, sir; the notes are in docs/assistant-setup.md." |
| never linked / refresh failed | "Spotify isn't linked yet, sir; run me with --spotify-login once." / "Spotify's link has lapsed, sir; …" |
| no device open anywhere | "Nothing's listening, sir — open Spotify on HPCOMPUTER or your phone." |
| a named device he cannot see | "I can't see toaster on Spotify, sir; I can see HPCOMPUTER, Hunter's iPhone." |
| Free account | "I'm afraid that needs Spotify Premium, sir." |
| nothing found | "I couldn't find X on Spotify, sir." |
| Spotify down / rate limited | "Spotify isn't answering, sir." / "Spotify's rate-limiting me, sir; give it a moment." |

Troubleshooting: "Nothing's listening" while Spotify is open on the PC usually means the PC's
Spotify app is not signed in to the same account or Connect is off — play any song there once
and it appears in the device list. Every Spotify call is logged under `jarvis.tools.spotify`
in `jarvis.log` without the secret or token.

---

## 12. Custom phrases (your own shortcuts)

A phrase you choose, straight to a tool — no model, no classifier, the same every time:

```json
"phrases": [
  {"say": ["drop my needle", "drop the needle"],
   "tool": "spotify_play",
   "args": {"query": "Jingle Bells Bombay Dub Orchestra Remix Joe Williams", "kind": "track"},
   "reply": "Dropping the needle, sir."}
]
```

- `say` is a string or a list; matching ignores case and punctuation, allows surrounding
  words ("jarvis, drop the needle please"), and is whole-word only ("play" does not fire
  inside "display"). The longest matching phrase wins.
- Checked before everything else that could guess, and on the bare transcript too: the wake
  word is consumed by the hotword, so "Jarvis, drop my needle" arrives as "drop my needle",
  which the intent classifier would otherwise drop as background chat.
- `reply` is spoken on success; a failing or unknown tool apologises aloud rather than
  going silent.

---

## 13. Canvas (coursework, grades, announcements)

Read-only. In Canvas: **Account → Settings → Approved Integrations → + New Access Token**,
copy the token once, then in `assistant.json`:

```json
"canvas": {"base_url": "https://canvas.tamu.edu", "token": "<paste the token>"}
```

Then: *"what's due this week?"*, *"any new grades?"*, *"any announcements?"*,
*"when's the biosensors midterm?"* The token is redacted from every log and repr. Unset:
"I'll need a Canvas access token set up, sir; the notes are in docs/assistant-setup.md."

### Deadlines in the briefing, and a heads-up before each one

With the token set, the briefing gains a **Due** section (the next two days: *"Due:
Lab 3 report for BIOSENSORS today 11:59 pm; Quiz 2 for CIRCUITS tomorrow 5:00 pm"*) and,
a few hours before every Canvas deadline, Jarvis says so unprompted, the way he does for
meetings — *"Sir, this is your reminder. Lab 3 report for BIOSENSORS is due in 3 hours."*

```json
"canvas": {"base_url": "https://canvas.tamu.edu", "token": "<paste the token>",
           "heads_up_hours": 3}
```

`heads_up_hours` is the lead (the meeting heads-up's ten minutes is no use for an 11:59 pm
deadline). He checks Canvas every fifteen minutes and files each reminder only once its
time is near, so work you hand in during the day is never announced; a restart never
repeats one (`~/.aiws_trainer/jarvis_memory/deadlines_state.json`). Without a token all of
this is silent: no Due line in the briefing, no nagging about the missing token.

### Exams: "when's my next exam?"

*"When's my next exam?"*, *"when is my next quiz"*, *"how long until the biosensors
midterm?"*, *"how many days until my final?"* are answered instantly, without the model,
from Canvas (30 days ahead) merged with the calendar cache (14 days) — so the calendar half
works before the token exists, and an exam that only the iCloud "Canvas" subscription
carries is found too. Anything titled exam / midterm / final is an exam, a quiz is a quiz;
*"next exam"* never answers with a quiz, and *"final"* / *"midterm"* must be in the title.
The briefing adds an **Exam** countdown (*"Midterm 1 for BIOSENSORS, in 6 days, Tuesday at
9:00 am"*), and at 7 pm the evening before he says *"Midterm 1 for BIOSENSORS is tomorrow
at 9:00 am"*. With Canvas read and nothing found: *"Nothing that looks like an exam on the
books, sir."* With no token and nothing on the calendar the question goes to the model,
which reaches `canvas_due` and its setup line.

## 14. Your own documents (fully local)

Drop PDFs, `.txt`, `.md` or `.docx` into **`~/Documents/Jarvis Docs`** (or list folders in
`docs.paths`). They are chunked, embedded with Ollama's `nomic-embed-text` and stored in a
chromadb index under `~/.aiws_trainer/docs_index`; the first ask kicks the indexing pass
("I'm indexing your documents now, sir; ask me again in a moment"), later asks are instant,
and changed files re-index on their own. *"What does the syllabus say about late work?"* is
answered from the text and cites the file. *"Reindex my documents"* forces a pass. Nothing
leaves the machine.

## 15. Screen Q&A (local vision model)

*"What's on my screen?"*, *"what does this error say?"*, *"summarise what I'm looking at."*
A screenshot of `DISPLAY=:1` goes to the local `llama3.2-vision` model through Ollama; the
answer is spoken directly. Nothing is written to disk unless `JARVIS_DEBUG_SCREEN=1`. The
first call after a while loads the model (~10 s); `screen.model` and `screen.max_width` tune it.

## 16. Spark health and the memory watchdog

*"How's the Spark doing?"* / *"system health"* reads memory, GPU, load, disk and the top
processes. A watchdog checks memory every 30 s and speaks ONCE per episode when free
memory drops under `health.warn_gb` (16) — "Memory is getting tight, sir: …" — again under
`health.critical_gb` (8), and when two processes each hold over `health.hog_gb` (20), the
pattern that ended in a hard power-off on 28 August. It never runs `nvidia-smi` itself.

### Lending the GPU to a trainer (opt-in)

```json
"health": {"yield_to_trainer": true}
```

With that on, the same watchdog lends the local model out the moment a training
process appears (anything an interpreter runs whose script or `-m` module says
`train` / `trainer` / `finetune` / `fine_tune` / `pretrain` — `train.py`,
`aiws_trainer.train`, `finetune_piper.py`; size does not matter): it unloads the
model through Ollama (`keep_alive: 0`, the call it already makes for foreign
models) and says *"I have lent the GPU to your trainer, sir; quick answers only
until it is done."* Until the trainer has been gone for two ticks (60 s) nothing
reloads the model — not a question, not a forced tool, not the five-minute
residency check — so every question that needs the local model gets *"My local
model is lent to your trainer at the moment, sir; quick answers only until it's
done."* The clock, timers, notes, to-dos, the standup and the Claude / web routes
still work. When the run ends: *"Your trainer has finished, sir; I'm loading my
model again."*

It is off by default on purpose: every tool answer (weather, calendar, mail,
Spotify, documents, briefings) runs through the local model, so a multi-hour run
leaves Jarvis with Tier 1 only. The manual doors work whether or not it is on:
*"lend the GPU"* / *"release the GPU"* / *"unload your model"*, and *"take the GPU
back"* / *"reclaim the GPU"* / *"load your model back"*. Taking it back while the
run is still going is an order — the watchdog will not lend to that run again,
though a second trainer that starts alongside it still gets the GPU.


## 17. Study sessions (pomodoro)

```json
"focus": {"block_min": 25, "break_min": 5, "halfway": true, "max_blocks": 4,
          "music": "pause", "playlist": ""}
```

Say *"study session biosensors"*, *"start a fifty-minute focus session"*,
*"pomodoro"* or *"25 minute study session for signals with a ten minute break"*.
He confirms ("25 minutes on biosensors, sir; I'll call the halfway mark and the
break"), says "Halfway, sir." in the middle of any block of ten minutes or more,
"Time for a break, sir; that's block 1 done. Five minutes off." at the end of it,
starts the break timer himself, and "Break's over, sir. Block 2." after it. It
repeats until you say *"end the session"* (or *"I'm done studying"*) — "Session
over, sir: three blocks of 25 minutes." — or until `max_blocks` blocks are done
(0 = only when told). *"How long left?"* answers from the running block or break.

- `music`: `pause` pauses Spotify for the block and resumes it for the break;
  `playlist` plays `playlist` (a playlist name) for the block and pauses it for the
  break; `off` leaves Spotify alone. No Connect device or no linked account is
  simply a session without music — he never apologises mid-study.
- The blocks are timekeeper items (silent ones: he speaks the session's own
  lines, not "your timer is up"), so a session survives a restart: a block that
  came due while the app was down moves into its break at boot; one missed by more
  than an hour is closed with the count. State: `~/.aiws_trainer/jarvis_memory/focus_session.json`.
- *"Any timers running?"* lists them as "focus block 1 in 20 minutes".
- There is no do-not-disturb: heads-ups and reminders still speak during a block.

## 18. Lecture notes by voice

```json
"lecture": {"window_s": 20}
```

*"Notes for biosensors"* (also *"take notes for …"*, *"lecture notes on …"*) opens a
capture. Everything you say to him after that is appended as a timestamped line to
`~/Documents/Jarvis Docs/notes/biosensors-2026-09-03.md` (the first folder in
`docs.paths`, so the next reindex makes it searchable — *"what did I write about
impedance?"* goes through `ask_docs`) and to the notes store tagged with the course.
*"End notes"* closes it — "Notes closed, sir: 12 lines for BIOSENSORS." — and kicks
the reindex. With a Canvas token set the course name is tidied against your roster;
otherwise it is filed as you said it.

**This is not a hands-free microphone.** Each line is one capture: say the wake
word, say the sentence. After every noted line the mic re-opens without the wake
word for `window_s` seconds (20; capped at 30 by the recorder) and closes quietly if
you say nothing — so a run of sentences flows, but a long stretch of listening
ends the window and the next note needs "Jarvis" again. Lines are not read back
(that would talk over the lecture), they do not enter the conversation memory, and
while notes are open nothing else routes until "end notes" — the same rule as
dictation. Speaker verification still gates the mic.

## 19. How he behaves now (the 2026-08-30 set)

- **Follow-ups without the wake word** — after an answer the mic stays open
  `followup_window` (4 s) seconds; say "…and Tuesday?" straight away. Nothing said → it
  closes quietly. Every follow-up is still speaker-verified.
- **He remembers the conversation** — the last few exchanges (10 min) go to the model, so
  "what about tomorrow?" follows a calendar question.
- **Interrupt him** — say the wake word while he is talking ("Jarvis, stop"). `barge_in`
  keeps the wake word live during speech; his own voice cannot wake him (the speaker gate
  scores it at −0.03..−0.06). Echo cancellation: `scripts/audio/aec-install.sh`.
- **"Say that again, sir?"** — a garbled transcription (confidence gate) is asked again
  instead of being routed.
- **First-wake briefing** — after the first thing you say each day past `briefing.after`
  (06:00) he gives the briefing; `briefing.on_first_wake` turns it off.
- **Good-night preview** — with briefings on, "good night" reads tomorrow in one breath
  and offers a wake-up alarm when the first event is early ("Shall I wake you at
  7:00 am, sir?" — say yes). "What does tomorrow look like" asks for it any time.
- **The week ahead** — "how's my week looking?": calendar, Canvas and reminders day by
  day; he names the heavy day and the clear ones.
- **Briefing preferences** — "no news in the morning", "put the sports back in my
  briefing", "be briefer" / "the full briefing" all stick (section 7).
- **Meeting heads-up** — "BIOSENSORS in ten minutes, sir" (`calendar.heads_up_min`).
- **Deadline heads-up** — "Lab 3 report for BIOSENSORS is due in 3 hours"
  (`canvas.heads_up_hours`; needs the Canvas token, §13) and, at 7 pm, "Midterm 1 for
  BIOSENSORS is tomorrow at 9:00 am" for anything exam-like on Canvas or the calendar.
- **Guests** — a clear wake word in another voice gets "I only answer to Hunter, sir."
- **He learns your voice** — a confident match joins the voiceprint (at most every 10 min).
- **"Run diagnostics"** — uptime, models, today's turns and median wait, memory, GPU.
- **Streamed replies** — the first sentence speaks while the rest generates (`stream_replies`).
- **"Anything wrong in your log?"** — see the next section.

## 18. Log triage: "anything wrong in your log?" / "why was that slow?"

You are Jarvis's developer, so the fastest bug report is Jarvis reading his own
log. Nothing to configure.

- **"Anything wrong in your log?"** (also "check your log", "any errors in your
  log today", "what's in your log", "log triage"). He clusters the WARNING and
  ERROR lines in the last 400 lines of `/tmp/vss_voice/jarvis.log` by logger and
  message (numbers, paths and ids folded, so two "watchdog fired after 45s / 61s"
  lines are one thing), drops the known noise (the tools-registry budget warning,
  the memory migrated-dir notice — the list is `NOISE` in `jarvis/logtriage.py`),
  and speaks two sentences: "Three things in the log, sir: 'audio processing
  failed' from app five times, 'speaker verify FAILED SHUT' from speaker once
  and one more on the card, last at 13:20. On the ledger, two turns waited over
  5 seconds." The text card lists every cluster with its count, last time and
  the traceback that followed it. A clean tail gets "Nothing wrong in the log,
  sir; the last 400 lines are clean."
- **"Why was that slow?"** (also "why did that take so long", "what took so
  long", "where did the time go", "break down that turn"). From the last real
  record in `turns.jsonl` (the ledger `jarvis/turnclock.py` writes; a closed
  follow-up window does not count, and the record may be hours old): "The last
  turn waited 6.3 seconds, sir: 2.7 seconds silence before the recorder stopped,
  0.7 seconds transcribing, 2.9 seconds working out the answer; the answer was
  the slow part. The energy timer stopped it, not the voice detector." The
  answer stage is the model and its tools together — the ledger has no mark
  between them. A turn that got no answer says why (uncertain, rejected…).
- The log line has no date, so the window is the tail of the file, never "the
  last N minutes" (it survives midnight and a quiet evening alike). The same
  clustering now feeds the local model's "recent errors" context slice.

## 18. Long-term memory that understands you (fully local)

*"Remember that my dentist is Dr Patel"* stores a fact; weeks later *"who's my dentist?"*
or *"what did I say about the thesis last week?"* finds it however you phrase it. Facts
live in `~/.aiws_trainer/jarvis_memory/facts.json` as before, and each one is also embedded
with Ollama's `nomic-embed-text` (the same local model the documents tool uses) into a
chromadb index beside it (`facts_index/`). Every turn, the model is shown the few facts that
bear on what you just said — not the last five — so a fact you stored months ago still
surfaces when it matters, and nothing is shown when nothing is close. *"Recall the thesis"*
/ *"what did I say about the move yesterday"* answer directly; a time phrase (last week,
yesterday, in the last three days) limits it to facts stored since then.

Nothing leaves the machine. If chromadb or Ollama is unavailable Jarvis falls back to the
old substring search and says nothing about it (one line in the log). `memory.semantic:
false` turns the index off. The embedder is kept loaded (`keep_alive: -1`, ~270 MB) so a
question after a quiet spell never pays the 7 s cold load; the first start after this
update indexes your existing facts in the background.

## 19. People: who "my advisor" and "Mom" are

*"My advisor is Dr Peyrovi, email hp@tamu.edu"* / *"my mom is Linda"* / *"remember that my
TA is Sam Ortiz, his email is sam@tamu.edu"* go into `people.json` beside the facts. From
then on *"who's my advisor?"* answers straight away, *"any email from my advisor?"* filters
the mailbox by that address, and *"add lunch with Mom tomorrow at noon"* lands on the
calendar as lunch with her name. The people block is shown to the model on every turn, so
the local model can connect "my advisor" to a name in anything else you ask.

A sentence is taken as a contact only when it plainly is one — an address, a title (Dr,
Prof, Mr…), a relation word (advisor, TA, mom, dentist, landlord…) or a capitalised
full name — so *"my favourite colour is blue"* is still just conversation. Edit or remove an
entry by hand in `~/.aiws_trainer/jarvis_memory/people.json` (alias → name / email /
relation).

## 20. The activity journal and "recap my day"

Everything Jarvis does with you is journaled, one JSON line per event, in
`~/.aiws_trainer/jarvis_memory/journal/YYYY-MM-DD.jsonl`: each exchange in full, each tool
call, each finished or failed Claude task, and the window you are working in (sampled every
`journal.window_interval_s` seconds, written only when it changes, never while the screen is
locked or there is no focused window). *"Recap my day"*, *"what was I doing before lunch?"*,
*"what did I get done this afternoon?"*, *"what have I been working on the last two hours?"*
and *"what did I do yesterday?"* read the journal back: the local model speaks a recap of at
most four sentences and the full hour-by-hour digest appears on a card. Files older than
`journal.keep_days` (90) are pruned at start; `journal.enabled: false` stops the window
sampler (exchanges and tool calls are always journaled). Nothing leaves the machine.
## 21. Explain a document, then have it read (fully local)

*"Explain the biosensors lab handout"* / *"summarize the CS101 syllabus"*. The name is
matched against the files in `docs.paths` (and cwd, `~`, `~/Jarvis`), so say the title, not
the file name. He acks ("Let me have a look at the biosensors lab handout, sir."), the local
model reads the first ~12 000 characters and, twenty seconds or so later, he gives a
two-sentence lead aloud and puts the fuller paragraph on screen. Say *"read it to me"* and
he reads the actual text in three-minute parts (*"continue reading"* for the next). PDFs
come through `pdftotext`, `.docx` through the standard library; *"read file handout.pdf"*
reads one directly. Nothing leaves the machine. If he says he can't find it, the name
needs at least most of the words in the file name (`Biosensors_Lab_Handout.pdf` answers to
"biosensors handout", not "the lab report").

## 22. Quiz mode (flashcards from your documents)

*"Quiz me on chapter three"* / *"test me on the syllabus"*. He pulls the matching chunks
from the documents index (a file name or a "chapter N" heading first, the embedding search
after), has the local model write `quiz.questions` (5) short questions, files them as
flashcards under `~/.aiws_trainer/jarvis_memory/flashcards.db` and asks the first. Answer in
the follow-up window (no wake word); *"skip"* or *"I don't know"* reveals the answer;
*"stop the quiz"* gives the tally. Right answers move a card up a Leitner box (due again in
1, 3, 7 then 14 days); wrong ones come straight back to box one, so *"review my
flashcards"* the next day asks exactly what you missed. Short answers are graded by
matching words and numbers; the model only judges the unclear ones, and when neither can
tell he names the answer and moves on without a mark. `quiz.chunks` (6) is how much study
text one round reads. Everything runs on the Spark.

## 25. Git standup ("what did I do today?")

*"What did I do today"* / *"what did I change today"* / *"what did I work on
yesterday"* / *"standup"* / *"yesterday's standup"* — with or without the wake
word. Jarvis walks every project cleared for Claude (`claude.allowed_dirs`, the
children of `claude.projects_root` when that folder exists) plus `~/vss_env`,
reads each repository's commits for the day, what is still uncommitted and how
far ahead of `origin` it is, adds the Claude Code sessions whose transcript was
touched that day (their titles, scratchpads and probes left out), and answers in
two sentences — *"Today you made 4 commits in Jarvis and haymaker-digest, sir,
with 3 files in Jarvis still uncommitted. Claude had 2 sessions today, on the
standup feature and the TTS shootout."* — with a card of the commit subjects per
repository. Nothing leaves the machine and no model turn is involved, so it
still answers while the GPU is lent out (section 16). Nothing to configure:
the repo list follows section 9. *"Git status"* now reports the Jarvis
repository (it used to be hard-wired to the VSS tree).

## 26. Faster calendar, weather and time-in-city answers

Nothing to set up. When the router already knows the tool — a read-only calendar
question, a weather question, "what's the time in London" — the commander forces the
tool call and only the render turn runs, so the answer lands ~1 s sooner. It stands
down for a write ("book a meeting"), a second clause ("… and what's the weather") or a
lowercase city, which take the full loop as before. Look for `route short-cut:` in the
log.

## 27. Corrections: "no, I said …"

Say "no, I said …", "I meant …", or "not X, Y" (with the comma) — typed, on Discord, or
by voice within a minute of the last turn. Jarvis stops talking, drops the misheard
exchange from his short-term memory and answers what you meant. Pairs are kept in
`~/.aiws_trainer/jarvis_memory/corrections.json`.

```json
"corrections": {"learn_vocab": false}
```

`learn_vocab: true` also adds new capitalised words from a correction ("Peyrovi") to the
Whisper vocabulary prompt (`~/.aiws_trainer/voice_vocab.txt`), so the next attempt decodes
the name. Off by default: Whisper's casing on a misheard name is itself a guess.

## 28. Teaching the intent gate: "that was for you"

If a command was dropped as background chat (nothing happens), say "Jarvis, that was for
you" within 20 s: he runs it and remembers the words as his. If he answered something you
said to someone else, "that wasn't for you" / "not you" within a minute stops him and
remembers that too. Labels go to `~/.aiws_trainer/intent_log.json` (what the classifier
reads) with an audit line in `~/.aiws_trainer/jarvis_memory/feedback.jsonl`. "That was for
you" also answers the spoken "Was that for me?" window.

## 29. Read-back before a bulk cancel or list wipe

```json
"confirm": {"read_back": true, "shaky_logprob": -0.7}
```

"Cancel all my alarms" with more than one alarm, or "clear my list" with more than one
to-do, is read back — "Cancel all three alarms, sir?" — and waits for a yes through the
follow-up window; "no" or any other subject drops it, and so does a yes a minute later. A
transcript scoring under `shaky_logprob` is read back even for one item. A single "cancel
the timer" never is. `read_back: false` turns it off.

## 30. Names and the listening vocabulary (Whisper + TTS)

Whisper is primed before every utterance with a prompt built from your own
world — nothing to enable. In priority order (the tail falls off first, near
Whisper's ~224-token prompt window):

1. `~/.aiws_trainer/voice_vocab.txt` — your manual vocabulary (the
   "Edit vocabulary…" button, one term per line) plus words learned from
   corrections (`corrections.learn_vocab`).
2. `~/.aiws_trainer/voice_names.txt` — names you teach by voice, newest first.
3. Names you taught the voice to say ("pronounce X as Y").
4. The built-in assistant seed (Jarvis, Canvas, Ollama, timers…). The old
   warehouse list (AGV, forklift, pallet) is gone.
5. Calendar event titles from the disk cache (BIOSENSORS and friends).
6. Canvas course names already cached by the Canvas tool — never a fresh fetch.

Rebuilt at most once a minute, except that teaching a name applies at once.

Teach a name in both directions in one sentence:

> "Jarvis, my advisor's name is spelled P-E-Y-R-O-V-I, say it pay-ROH-vee"

The letters go to the names file (so Whisper *hears* Peyrovi), the "say it …"
clause goes to the TTS dictionary (so he *says* pay-ROH-vee), and the reply
echoes the spelling back — "Noted, sir: P-E-Y-R-O-V-I. Peyrovi." — so a
misheard letter is caught on the spot. Simpler forms:

- "the name is spelled P E Y R O V I" (names file only)
- "Jarvis, add Librespot to your vocabulary"
- "pronounce Peyrovi as pay-ROH-vee" (the TTS half alone, as before)

Both files are plain text and safe to edit by hand.

## 18. Voice latency: the first-audio mark and streamed Fish playback

Nothing to configure. Two things changed on 2026-08-30 in `jarvis/tts.py`:

- **The `wait` in the turn ledger is now honest.** `SpeakingState(active=True)` — the
  "audio" mark `turns.jsonl` and "run diagnostics" report against — used to go out
  before the first chunk was even rendered, so a logged `wait 1.32 s` was really
  ~1.7–1.9 s to the ear. It now goes out when the player process is spawned. Expect the
  numbers in `/tmp/vss_voice/turns.jsonl` to read HIGHER after this restart; they are
  the same turns measured properly. Tune against the new ones.
- **Fish plays while it renders.** With `tts_engine: fish`, the API's bytes go straight
  into `paplay`'s stdin as they arrive (and into the speech cache file at the same time),
  so its ~186 ms time-to-first-audio is finally what you hear. If the connection drops
  mid-sentence you lose the tail of that one chunk; the next chunk comes from the local
  F5 fallback as before. To compare by ear, set `jarvis.tts.FISH_STREAM_PLAYBACK = False`
  (module constant) and restart. F5, XTTS and edge are unchanged (F5's sidecar writes a
  file per request and cannot stream).

## 19. Steering a read-aloud (skip / back / pause / go on)

Nothing to configure. While he is reading ("read the clipboard", "read file
~/syllabus.md", …) the transport words belong to the reading:

| say | he does |
|---|---|
| "skip" / "skip that" / "skip ahead" | cuts the chunk being spoken; the next one follows at once |
| "back" / "go back" / "previous" / "say that again" | re-reads the previous chunk, then carries on |
| "pause" / "hold on" / "hang on" / "wait" | "Paused, sir." — the rest is held |
| "go on" / "carry on" / "resume" / "keep going" | picks up from the chunk that was cut (it restarts; a chunk is ~25 s) |
| "quiet" / "stop" | ends the reading, as before |

Say them with or without the wake word — `barge_in` keeps the wake word live while he
speaks. The words are only his while a part is actually being spoken or the reading is
paused; between parts (waiting for "continue reading") and at every other time "pause",
"skip" and "back" mean what they always did — Spotify's transport and, with the prefix,
the media keys and the previous window. While paused, "skip" and "back" move the cursor
without speaking, so you can step to the bit you want and then say "go on".

- **He starts transcribing before you have finished pausing** — once the VAD has heard
  0.3 s of silence the clip is decoded while the 0.8 s endpoint silence runs out; if you
  said nothing more, that result is used the moment the recorder stops (about 0.4-0.5 s
  off a short question). The `turn:` line says `decode=speculative` when it was reused;
  `listening.speculative_stt: false` turns it off.
- **"Sir?" instead of silence** — after a wake word that captured nothing usable he says
  "Sir?" and listens again without the wake word; a clip that was not your voice, or a
  second garbled one in a row, gets a low beep instead. Never in a follow-up window, never
  from the mic button, and at most once every `listening.nudge_cooldown_s` (30) seconds;
  `listening.nudge: false` turns it off.

### Listening options (`listening` in assistant.json)

```json
"listening": {"speculative_stt": true, "nudge": true, "nudge_cooldown_s": 30}
```

Both are fully local and cost nothing new: the speculative decode is the same whisper pass
moved earlier (a pause that turns out to be mid-sentence wastes one decode, bounded like a
live-transcript preview), and the cue is a prewarmed line or a generated beep
(`/tmp/vss_voice/beep_nudge.wav`). Tune from `turns.jsonl`: a `no_audio` / `empty` /
`rejected:*` outcome followed by an `audio` turn within the window is a cue that worked.

---

## 18. Quiet hours and do not disturb

Jarvis holds his *proactive* lines — memory warnings, reminders and timers, meeting
heads-ups, anything from the hooks narrator — while you are busy, and reads them back
afterwards: *"While you were busy, sir: two reminders and one warning. …"* Answers to
what you ask, alarms and Claude's permission questions are never held.

- *"Do not disturb for an hour"* / *"don't disturb me until seven"* / *"hold my
  notifications for two hours"* / *"I'm busy for the rest of the day"* — a one-off window
  (`quiet.dnd_until`, survives a restart).
- *"Quiet hours from eleven to seven"* — every night (`quiet.hours`, 24 h `"HH:MM"`,
  overnight windows wrap); *"turn off quiet hours"*, *"what are my quiet hours"*.
- **Calendar** — a timed event running now whose title contains one of
  `quiet.calendar_keywords` (`class`, `exam`, `meeting`, `busy`) is quiet automatically;
  `quiet.calendar: false` turns that off.
- **Away** — with presence set up (section 19) he also holds while your phone is off the
  Wi-Fi; `quiet.hold_when_away: false` turns that off.
- *"I am free"* / *"what did I miss"* — ends the current window early and reads what was
  held; *"are you on do not disturb"* says why he is quiet.

Desktop banners are silenced in-process for the window (nothing is written to
gsettings, so a crash cannot leave the desktop mute). The first-wake briefing waits for
the window to close. A bare *"quiet"* is still the barge-in, not a DND request. The
backlog is capped at twelve lines.

```json
"quiet": {"hours": {"start": "23:00", "end": "07:00"}, "dnd_until": 0, "free_until": 0,
          "calendar": true, "calendar_keywords": ["class", "exam", "meeting", "busy"],
          "hold_when_away": true}
```

## 19. Presence (is anyone home?)

Give him your phone's Wi-Fi address and he knows whether you are in. Find it in the
router's client list or on the phone (Settings › Wi-Fi › the network › IP address; iPhones
use a per-network private MAC, which is fine — it is stable for that SSID). A DHCP
reservation keeps the IP steady; otherwise set `phone_mac` and he pings whatever the ARP
table has for it.

```json
"presence": {"enabled": true, "phone_ip": "192.168.50.42", "phone_mac": "",
             "away_after_min": 12, "poll_s": 60}
```

Every `poll_s` he reads `ip -4 neigh` (REACHABLE = present) and pings once when the
entry is stale or missing — no sudo, no Bluetooth (a phone does not stay connected to a
Linux host). "Away" needs `away_after_min` of silence first, because iPhones drop off
Wi-Fi power-save for minutes at a time; "home" is immediate. Leaving is silent; on the
return he says *"Welcome back, sir."* and reads anything held while you were out. Until
the first probe answers he assumes you are home. Check the log for
`presence: home` / `presence: away`.
## 23. Nightly self-review ("how did yesterday go")

Every quarter hour a small thread checks whether a day has ended without a review
and, if so, reads his own `jarvis.log` and `turns.jsonl` for that day and files a
digest to `~/.aiws_trainer/jarvis_memory/reviews/<date>.json` (kept 60 days — the log
itself lives in `/tmp` and is wiped at boot). Nothing to configure; nothing leaves the
box; no model call (every number is kept by construction).

- **At first wake** — before "Your briefing for today, sir." he says two sentences:
  *"Yesterday: 41 turns, median wait 1.4 seconds, worst 6.2. I dropped two clips of
  yours at the speaker gate, the turn watchdog let go once and one tool call failed,
  sir."* A day with nothing in it is not mentioned.
- **On demand** — *"how did yesterday go"*, *"what went wrong yesterday"*,
  *"yesterday's review"*; *"how's today going"* reads the open day so far.
- **Discord** — when section 6 is configured, the full table (turns, waits, aborts,
  uncertain / ignored, speaker-gate rejections, wake words refused, watchdog releases,
  tool exceptions, TTS fallbacks, model reloads, boots, errors / warnings, the top error
  lines) is posted when the digest is filed. No desktop banner.

What counts as "went wrong" comes from the exact lines the modules log (see
`COUNTERS` in `jarvis/dayreview.py`); a line from another process writing the same
file before the app's boot marker is ignored, so a stray test traceback cannot
show up as "a tool call failed".

## 24. Ask him from a shell (`jarvis`, SSH, tmux, scripts)

The running app listens on `/tmp/vss_voice/command.sock` (0600, next to
`approvals.sock`). `scripts/jarvis` sends one line and prints what he answers:

```bash
ln -s ~/Jarvis/scripts/jarvis ~/.local/bin/jarvis      # once
jarvis "what's due this week"                            # answered aloud AND printed
jarvis -q "how did yesterday go"                         # text only; soundbar silent
jarvis --status                                          # the "run diagnostics" line
jarvis --json "timer 5 minutes"                          # raw JSON lines (scripts)
ssh spark jarvis 'set an alarm for 7'                    # from the laptop or the phone
```

Replies go to stdout, status lines (`[info] Clock`) to stderr. Exit 0 means a reply
came back, 2 means Jarvis is not running, 3 means the turn ended without one — a
Claude task, for instance, answers with the acknowledgement and gets on with it. A
question you send this way goes through the same commander as the typed box (no
wake word, no intent guess), shows in the transcript, and is not stored in the
typed-box history. `-t 300` waits longer than the default 90 s. Everything stays on
the box: a UNIX socket, no port, no daemon beyond the app itself.

## What Jarvis says when something is missing

| section not set up | is_configured needs | spoken line |
|---|---|---|
| `home_location` | `lat` and `lon` numbers (the line is used only when the IP lookup is off or fails; with `location_lookup` true he looks it up) | "I'll need your home location set up, sir; the notes are in docs/assistant-setup.md." |
| `google_ical` | at least one `https://…` URL in `google_ical_urls` | "I'll need your Google calendar link set up, sir; the notes are in docs/assistant-setup.md." |
| `icloud` | `apple_id` and `app_password` | "I'll need your iCloud calendar set up, sir; the notes are in docs/assistant-setup.md." |
| `gmail` | `address` and `app_password` | "I'll need your Gmail app password set up, sir; the notes are in docs/assistant-setup.md." |
| `discord` | `bot_token` and `channel_id` | "I'll need the Discord bot set up, sir; the notes are in docs/assistant-setup.md." |
| `claude` | the `claude` command on PATH (or `JARVIS_CLAUDE_BIN`) and a non-empty `allowed_dirs` | "I'll need the Claude command line set up, sir; the notes are in docs/assistant-setup.md." |
| `spotify` | `client_id` and `client_secret`, then one `--spotify-login` | "I'll need spotify set up, sir; the notes are in docs/assistant-setup.md." / "Spotify isn't linked yet, sir; run me with --spotify-login once." |
| briefing disabled | `briefing.enabled` true | "The morning briefing is switched off, sir; the toggle is in settings under Briefing." |

Each line is one sentence, spoken verbatim (no model turn), and the same
line comes back until the section is filled in.

## Command line

```bash
cd ~/Jarvis
~/vss_env/bin/python -m jarvis.app                      # normal start (or the desktop icon)
~/vss_env/bin/python -m jarvis.app --install-autostart   # autostart entry + never-sleep, then exit
~/vss_env/bin/python -m jarvis.app --spotify-login       # one-time Spotify link, then exit
~/vss_env/bin/python -m jarvis.autostart --status        # is the login entry installed?
~/vss_env/bin/python -m jarvis.tools.spotify --status    # configured / linked / token path
~/vss_env/bin/python -m jarvis.ask "what's due today"     # ask the running app (scripts/jarvis)
~/vss_env/bin/python -m jarvis.ask --status              # the diagnostics line, no speech
```

Both `--install-autostart` and `--spotify-login` do their job and exit
without opening the window, so they are safe to run while Jarvis is already
up. A plain start while he is already running does not open a second window:
it raises the one that exists (and if it is still booting, waits for it and
toasts "Starting up, sir…" rather than doing nothing).

## Checking without leaking anything

```bash
cd ~/Jarvis && ~/vss_env/bin/python - <<'EOF'
from jarvis.assistant_config import AssistantConfig
cfg = AssistantConfig.load()
print(cfg.path, oct(cfg.path.stat().st_mode & 0o777))
print("missing:", cfg.missing_sections())
import json; print(json.dumps(cfg.redacted(), indent=2))   # secrets show as •••
EOF
```
