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

Reading is read-only by construction: `SELECT INBOX` read-only, `BODY.PEEK`,
so nothing is ever marked read. Message bodies are never logged. A Google
Workspace account uses the same host.

Jarvis can also **send** a file (section 81 below). Sending uses the same
app password over SMTP and adds one key:

```json
"gmail": {"address": "<you@gmail.com>", "app_password": "<16-char-app-password>",
          "imap_host": "imap.gmail.com", "smtp_host": "smtp.gmail.com"}
```

Leave `smtp_host` out and it is worked out from `imap_host`
(`imap.x` -> `smtp.x`). Nothing is ever sent without you saying yes out
loud first.

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
be absent): `"alerts": {"desktop": true, "discord": true, "claude_hooks": true}`
(`claude_hooks` is the spoken narration of your own Claude Code terminals, section 39).

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
event and where, Canvas work due within a day (once there is a Canvas source at
all: `canvas.token` OR a Canvas calendar feed, §13), open to-dos and the alarm
that is set. If the first event starts by
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
*"when's the biosensors midterm?"* The token is redacted from every log and repr.

### If your university blocks personal access tokens

Ours does, and no token can be minted at all — but Canvas will still hand you a **calendar
feed**, and it carries the same coursework. In Canvas: **Calendar → Calendar Feed**, copy
the `.ics` link, and add it to `google_ical_urls` alongside your other subscriptions
(section 4). Nothing else to configure: Jarvis recognises Canvas assignments by the bracket
of course codes Canvas appends to every title —

```
Lab 1: Introduction to the AD2 SDK [BMEN-427:501,502,503,504,BMEN-627:600,...]
HW#1 [MSEN-222:599,M99]
```

— and reads them as coursework (an all-day one is due 11:59 pm that day, which is what
Canvas means by it). *"What's due this week"*, *"when's my next exam"*, the briefing's
**Due** line, the week forecast and the deadline heads-ups below all work from the feed
alone. Two things the feed cannot carry: whether you already handed something in (a row
stands until its due time passes) and the course's spoken NAME, so it says "BMEN 427"
where a token would say "BIOSENSORS". **Grades and announcements** are the only features
that genuinely need the token.

With neither a token nor a Canvas feed: "I'll need a Canvas access token set up, sir; the
notes are in docs/assistant-setup.md." — and only then.

### Deadlines in the briefing, and a heads-up before each one

With a Canvas source in place (token or feed), the briefing gains a **Due** section (the next two days: *"Due:
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
repeats one (`~/.aiws_trainer/jarvis_memory/deadlines_state.json`). The calendar feed feeds
this too, so the Due line and the heads-ups work without a token; with BOTH, the token's
reading wins a duplicate, since it alone knows whether the work went in. With neither, all
of this is silent: no Due line in the briefing, no nagging about the missing token.

### Exams: "when's my next exam?"

*"When's my next exam?"*, *"when is my next quiz"*, *"how long until the biosensors
midterm?"*, *"how many days until my final?"* are answered instantly, without the model,
from Canvas (30 days ahead), the Canvas calendar feed (a whole term — the calendar cache
only keeps a fortnight, and a final is further out than that) and the calendar cache
(14 days) — so this works before the token exists, or without one ever, and an exam that
only the iCloud "Canvas" subscription carries is found too. Anything titled exam / midterm / final is an exam, a quiz is a quiz;
*"next exam"* never answers with a quiz, and *"final"* / *"midterm"* must be in the title.
The briefing adds an **Exam** countdown (*"Midterm 1 for BIOSENSORS, in 6 days, Tuesday at
9:00 am"*), and at 7 pm the evening before he says *"Midterm 1 for BIOSENSORS is tomorrow
at 9:00 am"*. With Canvas read and nothing found: *"Nothing that looks like an exam on the
books, sir."* Only with no token, no Canvas feed and nothing on the calendar does the
question go to the model, which reaches `canvas_due` and its setup line.

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
A screenshot of `DISPLAY=:1` goes to a local vision model through Ollama; the answer is
spoken directly. Nothing is written to disk unless `JARVIS_DEBUG_SCREEN=1`.

**Which model.** `screen.model` defaults to `""`, meaning *the chat model* (`local_model`,
gemma4:26b) — and that is the only free choice on this box: `OLLAMA_MAX_LOADED_MODELS=1`,
so naming any second vision model here evicts the chat model and costs ~7 s on the next
spoken turn. gemma4 carries a clip projector and reports the `vision` capability, so the
eyes are the model that is already resident. Measured 2026-09-01: 1.3 s per question once
warm, ~8 s on the first one (the projector loads into the running model).

`llama3.2-vision:latest` is pulled here (7.8 GiB) but **ollama 0.33.1 cannot load it at
all** — `/api/chat` answers 500 `unknown model architecture: 'mllama'`. If `screen.model`
still names it, the tool tries it once, remembers the refusal for ten minutes, falls back
to the chat model and logs `set screen.model … ("" = the chat model)`; the answer still
arrives. Loading mllama would need a newer ollama build. Any model ollama reports with the
`vision` capability works here (`ollama show <model>`); a model that is missing, text-only,
or unloadable is said aloud by name instead of a blanket "my vision model isn't answering".

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
- A block is a do-not-disturb window (`focus.dnd`, on by default): heads-ups,
  deadline warnings and the hooks narrator are held and read back as the usual
  catch-up digest when the break starts. See section 56.

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
  scores it at −0.03..−0.06). Echo cancellation is packaged but not enabled or measured —
  `docs/echo-cancellation.md` before `scripts/audio/aec-install.sh`.
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

## 20. Log triage: "anything wrong in your log?" / "why was that slow?"

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

## 21. Long-term memory that understands you (fully local)

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

## 22. People: who "my advisor" and "Mom" are

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

## 23. The activity journal and "recap my day"

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

## 24. Explain a document, then have it read (fully local)

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

## 25. Quiz mode (flashcards from your documents)

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

## 26. Git standup ("what did I do today?")

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

## 27. Faster calendar, weather and time-in-city answers

Nothing to set up. When the router already knows the tool — a read-only calendar
question, a weather question, "what's the time in London" — the commander forces the
tool call and only the render turn runs, so the answer lands ~1 s sooner. It stands
down for a write ("book a meeting"), a second clause ("… and what's the weather") or a
lowercase city, which take the full loop as before. Look for `route short-cut:` in the
log.

## 28. Corrections: "no, I said …"

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

## 29. Teaching the intent gate: "that was for you"

If a command was dropped as background chat (nothing happens), say "Jarvis, that was for
you" within 20 s: he runs it and remembers the words as his. If he answered something you
said to someone else, "that wasn't for you" / "not you" within a minute stops him and
remembers that too. Labels go to `~/.aiws_trainer/intent_log.json` (what the classifier
reads) with an audit line in `~/.aiws_trainer/jarvis_memory/feedback.jsonl`. "That was for
you" also answers the spoken "Was that for me?" window.

## 30. Read-back before a bulk cancel or list wipe

```json
"confirm": {"read_back": true, "shaky_logprob": -0.7}
```

"Cancel all my alarms" with more than one alarm, or "clear my list" with more than one
to-do, is read back — "Cancel all three alarms, sir?" — and waits for a yes through the
follow-up window; "no" or any other subject drops it, and so does a yes a minute later. A
transcript scoring under `shaky_logprob` is read back even for one item. A single "cancel
the timer" never is. `read_back: false` turns it off.

## 31. Names and the listening vocabulary (Whisper + TTS)

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

## 32. Voice latency: the first-audio mark and streamed Fish playback

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

## 33. Steering a read-aloud (skip / back / pause / go on)

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

## 34. Quiet hours and do not disturb

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
- **Away** — with presence set up (section 35) he also holds while your phone is off the
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

## 35. Presence (is anyone home?)

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

**Faster, if you want it: a room sensor.** The phone leg is slow and lies when the phone
sleeps. An ESP32 + LD2410 mmWave module (~$25, an evening) sits in the room and answers in
seconds — presence, not motion, so it does not decide you left because you sat still. It
composes with the phone rather than replacing it: the room seeing someone beats a sleeping
phone, the room seeing nobody never makes you away while the phone answers, and an
unplugged sensor is treated as no sensor at all. Off until configured. Parts list, wiring,
the ready-to-flash ESPHome file and a no-hardware test mode:
[docs/room-sensor.md](room-sensor.md).

## 36. Nightly self-review ("how did yesterday go")

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

## 37. Ask him from a shell (`jarvis`, SSH, tmux, scripts)

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

## 38. Local voice: the F5 sidecar as a service

F5-TTS is the local voice (blind-tied the hosted Fish voice, see
`~/voice-training/HANDOFF.md`) and the fallback whenever `tts_engine` is `fish`.
It runs as a sidecar in its own venv (`~/.local/share/jarvis-f5/venv`), and the
model takes up to ~180 s to become resident, so it should be up before anything
needs it. `jarvis-f5.service` is a user unit (no sudo) that keeps it resident
across Jarvis restarts and reboots:

```bash
cd ~/Jarvis && scripts/setup_f5_service.sh        # copy the unit, daemon-reload, enable
systemctl --user start jarvis-f5.service           # the live cutover: do this yourself
journalctl --user -u jarvis-f5.service -f          # "f5: listening on ..." means ready
systemctl --user is-active jarvis-f5.service
```

The script never starts the unit: the first start puts a few GB onto a GPU that
shares its memory with everything else here (see the 28 Aug deadlock), so that
moment is yours. The unit uses the same paths as `jarvis/config.py` PATHS —
socket `/tmp/vss_voice/f5.sock`, clip `~/.aiws_trainer/jarvis_voice_ref_f5.wav`
and its transcript `.txt` — and recreates `/tmp/vss_voice` first because `/tmp`
is wiped at boot. Swap the reference clip → `systemctl --user restart jarvis-f5`
(the server derives the short-utterance duration floor from the clip at start).

What Jarvis does with it:

- **Any engine**: `tts.py` pings the socket before ever starting a sidecar and
  adopts a running one. While the unit is active (or still loading) Jarvis
  refuses to spawn its own copy and waits for the unit's socket instead — the
  server unlinks and rebinds the socket path on start, so two copies would
  fight over it.
- **`tts_engine: fish`**: the F5 fallback is warmed on a background thread at
  load, so an outage costs one chunk, not a cold start. If it cannot be warmed
  you get a warn status ("Local voice fallback (F5) is not running") and a log
  line — the hosted voice keeps working.
- **`tts_engine: f5`**: if the sidecar is unavailable Jarvis says so (status
  "F5 voice down — using XTTS (local)") and loads XTTS, the other local voice;
  only with no XTTS reference clip does it fall to edge, announced as the cloud
  voice it is. It never drops to the cloud quietly.
- **On quit**: a sidecar Jarvis spawned itself is left running (it makes the
  next launch warm) unless the unit has taken over, in which case the
  redundant copy is stopped.

Flipping the default engine is still a manual step: set `"tts_engine": "f5"`
in `~/.aiws_trainer/voice_settings.json` (or the Engine picker) and restart;
the canned lines are re-rendered on F5 by the normal prewarm, and the Fish key
becomes optional. The speech cache keys F5 and Fish audio separately.

## 39. Claude Code hooks (your own terminals)

The sessions you start yourself (the `~/.bashrc` tmux wrapper, any plain `claude`)
never pass through Jarvis, so a failing test run or a permission prompt in another
window went unnoticed. A small stdlib-only hook script,
`scripts/claude_hooks/narrate.py`, fixes that: Claude Code runs it on a few events
and it appends one line to the speak queue Jarvis already tails
(`/tmp/vss_voice/speak_queue.txt`). You hear:

- after a `pytest` / `ruff` / `mypy` / `npm test` … command: "3 tests failed in
  Jarvis, sir." / "Tests passed in haymaker digest, sir." / "The tests errored in
  …, sir." — the same verdicts the Jarvis-driven sessions get;
- on a permission prompt: "Claude needs your say-so in Jarvis, sir."; when Claude is
  idle waiting for you: "Claude is waiting on you in Jarvis, sir.";
- when a turn ends, "Claude has finished in Jarvis, sir." — but only if the turn ran a
  test command or took at least 45 s (Stop fires after every reply; a chat answer
  would otherwise become a tic).

It stays quiet when Jarvis is not running (`jarvis.pid` absent or stale), when the
session was started BY Jarvis (his launch line sets `JARVIS_DRIVEN=1`, and those
panes are already narrated from the transcript), and when you turn it off:

```json
"alerts": {"desktop": true, "discord": true, "claude_hooks": false}
```

Repeats are limited per session (30 s; 2 min for the idle prompt) through a tiny
state file under `~/.cache/jarvis/claude_hooks/`. Nothing is ever printed to the
hook's stdout, and it always exits 0, so it can never block or fail the CLI.

**Install (once, yourself — Jarvis never edits `~/.claude/settings.json`):**

```bash
~/vss_env/bin/python ~/Jarvis/scripts/claude_hooks/install.py             # merge
~/vss_env/bin/python ~/Jarvis/scripts/claude_hooks/install.py --dry-run   # show only
~/vss_env/bin/python ~/Jarvis/scripts/claude_hooks/install.py --uninstall
```

The merge is idempotent, keeps every other hook and setting in the file, and
writes `settings.json.bak-claude-hooks` beside it the first time. Restart your
Claude sessions afterwards. This is exactly what it adds (the `hooks` block; paste it
by hand if you prefer):

```json
"hooks": {
  "PostToolUse": [
    {"matcher": "Bash",
     "hooks": [{"type": "command", "timeout": 10,
                "command": "/home/hunterp/vss_env/bin/python /home/hunterp/Jarvis/scripts/claude_hooks/narrate.py"}]}
  ],
  "Notification": [
    {"hooks": [{"type": "command", "timeout": 10,
                "command": "/home/hunterp/vss_env/bin/python /home/hunterp/Jarvis/scripts/claude_hooks/narrate.py"}]}
  ],
  "Stop": [
    {"hooks": [{"type": "command", "timeout": 10,
                "command": "/home/hunterp/vss_env/bin/python /home/hunterp/Jarvis/scripts/claude_hooks/narrate.py"}]}
  ],
  "UserPromptSubmit": [
    {"hooks": [{"type": "command", "timeout": 10,
                "command": "/home/hunterp/vss_env/bin/python /home/hunterp/Jarvis/scripts/claude_hooks/narrate.py"}]}
  ]
}
```

`UserPromptSubmit` only stamps when the turn began (for the 45 s gate); without it,
"finished" is spoken only after a test run.

These hooks are the *outbound* half — your sessions telling you what happened.
The inbound half (a session asking Jarvis for your calendar, due dates or memory
with `jarvis -q "..."`) is section 59; panes Jarvis starts get it automatically,
your own terminals need four lines in `~/.claude/CLAUDE.md`.

---

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

## 40. New-grade watch ("a grade just posted")

Nothing to set up beyond the Canvas token (section 13). Once it is in place
Jarvis reads the course list every fifteen minutes and tells you when a
total moves:

> "Quiz 2 posted for BIOSENSORS, sir: the course is now 94.2% (A), up from 90%."

The first run after a fresh install is deliberately silent — it only records
the baseline, because reading out every course's standing at boot is a
monologue, not news. From then on only changes are spoken.

Naming the assignment ("Quiz 2") costs one extra read-only Canvas call, made
only when a total has actually moved. If Canvas is slow, the token is
read-limited, or the newest graded submission is more than three days old,
the name is dropped and the line is still spoken:

> "A grade posted for Circuits, sir: the course is now 74% (C), down from 81%."

```jsonc
"watch": { "grades": true }
```

Set `watch.grades` to `false` to switch it off. Without a Canvas token it is
silent anyway — it never opens a socket. At most three courses are read out
per tick; anything beyond that is left in the log
(`grep 'grades:' /tmp/vss_voice/jarvis.log`). The snapshot lives in
`~/.aiws_trainer/jarvis_memory/grades_state.json`; delete it to re-baseline.

Like every unprompted line it goes through the quiet policy (section 34), so
during quiet hours, a do-not-disturb, or a lecture it is held for the
catch-up digest, and it reaches Discord when you are out (section 35).

## 41. Important-person mail heads-up

Mail was only ever fetched when you asked, so an advisor's email sat unseen.
This watch reads the unread inbox every ten minutes and speaks a line for
anything from someone **in the people book**:

> "Mail from Dr. Villalobos, sir — re: thesis draft."

Two things must be in place:

1. A Gmail app password (section 5) — one account or several.
2. The people themselves, entered by voice:

> "Jarvis, remember that my advisor is Dr. Villalobos, email villalobos@tamu.edu"

Only people in that book announce. That is the point: a poller that read out
every unread message would be a nuisance within the hour, and the people book
is the one contact list you have already curated. An entry with only a name
still works — it matches the sender's display name — but an address is
sharper.

```jsonc
"watch": { "people_mail": true }
```

At most three messages are read out per tick, then "And four more from your
contacts, sir." Everything matched is remembered for fourteen days
(`~/.aiws_trainer/jarvis_memory/mailwatch_state.json`), so nothing is said
twice, including the ones the cap held back. With no mailbox configured, or
an empty people book, the tick does nothing and no socket is opened.

## 42. Keyword watch ("watch for anything about the biosensors project")

A list of words or phrases to follow across unread mail and Canvas
announcements. The deadline watch follows due times and the grade watch
follows scores; this one follows a topic.

```jsonc
"watch": {
  "keywords": ["biosensors", "REU application", "Villalobos"]
}
```

Edit `~/.config/jarvis/assistant.json`, and hits are spoken on the next tick
(fifteen minutes):

> "Mail about biosensors, sir: Meeting about biosensors, from Dr. Villalobos."
> "Canvas announcement about biosensors, sir: BIOSENSORS — Project groups posted."

Matching is whole-word, so `"ai"` does not fire on "again" and `"lab"` does
not fire on "collaboration"; a multi-word entry matches as a phrase. Matching
is case-insensitive. An empty list means the watch never runs — it does not
read the mailbox at all.

It needs the Gmail app password (section 5) and/or the Canvas token
(section 13), and each source runs on its own: with only Canvas set up, only
announcements are watched. At most three hits are spoken per tick; the rest
are marked seen and counted in the log, because a hit read out every fifteen
minutes is worse than one missed. Seen hits live in
`~/.aiws_trainer/jarvis_memory/keyword_watch_state.json` for fourteen days.

## 43. Per-course scoping (quiz and lessons stay in one subject)

Nothing to configure. Until now *"quiz me on electrode transducers in signals and
systems"* matched no file by name, so the last resort was a global embedding query
over every indexed document — and the biosensors notes talk about electrodes and
transducers too, so the questions came out of the wrong course and the flashcards
were filed under a topic they were not about.

The topic is now read as a **course** first: the whole utterance, then each tail
after *of / in / for / from / on / about*, is looked up on the Canvas roster
(`lecture.resolve_course`; with no token it is simply the words as said), slugified
the way lecture notes are named — `biosensors-2026-08-29.md` — and matched as a
whole token against the file names already in the index. On a hit only that
course's files are read, in reading order, with no embedding call at all.

- **What it covers well**: lecture notes (`"notes for biosensors"` writes the
  slug-named files this depends on) and any document you name after its course.
- **What it does not**: a PDF saved under its publisher's name carries no slug, so
  it still takes the global path — exactly as before, no worse.
- A slug under three characters never scopes, so "cs" cannot pull in "physics".
- Naming a chapter the course does not have falls back rather than answering about
  the wrong week.

## 44. "Teach me X" — a lesson, then a quiz on the same material

> *"Teach me week four of signals and systems."*
> "Week four covers sampling: a signal sampled at twice its highest frequency can be
> reconstructed exactly, sir. Say quiz me and I'll test you on it, sir."
> *"Quiz me."*
> "Let me put some questions together on week four of signals and systems, sir."

The chunks are retrieved **once**. The lesson explains them and keeps that exact
study text; taking the offer builds the quiz from the same string, so the questions
are provably about what was just said (asking "quiz me on ..." separately retrieves
afresh and may land on other text). The misses go to box one and Leitner brings them
back tomorrow.

- Say yes with *"quiz me"*, *"go on"*, *"yes"*; decline with *"no"*. Anything else
  drops the offer, and so does three minutes' silence.
- *"teach me how to ..."*, *"teach me a lesson"* are not lessons and go to the model.
- Requires the documents index (section on `docs.paths`) and the local model.

## 45. Exam-week study in the morning briefing

When Canvas or your calendar has an exam inside the next `briefing.study_days`
days, the briefing adds a line about the deck for **that course**:

> "Exam: Midterm 1 for BIOSENSORS, in 2 days, Tuesday at 9:00 am.
> Study: 14 cards due on your BIOSENSORS deck, 9 of them in box one.
> Shall we run ten now, sir?"

*"Yes"* deals those cards straight away; *"no"* declines; anything else drops the
offer. With no cards yet it says so instead — *"Nothing on your BIOSENSORS deck yet,
sir; say quiz me on BIOSENSORS and I'll build one."*

```json
"briefing": {
  "sections": { "study": true },
  "study_days": 5,          // how close an exam has to be
  "study_offer": true,      // false: the line, never the offer
  "study_offer_n": 10       // cards the offer would start with
}
```

By voice: *"no study in the morning"*, *"put the flashcards back in my briefing"*.
The line is silent unless the exam carries a course name (without one there is
nothing to filter the deck by), and it rides under the `canvas` switch like the exam
countdown it depends on.

## 46. Study ledger: "how much did I study this week?"

Every finished focus session (section on focus/pomodoro) now appends one line to
`~/.aiws_trainer/jarvis_memory/focus_history.jsonl`. Before this, the block count
lived only in `focus_session.json`, which the next session overwrites — so *"That's
three blocks done, sir"* was said once and forgotten.

- *"How much did I study this week?"* — "Two hours and five minutes this week, sir,
  over five blocks on two days." Also *today*, *yesterday*, *last week*, *this month*.
- *"What's my streak?"* — "Four days running, sir." Yesterday still counts: at nine
  in the morning last night's streak is not broken yet.
- The nightly self-review (section 36) gains a line: "You studied four blocks, an
  hour and 40 minutes."

The answer also reads the timekeeper's own record. Every block was already a silent
`focus: block N` timer that reached `done` in `timekeeper.db`, which is never pruned,
so sessions from before this feature existed — or ones the app died in — still count;
a block that fired inside a session already in the ledger is not counted twice. To
fold that history into the ledger once:

```bash
~/vss_env/bin/python scripts/backfill_focus_history.py --dry-run
~/vss_env/bin/python scripts/backfill_focus_history.py
```

It is idempotent and never opens `timekeeper.db` for writing. Nothing leaves the box
and no model is called.

## 47. Sticky modes and the terminal (voice-CLI handoff)

Three modes are *sticky*: lecture notes, dictation and an open quiz
question. Once open they claim the next utterance instead of routing it —
that is the point, since a sentence spoken during a lecture has no other
meaning.

They claim the **microphone** only. A turn that arrives from the CLI
socket (`jarvis "…"`, `jarvis.ask`, SSH, a script) or from Discord was
typed on purpose, so it is answered normally. Before this, notes open on
the desk meant every `jarvis "what's due today"` from a tmux pane was
filed as a lecture line, and dictation mode typed it into whatever window
happened to be focused.

What still works from a terminal while a mode is open:

| from anywhere | effect |
|---|---|
| `jarvis "end notes"` | closes the lecture capture (and reindexes the file) |
| `jarvis "end dictation"` | leaves dictation mode |
| `jarvis "stop the quiz"` | ends the quiz with the tally |
| `jarvis "note: the demo is on friday"` | files one deliberate lecture line, verbatim |
| `jarvis --status` | names the open modes |

`--status` (the same line as "run diagnostics") ends with them:

```
All systems nominal, sir. Up 2 hours and 10 minutes; … Lecture notes open
for BIOSENSORS, 12 lines and a quiz open at question 3 of 5.
```

That line is the only way to notice a mode from a shell now that a CLI
turn no longer lands in one. Nothing to configure.

## 48. How long the mic waits for your answer

After Jarvis speaks he keeps listening for a few seconds so you can follow
up without the wake word. That window is `followup_window` in
jarvis/config.py — 4 seconds, which is right for "…and Tuesday?" and much
too short for a question he just asked you.

So the window depends on what is open:

| what is open | window | setting |
|---|---|---|
| lecture notes | 20 s | `lecture.window_s` |
| a flashcard question, a read-back ("Cancel all three alarms, sir?"), the good-night "Shall I wake you at seven?" | 15 s | `quiz.window_s` |
| everything else | 4 s | `followup_window` (jarvis/config.py) |

```json
{ "quiz": { "questions": 5, "chunks": 6, "window_s": 15 } }
```

Set it in `~/.config/jarvis/assistant.json`. It is a request, not a
promise: the recorder clamps any window to 30 s, the value never drops
below `followup_window`, and the long window only lasts while the question
does — a finished quiz, a read-back older than 60 s and a wake-alarm offer
older than three minutes all fall straight back to 4 s. Every follow-up is
still speaker-verified, so the open mic is still yours alone.

## 49. Read-backs when he isn't sure he heard you

Whisper returns a confidence for every utterance. Below
`confirm.shaky_logprob` (default -0.7, calibrated from the live log) the
words are doubtful — and a doubtful alarm is expensive, because "5:15" and
"5:50" differ by one phoneme and the mistake surfaces hours later.

So on a shaky transcript the three creation commands read the parse back
instead of committing:

```
you    "set an alarm for five fifteen"        (heard at -0.91)
jarvis "An alarm at 5:15 am tomorrow, sir?"
you    "yes"
jarvis "Alarm at 5:15 am tomorrow, sir."
```

- alarms — "An alarm at 5:15 am tomorrow, every day, sir?"
- timers — "A timer for 5 minutes, sir?"
- reminders — "A reminder to call mum at 5 pm, sir?"

The question names what he *understood*, not what he heard, which is the
point: you are checking the parse. A confident transcript is untouched, and
so are typed and CLI turns, which carry no confidence at all. Calendar adds
already had their own read-back and are unchanged.

An unanswered question sets nothing: a "no", a change of subject, or 60
seconds of silence all drop it (the same rule as "cancel all my alarms").
The yes rides the follow-up window, which is 15 s while a read-back is open
(§48).

```json
{ "confirm": { "read_back": true, "shaky_logprob": -0.7 } }
```

`read_back: false` turns off every read-back, the bulk cancels included.
Raise `shaky_logprob` toward 0 to be asked more often, lower it (-0.9) to be
asked only when the transcript is nearly garbled.

## 50. Two things at once

Tier-1 commands can be chained in one utterance, on the same conjunctions
desktop chains have always used — "and then", "then", "and", a comma:

```
you    "set a timer for ten minutes and add milk to my todo list"
jarvis "10 minutes, sir; I'll let you know. Added to your list, sir."
```

Both halves run, the replies are spoken as one, and the mic re-opens once.

The rules, which are deliberately strict:

- **The whole utterance is tried first.** Only if it means nothing as one
  command is it split. This is what protects a body that contains "and" —
  "remind me at 5 pm to buy milk and eggs" is one errand, always.
- **Two halves, no more.** "add milk, eggs and bread to my todo list" is a
  list, not three commands, so a three-way split is refused outright.
- **Both halves must be Tier-1 commands** (timers, alarms, reminders,
  to-dos, notes, focus, briefings — the ones answered without the model).
  "set a timer for ten minutes and call my mother" runs neither and goes to
  the model whole: half an answer is worse than none.

- **A doubtful transcript is never split.** If the words scraped in under
  `confirm.shaky_logprob` (§49), the compound goes to the model whole
  rather than running two actions off a guess.

Nothing to configure. If a pair you expect is not chaining, say each half
on its own first — if either one needs the model, the pair will too.

## 51. Named lists ("add milk to the shopping list")

Nothing to configure — the lists live in the same SQLite file as the notes
and to-dos (`~/.local/share/jarvis/memory/notes.db` unless `PATHS.MEMORY_DIR`
says otherwise) and the tables are created on the next start.

```
add milk to the shopping list          # creates the list if it is new
put sunscreen on my packing list
add milk, eggs and bread to the shopping list      # three items, not one
read my packing list
what's on the shopping list
take milk off the shopping list
cross the second one off the shopping list         # the order he read you
clear the shopping list                            # asks first
what lists do I have
```

Two rules worth knowing:

* **The name is required.** "Add milk to my list" is still the to-do list, and
  so are "my task list", "my to-do list" and "my notes list" — a named list
  may not shadow the built-in two.
* **Reading never creates.** Ask for a list you have not started and he says
  "You haven't a packing list, sir." rather than inventing an empty one.

From the aisle, with the phone:

```bash
ssh spark jarvis "what's on the shopping list"
ssh spark jarvis "add milk to the shopping list"
```

That path skips the wake word and the intent gate entirely (see section 37).

## 52. Taking it back: "scratch that"

Say **"scratch that"** (or "undo that", "undo", "take that back", "belay
that") within a minute of asking for something and he removes exactly that
thing:

| what you just did | what "scratch that" does |
|---|---|
| set a timer / an alarm / a reminder | cancels that one, by id |
| took a note | deletes the note |
| added a to-do or a list item | takes it back off |
| struck something off a list | puts it back |
| cleared a whole list | puts every row back, original order |

It runs once, and only within `UNDO_WINDOW_S` (60 s). With nothing to undo the
phrase keeps its old meaning — the dictation action that deletes the last
sentence in whatever window has focus — so "delete that" is untouched.

Calendar events are the exception: `write_event` is deliberately add-only, so
a calendar add cannot be undone by voice. It has its own read-back before it
writes; delete the event by hand if the answer was wrong.

## 53. Episodic recall ("when did I last talk to my advisor?")

The activity journal (section 23) already records every exchange, tool call, Claude
result and window title for 90 days; this asks it *when*.

- *"when did I last talk to my advisor"*, *"when did I last see Dr Peyrovi"*,
  *"when was the last time I worked on the thesis"*, *"how long since I emailed my
  advisor"* — with or without the wake word.
- The people book (section 22) resolves the alias first, so "my advisor" searches for
  **advisor**, **Dr Peyrovi** and **Peyrovi** at once.
- The answer is the day in words and what the row actually was:
  *"Tuesday afternoon, sir. You said, what did Dr Peyrovi say about the recommendation
  letter."* — "how long since" gets the gap instead: *"It's been three days, sir. …"*
- Nothing found → *"Nothing in the journal about my advisor, sir"*, plus any fact you
  **told** him about the same thing.

The wording is deliberate. The journal only sees what went **through Jarvis** — voice
and typed turns, tool calls, Claude results, sampled window titles — so it answers
"the last time you mentioned it here", never "the last time you met her". A meeting
you never spoke about is invisible to it.

Day files are walked newest-first and the scan stops at the first hit, so the usual
answer costs one file read rather than ninety. Term matching is whole-word but treats
`_` and `.` as boundaries, so "thesis" finds `thesis_draft.tex - TeXstudio`.

## 54. The weekly memory garden (what he learns about you on his own)

Nothing ever promoted what Jarvis **observed** into what he **knows** — the journal
filled up and `facts.json` only ever held what you told him out loud. Once a week it
does.

At the first tick after the ISO week closes, in the small hours (`garden.run_before_hour`,
default 06:00; a week still ungardened by Wednesday runs at any hour), the week's journal
is rendered as a digest, handed to the resident local model with a strict extraction
prompt, and up to `garden.max_facts` (4) durable facts are filed through the ordinary
long-term memory path, tagged `source: "garden"`.

```json
"garden": {"enabled": true, "max_facts": 4, "run_before_hour": 6}
```

- **Monday's first wake** — after the nightly review: *"I filed three things from this
  week, sir; say memory report to hear them."* A pass that filed nothing says nothing.
- *"memory report"* / *"what did you file this week"* reads them back.
- *"forget the last garden pass"* / *"forget what you filed this week"* takes every one
  of them out of memory again.

Four rules keep a hallucinating model out of your long-term memory: it **never**
overwrites a key that already exists (what you said out loud always wins), it files at
most four a week, every one carries provenance, and the undo is one sentence. A fact you
have since re-told by voice loses the tag and survives the undo.

It is **skipped, not queued**, whenever the model is lent to a trainer ("lend the GPU")
or busy with a turn — the week's journal is still there next tick, and a 26B extraction
must never sit in front of a real question. Delivery rides the first-wake path, which
already refuses to speak inside quiet hours, so a pass written at 3 am is heard at
breakfast. Fully local: the journal is a file and the model is Ollama.

## 55. The weekly self-review (the bugs he files about himself)

The nightly digest (section 36) now also records the day's WARNING / ERROR clusters —
it has to, because `/tmp` is wiped at boot and by Sunday there is nothing left to
re-read. Once the ISO week closes, the seven digests are aggregated into
`~/.aiws_trainer/jarvis_memory/reviews/weeks/<year>-W<nn>.json` (26 kept).

- **Monday's first wake** — two sentences: *"Last week: 96 turns over 7 days, median
  wait 2.1 seconds. The median wait rose from 1.4 to 2.1 seconds, and the speaker gate
  dropped you 9 times, against 3 last week, sir."* With nothing worsening he names the
  warning that recurred most nights instead, and with neither, *"Nothing is getting
  worse that I can see, sir."*
- *"weekly review"*, *"how was my week"*, *"week in review"*, *"what went wrong last
  week"* — asks for it on demand, with the full table on a card. (*"how's my week
  looking"* is still the calendar forecast, section 7.)
- **Discord** — the table is posted when the report is filed, like the nightly one.
- **feedback.jsonl** — every recurring warning cluster (2+ days, or 3+ occurrences) and
  every worsened number is appended to
  `~/.aiws_trainer/jarvis_memory/feedback.jsonl` as `{"kind": "regression", …}`: a
  standing bug list Jarvis wrote about himself, ready for the next Claude session. Once
  per week — the callback only fires on the tick that files the report.

A "trend" needs the median wait to move by 0.4 s **and** 20%, so a quiet week of three
turns cannot shout. All of it is arithmetic over JSON already on disk: no log
re-reading, no model, nothing leaving the box.

## 56. Do not disturb during a study block

```json
"focus": {"block_min": 25, "break_min": 5, "dnd": true}
```

Section 17's study sessions now hold Jarvis's tongue while a block is running.
Anything he decided to say on his own — a Canvas deadline heads-up, a memory
warning, a line from your Claude Code hooks, a reminder — is parked instead of
spoken, and read back at the break as the catch-up digest of section 34:

> "While you were busy, sir: one reminder and one warning. Your biosensors lab
> report is due in three hours. Memory is getting tight, sir."

What still gets through mid-block:

- **The session's own lines** — "Halfway, sir.", "Time for a break, sir", "Break's
  over, sir." They are marked non-proactive, exactly as alarms are, so the block
  can never silence the announcement that ends it.
- **Answers to you.** Asking him something is not an interruption.
- **Alarms**, as always.

The break is not held — that is the whole point of the break — so a session with
`break_min` of 5 gives him a five-minute window to read the backlog every block.
Set `"dnd": false` to go back to hearing everything as it happens; `"I am free"`
also ends the hold early and reads the digest immediately.

Ordering note: an explicit *"do not disturb for an hour"* outranks the block when
he is asked *why* he is quiet, but either one holds.

## 57. Interval nudges: "every 45 minutes"

Reminders have always repeated *daily* or *on weekdays*. They now also repeat on
an interval:

> *"Remind me to drink water every 45 minutes."*
> *"Stand up every hour."*
> *"Every two hours check the oven."*
> *"Take a break every 90 minutes."*

Anything from **one minute to twenty-four hours**. The first nudge is one
interval away, not immediate, and each one re-files itself as it fires. They
list and cancel like any other reminder:

> *"Any reminders?"* — "Drink water in 40 minutes, every 45 minutes, sir."
> *"Cancel the water reminder."*

Nothing to configure.

**They know when to shut up.** A nudge is only true at the moment it is due, so
when Jarvis is holding his tongue — quiet hours, do not disturb, a meeting on the
calendar, a study block (section 56), or simply because you are out of the house —
a nudge that comes due **expires** instead of joining the catch-up digest. Come
back from a two-hour meeting and you get your reminders and warnings, not four
stacked "drink water" lines. Real reminders, timers, alarms and warnings are held
as before; a nudge never takes up one of the twelve backlog slots either.

Two phrasings he deliberately refuses, because they are almost always a mis-hear:

- *"every 5 seconds"* — below the one-minute floor.
- *"every week"* — above the twenty-four-hour ceiling.

He says "I couldn't make out the time, sir" rather than quietly setting something
odd.

## 58. Sums and unit conversions, answered instantly

Nothing to configure — this one just works, and it works with the wake word or
without it.

> *"What's 18 percent of 74?"* — "18 percent of 74 is 13.32, sir."
> *"What's 43 times 17?"* — "43 times 17 is 731, sir."
> *"How many ounces in 300 grams?"* — "300 grams is 10.58 ounces, sir."
> *"Convert 5 miles to kilometres."* — "5 miles is 8.05 kilometres, sir."
> *"What's 100 Fahrenheit in Celsius?"*
> *"What's 15 percent off 80?"* — the discounted price, 68.
> *"What's the square root of 144?"*

These used to be a full local-model turn: seconds of waiting, and the 26B model
does not always get the arithmetic right. They are now answered in Python before
the model is ever asked.

**Numbers** may be spoken or written: *"eighteen percent of seventy-four"*,
*"a hundred and twenty times two"*, *"three point five"*, and a transcript's
"1,250" all work.

**Units** he knows:

| kind | units |
|---|---|
| mass | mg, g, kg, tonne, ounce, pound, stone |
| length | mm, cm, metre, km, inch, foot, yard, mile |
| temperature | Celsius, Fahrenheit, Kelvin |
| data | kB, MB, GB, TB (powers of 1000) and KiB, MiB, GiB, TiB (powers of 1024) |

**What he refuses out loud, rather than guessing:**

- *"How many ounces in five miles?"* — "Those don't convert, sir: miles is a
  length and ounces is a mass."
- *"What's 10 divided by 0?"* — "You can't divide by zero, sir."
- *"Convert 20 dollars to euros."* — "I can't do currency, sir; I've no exchange
  rate down here."

**What he hands to the model instead**, deliberately:

- **Volume.** A US pint is 473 ml and a UK pint is 568; fluid ounces differ too.
  There is no honest single answer, so he does not pretend there is.
- **Chains** like *"two plus three times four"* — spoken precedence is genuinely
  ambiguous.
- Anything else he does not recognise. The rule throughout is that he only claims
  a question he can actually answer; everything else carries on down the ladder
  exactly as before.

Note that *"five pounds in kilos"* is a weight (2.27 kg) while *"five pounds in
dollars"* is the currency refusal — the weight reading is tried first.

## 59. Claude Code sessions can ask Jarvis back

Section 37 gave you `jarvis -q "..."` from a shell. Section 39 made your own
Claude terminals talk to you through Jarvis. This closes the loop the other way:
every pane **Jarvis** starts is now told that the assistant driving it is itself
queryable, so a coding session can look up your calendar, Canvas due dates, notes
and long-term memory instead of guessing — or asking you to type it again.

Nothing to install: the note rides `--append-system-prompt-file` on the launch
line (the same file the "your final message is read aloud" rule already used),
and `Bash(*)` is already in the session allowlist, so no permission prompt. It
needs only the symlink from section 37:

```bash
ln -s ~/Jarvis/scripts/jarvis ~/.local/bin/jarvis      # once, if you haven't
```

What the session is told, in short:

- `jarvis -q "when is my next exam"` / `"what's due this week"` / `"what's on my
  list"` / `"what did I tell you about the venv"` — personal context it cannot
  read out of the repo;
- `jarvis -q "remember the F5 socket lives in /tmp/vss_voice"` — files a durable
  fact into the **same** semantic memory the voice assistant recalls from. Facts
  about your life and setup only; code notes belong in the repo;
- exit 2 means Jarvis is not running (carry on without him, do not retry in a
  loop), exit 3 means the turn produced no reply;
- it must **not** delegate coding, editing or web work back to Jarvis — he would
  route that straight back to `ClaudeSessionManager.submit()` and queue a task
  from inside a task.

`-q` is not decoration: without it the pane's lookup would be spoken aloud, on
the soundbar, over whatever Jarvis was saying at the time.

The suffix file under `/tmp/vss_voice/claude/system_suffix.txt` is rewritten
whenever the constant in `jarvis/claude_session.py` changes, so a pane can never
be launched with a stale contract; there is nothing to clear by hand.

**Your own terminals** (the `~/.bashrc` tmux wrapper, any plain `claude`) never
pass through Jarvis and so never see this suffix — the same reason the narration
hooks in section 39 exist. If you want the same behaviour there, paste the four
lines into `~/.claude/CLAUDE.md` yourself:

```markdown
Jarvis (Hunter's voice assistant) is queryable: `jarvis -q "..."` asks the
running assistant and prints his answer (-q keeps the soundbar silent).
Use it for personal context — calendar, Canvas due dates, notes, memory — and
`jarvis -q "remember <fact>"` to file a discovery. Exit 2 means he is not
running: carry on, do not retry. Never delegate coding or web work to him.
```

Jarvis never edits that file for you, for the same reason `install.py` is a
manual step: your Claude configuration is yours.

## 60. Ask him where something is in your own code

> "Jarvis, where does the mic arbiter live?"
> "It's in Jarvis, jarvis/recorder.py, lines 120 to 168, sir."

That question used to cost a twelve-second `claude -p` session and your
credits, for what is a lookup. It is now answered locally in about two
seconds from an embedding index over your own repositories — the same
machinery as the documents index in section 14, pointed at source instead
of PDFs.

**Setup: none, if Claude is already configured.** The folders come from
`claude.allowed_dirs`, because those already are your repos:

```json
"claude": {"allowed_dirs": ["~/Jarvis", "~/haymaker-digest"]},
"code": {"paths": [], "index_dir": "", "max_files": 3000}
```

Set `code.paths` only if you want a different list; leave `index_dir` empty
and the store lives in `~/.aiws_trainer/jarvis_memory/code_index` (a chroma
collection called `jarvis_code`, kept separate from the documents one so
quiz mode never draws a flashcard out of `app.py`).

`.py`, `.md` and `.sh` are indexed. `repo/`, `.git/`, `node_modules`,
`site-packages`, virtualenvs, caches and build output never are — the first
of those matters most here, since `~/Jarvis/repo/` holds 140 MB of StyleTTS2
weights. Files over 300 kB are skipped as generated rather than written.

Chunks are cut where a reader would start — at a `def`, a `class`, a
markdown heading, a shell function — and carry their line numbers, which is
where the `file:120-168` citation comes from. The first index of ~200 files
takes about a minute and a half in the background (measured on this box,
2026-08-30); after that each question is 20-30 ms of search, and the index
refreshes itself incrementally every fifteen minutes so a day of editing is
never invisible.

**What still goes to Claude.** The router only keeps a question local when
it is a *lookup*:

| stays local (`ask_code`) | still Claude |
|---|---|
| "where does the mic arbiter live" | "where should I add the new tool" |
| "which file has the speaker gate" | "why is the recorder test failing" |
| "what module holds the tool registry" | "refactor the module that owns the mic" |
| "show me the file that starts the tmux session" | "ask claude where the mic arbiter lives" |

The index can say where things are; it cannot say where they belong or why
they broke. Saying "ask Claude" explicitly always wins.

## 61. Syllabus dates: the exam that never reached Canvas

Canvas carries assignments. The midterm dates live in a PDF, and at TAMU
that PDF is often the only place they exist — so "when's my next exam" said
nothing, the evening-before heads-up never fired, and the briefing had
nothing to count down to.

Drop the syllabus into the documents folder from section 14 and say:

> "Jarvis, scan my syllabus."
>
> "I found 2 dates in your syllabus, sir: Midterm 1 for CS 101, in 30 days,
> Wednesday 1 Oct at 9:00 am; Final exam for CS 101, in 100 days, Wed 10 Dec.
> Shall I put them on the books?"
>
> "Yes." — "Filed, sir; 2 on the books."

Also: "read my syllabus", "go through my syllabus", "check my syllabus for
dates", "add my syllabus dates", "what's on my syllabus". No wake word
needed in jarvis mode.

**Nothing is filed until you say yes**, and every date is read back in full
rather than counted. That is deliberate: a model reading dates out of a PDF
is exactly where a wrong *year* files a reminder for the wrong week, and the
read-back is the only moment you can catch it. Saying anything other than a
clear yes or no drops the offer, as with any other read-back (section 30).

Three guards run before you even hear the question. A date the model could
not write as a plain `YYYY-MM-DD` is dropped rather than guessed at; a date
already in the past is dropped (scanning in October must not file
September's midterm); and a date more than 400 days out is dropped as the
hallucinated year it is.

Once accepted, the dates are a **third source** beside Canvas and your
calendar, and they are merged in both places that matter:

- `deadlines.tick` — the lead-hours reminder ("Lab 3 report for CS 101 is
  due in 3 hours") and the 7 pm evening-before exam call;
- `canvas.find_next_exam` — "when's my next exam" and the briefing's exam
  countdown.

Merging only the first would have him call an exam eve for a midterm he
would then deny having when asked, which is worse than no exam eve at all.
If a professor posts the midterm to Canvas *and* lists it in the syllabus,
Canvas wins: its due time is the authoritative one, and you get one
reminder rather than two.

Re-scanning the same syllabus is safe — already-known dates are recognised
and he says "Already on the books, sir."

```
~/.aiws_trainer/jarvis_memory/syllabus_deadlines.json   the accepted rows
```

Written atomically, one row per date (`title`, `course`, `due`, `all_day`).
Delete the file to forget everything a scan ever filed; the reminders
already handed to the timekeeper are separate and are cancelled the usual
way.

**When it says nothing useful.** "I can't find a syllabus in your documents,
sir" means the folder has nothing syllabus-shaped in it (or nothing at all);
"I'm indexing your documents now" means the file is there but not embedded
yet — ask again in a moment; "My document index isn't answering" means
Ollama is down, not that the folder is wrong.

## 62. Phone intercom: talk to him from bed (no wake word)

The wake word does not reach the next room, and the soundbar's answer would
wake the house. The intercom sends a *recording* over the command socket
instead: Jarvis transcribes it with the same Whisper the microphone uses and
answers in **text** on the phone, silently, unless you ask for speech.

Nothing new is installed on the Spark. On the phone you need Termux and
`termux-api` (the app AND `pkg install termux-api`), plus the SSH key you
already use.

```bash
# on the phone, in Termux
termux-microphone-record -q >/dev/null 2>&1     # make sure nothing is recording
termux-microphone-record -d -l 8 -e opus -f $HOME/j.ogg
sleep 9
ssh spark 'jarvis --send-audio -' < $HOME/j.ogg
```

`-e opus` is **not optional**: `termux-microphone-record` writes AAC by
default and libsndfile cannot read AAC, so the clip would come back
"I couldn't decode that clip, sir". wav, ogg, opus and flac all work.

Worth putting in a Termux `~/.bashrc` function:

```bash
ask() {
  termux-microphone-record -q >/dev/null 2>&1
  termux-microphone-record -d -l "${2:-8}" -e opus -f "$HOME/j.ogg" >/dev/null
  sleep "$(( ${2:-8} + 1 ))"
  ssh spark 'jarvis --send-audio -' < "$HOME/j.ogg"
}
```

From any box that has its own microphone (including Termux) the client can
do both legs itself:

```bash
jarvis --listen 8                     # record 8 s here, send it, print the answer
jarvis --send-audio clip.wav          # a clip you already have ("-" = stdin)
jarvis --send-audio clip.wav --speak  # ...and answer aloud in the room as well
```

What comes back:

```
[heard] what's my first thing tomorrow
Biosensors at nine, sir.
```

The `[heard]` line goes to stderr and is what he understood — a misheard
clip is otherwise indistinguishable from a wrong answer. Exit codes are the
CLI's: 0 answered, 2 Jarvis is not running, 3 no reply, 4 the clip could not
be recorded or read.

### Settings (`~/.config/jarvis/assistant.json`)

```json
"intercom": { "enabled": true, "verify_speaker": false, "max_mb": 10 }
```

| key | what it does |
|---|---|
| `enabled` | `false` refuses every clip ("The intercom is switched off, sir.") |
| `verify_speaker` | run the ECAPA speaker gate on the clip as the microphone path does |
| `max_mb` | the size of one clip; 10 MB is about five minutes of 16 kHz wav |

`verify_speaker` is **off** on purpose. The socket is mode 0600 and only
reachable through your own SSH session, which is already the
authentication; meanwhile the transcript gate *fails shut* once a voiceprint
exists, and a phone microphone through a lossy codec moves the ECAPA
embedding far enough that it would reject your own voice. Turn it on if the
box is shared — a rejected clip answers "That didn't sound like you, sir."

Anything longer than two minutes is truncated rather than refused, so a
phone left recording cannot park the resident Whisper. A clip arrives as one
JSON line: the request framing was 64 KB and is now the base64 of a whole
clip, and a runaway request answers "request too large" instead of looking
like a JSON bug.

## 63. Bedtime wind-down: "good night" dims the room

"Good night, sir. I'll be here." can be made physical. With the wind-down on,
saying good night also fades the music out, warms and dims the screen, and
arms do-not-disturb until quiet hours close. "Good morning" puts it all back.

It is **off by default** and every half has its own switch:

```json
"wind_down": {
  "enabled": false,
  "fade_s": 60,
  "brightness": 0.5,
  "night_light": true,
  "music": true,
  "dnd": true,
  "morning": "07:00"
}
```

| key | what it does |
|---|---|
| `enabled` | the whole thing; `false` leaves "good night" as a spoken line |
| `fade_s` | how long Spotify takes to reach silence (one volume step per 5 s) |
| `brightness` | the `xrandr` level for every connected output; floored at 0.2 |
| `night_light` | GNOME's warm screen (`night-light-enabled`) |
| `music` | fade and pause Spotify |
| `dnd` | hold proactive speech until quiet hours end |
| `morning` | when `dnd` ends if `quiet.hours` is not configured |

Say "good night", "night night" or "off to bed" and, in order: DND is armed,
the screen warms and dims, and the music fades over `fade_s` and pauses. The
Spotify **volume is put straight back after the pause** — the lasting effect
is the pause, and a device left at 0% is indistinguishable from a broken
speaker to anyone who presses play in the night.

Say "good morning" (or "hello", or let the first-wake briefing run) and the
screen, the night light and the volume go back to exactly what they were.

### If something goes wrong

Nothing is guessed. The state to restore is written to
`~/.aiws_trainer/jarvis_memory/winddown.json` **before** the first thing
changes, so:

* a failure part-way through dimming restores the screen immediately;
* a crash mid-fade leaves the file, and the next "good morning" undoes it;
* a second "good night" is a no-op — snapshotting then would record the
  *dimmed* screen as the brightness to go back to;
* starting Jarvis again after the window is over restores it (a restart at
  two in the morning deliberately does not, or the room would light up);
* a file older than a day and a half is always restored, whatever it says.

And there is a door from outside the app, for a screen left dim by something
that took Jarvis with it:

```bash
~/vss_env/bin/python -m jarvis.winddown --status     # what it would put back
~/vss_env/bin/python -m jarvis.winddown --restore    # put it back now
```

### Requirements

`xrandr` and `gsettings`, both already present, both without sudo; the screen
brightness is X's software gamma, not a backlight, so it works over HDMI on a
desktop monitor. Spotify only does anything when the account is linked
(section 11) and a device is reachable — otherwise it is a night without
music and no apology, exactly as in a focus session.

## 64. The Aside: one thing you did not ask for

> "Alarm for 7:00 am, sir. Incidentally, Lab 3 report for BIOSENSORS is due
> at 11:59 pm that night."

Everything else Jarvis says is an answer to something. This is the one door
that volunteers, and the budget is the whole product: **two a day, forty-five
minutes apart**, and one phrase ends it.

It ships **off**. Turn it on in `~/.config/jarvis/assistant.json` — this is
the configuration to use once you want it:

```json
"aside": {
  "enabled": true,
  "per_day": 2,
  "gap_min": 45,
  "horizon_hours": 18
}
```

| key | what it does |
|---|---|
| `enabled` | the whole feature; `false` (the shipped default) means he never volunteers |
| `per_day` | how many asides a day. Earn the third; do not start there |
| `gap_min` | the minimum minutes between two of them |
| `horizon_hours` | how far past the thing you just set still counts as "that night" |

### When he will and will not say something

He hangs an aside off a moment you have just pointed at — an **alarm** or a
**reminder**, never a timer. A timer is a kitchen device; a lecture an hour
after it has nothing to do with it, and saying so is exactly the tic this
budget exists to prevent.

The anchor has to be *real*. "Set for seven" only becomes an aside if "seven"
resolved to an actual datetime, so he reads the structured result the handler
produced — the timekeeper item with its epoch — and never re-reads his own
sentence looking for a time. No resolved anchor, no aside.

Then he looks for one thing between that moment and `horizon_hours` later:

* a Canvas or syllabus deadline, from the snapshot the deadline thread
  already had (never a fresh Canvas call — that would put the network in the
  middle of a spoken turn);
* an **exam** on the calendar. Only an exam. Your Tuesday lecture is not news,
  and the meeting heads-up is going to mention it ten minutes beforehand
  anyway.

And he stays quiet when:

* the budget is spent, or the last aside was less than `gap_min` ago;
* it is **quiet hours** — the line is *dropped*, not saved. An "incidentally"
  arriving three hours later with no question in front of it is the opposite
  of the effect, so it costs no budget either;
* the deadline heads-up is already going to say that item out loud. The two
  share a ledger, so the same lab report never earns a reminder at 09:00 and
  an aside at 09:20;
* he has already volunteered that exact thing.

### Making him stop

Say **"no more asides"** — or "that's enough of that", "enough asides", "stop
the asides". Deliberately not "stop that": that already means barge-in and
read-aloud steering, and a kill phrase the router can confuse with either is
worse than none. It zeroes the day's bucket only; tomorrow he starts again.

Every silencing is counted, and the nightly self-review reports it as a
problem: *"you told me to stop volunteering things."* If that line shows up
most days, the feature is wrong — turn `enabled` back to `false`, and the log
is the evidence for it.

### Reading the templates before you ship them

The correctness is easy; the taste is not. There is an offline harness that
replays twenty turns across two days against fake calendar and deadline data
and prints, for each one, either what he would have said or why he stayed
quiet:

```bash
~/vss_env/bin/python scripts/aside_dryrun.py
```

Read the output as a transcript, not as a test. If more than a couple of the
lines make you wince, the templates are wrong and no amount of ledger
correctness will save it.

## 65. Reasoned dissent: "I would advise against that, sir"

> "Wake me at two."
> "I would advise against a 2:00 am alarm, sir; your BIOSENSORS lecture is at
> 9:10 am and that leaves you under five hours. Shall I set it anyway?"
> "Set it anyway."
> "Setting it anyway, sir. Alarm at 2:00 am."

He is allowed to disagree once, out loud, with a reason — and then he does
what you told him. It is on by default:

```json
"confirm": {
  "read_back": true,
  "shaky_logprob": -0.7,
  "dissent": true,
  "sleep_floor_h": 5
}
```

| key | what it does |
|---|---|
| `dissent` | the whole feature; `false` sets every alarm without comment |
| `sleep_floor_h` | hours below which a small-hours alarm is worth a word |

### The four things he will object to

Each one names the actual row it read. An objection that cannot name its row
is a mood, and he does not make those.

| source | the objection |
|---|---|
| duplicate alarm | "you already have an alarm at 7:00 am" — one is already set within fifteen minutes |
| sleep window | "your BIOSENSORS lecture is at 9:10 am and that leaves you under five hours" — an alarm in the small hours, under `sleep_floor_h` away, with something on the calendar later that day |
| quiet conflict | "you're in BIOSENSORS then" — the alarm lands inside a window you told him to keep clear |
| deadline clash | "Lab 3 report for BIOSENSORS is due at 11:59 pm, before that" — the alarm is set for *after* something is due |

Only alarms. Never a timer, a track change or a volume nudge: objecting to
anything you can undo in under a minute is the failure mode itself.

### Answering him

The default is **yes**. This is the opposite of the read-back for a
destructive action ("Cancel all three alarms, sir?"), which drops on anything
that is not a clear yes — because there, doing nothing is the safe end. Here
you *asked* for the alarm and only the opinion was volunteered, so:

* **"no" / "don't"** — the only way to lose the alarm. *"Very good, sir. I'll
  leave it."*
* **"yes" / "set it anyway" / "anyway" / "I know" / "regardless" / "that's
  fine"** — it is set, and the reply is word-for-word what an unobjected alarm
  would have said.
* **anything else, including silence** — it is set, he says *"Setting it
  anyway, sir."*, and if you had actually said something else ("play some
  jazz") that sentence keeps its own meaning and gets its own turn.

A man who heard the objection, agreed with it and went to bed still wakes up
to his alarm.

### Not becoming insufferable

* Never twice for the same row in the same day. A second "wake me at two" is
  someone who has heard the reason and wants the alarm.
* Never over another open question — a shaky-transcript read-back, or the
  evening preview's "shall I wake you at seven?", is never talked over.
* Everything he reads is cached: the calendar's own cache and the deadline
  thread's snapshot. Nothing here puts the network inside "wake me at two".

Both halves are counted, and the nightly self-review flags the one number that
matters: if you overruled **all** of his objections, three or more times in a
day, the rule is simply wrong. "Why did you argue with me about that alarm?"
is answerable later too — every objection goes into the journal.

## 66. The debrief: "how did the midterm go, sir?"

The calendar says the BIOSENSORS midterm ended forty minutes ago and you are
not out. He asks. **Once.**

Whatever you say back is *filed* — into the activity journal and as a
remembered fact — and never routed to the model as chat. That restraint is the
feature: "it went badly, I ran out of time on the last question" is a fact
about your year, not a conversational turn to be met with sympathy and
forgotten when the window closes. It comes back in the day recap, in the
journal digest, and months later as an aside before the next one.

On by default:

```json
"debrief": {
  "enabled": true,
  "after_min": 15,
  "within_min": 180,
  "hold_hours": 14,
  "keywords": ["exam", "midterm", "final", "finals", "interview", "viva",
               "defense", "defence", "quiz", "test", "presentation",
               "audition"]
}
```

| key | what it does |
|---|---|
| `enabled` | the whole feature |
| `after_min` | how long after it ends before he asks — under this you may still be walking out of the room |
| `within_min` | after this the moment has passed and the question is an interrogation |
| `hold_hours` | how long a question held by quiet hours stays askable |
| `keywords` | whole words in the event title that make it worth asking about |

`keywords` is matched on **whole words**, so "contest" is not a test and
"finalise the slides" is not a final. Widen it if you like — but this is the
one feature that speaks without being spoken to, and "how did your lunch go"
is the version of it nobody wants.

### The four restraints

* **Never about something that did not happen.** An event only becomes askable
  if he saw it on the calendar *while it was still in the future*. A row added
  retroactively this afternoon — a meeting someone logged after the fact, a
  cancellation that never reached the cache — is never asked about.
* **Never twice.** The ledger is written the moment the question is asked, not
  when it is answered, and it survives a restart.
* **Never while you are out.** Presence gates it. If `presence.phone_ip` is
  not configured, presence is idle and you count as home — otherwise the
  feature would never fire at all.
* **Never in quiet hours** — but a quiet window *postpones* the question, it
  does not cancel it. An exam that finished at ten past eleven is asked about
  in the morning, up to `hold_hours` later.

### Answering

The question opens the mic behind it, so you answer without the wake word. He
files it and says *"Noted, sir."*

Say **"not now"** or "never mind" and nothing is filed — but he has been asked,
so he will not ask again. Say something that is plainly a command instead
("jarvis, set a timer for five minutes", or any Tier-1 phrase) and the debrief
steps aside and the command runs; the window also expires after two minutes.
Filing "set a timer for five minutes" as how the midterm went would poison a
record you are meant to be able to trust months from now.

### What you get back

Two sinks, and one failing does not lose the other:

* the activity journal, as its own `debrief` kind — pinned in the digest, so
  it is never the line the character budget trims away;
* long-term memory, under a key that reads as English ("how BIOSENSORS Midterm
  1 went on 14 September 2026"), so asking months later finds it.

Calendar first: Canvas exam rows carry a due time and no end, so there is
nothing there that can say *it is over*.
---

## 67. The arc: one named hour for the whole house

Time-of-day logic used to be five clocks that disagreed — quiet's hours, the
deadline nudges' "not yet evening", hardcoded morning/evening greetings in two
places, and the briefing's own `after`. `jarvis/arc.py` gives them one truth.

It names the hour as exactly one of:

```
pre-dawn   waking   working   afternoon   dusk   evening   night
```

from real sunrise and sunset, plus quiet hours, presence and whether a focus
block is running. There is nothing to say to it and nothing to configure to
make it work — it is a **state source**. Anything that wants to behave
differently at dusk than at ten in the morning reads `services.arc.phase` or
subscribes to `ArcChanged` on the bus.

**Sunrise and sunset are computed on the box.** Not fetched. The weather tool
does have them, but they arrive over the network and need `home_location`
lat/lon, which ships empty — an arc keyed off that degrades to fixed hours on
every cold boot and every network blip, silently. So the arc runs the NOAA
sunrise equation itself, stdlib only, accurate to a minute (checked against
published tables for New York at both solstices). If a forecast is *already*
cached it may override the local answer; nothing here ever triggers a fetch.
No coordinates, or a polar day, falls back to fixed hours.

**It never flaps.** A phase has to hold for 20 minutes before another can take
it, the solar boundaries carry a ±15-minute tolerance band, and within a local
date the phase only moves forward — so a cached forecast refreshing with a
sunset twenty minutes later cannot drag dusk back to afternoon. `ArcChanged`
is published on an accepted transition only, never per tick, so a subscriber
can treat every event as a real change.

Two overrides jump the queue in both directions: a running focus block pins
`working` (a man deliberately working at midnight is working), and quiet hours,
DND, a detected meeting or an empty house collapse it to `night`.

Two things it deliberately does **not** do. It never writes config — in
particular it will not touch `briefing.verbosity`, which is a preference you
set by voice and would read as a bug if the clock overwrote it. And it never
gates speech: `jarvis/quiet.py` is the owner of quiet, and the arc only
*consumes* `quiet.reason()`. Two policies disagreeing at 23:00 is exactly the
failure this avoids.

```jsonc
"arc": { "enabled": true, "tick_s": 60 }
```

State lives in `MEMORY_DIR/arc_state.json` so a restart does not re-announce
the hour you were already in.

## 68. The earcon lexicon: six tones in one family

He had three beeps already — the capture opening at 880 Hz, the capture
closing at 660, and the 440 Hz nudge that means "I heard you, and I have
nothing to answer". Those three turn out to be a root, a fifth and an octave,
so the family is *derived* from what the room already sounded like rather than
invented beside it: 220 / 330 / 440 / 660 / 880 / 1320, one timbre
(fundamental, a quiet octave, a trace of the twelfth), one 10 ms raised-cosine
envelope, every tone under 400 ms.

`jarvis/earcons.py` is now the only place a non-speech sound comes from;
`recorder.play_beep()` delegates to it. Two beep systems with different
accents is the precise failure a shared voice for the room exists to prevent.

Six names ship:

| tone | means | wired to |
| --- | --- | --- |
| `heard-you` | the wake word landed | the wake acknowledgement |
| `done` | the capture closed | end of recording |
| `held-back` | heard you, nothing to answer | the nudge policy |
| `arrival` | he came back | the arrival cue (§69) |
| `thinking` | — | nothing, on purpose |
| `warning` | — | nothing, on purpose |

The last two are rendered so the lexicon is complete and auditionable, and
connected to nothing deliberately. `BrainState` flips constantly and the
reactor already shows it visually, so a chirp per state transition reads as a
nervous tic rather than an accent. And the only candidate publisher for a
grave warning tone is the health watchdog, whose lines are *proactive* and
therefore held by quiet hours — the tone would sound at 3 a.m. exactly when
the sentence it accompanies was suppressed. Either gets wired when it has a
real publisher.

Audition all six back to back before you believe any of this:

```bash
~/vss_env/bin/python scripts/make_earcons.py --play
~/vss_env/bin/python scripts/make_earcons.py --out /tmp/tones   # just bake them
```

```jsonc
"sound": { "earcons": true, "cooldown_s": 4, "volume": 0.5 }
```

Note the gate is `sound.earcons` and **not** `CONFIG.sound` in
`voice_settings.json`: that flag defaults false and means "the old chimes", so
hanging the lexicon off it would have shipped it mute. Because the wake
acknowledgement now sounds by default, it earns the existing 200 ms mic guard
— the bloom fires while the mic is opening and would otherwise be recorded
straight back through the Snowball.

Rate limiting lives in the generator, per name, plus a 250 ms global gap. One
shared cooldown across all causes would swallow the `done` that closes a
capture two seconds after `heard-you` opened it, which is worse than the
false-wake metronome it prevents.

The WAVs bake deterministically into `MEMORY_DIR/earcons` — not `LOG_DIR`,
because `/tmp` is wiped at boot on this box.

## 69. Arrival and departure: the room notices the door

"Welcome back, sir" and the catch-up digest already shipped. What was missing
was that they landed as a bare line into a dead room. Arrival is now a
composed cue in a fixed order:

```
panel  ->  earcon  ->  "Welcome back, sir."  ->  the catch-up
```

The pre-existing quiet rule is kept exactly: `"you're out"` is stale by
definition on a returned event so it never defers the greeting, while any
other reason (hours, DND, a meeting) wakes the panel and leaves the voice to
the quiet policy's own tick.

**Departure is the mirror in the one way that matters: it says nothing at
all.** And it is asymmetric by construction — arrival fires on first sight,
because being late to notice him is the whole failure mode, while departure
clears three gates:

1. the sentinel's own away grace (`presence.away_after_min`, 12 min);
2. a further confirm window, `presence.departure_confirm_min`, floored at a
   minute and strictly *additional* to the grace;
3. a hard veto if the microphone heard him inside
   `presence.departure_mic_silence_min` — read from the existing turn ledger
   via `TurnLedger.idle_s()`, not a new field bolted onto the app.

A sleeping phone radio faking a departure while he is sitting in the room is
the one outcome worth spending latency to avoid.

`presence.poll_s_away` polls fast (10 s, floored at 5) **only while the house
is empty**. At the 60 s default, "the room notices the door" is in practice
"the room notices up to a minute after he sat down", by which point he may be
mid-utterance and a staged four-step cue reads as late rather than composed.
One ping per ten seconds, and only when nobody is home.

```jsonc
"presence": {
  "phone_ip": "192.168.1.42",
  "poll_s": 60, "poll_s_away": 10,
  "arrival_cue": true,
  "departure_confirm_min": 5,
  "departure_mic_silence_min": 10
}
```

Two things are deliberately absent. The **music follow-out** is not built:
`spotify.transfer_playback` defaults to `force_play=True`, and an unprompted
cue that *starts* audio in the pocket of a man walking to his car is far worse
than a paused song. And the panel does not go to standby — the night surface
is owned elsewhere, and two features writing one surface is how it ends up
flickering.

**This is dark until `presence.phone_ip` (or `phone_mac`) is set.** Read the
lease off the router, or find the phone in `ip -4 neigh` while it is on the
Wi-Fi. Before tuning the confirm windows, it is worth just watching the
`presence:` log lines for a couple of days of ordinary coming and going —
that ledger answers how late arrival detection really is, and how often the
radio naps, better than any guess.

## 70. Room tone: a house that is audibly awake

Under everything, a bed. One near-subliminal 12-second loop per arc phase:
pre-dawn is almost nothing, the work hours carry a faint HVAC-and-servers hum,
dusk warms, night thins to a single low drone. You are meant to notice it only
when it stops.

**It ships off, and it needs a spoken opt-in:**

```
"jarvis, room tone on"          "room tone off"          "is the room tone on?"
```

The switch is Tier 1 and reversible in one utterance, because the failure mode
of ambience is that it is quietly costing you wake words while you wonder why.

**Mic safety is the design here, not a risk section.** There is no AEC on this
box, the Snowball shares the room with the speaker, and Jarvis is armed for the
wake word continuously — so a bed enters *every* capture and puts three
thresholds that were tuned in a quiet room at risk: the openWakeWord threshold
(0.3), the Silero endpointer (whose own test asserts that room tone is not
speech), and the ECAPA speaker gate at 0.30, which has caused total rejection
once already. "Duck during TTS" does not cover any of that. So:

* the mute is **hard and immediate** — the stream is killed from the caller's
  own thread, not faded — and it is asserted by the wake word *before* the mic
  opens, then held through every capture and every spoken reply;
* the default volume is `0.05`, a placeholder for a number you measure at the
  Snowball: record the room at a few candidate levels, and require no
  regression in wake-word false rejects or in the ECAPA similarity
  distribution before raising it;
* holds are **named, not counted** — `SpeakingState` streams at ~12 Hz while
  he talks, so a depth counter would take twelve increments and one decrement
  per reply and latch the bed off for good. A hold nobody released expires
  after 90 s, because ambience failing silently until the next restart is the
  worst way for it to fail.

There is no `pactl` ducking on purpose. We own the process, so muting it
outright is stronger than lowering its volume and leaves no external state to
restore — which is also why this has no dependency on a mixer and ships
standalone.

Silence is a policy as much as a mic concern: the bed is quiet whenever
`quiet.reason()` is non-empty (quiet hours, DND, a calendar-detected class or
meeting, an empty house), and a long stretch with no microphone activity stops
it whatever the phone says — which also lets the Bluetooth speaker sleep.

```jsonc
"ambience": { "room_tone": false, "volume": 0.05, "away_stop_min": 20 }
```

Loops are baked once to `MEMORY_DIR/roomtone` from a fixed per-phase seed
(deterministic: a loop cached last month is the loop this code would make
today) and streamed raw into a single dedicated `paplay`. Nothing is ever
synthesized on a tick, and certainly not while Whisper is decoding. The loops
are periodic by construction — the drone rounded to whole cycles, the air
built in the frequency domain — so there is no crossfade anywhere and no seam
to hear.

## 71. The Board — mission control on the empty right of the desk

The console is 520x880 on a 3840x2160 panel. The Board fills the rest of the
right flank with a second borderless panel in the same theme.

Say **"bring up the board"** (or "show me the board", "board up") and it
slides in. **"Close the board"** takes it down. Both work without the wake
word, the way every other surface verb does.

Six slabs, top to bottom:

| slab | what it says |
|---|---|
| VITALS | memory free of total, GPU load / temperature / power, load average, and any training process. Amber under 16 GB free, red under 8. |
| TURNS | the turn ledger as a strip chart: the last turn, the **median** wait (a mean is dragged around by one cold model load), the worst, and how many. Amber when anything waited over four seconds. |
| CLAUDE | live tasks first — RUNNING, QUEUED, WAITING on a permission answer — then the most recent sessions on disk. |
| DUE | timekeeper items and Canvas deadlines, soonest first, as "in 12m" rather than a clock time you have to subtract from. |
| FOCUS | the block or break running, in the session's own words. |
| QUIET | why he is holding his tongue, how many lines are waiting, and where you are (only when presence is actually configured). |

### Asking about one panel

**"Focus on the sessions"** lights that slab *and* says the one-line read
aloud: "Two tasks running, sir." The other names work too — "the vitals" or
"the engines", "the turn ledger", "what's due", "the block", "quiet hours".
A "focus on ..." that names no panel is left alone: "focus on the thesis"
still means what it always meant.

### Over SSH

```bash
jarvis board
```

prints exactly the same state as plain text. That is not a convenience
bolted on afterwards — the whole state layer was built and proved through
that command before a single line of drawing code existed.

### What it costs

A 5 s poll thread while the Board is up, and nothing at all while it is
down. The Canvas half is cached five minutes, so raising the Board never
turns into a REST call every five seconds. Nothing expensive runs on the
drawing thread.

### Switching it off

```json
"console": { "board": false }
```

### Requirements

`xprop` and `xdotool`, both already present. Without them the panel falls
back to a "splash" window type — undecorated, but skipped by taskbars, which
for a docked panel is what you wanted anyway.


## 72. Standby and the ambient panel — the console when nobody is talking

Between conversations the console stops being an app.

**After about 45 seconds** of quiet the transcript recedes and one slab of
room state takes the stage: what is playing, the next thing on the calendar,
the next deadline, the outside temperature, the hour of the house, and where
you are. Quiet hours are not a row saying "quiet hours: on" — the whole slab
goes ember and low-contrast, which reads from across the room in a way a
label never does. GPU load is a bar for the same reason.

**After about twelve minutes** away from the keyboard the console becomes
the room's clock: a large soft clock with the day and date, the next
commitment, the next deadline and the temperature. The command bar and the
status strip go away. The reactor keeps turning at a third of its speed.
The whole thing dims on a curve toward the small hours and holds a floor at
35% — never black, and never below it, whatever the hour.

Touch anything, or say his name, and it is back.

### What it deliberately does not do

* **It never touches the screen's brightness or gamma.** The dimming is a
  colour blend inside Jarvis's own window. A crash while the desktop gamma
  is crushed would leave you with a dark screen and no obvious way back;
  a crash here leaves a normal window.
* **It never re-renders the reactor.** The avatar's frames are baked at a
  fixed size, and re-baking them at a mode change is exactly the kind of
  window churn that froze the desktop on 26 August.
* **It does not go fullscreen.** The room clock is the console's own
  window, for the same reason: a 4K standby surface would want the reactor
  at a new size, and every size change is a full re-bake.
* **Burn-in is handled by moving the window,** a few pixels a minute on a
  slow loop, which drifts the brightest thing on the panel — the reactor
  disc — without redrawing anything. The walk is undone the moment you come
  back, and again before the window position is saved at quit, so the
  console cannot creep across the desk one session at a time.

### How it knows you are away

In preference order: the desk-presence probe if it is wired, GNOME's own
idle monitor if that module is installed, and otherwise the X server's
screensaver idle counter. If none of them can answer, the console stays
awake — a machine that cannot see the keyboard should never decide nobody
is there.

A live turn or a ringing alarm keeps it awake regardless.

### Settings

```json
"console": {
  "ambient": true,
  "ambient_after_s": 45,
  "standby": true,
  "standby_after_min": 12,
  "standby_dim": 0.35,
  "drift_px_per_min": 3
}
```

`standby_dim` is a floor, not a target: 0.35 means it never goes below 35%.


## 73. Power-up — the room waking with you

The first time you touch the desk after the night — the phone coming back
onto the Wi-Fi, or simply the first wake word of the day — the panels
populate in sequence rather than being there all at once. About a second
apart, then the finished board holds for a beat and the console settles.

It happens **once a day**, and the moment you say anything the rest of it is
abandoned mid-sweep: the reply always wins.

It is ten seconds of theatre over data he already had. There is no new
information in it, and nothing waits on it.

Two triggers, because on this machine one is not enough: the proper one is
presence noticing you are back, but presence does nothing until `phone_ip`
is set (section 35), so the first wake word of the day is what actually
fires it here.

The "once a day" latch is a `boot_sweep` date written into the same
`briefing_state.json` the morning briefing already uses, so restarting
Jarvis five times before breakfast still gets you one sweep.

```json
"console": { "powerup": true, "powerup_gap_h": 6 }
```

`powerup_gap_h` is how long the machine must have been left alone to count
as "after the night". Set `powerup` to `false` and nothing sweeps.

## 74. The room: display light, level and scenes

There are no bulbs in this room, so the 43-inch panel is the light. Jarvis moves two
things and says so honestly — *"Dimming the display, sir."* The screen dims; the room
does not.

- *"Dim it a little"* / *"dim the screen"* / *"lights down"* / *"it's too bright"* —
  one step darker (0.15 of `xrandr --brightness`, an X **gamma scale**, not a backlight).
- *"Brighter"* / *"brighten the display"* — one step back up.
- *"Lights up"* / *"full brightness"* — everything back where you had it, in one word.
- *"Warmer"* / *"warm the screen"* / *"night light on"* — GNOME's night light, forced on
  whatever the hour; *"cooler"* / *"back to daylight"* walks it back.
- *"Power down the workshop"* / *"lights out"* / *"call it a night"* — the **wind-down
  scene**: screen warm and dim, Spotify paused, quiet hours armed, and a dry line.
- *"Wake up the workshop"* / *"good morning"* / *"lights up"* — the reverse, every knob.

The brightness floor is **0.55, never black**: Jarvis's own console lives on this panel,
so blanking it would take the cards, the reactor and any alarm prompt with it. Nothing
here touches DPMS or `xset`, which is also the answer to "what happens if a reminder
fires while the room is down" — the panel is dimmed, not off, so the card is still
readable. `ddcutil` (a real backlight) is not an option on this box: `/dev/i2c-*` is
root-only and there is no `i2c` group at all.

Everything is reversible, and the reversal is remembered on disk
(`~/.aiws_trainer/jarvis_memory/room_state.json` and `scene_state.json`, written
*before* the first change): Jarvis puts the display back at *"lights up"*, at **boot**
(so a crash at 0.55 heals at the next start), at quit, and immediately if any command
fails. Two ordering rules are load-bearing and are why this is a module rather than two
one-liners: night light and `--brightness` are the **same knob** (both end up in the CRTC
gamma ramp), so the night-light keys are written first and brightness is re-asserted
after them; and `night-light-schedule-automatic` is `true` by default, so forcing warmth
at three in the afternoon also means writing `automatic=false` plus a from/to window and
restoring all three afterwards.

Scenes are **data**. Edit `room.scenes` to add your own — the steps are `brightness`,
`temperature`, `music` (`pause` / `resume`), `quiet_hours` and `say`, applied in order,
and each one is reversed by *"good morning"*. A step that cannot run (no Spotify
credentials, no display) is skipped and logged; the rest of the scene still applies.

```json
"room": {
  "enabled": true,
  "wind_down_on_goodnight": false,
  "scenes": {
    "wind down": [
      {"do": "temperature", "kelvin": 2700},
      {"do": "brightness", "level": 0.6},
      {"do": "music", "action": "pause"},
      {"do": "quiet_hours", "start": "22:00", "end": "07:00"},
      {"do": "say", "line": "Powering down the workshop, sir."}
    ]
  }
}
```

`wind_down_on_goodnight` is **off** by default: *"good night"* already has its own
handler (the wind-down preview and tomorrow in one breath), and a scene is a change to
your desktop you did not ask for by saying good night. Turn it on and *"good night"*
runs the scene as well. `room.enabled: false` switches the spoken verbs off entirely,
and `JARVIS_ROOM_CONTROL=0` in the environment blocks every real `xrandr` / `gsettings` /
`pactl` call (the test suite sets it, because those commands act on your live session).

## 75. The Room Mixer (music bows under his voice)

While Jarvis speaks — and while he is listening — every stream on the box that is not
his own slides down to 30 % over about 200 ms, and slides back afterwards. Nothing to
configure to get it; it is on by default.

Two reasons it is worth having, and the second is the better one: he can answer at
conversational volume without shouting over the soundbar, and the microphone stops
hearing the music, which measurably helps Whisper, the endpoint detector and the speaker
gate.

- `audio.duck` — `false` turns it off entirely.
- `audio.duck_level` — the floor as a percentage of each stream's own volume (default
  30, clamped to 5–95).
- `audio.duck_ramp_ms` — how long the slide takes (default 200).

```json
"audio": {"duck": true, "duck_level": 30, "duck_ramp_ms": 200}
```

It only ever moves **individual streams** (`pactl set-sink-input-volume`), never the
sink: the default sink here is the Bluetooth soundbar, which is where Jarvis's own voice
comes out, so turning the sink down would turn him down with it. His own players are
recognised by **process ID**, not by name — librespot pipes through `pacat` and so does
`paplay`, so a name-based rule would either duck his voice or miss the music. That is
also why the alarm never bows under the spoken alarm line.

If the volume of a stream is ever left low by a Jarvis that died mid-sentence, the next
start puts it back: PipeWire remembers written volumes by application name, so the mixer
records what it moved before it moves it and heals it later — even a librespot restarted
the following day. When Spotify is playing on your phone or another Connect device there
is no local stream to duck, and the mixer stands down silently rather than pretending to
work.

### The first word, and the soundbar that went to sleep

One thing the mixer deliberately does **not** do is keep the Bluetooth link warm. On this
box WirePlumber suspends an idle node after three seconds
(`/usr/share/wireplumber/main.lua.d/90-enable-all.lua` loads `suspend-node.lua`), so if
the room has been quiet for a while the soundbar's A2DP link has to resume before the
first syllable comes out — and that resume eats the front of the word.

The tempting fix is a silent keepalive stream, and it is the wrong one: a stream that
plays forever is a stream the mixer then has to exempt from itself, and it holds the
radio open all night for nothing. The right fix is a WirePlumber drop-in, which is a
change to **your** session rather than to Jarvis, so Jarvis does not write it for you —
run it yourself, once (verified present and unprivileged here: WirePlumber 0.4.17, and
`~/.config/wireplumber/bluetooth.lua.d/` already exists):

```
mkdir -p ~/.config/wireplumber/bluetooth.lua.d
cat > ~/.config/wireplumber/bluetooth.lua.d/51-no-suspend.lua <<'LUA'
table.insert(bluez_monitor.rules, {
  matches = {{{ "node.name", "matches", "bluez_output.*" }}},
  apply_properties = { ["session.suspend-timeout-seconds"] = 0 },
})
LUA
systemctl --user restart wireplumber
```

The cost is honest: the soundbar's radio stays awake, so it will idle a little warmer and
a battery-powered speaker would drain. Undo it by deleting that one file and restarting
wireplumber again. Nothing in Jarvis depends on it — without it his first word is
occasionally clipped after a long silence, and that is all.
---

## Desk presence, the calendar anomaly watch, and learned walk times

Three additions of 2026-08-30. None of them needs a key, an account or a
network: everything below is local.

### Desk presence (`presence.desk*`)

Jarvis asks GNOME how long it has been since the keyboard or mouse moved:

```bash
gdbus call --session --dest org.gnome.Mutter.IdleMonitor \
  --object-path /org/gnome/Mutter/IdleMonitor/Core \
  --method org.gnome.Mutter.IdleMonitor.GetIdletime
(uint64 13478107,)      # milliseconds
```

No sudo, no configuration. It matters because the older Wi-Fi probe needs
`presence.phone_ip` or `presence.phone_mac`, and with those unset the
sentinel logs `presence: no phone_ip / phone_mac configured; sentinel idle`
— so the whole welcome-back and catch-up machine has never fired in the
room.

```json
"presence": {
  "desk": true,
  "desk_away_after_min": 25,
  "desk_poll_s": 30,
  "desk_standby": true,
  "desk_standby_alpha": 0.45
}
```

| key | what it does |
|---|---|
| `desk` | the whole feature. `false` and nothing below runs |
| `desk_away_after_min` | minutes of no keyboard/mouse before the chair counts as empty. Generous on purpose: idle time is keyboard and mouse only, so reading a paper at the desk looks like an empty chair |
| `desk_poll_s` | how often the idle monitor is asked (floor 5 s) |
| `desk_standby` | dim the Jarvis window while the chair is empty |
| `desk_standby_alpha` | how far it dims (1.0 = not at all, floor 0.2 so he can always find the window) |

What it does and — as importantly — what it does not:

* **Suppresses, never announces.** He will never say "nobody's home". The
  away signal goes into the same `quiet.hold_when_away` gate the phone
  probe already used, so proactive lines wait for the catch-up digest.
* **Greets the return once.** Both probes go through one greeter with a
  ten-minute damper, so if you ever do configure the phone as well, one
  walk through the door is one "Welcome back, sir".
* **A failure is silence, not a wrong answer.** No gdbus, no Mutter
  interface, a timeout, unparsable output — all read as "no signal", and
  the app behaves exactly as it did before this existed.
* **The board's dimming is reversible and always restored**: on any wake
  word, on any recording, and at quit.

Turn it off for one process without touching the config:

```bash
JARVIS_DESK_PRESENCE=0 ~/vss_env/bin/python -m jarvis.app
```

(The test suite sets that variable: the session bus is the developer's real
desktop and is the one piece of state that cannot be redirected into a
throwaway directory.)

### Calendar anomaly watch (`calendar.anomaly_watch`)

On by default. Every successful calendar refresh is diffed against the last
one, and a change to today or tomorrow earns one spoken line:

> Your 9:10 am BIOSENSORS has been cancelled, sir.
> Your 9:10 am BIOSENSORS has moved to 10:10 am, sir.
> Your 4:00 pm Chiro has moved to ETB 1020, sir.
> ADVISOR has been added to today at 3:10 pm, sir.

Only one line is spoken per refresh; the rest go into the catch-up digest,
and anything further out than tomorrow is filed silently. Quiet hours, DND
and a running class hold the line like any other unbidden speech.

It cannot cry wolf on a network outage: a failed source keeps its previous
events and the diff only runs when the refresh reports that some source
answered. Set `"calendar": {"anomaly_watch": false}` to switch it off. The
snapshot lives at `~/.aiws_trainer/jarvis_memory/calwatch_state.json`;
delete it and the next refresh simply relearns, silently.

### Learned walk times (`calendar.leave_times`)

The meeting heads-up speaks one global lead (`calendar.heads_up_min`, ten
minutes). Ten minutes is right for a call and useless for a lecture on the
far side of campus, so each building in your calendar gets its own lead —
**learned, never guessed, and never from a maps API**:

> **Jarvis:** BIOSENSORS in ten minutes, sir.
> **Jarvis:** How long do you need to get to Wisenbaker, sir?
> **You:** About twelve minutes.
> **Jarvis:** Wisenbaker, 12 minutes. I'll have you moving in good time, sir.

and from then on, before every event in that building:

> You want to be walking in 5 minutes, sir; Wisenbaker is a 12 minute walk.

```json
"calendar": {"heads_up_min": 10, "leave_times": true, "leave_notice_min": 5}
```

You can also teach, amend and query it outright:

| say | effect |
|---|---|
| "it takes ten minutes to get to Wisenbaker" | stores the walk |
| "it's a 12 minute walk to the ETB" | same |
| "the walk to Zachry is eight minutes" | same |
| "make that ten next time" | amends the building he last mentioned |
| "how long to Wisenbaker" | reads it back, or admits he does not know |

Rules worth knowing:

* He asks about a building **once, ever** — and only in passing, inside the
  heads-up window, never while quiet hours are on and never over a turn in
  flight. Answer "no idea" and he never asks again.
* He **never asks about a Zoom link or an empty location**, and never
  invents a building he has not seen in your calendar.
* The reply he accepts as an answer must be plainly a duration, so a timer
  you set in the same minute is still a timer.
* Two rooms in one building are one walk: `Emerging Technologies Building
  1003` and `… 1020` share a lead.

The walks are stored in long-term memory as `leave_lead.<building>`
preferences, not in `assistant.json` — they are facts about you, not
settings. To see or clear them:

```bash
cd ~/Jarvis && ~/vss_env/bin/python - <<'EOF'
from jarvis.memory import JarvisMemory
mem = JarvisMemory()
print({k: v for k, v in mem.get_all_preferences().items()
       if k.startswith("leave_lead.")})
EOF
```

## 76. The pre-class dossier and class-start staging

Two things that happen on their own around a lecture. Both hang off the
same idea: **a course is a calendar slot, not a Canvas course**. Canvas
cannot name your courses here — `canvas.token` is empty, and
`cached_course_names()` is read-only by design so the transcriber can never
trigger a fetch — so `jarvis/courses.py` reads identity off the calendar
instead: a timed title seen at the same weekday and clock time on two or
more distinct dates is a class. On the live cache that is exactly
BIOSENSORS, MAGNETIC RESONANCE ENGR and ELECTRICAL DESIGN LAB II, and none
of the one-off appointments.

### At T-10: the dossier

You used to hear "BIOSENSORS in 10 minutes" and nothing else. Now:

> Your 9:10 is BIOSENSORS, Wisenbaker 049, sir. Last time you noted
> electrode drift; Lab 3 report is due Thu 11:59 pm, and there's unread
> mail from Priya.

…and one card on the HUD:

```
CALENDAR   9:10 — BIOSENSORS
ROOM       College Station Wisenbaker Engineering Bldg 049
LAST TIME  electrode drift (biosensors-2026-08-26.md)
DUE        Lab 3 report — Thu 11:59 pm
MAIL       Priya — Lab 3 questions
```

Where the event's location is a URL — your Thursday Zoom is exactly that —
the card carries a **JOIN** row with the link instead of a room, and he
says "the link's on the card" rather than reading a URL aloud.

Every section is optional and is simply absent when its source is. Today
three of the four are dark: `~/Documents` does not exist, so there are no
previous notes to find, and `canvas.token` is empty, so there are no
deadlines. The room alone is still worth the line. Fill either one in and
that section starts appearing with no further setup.

```jsonc
"dossier": {
  "enabled": true,      // false: back to the bare "TITLE in ten minutes"
  "lead_min": 10,       // how early the line lands
  "notes": true,        // the LAST TIME row (reads the notes folder)
  "mail": true,         // the MAIL row (one IMAP pass, on a worker thread)
  "mail_hours": 72,     // how far back an unread message still counts
  "due_days": 7,        // the Canvas window for the DUE rows
  "budget_s": 25        // hard cap on the whole gather
}
```

Mail has to clear two bars, not one: the sender must be someone in your
people book ("my advisor", "Mom" — see section 22) **and** the course must
be named in the subject or the snippet. A newsletter with "biosensors" in
the subject line is not news.

The line is proactive, so quiet hours and do-not-disturb hold it for the
catch-up digest like everything else he decides to say on his own (section
34). The card is published either way.

### At the hour: the desk is already set

The moment the class starts, without a word:

* today's notes file exists at `<docs folder>/notes/<course>-<date>.md`
  with its dated header — the page is open, nothing is being recorded;
* the music is paused, and comes back when the class ends;
* the card reads `BIOSENSORS staged — notes ready`.

```jsonc
"class_flow": {
  "enabled": true,
  "auto_notes": false,  // arm voice capture for the hour (see below)
  "open_notes": false,  // xdg-open the file as well
  "duck_music": true,   // pause Spotify for the class, resume at the end
  "idle_min": 15        // how long away from the keyboard still counts as here
}
```

Two of those are off on purpose:

* **`auto_notes`.** `LectureNotes.add()` appends every accepted utterance
  to disk. Arming that from a calendar tick would record a room you never
  agreed to record, so it is yours to switch on — and when you do, Jarvis
  says "Taking notes for BIOSENSORS, sir" out loud, so nobody in the room
  is being written down silently. It closes itself at the end of the hour.
* **`open_notes`.** Opening an editor means a new window on `:1`, and
  window churn is what froze this desktop on 2026-08-26. Everything of
  value — a primed, dated file and quiet music — costs zero windows.

Neither needs a config edit in the moment: "notes for biosensors" (section
18) arms capture on the file that is already sitting there, one sentence
away, and closes it with "end notes".

**Nothing is staged into an empty room.** The phone-presence sentinel
cannot help here (`presence.phone_ip` is empty, so it never starts), so the
gate is the real one: `XScreenSaverQueryInfo` on `:1` reports true idle
milliseconds — keyboard and mouse, no sudo, no extra hardware — and falls
back to the timestamp of your last turn in `turns.jsonl`. If nothing can
measure it at all it fails **open** and stages anyway; a gate that says
"away" on a box it cannot read is the inert gate this replaces. Walk in two
minutes late and it still stages: a refused gate is not remembered, and the
offer stands for five minutes.

Do-not-disturb, quiet hours and being out stop it. A running calendar event
does **not** — because with course detection now feeding `quiet.py` the
class itself is a quiet window, and a gate the event trips the instant it
starts would never let the stager run.

Everything it changes is written down and put back: at the end of the
event, at quit, if the staging itself fails halfway, and at the first wake
word after the class is over (the safety net for an end tick that never ran
because Jarvis was down). Never mid-lecture — the music is meant to stay
down for the hour.

### The bug this fixed on the way past

`quiet.calendar_keywords` is `class / exam / meeting / busy`, matched as
whole words against the event title. None of your courses is *called* any
of those, so the calendar leg of quiet hours had never once fired. A
recurring course now counts as a running class whatever it is named
(`quiet.calendar_courses`, on by default; set it false for the old
keyword-only behaviour).

```bash
# what he thinks your courses are, read-only, no network
cd ~/Jarvis && ~/vss_env/bin/python - <<'PY'
import json
from jarvis import courses
from jarvis.tools.calendar import Event
from jarvis.tools.location import cache_dir
d = json.loads((cache_dir() / "calendar_cache.json").read_text())
evs = [Event.from_dict(e) for v in d["sources"].values()
       for e in v.get("events", [])]
print(courses.recurring_courses(evs))
PY
```

## Register, continuity and the self sheet

### "Formal mode" before your advisor arrives

Say **"formal mode"** (or "be more formal", "be serious", "cut the jokes")
and Jarvis drops the asides and the joke family for good — said once, it is
still formal tomorrow. **"Banter up"** ("more banter", "loosen up", "dial up
the wit") goes the other way; **"back to normal"** ("your usual", "banter
down", "less formal") returns him to the middle.

It is one line in the config, and you can set it by hand:

```jsonc
"persona": { "register": "normal" }        // "formal" | "normal" | "banter"
```

Why it is worth knowing about: the register is baked into the **static**
half of the Tier 2 prompt, which every question re-sends byte-for-byte so
Ollama can keep it cached. Changing it therefore costs exactly one
~2700-token reprocess, paid on a background thread the moment you say it,
while Jarvis answers with a fixed line straight away. The naive version of
this feature rebuilds the prompt every turn and pays that cost on every
question you ask. If you ever want to check the invariant yourself:

```bash
cd ~/Jarvis && JARVIS_SHOT_SEED=1 ~/vss_env/bin/python - <<'EOF'
from jarvis import brain
a = brain.static_system()
print("stable:", all(brain.static_system() == a for _ in range(50)))
print("changed once:", brain.set_register("formal"), brain.static_system() != a)
print("and no more:", brain.set_register("formal") is False)
EOF
```

Formal also changes the short courtesies ("Yes, sir." rather than "Always,
sir."), and it gets the plain diagnostics sheet instead of the film-register
one — the register that bans asides bans that one too. Every variant is a
fixed string, so they are all prewarmed into the speech cache and the
register costs no latency at all.

### "That would be the third coffee timer, sir"

Nothing to switch on. Every Tier 2 turn now carries a short **"Earlier
today"** block in its background: the same tool run with the same argument
more than once (the third *coffee* timer, the third weather check), and a
question you asked earlier and have come back to. The counts come from the
activity journal, which is the only place a tool's argument is kept —
`~/.aiws_trainer/jarvis_memory/journal/<date>.jsonl`, the same file "recap my day"
reads.

Two rules keep it tasteful: a count of one is never mentioned, at most two
clauses appear, and a repeated *question* only counts once the first asking
has scrolled out of the four recent exchanges the model can already see.
The block also carries its own instruction — the numbers colour his reply,
he never reads one aloud unless you ask him how many. If you want the
number, ask for it.

The block is re-read from disk at most every 30 seconds, and any journal
write drops that cache, so a timer you set this second is counted on the
very next thing you say.

### "How are you?" now has an answer

The old reply was one of three canned lines opening with "All systems
nominal, sir." — the one line in the product that sounded like a toy, and a
phrase his own voice rules forbid. It is gone from the courtesy and from the
diagnostics sheet.

"How are you?", "how do you feel?" and "are you busy?" are now answered from
what he actually is: whether the local model is lent to a trainer, whether
the GPU is throttled, whether quiet hours is holding messages for you,
whether your voiceprint was ever enrolled, and how many turns he has taken
today. It is one clause, it costs no model call, and the variants that carry
no number are prewarmed so the fastest exchange in the system stays fast.

### "Run diagnostics", in the register of the films

The same probes, spoken rather than read out:

> Power to the local model at full, sir. The GPU is running at 54 degrees,
> 2424 megahertz, 2 percent busy. 4 terminals of yours on the board, 2
> mid-turn. 9 turns today, and I've been up 3 hours and 12 minutes.

The plain sheet — the same numbers, in order — goes on the card, and stays
what `jarvis.ask --status` and the command socket's `status` return, so
nothing that scripts against it changes.

Four things it now knows that it did not before: the GPU's clock and
utilisation, whether the model is resident or lent out, how many of *your
own* `claude` tmux sessions are live (and how many are mid-turn), and the
kind of output your voice is actually leaving by. The clock is the one that
matters on this box: a wedged GB10 sits at 611 MHz against a healthy 2400,
and idle power draw reads about 15 W in both states, so the draw is not a
tell and is never spoken. Below 1200 MHz he says the GPU is dragging its
feet.

Nothing here is phrased by the model. Every figure is a reading, and if the
counters cannot be read he says so in a sentence rather than quietly
shortening the report.

## 77. Working sessions: "let us plan the week"

Say **"let's plan the week"** (or "plan my week", "sort out my week") and he
walks this week's Canvas deadlines and open to-dos one at a time, proposing
a day and an hour for each:

> — Lab 3 report for BIOSENSORS — Tuesday at four?
> — Move that to Thursday.
> — Thursday instead. Buy milk — Tuesday at four?
> — Skip it.
> — That's enough.
> — That's the week, sir: Thursday at four, the Lab 3 report. I've set a
>   reminder for each.

The four answers are **yes** (any of "yes / sure / go on / that works /
book it"), **a day** ("move that to Thursday", or just "Thursday"), **skip
it** ("no / skip / leave it / pass") and **that's enough** ("that'll do",
"we're done"), and none of them needs the wake word. Say anything else —
"what's the weather?" — and the session ends and your words are answered as
a normal command; nothing traps you in a dialogue.

What he writes: one **timekeeper reminder** per accepted slot, five minutes
before the hour, and nothing else. No calendar events — a plan built in
ninety seconds should stay cheap to undo. Cancel one the usual way
("cancel the reminder about the lab report").

Two behaviours worth knowing:

- The mic stays open ~18 s between turns of a session, rather than the
  usual 4 s follow-up window. That is the only reason a two-clause answer
  like "or push it to Wednesday" is possible at all.
- If you walk away, the session goes stale after 90 seconds and the next
  thing you say is a fresh command.

He refuses to start a session while dictation or lecture notes are open
("I can't plan while I'm taking notes, sir") — those modes swallow every
utterance, so the question could never be answered.

The machinery underneath (`jarvis/dialogue.py`) is a small protocol —
`ask` / `settle` / `stop` / `stale` / `finished` — that any future
multi-turn feature can rent. The week planner is its first tenant.


## 78. The fault lane: he tells you once, and the board remembers

The health watchdog already knew when memory was tight or two processes
were eating the unified pool. The problem was that it said so into a
four-second toast: a warning raised while you were out of the room left no
trace at all.

Now the engine card carries a **FAULT** row. It reads `--` almost always,
and when it does not it holds a short token until the fault actually
clears:

```
HEAR    SMALL
SPEAK   F5 · LOCAL
THINK   GEMMA4
DEVICE  GB10
FAULT   2 TRAINERS
```

Ask **"what's wrong"** (or "anything wrong", "what's the matter") and he
reads the live fault back with how long it has been standing. With the
board clear, that same question falls straight through to the existing log
triage, so it is never a dead end.

The new rule behind the token: **two distinct training runs on the pool at
once** — the exact shape of the 2026-08-28 hard power-off. It counts runs,
not processes, so a `torchrun` job spread over four workers is one run and
says nothing; `train.py` alongside `finetune_piper.py` is two, and raises
one error naming both.

He says each fault **once**. The watchdog's own latches handle the repeat
within a session; `~/.aiws_trainer/jarvis_memory/faults.json` handles it
across restarts, so starting Jarvis while the pool is still tight does not
re-announce the same episode. When the fault clears, that entry is dropped
and the next occurrence is news again.

Thresholds are the existing `health` block — nothing new to configure:

```json
"health": {"warn_gb": 16, "critical_gb": 8, "hog_gb": 20, "interval_s": 30}
```


## 79. The run ledger: "that's done, sir; twenty-two minutes"

Start a training run and, until now, the room went quiet for the twenty
minutes that mattered. The run ledger gives a run two beats and an honest
duration:

> Your finetune_piper.py has started, sir; I'll tell you when it's done.
> …
> That's your finetune_piper.py done, sir; 22 minutes.

The start time is read from `/proc/<pid>/stat`, not from when Jarvis
happened to look — so a run that was already going when you restarted him
still reports its true age. A run shorter than a minute is recorded but
not announced (that was a crash or a typo). A trainer that respawns between
epochs is not announced as finished: it has to be gone for two ticks.

If you have `health.yield_to_trainer` on, the two beats are folded INTO the
lines you already get, so the count of spoken lines per run does not go up
— the reclaim line simply carries the duration ("Your trainer has finished,
sir; that took 22 minutes.").

Say **"quietly please"** (also "keep it down", "stop narrating") to hold
the narration for the run in progress. It lifts by itself when that run
ends; there is nothing left switched off.

### Epoch narration (opt-in, needs a wrapper)

He cannot read a trainer's output on his own: `/proc/<pid>/fd/1` on a real
run is a socket or a pty, and his own log knows nothing about epochs. Start
runs through the wrapper instead:

```bash
scripts/runlog.sh python train.py --epochs 20
```

It tees stdout to `~/.cache/jarvis/runs/<pid>.log` — the trainer keeps the
wrapper's pid, so the filename is the one Jarvis sees in `/proc` — and
prunes logs older than a week. Then turn it on:

```json
"runwatch": {
  "narrate": true,
  "progress": true,
  "log_dir": "~/.cache/jarvis/runs",
  "min_run_s": 60,
  "progress_gap_s": 300
}
```

> Epoch 4, sir; the loss is still falling.

At most one such line every five minutes, and only when the epoch number
actually **changed** — never a heartbeat. "The loss is still falling" is
said only when two readings actually support it. Set `narrate: false` to
keep the ledger (and the board's lane) while saying nothing at all.

All of this rides the health watchdog's existing 30-second tick. There is
no extra thread and no `nvidia-smi` call — the wedge these features exist
to warn about is precisely the state in which `nvidia-smi` blocks forever.

## 80. The Oracle box: "how are the bots, sir?"

Your Oracle Cloud VM is **`demon-bot`** — `opc@163.192.101.18`, Oracle Linux
Server 9.6, up ten weeks, 20 GB of a 30 GB disk used. It runs **nine app
services under systemd**, behind nginx and fail2ban, with docker and
containerd alongside:

| service | unit | what it is |
|---|---|---|
| Haymaker | `haymaker-bot` | Haymaker Discord Bot |
| Court of Awe | `coa-bot` | Court of Awe Discord Bot |
| Exoshock | `exoshock-bot` | Exoshock Discord Bot |
| VRider | `vrider-bot` | VRider Discord Bot |
| Timecard | `timecard-bot` | Discord Timecard Bot |
| Knightfall | `knightfall-web` | Knightfall Protocol web — heartbeat endpoint + admin dashboard |
| the elevation API | `elevation-api` | Ditch Grade elevation-api — projects sync + log sink |
| Monday sync | `monday-sheets-sync` | Monday.com → Google Sheets sync |
| the dashboard | `bot-dashboard` | Discord Bot Dashboard |

Ask, and the box answers for itself:

> **You:** how are the bots?
> **Jarvis:** All nine services are up on demon-bot, sir; ten weeks and a day
> of uptime and a third of the disk free.

…with the sheet on the card beside it: hostname, uptime, load, memory, disk
and every one of the nine with its systemd state. When something is not
running, that leads the sentence instead:

> **Jarvis:** Haymaker is down, sir; the other eight are up.

That is one ssh round trip — it measures **1.0 s** against the real box — and
it answers the single-service questions too, out of the same roll-call.

This lane is **outbound only**. Jarvis asks the Oracle box questions. Nothing
opens the other way: there is no tunnel, no reverse tunnel, no port-forward
and no setting here that exposes the Spark to the internet — deliberately, and
there is a test that asserts it rather than trusting the comment.

### Switching it on is one line

There is **no credential problem**. The key already on this machine works:

```
~/Downloads/Oracle Cloud Service (2)/Oracle Cloud Service/Discord Bot/Keys/ssh-key-2025-08-15.key
```

It is mode 0600 and it logs in as `opc` — verified. `oracle.key_path` in the
shipped defaults already points at it, along with the right host and user, so
the whole edit in `~/.config/jarvis/assistant.json` is:

```json
"oracle": { "enabled": true }
```

The rest of the section (host, user, key_path, the nine services) is merged
in from the defaults, so you do not have to repeat any of it. No restart is
needed for the voice commands — the config is read fresh on every question —
but the `oracle_status` tool the local model can call is only registered at
boot, so restart Jarvis if you want that too.

If you would rather the key did not live in `~/Downloads` (it is a private
key, and that is not a home for one), move it and point `key_path` at the new
place:

```bash
mkdir -p ~/.ssh/oracle && chmod 700 ~/.ssh/oracle
cp ~/Downloads/"Oracle Cloud Service (2)"/"Oracle Cloud Service"/"Discord Bot"/Keys/ssh-key-2025-08-15.key \
   ~/.ssh/oracle/oracle-key
chmod 600 ~/.ssh/oracle/oracle-key
ssh -i ~/.ssh/oracle/oracle-key opc@163.192.101.18 'echo ok'
```

A key copied off Windows often lands mode 0644, and ssh refuses it outright
("UNPROTECTED PRIVATE KEY FILE"). Jarvis checks the mode himself and says so
in those words rather than blaming authentication.

Until `enabled` is true, every phrasing gets one honest line naming exactly
what is missing, and **nothing opens a socket**:

> The Oracle box is switched off in my settings, sir; set oracle.enabled to
> true in ~/.config/jarvis/assistant.json.

### What you can say

| say | what happens |
|---|---|
| "how are the bots" · "how's the Oracle box" · "are all the services up" · "what's running on Oracle" · "Oracle status" | the roll-call sentence + the card |
| "how's the haymaker bot" · "is knightfall up" · "how's the elevation api" · "what about monday sync" · "is vrider running" | that one service, out of the same round trip |
| "show me the haymaker logs" · "what do the knightfall logs say" · "the logs for monday sync" | `journalctl -u <unit> -n 20`, with how many of those lines mention an error |
| "restart the haymaker bot" · "Oracle restart knightfall" | **read back first** — "Restart Haymaker on the Oracle box, sir?" — and only a yes runs it |
| "show me the oracle logs" · "restart the bot" | nine journals are not one answer and nine bots are not one bot, so he asks which, with the list on the card |
| "is the server up" · "how's the bot" | the roll-call, but only once the lane is switched on. With it off these name no box, so they go to the model instead of Jarvis claiming your local dev server |

Naming a service is generous: "haymaker", "the haymaker", "haymaker bot" and
"haymaker-bot" are one name, and so are "coa" / "court of awe", "monday" /
"monday sync" / "monday sheets sync", "elevation" / "ditch grade" / "the
elevation api". A name that matches **more than one** service resolves to
none of them — restarting the wrong bot because two matched is the failure
this lane is built to avoid.

Note that a bare service name only answers once the lane is **on**. With it
off, "how's the haymaker" goes to the model, because `~/haymaker-digest` is a
job on *this* machine and answering for a server Jarvis has not been told
about would be a confident wrong answer.

Ask twice in a row and the second answer is instant: the last good reading is
kept for `cache_s` seconds, so "how are the bots" followed by "and
knightfall?" is one round trip, not two. Running any action throws that
reading away — a restart must never be reported off a reading taken before it.

### The rules it keeps

**Three actions, and nothing from your voice reaches a shell.** The only
things Jarvis does on that box are read the roll-call, tail one journal and
restart one service. A spoken name is resolved against the `oracle.services`
table and the command is then *built* from a fixed template plus that row's
unit name — which had to look like a systemd unit to be loaded at all. Not
one character of the transcript is ever interpolated into a command. Anything
else said at the box is refused out loud rather than guessed at or handed to
the model:

> **You:** run deploy on the Oracle box
> **Jarvis:** I've nothing called "deploy" on the Oracle box, sir.

> **You:** stop haymaker
> **Jarvis:** I only do status, logs and a restart on the Oracle box, sir.

That second one is refused *aloud* on purpose. Silence would leave you
believing the bot had been stopped.

**A restart needs sudo, and it has it.** As `opc`, a plain
`systemctl restart haymaker-bot` over ssh is refused — polkit wants
interactive authentication and a non-interactive ssh has no agent to give it
(`pkcheck --action-id org.freedesktop.systemd1.manage-units` says so in as
many words). `opc` does hold passwordless sudo, so the command Jarvis
actually sends is:

```
sudo -n systemctl restart <unit> && systemctl is-active <unit>
```

The `-n` is load-bearing on its own: if sudo were ever locked down on that
box, without it this would sit on a password prompt for the whole timeout
budget. The `is-active` tail is why he can tell you what happened rather than
just "done":

> Haymaker is back up, sir.

**It cannot hang the turn.** One ssh round trip, `timeout_s` seconds, and the
child is killed *without being waited for* — the same shape as the
`nvidia-smi` call in the health tool, and for the same reason: a
`subprocess.run` timeout kills the child and then blocks waiting for it, and
an ssh stuck in a TCP connect against a host that silently drops packets never
exits. Past the budget he says so:

> The Oracle box didn't answer in 6 seconds, sir; I've stopped waiting on it.

**The memory number is not the one /proc reports.** demon-bot's kernel says
`MemAvailable: 20512504 kB` on a box with `MemTotal: 5779324 kB` — nineteen
gigabytes free on a five-and-a-half gigabyte machine. That is a real reading
from a real OCI aarch64 kernel, and procps' own `free` guards against it by
falling back to `MemFree`, which is why `free -h` says 1.7 Gi available while
`/proc/meminfo` says 19.6. Jarvis does the same clamp; without it the card
would confidently report more free memory than the box has.

**There is no pm2 on that box.** An earlier version of this section described
a `game-news` bot under pm2 at `170.9.245.136`, out of a PDF cheat sheet. That
was an *older server*. Everything above was read off `demon-bot` itself.

## 80. Jarvis on your phone (home Wi-Fi only)

You asked for an app that gets you to him. This is that — a page the Spark
serves to your phone on the home network, with a text box that goes through
exactly the same door as the typed box and the `jarvis` command line, quick
buttons for the things you actually ask for, his own voice in your ear if
you want it, and an icon on your home screen that opens it like an app.

It is **off** until you turn it on, and it is **LAN only**: it binds one
private address and refuses to start on anything routable from outside the
house. There is no tunnel, no port-forward and no cloud leg anywhere in it.
Reaching him from off the network is a separate decision and it is yours to
make deliberately, not something this feature quietly does for you.

### Turning it on — the four steps

1. Open `~/.config/jarvis/assistant.json` and set:

   ```json
   "phone": {
     "enabled": true,
     "bind": "",
     "port": 8765,
     "token": "",
     "max_audio_mb": 8,
     "link_file": "~/jarvis-phone.txt",
     "qr_file": "~/jarvis-phone.svg"
   }
   ```

   Leave `bind` empty: he works out this box's own address on the home
   network at start. Leave `token` empty: a key is generated on the first
   start and written back into this file.

2. Restart Jarvis (`python -m jarvis.app`). The log line to look for is:

   ```
   phone client listening on http://192.168.50.x:8765/ (LAN only)
   phone client: the link is in /home/hunterp/jarvis-phone.txt
   ```

3. `cat ~/jarvis-phone.txt`. It holds the URL with the key in it, and it is
   mode 0600 — the link *is* the key, so treat that file the way you treat
   the config it came from. There is a QR beside it at `~/jarvis-phone.svg`;
   open it on the big panel and point the phone camera at it rather than
   typing forty-three characters on a phone keyboard.

4. On the phone: open that URL in Safari, wait for it to say "on the home
   network" at the top right, then **Share → Add to Home Screen**. It gets
   the roundel icon, opens full-screen with no browser chrome, and remembers
   the key.

If you ever want a new key: delete the `"token"` value from
`assistant.json` and restart. The old home-screen link stops working, and
step 3 gives you a new one.

### What is on the page

A transcript, a text box, and a row of taps for: **what's due**, **next
exam**, **calendar**, **briefing**, **timer 10m**, **diagnostics**, **the
board**, **lights up**, **lights down**. Each tap sends the same sentence
you would say out loud, so a button and your voice take the same route
through the commander — there is no second set of phone-only commands to
keep in sync.

`status`, `diagnostics` and `board` are **reads**: they print the same
sheets the CLI prints, without dispatching a turn, so they are not
remembered as a conversation and never wake the speaker.

### The two switches, and why they are two

Under the talk button there are two, and they are two rooms, not one
setting with a volume knob. **Both start off.**

* **Voice** plays his answer out of *this phone*, in his own voice.
* **Aloud in the room** makes the *Spark* answer the house as well — the
  same choice `jarvis --send-audio --speak` makes.

With both off you get text, and nothing anywhere makes a sound. With Voice
on and Aloud off — the useful setting away from the desk — he speaks in
your ear and the room stays silent. With both on he answers in both places,
which is what you want standing in the kitchen with the phone in your hand.

Aloud is off by default because a question asked from bed should not answer
the house. Voice is off by default because half of these turns happen in a
lecture, and a phone that starts talking on its own in a lecture theatre is
worse than one that says nothing. Neither switch changes anything else; he
still never barges in on a reply he is already speaking to somebody
standing in the room.

### His actual voice, on the phone

Tap **Voice** and the next answer arrives as audio as well as text. It is
the real thing — the same F5 voice, the same reference clip, the same
pronunciation dictionary and the same sentence splitting the Spark's
speaker gets — not the browser's built-in speech synthesis, which would be
a generic robot reading his lines and would make him sound like somebody
else.

It is also fast, for a reason worth knowing: the phone shares the Spark's
**speech cache**. Every line he renders is filed on disk under its engine,
voice settings and exact text, so anything he has said before — the canned
replies, a repeated question, a briefing preamble — is already there:

| | to the first audio byte | the whole clip |
|---|---|---|
| a line already in the cache | **0.6–1.7 ms** | the same (sent whole) |
| a fresh one sentence line | **350 ms** | 350 ms |
| a fresh two sentence reply | **366 ms** | 731 ms |
| a fresh four sentence reply | **368 ms** | 1589 ms |

Measured over real HTTP against the resident F5 sidecar. The interesting
column is the first one: a fresh reply is **streamed sentence by sentence**
rather than rendered, saved and then sent, so the phone starts playing the
first sentence while the fourth is still on the GPU — a second and a
quarter earlier, on a long answer, than waiting for the file.

**Tapping the switch is not decoration.** iOS will not let a page start
audio unless a real finger started it, so the audio pipeline is opened on
the tap that turns Voice on, and re-armed every time you send a question
(which is also a tap). That is why the switch exists as a switch instead of
the page just playing everything. Your choice is remembered on that phone,
so you tap it once. If the phone refuses anyway, the transcript says
"holding audio back — tap Voice once more" rather than going quiet and
leaving you to guess.

Turning Voice **off** cuts whatever is playing and stops any fetch in
flight, and while it is off nothing is requested and nothing is rendered —
there is no clip made and thrown away.

If the box has no speech engine at all, the switch renders dead and says
"Voice unavailable" instead of looking alive and answering in silence.

### "Why is the first question slow?"

It usually is not him. Each answer now carries the server's own timing, and
the page compares it with the wall clock: when the link added more than
0.7 s it says so, naming both halves. For reference, a Tier-1 command
("what time is it", "set a timer for ten minutes") takes **39–120 ms**
inside the Spark, on about **1 ms** of HTTP.

The rest is the network. Over a tailnet the first packets after a spell
away go through a relay while NAT traversal negotiates a direct path, and
that shows up as a slow first question and fast ones after it. Nothing in
this app configures Tailscale and nothing here should — but the page does
hold the path open on its own: while it is in front of you it pings once
every 20 s (and again the moment you come back from the lock screen), which
keeps both the tailnet's path and this server's 30 s keep-alive warm, so
the first question after a pause is not the one that pays for the
handshake.

### The microphone: the honest version

**Push-to-talk will not work on your iPhone over this link, and that is not
a bug you can wait out.** Browsers only hand a page the microphone on a
*secure context*. `http://192.168.50.x` is not one — not in Safari, not in
Chrome, not in Firefox. On iOS Safari the microphone API is not merely
refused, `navigator.mediaDevices` does not exist at all.

So the text path is the product here. It is first in the layout, it is what
the buttons drive, and it works on every device on the network today.

The talk button still ships, and it is wired to a real endpoint that hands
the clip to the same intercom code the SSH path uses (decode, resample,
speaker gate, Whisper). When the page is not in a secure context the button
renders visibly dead, with the reason and the fix written underneath it —
never a button that looks alive and silently does nothing. Two things would
bring it to life:

* **A certificate this phone trusts.** Serve the same page over https with a
  cert you have installed and trusted on the phone. That is a real chunk of
  work (a local CA, a profile installed on iOS, and a renewal you will
  forget about), which is why it is not step 5 above.
* **Opening it on the Spark itself**, at `http://127.0.0.1:8765/`. Localhost
  *is* a secure context, so the microphone works there — of no use from bed,
  but useful for testing the leg.

One format note for that day: `MediaRecorder` would rather give you a WebM
container, and libsndfile cannot read WebM. The page asks for
`audio/ogg;codecs=opus` first and falls back; if what arrives is
undecodable you get his own line about wav / ogg / opus / flac rather than
silence.

**If you are already reaching this page over https** — a reverse proxy in
front of it that terminates TLS with a certificate the phone trusts, which
is what a `tailscale serve` in front of `127.0.0.1:8765` gives you — then
the page *is* a secure context and the talk button comes alive by itself,
with no change here. That is the "certificate this phone trusts" case
above, arrived at from the other end. Nothing about it is configured from
this app, and the server underneath is unchanged: it still binds one
private address, still refuses a non-private peer, and still wants the key
on every `/api/*` call.

**Voice does not need any of that.** Playing his answer through the phone
uses `fetch` and Web Audio, neither of which is gated on a secure context,
so the Voice switch works over plain http on the Wi-Fi exactly as it does
over https. It is only the *microphone* that browsers hold back.

Until then, the fast way to talk to him from the phone is still the one in
§62 — record in Termux and pipe it over the SSH session you already have.

### What is guarding it

Worth knowing, because it is the whole reason this is a page and not a hole
in your network:

* **Off by default.** `phone.enabled` is false in the shipped config;
  nothing listens on a port until you change it.
* **One private address.** Never `0.0.0.0`, never a public interface. The
  check is "is this address routable on the internet" rather than a list of
  ranges someone wrote from memory — carrier-grade NAT would have slipped
  through such a list. A configured address that fails it does not start,
  and says so in the log.
* **A bearer key on every acting endpoint**, compared in constant time. The
  page itself is static markup with no data in it and cannot do anything;
  every `/api/*` route asks for the key.
* **Only private peers**, refused before anything is routed.
* **Rate limits and body caps.** Sixty requests a minute per device, twelve
  of them audio; 16 KB of text; `max_audio_mb` of audio, never more than the
  intercom's own ceiling. An oversized body is refused from its declared
  length, before it is read.

### If it does not come up

* `grep "phone client" /tmp/vss_voice/jarvis.log` — every refusal is a line
  there. "no private LAN address found" means the box could not work out its
  own address; set `phone.bind` to it by hand.
* "could not bind" with the port in it means something else already has
  8765; change `phone.port`.
* The page says "no key" or "that key was refused": the key in your
  home-screen link is stale. Re-read `~/jarvis-phone.txt` and add it to the
  home screen again.
* The page loads but nothing answers: Jarvis himself is down, or the phone
  has dropped onto cellular. It only answers on the home Wi-Fi.
* The Voice switch says "Voice unavailable": this Jarvis has no speech
  engine loaded at all — `grep "f5 sidecar" /tmp/vss_voice/jarvis.log` and
  `systemctl --user status jarvis-f5.service`.
* Voice is on and the text arrives but nothing is heard: look in the
  transcript. "holding audio back" is the phone refusing to start audio —
  tap Voice off and on again, which gives it the finger-press it wants.
  "His voice did not reach this phone" carries the server's own reason.
  `grep "phone say" /tmp/vss_voice/jarvis.log` shows every clip, how many
  of its sentences came from the cache, and the milliseconds to the first
  byte.
* Answers are slow the first time and quick after: read the line the page
  prints when the link is the slow half. See "Why is the first question
  slow?" above — it is the tailnet finding a direct path, not Jarvis.


## 81. Emailing a file: "email the lab report to Heather"

Say it the way you would say it to a person:

```
"Jarvis, email the lab report to Heather"
"send the PDF I just downloaded to my brother"
"email that file on my desktop to heather@example.com from my school account"
"send Heather the biosensors handout"
```

**Nothing is sent by the first sentence.** Jarvis finds the file, works out
who you mean and which of your accounts to use, and then reads the whole
thing back:

> "Biosensors Lab Handout v2.pdf, 5 kilobytes, to Heather, at heather at
> example dot com, from your school account. Send it, sir?"

Say **yes** (or "send it", "go ahead", "do it") and it goes. Say **no**
(or "not that one", "wrong file", "hold on") and it is dropped. Say
anything else — change the subject, ask a different question — and the draft
is thrown away and your sentence keeps its own meaning. A vague "okay" or
"sure" gets asked once more rather than obeyed: this is the one question in
the app where "probably yes" is not enough, because an email cannot be
recalled.

### When it asks instead of guessing

| What happened | What you hear |
|---|---|
| Two files fit the name | "I've 2 that could be the lab report, sir: lab report.pdf or lab report final.pdf. Which one?" |
| Nothing fits | "I can't find a file by that name, sir." |
| You did not name a file at all | "Which file, sir?" |
| The person is not in the book | "I've no address for Dana, sir. What is it?" |
| More than one account, and you did not say which | "Which account should I send from, sir — personal, work or school?" |

It never guesses an address from a name, and it never picks between two
files that fit equally well.

**"Which one?" is a real question and it waits for you.** Answer it with an
ordinal ("the second one", "the first one", "the last one", "the other
one") or with the name itself ("the final one", "lab report final") and
Jarvis reads that file back for the usual yes. Say "neither" and it is
dropped. The microphone stays open for the answer, as it does for every
other question he asks. Choosing a file confirms nothing: the read-back
still has to be answered — and repeating the phrase that was ambiguous in
the first place gets the question again, never a guess at which of the two
you meant.

**A yes has to come from where the question was asked.** The read-back is
spoken at the desk, so it is answered at the desk — by voice or by typing
into the same window. A "yes" arriving from Discord, the phone client, a
`jarvis "..."` in a terminal or the socket never heard the question and
sends nothing; the draft is left where it is, waiting for you.

**Naming the account takes a word, not a letter.** "from my school account"
works, and so does "sch"; a single letter does not, and neither does the
local part of the address on its own. An unrecognised hint gets a question
("I've no s account, sir."), never the nearest identity — sending as the
wrong one of your three is as irreversible as sending to the wrong person.

### When it refuses

* the file is a folder, is unreadable, or has gone;
* it is over **18 MB** — Gmail will not carry more once the attachment is
  encoded, so a bigger limit would only turn a spoken refusal into a bounce
  after Jarvis had already said it went;
* the name resolves **outside** `~/Desktop`, `~/Downloads` and
  `~/Documents` — a symlink on the desktop pointing somewhere else is
  refused, not followed;
* an explicit path you give outright (`~/projects/thesis.pdf`) **is**
  allowed outside those folders, but never into a dot-folder (`~/.ssh`,
  `~/.gnupg`, `~/.config`) or a system tree (`/etc`, `/usr`, ...).

### Configuration

```json
"send_file": {
  "roots": ["~/Desktop", "~/Downloads", "~/Documents"],
  "max_mb": 18,
  "from": "",
  "contacts": {"heather": "heather@example.com", "brother": "sam@example.com"},
  "body": "Sent from Jarvis."
}
```

* `roots` — the only folders a spoken file *name* may resolve inside.
* `max_mb` — may only be lowered; 18 is the Gmail ceiling.
* `from` — a label from `gmail.accounts`. Leave it blank with more than one
  account and Jarvis asks which identity to send as, which is usually what
  you want: personal, work and school are three different people to whoever
  receives the mail.
* `contacts` — a plain name-to-address map, checked before the people book
  ("my advisor is Dr Peyrovi"). Both are consulted; neither is guessed at.

### Things it deliberately will not do

* **Texts and calls.** Out of scope by decision, not by omission —
  "send Heather a text" is recognised and left alone.
* **Reply, forward, delete.** This makes a new message with an attachment
  and nothing else.
* **Anything the model decides.** There is no tool the local model can call
  to send mail; the only path is a sentence you said and a yes you gave.

Restart Jarvis after editing `assistant.json`, as with every other setting.


## 82. HPCOMPUTER: files both ways, and no shell

The other machine. Five things can be said to it and no more:

```
"put the lab report on HPCOMPUTER"          a file goes over
"get the report from HPCOMPUTER"            a file comes back
"is HPCOMPUTER up"                          reachability
"what's the disk on HPCOMPUTER"             one row of a fixed question list
"run the build on HPCOMPUTER"               refused, out loud
```

Both transfers are **read back and confirmed**, with the same strict answer
grammar the email lane uses — a passing "yeah" in a longer sentence does
not count, "sure" is asked again rather than obeyed, and the yes has to
come from the channel the question was asked on. A file on another machine
cannot be taken back any more than an email can.

The last line is the point of the lane: there is no path from speech to a
shell on that box. "Delete the logs on the HP" and "delete the block on the
HP" differ by one phoneme and only one of them is recoverable, so neither
runs. A sentence that merely mentions the machine ("the build failed on the
HP", "did you install anything on the HP") is left alone and goes to the
model — the refusal only fires on an actual imperative.

### Configuration

```json
"remote": {
  "enabled": false,
  "host": "",
  "user": "",
  "key_path": "~/.ssh/hpcomputer",
  "name": "HPCOMPUTER",
  "os": "windows",
  "socks_proxy": "127.0.0.1:1055",
  "inbox": "~/jarvis-inbox",
  "pull_dirs": {"outbox": "~/jarvis-outbox",
                "desktop": "~/Desktop",
                "downloads": "~/Downloads"},
  "max_mb": 100
}
```

* `enabled` ships **false**, and every door says exactly what is missing
  rather than opening a socket to find out.
* `host` is where HPCOMPUTER answers ssh: on this LAN that is
  `192.168.50.114` (or `hpcomputer.local`), `user` the account there
  (`h2pey`), `key_path` a key **ssh already owns** (`~/.ssh/hpcomputer`) —
  Jarvis never reads it, only hands over its path, so no new secret enters
  `assistant.json`. A blank `key_path` is refused out loud, not tried.
* `socks_proxy` applies to a **tailnet** address only — a `*.ts.net` name,
  a `100.x` address, or a bare MagicDNS name. tailscaled on the Spark runs
  `--tun=userspace-networking`, so there is no route to the tailnet at all
  and tailnet traffic goes through the daemon's own SOCKS5 port. For a LAN
  address it is ignored, and so is the tailnet's view of the host: "is
  HPCOMPUTER up" tries ssh and reports what ssh says. Leave it at the
  default.
* `os` is `windows` (shipped — HPCOMPUTER runs the built-in OpenSSH
  Server, whose login shell is cmd.exe or PowerShell, and there is no
  `ls`, `df` or `uptime` there) or `posix`. It picks the question list —
  each row has a command per OS, the Windows ones a single
  `powershell -Command "..."` — and how a folder is listed: over SFTP on
  Windows, the same channel scp uses, so no shell is involved; a plain
  `ls` on POSIX. If the far side answers "is not recognized as an internal
  or external command" (or "command not found" the other way round),
  Jarvis says which command it did not know and names this setting,
  rather than the generic "wouldn't answer that". The Windows commands
  have **not** been run against HPCOMPUTER yet — the first live one is
  yours.
* `inbox` is the **only** folder a push can land in, and `pull_dirs` the
  only ones a pull may read. Speech never names a remote path: the spoken
  words pick a *key* ("desktop"), never a directory.
* The paths may start with `~`, which scp and SFTP read against the login
  home on both OSes (`C:\Users\h2pey` on HPCOMPUTER); on Windows a
  backslash is written as a slash for you. Jarvis writes them the
  different ways the far side needs (`$HOME` for a POSIX shell,
  home-relative for scp/SFTP). Do not quote them yourself.
* A local file whose *name* contains a shell character (a backtick, a
  semicolon, a newline) is refused rather than escaped, in both directions.

Restart Jarvis after editing `assistant.json`.

## 83. Grab and throw: reach at the lens, close your hand, fling it

His words, 2026-09-03: *"lets have a gesture added where i basically reach
out and grab at the screen (in the air) where the camera is and then gesture
towards almost throwing the cast onto the HPCOMPUTER."*

It ships **off**, and it rides the camera preview (§ the Privacy section of
the settings drawer): the hand stage runs *inside* the preview's own capture,
on the frame the pane already pulled, so there is no second camera handle, no
second thread, and every way the preview shuts — the curfew, offline mode,
standby, the toggle, quit — shuts this too. The two hand models are the
opencv_zoo MediaPipe palm detector and hand-landmark graphs (Apache-2.0),
sha-verified under `~/.aiws_trainer/models/hand`; nothing here leaves the
box and no frame is ever written or logged.

### Turning it on

Settings → Privacy: switch **Camera preview** on, then **Grab and throw**.
Or in `~/.config/jarvis/assistant.json`:

```json
"camera":  {"preview": true},
"gesture": {"enabled": true}
```

Restart after a file edit, as always. Then say nothing — do it:

1. **Reach** at the lens with an open hand. He resolves what you are about
   to pick up while your hand is still on its way (a document you just had
   explained, the track that is playing, else the window in front of you),
   so the grab is instant.
2. **Close your hand** and hold it still for a third of a second. You hear
   the *heard-you* tone first, a chip appears in the console header with
   the name, and he says: **"Holding the thesis draft, sir."** (set
   `gesture.speak_grab` false for tone-and-chip only). With nothing in front
   of him there is no carry: one *held-back* tone, and only a second empty
   grab inside ten seconds earns **"I've nothing in hand, sir."**
3. **Fling it left or right** and open your hand. The chip slides that way
   and the throw lands:
   * on **the board** (the Spark's own docked panel — a CAST slab appears
     on it, the *done* tone plays, and if the console is not on top he says
     **"On the board, sir."**). Until you have taught him a side, *every*
     throw goes here and he tells you once: **"That went to the board, sir.
     Tell me which side HPCOMPUTER is on and I'll send it there."**
   * on **HPCOMPUTER**, once taught — and today that is **held**, out loud,
     with the live reason: **"HPCOMPUTER isn't answering, sir — no port
     answered. I've kept it here."** (a repeat inside a minute: **"Still
     nothing listening, sir."**). The *warning* tone plays and the payload
     falls back to the board so you are not left holding it. The one thing
     that *does* reach it is a playing Spotify track: **"Blue in Green, on
     HPCOMPUTER, sir."**
4. **Put it down** any of four ways: open your hand where it is, pull it
   back still closed, say **"drop it"** / **"put it down"**, or wait eight
   seconds. Each is the *held-back* tone and the chip reads *dropped*. A
   fling at the desk is a cancel too. Any other sentence you say while
   carrying puts it down quietly — a sentence outranks a gesture.

Nothing irreversible happens on a wave. A sink that needs a read-back (the
handoff page, and the SSH push once it exists) is only *proposed* — **"The
thesis draft to the page, sir. Shall I send it?"** — and runs on your spoken
yes inside a minute, the same machinery as a bulk cancel.

### The same verbs by voice, camera off

| say | he |
| --- | --- |
| "throw this on HPCOMPUTER" / "put it on the board" / "cast this to the pc" | resolves the subject now (or takes the one you are carrying) and casts it |
| "drop it" / "put that down" / "let go" | **"Put down, sir."** — or **"I've nothing in hand, sir."** |
| "what am I holding" / "what's in your hand" | **"Holding the thesis draft, sir."** |
| "HPCOMPUTER is on my right" / "the left is the board" | **"Right is HPCOMPUTER from now on, sir."** — written to `gesture.sinks` |
| "which side is HPCOMPUTER on" | **"HPCOMPUTER is on your right, sir."** or how to teach it |

Targets he knows: `hpcomputer` (also "the pc", "the desktop", "the Windows
machine"), `board` ("the board", "the spark", "my screen") and `handoff`
("the page"). Anything else: **"I don't know a target called the fridge,
sir."**

### Which side is HPCOMPUTER?

Nobody but you can see the room, so the direction map **ships empty**. One
sentence fixes it once: *"HPCOMPUTER is on my right."* Teaching a side
*moves* a machine, never doubles it.

### What HPCOMPUTER can actually catch today (measured 2026-09-03)

`192.168.50.114` answers ARP (REACHABLE: powered on, on the LAN), ping is
100% loss, and none of 22/445/3389/5900/8008/2343 answers (00:45); 22/445/
3389 again at 02:40 and 07:21 — a Windows firewall dropping every inbound
packet. It is not on the tailnet. So a thrown file or screen is **held** and
said so; a track lands by Spotify's own outbound connection. The unblocks,
in order, are in `scratchpad/ideas/cast-target.md`: OpenSSH Server on
HPCOMPUTER (user `h2pey`, key `~/.ssh/hpcomputer`, then `ssh hpcomputer
whoami`), or `phone.enabled` on so the Spark serves a handoff page it can
fetch. The SSH transport is a seam (`HpcomputerSink(transport=...)`) with
nothing behind it until it can be tested against the real host.

### Every number is a starting point

The thresholds in `gesture` (fist/open bars, the reach ratio, the dwell in
frames, the throw distances in hand-units, the 8 s carry cap) were measured
on a synthetic hand with the LifeCam's own lens constants — never on his
hand. The self-check prints what his hand actually measures, as numbers
only, and shows or saves nothing:

```bash
~/vss_env/bin/python scripts/gesture_selfcheck.py --seconds 30
```

Two guesses are named in the config comments: the hand-to-face anthropometry
behind the reach ratio (±15% on him) and the 3 s attention latch. The frame
counters follow the rate the camera *delivers* (~7.5 fps), not the
`preview_fps` you asked for.

### If it does not fire

* Settings → Privacy: both switches on, and the sensing badge reads SENSING
  (not CAMERA OFF / OFFLINE). The console must be active — the preview stops
  in ambient and standby.
* `grep "gesture" /tmp/vss_voice/jarvis.log` — "no hand tracker" names the
  missing model file; "hand tracker ready" means it loaded.
* Look at the lens for a moment first: attention is latched for 3 s and the
  face baseline needs three detections in the last five seconds.
* One hand. Two hands out at the lens is not this gesture, on purpose.
* A question on the floor (a read-back waiting on your yes) blocks a grab.

## 84. The brain: how much room he gets to remember in

Everything the local model is *given* now lives in one place you can edit,
under `brain` in `~/.config/jarvis/assistant.json`. It used to be four
numbers buried in the code. A value that makes no sense is logged and
replaced with the default rather than obeyed.

**How these six keys got into your live file, and why that cannot happen
again.** They are in your `assistant.json` because on 2026-09-04 at 15:08
an *agent's* `import jarvis.brain` rewrote it — not Jarvis. Two things
allowed that: importing the brain read the config, and reading the config
saved it back whenever the defaults had gained a key. Both are closed.
Loading the config **never writes** now — a missing file, a corrupt file,
a loose mode or new default keys are only noted and logged. The one write
is `ensure_defaults()`, which only the running app calls, once, at
startup (it creates the file, moves a corrupt one to `.bad`, tightens the
mode, fills in new keys — exactly what loading used to do, in the one
process that owns the file). And importing the brain reads nothing: the
app hands its already-loaded config over, and anything else gets a
read-only load the first time it actually needs a setting. A test imports
every one of the 152 jarvis modules against a stale, loose config and
checks the bytes, the mtime and the mode did not move (measured on the
old code: the same walk grew a 38-byte file to 12,632 bytes in 2 s).

```json
"brain": {
  "num_ctx": 16384,
  "num_predict": 160,
  "temperature": 0.7,
  "think": false,
  "answer_reserve_tokens": 128,
  "protect_question": true
}
```

**First, what the window was and was not.** The 09-04 change was sold as
"room to think". It is not: thinking is the `think` switch below, and it
is off. What the window holds is what he is *told* — and the truncation
it was meant to cure was rare: counted over a week of his log, **1 prompt
in 5,328 was truncated** (a count from the 09-04 review, not re-measured
since). Doubling the window buys room to *remember* — a long calendar and
a long inbox in the same turn, more of the conversation — and it was the
guard further down, not the size, that fixed the one turn that went wrong.

**`num_ctx` — how much he can hold in his head at once.**
Everything he sees on a turn shares this: the tool descriptions, his
persona, what he remembers about you, the last few exchanges, your
question, and whatever the tools came back with. 16384 is double what he
had. Costs **0.19 GB of memory** and about **11 milliseconds a turn** —
both measured, not guessed. It does *not* make his answers longer or
cleverer. Raising it further is untested: 16384 is the biggest window
anyone actually watched load on this box, so above that he logs a warning
and you should watch memory. Below 2048 or above 262144 he ignores you and
uses 16384.

**`num_predict` — how much he may *generate* in one round.**
160 tokens. This is **not** how long he speaks. What he says aloud is
clamped separately, after the model has answered, to four sentences and
about 450 characters (`MAX_SPOKEN_SENTENCES` / `MAX_SPOKEN_CHARS` in
`jarvis/brain.py`), and that clamp cuts at a sentence end. This budget
covers everything the model writes in a round — the reply text *and* the
JSON of any tool call it makes (and its reasoning, only if `think` is on)
— and when it binds, it cuts mid-word. His real replies come back at 8-28
tokens, so 160 has never once stopped him; a tool call with long
arguments is what would hit it first. Raising it makes long answers
*possible*, not *likely*.

**`temperature` — how much his wording varies.**
0.7. Lower is steadier and flatter; higher is livelier and less
predictable. Cheap to try, instantly reversible.

**`think` — whether he reasons to himself before answering. Leave it off.**
It was measured on 2026-09-04 and it does not work on this model yet: his
reasoning is charged to the same `num_predict` budget as his reply, so at
160 he spent the whole thing thinking and said **nothing at all, six times
out of six**. Given far more room, 6 of 10 were still empty and the ones
that finished took 11 to 33 seconds against his usual 1.3. If you turn it
on he warns you in the log and tells you what your `num_predict` is.

**`answer_reserve_tokens` — headroom kept clear for the reply.** 128.
You will not need to touch this: since round 3 it is not the only margin.
The guard works from an *estimate*, and the estimate has a measured error
— the static prefix was estimated at 3,556 tokens and Ollama counted 3,761,
5.8% low, which at the old 16,096 ceiling was ~930 tokens against a
288-token margin (160 + 128). So two more things now sit between the
estimate and the window: a **calibration factor** the estimate is
multiplied by, starting at the measured **1.06** and learning from every
round (below), and a fixed **4% of the window** (655 tokens) taken off the
ceiling. The arithmetic at the shipped settings: 16384 − 160 − 128 − 655 =
**15441** is the ceiling; a calibrated estimate passes at a raw estimate of
at most 14,566; even if the real cost ran 10% above that (16,022 — worse
than anything measured) the 160-token reply still fits under 16,384.

**`protect_question` — the safety catch. Leave it on.**
When a turn gets big — a long calendar plus a long email — something has
to give. Ollama's own way of giving is to delete the *oldest* messages,
and the oldest message is **your question**. Measured: a 9000-character
calendar result took his prompt from 8253 tokens down to 7754, and the
499 tokens that vanished were your question, the background and his
memory. He then answers something confident and unrelated, and nothing
anywhere says why. With this on he cuts the *tail of the largest tool
result* instead — a long calendar loses its evening, not its morning —
marks the cut in the result so the model knows it is reading part of it
(the "you may take up to N sentences" note a list-shaped result carries at
its end is lifted off and put back, so a cut calendar is still read out in
full sentences), keeps your question, and writes a line in the log saying
he did it. If the next round overflows again — the mail arrives after the
calendar — the same result is **cut again**, its marker lifted and put
back once, rather than the newest result being thrown away (round 3 as
first shipped had a once-only rule, and on the round after a cut it
dropped the newest result whole while hundreds of trimmable tokens sat
in the cut one; the review measured it). A result is never cut below
**400 characters**: when the largest cannot absorb the whole overflow
above that floor, every result goes down toward its floor in turn so each
tool keeps its head — unless even the floors would not fit, in which case
a drop is unavoidable and he takes it *first*, oldest result first, so the
newest (the one the model just asked for) is sent whole rather than cut
to its floor and then thrown away anyway. Round 2 dropped whole every
time, which for the one result a turn hinged on meant he answered "an
earlier result was dropped, sir" instead of reading you the morning. What
the model then sees of a cut result is also what his own checks judge the
reply against, not the full text it never had.

It guards **every** request he makes to the model, not only the tool
loop: the spoken summaries, the router's tie-breaker, "explain this
document", the quizzes, the syllabus reader, the Sunday memory garden,
both warm-ups — and the screen tool's vision question (`[screen]` in the
log), which used to post on its own. Those have no tool result to cut, so
there he cuts the *tail of the material* — the end of the document or
digest, or the question — and marks the cut, because the instruction in
front of it is what Ollama would have eaten first. A screenshot is costed
as a fixed allowance of 1,024 tokens, never as its base64, so an image
cannot trim the question — and that allowance is charged on the path the
screen tool actually takes (round 3 as first shipped charged it there at
zero; measured: the walk put a screen question at 1,122 tokens, the guard
at 116). A round that carries an image **never feeds the calibration**
below: whether Ollama's count includes the image, and at what price, is
not measured here, so its ratio says nothing about the text rate the
factor tracks. Before that rule, one screenshot question pinned the factor
at its 2.0 cap and halved the tool loop's trim threshold for the next
seven rounds.

### How he knows a prompt is too big

He estimates before he sends, with **two rates**: 4.1 characters a token
for prose and tool descriptions, and **2.25** for tool results — measured
on the 09-04 calendar turn, where 9,000 characters of calendar cost 3,993
tokens. Costed at 4.1 alone the estimate ran low by up to 1.8x, and on
that exact turn the guard would have slept. The estimate costs 0.055 ms a
round (measured), i.e. nothing.

Then he logs what it *really* cost, from Ollama's own reply, on every
path:

```
ctx: prompt 4265/16384 tokens (26%), answer 22/160 (estimated 4310, raw 4066 x1.060) [chat]
ctx-calibration: 1.060 -> 1.079 (Ollama counted 4265 against an estimate of 4066 [chat])
ctx: prompt 1900/16384 tokens (12%), answer 30/160 (estimated 1202, raw 1134 x1.060) (1 image at 1024) [screen]
ctx-calibration: unchanged at 1.060 -- 1 image in the prompt, costed at the 1024-token allowance (...)
```

The tag at the end names the path — `chat` (the tool loop), `persona`
(summaries), `route` (the tie-breaker), `json` (documents, quizzes, the
garden), `warm` and `rewarm` (the warm-ups), `screen` (the vision
question). Three numbers sit side by side: what Ollama counted, the
calibrated estimate the guard compared, and the raw estimate with the
factor it was multiplied by. The second line is the **calibration**: the
measured-over-raw ratio of that round moves the factor half-way toward
itself, and the *next* round's guard uses the new factor. It can only make
him more careful than the measured 1.06 baseline, never less (a count
below half the estimate is ignored as not a whole-prompt count, and so is
any round that carried an image), and it is capped at 2.0. Above 90% of
the window the `ctx:` line becomes a warning. The `warm` line is special: the warm-up sends the static prefix
and nothing else, so its prompt number **is** the true cost of the
persona plus the tool schemas.

### Two things to know

**A change here does nothing until Jarvis restarts.** That is deliberate,
not a missing feature. Ollama keys the loaded model on `num_ctx`, so
asking for a different one mid-run makes the 25-billion-parameter model
reload — nearly nine seconds — and on the live server three of four
attempts to do that hung outright. So the settings are read once, when he
starts, and are identical on every request until he restarts.

**He tells you what he is running on.** In `/tmp/vss_voice/jarvis.log`,
one line at startup:

```
brain: window 16384 tokens, generation per round capped at 160 (reply text
plus tool calls; speech is clamped separately), temperature 0.7, thinking
off (assistant.json brain.*; a change needs a restart). Static prefix
~3556 tokens (persona ~1462 + 28 tool schemas ~2094), 128 reserved ->
~12540 tokens left for his question, memory, history and tool results.
Question guard on (trims a round above ~15441 calibrated tokens: 4% of
the window kept as estimate margin, estimate x1.060 from the last
measured round).
```

None of this was visible before.

### Verifying after a restart

```bash
ollama ps                                          # CONTEXT should read 16384
grep "brain: window" /tmp/vss_voice/jarvis.log | tail -1
grep "ctx: prompt" /tmp/vss_voice/jarvis.log | tail -5   # estimate vs real
```
If the model will not load at all, put `num_ctx` back to 8192 and restart.
