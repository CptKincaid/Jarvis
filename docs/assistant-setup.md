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
               "sports_feeds": [], "stock_symbols": []},
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

Jarvis plays to your other Spotify Connect devices (HPCOMPUTER, your phone, a speaker), never
to the Spark itself. Needs Spotify Premium for anything that controls playback (Free accounts
get "I'm afraid that needs Spotify Premium, sir.").

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
