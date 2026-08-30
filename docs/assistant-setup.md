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

## 17. How he behaves now (the 2026-08-30 set)

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
- **Meeting heads-up** — "BIOSENSORS in ten minutes, sir" (`calendar.heads_up_min`).
- **Guests** — a clear wake word in another voice gets "I only answer to Hunter, sir."
- **He learns your voice** — a confident match joins the voiceprint (at most every 10 min).
- **"Run diagnostics"** — uptime, models, today's turns and median wait, memory, GPU.
- **Streamed replies** — the first sentence speaks while the rest generates (`stream_replies`).
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
