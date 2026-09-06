"""Assistant config: ``~/.config/jarvis/assistant.json`` (spec section 10).

One file holds every personal-assistant setting and secret: home location,
Google secret-iCal URLs, the iCloud / Gmail app passwords, the Discord bot
token, the Claude session rules (allowed dirs, models, skill phrases) and
the briefing / alarm / autostart options. ``load()`` is a READ: it never
touches the file, whatever it finds (missing, corrupt, loose mode, keys
added since it was written) -- it notes the finding and merges DEFAULTS in
memory. Writing the file back is ``ensure_defaults()``, which creates it
(placeholders, mode 0600) when missing, moves a corrupt one aside as
``assistant.json.bad`` and recreates it, tightens a loose mode and fills in
new keys; ONLY jarvis.app calls it, once, at startup. Every other reader --
a script, jarvis-breeze, a test, an agent's ``import jarvis.brain`` -- gets
a config it can read and cannot have rewritten. (Until 2026-09-04 load()
did the write itself, and an agent's import of jarvis.brain rewrote the
live, secret-bearing file at 15:08 that day; tests/test_config_readonly.py
imports every jarvis module and checks the bytes and mtime did not move.)

Loading never raises: with an unwritable directory the config lives in
memory and every ``save()`` logs the failure.

Two ways to read a value::

    cfg.get("alarms.snooze_min", 10)     # dotted, default when missing
    cfg.alarms.snooze_min                # live attribute view (spec 4-9 use this)

Every consumer of a section with placeholders missing asks
``cfg.is_configured("gmail")`` and speaks ``cfg.setup_line("gmail")``.
Secrets never reach a log: ``redacted()`` / ``repr(cfg)`` mask them and
``scrub(text)`` strips their values out of arbitrary text.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from jarvis.logs import get_logger

log = get_logger("assistant_config")

ENV_VAR = "JARVIS_ASSISTANT_CONFIG"
DEFAULT_PATH = Path.home() / ".config" / "jarvis" / "assistant.json"
DOCS_HINT = "docs/assistant-setup.md"
MASK = "•••"          # "•••"

# Spec 10.1 — kept literally; the file is created from this.
DEFAULTS: dict = {
    "version": 1,
    "user": {"name": "Hunter"},
    "units": "us",
    "local_model": "gemma4:26b",
    # HOW MUCH ROOM THE LOCAL MODEL GETS TO REMEMBER IN (jarvis/brain.py).
    # The window holds what he is TOLD -- persona, tools, memory, history,
    # the question, the tool results -- not how hard he thinks; think is
    # the separate switch below, and it is off.
    # Every value here is read ONCE, when Jarvis starts, and is then
    # identical on every single request the brain makes -- Ollama keys its
    # loaded runner on num_ctx, so asking for a different one mid-run makes
    # the 25 B model RELOAD (measured 8.8 s) and throws away the prompt
    # cache with it. A change here therefore needs a Jarvis RESTART; it can
    # never take effect per turn. See docs/assistant-setup.md, "The brain".
    "brain": {
        # The whole window, in tokens: everything the model can see at once
        # (tool descriptions + persona + memory + history + his question +
        # the tool results). 16384 is the largest size measured to load
        # safely on this box: +0.19 GB of KV cache, +0.011 s per turn.
        # Above it nothing has been watched loading -- raise with care and
        # watch MemAvailable.
        "num_ctx": 16384,
        # The cap on what the model may GENERATE in one round, in tokens:
        # its reply text, any tool-call JSON, and (only with think on) its
        # reasoning. It is NOT the cap on what is spoken -- speech is
        # clamped afterwards by MAX_SPOKEN_SENTENCES / MAX_SPOKEN_CHARS in
        # jarvis/brain.py, and a reply this budget cuts is cut mid-word.
        # Real replies come back at 8-28 tokens, so 160 has never yet
        # bound; a tool call with long arguments is what would hit it.
        "num_predict": 160,
        # How much the wording is allowed to vary. Lower is steadier and
        # flatter; higher is livelier and less predictable.
        "temperature": 0.7,
        # Let the model reason to itself before answering. MEASURED OFF for
        # a reason: at num_predict 160 the reasoning ate the whole budget
        # and the reply came back EMPTY 6 times out of 6, and the turns
        # that did finish took 10.9-33.0 s against a 1.3 s baseline. Do not
        # turn this on without also raising num_predict a long way.
        "think": False,
        # Tokens held back for the answer, on top of num_predict, when the
        # guard below decides whether a round still fits.
        "answer_reserve_tokens": 128,
        # THE GUARD. When a round would overflow the window, drop the
        # OLDEST TOOL RESULT. With this off, Ollama makes room its own way
        # -- by deleting the oldest messages, which is HIS QUESTION -- and
        # says nothing about it in any log.
        "protect_question": True,
    },
    "home_location": {"city": "", "region": "", "lat": None, "lon": None},
    "location_lookup": True,
    "google_ical_urls": [],
    "icloud": {"apple_id": "", "app_password": "",
               "url": "https://caldav.icloud.com"},
    # Reading uses imap_host; SENDING uses smtp_host (jarvis/outbox.py). A
    # per-account entry in `accounts` may carry its own smtp_host; without
    # one, mail.smtp_host() rewrites imap.x -> smtp.x, which is right for
    # Gmail and for everything else that names its servers that way.
    # smtp_host is deliberately NOT a default: load() writes every default
    # key into the file, and a written "smtp.gmail.com" then reached every
    # account that had none of its own, so the rewrite never ran (F22).
    "gmail": {"address": "", "app_password": "", "imap_host": "imap.gmail.com",
              "accounts": []},
    "claude": {
        "allowed_dirs": ["/home/hunterp/Jarvis", "/home/hunterp/haymaker-digest"],
        "projects_root": "/home/hunterp/projects",
        "permission_mode": "acceptEdits",
        "web_model": "haiku",         # one-shot web lookups; fast beats big here
        "dangerously_skip_permissions": False,
        # User decision 2026-08-26: work is auto-approved ANYWHERE, not only
        # under allowed_dirs ("even if its not in the base project directory
        # it can be auto approved").  Set false to restore the old behaviour
        # of refusing any project outside claude.allowed_dirs.
        "auto_approve_anywhere": True,
        # Verified live against Claude CLI 2.1.247 on 2026-08-27 (findings are
        # recorded at the top of jarvis/claude_session.py).  With this on, an
        # action outside the project asks Hunter aloud through the broker; set
        # it false to drop --mcp-config / --permission-prompt-tool, which also
        # makes work outside allowed_dirs refuse outright again.
        "permission_prompt_tool": True,
        "model": "opus", "big_model": "fable", "fast_mode": False, "effort": "",
        "skill_phrases": {
            "^review (this|my|the) code$": "/code-review",
            "^commit (this|it|that)$": "/commit",
            "^simplify (this|it|that)$": "/simplify",
            "^security review$": "/security-review",
            "^run a ralph loop on (.+)$": "/ralph-loop $1",
            "^plan a feature (.+)$": "/feature-dev $1",
        },
    },
    # Every desktop banner (the hub AND the direct notify-send sites) obeys
    # alerts.desktop; alerts.discord gates the Discord fan-out;
    # alerts.claude_hooks gates the spoken lines from the user's OWN Claude
    # Code sessions (scripts/claude_hooks/narrate.py reads this file with
    # plain json, so the key lives here rather than in a jarvis module).
    "alerts": {"desktop": True, "discord": True, "claude_hooks": True},
    # User-defined spoken shortcuts -> a tool call, matched ahead of the
    # classifier and the model. See docs/assistant-setup.md "Custom phrases".
    "phrases": [],
    # on_first_wake: deliver the briefing after the first thing you say to
    # Jarvis each day, once it is past `after` (24 h clock, local time).
    # heads_up_min: the meeting heads-up lead (jarvis/headsup.py).
    # anomaly_watch: diff each successful calendar refresh against the last
    # one and speak what changed today/tomorrow -- "your 9:10 has just been
    # cancelled, sir" (jarvis/calwatch.py).
    # leave_times: the per-building walk heads-up (jarvis/leavetime.py). The
    # walk itself is LEARNED by asking once and lives in long-term memory,
    # never in this file; leave_notice_min is how long before the walk
    # starts he is told ("you want to be walking in 5 minutes, sir").
    "calendar": {"heads_up_min": 10, "anomaly_watch": True,
                 "leave_times": True, "leave_notice_min": 5},
    # Canvas LMS: Account > Settings > New Access Token (read-only use).
    # heads_up_hours: the deadline heads-up (jarvis/deadlines.py) speaks
    # this long before each due time; the meeting heads-up's ten minutes
    # is no use for an 11:59 pm deadline.
    "canvas": {"base_url": "https://canvas.tamu.edu", "token": "", "heads_up_hours": 3},
    # Local document Q&A: drop PDFs / notes here; indexed with nomic-embed-text
    "docs": {"paths": ["~/Documents/Jarvis Docs"], "index_dir": "~/.aiws_trainer/docs_index",
             "max_files": 500, "embed_model": "nomic-embed-text",
             "ollama_url": "http://localhost:11434"},
    # Local code Q&A over his OWN repos (jarvis/tools/docs.py CodeIndex).
    # paths empty means "the folders in claude.allowed_dirs" -- those already
    # ARE his repos, and a second list would drift out of sync. index_dir
    # empty means PATHS.MEMORY_DIR/code_index, a chroma collection separate
    # from the documents one so quiz mode never draws a flashcard from app.py.
    "code": {"paths": [], "index_dir": "", "max_files": 3000},
    # Screen Q&A ("what's on my screen?"). model "" = the local_model above,
    # which is the ONLY free choice: OLLAMA_MAX_LOADED_MODELS=1, so naming a
    # second vision model here evicts the chat model and costs ~7 s on the
    # next spoken turn. It was llama3.2-vision:latest until 2026-09-01, and
    # that is pulled but unloadable -- ollama 0.33.1 answers 500 "unknown
    # model architecture: 'mllama'" -- so screen Q&A only ever apologised.
    "screen": {"model": "", "max_width": 1280},
    # Study / focus sessions (jarvis/focus.py): block and break lengths in
    # minutes, "Halfway, sir" for blocks of 10+ min, the session ends itself
    # after max_blocks (0 = until "end the session"). music: "pause" pauses
    # Spotify for the block and resumes it for the break, "playlist" plays
    # `playlist` for the block and pauses it for the break, "off" leaves it.
    # dnd: while a block is running, proactive lines (heads-ups, watchdog
    # warnings, the hooks narrator) are held by jarvis/quiet.py and read
    # back as the catch-up digest at the break -- not mid-pomodoro.
    "focus": {"block_min": 25, "break_min": 5, "halfway": True, "max_blocks": 4,
              "music": "pause", "playlist": "", "dnd": True},
    # Lecture notes (jarvis/lecture.py): after each noted line the mic
    # re-opens without a wake word for window_s seconds (capped at 30 by
    # the recorder; the 60 s hard cap on a capture must leave room to talk).
    "lecture": {"window_s": 20},
    # Quiz mode over the documents index: questions per round, chunks of
    # study text handed to the model per round (jarvis/tools/quiz.py).
    # window_s: the mic stays open this long for the ANSWER (and for a
    # yes/no read-back) instead of CONFIG.followup_window's 4 s, which is
    # sized for "...and Tuesday?" and not for thinking about a flashcard.
    # Capped at 30 by the recorder, like lecture.window_s.
    "quiz": {"questions": 5, "chunks": 6, "window_s": 15},
    # yield_to_trainer: when a GPU claimant appears, unload the local model
    # (brain.release) and speak the lent line; reload once it is gone for two
    # ticks. ON since 2026-08-30: it was off because a multi-hour run leaves
    # Jarvis with Tier 1 only, but the cost of NOT lending turned out to be
    # worse -- the haymaker digest timer (04:09, nightly) sat starved for
    # ~50 minutes waiting for qwen2.5:32b behind Jarvis's pinned gemma4:26b,
    # which is the same unified-memory contention that hard-power-off
    # wedged this box on 2026-08-28. Degraded answers are one spoken
    # sentence away from repair ("take the GPU back"); a starved nightly job
    # and a wedged box are not.
    # yield_to: GPU claimants that are NOT training runs, named because
    # nothing in their command line says "train" (jarvis/tools/health.py).
    "health": {"warn_gb": 16, "critical_gb": 8, "hog_gb": 20, "interval_s": 30,
               "yield_to_trainer": True, "yield_to": ["digest_llm"]},
    # The run ledger (jarvis/runwatch.py), driven off the health tick: two
    # spoken beats per training run (started / finished, with an honest
    # duration read from /proc). narrate=false keeps the board lane and the
    # log line but says nothing. progress needs scripts/runlog.sh, which
    # tees the trainer's stdout to log_dir/<pid>.log -- without the wrapper
    # there is nothing to read, so it is opt-in; epochs are narrated at
    # most once per progress_gap_s and only when the number CHANGES.
    "runwatch": {"narrate": True, "progress": False,
                 "log_dir": "~/.cache/jarvis/runs",
                 "min_run_s": 60, "progress_gap_s": 300},
    # Long-term memory (jarvis/memory.py): facts are also indexed with
    # nomic-embed-text so "who's my dentist" finds "my dentist is Dr Patel";
    # semantic=false keeps the substring store only.
    "memory": {"semantic": True},
    # Weekly memory garden (jarvis/garden.py): once the ISO week closes,
    # the week's activity journal is read by the local model and up to
    # max_facts durable facts are filed as long-term memory, tagged
    # source="garden" so "forget the last garden pass" can take them back.
    # run_before_hour keeps the 26B extraction in the small hours; a week
    # still ungardened by Wednesday runs at any hour instead.
    "garden": {"enabled": True, "max_facts": 4, "run_before_hour": 6},
    # Activity journal (jarvis/context.py + tools/journal.py): the focused
    # window is sampled every window_interval_s; day files older than
    # keep_days are pruned. enabled=false stops the sampler only -- exchanges
    # and tool calls are always journaled.
    "journal": {"enabled": True, "window_interval_s": 60, "keep_days": 90},
    # Spoken register (jarvis/brain.py REGISTERS), set by voice ("formal
    # mode", "banter up") and read back at app start. It is baked into the
    # STATIC Tier 2 prompt, so a change costs exactly one prefix reprocess
    # and every turn afterwards is cached again.
    # address_per_burst: how many times Jarvis may say "sir" in ONE spoken
    # burst, where a burst is several already-finished lines JOINED into one
    # utterance -- the catch-up digest, the arrival welcome plus catch-up,
    # the first-wake morning briefing, a compound turn's two answers, and
    # the speak-queue watcher. That join is the only place sirs were ever
    # measured to stack; a single line already carries exactly one, so this
    # knob changes nothing about ordinary replies (jarvis/address.py).
    # 1 is that measured rate; raise it to loosen the cut, or set
    # address_thinning false to turn the pass off. Only a sign-off is ever
    # dropped -- one that ends its fragment ("The build passed, sir.") or
    # ends a clause the same sentence runs on from ("Memory is tight, sir:
    # 3 gigabytes free."), and never one inside a quotation or with a new
    # SENTENCE behind it, which is the shape of an interpolated mail subject
    # or a quoted line. A "Sir, ..." summons goes only
    # when the SAME summons has already been spoken in that burst ("Sir,
    # this is your reminder" five times over is the chant; the first one
    # stays). A lone fragment is never rewritten, the first fragment of a
    # burst is never rewritten, and a burst that went in with an address
    # always comes out with one, so this cannot be edited into a Jarvis who
    # stops saying it. Read at start-up: a change needs a restart.
    "persona": {"register": "normal", "address_thinning": True,
                "address_per_burst": 1},
    "briefing": {"enabled": False, "on_first_wake": True, "after": "06:00", "hn_items": 3,
                 "news_feeds": ["https://www.theverge.com/rss/index.xml",
                                "https://feeds.arstechnica.com/arstechnica/index"],
                 "sports_feeds": [], "stock_symbols": [],
                 # Per-section switches, set by voice ("no news in the
                 # morning"); a missing name counts as on. verbosity "brief"
                 # halves every briefing view's sentence allowance.
                 "sections": {"weather": True, "calendar": True, "news": True,
                              "sports": True, "stocks": True, "canvas": True,
                              "todos": True, "alarms": True, "reminders": True,
                              # exam-week study: the flashcard deck for the
                              # course whose exam is inside study_days, and
                              # the offer to run some now
                              "study": True},
                 "study_days": 5, "study_offer": True, "study_offer_n": 10,
                 "verbosity": "normal",
                 # The good-night preview offers a wake-up alarm when
                 # tomorrow's first event starts by early_before and no alarm
                 # already covers it: wake_lead_min before the event.
                 "wake_offer": True, "early_before": "09:00", "wake_lead_min": 60},
    "alarms": {"sound": "", "volume": 0.8, "escalate": True,
               "max_ring_s": 300, "snooze_min": 10},
    # Destructive read-back (commander._try_destructive_confirm): "cancel
    # all alarms" / "clear my list" with more than one item is read back
    # and waits for a yes. shaky_logprob: a transcript whose Whisper
    # avg_logprob is below this also gets a read-back for a one-item
    # whole-list cancel. Calibrated 2026-08-30 from the live log (n=32
    # accepted turns: p10 -0.61, median -0.41; the garbled ones sat at
    # -0.94/-0.95), so -0.7 flags the doubtful tail without nagging.
    # dissent (jarvis/objections.py): before setting an alarm he checks four
    # cache-only rules -- a duplicate alarm, a small-hours alarm that leaves
    # under sleep_floor_h with something on the calendar later that day, a
    # quiet window it lands inside, a Canvas deadline it falls after -- and
    # names the row it objected from ("your BIOSENSORS lecture is at 9:10
    # am"). "Shall I set it anyway?" defaults to YES: an unclear reply, a
    # changed subject and silence all set the alarm, because the alarm was
    # asked for and only the opinion was volunteered. Never twice for the
    # same thing in a day, never for a timer.
    "confirm": {"read_back": True, "shaky_logprob": -0.7, "dissent": True,
                "sleep_floor_h": 5},
    # The Aside (jarvis/aside.py): one volunteered sentence after an answer,
    # hung off the datetime an alarm or reminder just resolved to ("Alarm for
    # 7:00 am, sir. Incidentally, Lab 3 report for BIOSENSORS is due at 11:59
    # pm that night"). Off for the first day on purpose -- the budget is the
    # product, and per_day/gap_min are here so tuning it never needs a code
    # edit. "No more asides" zeros the day's bucket. Snapshot data only: it
    # reads the deadline thread's last Canvas result and the calendar cache,
    # never a fetch, because it runs inside a spoken turn.
    "aside": {"enabled": False, "per_day": 2, "gap_min": 45,
              "horizon_hours": 18},
    # The debrief (jarvis/debrief.py): when a calendar event whose title
    # carries one of these words has ended between after_min and within_min
    # ago, and he is not out and not in a quiet window, Jarvis asks once --
    # "How did the midterm go, sir?" -- and FILES the answer to the journal
    # and to memory instead of routing it to the model as chat. A quiet
    # window postpones rather than cancels, up to hold_hours.
    "debrief": {"enabled": True, "after_min": 15, "within_min": 180,
                "hold_hours": 14,
                "keywords": ["exam", "midterm", "final", "finals", "interview",
                             "viva", "defense", "defence", "quiz", "test",
                             "presentation", "audition"]},
    # "No, I said X": re-dispatch X, drop the misheard exchange, log the
    # pair to corrections.json. learn_vocab additionally appends new
    # capitalised words from X to the Whisper vocabulary prompt -- off by
    # default because Whisper's casing on a misheard name is itself a guess.
    "corrections": {"learn_vocab": False},
    "discord": {"bot_token": "", "channel_id": "", "user_id": ""},
    # Spotify (jarvis/tools/spotify.py): the developer-app credentials and the
    # speaker Jarvis reaches for when nothing else is playing. The OAuth token
    # itself lives beside this file in spotify_token.json (0600), never here.
    "spotify": {"client_id": "", "client_secret": "",
                "default_device": "HPCOMPUTER", "liked_strategy": "uris",
                "market": "from_token"},
    "autostart": {"enabled": False},
    # Quiet hours / do-not-disturb (jarvis/quiet.py). hours: "HH:MM" 24 h,
    # overnight windows wrap ("23:00" -> "07:00"); dnd_until / free_until are
    # timestamps he sets himself ("do not disturb for an hour", "I am free");
    # calendar: a running timed event whose title contains one of the
    # keywords also holds proactive speech; hold_when_away needs presence.
    # calendar_courses: a recurring course also counts as a running class
    # even when its title matches no keyword (jarvis/courses.py) -- without
    # it the calendar leg never fired, because no course of his is CALLED
    # "class".
    "quiet": {"hours": {"start": "", "end": ""}, "dnd_until": 0, "free_until": 0,
              "calendar": True, "calendar_keywords": ["class", "exam", "meeting", "busy"],
              "calendar_courses": True, "hold_when_away": True},
    # Background watchers (jarvis/grades.py, mailwatch.py, keyword_watch.py):
    # unprompted lines, all held by the quiet policy. grades diffs the Canvas
    # course totals every 15 min; people_mail speaks unread mail from someone
    # in the people book every 10; keywords is a list of words or phrases to
    # watch for across unread mail and Canvas announcements -- empty means the
    # keyword watch never runs. All three are silent without their credentials.
    "watch": {"grades": True, "people_mail": True, "keywords": []},
    # Bedtime wind-down (jarvis/winddown.py): "good night" fades Spotify to
    # nothing over fade_s and pauses it, warms the screen (GNOME night
    # light) and dims it to `brightness`, and arms do-not-disturb until
    # quiet hours close (or `morning` when they are not configured).
    # "Good morning" -- or the next app start after the window -- puts it
    # all back. Off by default; every half has its own switch, and
    # brightness is floored well above black.
    "wind_down": {"enabled": False, "fade_s": 60, "brightness": 0.5,
                  "night_light": True, "music": True, "dnd": True,
                  "morning": "07:00"},
    # The Room Mixer (jarvis/mixer.py): while Jarvis speaks or listens,
    # every non-Jarvis stream slides to duck_level % over duck_ramp_ms and
    # slides back. Per STREAM (pactl set-sink-input-volume), never the sink
    # -- the default sink is the soundbar his own voice comes out of.
    # The Room Mixer's settings, and the sink sentinel's (jarvis/soundbar.py).
    # sink_watch: notice when the speaker his voice lands on changes, and say
    # so ONCE -- the soundbar's battery died on 2026-08-30 and Jarvis talked
    # into the HDMI monitor for hours. preferred_sink is a PipeWire sink name
    # or any fragment of one ("bluez", a MAC fragment); left empty the
    # sentinel learns the box's Bluetooth sink, since a monitor does not drop.
    # restore_sink is the ONE state change it may make -- pactl
    # set-default-sink back to the preferred sink -- and it is OFF because
    # that changes his desktop, not merely Jarvis's voice.
    "audio": {"duck": True, "duck_level": 30, "duck_ramp_ms": 200,
              "sink_watch": True, "preferred_sink": "", "restore_sink": False,
              "sink_poll_s": 30},
    # The room's light (jarvis/room.py) and scenes (jarvis/scenes.py).
    # There are no bulbs here: this is the 4K panel's brightness (an xrandr
    # gamma scale, floored at 0.55 -- Jarvis's own console lives on it) and
    # GNOME's night-light temperature. Every change is reversible and is
    # restored at "lights up", at boot and at quit.
    # wind_down_on_goodnight is OFF by default: "good night" already has a
    # handler (commander._goodnight_preview), and a scene is a change to
    # his desktop that he did not ask for by saying good night.
    # scenes: ordered lists of primitives -- brightness / temperature /
    # music / quiet_hours / say. Edit them here; no code change needed.
    "room": {
        "enabled": True,
        "wind_down_on_goodnight": False,
        "scenes": {
            "wind down": [
                {"do": "temperature", "kelvin": 2700},
                {"do": "brightness", "level": 0.6},
                {"do": "music", "action": "pause"},
                {"do": "quiet_hours", "start": "22:00", "end": "07:00"},
                {"do": "say", "line": "Powering down the workshop, sir. "
                                      "The screen's warm and the room is yours."},
            ],
        },
    },
    # The pre-class dossier (jarvis/dossier.py): at lead_min before a class,
    # ONE spoken line and one card with the room (or the join link), the
    # last lecture notes, that course's deadlines and the unread mail about
    # it. Every section is optional; each one is skipped when its source is
    # missing, so a box with no notes folder and no Canvas token still gets
    # "Your 9:10 is BIOSENSORS, Wisenbaker 049, sir." budget_s caps the
    # whole gather (IMAP + Canvas), which runs on a worker thread.
    "dossier": {"enabled": True, "lead_min": 10, "notes": True, "mail": True,
                "mail_hours": 72, "due_days": 7, "budget_s": 25},
    # Nightly flashcards (jarvis/studycards.py): in the small hours, each
    # course's newest lecture notes become Leitner cards, so the exam-week
    # briefing has a deck to count instead of "say quiz me and I'll build
    # one". Skipped whenever the GPU is lent or the model is busy. Silent:
    # the cards surface at "review my flashcards" and in the briefing.
    "study_cards": {"enabled": True, "per_course": 5, "max_courses": 3,
                    "run_before_hour": 5, "max_age_days": 7, "min_lines": 3},
    # Class-start staging (jarvis/classflow.py): at the start of a recurring
    # class, prime today's notes file, put the music down and show a card.
    # auto_notes arms voice capture for the hour and is OFF by default --
    # it records the room, so it is his to switch on; open_notes shells out
    # to xdg-open and is off because window churn froze the desktop once.
    # idle_min is how long he may have been away from the keyboard and
    # still count as at the desk (jarvis/desk.py).
    "class_flow": {"enabled": True, "auto_notes": False, "open_notes": False,
                   "duck_music": True, "idle_min": 15},
    # Presence (jarvis/presence.py): the phone's Wi-Fi address and/or MAC.
    # away_after_min is the grace before "out" -- iPhones nap off Wi-Fi for
    # minutes at a time, so anything under ~10 flaps.
    # desk* is the second, configuration-free probe (jarvis/deskpresence.py):
    # GNOME's Mutter idle monitor over the session bus. It only suppresses
    # (held lines, a dimmed board) and greets the return; it never says
    # "nobody's home" out loud. desk_away_after_min is generous because idle
    # time is keyboard/mouse only -- reading at the desk looks like an empty
    # chair. desk_standby dims the board while the chair is empty.
    "presence": {"enabled": True, "phone_ip": "", "phone_mac": "",
                 "away_after_min": 12, "poll_s": 60,
                 # Arrival/departure choreography (jarvis/arrival.py). The
                 # arrival cue is the ordered sequence panel -> earcon ->
                 # "Welcome back, sir" -> catch-up; departure is silent by
                 # design and additionally waits confirm_min past the away
                 # grace with no mic turn in mic_silence_min, because a
                 # sleeping phone radio fakes a departure and a wrong one
                 # would settle the room while he is sitting in it.
                 "poll_s_away": 10, "arrival_cue": True,
                 "departure_confirm_min": 5, "departure_mic_silence_min": 10,
                 "desk": True, "desk_away_after_min": 25, "desk_poll_s": 30,
                 "desk_standby": True, "desk_standby_alpha": 0.45,
                 # The room sensor (jarvis/roomsensor.py, docs/room-sensor.md):
                 # an ESP32 + LD2410 mmWave module read over plain HTTP from
                 # ESPHome's web_server. OFF until both keys are set, and it
                 # only ever ADDS presence: the room seeing someone makes him
                 # home immediately even with a sleeping phone, the room
                 # seeing nobody never makes him away while the phone
                 # answers. A restart is required after editing these --
                 # reload_if_changed has no callers.
                 # room_sensor_power_url is the OPTIONAL ESPHome switch
                 # that holds the LD2410's supply (the commented block in
                 # scripts/esphome/jarvis-room-sensor.yaml). With it set,
                 # offline mode CUTS the radar
                 # (POST <url>/turn_off) instead of merely not polling it.
                 "room_sensor_enabled": False, "room_sensor_url": "",
                 "room_sensor_power_url": "",
                 "room_sensor_timeout_s": 1.5,
                 # THE ZONE MODEL LIVES IN "zones" AT THE BOTTOM OF THIS
                 # FILE, and nowhere else. There used to be a second one
                 # here -- desk_band_m / room_band_m, a TWO-band ladder
                 # that assumed the desk was the NEARER band -- and it
                 # disagreed with the named N-band ladder under
                 # zones.rooms: the SENSORS page called 4 of 10 points "AT
                 # THE DESK" from the radar alone while the zone log had no
                 # desk band at all. They were different keys, so git saw
                 # no conflict, and in his actual office the two-band model
                 # was BACKWARDS -- the near space is empty and the desk is
                 # at 3.13 m (measured 2026-09-03).
                 #
                 # So the two keys are gone from here. jarvis/ui/
                 # sensors_page.py still READS them, once, as a migration
                 # source: a config that has them and no zones.rooms gets
                 # its bands carried across, the page says which key it
                 # used, and SAVE writes zones.rooms and clears them. They
                 # are deliberately absent from DEFAULTS so that a value
                 # here means "his file still holds it" and not "the
                 # shipped default is sitting here looking live".
                 #
                 # camera_overrules stays: it is the SENSORS page's own
                 # toggle for watching the fusion with the override off,
                 # and it is not a second copy of anything.
                 "camera_overrules": True,
                 # THREE ROOMS (jarvis/roomfabric.py). The plural
                 # of the four keys above, shaped exactly like
                 # gmail.accounts: a LIST of labelled entries, and while it
                 # is empty the singular keys above are used instead, so a
                 # config written before this existed keeps working with no
                 # edit. room_sensor_enabled stays the master switch over
                 # the whole fabric. Each entry is
                 #   {"name": "office",              # the identifier
                 #    "label": "the office",         # what gets spoken
                 #    "url": "http://192.168.50.60",
                 #    "power_url": "",               # optional; see above
                 #    "primary": true,               # where the Spark is
                 #    "enabled": true}
                 # An entry with no url or no name is skipped rather than
                 # fatal: one unfinished room must not take the others down.
                 #
                 # THE ONE LIST. jarvis/rooms.py (the satellite lease) and
                 # jarvis/roomaudio.py (which speaker a line comes out of)
                 # read THIS list too, and every lane spells the name with
                 # roomfabric.room_name, so "Kitchen", " kitchen " and
                 # "Kitchen!" are one room. Two lists was the shape that bit
                 # (F02, 2026-09-03): the other two lanes read a
                 # rooms.satellites key declared nowhere, so a config with
                 # this list got a fabric and no leases. The extra per-entry
                 # keys, and the lane that reads each:
                 #    "sensors": ["radar"],          # rooms: THE OPT-IN. Only
                 #                                   #   a box flashed with
                 #                                   #   jarvis-satellite.yaml
                 #                                   #   is leased; a plain
                 #                                   #   radar has no buttons
                 #                                   #   to press and is never
                 #                                   #   pressed
                 #    "username": "jarvis",          # rooms: web_server auth
                 #    "password": "",                # ditto -- masked; see
                 #                                   #   SECRET_LIST_FIELDS
                 #    "lease_ttl_s": 90.0,           # rooms: keep in step with
                 #                                   #   the YAML's lease_ttl
                 #    "say_url": "",                 # roomaudio: that room's
                 #                                   #   speaker; "" = none
                 #    "private": false               # roomaudio: never a
                 #                                   #   broadcast target
                 # The primary entry is the Spark's own room, and the one
                 # the voice falls through to.
                 "rooms": [],
                 # The fabric's four timers, argued in jarvis/roomfabric.py.
                 # enter: how long a new room must hold occupied before it
                 # takes over (a doorway pass-through is ~1 s in the beam).
                 # leave: how long the current room may read empty and still
                 # be believed -- the LD2410's own absence delay is already
                 # 5 s, so anything under that re-litigates the device.
                 # switch: the floor between room changes, the doorway
                 # anti-flap. stale: when the last known room stops being
                 # named at all. stuck: a room reading occupied this long
                 # without a break is a fan, not a man, and is dropped from
                 # the picture until it clears.
                 "rooms_poll_s": 2.0, "rooms_enter_hold_s": 2.0,
                 "rooms_leave_hold_s": 8.0, "rooms_switch_min_s": 6.0,
                 "rooms_stale_after_s": 90.0, "rooms_stuck_after_h": 12.0,
                 # HIS THREE-LEG VOTER (jarvis/presencevote.py): camera,
                 # then phone, then room sensor, in his order. OFF BY
                 # DEFAULT. Turning it on changes what "away" means: every
                 # poll asks all three legs, so a room reading occupied no
                 # longer stops the phone being asked and a latched radar
                 # cannot cost him the greeting (2026-09-05, 20:43). Off,
                 # the room-or-phone composition runs exactly as before.
                 "three_legs": False,
                 # With the voter on: how fresh an agreement (phone, camera
                 # or a spoken turn) must be, in minutes, for a room reading
                 # occupied to still count as him when his phone is silent
                 # and the camera cannot look. DERIVED from away_after_min,
                 # not measured (presencevote.RECENCY_S_PROVENANCE); the
                 # right value is how long his short trips are.
                 "corroboration_recency_min": 15,
                 # THE DOOR ROOM (jarvis/arrival.py, app._on_room_changed).
                 # His words: "kitchen to see if i enter my apartment since
                 # the kitchen and door are next to each other". That room
                 # going occupied after a WHOLE-HOME absence is the front
                 # door opening, and it is greeted straight away instead of
                 # waiting for his phone's radio to answer an ARP. It must
                 # match a `name` in `rooms` above; while no such room is
                 # configured nothing here fires. A kitchen trip while he
                 # is already home is not an arrival and never greets.
                 "door_room": "kitchen",
                 # "Welcome back from the dentist, sir" -- named only when
                 # a calendar event honestly covered the absence, and the
                 # plain "Welcome back, sir" otherwise. False keeps the
                 # plain line always.
                 "arrival_outing": True,
                 # The doorstep OFFER: unread count plus one clause on
                 # anything major, then a question. It never reads the mail
                 # -- that needs a yes (Commander._try_briefing_offer).
                 "arrival_offer": True},
    # The satellite LEASE lane's timers (jarvis/rooms.py). No room list
    # here, on purpose: the list is presence.rooms above, and an entry
    # there that lists "sensors" is what makes a room a leased satellite.
    # renew_s must stay in step with renew_every in scripts/esphome/
    # jarvis-satellite.yaml; timeout_s is jarvis/roomsensor.py's MEASURED
    # 3.0 s (max round trip seen 1186 ms), not the 1.5 s that used to be
    # the module's default.
    "rooms": {"renew_s": 25.0, "stale_after_s": 90.0, "timeout_s": 3.0},
    # Offline mode and the camera curfew (jarvis/sensing.py). ONE object
    # answers "may this sensor run", combining the manual switch (spoken:
    # "offline mode", "deactivate presence", "stop watching"), this daily
    # camera window, and the fail-safe. The switch itself is NOT here --
    # it lives in a state file under MEMORY_DIR, because a config this
    # file's own loader recreates from DEFAULTS on a corrupt read would
    # fail ONLINE, and the whole ruling is that it must fail OFFLINE.
    # The curfew closes the LENS only: the radar makes no image, so
    # switching it off at night would cost presence for no privacy.
    # start/end are "HH:MM" 24 h and wrap midnight, like quiet.hours.
    "sensing": {"curfew": {"enabled": True, "start": "21:00", "end": "07:00"}},
    # ZONES (jarvis/zones.py): WHERE in a room, and a written record of it
    # before anything is allowed to act on it -- his words, "just log it
    # first". Nothing reads this to decide what Jarvis says; the only
    # effect of turning it on is a JSONL file.
    #
    # A zone is a named DISTANCE BAND on one radar, because an LD2410C
    # reports range and no angle. The camera OVERRULES it: if the eye
    # recognises him in its cone he is at the desk whatever the range says.
    # camera_zone is what that verdict is called, and it defaults to "at
    # the desk" because the office is the only room with a lens -- a room
    # whose camera watches something else must set its own.
    #
    # The office ladder is the real geometry and is NOT a naive
    # "desk = nearest band". The profile in ~/.config/jarvis/room-sensors/
    # office.json records the module as sitting on the desk at the BACK
    # edge aimed OUT across the room at the door, so sitting in the chair
    # he is behind it and inside its 0.75 m blind zone, and the first 1.5 m
    # in front of it has no still-target sensitivity at all. The nearest
    # band is therefore the floor IN FRONT of the desk; the chair belongs
    # to the camera. Edges are multiples of one 0.75 m distance gate --
    # anything finer is a fiction the sensor cannot support -- and the
    # ladder stops at 4.5 m, the device's tuned far gate.
    #
    # dwell_s is the anti-chatter hold: a zone change commits only after it
    # has held this long. 3.0 s is three consecutive reads at the 2.0 s
    # poll cadence, longer than the ~0.6 s a walker spends inside the
    # narrowest band, and well inside the radar's own 10 s absence delay.
    # log_path "" means ~/.local/state/jarvis/zones.jsonl (0600 in a 0700
    # directory). The file rotates at log_max_bytes keeping one generation,
    # so 2 MB is the ceiling. No line may exceed jarvis/zones.py's
    # MAX_LINE_BYTES (640) -- append refuses one that would -- so 1 MB is
    # at least 1,562 transitions of any shape. What a real day writes is
    # not measured; earlier comments here quoted a per-record average that
    # would not reproduce, so it has been removed rather than restated.
    #
    # EDIT THIS SECTION AND MIND THE TYPO. Anything here of the wrong SHAPE
    # -- a rooms that is not a list, an entry that is not an object, a
    # near_m that is text, an enabled that is the STRING "false" -- is
    # REFUSED BY NAME and records nothing. It does NOT fall back to the
    # ladder built into jarvis/zones.py, and only a key that is absent
    # altogether falls back to anything. That is deliberate: a log written
    # against the bands you thought you had replaced looks exactly like a
    # log that worked.
    "zones": {"enabled": True, "dwell_s": 3.0, "log_path": "",
              "log_max_bytes": 1000000, "log_keep": 1,
              "rooms": [
                  {"name": "office", "enabled": True,
                   "camera_zone": "at the desk",
                   "bands": [
                       {"name": "just off the desk",
                        "near_m": 0.75, "far_m": 1.5},
                       {"name": "the middle of the room",
                        "near_m": 1.5, "far_m": 3.0},
                       {"name": "by the door",
                        "near_m": 3.0, "far_m": 4.5}]}]},
    # The camera (jarvis/eye.py, scratchpad/ideas/vision.md). OFF until he
    # turns it on, and there is deliberately NO SCHEDULE HERE: offline mode
    # and the 21:00-07:00 curfew belong to the single sensing-state owner,
    # which is the only place they can be enforced at the device. A second
    # copy of the window in this file is a copy that can disagree with the
    # first, and the one that disagrees quietly is the one that leaves the
    # lens open at 22:00.
    #
    # THE RESOLUTION CHAIN, written out because the first version of this
    # section shipped a detect size that could not resolve the face the mount
    # arithmetic produces, and nothing connected the two numbers.
    #
    # THE FIELD OF VIEW IS NOW A KEY, AND THAT IS THE FIX. Until 2026-09-02
    # nothing in this dict was a field of view: the 90 deg the arithmetic
    # below reasons from lived only in this comment, which is precisely how
    # it got applied to a camera that does not have it. hfov_deg is the
    # HORIZONTAL field of the camera actually plugged in, and every
    # pixel-to-angle claim in jarvis/camera.py and jarvis/visionrig.py is
    # derived from it. There is no default anywhere in the code: an unset
    # hfov_deg is an error, never a 90 (jarvis/camera.lens_from_config).
    #
    # 65.6 is the LifeCam Cinema he already owns. Its spec sheet says 73 deg,
    # which is the DIAGONAL; the conversion to horizontal is a ratio of
    # TANGENTS, not of numbers, and gives 65.64 (jarvis/facemodels.py). Set
    # diag_fov_deg instead and the conversion is done here. For reference:
    # a Logitech C930e is 82.2 deg horizontal, and the 98 deg-diagonal
    # Arducam docs/vision.md section 9 recommends BUYING is 90.1 -- the
    # number the old comment assumed for all of them.
    #
    # THE RESOLUTION THAT FOLLOWS FROM IT, at the ~95 cm mount section 9
    # recommends. The LifeCam spans 2*95*tan(32.82) = 122 cm, so 1280 px is
    # 10.4 px/cm and a 16 cm face is 167 px. The Arducam spans 191 cm, so
    # 1920 px is 10.05 px/cm and the same face is 162 px: the narrower field
    # and the lower resolution very nearly cancel at full frame. They do NOT
    # cancel after the downscale below -- 1280->320 is 4x against 1920->320's
    # 6x -- so the detector sees 42 px on the LifeCam and 27 px on the
    # Arducam. Both clear YuNet's documented 10-300 px range; only one has
    # margin. 640x480, which shipped first, would make the same face 53 px
    # against SFace's 112x112 input, which section 9 calls "far too small".
    #
    # width/height is 1280x720 BECAUSE THAT IS A MODE THE LIFECAM HAS.
    # 1920x1080 shipped here until 2026-09-02 and is not one of them; a
    # driver asked for a mode it does not have does not error, it quietly
    # grants a different one, which is why scripts/vision_selfcheck.py reads
    # the granted mode back and prints it.
    # Measured on /dev/video0 that night: the granted mode is nominally
    # 30 fps but delivers a frame every 130 ms (p50) -- ~7.5 fps, which is
    # the real ceiling any preview or detector loop has to live inside.
    #
    # detect_width/height is the size YuNet actually sees. 320x180, not the
    # 320x240 that shipped first, and the reason is aspect, not pixels: a
    # 16:9 frame squeezed into 4:3 is a 1.33x anisotropic horizontal squash of
    # every face in it. 1920x1080 -> 320x180 is an exact 6:1 in both axes,
    # while ->320x240 is 6:1 and 4.5:1; on the LifeCam's 1280x720 it is 4:1
    # and 4:1 against 4:1 and 3:1. The squash is the argument, and it is pure
    # arithmetic that holds for both cameras (tests/test_facemodels.py).
    #
    # AN EARLIER VERSION OF THIS COMMENT CARRIED A TABLE OF YuNet CONFIDENCE
    # SCORES (320x240 -> 0.703, 320x180 -> 0.840, 640x360 -> 0.894) ATTRIBUTED
    # TO A MEASUREMENT ON THIS BOX ON 2026-09-02. Those numbers cannot have
    # been measured: the weights were not on this machine until they were
    # downloaded on 2026-09-02 (find / -iname '*yunet*' returned nothing), and
    # a confidence score requires a real face, which nothing here is permitted
    # to look at. They have been removed rather than corrected. The real
    # per-frame COST, measured on synthetic frames once the weights existed
    # (scripts/measure_face_models.py, resize from 1280x720 + detect, p50, on
    # this 20-core aarch64 box) --
    #
    #   detect       1 thread   2 threads   4 threads
    #   320x180       2.41 ms     2.16 ms     1.50 ms
    #   320x240       3.25 ms     1.99 ms     1.35 ms
    #   640x360       9.46 ms     5.21 ms     3.06 ms
    #   1280x720     39.46 ms    21.30 ms    11.86 ms
    #
    # -- which does NOT reproduce the old claim that 320x180 is 2.0 ms cheaper
    # than 320x240. It is cheaper only single-threaded; at 2 and 4 threads the
    # taller frame is very slightly FASTER, presumably tiling. So 320x180 is
    # kept on the aspect argument alone, which is sound, and not on a speed
    # argument, which is not. Whether a real face clears min_conf at either
    # size is still unmeasured and needs a camera and a tape.
    #
    # min_conf 0.6, down from 0.7, for the asymmetry: a miss is SILENT and
    # disables the whole feature, while a false face has to survive faces==1
    # and 0.6 s of dwell before it can promote anything. Re-measure both with
    # a tape and a real camera (section 9's $0 test) before trusting them.
    #
    # threads 2, not the 4 that docs/vision.md section 2 first advised. 4 is
    # the latency win (SFace 20.9 -> 5.8 ms wall, re-measured 2026-09-02
    # against real weights) but it costs MORE total CPU, not less -- 21.1
    # CPU-ms at 1 thread against 22.6 at 4, and 2 threads is the sweet spot at
    # 10.1 ms wall for 20.4 CPU-ms. At the
    # armed tier's 8 fps the frame period is 125 ms and the whole chain is
    # ~20 ms even at 2, so latency is not the binding constraint; contention
    # with live Jarvis, ollama and F5 on a box that has already had one
    # unified-memory power-off is.
    #
    # Two frame rates because "is anyone there" and "is he addressing me" are
    # different questions with different budgets.
    #
    # identity=False is the phase gate: with it off, nothing about his face is
    # ever written down (jarvis/facegallery.py is not constructed at all).
    "camera": {"enabled": False, "device": "", "width": 1280, "height": 720,
               # The lens. hfov_deg is horizontal degrees; set diag_fov_deg
               # instead if the spec sheet quotes the diagonal (most webcams
               # do) and leave hfov_deg at 0. One of the two must be set --
               # nothing in the code guesses a field of view.
               "hfov_deg": 65.6, "diag_fov_deg": 0.0, "fourcc": "MJPG",
               # exposure: 0 leaves the camera on its own auto-exposure. A
               # value > 0 PINS manual exposure (v4l2 exposure_absolute
               # units, 100 us) at every open. Why the lever exists: the
               # LifeCam's sensor rate is set by its exposure tier -- 30 fps
               # at <=156, 15 at 312-625, 7.5 at >=1250 -- and its auto
               # metering stepped down to the slowest tier on 2026-09-03
               # evening, halving the preview to 3.7 fps; manual 156 was
               # MEASURED (09-04, grab only) to return it to 15-16 fps
               # through the app's single driver buffer. It trades the
               # camera's metering for a fixed rate, so whether the pane is
               # still watchable in evening light is read off the next
               # "campreview:" log line, not assumed. Values off the
               # camera's own table (50, 100, 200, 400) fall to the SLOWEST
               # tier -- 156 first, then read the line.
               "exposure": 0,
               "detect_width": 320, "detect_height": 180, "threads": 2,
               "idle_fps": 1.5, "armed_fps": 8.0, "min_conf": 0.6,
               # Where the downloaded weights live. Empty means
               # ~/.aiws_trainer/models/face (jarvis/facemodels.py). They are
               # NOT in the repo and must not be: 38 MB of SFace beside 140 MB
               # of TTS weights is how repo/ got swept into a commit once.
               "model_dir": "",
               # WHICH PAIR OF FACE MODELS. Empty means the shipped default,
               # which as of 2026-09-03 is "insightface": SCRFD-500m +
               # ArcFace-mbf, 512-D. Set it to "opencv" to go back to YuNet +
               # SFace, 128-D, and to the enrolment already on disk. That
               # reversal is why both pairs stay declared.
               #
               # WHY THE SWAP: his words, 2026-09-03, "he also is recognizing
               # me less from the side angle". A 128-D SFace embedding is weak
               # in profile. The new pair is SMALLER (15.4 MB against 37.1)
               # and FASTER (measured on this box: SCRFD 2.5 ms + ArcFace
               # 5.2 ms at 2 threads, against YuNet 1.5 + SFace 10.2) with a
               # 512-D embedding. Whether it is MORE ACCURATE ON HIS FACE is
               # NOT measured and cannot be measured without him: run
               # scripts/face_model_compare.py.
               #
               # LICENCE, AND IT IS NOT THE USUAL ANSWER. The InsightFace
               # weights are NON-COMMERCIAL RESEARCH ONLY -- that is the one
               # statement of terms that exists for them, and their repository
               # has no LICENSE file at all. Fine for Jarvis, which is his own
               # research use. NOT fine for VSS or anything that ships from
               # this machine; jarvis/facemodels.commercial_backends() is the
               # list to pick from there.
               #
               # SWAPPING TURNS IDENTITY OFF UNTIL HE RE-ENROLS. His gallery
               # holds SFace's 128-float vectors and an ArcFace vector is 512;
               # the cosine between them measures nothing, so the gallery
               # refuses to compare across models by name rather than scoring
               # noise. The old generations stay on disk, untouched, and the
               # startup log carries one line saying to re-enrol.
               "face_backend": "",
               # The attention cone, in degrees off the lens axis. 20 deg is
               # generous against the 47 deg separation an off-axis mount
               # gives (scratchpad/ideas/camera.md section 4) and useless on a
               # monitor-top mount, where the screen's own top edge is 1.8 deg
               # away.  Hysteresis on release so a blink does not drop it.
               # cone_centre_deg AIMS the cone off the lens axis, which an
               # off-axis mount needs: the camera sits beside the monitor, so
               # "looking at Jarvis" need not be "looking down the lens".
               "cone_deg": 20.0, "cone_hysteresis_deg": 5.0,
               "cone_centre_deg": 0.0, "dwell_s": 0.6,
               # Nose-tip protrusion over interocular distance -- the ONE
               # anthropometric constant standing between the measured
               # landmark ratio and a head angle in degrees. AN ASSUMPTION,
               # not a measurement of him: ~2.2 cm over ~6.3 cm. A 23% error
               # in it is ~5 deg at the cone edge, a quarter of the cone, so
               # calibrate it from the $0 photo test (docs/vision.md 9): the
               # rig prints the raw ratio t, and r = t / tan(known angle).
               "nose_ratio": 0.35,
               # There is deliberately NO wake-gate key here. eye.resolve_wake
               # exists and nothing calls it (docs/vision.md: the wiring "is
               # a two-line change when one exists"), so a camera wake-tiebreak
               # key sat here for two days promising a control the code did
               # not have. It comes back with the wiring, not before.
               # Face identity: a gallery of HIS FACE on disk. Opt-in, and the
               # threshold is OpenCV's own documented SFace cosine for "same
               # person".
               #
               # THIS NUMBER BELONGS TO SFACE AND DOES NOT CARRY ACROSS. It is
               # OpenCV's published figure for SFace's 128-D vectors, and it
               # was raised by hand on 2026-09-03 (to 0.47 in his own config)
               # from SFace scores measured on his face and on his wall.
               # ArcFace's cosines are a different model's distribution;
               # nothing has been measured for them on this machine, so with
               # camera.face_backend on insightface this bar is UNMEASURED and
               # jarvis/camera.identity_min_warning says so in the log every
               # start. scripts/face_model_compare.py is how the real number
               # arrives, and only he can run it -- it needs his face.
               "identity": False, "identity_min": 0.363,
               # And NO frame-to-disk key. A camera debug-frame key ("one
               # JPEG at 0600, for diagnosing a mount") sat here unread, and
               # an implementer following its comment would have written a
               # camera frame to disk -- the one thing the standing camera
               # rule and jarvis/campreview.py's docstring forbid. The mount
               # is diagnosed from numbers (scripts/vision_selfcheck.py).
               # The console's camera pane (jarvis/campreview.py,
               # jarvis/ui/preview.py). His words, 2026-09-02: "lets add a
               # small camera with visable tracking on the jarvis app but
               # make me be able to turn if off in settings" -- this is the
               # settings half, and the Privacy row in the drawer writes it.
               #
               # OFF by default, like every other lens key here. A pane that
               # switched itself on would be this feature introducing itself
               # by breaking the rule it lives under; and it is gated by
               # sensing.py on top, so offline mode and the curfew shut it
               # whatever this says.
               #
               # preview_fps is the PICTURE rate, a CEILING the device may
               # not reach, and THE SAME NUMBER AS campreview.DEFAULT_FPS --
               # pinned equal by tests/test_campreview.py, because this dict
               # is deep-merged into every config, so THIS is the default a
               # fresh install runs at and campreview's copy is only reached
               # with no config at all. The two were 15.0 and 7.5 for a day.
               #
               # 7.5 is the highest rate the running app has been measured to
               # get from the LifeCam's 1280x720 MJPG mode: 7.4-7.6 fps with
               # him at the desk at 10 requested (2026-09-03 00:04-00:22)
               # and 7.6 at 15 requested (19:47 the same day). A ceiling
               # above the delivered rate costs nothing but a parked read;
               # one below it throws pictures away, which is why the default
               # is the best measured rate and not the worst. An earlier
               # comment here said the 720p grab was 11 ms and that the
               # cadences kept the pane at ~12% of one core instead of 75%;
               # neither reproduced and both are gone -- the measured story,
               # including why the 7.5 is not understood, is the module
               # docstring of jarvis/campreview.py. His own config sets this
               # key explicitly, so his rate is whatever he last wrote there.
               # Capped at 30 in campreview, the mode's granted nominal; the
               # boxes and the name have their own slower cadences, and the
               # capture runs on its own thread off a latest-wins slot.
               "preview": False, "preview_fps": 7.5},
    # Grab and throw (jarvis/gesture.py, jarvis/handstage.py,
    # jarvis/cast.py, jarvis/gesturecast.py). His words, 2026-09-03: "reach
    # out and grab at the screen (in the air) where the camera is and then
    # gesture towards almost throwing the cast onto the HPCOMPUTER".
    #
    # OFF by default, like every other lens key. It RIDES THE CAMERA
    # PREVIEW: the hand stage runs inside the preview's own capture, on the
    # frame it already pulled, so camera.preview must be on and the console
    # active for a gesture to be seen at all -- there is no second device,
    # no second thread, and every way the preview shuts (curfew, offline,
    # standby, the toggle, quit) shuts this too. The models are the two
    # opencv_zoo MediaPipe hand graphs under ~/.aiws_trainer/models/hand
    # (Apache-2.0, sha-verified by jarvis/handpose.py); never in the repo.
    #
    # sinks: which side is which machine, taught by voice ("HPCOMPUTER is
    # on my right") and SHIPPED EMPTY -- nobody but Hunter can see the
    # room, so until he says, every throw lands on the board and Jarvis
    # tells him once how to teach a side. Keys are "left" / "right";
    # values are "board", "hpcomputer" or "handoff".
    #
    # EVERY NUMBER BELOW IS A CALIBRATED STARTING POINT, measured on a
    # synthetic hand with his own lens constants and never on his hand
    # (jarvis/gesture.py CastThresholds carries the measurements). The
    # self-check prints what his hand actually measures:
    #   ~/vss_env/bin/python scripts/gesture_selfcheck.py --seconds 30
    "gesture": {"enabled": False, "sinks": {},
                # ORT intra-op threads for the two hand graphs, SEPARATE
                # from camera.threads (cv2's global). 2 matches it: the
                # latency win of 4 costs total CPU on a box that has had
                # one unified-memory power-off already. model_dir empty
                # means ~/.aiws_trainer/models/hand. mirrored: the LifeCam
                # feed is NOT mirrored (cv2.flip appears nowhere), so image
                # +x is his LEFT; set true only if the feed is ever flipped.
                "hand_threads": 2, "model_dir": "", "mirrored": False,
                # Say "Holding <thing>, sir." on the grab (the tone plays
                # either way, and the chip names it either way).
                "speak_grab": True,
                # Attention is LATCHED, not sampled: one face attending
                # within attend_latch_s arms the hand stage, then a reach or
                # a carry keeps it armed -- the reaching arm crosses the face
                # at exactly the moment it matters. GUESSED (the design pass
                # offered 1.0 and 3.0). The reach ratio's scale is a rolling
                # MEDIAN of the interocular distance over face_window_s
                # with at least face_min_samples faces seen; no baseline
                # means no grab, stated as a refusal, never a default.
                "attend_latch_s": 3.0, "face_window_s": 5.0,
                "face_min_samples": 3,
                # The hand: C = mean fingertip-to-wrist / palm_diag. A fist
                # reads <= 0.680 and an open hand >= 0.919 over the pose
                # envelope; the two bars sit inside that gap with a dead
                # band between so a hand at the boundary cannot chatter.
                "closed_max": 0.70, "open_min": 0.85,
                # The reach: R = palm_diag / interocular. A hand at the FACE
                # PLANE never exceeded 2.03; 2.35 was the first bar with zero
                # false grabs in 288 everyday motions and full lateral
                # recall. reach_arm is where the stage starts preparing the
                # subject so the grab feels instant.
                "reach_min": 2.35, "reach_arm": 1.60,
                # FRAMES, not seconds, deliberately: at 7.5 fps a "300 ms
                # dwell" is 2.25 frames and the rounding decides whether it
                # works. dwell 3 = 400 ms of a still fist at reach; the hand
                # must have been seen OPEN within open_lookback_frames (1.6 s)
                # or a resting fist drifting into the zone becomes a grab;
                # anchor_drift_u is how still "still" is, in hand-units
                # (~7x the 0.05-0.09 landmark noise floor).
                "dwell_frames": 3, "open_lookback_frames": 12,
                "open_frames_req": 1, "anchor_drift_u": 0.60,
                # The throw, in hand-units of travel from the anchor: opened
                # in frame needs a full hand-width (he opens his hand
                # hundreds of times an hour); left the picture within
                # edge_frac of a half-field needs only 0.25 (leaving is the
                # evidence); vanished in open space needs 0.50 AND a last
                # step of exit_step_u. Anything less is a DROP -- the cheap,
                # reversible outcome.
                "throw_release_u": 1.00, "throw_exit_u": 0.25,
                "throw_lost_u": 0.50, "exit_step_u": 0.35, "edge_frac": 0.30,
                # A carry survives lost_grace_frames of missed detection,
                # and ends at carry_max_frames OR carry_max_s, whichever
                # first (the seconds are the wall-clock backstop for a
                # starved frame rate). cooldown_frames after any carry end.
                "lost_grace_frames": 2, "carry_max_frames": 30,
                "carry_max_s": 8.0, "cooldown_frames": 8,
                # Direction: four +/-sector_half_deg sectors in HIS frame
                # with 20 deg of "ambiguous" between them -- an unnameable
                # fling is a drop. Only left and right THROW (measured:
                # vertical throws scored 12/12 or 2/12 on finger direction
                # alone); down is the cancel, up is not a target. A second
                # hand at reach depth at least second_hand_frac the size of
                # the first makes the frame ambiguous: two hands out at the
                # lens is not this gesture.
                "sector_half_deg": 35.0, "target_sectors": ["left", "right"],
                "second_hand_frac": 0.70},
    # The arc (jarvis/arc.py): one name for the hour of the house --
    # pre-dawn / waking / working / afternoon / dusk / evening / night --
    # from locally computed sunrise/sunset plus quiet, presence and focus.
    # A state source only; the consumers are what you hear and see.
    "arc": {"enabled": True, "tick_s": 60},
    # Earcons (jarvis/earcons.py): the six-tone family that replaced the
    # scattered beeps. cooldown_s is shared across ALL causes, so a noisy
    # room cannot turn a false wake into a metronome.
    "sound": {"earcons": True, "cooldown_s": 4, "volume": 0.5},
    # Room tone (jarvis/roomtone.py): a near-subliminal generated bed that
    # follows the arc. OFF by design -- a continuous bed enters every
    # capture and the wake word, the endpointer and the ECAPA gate were all
    # tuned in a quiet room. Turn it on by voice ("room tone on") once you
    # have measured that it costs nothing at the mic.
    "ambience": {"room_tone": False, "volume": 0.05, "away_stop_min": 20},
    # The voice path's two latency/feedback policies (jarvis/app.py):
    # speculative_stt decodes the clip during the endpoint silence and reuses
    # the result when no more speech followed; nudge is the short spoken /
    # earcon cue after a wake-word turn that produced nothing to answer,
    # at most once per nudge_cooldown_s.
    "listening": {"speculative_stt": True, "nudge": True, "nudge_cooldown_s": 30},
    # The console's surfaces (jarvis/ui/console_mode.py, jarvis/ui/board.py).
    # board: allow "bring up the board", the docked mission-control panel on
    # the empty right flank. ambient/standby: between conversations the
    # console shows one room-state slab, and after standby_after_min away
    # from the keyboard it becomes the room's clock. standby_dim is the
    # BRIGHTNESS FLOOR, and it is a canvas-colour blend inside our own
    # window -- nothing here touches xrandr gamma or the desktop's
    # brightness, so a crash cannot leave the panel dark. drift_px_per_min
    # walks the window slowly so a static clock cannot burn in. powerup: the
    # staged sweep the first time he sits down after an overnight gap, at
    # most once a day (the date latch lives beside the briefing's, in
    # briefing_state.json). look: "holo" (the 2026-09-01 blue-holographic
    # overhaul) or "classic" (the 08-31 console, token for token -- the
    # fallback); read ONCE at window creation, so it applies after a
    # restart. The voice command "switch to classic visuals" writes it; a
    # JARVIS_LOOK in the environment outranks it (jarvis/ui/theme.py).
    "console": {"board": True, "ambient": True, "standby": True,
                "ambient_after_s": 45, "standby_after_min": 12,
                "standby_dim": 0.35, "drift_px_per_min": 3,
                "powerup": True, "powerup_gap_h": 6, "look": "holo"},
    # Phone intercom (jarvis/intercom.py): a recorded clip sent over the
    # command socket instead of a wake word. verify_speaker runs the same
    # ECAPA gate the microphone path uses -- off by default because the
    # 0600 socket behind the user's own SSH session is already the
    # authentication, and a phone codec moves the embedding far enough that
    # the gate (which fails SHUT) would reject his own voice. max_mb caps
    # one clip; the socket framing allows a little more than this.
    "intercom": {"enabled": True, "verify_speaker": False, "max_mb": 10},
    # Hunter's Oracle Cloud VM (jarvis/tools/oracle.py): `demon-bot`,
    # opc@163.192.101.18, Oracle Linux Server 9.6, running NINE app services
    # under systemd behind nginx. Verified by ssh 2026-08-31; there is no
    # pm2 on it (the pm2/game-news table this section used to hold described
    # an OLDER server). OUTBOUND ONLY -- Jarvis asks it questions over ssh;
    # nothing here opens anything the other way, and there is deliberately
    # no tunnel, reverse tunnel or port-forward setting to turn on.
    #
    # OFF by default. `key_path` points at the key that is verified to log
    # in as opc, so `"enabled": true` is the ONLY edit needed to switch the
    # lane on; until then every entry point answers one line naming what is
    # missing and NOTHING opens a socket. timeout_s is the whole budget for
    # one round trip -- the real round trip measures 1.0 s -- and cache_s is
    # how long the last good reading answers a second question for free.
    #
    # `services` is the ALLOW-LIST. A spoken name resolves to one of these
    # ROWS and the command is built in oracle.py from a fixed template plus
    # that row's unit; nothing from a transcript is ever interpolated into a
    # command, and a unit name that is not a plain systemd unit is dropped
    # at load. The only actions are status, logs and restart, and a restart
    # is read back for a yes first. `name` is what he is called out loud.
    "oracle": {
        "enabled": False,
        "host": "163.192.101.18",
        "user": "opc",
        "key_path": "~/Downloads/Oracle Cloud Service (2)/Oracle Cloud "
                    "Service/Discord Bot/Keys/ssh-key-2025-08-15.key",
        "timeout_s": 6,
        "cache_s": 25,
        "log_lines": 20,
        "services": {
            "haymaker": {"unit": "haymaker-bot", "name": "Haymaker"},
            "coa": {"unit": "coa-bot", "name": "Court of Awe",
                    "aliases": ["court of awe", "court of aw"]},
            "exoshock": {"unit": "exoshock-bot", "name": "Exoshock",
                         "aliases": ["exo shock"]},
            "vrider": {"unit": "vrider-bot", "name": "VRider",
                       "aliases": ["v rider", "the rider"]},
            "timecard": {"unit": "timecard-bot", "name": "Timecard",
                         "aliases": ["time card"]},
            "knightfall": {"unit": "knightfall-web", "name": "Knightfall",
                           "aliases": ["knightfall protocol", "nightfall"]},
            "elevation api": {"unit": "elevation-api",
                              "name": "the elevation API",
                              "aliases": ["elevation", "ditch grade"]},
            # no bare "monday": "what about Monday?" is about the week
            "monday sync": {"unit": "monday-sheets-sync",
                            "name": "Monday sync",
                            "aliases": ["sheets sync", "monday sheets"]},
            "dashboard": {"unit": "bot-dashboard", "name": "the dashboard",
                          "aliases": ["bot dashboard"]},
        },
    },
    # The phone client (jarvis/webapp.py): a page served to a device on the
    # home Wi-Fi, off the same dispatch_text the CLI uses. OFF by default
    # and deliberately so -- nothing listens on a network port until he
    # turns this on. `bind` empty means "this box's own LAN address",
    # discovered at start; whatever it resolves to must be a PRIVATE
    # address or the server refuses to start, and there is no wildcard
    # bind, no tunnel and no port-forward anywhere in that module. `token`
    # is generated on first enable, lives in this 0600 file and is in
    # SECRET_KEYS; delete it and restart to rotate the key. link_file /
    # qr_file are where the URL-with-key is left for him, 0600.
    "phone": {"enabled": False, "bind": "", "port": 8765, "token": "",
              "max_audio_mb": 8, "link_file": "~/jarvis-phone.txt",
              "qr_file": "~/jarvis-phone.svg"},
    # "Email this file to this person" (jarvis/outbox.py). Nothing here
    # switches the feature on or off: it is on, and what makes it safe is
    # the spoken read-back and the yes, not a flag.
    #
    # `roots` are the ONLY folders a spoken file NAME may resolve inside --
    # ~ is a whole filesystem and "the lab report" must not be able to reach
    # a README three levels down a checkout. An absolute path he gives
    # outright is allowed outside them (filephrase.DENY_ROOTS says what is
    # still refused there: anything behind a leading dot, anything under a
    # system tree).
    #
    # `max_mb` may only be lowered. 18 MB is what Gmail will actually
    # deliver once base64 has inflated the file by 4/3 inside a 25 MB
    # limit; a bigger number here would not send a bigger file, it would
    # move the refusal to the SMTP server AFTER the read-back had promised
    # the thing went.
    #
    # `contacts` is a name -> address map, checked before the people book
    # in jarvis/memory.py. Both are consulted and NEITHER is guessed at: an
    # unknown name is a question, never a plausible address.
    #
    # `from` is a label from gmail.accounts. Blank with more than one
    # account configured means Jarvis asks which identity to send as, which
    # is the right default -- personal, work and school are three different
    # people to whoever receives the mail.
    "send_file": {"roots": ["~/Desktop", "~/Downloads", "~/Documents"],
                  "max_mb": 18, "from": "", "contacts": {},
                  "body": "Sent from Jarvis."},
    # HPCOMPUTER -- files both ways and a short allow-list of read-only
    # questions (jarvis/tools/remote.py).  Ships OFF and EMPTY because as of
    # 2026-09-02 the host is not on the tailnet at all, has no sshd
    # reachable and has no key here; `host` stays blank until it joins, and
    # a blank host is refused by name ("it isn't on the tailnet yet").
    #
    # Deliberately NOT in SETUP_LINES/SECTIONS, exactly like `oracle`:
    # missing_sections() drives a spoken nag at boot, and nagging about a
    # machine he has not chosen to connect yet would be noise.
    #
    # `socks_proxy` is not optional here and not a preference.  tailscaled
    # on this box runs --tun=userspace-networking, so there is NO route to
    # 100.64/10 and a direct ssh to a tailnet name fails with "network is
    # unreachable" however healthy the tailnet is.  1055 is the daemon's
    # own SOCKS5 port.  Blank it only if this box ever gets a real tun.
    #
    # `inbox` is the ONLY directory a push can land in, and `pull_dirs` the
    # only ones a pull may read: speech never names a remote path.
    "remote": {
        "enabled": False,
        "host": "",                       # e.g. hpcomputer.tail5323b8.ts.net
        "user": "",
        "key_path": "",                   # a path ssh already owns; never a key
        "name": "HPCOMPUTER",
        "timeout_s": 12,
        "transfer_timeout_s": 120,
        "socks_proxy": "127.0.0.1:1055",
        "inbox": "~/jarvis-inbox",
        "pull_dirs": {"outbox": "~/jarvis-outbox",
                      "desktop": "~/Desktop",
                      "downloads": "~/Downloads"},
        "max_mb": 100,
    },
    # jarvis/gate.py -- whether Jarvis answers whoever speaks, or only the
    # people he recognises.  THIS IS RECOGNITION, NOT A LOCK: a photograph
    # defeats the face check and a recording defeats the voice check, and
    # the user was told that and accepted it.
    #
    # SHIPS IN SHADOW, and that is the whole point of the key.  In shadow
    # every verdict is logged and NOTHING is refused, so a day of real
    # turns can be read before the first refusal.  His voiceprint corrupted
    # once already this week (jarvis/config.py, PATHS.VOICEPRINT), and a
    # gate that had been enforcing that morning would have locked him out
    # of his own house for the rest of the day.  Move it to "enforce" when
    # the log says the verdicts are right; "off" switches it out entirely.
    #
    # NO SECRET LIVES HERE.  The salted hashes of the spoken passphrase and
    # the typed override code are in jarvis/identity.py's own 0600 file
    # (PATHS.OWNER_REGISTRY), never in this object -- it is deep-copied
    # into reports and its __repr__ prints redacted(), so the strongest way
    # to keep a hash out of a log is to keep it out of here.  SECRET_KEYS
    # is deliberately untouched.
    #
    # `gate_typed` reverses the one exemption that is a judgement call
    # rather than a ruling: typed input at the Tk box is exempt on the same
    # argument as the command socket -- he is physically at the machine.
    "owner": {
        "mode": "shadow",                 # shadow | enforce | off
        "gate_typed": False,
    },
}

SECRET_KEYS = ("icloud.app_password", "gmail.app_password", "discord.bot_token",
               "spotify.client_secret", "canvas.token",
               # the phone client's bearer key: it is the whole
               # authentication, so it must never reach a log
               "phone.token")
# Secrets that live inside a LIST of sections rather than at a dotted path:
# (list key, field). gmail.accounts[].app_password was invisible to redacted()
# and scrub(), so repr(cfg) printed three real app passwords in full.
SECRET_LIST_FIELDS = (("gmail.accounts", "app_password"),
                      # The satellite's web_server basic-auth password. It is
                      # the ONLY thing between anyone on the segment and the
                      # radar lease buttons -- scripts/esphome/jarvis-satellite
                      # .yaml says so in its own words ("world-writable"
                      # without it) AND told him it was already masked here,
                      # which it was not: repr(cfg) printed it in clear,
                      # secret_values() did not list it, and scrub() left it in
                      # a log line. MEASURED 2026-09-03 with a gmail
                      # app_password and a satellite password in one config:
                      # the gmail one was masked, the satellite one was not.
                      # It lives in presence.rooms[] since 2026-09-04, when
                      # that became the ONE room list (F02); the entry here
                      # moved with it.
                      ("presence.rooms", "password"))

# is_configured() / setup_line() sections and the film-JARVIS excuse for each.
SETUP_LINES: dict[str, str] = {
    "home_location": "I'll need your home location set up, sir; "
                     f"the notes are in {DOCS_HINT}.",
    "google_ical": "I'll need your Google calendar link set up, sir; "
                   f"the notes are in {DOCS_HINT}.",
    "icloud": "I'll need your iCloud calendar set up, sir; "
              f"the notes are in {DOCS_HINT}.",
    "gmail": "I'll need your Gmail app password set up, sir; "
             f"the notes are in {DOCS_HINT}.",
    "discord": "I'll need the Discord bot set up, sir; "
               f"the notes are in {DOCS_HINT}.",
    "claude": "I'll need the Claude command line set up, sir; "
              f"the notes are in {DOCS_HINT}.",
}
SECTIONS = tuple(SETUP_LINES)

# Values that mean "not filled in yet": empty, "<paste here>", "PASTE-…",
# "your-…", "changeme", "xxxx…".  Real secrets never look like these.
_PLACEHOLDER = re.compile(
    r"^\s*$|^<.*>$|^(paste|your|replace|todo|example)[-_ :]|"
    # "me" NOT followed by a letter, rather than \b: the satellite YAML ships
    # CHANGE_ME_32_RANDOM_CHARS, and "_" is a word character, so \b never
    # fired after "ME" and the shipped placeholder was treated as a real
    # secret -- scrub() would have blanked those words wherever they appeared.
    r"^change[ -_]?me(?![a-z])|^x{3,}$", re.I)


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    if not isinstance(value, str):
        return not value
    return bool(_PLACEHOLDER.search(value.strip()))


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` laid over it; dicts merge
    recursively, lists and scalars replace.  Unknown keys survive."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_path(path: Optional[os.PathLike | str] = None) -> Path:
    """Resolve the config file: explicit arg, else $JARVIS_ASSISTANT_CONFIG,
    else ``~/.config/jarvis/assistant.json``."""
    raw = path or os.environ.get(ENV_VAR) or ""
    if raw:
        return Path(os.path.expanduser(str(raw)))
    return Path.home() / ".config" / "jarvis" / "assistant.json"


def _claude_bin() -> str:
    """Seam for is_configured('claude'); mirrors MachineProfile.claude_bin
    without importing jarvis.config (keeps this module import-light).  A
    GNOME autostart launch may lack ~/.local/bin on PATH, so that install
    location is the last resort."""
    found = os.environ.get("JARVIS_CLAUDE_BIN") or shutil.which("claude")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "claude"
    return str(local) if os.access(local, os.X_OK) else ""


def _write_private(path: Path, text: str) -> None:
    """Atomic write with mode 0600: temp file in the same directory
    (created 0600 by mkstemp), fsync, os.replace, chmod to be sure."""
    # mode applies only when the directory is created (never widened later)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".assistant-", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def _file_stamp(path: Path):
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class _Section:
    """Live attribute view over one dict of the config: ``cfg.alarms.volume``
    reads through ``cfg.get("alarms.volume")`` every time, so it never goes
    stale after ``set()`` or ``reload_if_changed()``.  Assignment saves:
    ``cfg.alarms.volume = 0.5``."""
    __slots__ = ("_cfg", "_prefix")

    def __init__(self, cfg: "AssistantConfig", prefix: str):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_prefix", prefix)

    def _key(self, name: str) -> str:
        return f"{self._prefix}.{name}"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        sentinel = object()
        value = self._cfg.get(self._key(name), sentinel)
        if value is sentinel:
            raise AttributeError(
                f"assistant.json has no key {self._key(name)!r}")
        if isinstance(value, dict):
            return _Section(self._cfg, self._key(name))
        return value

    def __setattr__(self, name: str, value) -> None:
        self._cfg.set(self._key(name), value)

    def __getitem__(self, name: str):
        sentinel = object()
        value = self._cfg.get(self._key(name), sentinel)
        if value is sentinel:
            raise KeyError(self._key(name))
        return value

    def get(self, name: str, default=None):
        return self._cfg.get(self._key(name), default)

    def as_dict(self) -> dict:
        return self._cfg.get(self._prefix, {}) or {}

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def __contains__(self, name: str) -> bool:
        return name in self.as_dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def __eq__(self, other) -> bool:
        if isinstance(other, _Section):
            return self.as_dict() == other.as_dict()
        return self.as_dict() == other

    def __repr__(self) -> str:
        shown = self._cfg.redacted()
        for part in self._prefix.split("."):
            shown = shown.get(part, {}) if isinstance(shown, dict) else {}
        return f"<{self._prefix} {shown!r}>"


class AssistantConfig:
    """See the module docstring and spec 10.2."""

    DEFAULTS = DEFAULTS
    SECRET_KEYS = SECRET_KEYS
    SECTIONS = SECTIONS

    def __init__(self, data: Optional[dict] = None,
                 path: Optional[os.PathLike | str] = None):
        self._lock = threading.RLock()
        self.path: Optional[Path] = Path(path) if path else None
        self._data: dict = _deep_merge(DEFAULTS, data or {})
        self._stamp = _file_stamp(self.path) if self.path else None
        # What load() found on disk; ensure_defaults() acts on it.
        self.disk_state = {"missing": False, "corrupt": False,
                           "loose_mode": False, "new_keys": False}

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None) -> "AssistantConfig":
        """Never raises, and NEVER WRITES. Reads the file and merges DEFAULTS
        over it in memory; a missing or corrupt file, a loose mode and keys
        DEFAULTS has gained since the file was written are only NOTED (in
        ``disk_state``) and logged. ``ensure_defaults()`` is the write, and
        the app is its only caller."""
        p = config_path(path)
        raw: dict = {}
        state = {"missing": False, "corrupt": False, "loose_mode": False,
                 "new_keys": False}
        try:
            if p.exists():
                try:
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict):
                        raise ValueError(f"top level is {type(loaded).__name__}")
                    raw = loaded
                except (ValueError, UnicodeDecodeError) as exc:
                    log.warning("assistant config %s is corrupt (%s); using "
                                "defaults in memory (the app moves it aside "
                                "and recreates it at startup)", p, exc)
                    state["corrupt"] = True
                else:
                    try:
                        mode = p.stat().st_mode & 0o777
                        if mode & 0o077:
                            state["loose_mode"] = True
                            log.warning("assistant config %s has mode %o; the "
                                        "app tightens it to 600 at startup",
                                        p, mode)
                    except OSError:
                        log.warning("could not check mode of %s", p)
            else:
                state["missing"] = True
                log.info("assistant config missing at %s; using defaults in "
                         "memory (the app creates it at startup)", p)
        except OSError:
            log.exception("assistant config %s unreadable; using defaults in memory", p)
        cfg = cls(raw, p)
        if not state["missing"] and not state["corrupt"] and cfg._data != raw:
            state["new_keys"] = True        # new keys since the file was written
            log.info("assistant config %s lacks keys DEFAULTS has gained; "
                     "using their defaults in memory", p)
        cfg.disk_state = state
        return cfg

    def ensure_defaults(self) -> bool:
        """THE write that load() used to do, made explicit: create the file
        with placeholders (0600) when it is missing, move a corrupt one to
        ``<name>.bad`` and recreate it, tighten a loose mode to 0600 and
        write back any keys DEFAULTS has gained. Returns True when the file
        was touched. Only jarvis.app calls this, once, at startup -- so the
        one process that owns the file is the only one that rewrites it,
        and a script, a service or a test that merely loads the config can
        never race the running app's saves (os.replace, last writer wins).
        Never raises."""
        if self.path is None:
            return False
        state = getattr(self, "disk_state", None) or {}
        touched = False
        p = self.path
        try:
            if state.get("corrupt") and p.exists():
                bad = p.with_name(p.name + ".bad")
                log.warning("assistant config %s is corrupt; moved to %s and "
                            "recreated", p, bad)
                os.replace(p, bad)
            if state.get("missing") or state.get("corrupt") \
                    or state.get("new_keys"):
                if state.get("missing"):
                    log.info("assistant config missing; creating %s with "
                             "placeholders", p)
                elif state.get("new_keys"):
                    log.info("assistant config %s gained new default keys", p)
                touched = self.save()
            elif state.get("loose_mode"):
                mode = p.stat().st_mode & 0o777
                if mode & 0o077:
                    os.chmod(p, 0o600)
                    log.warning("assistant config had mode %o; tightened "
                                "to 600", mode)
                    touched = True
        except OSError:
            log.exception("assistant config %s could not be brought up to "
                          "date; continuing with the in-memory copy", p)
            return touched
        if touched:
            self.disk_state = {"missing": False, "corrupt": False,
                               "loose_mode": False, "new_keys": False}
        return touched

    # ------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = None) -> Any:
        """Dotted lookup (``"claude.model"``).  Dicts and lists come back as
        copies: mutate-and-forget can't silently change the config."""
        node: Any = self._data
        with self._lock:
            for part in dotted.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            if isinstance(node, (dict, list)):
                return copy.deepcopy(node)
            return node

    def set(self, dotted: str, value: Any) -> bool:
        """Set a dotted key (intermediate dicts are created) and save
        atomically.  Returns True when the file was written."""
        parts = dotted.split(".")
        with self._lock:
            node = self._data
            for part in parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    if child is not None:
                        log.warning("assistant config: %s was %r; now a section",
                                    part, child)
                    child = node[part] = {}
                node = child
            node[parts[-1]] = copy.deepcopy(value)
        return self.save()

    def unset(self, dotted: str) -> bool:
        """REMOVE a dotted key and save. True when the file was written.

        The counterpart ``set`` never had, and it exists for exactly one
        job: retiring a key that has been superseded. A superseded key that
        is merely ignored still sits in his file looking live, and the next
        person to read it -- him, at midnight, wondering why the bands are
        not what he typed -- has no way to tell it apart from one that
        still drives something.

        A key that is not there is not an error and is not a write: False
        with nothing changed, so a caller can call it unconditionally.
        Intermediate keys that are not mappings are the same case.
        """
        parts = dotted.split(".")
        with self._lock:
            node = self._data
            for part in parts[:-1]:
                child = node.get(part) if isinstance(node, dict) else None
                if not isinstance(child, dict):
                    return False
                node = child
            if not isinstance(node, dict) or parts[-1] not in node:
                return False
            node.pop(parts[-1], None)
        return self.save()

    def update(self, values: dict) -> bool:
        """Several dotted keys, one save."""
        with self._lock:
            for dotted, value in values.items():
                parts = dotted.split(".")
                node = self._data
                for part in parts[:-1]:
                    child = node.get(part)
                    if not isinstance(child, dict):
                        child = node[part] = {}
                    node = child
                node[parts[-1]] = copy.deepcopy(value)
        return self.save()

    def as_dict(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def __getattr__(self, name: str):
        # Only reached for names that are not real attributes: top-level
        # config keys.  cfg.claude -> section view, cfg.units -> "us".
        if name.startswith("_") or name in ("path",):
            raise AttributeError(name)
        data = self.__dict__.get("_data")
        if data is None or name not in data:
            raise AttributeError(f"{type(self).__name__} has no attribute "
                                 f"or config key {name!r}")
        value = self.get(name)
        if isinstance(value, dict):
            return _Section(self, name)
        return value

    # -------------------------------------------------------- persistence
    def save(self) -> bool:
        """Atomic 0600 write of the whole config.  Never raises."""
        if self.path is None:
            return False
        # Serialize AND replace under the lock: two concurrent saves must
        # land in snapshot order, or an older snapshot could win the race.
        with self._lock:
            text = json.dumps(self._data, indent=2, ensure_ascii=False) + "\n"
            try:
                _write_private(self.path, text)
            except OSError:
                log.exception("assistant config save failed: %s", self.path)
                return False
            self._stamp = _file_stamp(self.path)
        return True

    def reload_if_changed(self) -> bool:
        """Re-read the file when its mtime/size moved (the user edited it
        by hand).  A corrupt edit keeps the in-memory copy and logs.
        Returns True when the data was reloaded."""
        if self.path is None:
            return False
        stamp = _file_stamp(self.path)
        if stamp == self._stamp:
            return False
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("top level is not an object")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            log.warning("assistant config changed on disk but is unreadable "
                        "(%s); keeping the loaded copy", exc)
            self._stamp = stamp
            return False
        with self._lock:
            self._data = _deep_merge(DEFAULTS, loaded)
        self._stamp = stamp
        log.info("assistant config reloaded from %s", self.path)
        return True

    # ------------------------------------------------------ configured?
    def is_configured(self, section: str) -> bool:
        """True when the section has what its tool needs (placeholders and
        empty values do not count).  Unknown sections are False."""
        if section == "home_location":
            lat, lon = self.get("home_location.lat"), self.get("home_location.lon")
            return _is_number(lat) and _is_number(lon)
        if section == "google_ical":
            urls = self.get("google_ical_urls") or []
            return any(isinstance(u, str) and u.strip().lower().startswith(
                ("http://", "https://", "webcal://")) and not _is_placeholder(u)
                for u in urls)
        if section == "icloud":
            return not _is_placeholder(self.get("icloud.apple_id")) and \
                not _is_placeholder(self.get("icloud.app_password"))
        if section == "gmail":
            # Multi-account configs carry gmail.accounts and no top-level
            # pair; without this the setup line kept telling the user to
            # configure mail that was already working.
            for entry in (self.get("gmail.accounts") or []):
                if isinstance(entry, dict) and \
                        not _is_placeholder(entry.get("address")) and \
                        not _is_placeholder(entry.get("app_password")):
                    return True
            return not _is_placeholder(self.get("gmail.address")) and \
                not _is_placeholder(self.get("gmail.app_password"))
        if section == "discord":
            return not _is_placeholder(self.get("discord.bot_token")) and \
                not _is_placeholder(self.get("discord.channel_id"))
        if section == "claude":
            return bool(_claude_bin()) and bool(self.allowed_dirs)
        return False

    def missing_sections(self) -> list[str]:
        return [s for s in SECTIONS if not self.is_configured(s)]

    @staticmethod
    def setup_line(section: str) -> str:
        """The spoken film-JARVIS excuse for a section that is not set up."""
        line = SETUP_LINES.get(section)
        if line:
            return line
        what = section.replace("_", " ").strip() or "that"
        return f"I'll need {what} set up, sir; the notes are in {DOCS_HINT}."

    @staticmethod
    def setup_lines() -> list[str]:
        """Every fixed excuse, for the speech-cache prewarm."""
        return list(SETUP_LINES.values())

    # ------------------------------------------------------------ secrets
    def redacted(self) -> dict:
        """A deep copy with every SECRET_KEYS value masked as "•••" (an
        empty placeholder stays "" so a settings view can show it is unset)."""
        data = self.as_dict()
        for dotted in SECRET_KEYS:
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and parts[-1] in node:
                value = node[parts[-1]]
                if not _is_placeholder(value):
                    node[parts[-1]] = MASK
        for list_key, field in SECRET_LIST_FIELDS:
            node = data
            for part in list_key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            for entry in (node if isinstance(node, list) else []):
                if isinstance(entry, dict) and not _is_placeholder(entry.get(field)):
                    entry[field] = MASK
        return data

    def secret_values(self) -> list[str]:
        values = [v for v in (self.get(k) for k in SECRET_KEYS)
                  if isinstance(v, str) and not _is_placeholder(v)]
        for list_key, field in SECRET_LIST_FIELDS:
            for entry in (self.get(list_key) or []):
                v = entry.get(field) if isinstance(entry, dict) else None
                if isinstance(v, str) and not _is_placeholder(v):
                    values.append(v)
        return values

    def scrub(self, text: Any) -> str:
        """Replace every configured secret value inside ``text`` with "•••"
        (for log lines, subprocess argv echoes, error messages)."""
        text = "" if text is None else str(text)
        for value in sorted(self.secret_values(), key=len, reverse=True):
            text = text.replace(value, MASK)
        return text

    def __repr__(self) -> str:
        return f"AssistantConfig(path={str(self.path)!r}, {self.redacted()!r})"

    __str__ = __repr__

    # --------------------------------------------------------- properties
    @property
    def home_location(self) -> Optional[dict]:
        """``{"city","region","lat","lon"}`` or None until lat/lon are set
        (then the location tool falls back to the IP lookup)."""
        if not self.is_configured("home_location"):
            return None
        loc = self.get("home_location") or {}
        return {"city": loc.get("city", "") or "", "region": loc.get("region", "") or "",
                "lat": float(loc["lat"]), "lon": float(loc["lon"])}

    @property
    def user_name(self) -> str:
        return str(self.get("user.name") or "Hunter")

    @property
    def units(self) -> str:
        return str(self.get("units") or "us")

    @property
    def local_model(self) -> str:
        return str(self.get("local_model") or DEFAULTS["local_model"])

    @property
    def allowed_dirs(self) -> list[str]:
        dirs = self.get("claude.allowed_dirs") or []
        return [os.path.abspath(os.path.expanduser(str(d)))
                for d in dirs if isinstance(d, str) and d.strip()]

    @property
    def projects_root(self) -> str:
        root = self.get("claude.projects_root") or DEFAULTS["claude"]["projects_root"]
        return os.path.abspath(os.path.expanduser(str(root)))

    @property
    def skill_phrases(self) -> dict:
        phrases = self.get("claude.skill_phrases") or {}
        return {str(k): str(v) for k, v in phrases.items()}

    # ------------------------------------------------------------ paths
    def is_allowed_path(self, path: os.PathLike | str,
                        base: Optional[os.PathLike | str] = None) -> bool:
        """True when the real path (symlinks and ``..`` resolved) is one of
        the allowed dirs or under one.  A relative ``path`` is taken against
        ``base`` (the project cwd) when given, else the process cwd."""
        try:
            raw = os.path.expanduser(str(path))
            if base is not None and not os.path.isabs(raw):
                raw = os.path.join(os.path.expanduser(str(base)), raw)
            real = os.path.realpath(raw)
        except (OSError, ValueError):
            return False
        for allowed in self.allowed_dirs:
            root = os.path.realpath(allowed)
            if real == root or real.startswith(root.rstrip(os.sep) + os.sep):
                return True
        return False

    def add_allowed_dir(self, path: os.PathLike | str) -> bool:
        """Add a directory (absolute, ``~`` expanded) and save.  Returns
        True when it was new."""
        new = os.path.abspath(os.path.expanduser(str(path)))
        with self._lock:
            current = [os.path.abspath(os.path.expanduser(str(d)))
                       for d in (self.get("claude.allowed_dirs") or [])]
            real_new = os.path.realpath(new)
            if any(os.path.realpath(d) == real_new for d in current):
                return False
            current.append(new)
        self.set("claude.allowed_dirs", current)
        return True

    def remove_allowed_dir(self, path: os.PathLike | str) -> bool:
        target = os.path.realpath(os.path.expanduser(str(path)))
        current = self.get("claude.allowed_dirs") or []
        kept = [d for d in current
                if os.path.realpath(os.path.expanduser(str(d))) != target]
        if len(kept) == len(current):
            return False
        self.set("claude.allowed_dirs", kept)
        return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == value      # not NaN
