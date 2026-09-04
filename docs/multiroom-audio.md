# Multi-room audio: three rooms, all local

The office holds the Spark. Two more rooms get satellites. Nothing leaves the
LAN.

This document is the *audio* half of the three-room build. The sensing half
(offline mode, the curfew, the radar) is `docs/offline-mode.md` and
`jarvis/sensing.py`, and nothing here touches it.

---

## 0. What was measured on this box, 2026-09-02

Everything in the recommendation below rests on these, so they are first.

| Fact | Value | How |
| --- | --- | --- |
| PipeWire / pactl | 1.0.5 / 16.1 | `pipewire --version`, `pactl --version` |
| Wi-Fi | 5 GHz, EHT (Wi-Fi 7), −29 dBm, rx 1361 / tx 1441 Mbit/s | `iw dev wlP9s9 link` |
| LAN round trip to the gateway | min 1.17 / avg 2.66 / **max 5.53** ms, mdev 1.25, 0 % loss over 20 | `ping -c 20 -i 0.2 192.168.50.1` |
| Sinks today | `bluez_output.F4_4E_FC_95_BA_CB.1` (default, RUNNING) and `alsa_output...hdmi-stereo` (SUSPENDED) | `pactl list short sinks` |
| Network audio modules present | `rtp-sink`, `rtp-source`, `rtp-session`, `rtp-sap`, `pulse-tunnel`, `raop-sink`, `raop-discover`, `roc-sink`, `roc-source`, `combine-stream`, `zeroconf-discover`, `vban-*`, `netjack2-*` | `ls /usr/lib/aarch64-linux-gnu/pipewire-0.3/` |
| `rtp-sink` packetisation | `sess.min-ptime` 2 ms, `sess.max-ptime` 20 ms, MTU 1280, multicast 224.0.0.56, TTL 1 | `man 7 libpipewire-module-rtp-sink` on this box |
| `pulse-tunnel` latency | `pulse.latency` default **200 ms** | `man 7 libpipewire-module-pulse-tunnel` on this box |
| `combine-stream` | exists, has `combine.latency-compensate` (delay buffers only — no clock recovery) | `man 7 libpipewire-module-combine-stream` |
| avahi | **active**, `avahi-browse` present | `systemctl is-active avahi-daemon` |
| Snapcast without sudo | `apt-get download snapserver snapclient` succeeds (848 kB / 304 kB); `dpkg-deb -x` yields binaries with **zero** unresolved shared libraries | `ldd .../snapserver \| grep -c "not found"` → 0 |
| AirPlay / Cast receivers on the LAN | **none** (`_raop._tcp`, `_googlecast._tcp` both empty) | `avahi-browse -t -p` |
| What *is* on the LAN | `Spark` and `SpZc-D614E3` (`_spotify-connect`), `HPComputer`, three `_matter._tcp` devices, a Nanoleaf, the router advertising `_alexa._tcp` | `avahi-browse -at -p` |
| Live turn latency (the number to protect) | `wait` **1.33 – 2.56 s**, best 1.33 s | `grep "turn:" /tmp/vss_voice/jarvis.log` |

Two incidental findings worth a sentence:

* **`192.168.50.60` is already taken.** That is the static IP the shipped
  `scripts/esphome/jarvis-room-sensor.yaml` hands out as its example. There is
  a stale ARP entry for it (`4c:3b:df:19:ba:34`) and it does not answer today,
  but the address is not free by default. Pick per-room addresses deliberately.
* **The default sink is the SoundCore 2 over SBC** (`s16le 2ch 48000`). SBC
  adds roughly 100–200 ms of codec-plus-link delay that no transport in this
  document can remove, and it is the device whose battery died on 2026-08-30.
  It is not the model for the other rooms: **wire the satellites.**

---

## 1. Speakers

### 1.1 The candidates, judged on this box

**PipeWire RTP** (`module-rtp-sink` / `-source` / `-session`). Present.
Lowest latency available: packets every 2–20 ms, and the receiver's
`sess.latency.msec` defaults to 100. Sessions announce themselves over
mDNS/SAP and avahi is already running, so discovery is free. What it does not
have is inter-room clock discipline — two receivers on two crystals drift, and
PipeWire's answer (`combine.latency-compensate`) is a fixed delay buffer, not
the sample add/drop that keeps rooms locked over an hour.

**`module-pulse-tunnel`.** Present, and the simplest thing that works: one
virtual sink per room forwarding to `pipewire-pulse` on the satellite. Costs
**200 ms** by default, and it wants a full Linux audio stack in every room.

**Snapcast.** Not installed and `apt install` needs a root he does not have —
but that objection is **measured false**: the arm64 `.deb`s download as a
normal user and unpack into `~/.local` with every shared library already
present on this box. Snapcast is what synchronised multi-room actually means:
continuous client↔server time sync with drift corrected by adding and dropping
individual samples, typical deviation reported below 0.2 ms. The price is
buffer: the default is 1000 ms server *and* 1000 ms client (≈500 ms
end-to-end), the practical multi-room floor is ~100 ms, and the theoretical one
(~50 ms, `--buffer 20`) drops out on Wi-Fi. **The client buffer dominates; the
network does not.**

**RAOP / AirPlay** (`module-raop-sink`, `raop-discover`). Present, and there is
nothing to talk to: no `_raop._tcp` on his LAN today. AirPlay buffers on the
order of two seconds by design. A `shairport-sync` receiver on a Pi is
genuinely local; most commercial AirPlay speakers are not, and a cloud-tethered
speaker fails his first constraint outright.

**The existing Spotify Connect leg** (librespot, advertising as `Spark`).
Cannot carry Jarvis's voice at all. Connect is an account-scoped control plane
for Spotify's own catalogue; there is no "play these bytes" verb, and reaching
it goes through Spotify's servers. It also is not idle — `jarvis/mixer.py`
already leans on it as the **remote ducker**, and that leg is the one that
actually fires (see §5). Leave it exactly where it is: it is the music leg and
the duck, and it is never the voice leg.

### 1.2 The recommendation: two planes, not one transport

> **Voice plane** — replies, alarms, reminders, earcons: **a small HTTP `/say`
> endpoint per room**, one POST of a rendered wav, and the room ACKs with what
> it actually played.
>
> **Music plane** — Spotify and any "play it everywhere": **Snapcast**, later,
> if he ever wants the same song in two rooms. Never in the reply path.

The argument is one number. His measured `wait` — wake word to first audio — is
**1.33 s at its best**, and pulling it there was a fortnight of work. Every
network transport in §1.1 adds a *fixed* buffer to *every reply*:

| Transport | Added to every reply | As a share of his best turn |
| --- | ---: | ---: |
| HTTP push to a room endpoint | one LAN RTT (**≤5.5 ms** measured) + the satellite's own ALSA buffer (~30–60 ms) ≈ **<70 ms** | +5 % |
| PipeWire RTP | 100 ms receiver latency | +7.5 % |
| `pulse-tunnel` | 200 ms | +15 % |
| Snapcast, tuned hard | 100 ms | +7.5 % |
| Snapcast, out of the box | ~500 ms | **+37 %** |

And the deciding asymmetry, which is not about latency at all:

**A reply plays in exactly one room.** Sync between rooms is a music problem —
it only matters when two speakers playing the same thing are audible from one
spot. Paying a synchronisation tax on speech buys nothing and costs the number
he cares most about.

The second reason is §3: **only the HTTP plane returns a receipt.** RTP is fire
and forget into a multicast group. A tunnel sink reports that a *sink* exists.
Neither can tell you a sound came out of a speaker in a room — which is exactly
the thing that was not known for hours on 2026-08-30.

The third reason is that it is already written. `tts.Rendition` exists and its
docstring says what it is for: *"One reply, rendered as wav bytes for a client
that is not the room."* It shares the speech cache with the room's own voice,
so a canned line is a file read. `webapp.py`'s `POST /api/say` already streams
one to a phone chunk-by-chunk as each sentence lands. **A room satellite is a
headless phone client.** No new protocol, no new dependency, `urllib` and
`paplay` — the two things `roomsensor.py` and `tts.py` already use.

### 1.3 The room endpoint

Two shapes are possible. Recommend (a).

**(a) A Linux satellite (Pi).** ~120 lines: an `http.server` with

```
POST /say      body = wav        -> plays it, returns {"ok":true,"played_ms":3120,"peak":0.31}
POST /stop                       -> cuts what is playing (barge-in)
POST /duck     {"pct":30}        -> ducks local streams there
POST /unduck
GET  /health                     -> {"sink":"...","ok":true,"speaking":false}
```

Same bearer token as `phone.token`, same private-address bind rule as
`webapp.py`. Chunked request body so the satellite starts playing sentence one
while sentence two is still on the GPU — the mirror image of what `/api/say`
does toward the phone today.

**(b) An ESP32-S3 + I2S amp** running ESPHome's `speaker` / `media_player`,
handed a URL to fetch. Cheaper and smaller, but there is no `played_ms`
receipt without custom firmware, and no mic path later without the API problem
in §4.2. It is a fine *speaker-only* room; it is a dead end for a full one.

### 1.4 Config, and the day-one no-op

The audio side does **not** get a config block of its own. It reads
`presence.rooms` — THE room list, the one `jarvis/roomfabric.py` (which room
he is in) and `jarvis/rooms.py` (the satellite lease) read — one list, so a
room cannot exist for the radar and not for the voice, and a room name cannot
be spelled two ways (every lane spells it with `roomfabric.room_name`; until
2026-09-04 this lane read a `rooms.satellites` list declared nowhere, F02).
Two audio-only keys per entry:

```json
"presence": {
  "rooms": [
    {"name": "office", "url": "http://192.168.50.51", "primary": true},
    {"name": "kitchen", "url": "http://192.168.50.61",
     "say_url": "http://192.168.50.61:8765", "sensors": ["radar"]},
    {"name": "bedroom", "url": "http://192.168.50.62",
     "say_url": "", "private": true, "sensors": ["radar"]}
  ]
}
```

`url` is the ESPHome satellite (the radar, its lease, its power confirmation).
`say_url` is the audio endpoint, and it is **separate on purpose**: the radar
may be a $6 ESP32 on port 80 and the speaker a Pi on 8765, in the same room.
An empty `say_url` is a room that can be *seen* in but not *spoken* in — a
real intermediate state, and announcements for it fall back to `here`.

`here` is the Spark's room: the entry marked `primary`, else the first
(the fabric's own rule), and "office" when nothing is configured. It always
exists as a room, it needs no `say_url`, and it is the existing `paplay` path
untouched. **With no `say_url` anywhere, every line falls through to `here`
and the router is a no-op** — which is what makes this safe to merge before
any hardware exists.

The URL rule is not a second one. `roomaudio.say_url()` is built on
`rooms.check_url()`: `http(s)://` plus a **private IP literal**, no hostnames
(an mDNS answer is one poisoned packet away from pointing his voice at somebody
else's box), no public addresses, redirects refused. The `/say` endpoint
carries a bearer token, so a laxer rule on the audio path would be the hole.

The seam in the app is one argument. `JarvisApp._say(text, proactive, kind)` is
already documented as *"the one door to TTS"*, and `tts.speak(text, block)` is
the only thing behind it. Adding `room=` to both, defaulting to `here`, and
carrying it to the `dev = (CONFIG.playback_device or "").strip()` line in
`_play` / `_play_stream` is the whole plumbing change. Nothing else in the app
needs to know rooms exist.

### 1.5 How the two halves compose

| Owned by `jarvis/rooms.py` (sensing) | Owned by `jarvis/roomaudio.py` (voice) |
| --- | --- |
| `RoomSpec`, `Satellite`, `RoomMesh`, the lease and the power confirmation | `Room`, `VoiceRouter`, `RoomSpeaker`, the receipt |
| `check_url` — the one definition of an address we will talk to | `say_url` — the same rule, plus a default path |
| `Satellite.read()` → is someone in THIS room | `occupancy_from(mesh)` → `{room: True/False/None}` |
| `HouseView`, `spoken_status` — what is *sensing* | `Decision` — what is *speaking* |

`occupancy_from` is the only function in the audio half that knows the mesh's
shape, and it is deliberately defensive: a router that raised because an
attribute was renamed would take his voice down with it.

The snapshot is taken on the **mesh's** thread and handed to
`VoiceRouter.observe()`. `route()` then runs inside a spoken turn without a
single LAN round trip — `soundbar.status_line`'s rule, for `aside.py`'s reason.

**Offline mode reaches routing for free and correctly.** When sensing is denied,
`Satellite.read()` returns `None` (it does not poll — the `reads` counter is
that lane's assertion), so every room reports *no opinion*, and §2.2's rule
sends announcements to `here`. Offline mode therefore stops Jarvis *following*
him without stopping him *speaking* — which is what a privacy switch should do,
and it required no coordination between the two lanes to get right.

---

## 2. Which room answers

### 2.1 The rule, in one sentence

**An answer goes back where the question came from; an announcement goes where
he is; when nobody knows where he is, it goes to `here` — and if he is out, it
is HELD, never broadcast.**

Three classes of speech, and they route differently:

| Class | Target | Why |
| --- | --- | --- |
| **Answer** — a reply to something he just said | the room that asked, always | he asked two seconds ago and he is still standing there. Never re-route an answer on a sensor reading. |
| **Announcement** — proactive: timer, reminder, heads-up, watchdog | the occupied room | the person is the subject, not the room. |
| **Alarm** | every room | an alarm's entire purpose is to be heard from wherever he is. `tools/timekeeper.py` already marks it non-proactive so quiet hours cannot hold it; broadcasting is the same instinct. |

`source_room` is nearly free: `app.py` already tracks `_last_source`
(`voice` / `typed` / `cli` / `intercom` / `phone`). It gains a sibling
`_last_room`, set by whichever transport delivered the turn.

### 2.2 The awkward cases

| Case | What happens | Why |
| --- | --- | --- |
| **A timer set in the office fires while he is in the kitchen** | it speaks in the **kitchen** | he asked for a timer, not for the office. The timer belongs to the person. |
| The same timer, and the house reads empty | **held** by `quiet.should_hold()`, exactly as today (`quiet.hold_when_away`), and read back in the digest on his return | this is the existing rule and multi-room must not quietly break it into "shout in three rooms". |
| The same timer, and **no room has an opinion** (offline mode, all sensors down, radar unplugged) | it speaks in **`here`** — the office | broadcasting on ignorance is how a house shouts at 2 a.m. `roomsensor.read()` returning `None` means *no opinion*, never *empty*, and the routing policy has to honour that distinction the same way `presence.RoomOrPhone` does. |
| **Two rooms both occupied** | the one with the **most recent turn** inside `rooms.recency_s` (300 s); otherwise `here`. **Never both.** | two speakers saying the same sentence 40 ms apart is the comb filter that makes multi-room sound broken. Recency is the only cheap evidence of *which body is his*. |
| **A guest is present** | routing does not change; **content** does | the radar counts bodies, it cannot name them. The honest lever is `rooms.<name>.private`: a private room is never a broadcast target and never receives the held digest, mail or grades — only answers to questions asked in it. Face/body ID can later flip this per person; nothing here forecloses that. |
| **Offline mode** (radar off, mic live) | every room reports no opinion, so the policy degrades to "answer where asked, announce in `here`" | offline mode is a privacy switch. It must not silently relocate his assistant to a different room. |
| **The camera curfew** | nothing at all | routing never reads the camera. |
| A room's speaker is dead | it is demoted (§3) and the router falls back to `here`, with one spoken line | |

### 2.3 The one thing routing must not do

**Follow him mid-sentence.** A reply that starts in the office and finishes in
the kitchen is worse than either. The target is bound **once**, at `_say`, and
does not move for the duration of the utterance. If he walks out, he misses the
end — which is what happens with a human too.

---

## 3. Knowing the voice landed

`soundbar.py` exists because on 2026-08-30 the soundbar's battery died,
PipeWire moved the default sink to the HDMI monitor, and Jarvis talked into a
monitor behind the desk for hours. Its lesson generalises badly: with three
rooms there are three ways to talk to a wall.

**And `pactl` cannot answer the question for a remote room at all.** It sees a
*sink*, not a *speaker*. On the night in question the HDMI sink was perfectly
healthy — the sink was fine, the *room* was wrong. Scaling `pactl` to three
rooms scales a probe that already could not see the failure it was written for.

So every remote room owes a **receipt**:

1. **Per utterance.** `POST /say` returns `{"ok": true, "played_ms": 3120,
   "peak": 0.31}`. `played_ms` materially short of the wav's duration means it
   was cut off. `peak` near zero means it played into a muted or dead output —
   the remote equivalent of the HDMI monitor, and the one signal `pactl` never
   had.
2. **The silence audit.** Every routed reply records `(room, expected_ms,
   played_ms)`. Two consecutive utterances under 50 % of expected marks the
   room dead. This is the exact check the original incident had no way to make:
   hours of speech with nothing coming out, and nothing in the system that
   compared *intended* audio against *emitted* audio.
3. **Per tick.** `GET /health` on the 30 s pass `soundbar.py` already runs,
   with `roomsensor.py`'s circuit-breaker discipline copied verbatim — three
   consecutive failures and the room is skipped entirely on a growing cooldown,
   so an unplugged kitchen costs the loop **zero syscalls per tick**, not a
   timeout apiece. That breaker is why "unplug it and nothing changes" is
   literally true of the radar, and it is why it will be true of a speaker.
4. **One line per transition, never per tick.** A demoted room earns one spoken
   sentence in `here` — *"The kitchen speaker isn't answering, sir"* — latched
   in a state file so a restart into a still-dead kitchen does not say it again.
   That is `soundbar.py`'s rule and `faults.py`'s rule; it is not re-litigated.
5. **The Spark's own room keeps `soundbar.py` unchanged.** It stops being *the*
   sink sentinel and becomes *one room's* health probe among N. Its
   `read()`/`status` shape already fits the per-room health record.

The asymmetry worth stating plainly: **the local room is the one we can never
get a receipt from.** `paplay` exiting 0 proves a process ran, not that a
speaker moved. That is precisely why `soundbar.py` had to be written, and it is
an argument *for* the HTTP plane rather than against it — the remote rooms will
be better instrumented than the office.

---

## 4. Microphones

### 4.1 Push-to-talk: already built, and it is his cheapest satellite

`jarvis/intercom.py` is **enabled today** (`intercom.enabled = True`,
`verify_speaker = False`, `max_mb = 10`). A clip arrives as
`{"audio_b64": ...}` or a raw body, is decoded to exactly what the recorder
produces — mono float32 at 16 kHz — and goes through the **resident Whisper and
the same speaker gate, verbatim**. `webapp.py` serves the client to any device
on the home Wi-Fi: token-authed, bound to one private address, no cloud leg, no
tunnel.

**His phone is the cheapest satellite mic he already owns, and an old phone or
a tablet on a charging dock in the kitchen is a working second one for $0.**
It exercises 100 % of the routing, receipt and ducking design in this document
before he spends anything.

What multi-room changes about it is **one field**:

* the clip carries `room=`, so the answer comes back *in that room* and
  `last_heard_at` updates for §2.2's tie-break;
* the page's **Aloud** switch changes meaning from "answer in the office" to
  "answer in the room I am in". Today the two switches are *Voice* (out of the
  phone) and *Aloud* (out of the Spark); with rooms, Aloud is a room choice.

Nothing else. `intercom.settings()` already exists so that every transport
reads the same three switches and cannot disagree about them; a satellite is a
fourth transport reading the same three.

### 4.2 Always-listening satellites: the honest assessment

The requirement is: wake word decided **on the device**, nothing streams until
it fires, all-local, and it must land in the existing `intercom` contract.

**Option 1 — ESP32-S3 + ESPHome `micro_wake_word`. Best sound, worst fit.**

The remarkable fact is that micro_wake_word ships a pre-trained **`hey_jarvis`**
model — his exact wake word, already trained, free, running entirely in
TensorFlow Lite Micro on the device and reported faster than an ESP32 streaming
to openWakeWord. It needs an **ESP32-S3 with PSRAM** (a plain ESP32 "may run but
not as well").

The catch is load-bearing and this repo already knows about it. His own
`scripts/esphome/jarvis-room-sensor.yaml` carries the comment: *"THERE IS
DELIBERATELY NO `api:` BLOCK… with `api:` enabled and no client ever connecting,
ESPHome REBOOTS THE DEVICE EVERY 15 MINUTES by design."* ESPHome's
`voice_assistant` component runs **over that same native API**, so a mic
satellite must have `api:` and must have a permanently connected client. That
client is either Home Assistant — a second always-on service, and on this box a
GPU neighbour — or Jarvis itself learning the ESPHome protobuf/noise protocol
through `aioesphomeapi`, a dependency this stdlib-disciplined codebase has no
other reason to carry.

*Verdict: park it.* The wake word is better and the integration is a project.

**Option 2 — a small Linux satellite running openWakeWord and POSTing to the
intercom. Recommended.**

Jarvis already runs openwakeword 0.4.0 with **his own trained positives on
disk** (`~/.aiws_trainer/wakeword_training/positive/*.wav`). The same model runs
on a Pi 4/5 CPU. The satellite's whole job:

```
oww fires  ->  record until VAD says done  ->  POST /api/voice?room=kitchen
```

That is the intercom's existing contract. **Zero new server protocol.** The clip
goes through the same resident Whisper and the same gate as a clip from his
phone. And it is the same box that serves `/say`, so one satellite is both ends
of the room.

Two warnings, both current:

* **`rhasspy/wyoming-satellite` was archived by its owner on 2026-01-27.** Do
  not build on it. Its Pi Zero 2 W recipe was also reported unreliable —
  devices unresponsive after 4–6 hours, openWakeWord suspected. Use a Pi 4 or
  Pi 5 for a *mic* satellite (a Zero 2 W is fine for speaker-only, where the
  job is playing a wav).
* Wyoming is in any case a detour: it exists to talk to Home Assistant, and
  `POST /api/voice` is already the protocol.

**Option 3 — Home Assistant Voice Preview Edition (~$59).** ESP32-S3 plus an
**XMOS XU316** DSP doing echo cancellation, stationary noise suppression, AGC
and beamforming, with a two-mic array, a hardware mute switch and a speaker in
one box. The best far-field hardware at the price by a distance. It is Option 1's
integration problem in a nicer enclosure.

### 4.3 The measured wake-word problem, transposed to a satellite

This is where the honest answer is uncomfortable, and it splits in two.

**The bad half.** `scratchpad/wake-gate/wake_gate_diagnosis.md` measured it:
his wake word is **0.52–0.68 s** of trimmed speech, `speaker.score()` returns
`None` below 1.0 s, and the gate therefore **abstained on 52.9 % of live wakes**
(9 of 17 in one boot; 50 % reproduced offline). Worse, under competing audio his
own wake word scores **0.012–0.111** against a room-noise impostor band reaching
**0.138** — *no threshold separates owner from stranger there.* A satellite mic
is further from his mouth and in a noisier room. **On the raw wake buffer a
satellite will be strictly worse, and there is no threshold that rescues it.**

**The good half, which is structural rather than hopeful.** On a satellite the
speaker gate is not being asked to judge the wake word at all:

1. **The wake decision is acoustic, not speaker-verified.** openWakeWord and
   micro_wake_word are trained on the phrase; they do not need 1.0 s of speech
   and do not consult the voiceprint. Distance and noise degrade them, and the
   remedies are the ordinary ones — a per-room threshold, and (with the XMOS
   box) beamforming plus AEC in hardware.
2. **The clip Jarvis gates is the command, not the wake word.** Live `speech`
   durations in his own turn ledger are **1.0–6.9 s**, and the gate's measured
   curve on his voice is 0.43 at 2 s, 0.52 at 2.5 s, 0.57 at 3 s, **0.68 at
   6 s**. That is the regime where ECAPA works. **The satellite path is a
   better place for the speaker gate than the local wake gate has ever been.**
3. **The transport hands the gate the fix it was asking for.** R1/R2 of the
   diagnosis want an isolated word instead of a two-second ring buffer. A
   satellite has *already* decided where speech started and stopped; it sends an
   endpointed clip with ~200 ms of lead-in, so `trim_silence` receives a clean
   utterance by construction rather than by parameter.

**But do not switch the gate on for satellites yet.** The existing voiceprint is
six **near-field desk** samples, and it is stale in a way the code already
warns about (`format 1` against `VOICEPRINT_FORMAT = 2`, worth ~0.04 of margin).
Far-field moves the embedding further still. `intercom.verify_speaker = False`
already says the honest thing out loud — *"a phone microphone and a lossy codec
move the ECAPA embedding far enough that the transcript gate… would reject his
own voice"* — and a kitchen at four metres is the same argument, louder.

So, concretely:

* satellite clips are **scored and logged, not gated**, at first;
* the trust boundary stays what it is for the phone: the LAN, plus the bearer
  token, plus a device he owns;
* the way out is **per-room enrolment** — six lines spoken from where he
  actually stands in that room — and a per-room `rooms.<name>.speaker_min`
  threshold, because 0.25 was calibrated at a desk;
* only then is `verify_speaker` worth turning on per room.

---

## 5. Ducking and barge-in across rooms

### 5.1 The bug the naive design would have

`jarvis/mixer.py` ducks with `pactl set-sink-input-volume` — per stream, **on
this box**. A remote room's music is not a sink-input here. And the module
already carries the scar: the librespot pipe is an always-open, uncorked,
*silent* sink-input, so on 2026-09-01 pactl "always found exactly one stream,
ducked it — theatre — and the remote duck never fired once", while the music he
could actually hear was coming out of HPCOMPUTER. Multi-room makes that the
normal case rather than the exception.

### 5.2 The rule, which inverts the single-room one

> **Duck the room that is about to be spoken in, and only that room.**

Music in the kitchen has no business dropping because he asked a question in the
office. In a one-room house "the room" and "the box" are the same thing, so the
current rule cannot tell the difference; this is the one place the existing
mental model genuinely has to change.

Three ducking domains, and they must not be confused:

1. **Local sinks** — `mixer.py`, unchanged. Per stream, never the sink,
   Jarvis's own players exempt **by PID** (librespot and `paplay` both present
   as `pacat`, so only the PID tells them apart).
2. **The remote Connect device** — `spotify.duck()` / `unduck()`, unchanged.
   This is the leg that actually fires.
3. **Room endpoints** — `POST /duck` and `/unduck`, with the two rules
   `mixer.py` learned the hard way carried across verbatim: the `HOLD_MAX_S`
   backstop (no hold may hold a room down indefinitely because its publisher
   stopped talking to us) and the **state-file heal** (a duck that fails to
   *restore* stays on the books and is retried, because a room left at 30 % by a
   crash is a bug he will hear tomorrow).

### 5.3 Barge-in

* Barge-in is per-room. **The room that hears him is the room that gets cut.**
  `interrupt_speech()` becomes `POST /stop` to that room; a read-aloud in
  another room is not touched. Cutting all three rooms because he spoke in one
  is the multi-room version of talking over him.
* **Music in one room, speech in another: nothing ducks and nothing is
  interrupted, and that is correct.** The only cross-room interaction is two
  rooms audible from one spot, which is a placement problem, not a software one.
* **The self-wake hazard is the real one.** A satellite's speaker feeds its own
  mic; without AEC it will wake itself on Jarvis's voice. There is measured
  protection — Jarvis's own TTS scores −0.155…0.122 against the voiceprint
  (n=66), comfortably under the 0.25 bar, so the gate rejects it — *but only
  when the gate runs*, and §4.3 says it abstains half the time. So the
  belt-and-braces rule is the one the local path already uses: **a satellite
  must not stream while it is playing.** `/health` carries `speaking`, and the
  satellite's own wake detector is muted for the duration of a `/say`. Hardware
  AEC (the XMOS box) makes this a comfort rather than a necessity.
* `hotword.py` already relaxes its speaker threshold while `music_playing()` is
  true, because a wake word over a vocalist scores like a stranger. That cache
  read becomes **per room**: music playing in the kitchen must not relax the
  bar for a wake word spoken in the office.

---

## 6. Staging — what to build, in order

| # | Cost | What | What it proves |
| --- | --- | --- | --- |
| 0 | $0 | `presence.rooms` with **one** room, `office`, `say_url: ""` | the router is a no-op; nothing about today's behaviour changed |
| 1 | $0 | the old phone on a charger in the kitchen, web client open | **the only question that matters**: does he want an answer in the kitchen, or does he want to walk to the office? |
| 2 | ~$40 | one Pi satellite in the kitchen: `/say`, `/stop`, `/health` — **no mic** | routing, the receipt path, demotion, per-room duck |
| 3 | ~$10 | an LD2410 for that room (the existing YAML, a different IP) | the sensor fabric's room answer feeding real routing |
| 4 | ~$25 | openWakeWord on the same Pi → `POST /api/voice?room=kitchen` | always-listening, with the gate logging not gating |
| 5 | ~$0 | Snapcast from `~/.local`, music only | synchronised music, never in the reply path |

Step 1 is not a formality. It is the cheapest possible answer to the question
this whole document is downstream of, and it costs nothing to run for a week.

---

## 7. What this design deliberately does not do

* It does not touch `jarvis/roomsensor.py`, `jarvis/sensing.py`,
  `jarvis/presence.py`, `jarvis/mixer.py` or `jarvis/rooms.py`. The
  offline-mode and sensor-fabric lanes own those, and this design consumes
  their answers rather than reimplementing them. `jarvis/roomaudio.py` and
  `tests/test_roomaudio.py` are new files and touch nothing else.
* It does not add a dependency. The voice plane is `urllib` plus `paplay`.
* It does not foreclose face and body recognition or gestures: the room record
  has room for `who`, and the routing policy takes an occupancy answer it does
  not compute. When a room can name the body in it, `private` becomes per-person
  and the recency tie-break in §2.2 becomes unnecessary rather than wrong.

---

## 8. Hardware, per room, at two price points

**Buy nothing first.** Step 1 of §6 is not a formality: an old phone or a
tablet on a charger, page open, is a working satellite for **$0**, and it
exercises the routing, the receipt and the ducking design end to end. It is
push-to-talk rather than always-listening, and that is the *only* thing it
cannot answer.

### Speakers

| Tier | Parts | ~Cost | What it buys |
| --- | --- | ---: | --- |
| **free** | his old phone / a tablet / the HP laptop, `webapp.py` page open | $0 | routing, `/say`, ducking — everything but hands-free |
| **budget** | Pi Zero 2 W ($15) + MAX98357A I2S amp ($5) + a 3 W driver ($5) + PSU/case ($15) | **~$40** | a real `/say` endpoint with a `played_ms` receipt, `/health`, per-room duck, and it is the same box the mic goes on later. Playing a wav is trivial work — the Zero 2 W's reliability problem is openWakeWord, not playback |
| budget, no Pi | ESP32-S3 + MAX98357A + driver | ~$18 | plays a pushed URL via ESPHome `media_player`. **No receipt** and no mic path later. A dead end for a full room |
| **better** | Pi 4 (2 GB, ~$45) or Pi 5 + HiFiBerry DAC ($25–35) + powered speaker ($40+) | **~$110–130** | fidelity, headroom for the mic tier on the same box, and a Snapcast client if he ever wants synchronised music |

### Microphones

| Tier | Parts | ~Cost | What it buys |
| --- | --- | ---: | --- |
| **free** | the phone, `POST /api/voice` | $0 | push-to-talk that already reuses Whisper and the speaker gate verbatim |
| **budget** | a USB conference mic or a cheap USB array on the Pi | ~$12–25 | near-field hands-free. It will be **worse than the Snowball at range** — no beamforming, no AEC |
| budget+ | ReSpeaker 2-Mic Pi HAT | ~$25 | two mics and, more usefully, a real AEC reference channel |
| **better** | Home Assistant Voice Preview Edition (ESP32-S3 + **XMOS XU316**) | **~$59** | hardware AEC, stationary noise suppression, AGC, beamforming, mute switch, speaker — the best far-field hardware at the price. Costs the ESPHome-API integration problem of §4.2 |
| better, keeps the architecture | ReSpeaker Lite / 4-mic array on a Pi 4 | ~$70–90 | a real array while staying an HTTP satellite |

### Room by room

* **Office (the Spark).** Nothing to buy. It is `here` (the primary room), `say_url: ""`,
  and `soundbar.py` keeps watching it. One caveat that shapes the other rooms:
  the SoundCore 2 is measured today as the default sink
  (`bluez_output.F4_4E_FC_95_BA_CB.1`) and is **SBC-only Bluetooth**, which
  adds ~100–200 ms no transport in this document can remove and which is the
  device that died on 2026-08-30. **Wire the satellites; do not Bluetooth
  them.**
* **Kitchen.** The room most likely to want hands-free, and the noisiest. If he
  buys exactly one XMOS box, buy it here. Otherwise: Pi + I2S amp now, mic
  later.
* **Bedroom.** Speaker-first, `private: true`, and a low volume ceiling. The
  intercom's own docstring already names the use case: *"from bed, or from the
  next room, the wake word is out of range and the soundbar's answer would wake
  the house."* A bedroom is the room where **content** routing (§2.2) matters
  more than audio quality.

### The bill for the recommended path

| | |
| --- | ---: |
| Step 1 — phone in the kitchen | **$0** |
| Step 2 — kitchen speaker satellite | ~$40 |
| Step 3 — kitchen LD2410 (he owns the modules) | ~$6 for a spare ESP32 |
| Step 4 — kitchen mic (USB array on the same Pi) | ~$25 |
| Bedroom, same shape | ~$45 |
| **Three rooms, hands-free in one** | **~$120** |

Snapcast, if he ever wants synchronised music, is **$0** — measured, the arm64
`.deb`s unpack into `~/.local` with every shared library already on this box.
