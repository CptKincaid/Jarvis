# Three rooms — one plan

Reconciliation pass over the three multi-room design lanes, 2026-09-02. Read-only except this
file: nothing under `~/Jarvis` was touched, the running Jarvis (untouched all session) was never
signalled, no mic or radio was opened. Every code claim is cited `file:line` against the working
tree at `8b9c914` on `multiroom-design`.

The three lane reports are `docs/room-fabric.md`, `docs/multiroom-audio.md`,
`docs/multiroom-privacy.md`. They are all worth reading. This file is the part none of them
could do: **what is the same work twice, what contradicts what, and what to do first.**

The lanes worked blind to each other and it shows. They shipped **three** modules that each
answer "is he in this room" (`jarvis/rooms.py:720`, `jarvis/roomfabric.py:481`,
`jarvis/roomaudio.py:207`), **two** config schemas for the same wall of sensors, **two** ESPHome
firmwares for the same board, and **two** fixes for the same privacy hole. All of that is
resolvable, none of it is wasted, and §1 does it.

---

## BEFORE COFFEE — the one screen

| # | Evening(s) | $ | Hardware? | What it buys |
|---|---|---|---|---|
| **0** | **tonight, 5 min** | **0** | **no** | **Prop your phone on the kitchen counter with `webapp.py` open and leave it a week.** `intercom.enabled` is already `True` (`assistant_config.py:509`). This answers the only question that changes the budget by 3x: do you want an answer in the kitchen, or do you walk to the office anyway? Everything below runs while it does. |
| 1 | 2 | 0 | no | **Merge the three lanes into one spine.** One config key, one fusion, delete the duplicates (§1). Nothing works right until this happens and it gets harder every week. |
| 2 | 1 | 0 | no | **Three-room bench on localhost.** `scripts/roomsensor_stub.py --port 8781/8782/8783` (`:183`), three URLs, `curl .../on` to walk yourself around. Proves the handoff, the coverage rule and the console with zero solder. |
| 3 | 1–2 | 0 | no | **The console room strip + privacy matrix.** UNKNOWN rendered as a first-class state, not as OFF. This is the part that makes offline mode *checkable*. |
| 4 | 1 + solder | **~$25** | **yes — one kit** | **Flash ONE satellite: ESP32 + LD2410C + MOSFET + supply-rail LED.** Then unplug the Spark and watch the LED go out on its own. That is fail-to-offline being *true* rather than promised. Do not buy three before one works. |
| 5 | 1 | 0 | that kit | **Measure crosstalk through the wall**, then set the four timers from the measurement instead of from my arithmetic. Every number in §4 is a hypothesis about *your* rooms. |
| 6 | 1 | ~$30 | 2 more kits | **The other two rooms.** Now `where()` means something. |
| 7 | 1 | 0 | no | **Announcements follow the room.** Timers, reminders and heads-up lines land where he is. The cheapest real win in the whole plan, and it needs no speaker. |
| 8 | 2 | ~$40 | Pi kit | **A `/say` speaker satellite with a receipt** — *only if step 0 said yes.* |
| 9 | 2 | ~$25 | USB mic | Hands-free in that room. Score the speaker gate, do not gate on it. |

**The money.** **$0** for steps 0–3, which is four evenings and is where all the risk is.
**$31–64** buys all three rooms sensing (you already own the LD2410Cs; §5 itemises it). **$0–120** for the
network posture, and it may well be $0 — *check your router before buying anything*. Audio is
**$0–65 and deferred behind a week of free evidence**. Realistic all-in without a new router:
**~$105**. Prices are estimates from a search that returned listings but not reliable figures —
**this is the thinnest evidence in the document.** Treat it as a budget shape.

**Cut outright** (§1.3): Snapcast, RTP, pulse-tunnel and AirPlay; the ESPHome native API and
`micro_wake_word`; the ARP household roster; per-room offline-mode voice grammar; per-room
curfew windows; the device-side SNTP curfew; `RoomMesh.read`/`mesh_probe`;
`roomaudio.occupancy_from`; the router's recency tie-break; the `presence.rooms` config key; and
one of the two firmwares. That is roughly a third of what the three lanes proposed.

**The first thing, tomorrow:** delete `"rooms": []` from `assistant_config.py:435` and make
`rooms.satellites` the one roster. Everything in §1 falls out of that single decision.

**Waiting on you (§7).** ① Does your router do VLANs or a second SSID? ② Will you solder the
MOSFET and the supply-rail LED — because without them "offline mode" means "Jarvis stopped
asking", **and that is what it means on your single sensor today**. ③ House-wide offline mode
only, in v1? ④ Will you actually run the phone week before buying speakers?

---

## 1. Reconciliation

### 1.1 The seven contradictions, resolved

**C1 — Two config schemas for one wall of sensors. RULING: `rooms.satellites`.**

The fabric lane shipped `presence.rooms` and it is **live in DEFAULTS today**
(`assistant_config.py:435`, timers at `:447-449`). The privacy and audio lanes both read
`rooms.satellites` / `rooms.enabled` / `rooms.here`, which is **not** in DEFAULTS, so both are
inert right now.

`rooms.satellites` wins on three counts. It carries what a room actually needs — transport
credentials, the lease TTL, the per-room sensor list (`jarvis/rooms.py:245-260`), plus the
audio lane's `say_url` and `private`. Two of three lanes already read it. And the altitude is
right: once a room has a radar, a speaker and later a microphone, it is not a detail of
`presence`.

What survives under `presence.*` is only the **fusion's four timers**
(`rooms_enter_hold_s`, `rooms_leave_hold_s`, `rooms_switch_min_s`, `rooms_stale_after_s`, plus
`rooms_poll_s` and `rooms_stuck_after_h`), because those are presence tuning and nothing else
reads them. Delete `presence.rooms`. Keep `presence.room_sensor_url` as the legacy singular
path so a config written before any of this keeps working (`roomfabric.py:162-172` already
falls back to it).

Two master switches would be the same hazard one level up, so: **`rooms.enabled` is the master
switch** over the whole three-room system; `presence.room_sensor_enabled` governs only the
legacy singular sensor. `docs/room-sensor.md` §10 stays true for the one-sensor install and
`docs/multiroom-privacy.md` owns the plural.

**C2 — Three implementations of "is he in this room". RULING: `roomfabric` fuses, `rooms`
transports, `roomaudio` consumes.**

- `jarvis/rooms.py:839 RoomMesh.read()` + `:891 mesh_probe()` — the N-room presence answer.
  Its own docstring calls itself "THE FALLBACK, not the recommendation". **Delete both**, and
  their tests. ~40 lines. Keeping a second, dumber copy of the house answer alive next to the
  real one is exactly the duplicate-state trap this codebase warns about elsewhere.
- `jarvis/roomaudio.py:207 occupancy_from(mesh)` walks `mesh.satellites` and calls `sat.read()`
  itself. That is not just duplication, it is **a live bug**: `Satellite.read()`
  (`rooms.py:720`) issues *two* HTTP GETs — `confirm(RADAR)` then `presence.read()` — so a
  second caller doubles the LAN traffic per room and races two `RoomSensor` circuit-breaker
  counters. It is the precise hazard the fabric lane found and solved with `HouseView`'s cache
  (`roomfabric.py:602-616`). **Delete `occupancy_from`.**
- **`jarvis/roomfabric.py` is the only fusion.** `from_readers()` (`:676`) exists for exactly
  this: hand it `[(spec, satellite), …]` and the satellite lane's transport feeds the fabric's
  hysteresis. Both lanes independently derived the *same* coverage rule (`anywhere()` is False
  only when every room answered — `roomfabric.py:487`, `rooms.py:839`), which is good
  corroboration that the rule is right and no reason at all to ship it twice.

**C3 — Two arbitrations for "which room gets the voice". RULING: the fabric arbitrates rooms;
the router arbitrates speech.**

The fabric resolves two-occupied-rooms by newest ON edge plus a `switch_min_s` floor
(`roomfabric.py` module head). The audio lane resolves the same case by "most recent turn
inside `recency_s`". Running both means his voice could move on the router's rule while
presence moves on the fabric's, and the router's raw-occupancy input has no doorway anti-flap
at all — it would re-target every 2 s in a doorway.

So `VoiceRouter` stops reading occupancy and reads `fabric.where()`, which is already
arbitrated and already carries `confidence`. The router's job shrinks to three lines:

- **answer** → the room the question came from, always, never re-routed on a sensor;
- **announce** → `where().room` when `confidence == "certain"`, else `here`;
- **alarm** → every room, private ones included.

**Delete the recency tie-break.** One mechanism removed, and the surviving rule is shorter to
say out loud, which is how you know it is the right one.

**C4 — Two fixes for the attach-by-name hole. RULING: `Satellite` owns the attach; `RoomPolicy`
is for the legacy path only.**

The hole is real and I re-read it: `SensingPolicy.attach` does
`self._devices = [d for d in self._devices if d.name != name]` then appends
(`jarvis/sensing.py:379`). Three radars attaching as `"radar"` collapse to one and offline mode
cuts only the last one's power. The fabric lane measured the collapse; the privacy lane hit the
same wall from the other side.

Both fixes work and they namespace identically — `"kitchen radar"`. `Satellite.__init__`
(`rooms.py:581-587`) attaches per kind and binds `revoke`/`renew` as the stop/resume, which is
strictly better than `RoomPolicy`'s binding of `RoomSensor.stop`, because revoke actually cuts
the rail. So: build the fabric with **`from_readers()`, never `build(cfg, policy)`**, when the
mesh exists — then no policy reaches the fabric and `RoomPolicy` (`roomfabric.py:214`) is only
exercised on the one-sensor legacy path, where it is still the right fix. Never both.

The naming survives the trip to speech unchanged: `_sensing_join`
(`jarvis/commander.py:5470-5477`) turns `("office radar", "kitchen radar")` into "the office
radar and the kitchen radar", which is a sentence. A colon would not have been.

**C5 — Two ESPHome firmwares. RULING: `jarvis-satellite.yaml`, everywhere, from day one.**

`scripts/esphome/jarvis-room-sensor.yaml` (182 lines) and `scripts/esphome/jarvis-satellite.yaml`
(354 lines) target the same board and the same wiring. The satellite yaml is a strict superset:
same `/binary_sensor/presence` entity, plus HTTP auth (`:151`), `ota: false` (`:157`),
`local: true`, no `captive_portal:`/`ap:` fallback (`:126`), an `internal: true` power rail
(`:192`) with `restore_mode: ALWAYS_OFF` (`:195`), the lease script (`:197`) and the renew/revoke
buttons (`:211-235`).

This is not a preference, it is a **hard ordering dependency people will trip over**:
`Satellite.read()` returns `None` unless `confirm(RADAR)` came back True (`rooms.py:730-734`),
and `/binary_sensor/radar_powered` **only exists in the satellite firmware**. A satellite
pointed at the old firmware is permanently silent, and it fails silently — the room just never
reports anyone. Do not build a mixed fleet. Retire `jarvis-room-sensor.yaml` to a one-line
pointer, or keep it only as the documented "no lease, no MOSFET" variant with a warning that
`rooms.satellites` cannot use it.

**One live trap in that file:** its `static_ip` default is `192.168.60.61`
(`jarvis-satellite.yaml:67`), i.e. it *presumes the IoT VLAN already exists*. Your LAN measured
tonight is `192.168.50.109/24` on `wlP9s9`. If decision ① comes back "no VLAN", change the
subsitutions before flashing or the device joins nothing.

**C6 — The MOSFET: "mandatory" vs "defer". RULING: buy it with the boards, wire it on room one,
never let it block the software.**

The privacy lane is right that without it the lease governs only the polling — which is the one
thing that never needed a lease. The fabric lane is right that soldering before the software is
proven is how you get three unexplained brownouts.

Both are satisfied by sequencing: a MOSFET breakout is $1–2, so buy three now; wire **one** at
step 4 and prove the LED goes out when the Spark is unplugged; steps 1–3 need no hardware at all.

**This one is not academic.** `RoomSensor.stop()` returns `self._power(False) if self.power_url
else True` — with no power URL it returns **True** having only set `_stopped`. `SensingPolicy`
then files it under `stopped`, and `commander.py:5498` says **"THE RADAR is down."** while the
module is still radiating. That is live today on your single-sensor path, and it is the
strongest argument for the MOSFET there is.

**C7 — Offline mode scope, and satellite microphones. RULING: house-wide only in v1; mics stay
live and *visible*.**

All three lanes independently reached "microphone stays live" and they are right for the same
reason `docs/offline-mode.md` gives: the switch is spoken back off, so gating the mic makes it a
one-way door. The privacy lane's addition is the one to keep — a `mic` row in the matrix so
"offline mode left the kitchen microphone live" is **visible rather than inferred**
(`rooms.py:103-104` reserves `MIC` for display and never leases it).

On scope, the privacy lane designed a full room-scoped grammar ("offline mode in the kitchen",
with OFF widening and ON narrowing on ambiguity). It is good design and it is **cut from v1**:
it is real regex work in `commander.py`, and its failure mode is a room left live, which is
precisely what the switch exists to prevent. The per-room *invariant* is already written and
harmless (`room_allowed == house.allowed and not room_denies` — a room override can only ever
subtract); leave the code, wire no voice grammar. One switch he can say in one breath.

### 1.2 The dependency order, made explicit

```
  jarvis-satellite.yaml  ──────►  rooms.Satellite       (lease endpoints must exist,
   (firmware, step 4)                                    or read() is silent for ever)

  sensing.SensingPolicy  ──────►  rooms.Satellite       (allowed() before the socket;
   (66af3ea, merged)                                     the reads counter is the proof)

  rooms.Satellite        ──────►  roomfabric.from_readers  (the reader shape is the seam)

  roomfabric.where()     ──────►  roomaudio.VoiceRouter    (arbitrated room, with confidence)
  roomfabric.anywhere()  ──────►  presence.RoomOrPhone     (via HouseView; NO edit to presence)

  step 0 (the phone week) ─────►  steps 8–9 (buy audio, or do not)
  step 5 (crosstalk)      ─────►  the four timers are guesses until this runs
```

Two things are genuinely *independent* and can be done in any order: the console surfaces
(step 3) and the voice routing (step 7).

Two integration edits were deliberately left unmade by every lane, and they are yours:
`app.py:356` builds `self.sensing`, `app.py:361` builds `self.presence` — the mesh goes
**between** them, and `presence._make_sensor` (`presence.py:126`) returns
`fabric.house_view()` when the mesh is configured, before the singular path. Both are additive.
`RoomOrPhone` (`presence.py:165-200`) needs **no change whatsoever** for three rooms; its
asymmetry is already the right rule at house scale, and the coverage rule's `None` lands on its
existing dark-safe branch.

### 1.3 What I killed, and why

| Cut | Whose | Why it does not earn its place |
|---|---|---|
| Snapcast, PipeWire RTP, `pulse-tunnel`, RAOP | audio | A reply plays in **one** room, so inter-room sync — the only thing they buy — is worth nothing on speech, and each adds a fixed 100–500 ms to *every* turn. Your best measured turn is `wait 1.33 s` (`/tmp/vss_voice/jarvis.log`, 20:58:16). None of them can return a receipt. |
| ESPHome native API + `micro_wake_word` | audio | A pre-trained `hey_jarvis` model is a real prize, but it costs either Home Assistant as a second always-on service on a box whose GPU is lent to trainers, or `aioesphomeapi` in a stdlib-only codebase. And your own `jarvis-room-sensor.yaml:1-20` already documents why `api:` is poison here. |
| The ARP household roster for "someone else is here" | fabric | 30 honest lines that need a second regular person's MAC. You are one person. A spoken "we have company" costs one regex and no surveillance. |
| Per-room offline-mode voice grammar | privacy | See C7. Real work; failure mode is a live room. |
| Per-room curfew windows / `curfew_extra` | privacy | One window, one thing to mis-set. The privacy lane already recommends this; I am just deleting the escape hatch. |
| Device-side SNTP curfew | privacy | ESPHome's `sntp` defaults to `pool.ntp.org`; a satellite with no WAN and no resolver **never syncs and the curfew silently does not exist**. The lease already gives you the curfew. |
| `RoomMesh.read`, `mesh_probe`, `occupancy_from`, the recency tie-break, `presence.rooms` | all three | C1–C3. |
| `jarvis-room-sensor.yaml` as a fleet firmware | — | C5. |
| ESP32-S3 `media_player` speaker option | audio | No `played_ms` receipt, no mic path later. A dead end you would buy twice. |
| Pi 4 + HiFiBerry tier | audio | Fidelity you cannot judge until a $40 box has told you whether you want the room at all. |

---

## 2. Architecture

```
        IoT VLAN or second SSID  ── 192.168.60.0/24 ──  no WAN, no resolver, no mDNS
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │   KITCHEN  .61              BEDROOM  .62                OFFICE  .60           │
  │  ┌───────────────────┐     ┌───────────────────┐      ┌───────────────────┐   │
  │  │ ESP32-WROOM       │     │ ESP32-WROOM       │      │ ESP32-WROOM       │   │
  │  │ ESPHome web_server│     │      "            │      │      "            │   │
  │  │   :80  BASIC auth │     │                   │      │                   │   │
  │  │   ota:false       │     │                   │      │                   │   │
  │  │ ┌───────────────┐ │     │                   │      │                   │   │
  │  │ │lease_watchdog │ │     │  (the Spark is in this room, and it still     │   │
  │  │ │ mode: restart │ │     │   earns a lease: if Jarvis dies the radar     │   │
  │  │ │ 90 s countdown│ │     │   must stop, and one firmware everywhere      │   │
  │  │ └──────┬────────┘ │     │   is one thing to reason about)               │   │
  │  │   radar_power     │     │                   │      │                   │   │
  │  │   internal:true   │     │                   │      │                   │   │
  │  │        │ MOSFET   │     │        │          │      │        │          │   │
  │  │   LD2410C   ●LED  │     │   LD2410C   ●LED  │      │   LD2410C   ●LED  │   │
  │  │             └── across the SWITCHED supply: no firmware can light it   │   │
  │  └─────────┬─────────┘     └─────────┬─────────┘      └────────┬──────────┘   │
  └────────────┼─────────────────────────┼─────────────────────────┼──────────────┘
               │                         │                         │
     GET  /binary_sensor/presence            every 2 s   (the fabric's thread)
     GET  /binary_sensor/radar_powered       every 2 s   (Satellite.read confirms first)
     POST /button/radar_lease_renew/press    every 25 s  (the mesh's thread)
     POST /button/radar_lease_revoke/press   at 21:00 and on "offline mode"
               │                         │                         │
  ═════════════╪═════════════════════════╪═════════════════════════╪══════════════
               └─────────────────────────┴─────────────────────────┘
                                         │  firewall:  Spark → IoT ALLOW
                                         │             IoT → anything DENY
  ┌──────────────────────────────────────┴────────────────────────────────────────┐
  │  THE SPARK   192.168.50.109/24                                                │
  │                                                                                │
  │   sensing.SensingPolicy ─── one switch, fails to OFF, state in sensing.json    │
  │      │  allowed(camera|radar)          attach("kitchen radar", revoke, renew)  │
  │      ▼                                                                         │
  │   rooms.RoomMesh ─── thread @25 s: renew | revoke | confirm-anyway             │
  │      │   Satellite × 3        read() -> True | False | None                    │
  │      ▼                                                                         │
  │   roomfabric.RoomFabric ─── thread @2 s: enter 2 s / leave 8 s / switch 6 s    │
  │      │   where()  anywhere()  RoomChanged(room, label, previous, at)           │
  │      ├──► HouseView ──► presence.RoomOrPhone ──► PresenceSentinel @60 s        │
  │      │                     (no edit to presence.py; None is its dark-safe path)│
  │      ├──► roomaudio.VoiceRouter ──► app._say(text, room=…)                     │
  │      └──► UI: header room strip  ·  Settings ▸ Privacy matrix (room × sensor)  │
  │                                                                                │
  │   Blue Snowball ─── the only microphone ─── faster-whisper + ECAPA gate        │
  │   SoundCore 2 (SBC Bluetooth) ─── room "here" ─── soundbar.py, unchanged       │
  └────────────────────────────────────────────────────────────────────────────────┘
        ▲                                          ▲
        │ POST /api/voice  {audio_b64, room}       │ POST /say  (wav) → {played_ms, peak}
        │ POST /api/say                            │ POST /stop  /duck  /unduck  GET /health
        │                                          │
   his phone — webapp.py + intercom.py        Pi satellite   ── LATER, and only if
   ENABLED TODAY. This is step 0.                             step 0 says yes.
```

**Protocols, complete list:** plain HTTP GET and POST over the LAN, stdlib `urllib`, BASIC auth
pre-sent, redirects refused (`rooms.py:203-206`), private-IP-literal URLs only, 4 KB response
cap. That is the whole wire. No broker, no MQTT, no mDNS, no cloud leg, nothing new in
`vss_env`.

**Why polling and not `/events`:** an ESPHome SSE stream emits a packet the instant somebody
walks through a beam, so the *traffic timing alone* leaks room occupancy to anyone in radio
range, encryption notwithstanding. The fixed cadence is an accidental win. **Do not move the
presence leg to `/events`.**

**Cost of the two threads:** the fabric issues 2 GETs per room per 2 s tick (confirm + presence),
so 3 GETs/s across three rooms; the mesh adds ~1 POST + 1 GET per room per 25 s. Measured
gateway RTT tonight: **min/avg/max 1.453 / 3.782 / 12.252 ms, 0 % loss over 10 packets**. The
fabric lane measured one three-room tick at **1.77 ms**, and **0.74 ms** with a dead room's
breaker open — I did not re-measure those.

---

## 3. The config schema — one block

Add to `DEFAULTS` in `jarvis/assistant_config.py`, and **delete `"rooms": []` at line 435**:

```jsonc
"rooms": {
  "enabled": false,              // THE master switch for all three rooms
  "here": "office",              // the Spark's room; the fallback for every rule
  "renew_s": 25.0,               // keep in step with lease_ttl in the firmware
  "stale_after_s": 90.0,         // older than this renders UNKNOWN, never the last value
  "timeout_s": 1.5,
  "satellites": [
    {"name": "office",  "label": "the office",  "url": "http://192.168.60.60",
     "sensors": ["radar"], "username": "jarvis", "password": "…", "lease_ttl_s": 90.0},
    {"name": "kitchen", "label": "the kitchen", "url": "http://192.168.60.61",
     "sensors": ["radar", "mic"], "username": "jarvis", "password": "…",
     "say_url": ""},
    {"name": "bedroom", "label": "the bedroom", "url": "http://192.168.60.62",
     "sensors": ["radar"], "username": "jarvis", "password": "…",
     "private": true}
  ]
},
"presence": {
  "room_sensor_enabled": false,  // legacy singular only, once rooms.enabled is true
  "room_sensor_url": "",         //   "
  "rooms_poll_s": 2.0,           // the fusion's cadence
  "rooms_enter_hold_s": 2.0,     // a new room must hold this long to take over
  "rooms_leave_hold_s": 8.0,     // the active room may read empty this long
  "rooms_switch_min_s": 6.0,     // floor between room changes — the doorway anti-flap
  "rooms_stale_after_s": 90.0,   // when the last known room stops being named
  "rooms_stuck_after_h": 12.0    // a bit ON this long is a fan, not a man
}
```

**Field notes, each load-bearing:**

- `name` is the identifier everything matches on — the `RoomChanged` subscriber, the attach
  name, a future sink map. `label` is what gets spoken. Slug the name; never speak it raw.
- `primary` from the fabric's schema is **deleted**; it is derivable as `name == rooms.here`.
  Two keys for one concept is how a room ends up spelled two ways.
- `sensors` is what is *physically* in that room. A kind not listed renders **ABSENT**, not OFF,
  because "there is no camera in the kitchen" and "the kitchen camera is off" are different
  claims and only one of them is a guarantee (`rooms.py:245-249`).
- `mic` is listed but **never leased** (`rooms.py:103-104`). It exists so the matrix can show a
  live microphone during offline mode instead of hiding it.
- `say_url` is separate from `url` on purpose: the radar is a $6 ESP32 on port 80 and the
  speaker may be a Pi on 8765, in the same room. Empty is a real state — a room he can be
  *seen* in but not *spoken* in.
- `password` must join `SECRET_LIST_FIELDS`, which today is
  `(("gmail.accounts", "app_password"),)` at `assistant_config.py:586`. Add
  `("rooms.satellites", "password")` and the masking in `repr(cfg)` and the logs comes free —
  the same standard `phone.token` already gets.
- `lease_ttl_s` must match the firmware's `lease_ttl` substitution
  (`jarvis-satellite.yaml:74`). They are two copies of one number; there is no way around that,
  so put them next to each other in the comment and say so.

---

## 4. Build order, in evenings

**Step 0 — the phone week. 5 minutes, $0, tonight.**
Open `webapp.py`'s page on your phone, prop it in the kitchen, leave it. `intercom` is already
enabled with `verify_speaker: False` and `max_mb: 10` (`assistant_config.py:509`), and a clip
decodes to exactly what the recorder produces so the resident Whisper and the speaker gate are
reused verbatim. At the end of the week you know whether steps 8–9 are worth ~$65, and no code
was written to find out.

**Step 1 — merge the lanes. Two evenings, $0.**
The whole of §1: delete `presence.rooms`, add the `rooms` block and the secret field, delete
`RoomMesh.read` / `mesh_probe` / `occupancy_from` / the recency tie-break and their tests, point
`VoiceRouter` at `fabric.where()`, build the fabric with `from_readers()`. Then the two
integration edits: the mesh between `app.py:356` and `:361`, and `presence._make_sensor`
(`presence.py:126`) returning `fabric.house_view()`. Budget two evenings because it touches four
files you did not write today and deletes from two of them.

**Step 2 — the three-room bench. One evening, $0.**
`scripts/roomsensor_stub.py --port 8781`, `8782`, `8783` (`:183`), three localhost URLs in
`rooms.satellites`, `curl .../on|off` to walk yourself from room to room. `--mode
garbage|slow|error` rehearses the three failures. The fabric lane measured an office→kitchen
handoff at **4.6 s** this way. Note the stub does not serve the lease endpoints, so run the
fabric over plain `RoomSensor`s here and save `Satellite` for step 4.

**Step 3 — the console. One to two evenings, $0.**
A header room strip (`OFFICE ● KITCHEN ◌ BED ●`) and the room × sensor matrix in
Settings ▸ Privacy. The rule that matters: **UNKNOWN differs from OFF in word, colour *and*
shape** — a hollow dot, not just a different colour, because colour alone does not survive a
dimmed monitor. A confirmation older than `stale_after_s` becomes UNKNOWN, never the last value;
otherwise a satellite unplugged while OFF renders as a confirmed OFF for ever, which is the most
comfortable lie available here. And **inference never sets confirmed**: "its lease has expired
so it should be down" is a second line of text, never a second colour.

**Step 4 — one satellite, and the proof. One evening plus soldering, ~$25.**
Flash `jarvis-satellite.yaml` with your own substitutions. Wire the LD2410's ground through the
MOSFET, the LED with its 1 k across the *switched* supply, and the 100 k pull-**down** on the
gate so the rail stays off while the ESP32 boots. Then the acceptance test that is the entire
point of the lane: **stop Jarvis and watch the LED go out by itself within 90 s.** No packet was
sent. That is fail-to-offline being a property of the device.

**Step 5 — measure crosstalk. One evening, $0.**
Put the second sensor on the far side of one wall, stand in each room, log how often *both* read
ON. The LD2410 sees through plasterboard (`docs/room-sensor.md` §9). High crosstalk means
`enter_hold_s` and `switch_min_s` go up, or `max_move_distance_gate` comes down on the device
page. **Every timer in §3 is arithmetic about a generic room, not a measurement of yours.**

**Step 6 — the other two. One evening, ~$30.**
Distinct `device_name` per device so the OTA targets do not collide; a DHCP reservation or a
static IP outside the pool per room, because a URL that goes stale on a lease renewal is a
presence outage that looks like a bug.

**Step 7 — announcements follow the room. One evening, $0.**
`app._say(text, proactive, kind)` is already the one door to TTS; add `room=`, defaulting to
`here`, and carry it to the playback device line. Publish `RoomChanged` once per *settled*
change, never per reading — three radars at 2 s would be a metronome on the bus. Three rules:
latch the room at the start of a burst so a handoff never moves audio mid-sentence; hold the
line rather than announce into an empty house when `confidence != certain` (`quiet.py:401`
already holds on `hold_when_away`); and **say nothing about the move itself** — no "you're in
the kitchen now".

**Steps 8–9 — audio. Four evenings, ~$65, only if step 0 said yes.**
A ~120-line `http.server` on a Pi: `POST /say` (wav in, `{"ok":true,"played_ms":3120,"peak":0.31}`
out), `/stop`, `/duck`, `/unduck`, `GET /health`. The receipt is the whole reason to prefer this
over every network sink: `pactl` sees a *sink*, not a *speaker*, and could not have caught the
2026-08-30 incident it was written after. Two receipts under half the expected duration and the
room is demoted with one spoken line, latched so a restart into a still-dead kitchen does not
repeat it. Ducking **inverts** the single-room rule: duck only the room about to be spoken in.
Barge-in is per room. On the mic: **score the speaker gate and log it, do not gate on it** — the
voiceprint was enrolled at the desk, near-field, and a kitchen at four metres moves the ECAPA
embedding the same way a phone microphone does. `intercom.verify_speaker` is already `False`
(`assistant_config.py:509`) for exactly that reason; a satellite is the same argument one room
further out. The way in later is per-room enrolment and a per-room threshold, since 0.25 was
calibrated at a desk. *(I could not read the voiceprint file to confirm its sample count — that
detail is unverified.)*

---

## 5. Shopping list — three rooms, one list

| Item | Qty | Est. each | Est. total | Note |
|---|---:|---:|---:|---|
| ESP32-WROOM-32 DevKitC / NodeMCU-32S | 3 | $6–8 | **$18–24** | 3-packs run ~$20–25. The yaml targets `board: esp32dev` and GPIO16/17 for the radar UART. **Not WROVER** — its PSRAM occupies those pins. Not C3/S3 — different pin map, no benefit here. |
| HLK-LD2410C | 3 | — | **$0** | **owned.** ~$4–5 each if you need more. |
| Logic-level N-MOSFET breakout (AO3400 / 2N7002) | 3 | $1–2 | **$3–6** | The part that makes "off" mean off. See C6. |
| LED + 1 k resistor | 3 | <$1 | **~$2** | **The most important two dollars here.** Across the *switched* supply, never to a GPIO — the only promise that survives a reflashed satellite. |
| 100 R gate + 100 k pull-**down** resistors | 3 sets | <$1 | **~$2** | The pull-down is what keeps the radar off while the ESP32 boots, which is the direction a privacy switch must fail in. A resistor assortment covers it. |
| 5 V ≥1 A USB PSU + cable | 3 | $3–5 | **$0–15** | Very likely in a drawer. **Not from the Spark**: the sensor must not reboot when the Spark does, which is exactly the moment "he just walked in" matters most. A 500 mA charger browns out on Wi-Fi TX and looks like a code bug. |
| F–F Dupont jumpers (4/room) | 1 pack | $6 | **$6** | No soldering for the radar itself. |
| Small plastic enclosure | 3 | $0–3 | **$0–9** | Radar sees through plastic. Never metal, never above a radiator. |
| | | | **$31–64** | **all three rooms sensing** |

**Network — check before buying.**

| Item | Est. | Note |
|---|---:|---|
| VLAN-capable router, or a second AP on its own subnet | **$0–120** | **$0 if your current router already does VLANs or a segregated second SSID.** This line is worth more than everything above it combined: it turns "a cheap IoT device on my LAN" into "a cheap IoT device that can reach nothing". Fallbacks, ranked: second SSID on its own subnet → guest network plus an explicit Spark allow → same LAN with per-device WAN block, documented as residual risk. |

**Audio — deferred behind step 0.**

| Item | Est. | Note |
|---|---:|---|
| Old phone / tablet on a charger | **$0** | Already a working satellite. Step 0. |
| Pi Zero 2 W + MAX98357A I2S amp + 3 W driver + PSU/case | **~$40** | A real `/say` endpoint with a receipt, and the same box the mic goes on later. Playing a wav is trivial; the Zero 2 W's reported unreliability was openWakeWord, not playback. |
| USB conference mic on the same Pi | **~$25** | Will be worse than the Snowball at range — no beamforming, no AEC. |
| *(alternative to both)* HA Voice Preview Edition | *$59* | XMOS XU316 AEC + beamforming + hardware mute. Best far-field hardware at the price — and it drags the ESPHome-API problem in with it. Cut from v1; noted so the option is on the record. |

**Totals.** Sensing all three rooms: **$31–64**. With the network item at worst: **+$120**. With
audio in one room: **+$65**. Honest band: **$31 minimum, ~$105 realistic, ~$250 worst case.**
Every price here is an estimate — I could not retrieve reliable live listings, and this is the
weakest evidence in the document.

**One address warning.** `192.168.50.60` — the example static IP in the shipped
`jarvis-room-sensor.yaml:32` — has an ARP entry on your LAN. Tonight it reads `FAILED` (nothing
answers); earlier today it was reported `STALE` with a MAC. Thin evidence either way, so the
instruction is simply: pick per-room addresses deliberately rather than flashing a template
twice.

---

## 6. What this cannot guarantee

Stated plainly, because a privacy feature whose limits are not written down is a feature that
will be over-trusted.

1. **Jarvis cannot guarantee a remote sensor is off.** It can ask, verify by reading back, and
   arrange that the device stops on its own when unasked. The third is the strongest available
   and it is a promise made by the **firmware**, not by Jarvis. A network drop, a crash, a
   `kill -9`, a power cut — every one of them ends in a powered-down radar precisely *because*
   nobody is left to send anything.
2. **Between the revoke and the lease expiring, a remote sensor may still be running** — up to
   `lease_ttl_s` (90 s radar, 20 s camera). At 21:00 the curfew sends an explicit revoke and the
   lease is only the backstop; a satellite unreachable at that exact moment keeps its lens up
   for up to 20 s. Display that window, do not round it down.
3. **A reflashed or lying satellite defeats every software check in this design.** Only the LED
   across the switched supply and a physical switch survive it. That is why the LED is not
   optional.
4. **Without the MOSFET, "offline mode" means "Jarvis stopped asking".** And today, on your
   single sensor with no `room_sensor_power_url`, `RoomSensor.stop()` returns True regardless
   and `commander.py:5498` says *"THE RADAR is down."* while the module keeps radiating. This is
   a live overstatement, not a multi-room regression.
5. **The office radar was never actually "enforced at the device" either.** `sensing.py`'s
   promise is real for a camera on the Spark's own USB; the radar has always been an ESP32 over
   Wi-Fi. The lease does not weaken anything — it is the first time that gap has been closed.
6. **UNKNOWN is not OFF, and it is not ON.** "Unreachable" covers powered-off,
   unplugged-and-carried-elsewhere, and moved-to-another-network, and nothing here can tell them
   apart. Never style it as safe.
7. **A lying satellite can make Jarvis think you are home. It cannot make Jarvis think you are
   away.** That containment is `RoomOrPhone`'s existing asymmetry (`presence.py:165-190`) and
   the coverage rule, and it is worth preserving verbatim.
8. **A satellite that goes permanently unreachable silently stops contributing to "away".** It
   is safe — it can only degrade to the phone-only behaviour this box had before any sensor
   existed — but it is invisible without step 3's console surface.
9. **Radar cannot count people, and it sees through plasterboard.** "Someone else is here" is
   not answerable from these sensors, and even "two rooms occupied at once" is one person near a
   wall until the gates are trimmed. The module says so in words rather than inventing it.
10. **Voice identity is not authentication** — and the reason has changed since the wake-gate
    diagnosis, so here is the current state rather than the remembered one. The length floor **was
    fixed on 2026-09-02** (`speaker.py:71 MIN_SPEECH_SECONDS = 0.35`, wired at
    `hotword.py:482-483`), so your 0.52–0.68 s wake word now clears it and that half of the
    problem is closed. What remains, and what matters for satellites: the gate still **fails
    open** by construction — `score is None` becomes `accept` (`hotword.py:488-489`, logged at
    `:507-508`) and an exception logs *"waking anyway"* (`hotword.py:484-486`) — and it
    **deliberately abstains over music** (`hotword.py:497-498`), because your own voice measured
    0.012–0.111 there against an impostor band reaching 0.138, where no threshold separates owner
    from stranger. Both are live right now, not history: today's log carries **11**
    `wake speaker check unavailable -- waking anyway` lines (most recently 14:37:32) and **44**
    `speaker verify: … too little to judge; abstaining (fail-open)` lines, the shortest at
    **0.50 s** of speech. The satellite path is a *better* place for the gate — it judges the
    1.0–6.9 s command, not the wake word — but a gate that fails open is not an authenticator
    anywhere. Score and log; never gate capability on it.
11. **A Jarvis restart costs presence in every satellite room for up to `renew_s` (25 s).** That
    is the price of the lease and it is the right way round.
12. **Wi-Fi metadata leaks regardless of encryption.** The fixed poll cadence keeps occupancy out
    of the traffic pattern; an event-stream design would put it back.
13. **Nothing in this design has met real hardware.** Not one ESP32 is flashed. Every timer is a
    hypothesis about your rooms, and step 5 is where they become measurements.
14. **The prices in §5 are estimates**, not quotes.

---

## 7. Decisions — only the ones that change the build

**① Does your router do VLANs, or at least a second SSID on its own subnet?**
*Recommend:* find out tonight, before ordering. An IoT VLAN with `Spark → IoT ALLOW` and
everything else denied is worth more than every other line in the shopping list. *If not:* fall
back down the ranked list in §5 and write the residual risk into the doc rather than hiding it —
and change `jarvis-satellite.yaml:67`'s `192.168.60.61` to a `192.168.50.x` address before
flashing, or the device joins nothing. *Consequence of skipping:* three cheap boards with
listening HTTP servers share a subnet with the phone client and the Spark.

**② Will you solder the MOSFET and the supply-rail LED?**
*Recommend:* yes, and treat both as mandatory for a satellite. *Consequence of no:* the lease
governs only the polling — the one thing that never needed a lease — and the spoken confirmation
overstates what happened, exactly as it does today (§6.4). That is a fair trade if you make it
knowingly; it must not be discovered later.

**③ House-wide offline mode only, in v1?**
*Recommend:* yes. One switch you can say in one breath. *Consequence of per-room:* real regex
work in `commander.py`, plus a grammar that has to widen on an ambiguous OFF and narrow on an
ambiguous ON — and its failure mode is a room left live, which is the thing the switch exists to
prevent. "Turn off the bedroom radar" can be a separate small feature later.

**④ Will you actually run the phone week before buying speakers?**
*Recommend:* yes, and it starts tonight for free. *Consequence of skipping:* ~$65–120 of audio
hardware bought against a guess about your own behaviour, and the guess most likely to be wrong
is "I will want to talk to the kitchen" versus "I will walk to the office anyway".

**⑤ Three rooms named `office` / `kitchen` / `bedroom`, spoken as "the office" / "the kitchen" /
"the bedroom"?**
*Recommend:* yes, with `office` as `rooms.here`. *Consequence:* renaming later is a config edit
only — nothing in code hard-codes a room name — but the attach names, and therefore the spoken
line "I couldn't stop the bedroom radar", are built from these.

**⑥ Lease 90 s radar / 20 s camera, renewed every 25 s?**
*Recommend:* as set. It survives two lost renewals. *Consequence of lengthening:* a satellite
that loses contact keeps sensing longer. *Of shortening:* a Jarvis restart, which you do daily,
costs presence in the other rooms more often.

**⑦ Is the bedroom `private`?**
*Recommend:* yes — never a broadcast target, never receives the held digest, mail or grades;
still answers questions asked in it, still gets alarms. One boolean, and the cheapest
guest-privacy control available before face ID exists. *Consequence of no:* the mail digest can
read itself out at 23:40 to whoever is in there.

---

*Written 2026-09-02 against `8b9c914`. Verified before writing: `ruff check jarvis/ tests/` →
**6 errors, exactly the stated baseline**; `pytest -q` over
`test_rooms.py test_roomfabric.py test_roomaudio.py test_roomsensor.py test_sensing.py
test_offline_mode.py` → **338 passed in 2.08 s**. Note that `jarvis/roomaudio.py`,
`tests/test_roomaudio.py` and `docs/multiroom-audio.md` were still **uncommitted** in this
worktree when this was written.*
