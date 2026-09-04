# Privacy and control across three rooms

**One sentence: Jarvis cannot guarantee a sensor in another room is off, so
it stops trying to be the enforcer and becomes the requester — each
satellite holds a lease that runs out on its own, and anything Jarvis cannot
confirm is shown as UNKNOWN rather than as a green tick.**

This builds on `docs/offline-mode.md` (`jarvis/sensing.py`, commit 66af3ea).
Read that first. Nothing here replaces it: `SensingPolicy` stays the single
owner of "may this kind of sensor run", the curfew stays where it is, and
fail-to-offline stays the rule. What is added is the answer to the question
that appears the moment a sensor is not plugged into the Spark.

---

## 1. Where enforcement lives

### The problem, stated plainly

`jarvis/sensing.py` enforces at the device: `CameraGate` never calls the
opener, `RoomSensor.read()` issues no HTTP request. Both promises hold
because both devices are on this computer. A satellite in the kitchen is a
separate computer:

* if the Wi-Fi drops, Jarvis can stop *trusting* the readings; it cannot
  stop the *sensing*;
* if Jarvis is `kill -9`'d, nothing sends the "stop" that offline mode
  depends on;
* fail-to-offline is a promise made at start-up by a process. A process that
  is not running makes no promises about a machine it cannot reach.

### The options, and the recommendation

| | how it fails | verdict |
|---|---|---|
| **Central enforcement** — Jarvis pushes off/on, satellite obeys | network partition or a Jarvis crash leaves the remote sensor **live, indefinitely** | **No.** This is the design that looks right and is wrong. |
| **Device-side clock curfew** — satellite holds 21:00–07:00 itself | needs a synced clock. ESPHome's sntp defaults to `pool.ntp.org`; a satellite with no route to the internet **never syncs, `on_time` never fires, and the curfew silently does not exist** | **Optional hardening only**, and only after verifying a LAN time source. |
| **Lease / deadman** — sensing is powered only while a lease is renewed | a lost renewal costs presence in that room until contact returns | **Recommended.** Every failure mode ends with the sensor powered down, without Jarvis doing anything. |
| **PoE switch port control** | needs a managed PoE switch plus a splitter per ESP32; the control plane becomes another credentialed IoT box | Not now. Reconsider for audio satellites, where a hard kill of the *whole device* is worth more. |
| **Smart plug** | cuts the ESP32 too, so the device cannot report and cannot be brought back by voice from that room; most are cloud-tethered, which breaks "everything local" | **No**, except as a deliberately-labelled manual kill. |
| **Physical switch / lens cap** | he has to be in the room | **Yes, as the floor.** It is the only guarantee that survives a compromised satellite. |

### The lease, concretely

`scripts/esphome/jarvis-satellite.yaml`. Verified ESPHome capability, not
assumed:

* `switch: platform: gpio` with `restore_mode: ALWAYS_OFF` (which is also
  ESPHome's default) — a reboot, brown-out or crash comes up **not sensing**.
* `script:` with `mode: restart` — "start a new run after first stopping
  previous run". The script is `turn_on rail → delay ${lease_ttl} →
  turn_off rail`. Each renewal restarts the countdown; when renewals stop,
  the last one runs out and the rail drops.
* `button: platform: template` + the web REST API's
  `POST /button/<object_id>/press` — the renew and revoke endpoints.
* `binary_sensor: platform: template` reading the rail back —
  `GET /binary_sensor/Radar%20powered` is what Jarvis confirms against
  (the entity's NAME, percent-encoded — the only path ESPHome's web_server
  serves; the lower-cased object_id is a 404, measured 2026-09-03).
* The rail switch itself is `internal: true`. **There is no endpoint on the
  device that turns a sensor on and leaves it on.** The only way to make a
  satellite sense is to keep asking, every 25 s, forever — which is true for
  Jarvis, and true for anyone who steals Jarvis's credentials.

One lease **per sensor kind**, not per room. The 21:00 curfew closes the
lens and leaves the radar up (`docs/offline-mode.md`); a room-wide lease
could not express that, because stopping renewal would take the radar with
the camera. `jarvis/rooms.py` substitutes `{Kind}` into every entity name.

**Revoke is the mechanism; the lease is the backstop.** At the curfew edge
and on "offline mode" Jarvis sends an explicit revoke and retries it three
times. The lease TTL is what covers the case where the revoke *cannot* be
delivered — i.e. exactly the case where nothing else can help.

### The physical floor

Wire an **LED + 1k across the switched supply** — between the MOSFET-side
LD2410 VCC and its GND — *not* to a GPIO. The light is then powered by the
same rail as the radar. No firmware can light it and no firmware can hide
it, so "is it off" is answered by looking at the room. This is the only part
of the design that survives a satellite that has been reflashed, and it
costs about 20 cents.

---

## 2. What the console shows when it cannot confirm

### The state model

Per **(room, sensor)**, in `jarvis/rooms.py`:

| state | means | shown as |
|---|---|---|
| `LIVE` | the device says this sensor is powered, within the freshness window | **LIVE**, cyan, filled dot |
| `OFF` | the device says it is not, within the window | **OFF**, amber, filled dot |
| `UNKNOWN` | no fresh answer — unreachable, stale, breaker open, unparseable | **UNKNOWN**, muted, **hollow** dot |
| `DISAGREE` | we asked for off and it reports on | **STILL ON**, red, filled dot, and it raises a `FaultRaised` |
| `ABSENT` | no such sensor in that room | row not drawn |

Three rules make this honest:

1. **A confirmation older than `stale_after_s` (90 s, three renewal periods)
   becomes UNKNOWN, not the last value.** Otherwise a satellite unplugged
   while OFF renders as a confirmed OFF for ever — the most comfortable lie
   available here.
2. **UNKNOWN differs from OFF in word, colour AND shape.** Same grammar as
   `jarvis/ui/sensing_badge.py`: colour alone does not survive a dimmed
   monitor or a colour-blind glance, and this readout is the one where
   being wrong is not cosmetic. `tests/test_rooms.py` asserts all three axes
   differ.
3. **Inference never sets `confirmed`.** When a room is unreachable and its
   lease has expired, the caption says *"unreachable; last heard 4 min ago —
   its lease has expired, so it should have powered down unless its firmware
   was changed"*. That is a second line of text, not a second colour. The dot
   stays hollow.

### Where it appears

1. **Header strip**, left of the existing sensing badge: one chip per room.
   `OFFICE ● KITCHEN ◌ BED ●`. Tooltip carries the caption.
2. **Settings → Privacy**, under the offline toggle and the curfew pickers
   the offline lane already added: the full room × sensor matrix with age,
   lease seconds and a per-room "off now" button.
3. **The Board**: a row whenever anything is `DISAGREE`, or `UNKNOWN` for
   more than five minutes.
4. **Spoken**: `rooms.spoken_status(view)` answers "are you watching?" per
   room and names the unreachable rooms **first** — *"I can't reach the
   kitchen radar. The lease has run out, so it should be off — but I can't
   confirm that. The office camera and radar are off."*

### One change needed in the offline lane's file

The header badge (`jarvis/ui/sensing_badge.py`) currently has three tones and
they describe **what Jarvis asked for**, house-wide. That is still correct,
but a badge reading a clean `OFFLINE` beside an unreachable kitchen is
half a lie. The minimal fix, for whoever merges (about twelve lines, and
deliberately not made here to avoid a conflict):

```python
TONE_UNSURE = "unsure"
WORDS[TONE_UNSURE] = "OFFLINE?"          # hollow amber, plus a bar
# badge_tone(): if the house view has any UNKNOWN or DISAGREE row while the
# policy says offline, return TONE_UNSURE instead of TONE_OFF.
```

Until that lands, the room strip carries the honesty and the badge means
"what was asked for".

---

## 3. "Offline mode" by voice: this room or the house?

**Recommendation: a bare command means THE WHOLE HOUSE.**

Three reasons, in order of weight:

1. **A privacy command must fail toward more privacy.** If "offline mode"
   silenced only the room he is standing in, the rooms it left live are
   precisely the ones he cannot see. The point of the switch is the rooms he
   is not in.
2. **"Here" is not well defined.** The utterance can arrive from the desk
   mic, a satellite mic, the phone client (`webapp.py`), or the intercom
   (`intercom.py`, `source="intercom"`). A word whose meaning depends on
   which microphone happened to hear it is a bad privacy control.
3. **It is already shipped that way.** `SensingPolicy` is house-wide and the
   spoken family in `commander.py` acts on it. Making bare commands local
   would silently reinterpret an order he has already learned.

### The grammar

| | scope |
|---|---|
| "offline mode", "deactivate presence", "stop watching", "close your eyes" | **the house** |
| "offline mode **in the kitchen**", "stop watching **the bedroom**", "**kitchen** camera off" | that room |
| "offline mode **in here**", "stop watching **this room**" | the room the utterance came from — **and if that cannot be resolved with certainty, it escalates to the house and says so**: *"I couldn't tell which room that came from, sir, so I've taken all of them offline."* |
| "come back online", "reactivate presence" | the house — everything on |
| "bring the kitchen back online" | that room only |

**The asymmetry rule, which is the whole of it:** *an OFF command widens on
ambiguity; an ON command narrows on ambiguity.* A room-scoped command may
never re-enable more than it names.

### Where per-room state lives

`SensingPolicy` stays house-wide and untouched. A per-room override is a
thin additive layer with one invariant:

```
room_allowed(room, kind)  ==  house.allowed(kind)  and  not room_denies(room, kind)
```

**A room override can only ever be MORE restrictive.** It can never grant.
That is the same shape as his identity ruling (§6) and it means a bug in the
per-room layer cannot open a sensor the house policy has closed.

---

## 4. The 21:00–07:00 curfew across three rooms

**Recommendation: one house-wide window.**

* He set one window and stated it as a house rule.
* Three windows are three things to mis-set by voice, and the failure mode
  of a mis-set window is a lens open in a room he is not in.
* The readout gets three times harder to scan at a glance, which costs more
  than the flexibility buys.
* The room most likely to want a different window (the bedroom) wants a
  **longer** one, and that is expressible without three independent windows.

So: one window in `sensing.curfew.*`, plus an optional per-room
**extension** in `presence.rooms[].curfew_extra`, enforced as a union and
never an intersection — a room's window may start earlier and end later than
the house window, never later or earlier. No voice grammar for it in v1 (a
picker in Settings → Privacy is enough, and the sentence "camera curfew in
the bedroom from eight to eight" would have to *refuse* to shrink, which is
a confusing thing to be told). If he later asks for per-room by voice, that
refusal is the design.

**The honest limit:** at 21:00 Jarvis sends a revoke to every satellite that
carries a lens. If a satellite is unreachable at that moment, its camera runs
until its lease expires — up to `lease_ttl_s`, 20 s for cameras. That window
is displayed, not swallowed.

---

## 5. Network posture

### Do they need internet? No — and deny it actively

Nothing on a flashed ESPHome device needs the internet at runtime. Three
caveats that bite if you do not know them:

* **sntp defaults to `pool.ntp.org`.** If you add a `time:` block for a
  device-side curfew and the device cannot reach the internet, it will never
  sync and the curfew will never fire — silently. Point `servers:` at the
  router **by IP** (there is no DNS configured) and verify before believing
  the curfew is armed.
* **`web_server: version: 2` fetches its page assets from the internet.**
  Set `local: true` or the browsable page is blank and looks like a fault.
  Jarvis's own GET/POST calls are unaffected either way.
* **OTA is a push from the Spark**, so the *device* needs no internet; the
  *build machine* does.

Configure `manual_ip` with **no `dns1`/`dns2`**, and block the satellites'
addresses from WAN at the router. A device with no resolver and no route out
is a poor exfiltration platform.

### VLAN or guest network?

**An IoT VLAN, not a guest network.** Consumer guest networks enable client
isolation and block guest→LAN, which is exactly the direction Jarvis needs
(Spark → satellite). The rules that are actually wanted:

```
Spark → IoT          ALLOW   (HTTP 80, and the esphome OTA port when flashing)
IoT   → Spark        DENY    except established/related
IoT   → WAN          DENY
IoT   → IoT          DENY    (a compromised satellite cannot reach the others)
IoT   → LAN          DENY
```

Fallbacks, ranked, if his router cannot do VLANs:

1. a second SSID mapped to its own subnet with the rules above;
2. a guest network **plus** an explicit allow for the Spark's address (some
   routers expose this);
3. the same LAN with WAN blocked per device — documented as a known residual
   risk, not hidden.

Two Spark-side notes. `jarvis/webapp.py` binds one private address and
refuses a non-private one; make sure that address is **not** on the IoT
subnet. And `jarvis/rooms.py` mirrors that rule outward: a satellite URL must
be `http://` plus a **private IP literal** — no hostnames (an mDNS answer is
one poisoned packet from re-pointing the lease at somebody else's box) and no
public addresses.

### Does ESPHome's API encryption key need setting?

There is no `api:` block, and that is a decision rather than an omission.
The native API is Home Assistant's protobuf-over-Noise channel; Jarvis is
stdlib-only and does not speak it (`aioesphomeapi` is a dependency and an
async runtime). Enabling it would open a listening port nothing connects to —
and with no client connected ESPHome **reboots the device every
`reboot_timeout`, default 15 minutes**, which would reboot the enforcer four
times an hour. If it is ever enabled, it needs **both** `reboot_timeout: 0s`
**and** `encryption: key:`; an unencrypted native API on the device that
holds a privacy switch is not acceptable.

So HTTP is the control plane, and it must be locked down instead:

* **`web_server: auth:`** with a generated 32-char password. Verified:
  ESPHome wraps every registered handler in `AuthMiddlewareHandler`
  (`web_server_base/web_server_base.h`), so the JSON REST endpoints and
  `/events` are covered, not just the browsable page. Jarvis sends
  pre-emptive Basic (digest costs a 401 round trip per request through
  urllib, doubling the poll, against an attacker the LAN threat model does
  not include). Credentials go in `presence.rooms[].password`, added to
  `SECRET_LIST_FIELDS` so they are masked in logs and `repr(cfg)` — the same
  standard as `phone.token`.
* **`web_server: ota: false`** — the built-in firmware-upload form has no
  business on the surface that answers presence queries. OTA stays on the
  `esphome` push path with its own password.

### Anything listening that should not be?

On the shipped `jarvis-room-sensor.yaml`, yes — four things, all fixed in
`jarvis-satellite.yaml`:

| | why it matters | fix |
|---|---|---|
| `web_server` with no auth | anyone on the segment can read presence, reboot the device, or (with the power block) turn the radar on | `auth:` |
| `button: platform: restart` exposed | one unauthenticated request = a denial of presence | `internal: true` |
| the tuning `number:` entities writable | set the absence delay to maximum and the room reads "occupied" for ever; set the gates to zero and it is blind | `internal: true` |
| `captive_portal:` + `ap:` fallback with the password `jarvisroom` written in the repo | whenever Wi-Fi hiccups, the device raises an AP whose captive portal **accepts new Wi-Fi credentials from anyone who joins**. In the office that is a recovery convenience. In a shared room, in radio range of neighbours, it is a standing invitation | removed; recovery is the USB cable |

Also removed: `mdns:` (the address is static and `assistant.json` holds a
literal — one fewer service, one fewer way to be enumerated).

### If a satellite is compromised

| it can | containment |
|---|---|
| lie about presence | bounded by the fusion asymmetry: a room seeing someone makes Jarvis think he is *home*; **a room can never make him away**, because an empty room never overrides the phone. So the worst case is Jarvis speaking when it should not, not going silent for the evening. |
| lie about `radar_powered` | **no software containment exists.** The supply-rail LED is the answer. This is the single most important limit in this document. |
| attack the Spark | it is untrusted input: `RoomSensor` caps bodies at 4 KB, parses with stdlib JSON, and returns "no opinion" for anything odd; `rooms.py` refuses redirects (a 302 to `http://127.0.0.1:8765` would otherwise make the poll loop an SSRF gadget against the phone client) and refuses non-private URLs. The IoT→Spark firewall rule is the real containment. |
| exfiltrate | needs WAN; hence the block and the missing resolver. |
| stream audio, once mic satellites exist | this is the compromise that actually matters. VLAN, and a **hardware** mute on any mic satellite. |

One accidental win worth preserving: Jarvis **polls at a fixed cadence**
(60 s at home, 10 s while away) rather than subscribing to ESPHome's
`/events` SSE stream. A push stream would emit a packet every time someone
walks through a room, so the *traffic timing alone* would leak occupancy to
anyone in radio range, encrypted or not. **Do not switch the presence leg to
`/events`.**

---

## 6. Guests

His ruling: no voiceprint enrolment for family; identity may **remove**
capability or **add a name**, and must **never grant** capability.

For satellite microphones that means six concrete things.

1. **A satellite mic never enrols anyone.** `speaker.add_sample` (passive
   learning) must not be reachable from a satellite source. A guest's voice
   quietly joining his voiceprint pool is the 2026-09-02 incident in a new
   dress — and that one was unrecoverable.
2. **No capability is unlocked by a voice, anywhere.** Sending mail, writing
   the calendar, opening Claude sessions, reading his mail aloud: gate them
   on the desk, the phone client's bearer token, or an explicit approval on
   the console. This is not only his ruling, it is what the numbers support —
   `scratchpad/wake-gate/wake_gate_diagnosis.md` measured the gate abstaining
   on **~50 % of wakes** (his wake word is 0.52–0.68 s of speech; the
   embedding needs 1.0 s trimmed), it **fails open** by design, and over music
   his own voice scores inside the impostor band. "It sounded like Hunter" is
   not authentication and cannot be made into one.
3. **Identity may only subtract.** A voice that is *confidently* not his
   suppresses the personal lane — no mail read aloud, no calendar detail, no
   "welcome back". An abstain is not a confident anything and must change
   nothing.
4. **Commands that REDUCE sensing are unauthenticated. Commands that INCREASE
   it are not.** Anyone in the room may say "Jarvis, stop listening in here"
   and be obeyed, whoever they are, because it only ever removes capability.
   Bringing sensing back needs the desk, the console or the phone token.
   The cost, stated plainly: a guest — or the television — can take the house
   offline by saying so. That is the failure direction he chose, it is
   recoverable in one sentence, and the console records where it came from.
5. **A mic satellite needs a hardware tell and a hardware mute.** An
   indicator wired to the mic's supply, and a physical mute button anyone in
   the room can press. A guest who cannot see the microphone and cannot turn
   it off has not been given a choice.
6. **Retention.** No satellite audio is written to disk beyond the turn
   (`intercom.py` already decodes in memory under a 10 MB cap — keep that).
   Two live traps: `JARVIS_DEBUG_AUDIO=1` dumps captures to disk and must
   refuse satellite sources; and utterances land in searchable history, so an
   utterance the gate *confidently* attributes to someone else should be
   handled and dropped rather than stored. Because the gate abstains so
   often, most guest speech will still be kept — say that rather than promise
   otherwise.

**And the practical shape he will actually use:** a spoken "we have company"
guest mode — satellite mics muted, proactive speech suppressed, sensing
otherwise unchanged. Trying to *identify* guests is the wrong tool; letting
him declare them is one sentence and needs no biometrics at all.

### Does offline mode mute satellite microphones?

**No by default**, for the same reason the local mic is not gated: offline
mode is spoken back off, and a satellite mic is how he does that from the
kitchen. But the difference must not be *silent* — the room matrix carries a
`mic` row per room precisely so "offline mode left the kitchen microphone
live" is visible rather than inferred. "Mute the kitchen" is a separate,
named command, and guest mode mutes them all.

---

## What this system cannot promise

Stated now, because an honest limit is worth more than a promise that fails
silently later.

1. **Jarvis cannot guarantee a remote sensor is off.** It can ask, verify by
   reading back, and arrange that the device stops on its own when unasked.
   The third is the strongest available and it is a promise made by the
   *firmware*, not by Jarvis.
2. **A reflashed or lying satellite defeats every software check here.** Only
   the supply-rail LED and a physical switch survive it.
3. **Between the curfew edge and the revoke landing** — or for up to one
   lease TTL if Jarvis is dead — a remote sensor may still be running.
4. **Fail-to-offline for satellites is a promise the satellite keeps**, and
   only while its firmware is intact and its power is on.
5. **"Unreachable" covers three different worlds** — powered off, unplugged
   and carried to another room, moved to another network — and Jarvis cannot
   tell them apart. UNKNOWN must never be styled as safe.
6. **A lying satellite can make Jarvis think he is home.** It cannot make
   Jarvis think he is away; that asymmetry is already in `presence.py` and is
   worth keeping for exactly this reason.
7. **Voice identity is not authentication.** Measured: ~50 % abstain, fails
   open, impostor-band overlap over music.
8. **Nothing here defends against physical access to a satellite.**
9. **Wi-Fi metadata leaks regardless of encryption.** The fixed poll cadence
   keeps occupancy out of the traffic pattern; an event-stream design would
   put it back.
10. **A Jarvis restart costs presence in every satellite room for up to one
    renewal period** (~25 s). That is the price of the lease and it is the
    right way round.

---

## Wiring it in

`jarvis/rooms.py` and `tests/test_rooms.py` are additive and are **not wired
into the app** — deliberately, so this lane cannot conflict with the offline
lane's edits to `app.py`, `presence.py` and `commander.py`. The integration
is small:

1. `jarvis/assistant_config.py` — DONE 2026-09-04 (F02). There is ONE room
   list, `presence.rooms`, and an entry becomes a leased satellite by
   listing `sensors`; the `"rooms"` section holds only the lane's timers:

   ```json
   "presence": {"rooms": [
       {"name": "kitchen", "url": "http://192.168.60.61",
        "sensors": ["radar"], "username": "jarvis", "password": "",
        "lease_ttl_s": 90}]},
   "rooms": {"renew_s": 25, "stale_after_s": 90, "timeout_s": 3.0}
   ```

   `("presence.rooms", "password")` is in `SECRET_LIST_FIELDS`.

2. `jarvis/app.py`, immediately after `self.sensing`:

   ```python
   self.rooms = self._construct("rooms", self._make_rooms)
   # def _make_rooms(self):
   #     mod = _import_optional("jarvis.rooms")
   #     return None if mod is None else mod.RoomMesh.from_config(
   #         self.assistant, policy=self.sensing)
   ```
   started in `start_assistant`, stopped in `quit`, and passed to `Services`
   so the console can render `mesh.view()`.

3. `jarvis/presence.py` — when the mesh is configured, wrap the probe:
   `probe_fn = rooms.mesh_probe(self.rooms, phone=probe)`. With no satellite
   configured this is a no-op and the office keeps the exact code path it has
   today.

4. `jarvis/commander.py` — `_h_sensing_status` gains one line:
   `rooms.spoken_status(mesh.view())` appended to the existing status line.

Order matters in exactly one place: the mesh must be built **after**
`self.sensing`, because every satellite attaches its stop/resume into that
policy at construction under a room-qualified name (`"kitchen radar"`), and
that name is what the existing spoken confirmation already reads back.
