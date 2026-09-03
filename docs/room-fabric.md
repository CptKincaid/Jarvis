# The room fabric: one room becomes three

`docs/room-sensor.md` gets ONE mmWave sensor onto the wall and into
`presence.room_sensor_url`. This is the plural of it: three named rooms, one
fast poll, and the three answers the rest of Jarvis wants.

| question | who answers | shape |
|---|---|---|
| which room is he in | `RoomFabric.where()` | `Where(room, confidence, occupied, unknown)` |
| is he anywhere in the house | `RoomFabric.anywhere()` | `True` / `False` / `None` |
| is someone ELSE here | `RoomFabric.others()` | **always `None`** — see "What radar cannot tell you" |

Code: `jarvis/roomfabric.py`. Tests: `tests/test_roomfabric.py`.

## 1. Config

Shaped exactly like `gmail.accounts` — the codebase's existing plural. A LIST
of labelled entries, and while it is empty the singular `presence.room_sensor_*`
keys are used instead, so a config written before three rooms existed keeps
working with no edit at all.

```jsonc
"presence": {
  "room_sensor_enabled": true,          // still the master switch over ALL of it
  "rooms": [
    {"name": "office",  "label": "the office",  "url": "http://192.168.50.60",
     "primary": true},
    {"name": "kitchen", "label": "the kitchen", "url": "http://192.168.50.61"},
    {"name": "bedroom", "label": "the bedroom", "url": "http://192.168.50.62"}
  ],
  "rooms_poll_s": 2.0,
  "rooms_enter_hold_s": 2.0, "rooms_leave_hold_s": 8.0,
  "rooms_switch_min_s": 6.0, "rooms_stale_after_s": 90.0,
  "rooms_stuck_after_h": 12.0
}
```

`name` is the identifier everything else matches on (an audio sink map, a
`RoomChanged` subscriber); `label` is what gets spoken. `primary` is the room
the Spark is in — it is a tie-break and nothing more. An entry with no `url`,
no `name`, a duplicate name, or `"enabled": false` is skipped with one log
line: one unfinished room must not take the others down.

`presence.room_sensor_enabled: false` still turns the whole fabric off, so
`docs/room-sensor.md` section 10 stays true word for word with three sensors on
the wall.

## 2. The four timers, and why those numbers

The LD2410's OFF edge is late by design — its own "absence delay" holds a
target for a factory 5 s after the last sign of life — and its ON edge is
fast. Every timer is built around that asymmetry.

| timer | default | its one job |
|---|---|---|
| `enter_hold_s` | 2.0 s | a new room must hold occupied this long before it may take over. A doorway pass-through at walking pace is ~1 s inside the beam, so 2 s separates "walked through" from "walked in". |
| `leave_hold_s` | 8.0 s | the current room may read empty this long and still be believed. The device's own absence delay is already 5 s; 8 s is that plus poll jitter. Shorter re-litigates the device's timer and flaps every time he leans out of the beam. |
| `switch_min_s` | 6.0 s | a floor between room changes — the doorway anti-flap. Standing in a doorway BOTH sensors see him (radar goes through plasterboard) and newest-edge alone would ping-pong every poll. |
| `stale_after_s` | 90 s | when the last known room stops being named at all. Between `leave_hold_s` and this, `where()` still names the room and says `stale`. |

**Measured** on three `scripts/roomsensor_stub.py` instances (8781/8782/8783):
office → kitchen handoff **4.6 s** wall clock; one tick of three real HTTP GETs
**1.77 ms**; one tick with a dead room's breaker open **0.74 ms**.

Arbitration when two rooms are both occupied: the **newest ON edge wins** — the
room he walked *into* is the room whose bit turned over most recently — then
`primary` as a tie-break.

## 3. What radar cannot tell you

* **An LD2410 reports one presence bit. Not a count, not an identity.** A room
  that reads occupied holds one person or four.
* **It sees through plasterboard.** Two sensors either side of one wall can see
  the SAME body, so "two rooms occupied" is not evidence of two people.

So `others()` returns `None` and `others_reason` says why in words a spoken
line can use. The legs that *can* answer "is someone else here", cheapest
first: an ARP roster of known phones (works today, no new hardware), the
speaker gate on a voice, and face recognition later.

## 4. Failure, and what Jarvis believes

`anywhere()` returns `False` **only when every room answered `False`**. With one
sensor "empty" was a statement about the one room we could see; with three it is
a statement about coverage, and a room whose breaker is open means we do not
have it. A single unknown room makes the house `None`, which falls through to
the phone probe — exactly the behaviour this box had before any sensor existed.

| what breaks | the fabric | the house | safe? |
|---|---|---|---|
| one ESP32 reboots (~10 s) | that room `None` for ~10 s, then the breaker's 30 s cooldown | `None` unless another room sees him | yes |
| a satellite is unplugged | breaker open, cooldown doubles to 300 s, zero cost per tick | permanently `None`, so the fabric stops contributing to "away" | yes, but invisible — watch the `unknown` list |
| the LAN is congested | up to 1.5 s timeout × 3 rooms for 3 ticks, then the breakers open | `None` | yes |
| Wi-Fi drops entirely | every room `None` | phone-only, and his phone is off the LAN too, so 12 min later "away" | quiet and wrong; unavoidable |
| the Spark restarts | history empty; `where()` unknown until 2 s after the first sighting | phone-only for one poll | yes — and the ESP32s keep running, which is *why* they are not powered from the Spark |
| offline mode | every room `None`, **no HTTP request is sent at all** | phone-only — and with no phone leg the sentinel goes to **unknown** rather than freezing on its last verdict, via `HouseView.blocked` | yes |

`HouseView.blocked` is what carries that last row. `PresenceSentinel._blacked_out()`
reads `sensor.blocked` and nothing else asks the question, so the view shipping
without the property (2026-09-03) silently turned the dark-safe path off on the
one box shape that needs it — no phone leg, rooms only. It reports the house
blind only when EVERY configured room is blocked: one radar still permitted to
look is still a leg. Anything else wearing this interface owes the same
property, and presence now logs a missing one at ERROR instead of swallowing it.

A room stuck ON is the opposite risk — a pedestal fan inside the beam is the
documented failure and would make the house occupied for ever. A room whose bit
has run continuously for `rooms_stuck_after_h` is dropped from both answers with
one warning line and taken back the moment it reads false. The real fix is
placement and the `Max move gate` / `Max still gate` knobs (`docs/room-sensor.md`
section 9).

## 5. Offline mode with three radars

`SensingPolicy.attach` is keyed by device NAME and replaces a duplicate, so
three `RoomSensor`s built against one policy register as ONE device and only the
last one's power is cut. Measured 2026-09-02. `RoomPolicy` is the fix: it passes
`allowed()` straight through and namespaces the attach as `"<label> radar"`, so
every radar is registered and every one is stopped. The name is words and not
`radar:kitchen` because the commander *speaks* it — `_sensing_join` turns it into
"the office radar and the kitchen radar", which is a sentence. A single-sensor
install keeps the bare `"radar"` so the line he hears today does not change.

## 6. Trying it before the parts arrive

The whole fabric works with no hardware:

```bash
for p in 8781 8782 8783; do
  ~/vss_env/bin/python scripts/roomsensor_stub.py serve --port $p &
done
# put those three URLs in presence.rooms, restart Jarvis, then walk about:
curl -s http://127.0.0.1:8781/on    # he is in the office
curl -s http://127.0.0.1:8781/off; curl -s http://127.0.0.1:8782/on   # kitchen
curl -s http://127.0.0.1:8783/binary_sensor/presence   # what the bedroom serves
```

`--mode garbage|slow|error` rehearses the three failures worth rehearsing;
under all three the house must fall back to the phone probe and nothing else
may change.

## 7. The kitchen is the front door

His words: *"kitchen to see if i enter my apartment since the kitchen and door
are next to each other"*. So one room is nominated as the DOOR, and that room
going occupied **after a whole-home absence** is an arrival — greeted at once,
rather than when his phone's radio next answers an ARP.

```jsonc
"presence": {
  "door_room": "kitchen",     // must match a `name` in `rooms` above; slugged
                              // the same way, and a door room that names no
                              // configured room is warned about at startup
  "arrival_outing": true,     // "Welcome back from the dentist, sir"
  "arrival_offer": true       // "You've 3 unread emails. Shall I go through them, sir?"
}
```

It runs through the same `arrival_mod.run()` choreography as the phone and desk
probes — panel, earcon, greeting, catch-up — and through the same
`_greet_return` damper, so two sentinels noticing the same walk through the
door is still ONE welcome. Code: `app._on_room_changed`, `arrival.DoorWatch`.

**A kitchen trip mid-evening is not an arrival.** The gate is the absence, not
the room: `presence.state` must read `away` (not `unknown`, so a restart while
he is at his desk greets nobody). Making coffee at nine while he is already
home never fires.

**"Welcome back from X" needs evidence.** `arrival.outing()` names a calendar
event only when **both** halves of the coverage rule hold — he was out for at
least half of the EVENT, and the event accounts for at least half of the
ABSENCE — AND it ended no more than 45 minutes before he walked in (or was
still running). The second half was missing on the first cut, and without it a
four-minute entry that ended sixteen minutes before he got in named a four-hour
absence: *"Welcome back from take the bins out, sir."* The honest cost of the
fix is that a one-hour class inside a two-and-a-half-hour absence is now the
plain line — a long commute either side of a short event is exactly the case
the calendar cannot prove, and `absence_cover` is a keyword argument for anyone
who disagrees. No calendar, an unreachable one, an all-day event, no recorded
departure, a title too long to speak, or TWO events that both fit — every one
of those is the plain "Welcome back, sir", because a guessed event name is
worse than no event name. It reads the `CalendarSource` CACHE, so a homecoming
never waits on caldav.

**The catch-up OFFERS, it does not deliver.** The mail half is a count and a
question — never a sender, never a subject — plus one clause on anything major
(an `error` on the fault board, in its own words and its own case). He gets the
contents when he answers yes, and the yes is resolved by the offer protocol
that already exists: `services.briefing_offer` + `Commander._try_briefing_offer`,
60 s TTL, end-anchored yes/no. The quiet-hours digest in front of it is
unchanged, and the fault clause and the question go in as SEPARATE fragments of
the same single address pass, so the burst still says "sir" twice at most.

Three rules that are easy to get wrong, and were:

* **A fault is TOLD, not offered.** With no mailbox and a standing error the
  line is `"The disk is full."` and there is no question and nothing parked —
  there is nothing to "go through" in a broken disk, and the first cut asked
  anyway and then read the same sentence back when he said yes. The delivery
  never repeats what the offer already spoke.
* **The 60 s TTL starts when the digest has been SPOKEN**, not when the
  question was parked, or a long quiet-hours backlog eats the window he has to
  answer in.
* **The unread count runs on a worker thread** when a mailbox is configured.
  `bus.drain()` runs from the UI's Tk pump, so an IMAP round trip taken inline
  froze the window and every event behind it at the moment he walked in. The
  bound is `IMAP_TIMEOUT`: 15 s a mailbox, in parallel, so one timeout rather
  than their sum. What three of his mailboxes actually cost is **not
  measured** — the 8.1 s figure elsewhere in the tree is mail.py's *sequential*
  measurement from 2026-08-31 and predates the pool. With no mailbox nothing
  opens a socket and the cue stays synchronous.
* **A catch-up that lands too late is DROPPED, not spoken.** Off the pump the
  last step is no longer atomic, so before it speaks the worker asks whether
  the arrival it belongs to is still the current one and whether he has taken a
  turn since (`_dispatch_gen`, the same counter `_async_reply` reads). If
  either has moved, silence: nothing in the digest is news that keeps. The
  cue's log line says `catch-up (started)` on that path, because that is all
  `run()` can honestly claim.

Degradation is the point: with no kitchen sensor, no calendar and no mailbox he
gets exactly the "Welcome back, sir" he got before any of this was built.
