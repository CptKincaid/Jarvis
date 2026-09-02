# Offline mode

**One switch, one owner, and it fails to OFF.**

Jarvis's sensors are the camera (not yet wired) and the ESP32 + HLK-LD2410
mmWave radar (`jarvis/roomsensor.py`). Offline mode takes both down.

**The microphone stays live.** That is deliberate: offline mode is switched
back off by *speaking*, so gating the mic would make it a one-way door.
`jarvis/sensing.py` governs exactly two sensor names — `camera` and
`radar` — and an unknown name is denied rather than allowed, so nothing
can be gated through it by accident. The phone's Wi-Fi probe is not
governed either: it reads the kernel's ARP table for an address you
configured, and is not a sensor pointed at the room.

## Saying it

| You say | What happens |
| --- | --- |
| "offline mode" · "go offline" · "privacy mode" · "go dark" | camera + radar off, open-endedly |
| "deactivate presence" · "turn off the sensors" · "stop watching" | the same |
| "close your eyes" · "look away" · "no more cameras" | the same |
| "come back online" · "reactivate presence" · "start watching again" | sensing back on |
| "exit privacy mode" · "turn the sensors back on" · "wake the sensors" | the same |
| "are you watching?" · "is the camera on?" · "sensing status" | reads the state back; changes nothing |
| "no cameras for the next two hours" · "keep the camera off until noon" · "stop watching for ten minutes" | a **bounded** offline that ends by itself |
| "camera curfew from nine to seven" | sets the nightly window (overnight by convention, like quiet hours) |
| "start the camera curfew at ten tonight" | moves the start; the end is kept |
| "extend the camera curfew until noon" | moves the end; the start is kept |
| "turn off the camera curfew" | no curfew at all |

A leading "please" / "can you" / "could you" is allowed on all of them, and
so is naming both sensors at once ("turn off the camera and the radar").

Every one of those is **Tier 1** — matched by regex in `jarvis/commander.py`,
never routed to a model. A privacy switch that needs a 26B model resident
is not a privacy switch, and the GPU is lent out to trainers here daily.

Anything whose end is an *event* rather than a clock ("keep it off until
I'm back from class") deliberately falls through to the router, which can
ask a follow-up question; a regex cannot, and a mis-parsed time silently
rewrites a privacy schedule.

If a Tier-1 shape matches but the **time** does not parse, everything goes
off *now, open-endedly*, and the spoken line says the "until" was missed.
That is the only failure direction that cannot leave a lens open.

## The camera curfew

The camera is off **21:00 – 07:00** every day, from
`sensing.curfew.{enabled,start,end}` in `~/.config/jarvis/assistant.json`.
Change it by voice (above) or in the console's **Settings → Privacy**
pickers. Those rows are re-read every time the drawer opens, because
offline mode's primary control is the voice one and the switch moves while
the drawer is shut.

The curfew closes the **lens only**. The radar stays up: it produces no
image, so switching it off at night would cost presence for no privacy.

A malformed window in the config falls back to the shipped 21:00–07:00
rather than to "no curfew" — a typo must not quietly delete a privacy
control.

## Fail to offline

The switch lives in `~/.aiws_trainer/jarvis_memory/sensing.json`, **not**
in `assistant.json`, because that file's loader recreates a corrupt config
from defaults — which would fail *online*.

At start-up, anything that is not a readable, parseable, unambiguous
`"offline": false` comes up **OFFLINE**: missing file, empty file, truncated
JSON, wrong shape, unreadable permissions, `{"offline": "maybe"}`, **bytes
that are not valid UTF-8** (an interrupted write, a bad block — that one
raises `UnicodeDecodeError`, which is a `ValueError` and not an `OSError`,
so the read catches everything). A first run therefore starts offline; one
sentence ("come back online") clears it for good.

And if the policy object cannot be built **at all**, `jarvis/app.py` hands
the sensors `sensing.DENIED` instead of `None`: a governed sensor whose
policy is missing would otherwise decide that nobody is stopping it, which
is the same fail-online reached by the one path a fail-safe *inside* the
object cannot cover.

An **open-ended** offline does not expire — a switch that healed itself
overnight would reopen a lens nobody asked to reopen. A **bounded** one
("for the next two hours") is honoured even across a restart, because the
end time is his instruction and not a guess.

If the state file cannot be written, Jarvis still obeys the switch for the
rest of the run **and says so**: *"…I couldn't save that, so it won't hold
if I restart."*

## Enforcement is at the device

* `CameraGate` (`jarvis/sensing.py`) never calls the opener while sensing
  is denied. This is the interface the vision lane implements against;
  there is no camera on this box yet.
* **The curfew edge is enforced on a clock, not on the next `open()`.**
  `SensingPolicy.enforce()` walks the attached devices and closes the ones
  that have just lost permission (and resumes the ones it closed once it
  comes back); `SensingPolicy.start()` runs it on a daemon thread the
  policy owns, every 15 s. That thread — not the console's 5 s pass — is
  what shuts a lens opened at 20:59, because a privacy control that stops
  working when the Tk window is gone is not one. A stop that FAILED is
  retried on the next pass.
* `RoomSensor.read()` issues **no HTTP request at all** while offline — its
  `reads` counter is the assertion the tests make. "The readings are
  ignored" is not the same promise as "the radar was not polled".
* The spoken confirmation names what **actually** happened: which devices
  stopped, which refused ("I couldn't stop the radar, so it may still be
  running"), which were only stopped as far as this process reaches ("I've
  stopped reading the radar, but its power isn't switched, so it's still
  sensing the room" — see below), and never a device that is not
  physically there.

### A real power kill for the radar

**This is the state of the box today**: nothing is flashed, no MOSFET is
run, so the honest limit is that Jarvis stops asking — the LD2410 keeps
radiating and keeps serving presence to anyone on the LAN. `stop()` reports
that as `POLLING_ONLY` rather than as success, so the spoken line says it
out loud instead of claiming "the radar is down". To cut it for real, add
the optional MOSFET + GPIO block documented at the bottom of
`scripts/esphome/jarvis-room-sensor.yaml`, then set:

```json
"presence": { "room_sensor_power_url": "http://192.168.50.60/switch/radar_power" }
```

Offline mode then POSTs `<url>/turn_off`, and the firmware's
`restore_mode: ALWAYS_OFF` means a crash while offline leaves the radar
down.

## What degrades, and how

| Consumer | While offline |
| --- | --- |
| `jarvis/presence.py` | loses the radar leg only. The room reports **no opinion**, which is the module's existing dark-safe path, so it degrades to the phone's verdict on the same grace — exactly what it was before the sensor existed. On a **radar-only** box there is nothing left to fall back to, so the sentinel drops to **unknown** rather than freezing its last verdict: a stale "away" would make `quiet.py` answer "you're out" and swallow every proactive line for the whole blackout, and a stale "home" would speak into an empty room. A breaker outage (30 s) still holds — that is a transient, not a blackout. |
| `jarvis/quiet.py` | unchanged. It only ever *consumes* presence; gating it would turn a privacy switch into a silence switch. |
| `jarvis/deskpresence.py` | unchanged. GNOME's idle monitor is his own keyboard, not a sensor pointed at the room. |
| the console's ambient / standby modes | unchanged, for the same reason — they are driven by the desk probe. |
| the vision lane | refuses to open the device (`CameraGate`). |

## On the console

The header carries a badge beside the state pill, in every state:

* **SENSING** — cyan filled dot
* **CAMERA OFF** — amber filled dot (the curfew; the radar is still up)
* **OFFLINE** — amber **hollow** dot (nothing lit). Also what the badge
  shows when there is no policy object at all, because the app then hands
  the sensors a stand-in that denies everything.

Colour alone would not survive a dimmed monitor or a colour-blind glance,
so the three states differ in word, colour *and* shape. The badge refreshes
on the window's existing 5 s pass as well as on the spoken switch, because
the 21:00 curfew edge arrives with nobody having said anything.
