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
| "offline mode" · "go offline" · "privacy mode" | camera + radar off, open-endedly |
| "deactivate presence" · "turn off the sensors" · "stop watching" | the same |
| "close your eyes" · "look away" · "no more cameras" | the same |
| "come back online" · "reactivate presence" · "start watching again" | sensing back on |
| "exit privacy mode" · "turn the sensors back on" | the same |
| "are you watching?" · "is the camera on?" · "sensing status" | reads the state back; changes nothing |
| "no cameras for the next two hours" · "keep the camera off until noon" | a **bounded** offline that ends by itself |
| "camera curfew from nine to seven" | sets the nightly window (overnight by convention, like quiet hours) |
| "start the camera curfew at ten tonight" | moves the start; the end is kept |
| "extend the camera curfew until noon" | moves the end; the start is kept |
| "turn off the camera curfew" | no curfew at all |

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
pickers.

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
JSON, wrong shape, unreadable permissions, `{"offline": "maybe"}`. A first
run therefore starts offline; one sentence ("come back online") clears it
for good.

An **open-ended** offline does not expire — a switch that healed itself
overnight would reopen a lens nobody asked to reopen. A **bounded** one
("for the next two hours") is honoured even across a restart, because the
end time is his instruction and not a guess.

If the state file cannot be written, Jarvis still obeys the switch for the
rest of the run **and says so**: *"…I couldn't save that, so it won't hold
if I restart."*

## Enforcement is at the device

* `CameraGate` (`jarvis/sensing.py`) never calls the opener while sensing
  is denied, and closes a device that is already open the moment the
  curfew arrives. This is the interface the vision lane implements against;
  there is no camera on this box yet.
* `RoomSensor.read()` issues **no HTTP request at all** while offline — its
  `reads` counter is the assertion the tests make. "The readings are
  ignored" is not the same promise as "the radar was not polled".
* The spoken confirmation names what **actually** happened: which devices
  stopped, which refused ("I couldn't stop the radar, so it may still be
  running"), and never a device that is not physically there.

### A real power kill for the radar

Without extra wiring, the honest limit is that Jarvis stops asking — the
LD2410 keeps radiating. To cut it for real, add the optional MOSFET +
GPIO block documented at the bottom of
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
| `jarvis/presence.py` | loses the radar leg only. The room reports **no opinion**, which is the module's existing dark-safe path, so it degrades to the phone's verdict on the same grace — exactly what it was before the sensor existed. On a radar-only box the tick returns `None` and the sentinel **holds** its state: an unknown room is never reported as an empty one. |
| `jarvis/quiet.py` | unchanged. It only ever *consumes* presence; gating it would turn a privacy switch into a silence switch. |
| `jarvis/deskpresence.py` | unchanged. GNOME's idle monitor is his own keyboard, not a sensor pointed at the room. |
| the console's ambient / standby modes | unchanged, for the same reason — they are driven by the desk probe. |
| the vision lane | refuses to open the device (`CameraGate`). |

## On the console

The header carries a badge beside the state pill, in every state:

* **SENSING** — cyan filled dot
* **CAMERA OFF** — amber filled dot (the curfew; the radar is still up)
* **OFFLINE** — amber **hollow** dot (nothing lit)

Colour alone would not survive a dimmed monitor or a colour-blind glance,
so the three states differ in word, colour *and* shape. The badge refreshes
on the window's existing 5 s pass as well as on the spoken switch, because
the 21:00 curfew edge arrives with nobody having said anything.
