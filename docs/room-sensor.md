# Room sensor: an mmWave presence radar for "is he home?"

Today Jarvis works out whether you are in by pinging your phone on the Wi-Fi. That
answer is slow and occasionally a lie: an iPhone drops off Wi-Fi power-save for minutes
at a time, so the honest away grace is twelve minutes and the *arrival* edge lands
whenever the radio next feels like waking up. On 2026-08-31 that was still enough to
greet you at 14:32 — but only because the phone happened to rejoin promptly.

This adds a second, faster leg: a small radar in the room that Jarvis reads over the LAN
with one HTTP request. It costs about **$25** and an evening.

**mmWave, not PIR, and that is the whole point.** A PIR fires on *motion*: it decides the
room is empty the moment you stop moving, so a man reading in a chair disappears inside a
minute. The LD2410 is a *presence* radar — it reports a still target from chest movement
and breathing — so "nobody there" survives you sitting still. That is what "is he home"
actually needs.

---

## 1. What to buy

| Part | What to search for | Rough price |
| --- | --- | --- |
| ESP32 dev board | "ESP32 DevKitC" / "NodeMCU-32S" / "ESP32-WROOM-32" (30- or 38-pin, USB) | $6–10 (3-pack ~$18) |
| mmWave sensor | **HLK-LD2410C** (the "C" has the plug; LD2410B is the same radar) | $6–12 |
| Jumper wires | female–female DuPont, 10 cm | $5 for a bundle |
| Power | any 5 V USB charger + the board's USB cable | $5–8 |

Total **$22–35**. Nothing else — no soldering, no level shifter (both sides are 3.3 V
logic), no hub, no subscription.

Two things to avoid:
* **Not** an "LD2410 + PIR combo" board, and not an HC-SR501 PIR. See above.
* **Not** an ESP32-WROVER unless you have to — its PSRAM occupies GPIO16/17, which is
  where the wiring below lands. If you already own one, use GPIO32/33 and change the two
  pins in the YAML.

Optional: a 3D-printed or cardboard bracket. The radar likes being 1–1.5 m up, pointed
across the room like a small speaker.

## 2. Wire it (four wires, two of them crossed)

The LD2410C header, left to right when the antenna faces you: **VCC, GND, TX, RX, OUT**.

```
LD2410 VCC  ->  ESP32 5V (also labelled VIN)      the radar wants 5 V, not 3.3 V
LD2410 GND  ->  ESP32 GND
LD2410 TX   ->  ESP32 GPIO16      <- crossed: their transmit is our receive
LD2410 RX   ->  ESP32 GPIO17      <- crossed: their receive is our transmit
LD2410 OUT  ->  nothing           the raw presence line; the UART tells us more
```

The two crossed wires are the only place this goes wrong. If the device boots, joins
Wi-Fi and shows the entities but presence never turns on, you have TX and RX straight
through — swap them.

Power the ESP32 from the USB charger, not from a laptop that sleeps.

## 3. Flash ESPHome

ESPHome is the firmware. Install it **anywhere except `~/vss_env`** — that venv is shared
with the VSS project and this does not belong in it:

```bash
python3 -m venv ~/esphome-venv && ~/esphome-venv/bin/pip install esphome
```

Edit the six lines marked `<-- CHANGE ME` at the top of
[`scripts/esphome/jarvis-room-sensor.yaml`](../scripts/esphome/jarvis-room-sensor.yaml):
your Wi-Fi SSID and password, an OTA password (any string), and the static IP + gateway
you want the device to have. Pick an address **outside** your router's DHCP pool — that
is what stops the URL in `assistant.json` going stale.

Plug the ESP32 into the Spark by USB and:

```bash
~/esphome-venv/bin/esphome run scripts/esphome/jarvis-room-sensor.yaml
```

First flash is over USB and takes a few minutes (it downloads a toolchain). Every flash
after that can go over Wi-Fi. When it finishes it tails the device log; you want to see
`WiFi Connected` and then `binary_sensor.presence: ON` when you wave at it.

> **Why there is no `api:` block in that YAML.** The native API is for Home Assistant, and
> with it enabled and no client ever connecting, ESPHome **reboots the device every 15
> minutes** by design. Jarvis reads the HTTP endpoint instead. Do not add `api:` back.

## 4. Find it, and check what it serves

The static IP you set is the address. Confirm the device is up:

```bash
curl -s http://192.168.50.60/binary_sensor/presence
# {"id":"binary_sensor-presence","value":true,"state":"ON"}
```

Browsing to `http://192.168.50.60/` gives a page listing every entity with live values —
useful for watching `Still distance` while you decide where to mount it.

If `curl` cannot reach it: check the device is on the same subnet as the Spark
(`ip -4 addr | grep 192.168`), and that the static IP is not one the router has already
handed to something else.

## 5. Tell Jarvis about it

In `~/.config/jarvis/assistant.json`, in the **existing** `presence` section:

```json
"presence": {"enabled": true, "phone_ip": "192.168.50.42", "phone_mac": "",
             "away_after_min": 12, "poll_s": 60, "poll_s_away": 10,
             "room_sensor_enabled": true,
             "room_sensor_url": "http://192.168.50.60/binary_sensor/presence",
             "room_sensor_timeout_s": 1.5}
```

Both keys are needed: the URL alone does nothing while `room_sensor_enabled` is false
(the log says so, once, at startup). The URL needs its `http://` — a bare
`192.168.50.60` is rejected on purpose rather than guessed at, because a typo that turns
into an invented URL fails as a silent timeout every poll instead of as one loud line.

**Keep `phone_ip` set.** The radar covers one room; the phone covers the flat. Together
they are strictly better than either — see section 7.

**A config edit needs a restart.** `AssistantConfig.reload_if_changed()` has no callers,
so nothing re-reads the file while he is running. Quit and start `python -m jarvis.app`
again.

## 6. Tell that it is working

```bash
# what would Jarvis read from that URL, right now?
~/vss_env/bin/python scripts/roomsensor_stub.py check http://192.168.50.60

# walk out, wait, walk back in: every transition, timestamped
~/vss_env/bin/python scripts/roomsensor_stub.py watch http://192.168.50.60
```

`watch` is also how you measure the real thing: the gap printed on the `SOMEONE` line is
the radar's own arrival latency. Add Jarvis's poll interval for the rest (section 7).

In `/tmp/vss_voice/jarvis.log`, at startup:

```
presence: room sensor http://192.168.50.60/binary_sensor/presence
```

and then, when you walk in, the line that already existed:

```
presence: home (returned)
arrival (phone): panel -> earcon -> greeting -> catch-up
```

## 7. What he actually does with it

Two legs, composed with one deliberate asymmetry:

* **The room seeing someone beats a sleeping phone.** A hit on the radar makes you home
  on that tick, and the phone is not even pinged. This is the arrival win.
* **The room seeing nobody is not absence.** You might be in the kitchen. An empty room
  never overrides a phone that answers, so "away" remains exactly what it was — the
  phone's verdict, after the same twelve-minute grace.
* **A sensor with no opinion — unplugged, unreachable, serving nonsense — is treated as
  if it were not configured at all.** After three consecutive failures Jarvis stops
  asking it for 30 s, then 60, 120, up to 5 minutes, so a dead ESP32 costs the poll loop
  nothing per tick rather than a timeout apiece. One warning line when that starts, one
  info line when it comes back, and nothing in between. **A false "away" makes him hold
  his proactive speech and go quiet on you, so absence is never something a broken sensor
  can invent.**

**Arrival latency**, measured end-to-end through the real sentinel over real HTTP
(`scripts/roomsensor_stub.py serve` standing in for the device):

| `poll_s_away` | best | mean | worst |
| --- | --- | --- | --- |
| 10 (default) | 1.7 s | **5.7 s** | 9.7 s |
| 5 (the floor) | 0.7 s | **2.7 s** | 4.7 s |

Add the radar's own detection time — under a second for someone walking in, and the YAML
throttles republishing to 250 ms — for roughly **3–7 s from the doorway to "Welcome back,
sir"**, against a phone-only path that is *up to a minute* when the phone is awake and
genuinely unbounded when it is not. If you want the faster row, set `"poll_s_away": 5`;
the extra cost while you are out is one HTTP GET and one ping every five seconds.

Reading the sensor costs 0.5 ms on the LAN. A wedged device (accepting connections but
never answering) costs 1.5 s per tick for three ticks and then nothing.

## 8. Trying it before the parts arrive

The software half works with no hardware at all:

```bash
# terminal 1 - a fake ESP32 on localhost
~/vss_env/bin/python scripts/roomsensor_stub.py serve
#   serving http://127.0.0.1:8781/binary_sensor/presence   (mode=ok, state=OFF)

# terminal 2 - put THAT url in assistant.json, restart Jarvis, then:
curl -s http://127.0.0.1:8781/on      # "someone walked in"  -> he should greet you
curl -s http://127.0.0.1:8781/off     # "the room is empty"
```

The stub also rehearses the three failures worth rehearsing, and under all three Jarvis
must behave exactly as it does with no sensor:

```bash
scripts/roomsensor_stub.py serve --mode garbage   # HTML where JSON should be
scripts/roomsensor_stub.py serve --mode slow      # hangs past the timeout
scripts/roomsensor_stub.py serve --mode error     # HTTP 500
```

(It listens on 8781 because the phone web UI already owns 8765.)

## 9. Placement, and the two ways mmWave surprises you

* **It sees through things.** Plasterboard, a door, a sofa back. Point it *into* the room
  you care about, not at the wall you share with a corridor, or it will report the
  neighbour's hallway as your living room. Use the device's own web page to watch
  `Detection distance` while someone walks about.
* **It sees moving things that are not people.** A pedestal fan, a curtain over a vent, a
  hanging plant near an air conditioner, a large dog. If presence never goes OFF, look for
  something that moves all day inside the beam, then trim `Max move gate` / `Max still
  gate` on the device page.
* Mount it 1–1.5 m up, roughly chest height, pointing across the room. Not on the ceiling
  (that geometry is for a different product), not behind a metal object, not directly
  above a radiator.
* `Absence delay` on the device page is how long the radar holds ON after the last sign of
  life (factory: 5 s). That is a *departure* knob and it barely matters here — Jarvis's own
  twelve-minute grace dominates it. Raise it only if a still target flickers off while you
  read.
* **It locks on to furniture.** The third surprise, and the one trimming the gates cannot
  fix when the furniture is *behind* you: sensor → open space → the back of a chair → the
  person → desk and monitors → a wall reads OCCUPIED for ever (measured 2026-09-11 in the
  office: 527 of 527 samples with the flat empty, `Still distance` pinned at 306–313 cm,
  and shortening the gate would have cut at 300 cm, where the person actually sits —
  seated he reads 288–337 cm). Jarvis does not trust the bit alone in that case: every poll
  that reads occupied also reads `Still distance`, and once the readings cover a **180 s**
  window, a spread under **20 cm** means the room is read as **empty** (a body breathes
  and shifts; a desk does not — the empty office measured at most 7 cm over any full
  window, a seated person at least 35 cm). Nothing else can take an occupancy away: a
  missing or unreadable distance keeps the sensor's word, and the window is aged by the
  clock, so a distance entity that dies hands the bit back within three minutes. The costs,
  measured on the same recordings: a locked room reads occupied for the first ~3 minutes
  after Jarvis starts (or after the radar was unreachable for 8 s+), and a person who sits
  down in front of a locked radar is seen once the distance moves 20 cm — median 12 s,
  worst 82 s. The log says so, once, when it happens: `roomfabric: office reads occupied
  but its still distance has moved only 7 cm in 180 s … a fixture, not a body; reading it
  empty`, once when a body moves again, and once if the distance stops answering (`the
  fixture check is blind`). It is on for every radar in `presence.rooms`; put
  `"still_check": false` in a room's entry (or `presence.room_sensor_still_check` for the
  single-sensor keys) to switch that room off. It needs `presence.rooms_poll_s` at 3.6 s or
  faster (the default is 2.0) and warns at startup if it cannot fill its window. The numbers
  live in `jarvis/roomstill.py`; the instrument that produced them is `scripts/room_trace.py`.

## 10. Turning it off

Set `"room_sensor_enabled": false` and restart. You are back to exactly today's
behaviour, phone only, same states, same timings. Pulling the plug on the device gets you
there too, one warning line later.
