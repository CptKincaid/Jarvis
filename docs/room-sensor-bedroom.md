# Bedroom placement: putting the radar where it can hold a sleeper

**Yes — an LD2410C will detect you in bed; Hi-Link's own manual says the module senses
"static, micro-moving, sitting and lying human bodies", and the still channel is really a
breathing detector, which a sleeping man supplies continuously.** **Put it on the ceiling,
flat, directly above your sternum — *not* at chest height on the headboard wall, which is
what `docs/room-sensor.md` currently tells you and which is the one placement that
reliably fails.**

Everything below supports those two sentences.

> **Confidence key**, on every claim.
> **[datasheet]** a number, table or figure in Hi-Link's LD2410C manual or serial-protocol
> PDF. **[vendor doc]** Hi-Link prose, or ESPHome's own docs/source. **[community]** a
> field report from a stranger, named. **[inference]** reasoning, mine, unmeasured.
>
> **Nothing here was measured on hardware.** There is no radar attached to this machine.
> Every distance is Hi-Link's, a stranger's, or arithmetic. Section 4 exists because of
> that: the overnight log is the only thing that settles this.

---

## 0. The one number that decides the placement

**The LD2410 does no still-target detection inside 1.5 m.** Static sensitivity for distance
gates 0 and 1 is listed as *"(not settable)"* in Hi-Link's factory-default table — not a
threshold of zero, but not implemented. **[datasheet]** Inside 1.5 m the module is a motion
sensor, i.e. a PIR, i.e. the thing you rejected.

That number is *not* in `docs/room-sensor.md`, and it disqualifies the mount that most
online guides recommend.

**Where the two research lanes disagreed, and what I picked.** One lane read gates 0 and 1
as both covering 0–0.75 m (citing a forum post), making the still floor 0.75 m and a
headboard mount viable. The other read gate *N* as spanning 0.75×*N* to 0.75×(*N*+1),
making the floor 1.5 m. **I picked 1.5 m**, because Hi-Link's own worked example says
gates 3 and 4 cover *"2.25 to 3.75m"* **[datasheet]** — which only works if gate *N* starts
at 0.75×*N*, so gate 1 is 0.75–1.5 m. A manufacturer's worked example beats a forum post.
It is also the conservative choice: the recommended mount below clears 1.5 m anyway, so
picking the stricter reading costs nothing. (Hi-Link contradicts itself elsewhere by
exactly one gate on where a max-gate fence lands **[datasheet]** — resolve that on the
device with `Still distance`, not by reading.)

---

## 1. Where it goes: ceiling, above your sternum, flat

### The numbers to tape-measure

| | Value | Why |
|---|---|---|
| **Height** | the ceiling, flat against it | see below |
| **Distance to your chest** | ceiling height − (mattress top + 0.20 m). **Must be ≥ 1.55 m.** | the 1.5 m still floor, plus a hand's margin **[datasheet]** |
| **Tilt** | **0°** — face parallel to the ceiling, pointing straight down | a supine sleeper's chest moves vertically; overhead is the only geometry where the radar sees all of that movement rather than a fraction of it **[inference]** |
| **Horizontal position** | 0 m offset. Directly over the sternum: ~0.6–0.7 m out from the headboard wall, on the bed centreline. Shared bed → over *your* side, ~0.40 m off centre. | keeps you inside the strong part of the lobe |
| **Orientation** | antenna side (the etched patch array) faces the room; component side faces the ceiling | |
| **Behind it** | 60×60 mm aluminium plate or a patch of foil tape between module and ceiling | kills the back lobe into the flat above — Hi-Link's own remedy, and free at this mount **[datasheet]** |
| **Enclosure** | non-metallic; module held **12.4 mm** (±1.2 mm) off the inner face; cover wall **≤1.55 mm** | Hi-Link's radome spec at 24.125 GHz. A lid glued straight onto the antenna eats range silently. **[datasheet]** |
| **Fixing** | screwed or VHB-taped. Not dangling on its cable. | *"the shaking of the radar itself will affect the detection effect"* **[datasheet]** |

Worked, for common ceilings, with a 0.55 m mattress top (torso surface 0.75 m):

| Ceiling | Chest distance | Gate | Verdict |
|---|---|---|---|
| 2.3 m | 1.55 m | 2, just | **too tight** — a thicker mattress puts you inside the dead band |
| 2.4 m | 1.65 m | 2 | OK, 0.15 m of margin. Measure your actual mattress. |
| **2.5–2.8 m** | **1.75–2.05 m** | **2** | **the sweet spot** |
| 3.0 m | 2.25 m | 2/3 boundary | fine, but the still distance will flip between gates |

### Why the ceiling, when your own doc forbids it

`docs/room-sensor.md:223-225` says *"Mount it 1–1.5 m up, roughly chest height, pointing
across the room. Not on the ceiling (that geometry is for a different product)."*

**That line is correct for a desk and wrong for this part.** Hi-Link documents ceiling
mounting at 2.6–3 m as one of the LD2410C's two first-class installations, with its own
measured range plot (Figure 7 and Figure 8), and the LD2410 feature list says *"Support
ceiling, wall and other installation methods"*. **[datasheet]** Both research lanes reached
this independently; there was no disagreement.

More than sanctioned — it is the *better* geometry for a sleeper, by Hi-Link's own two
measurements. Reading the polar plots (±0.2 m by eye, so trust the ratio, not the metres):

* **Wall at 1.5 m (Figure 10):** Motion ≈4.9 m, Stand ≈4.6 m, **Sit-still ≈4.0 m** on
  boresight; all three collapse to ≈2.6 m at ±60°. Still-detection is the *weakest* mode.
* **Ceiling at 3 m (Figure 8):** near-circular; Move/Stand ≈3.4–4.0 m, **Sit-still
  ≈3.7–4.2 m — the outermost curve at most bearings.** Still-detection is the *strongest*
  mode. **[datasheet]**

In a bedroom, still-detection *is* the product. The reversal decides it.

The published beam is **±60°, with no axis named**, and there is **no published elevation
pattern**. **[datasheet]** Ignore the "±35° vertical" figure that circulates online — I
could not source it to Hi-Link.

### The picture

```
  SIDE VIEW                                              PLAN VIEW (from above)
  ═══════════════════════════ ceiling  2.5 m           ┌─────────────────────────┐
              ▓▓▓ ← module, FLAT, 0° tilt              │  headboard wall         │
              ║                                        ├─────────────────────────┤
              ║                                        │ ┌───────┬───────┐       │
              ║  1.75 m   (must be ≥ 1.55 m)           │ │       │  ▓▓   │       │  ▓ = module,
              ║           gate 2                       │ │       │       │       │      over HIS
              ║                                        │ │ her   │  his  │       │      sternum
              ▼ ← sternum                              │ │ side  │  side │       │
      ┌───────●───────────────┐  0.75 m                │ │       │       │       │  0.6–0.7 m
      │▂▂▂▂▂▂▂▂▂▂ mattress ▂▂▂│  0.55 m                │ └───────┴───────┘       │  from the
  ════╧═══════════════════════╧═══════ floor           │                         │  headboard
      ↑                       ↑                        │      bed                │  wall
   headboard                foot                       └─────────────────────────┘
```

**Nothing in that cone may move all day.** Walk section 5 before you drill.

---

## 2. Runner-up: foot-of-bed wall, and when to take it

**Take it if any of: there is a ceiling fan, there is a ceiling air diffuser, the ceiling is
below 2.4 m, or the ceiling run is genuinely impossible.** Do not compromise on the
ceiling mount — move to this one instead.

* **Position:** foot-of-bed wall, or a chest of drawers at the foot.
* **Height:** 1.2–1.4 m up.
* **Tilt:** 15–20° down, boresight landing on your chest.
* **Slant to sternum:** ~1.8 m horizontal + 0.45–0.65 m drop ≈ **1.86–1.91 m** — gate 2,
  clear of the 1.5 m floor.
* **What it buys:** it is **aimable**, which the ceiling mount is not (Figure 8 is a 360°
  azimuth pattern **[datasheet]**). If you share a wall with a corridor or a neighbour,
  this is the only mount that lets you point away from it. It also covers the rest of the
  room, not just the bed.
* **What it costs:** it looks along the body's long axis, so your chest wall is nearly
  edge-on — the weakest aspect for respiration sensing. **[inference]** And its boresight
  goes through the headboard wall into whatever is behind it.

**Do not use the headboard wall.** Mounted 1.6 m up with your sternum 0.6 m down the bed
and 0.75 m off the floor, the slant is **1.04 m — gate 1, where static sensitivity is not
settable.** **[datasheet]** It will hold you while you fidget and drop you when you settle:
a PIR with extra steps. Several published guides recommend exactly this; they are wrong for
still-holding. **[inference]** One research lane did offer "high on the headboard wall,
tilted down" as an acceptable second choice — that follows from the 0.75 m reading of the
still floor, which §0 rejects.

**Do not use a side wall or a nightstand.** Usually under 1.5 m, and worse, the aspect
angle **flips with sleeping posture** — good for a side-sleeper, near-null for a supine
one. **[inference]** An unattended overnight sensor must not depend on which way you
rolled.

---

## 3. What to change in `scripts/esphome/jarvis-room-sensor.yaml`

Two of these are YAML edits you must make before flashing. The rest are knobs on the
device's own web page — change them there, watch the effect live, and leave the YAML alone.
Settings **persist across power-off** **[datasheet]**, and ESPHome's `setup()` only *reads*
the module **[vendor doc]**, so nothing you set is lost on a reflash.

### YAML, before you flash

**A. Add engineering mode and the per-gate energies.** Without these, *"which gate is the
fan in?"* and *"what is my actual still energy in bed?"* are unanswerable and tuning
degenerates to guessing. The current file has no `switch:` block and no gate sensors.

```yaml
switch:
  - platform: ld2410
    ld2410_id: radar
    engineering_mode:
      name: Engineering mode      # turn ON to tune, OFF when done

sensor:
  - platform: ld2410
    ld2410_id: radar
    # ...keep the existing moving_distance / still_distance / detection_distance...
    g1: {still_energy: {name: g1 still energy}}
    g2: {still_energy: {name: g2 still energy}, move_energy: {name: g2 move energy}}
    g3: {still_energy: {name: g3 still energy}, move_energy: {name: g3 move energy}}
    g4: {still_energy: {name: g4 still energy}}
    g5: {still_energy: {name: g5 still energy}}

number:
  - platform: ld2410
    ld2410_id: radar
    # ...keep timeout / max_move_distance_gate / max_still_distance_gate...
    g2: {still_threshold: {name: g2 still threshold}}
    g3: {still_threshold: {name: g3 still threshold}}
```

Gate energies only report while engineering mode is on. **[vendor doc]** *If the compile
rejects a key, check the current names at esphome.io/components/sensor/ld2410 — I could not
compile this here.*

**B. Add the boot-time read**, so the web page shows the real stored values after a power
cycle instead of zeros. A user spent a day thinking his settings had been wiped; they had
not, HA simply was not querying them. **[community]**

```yaml
esphome:
  name: ${device_name}
  on_boot:
    priority: 600
    then:
      - lambda: 'id(radar).restart_and_read_all_info();'
```

### On the device web page, in this order

| Web page name | From | To | Why |
|---|---|---|---|
| **Absence delay** (`timeout`) | 5 s | **60 s** for night 1, then **300 s** | The single most valuable change. It is not cosmetic smoothing here — it is the defence against the latch described below. ESPHome accepts 0–65535 s. **[vendor doc]** Costs Jarvis nothing, because `room-sensor.md` §7 already forbids the radar from asserting absence at all. |
| **Max move gate** | 8 | **8** (leave it) | A through-wall *moving* target lasts seconds and the absence delay eats it. Keep whole-room motion — it is what re-arms the still bit. |
| **Max still gate** | 8 | **3** to start | A through-wall *still* target pins presence ON forever and never clears. This is the fence that matters. Tighten only after §4 Phase 1 tells you which gate you actually land in. |
| **g4–g8 still threshold** | 30/30/20/20/20 | **100** | Hi-Link's own technique: *"if the sensitivity of a certain distance gate is set to 100, the effect of not recognizing the target under the distance gate can be achieved."* **[datasheet]** Blinds the far gates to still targets — the corridor, the neighbour, the curtain. |
| **g2 / g3 still threshold** | 40 / 30 | **below your measured motionless energy** | The factory defaults are 2–3× too high to hold a motionless adult. The only field measurement anyone published: a completely still person reads **9–12** at gates 4–6, against factory thresholds of 30/30/20 there; the fix that worked was setting them to 8. **[community]** That is a stranger's number in a stranger's room — measure your own in Phase 1 and set below it. |
| **Distance resolution** | 0.75 m | leave at 0.75 m | The 0.2 m mode caps total range at 1.6 m **[datasheet]** — bedside-only, and it would put your fence inside the bed. |

**Free lever you already own and are not using:** the LD2410C's Bluetooth app
(**HLKRadarTools**, password `HiLink`, firmware V2.44+) has an automatic background-noise
calibration that ESPHome does not expose. Leave the room, press it, it learns the empty
room's clutter and sets its own thresholds. A user who hand-tuned for three days said the
auto-calibration beat him. **[community]** It survives a reflash. Try it before you
hand-tune.

### The one thing you cannot fix with settings

Four independent reports say **the LD2410's still bit will not re-assert on still energy
alone — once presence has cleared, a *moving* target must re-trigger it.** **[community]**
One is a controlled test (motion disabled on all gates → static stayed false despite static
energy over threshold; re-enabling motion made the same energy trigger correctly); one is an
open, uncommented ESPHome issue (#17620, July 2026). For a sleeper that is the difference
between a two-second dropout and a six-hour one.

I verified this cannot be patched in ESPHome: the component is a pure relay of protocol
byte 8 and does no thresholding of its own, and `LD2410Component::setup()` only reads.
**[vendor doc]** So the behaviour lives in module firmware.

**Two mitigations, in order:**

1. **The 300 s absence delay.** If presence never clears, you never need to re-acquire.
   This is free and it is why the timeout matters more here than anywhere else.
2. **If §4 says that is not enough:** build the bedroom's bit as an ESPHome template binary
   sensor OR-ing the per-gate still energies against your own thresholds, instead of
   reading `has_target`. A user has already done exactly this. **[community]** Don't
   pre-build it — measure first.

A sleeper does move: turning over, limb twitches. The latch only hurts if the gap between
movements exceeds what the delay covers. **That is precisely what the overnight log
measures.** **[inference]**

**Do not copy these settings to the office or living room.** A Max still gate of 3 there
would drop you reading in a chair at 4 m — the exact failure you bought this sensor to
avoid. Those rooms are the wall-mount case your existing doc already describes correctly.

---

## 4. The first night: what to log, and what PASS looks like

**Phase 0 — bench, 5 minutes, before anything goes on a ceiling.** Flash, set the static
IP, confirm it joins, and confirm the endpoints answer:

```bash
curl -s http://192.168.50.60/binary_sensor/presence
curl -s http://192.168.50.60/binary_sensor/still_target
curl -s http://192.168.50.60/sensor/still_distance
```

Everything after the first USB flash is OTA — which is what makes a ceiling mount
survivable. Do not mount an unverified board.

**Phase 1 — the ten minutes that save the night.** Mount it. Turn **Engineering mode ON**.
Lie still on the bed, awake, for **ten minutes**, watching the device's own web page on your
phone. Record:

* **`Still target`** — does it stay ON for all ten minutes?
* **`Still distance`** — the number. **This sets your Max still gate.**
* **`g2`/`g3` still energy** — the lowest value you see. **This sets your still thresholds
  (set them below it).**

**If it cannot hold you awake-and-still for ten minutes, it will not hold you asleep** — and
you find out now, standing up, rather than at 07:00.

**Phase 2 — empty-room baseline, 30–60 minutes. Do this BEFORE the sleep test.** A sensor
that says ON in an empty room passes a sleep test for the wrong reason. Leave the room with
everything running as it will be at night — fan, HVAC, humidifier, door in its usual
position, pet free to do what it does.

* **PASS:** presence OFF for **≥95%** of the hour; any ON bursts short and explainable.
* **FAIL:** presence pinned ON. Stop. The sleep test proves nothing until you find the
  moving thing with the per-gate energies.

**Phase 3 — the overnight log.** Set **Absence delay to 60 s** for this night (not 300 —
you want to *see* the real gaps). Two streams, both started before bed:

```bash
mkdir -p ~/jarvis-commissioning

# PRIMARY - through the real driver, so it tests the endpoint Jarvis actually reads
setsid nohup ~/vss_env/bin/python scripts/roomsensor_stub.py watch http://192.168.50.60 \
  > ~/jarvis-commissioning/watch-$(date +%F).log 2>&1 &

# SECONDARY - the range check, every 10 s
while true; do
  printf '%s ' "$(date -Is)"
  curl -s --max-time 2 http://192.168.50.60/sensor/still_distance || echo '{"error":true}'
  echo
  sleep 10
done >> ~/jarvis-commissioning/still-$(date +%F).log &
```

`watch` is the right primary because it already prints three distinct states — `SOMEONE`,
`empty`, `no answer` — which is what separates *"the radar dropped him"* from *"the Wi-Fi
dropped the log"*, the classic trap in this measurement. **[vendor doc: your own
`scripts/roomsensor_stub.py:149-172`]** It prints only on transitions, so a good night is a
two-line file.

**Phase 4 — PASS, in actual numbers.** Over a 23:30–07:00 window (27,000 s):

| | PASS | FAIL |
|---|---|---|
| `watch` log | **two lines**: `SOMEONE` at bedtime, `empty` in the morning `(after ~27000s of SOMEONE)` | more than a handful of transitions |
| Presence ON | **≥99%** of the window | below 95% |
| Longest OFF episode | **≤30 s.** A few sub-10 s blips at turn-overs are normal and Jarvis's hysteresis eats them. | **any single OFF episode over 2 minutes** |
| `still_distance` | within **one gate (±0.75 m)** of your Phase-1 figure for **>90%** of samples — e.g. 150–225 cm for a 1.75 m mount | parked somewhere that is not the bed |
| Morning `empty` edge | arrives within the absence delay of you leaving, and stays | |

**The single number that decides everything is the longest continuous OFF stretch while you
are demonstrably in bed.** Bounded at a minute or two → raise the absence delay to 300 s and
you are finished, with $9 of hardware. Running to tens of minutes → no threshold will save
it; change the geometry, then the part.

**Diagnosing a FAIL:**

* **Multi-minute OFF episodes, clustered** → you rolled out of the fenced gate, or the duvet
  came up. Widen Max still gate by one, or drop the g2/g3 still threshold, or re-centre the
  module over your torso. Not a sensor fault.
* **Presence never goes OFF, including after you leave** → a permanent moving target. Phase 2
  should have caught it. Go hunt it with the per-gate energies.
* **`still_distance` parked away from the bed** → a ghost. If it sits near *twice* the
  mirror distance, it is the mirror.
* **Presence ON but `Still target` false all night** → presence is coming from motion only.
  You have built a PIR. Check the 1.5 m floor first — this is exactly what §0 predicts.
* **`no answer` lines around every gap** → network or device, not radar. The night is void;
  re-run.

**Run it twice**, and make at least one of the two nights alone — sleeping posture varies,
aspect angle depends on posture, and a partner makes the result about the bed rather than
about you.

---

## 5. Walk the room before you drill

Stand where the module will go and look down its cone. Anything that moves all day is a
permanent target — this is the top false-positive risk of the whole build, and Hi-Link names
most of the list itself. **[datasheet]**

- [ ] **Ceiling fan.** **Disqualifying for a ceiling mount** — large periodic Doppler at
      fixed range, for hours. Take the runner-up placement instead and aim so the fan is
      above the beam. Hi-Link warns about fans *"on the top of the room"* even for a wall
      mount. **[datasheet]**
- [ ] **Ceiling air diffuser / HVAC vent.** Disqualifying overhead. Elsewhere: range-fence
      it out if it is further than the bed, or aim past it.
- [ ] **Curtain over a vent or draught.** Literally the datasheet's *"continuously swinging
      curtains"*. **[datasheet]**
- [ ] **Oscillating fan.** Worse than a ceiling fan — it sweeps *across* gates, so no single
      gate threshold catches it. It must be out of the beam or off.
- [ ] **Radiator convection plume**, and anything the plume moves.
- [ ] **Hanging plant, pendant lamp, curtain-track top** — anything that can sway.
- [ ] **Humidifier.** Probably fine — an aerosol plume is a low-RCS cloud and I found no
      report of one triggering an LD2410 **[inference]** — but check for an oscillating
      nozzle, and check the plume is not fluttering a curtain. Have it running during Phase 2.
- [ ] **Mirror or mirrored wardrobe door.** Does not break presence; **breaks range
      fencing.** A silvered mirror is near-perfect at 24 GHz, so expect a ghost of you at
      roughly twice the mirror distance — possibly in a gate you fenced out, or through a
      wall. Choose your fence *after* Phase 1 with the mirror in place. **[inference]**
- [ ] **Large flat TV.** Not the picture (that is light, not motion), but a big panel is a
      *"large area of strong reflectors"*, which the datasheet names as an interference
      source. **[datasheet]** From the ceiling this is automatic.
- [ ] **What is behind / above the module.** 24 GHz passes plasterboard, so the back lobe
      sees the flat upstairs. Foil tape or an aluminium plate against the ceiling. **[datasheet]**
- [ ] **What is beyond the far wall.** A corridor, a stairwell, a neighbour's hallway. The
      ceiling mount cannot aim away from it — only Max still gate and the 100s on g4–g8 fence
      it. If the party wall is closer than ~1.5 m, range-fencing is unavailable and only §6's
      software rule remains.
- [ ] **Metal bed frame.** Mostly fine — static, below you, removed by clutter processing.
      Two caveats: it can raise the specular return in that gate **[inference]**, and a frame
      that **rattles** when you turn over is a genuine moving target. Fix that with a spanner.
- [ ] **Metallised bedding.** A foil emergency blanket or a heated blanket with a wire mesh
      is a reflector, not a window. Ordinary textiles are transparent at 24 GHz — eight
      common fabrics measured *"usefully transparent"* up to 300 GHz **[community: Bjarnason
      et al., Appl. Phys. Lett. 85(4), 2004]**. A thick lofted duvet damping the *mechanical*
      displacement is my assumption and is **unmeasured**. **[inference]**
- [ ] **Nothing within 0.75 m of the module at all** — that is the blind zone. Including a
      person standing on the bed under a 2.5 m ceiling (~0.6 m away: invisible).
- [ ] **Power.** 5 V **1 A minimum** — the radar averages 79 mA **[datasheet]** but an ESP32
      Wi-Fi TX peak is ~250 mA, and a charger that sags presents as a flaky sensor, not as a
      power fault. Its own wall socket, not a Spark USB port. Budget **4–5 m** of cable for a
      ceiling run; buy one 5 m lead, not two joined.
- [ ] **Keep the ESP32 at the ceiling with the radar.** 256000 baud over metres of unshielded
      DuPont is where this build fails silently and unrepeatably. Keep the UART at 10 cm and
      lengthen the 5 V instead.

---

## 6. What this sensor cannot tell you

`has_target` is **one Boolean, an OR over everything alive in the fenced volume**.
`still_distance` reports **one** number — the dominant still target — so two people 1.70 m
and 1.75 m from the module produce a single figure that jitters between them. The protocol
frame has one moving distance, one still distance, one detection distance. **[datasheet]** No
gate setting, no ESPHome option and no firmware version changes this.

**Jarvis CAN conclude:** something alive is in the fenced bed volume. That is genuinely
valuable, and it is exactly what the phone leg is bad at.

**Jarvis CANNOT conclude:** that it is Hunter. That he is alone. That he is asleep. That he
left and someone else stayed. That the shape is human at all — a cat asleep on the duvet is
a still target in the same gate as you, breathing, and **no threshold separates them.** A HA
user spent weeks tuning distance, dwell time and ignore regions to stop a 40 cm dog reading
as occupied, and concluded *"it's clearly not that simple"*. **[community]**

**The design rule that makes every misdetection safe:**

> **The bedroom bit may only ever SUPPRESS, never AUTHORISE.**

Then a neighbour bleeding through plasterboard makes Jarvis quieter, never louder or
wronger. Applied:

* ✅ **"He's gone to bed → go quiet"** — safe. Being wrong in the occupied direction only
  makes Jarvis quieter. Fail-quiet.
* ❌ **"The bedroom is empty → he must be out"** — unsafe twice over: the radar can drop a
  still sleeper (§3), *and* `RoomOrPhone.__call__` ends `return False if seen is False else
  None` at `presence.py:184`. **With no phone leg configured, an empty room *does* become
  away, and `quiet.hold_when_away` fires.** `presence.phone_ip` is empty on this box.
  **Fix that or set `phone_ip` before you flash.**
* ❌ **Anything that reads personal content aloud because "the bedroom is occupied, so it's
  just him."** The radar is an occupancy signal, never an identity signal. Identity stays
  with speaker verification.
* ⚠️ **A "morning briefing when he stirs" keyed on `has_moving_target`** will also fire for
  the cat and for a partner.

A cheap detector for the silent failure — a phantom stuck ON is indistinguishable from a man
asleep, and survives every health check that only asks whether the sensor answers: **log the
ON duty cycle over 24 h.** A real bedroom is occupied perhaps 30–45% of a day. **≥99% ON for
24 h is a phantom, not a man.** **[inference]**

---

## 7. Do not buy a different part yet

The **LD2412** has one genuinely large, bed-relevant advantage: **dynamic background
correction** — an ESPHome button that learns the empty room's clutter and sets its own
thresholds, which is precisely the hard problem here and which the LD2410's ESPHome
component has never had (open feature request since May 2025). It also adds a **minimum**
distance gate, 14 gates to 9 m, and ±75°. **[vendor doc]**

**But nobody has published an LD2410-vs-LD2412 bed test**, and I could find **no first-person
full-night in-bed LD2410 log from anyone** — every page ranking for "LD2410 bed presence" is
vendor or SEO content asserting success with no data, and Reddit, where the first-hand
reports live, is unreachable from this machine. Recommending a part swap on a spec-sheet
inference is the move that has misfired on this project twice. **[inference]**

**Run the overnight log first.** If it fails, buy **one** LD2412 for the bedroom (~$6–8) and
keep the LD2410Cs for the other two rooms, where the wall geometry is exactly the "point it
across the room like a small speaker" case your existing doc describes correctly.

And a use for a spare LD2410C that routes around the firmware weakness with hardware you
already own: **put one at the bedroom door as a crossing detector.** Moving-target detection
is the half of the LD2410 that has never been in doubt. A door radar gives Jarvis an
independent *"he came in and has not left"* edge that stays true even when the bed radar's
still bit drops — which is the failure the field reports say to expect.

---

## Open risks, stated plainly

1. **Nothing here was measured on hardware.** No radar is attached to this machine.
2. **The 1.5 m still floor is documented but unconfirmed in the field.** It is Hi-Link's own
   *"(not settable)"* table entry, which is strong, but no community report independently
   confirms *"the LD2410 will not hold a still person inside 1.5 m"*. If it turns out to be
   a UI restriction rather than a detection limit, the headboard mount comes back and §2's
   ranking changes. **Phase 1 tests it directly in ten minutes.**
3. **The re-acquisition latch (§3) rests on four community reports**, one of them an open and
   uncommented ESPHome issue. It is consistent with the ESPHome source, which I did verify is
   a pure relay — so the behaviour must be in module firmware. But Hi-Link documents it
   nowhere and I could not reproduce it. If it is firmware-version-specific, the template-
   sensor workaround is unnecessary.
4. **The 9–12 still-energy figure is a single forum post** about an LD2410B, in someone else's
   room, with an unstated mount and no methodology. It is the only real number anyone
   published and I have leaned on it heavily. Treat it as an order of magnitude, never as a
   threshold to copy.
5. **The polar-plot metres were read off images by eye**, ±0.2–0.3 m, and Figure 8 does not
   state whether its radius is floor radius or slant range. The within-figure *ratio* —
   still-range vs move-range — is the robust part, and it is what the recommendation rests on.
6. **The aspect-angle argument is an extrapolation.** The measurement behind it (SPIE 9461,
   2015 — respiration signatures *"smaller from the sides"*) used 500 MHz–18 GHz on an upright
   subject on a turntable. Applying it to a supine sleeper at 24 GHz is my inference. It
   points the same way as Hi-Link's own two figures, which is why I trust the direction and
   not the magnitude. It is also weakened by side-sleeping, where the chest's largest
   displacement axis rotates toward horizontal.
7. **Duvet damping is unmeasured** and my assumption. Fabric *transparency* at 24 GHz is well
   documented; whether a thick lofted duvet reduces the detectable surface displacement is not.
8. **Hi-Link contradicts itself** on max-range (5 m in the introduction, 6 m in the spec
   table), on where a max-gate fence lands (by one full gate), and on the distance-resolution
   default (its table assigns 0x0001 to 0.2 m; the next line says 0x0001 is 0.75 m). Resolve
   all three on-device with `Still distance`, not by reading.

---

**Change to make in the sibling doc:** `docs/room-sensor.md:223-225` should become
room-specific rather than universal. The chest-height line is right for a desk and wrong for
a bed, against the LD2410C's own manual.
