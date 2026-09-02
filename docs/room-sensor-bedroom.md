# Bedroom placement: putting the radar where it can hold a sleeper

**Put it on the ceiling, flat, directly above your sternum — *not* at chest height on the
headboard wall, which is what `docs/room-sensor.md:223` currently tells you, and *not* on
the nightstand, which is what you are going to try first.** Both of those put your chest
inside the band where this module does no still-target detection at all, and both fail the
same way: they hold you while you fidget and drop you when you settle.

**If you will not drill: skip to §2b.** A box on the chest of drawers at the *foot* of the
bed works. A box on the nightstand beside your pillow does not, and §2a is the arithmetic.

> **Confidence key**, on every claim.
> **[datasheet]** a number, table or figure in Hi-Link's LD2410C manual or the LD2410
> serial-protocol PDF. **[vendor doc]** Hi-Link prose, or ESPHome's own docs/source.
> **[community]** a field report from a stranger, named. **[inference]** reasoning, mine,
> unmeasured.
>
> **Nothing here was measured on hardware.** There is no radar attached to this machine.
> Every distance is Hi-Link's, a stranger's, or arithmetic. Section 4 exists because of
> that: the overnight log is the only thing that settles this.
>
> **And Hi-Link never measured a sleeper.** Both range figures carry exactly three
> legends — Move, Stand, Sit still. The word "lying" appears twice in the whole manual,
> both times in a marketing bullet ("can also sensitively sense static, micro-moving,
> sitting and lying human bodies"; "can detect and identify the human body in motion,
> fretting, standing, sitting and lying down") **[vendor doc]**. A sitting measurement is
> the closest thing that exists to a sleeping one, and this document treats every
> sit-still number as a *stand-in*, not a sleep spec.

---

## 0. The two numbers that decide the placement

**1. Nothing at all inside 0.75 m.** Table 2: *"Detection distance 0.75m ~ 6m,
adjustable"* **[datasheet]**. Closer than that and the module is blind, full stop.

**2. No still-target detection inside 1.5 m.** Table 7 of the serial-protocol PDF lists
*"Rest sensitivity for distance gate 0"* and *"…gate 1"* as **"-(not settable)"** — not a
threshold of zero, but not implemented. **[datasheet]** Inside 1.5 m the module is a
motion sensor, i.e. a PIR, i.e. the thing you rejected.

Neither number appears anywhere in `docs/room-sensor.md`. Between them they disqualify the
headboard mount that most published guides recommend (§2b) **and** the nightstand you were
about to pick (§2a).

**Where the two research lanes disagreed, and what I picked.** One lane read gates 0 and 1
as both covering 0–0.75 m (citing a forum post), making the still floor 0.75 m and a
headboard mount viable. The other read gate *N* as spanning 0.75×*N* to 0.75×(*N*+1),
making the floor 1.5 m. **I picked 1.5 m**, because Hi-Link's own worked example says
gates 3 and 4 together cover *"2.25-3.75m"* **[datasheet]** — which only works if gate *N*
starts at 0.75×*N*, so gate 1 is 0.75–1.5 m. A manufacturer's worked example beats a forum
post. It is also the conservative choice: the recommended mount below clears 1.5 m anyway,
so picking the stricter reading costs nothing. (Hi-Link contradicts itself two paragraphs
earlier — *"if the farthest door is set to 2, only if there is a human body within 1.5m"*
implies gate *N* ends at 0.75×*N* **[datasheet]**. Resolve it on the device with
`Still distance`, not by reading.)

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
| **Behind it** | a metal plate or a patch of foil tape between module and ceiling; 60×60 mm is my guess at enough **[inference]** | Hi-Link's own remedy: *"A metal shield or metal backplane can be used to shield the radar back lobe"* **[datasheet]** — and free at this mount |
| **Enclosure** | non-metallic; module held **12.4 mm (±1.2 mm)** off the inner face; wall thickness **3.9 mm (±0.8 mm) in ABS** — or under 1 mm. **Never ~2 mm.** | see the derivation below **[datasheet]** |
| **Fixing** | screwed or VHB-taped. Not dangling on its cable. | *"the shaking of the radar itself will affect the detection effect"* **[datasheet]** |

**The enclosure numbers, derived** — because the first draft of this document got the wall
thickness backwards, and a lid that eats range does it silently.

λ at 24.125 GHz = 299 792 458 / 24.125×10⁹ = **12.43 mm**.

* **Standoff H** (antenna face → inner surface of the lid). §8.3: *"1 times or 1.5 times
  the wavelength … For example, 12.4 or 18.6mm is recommended for 24.125GHz … Error
  control: ±1.2mm"* **[datasheet, verbatim]**. §8.2 gives the general rule, H = (m/2)·(c₀/f),
  *"its half wavelength in air is about 6.2mm"* — so 12.4 mm is m=2, and **6.2 mm is a
  legal fallback if 12.4 mm will not fit** **[datasheet]**.
* **Wall thickness D** is a *different rule in a different medium*: D = (m/2)·(c₀/(f·√εᵣ)),
  *"For example, a certain ABS material εᵣ=2.5, its half wavelength is about 3.92mm"*,
  *"Recommended half wavelength, error control ±20%"* **[datasheet]**. So for ABS:
  12.43/√2.5 = 7.86 mm in the medium, half of that = **3.93 mm, ±20% → 3.1–4.7 mm**. Next
  legal value up is 7.9 mm.
* **The fallback**, and only if you cannot hit that: *"It is recommended to use low
  materials … Thickness recommended 1/8 wavelength or thinner"* **[datasheet]** — an eighth
  of the wavelength *in the medium*, so **≤0.98 mm for ABS**, not the 1.55 mm you get by
  doing that arithmetic in air.
* **Therefore the thing to avoid is the middle**, and the middle is exactly a cheap
  project box: **a ~2 mm ABS wall is the quarter-wave thickness** (7.86/4 = 1.96 mm),
  which is the maximum-reflection case, not the minimum **[inference, from the datasheet's
  own formula]**. Either machine the lid to ~3.9 mm, or use something genuinely thin.

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
measured range plot (Figures 7 and 8), and the feature list says *"Supports various
installation methods such as ceiling hanging and wall hanging"*. **[datasheet]** Both
research lanes reached this independently; there was no disagreement.

More than sanctioned — it is the *better* geometry for a still target, by Hi-Link's own two
measurements. **These are measured off the PDF, not eyeballed:** page 10 and page 11
rendered at 400 dpi, the grey polar grid least-squares-fitted (Figure 10: five circles at
112.1 / 224.6 / 336.5 / 451.2 / 562.8 px → **112.5 px per metre**; Figure 8: an octagon
whose vertices fall at 120 / 240 / 360 / 480 / 600 px along the labelled bearings →
**120.0 px per metre**, cross-checked against 111 px at the 22.5° edge bisectors =
120·cos 22.5°), then each of the three curve colours isolated and its outer radius taken
per bearing.

**Wall at 1.5 m (Figure 10)** — metres, dashes where the orange line is hidden under the
teal:

| bearing | −55° | −40° | −20° | **0°** | +20° | +40° | +55° |
|---|---|---|---|---|---|---|---|
| Motion | 4.85 | 4.57 | 4.21 | **4.92** | — | 4.74 | — |
| Stand up | 5.06 | 5.01 | 4.80 | **4.77** | 4.78 | 4.58 | 4.38 |
| **Sit still** | 4.29 | 4.23 | 3.99 | **3.96** | 3.87 | 3.91 | 4.30 |

Sit-still is the **innermost** curve at every bearing. And the pattern is a *fan with a
hard edge*, not a taper: all three hold near their boresight value out to about **±56°**
and then drop to the origin. That matches Table 2's *"Detection angle ±60°"*
**[datasheet]** and is worth knowing — off-axis costs you almost nothing until it costs you
everything.

> **Correction.** An earlier draft of this document said *"all three collapse to ≈2.6 m at
> ±60°"* and tagged it [datasheet]. **There is no 2.6 m anywhere in Figure 10.** The number
> was invented, and it happened to exaggerate how bad the wall mount is — i.e. it leaned
> toward this document's own conclusion. The table above replaces it. That draft's boresight
> figures held up (it said 4.9 / 4.6 / 4.0; measured 4.92 / 4.77 / 3.96), and so did its
> ceiling figures. Only the off-axis claim was fabricated — which is why it is called out
> here rather than quietly deleted.

**Ceiling at 3 m (Figure 8)** — an octagon, vertices on the labelled bearings, metres:

| bearing | 0° | 45° | 90° | 135° | 180° | 225° | 270° | 315° |
|---|---|---|---|---|---|---|---|---|
| Move | 3.88 | — | — | — | — | 3.67 | 3.39 | 3.60 |
| Stand | 3.78 | 3.99 | 3.98 | — | 4.08 | 3.79 | 3.28 | 3.47 |
| **Sit still** | **4.08** | **4.07** | **3.89** | **4.07** | **4.37** | **4.08** | **3.50** | **3.87** |

**Sit-still is the outermost curve, or tied, at 37 of the 38 bearings where all three can
be separated.** Near-circular, and the still channel wins by 0.2–0.4 m almost everywhere.

**That reversal is the whole argument.** On a wall, still-detection is the *weakest* mode;
on a ceiling it is the *strongest*. In a bedroom, still-detection *is* the product.

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

## 2a. The nightstand: no. Here is the arithmetic.

You said you would probably stand it on the nightstand pointing at yourself. It is the
obvious thing to try, it needs no drill, and **it is the one mount that fails twice over.**

Take the standard sizes and measure yours against them:

| | |
|---|---|
| nightstand top | 0.55–0.70 m; call the module centre **0.65 m** |
| your sternum, supine | mattress top 0.55 + ~0.20 = **0.75 m** |
| height difference | **0.10 m** — negligible, so slant ≈ horizontal |
| nightstand → mattress edge | ~0.10 m |
| mattress edge → your sternum | 0.40–0.50 m |

**Nightstand on your side:** horizontal ≈ 0.10 + (0.40 to 0.50) = **0.50–0.60 m**, and the
slant is the same to two decimal places (√(0.55² + 0.10²) = 0.56 m). **That is inside the
0.75 m blind zone** — the module does not see you at all, awake or asleep, moving or still.
Sleep further from the edge and you cross into gate 1, which is the next paragraph.
**[datasheet]**

**The far nightstand, across a 1.4 m double:** horizontal ≈ 0.10 + 1.4 − 0.45 =
**1.05 m** → **gate 1**, where rest sensitivity *"-(not settable)"*. Moving targets only. It
will hold you while you read and drop you the minute you settle. **[datasheet]**

**To clear the 1.55 m working minimum you need 1.55 m of horizontal separation from your
chest.** Every nightstand geometry lands between 0.5 m and 1.1 m — **blind at the near end,
PIR at the far end, and nothing in between is a nightstand any more.** It misses by 0.5–1.0
m, and it misses in the way that is hardest to notice, because it works perfectly during
the five minutes you stand there testing it.

**What the nightstand *is* good for:** moving targets, which is the half of the LD2410 that
has never been in doubt. A nightstand unit is an excellent **"he got up"** detector, and
§7 already wants a second moving-only radar for exactly that job. Use it there, not as the
thing that holds you asleep.

**If you are going to try it anyway — and you are — §4 Phase 1 costs ten minutes and
settles it.** Lie still and awake, watch `Still target` on the device's own web page. If it
goes false while you are lying there in front of it, that is the answer, and it does not
improve when you fall asleep. Do that *before* you run a night on it.

## 2b. The no-drill mount that works: foot of the bed

**Also take this if:** there is a ceiling fan, there is a ceiling air diffuser, the ceiling
is below 2.4 m, or the ceiling run is genuinely impossible.

* **Position:** the chest of drawers, shelf, bookcase or chair at the **foot** of the bed —
  a nightstand-shaped object in the right place. No drill, no cable run across a ceiling.
* **Height:** 1.2–1.4 m up. Stack books under it if the furniture is low.
* **Tilt:** 15–20° down, boresight landing on your chest.
* **Slant to sternum:** ~1.8 m horizontal + 0.45–0.65 m drop = √(1.8² + 0.45²) to
  √(1.8² + 0.65²) = **1.86–1.91 m** — gate 2, clear of both floors.
* **What it buys:** it is **aimable**, which the ceiling mount is not (Figure 8 is a 360°
  azimuth pattern **[datasheet]**). If you share a wall with a corridor or a neighbour, this
  is the only mount that lets you point away from it. It also covers the rest of the room,
  not just the bed.
* **What it costs you against the ceiling:** it looks along the body's long axis, so your
  chest wall is nearly edge-on — the weakest aspect for respiration sensing
  **[inference]**. Expect a *smaller* still-energy margin than the ceiling would give, so
  §4 Phase 1 matters more here, not less. And its boresight goes through the headboard wall
  into whatever is behind it.

**Do not use the headboard wall.** Mounted 1.6 m up with your sternum 0.6 m down the bed
and 0.75 m off the floor, the slant is √(0.6² + 0.85²) = **1.04 m — gate 1, where rest
sensitivity cannot be set.** **[datasheet]** Same failure as the nightstand. Several
published guides recommend exactly this; they are wrong for still-holding. **[inference]**
One research lane did offer "high on the headboard wall, tilted down" as an acceptable
second choice — that follows from the 0.75 m reading of the still floor, which §0 rejects.

**Do not use a side wall either.** Usually under 1.5 m, and worse, the aspect angle **flips
with sleeping posture** — good for a side-sleeper, near-null for a supine one.
**[inference]** An unattended overnight sensor must not depend on which way you rolled.

---

## 3. What to change in `scripts/esphome/jarvis-room-sensor.yaml`

### Which mistake this tuning chooses to make — read this before the table

The bedroom bit only ever **suppresses** (§6). So:

* **A false OCCUPIED** costs a held briefing and a late heads-up. Recoverable; he can
  always ask.
* **A false EMPTY at 03:00** is a voice in a dark bedroom. Not recoverable.

**So these settings deliberately err toward OCCUPIED.** That is the cheap mistake and it is
the right direction.

**But "err toward occupied" is not "pinned ON", and the first draft of this document
confused the two.** It left `Max move gate` at 8 — from a ceiling, a 6 m sphere through the
party wall, the floor above and every adjacent room — on the stated grounds that *"a
through-wall moving target lasts seconds and the absence delay eats it"*. **That is exactly
backwards.** Hi-Link:

> *"When the radar outputs the result from man to no man, it will report man for a period of
> time. If there is no man in the radar test range during this time period, the radar will
> report no man; **if the radar detects man during this time period, it will be refreshed
> again**."* **[datasheet]**

The delay does not absorb a transient — it **extends** it, and every fresh transient
restarts the clock. One neighbour walking past the party wall, at the 300 s that draft
recommended, is five minutes of "he's in bed"; a stream of them is permanently occupied. A
bit that is ON 24 hours a day carries no information, gates nothing, and fails this
document's own Phase-2 and 24-hour duty-cycle checks.

**So: err toward occupied, but bound it.** The corrected settings err by **at most one
absence delay past the last sign of life, inside a fenced volume that stops at the walls of
your own bedroom** — not by pinning ON off a neighbour six metres away. Concretely: the
range fence does the rejecting, the delay does only the bridging, and the delay is set from
a measured gap rather than a round number.

### YAML, before you flash

**A. Add engineering mode and the energies.** Without these, *"which gate is the fan in?"*
and *"what is my actual still energy in bed?"* are unanswerable and tuning degenerates to
guessing. The file has no *active* `switch:` block and no energy sensors.

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
    still_energy:                          # <- ALWAYS published, engineering mode or not
      name: Still energy
      id: still_e
    g1: {still_energy: {name: g1 still energy}}
    g2: {still_energy: {name: g2 still energy}, move_energy: {name: g2 move energy}}
    g3: {still_energy: {name: g3 still energy}, move_energy: {name: g3 move energy}}
    g4: {still_energy: {name: g4 still energy}}
    g5: {still_energy: {name: g5 still energy}}

number:
  - platform: ld2410
    ld2410_id: radar
    # ...keep timeout / max_move_distance_gate / max_still_distance_gate...
    g2: {move_threshold: {name: g2 move threshold}, still_threshold: {name: g2 still threshold}}
    g3: {move_threshold: {name: g3 move threshold}, still_threshold: {name: g3 still threshold}}
```

⚠️ **The file already contains a commented-out `switch:` block** (the MOSFET power kill at
the bottom). If you ever uncomment that, **merge the two — YAML will not accept two
`switch:` keys** and the failure is a parse error, not a warning.

The **top-level `still_energy`** is the one worth adding even if you never touch
engineering mode: in ESPHome's component it is published on every frame, above the
`if (engineering_mode)` branch in `handle_periodic_data_()`, whereas the per-gate `gN`
energies are only published inside it. **[vendor doc: esphome/components/ld2410/ld2410.cpp]**
That means you can log the still energy all night with engineering mode OFF — which is what
Phase 3 wants, and what mitigation 2 below is built on. *I could not compile any of this
here; if a key is rejected, check the current names at esphome.io/components/sensor/ld2410.*

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

Settings **persist across power-off** **[datasheet]**, and ESPHome's `setup()` only *reads*
the module — it is one line, `{ this->read_all_info(); }` **[vendor doc]** — so nothing you
set on the web page is lost on a reflash.

### On the device web page, in this order

Factory defaults below are Table 7 of the serial-protocol PDF **[datasheet]**.

| Web page name | From | To | Why |
|---|---|---|---|
| **Max still gate** | 8 | **3** to start | A through-wall *still* target pins presence ON and never clears. **This is the fence that matters — set it before you touch the delay.** Tighten or widen only after §4 Phase 1 tells you which gate you actually land in. |
| **Max move gate** | 8 | **still gate + 1** (so **4**) | ⚠️ *Changed from the first draft, which said "leave it at 8".* The move channel's only job here is to re-arm the still bit after the latch below; a moving target that matters is one in your bed. Gate 4 from a 1.75 m ceiling mount is 3.75 m of slant = a floor circle of radius √(3.75² − 1.75²) = **3.32 m** at chest height, which still covers a whole normal bedroom. Gate 8 is 6 m of slant = **5.74 m** of floor radius, which does not stay in the room — and every transient it collects gets multiplied by the absence delay. |
| **g4–g8 still threshold** | 30/30/20/20/20 | **100** | Hi-Link's own technique: *"if the sensitivity of a certain distance gate is set to 100, the effect of not recognizing the target under the distance gate can be achieved."* **[datasheet]** Blinds the far gates to still targets — the corridor, the neighbour, the curtain. |
| **g2 / g3 still threshold** | **40 / 40** | **your Phase-1 minimum − 5, floor of 3** | The factory defaults belong to Hi-Link's own test subject, who in Figures 9 and 10 is *sitting up on a stool* **[datasheet]**; a motionless adult under a duvet is a smaller signal than that **[inference]**. Do not copy a number from anywhere, including here — measure yours in Phase 1 and go under it. *(The first draft said the defaults were 40/30; Table 7 says 40/40.)* |
| **Absence delay** (`timeout`) | 5 s | **60 s**, and then whatever §4 Phase 4 measures | ⚠️ *Changed from the first draft, which said 300 s.* This bridges the gaps between a sleeper's movements — nothing else. It is not a filter, it is a stretcher (see above). Run the night at 60 s, then set it from §4 Phase 4's formula — **capped at 180 s**. The protocol field is two bytes, so 0–65535 s is legal **[datasheet]**; that is not a reason to use it. |
| **Distance resolution** | 0.75 m | leave at 0.75 m | ESPHome exposes `0.2m` / `0.75m` as a select **[vendor doc: ld2410.cpp]**. Eight gates at 0.2 m is 1.6 m of total range — bedside-only, and it would put your fence inside the bed. That is arithmetic, **[inference]**, not a datasheet line; the first draft tagged it [datasheet] and should not have. |

**The only reason 9–12 gets mentioned at all.** The single field measurement anyone
published: a completely still person read **9–12** at gates 4–6, against factory thresholds
of 30/30/20 there, and the fix that worked was setting them to 8. **[community]** That is
one forum post, about an **LD2410B**, in a room you have never seen, with an unstated mount
and no methodology. It is here as an order of magnitude and nothing else. **Do not copy 8.**
If your Phase-1 minimum is 30, set 25. If it is 6, set 3. If you never see a number at all,
engineering mode is off.

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

1. **The absence delay, sized to your measured gaps.** If presence never clears, you never
   need to re-acquire. This is why keeping `Max move gate` sane matters *more*, not less:
   the delay has to be long enough to bridge you, and a long delay is only affordable if the
   fence has already thrown out the neighbours.
2. **If §4 says that is not enough:** build the bedroom's bit as an ESPHome template binary
   sensor OR-ing `has_target` with your own threshold on `still_e` (the top-level
   `still_energy` added above), instead of reading `has_target` alone. Because that sensor
   publishes outside engineering mode, this needs **no** all-night engineering mode. A user
   has already done the per-gate version of this. **[community]** Two cautions: that energy
   is the *whole-target* figure and inherits your gate fence, so a bad fence turns this into
   a bit that never goes off; and it is an OR, so it can only ever make presence stickier,
   never less sticky. Don't pre-build it — measure first.

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
curl -s http://192.168.50.60/sensor/still_energy
```

Everything after the first USB flash is OTA — which is what makes a ceiling mount
survivable. Do not mount an unverified board.

**Phase 1 — the ten minutes that save the night.** Mount it, *or just hold it where you
are thinking of putting it*. Turn **Engineering mode ON**. Lie still on the bed, awake, for
**ten minutes**, watching the device's own web page on your phone. Record:

* **`Still target`** — does it stay ON for all ten minutes?
* **`Still distance`** — the number. **This sets your Max still gate.**
* **`g2`/`g3` still energy** — the lowest value you see. **This sets your still thresholds
  (go 5 below it, floor of 3).**

**If it cannot hold you awake-and-still for ten minutes, it will not hold you asleep** — and
you find out now, standing up, rather than at 07:00. **This is also the ten minutes that
settle §2a**: run it on the nightstand first if you must, and let the box tell you.

**Phase 2 — empty-room baseline, 30–60 minutes. Do this BEFORE the sleep test.** A sensor
that says ON in an empty room passes a sleep test for the wrong reason. Leave the room with
everything running as it will be at night — fan, HVAC, humidifier, door in its usual
position, pet free to do what it does.

* **PASS:** presence OFF for **≥95%** of the hour; any ON bursts short and explainable.
* **FAIL:** presence pinned ON. Stop. The sleep test proves nothing until you find the
  moving thing with the per-gate energies. **If it clears when you drop `Max move gate`
  from 8 to 4, it was never in the room** — that is the through-wall case, and it is why
  the fence changed.

**Phase 3 — the overnight log.** Set **Absence delay to 60 s** for this night — you want to
*see* the real gaps, not a number that hides them. Two streams, both started before bed:

```bash
mkdir -p ~/jarvis-commissioning

# PRIMARY - through the real driver, so it tests the endpoint Jarvis actually reads
setsid nohup ~/vss_env/bin/python scripts/roomsensor_stub.py watch http://192.168.50.60 \
  > ~/jarvis-commissioning/watch-$(date +%F).log 2>&1 &

# SECONDARY - range and energy, every 10 s
while true; do
  printf '%s ' "$(date -Is)"
  curl -s --max-time 2 http://192.168.50.60/sensor/still_distance || printf '{"error":true}'
  printf ' '
  curl -s --max-time 2 http://192.168.50.60/sensor/still_energy   || printf '{"error":true}'
  echo
  sleep 10
done >> ~/jarvis-commissioning/still-$(date +%F).log &
```

`watch` is the right primary because it already prints three distinct states — `SOMEONE`,
`empty`, `no answer` — which is what separates *"the radar dropped him"* from *"the Wi-Fi
dropped the log"*, the classic trap in this measurement. **[vendor doc: your own
`scripts/roomsensor_stub.py:149-172`]** It prints only on transitions, so a good night is a
two-line file. The energy trace in the secondary log is what tells you whether a dropout was
*"the energy fell below threshold"* or *"the latch fired"* — those need opposite fixes.

**Phase 4 — PASS, in actual numbers.** Over a 23:30–07:00 window (27,000 s):

| | PASS | FAIL |
|---|---|---|
| `watch` log | **two lines**: `SOMEONE` at bedtime, `empty` in the morning `(after ~27000s of SOMEONE)` | more than a handful of transitions |
| Presence ON | **≥99%** of the window | below 95% |
| Longest OFF episode | **≤30 s.** A few sub-10 s blips at turn-overs are normal and Jarvis's hysteresis eats them. | **any single OFF episode over 2 minutes** |
| | *Remember what you are reading: at the 60 s delay you ran this at, a 30 s logged OFF means your real worst still-gap was **90 s**.* | |
| `still_distance` | within **one gate (±0.75 m)** of your Phase-1 figure for **>90%** of samples — e.g. 150–225 cm for a 1.75 m mount | parked somewhere that is not the bed |
| Morning `empty` edge | arrives within the absence delay of you leaving, and stays | still ON an hour after you left → go back to Phase 2 |
| **24 h duty cycle** (day 2, not night 1 — it needs a whole day, working hours included) | **30–60% ON** | **≥99% ON is a phantom, not a man** (§6) |

The first two rows are about the 7.5 h in bed; the last is about the other 16.5 h. They are
not in tension — a good sensor is nearly always ON while you are in bed and mostly OFF while
you are not, and it is the *second* half that a tuning erring toward "occupied" quietly
breaks.

**Then set the absence delay from what you measured**, and only then. The delay you ran the
test at is already baked into every OFF episode you logged, so:

> **new delay = (longest logged OFF episode) + (the 60 s you tested at) + 30 s of margin,
> capped at 180 s.**

A 30 s worst OFF → your real gap was 90 s → **set 120 s**. **If the night produced no OFF
episodes at all, leave it at 60 s** — you have no evidence you need more, and every second
you add is a second of false-positive too. If the arithmetic wants more than 180 s, stop:
that is not a delay problem, it is §3's latch, and mitigation 2 is the answer.

**The single number that decides everything is that longest continuous OFF stretch.**
Bounded at a minute or two → set the delay and you are finished, with $9 of hardware.
Running to tens of minutes → no threshold will save it; change the geometry, then the part.

**Diagnosing a FAIL:**

* **Multi-minute OFF episodes, clustered** → you rolled out of the fenced gate, or the duvet
  came up. Widen Max still gate by one, or drop the g2/g3 still threshold, or re-centre the
  module over your torso. Not a sensor fault.
* **OFF episodes where `still_energy` was still healthy** → that is the latch, not the
  threshold. Go to mitigation 2; widening gates will not help.
* **Presence never goes OFF, including after you leave** → a permanent moving target, or a
  fence that reaches through a wall. Phase 2 should have caught it. Hunt it with the
  per-gate energies before you blame the sensor.
* **`still_distance` parked away from the bed** → a ghost. If it sits near *twice* the
  mirror distance, it is the mirror.
* **Presence ON but `Still target` false all night** → presence is coming from motion only.
  You have built a PIR. Check the 1.5 m floor first — this is exactly what §0 predicts, and
  exactly what §2a predicts if you tried the nightstand.
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
      fixed range, for hours. Take §2b instead and aim so the fan is above the beam. Hi-Link
      warns about *"electric fans on the top of the room"* even for a wall mount.
      **[datasheet]**
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
      sees the flat upstairs. Foil tape or a metal backplane against the ceiling.
      **[datasheet]**
- [ ] **What is beyond the far wall.** A corridor, a stairwell, a neighbour's hallway. Work
      out the slant, not the horizontal: a neighbour standing 2.0 m horizontally from a
      2.5 m ceiling mount, chest 1.2 m below it, is at √(2.0² + 1.2²) = **2.33 m — inside
      gate 3**, i.e. inside the fence this document recommends. **So if the bed is against a
      party wall, no fence excludes them.** Set g4–g8 to 100 anyway, take §2b so you can aim
      away, and fall back on §6's software rule. **[inference, from the datasheet's gate
      arithmetic]**
- [ ] **Metal bed frame.** Mostly fine — static, below you, removed by clutter processing.
      Two caveats: it can raise the specular return in that gate **[inference]**, and a frame
      that **rattles** when you turn over is a genuine moving target. Fix that with a spanner.
- [ ] **Metallised bedding.** A foil emergency blanket or a heated blanket with a wire mesh
      is a reflector, not a window. Ordinary textiles are transparent at 24 GHz — eight
      common fabrics measured *"usefully transparent"* up to 300 GHz **[community: Bjarnason
      et al., Appl. Phys. Lett. 85(4), 2004]**. A thick lofted duvet damping the *mechanical*
      displacement is my assumption and is **unmeasured**. **[inference]**
- [ ] **Nothing within 0.75 m of the module at all** — that is the blind zone
      **[datasheet]**. Including a person standing on the bed under a 2.5 m ceiling
      (chest ~0.6 m away: invisible). It is also what kills the nightstand, §2a.
- [ ] **Power.** 5 V **1 A minimum**. Hi-Link asks for *"DC 5V, power supply capacity
      >200mA"* and measures **79 mA average** **[datasheet]**, but an ESP32 Wi-Fi TX peak is
      ~250 mA, and a charger that sags presents as a flaky sensor, not as a power fault. Its
      own wall socket, not a Spark USB port. Budget **4–5 m** of cable for a ceiling run; buy
      one 5 m lead, not two joined.
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
wronger. Fail-quiet. Applied:

* ✅ **"He's gone to bed → go quiet"** — safe. Being wrong in the occupied direction only
  makes Jarvis quieter. This is the direction §3 deliberately errs in.
* ❌ **"The bedroom is empty → he must be out"** — unsafe in principle, because the radar can
  drop a still sleeper (§3). **In practice, on this box, it is already blocked — and the
  first draft of this document was wrong about why.** Two things stand between an empty
  bedroom and `quiet.hold_when_away`:
  * `RoomOrPhone.__call__` (`jarvis/presence.py:202`) ends
    `return False if seen is False else None`. **That `False` is reached only when there is
    no phone leg.** `~/.config/jarvis/assistant.json` has
    **`"phone_ip": "192.168.50.34"`** — so an empty room hands the verdict to the phone and
    the radar cannot assert absence at all. This is what section 7 of the sibling doc
    `docs/room-sensor.md` promises (*"The room seeing nobody is not absence"*) and what §3 above relies on when it
    calls the delay free on the absence side. Both statements are now consistent, and both
    are true.
  * `JarvisApp._is_home` (`jarvis/app.py:643`) fails OPEN and needs *both* the phone probe
    and the desk probe to say gone before `hold_when_away` fires.
  * **So the action item is not "set `phone_ip`" — it is already set. It is: do not clear
    it.** Run sensor-only and that `False` branch goes live, and an empty bedroom becomes
    "away". *(Several docstrings still assert the phone leg is unconfigured on this box —
    `jarvis/desk.py:5`, `jarvis/deskpresence.py:4`, `jarvis/classflow.py:20`,
    `jarvis/app.py:1896`. They are stale. The first draft of this document inherited the
    claim from them instead of reading the config. Trust the config.)*
* ❌ **Anything that reads personal content aloud because "the bedroom is occupied, so it's
  just him."** The radar is an occupancy signal, never an identity signal. Identity stays
  with speaker verification.
* ⚠️ **A "morning briefing when he stirs" keyed on `has_moving_target`** will also fire for
  the cat and for a partner.

A cheap detector for the silent failure — a phantom stuck ON is indistinguishable from a man
asleep, and survives every health check that only asks whether the sensor answers: **log the
ON duty cycle over 24 h.** A real bedroom is occupied perhaps 30–45% of a day. **≥99% ON for
24 h is a phantom, not a man.** **[inference]** This is the check that a tuning erring
toward "occupied" must not be allowed to fail, and it is why §3 no longer recommends a 6 m
move fence with a 300 s stretcher on the end of it.

---

## 7. Do not buy a different part yet

The **LD2412** has one genuinely large, bed-relevant advantage: **dynamic background
correction** — an ESPHome button that learns the empty room's clutter and sets its own
thresholds **[vendor doc]**, which is precisely the hard problem here and which the
LD2410's ESPHome component has never had (open feature request since May 2025). It also
adds a **minimum distance gate** (*"Set the nearest detection distance … value range 1 to
14"*), 14 gates out to 9 m, and *"coverage range up to ±75 degrees"*. **[datasheet: LD2412
manual, feature list and spec table]**

**But nobody has published an LD2410-vs-LD2412 bed test**, and I could find **no first-person
full-night in-bed LD2410 log from anyone** — every page ranking for "LD2410 bed presence" is
vendor or SEO content asserting success with no data, and Reddit, where the first-hand
reports live, is unreachable from this machine. Recommending a part swap on a spec-sheet
inference is the move that has misfired on this project twice. **[inference]**

**Run the overnight log first.** If it fails, buy **one** LD2412 for the bedroom (~$6–8) and
keep the LD2410Cs for the other two rooms, where the wall geometry is exactly the "point it
across the room like a small speaker" case your existing doc describes correctly.

And a use for a spare LD2410C that routes around the firmware weakness with hardware you
already own: **put one at the bedroom door as a crossing detector** — or, per §2a, **on the
nightstand**, which is exactly where a moving-target-only radar belongs. Moving-target
detection is the half of the LD2410 that has never been in doubt. A second radar gives
Jarvis an independent *"he came in and has not left"* / *"he just got up"* edge that stays
true even when the bed radar's still bit drops — which is the failure the field reports say
to expect.

---

## Open risks, stated plainly

1. **Nothing here was measured on hardware.** No radar is attached to this machine.
2. **Hi-Link never measured a lying person.** Both polar figures carry three legends —
   Move, Stand, Sit still. Every "sleeping" claim in this document is a sitting measurement
   used as a stand-in, and the two places "lying" appears in the manual are marketing
   bullets, not data. **[vendor doc]** Phase 1 is the only thing that converts the stand-in
   into a fact about you.
3. **The 1.5 m still floor is documented but unconfirmed in the field.** It is Hi-Link's own
   *"-(not settable)"* table entry, which is strong, but no community report independently
   confirms *"the LD2410 will not hold a still person inside 1.5 m"*. If it turns out to be
   a UI restriction rather than a detection limit, the headboard and far-nightstand mounts
   come back and §2 changes. **Phase 1 tests it directly in ten minutes.** The 0.75 m blind
   zone that kills the near nightstand is a separate and much harder number (Table 2).
4. **The re-acquisition latch (§3) rests on four community reports**, one of them an open and
   uncommented ESPHome issue. It is consistent with the ESPHome source, which I did verify is
   a pure relay — so the behaviour must be in module firmware. But Hi-Link documents it
   nowhere and I could not reproduce it. If it is firmware-version-specific, the template-
   sensor workaround is unnecessary.
5. **The 9–12 still-energy figure is a single forum post** about an LD2410B, in someone
   else's room, with an unstated mount and no methodology. **The first draft leaned on it
   hard enough to recommend a threshold from it; this one does not** — every threshold in §3
   is now derived from your own Phase-1 measurement, and 9–12 survives only as a sanity
   check on the order of magnitude.
6. **The polar-plot metres are now measured, not eyeballed** — 400 dpi render, grid circles
   least-squares-fitted, curves isolated by colour, ~±0.05 m repeatability on the fit. What
   remains unknown is what the radius *means*: Figure 8 does not state whether it is floor
   radius or slant range, which matters for a ceiling mount. The within-figure *ratio* —
   still-range vs move-range — is unaffected by that, and it is what the recommendation
   rests on.
7. **The aspect-angle argument is an extrapolation.** The measurement behind it (SPIE 9461,
   2015 — respiration signatures *"smaller from the sides"*) used 500 MHz–18 GHz on an upright
   subject on a turntable. Applying it to a supine sleeper at 24 GHz is my inference. It
   points the same way as Hi-Link's own two figures, which is why I trust the direction and
   not the magnitude. It is also weakened by side-sleeping, where the chest's largest
   displacement axis rotates toward horizontal.
8. **Duvet damping is unmeasured** and my assumption. Fabric *transparency* at 24 GHz is well
   documented; whether a thick lofted duvet reduces the detectable surface displacement is not.
9. **None of the YAML in §3 was compiled.** There is no ESPHome install on this machine. The
   entity names come from ESPHome's `ld2410.cpp`, which is strong evidence that the entities
   exist and weak evidence about the YAML keys that expose them.
10. **Hi-Link contradicts itself** on max-range (5 m in the introduction, 6 m in Table 2), and
    on where a max-gate fence lands (*"farthest door set to 2 → within 1.5m"* vs *"gates 3
    and 4 → 2.25-3.75m"*, one full gate apart). Resolve both on-device with
    `Still distance`, not by reading.

---

**Change to make in the sibling doc:** `docs/room-sensor.md:223-225` should become
room-specific rather than universal. The chest-height line is right for a desk and wrong for
a bed, against the LD2410C's own manual. Its `Absence delay` note at line 226 should also
lose *"it barely matters here"* — that is true for the desk, and §3 above is why it is not
true for a bed.
