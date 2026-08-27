# Stark / JARVIS HUD — Binding Design Reference for the Jarvis V3 UI

Synthesized 2026-08-25 from designer-primary sources (Jayse Hansen / Cantina
Creative, The Orphanage's Dav Rauch & Kent Seki, Prologue) and the best
technical recreations. This document is BINDING for UI passes: where it
conflicts with generic sci-fi instincts, this wins. Reference stills live in
the session scratchpad `stark_refs/` directory.

## The one rule that separates real from cheap

**The mature Stark HUD is WHITE on near-black; cyan is structure, not
star.** The film designers said it outright ("cyan is passé" — Mark III+ is
"predominantly white with colours to accent information"). Saturated
neon-cyan-on-black everywhere is the #1 tell of a cheap recreation.

Color budget (approximate coverage of all lit pixels):
~70% dim structural cyan · ~20% white/ice highlights · ~8% amber · ~2% red.
Color is *state*, never decoration.

## Palette (pre-blended for Tk canvas — no alpha compositing)

Ground: `BG #06090f` (keep ours; cold near-black, never pure black).

Cyan structure ramp (JARVIS cyan `#35e0ff` over BG) — use LOW steps for most
linework, the top step only at focal moments:
```
a=0.08 #0a1a23   a=0.13 #0c2530   a=0.20 #0f3442   a=0.27 #124357
a=0.33 #145064   a=0.40 #175f77   a=0.47 #1a6d89   a=0.60 #218ab0
a=0.73 #28a8d7   a=0.87 #2fc5f3   a=1.00 #35e0ff
```
White-hot focal ramp (text, needles, cores): `#eaf7ff → #ffffff`; band cores
as concentric fills `#ffffff → #d4f4ff → #a8e9ff → #7de4ff → #35e0ff`.
Amber secondary (thinking/caution/power): `#f2a33c` (ramp toward BG for dim
uses: `#3a2c14 / #6d5121 / #b57b2e / #f2a33c`).
Alert red `#ff5533` — target-lock/error ONLY, pulses at ~2Hz by glow not hue.

Most linework sits at the 13–40% steps. If a decor stroke uses the a=1.0
cyan, it had better be the single focal element of its region.

## Ring construction (the #1 identifier — nested counter-rotating gimbal)

5–7 concentric radii; radius ratios cluster near the rim with a jump to the
core (≈ 1 : 0.92 : 0.76 : 0.50 : 0.33). ONLY TWO rings move:
- outer scan arc: **partial** (~144° span), dash rhythm [6,10], rotates ~4s/rev
- counter-scan arc: **partial** (~198°), dash [4,14], rotates opposite at
  **−0.7×** the speed (~6s/rev)
Static rings *breathe* ±3–4% radius on two incommensurate sine frequencies
(e.g. sin(3t) and sin(2t+1)). Tick ruler: 72 ticks, length rhythm 6/3/1
(every 6th long, every 3rd medium), with zero-padded bearing labels
**000 / 090 / 180 / 270** just outside in small mono. Core: banded white-hot
per the ramp above.

## Layout laws

**Frame-verified correction (from the actual stills in `stark_refs/`):** the
real frames are NOT minimal — they are *dense at the margins, calm at the
focus*. Edges and corners are packed with micro-text blocks, numeric
ladders, and tick scales that function as TEXTURE (4–6pt-equivalent, dim,
not meant to be read); one or two large focal elements sit in that ocean.
Hierarchy is carried by scale and brightness, not whitespace. "Bare" is
wrong; uniform density is also wrong. Dense rim, quiet center.

- Elements hug the rim/edges; centers stay calm around the focal object.
- **Asymmetric by design**: different widgets per corner/edge (compass tape
  top, tapes at sides, consolidated status cluster bottom-center in the
  films). Symmetric everything reads as a video game.
- **One consolidated status cluster** beats scattered gauges: for Jarvis,
  a bottom-of-stage system avatar — a small radial "suit status" analog
  showing mic / STT / TTS / LLM / memory as per-spoke highlights.
- Panels: chamfered (one or two 45° cut corners), 1px borders at the 27–47%
  ramp, fills at most 5–15% equivalent, **corner brackets instead of full
  boxes** for micro-readouts (12–18px L-brackets).
- Wireframe motifs: line-only, no fills; density implies mass.

## Text & data

- Condensed technical sans for HUD text (we have **Rajdhani**; Arame/
  Eurostile are the film matches). ALL-CAPS micro-labels, wide tracking,
  label DIM (40–60% ramp) / value BRIGHT (white).
- **Leader lines**: no floating text — 1px line with one 45° elbow and a dot
  or bracket terminator tying each label to its geometry. Callout arrival:
  bracket first, line draws ~0.15s, text types on ~0.2s.
- Numbers are tabular and *tick* (odometer scramble ~15 updates/s settling
  in ~0.5s) when values change.
- **Every readout is REAL** (state, uptime, backends, temps, utilization,
  latency, model names, reminder counts). The film designers made even
  micro-text story-true; fake hexdumps/matrix rain are the cheese line.

## Motion laws

- **Everything idles** — nothing is frozen: slow perpetual ring rotation
  (6–30s/rev, adjacent rings opposite directions), small elements pulse at
  1–3s periods, blinking indicators at ~3s sine.
- Arrivals are **snappy**: scale 60%→100% in 0.2–0.35s ease-out with 3–5%
  overshoot; reticles/highlights jump (<100ms) with data catching up a beat
  later.
- **Assembly on boot**: arcs draw on, ticks populate sequentially, panels
  resolve over 1.5–4s. The app's startup should assemble the HUD, not pop it.
- Widget promotion: peripheral gauges grow toward center when relevant
  (position+scale together, ~0.3s).
- Flicker is **sparing and meaningful**: 2–3 frame dips on state changes or
  "signal" moments; never constant, never on a healthy idle system beyond a
  rare (7–11s) subtle shimmer. Warnings pulse glow at 2Hz, hue stays.

## Depth & glow (Tk-feasible)

- Two/three brightness+weight tiers imply depth: near = heavier + brighter,
  far = hairline + dim (the IM3 signature).
- Glow = white-cored stacked strokes: same arc drawn 3–4× (w7@8%, w5@15%,
  w3@30%, w1@100%-white-core) using pre-blended colors; or a pre-rendered
  radial-gradient PhotoImage disc behind ring stacks (render once, blit).
- Scanline/vignette: ≤ 6–12% intensity, 3–4px period; a radial vignette
  seats graphics "behind glass". Subtlety is the whole trick.
- Optional hologram fringe: main text/line drawn once, plus 1px offset ghost
  in dim blue at ~15% — focal elements only.

## Frame-verified tells (each faithful pastiche needs at least one)

- **Line weight, not fill**: almost nothing is a solid filled shape —
  hairline strokes, dots, and 10–30% glass panels; glow blooms off thin
  lines, surfaces stay transparent. The IM1 HUD is barely blue at all:
  mostly white/monochrome hairlines.
- Rectangles exist only as thin-cornered "cards" (video inset, dossier
  panel); everything else is concentric/radial or triangulated lattice.
- Diegetic hologram tells: scanline shimmer on volumetric elements, subtle
  RGB channel-split fringing at high-contrast edges (1px dim blue/red ghost
  offsets), data anchored to space rather than boxed.
- Margin texture: numeric ladders (altitude-style tick columns with
  values), boxed alphanumerics ("MACH 0.14" style), tiny circular
  micro-gauges in a row — all REAL values in our case.

## Anti-cheese checklist (reject a change if it hits any)

1. Saturated cyan everywhere / no white focal values.
2. Glow radii huge, everything blooming.
3. All rings spinning, same speed, full 360° dashes.
4. Symmetric, uniformly dense layout; filled quiet zones.
5. Fake data (lorem, hexdumps, binary rain, "00 ORCHESTRATION LAYER").
6. Full glowing borders on panels (video-game); use corner brackets.
7. Constant flicker/glitch on a healthy system.
8. Uniform line weight everywhere (no depth tiers).

## Mapping to Jarvis V3 components

- **Reactor stage** = workshop hologram + helmet aperture: nested gimbal
  ring cluster per the construction recipe; degree ruler with 000/090/180/270;
  corner brackets; two scanlines; radar sweep already present → retune to
  partial-arc counter-rotation and white-cored strokes; telemetry corners
  keep REAL data with leader-line elbows into the stage.
- **Header** = compass-tape band: thin ruler ticks under the full width,
  white needle at center; wordmark stays display-face.
- **Transcript cards** = chamfered holo panels: corner brackets, dim
  borders, white text, cyan structure; JARVIS cards may carry the subtle
  left glow.
- **Command bar** = input rail: chamfered field, `❯` prompt, white caret;
  mic ring segments close while recording (already specced).
- **Status strip** = bottom gauge rail: segmented micro-gauges with tiny
  meters (real cpu/gpu), plus the radial system-status cluster if space
  allows.
- **Boot** = assembly: stage draws on over ~2s (arcs trim in, ticks
  populate, telemetry types on). Full-window fades are not available in Tk;
  sequence item creation instead.
