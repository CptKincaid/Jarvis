# Face and body recognition on this machine

Design pass, 2026-09-02, worktree `camera-vision` cut from `2df866c`. Companion to
`scratchpad/ideas/camera.md`, which settled the *geometry* and the *attention* half of the
camera question earlier today. **This document is the recognition half** — who the camera
thinks it is looking at, where that lives on disk, and what may depend on it.

Every model timing below was **measured on this box today**, single-process, with
`cv2.setNumThreads()` pinned. Nothing is extrapolated from a vendor's benchmark unless it is
labelled as such. The models were fetched into a scratch directory outside the repo, timed,
and left there; **nothing was downloaded into `~/Jarvis`.**

---

## The verdict, first

1. **Face: YuNet + SFace, both through OpenCV's own DNN module. Zero new dependencies, zero
   pip installs, zero GPU.** Measured end to end at **7.7 ms per frame on four threads** —
   detect, align and identify. Licences MIT and Apache-2.0. This is not a research project;
   it is two `cv2` calls the box can already make.
2. **Body: do not buy the re-identification model.** It is measured here at **30.8 ms a crop
   on four threads for a 106 MB network**, and what it encodes is mostly *clothing*. The
   honest answer to "recognise him when his face is turned away" is not a body-identity model
   — it is to **carry an identity the face already established**, on a clock, and drop it the
   moment the room empties. That is `SessionIdentity` in `jarvis/eye.py`, and it costs nothing.
3. **The daily win is the wake gate, and the specific failure is now nameable.**
   `jarvis/hotword.py:428-448` documents a still-open hole: under an *unflagged* bed — a
   television, a browser, anything that is not Spotify — the verifier's trim widens to cover
   the room, so the too-little-speech abstention disengages, and `hey_jarvis_05` reports 1.46 s
   and scores **0.183 → suppress**, on exactly the 0.52 s of him that scores **0.277 → accept**
   when dry. Audio has no more information to give there. **A television cannot put a face in
   his chair.** That is the fusion rule, and it is built and tested.
4. **One mount: eye level, ~70 cm to the side, wide lens, capturing 1080p.** Not the high
   shelf. The high shelf is the better *presence* camera and the worse *face* camera, and the
   presence coverage it buys can be bought back with optics instead — a wide lens costs face
   recognition nothing, while a downward pitch costs it a great deal.
5. **Offline mode and the curfew are consumed, never re-implemented.** `Eye.permitted()` asks
   the sensing-state owner twice per frame — before the device is opened and again after the
   grab, before the frame is handed to anyone — so a deny mid-pipeline drops the frame in
   flight. Anything but exactly `True` is a no, exceptions included.

---

## 1. What is actually on this box

All verified today, in `~/vss_env`, on this machine.

| Thing | Measured | Consequence |
|---|---|---|
| Architecture | `aarch64`, **20 cores: 10 × Cortex-X925 + 10 × Cortex-A725** | threads are the lever; see §2 |
| torch | **2.12.1+cu130**, `cuda.is_available() True` | *the brief said 2.9.1+cu130; it is 2.12.1.* Irrelevant to this design — nothing here uses torch |
| OpenCV | `cv2` **4.12.0**, `cv2.FaceDetectorYN` ✅, `cv2.FaceRecognizerSF` ✅ | the whole face stack is already importable |
| `cv2.cuda` | **0 devices** | OpenCV DNN runs on CPU here, full stop |
| onnxruntime | 1.23.2, providers `['AzureExecutionProvider', 'CPUExecutionProvider']` | **no CUDA EP, no TensorRT.** Another reason to load through `cv2`, not `ort` |
| insightface | **not installed** | good; see §3 on why it should stay that way |
| ultralytics | 8.4.6, **AGPL-3.0** | avoidable — see §4 |
| `/dev/video*`, `/dev/v4l` | **do not exist** | no camera attached; every test here is hardware-free |
| `uvcvideo` | `uvcvideo.ko.zst` present, **0 modules loaded** | loads on hotplug |

The vision stack proposed here takes **0 bytes of GPU** and adds **no package to `~/vss_env`**,
which matters more than it sounds: this box had a hard power-off on 2026-08-28 from unified-memory
exhaustion, and `~/vss_env` is shared with VSS.

---

## 2. The benchmark — measured here, today

`opencv_zoo` models, loaded through `cv2.dnn` / `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`,
timed over 200-300 iterations after warm-up, median and 95th percentile in milliseconds.

| Model | On-disk bytes | Licence | 1 thread | 2 | 4 | Published, Jetson Orin Nano CPU |
|---|---|---|---|---|---|---|
| **YuNet** @160×120 | 229,738 | **MIT** | **0.86** / 0.94 | — | — | 2.59 |
| **YuNet** @320×240 | " | " | **3.10** / 3.13 | 2.60 | **1.65** | — |
| **YuNet** @640×480 | " | " | **12.63** / 14.97 | — | — | — |
| **SFace** @112×112 | 38,696,353 | **Apache-2.0** | **19.63** / 19.98 | 10.36 | **5.85** | 20.05 |
| SFace `alignCrop` | " | " | 0.17 / 0.18 | — | — | — |
| **YoutuReID** @128×256 | 106,878,407 | **Apache-2.0** | **105.3** / 106.7 | — | **30.8** | 93.58 |
| **YuNet** @320×180 (the shipped detect size) | " | " | **2.24** | 1.36 | **1.00** | — |
| MJPEG decode 1920×1080, *smooth upscale* | — | — | **3.42** | — | — | — |
| MJPEG decode 1920×1080, *detail-preserving* | — | — | **5.45** | — | — | — |
| MJPEG decode 1280×720, detail-preserving | — | — | **2.16** | — | — | — |
| resize 1080p→320×240, `INTER_AREA` | — | — | — | **2.54** | — | — |
| resize 1080p→320×180, `INTER_AREA` | — | — | — | **1.30** | — | — |

Sizes are the exact git-lfs object sizes from `opencv/opencv_zoo@main`; licences are the
per-model `LICENSE` files in that repo, read today.

**The decode row is two rows because the first one was measured wrong.** The original
1080p figure came from a 512 px photo upscaled to 1080p, which encodes to 226 KB and decodes in
3.42 ms; a frame that actually carries 1080p worth of detail encodes to 544 KB and decodes in
**5.45 ms**, ~1.6× more. And the whole line item is an assumption about a part nobody has: I could
not confirm that the recommended Arducam streams **MJPEG** at 1080p rather than uncompressed
**YUY2** over USB 3.0 (arducam.com returns 403, the Amazon listing is JS-rendered). **If it is
YUY2 the decode term disappears entirely.** Check with `v4l2-ctl --list-formats-ext` on the day it
arrives, before trusting any budget below.

**Three findings worth more than the raw numbers.**

- **YuNet is 3× faster here than on an Orin Nano CPU (0.86 vs 2.59 ms); SFace is not faster at
  all (19.63 vs 20.05).** A tiny detector is compute-bound and rides the X925's clock; a 38 MB
  network is memory-bandwidth-bound and does not. **So the lever on this box is threads, not
  cores' speed**: SFace goes 19.63 → 5.85 (3.4×) across four threads while YuNet only manages
  3.10 → 1.65 (1.9×). Budget accordingly, and pin `cv2.setNumThreads(4)` in the sidecar rather
  than leaving it at 20, which would fight the rest of the box for no gain.
- **SFace's cost does not depend on how many pixels the face had.** A 48 px crop upscaled to
  112 costs 19.76 ms; a 160 px crop costs 19.66. The model always resamples to 112×112. **So a
  small face is not cheaper, it is only worse** — which is what makes the capture-resolution
  arithmetic in §9 load-bearing rather than fussy.

- **Threads buy latency and cost CPU, and the headline number used to confuse the two.** This
  document first summed a 4-thread budget and called it "13 % of one core", which is wall time
  presented as core occupancy. Measured directly: SFace at 1 thread is **22.13 ms wall / 22.13
  CPU-ms** (1.00×); at 2 threads **12.12 / 24.15** (1.99×); at 4 threads **7.00 / 25.93** (3.71×).
  More threads is *more* total CPU, not less. **That is why `camera.threads` ships at 2, not the 4
  first advised here** — the armed tier's frame period is 125 ms and the whole chain is ~20 ms even
  at 2, so latency is not the binding constraint; contention with the live Jarvis, ollama and F5 on
  a box that has already had one unified-memory power-off is.

**Frame budget, armed tier, 1080p capture, 320×180 detect, 2 threads — wall *and* CPU, because
they are different questions:**
decode 5.45 + downscale 1.30 + YuNet@320×180 1.36 + alignCrop 0.17 + SFace 12.12 ≈ **20.4 ms of
wall latency**, and ≈ **35 CPU-ms**. At 8 fps that is **≈280 CPU-ms/s — about 28 % of one core of
twenty**, not 13 %. The wall figure is the right one for latency; the CPU figure is the right one
for contention. **The conclusion survives — it is affordable on a 20-core box — but the old number
did not.** With `identity: False` (the phase-1 default) SFace never runs and the armed tier is
≈8 ms wall / ≈11 CPU-ms.
**Idle tier, 720p, 1.5 fps, detection only:** (2.16 + 0.15 + 1.36) × 1.5 ≈ **6 ms/s, well under 1 %.**

---

## 3. Face — detection and identity

### Detection: YuNet, and it is not a close call

`face_detection_yunet_2026may.onnx`, **229,738 bytes**, **MIT** (Shiqi Yu), loaded by
`cv2.FaceDetectorYN` — OpenCV's own DNN module, **not** onnxruntime, which is what dodges the
aarch64 CPU-only-provider trap entirely. This is the same load path VSS already wrote and has
never once run: `aiws_system/face_obscurer.py:164-190` builds the detector exactly this way and
falls back silently to a head-box heuristic because the weights were never fetched
(`face_obscurer.py:178-182`). **The pattern is proven code in this house; only the 229 KB is new.**

`detect()` returns N×15. VSS reads columns 0-3. Columns **4-13 are five landmarks** — both eyes,
nose tip, both mouth corners — and they are what makes the attention half of `camera.md` free.
They are also what `cv2.FaceRecognizerSF.alignCrop` requires, so **detection and identity share
one forward pass**: there is no separate landmark model, and no MediaPipe.

> **Do not install mediapipe into `~/vss_env`.** It depends on `opencv-contrib-python`; the venv
> has `opencv-python`. They are two distributions of the same `cv2` import, and `pip install`
> would silently replace the `cv2` that **Jarvis and VSS both use**, at install time. This design
> needs no new package at all, which is the cheapest possible way to be safe from that.

### Identity: SFace

`face_recognition_sface_2021dec.onnx`, **38,696,353 bytes**, **Apache-2.0**, via
`cv2.FaceRecognizerSF`. Measured output: **128-D float32, L2 norm 10.41 — not unit length**,
which is why `jarvis/facegallery.py` normalises inside `cosine()` rather than dotting raw
vectors. OpenCV's documented "same person" cosine threshold is **0.363**, and that is the
default in `assistant_config` (`camera.identity_min`).

### What is rejected, and why

| Option | Verdict |
|---|---|
| **InsightFace / buffalo_l (ArcFace)** | **No.** More accurate, and the pretrained weights are research-only. VSS lives in the same venv on the same box and has a commercial ship-gate plus a label-provenance firewall; a research-licensed face model sitting beside it is exactly how a licence leak happens. It is not installed today. Keep it that way. |
| **MediaPipe Face Landmarker** | Only in a sidecar venv, and not needed: YuNet's five points already do the yaw proxy. See the `cv2` trap above. |
| **A hosted face API** | **Forbidden by ruling.** All camera processing is local. There is no cloud path to disable here because there is no cloud path — the sidecar makes no network calls at all. |
| **`ediffiqa`** (opencv_zoo face image quality) | Worth remembering later. A quality score would let enrolment refuse a bad take rather than poisoning the pool. Not phase 1. |

---

## 4. Body — detection and re-identification, honestly

He asked for recognition when his face is *not* toward the lens, "which at a desk is most of the
time". That is the right problem. The usual answer to it is wrong.

### Person detection

Two Apache-2.0 options load through `cv2.dnn` with no new dependency: **NanoDet**
(3,800,954 bytes) and **MediaPipe person detection** (11,990,159 bytes). Either avoids
`ultralytics` 8.4.6, which is **AGPL-3.0** — irrelevant to him personally, relevant the moment
any of this drifts toward VSS.

**But at a desk, prefer the face count.** YuNet already returns *how many faces*, at
**1.65 ms**, in a pass that has to happen anyway. NanoDet's published Orin CPU time is 125 ms.
A person detector earns its place when someone is in the room but out of face range — the
doorway, the far chair — and that is a §9 field-of-view question, not a phase-1 one.

### Re-identification: the honest answer is "don't"

**YoutuReID** (opencv_zoo, Apache-2.0, **106,878,407 bytes**, 768-D output, L2 = 17.97) measured
here at **105.3 ms single-threaded / 30.8 ms on four**. Set the cost aside; the accuracy argument
is the one that decides it.

Person re-ID models are trained to match the same person across cameras minutes apart
(Market-1501, DukeMTMC and their descendants). In that setting clothing is a nearly perfect cue,
so that is largely what the network learns. The published cloth-changing benchmarks — **PRCC**
and **LTCC** — exist precisely because rank-1 accuracy collapses when the clothes change, and
purpose-built cloth-changing methods still trail same-clothes numbers by a wide margin. Applied
to his question:

- **"Is this the same body as 200 ms ago?"** — yes, reliably. But that question does not need a
  106 MB network. **An IoU tracker plus a torso colour histogram answers it for well under
  0.1 ms** — measured 2026-09-02 rather than asserted: a 16×16 HSV histogram of a 128×256 torso
  crop is **p50 0.054 ms / p95 0.055 ms** on two threads, against YoutuReID's 30.8 ms on four —
  and is not meaningfully worse at a fixed camera watching one chair. *The cost was never the
  problem with the histogram; the threshold is, see below.*
- **"Is this Hunter or a stranger, from the body alone?"** — **no**, not dependably. Same-day, a
  stranger in a similar dark hoodie will match; next-day in a different shirt he will not. A
  system that says "welcome back" to a visitor in the right jumper is worse than one that says
  nothing.

**So: the smallest thing that distinguishes "Hunter at his desk" from "someone else at his desk"
from body alone is nothing.** There is no honest small answer. What *is* available, and what is
implemented today, is a different shape:

> **`SessionIdentity` (`jarvis/eye.py`).** A face sighting *anchors* a body vector to a label.
> While the anchor holds, the body answers "still him" through every head turn — which is the
> actual gap he described. Two clocks bound the damage: a **TTL** (15 min default), because
> clothes change; and **`room_empty()`**, which drops the anchor the instant nobody is in frame,
> because that is exactly the window in which a different person can sit down wearing anything.
> Nothing is persisted — **a body vector never reaches the disk**, so there is no body gallery to
> back up, leak or delete.

That gives him the behaviour he wanted without pretending to an accuracy nobody has. And it means
**phase 1 needs no re-ID model at all**: the anchor can be a colour histogram to start, and be
upgraded to YoutuReID later without changing a single consumer, because the interface is
"vector in, label out".

**One thing does change, and this document first got it wrong: the threshold.** `BODY_MATCH_MIN
= 0.75` is a cosine over the 768-D YoutuReID vector. It does **not** transfer to a histogram, and
the reason is arithmetic rather than tuning. Cosine over **non-negative** vectors lives in a
compressed range: for iid uniform components, two *completely unrelated* vectors have expected
cosine `E[x]²/E[x²] = 0.25/(1/3) = 0.750` — **exactly the shipped bar**. Measured here 2026-09-02
over 2000 pairs at d = 256, 768 and 4096: mean 0.750 every time, ~50 % at or above 0.75. Sparse
HSV histograms do better than that worst case (nine unlike shirts, 36 pairs, median 0.000) but
still put **3 of 36 unlike pairs over the bar** — and, in the other direction, **the same shirt
under a changed desk lamp scored 0.692–0.706, i.e. *under* it.** A histogram at 0.75 both welcomes
strangers and forgets him when the lamp changes.

So `SessionIdentity.match_min` is now a **required keyword argument with no default**: a caller
must state which vector it is holding. And the honest phase-1 position is that **the histogram
anchor is not calibrated** — there is no number here that separates those two distributions, and
finding one needs his room, his lamps and his wardrobe, not more arithmetic. Until then the
tiebreaker's identity term stays `""` and the anchor carries nothing.

---

## 5. Enrolment and storage — the part that goes wrong quietly

A face embedding is a measurement of one specific person that cannot be re-issued if it leaks.
It gets a stricter standard than the rest of the app, and the standard comes from an incident
that happened **today**, not from principle.

### The incident, stated plainly

On 2026-09-02 a test built a real `SpeakerVerifier` and called `enroll_from_audio`. `save()`
(`jarvis/speaker.py:276`) writes a module-global path, so his **six-sample voiceprint was
replaced with two copies of the fixture's constant vector — every element 0.07216878**. His own
enrolment clips then scored 0.055-0.125 against a 0.30 threshold. **The copy kept beside it held
only those fixtures. The original was unrecoverable and he re-enrolled by hand.**
(`jarvis/config.py:46-55`; the `JARVIS_VOICEPRINT` paragraph in `tests/conftest.py`.)

`speaker.save` is *atomic* — tmp file plus `os.replace`. **Atomicity protects against a crash in
the middle of a write and against nothing else. The write that destroyed the voiceprint
succeeded.** Every guard below is a thing atomicity does not give you.

### The design — `jarvis/facegallery.py`, built and tested today

**Where.** `~/.aiws_trainer/face_gallery/`, directory mode **0700**, files **0600**. Beside the
voiceprint, because they are the same class of object and should be found, backed up and deleted
together. `PATHS.FACE_GALLERY` (`jarvis/config.py`) reads **`JARVIS_FACE_GALLERY`**.

**A directory, not a file, because saves are generational.** A save never overwrites. It writes
`gen-00002.npz` beside `gen-00001.npz`; `load()` takes the newest that *parses* and falls back
down the stack when it does not. **The old generations are the backup** — there is no separate
`.bak` that can quietly come to hold the same bad data the live file does, which is exactly the
shape the voiceprint's backup had. Five are kept, never fewer than two: one generation is no
backup at all — **and the richest generation is never one of the five that fall out.** That last
clause is a correction: oldest-first pruning alone made the claim false, because five saves of any
size evicted the enrolment and reproduced the incident's end state one loop later (measured: one
6-sample enrolment plus six 2-sample saves left generations [4,5,6,7,8], every one n=2). Ties go
to the newest, so in the normal case — saves the same size or growing — pruning behaves exactly as
it did and the window still moves.

**Three guards, each refusing the shape the incident actually had:**

| Guard | Refuses | Because |
|---|---|---|
| `degenerate_reason()` at `add()` | a **constant vector**, a wrong dimension, a non-finite element | the fixture that destroyed the voiceprint was a constant vector; a real SFace embedding never is |
| `_check_not_collapsed()` at `save()` | a pool whose samples are all the **same vector** | two copies of one vector is what the incident *left behind*, and it is indistinguishable from a working enrolment until the day it refuses him |
| the **shrink guard** at `save()` | going from N samples to fewer without `allow_shrink=True`, measured against **the larger of what this object loaded and what is on disk** | six became two and nothing objected — and the caller that did it had loaded nothing, so an in-memory baseline abstains on its own motivating case |
| `_prune()` protecting `max(n)` | evicting the **richest** generation, ever | oldest-first pruning made five ordinary saves enough to delete a six-take enrolment and leave the store holding only the bad writes |

**`rollback()`** deletes the newest generation and reloads the one before — the step that did not
exist today — and **reports the generation it is actually holding**, which is not the same thing
when the predecessor is corrupt too (measured: it returned 2 while holding 1). **`purge()`**
deletes *every* generation, not just the newest, **and any `gen-NNNNN.npz.tmp` a crashed save left
behind**: a tmp holds a full set of embeddings under a name the generation pattern does not match,
so the first version reported success and left one on disk at 0600. A store whose "delete" leaves
an older copy of his face on disk has not deleted anything. `save()` also unlinks its own tmp on
failure, and creates it `0600` **from the open() call** rather than chmod-ing after `np.savez`
returns — under his 0002 umask the plain form is 0664 for the whole write.

**Versioning.** `FORMAT = 1` is stored in the file and a *future* format is refused rather than
misread — reading a newer pool as if it were this one is how embeddings silently stop comparing,
which `jarvis/speaker.py:264-272` already warns about for the voiceprint. Each generation also
carries **provenance**: `created_ns`, sample count, and a **`reason` string** the caller must
pass. When a gallery turns out to be wrong the only question that matters is what wrote it, and
today nothing on disk could answer that.

**Backup — checked, and the answer is worse than "should be".** `restic` is in `~/.local/bin`
(28 MB, 30 Jun). **There is no backup set for the gallery to join.** No `~/.config/restic`, no
`~/.cache/restic`, no `RESTIC_*` in `~/.bashrc` or `~/.profile`, no restic reference in any user
systemd unit, no backup script, and the only user timer on the box is `haymaker-digest.timer`.
**Nothing under `~/.aiws_trainer/` is backed up today — which is why this morning's loss was
unrecoverable.** A set has to be *created* before the gallery exists, not joined. (Also:
`voiceprint.npz` is currently mode 0664; the gallery is 0600.) This is no longer open question 3;
it is a finding, and it is the most urgent item in this document.

**The test firewall.** `tests/conftest.py` now forces `JARVIS_FACE_GALLERY` into the throwaway
directory, forced rather than `setdefault`, exactly as it now does for `JARVIS_VOICEPRINT`. **A
firewall and a recoverable format are different defences and this data warrants both** — the
voiceprint had neither on the day it was lost.

### Enrolment procedure

- **A deliberate script, never passive learning.** `jarvis/app.py:2598 _maybe_learn_voice`
  quietly adds accepted utterances to the voice pool. **Do not do that for faces.** A passive
  learner that drifts onto a visitor's face is silent, cumulative, and produces a gallery that
  identifies the wrong person with high confidence.
- **8-12 takes**, varying what actually varies: with and without glasses, desk lamp and overhead,
  head slightly left/right/up/down. Not 12 frames of one pose — that is a collapsed pool with
  extra steps, and `save()` will say so.
- **Refuse a take whose face box is under ~112 px** on the long side. SFace costs the same either
  way (§2), so a small take is pure loss.
- `--reset` purges before enrolling; `--rollback` undoes the last write.

---

## 6. What it is for, in priority order

**1 — Wake-gate tiebreaker. This is the measured win and the reason to spend the money.**
Built today as `resolve_wake()` in `jarvis/eye.py`. The gate's only remaining `False` is "clean
audio, enough measured speech, no music *known*, score under the bar" — and `hotword.py:428-448`
documents that verdict being **wrong on his own voice** under an unflagged bed, because
`music_playing` is Spotify's cache and nothing else. The camera is a sensor a television cannot
touch.

The rule is deliberately one-directional and the invariant is asserted **exhaustively** over the
whole input space in `tests/test_eye.py::test_the_camera_cannot_subtract_over_the_whole_input_space`:

> **The camera may promote a suppressed wake. It may never suppress an accepted one.**
> `jarvis/hotword.py:373-375` already states why — "a wake word that cannot be triggered is worse
> than one that triggers too often" — and a second sensor must not quietly undo that.

Promotion requires `faces == 1` (a second person could be the one who spoke), a real dwell
(≥0.4 s; a glance past the lens on the way to a mug is not an address), and an identity that is
not *someone else's*. **The cost is stated rather than hidden:** a stranger at his desk, facing
the camera, whose voice scores under the bar, now wakes Jarvis. The transcript gate in `app.py`
still fails shut behind it.

**And so the whole of rule 1 rests on the yaw proxy, which is the untested part.** "A television
cannot put a face in his chair" is a *presence* argument, but this is implemented as an
*attention* gate — `attending` is a hard requirement, not a bonus term. If §9's tape-and-photos
test shows the mount cannot separate "looking at Jarvis" from "reading the tab bar", rule 1
promotes nothing and there is no weaker version to fall back to (§11.1). Rules 2 and 3 survive
that outcome; rule 1 does not.

A smaller correction while here: **`camera.md`'s "Rule 2" is stale.** It targets
`SPEAKER_WAKE_MIN_MUSIC = 0.10`, which no longer exists — `2df866c` replaced it with an
abstention (`hotword.py:300-327`). Over *known* music the gate now already waves him through, so
there is nothing there for the camera to fix. The unflagged-bed case above is the live one.

**2 — Presence, as positive evidence only.** A face in frame means he is here. **No face in frame
means nothing at all** — he may be in the kitchen. This is the same asymmetry
`jarvis/presence.py:14-19` already states for the room sensor, and the camera must inherit it
rather than invent a second rule.

**3 — Extending the follow-up window.** `app._start_followup` (`app.py:2394`) already opens the
mic with no wake word for `CONFIG.followup_window = 4.0` s, `jarvis/config.py:141`. That window is blind today. Extend it
while he is attending and close it when he looks away. It needs no new trigger path and lives
inside a window Jarvis itself just opened.

**4 — A cold trigger. Opt-in, default off, and only if phase 2's logged numbers justify it.** A
false wake from a wake word costs a chime; a false wake from *attention* costs the same and is
undiagnosable, because he did nothing. Rules 1-3 cannot produce a wake that a wake word did not
already produce. Only this one can.

### What must NOT depend on the camera

- **The wake word.** Ever. Unplug the camera and wake behaviour must be byte-for-byte identical.
- **The transcript gate.** It fails shut and stays the backstop; the camera never relaxes it.
- **Absence.** Nothing may conclude "he is away" from an empty frame.
- **Any proactive speech**, quiet hours, or presence *departure* logic.

`Eye.capture()` returns `None` for *every* way it can decline — denied, no device, unplugged,
failed grab, offline mid-grab — because a consumer that has to tell those apart will eventually
get one of them wrong, and the safe reading of all five is identical.

---

## 7. Offline mode and the curfew

**The sensing-state owner is a separate module built in the `offline-mode` lane. This design
consumes it and deliberately re-implements none of it** — not the 21:00-07:00 window, not the
voice phrasings, not the UI buttons. `assistant_config`'s new `camera` section carries **no
schedule keys at all**, and there is a test that fails if one appears
(`test_the_camera_section_carries_no_schedule_of_its_own`). A second copy of the window is a copy
that can disagree with the first, and the one that disagrees quietly is the one that leaves the
lens open at 22:00.

### Exactly where `open()` consults it

`jarvis/eye.py`, class `Eye`. It is **the only thing in the process that opens the video device**,
and it asks **twice per frame**:

```
Eye.capture()
  1. permitted()?  ──no──►  denials++ ; close() any open device ; return None
     │                      (the device is NEVER opened — nothing to ignore)
     yes
  2. open the device if not already open   (any exception -> None, no latch)
  3. device.read()                          (a failed grab -> close(), return None)
  4. permitted()?  ──no──►  DROP THE FRAME ; close() ; log ; return None
     yes
  5. return frame
```

**Step 1 is Hunter's ruling.** He asked for enforcement "so the device is not opened at all — not
merely a software flag that a later code path could ignore". A denied gate costs **zero device
opens**, so there is no flag downstream to ignore.

**Step 4 is the frame in flight.** If he says "offline mode" while a grab is in progress, the
frame exists in memory and **must not be used**. It is discarded before any inference, before any
event, and the device is released in the same tick. Test:
`test_going_offline_mid_pipeline_drops_the_frame_in_flight`.

### Fail to offline

He chose this over persisting state and over failing online. `permitted()` returns True **only**
when `allow()` returns exactly `True`. A missing callable, `None`, a truthy `1`, the string
`"yes"`, or an exception out of the owner are all **no**. *A camera that watches because nobody
told it not to is precisely the bug the ruling was about*, and "the sensing owner is broken" is
not evidence that watching is wanted. Tests: `test_it_fails_to_offline_when_the_owner_raises`,
`test_it_fails_to_offline_with_no_owner_wired_at_all`, `test_only_a_real_true_is_permission`.

Because there is **no persisted camera state**, a Jarvis restart while offline cannot come back
watching: with the owner unreachable during start-up, `permitted()` is False and the device stays
shut until the owner says otherwise.

### Two things the sidecar must add

1. **Poll the gate on its own timer, not only on the frame timer.** At the idle tier (1.5 fps) a
   deny would otherwise wait up to 667 ms for the next scheduled frame. Poll at ~5 Hz so
   "offline" closes the device in ≤200 ms — he should hear the command and see the lamp go dark
   in the same beat.
2. **The UI lamp reads `Eye.device_open`, never the config.** `StatusStrip`
   (`jarvis/ui/views.py:2066`) already owns the wake-word segment; an EYE segment beside it must
   be driven by whether a device handle is actually held. **If the strip is dark, the camera is
   closed — by construction, not by intention.**

### Curfew, mmWave, and the one thing to watch

The ruling puts the *camera* under curfew 21:00-07:00 and keeps the **microphone live**. Note the
interaction: the camera's wake-gate tiebreaker (§6, rule 1) is therefore **unavailable overnight**
— and the unflagged-bed failure it fixes is most likely in the evening. That is not an argument
against the curfew; it is an argument for **not letting anything regress into depending on the
camera**, which §6 already requires. Overnight, behaviour is exactly today's.

---

## 8. Gesture control — what would foreclose it

Not designed here. What matters is that nothing chosen now makes it hard.

**Already easy.** `opencv_zoo` ships `handpose_estimation_mediapipe`, `palm_detection_mediapipe`
and `pose_estimation_mediapipe`, all Apache-2.0, all loadable through **the same `cv2.dnn` path
with no new dependency**. Choosing OpenCV DNN over onnxruntime today keeps that door open; it is
the single most gesture-friendly decision in this document.

**Three things that would foreclose it, and how they are avoided:**

| Risk | Avoided by |
|---|---|
| Hard-coding capture resolution to the detect resolution. A hand at 1.5 m needs far more pixels than a face at 70 cm. | `camera.width/height` and `camera.detect_width/detect_height` are **separate config keys**, already |
| A fixed `Attention`-shaped socket message. Gestures are a different event with different fields. | make the sidecar protocol a **tagged message** (`{"t": "attention", ...}`), not a fixed struct. Not yet written — this is the note that says to. |
| A narrow lens. | §9 recommends a wide one anyway |

**The honest cost of the recommended mount:** an eye-level side camera sees his hands at the
keyboard **partly occluded by the monitor**. Deliberate mid-air gestures raised toward the camera
are fine; subtle at-the-keyboard gestures are not. A high mount would be better for hands and
worse for faces (§9). If gesture control later matters more than face identity, that is the trade
to revisit — and it is a screwdriver, not a rewrite.

---

## 9. The shopping list and the mount

### One placement, and why it is not the high shelf

Three uses, one mount: **addressing by head turn**, **presence**, **face recognition**.
`camera.md` §4 established the geometry — on the monitor, "looking at Jarvis" and "reading the tab
bar" differ by **1.8°** against 4-10° of model error, so an on-monitor mount is dead; 70 cm to the
side gives **47°** separation with 22° of clearance, and 60 cm above the monitor gives **50°**
with 33°. Both work for attention. They differ on the other two uses:

| | Eye level, ~70 cm to the side | High shelf, 60 cm above the monitor |
|---|---|---|
| Head-turn separation | 47°, **22°** clear of the screen | 50°, **33°** clear |
| Presence coverage | narrower — unless the lens is wide | **better**: a downward view sees the room |
| **Face recognition** | **best: face at eye level, no pitch** | **worst: a persistent 30-35° downward pitch onto his face** |
| Gestures at the keyboard | partly occluded by the monitor | better |

**Recommendation: eye level, ~70 cm to the side, angled in ~30-40°, with a wide (95-100° dFOV)
lens.** The reasoning is that the two disadvantages are not symmetric: **the presence coverage the
high mount buys can be bought back with optics, and the face-recognition quality the high mount
costs cannot be bought back at all.** A wide lens costs face recognition nothing but pixels — and
pixels are a resolution setting. A downward pitch costs pose, and there is no setting for that.

**What it costs the other two, said plainly:** 22° of clearance instead of 33° (still roughly 2×
the best gaze estimator's error, but tighter — this is the number to re-measure with a tape before
buying); a worse view of the doorway than a high mount, partly recovered by the wide lens; and
hands at the keyboard partly occluded (§8). It also **points across the room**, so compose the
shot so that whatever is on that side is out of frame — composition is a privacy control.

### The resolution this forces — and it is the concrete consequence

At 98° diagonal on 16:9 the horizontal field is ≈90°. A camera 70 cm to the side of a face 65 cm
from the screen is **≈95 cm from that face**, spanning ≈191 cm horizontally. So:

| Capture | Pixels per cm | A 16 cm face | Verdict |
|---|---|---|---|
| 640×480 | 3.4 | **53 px** | far too small |
| 1280×720 | 6.7 | **107 px** | just under SFace's 112 |
| **1920×1080** | **10.1** | **161 px** | **comfortable** |

**So: capture 1080p, detect on a 320×180 downscale, and crop the face for SFace from the
full-resolution frame.** Measured cost: 5.45 ms decode + 1.30 ms downscale, inside the 20.4 ms
armed-tier budget in §2. Had the mount been on the monitor at 58° FOV, 720p would have done —
**the mount decision changes the camera you buy.**

**320×180, not 320×240, and the reason is aspect rather than pixels.** This document originally
recommended a 4:3 detect target for a 16:9 capture, which is a **1.33× anisotropic horizontal
squash** of every face in the frame — it spent 1080p on 161 px of face in §9 and then threw the
shape of it away at the downscale. Measured 2026-09-02 with the real YuNet weights, on a 1080p
frame carrying exactly the 161 px face the table above produces, 4 threads:

| Detect target | Aspect | YuNet score | resize + detect |
|---|---|---|---|
| 320×240 | 4:3 — **squashed** | **0.703** | 2.54 + 1.74 ms |
| **320×180** | **16:9** | **0.840** | **1.30 + 1.00 ms** |
| 416×234 | 16:9 | 0.839 | — + 2.10 ms |
| 640×360 | 16:9 | 0.894 | 1.06 + 2.97 ms |

**The aspect-correct target scores better on fewer pixels and costs 2.0 ms less**, because
1920×1080 → 320×180 is an exact 6:1 in both axes while → 320×240 is 6:1 and 4.5:1. And 0.703
against the old 0.700 `min_conf` is not a margin: two independent reconstructions of the same
scene put 320×240 at **0.62** and at **no detection at all**. `min_conf` therefore drops to
**0.6** as well, for the asymmetry — **a miss is silent and disables the feature outright**, while
a false face still has to survive `faces == 1` and 0.6 s of dwell before it can promote anything.

**One more thing the detect size moves, and it is not free either.** The five-point yaw proxy —
the discriminant the `cone_deg` threshold is applied to — is **not resolution-invariant**. On one
face, `yaw_u = (nose_x − eye_mid_x) / eye_span` measured 0.293 at 320×240, 0.281 at 320×180, 0.289
at 640×360 and **0.207 at 960×540**. Across the sizes actually in play it moves ~3 %, which is
tolerable; across the full range it moves 29 %. **So `cone_deg = 20` cannot be calibrated once and
reused at a different detect size** — whatever the $0 test measures, it measures at one size.
Write the size down beside the number.

All of these are synthetic frames. **The $0 test below is what settles them**, and
`tests/test_eye.py` now pins the *ratio* rather than the numbers so the capture and detect
settings cannot drift apart again.

### Parts

| | Part | Price | Why |
|---|---|---|---|
| **Buy** | **Arducam 2MP IMX462 STARVIS USB 3.0, 98° dFOV** (Amazon `B0CXXBD7KX`) | $50-70 | The wide lens the mount needs, plus a back-illuminated STARVIS sensor that actually sees in a lamp-lit room. UVC. Board camera: **no case, no shutter.** |
| Add | Slide-on lens shutter | ~$6 | A privacy kill he can *see*. Non-negotiable for a board camera |
| Add | Clamp/arm mount (mic-boom clamp or similar) | $15-25 | It is off-axis by 70 cm; it needs somewhere to live |
| Add | USB-A → USB-C adapter | ~$6 | The Spark's rear ports are USB-C |
| Separate | ESP32 + **HLK-LD2410C** — *already bought* | — | Owns presence in the dark and produces **no image**. `jarvis/roomsensor.py` and `scripts/esphome/jarvis-room-sensor.yaml` are written and never flashed |

**Do not buy the IR/night version**, even though `camera.md` listed it as an option. Three reasons,
and the first is new to this document: **NIR imagery breaks face identity.** Embeddings trained on
RGB degrade badly on near-infrared (the NIR-VIS domain gap), so night identity would need a second
gallery — a second copy of his biometric data, for a camera that is under curfew anyway. Second,
this mount points across the room at eye level. Third, **the LD2410 already owns the dark**, sees
through a duvet, and produces no image at all, which is the correct property.

**Do not share a USB hub with the Blue Snowball.** UVC and USB audio both reserve isochronous
bandwidth up front. Direct into a rear port.

### Before spending anything

`camera.md` §8 already specifies the $0 test and it is still the right first step: measure the real
angles with a tape, then take 20 phone photos from the candidate position — 10 looking at the
phone, 10 at the monitor — and check whether YuNet's five-point yaw proxy separates the two
clusters. **Take a third set from the recommended eye-level side position specifically**, and
while you are there, run one face through `cv2.FaceRecognizerSF` at the distance and lighting that
mount implies. That answers the resolution table above with his room instead of my arithmetic.

---

## 10. What was built today

TDD, camera-free, display-free, network-free. **67 new tests, all green;** whole suite passing
with only the known worktree-only `test_autostart.py` failure; ruff unchanged at 6.

**Then an adversarial read found nine real defects in it, and they are fixed** — five in the
store, three in the camera interface, one in the config, each with a test that fails against the
version before it. They are worth listing because every one of them let the 2026-09-02 shape recur
through a door this document claimed was shut:

| # | Where | What was wrong |
|---|---|---|
| 1 | `facegallery._prune` | oldest-first pruning meant **five ordinary saves deleted a six-take enrolment**, reproducing the incident's end state one loop later. The richest generation is now never pruned |
| 2 | `facegallery.save` | the shrink guard measured against what *this object* loaded, which is 0 for a freshly constructed caller — **the incident's own shape**. The baseline now also comes off the disk |
| 3 | `facegallery.save` | the tmp file was created **0664** under his umask and chmod-ed to 0600 only after `np.savez` returned. It is now 0600 from the `open()` |
| 4 | `facegallery.purge` | a crashed save's `gen-NNNNN.npz.tmp` holds a full set of embeddings and did not match the generation pattern, so **purge reported success and left one on disk**. Both purge and save now handle it |
| 5 | `facegallery.rollback` | returned `gens[-2]` regardless of what `load()` actually recovered — it reported 2 while holding 1 |
| 6 | `assistant_config` | **the shipped detect size could not resolve the face §9's own arithmetic produces**: 4:3 for a 16:9 capture, scoring 0.703 against a 0.700 bar. Now 320×180 + `min_conf` 0.6, and a test pins the ratio |
| 7 | `assistant_config` | capture defaulted to 640×480, which §9 itself calls "far too small" (a 53 px face against SFace's 112). Now 1920×1080 |
| 8 | `eye.SessionIdentity` | `room_empty()` had **no caller anywhere**, so its bound was unowned and a body vector outlived "offline mode" in RAM. `Eye` now fires `on_blind` on the deny edge |
| 9 | `eye.SessionIdentity` | `match_min` defaulted to a YoutuReID threshold that is **a coin flip for the phase-1 colour histogram** (§4). It is now a required keyword argument |

Three of the document's own numbers were also wrong and are corrected in place: the "13 % of one
core" budget (wall time presented as CPU occupancy — it is ~28 %), the 1080p MJPEG decode
(measured on an upscaled photo, ~1.6× optimistic), and "restic … should be in the same set"
(there is no set).

| File | What |
|---|---|
| `jarvis/facegallery.py` | the biometric store — generational saves, three guards, rollback, purge, provenance, format check |
| `jarvis/eye.py` | `Eye` (the device gate, both permission checks), `Attention` (the only thing that crosses the boundary — no pixels), `resolve_wake` (the fusion rule), `SessionIdentity` (the body anchor) |
| `jarvis/config.py` | `PATHS.FACE_GALLERY`, honouring `JARVIS_FACE_GALLERY` |
| `jarvis/assistant_config.py` | the `camera` section — and deliberately no schedule |
| `tests/conftest.py` | `JARVIS_FACE_GALLERY` forced into the throwaway dir |
| `tests/test_facegallery.py`, `tests/test_eye.py` | 23 + 44 tests |

**Not built, deliberately:** the sidecar itself, any model download, any cv2 inference, any wiring
into `hotword.py` or `app.py`. All of it needs a camera to be worth anything, and
`resolve_wake` is a pure function precisely so the wiring is a two-line change when one exists.

---

## 11. What could go wrong

1. **The geometry kills the attention half.** Free to test (§9). **The tiebreaker dies with the
   mount** — an earlier version of this line said to "build the wake tiebreaker and presence only",
   and the code does not support that: `resolve_wake` hard-requires `attending`
   (`jarvis/eye.py`), so with no usable attention signal it promotes nothing. A faces-only
   promotion is not the weaker version of it — it is the cost paragraph in `resolve_wake`'s
   docstring (*a stranger at his desk whose voice scores under the bar now wakes Jarvis*) with its
   only mitigation deleted. What survives an unusable mount is **presence**, which is worth the
   money on its own. Pinned by `test_the_tiebreaker_promotes_nothing_without_attention`.
2. **`pip install mediapipe` into `~/vss_env` replaces `cv2`** and takes down Jarvis and VSS
   simultaneously, silently, at install time. This design needs no new package at all.
3. **A second always-on process on a box that has already had one unified-memory power-off.**
   Mitigation: CPU-only by construction (measured: no CUDA in this `cv2`, no CUDA EP in this
   onnxruntime), `cv2.setNumThreads(4)` not 20, and a `StartLimitBurst` like `jarvis-f5.service`.
4. **Passive face learning drifts onto a visitor.** Mitigation: there is none in the design, and
   §5 says why there must not be.
5. **The gallery is not in the backup set — and neither is the voiceprint.** *Checked: there is no
   backup set at all* (§5). This is the live one.
6. **"No face" silently reads as "not attending".** `Attention.usable()` and the `None` return
   from `capture()` exist for this; the test that catches a regression is unplugging the camera
   mid-session and confirming wake behaviour is unchanged.
7. **A camera-triggered capture bypasses the mic arbiter.** `CLAUDE.md` is explicit that the
   arbiter "is a re-entrant depth counter, NOT a mutex". Any future camera-opened capture must go
   through `recorder.start` and respect `app._audio_busy` exactly as `_on_hotword` does
   (`app.py:1835-1850`).

---

## 12. Open questions

1. **Eye-level side mount, or the high shelf?** I recommend eye level and gave the reasoning
   (§9) — face recognition cannot be recovered from a downward pitch, presence can be recovered
   with a wide lens. But it is *his* desk, and the high shelf is tidier. This decides the camera.
2. **Identity at all, in phase 1?** I recommend **no**. The wake tiebreaker works on
   `faces == 1` + attending, with no gallery and **nothing about his face written down anywhere**.
   Turn `camera.identity` on only once a second person in frame demonstrably causes a false
   accept. The store is built so that day is cheap; that is not a reason to bring it forward.
3. ~~**Is `~/.aiws_trainer/` in the restic set?**~~ **Answered, and the answer is no** — see §5.
   There is no restic set at all on this box: the binary, and nothing else. This is a finding, not
   a question, and it is the most urgent item here. **A set has to be created before the gallery
   exists.** It needs one decision from him: where the repository lives (a second disk, or
   somewhere off this machine).
4. **Who owns the "close your eyes" phrasing?** The offline lane owns the command family. The
   camera-side requirement is only that stopping it stops **the sidecar**, not merely the
   consumer — stopping the consumer while the device stays open is the dishonest version, and the
   easy one to write by accident.
5. **Does the wake tiebreaker go in before or after a week of `eye=` logging?** I recommend
   **after**: add the field to the existing one-line `wake candidate:` log (`hotword.py:502`),
   run a week, and count what it would have promoted. n ≥ 30 before changing a default — the same
   discipline as the voice rounds.

### Sources

- [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) — model sizes (git-lfs), per-model `LICENSE` files, and the published ARM benchmark table
- Cloth-changing person re-ID benchmarks: **PRCC** (*Person Re-identification by Contour Sketch under Moderate Clothing Change*) and **LTCC** (*Long-Term Cloth-Changing Person Re-identification*)
- [Rethinking the Domain Gap in Near-infrared Face Recognition (arXiv 2312.00627, CVPRW 2024)](https://arxiv.org/abs/2312.00627)
- In-repo: `scratchpad/ideas/camera.md`, `scratchpad/wake-gate/wake_gate_diagnosis.md`, `docs/room-sensor.md`, `jarvis/hotword.py`, `jarvis/speaker.py`, `jarvis/presence.py`, `jarvis/roomsensor.py`, `tests/conftest.py`; and VSS's `aiws_system/face_obscurer.py`
