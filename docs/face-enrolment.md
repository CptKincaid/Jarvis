# Face enrolment — the command, the numbers, and what can go wrong

*Built 2026-09-02. Companion to `docs/vision.md` §5 (enrolment and storage), which
states the incident this design answers.*

```bash
~/vss_env/bin/python scripts/face_enrol.py                 # enrol
~/vss_env/bin/python scripts/face_enrol.py --status        # opens no device
~/vss_env/bin/python scripts/face_enrol.py --verify        # does it match me?
~/vss_env/bin/python scripts/face_enrol.py --backup ~/face-backup
~/vss_env/bin/python scripts/face_enrol.py --restore ~/face-backup
~/vss_env/bin/python scripts/face_enrol.py --rollback      # undo the last save
~/vss_env/bin/python scripts/face_enrol.py --delete        # destroy it
```

---

## The rule this is built around

**Nothing looks at what the camera sees.** Not to debug it, not to confirm a
detection, not once. The consequence is that enrolment cannot be "watch the
preview until it looks right", so the whole command is designed around a
different workflow:

> **He runs it. He pastes the output. Somebody who has never seen him says
> whether the enrolment is good.**

Everything is arranged to make that work. Per sample: a detector confidence, a
face size in capture pixels, an interocular distance, a head yaw and roll, a
sharpness, and accepted/dropped with the reason. Per gallery: the sample count,
the yaw spread, the full pairwise-cosine distribution, the per-sample cohesion,
and a PASS/FAIL on each. `jarvis/visionrig.assert_numbers_only` walks the whole
report before a line of it is printed and raises on anything that is not a
number, a string or a container of those — so "no image data leaves this
script" is a property of the structure, not of the print statements.

**What reaches the disk is one thing: the 128-float embedding.** No frame, no
crop, no thumbnail, no debug JPEG. Crops exist inside `SFaceRecogniser.embed`
and inside `faceenrol.sharpness` and die when those calls return.

---

## Why 13 samples, and why not one head position

The precedent that exists on this box is his **voiceprint: 11 ECAPA embeddings,
pairwise cosine 0.289–0.735.** That is a real *spread* — the enrolment script
that produced it deliberately varies distance, pace and loudness
(`scripts/enroll_voice.py:41-61`), because eleven readings of one sentence
would have been eleven copies of one measurement.

The face equivalent of "one sentence eleven times" is **one head position**, and
his own usage guarantees it goes wrong: measured on his camera 2026-09-02, his
yaw is **~14° looking at the lens and ~54° looking at his screen.** A gallery
built at 14° stops working the moment he turns to work — and it fails
*silently*, presenting as "Jarvis doesn't know me any more".

So the run walks five stations, 13 samples:

| station | what he is asked to do | yaw window | samples |
|---|---|---|---|
| `lens` | look straight into the lens | −15…+15° | 3 |
| `screen` | look at the screen as when working | +18…+62° | 3 |
| `across` | turn the other way | −62…−18° | 3 |
| `back` | at the lens, sitting further back | −15…+15° | 2 |
| `close` | at the lens, leaning in | −15…+15° | 2 |

The stations **guide**; they do not judge. A station his desk cannot produce —
the camera clamped on the side his screen is not on — costs that station and not
the enrolment, and the run says "only 1 of 3 here — moving on". The **pose
distribution that actually came out** is what gets judged, at the end.

A minimum 0.35 s gap between accepted samples is enforced, because eight frames
at 8 fps is one second of one pose: eight near-duplicates that would then fail
the variation check for a reason he could not act on.

---

## The quality gate — a bad gallery is worse than none

Each candidate frame is judged **before** any embedding is computed. The order
matters and the first bar is not negotiable:

| bar | default | why |
|---|---|---|
| one face in frame | 1 | a gallery that learned a visitor identifies the wrong person confidently, and nothing about it looks wrong afterwards |
| **detector confidence** | `camera.min_conf` (0.6) | **see below — this is the whole gate** |
| eye landmarks distinct | — | coincident eyes means the crop cannot be aligned and every angle below is invented |
| face size | ≥ 112 px | SFace's input is 112×112; smaller is being upsampled, and the model costs the same either way |
| interocular | ≥ 35 px | ArcFace's 112×112 alignment template puts the eye centres 35.2 px apart; under that the aligner is magnifying |
| yaw | ≤ 62° | above his 54° working pose, so it does not refuse his most common head position |
| roll | ≤ 30° | a head resting on a hand, and the aligned crop is rotated by the same amount |
| sharpness | ≥ 0.015 | **provisional** — gross motion blur only, see below |

### The confidence bar is the only thing standing between a bad crop and a poisoned gallery

Measured on this box 2026-09-02 (`jarvis/facemodels.py`): SFace's 128-D
embedding **collapses on out-of-distribution input**. Unrelated *non-face* crops
match each other at mean cosine 0.66–0.92, with **94–100% of pairs above the
0.363 "same person" bar**:

| input family | mean cosine | fraction ≥ 0.363 |
|---|---|---|
| uniform noise | 0.848 | 1.000 |
| flat colour | 0.923 | 1.000 |
| random blobs | 0.714 | 1.000 |
| smooth gradients | 0.660 | 0.937 |

So a false-positive box, a motion blur or a bad alignment **does not score low
— it scores confidently.** The embedding is not a second opinion on whether this
is a face and must never be used as one. That rule is enforced in three places,
deliberately: `EnrolmentSession.offer` judges before it embeds,
`SFaceRecogniser.embed` refuses a row under the bar
(`jarvis/facedetect.py:212-226`), and `FaceIdentifier.identify` refuses it again
on the matching path. A rejected sample costs **zero** embeddings, and a test
counts the recogniser's calls to prove it.

### The sharpness number is provisional and says so

Nobody has measured a blur bar on his face, and the asymmetry is one-sided: a
bar set too high makes enrolment impossible with a confusing message, while a
bar set too low lets a slightly soft sample through where the cohesion check
catches it. So 0.015 catches **gross motion blur and nothing finer**, the report
prints the whole distribution, and `--min-sharpness` moves it once there is
data. The measure is gradient energy over the face box normalised by its own
mean level, subsampled to ~64 px on the long side so a 300 px face and a 130 px
face are comparable — otherwise a "blur bar" would really be a distance bar.

---

## The two refusals

The gallery is judged as a whole, and **it is not saved if it fails.**

**Too tight** — near-duplicate samples (pairwise cosine p50 > 0.98) or no pose
variation (yaw range < 25°, or fewer than 2 frontal and 2 off-axis samples).
This gallery knows him in one position and rejects him in every other.

**Too loose** — some sample's *median* cosine to the others is under the 0.363
"same person" bar. That is either a different person who walked through frame or
a crop that is not a face. Median rather than mean, so one genuinely distant
pose does not condemn a sample; per-sample rather than pool-wide, because
`FaceGallery.match` scores against the pool's **best** member, so one poisoned
sample is all it takes and the pool's own median stays healthy while it sits
there.

Both failures are *silent* in use, which is why neither may be saved. The report
names which one, with the numbers. `--force` exists and records itself in the
generation's provenance string, so a forced save can be identified later.

---

## Storage, backup and deletion — and the honest part

**Where.** `~/.aiws_trainer/face_gallery/`, directory **0700**, files **0600**,
created at those modes rather than chmod-ed afterwards (under his 0002 umask the
plain form is 0664 for the whole of `np.savez`). `--status` prints the modes it
found and repairs them.

**Generational.** A save never overwrites: it writes `gen-00002.npz` beside
`gen-00001.npz`, and `load()` takes the newest that *parses*. Five are kept,
never fewer than two, and **the richest generation is never pruned**.
`--rollback` undoes the last write.

### Say this part plainly

> **The generations are not a backup.** Every one of them lives in the same
> directory on the same disk, so a single disk failure takes all of them at
> once. They make a *bad write* recoverable. They do nothing about hardware.
>
> **There is no backup system on this box.** `restic` is installed in
> `~/.local/bin` (28 MB, 30 Jun); there is no repository, no `~/.config/restic`,
> no `RESTIC_*` in the shell profile, no user timer, no script. **Nothing under
> `~/.aiws_trainer/` is backed up** — which is precisely why the voiceprint loss
> on 2026-09-02 was unrecoverable.
>
> `--backup DIR` is the only thing that puts this enrolment somewhere else, and
> **it is only as good as where you point it: another directory on the same disk
> is a copy, not a backup.**

`--backup` copies every generation and then **reads each copy back** and checks
it parses. That check is the point: a backup nobody has read back is exactly the
shape the voiceprint's `.corrupt-<date>` copy had — it existed, it was named like
a backup, and it held only the fixtures.

`--restore DIR` loads the newest generation from the backup and saves it as a
**new generation** of the live gallery. It replaces the live pool rather than
merging into it (a pool that is two enrolments at once is one nobody can reason
about), and it overwrites nothing — so a restore of the wrong thing is one
`--rollback` away.

`--delete` destroys **every** generation *and* any `gen-NNNNN.npz.tmp` a crashed
save left behind — a tmp holds a full set of embeddings under a name the
generation pattern does not match, so a "delete" that skipped it would leave a
measurement of his face on the disk at 0600. It then reports what is left in the
directory, and sets `camera.identity` to false, because recognition running
against a gallery he has just destroyed is a feature that is on and cannot work.

### The test firewall

`tests/conftest.py` forces `JARVIS_FACE_GALLERY` (and `JARVIS_FACE_MODEL_DIR`)
into a throwaway directory — *forced*, not `setdefault`, so a shell that exported
the real path cannot defeat it. `tests/test_faceenrol.py` asserts the redirect
holds rather than trusting it was set, including through `default_gallery()`,
which is the exact call the 2026-09-02 accident made against the voiceprint.

---

## The matching path

`jarvis/eye.FaceIdentifier` is the gallery asked "who is this", and it is the
only thing that asks:

* a row under `camera.min_conf` returns `("", 0.0)` with **no embedding
  computed**, and `gated_out` counts how often that happened so the number shows
  up in a report rather than being inferred;
* an empty gallery returns `("", 0.0)` without paying for an embedding;
* a score under `camera.identity_min` returns `("", 0.0)`. The default 0.363 is
  **OpenCV's own documented SFace cosine threshold for "same person"** — the same
  number `jarvis/facegallery.SFACE_COSINE_SAME` carries and the config ships;
* a broken recogniser is no opinion, never an exception.

### What a name may do

**Identity may REMOVE capability or ADD a name. It must never GRANT capability
the existing gates do not already grant.** That is his standing ruling, and it is
pinned by test rather than by comment:

* `resolve_wake` promotes on `identity in ("", owner)`, so a recognised **him**
  gets exactly what an anonymous single attending face already got — only the log
  line differs. `test_recognising_him_grants_nothing_an_anonymous_face_lacked`
  asserts equal `ok` and `guest_ok` across every combination of verdict, face
  count, attention and dwell.
* A recognised **stranger** blocks a promotion an anonymous face would have got,
  and drops the `SessionIdentity` body anchor. That is the direction identity is
  allowed to act in.
* A face that is present but *not confirmed* to be him also drops the anchor.
  Below the bar the nearest label means nothing — the gallery always has a
  nearest member — so "matched him weakly" and "matched somebody else" are the
  same state.

`--verify` runs the live camera through `visionrig.Rig` with the identifier
attached and reports matched frames, unknown frames, how many detections were
under the detector bar, and the match-cosine distribution. That is the proof the
matching path works, obtained without looking at anything.

---

## Sensing owns the lens

Offline mode, the 21:00–07:00 curfew and the fail-safe are read from
`jarvis/sensing.py`, and the device is opened through `jarvis/camera.CameraFeed`
— the same objects the running Jarvis uses, not a copy. **Sensing is asked
first**, before the models are even looked for: a run that reported "the weights
are missing" while the real answer was "you are in the curfew" would send him to
fix the wrong thing.

If the curfew arrives mid-enrolment, `_GatedDevice.read` returns `(False, None)`,
the run **stops and says so**, and nothing is saved. It does not finish quietly
with half a gallery.

Exit codes: `0` fine, `1` something failed a check, `2` sensing said no, `3`
nothing to work with (no camera, no models, no gallery).

---

## Him only

`camera.identity` is the phase gate: with it false, nothing about his face is
written down at all, and `--enable-identity` is the deliberate act that turns it
on. The label written is his, derived from `user.name`, and there is no
`--label` flag. **Enrol him only — no family.** That is his ruling from the
09-02 decisions, and the reason is the one above: the rule that identity may
never grant capability gets harder to hold the more names exist.

---

## Still unmeasured, and only he can measure it

* **Whether a real face clears `camera.min_conf` at 320×180.** Untouched by any
  of this; it needs the camera and a face.
* **The sharpness bar**, above.
* **The SFace cosine his own face produces across the 14°–54° yaw range.** The
  0.363 bar is OpenCV's documented figure, not a measurement of him. `--verify`
  is how that number arrives, and if it turns out his off-axis samples land under
  it, the answer is `camera.identity_min`, from data.
* **`camera.nose_ratio` = 0.35** is an anthropometric assumption, so every yaw in
  degrees inherits it. The report prints the raw ratio too; the $0 photo test in
  `docs/vision.md` §9 calibrates it.
