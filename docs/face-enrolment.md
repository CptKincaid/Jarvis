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
~/vss_env/bin/python scripts/face_enrol.py --append        # add to the pool
~/vss_env/bin/python scripts/face_enrol.py --reset         # replace the pool
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
| the bar is a bar | ≥ 0.30 | `camera.min_conf` is user-editable and at 0 it removed the gate entirely — a 0.01 detection was accepted, embedded and stored, with nothing saying so. Below the detector's own filter it cannot reject anything, so it is refused; a bar merely *lowered* from 0.60 is allowed and printed in the report header |
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

Two things had to be true for "three gates on one rule" to mean anything, and
neither was:

* **They must fail shut.** Every bar was spelled `value < bar`, and `NaN < 0.6`
  is False — so a non-finite score cleared the confidence gate *and then every
  bar under it*, in all three places at once. They are now spelled
  `not (value >= bar)`, so a NaN passes nothing.
* **They must read the same number.** The judge read `row[IDX_SCORE]` (column
  14); the recogniser and the identifier read `row[-1]`. Identical for YuNet's
  15 columns and divergent for anything longer, in *both* directions — with a
  16-column row a 0.45 detection the judge would refuse got embedded, and a 0.99
  detection got dropped. All three now read column 14, and a row of any other
  shape is refused loudly at the seam rather than having some other number
  silently read as its confidence.

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

**Too loose** — some member's *median* cosine to the others is under the 0.363
"same person" bar, **or** far below the pool's own median. Median rather than
mean, so one genuinely distant pose does not condemn a member; per-member rather
than pool-wide, because `FaceGallery.match` scores against the pool's **best**
member, so one poisoned member is all it takes and the pool's own median stays
healthy while it sits there.

### Why the cohesion check has two terms, and what it still cannot see

0.363 is OpenCV's **verification** threshold — "are these two the same person,
at some FAR" — and a healthy pool's own median sits near **0.80**. An absolute
0.363 bar is therefore 0.44 *below* the distribution it is supposed to police:
measured on synthetic pools, it only fires below ~0.40. A foreign member at
cosine 0.45, 0.50 or 0.60 to the pool cleared every check.

So the check is relative as well as absolute. A member fails if it is under
0.363 **or** under `pool median − max(0.20, 6 × pool MAD)`. Scaled by the pool's
own MAD because his yaw runs 14°–54° and an enrolment that covers it is
*supposed* to be spread — a fixed margin alone would refuse the thing the
stations exist to produce. With that second term the refusal now fires from
about 0.62 downward instead of 0.40.

**Say what it still does not catch.** This is an outlier test: it finds one
member unlike a pool. It does **not** find a pool that is *two* tight clusters
at a cross-cosine above the bar — with an even split every member's median *is*
the cross value, so the shape is invisible to any per-member statistic. What
stands there instead is the one-face-in-frame rule (a second face is refused
before any embedding), the single-label rule, and judging the pool that will
actually be written rather than the batch that was captured. The 0.66–0.92 band
that non-face crops occupy is likewise only partly covered; the detector
confidence bar, not this check, is what keeps non-faces out.

Both failures are *silent* in use, which is why neither may be saved. The report
names which one, with the numbers — including the relative floor and the margin
it was computed from, so a refusal is arithmetic he can check rather than a
verdict he has to trust. `--force` exists and records itself in the generation's
provenance string, so a forced save can be identified later.

> **`camera.identity_min` is two knobs wearing one name.** It is the enrolment's
> absolute cohesion floor *and* the live match bar. Lowering it because his
> off-axis samples land under 0.363 also lowers the bar that keeps a stranger
> out of the gallery, by the same amount, silently. The relative term above is
> deliberately independent of it, so tightening or loosening `identity_min` no
> longer moves the whole refusal. If they ever need to move separately, they
> need to be two keys.

---

## Enrolling again: three commands, and what each destroys

| command | captures | what it saves | what it destroys |
|---|---|---|---|
| *(no flag)* | a fresh pool | a **new** generation | nothing |
| `--append` | a fresh pool | the loaded pool **plus** the new one | nothing |
| `--reset` | a fresh pool | a **new** generation | the older generations, **after** the save |

**Plain re-run is the one you want.** The previous enrolment stays on disk and
is one `--rollback` away.

**`--append` judges the merged pool, not the batch.** It loads the existing pool
first and `save()` writes loaded+new, so the checks run over what will actually
reach the disk. They used to run over the captured batch alone, and the gap was
not cosmetic: demonstrated synthetically, 13 embeddings of him as generation 1
plus an `--append` of 13 embeddings of a **different identity** (cross cosine
0.10) passed all four checks and was written under his label — while the same
checks over the 26 that landed say `[FAIL] cohesion` with a worst median of
0.075, and `match()` then returned `('hunter', 1.0000)` for the stranger. The
report's `pool` line says which set the cosines describe.

**`--reset` captures first and destroys last.** It used to purge before the
capture, so a run that then failed a check — 'too tight' / 'too loose', the
outcome this whole design exists to produce — left an empty directory and
nothing to roll back to. Now: it asks him to type `reset` (the same standard
`--delete` has, on the same data), says how many generations are going, captures
and saves, and **only then** drops the generations that predate the new one. If
anything fails, it destroys nothing and says so. It also no longer side-steps
the shrink guard by emptying the disk first — a smaller replacement still needs
`--allow-shrink`, which is the guard doing its job.

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

Two things it refuses or reports, both from real failure shapes:

* **A destination that is the gallery, or inside it, is refused before a byte is
  written.** `--backup ~/.aiws_trainer/face_gallery` — one typo from the path in
  every doc — used to open each live generation with `O_TRUNC` and rewrite it
  from itself, reporting `copied: 1, verified: 1, ok: True` while doing it. An
  interruption after the truncate left `gen-00001.npz` at 0 bytes with `load()`
  returning False: the enrolment gone, from the command whose job is to keep it.
  Each copy is now written tmp + `os.replace`, the way `FaceGallery.save` is, so
  a crash mid-copy costs the tmp and never the file it replaces.
* **It reports what the destination HOLDS, not only what was copied into it.** A
  backup directory is never reconciled with the source, so after a `--rollback`
  the discarded generation is still sitting there — and it is still the *newest*,
  which is the only thing `--restore` looks at. The report names the generation
  list, the newest and its sample count, and flags any generation that is in the
  backup but no longer in the live gallery.

`--restore DIR` loads the newest generation from the backup and saves it as a
**new generation** of the live gallery. It replaces the live pool rather than
merging into it (a pool that is two enrolments at once is one nobody can reason
about), and it overwrites nothing — so a restore of the wrong thing is one
`--rollback` away. It prints which generation it took, how many samples that
holds, and how many were live before, because those three numbers are what say
whether it just resurrected an enrolment he had deliberately rolled back.

`--delete` destroys **every** generation *and* any `gen-NNNNN.npz.tmp` a crashed
save left behind — a tmp holds a full set of embeddings under a name the
generation pattern does not match, so a "delete" that skipped it would leave a
measurement of his face on the disk at 0600. Each file is **overwritten with
zeros and then unlinked**: unlink alone removes the directory entry and leaves
the vectors in the extents. It then reports what is left in the directory, and
sets `camera.identity` to false, because recognition running against a gallery
he has just destroyed is a feature that is on and cannot work.

**What `--delete` is allowed to claim, and what it is not.** It says "the gallery
at *path* is gone: N files overwritten and unlinked", not "your face is no longer
on this disk" — that older line was wrong twice over. A `--backup` copy survives
it fully loadable (the command's own confirm prompt already conceded that three
lines earlier), so the closing text now says copies are **not** touched and he
must delete them himself. And the overwrite is a *filesystem*-level erase: on a
copy-on-write filesystem, on a journalled one that already wrote the block
elsewhere, and on any SSD whose controller remaps rather than rewrites, the old
blocks can survive. What is true is that the bytes are gone from the path
anything reads, and that is exactly what it says.

### The test firewall

`tests/conftest.py` forces `JARVIS_FACE_GALLERY` (and `JARVIS_FACE_MODEL_DIR`)
into a throwaway directory — *forced*, not `setdefault`, so a shell that exported
the real path cannot defeat it. `tests/test_faceenrol.py` asserts the redirect
holds rather than trusting it was set, including through `default_gallery()`,
which is the exact call the 2026-09-02 accident made against the voiceprint.

The redirect is **also** asserted in the session-scoped `_firewall_live_log_dir`
fixture, beside the two voiceprint assertions. A forced env var is one belt; the
voiceprint was given two *after* it was lost, and a face embedding is the same
kind of irreplaceable measurement. The difference is when it fires: a fixture
fails at session start, an ordinary test fails somewhere inside a 7,000-test run
— possibly after something has already written. And the real gallery does not
exist on this box yet, so a leak now would silently *create* a fixture gallery at
the real path, which is harder to notice than corrupting one.

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
* **The false-accept rate. Nothing measures it.** Everything above is validated
  on the false-*reject* side: `--verify`'s gallery check asks only whether the
  gallery matches **him** (`id_matched_frames > id_unknown_frames`). No path in
  `jarvis/` or `scripts/` measures whether it rejects anybody else. A number can
  be had without enrolling a second person and without breaking the him-only
  ruling: run `--verify` while somebody else sits in the chair, and read the
  match distribution against the bar — the report already prints
  `match_p50`/`match_min_score` and the gallery bar. That is a measurement, not
  an enrolment.
* **Liveness. There is none, at any stage.** The detector gate is a *face-ness*
  test, not a *live-ness* test, so a face on a monitor, in a video call, or in a
  printed photograph is indistinguishable from a face in the room — at enrolment
  and at match. Nothing here claims otherwise, and nothing downstream grants a
  capability on a name (see **What a name may do**), which is what keeps that
  from being urgent rather than what makes it untrue.
