"""Speaker verification using SpeechBrain ECAPA-TDNN.

V3 evolution of speaker_verification.py (same math, same thresholds):
- no reverse import of voice_input_gui (uses jarvis.logs)
- atomic voiceprint save (np.savez to .tmp file handle + os.replace)
- gpu default 0 with automatic CPU fallback (GB10 / no-CUDA machines)
- fail-open verify kept, but logged at WARNING and surfaced once per
  session as a Status(kind="warn") event on the bus

Stores the voiceprint as a set of 192-dim embeddings in a .npz file.
Matching uses cosine similarity against the centroid of all embeddings.

Usage:
    verifier = SpeakerVerifier()          # gpu=0 default, CPU fallback
    verifier.load_model()
    verifier.load()

    ok, n = verifier.enroll_from_audio(audio_16k_np_array)
    is_user, score = verifier.verify(audio_16k_np_array)
    verifier.add_sample(audio_16k_np_array)     # passive learning
"""
from __future__ import annotations

import dataclasses
import os
import threading
import time

import numpy as np

from jarvis import voicegallery as vgal
from jarvis.config import PATHS
from jarvis.events import Status, bus
from jarvis.logs import get_logger

log = get_logger("speaker")

VOICEPRINT_FILE = PATHS.VOICEPRINT                    # ~/.aiws_trainer/voiceprint.npz
SPEAKER_MODEL_DIR = PATHS.AIWS / "speaker_model"

# Cosine similarity threshold for accepting a speaker match
# Lower = more permissive, higher = stricter
# 0.30, measured (2026-08-29): with silence no longer pooled into the
# embedding, this speaker's genuine short utterances score 0.377-0.397 and
# every non-match seen (silence, TTS, other voices) sits at -0.03..-0.10.
# speaker_tuning.recommend() on that data puts 0.40 at a 40% false-reject rate
# and 0.30 at 0%/0%. The live value lives in voice_settings.json; this default
# is what a fresh box or a reset gets, so it must not be the number that was
# rejecting the user.
DEFAULT_THRESHOLD = 0.30

# Maximum stored embeddings (oldest beyond this are dropped)
MAX_EMBEDDINGS = 100

# Passive samples one PROCESS may add to the voiceprint. See add_sample for
# the drift measurement that produced the number: at the label's own genuine
# floor a single passive sample rotates a 14-take centroid to cos 0.997 and
# leaves an intruder at 0.220, two leave it at 0.991 / 0.281, and three put
# the intruder INSIDE a 0.30 bar at 0.334. Two is the last safe value and it
# is the one taken. The app's own 10-minute cooldown and 0.55 bar sit in
# front of this (app._maybe_learn_voice); this is the floor under them.
PASSIVE_CAP = 2

# Minimum audio length (seconds) for a useful embedding
MIN_AUDIO_SECONDS = 1.0
# Below this much TRIMMED speech a score is not evidence (see verify): the
# 2026-08-31 study measured 30% false rejects at 1.0 s against 0% at 3.0 s
# on Hunter's own voice. Between the two, abstaining beats guessing.
ABSTAIN_SECONDS = 1.5
SAMPLE_RATE = 16000

# Silence is not ignored by ECAPA -- it is pooled in like any other frame, so
# a window's score lands near the speech-fraction-weighted average of the
# speaker's true score and the score of silence (about -0.10 here). Measured
# on a real rejection: a 3.9 s capture holding 1.4 s of speech scored 0.253
# where the same voice scores 0.63 clean; 0.47*0.63 + 0.53*-0.10 = 0.25. With
# a 0.40 threshold that demands ~68% of the window be speech, which no short
# utterance survives once the recorder's 2.5 s silence auto-stop is appended.
FRAME_MS = 20
SPEECH_PEAK_FRACTION = 0.08      # of the clip's own loudest frame
SPEECH_FLOOR_MULTIPLE = 3.0      # of the clip's own noise floor
MIN_SPEECH_SECONDS = 0.35        # below this we cannot tell, so change nothing

# Bumped when the embedding pipeline changes in a way that makes stored
# embeddings incomparable with fresh ones. 2 = probes are silence-trimmed.
VOICEPRINT_FORMAT = 2

# EVERY FORMAT THIS BUILD CAN READ, and the refusal of anything else is the
# whole point of the tuple. 1 is the pre-trim pool (loadable, and load() says
# out loud that it scores low); 2 is the current one.
#
# WHY A REFUSAL AND NOT ANOTHER WARNING. This file is a SINGLE-SPEAKER pool:
# load() selects its keys by ``k.startswith("emb_")``, which also matches
# ``emb_mara_0000``, and the only format complaint it had fired when
# ``fmt < VOICEPRINT_FORMAT`` -- i.e. never for a NEWER file. So a multi-label
# voiceprint written by a later build, read by this one after a rollback or by
# a stale process, pooled every person into one centroid without a word.
# Measured 2026-09-04 on synthetic vectors (tests/test_speaker_format.py):
# a _format=3 file holding two people loaded as 20 pooled samples and 6 of 6
# of the OTHER person's takes then scored past the 0.30 bar as him -- a silent
# universal false accept, arriving by a path nobody would think to look at.
# That is the television-reaches-the-commander failure, so an unrecognised
# format loads NOTHING and says which number it saw.
#
# Refusing leaves the pool empty, so both gates fall back to their
# nothing-enrolled behaviour: the wake gate and the transcript gate both fail
# OPEN and voice keeps working, unfiltered and loudly logged. That is the
# right direction -- an unreadable pool is "no instrument", and no label is
# minted from it, where pooling would have named a stranger as him.
KNOWN_VOICEPRINT_FORMATS = (1, 2)

# THE OWNER'S TWO POOLS ARE ONE POOL, AND THE MEASUREMENT IS WHAT SAYS SO.
# ``voiceprint.npz`` has no label, and ``--migrate`` copies its vectors
# unchanged into the gallery under his label, so a window of his voice scores
# the two centroids identically up to float rounding and the maximum lands on
# whichever came first. Identity is decided PER WINDOW (see filter_segments),
# and a clip of his that split between "the voiceprint" and "hunter" would
# have half its windows dropped as somebody else's. So a gallery label is
# folded into the voiceprint's pool when its centroid MEASURES as that pool.
#
# IT USED TO BE FOLDED FOR THE LABEL STRING ALONE -- ``if label ==
# self.owner_label`` with no cosine, no provenance and no consent -- and that
# was the hole the 2026-09-05 review reproduced at 100/100 through two routes:
# anybody's takes recorded under his name WERE him. His name is not a
# credential; ``voiceprint.npz`` is. See voicegallery.OWNER_POOL_COSINE for
# the number and the measurement behind it. The one thing the label string is
# still trusted for is BOOTSTRAP: with no voiceprint at all there is nothing
# to measure against, and the gallery is the only anchor identity has.
MIGRATED_ALIAS_COSINE = vgal.OWNER_POOL_COSINE


def _frame_rms(audio_16k, n):
    frames = np.asarray(audio_16k[:len(audio_16k) // n * n],
                        dtype=np.float32).reshape(-1, n)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)


def _speech_threshold(rms):
    """Per-frame RMS above which a frame counts as speech, or None when the
    clip is too flat to tell (the quietest tenth of frames sits within 3x of
    the loudest -- all speech, all room, or a sustained tone). Thresholds are
    relative to the clip's own peak and floor, so quiet speech survives and a
    loud room does not swallow it."""
    if rms.size == 0:
        return None
    peak = float(rms.max())
    if peak <= 0.0:
        return None
    thr = max(peak * SPEECH_PEAK_FRACTION,
              float(np.percentile(rms, 10)) * SPEECH_FLOOR_MULTIPLE)
    return thr if thr <= peak else None


def speech_bounds(audio_16k, frame_ms=FRAME_MS, min_speech_s=MIN_SPEECH_SECONDS):
    """(start, end) sample bounds of the speech in a clip, or None when no
    speech can be told apart from the room (too short, digital silence, or a
    flat clip -- see _speech_threshold). Endpoints only: interior pauses are
    speech rhythm and are present in the enrolment audio too."""
    n = int(SAMPLE_RATE * frame_ms / 1000)
    if n <= 0 or len(audio_16k) < 2 * n:
        return None
    rms = _frame_rms(audio_16k, n)
    thr = _speech_threshold(rms)
    if thr is None:
        return None
    voiced = np.nonzero(rms >= thr)[0]
    lo = max(0, int(voiced[0]) - 1)                  # one frame of margin so a
    hi = min(len(rms), int(voiced[-1]) + 2)          # soft onset is not clipped
    if (hi - lo) * n < int(SAMPLE_RATE * min_speech_s):
        return None
    return lo * n, hi * n


def trim_silence(audio_16k, frame_ms=FRAME_MS, min_speech_s=MIN_SPEECH_SECONDS):
    """Strip leading and trailing silence before an embedding is taken.

    Returns the audio unchanged whenever speech_bounds cannot confidently
    find speech -- a trim that guessed wrong would fail shut on the
    transcript gate, and unchanged is exactly the pre-trim behaviour.
    """
    bounds = speech_bounds(audio_16k, frame_ms, min_speech_s)
    if bounds is None:
        return audio_16k
    return audio_16k[bounds[0]:bounds[1]]


class SpeakerVerifier:
    """ECAPA-TDNN speaker verification with enrollment and passive learning."""

    # CLASS-LEVEL DEFAULTS, not only __init__ ones. Several tests build a bare
    # verifier with ``object.__new__(SpeakerVerifier)`` to exercise the pure
    # arithmetic without a model, and an attribute that exists only in
    # __init__ turns every one of them into an AttributeError the moment a new
    # field is added. The defaults are also the SAFE state: no gallery, no
    # frozen reference, nothing learned.
    gallery = None
    _frozen_centroid = None
    _passive_added = 0
    format_fault = ""
    # The owner's gallery label, so his migrated pool and his voiceprint are
    # read as ONE pool (see MIGRATED_ALIAS_COSINE). "" is "nobody said".
    owner_label = ""

    def __init__(self, gpu=0, threshold=DEFAULT_THRESHOLD, owner_label=""):
        self.gpu = gpu
        self.threshold = threshold
        self.owner_label = str(owner_label or "")
        self._model = None
        self._embeddings = []       # List of 192-dim numpy arrays
        self._centroid = None       # Mean of all embeddings
        self._format = VOICEPRINT_FORMAT   # of the pool on disk; see load()
        self._lock = threading.Lock()
        self._loaded = False
        self._model_loaded = False
        self._device = None         # resolved by _resolve_device()
        self._warned_fail_open = False   # one Status(warn) per session
        self._warned_disowned = False    # one warning per session
        self._model_failed = False       # a failed load is not retried
        # "" or one plain sentence naming the format on disk that this build
        # refused to read. Held rather than only logged so a startup line and
        # the instrument can say WHY the voice leg is dark, the way
        # facegallery.reenrol_message does for a cross-model gallery.
        self.format_fault = ""
        # THE MULTI-SPEAKER STORE, BESIDE THE VOICEPRINT AND NOT INSTEAD OF IT.
        # ``voiceprint.npz`` stays exactly what it was -- one pool, one
        # centroid, his -- and is the rollback. The gallery adds labels. Both
        # are consulted; see ``_all_centroids``.
        self.gallery = None
        # The reference passive learning is measured against, captured the
        # first time it is needed and never moved afterwards. See add_sample.
        self._frozen_centroid = None
        self._passive_added = 0

    # ------------------------------------------------------------ state
    @property
    def is_enrolled(self):
        """Whether THE OWNER has a pool to be verified against -- the
        voiceprint, or his own label in the gallery. ``enrolment_gap`` is
        the same question with the sentence attached.

        BOTH HALVES OF THE "OR" ARE LOAD-BEARING AND IN OPPOSITE DIRECTIONS.
        False here makes the wake gate and the transcript gate BOTH fail open,
        which is what keeps a fresh box from being mute; True makes the
        transcript gate fail shut, which is what stops a television reaching
        the commander. So a box with only a gallery must read True WHEN HE
        IS IN IT (enroll_voice --reset after --migrate), and a box with
        neither must read False (or it goes silent on its first day).

        AND A GALLERY HOLDING ONLY OTHER PEOPLE IS NO INSTRUMENT FOR HIM.
        The first version read any gallery label as an enrolment, arguing
        the person who just enrolled must be filtered for. Measured
        2026-09-04 (tests/test_voice_owner_lockout.py) on the layout that
        produces -- a guest enrolled first on a box with no voiceprint --
        that reading woke him 0 of 50 (the wake gate's fail-open None had
        become a number under the bar, against HER centroid) and admitted
        him 0 of 50 on the transcript gate. Filtering for her meant
        refusing him, which is the one thing this feature is not allowed to
        do; so a verifier that was TOLD who the owner is (``owner_label``,
        which app.py sets) reads that gallery as nothing enrolled, loudly.
        A verifier nobody told cannot tell whose the gallery is and keeps
        the old reading -- the explicit label is the mechanism.
        """
        return not self.enrolment_gap()

    def enrolment_gap(self) -> str:
        """"" when there is a pool of HIS to verify against, else one
        sentence saying what is missing and what to run. The reason every
        fail-open in this file logs, so the log says WHY the box is open."""
        if self._embeddings:
            return ""
        labels = self._gallery_labels()
        if not labels:
            return "no voiceprint enrolled"
        if not self.owner_label or self.owner_label in labels:
            return ""
        return ("the owner (%s) has no voice pool: no voiceprint, and the "
                "voice gallery holds only %s. Voice verification is OFF -- "
                "everyone is answered -- until he is enrolled: "
                "scripts/enroll_voice.py, then scripts/voice_enrol.py "
                "--migrate" % (self.owner_label, ", ".join(labels)))

    def _gallery_labels(self):
        return self._gallery_state()[0]

    def _gallery_state(self):
        """``(labels, fault)``: every label the gallery holds, and "" -- or
        one sentence when the gallery could not even say.

        THE FAULT IS RETURNED, NOT ONLY LOGGED, because ``gate._voice_leg``
        needs it: a gallery that raises used to reach the gate as
        ``who=""`` -- byte-identical to a legitimate non-match -- and a
        nameless match is the owner on a one-person box. A wedged instrument
        must arrive as a wedged instrument, which the gate treats as NOT
        RUNNING (loud, dead-man counted), never as a name and never as a
        rejection it did not measure.
        """
        if self.gallery is None:
            return (), ""
        try:
            return tuple(self.gallery.labels()), ""
        except Exception as exc:  # noqa: BLE001 - a broken gallery is no gallery
            log.warning("voice gallery labels unreadable: %s: %s",
                        type(exc).__name__, exc, exc_info=True)
            return (), ("the voice gallery could not list its labels: %s"
                        % type(exc).__name__)

    def _all_centroids(self, matchable=False):
        """``{label_or_"": centroid}`` over everything enrolled.

        The voiceprint's pool is keyed "" -- it has no label and inventing one
        here would put a name into the identity chain that no store agrees on.
        ``_best_match`` reports WHICH pool a score came from (the stats dict's
        ``matched_label``), and ``gate._voice_leg`` turns a nameless match
        into the owner ONLY for a match on that pool or on his own migrated
        label; that is where the owner's label is actually known.

        ``matchable=True`` LEAVES OUT EVERY PROVISIONAL LABEL, and that is the
        difference between a store that scores and a door. A label with fewer
        than ``voicegallery.MIN_TAKES_TO_NAME`` takes is never NAMED, but its
        centroid used to sit inside the maximum ``_best_score`` takes, so a
        second person with six takes cleared the 0.30 bar on her own centroid,
        set ``matched=1`` with no name, and the gate's owner fallback made her
        HIM. Measured 2026-09-04: 150 of 150 of her clips admitted as the
        owner, at a centroid cosine to his pool of 0.246. A label that cannot
        be named cannot match as anybody; it still scores and logs through
        ``identify``.
        """
        out = {}
        if self._centroid is not None:
            out[""] = self._centroid
        if self.gallery is not None:
            try:
                cents = self.gallery.centroids()
                if matchable:
                    cents = {k: c for k, c in cents.items()
                             if not self.gallery.provisional(k)}
                out.update(cents)
            except Exception:  # noqa: BLE001
                log.warning("voice gallery centroids unreadable; matching "
                            "against the voiceprint alone", exc_info=True)
        if matchable:
            # AND A LABEL WEARING HIS NAME THAT IS NOT HIS POOL MATCHES
            # NOBODY -- the same rule a provisional label already lives
            # under, for the same reason. It cannot be folded into his pool
            # (``_owner_pools``), and leaving it matchable would hand the
            # gate a ``matched_label`` spelled like the owner, which
            # ``gate._voice_leg`` compares to his label as a STRING. Dropping
            # it is what makes the fix hold at the gate as well as here.
            for label in self._disowned(out):
                out.pop(label, None)
        return out

    # Legacy alias (voice_input_gui / hotword_daemon used .enrolled)
    @property
    def enrolled(self):
        return self.is_enrolled

    @property
    def num_samples(self):
        """Number of stored voice samples."""
        return len(self._embeddings)

    # ------------------------------------------------------- fail-open
    def _fail_open(self, reason):
        """Log a fail-open acceptance loudly; publish one warn Status/session."""
        log.warning("speaker verify fail-open (%s) — accepting audio", reason)
        if not self._warned_fail_open:
            self._warned_fail_open = True
            try:
                bus.publish(Status(
                    text=f"Speaker verify inactive ({reason}) — accepting all audio",
                    kind="warn"))
            except Exception:
                log.exception("failed to publish fail-open Status")

    def _fail_shut(self, reason):
        """Reject audio we could not verify, loudly.

        Only reachable once a voiceprint exists: the user has explicitly asked
        for other voices to be filtered, so accepting everything because the
        model broke defeats the feature -- that is exactly how a television
        reached the commander. Published every time rather than once per
        session, because this blocks the voice path and a silent block is the
        experience being avoided. Typed input is unaffected.
        """
        # The recorder polls verify() at 1 Hz while a capture is open, so with
        # a broken model this fired every second. Still loud, still repeated
        # (a silent block is the experience being avoided), but paced.
        now = time.monotonic()
        if now - getattr(self, "_fail_shut_last", 0.0) < 5.0:
            log.debug("speaker verify failed shut (%s), paced", reason)
            return
        self._fail_shut_last = now
        log.error("speaker verify FAILED SHUT (%s) -- rejecting audio; "
                  "typed input still works", reason)
        try:
            bus.publish(Status(
                text=f"Voice blocked: speaker check unavailable ({reason})",
                kind="error"))
        except Exception:
            log.exception("failed to publish fail-shut Status")

    # ---------------------------------------------------------- device
    def _resolve_device(self):
        """Pick cuda:{gpu} when torch CUDA is usable, else cpu."""
        if self._device is not None:
            return self._device
        try:
            import torch
            if torch.cuda.is_available() and 0 <= self.gpu < torch.cuda.device_count():
                self._device = f"cuda:{self.gpu}"
            else:
                self._device = "cpu"
        except Exception:
            log.exception("torch device probe failed; using cpu")
            self._device = "cpu"
        return self._device

    # ----------------------------------------------------------- model
    def load_model(self):
        """Load the ECAPA-TDNN model. Call once at startup.

        Tries the resolved device (cuda:{gpu} when available, else cpu) and
        falls back to cpu when a CUDA load fails.
        """
        if self._model_loaded:
            return True
        device = self._resolve_device()
        for attempt_device in ([device, "cpu"] if device != "cpu" else ["cpu"]):
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": attempt_device},
                    savedir=str(SPEAKER_MODEL_DIR),
                )
                self._device = attempt_device
                self._model_loaded = True
                log.info("speaker model loaded on %s", attempt_device)
                return True
            except Exception:
                log.exception("speaker model load failed on %s", attempt_device)
        return False

    # ----------------------------------------------------- persistence
    def load_gallery(self):
        """Load the multi-speaker store, if there is one. Never raises.

        SEPARATE FROM ``load()``'s BODY so a gallery that will not read cannot
        cost him his voiceprint. The two stores are independent on purpose:
        one is his rollback and the other is the new feature.
        """
        try:
            gal = vgal.default_gallery()
            if gal.load():
                self.gallery = gal
                log.info("voice gallery: %d label(s) enrolled (%s)",
                         len(gal.labels()), ", ".join(gal.labels()) or "-")
                labels = gal.labels()
                if labels and self.owner_label and \
                        self.owner_label not in labels:
                    # scripts/voice_enrol.py refuses to build this layout;
                    # a gallery from an older build can still hold it.
                    # He is not locked out by it (a match on the voiceprint
                    # is still his), but identify() cannot rank him against
                    # anybody, so a guest whose voice also clears his bar
                    # is answered by the voiceprint alone.
                    log.warning("voice gallery holds %s but not the owner's "
                                "label %r: run scripts/voice_enrol.py "
                                "--migrate so the gallery can tell him from "
                                "them", ", ".join(labels), self.owner_label)
            else:
                self.gallery = None
                if gal.foreign_generations:
                    log.warning("voice gallery: %d generation(s) written by "
                                "another encoder; nothing loaded and nothing "
                                "touched", len(gal.foreign_generations))
        except Exception:
            self.gallery = None
            log.exception("voice gallery load error; multi-speaker voice ID "
                          "is off and the voiceprint is unaffected")

    def _warn_if_owner_has_no_pool(self):
        """Said at LOAD, not only on the first fail-open: a gallery that
        holds somebody while the owner is enrolled nowhere is the layout
        that used to lock him out (see is_enrolled), and the fix is a
        command he has to run."""
        gap = self.enrolment_gap()
        if gap and self._gallery_labels():
            log.warning("voice verification is OFF: %s", gap)

    def load(self):
        """Load saved voiceprint AND the multi-speaker gallery from disk."""
        self.load_gallery()
        if not VOICEPRINT_FILE.exists():
            self._loaded = True
            self._warn_if_owner_has_no_pool()
            return
        self.format_fault = ""
        try:
            data = np.load(VOICEPRINT_FILE)
            # THE FORMAT IS READ BEFORE THE VECTORS, because it decides
            # whether these vectors may be read at all -- facegallery._read
            # settles the same question in the same order and for the same
            # reason. Reading an unknown pool as if it were this one is how
            # embeddings silently stop meaning what the code thinks.
            fmt = int(data["_format"][0]) if "_format" in data.files else 1
            if fmt not in KNOWN_VOICEPRINT_FORMATS:
                self.format_fault = (
                    "voiceprint.npz is format %d and this build reads %s. "
                    "Loading NOTHING from it: its keys may name several "
                    "people and this build would pool them into one "
                    "centroid, which accepts every one of them as you. The "
                    "file is left exactly where it is."
                    % (fmt, ", ".join(str(f) for f in KNOWN_VOICEPRINT_FORMATS)))
                log.error("voice verification is OFF: %s", self.format_fault)
                self._embeddings = []
                self._recompute_centroid()
                self._loaded = True
                return
            keys = [k for k in sorted(data.files) if k.startswith("emb_")] or \
                [k for k in sorted(data.files) if not k.startswith("_")]
            self._embeddings = [data[k] for k in keys]
            self._recompute_centroid()
            self._format = fmt
            log.info("voiceprint loaded: %d samples (format %d)",
                     len(self._embeddings), fmt)
            if fmt < VOICEPRINT_FORMAT:
                # Embeddings saved before trim_silence were pooled with the
                # silence of fixed-length enrolment takes; probes are trimmed
                # now, and the asymmetry measurably lowers genuine scores.
                log.info("voiceprint format %d < %d (fresh pools measured no "
                         "better on 2026-08-31; nothing to act on) "
                            "genuine scores run low until you re-enrol: "
                            "scripts/enroll_voice.py --reset", fmt, VOICEPRINT_FORMAT)
        except Exception:
            log.exception("voiceprint load error")
        self._loaded = True
        self._warn_if_owner_has_no_pool()

    def save(self):
        """Save voiceprint to disk atomically (.tmp + os.replace)."""
        with self._lock:
            if not self._embeddings:
                return
            try:
                VOICEPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
                arrays = {f"emb_{i:04d}": emb for i, emb in enumerate(self._embeddings)}
                # The pool's own format, not the code's: saving a pre-trim
                # pool under the current number would silence the re-enrol
                # warning while the embeddings stayed incomparable.
                arrays["_format"] = np.array([self._format])
                tmp = VOICEPRINT_FILE.with_name(VOICEPRINT_FILE.name + ".tmp")
                # savez appends ".npz" to bare paths; a file handle keeps the
                # exact tmp name so os.replace targets the right file.
                with open(tmp, "wb") as fh:
                    np.savez(fh, **arrays)
                os.replace(tmp, VOICEPRINT_FILE)
            except Exception:
                log.exception("voiceprint save error")

    def _ensure_model(self) -> bool:
        """Load the speaker model on first use, at most once.

        The app warms this on its model-loader thread, but audio can arrive
        before that finishes; now that verification fails SHUT, rejecting
        merely because the load had not happened yet would make voice dead for
        the opening seconds of every session -- indistinguishable from the
        feature being broken. A load that genuinely FAILS is not retried: a
        broken model is almost always permanently broken, and retrying would
        add seconds to every rejection.
        """
        if self._model_loaded:
            return True
        if self._model_failed:
            return False
        if self.load_model():
            return True
        self._model_failed = True
        return False

    # ------------------------------------------------------ embeddings
    def _extract_embedding(self, audio_16k, min_seconds=MIN_AUDIO_SECONDS):
        """Extract a 192-dim speaker embedding from 16kHz audio.

        Args:
            audio_16k: numpy array, mono, 16kHz float32
            min_seconds: trimmed speech below which no embedding is taken.
                The default is the transcript-length floor every caller here
                wants; the wake gate passes MIN_SPEECH_SECONDS, because its
                2 s ring buffer holds 0.40-0.88 s of speech on 6 of his 10
                own wake clips (measured 2026-09-02) and a None there leaves
                the gate blind on most wakes.

        Returns:
            numpy array of shape (192,) or None on failure
        """
        if not self._ensure_model():
            return None
        # Trim FIRST, then apply the length floor to what remains: a 3 s
        # capture holding 0.58 s of speech used to pass the raw-length check
        # and get embedded anyway -- exactly the clip whose score is a coin
        # flip (measured FRR@0.30: 30% at 1.0 s of speech, 0% at 3.0 s).
        # Every score, verify, enrol and window passes through here, so the
        # trim lands on all of them at once -- including the wake-word gate,
        # which scores a 1 s buffer that is mostly pre-speech silence.
        audio_16k = trim_silence(audio_16k)
        if len(audio_16k) < int(SAMPLE_RATE * min_seconds):
            return None
        try:
            import torch
            # SpeechBrain expects (batch, time) tensor
            waveform = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0)
            waveform = waveform.to(self._device or self._resolve_device())
            with torch.no_grad():
                embedding = self._model.encode_batch(waveform)
            # Shape: (1, 1, 192) -> (192,)
            return embedding.squeeze().cpu().numpy()
        except Exception:
            log.exception("embedding extraction error")
            return None

    def _recompute_centroid(self):
        """Recompute the mean embedding (centroid) from all samples."""
        if self._embeddings:
            self._centroid = np.mean(self._embeddings, axis=0)
        else:
            self._centroid = None

    def _cosine_similarity(self, a, b):
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------ enrollment
    def enroll_from_audio(self, audio_16k):
        """Add an enrollment sample. Returns (success, num_total_samples).

        Args:
            audio_16k: numpy array, mono, 16kHz float32
        """
        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            return False, len(self._embeddings)

        with self._lock:
            if self._embeddings and self._format < VOICEPRINT_FORMAT:
                # A fresh enrolment supersedes a pre-trim pool rather than
                # mixing into it: the old embeddings were pooled with silence
                # and would drag the centroid away from the trimmed takes.
                log.warning("replacing a voiceprint enrolled before silence "
                            "trimming (%d samples); this enrolment starts a "
                            "fresh pool", len(self._embeddings))
                self._embeddings = []
            self._format = VOICEPRINT_FORMAT
            self._embeddings.append(embedding)
            self._recompute_centroid()
            # A DELIBERATE ENROLMENT IS A NEW REFERENCE. The frozen centroid
            # exists to stop passive samples walking the pool; it must not
            # freeze him out of moving it himself, on purpose, at the mic.
            self._frozen_centroid = None
            self._passive_added = 0

        self.save()
        log.info("enrolled sample #%d (embedding norm: %.3f)",
                 len(self._embeddings), np.linalg.norm(embedding))
        return True, len(self._embeddings)

    # Legacy alias (voice_input_gui called .enroll)
    enroll = enroll_from_audio

    # ---------------------------------------------------- verification
    def score(self, audio_16k, min_seconds=MIN_AUDIO_SECONDS):
        """Cosine similarity to the voiceprint, or None if it cannot be computed.

        Applies no accept/reject policy, unlike verify(), which collapses "not
        you" and "could not tell" into the same False. Callers needing their
        own fallback use this -- the wake-word gate fails OPEN where the
        transcript gate fails SHUT, so it cannot reuse verify()'s verdict.

        min_seconds lets such a caller trade a weaker number for having one at
        all. It is the caller's job to then treat a weak number as weak: the
        wake gate scores at MIN_SPEECH_SECONDS and, where it can MEASURE that
        the buffer held less than MIN_AUDIO_SECONDS of speech, refuses to
        reject on the result. Measured 2026-09-02, that relief is confined to
        the log on the wake path -- both of its branches wake him below the
        floor -- so this argument is instrumentation, not a policy hook.
        """
        if not self.is_enrolled:
            return None
        embedding = self._extract_embedding(audio_16k, min_seconds)
        if embedding is None:
            return None
        with self._lock:
            return self._best_score(embedding)

    def _best_score(self, embedding):
        """The highest cosine against ANY enrolled centroid.

        A MAXIMUM, AND THAT IS WHAT KEEPS THE WAKE GATE FAILING OPEN. The wake
        gate (hotword._speaker_ok) asks this for a BOOLEAN and never for a
        name. Adding a label can only ever raise a maximum, so enrolling
        somebody can only make that gate MORE permissive -- never less. An
        unwakeable assistant is the worse failure and this feature is not
        allowed to create one.

        THE ONE TRANSITION A MAXIMUM DOES NOT COVER is nothing -> somebody
        else: with no pool of his at all there is no maximum to raise, and
        the first guest's centroid turned the wake gate's ``None`` (fail
        open) into a number under the bar (suppress) -- measured 2026-09-04,
        woken 0 of 50. ``is_enrolled`` closes it: a gallery holding only
        other people is not an enrolment for a verifier that knows who the
        owner is, so ``score`` keeps returning None there.

        OVER THE MATCHABLE CENTROIDS ONLY -- a provisional label is left out
        (see ``_all_centroids``), so enrolling somebody with too few takes to
        be named raises nothing here and lowers nothing here. Nothing enrolled
        but provisional labels returns None, which the wake gate reads as its
        fail-open.

        Callers hold ``self._lock``.
        """
        best = self._best_match(embedding)
        return None if best is None else best[1]

    def _owner_pools(self, cents):
        """The gallery labels that ARE the voiceprint's pool -- his, whatever
        they are called. ``cents`` is the dict ``_all_centroids`` built, so
        this costs one cosine per label and no lock.

        ONE QUESTION, ASKED OF EVERY LABEL INCLUDING HIS OWN: does this
        centroid measure as the voiceprint's pool? The previous version asked
        a second question first -- "is this label spelled like the owner?" --
        and answered YES with no measurement at all, which made his name a
        credential anybody at the microphone could type. Measured 2026-09-05,
        that admitted a guest with owner scope 100/100 by name and 100/100
        again through the abstention fail-open (tests/
        test_voice_owner_label_guard.py). There is no label test here now.

        THE BOOTSTRAP IS THE ONE EXCEPTION AND IT IS NOT A LOOPHOLE. With no
        voiceprint there is no pool to measure against, so the gallery is the
        only anchor identity has and his label is it -- the layout a box has
        after ``enroll_voice.py --reset``, and refusing it would silence him.
        Nothing can be stolen there, because there is nothing yet to steal.
        """
        out = set()
        mine = cents.get("")
        for label, c in cents.items():
            if not label:
                continue
            if mine is None:
                if label and label == self.owner_label:
                    out.add(label)
            elif self._cosine_similarity(c, mine) >= MIGRATED_ALIAS_COSINE:
                out.add(label)
        return out

    def _disowned(self, cents):
        """The gallery labels that CLAIM the owner's name without measuring
        as his pool. Everything below must refuse to read one as him.

        A frozenset, so the empty case -- which is every healthy box -- costs
        nothing. ``cents`` must be the UNFILTERED dict: a disowned label is
        dropped from the matchable set by ``_all_centroids`` and cannot be
        used to look itself up afterwards.
        """
        mine = cents.get("")
        if mine is None or not self.owner_label:
            return frozenset()          # bootstrap: nothing to measure against
        c = cents.get(self.owner_label)
        if c is None:
            return frozenset()
        score = float(self._cosine_similarity(c, mine))
        if score >= MIGRATED_ALIAS_COSINE:
            return frozenset()
        if not self._warned_disowned:
            self._warned_disowned = True
            log.warning(
                "voice gallery: the label %r does not match voiceprint.npz "
                "(cos %.3f, needs %.2f) -- it is NOT being read as the owner. "
                "Takes recorded under his name by somebody else look exactly "
                "like this. Check with: scripts/voice_enrol.py --status",
                self.owner_label, score, MIGRATED_ALIAS_COSINE)
        return frozenset({self.owner_label})

    def _pool_of_label(self, label):
        """Which pool a gallery label belongs to: "" when it is the owner's
        (see ``_owner_pools``), otherwise itself."""
        label = str(label or "")
        if not label:
            return ""
        with self._lock:
            cents = self._all_centroids()
        return "" if label in self._owner_pools(cents) else label

    def _best_match(self, embedding):
        """``(pool, score)`` of the best MATCHABLE centroid, or None when
        nothing matchable is enrolled.

        THE POOL IS WHICH PERSON'S CENTROID THE SCORE CAME FROM, and it is
        the fact the gate was missing. ``_best_score`` used to return the
        number alone, so a second person clearing the bar on HER OWN
        centroid arrived at the gate as "a nameless match" -- byte-identical
        to a match on his voiceprint -- and the gate's owner fallback made
        her him (measured 2026-09-04: 50 of 50 short commands of hers
        admitted with owner scope). "" is the voiceprint's pool, and the
        owner's migrated label is folded into it (``_owner_pools``); any
        other label is that person's pool and can never become the owner.

        Callers hold ``self._lock``.
        """
        cents = self._all_centroids(matchable=True)
        if not cents:
            return None
        his = self._owner_pools(cents)
        best = None
        for label, c in cents.items():
            score = float(self._cosine_similarity(embedding, c))
            if best is None or score > best[1]:
                best = ("" if label in his else label, score)
        return best

    def _who(self, embedding, speech_s):
        """``(verdict, fault)`` from the gallery.

        No gallery is ``(None, "")`` -- nobody to name, nothing wrong. A
        gallery that RAISES is ``(None, "<sentence>")``, and the second
        value is the whole point: the two used to be the same, so a wedged
        store reached the gate looking exactly like a voice it had measured
        and declined to name.

        THE WHOLE VERDICT COMES BACK, not the name alone. This used to
        return ``(who, scores, "")`` and drop ``verdict.abstained`` on the
        floor, so an abstention on a short window reached the gate as a
        MEASURED nameless match -- which is the route a guest's 1.2 s
        command took to owner scope, and the route his own 1.2 s command
        took to a refusal (both measured 50/50, 2026-09-04).

        Every bar lives in ``voicegallery.identify`` -- the accept bar, the
        margin, the provisional rule and the abstain window. Nothing here
        second-guesses it, and ``gate.py`` holds no threshold at all.
        """
        if self.gallery is None:
            return None, ""
        try:
            verdict = self.gallery.identify(embedding, speech_s, self.threshold)
        except Exception as exc:  # noqa: BLE001 - a broken gallery names nobody
            log.exception("voice gallery identify failed; naming nobody")
            return None, ("the voice gallery could not identify: %s"
                          % type(exc).__name__)
        verdict = self._strip_disowned(verdict)
        if verdict.who:
            log.info("voice gallery: %s (%.3f%s)", verdict.who, verdict.score,
                     "" if verdict.margin is None
                     else ", margin %.3f" % verdict.margin)
        elif verdict.provisional:
            log.info("voice gallery: probably %s (%.3f) but only %d take(s); "
                     "naming nobody", verdict.provisional, verdict.score,
                     self.gallery.count(verdict.provisional))
        elif verdict.why and not verdict.abstained:
            log.info("voice gallery: naming nobody -- %s", verdict.why)
        return verdict, ""

    def _strip_disowned(self, verdict):
        """His name, taken back off a pool that is not his.

        ``identify`` ranks GALLERY LABELS and knows nothing of
        ``voiceprint.npz``, so it will happily name a disowned label -- his
        name, on somebody else's voice. The escalation itself is already shut
        by ``_all_centroids`` (a disowned label cannot be ``matched_label``,
        which is the only thing ``gate._voice_leg`` turns into scope), so this
        is not the lock; it is the LIE. A rejected clip still carries the
        gallery's opinion into ``stats`` and into the log, and "probably
        hunter" said about a stranger is exactly the sentence that would get
        the guard hand-waved away as a false alarm next time.

        ``score``, ``second`` and ``margin`` are left alone deliberately.
        They feed only the accept bar and the near-miss test, and both of
        those can only ever WITHHOLD a name downstream -- never grant one --
        so a stale number there is conservative in the safe direction.

        The lock is taken here and is never held by a caller: all three
        ``_who`` call sites release it before asking.
        """
        if verdict is None:
            return verdict
        with self._lock:
            gone = self._disowned(self._all_centroids())
        if not gone:
            return verdict
        scores = tuple((lab, sc) for lab, sc in (verdict.scores or ())
                       if lab not in gone)
        who = "" if verdict.who in gone else verdict.who
        prov = "" if verdict.provisional in gone else verdict.provisional
        if (who == verdict.who and prov == verdict.provisional
                and len(scores) == len(verdict.scores or ())):
            return verdict
        log.warning("voice gallery: withholding %r -- that label does not "
                    "match voiceprint.npz and is not the owner",
                    verdict.who or verdict.provisional or self.owner_label)
        return dataclasses.replace(verdict, who=who, provisional=prov,
                                   scores=scores)

    def _reconcile(self, verdict, pool):
        """The gallery's NAME must agree with the POOL the bar was cleared
        on, or the name is withheld and ``top`` carries it as a guess.

        Reachable with the owner un-migrated: ``identify`` ranks gallery
        labels only, so with a guest in the gallery and him only in
        ``voiceprint.npz`` it cannot rank him against her and names the one
        label it has -- on HIS voice, whenever it also clears her bar
        (measured 2026-09-04 at apart 1.0: 43 of 150 of his clips named
        "mara" outright, 150 of 150 at apart 2.0). The pool is the measured
        fact: his voiceprint out-scored her centroid, so the name is
        withheld and the gate answers nobody rather than her. ONE helper for
        the whole-clip path and the windowed path, so a clip under 3 s
        cannot be named what a clip over 3 s would not be.
        """
        if verdict is None or not verdict.who:
            return verdict
        if self._pool_of_label(verdict.who) == pool:
            return verdict
        log.info("voice gallery named %s but the bar was cleared on %s's "
                 "pool; naming nobody", verdict.who, repr(pool or "voiceprint"))
        return dataclasses.replace(verdict, who="")

    def _ident(self, verdict=None, *, pool="", abstained=False, fault="",
               labels=None):
        """The identity half of a stats dict. ONE builder, so every path out
        of ``verify`` and ``filter_segments`` carries the same keys:

        who           the label the gallery NAMED (bar, margin and takes all
                      cleared), or ""
        who_scores    {label: cosine} the gallery measured
        labels        every label the gallery holds, provisional included
                      (for the log; the gate no longer counts them)
        matched_label WHOSE POOL the kept windows cleared the bar on: "" for
                      the voiceprint (his), a label for that person. The
                      gate's owner fallback is legal ONLY for "" or his own
                      label -- never for anybody else's pool.
        top           the label identify() ranked first ABOVE THE BAR, named
                      or not (provisional, or a failed margin). "" when
                      nothing cleared it. A nameless match whose best guess
                      is somebody else may not become the owner.
        provisional   ``top`` when it was withheld only for lack of takes
        near_miss     True when the margin failed: "I can't tell which of
                      you", which is the line the gate says for it
        abstained     True when identify() measured nothing on the speech
                      it was given (too little): the documented fail-open,
                      carried through EVERY path, and NOT a recognition
        who_fault     "" or one sentence when the gallery raised
        """
        if labels is None:
            labels, lab_fault = self._gallery_state()
            fault = fault or lab_fault
        who = who_top = provisional = ""
        scores = {}
        near_miss = False
        if verdict is not None:
            abstained = abstained or bool(verdict.abstained)
            scores = dict(verdict.scores)
            who = str(verdict.who or "")
            provisional = str(verdict.provisional or "")
            if verdict.scores and float(verdict.score) >= self.threshold:
                who_top = str(verdict.scores[0][0] or "")
            near_miss = bool(not who and not provisional and not abstained
                             and verdict.margin is not None
                             and who_top
                             and verdict.margin < vgal.MARGIN)
        return {"who": who, "who_scores": scores, "labels": tuple(labels),
                "matched_label": str(pool or ""), "top": who_top,
                "provisional": provisional, "near_miss": near_miss,
                "abstained": bool(abstained), "who_fault": str(fault or "")}

    def verify(self, audio_16k):
        """Check if audio matches the enrolled voiceprint.

        Args:
            audio_16k: numpy array, mono, 16kHz float32

        Returns:
            (is_match: bool, score: float)
            score is cosine similarity (0-1), higher = more similar

        Fail-open: with no voiceprint, no model, or unextractable audio the
        clip is ACCEPTED (True, 1.0) — logged at WARNING, with one
        Status(kind=warn) event per session.
        """
        is_match, score, _ident = self._verify_named(audio_16k)
        return is_match, score

    def _verify_named(self, audio_16k):
        """``verify()`` plus the identity. ``(is_match, score, ident)`` where
        ``ident`` is the dict ``_ident`` builds.

        Split out rather than folded in because ``filter_segments`` falls back
        to whole-clip verification on a short capture and needs the name too;
        returning it through a shared attribute would race the recorder's
        1 Hz polling."""
        gap = self.enrolment_gap()
        if gap:
            # Nobody enrolled -- or nobody who is HIM -- accept all audio
            self._fail_open(gap)
            return True, 1.0, self._ident()

        speech_s = len(trim_silence(audio_16k)) / SAMPLE_RATE
        if speech_s < ABSTAIN_SECONDS:
            # Duration -- not the threshold, not the model -- is what drives
            # false rejects: measured 2026-08-31 on his own clips, FRR@0.30
            # is 30% at 1.0 s of trimmed speech and 0% at 3.0 s, and his
            # genuine scores bottom at 0.330 against a 0.30 threshold. A
            # score from this little speech is a coin flip, so ABSTAIN --
            # fail open the way an unenrolled box does, and let the words
            # themselves be judged -- rather than reject him silently.
            log.info("speaker verify: %.2fs of speech is too little to judge; "
                     "abstaining (fail-open)", speech_s)
            # NO NAME FROM AN ABSTENTION, and that is the trap a naive label
            # change springs. The clip is accepted (fail-open) with who="",
            # and SAID TO BE an abstention: gate._voice_leg keeps the owner
            # fallback for one -- nothing was measured, so enrolling a second
            # person moves nothing about it -- where a MEASURED nameless
            # match on a two-person box is unknown. An abstention must never
            # be narrated as a recognition.
            return True, 0.0, self._ident(abstained=True)
        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            # Can't extract embedding (model missing, audio too short) — accept
            reason = ("model not loaded" if not self._model_loaded
                      else "no embedding (audio too short?)")
            self._fail_shut(reason)
            return False, 0.0, self._ident()

        with self._lock:
            best = self._best_match(embedding)
        # None: nothing matchable is enrolled (only provisional labels). That
        # is a REJECT on the transcript gate, not an accept -- somebody IS
        # enrolled, so the gate fails shut exactly as it did before labels.
        pool, score = ("", 0.0) if best is None else best
        verdict, fault = self._who(embedding, speech_s)
        verdict = self._reconcile(verdict, pool)

        # Duration beside the score, always: it is the variable that actually
        # drives rejection, and it was invisible in the log until now.
        log.info("speaker verify: score=%.3f on %.2fs of speech (threshold %.2f)",
                 score, speech_s, self.threshold)
        is_match = score >= self.threshold
        who = "" if verdict is None else verdict.who
        log.info("speaker verify: score=%.3f threshold=%s %s%s",
                 score, self.threshold, "MATCH" if is_match else "REJECT",
                 " (%s)" % (who or pool or "voiceprint") if is_match else "")
        return is_match, score, self._ident(verdict, pool=pool, fault=fault)

    # ------------------------------------------------ passive learning
    def add_sample(self, audio_16k):
        """Passively improve voiceprint with a confirmed recording.

        Only call this after the user has accepted the transcription
        (didn't reject or cancel it).

        Args:
            audio_16k: numpy array, mono, 16kHz float32

        Returns:
            True if sample was added

        GATED AGAINST A FROZEN CENTROID, NOT THE LIVE ONE, and the difference
        is a measured one. This method used to score a candidate against the
        centroid it was about to move, so the bar bounded one STEP and not the
        WALK: every accepted sample buys the next one more room. Measured
        2026-09-04 on synthetic vectors at his pool's spread
        (tests/test_voice_passive.py), worst-case attacker, 14-take enrolment,
        an intruder starting at 0.149 against a 0.30 bar --

            gate on the live centroid, 20 accepts   cos 0.657, intruder 0.843
            gate on the frozen centroid, 20         cos 0.724, intruder 0.790

        -- so freezing is better and IS NOT ENOUGH. Twenty new vectors against
        fourteen originals is a 59% swing in a mean whatever each one scores;
        only the COUNT bounds it, hence PASSIVE_CAP.

        AND SAY THE LIMIT OUT LOUD. ``voiceprint.npz`` has no per-sample
        provenance, so it cannot tell an enrolment take from a passively
        learned one. This "frozen" centroid is frozen for the life of the
        PROCESS; after a restart it is recomputed over a pool that already
        contains the passive samples, and the walk resumes from wherever it
        got to. Closing that needs a per-sample key, which is a format bump
        this file may not take -- ``jarvis/voicegallery.py`` has the key
        (``src_``), and passive learning into that store is OFF for exactly
        the reasons above.
        """
        if self._format < VOICEPRINT_FORMAT:
            log.info("passive sample skipped: voiceprint predates silence "
                     "trimming; re-enrol with scripts/enroll_voice.py --reset")
            return False
        if self._passive_added >= PASSIVE_CAP:
            log.info("passive sample skipped: %d already added this session, "
                     "the cap (see add_sample for the measured drift)",
                     self._passive_added)
            return False
        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            return False

        # Only add if it matches the FROZEN reference (sanity check)
        if self.is_enrolled:
            with self._lock:
                if self._frozen_centroid is None and self._centroid is not None:
                    self._frozen_centroid = np.array(self._centroid, copy=True)
                ref = self._frozen_centroid
                score = (self._cosine_similarity(embedding, ref)
                         if ref is not None else None)
            if score is None:
                log.info("passive sample skipped: no frozen reference to "
                         "measure it against")
                return False
            if score < self.threshold:
                log.info("passive sample rejected (score=%.3f < %s against "
                         "the frozen enrolment centroid)",
                         score, self.threshold)
                return False

        with self._lock:
            self._passive_added += 1
            self._embeddings.append(embedding)
            # Trim oldest if over limit (keep first 10 enrollment + newest)
            if len(self._embeddings) > MAX_EMBEDDINGS:
                # Keep first 10 (original enrollment) + newest
                keep_first = min(10, len(self._embeddings) // 2)
                keep_recent = MAX_EMBEDDINGS - keep_first
                self._embeddings = (
                    self._embeddings[:keep_first]
                    + self._embeddings[-keep_recent:]
                )
            self._recompute_centroid()

        self.save()
        return True

    # ------------------------------------------------ segment filtering
    def _dump_reject(self, audio_16k, windows, scores):
        """Save a rejected capture so a rejection can be diagnosed from the
        audio instead of from the score alone. Off unless JARVIS_DEBUG_AUDIO=1:
        these are recordings of the user's room and do not accumulate silently.
        """
        try:
            import soundfile as sf
            out = PATHS.LOG_DIR / "reject_last.wav"
            sf.write(out, audio_16k, SAMPLE_RATE)
            parts = []
            for w, sc in zip(windows, scores):
                kept = len(trim_silence(w)) / SAMPLE_RATE
                parts.append("%.2fs->%.2fs=%.3f" % (len(w) / SAMPLE_RATE, kept, sc))
            log.info("reject dump: %s  windows: %s", out, "  ".join(parts))
        except Exception:
            log.exception("reject dump failed")

    def filter_segments(self, audio_16k, window_sec=3.0, hop_sec=1.5):
        """Filter audio to keep only segments matching the enrolled voice.

        Splits audio into overlapping windows, verifies each against the
        voiceprint, and returns only the windows that match. This allows
        the user to speak while a TV/YouTube is playing — their voice
        segments pass, background voices are dropped.

        Args:
            audio_16k: numpy array, mono, 16kHz float32
            window_sec: window size in seconds (default 3.0)
            hop_sec: hop between windows in seconds (default 1.5)

        Returns:
            (filtered_audio, stats_dict)
            filtered_audio: numpy array of concatenated matching segments,
                            or None if no segments matched
            stats_dict: {'total': N, 'matched': M, 'scores': [...]} plus
                        the identity keys ``_ident`` documents (who,
                        matched_label, top, abstained, ...). ``matched``
                        counts the windows KEPT: those that cleared the bar
                        on the clip's own speaker's pool.
        """
        gap = self.enrolment_gap()
        if gap:
            # Unconfigured: pass through, or voice never works on a fresh box.
            self._fail_open(gap)
            return audio_16k, {"total": 0, "matched": 0, "scores": [],
                               **self._ident()}
        if not self._ensure_model():
            self._fail_shut("model not loaded")
            return None, {"total": 0, "matched": 0, "scores": [],
                          **self._ident()}

        window_samples = int(window_sec * SAMPLE_RATE)
        hop_samples = int(hop_sec * SAMPLE_RATE)
        total_samples = len(audio_16k)

        if total_samples < window_samples:
            # Audio shorter than one window — fall back to whole-clip verify
            is_match, score, ident = self._verify_named(audio_16k)
            if is_match:
                return audio_16k, {"total": 1, "matched": 1, "scores": [score],
                                   **ident}
            # No name on a rejection; the fault and the labels still travel.
            return None, {"total": 1, "matched": 0, "scores": [score],
                          **dict(ident, who="")}

        windows = []
        positions = []
        pos = 0
        while pos + window_samples <= total_samples:
            windows.append(audio_16k[pos:pos + window_samples])
            positions.append(pos)
            pos += hop_samples
        # Include tail if it's at least 1.5 seconds
        if pos < total_samples and (total_samples - pos) >= int(1.5 * SAMPLE_RATE):
            windows.append(audio_16k[pos:total_samples])
            positions.append(pos)

        # A window holding no speech can only ever score the silence embedding
        # (about -0.06 against a voiceprint) -- it cannot match, and scoring it
        # spends a GPU pass to put a misleading number in the log. Drop those,
        # unless that would leave nothing, in which case score them all and let
        # the threshold decide as before.
        #
        # Judged against the threshold of the WHOLE capture, not the window's
        # own: a window of pure room tone is flat, and a flat clip has no
        # floor to measure against, so per-window it reads as "cannot tell".
        # The capture as a whole holds both the speech and the room, which is
        # exactly the contrast the threshold needs. (The first version asked
        # len(trim_silence(w)) -- which never shrinks on no-speech, by design,
        # so it skipped nothing.)
        n = int(SAMPLE_RATE * FRAME_MS / 1000)
        rms_all = _frame_rms(audio_16k, n)
        thr = _speech_threshold(rms_all)
        min_frames = max(1, int(SAMPLE_RATE * MIN_SPEECH_SECONDS) // n)

        def _has_speech(pos, length):
            if thr is None:
                return True                  # flat capture: cannot tell, score it
            seg = rms_all[pos // n:(pos + length) // n]
            return int((seg >= thr).sum()) >= min_frames

        voiced = [i for i, (w, pos) in enumerate(zip(windows, positions))
                  if _has_speech(pos, len(w))]
        if voiced and len(voiced) < len(windows):
            windows = [windows[i] for i in voiced]
            positions = [positions[i] for i in voiced]

        # IDENTITY IS DECIDED PER WINDOW, AND THE CLIP IS ONE PERSON'S.
        #
        # The first version scored every window against the MAXIMUM over all
        # centroids, kept every window that cleared it, and ran ONE identify()
        # on the best window. So a 6 s capture holding him for 3 s and her
        # for 3 s kept BOTH halves (each cleared the bar on its own
        # centroid) and ran the whole thing under whichever of them scored
        # higher: measured 2026-09-04, her half passed to the commander under
        # HIS name 61 of 100 times. Enrolling a guest had turned the filter
        # into a laundry.
        #
        # Now every window records WHOSE pool it cleared the bar on
        # (``_best_match``), the clip is attributed to the pool of its
        # strongest window, and a matched window on anybody else's pool is
        # DROPPED -- its audio does not reach Whisper and it does not count
        # toward ``matched``. The name comes from the kept windows' own
        # verdicts, a named one first, and an abstention on every kept window
        # is carried as an abstention rather than laundered into a match.
        scores = []
        pools = []
        embs = []
        speech = []
        try:
            for chunk in windows:
                emb = self._extract_embedding(chunk)
                if emb is not None:
                    with self._lock:
                        best = self._best_match(emb)
                    pool, score = ("", 0.0) if best is None else best
                    scores.append(score)
                    pools.append(pool)
                    embs.append(emb)
                    speech.append(len(trim_silence(chunk)) / SAMPLE_RATE)
                else:
                    scores.append(0.0)
                    pools.append("")
                    embs.append(None)
                    speech.append(0.0)
            cleared = [i for i, s in enumerate(scores) if s >= self.threshold]
            if os.environ.get("JARVIS_DEBUG_AUDIO") == "1" and not cleared:
                self._dump_reject(audio_16k, windows, scores)
        except Exception:
            log.exception("segment verification error")
            self._fail_shut("segment verification error")
            return None, {"total": len(windows), "matched": 0, "scores": [],
                          **self._ident()}

        total_count = len(windows)
        if not cleared:
            # No name on a rejection, whatever the gallery thought: "matched"
            # is the pipeline's verdict and a label may never contradict it.
            # The gallery still SCORES the strongest window so the log can
            # say "probably mara, 6 takes" about a clip that went nowhere.
            log.info("segment filter: 0/%d windows matched (scores: %s)",
                     total_count, ", ".join(f"{s:.2f}" for s in scores))
            verdict, fault = None, ""
            scored = [i for i, e in enumerate(embs) if e is not None]
            if scored:
                i = max(scored, key=lambda i: scores[i])
                verdict, fault = self._who(embs[i], speech[i])
                if verdict is not None:
                    verdict = dataclasses.replace(verdict, who="")
            return None, {"total": total_count, "matched": 0, "scores": scores,
                          **self._ident(verdict, fault=fault)}

        strongest = max(cleared, key=lambda i: scores[i])
        clip_pool = pools[strongest]
        kept = [i for i in cleared if pools[i] == clip_pool]
        dropped = [i for i in cleared if pools[i] != clip_pool]
        if dropped:
            log.info("segment filter: %d window(s) cleared the bar on %s, "
                     "not on %s; dropped as somebody else's",
                     len(dropped),
                     ", ".join(repr(pools[i] or "voiceprint") for i in dropped),
                     repr(clip_pool or "voiceprint"))

        # The verdict for the clip, from ITS OWN windows: a named verdict
        # beats a nameless one, a measured one beats an abstention, and the
        # strongest evidence breaks ties. Every kept window is asked, so a
        # short window that abstains cannot hide a longer one that names him.
        verdict, fault = None, ""
        ranked = []
        for i in kept:
            v, f = self._who(embs[i], speech[i])
            if f:
                fault = fault or f
                continue
            if v is not None:
                ranked.append((bool(v.who), not v.abstained, scores[i], i, v))
        if ranked:
            ranked.sort(key=lambda r: r[:3], reverse=True)
            verdict = self._reconcile(ranked[0][4], clip_pool)
        ident = self._ident(verdict, pool=clip_pool, fault=fault)

        matched_count = len(kept)
        matched_mask = [i in kept for i in range(total_count)]
        log.info("segment filter: %d/%d windows matched on %s (scores: %s)",
                 matched_count, total_count, repr(clip_pool or "voiceprint"),
                 ", ".join(f"{s:.2f}" for s in scores))

        # Reconstruct audio from matched segments using a mask over the
        # original audio to preserve continuity where possible
        # Mark each sample as "keep" if it belongs to any matched window
        keep = np.zeros(total_samples, dtype=bool)
        for i, pos in enumerate(positions):
            if matched_mask[i]:
                end = min(pos + len(windows[i]), total_samples)
                keep[pos:end] = True

        filtered = audio_16k[keep]

        return filtered, {"total": total_count, "matched": matched_count,
                          "scores": scores, **ident}

    # ------------------------------------------------------------ reset
    def clear(self):
        """Delete all voiceprint data."""
        with self._lock:
            self._embeddings.clear()
            self._centroid = None
            self._format = VOICEPRINT_FORMAT
            self._frozen_centroid = None
            self._passive_added = 0
        try:
            VOICEPRINT_FILE.unlink(missing_ok=True)
            log.info("voiceprint cleared")
        except Exception:
            log.exception("voiceprint clear error")
