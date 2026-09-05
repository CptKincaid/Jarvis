"""Instrument for the live preview ("ghost card") — the one display path
in Jarvis that shows the user text and writes nothing down.

WHY THIS EXISTS
---------------
2026-09-03: words repeated on the console transcript for a short while and
then stopped on their own. Three independent diagnoses ran and NONE reached
high confidence, for one shared reason: the text was never recorded. A
case-insensitive grep of the two tokens the user himself disclosed found 0
and 1 hits across jarvis.log and jarvis.log.1, while he had watched them
appear repeatedly. Rotation, log level, truncation and a second log file
were each checked and ruled out. What was left is the path:

    app._partial_loop -> Transcriber.partial() -> bus.publish(PartialText)
        -> ui/main_window._ev_partial -> views.show_partial

``Transcriber.partial()`` returns text with NO log call on the success path,
and ``_partial_loop`` publishes it with none either. It is also the only
decode in the system that skips the speaker gate, the confidence gate, the
compression-ratio loop gate and collapse_repeats. (Since 2026-09-04 it
does apply the prompt-echo half of the loop gate -- transcriber.prompt_echo
-- and returns "" for an echo, so a name from the prompt is never shown;
that blank is counted here as a decode with no emission, the same as
"whisper returned the same text".) So the preview can put text on screen
that nothing else in the system has ever seen, scored, or written down.

This module does not fix that. The cause is UNPROVEN, and a guard aimed at
an unproven cause is how this project got burned before: a silent Whisper
confidence gate that ate the user's commands. So this is an INSTRUMENT.
It suppresses nothing, drops nothing, and changes no decode. It counts,
and it records the SHAPE of a repeat so the next occurrence diagnoses
itself instead of needing another archaeology pass.

WHAT "SHAPE" MEANS, AND WHY IT IS NOT THE TEXT
----------------------------------------------
A repeat is characterised by (how many times, over what span, from which
path, how long the audio was, how many words) plus an identity token that
says "this is the same string as that one". It does NOT need the string.

The identity token is a SALTED hash, and the salt matters more than the
hash: an unsalted sha256 of a single short word is reversible in seconds
against a wordlist, so an unsalted digest of preview text is the text. The
salt is random per process (see ``_new_salt``), which is exactly the
lifetime the instrument needs — correlating emissions WITHIN one run is
the whole job — and it makes the recorded token meaningless to anyone
holding the log afterwards, including a future agent reading it.

That is deliberate beyond privacy hygiene. This preview is the one decode
that never passed the speaker gate, so its text is not established to be
the user's speech at all; it may be a decode of room noise, of a passer-by,
or of the assistant's own reply. Writing shape rather than content keeps
the instrument readable by someone who has no grant to read transcripts.

The full per-emission stream is DEBUG (the app runs the "jarvis" logger at
DEBUG, so it lands in the file). The repeat signature — the bug itself —
is INFO, so it cannot be missed and it is greppable as "preview repeat:".

Pure and thread-agnostic, like jarvis/turnclock.py: no bus import, clock
and emit injectable, and every public method swallows its own exceptions.
The preview loop runs on a best-effort thread whose whole contract is that
it must never delay, disturb or fail the real transcription that follows;
an instrument bolted onto it inherits that contract.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Deque, Optional, Tuple

from jarvis.logs import get_logger

log = get_logger("preview")

# Which code path put the text on screen. Both publish the same PartialText
# event, and telling them apart in hindsight was impossible -- the greedy
# preview is ungated and unlogged, the speculative pass runs through
# _decode_clip (speaker filter + full transcribe) and logs at DEBUG. A
# repeat means very different things depending on which one produced it.
PATH_GREEDY = "greedy"              # app._partial_loop -> Transcriber.partial()
PATH_SPECULATIVE = "speculative"    # app._maybe_speculate -> _decode_clip

WINDOW_S = 60.0          # how far back a repeat is allowed to reach
REPEAT_THRESHOLD = 3     # emissions of one identical string before it is a "repeat"
FP_CHARS = 8             # hex digits of the salted digest kept in the record
MAX_TRACKED = 256        # distinct strings held in the window; bounds a runaway loop

# Fingerprint normalisation. Case and punctuation must not split a repeat --
# whisper alternates "Massage." / "massage" / " Massage" across passes, and
# three spellings of one hallucination is still one hallucination.
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACES = re.compile(r"\s+")


def _new_salt() -> bytes:
    """A fresh random salt per process.

    Deliberately NOT stable across runs. A stable salt would let two runs be
    correlated, which this instrument never needs, at the cost of making the
    digest a persistent pseudonym for a word the user said.
    """
    return secrets.token_bytes(16)


def normalise(text: str) -> str:
    """Casefolded, punctuation-stripped, whitespace-collapsed form."""
    return _SPACES.sub(" ", _NON_WORD.sub(" ", (text or "").casefold())).strip()


@dataclass(frozen=True)
class PreviewRepeat:
    """One reported repeat. Shape only -- there is no text field, by design."""
    fp: str             # salted digest, meaningful only within this process
    count: int          # emissions of this exact string inside the window
    span_s: float       # first to last of those emissions
    path: str           # PATH_GREEDY or PATH_SPECULATIVE (the last emitter)
    words: int
    chars: int
    audio_s: Optional[float]    # buffer length of the most recent emission

    def line(self) -> str:
        audio = "—" if self.audio_s is None else f"{self.audio_s:.2f}s"
        return (f"preview repeat: {self.count}x in {self.span_s:.1f}s · "
                f"path={self.path} · {self.words}w · {self.chars}c · "
                f"audio={audio} · fp={self.fp}")


class PreviewProbe:
    """Counts preview decodes and emissions; reports repeated strings.

    ``emit`` receives each finished PreviewRepeat and is where logging and
    JSONL writing happen; the default logs the INFO line and appends to
    ``jsonl_path`` when one is given. ``clock`` and ``salt`` are injectable
    so tests are deterministic without touching the real clock or entropy.
    """

    def __init__(self, window_s: float = WINDOW_S,
                 repeat_threshold: int = REPEAT_THRESHOLD,
                 clock: Callable[[], float] = time.monotonic,
                 emit: Optional[Callable[[PreviewRepeat], None]] = None,
                 jsonl_path=None,
                 salt: Optional[bytes] = None):
        self._window_s = float(window_s)
        self._threshold = max(2, int(repeat_threshold))
        self._clock = clock
        self._emit = emit or self._default_emit
        self._jsonl_path = jsonl_path
        self._salt = _new_salt() if salt is None else salt
        self._lock = threading.Lock()
        # (when, fp) newest last; pruned to the window on every touch.
        self._seen: Deque[Tuple[float, str]] = deque()
        # fp -> emissions already reported, so a long run escalates instead
        # of writing one line per emission.
        self._reported: dict[str, int] = {}
        self._counts = {
            "decodes": 0,           # partial() calls that ran, published or not
            "emissions": 0,         # PartialText publishes actually shown
            "repeats": 0,           # PreviewRepeat records emitted
            "retractions": 0,       # ghost cards taken down
        }
        self._by_path: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------ recording

    def fingerprint(self, text: str) -> str:
        """Salted digest of the normalised text. Not reversible to the text."""
        h = hashlib.sha256()
        h.update(self._salt)
        h.update(normalise(text).encode("utf-8", "replace"))
        return h.hexdigest()[:FP_CHARS]

    def decoded(self, path: str = PATH_GREEDY) -> None:
        """A preview decode ran. Counted whether or not it published.

        This is the number every diagnosis wanted and none could get: the
        preview fires roughly once a second during a capture, and only the
        ones that CHANGED the card were ever visible downstream.
        """
        try:
            with self._lock:
                self._bump(path, "decodes")
        except Exception:               # noqa: BLE001 - never disturb the preview
            pass

    def observe(self, text: str, *, path: str = PATH_GREEDY,
                audio_s: Optional[float] = None) -> Optional[PreviewRepeat]:
        """Record that ``text`` was published to the ghost card.

        Returns a PreviewRepeat when this emission crosses a reporting
        threshold, else None. The return value is informational: NOTHING
        here gates, delays or alters the publish, and the caller must not
        treat a repeat as a reason to withhold the text. Whether a repeat
        deserves suppression is unproven, and suppressing on a guess is the
        documented failure this instrument exists to avoid repeating.
        """
        try:
            text = text or ""
            if not text.strip():
                return None
            now = self._clock()
            with self._lock:
                self._bump(path, "emissions")
                fp = self.fingerprint(text)
                self._prune(now)
                self._seen.append((now, fp))
                hits = [t for t, f in self._seen if f == fp]
                count = len(hits)
                if count < self._threshold:
                    return None
                # Escalate by doubling. A 40-emission run writes ~4 lines,
                # not 38, and each line still says the run got worse.
                already = self._reported.get(fp, 0)
                if already and count < already * 2:
                    return None
                self._reported[fp] = count
                self._counts["repeats"] += 1
                self._bump(path, "repeats")
                norm = normalise(text)
                rec = PreviewRepeat(
                    fp=fp, count=count, span_s=hits[-1] - hits[0], path=path,
                    words=len(norm.split()), chars=len(text.strip()),
                    audio_s=audio_s)
            self._emit(rec)             # outside the lock: emit does file I/O
            return rec
        except Exception:               # noqa: BLE001 - never disturb the preview
            log.debug("preview probe failed", exc_info=True)
            return None

    def shown(self, text: str, *, path: str = PATH_GREEDY,
              audio_s: Optional[float] = None) -> Optional[PreviewRepeat]:
        """observe() plus the per-emission DEBUG line.

        The DEBUG line is the fix for "most of what he saw never reached the
        log": every string the ghost card displays now leaves a trace, in
        shape, whichever path produced it.
        """
        rec = self.observe(text, path=path, audio_s=audio_s)
        try:
            norm = normalise(text or "")
            audio = "—" if audio_s is None else f"{audio_s:.2f}s"
            log.debug("preview shown: path=%s · %dw · %dc · audio=%s · fp=%s",
                      path, len(norm.split()), len((text or "").strip()),
                      audio, self.fingerprint(text or ""))
        except Exception:               # noqa: BLE001
            pass
        return rec

    def retracted(self, path: str = PATH_GREEDY) -> None:
        """The ghost card was taken down (PartialText("")).

        Counted because "it stopped on its own" is half the report, and a
        retraction that never fires is its own bug -- that is exactly the
        2026-08-31 card that stayed up because nothing could remove it.
        """
        try:
            with self._lock:
                self._bump(path, "retractions")
            log.debug("preview retracted: path=%s", path)
        except Exception:               # noqa: BLE001
            pass

    # ------------------------------------------------------------- readback

    def counters(self) -> dict:
        """Totals plus a per-path breakdown. Safe to call from any thread."""
        with self._lock:
            out = dict(self._counts)
            out["tracked"] = len(self._reported)
            out["by_path"] = {p: dict(c) for p, c in self._by_path.items()}
            return out

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()
            self._reported.clear()
            for key in self._counts:
                self._counts[key] = 0
            self._by_path.clear()

    # -------------------------------------------------------------- private

    def _bump(self, path: str, key: str) -> None:
        """Caller holds the lock."""
        self._counts[key] = self._counts.get(key, 0) + 1
        by = self._by_path.setdefault(path, {})
        by[key] = by.get(key, 0) + 1

    def _prune(self, now: float) -> None:
        """Caller holds the lock."""
        cutoff = now - self._window_s
        while self._seen and self._seen[0][0] < cutoff:
            self._seen.popleft()
        if len(self._reported) > MAX_TRACKED:
            live = {f for _, f in self._seen}
            self._reported = {f: n for f, n in self._reported.items() if f in live}

    def _default_emit(self, rec: PreviewRepeat) -> None:
        log.info("%s", rec.line())
        if not self._jsonl_path:
            return
        try:
            payload = asdict(rec)
            payload["at"] = time.time()
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:               # noqa: BLE001 - a ledger must not take the app down
            log.debug("preview ledger write failed", exc_info=True)
