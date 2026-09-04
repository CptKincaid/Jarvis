"""Admission: whether this turn is answered, by whose name, and on which leg.

JARVIS ANSWERS THE PERSON HE RECOGNISES. That is the whole claim and it is
deliberately a small one. A photograph defeats the face check and a recording
defeats the voice check; the user was told that and accepted it. The threat
model he chose is ACCIDENTS AND CONFUSION, not somebody determined to get in,
and every trade below is made under that heading.

ONE HOOK, NOT A SPRINKLING OF CALL SITES
    He chose to gate EVERYTHING including ordinary chat, so the decision is
    taken once, where a turn is admitted (``JarvisApp._process_audio``), and
    the answer travels with the turn. Scattering the question over call sites
    is how half of them end up asking a slightly different one.

GATED_SOURCES IS AN ALLOW-LIST, AND THAT IS THE LOAD-BEARING CHOICE
    Only ``voice`` is gated. ``cli``, ``intercom``, ``phone``, ``typed``,
    ``discord`` and anything added next year are exempt BY CONSTRUCTION,
    rather than by a list of carve-outs somebody has to remember to extend.
    Per source:

    * THE COMMAND SOCKET. His words: "excluding you for now of course since
      you are helping me build him." The argument is written down rather
      than left as an accident: ``cmdsock.py`` chmods the socket 0600, so
      THE FILESYSTEM IS ALREADY THE CREDENTIAL there -- anybody who can open
      it can already read ``assistant.json``, ``people.json`` and the
      voiceprint, and gating it would cost him his way in when the mic is
      dead while buying nothing.
    * THE PHONE. His ruling: "Leave the phone client alone." The bearer
      token stands as the credential (``webapp.py``, compare_digest plus a
      private-peer check plus its own rate limit). OWED, recorded and not
      built: a new phone needs that token re-issued.
    * TYPED, at the Tk box. Same argument as the socket: he is physically at
      the machine. Reversible with one key, ``owner.gate_typed``.
    * DISCORD. His ruling: "Leave discord be for now." It is dormant
      (``discord.user_id`` is empty) and nothing is wired. See the seam
      comment in ``jarvis/commander.py``.

WHY THIS GATE OWNS NO THRESHOLD
    His own verification scores sit in the 0.377-0.397 band against a bar
    already moved down from measured data, and a gate that demanded more
    than the wake word already demands would lock him out of his own house.
    So it demands nothing. The voice leg is the verdict
    ``speaker.filter_segments`` ALREADY REACHED -- it reads ``matched`` and
    nothing else. It never reads the per-window score list, and it never
    reads ``best_score``, which ``filter_segments`` does not set at all
    (``app.py`` and ``intercom.py`` have therefore been reading a constant
    zero from it since it was written -- a pre-existing bug, reported and
    not fixed here). A test enumerates this module's constants and fails if
    any new bar appears, because a second copy of a bar is how a gate
    silently stops meaning what it says.

WHY THERE IS NO CARRY WINDOW, WHICH THE DESIGN PROPOSED AND I DROPPED
    A carry window exists to rescue a follow-up too short to score. But
    ``speaker.verify`` ABSTAINS below its own minimum of speech and returns
    a match, so the short follow-up arrives here with ``matched >= 1``
    already -- the pipeline resolved it in his favour before this module was
    called. A carry window would have had no cases of its own, and every
    extra mechanism in a gate is another way to lock him out. The cost of
    inheriting the abstention rather than papering over it is stated plainly
    and is not new: for a clip under the verifier's speech minimum, whoever
    said it is answered as him. That is the pipeline's existing documented
    fail-open, unchanged by this feature, and it is reachable only inside a
    follow-up window because every other turn needs a wake word, which the
    hotword verifier scores separately.

ONE DOCUMENTED INVARIANT IS DELIBERATELY SUSPENDED HERE, AND SAID OUT LOUD
    ``jarvis/eye.py`` and ``scripts/face_enrol.py`` both write down the
    standing ruling: *identity may REMOVE capability or ADD a name; it must
    never GRANT capability the existing gates do not already grant.* INSIDE
    THIS GATE THE FACE LEG GRANTS. A clip the speaker filter dropped is
    answered when the camera names him, and today's Jarvis would have
    dropped it.

    That is his explicit instruction -- "EITHER voice OR face is enough" --
    and without it the face leg would be decorative, since the voice leg
    already admits everything it is going to admit. The design pass
    reconciled this by claiming the gate "only ever subtracts"; on this
    implementation that is simply not true, and saying so is better than a
    reconciliation that does not hold.

    WHAT BOUNDS THE GRANT, and it is worth stating because it is what makes
    it acceptable: the face leg can only ever name somebody the REGISTRY
    already holds, it can only grant that person's OWN role (a KNOWN face
    rescues a clip into KNOWN scope, not his), and ``eye.resolve_wake`` is
    untouched -- the wake tiebreaker keeps its one-directional rule.

    OWED: the reconciliation belongs in ``eye.py``'s docstring too, and it
    is not written there because another branch is live in that file. It is
    a merge-time edit, deliberately not taken here.

FAILING OPEN ON ITS OWN FAILURE IS NOT FAILING OPEN ON A NEGATIVE
    A missing, corrupt or unreadable registry, nothing enrolled, no leg
    running at all, a raised exception anywhere inside ``judge`` -- all of
    those admit, loudly. Somebody the registry does not know is still
    refused.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from jarvis import passphrase as pp
from jarvis import signinlines
from jarvis.identity import (ROLE_KNOWN, ROLE_OWNER, ROLE_UNKNOWN, Registry,
                             startup_line)
from jarvis.logs import get_logger
from jarvis.recognise import (HOW_FACE, HOW_NOBODY, HOW_PHRASE, HOW_VOICE,
                              Legs, recognise)

log = get_logger("gate")

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"
MODES = (MODE_OFF, MODE_SHADOW, MODE_ENFORCE)

# The ONLY gated source. An allow-list, never an exception list.
GATED_SOURCES = ("voice",)

# Gate-layer outcomes. They are not recognition facts -- a socket, a
# configuration, a broken instrument -- which is why they are here and not
# in jarvis/recognise.py.
HOW_EXEMPT = "exempt"
HOW_GRANT = "grant"
HOW_OFF = "off"
HOW_BLIND = "blind"
HOW_FAULT = "fault"

# Three turns in a row where NOTHING was measuring is a broken gate, not a
# besieged one, and it should stand down out loud rather than hold the door
# against its owner. Abstentions only: a persistent stranger being refused
# is the gate working, and must never be able to talk it into standing down.
DEADMAN_TURNS = 3

# What the transcript is replaced with before anything can publish it.
REDACTED_TEXT = "«passphrase»"

# HOW LONG THE PASSPHRASE BUYS, and why it has to buy anything at all.
# Admitting only the turn that CONTAINED the phrase would be a dead end of a
# new shape: the phrase is not a command, so the next thing he says is the
# thing he wanted, and that clip is rejected exactly as the last one was. He
# would have to say the phrase before every sentence.
#
# 300 s is one sitting, and it is the same number as the attempt window
# below, which is already in this file. THE COST, PLAINLY: for five minutes
# after the phrase is spoken, ANYBODY on the voice path is answered as him.
# Under the threat model he chose -- accidents and confusion, not a
# determined adversary -- that is the right trade; against somebody
# determined it is a hole, and it cannot be closed while the phrase is the
# fallback for a voice the verifier will not match. It lives in memory, so a
# restart ends it.
GRANT_S = 300.0

# ------------------------------------------------------------- the wording
# Replacing app.py's "I only answer to Hunter, sir.", which named no way back
# in -- that was the whole complaint -- and named him to a stranger besides.
UNKNOWN_LINE = ("I don't recognise your voice. The owner can enrol you here "
                "at the keyboard.")
UNKNOWN_PHRASE_LINE = ("I don't recognise your voice. Say the passphrase and "
                       "I'll open up, or the owner can enrol you here at the "
                       "keyboard.")
STANDDOWN_LINE = ("My recognition isn't working, sir — I've stood the gate "
                  "down and I'm answering everyone until you look at it.")
# A {name} template, so never prewarmed -- the same rule commander's
# BUSY_LINE follows.
KNOWN_SCOPE_LINE = ("That one's the owner's, {name}. I can give you the time, "
                    "the weather, and work the music.")
# What he hears when the passphrase lands. Deliberately says nothing about
# the phrase, how close it was, or how long the floor stays open: it is
# spoken out loud in a room that may hold whoever the gate just refused.
PHRASE_OK_LINE = "Thank you, sir. I'm listening."
# The refusal for a voice the gallery could not tell from another enrolled
# one. Imported so there is one wording, in jarvis/signinlines.py.
NEAR_MISS_LINE = signinlines.NEAR_MISS_LINE
PREWARM_LINES = (UNKNOWN_LINE, UNKNOWN_PHRASE_LINE, STANDDOWN_LINE,
                 PHRASE_OK_LINE, NEAR_MISS_LINE)

# ------------------------------------------------------------- the scope
# What a KNOWN person may do. DEFAULT-DENY: an explicit allow-list, and
# everything not on it is refused. That is not over-tightening -- it is his
# three examples ("the time, the weather, music") read as what they are, a
# list. He took the narrower reading over his own earlier "everything".
_OPEN = (
    re.compile(r"\bwhat(?:'s| is)?\s+the\s+(?:time|date|day)\b", re.I),
    re.compile(r"\bwhat\s+time\b", re.I),
    re.compile(r"\bwhat\s+day\s+is\s+it\b", re.I),
    re.compile(r"\b(?:weather|forecast|temperature|raining|snowing)\b", re.I),
    re.compile(r"\b(?:play|pause|resume|skip|next|previous|shuffle|louder|"
               r"quieter|volume|mute|unmute)\b", re.I),
)
# ...and a veto that wins even when an open pattern matched, because
# "play my voicemail" is a music verb reaching his mail. Settings, personal
# data, and anything that leaves the machine.
_HIS = re.compile(
    r"\b(?:mail|email|inbox|calendar|notes?|briefing|memory|remember|"
    r"reminder|grades?|canvas|deadlines?|messages?|voicemail|"
    r"settings?|config(?:ure|uration)?|enrol|enroll|volume\s+of\s+my|"
    r"send|text|cast|remote|hpcomputer|ssh|shut\s*down|restart|"
    r"camera|curfew|passphrase|override)\b", re.I)


@dataclass
class Decision:
    """What the gate decided, and enough of why to debug a wrong answer.

    ``redact`` is the text to publish INSTEAD of the transcript, or "" to
    publish it as spoken. It exists because the spoken passphrase arrives as
    a Whisper transcript, and by the time a transcript reaches ``_dispatch``
    it has already been through the bus, the history, the transcript pane
    and commander's own log line.
    """

    admit: bool = True
    who: str = ""
    role: str = ROLE_UNKNOWN
    how: str = HOW_EXEMPT
    line: str = ""
    redact: str = ""
    why: str = ""
    would_refuse: bool = False


class OwnerGate:
    """One instance, held by the app. Constructing it cannot raise."""

    def __init__(self, *, registry=None, get_option: Optional[Callable] = None,
                 owner: str = ""):
        self.registry = registry
        self.get_option = get_option
        self.owner = str(owner or "")
        self.stood_down = False
        self.kdf_calls = 0
        self._blind_run = 0
        self._grant_who = ""
        self._grant_until = 0.0
        # TWO limiters, never shared, different limits. Burning the
        # passphrase attempts must not lock out the break-glass, or the
        # fallback of last resort fails exactly when it is needed.
        self.phrase_attempts = pp.Attempts(limit=pp.PHRASE_LIMIT,
                                           window_s=pp.PHRASE_WINDOW)
        self.code_attempts = pp.Attempts(limit=pp.CODE_LIMIT,
                                         window_s=pp.CODE_WINDOW)
        if self.registry is None:
            self.registry = Registry.load()

    # -------------------------------------------------------------- state
    def reload(self) -> None:
        path = getattr(self.registry, "path", None)
        self.registry = Registry.load(path)

    def _get(self, key, default=None):
        fn = self.get_option
        if not callable(fn):
            return default
        try:
            return fn(key, default)
        except Exception:  # noqa: BLE001 - a config that cannot say is a default
            log.debug("gate: %s unreadable", key, exc_info=True)
            return default

    def _owner_label(self) -> str:
        reg = self.registry
        if self.owner and reg.role_of(self.owner) == ROLE_OWNER:
            return self.owner
        owners = reg.owners()
        return owners[0].label if owners else ""

    def _mode_unsafe(self) -> str:
        """The live mode. Raises only if the registry does; ``judge`` owns
        that boundary so a broken registry becomes an admit, not a crash."""
        mode = str(self._get("owner.mode", MODE_SHADOW) or MODE_SHADOW).lower()
        if mode not in MODES:
            log.warning("gate: owner.mode is %r, which is not a mode; "
                        "using shadow", mode)
            mode = MODE_SHADOW
        if mode == MODE_OFF:
            return MODE_OFF
        if not getattr(self.registry, "usable", False):
            return MODE_OFF
        if not self.registry.owners():
            return MODE_OFF
        if self.stood_down and mode == MODE_ENFORCE:
            return MODE_SHADOW
        return mode

    def effective_mode(self) -> str:
        try:
            return self._mode_unsafe()
        except Exception:  # noqa: BLE001 - a gate that cannot decide is off
            log.exception("gate: the mode could not be read; the gate is off")
            return MODE_OFF

    def standdown_line(self) -> str:
        return STANDDOWN_LINE

    def startup_line(self, *, voice_ok: bool = False, face_ok: bool = False,
                     face_why: str = "") -> str:
        try:
            return startup_line(self.registry, self.effective_mode(),
                                voice_ok=voice_ok, face_ok=face_ok,
                                face_why=face_why)
        except Exception:  # noqa: BLE001 - the line is never worth a crash
            log.exception("gate: the startup line could not be built")
            return "owner-gate: OFF -- it could not describe itself."

    # --------------------------------------------------------- the scope
    def allowed_for(self, role, text) -> Tuple[bool, str]:
        """What this role may ask for. ``(ok, refusal line)``.

        OWNER: everything, exactly as today. KNOWN: recognised and greeted
        by name, plus a named allow-list of open intents -- and nothing that
        touches a setting, his personal data, or that leaves the machine.
        UNKNOWN never reaches here; it is refused at admission.
        """
        if role == ROLE_OWNER:
            return True, ""
        if role != ROLE_KNOWN:
            return False, UNKNOWN_LINE
        words = str(text or "")
        if _HIS.search(words):
            return False, KNOWN_SCOPE_LINE
        for rx in _OPEN:
            if rx.search(words):
                return True, ""
        return False, KNOWN_SCOPE_LINE

    # ----------------------------------------------------------- the legs
    def _voice_leg(self, stats, rejected) -> Tuple[str, bool]:
        """``(who the voice named, whether it was running at all)``.

        THE ONE LINE THAT STOPS THIS FEATURE LOCKING HIM OUT. ``matched``
        is the pipeline's own verdict and this adds nothing to it. An empty
        stats dict means the verifier was never run (verification off, or no
        voiceprint) -- which is NO INSTRUMENT, never a negative.
        """
        if not isinstance(stats, dict) or "matched" not in stats:
            return "", False
        fault = str(stats.get("who_fault") or "")
        if fault:
            # THE NAMING INSTRUMENT IS BROKEN, and a broken instrument is NO
            # INSTRUMENT -- not a name, not a negative. Before this key
            # existed a gallery that raised arrived here as who="", which is
            # byte-identical to a voice it measured and declined to name, so
            # a wedged store either minted the owner (one label) or refused
            # him (two). Not running is loud: it counts toward the dead-man
            # and stands the gate down rather than guessing either way.
            log.warning("gate: the voice leg's naming instrument faulted "
                        "(%s); treating the leg as not running", fault)
            return "", False
        try:
            hits = int(stats.get("matched") or 0)
        except (TypeError, ValueError):
            return "", False
        if rejected or hits < 1:
            # The clip was dropped, or nothing matched. That is not evidence
            # against anybody else's leg: "no name from me" is all it says.
            return "", True
        who = str(stats.get("who") or "")
        if who:
            # A NAMED match means the person the gallery named. No threshold
            # is applied here and none ever may be. Both bars live in
            # jarvis/voicegallery.py, which is where the numbers were measured.
            return who, True
        owner = self._owner_label()
        # A NAMELESS MATCH IS THE OWNER ONLY WHEN IT WAS MEASURED ON HIS POOL
        # AND THE GALLERY'S BEST GUESS IS NOT SOMEBODY ELSE. Two facts, both
        # from speaker.filter_segments, and neither is a count of labels:
        #
        #   matched_label  whose centroid the kept windows cleared the bar on.
        #                  "" is the voiceprint -- his -- and it is also what
        #                  an abstention carries, since nothing was scored.
        #   top            the label identify() ranked first above the bar,
        #                  named or not: a provisional label ("probably
        #                  mara"), or the winner of a margin that failed.
        #
        # The rule this replaces counted labels: at most one label meant "the
        # only nameless pool is his", two meant "nobody". Measured 2026-09-04
        # it was wrong both ways. A guest's 1.2 s command inside a 4 s clip
        # cleared the bar on HER centroid, identify() abstained, the flag was
        # lost, one label -> the owner, 50 of 50 with owner scope. The same
        # path with him migrated -> two labels -> HIS short commands refused
        # 50 of 50; two guests enrolled without --migrate -> every turn of
        # his refused 50 of 50, provisional labels counting though they can
        # match nobody. And a 6-take "probably mara" was minted as him
        # 150 of 150 at a separation where her voice clears his bar.
        top = str(stats.get("top") or "")
        pool = str(stats.get("matched_label") or "")
        if stats.get("near_miss"):
            # THE MARGIN FAILED: the gallery could not tell two enrolled
            # people apart on this voice, and that is nobody WHICHEVER of
            # them was on top. The first draft of this rule admitted a near
            # miss as the owner when he was on top and the bar had been
            # cleared on his pool ("nothing says it is anybody else"), and
            # measured 2026-09-04 that minted a confusable guest as him 5 of
            # 150 (10 takes) and 22 of 150 (6 takes) at the widest synthetic
            # overlap with him migrated, where the count rule it replaced
            # had minted 0. A refusal that says "say a little more" is the
            # documented answer to a coin flip. It costs him at separations
            # scripts/voice_enrol.py refuses to enrol; at the ones it admits
            # the measured cost is 1 of 150 of his turns, the same margin
            # failure as before this branch.
            log.info("gate: the voice matched with no name and the gallery's "
                     "margin failed (%s on top); that is nobody, not the "
                     "owner", top or "?")
            return "", True
        if top and top != owner:
            # The gallery's best guess above the bar is somebody else --
            # withheld only for takes or for a margin. A recognition that
            # says "probably her" may not become "him".
            log.info("gate: the voice matched with no name and the gallery's "
                     "best guess is %s%s; that is nobody, not the owner", top,
                     " (provisional)" if stats.get("provisional") else
                     " (margin failed)" if stats.get("near_miss") else "")
            return "", True
        if pool and pool != owner:
            # Measured on somebody else's pool. Whatever identify() could not
            # say about it (too little speech, most often), it is not his.
            log.info("gate: the voice cleared the bar on %s's pool with no "
                     "name; that is nobody, not the owner", pool)
            return "", True
        # HIS POOL, OR NOTHING MEASURED. The voiceprint has no label, so a
        # match on it arrives nameless and still means him, exactly as before
        # this feature existed -- however many labels the gallery holds and
        # whatever else is enrolled; nobody else enrolling can move this. An
        # abstention lands here too: every "Yes." he says is under 1.5 s of
        # speech and fails open by design (tests/test_owner_gate.py:74 is the
        # concrete lockout), and it is his whether the clip was short or a
        # short window of a long one -- the pipeline's documented fail-open,
        # unchanged by the gallery.
        if stats.get("abstained"):
            log.info("gate: an abstention on his pool (%d label(s): %s); "
                     "the documented fail-open stands",
                     len(stats.get("labels") or ()),
                     ", ".join(str(x) for x in (stats.get("labels") or ())))
        return owner, True

    @staticmethod
    def _near_miss(stats, rejected) -> bool:
        """A MATCHED clip whose margin failed. A rejected clip (nothing
        cleared the pipeline's bar) may carry the flag from the gallery's
        scoring of its strongest window -- two provisional labels can both
        sit above the gallery's bar while matching nobody -- and telling a
        stranger "I can hear someone I know" on the strength of that would
        be a small lie; the ordinary unknown line answers a rejection."""
        if rejected or not isinstance(stats, dict):
            return False
        try:
            hits = int(stats.get("matched") or 0)
        except (TypeError, ValueError):
            return False
        return hits >= 1 and bool(stats.get("near_miss"))

    def _face_leg(self, face, running) -> Tuple[str, bool]:
        """The gallery's name, mapped to a registry label.

        NO THRESHOLD IS APPLIED HERE. ``eye.Attention.identity`` is already
        past ``camera.identity_min`` -- re-checking it would be a second
        copy of a bar, which is the same mistake this module refuses to make
        on the voice leg.
        """
        if not running:
            return "", False
        name = str(face or "")
        if not name:
            return "", True
        person = self.registry.person(name)
        if person is not None:
            return person.label, True
        return self.registry.face_labels().get(name, ""), True

    def _now(self, now) -> float:
        return time.monotonic() if now is None else float(now)

    def _granted(self, now) -> str:
        """Whom the passphrase window is currently open for, or ""."""
        if not self._grant_who:
            return ""
        if self._now(now) >= self._grant_until:
            self._grant_who = ""
            self._grant_until = 0.0
            return ""
        return self._grant_who

    def _try_phrase(self, text, now) -> str:
        """The owner's way back in, or "". Runs the key derivation ONLY
        when the words were phrase-shaped and recognition has already
        failed, so an ordinary refused sentence pays nothing."""
        if not pp.phrase_shaped(text):
            return ""
        owners = [p for p in self.registry.owners() if p.phrase_hash]
        if not owners:
            return ""
        ok, wait = self.phrase_attempts.allow(now=now)
        if not ok:
            log.info("gate: too many passphrase attempts; %.0fs to wait", wait)
            return ""
        offered = pp.normalise_spoken(text)
        self.phrase_attempts.record(now=now)
        for person in owners:
            self.kdf_calls += 1
            if pp.check_secret(offered, person.phrase_hash):
                return person.label
        return ""

    # -------------------------------------------------------- the verdict
    def judge(self, source, text, *, stats=None, rejected: bool = False,
              face: str = "", face_running: bool = False,
              now: Optional[float] = None) -> Decision:
        """Admit this turn, or refuse it with a way back in.

        EVERY exception in here resolves to an admit. The gate failing must
        never be the reason he cannot speak to his own assistant.
        """
        try:
            return self._judge(source, text, stats=stats, rejected=rejected,
                               face=face, face_running=face_running, now=now)
        except Exception as exc:  # noqa: BLE001 - the gate's own boundary
            log.exception("gate: judging failed; admitting the turn")
            return Decision(admit=True, how=HOW_FAULT,
                            why="the gate failed: %s: %s"
                                % (type(exc).__name__, exc))

    def _judge(self, source, text, *, stats, rejected, face, face_running,
               now) -> Decision:
        src = str(source or "")
        if src not in GATED_SOURCES:
            return Decision(admit=True, how=HOW_EXEMPT,
                            why="%s is not a gated source" % (src or "?"))

        mode = self._mode_unsafe()
        if mode == MODE_OFF:
            return Decision(admit=True, how=HOW_OFF,
                            why=getattr(self.registry, "fault", "")
                                or "the gate is switched off")

        owner = self._owner_label()
        roles = self.registry.roles()
        voice_says, voice_running = self._voice_leg(stats, rejected)
        face_says, face_on = self._face_leg(face, face_running)

        if not voice_running and not face_on:
            # NOTHING IS MEASURING. That is not "nobody is here", and it
            # must never be read as one. It is also what the dead-man's
            # switch counts.
            self._blind_run += 1
            tripped = (self._blind_run >= DEADMAN_TURNS and not self.stood_down)
            if tripped:
                self.stood_down = True
                log.warning("gate: %d turns with no leg running; standing "
                            "down to shadow", self._blind_run)
            return Decision(admit=True, how=HOW_BLIND,
                            line=STANDDOWN_LINE if tripped else "",
                            why="no leg was running")

        verdict = recognise(Legs(voice_says=voice_says,
                                 voice_running=voice_running,
                                 face_says=face_says, face_running=face_on),
                            roles, owner)
        granted = False
        if verdict.how == HOW_NOBODY:
            who = self._granted(now)
            if who:
                granted = True
            else:
                who = self._try_phrase(text, now)
                if who:
                    self._grant_who = who
                    self._grant_until = self._now(now) + GRANT_S
                    log.info("gate: the passphrase opened the floor to %s "
                             "for %.0fs", who, GRANT_S)
            if who:
                verdict = recognise(Legs(phrase_says=who), roles, owner)

        if verdict.how in (HOW_VOICE, HOW_FACE, HOW_PHRASE, HOW_GRANT):
            self._blind_run = 0
            ok, line = self.allowed_for(verdict.role, text)
            person = self.registry.person(verdict.who)
            name = person.display() if person is not None else verdict.who
            if granted and verdict.how == HOW_PHRASE:
                # It was the window the phrase opened, not the phrase: the
                # words are ordinary and must not be redacted, and the log
                # must not claim the phrase was said again.
                verdict = type(verdict)(who=verdict.who, role=verdict.role,
                                        how=HOW_GRANT,
                                        why="inside the passphrase window")
            redact = REDACTED_TEXT if verdict.how == HOW_PHRASE else ""
            if ok:
                log.info("gate: %s (%s) on the %s leg", verdict.who,
                         verdict.role, verdict.how)
                return Decision(admit=True, who=verdict.who, role=verdict.role,
                                how=verdict.how, redact=redact,
                                why=verdict.why)
            log.info("gate: %s is %s; that one is out of scope", verdict.who,
                     verdict.role)
            refuse = Decision(admit=False, who=verdict.who, role=verdict.role,
                              how=verdict.how, line=line.format(name=name),
                              redact=redact, why="out of scope for %s"
                                                 % verdict.role)
            if mode != MODE_ENFORCE:
                return Decision(admit=True, who=verdict.who,
                                role=verdict.role, how=verdict.how,
                                redact=redact, would_refuse=True,
                                why=refuse.why)
            return refuse

        # Nobody. The refusal NAMES THE WAY BACK IN rather than being a dead
        # end, and it never names an owner to somebody it does not know.
        has_phrase = any(p.phrase_hash for p in self.registry.owners())
        line = UNKNOWN_PHRASE_LINE if has_phrase else UNKNOWN_LINE
        if self._near_miss(stats, rejected):
            # The voice IS somebody enrolled and the margin could not say
            # which. "Enrol you at the keyboard" would be the wrong answer
            # to a person who already is; the line names nobody.
            line = NEAR_MISS_LINE
        if mode != MODE_ENFORCE:
            log.info("gate: shadow -- would have refused this turn")
            return Decision(admit=True, how=HOW_NOBODY, would_refuse=True,
                            why=verdict.why)
        log.info("gate: refusing a turn nobody claimed")
        return Decision(admit=False, how=HOW_NOBODY, line=line,
                        why=verdict.why)


def check_override_code(registry, code, *, attempts=None) -> Tuple[str, str]:
    """THE BREAK-GLASS, and it lives on the KEYBOARD/SSH SIDE ONLY.

    ``OwnerGate.judge`` never calls this, by construction: the microphone
    path and this function are on opposite sides of the process, so the code
    cannot be overheard or replayed. Presenting it aloud is worth exactly as
    much as any other sentence, and a test pins that the two refusals are
    identical -- no timing branch, no hint that it was close.

    It is not a wall and must not be described as one. Anybody already at
    the keyboard can edit ``people.json`` or delete it, which turns the gate
    off. This is HIS way back in, nothing more.
    """
    plain = str(code or "").strip()
    if not plain:
        return "", "no code was given"
    try:
        owners = [p for p in registry.owners() if p.code_hash]
    except Exception:  # noqa: BLE001 - a registry that cannot say has no code
        log.exception("gate: the registry could not be asked for a code")
        return "", "the registry could not be read"
    if not owners:
        return "", "no override code has been set"
    if attempts is not None:
        ok, wait = attempts.allow()
        if not ok:
            return "", ("too many tries; wait %.0f seconds and try again"
                        % wait)
        attempts.record()
    for person in owners:
        if pp.check_secret(plain, person.code_hash):
            log.info("gate: the override code admitted %s at the keyboard",
                     person.label)
            return person.label, ""
    return "", "that is not a code I know"
