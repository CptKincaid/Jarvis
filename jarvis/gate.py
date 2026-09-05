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
# A turn inside the window the TYPED code opened (Knightfall, 2026-09-04).
# Its own name rather than a second "grant" so the log says which path let
# him in -- that is all a log line about this feature may ever carry: WHO
# and WHICH PATH, never the code and never the phrase.
HOW_CODE = "code"
HOW_OFF = "off"
HOW_BLIND = "blind"
HOW_FAULT = "fault"
# The legs a turn may be admitted on. HOW_CODE joins the two the window
# already had; the app's rescue set reads the same tuple.
ADMITTED_HOWS = (HOW_VOICE, HOW_FACE, HOW_PHRASE, HOW_GRANT, HOW_CODE)
# THE TWO LEGS HE HIMSELF OPENED, as against the ones a sensor produced.
# Every other leg in ADMITTED_HOWS is something the machine decided about
# him; these two exist only because he said the phrase or typed the code,
# and that difference is what lets them act in shadow (jarvis/app.py,
# _gate_rescue_inner) while the face leg still only logs.
WINDOW_HOWS = (HOW_GRANT, HOW_CODE)
# The two ways the window can be opened, worded for the log.
_OPENER_NAMES = {HOW_PHRASE: "the phrase", HOW_CODE: "the code"}

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
# ...AND WHAT MODE_OFF SAYS INSTEAD, because in off the phrase opens no
# window (see _phrase_consumed) and the speaker filter -- which is not the
# gate and does not care what mode the gate is in -- goes on dropping his
# clips exactly as it did a second earlier. "I'm listening." was therefore
# a promise the code did not keep in that mode: measured 2026-09-05, the
# turn after the phrase dispatched nothing at all in off. This says thank
# you and claims nothing. It is deliberately still opaque to a bystander --
# it does not name the phrase, the mode, or the fact that anything was
# recognised -- and deliberately DIFFERENT from PHRASE_OK_LINE, so that he,
# who knows what the two mean, can hear which mode he is in.
PHRASE_OFF_LINE = "Thank you, sir."
PREWARM_LINES = (UNKNOWN_LINE, UNKNOWN_PHRASE_LINE, STANDDOWN_LINE,
                 PHRASE_OK_LINE, PHRASE_OFF_LINE)

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

    ``consumed`` (Knightfall, 2026-09-04) says the gate ANSWERED THIS TURN
    ITSELF: the words were the spoken phrase, the window is open, ``line``
    is what to say, and the app must dispatch NOTHING -- no commander, no
    model, no transcript but ``redact``. It is set in every mode, because
    the phrase he chose is also the name of an Oracle service, and a phrase
    that travelled on as a sentence would be answered as "is knightfall
    up" or handed to the model.
    """

    admit: bool = True
    who: str = ""
    role: str = ROLE_UNKNOWN
    how: str = HOW_EXEMPT
    line: str = ""
    redact: str = ""
    why: str = ""
    would_refuse: bool = False
    consumed: bool = False


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
        self._grant_how = HOW_PHRASE      # which path opened the window
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
        try:
            hits = int(stats.get("matched") or 0)
        except (TypeError, ValueError):
            return "", False
        if rejected or hits < 1:
            # The clip was dropped, or nothing matched. That is not evidence
            # against anybody else's leg: "no name from me" is all it says.
            return "", True
        return self._owner_label(), True

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

    def open_window(self, who: str, how: str, now=None) -> None:
        """THE ONE WINDOW OPENER, for the spoken phrase and the typed code
        alike. Opens the floor to ``who`` for GRANT_S; ``how`` is HOW_PHRASE
        or HOW_CODE and is what a turn inside the window is attributed to.
        The log line carries who, which path and the live mode -- nothing
        else, ever."""
        self._grant_who = str(who or "")
        self._grant_how = HOW_CODE if how == HOW_CODE else HOW_PHRASE
        self._grant_until = self._now(now) + GRANT_S
        log.info("gate: %s opened the floor to %s for %.0fs (mode=%s)",
                 _OPENER_NAMES[self._grant_how], self._grant_who, GRANT_S,
                 self.effective_mode())

    def _phrase_consumed(self, who: str, now, *,
                         window: bool = True) -> "Decision":
        """The one verdict the phrase produces, in EVERY mode: the turn
        ends here. Nothing of the words travels -- the text is
        REDACTED_TEXT and the app dispatches nothing.

        ``window`` is False in MODE_OFF only. Off already admits every
        turn, so the window would buy him nothing -- while the phrase
        check in that mode is deliberately unlimited, and an unlimited
        path that leaves a five-minute admission behind it would still be
        open if the mode were moved to enforce a minute later."""
        if not window:
            log.info("gate: the phrase was consumed for %s; the gate is off, "
                     "so no floor was opened", who)
            return Decision(admit=True, who=who, role=ROLE_OWNER,
                            how=HOW_PHRASE, line=PHRASE_OFF_LINE,
                            redact=REDACTED_TEXT, consumed=True,
                            why="the phrase; the gate is off and answered "
                                "this turn")
        self.open_window(who, HOW_PHRASE, now=now)
        return Decision(admit=True, who=who, role=ROLE_OWNER,
                        how=HOW_PHRASE, line=PHRASE_OK_LINE,
                        redact=REDACTED_TEXT, consumed=True,
                        why="the phrase; the gate answered this turn")

    def _try_phrase(self, text, now, *, limited: bool = True) -> str:
        """The owner's phrase, or "". The key derivation runs ONLY when the
        words were phrase-shaped (the cheap pre-filter, unchanged), so
        "yes", "no" and silence pay nothing.

        ``limited`` is whether this counts against the five-per-window
        limiter: it does for a voice nobody recognised (a stranger
        guessing, exactly as before) and it does NOT for a turn a leg has
        already named an owner on. He is not attempting anything by
        talking; if his own sentences burned attempts, the sixth sentence
        in five minutes would close the phrase for the rest of the window
        and "Knightfall protocol" would reach the Oracle lane.

        THE COST OF CHECKING EVERY TURN, stated rather than hidden: a
        recognised owner's phrase-shaped sentence now pays one scrypt per
        owner with a phrase set (~20 ms at n=2**14, measured in
        jarvis/passphrase.py). A cheaper bound that excludes ordinary
        sentences would have to know something about the phrase -- its
        length, its word count -- and storing that beside the hash is a
        leak of the phrase's shape. The 20 ms was taken instead; see
        tests/test_knightfall_kdf_cost.py for the count per 100 turns.
        """
        if not pp.phrase_shaped(text):
            return ""
        owners = [p for p in self.registry.owners() if p.phrase_hash]
        if not owners:
            return ""
        if limited:
            ok, wait = self.phrase_attempts.allow(now=now)
            if not ok:
                log.info("gate: too many passphrase attempts; %.0fs to wait",
                         wait)
                return ""
        offered = pp.spoken_candidates(text)
        if not offered:
            return ""
        if limited:
            self.phrase_attempts.record(now=now)
        for person in owners:
            for candidate in offered:
                self.kdf_calls += 1
                if pp.check_secret(candidate, person.phrase_hash):
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
            # OFF ADMITS EVERYTHING; IT DOES NOT REPEAT HIS PASSPHRASE.
            # This return used to be the first thing in the function, so
            # the one mode where the gate is switched off was the one mode
            # that dispatched the phrase as an ordinary command -- onto the
            # bus, into the history, the transcript pane and the model
            # (verdict, 2026-09-05). Consuming it here costs nothing but
            # the derivation and gives away nothing: the window it opens
            # grants what off already grants.
            #
            # UNLIMITED, and that is the point of it: the limiter exists to
            # stop a stranger guessing his way past a gate, and there is no
            # gate here to guess past. If his own sentences burned attempts
            # in this mode, the sixth phrase-shaped one in five minutes
            # would put the phrase back on the bus -- the leak this closes.
            spoke = self._try_phrase(text, now, limited=False)
            if spoke:
                return self._phrase_consumed(spoke, now, window=False)
            return Decision(admit=True, how=HOW_OFF,
                            why=getattr(self.registry, "fault", "")
                                or "the gate is switched off")

        owner = self._owner_label()
        roles = self.registry.roles()
        voice_says, voice_running = self._voice_leg(stats, rejected)
        face_says, face_on = self._face_leg(face, face_running)
        named = recognise(Legs(voice_says=voice_says,
                               voice_running=voice_running,
                               face_says=face_says, face_running=face_on),
                          roles, owner)

        # THE PHRASE, BEFORE ANYTHING ELSE CAN ACT ON THE WORDS. Whether a
        # leg named him or not, in shadow and in enforce, a matching phrase
        # is CONSUMED here: the window opens, the line is what he hears,
        # and the app dispatches nothing. The limiter counts only a voice
        # nobody recognised (see _try_phrase).
        # THE EXEMPTION IS THE OWNER'S, AND ONLY HIS. This used to read
        # ``limited=not named.who`` -- "somebody was named" -- which is not
        # the same sentence at all: a KNOWN non-owner in front of the lens
        # (a guest, the cleaner, anybody the camera has a row for) turned
        # the five-per-window limiter OFF and bought a guesser unlimited
        # attempts at one scrypt each, with the owner nowhere in the room.
        # Measured 2026-09-05: 40 phrase-shaped guesses cost 40 key
        # derivations with a KNOWN face in view, against 5 with nobody
        # named. The reason for the exemption was only ever HIS OWN
        # sentences not burning his own way back in, so it is his alone.
        owner_named = bool(named.who) and named.role == ROLE_OWNER
        spoke = self._try_phrase(text, now, limited=not owner_named)
        if spoke:
            return self._phrase_consumed(spoke, now)

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

        verdict = named
        if verdict.how == HOW_NOBODY:
            who = self._granted(now)
            if who:
                # The window, not the phrase: the words are ordinary and
                # must not be redacted, and the log must not claim the
                # phrase was said again. Attributed to whichever path
                # opened it, so the log says which.
                verdict = recognise(Legs(phrase_says=who), roles, owner)
                verdict = type(verdict)(who=verdict.who, role=verdict.role,
                                        how=self._grant_how if
                                        self._grant_how == HOW_CODE
                                        else HOW_GRANT,
                                        why="inside the %s window"
                                            % _OPENER_NAMES[self._grant_how])

        if verdict.how in ADMITTED_HOWS:
            self._blind_run = 0
            ok, line = self.allowed_for(verdict.role, text)
            person = self.registry.person(verdict.who)
            name = person.display() if person is not None else verdict.who
            if ok:
                log.info("gate: %s (%s) on the %s leg", verdict.who,
                         verdict.role, verdict.how)
                return Decision(admit=True, who=verdict.who, role=verdict.role,
                                how=verdict.how, why=verdict.why)
            log.info("gate: %s is %s; that one is out of scope", verdict.who,
                     verdict.role)
            refuse = Decision(admit=False, who=verdict.who, role=verdict.role,
                              how=verdict.how, line=line.format(name=name),
                              why="out of scope for %s" % verdict.role)
            if mode != MODE_ENFORCE:
                return Decision(admit=True, who=verdict.who,
                                role=verdict.role, how=verdict.how,
                                would_refuse=True, why=refuse.why)
            return refuse

        # Nobody. The refusal NAMES THE WAY BACK IN rather than being a dead
        # end, and it never names an owner to somebody it does not know.
        has_phrase = any(p.phrase_hash for p in self.registry.owners())
        line = UNKNOWN_PHRASE_LINE if has_phrase else UNKNOWN_LINE
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
