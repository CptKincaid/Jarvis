"""Email a file, by voice, with a read-back that has to be answered.

Everything else this assistant does can be taken back. A timer is
cancelled, a note is struck, a calendar event has an undo closure hanging
off its CommandResult. An attachment that has left the machine has none of
that, and neither does sending it as the wrong one of his three identities
— personal, work and school are three different people as far as the
recipient is concerned.

So the shape is: NOTHING happens on the first utterance. The request
becomes a :class:`Draft`, the draft is read back — the file NAME, its SIZE,
the recipient ADDRESS and the SENDING ACCOUNT, in that order, because that
is the order in which a mistake gets worse — and the draft is spent only by
an explicit yes (``commander._try_send_confirm``). Every gap in the request
becomes a QUESTION rather than a default:

    unknown recipient       -> ask; an address is never inferred from a name
    two matching files      -> ask, naming them
    nothing matched         -> ask
    three accounts, no hint -> ask which

and six conditions refuse outright (nothing said, not found, unreadable,
a folder, over the cap, resolving outside the search roots).

The pieces underneath are shared with the HPCOMPUTER lane on purpose:
:mod:`jarvis.tools.filepick` owns containment and the refusal vocabulary,
:mod:`jarvis.filephrase` adds the spoken shapes ("that file on my desktop"),
and ``jarvis.tools.mail.send_message`` is the transport. None of it is
reachable by the model: there is no ToolSpec on this path, deliberately —
an irreversible action must be something he asked for out loud and then
confirmed, never something a tool loop decided.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from jarvis import contacts as contacts_mod
from jarvis.config import PATHS
from jarvis import filephrase
from jarvis import spelling
from jarvis.logs import get_logger
from jarvis.tools import filepick
from jarvis.tools import mail as mail_mod

log = get_logger("outbox")

# How long a read-back draft stays answerable. Longer than the 60 s of
# commander.DESTRUCTIVE_TTL_S because this read-back is longer to hear and
# more to check: four facts, one of them an address he has to match against
# the person in his head. Short enough that a "yes" said minutes later, to
# something else entirely, cannot land on it.
DRAFT_TTL_S = 90.0
# Gmail's SMTP refuses a message over 25 MB, and base64 inflates an
# attachment by 4/3 plus headers — so 25 MB on the wire is about 18.3 MB of
# file. 18 MB is therefore the largest attachment that can actually be
# DELIVERED. A larger cap would not send more; it would only move the
# refusal from Jarvis, who can say why in one sentence BEFORE the read-back
# promises anything, to the SMTP server, who says it in a bounce after
# Jarvis has already reported the file sent.
MAX_ATTACHMENT_MB = 18
SUBJECT_MAX = 120
DEFAULT_BODY = "Sent from Jarvis."

SENT_LINE = "Sent to {who}, sir."
DROPPED_LINE = "Very good, sir; nothing sent."
SELF_LINE = ("That draft is to {who}, sir, not to you. Yes to send it there, "
             "no to drop it.")
# A pronoun of the OTHER gender after a read-back (Hunter's 19:00 ruling,
# 09-04): "send it to him" with Heather pending is not her, so it is not a
# confirmation. The re-ask names the pending person and the pronoun that
# IS theirs, so the yes he gives next is to a sentence he has just heard.
GENDER_LINE = "The draft is to {who}, sir. Send it to {pron}?"
UNSURE_LINE = ("I'd rather be certain, sir — say yes and I'll send it, "
               "or no and I'll let it go.")
CHANGED_LINE = ("That file has changed since I read it back, sir; "
                "nothing was sent. Ask me again and I'll take another look.")
GONE_LINE = "That file has gone, sir; nothing was sent."
EMPTY_LINE = "{what} is empty, sir; there'd be nothing to attach."
NO_RECIPIENT_LINE = "I've no address for {who}, sir. What is it?"
# An address he SAID that the parser cannot read whole -- "heather tilde
# smith at example dot com" (round 4, 09-05). Said back as heard and asked
# for again; never cut to the part after the word it did not know.
HEARD_LINE = "I heard {heard}, sir — I can't make an address of that. What is it?"
WHO_LINE = "Who should I send it to, sir?"
WHICH_FILE_LINE = "Which file, sir?"
# The answers to the two questions above, when they miss (F21). One re-ask
# each, the way the read-back and "Which one, sir?" already get one.
ADDRESS_REASK_LINE = ("I didn't catch an address there, sir — say it as "
                      "name at domain dot com, or a name I know.")
ACCOUNT_REASK_LINE = "I've no {hint} account, sir — {names}?"
ACCOUNT_WHICH_LINE = "Which of them, sir — {names}?"
ASK_DROPPED_LINE = "I'll leave it there, sir; ask me again when you have it."
ASK_SPENT_LINE = "Very good, sir; nothing sent."
# "Which Heather, sir — Heather Smith or Heather Jones?" -- the address
# book's question (jarvis/contacts.py), when two rows answer to the name he
# said. The status strip carries this string and the commander branches on
# it BEFORE the file offer, because the file offer's answer grammar is the
# fuzzy one and a list of people must never reach it.
WHICH_PERSON_STATUS = "Which person?"
# The cap refusal is the one place this lane does NOT reuse
# filepick.reason_line: filepick's wording ("past the N I'll put on the wire
# without you saying so plainly") offers an override, and for mail there is
# none to offer — Gmail will not carry it however plainly he says so.
TOO_BIG_LINE = ("{what} is {size}, sir — Gmail won't carry more than "
                "{cap} as an attachment, so I've not sent it.")

# The rehearsal and the two send failures. AUTH and WIRE are separate
# sentences because they are separate problems and only one of them is
# his to fix: a revoked app password reported as "that didn't send" sends
# him looking at his network. Every one of them promises, in words, that
# nothing left the machine -- so a failure can never be mistaken for a
# send.
REHEARSAL_LINE = "Rehearsal only, sir — nothing left the machine."
AUTH_FAILED_LINE = ("Gmail wouldn't take your {label} password, sir; "
                    "nothing was sent.")
WIRE_FAILED_LINE = ("That didn't send, sir. Nothing has left the machine — "
                    "the file is still here, and you can ask me again.")
# Past filepick.MAX_CANDIDATES the offer is REPLACED rather than shortened:
# a cut list invites him to pick from candidates the right file may not be
# in, and he has no way to know it was cut. filephrase.Match.total is the
# count this line reads.
TOO_MANY_LINE = "I've {n} it could be, sir — give me more of the name."
# The read-back's own question was answered with something that is not an
# address. One re-ask, then the slot is spent.
NOT_AN_ADDRESS_LINE = ("I didn't catch an address there, sir. Say it as "
                       "heather at example dot com, or say never mind.")
EXPIRED_STATUS = "Not sent — expired"

# THE AUDIT. There is no unsend over SMTP -- Gmail's is a client-side delay
# its web app implements and its submission server does not -- so a record
# of what left is the only thing that replaces an undo. Append-only, one
# JSON object per line, and it holds NO body and NO password.
SENT_LOG = PATHS.MEMORY_DIR / "sent.jsonl"


@dataclass
class Draft:
    """A message that has been read back and not sent."""
    path: Path
    size: int
    mtime: float
    to_addr: str
    to_name: str
    account: dict
    subject: str
    body: str = DEFAULT_BODY
    made_at: float = 0.0             # time.monotonic() when it was armed
    # The roots the file was resolved inside, carried so send() can re-run
    # the SAME containment check the draft passed rather than a weaker one.
    roots: list = field(default_factory=list)
    # One vague answer ("okay") gets one re-ask before the draft is spent.
    # See commander._try_send_confirm: a silent drop on a word he meant as
    # a yes is how a man learns the feature does not work, and treating it
    # AS a yes is how a file reaches the wrong person.
    reasked: bool = False
    # WHICH CHANNEL the read-back was spoken on. A question put out loud at
    # the desk cannot be answered from Discord, a phone client, a tmux
    # shell or a socket -- those turns never heard it. commander.stash_send
    # records the turn's source here and _try_send_confirm requires the
    # answer to come back the same way, the rule _try_briefing_offer
    # already applies to a question that is entirely reversible.
    asked_from: str = "voice"
    # The recipient came out of the ADDRESS BOOK (jarvis/contacts.py): a
    # person he typed and validated, with a name to say. The read-back then
    # speaks the name and SHOWS the address (CommandResult.display_only) --
    # an address is a bad minute of TTS, and he ruled it. A spoken address,
    # a legacy send_file.contacts hit and a memory hit keep the spelled-out
    # wording: they have no validated name to say instead.
    from_book: bool = False
    honorific: str = ""              # "Dr" -- spoken before the name only
    # THE ONE SEAM for the recipient's gender (Hunter's 19:00 ruling,
    # 09-04): "f" / "m" / None. None by default -- and None means either
    # pronoun confirms, exactly as before the ruling. Filled by whoever
    # KNOWS, from an explicit source only: a stored honorific on the person
    # or on a book row (prepare -> recipient_gender), or a pronoun Hunter
    # himself used about the person earlier in the same draft conversation
    # ("her address is dana at ..." -> commander's address answer). Never
    # from the name: nothing in this file guesses a gender from "Heather".
    # prepare fills it: the ADDRESS BOOK row's stored honorific first (the
    # row he typed and validated), then recipient_gender's sources. The
    # confirmation grammar reads only this field.
    to_gender: Optional[str] = None

    @property
    def account_label(self) -> str:
        return mail_mod.account_label(self.account)

    def stale(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        return (now - float(self.made_at or 0.0)) > DRAFT_TTL_S


@dataclass
class Prepared:
    """Either a draft to read back, or the question/refusal to speak."""
    draft: Optional[Draft] = None
    ask: str = ""
    status: str = ""
    candidates: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.draft is not None


# One config reader for the whole mail path: mail._cfg_get already copes
# with a dict, an AssistantConfig and a duck-typed double, and a second
# implementation here would drift from it.
_cfg_get = mail_mod._cfg_get


# ------------------------------------------------------------- wording
def spoken_name(path) -> str:
    """A file name a voice can read: separators become spaces.

    "Biosensors_Lab-Report_v2.pdf" -> "Biosensors Lab Report v2.pdf". The
    extension stays on — it is half of how he tells two files apart — and
    the exact name still goes to the status strip for his eyes.
    """
    name = Path(path).name
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    stem = re.sub(r"[_\-.]+", " ", stem).strip()
    stem = re.sub(r"\s{2,}", " ", stem)
    return f"{stem}.{suffix}" if suffix else stem


def spoken_size(n: int) -> str:
    """"2.4 megabytes" / "312 kilobytes" / "under a kilobyte"."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "an unknown size"
    if n < 1024:
        return "under a kilobyte"
    if n < 1024 * 1024:
        return f"{round(n / 1024)} kilobytes"
    mb = n / (1024 * 1024)
    if mb < 10:
        return f"{mb:.1f} megabytes"
    return f"{round(mb)} megabytes"


def spoken_address(addr: str) -> str:
    """"heather@example.com" -> "heather at example dot com".

    The read-back exists so he can HEAR where the file is going, and
    neither engine is reliable on a raw address (edge-tts spells some
    domains letter by letter; F5 clones prosody from a reference clip that
    has never said an "@"). Written out, the address is unambiguous in the
    ear and still readable on screen.
    """
    text = str(addr or "").strip()
    if not text:
        return ""
    return (text.replace("@", " at ").replace(".", " dot ")
                .replace("_", " underscore ").replace("-", " dash "))


_ADDRESS_IN_TEXT_RX = re.compile(r"[^\s@<>,;]+@[^\s@<>,;]+")

# The same address SAID -- "dana at example dot com", "d dot ruiz at mail
# dot tamu dot edu", Whisper's "Dana at gmail. com" -- which is how it
# reaches the commander far more often than the symbols do. The local part
# may be joined by EVERY word address_span reads as a joiner (dot, period,
# full stop, underscore / under score, dash, hyphen): a shape the parser
# will send to must be a shape this masks, or the log carries what the
# mailbox gets. The domain is one or more labels (a label may carry a
# said dash: "my dash host") and an alphabetic top level, said ("dot",
# "period") or punctuated (". ", ".").
#
# This regex is now the SECOND of the two the masker runs, and it earns
# its place on the punctuated shapes the PARSER still refuses: a domain
# punctuated with a SPACE after the stop ("Dana at gmail. com") and a
# local part joined by a bare dot ("dana.ruiz at example dot com").
# (09-05, HIS RULING B) It no longer earns it on "dana at gmail.com": a
# TIGHT punctuated domain is an address the parser reads now, so
# _SPOKEN_ADDR_RX matches it and _one_drafted masks it either way.
# Everything said with the word "dot" is masked by the parser's own regex
# too, so no hand-written list decides whether an address is an address.
_SAID_JOIN = (r"(?:\s+(?:dot|period|full\s+stop|under\s*score|dash|hyphen)\s+"
              r"|\.(?!\s))")
_SAID_SEP = r"(?:\s+(?:dot|period|full\s+stop)\s+|\.\s*)"
_SAID_LABEL = r"[A-Za-z0-9][\w\-]*(?:\s+(?:dash|hyphen)\s+[A-Za-z0-9][\w\-]*)*"
_SAID_ADDR_RX = re.compile(
    r"(?<![\w@.\-])"
    r"(?P<local>[A-Za-z0-9][\w+\-]*(?:" + _SAID_JOIN + r"[A-Za-z0-9][\w+\-]*)*)"
    r"(?P<rest>\s+at\s+"
    r"(?P<domain>" + _SAID_LABEL + r"(?:" + _SAID_SEP + _SAID_LABEL + r")*"
    + _SAID_SEP + r"(?P<tld>[A-Za-z]{2,24})))"
    r"(?![\w\-])", re.I)
_AT_HINT_RX = re.compile(r"\bat\b", re.I)
# The SPELLED shape: how whisper writes an address he said one character
# at a time, in every punctuation it has been seen to hang on the letters
# -- "q. z. v. k. b. w. 7. at example.com" (the 09-06 verdict), "q-z-v,
# k-b-w-7, at example dot com" (the fourth fragment he was left with on
# 09-05), "q z v k b w seven at example dot com", "d.a.n at example.com".
# The said-shape mask below reads ONE label as the local part, so on a
# run it keeps the last group's first letter and leaves the rest raw:
# "q-z-v, k… at example.com", three characters of seven in a log line.
# Here the local part is the whole run -- two or more single characters,
# whatever separates them, plus at most one trailing label ("seven") --
# and the whole of it is cut to its first character, the way every other
# local part is. The domain has to be a domain (said "dot", or punctuated
# and ending on a top level in use, the _one_said rule), so "grades a, b,
# c at noon. Come by" is left alone.
_SPELLED_CHAR = r"[A-Za-z0-9]"
_SPELLED_SEP = r"(?:\s*[.,;:\-–—]\s*|\s+)"
_SPELLED_ADDR_RX = re.compile(
    r"(?<![\w@.\-'’])"
    r"(?P<local>" + _SPELLED_CHAR + r"(?:" + _SPELLED_SEP + _SPELLED_CHAR + r")+"
    r"(?:\s+[A-Za-z0-9][\w+\-]*)?)"
    r"(?![\w\-])[.,;:]?"
    r"(?P<rest>\s+at\s+"
    r"(?P<domain>" + _SAID_LABEL + r"(?:" + _SAID_SEP + _SAID_LABEL + r")*"
    + _SAID_SEP + r"(?P<tld>[A-Za-z]{2,24})))"
    r"(?![\w\-])", re.I)
# Top levels a SPACED punctuated domain may end on ("gmail. com").
# English words that are also top levels (in, me, us, it, is, be, no, to,
# at, so, info) are left out on purpose: after a full stop they are the
# next sentence far more often than an address -- "we stopped at noon.
# Then we left". Said with the WORD "dot", and (09-05) written with a
# TIGHT dot, they need no list at all: the parser reads them as an
# address, so the parser rule below masks them. This list is therefore
# down to ONE job -- the spaced shape the parser still will not read.
_SAID_TLDS = frozenset("""
    com org net edu gov mil int io co ai dev app biz tv uk ca au de fr
    nl es ie ch eu nz jp cn br mx ru se fi dk pl cz pt gr tr za kr
""".split())


def _mask_local(local: str) -> str:
    """A local part cut to its first letter -- "dana" -> "d…" -- and a
    ONE-letter local part dropped whole, because keeping the first letter
    of "q" keeps the whole of it. That single case is the only place this
    mask could ever say more than jarvis-v3's, which drops every local
    part; everywhere else the kept letter is what makes the log line
    readable."""
    return local[:1] + "…" if len(local) > 1 else "…"


def _one_drafted(m) -> str:
    """A run the PARSER would draft an address from, masked -- found with
    the parser's OWN compiled regex (_SPOKEN_ADDR_RX), never a copy and
    never a list.

    THREE ROUNDS tuned hand-written word lists here and every round a
    verdict found a class they let through raw into four jarvis.commander
    INFO lines: round 5 the top levels no list contains (.site .xyz
    .info), round 6 the eight real ccTLDs that are also English function
    words (.at .be .in .is .it .no .so .to -- "dana at my dot in" drafted
    dana@my.in, was mailed by a yes, and was logged verbatim). The design
    was the problem. The invariant Hunter wants -- an address he SAID
    never reaches a log line raw -- needs no list, because the PARSER
    already decides what an address is: if address_span drafts a recipient
    from a span, that span IS an address, so masking exactly what the
    parser reads makes it impossible for the two to disagree. Anything the
    parser learns to read tomorrow, this masks in the same edit.

    It is also jarvis-v3's floor by construction: d665b0f's mask_addresses
    is this same regex (its `"… at " + m.group(2)`), so nothing v3 masks
    today can come out less masked here."""
    return _mask_local(m.group(1)) + m.group(0)[len(m.group(1)):]


def _one_said(m) -> str:
    """The runs the parser does NOT read, and this still must: a domain
    Whisper punctuated with a SPACE after the stop ("Dana at gmail. com")
    and a local part joined by a bare dot ("dana.ruiz at example dot
    com"). address_span refuses these, so no mail can go to them, but they
    are still the address he said and the log is read by more eyes than
    the mailbox is.

    (09-05, HIS RULING B) "dana at gmail.com" is no longer in that set --
    a TIGHT punctuated domain is one the parser reads. This pass still
    matches it and still masks it; _one_drafted would now mask it anyway,
    so the two agree instead of one covering for the other.

    Ordinary prose has the same skeleton ("we stopped at noon. Then we
    left"), so a punctuated domain has to end on a top level actually in
    use. That exemption is safe HERE and only here: it can never leave a
    draftable span raw, because _one_drafted runs after it over every span
    the parser reads, unconditionally and without consulting any list."""
    if "." in m.group("domain") and m.group("tld").lower() not in _SAID_TLDS:
        return m.group(0)
    return _mask_local(m.group("local")) + m.group("rest")


def mask_addresses(text) -> str:
    """"yes, to hjones@example.com" -> "yes, to h…@example.com", and
    "send it to dana at example dot com" -> "send it to d… at example dot
    com": every address-shaped run in a sentence, typed or SAID, masked
    the way mail and this module already mask a single address, for a log
    line that carries what he said. The rest of the sentence is kept -- it
    is the line's whole point -- the domain stays as he put it, and a
    trailing full stop stays outside the mask.

    Four passes, in this order: the typed address; then the SPELLED run
    (_SPELLED_ADDR_RX: the characters he said one at a time, in whatever
    whisper hung on them, which the one-label rule below would leave
    mostly raw); then the punctuated shapes the parser cannot read
    (_one_said, which may exempt prose); then
    -- last, and over everything -- every span the PARSER would draft from,
    masked with the parser's own regex, so the exemption above can never
    leave an address raw. Prose pays for that: a sentence the parser would
    have drafted an address out of ("look at the dot in the corner" ->
    "l… at the dot in the corner") loses its head word down to one letter
    in a log line. That is the trade, taken deliberately and for the third
    time: a letter is cheaper than an address."""
    text = str(text or "")
    if "@" in text:
        def _one(m):
            token = m.group(0)
            tail = ""
            while token and token[-1] in ".,;:!?":
                tail, token = token[-1] + tail, token[:-1]
            return mail_mod._mask_address(token) + tail

        text = _ADDRESS_IN_TEXT_RX.sub(_one, text)
    if _AT_HINT_RX.search(text):
        text = _SPELLED_ADDR_RX.sub(_one_said, text)
        text = _SAID_ADDR_RX.sub(_one_said, text)
        text = _SPOKEN_ADDR_RX.sub(_one_drafted, text)
    return text


def spoken_who(who: str) -> str:
    """A recipient as he said it, fit to be said back: an address, or the
    half of one ("heather@example", "heather@"), is spoken; a name --
    "Mary-Jane" included -- is left alone. Attack 2 (round 4): the
    no-address line echoed a typed half-address with its "@" in it."""
    w = " ".join(str(who or "").split())
    return spoken_address(w).strip() if "@" in w else w


def spoken_recipient(name: object = "", addr: object = "") -> str:
    """THE ONE PLACE a recipient becomes something Jarvis SAYS OUT LOUD.

    Name first when there is one, the address in words when there is not
    -- and NOTHING that leaves here carries an "@", whichever slot it
    arrived in. Jarvis must never say an at sign: neither engine is
    reliable on a raw address (edge-tts spells some domains letter by
    letter; F5 clones prosody from a reference clip that has never said
    one), so an address he HEARS has to be an address in words.

    Every spoken line that names a recipient goes through here -- the
    read-back, the correction, the re-ask, the sent line, the day's audit
    summary. It exists because ``to_name or spoken_address(to_addr)`` was
    written out by hand in four places and each copy spoke ``to_name``
    UNTOUCHED: a half address that reached the name slot ("hjones@example"
    -- round 4's attack), or a masked audit row ("d…@example.com" -- the
    sent-log summary, measured), was said with its "@" in it. Masking is
    not speaking. A field being safe on disk says nothing about whether
    it is safe in the ear, and only this function decides that.
    """
    said = spoken_who(str(name or ""))
    return said or spoken_address(str(addr or "")).strip()


def auth_failed_line(label: object = "") -> str:
    """The refused-password sentence, with the account label SPOKEN.

    account_label falls back to whatever ``label`` the config carries,
    raw, and this string is read out: a label he set to an address would
    put an "@" in a spoken line by the same route the recipient did.
    """
    return AUTH_FAILED_LINE.format(label=spoken_who(str(label or "")) or "mail")


def pronoun_for(gender: Optional[str]) -> str:
    """"her" / "him" for a known gender, "them" for none."""
    return {"f": "her", "m": "him"}.get(str(gender or "").lower(), "them")


def account_words(account: dict) -> str:
    """"your school account".

    Through spoken_who, because account_label returns the configured
    label RAW and this sentence is said out loud.
    """
    label = spoken_who(mail_mod.account_label(account))
    return f"your {label} account" if label else "your account"


def _to_words(draft: Draft) -> str:
    # spoken_who, not the raw field: a half address that landed in the
    # name slot ("hjones@example") is said here, and it must be said in
    # words like every other address.
    who = spoken_who(str(draft.to_name or ""))
    if getattr(draft, "from_book", False) and who:
        hon = str(getattr(draft, "honorific", "") or "").strip()
        return f"{hon} {who}".strip()
    addr = spoken_address(draft.to_addr)
    return f"{who}, at {addr}" if who else addr


def shown_address(draft: Draft) -> str:
    """The half of a book read-back that is SHOWN and never spoken:
    "to heather@example.com". "" for every other kind of recipient, whose
    address is already in the spoken line."""
    if getattr(draft, "from_book", False) and draft.to_name:
        return f"to {draft.to_addr}"
    return ""


def read_back(draft: Draft) -> str:
    """The one sentence standing between the file and the recipient.

    File, size, address, account, then the question — nothing else, and the
    question LAST so the yes he gives is to a sentence he has heard all of.
    """
    return (f"{spoken_name(draft.path)}, {spoken_size(draft.size)}, "
            f"to {_to_words(draft)}, from {account_words(draft.account)}. "
            f"Send it, sir?")


def gender_line(draft: Draft) -> str:
    """The re-ask for a pronoun that is not the pending person's (Hunter's
    19:00 ruling): names them, and the pronoun that is theirs."""
    who = spoken_recipient(draft.to_name, draft.to_addr)
    return GENDER_LINE.format(who=who, pron=pronoun_for(getattr(draft, "to_gender", None)))


def unsure_line(draft: Draft) -> str:
    """The one re-ask a vague answer gets -- and it names the file and the
    recipient again (F23, 09-03). The generic UNSURE_LINE asked for a yes
    without saying what the yes was to, so the second yes was to a sentence
    he had not heard since the first one; the read-back exists so he hears
    where the file is going, and the re-ask is the same question."""
    return (f"I'd rather be certain, sir — that's {spoken_name(draft.path)} "
            f"to {_to_words(draft)}. Say yes and I'll send it, or no and "
            f"I'll let it go.")


def offer_line(candidates, what: str = "") -> str:
    """The question for an ambiguous match, naming the rivals.

    Names only, never paths: "lab report.pdf or lab report final.pdf" is
    the distinction he has to make, and a spoken absolute path is four
    seconds of /home/hunterp he already knows. The names go through
    spoken_name rather than filepick.describe's raw ones, because this
    question is ASKED OUT LOUD and an underscore read as "underscore"
    buries the difference between the two files it is asking about.
    """
    names = [spoken_name(p) for p in (candidates or ())]
    if not names:
        return WHICH_FILE_LINE
    listed = (names[0] if len(names) == 1
              else ", ".join(names[:-1]) + " or " + names[-1])
    if len(names) == 1:
        return f"Do you mean {listed}, sir?"
    lead = f"I've {len(names)} that could be {what}" if what else \
        f"I've {len(names)} it could be"
    return f"{lead}, sir: {listed}. Which one?"


def refusal_line(match, said: str = "", cap_mb: float = MAX_ATTACHMENT_MB) -> str:
    """One sentence for a Match that is neither ok nor ambiguous."""
    what = spoken_name(match.path) if match.path is not None else \
        (said.strip() or "that file")
    if match.reason == "too-big":
        return TOO_BIG_LINE.format(what=what, size=spoken_size(match.size),
                                   cap=f"{cap_mb:.0f} megabytes")
    return filepick.reason_line(match.reason, cap_mb,
                                match.size / (1024 * 1024))


# --------------------------------------------------------- recipients
# "heather at example dot com" is how an address is SAID, and Whisper
# transcribes it that way far more often than it produces the symbols.
# Both halves may be dotted: "h dot peyrovi at tamu dot edu". Reading only
# the domain's dots (which is what a simpler pattern does) turns that into
# peyrovi@tamu.edu -- a DIFFERENT, possibly real address, and the one class
# of mistake the read-back is least likely to catch, because it sounds
# almost right.
#
# Round 4 (09-05) found the same mistake one joiner over: "dot" was the
# only spoken joiner the parser knew, so "heather underscore smith at
# example dot com" matched from "smith" and the file went to
# smith@example.com -- and spoken_address() itself says "_" as
# "underscore", so Jarvis's OWN read-back of heather_smith@... said back
# to it word for word went to a stranger. Every joiner the speaker speaks
# is read here, and a local part with a word in it that is NOT read is
# never cut to its tail: address_span refuses it and unresolved_address
# hands it back as heard, for the re-ask.
_SPOKEN_JOINERS = {"dot": ".", "period": ".", "fullstop": ".",
                   "underscore": "_", "dash": "-", "hyphen": "-"}
_LOCAL_JOINER = r"(?:dot|period|full\s+stop|under\s*score|dash|hyphen)"
_DOMAIN_DOT = r"(?:dot|period|full\s+stop)"
_DOMAIN_DASH = r"(?:dash|hyphen)"
_LOCAL_LABEL = r"[A-Za-z0-9][\w+\-]*"
_DOMAIN_LABEL = r"[A-Za-z0-9][\w\-]*"
# HIS RULING (B), 2026-09-05: "example.com" said as ONE WORD is a domain.
# Whisper wrote his domain down exactly like that on 09-05 and the parser
# refused it -- which cost him more than the draft, because
# unresolved_address reads this same regex, so there was nothing to hand
# back and he got NO RE-ASK AT ALL. That refusal was deliberate once (see
# the _SAID_TLDS note above: ordinary prose has the same skeleton) and he
# has now overruled it.
#
# It is read under TWO guards, and they are what keep prose out:
#   * the dot must be TIGHT -- no space on either side. Every sentence
#     boundary has a space after the full stop ("we stopped at noon. Then
#     we left", "I'm at home. In the morning"), so no prose row in
#     tests/test_contacts.py's corpus changed by a byte.
#   * the last label must be ALPHABETIC, 2-24 characters. That is what
#     keeps "meet me at 4.30" and "at 3.5 tomorrow" from becoming
#     me@4.30 -- and it is a SHAPE, not a hand-written list of top levels,
#     because rounds 5 and 6 proved a list here is always missing one
#     (.site, .xyz, .info were all missing, and all three were drafted).
# The trailing (?!\.?\w) lets a full stop that ENDS the sentence sit
# after the domain ("dana at example.com.") without eating into it, while
# still preferring the longest real domain ("example.co.uk").
#
# WHAT RULING (B) COSTS, MEASURED (09-06, an invented 93-phrase corpus of
# ordinary non-spelling sentences, numbers only). A verb before "at" and
# a domain after it drafts a mailbox out of prose: "have a look at
# example.com" -> look@example.com. That class is NOT new -- jarvis-v3
# already drafted look@example.com from "have a look at example dot com"
# and is@example.com from "the site is at example dot com"; the tight
# dot only adds the same sentence spelt the way whisper spells it. The
# ONE sub-shape that is new is a file extension read as a top level:
# "have a look at notes.txt" -> look@notes.txt. On that corpus: 2 of 93
# drafted at HEAD, both this class; 0 of 93 at 996408d for the tight
# shape. Neither can reach the wire without a read-back
# (tests/test_send_file.py section 27), and a hand-written extension
# list here would be the always-missing-one list rounds 5 and 6 buried.
# Pinned, not hidden: tests/test_spelling_hold.py
# ::test_what_ruling_b_costs_is_measured_and_pinned.
_DOMAIN_TIGHT = (_DOMAIN_LABEL + r"(?:\." + _DOMAIN_LABEL + r")*"
                 r"\.[A-Za-z]{2,24}\b(?!\.?\w)")
_DOMAIN_SPOKEN = (_DOMAIN_LABEL + r"(?:\s+" + _DOMAIN_DASH + r"\s+" + _DOMAIN_LABEL + r")*"
                  r"(?:\s+" + _DOMAIN_DOT + r"\s+" + _DOMAIN_LABEL
                  + r"(?:\s+" + _DOMAIN_DASH + r"\s+" + _DOMAIN_LABEL + r")*)+")
# THE TIGHT DOT IN A LOCAL PART (09-06, a default taken for him): "d.a.n at
# example.com" and "j.r.smith at example.com" are read WHOLE -- d.a.n@,
# j.r.smith@ -- and read back "d dot a dot n at example dot com". Before,
# the fold flattened "d.a.n" to "dan" and _LOCAL_LABEL (no dot in it) took
# the LAST label of "j.r.smith" as the local part: dan@ and smith@, two
# silently different mailboxes, drafted, read back almost right, and sent
# on a yes. Whisper wrote his spoken "dot" as "." for the domain on 09-05,
# so it will for a local part too. Keeping the dots means his ear hears
# exactly what will be sent, whichever he meant (jarvis.spelling.fold_spans
# keeps them; this alternative reads them). The tight dot is the same
# shape _DOMAIN_TIGHT already reads on the other side of the "at".
_SPOKEN_ADDR_RX = re.compile(
    r"\b(" + _LOCAL_LABEL + r"(?:(?:\s+" + _LOCAL_JOINER + r"\s+|\.)" + _LOCAL_LABEL + r")*)"
    r"\s+at\s+"
    r"((?:" + _DOMAIN_TIGHT + r")|(?:" + _DOMAIN_SPOKEN + r"))", re.I)
_JOINER_RX = re.compile(
    r"\s+(dot|period|full\s+stop|under\s*score|dash|hyphen)\s+", re.I)
# Words that name a character an address cannot carry, or one this parser
# does not read -- and "plus": it IS a character an address can carry, but
# it is also the second-recipient connector ("send it to her, plus Dana"),
# and read as "+" it makes one address out of two people. A local part
# with any of these in it is handed back as heard, never resolved.
_SPOKEN_SYMBOLS = frozenset((
    "plus", "minus", "point", "tilde", "squiggle", "apostrophe", "slash",
    "backslash", "star", "asterisk", "hash", "hashtag", "pound", "ampersand",
    "percent", "equals", "colon", "semicolon", "comma", "space", "caret",
    "pipe", "bang", "exclamation", "quote", "quotes", "bracket", "brace",
    "paren", "parenthesis", "dollar", "sign"))
_ADDR_RX = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
# What may stand directly in front of a TYPED address: a space, a bracket,
# a quote, "mailto:", a comma. Anything else ("heather~smith@example.com")
# is a character the address cannot carry, and the match is its tail.
_ADDR_LEAD_OK = frozenset(" \t\n\r<([{\"':;,=>")


def _joined(m: "re.Match") -> str:
    word = re.sub(r"\s+", "", m.group(1).lower())
    return _SPOKEN_JOINERS.get(word, "_" if word == "underscore" else ".")


def _spoken_lead(raw: str, start: int) -> Optional[int]:
    """Where a spoken local part REALLY starts when the parser's match at
    ``start`` has ``<word> <symbol word>`` in front of it -- the start of
    the earliest such pair -- or None when the match stands on its own."""
    at = None
    pre = raw[:start]
    while True:
        m = re.search(r"(\S+)\s+(\S+)\s+$", pre)
        if not m or m.group(2).lower().strip(",.") not in _SPOKEN_SYMBOLS:
            return at
        if not re.fullmatch(_LOCAL_LABEL, m.group(1)):
            return at
        at = m.start(1)
        pre = raw[:at]


def _typed_lead(raw: str, start: int) -> Optional[int]:
    """The start of the word a typed address match at ``start`` is the
    tail of, when the character in front of it is one an address cannot
    carry; None when the match stands on its own."""
    if start == 0 or raw[start - 1] in _ADDR_LEAD_OK:
        return None
    j = start
    while j > 0 and not raw[j - 1].isspace():
        j -= 1
    return j


def address_span(text: str) -> Optional[tuple]:
    """(address, start, end) of the first address in the text, typed or
    spoken, or None. The span is in the text AS GIVEN (no whitespace
    normalising), so a caller can cut the address out of the sentence --
    which is how the commander's fold keeps its hands off one.

    A match that is only the TAIL of what he said (a joiner word the
    parser does not read in front of it, a character an address cannot
    carry) is no address at all: None, and unresolved_address says what
    was heard.

    SPELLED ALOUD (09-05). The spoken pass runs over the text with every
    spelling run folded into one word (jarvis.spelling), so "q-z-v at
    example dot com" -- and every other way whisper punctuates a man
    saying letters -- reads as "qzv at example dot com" and needs no new
    grammar here. That is not cosmetic: BEFORE the fold this parser read
    "q z v at example dot com" as the local part "v", because
    _LOCAL_LABEL matched the last single letter and _spoken_lead only
    refuses a SYMBOL word in front of it. It drafted v@example.com -- a
    different, possibly real mailbox, and the one class of mistake a
    read-back is least likely to catch, because it sounds almost right.
    The returned span is mapped back through fold_spans's index map, so
    it still cuts the raw sentence. Text with no run in it folds to
    itself and every existing caller is unchanged.

    The TYPED pass runs on the raw text first and is never folded:
    "a.b@example.com" is already an address and rewriting it to
    "ab@example.com" would be a fold inventing a mailbox.

    A RUN-TOGETHER DOMAIN (09-05, HIS RULING B). "dana at example.com" --
    whisper's own transcript of a domain he SAID -- is read now. It used
    to be refused on purpose, and the refusal cost him more than the
    draft: unresolved_address reads this same regex, so there was nothing
    to hand back either and he got no re-ask at all. The two guards that
    keep prose out are in the _DOMAIN_TIGHT comment above: the dot must be
    tight (a sentence boundary has a space after it) and the last label
    must be alphabetic (so "meet me at 4.30" is not me@4.30)."""
    raw = str(text or "")
    if not raw.strip():
        return None
    m = _ADDR_RX.search(raw)
    if m:
        if _typed_lead(raw, m.start()) is not None:
            return None
        addr = m.group(0).rstrip(".,;:")
        return addr, m.start(), m.start() + len(addr)
    folded, imap = spelling.fold_spans(raw)
    m = _SPOKEN_ADDR_RX.search(folded)
    if m:
        if _spoken_lead(folded, m.start()) is not None:
            return None
        local = _JOINER_RX.sub(_joined, " ".join(m.group(1).split())).strip()
        domain = _JOINER_RX.sub(_joined, " ".join(m.group(2).split())).strip()
        addr = f"{local}@{domain}".replace(" ", "")
        if re.fullmatch(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+", addr):
            return addr, imap[m.start()], imap[m.end() - 1] + 1
    return None


def unresolved_address(text: str) -> str:
    """The address he SAID, when it has an address's shape and a local
    part this parser cannot read whole -- "heather tilde smith at example
    dot com", "heather~smith@example.com" -- as words fit to be said back
    (never an "@" in it), or "" when there is no such thing."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return ""
    m = _ADDR_RX.search(raw)
    if m:
        lead = _typed_lead(raw, m.start())
        if lead is None:
            return ""
        return spoken_address(raw[lead:m.end()].rstrip(".,;:")).strip()
    folded = spelling.fold(raw)
    m = _SPOKEN_ADDR_RX.search(folded)
    if m:
        lead = _spoken_lead(folded, m.start())
        if lead is not None:
            return folded[lead:m.end()]
    return ""


def parse_address(text: str) -> str:
    """An email address out of spoken or written text, or ""."""
    span = address_span(" ".join(str(text or "").split()))
    return span[0] if span else ""


# ---- gender, from an EXPLICIT source only (Hunter's 19:00 ruling) -------
# A stored honorific on the person or a book row, or a pronoun he himself
# used about the person. No name-based guessing: "Heather" says nothing.
_HONORIFIC_RX = re.compile(
    r"^(?P<h>mr|mister|mrs|missus|ms|miss|madam|ma'am|sir|dame|lady|lord)\b\.?\s*",
    re.I)
_MALE_HONORIFICS = frozenset(("mr", "mister", "sir", "lord"))
_FEMALE_HONORIFICS = frozenset(("mrs", "missus", "ms", "miss", "madam", "ma'am",
                                "dame", "lady"))
_GENDER_WORDS = {"f": "f", "female": "f", "woman": "f", "she": "f", "her": "f",
                 "m": "m", "male": "m", "man": "m", "he": "m", "him": "m"}
_SHE_RX = re.compile(r"\b(?:she|her|hers|herself)\b", re.I)
_HE_RX = re.compile(r"\b(?:he|him|his|himself)\b", re.I)


def strip_honorific(text: str) -> str:
    """"Mrs Jones" -> "Jones"; "Heather" -> "Heather"."""
    t = " ".join(str(text or "").split())
    return _HONORIFIC_RX.sub("", t, count=1).strip()


def gender_from_honorific(text) -> Optional[str]:
    """"Mrs Jones" -> "f", "Mr Jones" -> "m"; "Dr Jones" and a bare name ->
    None. A stored gender VALUE ("f", "female", "she", "m", "male", "he")
    is read the same way, so a people-book field can hold either."""
    t = " ".join(str(text or "").split()).strip(" .,")
    if not t:
        return None
    hit = _GENDER_WORDS.get(t.lower())
    if hit:
        return hit
    m = _HONORIFIC_RX.match(t)
    if not m:
        return None
    h = m.group("h").lower()
    if h in _MALE_HONORIFICS:
        return "m"
    if h in _FEMALE_HONORIFICS:
        return "f"
    return None


def gender_from_pronouns(text) -> Optional[str]:
    """The gender of the ONE pronoun a sentence uses about somebody:
    "her address is ..." -> "f"; "he's at ..." -> "m"; none, or both
    ("her and his") -> None."""
    t = str(text or "")
    she, he = _SHE_RX.search(t), _HE_RX.search(t)
    if she and not he:
        return "f"
    if he and not she:
        return "m"
    return None


def _book_row(book: dict, key: str) -> Optional[tuple]:
    """(address, the row's key) for a name, looking THROUGH a stored
    honorific either way: "heather" finds the row "ms heather", and "Mrs
    Jones" finds the row "jones"."""
    low = key.lower()
    if low in book:
        return book[low], low
    bare = strip_honorific(low)
    for k, addr in book.items():
        if strip_honorific(k) == bare and bare:
            return addr, k
    return None


def names_a_real_file(cfg, file_query: str, search_roots=None,
                      now: Optional[float] = None) -> bool:
    """Does this spoken phrase point at something actually on his disk?

    Used by the commander to decide whether a sentence is a send request at
    all when the RECIPIENT did not resolve. "Send the kids to bed" and
    "email the biosensors handout to Dana" have the same grammar and name
    nobody Jarvis can write to; the difference between them, and the only
    one available, is that one of them names a real file.

    True for a refusal as well as a match (too big, empty, a folder): those
    are real things he pointed at, and they deserve the sentence that says
    so rather than silence.
    """
    said = " ".join(str(file_query or "").split())
    if not said:
        return False
    match = filephrase.resolve(
        said, roots=search_roots if search_roots is not None else roots(cfg),
        max_mb=max_mb(cfg), now=now)
    return bool(match.ok or match.ambiguous or match.path is not None)


def contacts(cfg) -> dict:
    """assistant.json ``send_file.contacts``: {"heather": "h@x.com"}."""
    raw = _cfg_get(cfg, "send_file.contacts", None)
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        addr = parse_address(value)
        if addr:
            out[str(key).strip().lower()] = addr
    return out


def resolve(cfg, memory, who: str) -> contacts_mod.Resolution:
    """What a spoken recipient comes to: found, ambiguous, or unknown.

    Four sources, most explicit first: an address he actually said; the
    ADDRESS BOOK (jarvis/contacts.py -- consulted first among the books
    because it is the only one that can say "ambiguous"); the legacy
    ``send_file.contacts`` map; then the people book Jarvis already keeps
    ("my brother" -> whatever memory.resolve_person returns). NOTHING
    infers an address from a name -- a plausible guess here is a stranger
    holding his file, and there is no undo. Two rows that answer to one
    name are a QUESTION (``candidates``), never the first row.
    """
    raw = " ".join(str(who or "").split()).strip(" .,;:?!")
    if not raw:
        return contacts_mod.Resolution()
    said = parse_address(raw)
    if said:
        return contacts_mod.Resolution(addr=said)
    key = re.sub(r"^(?:my|our|the)\s+", "", raw, flags=re.I).strip()
    try:
        res = contacts_mod.resolve(raw)
    except Exception:                                  # noqa: BLE001 - a file
        log.exception("outbox: the address book could not be read")
        res = contacts_mod.Resolution()
    if res.addr or res.candidates:
        return res
    book = contacts(cfg)
    row = _book_row(book, key) or _book_row(book, raw)
    if row:
        return contacts_mod.Resolution(addr=row[0], name=key)
    resolve_person = getattr(memory, "resolve_person", None) if memory is not None else None
    if callable(resolve_person):
        person = None
        for probe in (raw, key):
            try:
                person = resolve_person(probe)
            except Exception:                          # noqa: BLE001 - store
                log.debug("outbox: resolve_person failed", exc_info=True)
                person = None
            if isinstance(person, dict):
                break
        if isinstance(person, dict):
            addr = parse_address(person.get("email") or "")
            name = str(person.get("name") or key)
            return contacts_mod.Resolution(addr=addr, name=name)
    return contacts_mod.Resolution(name=key)


def resolve_recipient(cfg, memory, who: str) -> tuple[str, str]:
    """(address, what to call them). ("", name) when he has to be asked --
    and ("", name) for an AMBIGUOUS name too: the callers that only want
    an address get none, and ``resolve`` is there for the one that needs
    to know why."""
    res = resolve(cfg, memory, who)
    return res.addr, res.name


def said_gender(who: str) -> Optional[str]:
    """The gender of an honorific HE SAID in the name itself -- "Mrs
    Jones" -> "f". Nothing is looked up: this is only what is in the
    words. An address says nothing, and so does a bare name."""
    raw = " ".join(str(who or "").split()).strip(" .,;:?!")
    if not raw or parse_address(raw):
        return None
    key = re.sub(r"^(?:my|our|the)\s+", "", raw, flags=re.I).strip()
    return gender_from_honorific(key)


def recipient_gender(cfg, memory, who: str) -> Optional[str]:
    """The recipient's gender from an EXPLICIT source, or None: an
    honorific he said ("Mrs Jones"), an honorific on the LEGACY
    send_file.contacts key that resolved the name ("mr jones" for
    "Jones"), or the people book's own honorific / title / gender field.
    An address says nothing, and so does a bare name.

    Both lookups are keyed by the name he SAID, so this is for a
    recipient the address book did NOT resolve; ``draft_gender`` is what
    a book row goes through.
    """
    raw = " ".join(str(who or "").split()).strip(" .,;:?!")
    if not raw or parse_address(raw):
        return None
    key = re.sub(r"^(?:my|our|the)\s+", "", raw, flags=re.I).strip()
    hit = said_gender(raw)
    if hit:
        return hit
    row = _book_row(contacts(cfg), key) or _book_row(contacts(cfg), raw)
    if row:
        hit = gender_from_honorific(row[1])
        if hit:
            return hit
    resolve = getattr(memory, "resolve_person", None) if memory is not None else None
    if callable(resolve):
        for probe in (raw, key):
            try:
                person = resolve(probe)
            except Exception:                          # noqa: BLE001 - store
                person = None
            if isinstance(person, dict):
                for field_name in ("gender", "honorific", "title", "name"):
                    hit = gender_from_honorific(person.get(field_name) or "")
                    if hit:
                        return hit
                break
    return None


def draft_gender(cfg, memory, who: str,
                 res: contacts_mod.Resolution) -> Optional[str]:
    """What Draft.to_gender is filled with -- the ONE seam both lanes
    designed for (Hunter's 19:00 ruling, 09-04). The ADDRESS BOOK row's
    stored honorific comes first: a row he typed and validated, and
    "Mr" / "Mrs" / "Ms" / "Miss" / "Sir" / "Madam" say which pronoun is
    theirs ("Dr", and a row with no honorific, say nothing). Then
    recipient_gender's sources -- an honorific he said, a legacy
    send_file.contacts key, the people book. None when nobody knows, and
    then either pronoun confirms, exactly as before the ruling. The name
    itself is never read.

    For a BOOK-resolved person those lookups are shut off (w5d). They are
    keyed by the name he SAID, not by the row the book picked, so they
    answer about a DIFFERENT record: with a legacy key "mr jones" beside
    a book row "Heather Jones", saying "Jones" resolved to Heather and
    then took the Mr, and "send it to her" was refused with "Send it to
    him?" (measured). Her own row, or an honorific he said in the same
    breath, are the only things that know who she is; with neither, the
    seam stays None and either pronoun confirms.
    """
    if getattr(res, "from_book", False):
        return (gender_from_honorific(getattr(res, "honorific", "") or "")
                or said_gender(who))
    return recipient_gender(cfg, memory, who)


# ------------------------------------------------------------- config
def roots(cfg) -> list:
    """Where a spoken file may be looked for."""
    raw = _cfg_get(cfg, "send_file.roots", None)
    if not isinstance(raw, (list, tuple)) or not raw:
        raw = list(filepick.DEFAULT_ROOTS)
    return [str(r) for r in raw]


def max_mb(cfg) -> float:
    """The attachment cap, config-overridable DOWNWARDS only.

    MAX_ATTACHMENT_MB is what Gmail will actually deliver; a config that
    raised it would not send a larger file, it would only move the refusal
    to the SMTP server after the read-back had promised the file went.
    """
    raw = _cfg_get(cfg, "send_file.max_mb", None)
    try:
        want = float(str(raw))
    except (TypeError, ValueError):
        return float(MAX_ATTACHMENT_MB)
    if want <= 0:
        return float(MAX_ATTACHMENT_MB)
    return min(want, float(MAX_ATTACHMENT_MB))


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def default_subject(path) -> str:
    stem = Path(path).stem
    subject = re.sub(r"[_\-.]+", " ", stem).strip()
    return (subject or Path(path).name)[:SUBJECT_MAX]


# ------------------------------------------------------------- prepare
def prepare(cfg, memory, file_query: str, recipient: str,
            account_hint: str = "", subject: str = "",
            now: Optional[float] = None,
            search_roots: Optional[list] = None,
            chosen=None,
            resolved: Optional[contacts_mod.Resolution] = None) -> Prepared:
    """Turn the request into a Draft, or into the question to ask.

    Checked in the order a failure is cheapest to say: is there a mailbox
    at all, then the FILE (the hard half, and the half he named), then the
    recipient, then which identity to send as.

    ``chosen`` is the answer to "Which one, sir?" -- one of the paths this
    function itself offered a moment ago. It goes through the SAME resolver
    as a spoken phrase rather than round the side of it, so the containment
    check, the cap and the mtime the draft records are the ones every other
    send passed; the only thing it skips is the guessing.

    ``resolved`` is the recipient already settled: the answer to "Which
    Heather, sir?" names ONE row of the list that was read out, and the
    commander resolves it by that row's identity (contacts.Book.pick).
    Going back through ``resolve`` with the row's name would, for a
    one-word row beside "Heather Jones", ask the question again -- the
    reviewed loop. Everything after the recipient is unchanged.
    """
    accounts = mail_mod.mail_accounts(cfg)
    if not accounts:
        return Prepared(ask=mail_mod.setup_line(cfg, "gmail"),
                        status="Mail not set up")

    cap = max_mb(cfg)
    match = filephrase.resolve(str(chosen) if chosen is not None else file_query,
                               roots=search_roots if search_roots is not None
                               else roots(cfg),
                               max_mb=cap, now=now)
    if match.ambiguous:
        return Prepared(ask=offer_line(match.candidates, file_query.strip()),
                        status=f"{len(match.candidates)} files match",
                        candidates=list(match.candidates))
    if not match.ok:
        line = (WHICH_FILE_LINE if match.reason == "empty"
                else refusal_line(match, file_query, cap))
        return Prepared(ask=line, status=f"Refused: {match.reason or 'none'}")

    # Zero bytes. filepick has no opinion on this (a 0-byte file is a
    # perfectly good file to COPY), but reading back "under a kilobyte" and
    # then failing inside the SMTP call is the worst of both: he hears a
    # promise and then a class name. Refused here, before the read-back.
    if match.size <= 0:
        return Prepared(ask=EMPTY_LINE.format(what=spoken_name(match.path)),
                        status="Refused: empty")

    res = resolved if resolved is not None else resolve(cfg, memory, recipient)
    addr, who = res.addr, res.name
    if res.ambiguous:
        # Two rows in the address book answer to the name he said. A
        # question, in file order, never the first one -- and its OWN
        # status, so the commander parks it as a person question and not
        # as the file offer (whose answer grammar is the fuzzy one).
        said = " ".join(str(recipient or "").split()).strip(" .,;:?!")
        return Prepared(ask=contacts_mod.which_line(said, res.candidates),
                        status=WHICH_PERSON_STATUS,
                        candidates=list(res.candidates))
    if not addr:
        heard = unresolved_address(recipient)
        if heard:
            line = HEARD_LINE.format(heard=heard)
        else:
            line = NO_RECIPIENT_LINE.format(who=spoken_who(who)) if who else WHO_LINE
        return Prepared(ask=line, status="No address")

    account, why = mail_mod.choose_account(
        accounts, account_hint,
        default_label=str(_cfg_get(cfg, "send_file.from", "") or ""))
    if account is None:
        if why.startswith("which account:"):
            names = why.split(":", 1)[1].strip()
            line = f"Which account should I send from, sir — {names}?"
        elif why.startswith("no account called"):
            line = f"I've no {account_hint} account, sir."
        else:
            line = mail_mod.NO_ACCOUNT_LINE
        return Prepared(ask=line, status="Which account?")

    try:
        mtime = match.path.stat().st_mtime
    except OSError:
        return Prepared(ask=GONE_LINE, status="File gone")

    # An explicit path he gave outright can legitimately sit outside the
    # search roots (filephrase.DENY_ROOTS is what guards those), so the
    # draft remembers the tree it was actually resolved in.
    kept = filepick.expand_roots(search_roots if search_roots is not None
                                 else roots(cfg))
    if not any(_inside(match.path, r) for r in kept):
        kept = [match.path.parent]
    draft = Draft(path=match.path, size=match.size, mtime=mtime,
                  to_addr=addr, to_name=who, account=account,
                  subject=subject.strip() or default_subject(match.path),
                  body=str(_cfg_get(cfg, "send_file.body", "") or DEFAULT_BODY),
                  made_at=time.monotonic(), roots=kept,
                  from_book=bool(res.from_book), honorific=res.honorific,
                  to_gender=draft_gender(cfg, memory, recipient, res))
    log.info("outbox: drafted %s (%d bytes) to %s from %s", match.path.name,
             match.size, mail_mod._mask_address(addr),
             mail_mod.account_label(account))
    return Prepared(draft=draft, status=f"Confirm: {match.path.name}")


# ---------------------------------------------------------------- send
class DraftChanged(mail_mod.MailSendFailed):
    """The file moved between the read-back and the yes. Its message IS the
    line to speak, unlike a transport failure whose text is a class name."""


def send_notice(account, subject: str, body: str, *, smtp=None,
                mail=None) -> str:
    """A short note an account sends TO ITSELF; returns the Message-ID.

    WHY THIS LIVES HERE. ``tests/test_send_file.py`` pins that the only
    module in the package that calls ``mail.send_message`` is this one --
    a guard, not a coding-style rule: one door to an irreversible action,
    so that "can a tool loop reach the transport?" is a question with one
    place to look. Knightfall's rotated code needs to be mailed, and the
    first cut of it reached round the side and left that guard red
    (verdict, 2026-09-05). It comes through here instead.

    The door is NARROWER than :func:`send`'s: there is no recipient
    parameter and no attachment. Nothing that gets hold of this seam can
    use it to mail a stranger. ``mail`` is the module seam the tests
    substitute.

    WHERE IT GOES (2026-09-05). The account's own address, unless HIS
    CONFIG named another one -- a key ``mail.mail_accounts()`` mints onto
    the account dict beside the SMTP credential, spelled and specified in
    :func:`mail.notice_destination`, which is the one place in the package
    that knows its name. That is still NOT a recipient parameter: the
    address the transport is handed is a pure function of the
    account object, and the account object is a pure function of his
    config file. A caller supplies the subject and the body and nothing
    else, and neither is parsed for an address.

    NOTE THE ``mail_mod`` ON THE NEXT LINE, not ``mail``. The destination
    is resolved against the REAL module, never against the injected seam:
    resolving it through ``mail`` would let a caller that passes a stub
    module choose the recipient after all, which is the banned thing
    wearing a different hat.
    """
    mail = mail or mail_mod
    to_addr = mail_mod.notice_destination(account)
    if not to_addr:
        raise mail_mod.MailSendFailed("that account has no address")
    # MASKED, like the file lane's line: the destination is now something
    # he can configure, so "the account itself" stopped being true and a
    # log that says where a break-glass code went should not spell it out.
    log.info("outbox: notice %r to %s (from %s)", subject,
             mail_mod._mask_address(to_addr), mail_mod.account_label(account))
    return mail.send_message(account, to_addr, subject, body, smtp=smtp)


def send(draft: Draft, smtp=None, cap_mb: float = MAX_ATTACHMENT_MB) -> str:
    """Send a confirmed draft; returns the line to speak.

    The file is re-checked FIRST. Between the read-back and the yes it can
    be deleted, rewritten or replaced, and the whole value of a read-back
    is that what was described is what goes: a size or mtime that has moved
    means he confirmed a different file, so this refuses and makes him ask
    again. ``allow_outside`` is not a thing here — the path was already
    contained by filepick when the draft was made, and it is re-checked
    against the same roots, so a swap for a symlink out of the roots fails.
    """
    why = filepick.check_file(draft.path,
                              draft.roots or [draft.path.parent], cap_mb)
    if why == "not-found":
        log.warning("outbox: %s vanished before the send", draft.path.name)
        raise DraftChanged(GONE_LINE)
    if why:
        log.warning("outbox: %s no longer sendable (%s)", draft.path.name, why)
        raise DraftChanged(CHANGED_LINE)
    try:
        st = draft.path.stat()
    except OSError as exc:
        raise DraftChanged(GONE_LINE) from exc
    if st.st_size != draft.size or st.st_mtime != draft.mtime:
        log.warning("outbox: %s changed between the read-back and the yes",
                    draft.path.name)
        raise DraftChanged(CHANGED_LINE)

    # THE REHEARSAL SWITCH, applied last and over everything. It wins over
    # the transport it was handed on purpose: it can only ever turn a send
    # into no-send, never the other way round, so there is no arrangement
    # of flags in which setting it causes a message to leave. A rehearsal
    # is then said in its OWN sentence -- it must be impossible to mistake
    # for "Sent to Heather, sir."
    if mail_mod.dryrun_enabled():
        smtp = mail_mod.DryRunSMTP
    rehearsal = mail_mod.is_dryrun(smtp)

    message_id = mail_mod.send_message(draft.account, draft.to_addr,
                                       draft.subject, draft.body,
                                       attachment=draft.path, smtp=smtp)
    # THE AUDIT, written only once the send has actually returned. A
    # failure raises above this line, so the log can never claim a message
    # that did not go -- and record_sent never raises, so a log that
    # cannot be written can never turn a delivered message into a
    # reported failure.
    record_sent(draft, message_id=message_id, dry_run=rehearsal)
    if rehearsal:
        return REHEARSAL_LINE
    who = spoken_recipient(draft.to_name, draft.to_addr)
    return SENT_LINE.format(who=who)


# ----------------------------------------------------------- the audit
#: The fields of a sent-log row that are free text and could therefore
#: carry an address. Masked on the way IN, never on the way out: masking
#: at read time would still leave the raw address sitting on disk.
_MASKED_FIELDS = ("to", "to_name", "subject")


def record_sent(draft: "Draft", message_id: str = "",
                dry_run: bool = False, path: Optional[Path] = None) -> bool:
    """Append one line to sent.jsonl. Never raises.

    THIS IS WHAT REPLACES AN UNDO. There is no unsend over SMTP -- Gmail's
    is a delay its own web client implements and its submission server
    knows nothing about -- so the honest substitute is a record of what
    left, when, to whom and as which identity. Deliberately NOT recorded:
    the message body, the app password, and anything the server said.

    Every address-bearing field goes through mask_addresses FIRST. The
    branch this was carried from wrote ``to`` raw; the address-book lane
    rewrote mask_addresses tonight so that an address the parser can draft
    never reaches a log line raw, and writing one raw into a new file
    would reopen that hole rather than inherit the fix. The file name and
    path are NOT masked -- they are filesystem facts, not addresses, and
    the audit is worthless if it cannot say which file went.
    """
    line = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "path": str(draft.path),
        "name": Path(draft.path).name,
        "bytes": int(draft.size),
        "to": str(draft.to_addr),
        "to_name": str(draft.to_name or ""),
        "account": mail_mod.account_label(draft.account),
        "subject": str(draft.subject),
        "message_id": str(message_id or ""),
        "dry_run": bool(dry_run),
    }
    for field_name in _MASKED_FIELDS:
        line[field_name] = mask_addresses(line[field_name])
    target = Path(path) if path is not None else SENT_LOG
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        return True
    except OSError:
        # An audit that fails must never be the reason a send is reported
        # as failed: the message HAS gone, and saying otherwise is the
        # exact lie this lane must not tell.
        log.warning("outbox: could not write the sent log", exc_info=True)
        return False


def sent_rows(path: Optional[Path] = None, limit: int = 200) -> list:
    """The audit lines, oldest first. A bad line is skipped, not fatal."""
    target = Path(path) if path is not None else SENT_LOG
    out: list = []
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out[-limit:]


def sent_today_line(now: Optional[datetime] = None,
                    path: Optional[Path] = None) -> str:
    """"What did I email today?" -- one spoken sentence over the audit.

    Rehearsals are excluded: nothing left the machine, so listing one
    among the day's sends would be the audit telling the same lie the
    spoken line is written to avoid.

    The address is already masked on DISK, so nothing here can say one in
    full -- but masked is not spoken, and this line read the masked value
    straight for one commit: "One, sir: lab report.pdf to d…@example.com",
    measured. The recipient goes through spoken_recipient like every other
    spoken recipient, and the "@" is said as "at".
    """
    ref = now or datetime.now()
    day = ref.date().isoformat()
    rows = [r for r in sent_rows(path)
            if str(r.get("at", ""))[:10] == day and not r.get("dry_run")]
    if not rows:
        return "Nothing today, sir."
    said = []
    for row in rows:
        who = spoken_recipient(row.get("to_name"), row.get("to"))
        said.append(f"{spoken_name(str(row.get('name') or 'a file'))} to {who}")
    if len(said) == 1:
        return f"One, sir: {said[0]}."
    listed = ", ".join(said[:-1]) + " and " + said[-1]
    return f"{len(said)}, sir: {listed}."


def spoken_when(mtime: float, now: Optional[float] = None) -> str:
    """"saved yesterday" / "saved this morning" / "saved on 30 August".

    The modified date is in the read-back because it is HOW TWO VERSIONS
    OF ONE REPORT ARE TOLD APART. The tie band already turns
    lab_report.pdf and lab_report_final.pdf into a question, but the case
    it cannot help with is the same NAME saved twice -- he overwrote it
    after the lecture, and the only fact that distinguishes what he means
    from what is on disk is when it was written. Said in words, never as a
    timestamp: "modified 2026-09-03 18:42" is on the card for his eyes.
    """
    try:
        stamp = datetime.fromtimestamp(float(mtime))
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
    ref = datetime.fromtimestamp(float(now)) if now is not None \
        else datetime.now()
    days = (ref.date() - stamp.date()).days
    if days < 0:
        return "saved today"
    if days == 0:
        if (ref - stamp).total_seconds() < 3600:
            return "saved in the last hour"
        return "saved this morning" if stamp.hour < 12 else \
            ("saved this afternoon" if stamp.hour < 18
             else "saved this evening")
    if days == 1:
        return "saved yesterday"
    if days < 7:
        return f"saved on {stamp.strftime('%A')}"
    return f"saved on {stamp.day} {stamp.strftime('%B')}"


def subfolder_words(draft: "Draft") -> str:
    """", in Fall2026" when the file is NOT sitting directly in a root.

    A folder is the other thing that separates two files with the same
    name, and it is the one he can check without looking: he knows which
    folder he put it in. Empty when the parent IS a search root, because
    "in Desktop" adds nothing he did not already say.
    """
    try:
        parent = Path(draft.path).parent.resolve()
    except (OSError, RuntimeError):
        return ""
    for root in (draft.roots or ()):
        try:
            if Path(root).resolve() == parent:
                return ""
        except (OSError, RuntimeError):
            continue
    return parent.name


def clean_subject(said: str) -> str:
    """A spoken subject line, trimmed and capped, or "".

    Capped at SUBJECT_MAX because it is READ BACK VERBATIM and a subject
    long enough to lose him is a subject he stops checking. Newlines are
    removed outright: a header may not contain one, and a mis-transcribed
    line break would be a header-injection shape rather than a subject.
    """
    text = " ".join(str(said or "").split()).strip(" ,.;:!?\"'")
    return text[:SUBJECT_MAX]
