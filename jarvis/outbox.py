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

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from jarvis import filephrase
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
# The cap refusal is the one place this lane does NOT reuse
# filepick.reason_line: filepick's wording ("past the N I'll put on the wire
# without you saying so plainly") offers an override, and for mail there is
# none to offer — Gmail will not carry it however plainly he says so.
TOO_BIG_LINE = ("{what} is {size}, sir — Gmail won't carry more than "
                "{cap} as an attachment, so I've not sent it.")


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
    # THE ONE SEAM for the recipient's gender (Hunter's 19:00 ruling,
    # 09-04): "f" / "m" / None. None by default -- and None means either
    # pronoun confirms, exactly as before the ruling. Filled by whoever
    # KNOWS, from an explicit source only: a stored honorific on the person
    # or on a book row (prepare -> recipient_gender), or a pronoun Hunter
    # himself used about the person earlier in the same draft conversation
    # ("her address is dana at ..." -> commander's address answer). Never
    # from the name: nothing in this file guesses a gender from "Heather".
    # The address-book branch fills it from its rows later without touching
    # the confirmation grammar, which reads only this field.
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


def spoken_who(who: str) -> str:
    """A recipient as he said it, fit to be said back: an address, or the
    half of one ("heather@example", "heather@"), is spoken; a name --
    "Mary-Jane" included -- is left alone. Attack 2 (round 4): the
    no-address line echoed a typed half-address with its "@" in it."""
    w = " ".join(str(who or "").split())
    return spoken_address(w).strip() if "@" in w else w


def pronoun_for(gender: Optional[str]) -> str:
    """"her" / "him" for a known gender, "them" for none."""
    return {"f": "her", "m": "him"}.get(str(gender or "").lower(), "them")


def mask_addresses(text: str) -> str:
    """The sentence with every address in it masked, typed or spoken, for
    a log line. "yes, to dana@example.com" -> "yes, to d…@example.com";
    "dana at example dot com" -> "… at example dot com". The address-book
    review (09-04) found the commander's INFO lines carrying a typed
    address whole; the log is read by more eyes than the mailbox is."""
    t = str(text or "")
    if not t:
        return t
    t = _ADDR_RX.sub(lambda m: mail_mod._mask_address(m.group(0).rstrip(".,;:"))
                     + m.group(0)[len(m.group(0).rstrip(".,;:")):], t)
    t = _SPOKEN_ADDR_RX.sub(lambda m: "… at " + m.group(2), t)
    return t


def account_words(account: dict) -> str:
    """"your school account"."""
    label = mail_mod.account_label(account)
    return f"your {label} account" if label else "your account"


def _to_words(draft: Draft) -> str:
    who = draft.to_name or ""
    addr = spoken_address(draft.to_addr)
    return f"{who}, at {addr}" if who else addr


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
    who = draft.to_name or spoken_address(draft.to_addr)
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
_SPOKEN_ADDR_RX = re.compile(
    r"\b(" + _LOCAL_LABEL + r"(?:\s+" + _LOCAL_JOINER + r"\s+" + _LOCAL_LABEL + r")*)"
    r"\s+at\s+"
    r"(" + _DOMAIN_LABEL + r"(?:\s+" + _DOMAIN_DASH + r"\s+" + _DOMAIN_LABEL + r")*"
    r"(?:\s+" + _DOMAIN_DOT + r"\s+" + _DOMAIN_LABEL
    + r"(?:\s+" + _DOMAIN_DASH + r"\s+" + _DOMAIN_LABEL + r")*)+)", re.I)
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
    was heard."""
    raw = str(text or "")
    if not raw.strip():
        return None
    m = _ADDR_RX.search(raw)
    if m:
        if _typed_lead(raw, m.start()) is not None:
            return None
        addr = m.group(0).rstrip(".,;:")
        return addr, m.start(), m.start() + len(addr)
    m = _SPOKEN_ADDR_RX.search(raw)
    if m:
        if _spoken_lead(raw, m.start()) is not None:
            return None
        local = _JOINER_RX.sub(_joined, " ".join(m.group(1).split())).strip()
        domain = _JOINER_RX.sub(_joined, " ".join(m.group(2).split())).strip()
        addr = f"{local}@{domain}".replace(" ", "")
        if re.fullmatch(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+", addr):
            return addr, m.start(), m.end()
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
    m = _SPOKEN_ADDR_RX.search(raw)
    if m:
        lead = _spoken_lead(raw, m.start())
        if lead is not None:
            return raw[lead:m.end()]
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


def resolve_recipient(cfg, memory, who: str) -> tuple[str, str]:
    """(address, what to call them). ("", name) when he has to be asked.

    Three sources, most explicit first: an address he actually said, the
    ``send_file.contacts`` map, then the people book Jarvis already keeps
    ("my brother" -> whatever memory.resolve_person returns). NOTHING
    infers an address from a name — a plausible guess here is a stranger
    holding his file, and there is no undo.
    """
    raw = " ".join(str(who or "").split()).strip(" .,;:?!")
    if not raw:
        return "", ""
    said = parse_address(raw)
    if said:
        return said, ""
    key = re.sub(r"^(?:my|our|the)\s+", "", raw, flags=re.I).strip()
    book = contacts(cfg)
    row = _book_row(book, key) or _book_row(book, raw)
    if row:
        return row[0], key
    resolve = getattr(memory, "resolve_person", None) if memory is not None else None
    if callable(resolve):
        person = None
        for probe in (raw, key):
            try:
                person = resolve(probe)
            except Exception:                          # noqa: BLE001 - store
                log.debug("outbox: resolve_person failed", exc_info=True)
                person = None
            if isinstance(person, dict):
                break
        if isinstance(person, dict):
            addr = parse_address(person.get("email") or "")
            name = str(person.get("name") or key)
            return (addr, name) if addr else ("", name)
    return "", key


def recipient_gender(cfg, memory, who: str) -> Optional[str]:
    """The recipient's gender from an EXPLICIT source, or None: an
    honorific he said ("Mrs Jones"), an honorific on the book row that
    resolved the name ("mr jones" for "Jones"), or the people book's own
    honorific / title / gender field. An address says nothing, and so does
    a bare name."""
    raw = " ".join(str(who or "").split()).strip(" .,;:?!")
    if not raw or parse_address(raw):
        return None
    key = re.sub(r"^(?:my|our|the)\s+", "", raw, flags=re.I).strip()
    hit = gender_from_honorific(key)
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
            chosen=None) -> Prepared:
    """Turn the request into a Draft, or into the question to ask.

    Checked in the order a failure is cheapest to say: is there a mailbox
    at all, then the FILE (the hard half, and the half he named), then the
    recipient, then which identity to send as.

    ``chosen`` is the answer to "Which one, sir?" -- one of the paths this
    function itself offered a moment ago. It goes through the SAME resolver
    as a spoken phrase rather than round the side of it, so the containment
    check, the cap and the mtime the draft records are the ones every other
    send passed; the only thing it skips is the guessing.
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

    addr, who = resolve_recipient(cfg, memory, recipient)
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
                  to_gender=recipient_gender(cfg, memory, recipient))
    log.info("outbox: drafted %s (%d bytes) to %s from %s", match.path.name,
             match.size, mail_mod._mask_address(addr),
             mail_mod.account_label(account))
    return Prepared(draft=draft, status=f"Confirm: {match.path.name}")


# ---------------------------------------------------------------- send
class DraftChanged(mail_mod.MailSendFailed):
    """The file moved between the read-back and the yes. Its message IS the
    line to speak, unlike a transport failure whose text is a class name."""


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

    mail_mod.send_message(draft.account, draft.to_addr, draft.subject,
                          draft.body, attachment=draft.path, smtp=smtp)
    who = draft.to_name or spoken_address(draft.to_addr)
    return SENT_LINE.format(who=who)
