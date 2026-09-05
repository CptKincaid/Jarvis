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

from jarvis import contacts as contacts_mod
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
UNSURE_LINE = ("I'd rather be certain, sir — say yes and I'll send it, "
               "or no and I'll let it go.")
CHANGED_LINE = ("That file has changed since I read it back, sir; "
                "nothing was sent. Ask me again and I'll take another look.")
GONE_LINE = "That file has gone, sir; nothing was sent."
EMPTY_LINE = "{what} is empty, sir; there'd be nothing to attach."
NO_RECIPIENT_LINE = "I've no address for {who}, sir. What is it?"
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
# may be joined the way spoken_address() reads one out (dot, underscore,
# dash); the domain is one or more labels and an alphabetic top level, said
# ("dot", "period") or punctuated (". ", "."). Ordinary prose has the same
# skeleton ("look at the dot on the map", "at 4 dot 30"), so a match is
# then CHECKED: a domain that opens on a function word, or ends on a
# number, is not an address; and a punctuated one has to end on a top
# level actually in use, because "at home. See you" is not one either.
_SAID_JOIN = r"(?:\s+(?:dot|period|underscore|dash)\s+|\.(?!\s))"
_SAID_SEP = r"(?:\s+(?:dot|period)\s+|\.\s*)"
_SAID_ADDR_RX = re.compile(
    r"(?<![\w@.\-])"
    r"(?P<local>[A-Za-z0-9][\w+\-]*(?:" + _SAID_JOIN + r"[A-Za-z0-9][\w+\-]*)*)"
    r"(?P<rest>\s+at\s+"
    r"(?P<domain>[A-Za-z0-9][\w\-]*(?:" + _SAID_SEP + r"[A-Za-z0-9][\w\-]*)*"
    + _SAID_SEP + r"(?P<tld>[A-Za-z]{2,24})))"
    r"(?![\w\-])", re.I)
_SAID_SEP_RX = re.compile(_SAID_SEP, re.I)
_AT_HINT_RX = re.compile(r"\bat\b", re.I)
_NOT_A_DOMAIN_WORD = frozenset("""
    the a an this that these those my your his her our their its it
    on in of to for from and or but is was are were be been so as at by
    up out if then than there here what which who whom when where why how
    not no yes now just also very too all any some each every both few
    more most other such only own same do does did done can could will
    would shall should may might must have has had am we you they he she
    him them i
""".split())
# ("me" and "us" are not in that list: "dana at me dot com" is an address
# -- me.com is a mail domain -- and no sentence says "at me dot".)
# Top levels a PUNCTUATED domain may end on ("gmail. com", "example.edu").
# English words that are also top levels (in, me, us, it, is, be, no, to,
# at, so, info) are left out on purpose: after a full stop they are the
# next sentence far more often than an address. Said with the word "dot"
# they are still masked -- "dana at example dot in" is an address.
_SAID_TLDS = frozenset("""
    com org net edu gov mil int io co ai dev app biz tv uk ca au de fr
    nl es ie ch eu nz jp cn br mx ru se fi dk pl cz pt gr tr za kr
""".split())


def _one_said(m) -> str:
    domain = m.group("domain")
    labels = [x for x in _SAID_SEP_RX.split(domain) if x]
    first, tld = labels[0].lower(), m.group("tld").lower()
    if first in _NOT_A_DOMAIN_WORD:
        return m.group(0)
    if "." in domain and tld not in _SAID_TLDS:
        return m.group(0)
    return m.group("local")[:1] + "…" + m.group("rest")


def mask_addresses(text) -> str:
    """"yes, to hjones@example.com" -> "yes, to h…@example.com", and
    "send it to dana at example dot com" -> "send it to d… at example dot
    com": every address-shaped run in a sentence, typed or SAID, masked
    the way mail and this module already mask a single address, for a log
    line that carries what he said. The rest of the sentence is kept -- it
    is the line's whole point -- the domain stays as he put it, and a
    trailing full stop stays outside the mask."""
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
        text = _SAID_ADDR_RX.sub(_one_said, text)
    return text


def account_words(account: dict) -> str:
    """"your school account"."""
    label = mail_mod.account_label(account)
    return f"your {label} account" if label else "your account"


def _to_words(draft: Draft) -> str:
    who = draft.to_name or ""
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
_SPOKEN_ADDR_RX = re.compile(
    r"\b([A-Za-z0-9][\w+\-]*(?:\s+dot\s+[A-Za-z0-9][\w+\-]*)*)"
    r"\s+at\s+"
    r"([A-Za-z0-9][\w\-]*(?:\s+dot\s+[A-Za-z0-9][\w\-]*)+)", re.I)
_DOT_RX = re.compile(r"\s+dot\s+", re.I)
_ADDR_RX = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")


def parse_address(text: str) -> str:
    """An email address out of spoken or written text, or ""."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return ""
    m = _ADDR_RX.search(raw)
    if m:
        return m.group(0).strip(".,;:")
    m = _SPOKEN_ADDR_RX.search(raw)
    if m:
        local = _DOT_RX.sub(".", m.group(1)).strip()
        domain = _DOT_RX.sub(".", m.group(2)).strip()
        addr = f"{local}@{domain}".replace(" ", "")
        if re.fullmatch(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+", addr):
            return addr
    return ""


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
    hit = book.get(key.lower()) or book.get(raw.lower())
    if hit:
        return contacts_mod.Resolution(addr=hit, name=key)
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
        line = NO_RECIPIENT_LINE.format(who=who) if who else WHO_LINE
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
                  from_book=bool(res.from_book), honorific=res.honorific)
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
