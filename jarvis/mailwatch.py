"""Important-person mail heads-up: "Mail from Dr. Villalobos, sir — re:
thesis draft."

Everything this needs already existed and only ever ran on demand:
``mail.fetch_unread`` reads the inbox, the people book (jarvis/memory.py)
stores "my advisor" -> name + address, and ``mail.sender_matches`` decides
whether a message is from them. So an advisor's mail sat unseen until he
thought to ask. This thread closes that: every INTERVAL_S it reads the
unread mail once and speaks a canned line for anything from someone in the
people book.

ONLY people-book senders announce. That is the whole point of the filter --
a poller that read out every unread message would be a nuisance within an
hour, and the people book is the one list he has already curated by voice
("remember that my advisor is Dr. X, email …").

The line is built here, never by the model: it must still work while the GPU
is lent to a trainer, and a heads-up is not worth a model turn.

DARK-SAFE twice over: no configured mailbox, or an empty people book, and
tick() returns 0 without opening a socket.
"""
from __future__ import annotations

from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools import mail as mail_mod
from jarvis.watchers import PollingWatcher, cfg_get

log = get_logger("mailwatch")

INTERVAL_S = 600.0             # 10 min
SINCE_HOURS = 24               # an unread mail older than this is not news
FETCH_LIMIT = 25
MAX_LINES = 3                  # the rest are marked seen and counted, not read
SUBJECT_CHARS = 90


def _subject_words(subject: str) -> str:
    text = " ".join(str(subject or "").split())
    if not text:
        return ""
    if len(text) > SUBJECT_CHARS:
        text = text[:SUBJECT_CHARS].rsplit(" ", 1)[0].rstrip(",;:-") + "…"
    return text


class PeopleMailHeadsUp(PollingWatcher):
    NAME = "mailwatch"
    INTERVAL_S = INTERVAL_S

    def __init__(self, cfg, people: Optional[Callable] = None,
                 announce: Optional[Callable] = None, state_path=None,
                 now: Callable = None, fetch_unread: Optional[Callable] = None):
        super().__init__(state_path=state_path, announce=announce, now=now)
        self._cfg = cfg
        # services.memory.people; a callable so a people book edited by voice
        # is picked up on the next tick without a restart.
        self._people = people
        self._fetch_unread = fetch_unread or mail_mod.fetch_unread

    @property
    def enabled(self) -> bool:
        return bool(cfg_get(self._cfg, "watch.people_mail", True))

    # ---------------------------------------------------------- contacts
    def _contacts(self) -> list[tuple[str, str]]:
        """[(what sender_matches should match on, how to name them)] from the
        people book. An address is preferred; an entry with only a name still
        counts, matched loosely the way "mail from my advisor" already is."""
        book = self._people() if callable(self._people) else self._people
        out = []
        for entry in (book or {}).values():
            if not isinstance(entry, dict):
                continue
            name = " ".join(str(entry.get("name") or "").split())
            email = str(entry.get("email") or "").strip()
            wanted = email or name
            if wanted:
                out.append((wanted, name or email))
        return out

    # ------------------------------------------------------------ source
    def _unread(self) -> list:
        if not mail_mod.mail_accounts(self._cfg):
            return []                             # not set up: silent
        try:
            return list(self._fetch_unread(self._cfg, since_hours=SINCE_HOURS,
                                           limit=FETCH_LIMIT))
        except mail_mod.MailNotConfigured:
            return []
        except Exception:                         # noqa: BLE001 - IMAP boundary
            log.debug("mailwatch: inbox unreachable", exc_info=True)
            return []

    @staticmethod
    def _key(m) -> str:
        """A stable id for one message. mail.Mail carries no Message-ID, so
        the mailbox, sender, subject and timestamp stand in -- the same four
        fields a person would use to say "that one again"."""
        when = m.date.isoformat() if getattr(m, "date", None) else ""
        return "|".join((getattr(m, "account", "") or "",
                         (getattr(m, "from_addr", "") or "").lower(),
                         " ".join(str(getattr(m, "subject", "") or "").split()),
                         when))

    # ------------------------------------------------------------- tick
    def tick(self) -> int:
        if not self.enabled:
            return 0
        contacts = self._contacts()
        if not contacts:
            return 0                              # nobody to watch for
        mails = self._unread()
        if not mails:
            return 0
        hits = []
        for m in mails:
            who = next((name for wanted, name in contacts
                        if mail_mod.sender_matches(m, wanted)), "")
            if not who:
                continue
            key = self._key(m)
            if key in self._state:
                continue
            hits.append((key, who, m))
        if not hits:
            return 0
        said = 0
        for key, who, m in hits:
            # Marked seen whether or not it was spoken: a message held back
            # by the cap (or lost to a Discord outage) must not come back
            # every ten minutes for the rest of the day.
            self._mark(key)
            if said >= MAX_LINES:
                continue
            subject = _subject_words(getattr(m, "subject", ""))
            line = f"Mail from {who}, sir" + (f" — re: {subject}." if subject else ".")
            if self._speak("Mail", line):
                said += 1
        extra = len(hits) - MAX_LINES
        if extra > 0 and self._speak(
                "Mail", f"And {extra} more from your contacts, sir."):
            said += 1
        self._prune_seen()
        self._save()
        return said
