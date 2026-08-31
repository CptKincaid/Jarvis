"""Keyword watch: "Watch for anything about the BIOSENSORS project."

``watch.keywords`` in assistant.json is a plain list of words or phrases. Every
INTERVAL_S this thread reads the unread mail (subject + snippet) and the recent
Canvas announcements (title + snippet) and speaks one line for each new hit.

Deliberately NOT a search: the deadline watch follows due times and the grade
watch follows scores, but nothing followed a TOPIC across the two places a
student's news actually arrives. Matching is whole-word (jarvis/watchers.
keyword_hit), so "AI" does not fire on "again" and "lab" does not fire on
"collaboration"; a phrase matches as a phrase.

Hits go out through the app's announce callback, which means the proactive
speech path AND the alerts hub: quiet hours fold them into "while you were
busy, sir", presence sends them to Discord when he is out, and the hub's
repeat window dedupes.

DARK-SAFE: an empty keyword list returns 0 before either source is touched,
and each source is skipped on its own when it is not configured.
"""
from __future__ import annotations

from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools import canvas as canvas_mod
from jarvis.tools import mail as mail_mod
from jarvis.tools.canvas import CanvasError, canvas_settings
from jarvis.watchers import PollingWatcher, cfg_get, keyword_hit

log = get_logger("keyword_watch")

INTERVAL_S = 900.0             # 15 min
MAIL_SINCE_HOURS = 24
MAIL_LIMIT = 25
ANNOUNCE_DAYS = 2              # the announcement poll's own lookback
MAX_LINES = 3                  # per tick; the rest are marked seen and counted
TITLE_CHARS = 90


def _trim(text: str, limit: int = TITLE_CHARS) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) > limit:
        flat = flat[:limit].rsplit(" ", 1)[0].rstrip(",;:-") + "…"
    return flat


class KeywordWatch(PollingWatcher):
    NAME = "keyword_watch"
    INTERVAL_S = INTERVAL_S

    def __init__(self, cfg, announce: Optional[Callable] = None, state_path=None,
                 now: Callable = None, fetch_unread: Optional[Callable] = None,
                 fetch_announcements: Optional[Callable] = None,
                 fetch: Callable = None):
        super().__init__(state_path=state_path, announce=announce, now=now)
        self._cfg = cfg
        self._fetch_unread = fetch_unread or mail_mod.fetch_unread
        self._fetch_announcements = fetch_announcements or canvas_mod.fetch_announcements
        self._fetch = fetch or canvas_mod._fetch

    def keywords(self) -> tuple:
        """Read the way QuietPolicy.keywords() reads its own list: a bare
        string counts as a one-item list, blanks are dropped, lower-cased."""
        kws = cfg_get(self._cfg, "watch.keywords", [])
        if isinstance(kws, str):
            kws = [kws]
        if not isinstance(kws, (list, tuple)):
            return ()
        return tuple(" ".join(str(k).split()).lower() for k in kws if str(k).strip())

    # ----------------------------------------------------------- sources
    def _mail_hits(self, keywords) -> list[tuple[str, str, str]]:
        if not mail_mod.mail_accounts(self._cfg):
            return []                             # no mailbox: silent
        try:
            mails = list(self._fetch_unread(self._cfg, since_hours=MAIL_SINCE_HOURS,
                                            limit=MAIL_LIMIT))
        except mail_mod.MailNotConfigured:
            return []
        except Exception:                         # noqa: BLE001 - IMAP boundary
            log.debug("keyword watch: inbox unreachable", exc_info=True)
            return []
        out = []
        for m in mails:
            subject = " ".join(str(getattr(m, "subject", "") or "").split())
            kw = keyword_hit(f"{subject} {getattr(m, 'snippet', '') or ''}", keywords)
            if not kw:
                continue
            when = m.date.isoformat() if getattr(m, "date", None) else ""
            key = "mail|" + "|".join(((getattr(m, "from_addr", "") or "").lower(),
                                      subject, when))
            who = getattr(m, "sender", "") or "someone"
            line = f"Mail about {kw}, sir: {_trim(subject) or 'no subject'}, from {who}."
            out.append((key, kw, line))
        return out

    def _canvas_hits(self, keywords) -> list[tuple[str, str, str]]:
        settings = canvas_settings(self._cfg)
        if settings is None:
            return []                             # no token: silent
        try:
            items = list(self._fetch_announcements(settings, ANNOUNCE_DAYS,
                                                   self._fetch, self._now()))
        except CanvasError as exc:
            log.debug("keyword watch: Canvas unavailable (%s)", exc.kind)
            return []
        except Exception:                         # noqa: BLE001 - source boundary
            log.debug("keyword watch: Canvas fetch failed", exc_info=True)
            return []
        out = []
        for a in items:
            if not isinstance(a, dict):
                continue
            title = " ".join(str(a.get("title") or "").split())
            kw = keyword_hit(f"{title} {a.get('snippet') or ''}", keywords)
            if not kw:
                continue
            course = str(a.get("course") or "Canvas")
            posted = a.get("posted")
            key = "canvas|" + "|".join(
                (course, title, posted.isoformat() if posted is not None else ""))
            out.append((key, kw, f"Canvas announcement about {kw}, sir: "
                                 f"{course} — {_trim(title)}."))
        return out

    # ------------------------------------------------------------- tick
    def tick(self) -> int:
        keywords = self.keywords()
        if not keywords:
            return 0                              # nothing being watched
        hits = [h for h in self._mail_hits(keywords) + self._canvas_hits(keywords)
                if h[0] not in self._state]
        if not hits:
            return 0
        said = 0
        for key, _kw, line in hits:
            # Marked seen even when it is not spoken (cap reached, or the
            # sink failed): a hit read out every fifteen minutes is worse
            # than a hit missed once.
            self._mark(key)
            if said < MAX_LINES and self._speak("Keyword watch", line):
                said += 1
        extra = len(hits) - MAX_LINES
        if extra > 0:
            log.info("keyword watch: %d more hits, not spoken", extra)
        self._prune_seen()
        self._save()
        return said
