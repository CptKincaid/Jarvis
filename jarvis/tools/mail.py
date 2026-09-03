"""Gmail over IMAP (reading, spec 6.6) and SMTP (sending, 2026-09-02).

The READ side is read-only by construction: ``SELECT INBOX`` readonly,
``BODY.PEEK`` for every fetch, so nothing is ever marked read. Stdlib only
(``imaplib``, ``email``). The IMAP class is an argument
(``imap=imaplib.IMAP4_SSL``) so tests drive it with a fake returning canned
RFC 822 bytes.

The SEND side (``send_message``, at the bottom) is the one thing in this
module that cannot be undone, and it is reachable only from
jarvis/outbox.py behind a spoken read-back — never from a ToolSpec, so the
model can never decide to send anything. Its transport is an argument too
(``smtp=smtplib.SMTP_SSL``).

Never logs the password or message bodies — only counts, hosts and masked
addresses.
"""
from __future__ import annotations

import email
import email.utils
import html as _html
import imaplib
import mimetypes
import re
import smtplib
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Optional

from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.mail")

DEFAULT_IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
IMAP_TIMEOUT = 15.0
BODY_BYTES = 2000                      # partial body fetch (never the whole mail)
# One thread per mailbox, capped: Hunter runs three (personal, work,
# school) and the cap only exists so a config that grows to a dozen does
# not open a dozen sockets at once.
MAX_MAIL_WORKERS = 4
SNIPPET_CHARS = 200
SHEET_SNIPPET_CHARS = 120    # what a browse-the-inbox listing shows per item
# A search BY NAME is not inbox triage. LIVE 2026-08-31 21:07: "Any emails
# from Evolving AI Insights?" was answered "I'm afraid there are no emails
# from Evolving AI Insights, sir" -- the newsletter had arrived at 5:47 am
# that morning and had simply been READ, so the UNSEEN + 24 h default hid
# it. A named search looks a week back and counts read mail.
NAMED_SEARCH_HOURS = 24 * 7
NOTHING_NEW_LINE = "Nothing new in the inbox, sir."
UNREACHABLE_LINE = "I can't reach your mailbox, sir."
FALLBACK_SETUP_LINE = ("I'll need your Gmail app password set up, sir; "
                       "the notes are in docs/assistant-setup.md.")
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Template values the setup doc / W1 leave behind: empty, "<your address>",
# "your-app-password", "changeme", "xxxx…", "placeholder".
_PLACEHOLDER = re.compile(r"^\s*$|^<.*>$|placeholder|^(your|my)[-_ ]|^x{3,}$|changeme",
                          re.I)


class MailNotConfigured(RuntimeError):
    """gmail.address / gmail.app_password missing or placeholders."""


@dataclass
class Mail:
    from_name: str
    from_addr: str
    subject: str
    date: Optional[datetime]           # aware, local tz when parsable
    snippet: str
    account: str = ""                  # which mailbox; "" when only one

    @property
    def sender(self) -> str:
        return self.from_name or self.from_addr or "unknown sender"


# ------------------------------------------------------------- config
def _cfg_get(cfg, dotted: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        cur = cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return default if cur is None else cur
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            val = get(dotted, default)
            return default if val is None else val
        except Exception:
            log.debug("cfg.get(%s) failed", dotted, exc_info=True)
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _is_placeholder(value) -> bool:
    return not isinstance(value, str) or bool(_PLACEHOLDER.search(value))


def gmail_settings(cfg) -> Optional[dict]:
    """(address, password, host) or None when not configured."""
    is_conf = getattr(cfg, "is_configured", None)
    if callable(is_conf):
        try:
            if not is_conf("gmail"):
                return None
        except Exception:
            log.debug("cfg.is_configured failed", exc_info=True)
    address = _cfg_get(cfg, "gmail.address", "")
    password = _cfg_get(cfg, "gmail.app_password", "")
    if _is_placeholder(address) or _is_placeholder(password):
        return None
    host = _cfg_get(cfg, "gmail.imap_host", "") or DEFAULT_IMAP_HOST
    return {"address": address.strip(), "password": password,
            "host": str(host).strip(),
            # Carried even on the read path so a caller that later sends
            # from this account does not have to re-read the config; blank
            # is fine, smtp_host() derives one from the IMAP host.
            "smtp_host": str(_cfg_get(cfg, "gmail.smtp_host", "") or "").strip(),
            "from_name": str(_cfg_get(cfg, "gmail.from_name", "") or "").strip()}


def mail_accounts(cfg) -> list[dict]:
    """Every configured mailbox, in config order.

    `gmail.accounts` is a list of {label, address, app_password, imap_host}.
    When it is absent or empty the legacy top-level gmail.address /
    gmail.app_password pair is used instead, so existing configs keep working
    untouched. Incomplete or placeholder entries are skipped rather than
    raising: one unfinished mailbox must not take the others down with it.
    """
    raw = _cfg_get(cfg, "gmail.accounts", None)
    default_host = _cfg_get(cfg, "gmail.imap_host", "") or DEFAULT_IMAP_HOST
    default_smtp = str(_cfg_get(cfg, "gmail.smtp_host", "") or "").strip()
    if not isinstance(raw, (list, tuple)) or not raw:
        single = gmail_settings(cfg)
        if single is None:
            return []
        single.setdefault("label", single["address"].partition("@")[0])
        return [single]

    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        address = str(entry.get("address") or "")
        password = entry.get("app_password") or ""
        if _is_placeholder(address) or _is_placeholder(password):
            continue
        address = address.strip()
        out.append({
            "label": str(entry.get("label") or "").strip()
            or address.partition("@")[0],
            "address": address,
            "password": password,
            "host": str(entry.get("imap_host") or default_host).strip(),
            # Per-account submission host and display name, for the SEND
            # path. Blank smtp_host is derived from the IMAP host below.
            "smtp_host": str(entry.get("smtp_host") or default_smtp).strip(),
            "from_name": str(entry.get("from_name") or "").strip(),
        })
    return out


def _truthy_flag(value, default: bool = True) -> bool:
    """Tool arguments arrive as whatever the model emitted -- bool, "false",
    "no", 0. Anything unrecognised keeps the safer default (unread only)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("false", "0", "no", "off", "all"):
        return False
    if text in ("true", "1", "yes", "on", "unread"):
        return True
    return default


def setup_line(cfg, section: str = "gmail") -> str:
    fn = getattr(cfg, "setup_line", None)
    if callable(fn):
        try:
            line = fn(section)
            if line:
                return line
        except Exception:
            log.debug("cfg.setup_line failed", exc_info=True)
    return FALLBACK_SETUP_LINE


# --------------------------------------------------------------- fetch
def imap_date(dt: datetime) -> str:
    """IMAP date-text, always English month names (26-Aug-2026)."""
    return f"{dt.day:02d}-{_MONTHS[dt.month - 1]}-{dt.year}"


def _parse_fetch(data) -> list[tuple[bytes, bytes]]:
    """imaplib FETCH payload -> [(header_bytes, body_bytes)] per message.

    Shape: [(b'1 (BODY[HEADER.FIELDS (...)] {n}', b'...'),
            (b' BODY[TEXT]<0> {m}', b'...'), b')', ...]"""
    out: list[tuple[bytes, bytes]] = []
    headers = body = None
    for item in data or ():
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        label = item[0] if isinstance(item[0], bytes) else bytes(str(item[0]), "ascii")
        payload = item[1] if isinstance(item[1], bytes) else bytes(str(item[1]), "utf-8")
        if re.match(rb"^\s*\d+\s*\(", label) and \
                (headers is not None or body is not None):
            out.append((headers or b"", body or b""))
            headers = body = None
        up = label.upper()
        if b"HEADER" in up:
            headers = payload
        elif b"TEXT" in up or b"BODY[" in up:
            body = payload
    if headers is not None or body is not None:
        out.append((headers or b"", body or b""))
    return out


def _parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Bare astimezone(), never astimezone(datetime.now().astimezone().tzinfo):
    # that tzinfo is a FIXED-offset snapshot of today's offset, so a message
    # sent 2026-08-26 14:10 CDT read back in November came out 13:10 "CST".
    # This converts an instant, so the platform's rules for THAT instant win.
    return dt.astimezone()


_TAG = re.compile(r"<[^>]+>")
_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)


def _html_to_text(raw: str) -> str:
    raw = _STYLE.sub(" ", raw)
    raw = re.sub(r"<br\s*/?>|</p>|</div>|</tr>|</li>", "\n", raw, flags=re.I)
    return _html.unescape(_TAG.sub(" ", raw))


def _text_part(headers: bytes, body: bytes) -> str:
    """Reassemble headers + (truncated) body and pull the first text/plain
    part, else text/html stripped of tags."""
    hdr = headers.rstrip(b"\r\n")
    msg = email.message_from_bytes(hdr + b"\r\n\r\n" + body, policy=policy.default)
    plain = html = ""
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
        except Exception:
            text = str(part.get_payload() or "")
        if ctype == "text/plain" and not plain:
            plain = text
        elif ctype == "text/html" and not html:
            html = text
        if plain:
            break
    if plain.strip():
        return plain
    if html.strip():
        return _html_to_text(html)
    if not msg.is_multipart() and not msg.get_content_type().startswith("text/"):
        return ""
    return plain or ""


_QUOTE_LINE = re.compile(r"^\s*>")
_WROTE_LINE = re.compile(r"^\s*On .{3,120}\bwrote:\s*$|^\s*-{2,}\s*(Original|Forwarded) message\s*-{2,}",
                         re.I)
_SIG_LINE = re.compile(r"^\s*(--\s*$|__+\s*$|Sent from my |Get Outlook for |"
                       r"Best regards,?\s*$|Kind regards,?\s*$|Regards,?\s*$|"
                       r"Thanks,?\s*$|Cheers,?\s*$)", re.I)


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """First ``limit`` chars of the readable body: quoted lines, the
    'On … wrote:' line and everything after a signature marker dropped."""
    kept: list[str] = []
    for line in (text or "").splitlines():
        if _WROTE_LINE.match(line) or _SIG_LINE.match(line):
            break
        if _QUOTE_LINE.match(line):
            continue
        kept.append(line)
    flat = " ".join(" ".join(kept).split())
    if len(flat) > limit:
        cut = flat[:limit].rsplit(" ", 1)[0]
        flat = cut.rstrip(",;:") + "…"
    return flat


def _decode_str(value) -> str:
    return " ".join(str(value or "").split())


def _parse_message(headers: bytes, body: bytes) -> Mail:
    hdr = email.message_from_bytes(headers.rstrip(b"\r\n") + b"\r\n\r\n",
                                   policy=policy.default)
    try:
        from_hdr = _decode_str(hdr.get("From", ""))
    except Exception:
        from_hdr = ""
    name, addr = email.utils.parseaddr(from_hdr)
    try:
        subject = _decode_str(hdr.get("Subject", "")) or "(no subject)"
    except Exception:
        subject = "(no subject)"
    date = _parse_date(str(hdr.get("Date", "") or ""))
    try:
        text = _text_part(headers, body)
    except Exception:
        log.debug("body parse failed", exc_info=True)
        text = ""
    return Mail(from_name=_decode_str(name).strip('"'), from_addr=addr,
                subject=subject, date=date, snippet=make_snippet(text))


def fetch_unread(cfg, since_hours: int = 24, limit: int = 20,
                 imap=imaplib.IMAP4_SSL, now: Optional[datetime] = None,
                 timeout: float = IMAP_TIMEOUT,
                 unread_only: bool = True) -> list[Mail]:
    """INBOX mail newer than ``since_hours``, newest first.

    ``unread_only=False`` drops the UNSEEN filter, which is what answers
    "what was my last email about?" -- previously unanswerable, because the
    only query this module could make was UNSEEN and a read message was
    therefore invisible no matter how recent.

    Raises MailNotConfigured; IMAP/socket errors propagate."""
    accounts = mail_accounts(cfg)
    if not accounts:
        raise MailNotConfigured("gmail address or app password not set")
    now = now or datetime.now().astimezone()
    since = now - timedelta(hours=int(since_hours))
    limit = max(1, int(limit))

    if len(accounts) > 1:
        merged, failures = _fetch_every(accounts, since, limit, imap,
                                        timeout, unread_only)
        # One mailbox with a stale app password must not blind Jarvis to the
        # rest, so failures are collected and only re-raised if EVERY mailbox
        # failed -- silence there would look identical to an empty inbox.
        if failures and len(failures) == len(accounts):
            raise failures[0]
        merged.sort(key=lambda m: m.date or since, reverse=True)
        return merged[:limit]

    settings = accounts[0]
    log.info("mail: connecting to %s for %s", settings["host"],
             _mask_address(settings["address"]))
    conn = imap(settings["host"], IMAP_PORT, timeout=timeout)
    try:
        conn.login(settings["address"], settings["password"])
        conn.select("INBOX", readonly=True)
        criteria = (["UNSEEN"] if unread_only else []) + \
            [f"SINCE {imap_date(since)}"]
        typ, data = conn.search(None, *criteria)
        if typ != "OK":
            raise imaplib.IMAP4.error(f"search failed: {typ}")
        ids = (data[0] or b"").split() if data else []
        log.info("mail: %d %s since %s", len(ids), "unseen" if unread_only else "messages", imap_date(since))
        if not ids:
            return []
        ids = ids[-limit:]
        typ, data = conn.fetch(
            b",".join(ids),
            "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE CONTENT-TYPE "
            "CONTENT-TRANSFER-ENCODING)] "
            f"BODY.PEEK[TEXT]<0.{BODY_BYTES}>)")
        if typ != "OK":
            raise imaplib.IMAP4.error(f"fetch failed: {typ}")
    finally:
        for closer in (conn.logout,):
            try:
                closer()
            except Exception:
                log.debug("imap logout failed", exc_info=True)
    mails = [_parse_message(h, b) for h, b in _parse_fetch(data)]
    fresh = [m for m in mails if m.date is None or m.date >= since]
    fresh.sort(key=lambda m: m.date or since, reverse=True)
    return fresh


def _fetch_every(accounts: list[dict], since: datetime, limit: int,
                 imap, timeout: float, unread_only: bool
                 ) -> tuple[list[Mail], list[Exception]]:
    """Every mailbox at once -> (merged mail, failures).

    LIVE 2026-08-31 14:35, "what's on my calendar and what's on my latest
    email?": the three accounts were opened one after another (14:35:40.0 ->
    14:35:43.5 just to reach them), get_mail took 8.1 s, and brain's whole
    CHAT_WALL_BUDGET_S was gone before the model could put the result into
    words -- Jarvis answered with TOOL_ONLY_LINE. The connections are
    independent, so the wall cost is now the SLOWEST mailbox, not their sum.

    Two properties the sequential loop had, kept deliberately:

    * per-account failure isolation -- a future that raises loses only its
      own rows, and the exception is returned so the caller can tell "every
      mailbox failed" (which must raise) from "one did" (which must not);
    * deterministic ORDER -- results are collected per account in config
      order, never in completion order, so the spoken answer does not
      change depending on which IMAP server happened to answer first.
    """
    per_account: list[list[Mail]] = [[] for _ in accounts]
    failures: list[Exception] = []
    workers = max(1, min(len(accounts), MAX_MAIL_WORKERS))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="mail") as pool:
        futures = [pool.submit(_fetch_one, account, since, limit, imap,
                               timeout, unread_only) for account in accounts]
        for i, (account, future) in enumerate(zip(accounts, futures)):
            try:
                per_account[i] = future.result()
            except Exception as exc:                     # noqa: BLE001
                failures.append(exc)
                log.warning("mail: %s failed: %s", account["label"], exc)
    return [m for chunk in per_account for m in chunk], failures


def _fetch_one(settings: dict, since: datetime, limit: int,
               imap, timeout: float, unread_only: bool = True) -> list[Mail]:
    """One mailbox. Same conversation as the single-account path, with each
    Mail tagged so a merged briefing can say which inbox it came from."""
    log.info("mail: connecting to %s for %s (%s)", settings["host"],
             _mask_address(settings["address"]), settings["label"])
    conn = imap(settings["host"], IMAP_PORT, timeout=timeout)
    try:
        conn.login(settings["address"], settings["password"])
        conn.select("INBOX", readonly=True)
        criteria = (["UNSEEN"] if unread_only else []) + \
            [f"SINCE {imap_date(since)}"]
        typ, data = conn.search(None, *criteria)
        if typ != "OK":
            raise imaplib.IMAP4.error(f"search failed: {typ}")
        ids = (data[0] or b"").split() if data else []
        log.info("mail: %s has %d %s since %s", settings["label"],
                 len(ids), "unseen" if unread_only else "messages", imap_date(since))
        if not ids:
            return []
        ids = ids[-limit:]
        typ, data = conn.fetch(
            b",".join(ids),
            "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE CONTENT-TYPE "
            "CONTENT-TRANSFER-ENCODING)] "
            f"BODY.PEEK[TEXT]<0.{BODY_BYTES}>)")
        if typ != "OK":
            raise imaplib.IMAP4.error(f"fetch failed: {typ}")
    finally:
        try:
            conn.logout()
        except Exception:
            log.debug("imap logout failed", exc_info=True)
    mails = [_parse_message(h, b) for h, b in _parse_fetch(data)]
    for m in mails:
        m.account = settings["label"]
    return [m for m in mails if m.date is None or m.date >= since]


def _mask_address(addr: str) -> str:
    if "@" not in addr:
        return "…"
    user, _, dom = addr.partition("@")
    return f"{user[:1]}…@{dom}"


# ----------------------------------------------------------- wording
def when_text(dt: Optional[datetime], now: Optional[datetime] = None) -> str:
    if dt is None:
        return ""
    now = now or datetime.now().astimezone()
    clock = dt.strftime("%I:%M %p").lstrip("0").lower()
    if dt.date() == now.date():
        return clock
    if dt.date() == (now - timedelta(days=1)).date():
        return f"yesterday {clock}"
    return f"{dt.strftime('%a')} {clock}"


def window_words(since_hours: int) -> str:
    """"24 hours" / "7 days" -- what the model repeats back. A named search
    looks 168 hours back and "in the last 168 hours" is not English."""
    hours = int(since_hours)
    if hours % 24 == 0 and hours >= 48:
        return f"{hours // 24} days"
    return f"{hours} hours"


def fact_sheet(mails: list[Mail], total: int, since_hours: int = 24,
               now: Optional[datetime] = None, unread: bool = True,
               snippet_chars: int = SHEET_SNIPPET_CHARS) -> str:
    """'5 unread since yesterday: 1) Jane Doe — Invoice 4471 due Friday
    (2:10 pm): snippet …' — plain text for the model, one item per line.

    ``unread=False`` when the search included read mail: the model renders
    this sheet as fact, so calling twenty read messages "unread" had Jarvis
    reporting a full inbox of new mail the user had already seen."""
    # "since yesterday" is English; "since 7 days" is not, and the model
    # repeats this head back as fact -- a named search defaults to a week now,
    # so the long window is no longer the rare case.
    since = ("since yesterday" if since_hours <= 24
             else f"in the last {window_words(since_hours)}")
    kind = "unread" if unread else "messages"
    if total > len(mails):
        head = f"{total} {kind} {since}, latest {len(mails)}:"
    else:
        head = f"{total} {kind} {since}:"
    lines = [head]
    for i, m in enumerate(mails, 1):
        when = when_text(m.date, now)
        item = f"{i}) {m.sender} — {m.subject}"
        if when:
            item += f" ({when})"
        if m.snippet:
            # A narrowed search (a sender or a subject) passes the FULL
            # snippet: "What specifically was the undergrad engineering
            # update about?" cannot be answered from 120 characters, and
            # Jarvis answered "I only have the subject line and sender"
            # about a body that had already been fetched.
            item += f": {m.snippet[:snippet_chars]}"
        lines.append(item)
    return "\n".join(lines)


# ------------------------------------------------------------ sending
# Everything above this line is READ-ONLY by construction (readonly SELECT,
# BODY.PEEK). Everything below it leaves the machine and cannot be taken
# back, so it is deliberately kept off the model's path: there is NO
# ToolSpec for send_mail and there never should be one. The only caller is
# jarvis/outbox.py, behind a spoken read-back and an explicit yes
# (commander._try_send_confirm). A tool the model can call is a tool the
# model can call by mistake, and "sorry, I sent your transcript to the
# wrong Heather" has no undo.
SMTP_HOST_DEFAULT = "smtp.gmail.com"
SMTP_PORT = 465                    # implicit TLS; never 25, never STARTTLS
SMTP_TIMEOUT = 30.0                # an attachment is slower than a header fetch
SEND_FAILED_LINE = "I couldn't send that, sir."
NO_ACCOUNT_LINE = "I'm not sure which account to send from, sir."


class MailSendFailed(RuntimeError):
    """SMTP refused, or the attachment could not be read."""


def smtp_host(account: dict) -> str:
    """The submission host for an account.

    An explicit ``smtp_host`` wins; otherwise the IMAP host is rewritten
    (imap.gmail.com -> smtp.gmail.com), which is right for Gmail and for
    every provider that follows the same naming. A host that fits neither
    falls back to Gmail's, because these are Gmail app passwords.
    """
    explicit = str((account or {}).get("smtp_host") or "").strip()
    if explicit:
        return explicit
    host = str((account or {}).get("host") or "").strip()
    if host.startswith("imap."):
        return "smtp." + host[len("imap."):]
    return SMTP_HOST_DEFAULT


def account_label(account: dict) -> str:
    return str((account or {}).get("label") or "").strip() or \
        str((account or {}).get("address") or "").partition("@")[0]


def account_by_label(accounts: list[dict], hint: str) -> Optional[dict]:
    """The account a spoken hint names, or None.

    Matches the label ("work"), the address, or the local part -- exactly,
    then as a prefix. Never fuzzily: sending from the wrong identity is one
    of the two irreversible halves of this feature, so an unrecognised hint
    must produce a question, not a near miss.
    """
    want = " ".join(str(hint or "").split()).lower()
    if not want or not accounts:
        return None
    for account in accounts:
        label = account_label(account).lower()
        addr = str(account.get("address") or "").lower()
        if want in (label, addr, addr.partition("@")[0]):
            return account
    for account in accounts:
        label = account_label(account).lower()
        if label and (label.startswith(want) or want.startswith(label)):
            return account
    return None


def choose_account(accounts: list[dict], hint: str = "",
                   default_label: str = "") -> tuple[Optional[dict], str]:
    """(account, why-there-is-none).

    Order: what he SAID, then what he CONFIGURED, then -- only when there
    is no choice to get wrong -- the single account. With three identities
    configured and nothing said, this returns None and the caller asks;
    guessing from the recipient's domain was considered and rejected,
    because "it looked like a university address" is not a reason to put
    his school identity on a message he meant to send as himself.
    """
    if not accounts:
        return None, "no mailbox is configured"
    if hint:
        picked = account_by_label(accounts, hint)
        if picked is not None:
            return picked, ""
        return None, f"no account called {hint}"
    if default_label:
        picked = account_by_label(accounts, default_label)
        if picked is not None:
            return picked, ""
    if len(accounts) == 1:
        return accounts[0], ""
    names = ", ".join(account_label(a) for a in accounts)
    return None, f"which account: {names}"


def _attachment_parts(path) -> tuple[bytes, str, str, str]:
    """(bytes, maintype, subtype, filename). Raises MailSendFailed."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise MailSendFailed(f"cannot read {p.name}") from exc
    if not data:
        raise MailSendFailed(f"{p.name} is empty")
    ctype, _ = mimetypes.guess_type(p.name)
    maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
    return data, maintype, subtype or "octet-stream", p.name


def build_message(from_addr: str, to_addr: str, subject: str, body: str,
                  attachment=None, from_name: str = "") -> EmailMessage:
    """The RFC 822 message, attachment included. No network."""
    msg = EmailMessage()
    msg["From"] = formataddr((from_name, from_addr)) if from_name else from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject or "(no subject)"
    msg["Date"] = email.utils.formatdate(localtime=True)
    # A real Message-ID, from the SENDING domain: Gmail supplies one when a
    # message has none, but the value is what the caller logs and reports,
    # so it is generated here where the address is known.
    msg["Message-ID"] = email.utils.make_msgid(
        domain=from_addr.partition("@")[2] or None)
    msg.set_content(body or "")
    if attachment is not None:
        data, maintype, subtype, filename = _attachment_parts(attachment)
        msg.add_attachment(data, maintype=maintype, subtype=subtype,
                           filename=filename)
    return msg


def send_message(account: dict, to_addr: str, subject: str, body: str,
                 attachment=None, smtp=None, timeout: float = SMTP_TIMEOUT,
                 host: str = "", port: int = SMTP_PORT) -> str:
    """Send one message. Returns its Message-ID.

    ``smtp`` is the class, exactly as ``imap`` is on the read side, so the
    tests drive a fake and NOTHING in the suite can reach a mail server
    (tests/conftest.py refuses the submission ports as well, belt and
    braces). Raises MailSendFailed for anything that goes wrong; the
    password is never logged, and neither is the body.
    """
    if not account:
        raise MailSendFailed("no account")
    to_addr = str(to_addr or "").strip()
    if "@" not in to_addr:
        raise MailSendFailed(f"not an address: {to_addr!r}")
    smtp = smtp or smtplib.SMTP_SSL
    host = host or smtp_host(account)
    msg = build_message(account["address"], to_addr, subject, body,
                        attachment=attachment,
                        from_name=str(account.get("from_name") or ""))
    label = account_label(account)
    log.info("mail: sending to %s from %s (%s), attachment %s",
             _mask_address(to_addr), _mask_address(account["address"]), label,
             Path(attachment).name if attachment is not None else "none")
    try:
        conn = smtp(host, port, timeout=timeout)
    except Exception as exc:                       # noqa: BLE001 - transport
        raise MailSendFailed(f"cannot reach {host}") from exc
    try:
        conn.login(account["address"], account["password"])
        conn.send_message(msg)
    except Exception as exc:                       # noqa: BLE001 - transport
        # str(exc) on an SMTPAuthenticationError carries the server's reply,
        # never the credential, so this is safe to log -- but the exception
        # is re-raised with the TYPE only, because it travels into a spoken
        # line and a server that echoes the username would put it there.
        log.warning("mail: send failed (%s)", type(exc).__name__)
        raise MailSendFailed(type(exc).__name__) from exc
    finally:
        for closer in (getattr(conn, "quit", None),):
            if callable(closer):
                try:
                    closer()
                except Exception:
                    log.debug("smtp quit failed", exc_info=True)
    log.info("mail: sent %s", msg["Message-ID"])
    return str(msg["Message-ID"])


# ---------------------------------------------------------------- tool
_IMAP_ERRORS = (imaplib.IMAP4.error, OSError, socket.timeout, ssl.SSLError,
                EOFError, ConnectionError)


SENDER_FETCH_LIMIT = 60      # a sender filter over only the latest 20 is thin


def sender_matches(mail: Mail, wanted: str) -> bool:
    """Case-insensitive match on the address or the display name. An
    address in ``wanted`` (from the people book) must match the address;
    a bare name matches either field as a substring."""
    w = (wanted or "").strip().lower()
    if not w:
        return True
    addr = (mail.from_addr or "").lower()
    name = (mail.from_name or "").lower()
    if "@" in w:
        return addr == w or addr.endswith("<" + w + ">") or w in addr
    return w in name or w in addr


# Filler in "what was the undergrad engineering update about" that must
# not become a required keyword.
_SUBJECT_STOP = {"the", "a", "an", "my", "about", "email", "e-mail", "mail",
                 "message", "from", "of", "on", "for", "was", "is", "what",
                 "whats", "what's", "specifically", "sir", "jarvis", "that",
                 "this", "it", "and", "to", "in", "one", "last", "latest"}
_SUBJECT_WORD_RX = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def subject_matches(mail: Mail, wanted: str) -> bool:
    """Every content word of ``wanted`` somewhere in the subject or the
    snippet. Substring, not whole-word: "undergrad" has to find
    "Undergraduate Advising", which is how Hunter refers to that sender."""
    words = [w for w in _SUBJECT_WORD_RX.findall((wanted or "").lower())
             if w not in _SUBJECT_STOP and len(w) > 2]
    if not words:
        return True
    hay = f"{mail.subject} {mail.snippet}".lower()
    return all(w in hay or w.rstrip("s") in hay for w in words)


def resolve_sender(services, sender: str) -> tuple[str, str]:
    """-> (what to match on, how to name it). "my advisor" goes through the
    people book (jarvis.memory) to the stored address, else the name; an
    unknown alias matches literally."""
    raw = (sender or "").strip()
    if not raw:
        return "", ""
    memory = getattr(services, "memory", None) if services is not None else None
    resolve = getattr(memory, "resolve_person", None)
    if callable(resolve):
        try:
            person = resolve(raw)
        except Exception:                            # noqa: BLE001
            log.debug("resolve_person failed", exc_info=True)
            person = None
        if person:
            who = person.get("name") or raw
            return (person.get("email") or who), who
    return raw, raw


def make_tools(cfg, services) -> list[ToolSpec]:
    imap_cls = getattr(services, "imap", None) if services is not None else None
    imap_cls = imap_cls or imaplib.IMAP4_SSL

    def get_mail(limit=5, since_hours=None, unread_only=None, sender="",
                 subject="", **_) -> ToolResult:
        try:
            limit = max(1, min(20, int(float(str(limit)))))
        except (TypeError, ValueError):
            limit = 5
        wanted, who = resolve_sender(services, str(sender or ""))
        about = " ".join(str(subject or "").split())
        # A NAMED search -- "any emails from X", "what was the Y about" --
        # is a lookup, not a look at what is new, so read mail counts and a
        # day is too short a memory. Only the DEFAULTS move: an explicit
        # unread_only / since_hours from the model still wins.
        named = bool(wanted or about)
        if since_hours is None:
            since_hours = NAMED_SEARCH_HOURS if named else 24
        try:
            since_hours = max(1, min(24 * 14, int(float(str(since_hours)))))
        except (TypeError, ValueError):
            since_hours = NAMED_SEARCH_HOURS if named else 24
        # mail_accounts(), NOT gmail_settings(): the latter only knows the
        # LEGACY top-level gmail.address / gmail.app_password pair, so on a
        # multi-account config (gmail.accounts) it returns None and Jarvis
        # asked for credentials it already had -- while fetch_unread below
        # would have read all three mailboxes perfectly well.
        if not mail_accounts(cfg):
            line = setup_line(cfg, "gmail")
            return ToolResult(text=line, ok=False, speak=line)
        unread = (not named) if unread_only is None else _truthy_flag(unread_only)
        try:
            mails = fetch_unread(cfg, since_hours=since_hours,
                                 limit=SENDER_FETCH_LIMIT if named else 20,
                                 imap=imap_cls, unread_only=unread)
        except MailNotConfigured:
            line = setup_line(cfg, "gmail")
            return ToolResult(text=line, ok=False, speak=line)
        except _IMAP_ERRORS as exc:
            log.warning("mail: unreachable (%s)", type(exc).__name__)
            return ToolResult(text="mailbox unreachable: IMAP login or "
                                   "connection failed", ok=False,
                              speak=UNREACHABLE_LINE)
        if named:
            if wanted:
                mails = [m for m in mails if sender_matches(m, wanted)]
            if about:
                mails = [m for m in mails if subject_matches(m, about)]
            if not mails:
                # The model says it from the fact: "nothing from Dr Peyrovi
                # this week, sir" -- NOTHING_NEW_LINE would claim an empty
                # inbox.
                kind = "unread mail" if unread else "mail"
                scope = f"from {who}" if who else f"about {about}"
                if who and about:
                    scope = f"from {who} about {about}"
                return ToolResult(text=f"no {kind} {scope} in the last "
                                       f"{window_words(since_hours)}",
                                  max_sentences=2)
        if not mails:
            if unread:
                return ToolResult(text="no unread mail in the last "
                                       f"{window_words(since_hours)}",
                                  speak=NOTHING_NEW_LINE)
            # "Nothing new in the inbox" would be the wrong claim here: this
            # search included read mail, so the inbox is simply empty for the
            # period. Let the model say that from the fact.
            return ToolResult(text="no mail at all, read or unread, in the last "
                                   f"{window_words(since_hours)}", max_sentences=2)
        sheet = fact_sheet(mails[:limit], len(mails), since_hours, unread=unread,
                           snippet_chars=SNIPPET_CHARS if named
                           else SHEET_SNIPPET_CHARS)
        if who:
            sheet = f"From {who}: " + sheet
        return ToolResult(text=sheet, max_sentences=4)

    spec = ToolSpec(
        name="get_mail",
        # <= 20 words: this rides in every prompt (test_get_mail_tool).
        # The per-parameter descriptions carry the detail.
        description=("Gmail: sender, subject, body snippet. Unread from the "
                     "last day, or search by sender or subject."),
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer",
                          "description": "how many to report (default 5)"},
                "since_hours": {
                    "type": "integer",
                    "description": ("how far back to look, hours "
                                    "(default 24, max 336)")},
                "unread_only": {
                    "type": "boolean",
                    "description": ("true = unread only (the default for a "
                                    "bare 'any mail?'); false = include "
                                    "already-read mail. A sender or subject "
                                    "search already includes read mail")},
                "sender": {
                    "type": "string",
                    "description": ("only mail from this person: a name, an "
                                    "address, or how Hunter refers to them "
                                    "('my advisor')")},
                "subject": {
                    "type": "string",
                    "description": ("keywords the message is about; use it "
                                    "to fetch the body of a message already "
                                    "mentioned ('undergrad engineering')")},
            },
        },
        handler=get_mail,
    )
    return [spec]
