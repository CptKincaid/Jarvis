"""Gmail summaries over IMAP (spec section 6.6).

Read-only by construction: ``SELECT INBOX`` readonly, ``BODY.PEEK`` for
every fetch, so nothing is ever marked read. Stdlib only (``imaplib``,
``email``). The IMAP class is an argument (``imap=imaplib.IMAP4_SSL``) so
tests drive it with a fake returning canned RFC 822 bytes.

Never logs the password or message bodies — only counts and the host.
"""
from __future__ import annotations

import email
import email.utils
import html as _html
import imaplib
import re
import socket
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from typing import Optional

from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.mail")

DEFAULT_IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
IMAP_TIMEOUT = 15.0
BODY_BYTES = 2000                      # partial body fetch (never the whole mail)
SNIPPET_CHARS = 200
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
            "host": str(host).strip()}


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
        })
    return out


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


def _local_tz():
    return datetime.now().astimezone().tzinfo


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
    return dt.astimezone(_local_tz())


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
                 timeout: float = IMAP_TIMEOUT) -> list[Mail]:
    """Unread INBOX mail newer than ``since_hours``, newest first.
    Raises MailNotConfigured; IMAP/socket errors propagate."""
    accounts = mail_accounts(cfg)
    if not accounts:
        raise MailNotConfigured("gmail address or app password not set")
    now = now or datetime.now().astimezone()
    since = now - timedelta(hours=int(since_hours))
    limit = max(1, int(limit))

    if len(accounts) > 1:
        # One mailbox with a stale app password must not blind Jarvis to the
        # rest, so failures are collected and only re-raised if EVERY mailbox
        # failed -- silence there would look identical to an empty inbox.
        merged: list[Mail] = []
        failures = []
        for account in accounts:
            try:
                merged.extend(_fetch_one(account, since, limit, imap, timeout))
            except Exception as exc:                     # noqa: BLE001
                failures.append(exc)
                log.warning("mail: %s failed: %s", account["label"], exc)
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
        typ, data = conn.search(None, "UNSEEN", f"SINCE {imap_date(since)}")
        if typ != "OK":
            raise imaplib.IMAP4.error(f"search failed: {typ}")
        ids = (data[0] or b"").split() if data else []
        log.info("mail: %d unseen since %s", len(ids), imap_date(since))
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


def _fetch_one(settings: dict, since: datetime, limit: int,
               imap, timeout: float) -> list[Mail]:
    """One mailbox. Same conversation as the single-account path, with each
    Mail tagged so a merged briefing can say which inbox it came from."""
    log.info("mail: connecting to %s for %s (%s)", settings["host"],
             _mask_address(settings["address"]), settings["label"])
    conn = imap(settings["host"], IMAP_PORT, timeout=timeout)
    try:
        conn.login(settings["address"], settings["password"])
        conn.select("INBOX", readonly=True)
        typ, data = conn.search(None, "UNSEEN", f"SINCE {imap_date(since)}")
        if typ != "OK":
            raise imaplib.IMAP4.error(f"search failed: {typ}")
        ids = (data[0] or b"").split() if data else []
        log.info("mail: %s has %d unseen since %s", settings["label"],
                 len(ids), imap_date(since))
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


def fact_sheet(mails: list[Mail], total: int, since_hours: int = 24,
               now: Optional[datetime] = None) -> str:
    """'5 unread since yesterday: 1) Jane Doe — Invoice 4471 due Friday
    (2:10 pm): snippet …' — plain text for the model, one item per line."""
    since = "yesterday" if since_hours <= 24 else f"{since_hours // 24} days"
    if total > len(mails):
        head = f"{total} unread since {since}, latest {len(mails)}:"
    else:
        head = f"{total} unread since {since}:"
    lines = [head]
    for i, m in enumerate(mails, 1):
        when = when_text(m.date, now)
        item = f"{i}) {m.sender} — {m.subject}"
        if when:
            item += f" ({when})"
        if m.snippet:
            item += f": {m.snippet[:120]}"
        lines.append(item)
    return "\n".join(lines)


# ---------------------------------------------------------------- tool
_IMAP_ERRORS = (imaplib.IMAP4.error, OSError, socket.timeout, ssl.SSLError,
                EOFError, ConnectionError)


def make_tools(cfg, services) -> list[ToolSpec]:
    imap_cls = getattr(services, "imap", None) if services is not None else None
    imap_cls = imap_cls or imaplib.IMAP4_SSL

    def get_mail(limit=5, since_hours=24, **_) -> ToolResult:
        try:
            limit = max(1, min(20, int(float(str(limit)))))
        except (TypeError, ValueError):
            limit = 5
        try:
            since_hours = max(1, min(24 * 14, int(float(str(since_hours)))))
        except (TypeError, ValueError):
            since_hours = 24
        if gmail_settings(cfg) is None:
            line = setup_line(cfg, "gmail")
            return ToolResult(text=line, ok=False, speak=line)
        try:
            mails = fetch_unread(cfg, since_hours=since_hours, limit=20,
                                 imap=imap_cls)
        except MailNotConfigured:
            line = setup_line(cfg, "gmail")
            return ToolResult(text=line, ok=False, speak=line)
        except _IMAP_ERRORS as exc:
            log.warning("mail: unreachable (%s)", type(exc).__name__)
            return ToolResult(text="mailbox unreachable: IMAP login or "
                                   "connection failed", ok=False,
                              speak=UNREACHABLE_LINE)
        if not mails:
            return ToolResult(text=f"no unread mail in the last {since_hours} hours",
                              speak=NOTHING_NEW_LINE)
        sheet = fact_sheet(mails[:limit], len(mails), since_hours)
        return ToolResult(text=sheet, max_sentences=4)

    spec = ToolSpec(
        name="get_mail",
        description="Unread Gmail from the last day: sender, subject, snippet.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer",
                          "description": "how many to report (default 5)"},
            },
        },
        handler=get_mail,
    )
    return [spec]
