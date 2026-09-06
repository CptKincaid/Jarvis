# Oracle's half of the weekly Knightfall code

His ruling, 2026-09-06 11:00, verbatim: **"I want the nightfall code to send
weekly in the same email as the encrypted back up."**

Oracle (`opc@163.192.101.18`, host `demon-bot`) runs the Knightfall Protocol —
a dead-man's switch with nine systemd services that would send real messages
to real people. Its `knightfall-backup.timer` already mails him
"Knightfall encrypted backup" every Sunday at 08:30 UTC. This lane puts **one
extra line** in that email and changes nothing else.

**With no spool file the email is byte-for-byte what it has always been.**
That is the property everything here is built around.

## What is deployed, and what it was on 2026-09-06

| Path on Oracle | What |
|---|---|
| `app/jarvis_override.py` | **new**, this directory's copy. Stdlib only, never raises. |
| `app/backup_main.py` | **edited**, +26 lines, one line changed. |
| `app/backup_main.py.bak-before-jarvis-code-20260906` | the file as it was (CRLF, 2689 bytes). |
| `spool/` | **new**, `0700`, owned by `opc`. Empty until the Spark pushes. |

Nothing else was touched. No unit was restarted, enabled, disabled or edited;
`crypto.py`, `auth.py`, `dispatcher.py`, `canary_main.py`,
`heartbeat_monitor.py`, `liveness.py`, `db.py`, `config.py`, `links.py` and
`/etc/knightfall.env` were not opened. `knightfall-backup.timer` still shows
next `Sun 2026-09-13 08:30 GMT`. The real backup job was **never run**.

## The change to `backup_main.py`

Two hunks, both additive apart from a single `+ extra`:

```python
try:
    from app import jarvis_override
except Exception:          # the backup must not need this feature
    class jarvis_override: ...   # claim() -> ("", None); settle() -> None
```

```python
        extra, jarvis_token = jarvis_override.claim(now)
        result = sender.send_with_attachment(
            ...
                  f"Encrypted size: {len(blob)} bytes." + extra),
            filename=filename, data=blob)
        jarvis_override.settle(jarvis_token, result.ok, now)
```

The extra text lands **after** "Encrypted size: N bytes.", so the Knightfall
paragraphs are byte-identical up to that point:

```
Jarvis override code for this week: <code>
Typed only, never spoken; it replaces the previous code once Jarvis
confirms this email went out.
```

It never uses the word *key*. **Correction to the original brief:** the master
key does not ride in this email and never did — the body only names it, and
the key lives solely in `/etc/knightfall.env`. So the new line cannot be
confused with the key by position; it only has to avoid the word, and it does.

`app/` is **CRLF** throughout. Edit it preserving that: a naive
`read_text`/`write_text` normalises the file to LF and turns a three-line
change into a 61-line rewrite in every future diff. (It did, once, here;
it was restored from the backup and redone.)

## Rehearsing, which is the only way to exercise it here

`Config.from_env()` raises without `/etc/knightfall.env`, and
`KNIGHTFALL_DRY_RUN` deliberately does **not** stop backups
("Backups should go out even while the protocol is in dry-run"), so
`app.backup_main` cannot be rehearsed as `opc` at all. These need no config
and send nothing:

```bash
cd /home/opc/knightfall
.venv/bin/python -m app.jarvis_override --check      # what claim() would do
.venv/bin/python -m app.jarvis_override --dry-run    # composes; prints one line
.venv/bin/python -m app.jarvis_override ping         # is the gate alive
```

Measured on Oracle, 2026-09-06, with the spool empty:

```
--check   : no spool: the email would be exactly as today
--dry-run : extra line: would not
```

and with a fresh spool **in a temp directory** (never the real one — a spool
holding an invented code would put a dead code in his Sunday email):

```
--check   : a line WOULD be included (id ab12cd34, age 1.0 h)
--dry-run : extra line: would include
```

Neither rehearsal deletes anything, and neither ever prints the code.

## The restricted key — still his call

`jarvis-override-gate.sh` is the forced command for a **dedicated** ssh key,
so the Spark's timer need not hold `~/.ssh/oracle/oracle-key`, which is
effectively **root on the dead-man's switch** (`opc` has NOPASSWD sudo,
measured). It allows exactly `put`, `receipt`, `ping` and `revoke <8 hex>`.
Not installed; the script's own header carries the two commands. Until then
the lane falls back to the existing key, and `knightfall_weekly.key_path` in
`assistant.json` switches it over with no code change.

## Rolling back

```bash
cd /home/opc/knightfall
cp -p app/backup_main.py.bak-before-jarvis-code-20260906 app/backup_main.py
rm -f app/jarvis_override.py && rm -rf spool
```

Or just leave it: with no spool the email is unchanged, so the Oracle half is
inert on its own and can be deployed long before the Spark half.
