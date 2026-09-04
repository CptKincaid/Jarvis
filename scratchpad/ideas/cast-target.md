# Where a throw can land — HPCOMPUTER, measured, and how to unblock it

He asked (2026-09-02, verbatim): *"reach out and grab at the screen (in the
air) where the camera is and then gesture towards almost throwing the cast
onto the HPCOMPUTER."*

The gesture half (jarvis/gesture.py) and the payload/target half
(jarvis/cast.py) are built. This note is about the TARGET: what is true
about HPCOMPUTER right now, what Jarvis does about it, and the exact steps
that turn "held" into "landed". Every number below says whether it was
measured or guessed. Nothing here came from a camera frame.

---

## What was MEASURED about HPCOMPUTER

Design pass 2026-09-03 ~00:45 (four read-only design agents), re-checked
02:40 and 07:21 from the cast module's own probe. All
network/filesystem/arithmetic. Only ports with a named probe are listed;
an earlier draft of this note carried a 13-port list nobody had run.

| Fact | Reading | When |
|---|---|---|
| `hpcomputer.local` | resolves to **192.168.50.114** | 00:45 |
| ARP (`ip neigh`) | `60:cf:84:ad:fd:91`, went **REACHABLE** under probe; STALE-but-present at 02:40 | 00:45 / 02:40 |
| `ping -c 3 -W 1` | 3 sent, **0 received, 100% loss** | 00:45 |
| TCP 22 445 3389 5900 8008 | **no answer** on any (design-pass probe, one attempt each) | 00:45 |
| TCP 2343 (the one port it advertises via avahi, `_ni-logos._tcp`) | no answer, 3 attempts | 00:45 |
| `cast.probe_tcp("192.168.50.114", (22, 445, 3389), 0.35)` | all three **timed out** (no RST) in **0.351 s** | 02:40 |
| same probe, again | all three **timed out** in **0.351 s**; ARP STALE before, DELAY after (the kernel re-asking; not yet REACHABLE at that instant) | 07:21 |
| Tailnet (`tailscale status`, 15 s) | two nodes: `spark` 100.70.145.63, `iphone172` offline 2 d. **HPCOMPUTER is not a node.** | 00:45 |
| `tailscaled` argv | `--tun=userspace-networking --socks5-server=localhost:1055` — **no tun device**; 127.0.0.1:1055 is listening (02:40) | 00:45 / 02:40 |
| `~/.ssh/hpcomputer` | private key + `.pub` exist (00:44 today, 399 / 94 bytes). **No `~/.ssh/config` entry** for the host. Key contents not read. | 02:40 |
| Spark LAN address | `192.168.50.109/24` on `wlP9s9`, same /24 as HPCOMPUTER | 02:40 |
| `avahi-resolve -n spark-509f.local` | **172.17.0.1** — the docker0 bridge, NOT the LAN. Any URL built from the `.local` name is unreachable from HPCOMPUTER. | 00:45 |
| `phone.enabled` (webapp, port 8765) | block absent → **False**; nothing listening on 8765 (02:40, 07:21) | 02:40 / 07:21 |

**Diagnosis (measured, not guessed):** powered on, answering ARP, dropping
every IP packet including ICMP echo. Timeouts rather than RSTs on every
port is exactly what a Windows Defender Firewall DROP looks like (a host
that was off would not answer ARP; a host with the firewall off would RST
a closed port). It is a *setting*, not a dead box. TCP alone cannot
distinguish "firewalled" from "off", so the spoken refusal does not claim
either — it says "nothing on it is listening".

The earlier claim (file-and-remote commit 4f07da0) that `hpcomputer.local`
is "a stale mDNS record" is **refuted** by the REACHABLE ARP state.

---

## What Jarvis does TODAY (jarvis/cast.py, honest status per sink)

| Sink | Status | Read-back? | What he hears |
|---|---|---|---|
| `board` (his own console surface on the Spark) | **WORKS.** Fires immediately, reversible, no window opened. | No | silent if the console is on top, else "On the board, sir." |
| `handoff` (Spark SERVES a token URL on 192.168.50.109:8765; HPCOMPUTER FETCHES it) | **Built but DARK today** — `phone.enabled` is False so nothing serves. Works in principle because a Windows firewall blocks inbound only. Needs him to open a URL there, so it is a handoff, not a cast. | Yes (it leaves the box) | "It's waiting on the Spark's page, sir — the address is on the board." |
| `hpcomputer` — a FILE or the SCREEN | **HELD.** Live probe (cached 15 s, 0.35 s budget), never a constant. Payload falls back to the board and the outbox keeps it for 30 min; when the target comes back Jarvis OFFERS, never sends. | Yes, once a transport exists | "HPCOMPUTER isn't answering, sir — nothing on it is listening. I've kept it here." Repeat inside 60 s: "Still nothing listening, sir." |
| `hpcomputer` — a TRACK | **LANDS** via Spotify Connect (`transfer('HPCOMPUTER')`): its Spotify client connects OUTBOUND, which the firewall does not block; "play it on hpcomputer" already works by voice. | No | "Kashmir by Led Zeppelin, on HPCOMPUTER, sir." |

Never a fake success: `CastResult` refuses `landed and held` at
construction, and the suite asserts it for every sink and every subject.

**Direction → sink map ships EMPTY** (`gesture.sinks`), so every fling
lands on the board until he says which side HPCOMPUTER is on. Teachable by
voice once the commander wiring lands: "HPCOMPUTER is on my right" /
"right is HPCOMPUTER" / "throw left to the board" →
`cast.parse_side_teaching` + `cast.teach_sink`. Until taught, the wiring
should say `cast.UNTAUGHT_LINE` once.

---

## The seam for the SSH transport (do NOT implement blind)

`HpcomputerSink(transport=...)` takes ONE callable, `transport(subject) ->
str`, that pushes `subject.path` and returns a detail string, raising on
failure. `available()` becomes True only when a transport is wired AND the
live probe answers; a fling then PROPOSES ("The lab report to HPCOMPUTER,
sir. Shall I send it?") and the spoken yes runs it. `needs_identity=True`
by default for that path (a file leaving the box is the "scp" case the
safety lane ruled irreversible); relax with `needs_identity=False` once the
speaker-verified yes is judged enough.

The shape to plug in is `remote.push(conf, local)` on the unmerged
`file-and-remote` branch (jarvis/tools/remote.py): one configured inbox,
containment check, `run_copy` seam. Over the plain LAN IP no
ProxyCommand is needed. If it ever goes via the tailnet instead, it MUST go
through the SOCKS5 proxy on 127.0.0.1:1055 (`nc -X 5 -x 127.0.0.1:1055 %h
%p` as ProxyCommand) — there is no tun device here, and a plain
`ssh user@hpcomputer.tail…ts.net` will not route.

Nothing here has been run against the real host, by instruction. Hunter
runs the first live command himself.

---

## Ranked unblocks — the exact steps HE takes

### 1. OpenSSH Server on HPCOMPUTER (already in progress on his side — least work, a setting not a purchase)

State as of 02:40: `InstallPending` on his side (his report), key pair
already generated here at 00:44.

1. **Reboot HPCOMPUTER** to clear `InstallPending`.
2. EITHER by clicks — Settings → System → Optional features → View
   features → search "OpenSSH Server" → tick → Next → Install; then Win+R,
   `services.msc`, find "OpenSSH SSH Server", right-click → Start, and set
   Startup type to Automatic (the installer opens the firewall port itself)
   — OR in PowerShell **as Administrator**:
   ```powershell
   Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
   # expect State: Installed. If NotPresent:
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Set-Service -Name sshd -StartupType Automatic
   Start-Service sshd
   Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' | Select Name, Enabled
   # if it is missing:
   New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
     -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
   ```
3. Put the Spark's public key on HPCOMPUTER. Paste the ONE line of
   `~/.ssh/hpcomputer.pub` (from the Spark) into:
   - `C:\Users\h2pey\.ssh\authorized_keys` **if h2pey is a standard user**, or
   - `C:\ProgramData\ssh\administrators_authorized_keys` **if h2pey is an
     Administrator** (Windows ignores the per-user file for admins), then
     fix its ACL:
     ```powershell
     icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
     ```
4. On the Spark, add a host entry (he does this, not an agent):
   ```
   Host hpcomputer
       HostName 192.168.50.114
       User h2pey
       IdentityFile ~/.ssh/hpcomputer
       IdentitiesOnly yes
   ```
   then `ssh hpcomputer whoami` — expect `hpcomputer\h2pey`.
5. Tell Jarvis (or the next agent) it works. From then on `cast.probe_tcp`
   answers `(True, "", "22")` on its own and the refusal line changes by
   itself; the transport seam above is what still needs wiring.

**What this buys:** a real cast — a file pushed into one inbox folder on
HPCOMPUTER, read back and confirmed. What it does NOT buy: anything
appearing on HPCOMPUTER's *screen*. Opening the pushed file there is a
second step (a `ssh hpcomputer start <file>` line in the same transport)
and a second decision.

### 2. Turn the handoff page on (no change on Windows at all)

Set `phone.enabled: true` in `~/.config/jarvis/assistant.json` and
**restart Jarvis** (a config edit does nothing until restart). The webapp
binds 192.168.50.109 only, refuses a public address, and needs the token in
the URL. Cost: a listening socket on the home LAN — the only genuinely new
exposure in the whole design, which is why it is his call and not an
inference from "he wanted a cast".

Then a thrown file/screen goes to the board with a URL he opens on
HPCOMPUTER. The page endpoint (`/cast/<id>?t=<token>`) is a later step;
`HandoffSink.path_for(ident, token)` is what it answers from.

### 3. Tailscale client on HPCOMPUTER (valid, more moving parts)

Install the Windows client, sign in as h2peyrovi@. HPCOMPUTER becomes a
tailnet peer and the firewall question goes away on that path — but every
connection from the Spark must go through the SOCKS5 proxy on
127.0.0.1:1055 (userspace networking, no tun). Only worth it if #1 stalls.

### Named and dismissed
- **Allow ICMP + File Sharing on the private profile:** it would ping and
  give SMB, but SMB is a disk, not a screen — it does not deliver a cast.
- **A receiver agent on Windows:** most code, most to maintain, and the
  only option that needs software he does not already have.

---

## Still his to decide (nothing here can be inferred)

1. **Which side is HPCOMPUTER — left or right of his chair?** Without it
   every throw goes to the board. One sentence to Jarvis fixes it once
   the wiring lands.
2. **`phone.enabled` on?** (unblock #2 above — the one new exposure.)
3. **Board-only, or may a cast open a real window on the Spark?** Ruled
   board-only: the no-read-back rule depends on a wrong cast being cheap,
   and the 2026-08-26 freeze came from window churn on :1.
4. **When SSH lands: must the camera positively recognise him before a
   file is pushed by gesture** (`needs_identity=True`, the default), or is
   the speaker-verified spoken "yes" enough? With `camera.identity` off,
   the default means a gesture can never push a file — it will always
   refuse out loud — until he either turns identity on or relaxes the flag.
   The handoff sink defaults the OTHER way (`needs_identity=False`): it is
   read back, but nothing leaves until he fetches it himself, so identity
   is not demanded. That is a judgement, not a measurement, and it is his
   to overrule with one constructor flag.

## GUESSED numbers in cast.py (none tuned against anything real)
- `PROBE_TIMEOUT_S = 0.35`, `PROBE_CACHE_S = 15` — the probe budget and cache.
- `REFUSAL_REPEAT_S = 60` — when the refusal shortens.
- `HELD_TTL_S = 1800`, `HELD_CAPACITY = 3` — how long/many held casts are kept.
- `HANDOFF_TTL_S = 900` — how long a served URL stays live.

---

## Wired 2026-09-03 (branch gesture-cast)

The gesture, the sinks and the console are joined: `jarvis/handstage.py`
runs the hand tracker inside `PreviewPipeline.grab()` (one lens, one
consumer), `jarvis/gesturecast.py` is the courier (tones, chip, spoken
lines, the read-back through `commander.stash_destructive`, the voice API),
`jarvis/ui/carry_chip.py` is the header chip, `jarvis/board.py` grew a CAST
slab, and `jarvis/commander.py` has the voice verbs. Config: the `gesture`
block in `jarvis/assistant_config.py`, OFF by default, `sinks` EMPTY.

**HPCOMPUTER facts, corrected and as wired.** The 13-port list an earlier
note claimed was never probed. What WAS probed: 22/445/3389/5900/8008/2343
at 00:45 (no answer), 22/445/3389 at 02:40 and 07:21 (no answer). ARP
REACHABLE throughout; ping 100% loss; not on the tailnet. `HpcomputerSink`
probes 22/445/3389 live (0.35 s, cached 15 s) at throw time, never at
construction; a non-track throw is HELD and said, the payload falls back to
the board; a Spotify track lands by `control("transfer")`. The SSH transport
is `HpcomputerSink(transport=...)` with NOTHING behind it: `_make_gesture`
in jarvis/app.py passes `transfer=` only. When `ssh hpcomputer whoami`
answers, the transport is the next change, and its first live run is
Hunter's.

**Which side.** Unknown, and the default is the safe one: `gesture.sinks`
ships `{}`, every throw lands on the board, and `UNTAUGHT_LINE` is said
once per session. "HPCOMPUTER is on my right" writes `{"right":
"hpcomputer"}`.

**Identity.** `app._eye_identity` reads `services.camera_feed.eye.state`
when a feed is attached; nothing attaches one on this tree, so it answers
"" (no opinion) today. `HpcomputerSink` keeps `needs_identity=True`, the
handoff `False` -- both his to overrule (decision 4 above).
