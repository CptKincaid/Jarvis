# Running idea backlog

Ideas Hunter has raised, kept in **his words first**, with only what is already
known from the code or the machine added underneath. Nothing here is designed,
vetted or scheduled until he says so — this is a place to put things down so
they stop living in a conversation.

The 2026-08-30 backlog (53 ideas → 34 vetted → all built) is the precedent, and
its standing constraint still applies unless he says otherwise: **fully local,
no new cloud dependencies.**

---

## 2026-09-02 — five new ideas

### 1. "is there a way to connect my google drives files?"
Read (and maybe write) his Google Drive from Jarvis.

*What is already known:* this is the first idea in the set that genuinely needs
a cloud account — there is no local substitute for files that live in Drive.
Note the shape of the calendar answer earlier today: we assumed Google OAuth
was required, measured, and found iCloud already writable, so OAuth was avoided.
**Drive has no such escape hatch.** If he wants it, it is a standing OAuth grant
in a file on the box that runs agent code — the same credential objection raised
against Google Calendar, and worth making deliberately rather than by drift.
Read-only scope would be a materially smaller grant than read-write.

### 2. "email this file to this person from this location on my desktop"
Send an email with an attachment, naming the file by where it sits.

*What is already known:* `jarvis/tools/mail.py` reads IMAP across three accounts
(personal / work / school) and **send with read-back was already on the agreed
queue** — this extends it to attachments plus a spoken file path. The hard part
is not SMTP, it is resolving "that file on my desktop" from speech to one exact
path, and confirming it out loud before anything leaves the machine. Sending a
file is irreversible, so the read-back matters more here than for plain text.

### 3. "connection to my HPCOMPUTER desktop?"
Reach his other machine from Jarvis.

*What is already known:* HPCOMPUTER is **already a known entity in the code** —
`jarvis/mixer.py:36,41,704` handles it as a Spotify Connect target and ducks its
audio to 30%. So Jarvis already talks to it for one purpose. Also relevant:
`tailscaled.service` is loaded and running as a userspace node agent, which is a
ready-made private path between his machines with no port forwarding. Worth
establishing what he actually wants — files, a remote shell, screen control, or
just "put this on that screen" — because those are four different builds.

### 4. "encrypting my wifi or searching or online communication"
Three separate things bundled in one sentence; each needs its own answer.

*What is already known:* this is the vaguest of the five and the one most likely
to mean something different to him than to me. Candidates: WPA3 on the router,
DNS-over-HTTPS or a resolver change, a VPN (tailscale is already here), or
encrypted messaging. **Ask before designing.** Note his existing posture is
already strong locally — `jarvis/webapp.py` refuses to bind a non-private
address and keeps its token in SECRET_KEYS.

### 5. "building our own way to text instead of twilio SMS service"
Self-hosted messaging rather than a paid SMS gateway.

*What is already known:* **this revisits an earlier decision.** On 2026-08-31 he
ruled "email send with read-back, **NO texts/calls**". That ruling should be
explicitly re-opened rather than quietly overwritten. Options split sharply:
carrier SMS essentially requires a gateway (Twilio, or an Android phone acting
as one); everything else — Matrix, XMPP, Signal, ntfy/Gotify push — is not SMS
and only reaches people who install something. Worth deciding **who he needs to
reach**, because that answer eliminates most of the options immediately.

---

## How to use this list

Nothing here is committed work. When he wants one moved forward, the pattern
that has worked is a design pass grounded in the actual code and hardware
(`scratchpad/ideas/PLAN.md` is the example), producing a costed build order and
the decisions only he can make — not an implementation.

---

## 2026-09-02, later — his clarifications

He answered every open question in the same breath. Recording them here because
two of the five changed shape entirely.

### 1. Google Drive — **read, download, and send files to people**
Not "sync my Drive". He wants to PULL files out and SEND them onward.

That is a materially smaller ask than it looked. **Read-only Drive scope plus
the existing mail path covers it** — no write scope, no standing permission to
modify or delete anything in Drive. It also merges with idea 2: "email this file
to this person" and "send this Drive file to this person" are the same feature
with two different file sources. Build the file-resolution and read-back layer
once and point it at both.

### 3. HPCOMPUTER — **all four: files, remote shell, screen control, cast-to-screen**
Not a subset. So this is a real remote-machine integration, not a one-off.

*What is already true:* `tailscaled.service` is loaded and running as a
userspace node agent, which already gives an authenticated private path between
his machines with no port forwarding and no open ports on the LAN. That is the
transport for all four. On top of it: SFTP/rsync for files, SSH for the shell,
and for screen the two halves are different — CONTROL wants VNC/RDP, while
"put this on that screen" is a one-way cast and is much cheaper. Worth splitting
those two in any design, because he may want casting far more often than control.

### 4. Encryption — **all four**, i.e. WPA3, DNS-over-HTTPS, VPN, encrypted messaging
Ambiguity resolved: he meant the lot.

Note how little of this is Jarvis work. WPA3 is a router setting. DNS-over-HTTPS
is a system resolver change. **The VPN already exists** — tailscale is running.
So three of the four are configuration he could have this week, and only the
fourth is a build. That fourth one is the same thing as idea 5, so they should
be designed together rather than twice.

### 5. Self-hosted texting — **the ruling was about COST AND CLOUD, not about texting**
His words: *"i know i made that ruling because it was paid and not local."*

That is the whole reframe. The 2026-08-31 "NO texts/calls" decision was not a
judgement that messaging is undesirable — it was a rejection of Twilio as a paid
cloud dependency. **A local, free path is not covered by that ruling.**

Which changes what to evaluate:
- **An old Android phone as an SMS gateway** deserves first look. It sends REAL
  SMS from his own number on his existing plan, costs nothing per message, runs
  on his LAN, and needs no third-party account. It is the only option that is
  simultaneously real SMS, free, and local.
- **Matrix (self-hosted)** covers idea 4's encrypted-messaging half at the same
  time — end-to-end encrypted, self-hosted, no per-message cost. But it only
  reaches people who install a client.
- Signal / XMPP / ntfy sit between those and should be judged on the same axis.

**The question that still decides it: who does he need to reach?** People who
will install an app, or anyone with a phone number. That answer eliminates most
of this list immediately, and he has not answered it yet.

### 5, narrowed further — 2026-09-02
**He says carrier email-to-SMS gateways no longer work, from his own knowledge.**
Treat that as settled; do not re-research it.

That eliminates the only free, no-hardware route to arbitrary phone numbers. What
remains, honestly:

| path | reaches anyone? | recurring cost? | local? |
|---|---|---|---|
| USB LTE modem + prepaid SIM | **yes** | SIM only | **yes** |
| Matrix / Signal / ntfy | no — installers only | no | yes |
| Twilio and friends | yes | **yes** | **no** |

So the modem is the ONLY option that satisfies both halves of his objection
(cost AND cloud) while still reaching people who have not installed anything.
The open questions are now all practical rather than strategic:

  * which modem actually works on **aarch64 with in-kernel drivers and no sudo**
    (he is not in `dialout`, which is the classic trap),
  * avoiding 2G/3G-only modules — those networks are **shut down** in the US and
    a cheap SIM800-class board will never connect,
  * a prepaid SIM that suits a handful of messages a day and does not expire
    from inactivity,
  * and **carrier terms**: automated messaging from a consumer SIM can get a
    number blocked. Low-volume personal texting to people he knows is a
    different risk profile from anything bulk, and that distinction should be
    made explicitly before he buys.

### 5, ANSWERED — 2026-09-02 research verdict

**Buy the LTE modem. Gateways are dead, and were never viable anyway.**

*Why gateways fail on their own premise, before any shutdown:* to address a
message you must know the recipient's CURRENT carrier. Portability means the
number does not tell you, the authoritative data is sold not published, and every
free lookup is a metered third-party cloud API. **So step one of every message is
a cloud call** — it fails his own "not paid, not cloud" test regardless of
health. Separately: AT&T states verbatim "Effective June 17, 2025, you can no
longer send or receive texts using email" and `txt.att.net` has zero MX/A records;
Verizon publishes a 2027-03-31 shutdown; **T-Mobile and Verizon ACCEPT THE MAIL
AND SILENTLY DISCARD IT** — Jarvis would report "sent" with nothing delivered.

**The buy:**
| | |
|---|---|
| Modem | Waveshare **SIM7600NA-H 4G Dongle**, ~$65 (Amazon ASIN B09R1VP7Q5; Waveshare direct was out of stock) |
| SIM | **Red Pocket 360-day, $30/yr**, unlimited talk/text — prefer the **T-Mobile (GSMT)** variant |
| Also | one USB-A-to-C adapter; the powered hub is **required, not optional** — LTE transmit bursts exceed a bare USB 2.0 port |

**No software to install, and no sudo — verified on this box:** ModemManager is
already running, the SimTech plugin is present, the kernel already carries the
`1e0e:9001` ID, and `pkcheck` returns AUTHORIZED for
`org.freedesktop.ModemManager1.Messaging` as hunterp. So `mmcli` sends SMS with
**no sudo, no `dialout`, no kernel module** — which matters because his `dialout`
group is empty. This is why mmcli wins over gammu.

**DO NOT BUY:** SIM7600**A**-H (bands too narrow), SIM7600**G**-H (no B71, +$23),
any Huawei E3372 (wrong-continent bands, and HiLink firmware exposes no AT port),
any SIM800/SIM900 (**2G is dead** — T-Mobile's sunset completed 2026-08-03), and
**no IoT/M2M SIM** (Soracom, Hologram) — they structurally cannot SMS ordinary
phone numbers, so the device would arrive unable to text anyone.

**Cost honesty:** $75 up front, $30/yr. At ~5 msgs/day Twilio is $4.29/mo, so the
modem does not break even until partway through **year 4** — and with a $5/mo SIM
it never does. **This is a principle purchase, not a savings purchase.**

**Biggest risk, unresolved:** the MVNO may refuse to activate a SIM against a
MODEM IMEI rather than a phone IMEI. The usual workaround is activating in a
phone first — **but US iPhone 14 and newer are eSIM-only**, so that escape hatch
may be closed. **The one call that settles the purchase: ask Red Pocket/Tello
directly whether they activate a SIM in a USB LTE modem (SIMCom SIM7600 TAC).**

**Second risk:** this violates AT&T/T-Mobile prohibitions on "unattended use" and
"automated machine-to-machine communications". At a handful of messages a day to
people he knows, enforcement is unlikely — but no carrier publishes a safe
threshold and the penalty is a dead line. **Treat the number as disposable;
never use it for 2FA or account recovery.**

**Build size:** small — `jarvis/tools/sms.py` shelling out to `mmcli` via
subprocess (stdlib only), a contacts map, intent routing, the same read-back-
before-send he wanted for email, a polkit-active precondition guard, and an
inbound poll so replies reach him. About half a day with tests. **Replies work**,
so this is genuinely two-way.

**One silent failure to guard:** polkit grants Messaging on `allow_active` only.
If he fully logs out of the desktop, or Jarvis is converted to a SYSTEM systemd
unit, SMS stops working and nothing else does. Needs an explicit precondition
check with a clear error.

---

## 2026-09-02, late — his rulings on the backlog

**DOING NOW:**
  * **Email a file to someone** (idea 2). Free, no new accounts.
  * **HPCOMPUTER files + remote shell** (part of idea 3). Tailscale already
    running is the transport.
  * **Screen control on the SPARK itself** — new, his words: *"lets do screen
    control on the spark for jarvis. that would be a hard but useful one."*

**DROPPED — do not raise again unless he does:**
  * **Encryption (idea 4)** — disregarded. Three of the four were router/system
    settings anyway, and the VPN already exists.
  * **Texting (idea 5)** — disregarded. Answer stands if he ever wants it: a
    Waveshare SIM7600NA-H (~$65) + Red Pocket $30/yr, mmcli, no sudo needed.

**BACKLOG, not now:**
  * **HPCOMPUTER SCREEN control** — deliberately split from files/shell, which
    are cheap. Note when it comes up: CASTING to that screen is far cheaper than
    CONTROLLING it, and is probably what he wants more often.
  * ~~**Google Drive (idea 1)**~~ — **DROPPED 2026-09-03. His answer: "no to
    google drive."** It was the only backlog item that would have put a standing
    cloud token on the box that runs agent code, and it stayed open from 09-02
    to 09-03 for exactly that reason. Do not raise it again unless he does.

### Spark screen control — what already exists, measured 2026-09-02
Do not start from scratch. `jarvis/desktop.py` already has `DesktopControl`,
`press_key`, `launch_app`, `list_windows`, `find_claude_terminal` and
`parse_desktop_action`; `jarvis/tools/screen.py` handles screen Q&A.
**The session is X11 (`DISPLAY=:1`, `Type=x11`), not Wayland** — which is why
this is tractable at all: `xdotool` has full synthetic input on X11 and none on
Wayland. Present: `xdotool`, `xclip`, `gnome-screenshot`. **Missing:** `wmctrl`,
`scrot`, `import`, `ydotool` — and there is no sudo, so anything needing an
apt install is a blocker to raise, not to assume.

### Interrupt by talking, not by saying "Jarvis" — BACKLOG, 2026-09-02
His words: *"no just by saying jarvis for now but put that on backlog."*

**Barge-in already works, but only on the WAKE WORD.** Measured live at
23:23:54: he said "Jarvis" mid-reply and `speech interrupted (barge-in #1)`
fired in the SAME MILLISECOND as the wake detection. Talking over Jarvis does
nothing, and that is what made it feel broken.

Doing it properly means an always-open mic during speech, which needs
**acoustic echo cancellation** — otherwise Jarvis hears himself and barges in on
his own voice constantly. Prior note: AEC was *prepared but never installed* on
this box (the `aec-prep` branch, merged as docs/scripts only). So this is a real
piece of work, not a config flag, and it is the reason wake-word-only barge-in
was the right first shape.

When it is picked up: the earlier aec-prep lane found a live blocker worth
knowing — the mixer ducks by PID, so an AEC playback stream must be exempted
(sink-input `jarvis_aec_playback`) or it will duck the very audio it is trying
to cancel.


## 2026-09-03 — travel time, APPROVED (local only)

His words: *"can we also do GPS for me asking how jarvis how long it would take
to go somewhere?"* and, given the choice, *"local router"*.

**Approved shape, and it is deliberately offline:**
* POSITION comes from the phone client, which is already a browser page and
  already uses `navigator.*` (audioSession, mediaDevices). `navigator.geolocation`
  gives real GPS and it reaches only the Spark -- no cloud, no account, no key.
* ROUTING is a LOCAL engine (OSRM or Valhalla) over an OpenStreetMap extract.
  MEASURED 2026-09-03: /home has 3.0 TB free of 3.7 TB; a Texas extract is a
  couple of GB and the prepared graph a few more.

**The limitation, stated up front because it will otherwise be discovered as a
bug:** a local router returns FREE-FLOW times. It does not know about traffic.
"22 minutes" means 22 minutes on empty roads, which is optimistic at 08:00.
Live traffic needs a hosted service, which he has consistently declined.

**Bonus, and it is the reason to prefer this over a maps API even ignoring the
privacy rule:** `jarvis/leavetime.py` currently refuses to guess a walk and asks
him once per building ("How long do you need to get to Wisenbaker, sir?"). A
local router retires that asking without breaking its rule that nothing is
guessed -- a measured route is not a guess.
