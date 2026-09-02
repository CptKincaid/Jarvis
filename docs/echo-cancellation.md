# Echo cancellation: hearing Hunter through the music Jarvis is playing

Status on 2026-09-01: **packaged, not enabled, not measured.** Everything under
`scripts/audio/` installs and reverts cleanly without restarting PipeWire, and the
measurement tool exists — but the one number that matters (how many dB of music the
canceller removes from the Snowball) has never been taken, because taking it means
playing music on the soundbar, and that is a decision for a person in the room. Do
not read "installed" as "working" until section 5 has been run.

## 1. What went wrong on 2026-09-01, and what this can and cannot fix

Hunter asked for his liked songs, Spotify played them on **HPCOMPUTER** (the active
Connect device — another machine), and then "Jarvis" over the music was heard by the
wake word (oww 0.71 and 0.86) but thrown out by the wake speaker gate: `wake
suppressed: speaker score 0.135 < 0.25`, then 0.158. Pure music vocals score the ECAPA
gate near or below zero (`speaker verify: score=-0.084 ... REJECT`), so his voice mixed
with a vocalist landed in between and lost. The second rejection was then read as a
stranger: `guest wake (score=0.86): declining politely` — "I only answer to Hunter, sir."

**Echo cancellation cannot touch that incident.** A canceller subtracts a signal it is
*given* — the reference — from the mic. Music coming out of HPCOMPUTER's speakers never
passes through this box, so there is no reference to subtract, and no AEC configuration
here will ever make a difference to it. That case belongs to the Room Mixer's remote
duck (`SpotifyTool.duck` over the Connect volume endpoint — the mixer never reached it
that night because the idle librespot pipe on this box counted as a local stream) and to
a wake gate that does not mistake a rejected-on-music wake for a guest.

**What AEC does fix** is the case where the sound comes out of *this* box: Jarvis's own
speech (barge-in — "Jarvis, stop" while he is talking) and Spotify when the **Spark**
Connect device is the one playing (librespot → `pacat` → PipeWire → soundbar). Both of
those are audio PipeWire can hand the canceller as a reference. Note that the Spark
device has never been authenticated (`~/.cache/librespot` is empty), so today it is an
always-open silent pipe; the AEC path for music only earns its keep once he actually
picks "Spark" in a Spotify app.

## 2. Facts established live (2026-09-01)

- **PipeWire 1.0.5**, `libpipewire-module-echo-cancel` present, canceller plugin
  `libspa-aec-webrtc` linked against **webrtc-audio-processing 0.3.1** — the older
  "AEC2" (`webrtc.extended_filter`, `webrtc.delay_agnostic`), not AEC3. WirePlumber
  0.4.17.
- **The stock user unit `filter-chain.service`** (`/usr/bin/pipewire -c
  filter-chain.conf`) loads `~/.config/pipewire/filter-chain.conf.d/*.conf`, and
  `systemctl --user restart filter-chain` reloads it **without restarting pipewire**.
  A pipewire restart drops the Bluetooth soundbar (`bluez_output.F4_4E_FC_95_BA_CB.1`,
  SoundCore 2) — that is what the previous `aec-install.sh` did (it wrote into
  `pipewire.conf.d` and restarted pipewire, pipewire-pulse and wireplumber), which is
  why it was replaced. Nothing under `scripts/audio/` touches those three units now.
- **`monitor.mode` is unusable here.** The obvious design (reference = the default
  sink's monitor ports, nothing re-routed) was tried first: the bluez A2DP sink's
  monitor captured **-180 dBFS** — silence — via both `parecord` and `pw-record`, while
  the mic heard the music at -19 dBFS. The reference therefore has to be **routed into**
  the canceller, which is the topology below.
- **Live attenuation is UNMEASURED.** The 0.8 dB in the scratch recordings
  (`raw_s0.wav` vs `aec_s0.wav`) is meaningless: the only music that day came from
  HPCOMPUTER, so the canceller had no reference, and `aec_measure.py lag` on that
  session's files reports `ncc 0.000` — the two recordings do not share a programme.
  Measuring properly means playing audio on Hunter's speakers, which is not done
  unannounced.

## 3. Topology

```
Snowball ──► jarvis_aec_capture ─┐
                                 ├─ cancel ──► jarvis_aec_source ──► Jarvis (default source)
paplay / pacat ──► jarvis_aec_sink ┘   └────► jarvis_aec_playback ──► default sink (soundbar)
```

`scripts/audio/99-jarvis-echo-cancel.conf` creates four nodes:

| node | role |
| --- | --- |
| `jarvis_aec_capture` | pinned to the Snowball (`target.object`, `node.dont-reconnect = true`) so it never follows the default source — which is about to become the canceller's own output. The pin has a failure mode of its own, below |
| `jarvis_aec_source` | the cancelled mic; what Jarvis should capture from |
| `jarvis_aec_sink` | the reference: anything played here is subtracted from the mic. `priority.session = 100` keeps it from ever becoming the default sink |
| `jarvis_aec_playback` | forwards the sink's audio to the default output, so the soundbar still hears everything |

**The pin's failure mode: an unplugged Snowball takes the whole canceller down, silently.**
`node.dont-reconnect` does not make the capture *wait* for its target. WirePlumber 0.4.17's
`policy-node.lua` handles a stream whose `target.object` is not in the graph by logging
`... target not found, reconnect:false` and calling `node:request_destroy()` — the
`... waiting reconnect` branch is the one `dont-reconnect` opts *out* of. The capture
stream then goes unconnected, and `module-echo-cancel` answers `capture unconnected` with
`pw_impl_module_schedule_destroy`: all four nodes vanish. Two consequences:

- an install with the Snowball absent can only end in the installer's 10 s rollback, so
  `aec-install.sh` refuses outright when the mic is not in the source roster;
- once enabled, a USB drop/replug of the Snowball kills `jarvis_aec_source` and
  `jarvis_aec_sink` **for good**: the `filter-chain` *process* stays up, so
  `Restart=on-failure` never fires, WirePlumber falls the default source back to the raw
  Snowball, and Jarvis is on the raw mic with nothing in his log saying so — until someone
  runs `systemctl --user restart filter-chain` (section 7). That is the "setting that
  silently stopped applying" trap, and section 8 says what has to exist before this goes
  live. The same mechanism is a plausible login-time race, untested: `filter-chain.service`
  is ordered `After=pipewire-session-manager.service`, but WirePlumber enumerates the USB
  Snowball asynchronously, so a capture handled before that node exists is destroyed the
  same way. Until that is observed either way, `pactl list short sources | grep jarvis_aec`
  after each login is the check.

Who has to play into `jarvis_aec_sink` for it to be a reference:

- **Jarvis's speech** — `playback_device` in `~/.aiws_trainer/voice_settings.json`
  (the `Config.playback_device` field, default `""` = the default sink). `tts._play` and
  the streaming player pass it as `paplay --device` / `pw-play --target`. A hand edit
  does nothing until Jarvis restarts.
- **librespot** — `~/.local/bin/jarvis-spotify-device` (source copy at
  `scripts/jarvis-spotify-device`) adds `--device=jarvis_aec_sink` when that sink is in
  `pactl list short sinks` **at service start**, so `systemctl --user restart
  jarvis-spotify` after the sink exists (and again after it is removed).
- **Not routed, so not cancelled:** earcons (`jarvis/earcons.py`), the alarm loop
  (`jarvis/tools/timekeeper.py`) and the room tone (`jarvis/roomtone.py`) all play to
  the default sink with no device argument. They are short chimes, except the alarm,
  which is the one worth revisiting if "Jarvis, stop" over an alarm turns out to be a
  problem.

Jarvis's **capture** side needs no config change: the recorder and hotword open the
PipeWire default source (PortAudio only sees `pipewire`/`default` while PipeWire holds
the USB mic — see CLAUDE.md), so `pactl set-default-source jarvis_aec_source` is the
switch, and `aec-install.sh --default-source` is what flips it. That holds only while
`mic` in `voice_settings.json` is `"Default"` (it is, as of 2026-09-01): both
`Recorder._resolve_mic` and `JarvisApp._mic_index` map that name to PortAudio's default
device, whereas a `[N] name` entry pins a PortAudio index and the default-source switch
never reaches Jarvis — he would keep capturing the raw Snowball with the canceller
running beside him, and nothing in the log would say so.

## 4. Enable

Jarvis stopped or about to be restarted; nothing playing.

```bash
cd ~/Jarvis
scripts/audio/aec-install.sh                  # conf into filter-chain.conf.d, reload filter-chain
pactl list short sources | grep jarvis_aec    # jarvis_aec_source present
pactl list short sinks   | grep jarvis_aec    # jarvis_aec_sink present, NOT the default
pactl get-default-sink                        # still the soundbar
```

That alone changes nothing Jarvis hears — *provided* the configured default source is
not already the canceller. `pactl get-default-source` shows WirePlumber's **effective**
default, but `pactl set-default-source` writes the **configured** one, and WirePlumber
0.4.17 keeps those as a most-recent-first stack (`default.configured.audio.source.N` in
`~/.local/state/wireplumber/default-nodes`; `.1 = jarvis_aec_source` is already there from
the 09-01 attempt) that it re-applies the moment a stacked node reappears. If an earlier
`--default-source` was undone by anything other than `aec-uninstall.sh`, the plain install
above brings the node back and WirePlumber flips the default onto it unasked; the
installer checks for this and says `default source is already jarvis_aec_source` instead
of advising the flag. To route him through the canceller on purpose:

```bash
scripts/audio/aec-install.sh --default-source # default source -> jarvis_aec_source
# voice_settings.json: "playback_device": "jarvis_aec_sink"
systemctl --user restart jarvis-spotify       # librespot picks up --device=jarvis_aec_sink
# restart Jarvis (CLAUDE.md, "Restarting it")
```

The installer is idempotent (identical conf + nodes present = no restart), refuses to
run beside a legacy `pipewire.conf.d/99-jarvis-echo-cancel.conf`, refuses when the
Snowball is not in the source roster (the conf cannot come up without it — section 3),
and backs the conf out again if the AEC nodes do not appear within 10 s (a rejected conf
would otherwise leave `filter-chain` in its `Restart=on-failure` loop).

## 5. Measure — the step that has not been done

Target: **>= ~10 dB** of music removed from the Snowball. Announce it, then:

```bash
# 1. start music on the SPARK device (pick "Spark" in a Spotify app), or have Jarvis
#    speak a long line: jarvis --quiet "read me the week ahead"
# 2. with it playing, record raw / cancelled / reference side by side:
~/vss_env/bin/python scripts/audio/aec_measure.py record --seconds 12 --out scratchpad/aec
```

It prints `raw ... dBFS  aec ... dBFS  attenuation N dB`, a per-second table (the
canceller converges over the first second or two — a flat table near 0 dB means it
never locked on), and the lag of the mic behind the reference with a normalised
correlation `ncc`. `ncc` under ~0.1 means the reference recording was silent or
unrelated, i.e. the music was not going through `jarvis_aec_sink` — check the `--device`
on the playing stream (`pactl list sink-inputs | grep -B3 -A12 pacat`) before believing
any attenuation figure. The tool never plays audio; it only records.

Then check the two gates on the cancelled signal, because both thresholds were measured
on the raw Snowball and a canceller reshapes what is left:

- say "Jarvis" over the music and read `speaker score` on the wake line (gate:
  `SPEAKER_WAKE_MIN` 0.25);
- `~/vss_env/bin/python scripts/tune_speaker_threshold.py` if the transcript gate
  (0.30) starts rejecting him.

`webrtc.noise_suppression` and `webrtc.gain_control` are off in the conf for exactly this
reason; leave them off unless the measurement says otherwise.

## 6. The Bluetooth latency caveat

A2DP adds **~150–300 ms** between audio entering `jarvis_aec_sink` and the soundbar
producing it, so the echo reaches the Snowball that much *after* the reference. The
WebRTC canceller on this box is the older AEC2, whose delay estimator was built for
sound-card latencies of tens of ms. If `attenuation` is poor while `lag` reports a clean
peak at a few hundred ms with a healthy `ncc`, alignment is the problem, not the
canceller. Two knobs, in order, re-measuring after each (both are commented in the conf):

1. `webrtc.delay_agnostic = true` in `aec.args` — lets the canceller estimate the
   delay itself.
2. `buffer.play_delay = <ms>/1000` at the module level (a fraction of a second; set
   `audio.rate = 48000` beside it so the module can turn it into samples). Start from
   the `lag` number. Its direction — whether it delays the reference copy the
   canceller sees or the copy going to the speakers — is to be confirmed by the
   measurement, not assumed; if attenuation gets *worse*, it is the wrong sign for
   this build and only knob 1 is left.

`jarvis-spotify-device` already runs `pacat --latency-msec=200`; the two recordings the
lag tool makes are started by separate processes, so treat the lag as ±30 ms.

## 7. Revert

```bash
scripts/audio/aec-uninstall.sh   # removes the conf, reloads filter-chain, Snowball back as default source
# voice_settings.json: "playback_device": ""   (paplay fails on a sink that is gone)
systemctl --user restart jarvis-spotify
# restart Jarvis
```

The uninstaller moves the default source only onto a Snowball that is really in the
roster, and only when the default points at the canceller, at a node that no longer
exists, **or at the Snowball itself** — that last case is not a no-op: right after the
canceller vanishes the effective default has already fallen back to the Snowball while
`jarvis_aec_source` still sits on top of WirePlumber's configured stack (section 4), and
re-writing the Snowball is what puts it back on top. A default WirePlumber parked on the
HDMI monitor in between is left alone (not the canceller's doing, not ours to move).

**Recovery, not revert — the Snowball was unplugged while the canceller was live** (the
failure mode in section 3): the nodes are gone but the conf is still installed. With the
mic back in the roster (`pactl list short sources | grep Snowball`):

```bash
systemctl --user restart filter-chain          # module-echo-cancel reloads; nodes return
pactl get-default-source                       # jarvis_aec_source again, if --default-source had been run
# restart Jarvis: his capture re-opened on the raw mic when the default fell back
```

## 8. Known interactions to settle BEFORE enabling

### 8.1 Nothing in Jarvis's log says which source he is on

Two independent paths put Jarvis on the raw Snowball with the canceller apparently
running: a `[N] name` mic pin in `voice_settings.json` (section 3) and the mic-unplug
failure mode (section 3, recovery in section 7). Neither produces a log line today. Before
`--default-source` is run for real, the recorder or the wake gate needs one that names the
capture source at open time and complains when the default source is not
`jarvis_aec_source` while `~/.config/pipewire/filter-chain.conf.d/99-jarvis-echo-cancel.conf`
is installed — the same shape as `voiceprint loaded: N samples`, which exists because a
silently dead gate was worse than a loud one. Not done in this change (recorder.py is not
this lane's file).

### 8.2 The Room Mixer would duck Jarvis under his own voice

`jarvis/mixer.py` ducks every sink-input that is not Jarvis's, and it knows Jarvis's
streams **by PID** (`register_own_pid` plus descendants of the app). With the canceller
live, Jarvis's speech takes the path paplay → `jarvis_aec_sink` → `jarvis_aec_playback`
→ soundbar, and `jarvis_aec_playback` is a sink-input owned by the **filter-chain
process**, not by Jarvis. On `SpeakingState(active=True)` the mixer will find it
uncorked at 100 % and duck it to 30 % — **ducking Jarvis's own voice under his own
voice**, the one failure the mixer's docstring says it must never have. The
`module-stream-restore.id` on that stream means a Jarvis killed mid-duck would leave the
whole cancelled playback path at 30 % until `heal()` sees it again.

Before `--default-source` and `playback_device` go live, the mixer needs an exemption
for the AEC playback stream (by `node.name`/`media.name` `jarvis_aec_playback`, or by
the filter-chain unit's PID). Not done in this change: the mixer is being reworked for
the remote-duck fix at the same time, and two edits to the same module in parallel is
how a wiring gets lost.

## 9. Files

| path | what |
| --- | --- |
| `scripts/audio/99-jarvis-echo-cancel.conf` | the module config (filter-chain.conf.d shape) |
| `scripts/audio/aec-install.sh` | install + reload filter-chain; `--default-source` routes Jarvis |
| `scripts/audio/aec-uninstall.sh` | remove + reload + Snowball back |
| `scripts/audio/aec_measure.py` | `record` / `attenuation` / `lag`; records only, never plays |
| `tests/test_aec_prep.py` | the conf parses (spa-json-dump) and says what section 3 says; the scripts never restart pipewire; both scripts run for real against stub `pactl`/`systemctl` under a tmp HOME — refusal without the Snowball, rollback, the re-pin, the remembered-default message |
