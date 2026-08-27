# First-mic-session checklist (Jarvis V3 on the DGX Spark)

The voice *input* paths (recorder, hotword, speaker verification) have never
run against a real microphone on this machine. Everything that could be
checked without one has been (26 Aug 2026, synthetic audio — numbers below);
this is the ordered list for the day a mic is plugged in. Budget: ~20 min.

Baselines measured offline on the GB10 (so you know what "normal" looks like):

| path | result |
|---|---|
| Edge TTS (en-GB-RyanNeural) | 2.6 s cold / 1.0 s warm per sentence; output is MP3 24 kHz (despite `.wav` name) |
| XTTS v2 on CUDA:0 | load ~30 s (app) — 159 s when the GPU is shared; first synth 16 s (warm-up), then 1.5-1.8 s per sentence (RTF 0.48) |
| whisper `small` GPU fp16 | load 7 s; 1.0 s per 3-5 s clip; transcribed both engines' output verbatim (avg_logprob -0.32) |
| OpenWakeWord base `hey_jarvis` | 0.998-0.999 on three synthetic voices, 0.000-0.003 on other speech; 3.6-4.9 ms per 80 ms chunk |
| Custom verifier (`hey_jarvis_verifier.pkl`) | 1536 features, compatible; needs the shim in `jarvis/hotword.py` (oww 0.4.0 bug) — fires 0.975-0.993 |
| ECAPA speaker verifier on CUDA:0 | load 0.8 s; verify 61 ms; same voice 0.74, different voice 0.11, XTTS clone of the Jarvis voice 0.22 (threshold 0.40) |
| Audio output | **only a Dummy sink exists** — card profile `off`, every HDMI profile `available: no` |

## 0. Before plugging anything in

```bash
~/vss_env/bin/python -m jarvis.voice_check          # offline: sink, devices, assets, imports, CUDA
```

Expect `!! audio output sink` and `!! microphone` today; everything else `OK`.
If the app is running, its status strip shows "No audio output device —
speech will be silent" for the same reason.

## 1. Speakers first

Jarvis can be heard only when PipeWire has a real sink. Attach a display with
audio to HDMI (or a USB speaker/DAC), then:

```bash
wpctl status            # Sinks: should list an HDMI/USB sink, not "Dummy Output"
pactl list short sinks  # not just "auto_null"
paplay --volume=20000 /usr/share/sounds/alsa/Front_Center.wav   # quiet test tone
```

If the HDMI sink is missing while a display is attached:
`pactl list cards` → `Active Profile: off` → `pactl set-card-profile <card> output:hdmi-stereo`.

Then in the app: type `jarvis, are you there` — the reply "Always, sir." should
be audible (it is in the speech cache after the first play).

## 2. Microphone enumeration

```bash
~/vss_env/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```

A real capture device shows `(N in, ...)` with a hardware name (USB, hw:1,0).
`pipewire`/`default` always advertise 64 inputs and do not count.
`MachineProfile.detect()` (config.py) applies the same rule → `MACHINE.has_mic`
flips to True on the next app start; the mic button enables; the hotword
thread starts (if `hotword` is on in settings).

## 3. Live loop in one command

```bash
~/vss_env/bin/python -m jarvis.voice_check --mic --seconds 4
```

Say "Hey Jarvis, what time is it" when prompted. Expect:

- `record_fixed`  ≥ 3.9 s @16k, `rms` well above 0.003 (below that: input muted / gain)
- `transcribe`    the sentence, `accepted=True`, conf ≥ -1.5
- `hotword score` ≥ 0.3 (rule: max(hey_jarvis, 0.7·hey_mycroft))
- `speaker verify` "no voiceprint enrolled (fail-open)" until step 5
- `calibrate_noise` sets `noise_threshold` in voice_settings.json (0.01-0.15)

## 4. Recorder inside the app

1. Settings > Audio: pick the mic; "Calibrate" with the room quiet (3 s).
2. Press F5 (or the mic button), speak, stop: the transcript card appears and
   the command routes (try "what time is it").
3. Leave it recording and say nothing: it auto-stops after `silence_timeout`
   (8 s) but never in the first 5 s (grace period); the 60 s cap works.
4. `sound` toggle: start/stop beeps are pre-rendered under the log dir.

Watch `/tmp/vss_voice/jarvis.log` for `Recording started` / `Stopped: N.Ns audio`
/ `Transcribed:`.

## 5. Voiceprint

1. Settings > Voice ID > Enroll: 15 s of normal speech → "Voice enrolled (1 samples)".
   Repeat 2-3 times in different tones; ~/.aiws_trainer/voiceprint.npz appears.
2. Turn on `speaker_verify`. Speak: log shows `speaker verify: score=0.xx MATCH`.
3. Have someone else (or a phone playing speech) talk: `REJECT`, transcript card
   marked rejected (reason "speaker"). Scores: yours should sit > 0.6, others < 0.3;
   `speaker_threshold` 0.40 is the accept line. Passive learning adds accepted
   clips (kept to 100 embeddings).
4. With `speaker_verify` on and a TV playing, the recorder auto-stops when *you*
   stop talking (relaxed 0.7·threshold, 2 consecutive misses, `silence_timeout`).

## 6. Wake word

1. Log line at start: `Custom wake word verifier loaded: ... (1536 features, threshold 0.3)`.
   If instead `Custom wake word verifier NOT used`, retrain (step 6.3).
2. Say "Hey Jarvis" from 1-2 m: `Hotword detected (score=0.xx)` → recording starts
   after 0.2 s (beep if `sound` on). Debounce 1.5 s. While Jarvis is speaking
   the mic is released (`Hotword stream paused`) — the wake word is deaf during
   his own speech; that is by design, type to barge in.
3. False triggers from TV/other people: the base model fires on any voice
   (verified: three synthetic voices scored 0.998). Train the verifier on your
   own voice — Settings > Voice ID > Train wake word (3 × 3 s) — the current file
   also accepts synthetic voices (it was trained against noise negatives).
4. If the hotword never fires but `voice_check --mic` scores ≥ 0.3: check
   the app log for `hotword predict failed` (the guard added in audit C now
   logs it instead of swallowing it).

## 7. Talk-back etiquette

- Type while he speaks: he stops mid-sentence (barge-in). Say/type "quiet"
  or "stop": same, with "Very good, sir." in the transcript only.
- "say again": the last line, instantly, from the cache.
- "read the clipboard" / "read this" / "read file ~/x.md": long text in parts;
  "continue reading" for the next part.
- Pronunciation wrong? "Jarvis, pronounce <word> as <how>".

## 8. Streaming preview (not wired yet)

`Transcriber.partial()` (0.23-0.32 s per clip on GPU) exists and is tested,
but nothing feeds it 2-second windows during recording yet (`STREAMING_INTERVAL`).
When a mic exists, the recorder poll loop can hand `self._audio_frames[-N:]`
to `partial()` and publish `PartialText` — the UI already subscribes.

## Known limits

- No voice barge-in (mic paused during speech); no echo cancellation.
- `aplay` fallback cannot play Edge's MP3 stream (paplay/pw-play can).
- The recorder's mic picker lists `pipewire`/`default` as inputs even with no
  hardware mic; "Default" is the safe choice.
