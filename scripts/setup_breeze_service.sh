#!/usr/bin/env bash
# Install jarvis-breeze.service as a systemd --user unit: copy + daemon-reload.
#
# It does NOT enable and does NOT start the unit, and that is the difference
# from scripts/setup_f5_service.sh. Enabling would bring a 13.5 GB resident
# model up at the next login without anyone choosing it, and its one-time
# CUDA-graph capture transiently demands ~18.3 GB on a box whose GPU memory
# IS the system memory. Both moments are chosen by hand, and the commands to
# do it are printed below. Until then Jarvis speaks exactly as it does today.
#
# No sudo anywhere: user units live under ~/.config/systemd/user.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
UNIT=jarvis-breeze.service
SRC="$HERE/systemd/$UNIT"
DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

# The unit hard-codes %h/Jarvis because systemd has no "where the repo is"
# specifier. A checkout elsewhere gets the real path substituted in.
render() {
    if [ "$REPO" = "$HOME/Jarvis" ]; then
        cat "$SRC"
    else
        sed "s|%h/Jarvis/|$REPO/|g" "$SRC"
    fi
}

# Prerequisites, checked up front so a missing one is one clear line here
# rather than a restart loop in journalctl. Missing pieces are a warning,
# not a failure: the unit can be installed ahead of the engine.
warn() { echo "warning: $*" >&2; }
[ -x "$HOME/voice-training/engines/breeze/venv/bin/python" ] \
    || warn "breeze venv missing at ~/voice-training/engines/breeze/venv (the unit will fail until it exists)"
[ -d "$HOME/voice-training/engines/breeze-q4/ckpt-q4" ] \
    || warn "pre-quantized checkpoint missing: ~/voice-training/engines/breeze-q4/ckpt-q4"
[ -d "$HOME/voice-training/engines/breeze-q4/repo" ] \
    || warn "breeze source tree missing: ~/voice-training/engines/breeze-q4/repo"
[ -x /usr/local/cuda/bin/ptxas ] \
    || warn "/usr/local/cuda/bin/ptxas missing -- CUDA graph capture will fail and the sidecar will report ready:false"
[ -f "$HOME/.aiws_trainer/jarvis_voice_ref_f5.wav" ] \
    || warn "reference clip missing: ~/.aiws_trainer/jarvis_voice_ref_f5.wav"
[ -f "$HOME/.aiws_trainer/jarvis_voice_ref_f5.txt" ] \
    || warn "reference transcript missing: ~/.aiws_trainer/jarvis_voice_ref_f5.txt"
[ -f "$REPO/scripts/breeze_server.py" ] || { echo "scripts/breeze_server.py not found under $REPO" >&2; exit 1; }

mkdir -p "$DEST_DIR"
render > "$DEST_DIR/$UNIT.tmp"
mv -f "$DEST_DIR/$UNIT.tmp" "$DEST_DIR/$UNIT"
chmod 644 "$DEST_DIR/$UNIT"
systemctl --user daemon-reload

cat <<MSG
installed $DEST_DIR/$UNIT (NOT enabled, NOT started).

Turn the voice on -- three steps, in this order:
  1. start the sidecar (needs MemFree >= 35 GB; it refuses otherwise and says so):
       systemctl --user start $UNIT
       journalctl --user -u $UNIT -f     # "breeze: listening on ... (ready=True)"
  2. check it is ready AND its CUDA graphs are captured:
       python3 -c 'import json,socket; s=socket.socket(socket.AF_UNIX); s.connect("/tmp/vss_voice/breeze.sock"); s.sendall(b"{\\"ping\\": true}\\n"); print(s.recv(65536).decode())'
  3. point Jarvis at it and restart him:
       python3 -c 'import json,pathlib; p=pathlib.Path.home()/".aiws_trainer/voice_settings.json"; d=json.loads(p.read_text()); d["tts_engine"]="breeze"; p.write_text(json.dumps(d, indent=2))'
       kill -TERM \$(cat /tmp/vss_voice/jarvis.pid)   # then start Jarvis as usual

Turn it off again (either step alone is enough; the first is instant):
       python3 -c 'import json,pathlib; p=pathlib.Path.home()/".aiws_trainer/voice_settings.json"; d=json.loads(p.read_text()); d["tts_engine"]="f5"; p.write_text(json.dumps(d, indent=2))'
       kill -TERM \$(cat /tmp/vss_voice/jarvis.pid)   # then start Jarvis as usual
       systemctl --user stop $UNIT                    # frees 13.5 GB
       systemctl --user disable $UNIT                 # if you had enabled it

Jarvis falls back to F5 on his own whenever the sidecar is absent, not
ready, or its graphs were not captured -- he never stutters and never
persists the fallback, so fixing the sidecar is all it takes to get the
voice back.
MSG
