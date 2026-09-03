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
# BOOT ORDERING. Once you do want it on every boot, `--enable-at-boot` is the
# whole switch: it enables the unit at default.target AND replaces
# ~/.config/autostart/jarvis.desktop with one that runs scripts/jarvis-autostart,
# so Jarvis (and therefore the 19 GB gemma4:26b his residency loop pulls in)
# waits for the sidecar's 33.0 s load-and-capture instead of landing inside it.
# `--revert-boot` undoes exactly those two things. Neither flag starts anything.
#
# No sudo anywhere: user units live under ~/.config/systemd/user.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
UNIT=jarvis-breeze.service
SRC="$HERE/systemd/$UNIT"
DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
AUTOSTART_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
DESKTOP=jarvis.desktop
DESKTOP_BAK="$AUTOSTART_DIR/$DESKTOP.pre-breeze"
# The entry is rendered by jarvis.autostart and by nothing else. app.py re-runs
# autostart.install() at every start once "autostart.enabled" is on, so a
# second renderer here would be overwritten by the first one -- or, worse,
# would agree until one of them drifted and then rewrite the file on every
# login. jarvis.autostart already prefers scripts/jarvis-autostart when it is
# executable; all this has to do is ask it to write the file now.
JARVIS_PYTHON="${JARVIS_PYTHON:-$HOME/vss_env/bin/python}"

MODE=install
case "${1:-}" in
    --enable-at-boot) MODE=enable ;;
    --revert-boot)    MODE=revert ;;
    "")               ;;
    *) echo "usage: $0 [--enable-at-boot|--revert-boot]" >&2; exit 1 ;;
esac

if [ "$MODE" = revert ]; then
    systemctl --user disable "$UNIT" 2>/dev/null || true
    cat <<REVERTED
$UNIT is disabled: it will not come up at the next login.

The autostart entry is LEFT NAMING scripts/jarvis-autostart, deliberately.
That wrapper is a pass-through whenever this unit is not enabled -- it asks
systemd, gets "not enabled", and execs Jarvis, measured at about a
millisecond -- so there is nothing here to undo. Putting the bare command
back would not stick anyway: jarvis.autostart renders the wrapper whenever
it is executable, and app.py rewrites the entry at every start once
"autostart.enabled" is on. Your entry as it was before any of this is kept
at $DESKTOP_BAK.

$UNIT is NOT stopped -- \`systemctl --user stop $UNIT\` frees the 13.5 GB now.
REVERTED
    exit 0
fi

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
# The two boot-ordering pieces. The unit names the first in an ExecStartPre,
# so a missing one is a failed start, not a warning.
[ -f "$REPO/scripts/breeze_make_room.py" ] || { echo "scripts/breeze_make_room.py not found under $REPO" >&2; exit 1; }
[ -x "$REPO/scripts/jarvis-autostart" ] || { echo "scripts/jarvis-autostart missing or not executable under $REPO" >&2; exit 1; }

mkdir -p "$DEST_DIR"
render > "$DEST_DIR/$UNIT.tmp"
mv -f "$DEST_DIR/$UNIT.tmp" "$DEST_DIR/$UNIT"
chmod 644 "$DEST_DIR/$UNIT"
systemctl --user daemon-reload

if [ "$MODE" = enable ]; then
    mkdir -p "$AUTOSTART_DIR"
    # Keep exactly one backup: the FIRST one, which is the file that predates
    # any of this. Re-running must not overwrite it with our own.
    if [ -f "$AUTOSTART_DIR/$DESKTOP" ] && [ ! -f "$DESKTOP_BAK" ]; then
        cp -p "$AUTOSTART_DIR/$DESKTOP" "$DESKTOP_BAK"
        echo "backed up $AUTOSTART_DIR/$DESKTOP -> $DESKTOP_BAK"
    fi
    if [ -x "$JARVIS_PYTHON" ]; then
        ( cd "$REPO" && "$JARVIS_PYTHON" -c \
            'import sys; from jarvis import autostart; print("wrote", autostart.install(path=sys.argv[1]))' \
            "$AUTOSTART_DIR/$DESKTOP" )
    else
        echo "warning: $JARVIS_PYTHON is not there, so the autostart entry was NOT" >&2
        echo "         rewritten. Run this from $REPO with the interpreter Jarvis uses:" >&2
        echo "         python -c 'from jarvis import autostart; autostart.install()'" >&2
    fi
    systemctl --user enable "$UNIT"
    cat <<ENABLED
enabled $UNIT at default.target and pointed $AUTOSTART_DIR/$DESKTOP at
$REPO/scripts/jarvis-autostart (jarvis.autostart renders it).

NOTHING WAS STARTED. Both take effect at the next login/reboot, which is the
point: the sidecar's 31.2 GB start belongs in an empty pool, and at a fresh
login there is more free than the 49.9 GB a start was measured surviving
from. From then on the order is fixed --
jarvis-breeze loads and captures, jarvis-autostart waits for its ping (90 s
cap), then Jarvis starts and his residency loop pulls in gemma4:26b.

To take it back:
  $0 --revert-boot        # disables the unit, restores the old $DESKTOP
ENABLED
    exit 0
fi

cat <<MSG
installed $DEST_DIR/$UNIT (NOT enabled, NOT started).

To have it come up on every boot -- the supported answer to "MemFree is never
33 GB once Jarvis is up", because a login has more free than the 49.9 GB a
start was measured surviving from:
  $0 --enable-at-boot     # enables the unit AND makes Jarvis wait for it
  $0 --revert-boot        # takes it back

Turn the voice on -- three steps, in this order:
  1. start the sidecar. It needs MemFree >= 33 GB AND MemAvailable >= 60 GB and
     refuses otherwise, with the numbers, on one line. MemFree measured
     19.2-29.2 GB on 2026-09-02 with ollama, F5, Jarvis and the desktop up, so
     expect the refusal until you free some: the 18.6 GB ollama pins at
     keep_alive -1 is the lever (\`ollama stop <model>\`), not the floor.
       grep -e MemFree -e MemAvailable /proc/meminfo
       systemctl --user start $UNIT
       journalctl --user -u $UNIT -f     # "breeze: listening on ... (ready=True)"
     A refusal exits 2 and the unit sets RestartPreventExitStatus=2, so it
     stops cleanly rather than looping -- but if you ever do see "start request
     repeated too quickly", clear it with:
       systemctl --user reset-failed $UNIT
  2. check it is ready AND its CUDA graphs are captured:
       python3 -c 'import json,socket; s=socket.socket(socket.AF_UNIX); s.connect("/tmp/vss_voice/breeze.sock"); s.sendall(b"{\\"ping\\": true}\\n"); print(s.recv(65536).decode())'
  3. point Jarvis at it and restart him:
       python3 -c 'import json,pathlib; p=pathlib.Path.home()/".aiws_trainer/voice_settings.json"; d=json.loads(p.read_text()); d["tts_engine"]="breeze"; p.write_text(json.dumps(d, indent=2))'
       kill -TERM \$(cat /tmp/vss_voice/jarvis.pid)   # then start Jarvis as usual

Turn it off again (either step alone is enough; the first is instant):
       python3 -c 'import json,pathlib; p=pathlib.Path.home()/".aiws_trainer/voice_settings.json"; d=json.loads(p.read_text()); d["tts_engine"]="f5"; p.write_text(json.dumps(d, indent=2))'
       kill -TERM \$(cat /tmp/vss_voice/jarvis.pid)   # then start Jarvis as usual
       systemctl --user stop $UNIT                    # frees 13.5 GB
       $0 --revert-boot                               # if you had enabled it

Jarvis falls back to F5 on his own whenever the sidecar is absent, not
ready, or its graphs were not captured -- he never stutters and never
persists the fallback, so fixing the sidecar is all it takes to get the
voice back.
MSG
