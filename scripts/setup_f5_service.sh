#!/usr/bin/env bash
# Install jarvis-f5.service as a systemd --user unit: copy, daemon-reload,
# enable. It does NOT start the unit -- the first start loads a ~5 GB model
# onto a GPU that shares its memory with everything else on the box, so that
# moment is chosen by a person: `systemctl --user start jarvis-f5.service`.
#
# No sudo anywhere: user units live under ~/.config/systemd/user and the
# sidecar runs as the login user, like jarvis-spotify.service.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
UNIT=jarvis-f5.service
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
# not a failure: the unit can be installed ahead of the venv or the clip.
warn() { echo "warning: $*" >&2; }
[ -x "$HOME/.local/share/jarvis-f5/venv/bin/python" ] \
    || warn "F5 venv missing at ~/.local/share/jarvis-f5/venv (the unit will fail until it exists)"
[ -f "$HOME/.aiws_trainer/jarvis_voice_ref_f5.wav" ] \
    || warn "reference clip missing: ~/.aiws_trainer/jarvis_voice_ref_f5.wav"
[ -f "$HOME/.aiws_trainer/jarvis_voice_ref_f5.txt" ] \
    || warn "reference transcript missing: ~/.aiws_trainer/jarvis_voice_ref_f5.txt"
[ -f "$REPO/scripts/f5_server.py" ] || { echo "scripts/f5_server.py not found under $REPO" >&2; exit 1; }

mkdir -p "$DEST_DIR"
render > "$DEST_DIR/$UNIT.tmp"
mv -f "$DEST_DIR/$UNIT.tmp" "$DEST_DIR/$UNIT"
chmod 644 "$DEST_DIR/$UNIT"
systemctl --user daemon-reload
systemctl --user enable "$UNIT"

echo "installed $DEST_DIR/$UNIT and enabled it for the next login."
echo "start it now with:   systemctl --user start $UNIT"
echo "watch it come up:    journalctl --user -u $UNIT -f   (\"f5: listening on\" = ready)"
echo "Jarvis adopts a running unit automatically; nothing in voice_settings.json changes."
