#!/usr/bin/env bash
# Install jarvis-foldersync.service as a systemd --user unit: create the
# folders, copy the unit, daemon-reload, enable.  It does NOT start it -- the
# first start begins copying his files between two machines, and that moment
# is chosen by a person.
#
# No sudo anywhere: user units live under ~/.config/systemd/user and this runs
# as the login user, like jarvis-f5.service and jarvis-spotify.service.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
UNIT=jarvis-foldersync.service
SRC="$HERE/systemd/$UNIT"
DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

# The unit hard-codes %h/Jarvis because systemd has no "where the repo is"
# specifier.  A checkout elsewhere gets the real path substituted in.
render() {
    if [ "$REPO" = "$HOME/Jarvis" ]; then
        cat "$SRC"
    else
        sed "s|%h/Jarvis|$REPO|g" "$SRC"
    fi
}

warn() { echo "warning: $*" >&2; }
[ -x "$HOME/vss_env/bin/python" ] \
    || warn "venv missing at ~/vss_env/bin/python (the unit will fail until it exists)"
[ -f "$HOME/.ssh/hpcomputer" ] \
    || warn "no key at ~/.ssh/hpcomputer -- the lane refuses before a socket opens"

# The three folders, made here so the first run is not the thing that creates
# his desktop furniture.  Sent is created on the first successful send.
mkdir -p "$HOME/Desktop/Jarvis/Outbox" "$HOME/Desktop/Jarvis/Inbox"

mkdir -p "$DEST_DIR"
render > "$DEST_DIR/$UNIT.tmp"
mv -f "$DEST_DIR/$UNIT.tmp" "$DEST_DIR/$UNIT"
chmod 644 "$DEST_DIR/$UNIT"
systemctl --user daemon-reload
systemctl --user enable "$UNIT"

echo
echo "installed $DEST_DIR/$UNIT and enabled it for the next boot."
echo
echo "BEFORE it will do anything, set foldersync.enabled to true in"
echo "  ~/.config/jarvis/assistant.json"
echo "then check the folders and the config with:"
echo "  cd $REPO && ~/vss_env/bin/python -m jarvis.foldersync --check"
echo
echo "start it:       systemctl --user start $UNIT"
echo "watch it:       journalctl --user -u $UNIT -f"
echo "what it did:    cd $REPO && ~/vss_env/bin/python -m jarvis.foldersync --status"
echo "or just open:   ~/Desktop/Jarvis/status.txt"
