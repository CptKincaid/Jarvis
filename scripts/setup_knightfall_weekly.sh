#!/usr/bin/env bash
# Install the two weekly Knightfall timers as systemd --user units (push
# Sunday 07:30 UTC, pull Sun..Wed 09:30 UTC) and START the timers.
#
# NOT RUN BY THE BUILD, ON PURPOSE. Nothing goes under ~/.config/systemd/
# user until the attack has cleared and he has restarted the live app on a
# build that knows the pending fields (an older app that saves people.json
# would silently drop them). He runs this himself, once, when both are
# true. No sudo anywhere: user units, like jarvis-f5 and jarvis-spotify.
#
# Before it: rehearse both halves --
#   cd ~/Jarvis && ~/vss_env/bin/python -m jarvis.knightfall_weekly --rehearse
#   cd ~/Jarvis && ~/vss_env/bin/python -m jarvis.knightfall_weekly --probe
#   ssh opc@163.192.101.18 'cd /home/opc/knightfall && .venv/bin/python -m app.jarvis_override --dry-run'
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNITS="jarvis-knightfall-push.service jarvis-knightfall-push.timer \
jarvis-knightfall-pull.service jarvis-knightfall-pull.timer"

# The units hard-code %h/Jarvis because systemd has no "where the repo is"
# specifier. A checkout elsewhere gets the real path substituted in.
render() {
    if [ "$REPO" = "$HOME/Jarvis" ]; then
        cat "$1"
    else
        sed "s|%h/Jarvis|$REPO|g" "$1"
    fi
}

warn() { echo "warning: $*" >&2; }
[ -x "$HOME/vss_env/bin/python" ] \
    || warn "venv missing at ~/vss_env/bin/python (the units will fail until it exists)"
if [ -f /tmp/vss_voice/jarvis.pid ]; then
    echo "note: Jarvis is running (pid $(cat /tmp/vss_voice/jarvis.pid)). It must be"
    echo "      on a build that knows the pending code fields BEFORE the first push,"
    echo "      or its next save of people.json drops them. Restart it on this branch"
    echo "      first if you have not."
fi

mkdir -p "$DEST_DIR"
for unit in $UNITS; do
    render "$HERE/systemd/$unit" > "$DEST_DIR/$unit.tmp"
    mv -f "$DEST_DIR/$unit.tmp" "$DEST_DIR/$unit"
    chmod 644 "$DEST_DIR/$unit"
done
systemctl --user daemon-reload
systemctl --user enable --now jarvis-knightfall-push.timer jarvis-knightfall-pull.timer

echo
echo "installed and started both timers. Next fires:"
systemctl --user list-timers 'jarvis-knightfall-*' --no-pager || true
echo
echo "what it did:    cd $REPO && ~/vss_env/bin/python -m jarvis.knightfall_weekly --status"
echo "the journal:    journalctl --user -u jarvis-knightfall-push -u jarvis-knightfall-pull"
echo "the log lines:  grep knightfall-weekly /tmp/vss_voice/jarvis.log"
echo "stop it all:    systemctl --user disable --now jarvis-knightfall-push.timer jarvis-knightfall-pull.timer"
