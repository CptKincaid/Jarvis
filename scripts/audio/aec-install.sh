#!/usr/bin/env bash
# Enable Jarvis's echo canceller: install 99-jarvis-echo-cancel.conf into
# ~/.config/pipewire/filter-chain.conf.d/ and reload the stock filter-chain
# user unit.  With --default-source it also makes jarvis_aec_source the
# default mic, which is the step that actually routes Jarvis's capture
# through the canceller (the recorder opens PipeWire's default source).
#
# Only `filter-chain` is ever restarted.  The previous version of this
# script wrote into pipewire.conf.d and restarted pipewire + wireplumber,
# and a pipewire restart drops the Bluetooth soundbar (2026-08-30); the
# filter-chain unit is a separate `pipewire -c filter-chain.conf` process,
# so reloading it costs the room nothing.  This script never touches the
# pipewire, pipewire-pulse or wireplumber units, and never the sink side.
#
# Idempotent: an identical conf with the AEC nodes already present is a
# no-op (a needless filter-chain restart would yank jarvis_aec_source out
# from under a running Jarvis).  Reversible with aec-uninstall.sh.
#
# Usage: aec-install.sh [--default-source]
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
CONF_NAME=99-jarvis-echo-cancel.conf
SRC="$HERE/$CONF_NAME"
CONF_D="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/filter-chain.conf.d"
DEST="$CONF_D/$CONF_NAME"
# The hazardous location the old design used.  Never written to here.
LEGACY="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire.conf.d/$CONF_NAME"
SOURCE_NODE=jarvis_aec_source
SINK_NODE=jarvis_aec_sink

usage() {
    cat <<EOF
usage: $(basename "$0") [--default-source]

Install $CONF_NAME into $CONF_D and reload filter-chain
(pipewire/wireplumber are never restarted). --default-source additionally
makes $SOURCE_NODE the default mic, i.e. routes Jarvis through the canceller.
Reverse with aec-uninstall.sh. Details: docs/echo-cancellation.md
EOF
}

SET_DEFAULT=0
for arg in "$@"; do
    case "$arg" in
        --default-source) SET_DEFAULT=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

[ -f "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
command -v pactl >/dev/null || { echo "pactl not found; is pipewire-pulse installed?" >&2; exit 1; }

# The Snowball's node name is read from the conf so there is exactly one
# copy of that string; the same trick lets aec-uninstall.sh restore it.
MIC="$(sed -n 's/^[[:space:]]*target\.object = "\(.*\)".*/\1/p' "$SRC" | head -1)"
[ -n "$MIC" ] || { echo "no target.object in $SRC" >&2; exit 1; }

# Two echo-cancel modules with the same node names would fight; the legacy
# one also lives inside the pipewire daemon, which only a pipewire restart
# can unload -- a person chooses that moment, not this script.
if [ -e "$LEGACY" ]; then
    echo "refusing: legacy conf present at $LEGACY" >&2
    echo "  remove it and restart pipewire at a time of your choosing" >&2
    echo "  (that restart drops the Bluetooth soundbar), then re-run." >&2
    exit 1
fi

have_node() { pactl list short "$1" 2>/dev/null | awk '{print $2}' | grep -qx "$2"; }

if ! have_node sources "$MIC"; then
    echo "warning: $MIC is not in the source roster; the canceller will have" >&2
    echo "  no mic until it appears (node.dont-reconnect keeps it waiting)." >&2
fi

if [ -f "$DEST" ] && cmp -s "$SRC" "$DEST" && have_node sources "$SOURCE_NODE" && have_node sinks "$SINK_NODE"; then
    echo "already installed: $DEST is current and $SOURCE_NODE is up"
else
    mkdir -p "$CONF_D"
    install -m644 "$SRC" "$DEST.tmp"
    mv -f "$DEST.tmp" "$DEST"
    echo "installed $DEST"
    if ! systemctl --user is-enabled --quiet filter-chain; then
        # Stock unit, enabled here so the canceller survives the next login.
        systemctl --user enable filter-chain
        echo "enabled filter-chain.service (it was not enabled)"
    fi
    systemctl --user restart filter-chain
    echo "restarted filter-chain (pipewire and wireplumber untouched)"
    for _ in $(seq 20); do
        have_node sources "$SOURCE_NODE" && have_node sinks "$SINK_NODE" && break
        sleep 0.5
    done
    if ! have_node sources "$SOURCE_NODE" || ! have_node sinks "$SINK_NODE"; then
        # A conf the module rejects leaves filter-chain in a Restart=on-failure
        # loop; back it out so the box is where it started.
        rm -f "$DEST"
        systemctl --user restart filter-chain || true
        echo "AEC nodes did not appear; removed $DEST and reloaded filter-chain." >&2
        echo "  journalctl --user -u filter-chain -n 30   shows why" >&2
        exit 1
    fi
    echo "up: source $SOURCE_NODE, sink $SINK_NODE"
    # The nodes vanished for a moment during the reload; a Jarvis that was
    # capturing from the canceller got moved by WirePlumber and may not have
    # been moved back.
    echo "  (a Jarvis already on $SOURCE_NODE should be restarted)"
fi

if [ "$SET_DEFAULT" = 1 ]; then
    CUR="$(pactl get-default-source 2>/dev/null || true)"
    if [ "$CUR" = "$SOURCE_NODE" ]; then
        echo "default source already $SOURCE_NODE"
    else
        pactl set-default-source "$SOURCE_NODE"
        echo "default source: $CUR -> $SOURCE_NODE"
    fi
    echo "restart Jarvis so its capture re-opens on the new default source"
    # The recorder maps mic "Default" to PortAudio's default device; a "[N]
    # name" entry pins an index and the switch above never reaches Jarvis.
    echo "  (this reaches Jarvis only while voice_settings.json has \"mic\": \"Default\")"
else
    echo "default source left as $(pactl get-default-source 2>/dev/null || echo '?');"
    echo "  re-run with --default-source to route Jarvis's mic through the canceller"
fi
echo "then: set playback_device to $SINK_NODE in ~/.aiws_trainer/voice_settings.json,"
echo "  systemctl --user restart jarvis-spotify (it picks the sink up at start),"
echo "  and measure with scripts/audio/aec_measure.py -- see docs/echo-cancellation.md"
