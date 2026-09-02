#!/usr/bin/env bash
# Disable Jarvis's echo canceller: remove the conf from
# ~/.config/pipewire/filter-chain.conf.d/, reload filter-chain, and put the
# Snowball back as the default source if the canceller had taken it.
#
# Mirror of aec-install.sh with the same rule: only the filter-chain unit
# is restarted, never pipewire/pipewire-pulse/wireplumber (a pipewire
# restart drops the Bluetooth soundbar).  Idempotent -- run it twice and the
# second run says there was nothing to do.  A legacy conf in
# pipewire.conf.d (the old design) is reported, not removed: deleting the
# file does not unload a module the pipewire daemon already holds, so that
# is a decision for a person with the soundbar in mind.
#
# Usage: aec-uninstall.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CONF_NAME=99-jarvis-echo-cancel.conf
SRC="$HERE/$CONF_NAME"
CONF_D="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/filter-chain.conf.d"
DEST="$CONF_D/$CONF_NAME"
LEGACY="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire.conf.d/$CONF_NAME"
SOURCE_NODE=jarvis_aec_source

case "${1:-}" in
    "") ;;
    -h|--help)
        echo "usage: $(basename "$0")"
        echo "Remove $DEST, reload filter-chain, restore the Snowball as default source."
        exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

# Same single source of truth for the mic name as the installer: the conf
# beside this script, which is still here after the installed copy is gone.
MIC="$(sed -n 's/^[[:space:]]*target\.object = "\(.*\)".*/\1/p' "$SRC" 2>/dev/null | head -1)"

have_node() { pactl list short "$1" 2>/dev/null | awk '{print $2}' | grep -qx "$2"; }

if [ -f "$DEST" ]; then
    rm -f "$DEST"
    echo "removed $DEST"
    systemctl --user restart filter-chain
    echo "restarted filter-chain (pipewire and wireplumber untouched)"
    for _ in $(seq 20); do
        have_node sources "$SOURCE_NODE" || break
        sleep 0.5
    done
    if have_node sources "$SOURCE_NODE"; then
        echo "warning: $SOURCE_NODE is still in the roster; another conf may define it" >&2
    fi
else
    echo "nothing installed at $DEST"
fi

# WirePlumber picks SOME source when the default vanishes, not necessarily
# the Snowball -- the HDMI monitor is a candidate on this box.  Only move the
# default when it points at the canceller (or at nothing that exists), and
# only onto a Snowball that is really there.
CUR="$(pactl get-default-source 2>/dev/null || true)"
if [ -n "$MIC" ] && have_node sources "$MIC"; then
    if [ "$CUR" = "$MIC" ]; then
        echo "default source already $MIC"
    elif [ "$CUR" = "$SOURCE_NODE" ] || [ -z "$CUR" ] || ! have_node sources "$CUR"; then
        pactl set-default-source "$MIC" && echo "default source: ${CUR:-none} -> $MIC"
    else
        echo "default source left as $CUR (not the canceller's; not ours to move)"
    fi
else
    echo "warning: Snowball not in the source roster; default source left as ${CUR:-none}" >&2
fi

if [ -e "$LEGACY" ]; then
    echo "note: legacy conf still present at $LEGACY;" >&2
    echo "  its module stays loaded until pipewire restarts (which drops the soundbar)." >&2
fi
echo "AEC removed. If voice_settings.json still pins playback_device=jarvis_aec_sink,"
echo "  clear it (paplay fails on a sink that no longer exists and tts falls down its"
echo "  player chain), restart jarvis-spotify so librespot leaves the vanished sink,"
echo "  and restart Jarvis."
