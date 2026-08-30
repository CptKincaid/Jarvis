#!/usr/bin/env bash
# Install the Jarvis echo-cancel PipeWire config, restart the user audio
# services, and make the AEC source the default mic. Reversible with
# aec-uninstall.sh. Run when Jarvis is stopped or about to be restarted:
# a PipeWire restart drops every client's stream.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
install -m644 "$HERE/99-jarvis-echo-cancel.conf" ~/.config/pipewire/pipewire.conf.d/99-jarvis-echo-cancel.conf
systemctl --user restart pipewire pipewire-pulse wireplumber
for i in $(seq 20); do pactl list short sources 2>/dev/null | grep -q jarvis_aec_source && break; sleep 0.5; done
pactl list short sources | grep -q jarvis_aec_source || { echo "AEC source did not appear"; exit 1; }
pactl set-default-source jarvis_aec_source
echo "AEC live: default source -> jarvis_aec_source; play Jarvis through sink jarvis_aec_sink"
