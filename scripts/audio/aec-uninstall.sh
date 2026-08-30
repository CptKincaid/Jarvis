#!/usr/bin/env bash
# Remove the Jarvis echo-cancel config and put the Snowball back as the mic.
set -u
rm -f ~/.config/pipewire/pipewire.conf.d/99-jarvis-echo-cancel.conf
systemctl --user restart pipewire pipewire-pulse wireplumber
sleep 2
pactl set-default-source alsa_input.usb-BLUE_MICROPHONE_Blue_Snowball_201506-00.analog-stereo 2>/dev/null || true
echo "AEC removed; default source -> Snowball"
