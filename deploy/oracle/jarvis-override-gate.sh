#!/bin/sh
# The forced command for a DEDICATED, RESTRICTED ssh key on Oracle, so the
# Spark's weekly timer does not hold a key that is root on the dead-man's
# switch (opc has NOPASSWD sudo; the existing oracle-key is that key).
#
# NOT INSTALLED BY THE BUILD. His call. When approved, on Oracle as opc:
#   install -m 0755 jarvis-override-gate.sh /home/opc/knightfall/jarvis-override-gate.sh
#   cp -p ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak-before-jarvis-code-20260906
#   printf 'command="/home/opc/knightfall/jarvis-override-gate.sh",restrict %s\n' \
#       "$(cat jarvis-knightfall.pub)" >> ~/.ssh/authorized_keys
# and on the Spark: ssh-keygen -t ed25519 -f ~/.ssh/jarvis-knightfall -N ''
# then set knightfall_weekly.key_path to ~/.ssh/jarvis-knightfall in
# ~/.config/jarvis/assistant.json. oracle.key_path stays as it is: the
# voice lane's status/logs/restart still use the full key.
#
# What it allows: exactly the forms jarvis/knightfall_weekly.py sends
# (put / receipt / ping / revoke <8 hex> / receipt --ack <8 hex>), run
# through the app's own module. Anything else -- a shell, scp, sftp, a
# different module -- is refused with exit 2. `restrict` in
# authorized_keys removes forwarding, pty and X11 besides.
#
# `receipt --ack <id>` is the only verb that DELETES a receipt, and it is
# sent only after the Spark has written the receipt's meaning into
# people.json; a plain `receipt` reads and removes nothing.
set -eu
cmd="${SSH_ORIGINAL_COMMAND:-}"
# The Spark sends "cd /home/opc/knightfall && exec .venv/bin/python -m
# app.jarvis_override <verb>"; only the verb after the module name counts.
verb="${cmd##*app.jarvis_override }"
case "$verb" in
    put|receipt|ping) ;;
    revoke\ [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
    receipt\ --ack\ [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
    *) echo '{"ok": false, "why": "not a verb this gate allows"}'; exit 2 ;;
esac
cd /home/opc/knightfall
# shellcheck disable=SC2086
exec .venv/bin/python -m app.jarvis_override $verb
