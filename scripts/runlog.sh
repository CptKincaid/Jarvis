#!/usr/bin/env bash
# Run a trainer under Jarvis's eye, so he can narrate its epochs.
#
#   scripts/runlog.sh python train.py --epochs 20
#
# WHY a wrapper at all: Jarvis can see that a training run exists (the
# health watchdog scans /proc for interpreter command lines) but it cannot
# read its output. /proc/<pid>/fd/1 on a real run is a socket or a pty, not
# a tailable file, and jarvis.log knows nothing about epochs or loss. This
# tees the run's stdout to <log_dir>/<pid>.log, which is the only thing the
# run ledger (jarvis/runwatch.py) reads.
#
# The `exec` matters twice. `exec > >(tee ...)` redirects this shell's own
# stdout, which the trainer then inherits; `exec "$@"` REPLACES the shell
# with the trainer, so the trainer keeps this shell's pid -- and the log is
# named for the pid Jarvis will see in /proc. Rename the file and the
# narration goes quiet.
#
# Turn the narration on with runwatch.progress = true in
# ~/.config/jarvis/assistant.json; log_dir there must match JARVIS_RUN_LOG_DIR
# (both default to ~/.cache/jarvis/runs).
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 64
fi

DIR="${JARVIS_RUN_LOG_DIR:-$HOME/.cache/jarvis/runs}"
mkdir -p "$DIR" || {
    echo "runlog: cannot write $DIR; running without a log" >&2
    exec "$@"
}

# Housekeeping: a run log per pid accumulates forever otherwise. Anything
# untouched for a week is from a run nobody is narrating any more.
find "$DIR" -maxdepth 1 -name '*.log' -mtime +7 -delete 2>/dev/null || true

LOG="$DIR/$$.log"
echo "runlog: $LOG" >&2
exec > >(tee -a "$LOG") 2>&1
exec "$@"
