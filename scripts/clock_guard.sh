#!/usr/bin/env bash
# THE CLOCK GUARD, runtime half -- prove the suite does not care what time it is.
#
# Why this exists: on 2026-09-05 one commit was 12217 passed / 0 failed at
# 03:50 and 4 failed / 197 passed at 09:48. Nothing had changed but the wall
# clock. A suite that is only green in the evening cannot be a merge gate.
#
# How it works: it does NOT freeze the clock. It moves the process's local
# TIMEZONE so datetime.now() reads the hour we want, while the clock keeps
# ticking at the normal rate and every absolute instant is unchanged. That
# matters -- a frozen clock hangs every test that waits on a deadline (12
# files in this suite do) and breaks anything subclassing date/datetime.
# Shifting the zone costs nothing and needs no library.
#
# The system clock and the machine's timezone are NEVER touched: TZ is set
# for the pytest child process only.
#
#   scripts/clock_guard.sh              # the 6 decisive configurations (~30 min)
#   scripts/clock_guard.sh --hours      # all 24 hours, the thorough sweep
#   scripts/clock_guard.sh 09:00 UTC    # just these configurations
#
# Exit 0 = every configuration agreed. Exit 1 = a test's verdict moved with
# the clock; the offending configuration and test names are printed.
set -uo pipefail
cd "$(dirname "$0")/.."

# PYTEST_ARGS scopes the sweep while you iterate, e.g.
#   PYTEST_ARGS=tests/test_claim_guard_memory.py scripts/clock_guard.sh
# Unset, it sweeps the whole suite, which is the only form that proves the
# merge gate.
PYTEST=(~/.local/bin/memcap timeout 1800 "${JARVIS_PY:-$HOME/vss_env/bin/python}"
        -m pytest -q -p no:cacheprovider ${PYTEST_ARGS:-})

# The configurations that between them would have caught every clock bug
# found on 2026-09-05: one in each greeting band (morning / afternoon /
# evening -- the bands live in jarvis/arc.py), one that rolls the DATE over
# in the middle of the run, and one in a foreign zone. "real" is the
# unsimulated clock, so a difference between real and simulated counts too.
DEFAULT=(real --clock-at=09:00 --clock-at=13:00 --clock-at=21:00
         --clock-at=23:59:30 --clock-tz=UTC)

case "${1:-}" in
  --hours) CONFIGS=(real); for h in $(seq -w 0 23); do CONFIGS+=("--clock-at=$h:30"); done ;;
  "")      CONFIGS=("${DEFAULT[@]}") ;;
  *)       CONFIGS=(); for a in "$@"; do
             case "$a" in real)         CONFIGS+=(real) ;;
                          *[/A-Za-z]*)  CONFIGS+=("--clock-tz=$a") ;;
                          *)            CONFIGS+=("--clock-at=$a") ;; esac
           done ;;
esac

out=$(mktemp -d); trap 'rm -rf "$out"' EXIT
printf 'clock guard: %d configurations, real local time now %s\n\n' \
       "${#CONFIGS[@]}" "$(date '+%H:%M:%S %Z')"

# The guard does NOT demand a green suite: a worktree always carries the
# one documented test_autostart failure, and demanding green would make the
# guard cry wolf. What it demands is that the set of failures is the SAME
# in every configuration. A test that fails in some hours and not others is
# the bug this exists to find; a test that fails in all of them is somebody
# else's problem and is reported separately.
names=()
for cfg in "${CONFIGS[@]}"; do
  tag=$(echo "$cfg" | tr -c 'A-Za-z0-9' '_')
  log="$out/$tag.log"
  if [ "$cfg" = real ]; then "${PYTEST[@]}" >"$log" 2>&1; else "${PYTEST[@]}" "$cfg" >"$log" 2>&1; fi
  rc=$?
  if [ "$rc" -eq 137 ]; then
    printf '  %-22s MEMORY CAP EXCEEDED (exit 137) -- a finding, not a flake\n' "$cfg"
    exit 1
  fi
  grep -E '^(FAILED|ERROR) ' "$log" | awk '{print $2}' | sort -u >"$out/$tag.fails"
  printf '  %-22s %s\n' "$cfg" "$(grep -E 'passed|failed' "$log" | tail -1)"
  names+=("$tag")
done

first="$out/${names[0]}.fails"
moved=0
echo
for tag in "${names[@]}"; do
  if ! diff -q "$first" "$out/$tag.fails" >/dev/null; then moved=1; fi
done

if [ "$moved" -eq 0 ]; then
  n=$(wc -l <"$first")
  echo "PASS: identical verdicts in all ${#CONFIGS[@]} configurations."
  if [ "$n" -gt 0 ]; then
    echo "      ($n failure(s) present in EVERY configuration, so not clock-related:"
    sed 's/^/        /' "$first"; echo "      )"
  fi
  exit 0
fi

echo "FAIL: a test changed its verdict with the clock."
echo
cat "$out"/*.fails | sort | uniq -c | sort -rn | while read -r count name; do
  [ "$count" -eq "${#CONFIGS[@]}" ] && continue          # constant: not the clock
  echo "  $name"
  for tag in "${names[@]}"; do
    grep -qx "$name" "$out/$tag.fails" && echo "        RED  $tag" || echo "        ok   $tag"
  done
done
cat <<'MSG'

That test asserts against a clock it does not control. Fix the TEST, not the
product -- Jarvis is right to reground a stale greeting against the hour, to
render a timestamp in the machine's local zone, and to read today's date.
Give the test the clock through the seam the product already has
(ground_greeting(now=), DayReviewer(now=), calendar.now_local,
SensingPolicy(now=), Arc(now=)), or assert on the part the test is actually
about. See tests/test_clock_hygiene.py and docs/clock-guard.md.
MSG
exit 1
