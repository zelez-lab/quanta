#!/usr/bin/env bash
# Step 085 compile-time payoff: how much of quanta's own compile a headless
# consumer skips by building with render OFF. Rebuilds ONLY the quanta crate
# each way (deps cached) so the delta is render code, not shared deps.
#
# Run from repo root: scripts/bench-render-compiletime.sh
set -euo pipefail
cd "$(dirname "$0")/.."

time_build() {
  # $1 = label, rest = cargo args
  local label="$1"; shift
  cargo clean -p quanta >/dev/null 2>&1
  # warm deps once (uncounted) so only quanta itself is timed
  cargo build "$@" >/dev/null 2>&1 || true
  cargo clean -p quanta >/dev/null 2>&1
  local t0 t1
  t0=$(date +%s.%N)
  cargo build "$@" >/dev/null 2>&1
  t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" -v l="$label" 'BEGIN{printf "%-28s %.2fs\n", l, b-a}'
  echo "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')"
}

echo "== quanta crate compile time: render OFF vs ON =="
off=$(time_build "headless (render off)" -p quanta --no-default-features --features software | tail -1)
on=$(time_build  "full (render on)"      -p quanta --features software,render          | tail -1)

awk -v off="$off" -v on="$on" 'BEGIN{
  if (on>0) printf "delta: %.0f%% less compile for the quanta crate with render off\n", (on-off)/on*100
}'
