#!/usr/bin/env bash
# Build a Quanta WebGPU smoke test for the browser.
#
# Step 050 + B⁰ (2026-04-28). Replaces the old `wasm-pack` flow:
#   - cargo build --target wasm32-unknown-unknown -p <crate>
#   - tsc (compile web/src/glue.ts → web/dist/glue.js)
#   - copy .wasm + glue.js next to the example's index.html
#
# The output is a stand-alone directory you can serve via any static
# HTTP server (e.g. `python3 -m http.server` from the example dir).
#
# Usage:
#   ./scripts/build-web.sh web_add_one
#   ./scripts/build-web.sh web_triangle
#   ./scripts/build-web.sh        # builds both
#
# No npm runtime deps are bundled; only `glue.js` (the compiled
# `web/src/*.ts` tree) and the wasm binary ship.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${PROFILE:-release}"
CARGO_PROFILE_FLAG=""
if [ "$PROFILE" = "release" ]; then
  CARGO_PROFILE_FLAG="--release"
fi

# Examples we know how to build.
ALL_EXAMPLES=(web_add_one web_triangle)
TARGETS=()
if [ "$#" -eq 0 ]; then
  TARGETS=("${ALL_EXAMPLES[@]}")
else
  TARGETS=("$@")
fi

# 1. Compile glue.ts → glue.js (idempotent — tsc skips up-to-date files).
echo "[build-web] tsc web/src → web/dist"
( cd web && npm run --silent build )

GLUE_JS="$ROOT/web/dist/glue.js"
if [ ! -f "$GLUE_JS" ]; then
  echo "[build-web] error: tsc did not produce $GLUE_JS"
  exit 1
fi

# 2. Build wasm + copy outputs for each example.
for name in "${TARGETS[@]}"; do
  case "$name" in
    web_add_one|web_triangle) ;;
    *)
      echo "[build-web] error: unknown example '$name' (try: ${ALL_EXAMPLES[*]})"
      exit 1
      ;;
  esac
  CRATE_NAME="${name//_/-}"          # web_add_one → web-add-one
  WASM_FILE="${name}.wasm"

  echo "[build-web] cargo build -p $CRATE_NAME ($PROFILE)"
  cargo build $CARGO_PROFILE_FLAG --target wasm32-unknown-unknown -p "$CRATE_NAME"

  SRC_WASM="$ROOT/target/wasm32-unknown-unknown/$PROFILE/${name}.wasm"
  DST_DIR="$ROOT/examples/$name"
  if [ ! -f "$SRC_WASM" ]; then
    echo "[build-web] error: wasm artifact $SRC_WASM not found"
    exit 1
  fi

  cp "$SRC_WASM" "$DST_DIR/$WASM_FILE"
  cp -R "$ROOT/web/dist/." "$DST_DIR/"
  echo "[build-web] ready: $DST_DIR/index.html (wasm + glue.js in place)"
done
