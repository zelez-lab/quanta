#!/usr/bin/env bash
# Step 085 separation invariants — fail CI if the compute/render boundary
# regresses. Two checks:
#
#   1. Surface-disjointness: with `render` OFF, the public surface of the
#      `quanta` crate must expose NO render type name. Mechanically checked
#      from rustdoc JSON (nightly).
#   2. Dependency-acyclicity: there must be no `quanta -> quanta-render`
#      edge (render depends on compute, never the reverse). Checked from
#      `cargo metadata`.
#
# Run from the repo root: scripts/check-render-boundary.sh
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

# Render type names that must never appear on the render-OFF quanta surface.
RENDER_TYPES=(
  RenderPass RenderBuilder ColorTarget DepthTarget RenderOp
  Pipeline PipelineDesc TessellationPipeline TessellationPipelineDesc
  MeshPipeline MeshPipelineDesc VrsState ShadingRate IndirectRenderBundle
  GeometryDesc RayTracingPipelineDesc AccelerationStructure
)
# Render Gpu methods that must not be reachable render-off.
RENDER_METHODS=(render_target msaa_target resolve_texture stencil_read render_begin pipeline_create)

echo "== check 1: surface-disjointness (render OFF) =="
# rustdoc JSON is nightly-only and its schema drifts; we only do a
# name-membership grep, which is stable across schema versions.
JSON=target/doc/quanta.json
rm -f "$JSON"
cargo +nightly rustdoc -p quanta --no-default-features --features software \
  -- -Z unstable-options --output-format json >/dev/null 2>&1
if [[ ! -f "$JSON" ]]; then
  echo "  ERROR: rustdoc JSON not produced (need a nightly toolchain)"; exit 2
fi
for name in "${RENDER_TYPES[@]}" "${RENDER_METHODS[@]}"; do
  if grep -qE "\"name\":\"$name\"" "$JSON"; then
    echo "  LEAK: render name '$name' is reachable from render-off quanta"
    fail=1
  fi
done
[[ $fail -eq 0 ]] && echo "  ok: zero render names on the render-off surface"

echo "== check 2: dependency-acyclicity =="
# Assert quanta's dependency list does not contain quanta-render.
if cargo metadata --no-deps --format-version 1 \
  | grep -oE '"name":"quanta"[^}]*' >/dev/null 2>&1; then :; fi
# Robust: list quanta's direct deps via cargo tree and ensure no quanta-render.
if cargo tree -p quanta --no-default-features --features software -e normal 2>/dev/null \
  | grep -q "quanta-render"; then
  echo "  CYCLE: quanta depends on quanta-render"
  fail=1
else
  echo "  ok: no quanta -> quanta-render edge"
fi

# And the converse must hold: quanta-render DOES depend on quanta.
if cargo tree -p quanta-render -e normal 2>/dev/null | grep -q "quanta v"; then
  echo "  ok: quanta-render -> quanta edge present"
else
  echo "  ERROR: quanta-render does not depend on quanta"; fail=1
fi

if [[ $fail -ne 0 ]]; then
  echo "render-boundary check FAILED"; exit 1
fi
echo "render-boundary check passed"
