# Render Conformance

`crates/tools/quanta-conform` answers one question: **does the same draw
produce the same pixels on every backend?** It is the frame-level
counterpart to the shader-parity suites — those prove two emitters agree
on what a shader body *means*, this proves the whole pipeline (vertex
fetch, rasterization, blending, resolve, readback) lands the same
picture.

## The runner

The corpus is a list of named draws, each a plain
`fn(&Gpu) -> Result<Frame, QuantaError>` written against the public API
only. Every case renders offscreen into a fresh RGBA8 target and returns
the readback.

Each backend runs the whole corpus **in its own child process**, with
`QUANTA_BACKEND` set, so no driver's state can leak into another's
frames. The parent compares every backend's frames against the
reference and prints the matrix.

```
quanta-conform --bless <backend> --golden-dir <dir>
    run the corpus on one backend, write its frames as the goldens

quanta-conform --backends a,b --golden-dir <dir> [--out matrix.md]
    run each backend, compare every frame against the committed
    goldens, print the matrix, exit non-zero on any failure

quanta-conform --backends a,b --reference a
    no goldens: compare the listed backends live against one of them
    (for a box with two real backends)
```

### Why goldens and not a CPU reference

The CPU device has no rasterizer — it refuses render passes outright.
There is no software oracle to be right, so the reference is a committed
set of frames, blessed from a named backend and recorded in the goldens'
`BLESSED_FROM` manifest. A divergence is a divergence between two real
rasterizers either way; the manifest keeps the matrix honest about which
one is being called the reference.

**Reference policy:** goldens are blessed from **Metal** on an Apple
machine, and cross-checked against **lavapipe** in the `gpu-tests-vulkan`
CI lane.

## Tolerance and budget

A pass never hides its terms — both knobs print in the matrix row.

| Term | Meaning |
|---|---|
| `channel` | Maximum per-channel difference, in unorm LSBs, a pixel may show and still count as matching. |
| `edge_budget_permille` | How many out-of-tolerance pixels the case may have, in per-mille of the frame. |

The budget exists because rasterizers legitimately disagree about
triangle-edge coverage — tie-breaking rules differ per implementation,
and a pixel whose centre sits exactly on an edge is a coin toss the
specs do not settle. A case with an interior primitive edge grants a
small allowance for it. A case with **no** interior edge runs at budget
zero: there is no coverage decision to be made, so any structural
divergence is a bug.

Two presets cover almost everything:

- `Tolerance::EXACT` — 1 LSB, budget 0. Clears, fullscreen quads,
  axis-aligned state.
- `Tolerance::EDGED` — 1 LSB, budget 5‰. Anything whose primitives
  leave a visible edge inside the frame.

A case whose own math differs legitimately between backends names its
own terms instead of borrowing a preset. `msaa4_resolve` is the one in
the corpus today: multisample resolve averages over sample positions the
specs do not pin, so it runs at 2 LSBs and 10‰, with the reason in the
constant's doc comment.

**Never widen a tolerance to make a backend pass against its own
goldens.** A backend compared to frames it blessed itself must be Δ≤0;
anything else means the case is nondeterministic, and the fix is the
case.

## Adding a case

1. Pick the family module under `src/cases/` — or add one. The family's
   fragments live in it; geometry, vertex stages, and pass plumbing that
   more than one family needs live in `src/cases/common.rs`.
2. Write the draw as `fn(&Gpu) -> Result<Frame, QuantaError>`, public
   API only, offscreen, RGBA8, 64×64 unless the case is about size.
   Row 0 is the top row — the cross-backend orientation contract.
3. Keep it deterministic: no time, no randomness, no dependence on what
   ran before. Every case builds its own target.
4. Pick the terms by the rule above and register the case in
   `cases::all()`, which is the matrix's row order.
5. Re-bless and commit the new `.frame` alongside the code.

A fragment that reads the shared `common::QuadVary` interface has to
import its `__quanta_varyings_QuadVary` trampoline: the shader macros
reach the varyings metadata through it, and the struct itself is erased
along with the shader function.

If a case's API path is `NotSupported` on some backend, keep it — the
matrix prints `NotSupported` with the driver's reason, which is a more
useful row than a missing one.

## Re-blessing

```bash
just conform-bless    # build with metal, bless from metal
just conform          # build with metal, check metal against the goldens
```

`conform` must exit 0 with every case Δ≤0 — a backend against its own
goldens has nothing to round differently. Run it twice: two identical
matrices is the determinism check, and it is cheap.

## In CI

The `gpu-tests-vulkan` lane builds the runner with `--features vulkan`
and runs the corpus on lavapipe against the committed goldens. The
matrix goes into the job summary and is uploaded as the
`conform-matrix` artifact.

That step is **advisory** (`continue-on-error: true`, plus a `::warning::`
annotation when it diverges) until first contact with lavapipe comes
back green. The goldens have only ever been compared against the backend
that blessed them, so the first cross-backend run is a measurement, not
a gate. Once a run is clean the `continue-on-error` comes off and the
step becomes the parity gate.
