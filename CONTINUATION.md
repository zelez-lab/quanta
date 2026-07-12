# Continuation — branch `dsl/swizzle-and-indexing`

State as of the `lower: fix loop-carried addresses and unconditional-br
joins` commit (rebased onto origin/main at `f9c4de9`).

## Done and verified

The loop-lowering miscompile class (spike CF0/CF1) is root-caused and
fixed in `quanta-wasm-lowering/src/lower.rs`. See that commit message
for the three mechanisms. Verified against:

- The filed repro (`dija_project/spikes/tile-raster/repro/
  loop_accumulate_bug.rs`): plain=5.2, guarded=0.7 on BOTH Metal and
  the software backend (was 4.0 / 1.0).
- Four probe kernels (`repro/probes.rs`, added this session): P1
  two-accumulator branches, P2 in-loop conditional swap + stride-2
  induction pointer, P3 bool-cast AND guard, P4 nested else-if chain.
  8/8 green on both backends. P4 previously "passed" only by a
  numerical coincidence (frozen gather at element 0 happened to sum to
  the same 4.5); with the pointer fix the br-join bug surfaced and is
  now genuinely fixed.
- The spike's K3 fine-coverage kernel: with host binning
  (`SPIKE_HOST_BIN=1`), coverage is now max_err=0.0000, bad_px=0,
  interior/exterior exact, all three scales, coverage + fill + storage
  image variants. Before the fix it was max_err=1.0 wholesale.
- `cargo test -p quanta-compiler` (29) and `-p quanta --lib` green.
  The umbrella `differential`/`op_matrix` tests and `smoke_*` examples
  fail to COMPILE on untouched origin/main too (unresolved
  `quanta_ir::op_matrix_cases`, missing `init_cpu` without the
  `software` feature) — pre-existing, not from this branch.

## Open: spike K1 binning still diverges (why RESULT is still FAIL)

`k1_bin` (doubly-nested tile deposit loop, atomic counter + computed
store) no longer misaddresses grossly — after the atomic-scale fix its
non-empty-tile count agrees with K1b — but its per-tile counts still
differ from a host model with the kernel's exact clamp semantics
(`SPIKE_BIN_DIFF=1`, 678/1024 tiles at scale 1). Coverage is winding
math, so a couple of missing left-edge segments invert whole spans:
that residual is the remaining distance to `RESULT: PASS`.

What is already ruled out:

- Per-segment range math: `k1_range_debug` (added to the spike's
  kernels.rs) recomputes (tx0, ty0, ty1) per segment on the GPU;
  `SPIKE_RANGE_DEBUG=1` diffs vs host: **0 mismatches of 220/160/110**
  at all scales. The divergence is in the deposit LOOPS' execution,
  not the clamped range computation.
- The lowered op tree (`QUANTA_LOWER_DUMP_OPS=k1_bin`) decodes
  correctly by hand: byte-canonical row bases, correct row strides
  (counts += tiles_x*4, lists += tiles_x*max_segs*4), correct inner
  advances (+4 / +max_segs*4), atomic and store indices reconciled
  back to elements, loop-crossing exit flag declared-false at entry.
- The per-local event dump (`QUANTA_LOWER_DEBUG=k1_bin`) shows the
  expected stable-reg assignments; the heavily slot-reused locals
  (6 and 10 — the two induction pointers) rebase onto Reg(0)/Reg(4)
  consistently.

Diff fingerprint at scale 1 (SPIKE_BIN_DIFF): tile-row 2 is short by
exactly 2 deposits from tx≈15/16 rightward, and row 3 gains exactly 2
from tx=0 — i.e. two multi-row segments' first-row deposit ranges
appear displaced roughly one row-length forward. Since the static op
tree looks right, suspect the RUNTIME of the doubly-nested
`KernelOp::Loop` (sentinel count 10000 + internal Break) on the Metal
emitter — or a subtle interaction the op-dump can't show (register
lifetime across the nested Loop bodies in the MSL emission). Next
probe I would write: the exact k1_bin shape (nested loops + atomic +
guarded store, multi-row ranges) at tiny scale (4x4 tiles, 3
segments) on BOTH backends — if software agrees with host and Metal
doesn't, it's the Metal emitter, not lower.rs.

Spike-side diagnostics added (uncommitted, in
`dija_project/spikes/tile-raster`): `SPIKE_BIN_DIFF`,
`SPIKE_RANGE_DEBUG`, `k1_range_debug` kernel, `SPIKE_CPU` (needs
`software` added to the spike's own Cargo.toml quanta features — the
repro/Cargo.toml already has it).

## Not started

- Fragment `&[T]` storage-buffer params (array indexing in shader
  DSL). The maintainer's vertex-uniform SSBO machinery (`7782714`) is
  the intended base: extend the binding model to SSBO-per-slot for
  arrays, grammar `param[index]` in emit_msl + emit_spirv. Nothing on
  this branch touches it yet.
- Still-unsupported wasm forms (fail loudly, no silent miscompile):
  consecutive const stores coalesced to `i64.store`, and `br_table`
  from small-int matches. Both documented in the spike's kernels.rs
  header workarounds.

## Diagnostics reference

- `QUANTA_LOWER_DUMP_INSTRS=<kernel|*>` — decoded wasm instruction
  stream with block nesting (new this session).
- `QUANTA_LOWER_DUMP_OPS=<kernel|*>` — final lowered op tree (new).
- `QUANTA_LOWER_DEBUG=<kernel|*>` — per-local get/set events with
  loop depth and stable-reg assignments (pre-existing).
- Kernels lower at BUILD time inside the quanta-compiler binary;
  `QUANTA_COMPILER=<path>` overrides the installed one (rev handshake
  from `df0c57c` applies). Touch the kernel source to force re-lower.
