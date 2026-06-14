# quanta-wasm-lowering — Verus refinement arm (step 059, Phase V)

The Verus half of step 059's framework claim. The Lean arm
(`specs/verify/lean/Quanta/Wasm/`) proves the *abstract*
WASM→KernelOps translator preserves semantics; this crate proves the
*production* translator in `crates/quanta-wasm-lowering/src/lower.rs`
refines that abstract spec — closing the spec↔implementation gap.

Each file is self-contained and verifies standalone:

```sh
just verus specs/verify/verus/quanta-wasm-lowering/<file>.rs --crate-type=lib
```

(`just verus` redirects the side-effect rlibs to `target/verus/`; see
the project `CLAUDE.md`.)

## Milestone map (per `roadmap/059_source_to_ir_proof/endgame.md`)

| File | Milestone | Covers |
|------|-----------|--------|
| `skeleton.rs` | V1 | Toolchain smoke test — crate verifies clean. |
| `spec_types.rs` | V2 | `#[spec]` types mirroring Lean `SymVal` / `LowerState` / `WasmInstr` subset / `KernelOp` / `Reg` / `Scalar`, with structural lemmas. |
| _(planned)_ `lower_instr_spec.rs` | V3 | Verus `#[spec]` `lower_instr` mirroring the Lean def. |
| _(planned)_ per-op refinement files | V5 | The Rust per-op lowering returns the same `(s', ops)` as the spec. |
| _(planned)_ `commit_refine.rs` | V6 | `commit` refinement. |
| _(planned)_ `lower_instructions_refine.rs` | V7 | Top-level composition. |

## Boundary / TCB

The translator consumes `wasmparser::Operator` at its input edge;
that parse step is axiomatized at the boundary (`#[verifier::trusted]`),
matching how the other Verus crates handle their external interfaces.
rustc's Rust→wasm32 step is accepted as TCB (out of scope, same as the
Lean arm).
