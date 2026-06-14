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
| `lower_instr_spec.rs` | V3 | Verus `spec` `lower_instr` mirroring the Lean def (straight-line slice-1 subset: consts, i32 binop/cmp family, the shl/add buffer-pattern recognizers, drop/nop/wreturn, `commit`). Local-binding + memory arms deferred to V5. |
| `lower_instr_refine.rs` | V5 (bridge + straight-line, full) | The imperative→functional bridge (`view`: production `Vec`-end stack ↔ spec `Seq`-head; `reverse_push`; `pops_leave_rest`) plus **full** per-op refinements — exact state + ops equality — for i32Const, the whole register-operand binop family (op-parametric, all 9 binops in one proof), the cmp family (all 6 cmps in one proof), and the shl/add fast-path view-alignment. Local-binding/memory arms join with the extended `LowerState`; the Rust `i32Add`'s chained-address-arithmetic cases are out of the Lean slice-1 subset. |
| `local_arms_spec.rs` | V5 (local/memory) | The extended `LowerState` (adds the four binding maps — `local_reg`/`local_ty`/`current_reg`/`buffer_slots` as `Map`s) and the spec arms `localGet` (buffer / current-binding / stable-fallback / uninit-refuse), `localSet` (the dual-Copy, existing-stable and fresh-local cases), and `i32Load` (typed Load on a 4-byte BufferAccess). Eight definitional lemmas pin each arm's shape, including the two-alloc `next_reg + 2` behavior of a first localSet. Production refinement of these arms (the `view`/`step` layer) composes against these. |
| `local_arms_refine.rs` | V5 (local prod refinement) | Production `localSet` (existing-stable), `localGet` (current-binding read + buffer-ptr push), and `i32Load` (typed Load on a 4-byte BufferAccess) refined against the spec arms, via an extended `view` mapping the production `Vec<LocalInfo>` (`{cur, stable, stable_ty}`) onto the spec binding `Map`s. `view_exposes_{stable,current,buffer_slot}` pin the field correspondence. **Documents a real gap**: production inserts frame-0 `Const fresh 0` zero-inits the Lean spec abstracts (semantically inert — the Copy overwrites before any read) — recorded for V8. |
| _(planned)_ `commit_refine.rs` | V6 | `commit` refinement. |
| _(planned)_ `lower_instructions_refine.rs` | V7 | Top-level composition. |

### The model↔Rust correspondence (trust boundary)

V5 proves the *transcribed* per-op effect (the `step_<op>` /
view-aligned spec arms) equals the V3 spec. That the `step_<op>`
transcription faithfully reflects the actual `&mut self` body in
`lower.rs` is the documented manual obligation — the same status the
other Verus crates' production mirrors carry (endgame.md §5c/§8). The
differential test suite is the standing safety net underneath it.

## Boundary / TCB

The translator consumes `wasmparser::Operator` at its input edge;
that parse step is axiomatized at the boundary (`#[verifier::trusted]`),
matching how the other Verus crates handle their external interfaces.
rustc's Rust→wasm32 step is accepted as TCB (out of scope, same as the
Lean arm).
