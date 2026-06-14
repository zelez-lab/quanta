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
| `parse_boundary.rs` | **V4 — input edge** | The `wasmparser::Operator → RawInstr` decode boundary (`from_operator`), named explicitly. `parse_op` is the `uninterp` spec image of the production `match`; two `external_body` boundary axioms pin its production claims — **clean refusal** (unrecognized operators decode to the refusing `Other`, image of `_ => Unsupported(name)`, so the fold refuses rather than mislowers) and **determinism** (equal operators decode equally, so the decoded list is a well-defined `Seq<WasmInstr>`). `parse_ops` lifts the decode over a stream (defined, not trusted — only the per-element decode is the edge); `parse_ops_{len,cons,deterministic}` are proved list-plumbing lemmas the V7 fold lines up with. |
| `spec_types.rs` | V2 | `#[spec]` types mirroring Lean `SymVal` / `LowerState` / `WasmInstr` subset / `KernelOp` / `Reg` / `Scalar`, with structural lemmas. |
| `lower_instr_spec.rs` | V3 | Verus `spec` `lower_instr` mirroring the Lean def (straight-line slice-1 subset: consts, i32 binop/cmp family, the shl/add buffer-pattern recognizers, drop/nop/wreturn, `commit`). Local-binding + memory arms deferred to V5. |
| `lower_instr_refine.rs` | V5 (bridge + straight-line, full) | The imperative→functional bridge (`view`: production `Vec`-end stack ↔ spec `Seq`-head; `reverse_push`; `pops_leave_rest`) plus **full** per-op refinements — exact state + ops equality — for i32Const, the whole register-operand binop family (op-parametric, all 9 binops in one proof), the cmp family (all 6 cmps in one proof), and the shl/add fast-path view-alignment. Local-binding/memory arms join with the extended `LowerState`; the Rust `i32Add`'s chained-address-arithmetic cases are out of the Lean slice-1 subset. |
| `local_arms_spec.rs` | V5 (local/memory) | The extended `LowerState` (adds the four binding maps — `local_reg`/`local_ty`/`current_reg`/`buffer_slots` as `Map`s) and the spec arms `localGet` (buffer / current-binding / stable-fallback / uninit-refuse), `localSet` (the dual-Copy, existing-stable and fresh-local cases), and `i32Load` (typed Load on a 4-byte BufferAccess). Eight definitional lemmas pin each arm's shape, including the two-alloc `next_reg + 2` behavior of a first localSet. Production refinement of these arms (the `view`/`step` layer) composes against these. |
| `local_arms_refine.rs` | V5 (local prod refinement) | Production `localSet` (existing-stable), `localGet` (current-binding read + buffer-ptr push), and `i32Load` (typed Load on a 4-byte BufferAccess) refined against the spec arms, via an extended `view` mapping the production `Vec<LocalInfo>` (`{cur, stable, stable_ty}`) onto the spec binding `Map`s. `view_exposes_{stable,current,buffer_slot}` pin the field correspondence. **Documents a real gap**: production inserts frame-0 `Const fresh 0` zero-inits the Lean spec abstracts (semantically inert — the Copy overwrites before any read) — recorded for V8. |
| `commit_refine.rs` | V6 | `commit` refinement: Reg case full equality, BufferPtr both-refuse, const register/state agreement. **Surfaces two real divergences (pinned, not hidden)**: (1) the const scalar TAG — production emits `Const(dst, I32 n)`, the Lean spec `Const(dst, U32 n)` (same value, proven to differ); (2) production `commit` materializes `ScaledIdx`/`BufferAccess`/`I64Const` that the Lean spec refuses (rustc-optimizer pointer-arith shapes outside the slice-1 Lean subset). Both recorded for V8. |
| `lower_instructions_refine.rs` | **V7 — closes the arm** | **Top-level composition.** The production list fold `lower_instructions` (folding the per-instruction `step` over a `Seq<WasmInstr>`, threading `ProdCtx`, concatenating ops) refines the Lean `lowerInstrs` straight-line layer (`spec_lower_instrs`). The theorem `refine_lower_instructions` — `view ∘ lower_instructions == spec_lower_instrs ∘ view`, on state and ops, lifted through `Option` (refuse-iff-refuse) — is proved by **induction on the list**: the inductive step is `refine_step` (the head, the V5 per-op refinement lifted to the uniform `step`) composed with the IH (the tail). `view`/`unview` are mutual inverses (`reverse_reverse`), so the threaded production state slots back through `view` cleanly at each position. Two corollaries (`_refuse_iff`, `_ops`) specialize the closer for the framework-claim write-up. **Scope: the slice-1 straight-line subset** — every instruction the V5 per-op refinements cover, over arbitrary-length lists. The structured-control layer (block/wloop/wif/br/brIf/wreturn; fuel + frames + `splitAtEnd`) is mechanized in `structured_refine.rs` (below); on the straight-line subset, fuel/frames are inert and `lowerInstrs` reduces definitionally to this fold (proved there as `straightline_agrees`). |
| `structured_refine.rs` | **V7-structured** | The structured-control layer of `lowerInstrs` (`block`/`wloop`/`wif`/`br`/`brIf`/`wreturn`; Translate.lean:555-728). Mechanizes: the **splitters** (`split_at_end`/`split_at_else_or_end`/`closer_index`, mirrors of `Quanta.Wasm.Structured`) with the **progress lemma** `split_at_end_shrinks` (body + post-suffix strictly shorter than input — what makes the fuel-bounded fold well-founded); the **frame predicates** `has_loop_above`/`loops_above`; the full fuel-bounded fold `lower_instrs` with every arm transcribed; **conservativity** `straightline_agrees` (on a list with no structured openers/branches, the structured fold equals the V7 straight-line fold — V7-structured is a conservative *extension*); and **per-arm shape lemmas** — `block_splices` (body splices, no wrapper), `loop_wraps` (`[LoopOp body_ops]`), `br_loop0_no_ir`, `br_cross_loop_breaks`, plus the **not-yet-modeled refusals** `br_exitflag_refuses` (the `emit_loop_crossing_exit` shape) / `br_record_refuses` (`record_br_at`) / `br_oob_refuses`. The per-arm *production* refinement (streaming-`Vec<Frame>` ↔ recursive-descent equivalence at the body boundary) is the documented transcription obligation, the same status the straight-line `step_<op> ≈ Rust arm` correspondence carries. |

### Recorded production↔spec divergences (for V8)

The refinement has surfaced three gaps where production does more than
the Lean spec models. Each is recorded with a **mechanized witness**
(the production effect is modeled and its shape pinned), not just prose:

1. **Frame-0 zero-inits** (`local_arms_refine.rs`): production inserts
   `Const fresh 0` at the function-frame head before each dual-Copy
   (for the Metal/WGSL `uint rN = 0u;` decls). The Copy overwrites
   before any read — inert.
2. **`commit` const tag + address materialization** (`commit_refine.rs`):
   production tags consts `I32`/`I64` and materializes `ScaledIdx`/
   `BufferAccess`; the Lean spec uses `U32` and refuses the address
   forms. Same values; the spec's `commit` domain is narrower.
3. **`i32.add` chained-address arithmetic** (`i32add_chained_refine.rs`):
   beyond the Lean `BufferPtr + ScaledIdx → BufferAccess` no-IR
   fast-path, production folds three IR-emitting chained shapes —
   same-scale add, rescale add (larger BufferAccess scale, power-of-two
   ratio), and const-offset add — when rustc precomputes part of a byte
   offset. The Lean `lowerI32Add` refuses these (commits the address
   SymVals); production composes them. Each arm's shape (reg count + ops
   + pushed `BufferAccess`) is pinned; the const-offset arm re-exhibits
   the gap-2 `I32` const tag.

None affects semantics; all three are candidate spec syncs for V8.

### The model↔Rust correspondence (trust boundary)

V5 proves the *transcribed* per-op effect (the `step_<op>` /
view-aligned spec arms) equals the V3 spec. That the `step_<op>`
transcription faithfully reflects the actual `&mut self` body in
`lower.rs` is the documented manual obligation — the same status the
other Verus crates' production mirrors carry (endgame.md §5c/§8). The
differential test suite is the standing safety net underneath it.

## Boundary / TCB

The translator consumes `wasmparser::Operator` at its input edge; that
parse step is axiomatized in `parse_boundary.rs` (V4) — two
`external_body` boundary axioms (clean refusal + determinism), matching
how the other Verus crates handle their external interfaces (e.g.
`quanta-api`'s `external_body` driver obligation). rustc's Rust→wasm32
step is accepted as TCB (out of scope, same as the Lean arm). The
`parse_op`/`step_<op>` arm-by-arm faithfulness to the real Rust `match`
bodies is the manual transcription obligation, with the differential
test suite as the standing net underneath.
