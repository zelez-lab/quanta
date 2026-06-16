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
| `local_arms_spec.rs` | V5 (local/memory) + **V8-#1** | The extended `LowerState` (adds the four binding maps — `local_reg`/`local_ty`/`current_reg`/`buffer_slots` as `Map`s) and the spec arms `localGet` (buffer / current-binding / stable-fallback / uninit-refuse), `localSet` (the **zero-init `Const(fresh, zeroConst ty)` then dual-Copy**, existing-stable and fresh-local cases — synced to the post-V8 Lean spec), and `i32Load` (typed Load on a 4-byte BufferAccess). Definitional lemmas pin each arm's shape, including the two-alloc `next_reg + 2` behavior of a first localSet. Production refinement of these arms (the `view`/`step` layer) composes against these. |
| `local_arms_refine.rs` | V5 (local prod refinement) + **V8-#1 closed** | Production `localSet` (existing-stable), `localGet` (current-binding read + buffer-ptr push), and `i32Load` (typed Load on a 4-byte BufferAccess) refined against the spec arms, via an extended `view` mapping the production `Vec<LocalInfo>` (`{cur, stable, stable_ty}`) onto the spec binding `Map`s. `view_exposes_{stable,current,buffer_slot}` pin the field correspondence. **V8 divergence #1 folded in**: the Lean spec (and these spec arms) now emit the frame-0 zero-init `Const(fresh, zeroConst ty)` inline. `refine_local_set_existing_stable` proves the spec body equals production's full emission `prelude ++ body` (`[Const]` ++ `[Copy,Copy]`) as sequences; `local_set_placement_residual` pins the **only** remaining gap — production routes the const to the frame-0 head rather than inline, a dead-write hoist (same op multiset, semantically inert position). |
| `commit_refine.rs` | V6 + **V8-#2 closed** | `commit` refinement: Reg case full equality, BufferPtr both-refuse, and — after V8-#2 — **const case FULL equality** (`commit_const_full_refine`): production and spec both emit `Const(dst, I32 n)`. `commit_const_tag_agrees` pins the now-matching tag (formerly a recorded divergence). Remaining gap is **domain only**: production materializes `ScaledIdx`/`BufferAccess`/`I64Const` that the Lean spec refuses (rustc-optimizer pointer-arith shapes outside the slice-1 Lean subset). |
| `lower_instructions_refine.rs` | **V7 — closes the arm** | **Top-level composition.** The production list fold `lower_instructions` (folding the per-instruction `step` over a `Seq<WasmInstr>`, threading `ProdCtx`, concatenating ops) refines the Lean `lowerInstrs` straight-line layer (`spec_lower_instrs`). The theorem `refine_lower_instructions` — `view ∘ lower_instructions == spec_lower_instrs ∘ view`, on state and ops, lifted through `Option` (refuse-iff-refuse) — is proved by **induction on the list**: the inductive step is `refine_step` (the head, the V5 per-op refinement lifted to the uniform `step`) composed with the IH (the tail). `view`/`unview` are mutual inverses (`reverse_reverse`), so the threaded production state slots back through `view` cleanly at each position. Two corollaries (`_refuse_iff`, `_ops`) specialize the closer for the framework-claim write-up. **Scope: the slice-1 straight-line subset** — every instruction the V5 per-op refinements cover, over arbitrary-length lists. The structured-control layer (block/wloop/wif/br/brIf/wreturn; fuel + frames + `splitAtEnd`) is mechanized in `structured_refine.rs` (below); on the straight-line subset, fuel/frames are inert and `lowerInstrs` reduces definitionally to this fold (proved there as `straightline_agrees`). |
| `structured_refine.rs` | **V7-structured** | The structured-control layer of `lowerInstrs` (`block`/`wloop`/`wif`/`br`/`brIf`/`wreturn`; Translate.lean:555-728). Mechanizes: the **splitters** (`split_at_end`/`split_at_else_or_end`/`closer_index`, mirrors of `Quanta.Wasm.Structured`) with the **progress lemma** `split_at_end_shrinks` (body + post-suffix strictly shorter than input — what makes the fuel-bounded fold well-founded); the **frame predicates** `has_loop_above`/`loops_above`; the full fuel-bounded fold `lower_instrs` with every arm transcribed; **conservativity** `straightline_agrees` (on a list with no structured openers/branches, the structured fold equals the V7 straight-line fold — V7-structured is a conservative *extension*); and **per-arm shape lemmas** — `block_splices` (body splices, no wrapper), `loop_wraps` (`[LoopOp body_ops]`), `br_loop0_no_ir`, `br_cross_loop_breaks`, plus the **not-yet-modeled refusals** `br_exitflag_refuses` (the `emit_loop_crossing_exit` shape) / `br_record_refuses` (`record_br_at`) / `br_oob_refuses`. The per-arm *production* refinement (streaming-`Vec<Frame>` ↔ recursive-descent equivalence at the body boundary) is **mechanized** in `streaming_equiv.rs` (below) for the wrapper-assembly shape — the one place the two strategies could diverge. |
| `streaming_equiv.rs` | **hardening** | The streaming-`Vec<Frame>` ↔ recursive-descent equivalence — the largest trust step under V7-structured, formerly a prose note in `Quanta.Wasm.Structured`. Models op-assembly on both sides (`step_stream`/`run_stream` = production's push-frame/accumulate/pop-and-fold walk; `descend` = the `split_at_end` recursive descent) routed through a single `wrap` (Block→splice, Loop→`[LoopOp]`, If→`[Branch cond · []]`). Proves: the three **close lemmas** (`close_block_splices`/`close_loop_wraps`/`close_if_branches` — a `wend` close folds exactly `wrap(kind, body)` into the parent), plain-accumulation + height preservation, and the concrete **`stream_equiv_recursive_flat`** (`run_stream` from a Function frame == `descend`, by induction, on flat streams). **Honest scope:** the full *nested* `run_stream == descend` (arbitrary depth) is not driven end-to-end here; what's proved is (1) flat-stream equality and (2) per-`wend` `wrap`-agreement — the only divergence point — so the nested composition is mechanical but un-driven. Shrinks, does not fully retire, the body-boundary obligation. |

### Production↔spec divergences (V8) — all closed

The refinement surfaced three gaps where production did more than the
Lean spec modeled. Each was first recorded with a **mechanized witness**
(the production effect modeled and its shape pinned), then closed by
adding the matching Lean arm with proven preservation. All three (V8-#1
frame-0 zero-inits, V8-#2 const-tag + type threading, V8-#3
chained-address folds) are now synced into the spec.

1. **Frame-0 zero-inits** — **CLOSED (V8-#1)**. The Lean spec
   (`Translate.lean`'s `localSet`/`localTee`, via the new `zeroConst`
   helper) now emits the `Const(fresh, zeroConst ty)` zero-init inline
   before each dual-Copy; the preservation proofs absorb it through the
   new `evalOps_const_copy_absorb` dead-write lemma (the const writes
   `fresh`, overwritten by `Copy(fresh, src)` before any read), backed
   by `regWrite_regWrite_self`. The Verus arm (`local_arms_spec.rs`,
   `local_arms_refine.rs`) is synced: `refine_local_set_existing_stable`
   proves the inline spec body equals production's `prelude ++ body`,
   and `local_set_placement_residual` pins the **sole** remaining
   difference — production routes the const to the frame-0 head, not
   inline (a dead-write hoist; identical op multiset, inert position).
2. **`commit` const tag** — **CLOSED (V8-#2)** (`commit_refine.rs`).
   The Lean spec now tags committed `i32.const`s `I32` (the wrapped
   32-bit pattern), coherent with the `.reg dst .i32` encoding; the op
   is byte-identical to production. `commit_const_full_refine` proves
   full `commit` equality on the const arm and `commit_const_tag_agrees`
   pins the now-matching tag. The remaining `commit` gap is domain only:
   production materializes `ScaledIdx`/`BufferAccess`/`I64Const` the
   slice-1 Lean subset refuses (rustc pointer-arith shapes).
3. **`i32.add` chained-address arithmetic** — **CLOSED (V8-#3)**
   (`i32add_chained_refine.rs`). Beyond the Lean `BufferPtr + ScaledIdx
   → BufferAccess` no-IR fast-path, production folds three IR-emitting
   chained shapes — same-scale add, rescale add (larger BufferAccess
   scale, power-of-two ratio), and const-offset add — when rustc
   precomputes part of a byte offset. `Translate.lean`'s `lowerI32Add`
   now has matching arms for all three (either operand order), and
   `Preservation.lean` proves each one's encoding correctness
   (`preservation_i32Add_chained_{sameScale,constOff,rescale}`): the
   emitted `Add` / `Shl`+`Add` / `Const`+`Add` make the merged
   `BufferAccess` encode the WASM `addr + idx` by distributivity /
   shift-as-multiply / divisibility (no-overflow via per-arm
   preconditions). Scope-validity, well-scopedness, nextReg-monotonicity
   and bufferSlots are proven for the new arms; `evalBinOp_add_asU32Bits`
   / `evalBinOp_shl_asU32Bits` give the tag-agnostic bit-pattern results
   the folds read. This Verus file now stands as the production
   transcription matched by a proven Lean arm, not an open gap.

Additionally, **V8-#2 closed the binop/cmp result-type threading on the
Lean side**: `lowerI32Bin`/`lowerI32Cmp` derive the pushed reg's scalar
from the committed operands (`binResultTy = if tags match then tag else
.i32`) instead of hardcoding `.u32`, matching production's
`ty = if ty_a == ty_b { ty_a } else { I32 }`. Locals carry the
committed value's type (`commitTy`), and a `local.get` reads it back at
that tag — the full local surface (`LocalsRefines`/`CurrentRegRefines`
generalized to the recorded `localTy`) is proven in `Preservation.lean`.

None affects semantics. #1, #2, and #3 are all now synced into the Lean
spec with proven preservation (#1's residual is op *placement*, not the
op set). No open production↔spec divergences remain on the modeled
surface; the Verus files stand as the production transcriptions matched
by proven Lean arms.

### The model↔Rust correspondence (trust boundary)

V5 proves the *transcribed* per-op effect (the `step_<op>` /
view-aligned spec arms) equals the V3 spec. That the `step_<op>`
transcription faithfully reflects the actual `&mut self` body in
`lower.rs` is the documented manual obligation — the same status the
other Verus crates' production mirrors carry (endgame.md §5c/§8). The
differential test suite is the standing safety net underneath it.

The structured arms carried an additional trust step — that
production's streaming `Vec<Frame>` walk assembles the same op list as
the recursive-descent `lower_instrs`. `streaming_equiv.rs` mechanizes
the wrapper-assembly core of that (both sides route every construct
through one `wrap`; flat streams are proved equal end-to-end), shrinking
the obligation to the un-driven nested composition.

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
