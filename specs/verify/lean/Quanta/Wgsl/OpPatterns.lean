/-
# WGSL emitter patterns — per-op structural witnesses

Step **B.3** of the FFI TCB shrink track. For every `KernelOp`
variant the JIT emitter handles, this module pins down a
*representative* WGSL `Source` fragment in the Lean grammar mirror
and proves it is structurally well-formed. The shapes mirror
`crates/gpu/quanta-ir/src/emit_wgsl/ops.rs` op-for-op; identifiers and
scalar choices are fixed to canonical placeholders since structural
well-formedness is invariant under register IDs and field names.

Why this is the right next step:
- T410 says "for every `KernelDef k`, `emit_wgsl_jit k` is well-
  formed." B.4 will reduce that to: "for every kernel, the
  emitter's output is built up from these per-op patterns." So
  T410 = (B.3 per-op witnesses) ∘ (composition lemma) ∘ A12.
- Each per-op claim is a closed `Bool` — `native_decide` evaluates
  `Stmt.wellFormed` over a finite list and produces `rfl`. No
  hand-written proofs needed.

If a future emitter change touches a tag, regenerating this Lean
view forces a recheck — exactly like B″'s discipline for the FFI
boundary.
-/

import Quanta.Wgsl.Grammar

namespace Quanta.Wgsl.OpPatterns

open Quanta.Wgsl

-- ════════════════════════════════════════════════════════════════════
-- KernelOpTag — the finite set of variants the emitter dispatches on
-- ════════════════════════════════════════════════════════════════════
--
-- Mirrors `pub enum KernelOp` in `crates/gpu/quanta-ir/src/types.rs`.
-- The tag enumeration is what `native_decide` ranges over; the
-- payload of each variant (register IDs, scalar types, field names)
-- does not affect structural well-formedness, so a canonical
-- placeholder per tag suffices.

inductive KernelOpTag where
  -- Memory
  | load | store | sharedDecl | sharedLoad | sharedStore
  -- Arithmetic
  | binOp | unaryOp | cmp | cast | bitcast | copy
  -- Control flow
  | branch | loopOp | breakOp | barrier
  -- Calls
  | mathCall | deviceCall
  -- Atomics
  | atomicOp | atomicCas
  -- Wave / subgroup
  | waveShuffle | waveBallot | waveAny | waveAll
  -- Vector / matrix
  | vecConstruct | vecExtract | matMul | cooperativeMMA
  -- Texture
  | textureSample2D | textureSample3D | textureWrite2D | textureSize
  -- IDs
  | quarkId | quarkCount | protonId | nucleusId | protonSize
  -- Misc
  | constLit | dispatch
  | countTrailingZeros | countLeadingZeros
  deriving Repr, DecidableEq

/-- Enumeration witness — every tag once. `native_decide` ranges
    over this list rather than asking for a `Fintype` instance. -/
def allKernelOpTags : List KernelOpTag :=
  [ .load, .store, .sharedDecl, .sharedLoad, .sharedStore,
    .binOp, .unaryOp, .cmp, .cast, .bitcast, .copy,
    .branch, .loopOp, .breakOp, .barrier,
    .mathCall, .deviceCall,
    .atomicOp, .atomicCas,
    .waveShuffle, .waveBallot, .waveAny, .waveAll,
    .vecConstruct, .vecExtract, .matMul, .cooperativeMMA,
    .textureSample2D, .textureSample3D, .textureWrite2D, .textureSize,
    .quarkId, .quarkCount, .protonId, .nucleusId, .protonSize,
    .constLit, .dispatch,
    .countTrailingZeros, .countLeadingZeros ]

-- ════════════════════════════════════════════════════════════════════
-- Per-tag representative pattern
-- ════════════════════════════════════════════════════════════════════
--
-- For each tag, return the WGSL statement shape the emitter
-- produces. Register IDs are placeholders (0, 1, 2 …); the scalar
-- type is `f32` where the emitter's choice would otherwise depend
-- on `KernelOp` payload. Branches/loops/atomic-CAS carry their
-- nested statement lists empty for the structural check; the
-- recursive cases are covered when their inner ops are looked up
-- separately.

private def regE (n : Nat) : Expr := Expr.reg n

private def f32Ty : Ty := Ty.scalar Scalar.f32

/-- One representative WGSL `Stmt` list per `KernelOpTag`.
    Mirrors the patterns in `emit_wgsl/ops.rs`; the contents are
    purely structural — actual emitter output substitutes the
    appropriate identifiers and types but the constructor shape
    is invariant. -/
def opPattern : KernelOpTag → List Stmt
  -- Memory
  | .load =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.indexBuf "field" (regE 1)) ]
  | .store =>
      [ Stmt.assignIdx "field" (regE 1) (regE 2) ]
  | .sharedDecl =>
      []  -- shared decls are module-scope; no per-op statement
  | .sharedLoad =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.indexBuf "shared0" (regE 1)) ]
  | .sharedStore =>
      [ Stmt.assignIdx "shared0" (regE 1) (regE 2) ]
  -- Arithmetic
  | .binOp =>
      [ Stmt.letDecl 0 none
          (Expr.binOp BinOp.add (regE 1) (regE 2)) ]
  | .unaryOp =>
      [ Stmt.letDecl 0 none
          (Expr.unaryOp UnaryOp.neg (regE 1)) ]
  | .cmp =>
      [ Stmt.letDecl 0 none
          (Expr.binOp BinOp.lt (regE 1) (regE 2)) ]
  | .cast =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.castTo (regE 1) f32Ty) ]
  | .bitcast =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "bitcast" [regE 1]) ]
  | .copy =>
      [ Stmt.letDecl 0 none (regE 1) ]
  -- Control flow
  | .branch =>
      [ Stmt.ifThenElse (regE 0) [] [] ]
  | .loopOp =>
      [ Stmt.loopBody [] ]
  | .breakOp =>
      [ Stmt.breakStmt ]
  | .barrier =>
      [ Stmt.callStmt "workgroupBarrier" [] ]
  -- Calls
  | .mathCall =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "sqrt" [regE 1]) ]
  | .deviceCall =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "device_helper_0" [regE 1]) ]
  -- Atomics
  | .atomicOp =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "atomicAdd"
            [ Expr.addrOfIdx "field" (regE 1), regE 2 ]) ]
  | .atomicCas =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "atomicCompareExchangeWeak"
            [ Expr.addrOfIdx "field" (regE 1), regE 2, regE 3 ]) ]
  -- Wave / subgroup
  | .waveShuffle =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "subgroupBroadcast" [regE 1, regE 2]) ]
  | .waveBallot =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "subgroupBallot" [regE 1]) ]
  | .waveAny =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "subgroupAny" [regE 1]) ]
  | .waveAll =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "subgroupAll" [regE 1]) ]
  -- Vector / matrix
  | .vecConstruct =>
      [ Stmt.letDecl 0 (some (Ty.vec Scalar.f32 4))
          (Expr.vecCtor Scalar.f32 4 [regE 1, regE 2, regE 3, regE 0]) ]
  | .vecExtract =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.indexBuf "v0" (regE 1)) ]
  | .matMul =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "fma" [regE 1, regE 2, regE 3]) ]
  | .cooperativeMMA =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "fma" [regE 1, regE 2, regE 3]) ]
  -- Texture
  | .textureSample2D =>
      [ Stmt.letDecl 0 (some (Ty.vec Scalar.f32 4))
          (Expr.callBuiltin "textureSampleLevel" [regE 1, regE 2, regE 3]) ]
  | .textureSample3D =>
      [ Stmt.letDecl 0 (some (Ty.vec Scalar.f32 4))
          (Expr.callBuiltin "textureSampleLevel" [regE 1, regE 2, regE 3]) ]
  | .textureWrite2D =>
      [ Stmt.callStmt "textureStore" [regE 0, regE 1, regE 2] ]
  | .textureSize =>
      [ Stmt.letDecl 0 (some (Ty.vec Scalar.u32 2))
          (Expr.callBuiltin "textureDimensions" [regE 1]) ]
  -- IDs
  | .quarkId       =>
      [ Stmt.letDecl 0 none (Expr.identE "_quark_id") ]
  | .quarkCount    =>
      [ Stmt.letDecl 0 none (Expr.identE "_quark_count") ]
  | .protonId      =>
      [ Stmt.letDecl 0 none (Expr.identE "_proton_id") ]
  | .nucleusId     =>
      [ Stmt.letDecl 0 none (Expr.identE "_nucleus_id") ]
  | .protonSize    =>
      [ Stmt.letDecl 0 none (Expr.identE "_proton_size") ]
  -- Misc
  | .constLit =>
      [ Stmt.letDecl 0 none (Expr.litI32 0) ]
  | .dispatch =>
      [ Stmt.callStmt "dispatchWorkgroupsIndirect" [regE 0, regE 1] ]
  | .countTrailingZeros =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "countTrailingZeros" [regE 1]) ]
  | .countLeadingZeros =>
      [ Stmt.letDecl 0 (some f32Ty)
          (Expr.callBuiltin "countLeadingZeros" [regE 1]) ]

-- ════════════════════════════════════════════════════════════════════
-- T420 — every per-op pattern is structurally well-formed
-- ════════════════════════════════════════════════════════════════════

/-- True iff every tag in the enumeration produces a `List Stmt`
    whose `Stmt.wellFormed` predicate holds pointwise. -/
def allPatternsWellFormed (tags : List KernelOpTag) : Bool :=
  tags.all (fun t => (opPattern t).all Stmt.wellFormed)

/-- **T420 — wgsl_op_patterns_well_formed**: every WGSL statement
    list our JIT emitter produces, modelled by `opPattern`, is
    structurally well-formed per `Quanta.Wgsl.Grammar`.

    The proof discharges by `native_decide`: the predicate is a
    ground `Bool` over a 40-element tag enumeration; the kernel
    evaluates each `Stmt.wellFormed` and conjuncts them.

    With T420 + the kernel-level Source assembly (B.4) + A12, the
    T410 axiom collapses to a Lean theorem: "every kernel
    `emit_wgsl_jit` produces is structurally well-formed → its
    serialization is WGSL-spec-well-formed." -/
theorem t420_wgsl_op_patterns_well_formed :
    allPatternsWellFormed allKernelOpTags = true := by
  native_decide

end Quanta.Wgsl.OpPatterns
