/-
# WASM → KernelOps lowering — Lean specification (step 059)

Mirrors the production translator at
`crates/quanta-wasm-lowering/src/lower.rs`. The Rust pass simulates
WASM's stack with a richer symbolic abstract domain (buffer pointers,
scaled indices, etc.) so it can recognize buffer-access patterns.
The Lean port now carries the full `SymVal` alphabet on the stack;
existing slice-1/2/3 ops push only `.reg r .u32` and pop only on
that shape. The richer SymVals (`bufferPtr`, `scaledIdx`,
`i32ConstSym`, `bufferAccess`) are introduced by buffer-pattern
recognition arms (slice-4 step 7).

State carried during lowering:
* `nextReg` — fresh-register counter (matches `LowerCtx::next_reg`).
* `stack`  — list of `SymVal`s, top first.
* `localReg` — `localIdx → Reg` map (`stable_reg` in the Rust pass).
  Allocated lazily on first read; later writes emit `KernelOp.copy`.
* `localTy` — Quanta IR scalar type for each local (slice 1 only
  models `i32`, but the field is here so future slices can lift to
  `f32`/etc.).

The lowering function returns the final state plus the emitted
`KernelOp` list. Slice 1 covers `i32Const`, `i32Add`, `i32Sub`,
`i32Mul`, the bitwise i32 family, `localGet`, `localSet`,
`localTee`, `wreturn`. Everything outside the subset returns `none`,
matching `LoweringError::UnsupportedOp` in production.
-/

import Quanta.Wasm.Syntax
import Quanta.Wasm.Structured
import Quanta.KOps.Syntax

namespace Quanta.Wasm

open Quanta.KOps (KernelOp Reg ConstValue Scalar BinOp CmpOp)

-- ════════════════════════════════════════════════════════════════════
-- Lowering state
-- ════════════════════════════════════════════════════════════════════

/-- Symbolic stack value during lowering. Mirrors `SymVal` in
    `crates/quanta-wasm-lowering/src/lower.rs`. The lowering pass
    simulates WASM's stack with these abstract values to recognize
    canonical buffer-access patterns.

    For slices 1-3, every stack slot was a plain `Reg`. From slice 4
    on, we track the richer alphabet so the `<buffer_ptr> + <byte_offset> .load`
    pattern can be reduced to `KernelOp.Load { field, index }`. -/
inductive SymVal where
  /-- A virtual register holding a value of given scalar type. -/
  | reg          (r : Reg) (ty : Scalar)
  /-- Pointer to buffer slot N (tracked from WASM parameter analysis).
      `BufferPtr` slots never carry a runtime register because the
      lowering folds the `+ byte_offset .load` chain into a typed
      `Load` directly. -/
  | bufferPtr    (slot : Nat)
  /-- Result of `<base_reg> << log2(scale)` — the canonical
      "element index → byte offset" encoding rustc emits. `scale` is
      the element size in bytes. -/
  | scaledIdx    (base : Reg) (scale : Nat)
  /-- A WASM i32 constant — kept symbolically so we can recognize
      `i32.const 2; i32.shl` as a left-shift-by-2 (= scale by 4). -/
  | i32ConstSym  (n : Int)
  /-- `BufferPtr(slot) + ScaledIdx{base, scale}` — emitted by
      recognizing the canonical `<ptr> <byte_offset> i32.add` pattern.
      Consumed by the next `i32.load` / `i32.store` op into a typed
      `KernelOp.Load` / `Store`. -/
  | bufferAccess (slot : Nat) (base : Reg) (scale : Nat)
  deriving Repr, DecidableEq

namespace SymVal

/-- Registers referenced by a SymVal — used by the freshness and
    alias-free invariants in `Quanta.Wasm.Preservation`. The non-reg
    SymVals (`bufferPtr`, `i32ConstSym`) carry no register, so they
    contribute the empty list. -/
def regs : SymVal → List Reg
  | .reg r _              => [r]
  | .bufferPtr _          => []
  | .scaledIdx base _     => [base]
  | .i32ConstSym _        => []
  | .bufferAccess _ base _ => [base]

end SymVal

structure LowerState where
  nextReg  : Nat
  /-- Symbolic stack — top first. Slice 4 lifts this to `List SymVal`
      so buffer-pattern recognition (the `<bufferPtr> + <byte_offset>
      i32.add | i32.load` chain) can fold into a typed `KernelOp.Load`.
      Every value-typed slot ops outside the buffer-pattern arms push
      `SymVal.reg r .u32`; the richer SymVals only appear in the
      transient sequence introduced by buffer-pattern recognition. -/
  stack    : List SymVal
  /-- `localIdx → Reg`. Stored as an association list keyed by `Nat`. -/
  localReg : List (Nat × Reg)
  /-- `localIdx → Scalar` — kept in lockstep with `localReg`. -/
  localTy  : List (Nat × Scalar)
  /-- `localIdx → bufferSlot`. Records which locals were seeded as
      `#[quanta::shared]` buffer-pointer parameters (production: the
      `ParamKind::BufferRead | BufferWrite` arms in
      `crates/quanta-wasm-lowering/src/lower.rs` `LowerCtx::new`).
      Used by `localGet` to push `SymVal.bufferPtr slot` instead of
      reading a stable register, so the subsequent `<scaledIdx>
      i32.add | i32.load` chain can fold into a typed
      `KernelOp.Load`. -/
  bufferSlots : List (Nat × Nat)
  /-- `localIdx → Reg` for the **current** (post-set) binding inside
      a structured-control frame. Updated by every `localSet`/
      `localTee`: a fresh register is allocated, and `currentReg[i]`
      becomes that fresh reg. `localGet` reads `currentReg` first,
      `localReg` (the stable merge-anchor) as fallback for locals
      never written in the current frame.

      At frame close (wif / wloop), a `merge_locals_post_frame`
      helper diffs `currentReg` against the snapshot taken at frame
      entry and emits `Copy { dst := localReg[i], src := currentReg[i] }`
      for every modified local; then `currentReg` is reset by
      removing those entries so post-frame reads see the merged
      value via `localReg`.

      Mirrors production's `LocalInfo.val = Some(SymVal::Reg(fresh, ty))`
      after each set (`write_local_via_copy` in
      `crates/quanta-wasm-lowering/src/lower.rs`, lines 625-685).
      Production overloads `val` for this purpose; the Lean spec
      keeps a separate map so the existing `localReg`-as-stable
      proofs survive. -/
  currentReg : List (Nat × Reg)
  deriving Repr, DecidableEq

def LowerState.empty : LowerState :=
  { nextReg := 0, stack := [], localReg := [], localTy := [],
    bufferSlots := [], currentReg := [] }

namespace LowerState

def alloc (s : LowerState) : Reg × LowerState :=
  (s.nextReg, { s with nextReg := s.nextReg + 1 })

/-- Push a plain `.reg r .u32` SymVal onto the stack. The default
    push every value-producing op uses; buffer-pattern recognition
    uses `pushSym` directly with a richer SymVal. -/
def push (s : LowerState) (r : Reg) : LowerState :=
  { s with stack := SymVal.reg r .u32 :: s.stack }

/-- Push a generic SymVal onto the stack. Used by the buffer-pattern
    recognition arms (slice-4 step 7) — produces e.g. `bufferPtr`,
    `scaledIdx`, `bufferAccess` entries. -/
def pushSym (s : LowerState) (sv : SymVal) : LowerState :=
  { s with stack := sv :: s.stack }

/-- Pop the top stack slot as a plain register. Succeeds only when
    the top is `.reg r _`; richer SymVals are consumed by the
    buffer-pattern arms via `popSym` instead. -/
def pop (s : LowerState) : Option (Reg × LowerState) :=
  match s.stack with
  | SymVal.reg r _ :: rs => some (r, { s with stack := rs })
  | _                    => none

/-- Pop any SymVal off the top — used by buffer-pattern recognition
    arms that need to inspect the symbolic shape (slice-4 step 7). -/
def popSym (s : LowerState) : Option (SymVal × LowerState) :=
  match s.stack with
  | []        => none
  | sv :: rs  => some (sv, { s with stack := rs })

def lookupLocal (s : LowerState) (i : Nat) : Option Reg :=
  s.localReg.find? (fun p => p.fst = i) |>.map Prod.snd

def lookupLocalTy (s : LowerState) (i : Nat) : Option Scalar :=
  s.localTy.find? (fun p => p.fst = i) |>.map Prod.snd

/-- Lookup which buffer slot a local was seeded as. `none` for a
    plain scalar local; `some slot` for a `#[quanta::shared]` buffer
    parameter. Used by `localGet` to dispatch into the buffer-pattern
    arm instead of the generic stable-reg read. -/
def lookupBufferSlot (s : LowerState) (i : Nat) : Option Nat :=
  s.bufferSlots.find? (fun p => p.fst = i) |>.map Prod.snd

def setLocalReg (s : LowerState) (i : Nat) (r : Reg) (ty : Scalar) : LowerState :=
  let regs' := (i, r) :: s.localReg.filter (fun p => p.fst ≠ i)
  let tys'  := (i, ty) :: s.localTy.filter (fun p => p.fst ≠ i)
  { s with localReg := regs', localTy := tys' }

/-- Lookup the per-frame current-binding register for local `i`.
    `none` means the local hasn't been written in the current frame;
    `localGet` then falls back to `localReg[i]` (the stable merge
    anchor). Mirrors production's `locals[idx].val = SymVal::Reg(fresh, _)`
    after every `write_local_via_copy`. -/
def lookupCurrentReg (s : LowerState) (i : Nat) : Option Reg :=
  s.currentReg.find? (fun p => p.fst = i) |>.map Prod.snd

/-- Update the per-frame current-binding register for local `i` to
    `r`. Overwrites any prior entry for `i`. Frame-close fixups
    (wif/wloop merge) emit `Copy { dst := localReg[i], src := r }`
    for each `(i, r) ∈ currentReg` then reset currentReg by
    removing those entries. -/
def setCurrentReg (s : LowerState) (i : Nat) (r : Reg) : LowerState :=
  let regs' := (i, r) :: s.currentReg.filter (fun p => p.fst ≠ i)
  { s with currentReg := regs' }

/-- Zero literal for a scalar type, as production's
    `write_local_via_copy` emits it for the frame-0 pre-declaration.
    Production switches on the local's stable type: integer-unsigned →
    `ConstValue::U32(0)`, integer-signed → `ConstValue::I32(0)`, etc.
    Slice 1 only reaches `.u32` (the `localSet` arm's `ty` defaults to
    `.u32`), so the `.u32` case is the live one; the others mirror
    production for faithfulness. -/
def zeroConst : Scalar → ConstValue
  | .i8 | .i16 | .i32 | .i64 => .i32 0
  | _ => .u32 0

/-- Materialize a `SymVal` into a real `Reg` + the ops needed to
    produce that reg's value. Mirrors production's `commit()`:
    * `.reg r _` is already a register — no ops, no alloc.
    * `.i32ConstSym n` allocates a fresh reg and emits `.const r ...`
      (the const op the eager `i32.const` arm used to emit).
    * Address SymVals (`.bufferPtr`, `.scaledIdx`, `.bufferAccess`)
      cannot commit to a value reg — they're consumed by the buffer-
      pattern load/store arms instead. (Future: scaledIdx could
      commit by emitting a shift; deferred until needed.) -/
def commit (s : LowerState) (sv : SymVal) : Option (Reg × LowerState × List KernelOp) :=
  match sv with
  | .reg r _              => some (r, s, [])
  | .i32ConstSym n        =>
      -- Production tags a committed const I32 (lower.rs `commit()`
      -- returns `ScalarType::I32`). The materialized value is the
      -- WRAPPED 32-bit pattern as a non-negative Int, so `evalConst`
      -- yields `vI32 bits` — coherent with the `.reg dst .i32` encoding
      -- of `wI32 (UInt32.ofNat n.toNat)`. The i32-tag SEED (V8-#2).
      let (dst, s1) := s.alloc
      some (dst, s1, [.const dst (.i32 ((UInt32.ofNat n.toNat).toNat))])
  | .bufferPtr _          => none
  | .scaledIdx _ _        => none
  | .bufferAccess _ _ _   => none

/-- The scalar type a committed `SymVal` carries — mirrors production's
    `commit()` return type. A `.reg r ty` keeps `ty`; an `i32ConstSym`
    materializes `.i32`. Used by `lowerI32Bin` to derive the binop
    result type the way production does. -/
def commitTy : SymVal → Scalar
  | .reg _ ty             => ty
  | .i32ConstSym _        => .i32
  | .bufferPtr _          => .u32
  | .scaledIdx _ _        => .u32
  | .bufferAccess _ _ _   => .u32

/-- Binop result type on two committed operands: matched types pass
    through, mixed falls to `.i32` (production `ty = if ty_a == ty_b
    { ty_a } else { I32 }`). -/
def binResultTy (a b : SymVal) : Scalar :=
  if commitTy a = commitTy b then commitTy a else .i32

end LowerState

-- ════════════════════════════════════════════════════════════════════
-- Per-instruction lowering
-- ════════════════════════════════════════════════════════════════════

/-- Allocate a fresh register, emit a single op writing into it,
    push it on the symbolic stack. The composite move every
    arithmetic / const lowering performs. -/
@[inline] def freshAndPush (s : LowerState) (mk : Reg → KernelOp) : Reg × LowerState × List KernelOp :=
  let (r, s1) := s.alloc
  let s2 := s1.push r
  (r, s2, [mk r])

/-- Lower a single i32 binary op: pop two SymVals, materialize each to
    a real register via `commit` (no-op for `.reg`, fresh-alloc + const
    op for `.i32ConstSym`, refusal for the address SymVals), allocate a
    result reg, emit the matching `binOp`, push the result. `none` on
    stack underflow or on either operand being a non-committable
    SymVal (the buffer-pattern arms intercept those before this fires).

    Mirrors production `lower.rs` `RawInstr::I32Add`: pop b then a,
    commit a then b, alloc dst, emit. The op order in the emitted
    list is `opsA ++ opsB ++ [binOp]` so that ra/rb are written before
    the binOp reads them. -/
def lowerI32Bin (s : LowerState) (op : BinOp) : Option (LowerState × List KernelOp) := do
  let (svb, s1) ← s.popSym
  let (sva, s2) ← s1.popSym
  let (ra, s3, opsA) ← s2.commit sva
  let (rb, s4, opsB) ← s3.commit svb
  let (dst, s5) := s4.alloc
  -- Result type derived from operand types (production's rule). All-u32
  -- subset keeps `.u32`; a committed i32 const operand makes it `.i32`.
  let ty : Scalar := LowerState.binResultTy sva svb
  let s6 := s5.pushSym (.reg dst ty)
  pure (s6, opsA ++ opsB ++ [.binOp dst ra rb op ty])

/-- Lower a single i32 comparison. KOps `Cmp` produces a `vBool`, but
    WASM's `i32.{eq,ne,lt,le,gt,ge}` push an `wI32 0/1` — so we emit
    `Cmp` followed by a `Cast bool→u32` to re-enter the u32 alphabet
    before the value flows back onto the stack as `.reg _ .u32`.

    Production's lowering pushes a `.reg _ .bool` slot and casts at
    consume-time via `commit()`. The Lean port casts eagerly here to
    keep `WasmValue.encodes` single-shape (always `.u32`) and avoid a
    cascade through every existing per-op preservation proof. The
    end-to-end IR shape is identical (cmp + cast); only the placement
    of the cast in the lowering pass differs.

    Operands flow through `popSym + commit` (same as `lowerI32Bin`),
    so an `i32ConstSym` operand materializes via a const op prefix. -/
def lowerI32Cmp (s : LowerState) (op : CmpOp) : Option (LowerState × List KernelOp) := do
  let (svb, s1) ← s.popSym
  let (sva, s2) ← s1.popSym
  let (ra, s3, opsA) ← s2.commit sva
  let (rb, s4, opsB) ← s3.commit svb
  let (boolReg, s5) := s4.alloc
  let (dst, s6) := s5.alloc
  let s7 := s6.push dst
  pure (s7, opsA ++ opsB ++ [.cmp boolReg ra rb op .bool, .cast dst boolReg .bool .u32])

/-- Lower `i32.shl`. Production fast-path: if the popped operands are
    `<reg base> <i32ConstSym k>`, fold to `SymVal.scaledIdx base
    (1 <<< k)` with no IR emitted (the canonical "element index → byte
    offset" pattern rustc emits). Otherwise fall through to the
    generic `lowerI32Bin .shl`. Mirrors `RawInstr::I32Shl` in
    `crates/quanta-wasm-lowering/src/lower.rs`. -/
def lowerI32Shl (s : LowerState) : Option (LowerState × List KernelOp) :=
  match s.stack with
  | .i32ConstSym k :: .reg base _ :: rest =>
      let scale := 1 <<< k.toNat
      some ({ s with stack := .scaledIdx base scale :: rest }, [])
  | _ => lowerI32Bin s .shl

/-- `log2 n` for a power-of-two `n` (mirrors production's
    `(scale / s2).trailing_zeros()`): the shift amount that recovers a
    power-of-two scale ratio. Defined for all `n`; only the power-of-two
    case is exercised, where `1 <<< log2 n = n`. -/
def log2 : Nat → Nat
  | 0 => 0
  | 1 => 0
  | (n + 2) => 1 + log2 ((n + 2) / 2)
decreasing_by exact Nat.div_lt_self (by omega) (by omega)

/-- Lower `i32.add`. Production fast-paths (`RawInstr::I32Add` in
    `lower.rs`), tried in order on the top two symbolic stack slots
    (either operand order, mirroring production's `match (a, b)`):

    * `<bufferPtr slot> + <scaledIdx base scale>` → `bufferAccess slot
      base scale`, NO IR (the canonical pointer + element-offset fold).
    * **same-scale chained add**: `<bufferAccess slot base scale> +
      <scaledIdx b2 scale>` (equal scale) → emit `Add(dst, base, b2)`,
      push `bufferAccess slot dst scale`. Combines two runtime indices.
    * **rescale chained add**: `<bufferAccess slot base scale> +
      <scaledIdx b2 s2>` with `scale > s2`, `s2 ∣ scale`, `scale / s2`
      a power of two → emit `Const(shift, log2(scale/s2))`,
      `Shl(scaled, base, shift)`, `Add(dst, scaled, b2)`, push
      `bufferAccess slot dst s2` at the SMALLER scale.
    * **const-offset chained add**: `<bufferAccess slot base scale> +
      <i32ConstSym c>` with `scale ∣ c` → emit `Const(off, c/scale)`
      [I32-tagged], `Add(dst, base, off)`, push `bufferAccess slot dst
      scale`. Folds a precomputed constant element offset.

    These chained arms are the address shapes rustc emits when it has
    precomputed part of a byte offset and leaves the rest runtime
    (`out + block_off + pos_off`). Anything else falls through to the
    generic `lowerI32Bin .add`. -/
def lowerI32Add (s : LowerState) : Option (LowerState × List KernelOp) :=
  match s.stack with
  -- ── pointer + element-offset fold (no IR) ──
  | .scaledIdx base scale :: .bufferPtr slot :: rest =>
      some ({ s with stack := .bufferAccess slot base scale :: rest }, [])
  | .bufferPtr slot :: .scaledIdx base scale :: rest =>
      some ({ s with stack := .bufferAccess slot base scale :: rest }, [])
  -- ── same-scale chained add ──
  | .scaledIdx b2 s2 :: .bufferAccess slot base scale :: rest =>
      if scale = s2 then
        let (dst, s1) := s.alloc
        some ({ s1 with stack := .bufferAccess slot dst scale :: rest },
              [.binOp dst base b2 .add .u32])
      else if scale > s2 ∧ scale % s2 = 0 ∧ (1 <<< log2 (scale / s2)) = scale / s2 then
        let (shift, s1) := s.alloc
        let (scaled, s2') := s1.alloc
        let (dst, s3) := s2'.alloc
        some ({ s3 with stack := .bufferAccess slot dst s2 :: rest },
              [.const shift (.u32 (UInt32.ofNat (log2 (scale / s2)))),
               .binOp scaled base shift .shl .u32,
               .binOp dst scaled b2 .add .u32])
      else lowerI32Bin s .add
  | .bufferAccess slot base scale :: .scaledIdx b2 s2 :: rest =>
      if scale = s2 then
        let (dst, s1) := s.alloc
        some ({ s1 with stack := .bufferAccess slot dst scale :: rest },
              [.binOp dst base b2 .add .u32])
      else if scale > s2 ∧ scale % s2 = 0 ∧ (1 <<< log2 (scale / s2)) = scale / s2 then
        let (shift, s1) := s.alloc
        let (scaled, s2') := s1.alloc
        let (dst, s3) := s2'.alloc
        some ({ s3 with stack := .bufferAccess slot dst s2 :: rest },
              [.const shift (.u32 (UInt32.ofNat (log2 (scale / s2)))),
               .binOp scaled base shift .shl .u32,
               .binOp dst scaled b2 .add .u32])
      else lowerI32Bin s .add
  -- ── const-offset chained add ──
  | .i32ConstSym c :: .bufferAccess slot base scale :: rest =>
      if scale ≠ 0 ∧ c % (scale : Int) = 0 then
        let (off, s1) := s.alloc
        let (dst, s2') := s1.alloc
        some ({ s2' with stack := .bufferAccess slot dst scale :: rest },
              [.const off (.i32 (c / (scale : Int))),
               .binOp dst base off .add .u32])
      else lowerI32Bin s .add
  | .bufferAccess slot base scale :: .i32ConstSym c :: rest =>
      if scale ≠ 0 ∧ c % (scale : Int) = 0 then
        let (off, s1) := s.alloc
        let (dst, s2') := s1.alloc
        some ({ s2' with stack := .bufferAccess slot dst scale :: rest },
              [.const off (.i32 (c / (scale : Int))),
               .binOp dst base off .add .u32])
      else lowerI32Bin s .add
  | _ => lowerI32Bin s .add

/-- Lower `i32.load`. Only succeeds on a `BufferAccess { scale := 4 }`
    address — the buffer-pattern arms above (`localGet` for buffer
    locals, `i32.shl` for scaledIdx, `i32.add` for bufferAccess) are
    the only producers of that shape, so a successful match certifies
    the source is a `<bufferPtr>+<i*4>` chain. Allocates a fresh reg
    and emits `KernelOp.Load { field := slot, index := base, ty := .u32 }`.
    `none` on any other address shape (no plain-address fallback —
    matches production `lower.rs::I32Load`'s `UnsupportedOp` arm). -/
def lowerI32Load (s : LowerState) : Option (LowerState × List KernelOp) :=
  match s.stack with
  | .bufferAccess slot base 4 :: rest =>
      let (dst, s1) := s.alloc
      some ({ s1 with stack := .reg dst .u32 :: rest }, [.load dst slot base .u32])
  | _ => none

/-- Lower `i32.store`. Pops val (top), then addr; commits val into a
    real register (`commit` materializes a `.i32ConstSym` source via a
    prefix const op), then emits `KernelOp.Store` against the
    `BufferAccess { scale := 4 }` addr. `none` on any other address
    shape — matches `lower.rs::I32Store`. -/
def lowerI32Store (s : LowerState) : Option (LowerState × List KernelOp) := do
  let (sv_val, s1) ← s.popSym
  let (sv_addr, s2) ← s1.popSym
  let (src, s3, opsCommit) ← s2.commit sv_val
  match sv_addr with
  | .bufferAccess slot base 4 =>
      pure (s3, opsCommit ++ [.store slot base src .u32])
  | _ => none

/-- Lower one WASM instruction. Returns the new state and the emitted
    KOps. `none` for ops outside the subset (matches the production
    pass's `UnsupportedOp` error). -/
def lowerInstr (s : LowerState) : WasmInstr → Option (LowerState × List KernelOp)
  -- Constants. WASM `i32.const n` pushes the constant *symbolically*
  -- as `SymVal.i32ConstSym n` and emits no IR ops. The const is
  -- materialized later, either by a buffer-pattern arm consuming it
  -- (e.g., `<reg> <i32ConstSym k> i32.shl` → `ScaledIdx { base, 1<<k }`,
  -- no const op needed) or by a generic consumer that demands a real
  -- register, which calls `commit` to emit the const op then. This
  -- matches the production translator's pull-based materialization;
  -- before this change, every i32.const eagerly emitted a `.const`
  -- op, which would defeat the buffer-pattern recognition.
  | .i32Const n =>
      some ({ s with stack := .i32ConstSym n :: s.stack }, [])
  -- Locals
  --
  -- `localGet` allocates a fresh register and emits `Copy { dst, src }`
  -- copying from the local's stable_reg into the fresh reg, then pushes
  -- the fresh reg. This breaks register aliasing between the symbolic
  -- stack and the local's stable_reg — necessary so that a subsequent
  -- `localSet` writing to the stable_reg doesn't clobber stack-aliased
  -- copies of an older value. The production translator shares the
  -- stable_reg directly; the Lean port differs here to make the
  -- preservation proof tractable. The semantic effect is identical
  -- (one extra IR copy); production likely doesn't hit the alias bug
  -- because rustc-emitted WASM avoids the pattern.
  | .localGet i =>
      -- Buffer-typed locals (seeded from `#[quanta::shared]`
      -- parameters) push their `SymVal.bufferPtr slot` symbolically:
      -- no register allocated, no IR emitted. The subsequent
      -- `<scaledIdx> i32.add | i32.load` chain folds into a typed
      -- `KernelOp.Load` against `slot`. Mirrors production
      -- `LowerCtx::process_instruction`'s LocalGet path which reads
      -- the local's initial `Some(SymVal::BufferPtr(slot))` if any.
      match s.lookupBufferSlot i with
      | some slot =>
          some (s.pushSym (.bufferPtr slot), [])
      | none => do
          -- Read from currentReg first (the post-write binding
          -- inside the current frame). Fall back to localReg (the
          -- stable merge anchor) for locals never written in this
          -- frame. Allocates a fresh reg and emits Copy { dst :=
          -- fresh, src := source } to break aliasing — necessary
          -- so a subsequent localSet writing to the source-reg
          -- doesn't clobber stack-aliased copies of an older value.
          let source ← (s.lookupCurrentReg i).orElse (fun _ => s.lookupLocal i)
          let (fresh, s1) := s.alloc
          -- Push `fresh` at the local's recorded type: an i32-set local
          -- reads back `.i32`, a u32 local `.u32` (default for unset).
          -- The `.copy` preserves the value, so the fresh reg holds the
          -- same tag the source did (V8-#2 local read).
          let ty : Scalar := (s.lookupLocalTy i).getD .u32
          let s2 := s1.pushSym (.reg fresh ty)
          pure (s2, [.copy fresh source])
  | .localSet i => do
      -- popSym + commit (matches binop/cmp/localTee): a popped
      -- `.i32ConstSym` materializes via a const-op prefix, while
      -- buffer SymVals refuse at `commit` (and never reach localSet
      -- in well-formed code — the buffer-pattern arms intercept
      -- them earlier).
      let (sv, s1) ← s.popSym
      let (src, s2, opsCommit) ← s1.commit sv
      -- The local takes the COMMITTED value's type (`commitTy sv`): an
      -- i32-const set tags the local `.i32`, a u32 reg keeps `.u32`.
      -- This is what makes a later `local.get` of an i32-set local
      -- read back at the right tag (V8-#2 local surface).
      let ty : Scalar := LowerState.commitTy sv
      -- Dual-Copy pattern (mirrors production's `write_local_via_copy`):
      --   1. allocate `fresh` reg — the new per-set binding
      --   2. emit Copy { dst := fresh, src }
      --   3. emit Copy { dst := stable_reg, src := fresh }
      --   4. setCurrentReg[i] := fresh so subsequent localGet in this
      --      frame sees the new binding
      --   5. stable_reg stays the merge anchor (set on first write)
      let (fresh, s3) := s2.alloc
      -- Frame-0 zero-init: production's `write_local_via_copy` inserts
      -- a `Const fresh (zeroConst ty)` at the function-frame head before
      -- the dual-Copy, so the Metal/WGSL emitters get a `uint rN = 0u;`
      -- declaration to assign into. The very next `Copy fresh src`
      -- overwrites `fresh`, so the const is a dead write — semantically
      -- inert. (Production routes it to frame 0; the flat op-list model
      -- here places it inline. Placement differs, abstract effect does
      -- not, because `fresh` is freshly allocated and immediately
      -- re-bound.)
      let zinit : KernelOp := .const fresh (LowerState.zeroConst ty)
      match s3.lookupLocal i with
      | some stable =>
          -- Local already has a stable reg from a prior set; keep it.
          let s4 := (s3.setLocalReg i stable ty).setCurrentReg i fresh
          pure (s4, opsCommit ++ [zinit, .copy fresh src, .copy stable fresh])
      | none =>
          -- First write: allocate the local's stable reg too.
          let (stable, s4) := s3.alloc
          let s5 := (s4.setLocalReg i stable ty).setCurrentReg i fresh
          pure (s5, opsCommit ++ [zinit, .copy fresh src, .copy stable fresh])
  | .localTee i => do
      -- `local.tee` = `local.set i` followed by `local.get i`. The
      -- `localGet` half breaks aliasing by emitting a Copy into a
      -- fresh register, so the post-tee stack value is `post_fresh`,
      -- not the local's stable register. Same alias-free invariant
      -- as `localGet`.
      --
      -- popSym + commit to materialize the popped SymVal into a real
      -- register (matches localSet / binop / cmp). `i32ConstSym`
      -- emits a const-op prefix in `opsCommit`.
      let (sv, s1) ← s.popSym
      let (src, s2, opsCommit) ← s1.commit sv
      let ty : Scalar := LowerState.commitTy sv
      -- Dual-Copy + tee re-read pattern:
      --   1. allocate `fresh` reg — new per-set binding
      --   2. emit Copy { dst := fresh, src }
      --   3. emit Copy { dst := stable, src := fresh } — keep stable in sync
      --   4. setCurrentReg[i] := fresh
      --   5. The tee re-read: emit Copy { dst := post_fresh, src := fresh }
      --      and push post_fresh on the stack (alias-free)
      let (fresh, s3) := s2.alloc
      -- Same frame-0 zero-init as `localSet` (production routes
      -- `local.tee` through the same `write_local_via_copy`).
      let zinit : KernelOp := .const fresh (LowerState.zeroConst ty)
      match s3.lookupLocal i with
      | some stable =>
          let s4 := (s3.setLocalReg i stable ty).setCurrentReg i fresh
          let (post_fresh, s5) := s4.alloc
          -- The tee re-read copies `fresh` (which holds the committed
          -- value at tag `ty`) into `post_fresh`, so push it at `ty`.
          pure (s5.pushSym (.reg post_fresh ty),
                opsCommit ++ [zinit, .copy fresh src, .copy stable fresh,
                              .copy post_fresh fresh])
      | none =>
          let (stable, s4) := s3.alloc
          let s5 := (s4.setLocalReg i stable ty).setCurrentReg i fresh
          let (post_fresh, s6) := s5.alloc
          pure (s6.pushSym (.reg post_fresh ty),
                opsCommit ++ [zinit, .copy fresh src, .copy stable fresh,
                              .copy post_fresh fresh])
  -- i32 arithmetic. `i32.shl` and `i32.add` carry the buffer-pattern
  -- recognition fast-paths; the others dispatch directly to the
  -- generic binop lowering.
  | .i32Add  => lowerI32Add s
  | .i32Sub  => lowerI32Bin s .sub
  | .i32Mul  => lowerI32Bin s .mul
  | .i32And  => lowerI32Bin s .bAnd
  | .i32Or   => lowerI32Bin s .bOr
  | .i32Xor  => lowerI32Bin s .bXor
  | .i32Shl  => lowerI32Shl s
  | .i32ShrU => lowerI32Bin s .shr
  | .i32DivU => lowerI32Bin s .div
  | .i32RemU => lowerI32Bin s .rem
  -- Memory: typed loads/stores against `#[quanta::shared]` buffers.
  -- Only the buffer-pattern address shape (BufferAccess from the
  -- recognized `<bufferPtr>+<i*4>` chain) is accepted — production
  -- refuses any other shape with `UnsupportedOp`.
  | .i32Load _offset _align  => lowerI32Load s
  | .i32Store _offset _align => lowerI32Store s
  -- i32 comparisons (unsigned only — signed lift in a later slice).
  | .i32Eq  => lowerI32Cmp s .eq
  | .i32Ne  => lowerI32Cmp s .ne
  | .i32LtU => lowerI32Cmp s .lt
  | .i32LeU => lowerI32Cmp s .le
  | .i32GtU => lowerI32Cmp s .gt
  | .i32GeU => lowerI32Cmp s .ge
  -- Return / nop / drop. (`wreturn` is intercepted by `lowerInstrs`
  -- with frame context; this arm only serves standalone `lowerInstr`
  -- uses and the per-op lemmas about them.)
  | .wreturn => some (s, [])
  | .nop     => some (s, [])
  | .drop    => do
      -- `popSym` (not `pop`): accept any SymVal on top, including
      -- `i32ConstSym` and the buffer-pattern address SymVals. Drop
      -- emits no IR — discards the popped value with no
      -- materialization.
      let (_, s1) ← s.popSym
      pure (s1, [])
  -- Outside slice 1 — refused, matching `UnsupportedOp` in production.
  | _ => none

/-- True if any frame strictly above the target depth (i.e. inside
    the current emission scope's view of the frame stack) is a
    `loopK`. Mirrors `LowerCtx::has_loop_between_top_and_depth` in
    `lower.rs`. Used by `br`/`brIf` to decide whether the branch
    crosses a loop boundary — if so, we emit `KernelOp.breakOp`
    rather than the (unimplemented) redirect-chain approach. -/
def hasLoopAbove (frames : List FrameKind) (depth : Nat) : Bool :=
  (frames.take depth).any (· = .loopK)

/-- Number of `loopK` frames strictly above the target depth.
    Mirrors `LowerCtx::loops_between_top_and_depth` in `lower.rs`.
    Production gives the single-loop-crossing-to-Block case the
    exit-flag treatment (`emit_loop_crossing_exit`); only multi-loop
    crossings and non-Block targets keep the label-lossy `Break`. -/
def loopsAbove (frames : List FrameKind) (depth : Nat) : Nat :=
  ((frames.take depth).filter (· = .loopK)).length

/-- Does an op list end in an unconditional `.breakOp`? Mirrors the
    `matches!(else_ops.last(), Some(KernelOp::Break))` half of
    `tail_exits_or_continues` in `reconstruct_loop_backedges`
    (lower.rs): a backedge wrap whose lowered tail already exits
    must not append a second exit `Break`. -/
def endsInBreak : List KernelOp → Bool
  | [] => false
  | [.breakOp] => true
  | [_] => false
  | _ :: op :: rest => endsInBreak (op :: rest)

/-- Scan the instruction tail after a depth-0 loop backedge for a
    same-level branch back to the enclosing loop: an explicit
    continue (`br 0` — production's `continue_pos` recording in the
    `RawInstr::Br` loop-target arm) or a LATER backedge record
    (`br_if 0` — production's reverse `last_record` walk in
    `reconstruct_loop_backedges` appends the exit `Break` only to the
    last record's tail; every earlier record's tail terminates in the
    later record's composite). Either way the wrap for THIS backedge
    must not append its exit `Break`: the tail's fall-through is the
    loop's continue path, not the wasm loop-end exit.

    `d` tracks the nesting depth of structured openers within the
    tail — only same-level (`d = 0`) branches at label depth 0 target
    the enclosing loop; a `br 0`/`br_if 0` inside a nested frame
    targets that frame, and deeper-label branches to the loop are not
    recorded by production either (they keep the historical eager
    emission). -/
def tailReenters (d : Nat) : List WasmInstr → Bool
  | [] => false
  | i :: rest =>
    match i with
    | .block _ | .wloop _ | .wif _ => tailReenters (d + 1) rest
    | .wend =>
        match d with
        | 0 => false          -- current frame closed; nothing later
        | d' + 1 => tailReenters d' rest
    | .br 0 => if d = 0 then true else tailReenters d rest
    | .brIf 0 => if d = 0 then true else tailReenters d rest
    | _ => tailReenters d rest

/-- The exit-`Break` suffix a depth-0 loop-backedge wrap appends to
    its else arm. Mirrors the `last_record && !tail_exits_or_continues`
    append in `reconstruct_loop_backedges` (lower.rs): the wrapped
    tail runs on the `!cond` exit path, and falling off the wasm loop
    end exits — but Quanta's structured Loop auto-continues on body
    fall-through, so an explicit `Break` seals the exit. Skipped when
    the tail re-enters the loop (`reenters` — explicit continue or a
    later backedge record) or already exits (`endsInBreak`). -/
def backedgeEndBreak (reenters : Bool) (postOps : List KernelOp) : List KernelOp :=
  match reenters with
  | true  => []
  | false => if endsInBreak postOps then [] else [.breakOp]

@[simp] theorem tailReenters_nil (d : Nat) : tailReenters d [] = false := rfl

@[simp] theorem endsInBreak_nil : endsInBreak [] = false := rfl

@[simp] theorem backedgeEndBreak_false_nil :
    backedgeEndBreak false [] = [.breakOp] := rfl

/-- Lower a list of WASM instructions, threading state. Concatenates
    the per-instr op lists. `none` if any single op refuses or stack
    underflows.

    Structured-control ops are handled here (not in `lowerInstr`)
    because they consume the inner body out of the instruction
    stream. We use the splitter helpers in `Quanta.Wasm.Structured`
    to pre-extract the body, recurse with a `frames : List FrameKind`
    stack (innermost = head), then wrap into the appropriate
    `KernelOp`:

    * `block ... wend` is a label scope — its body's ops splice
      directly into the parent's op list (matches the
      `FrameKind::Block` arm in `lower.rs::End`).
    * `wloop ... wend` lowers to `[KernelOp.loopOp body_ops]`. Inner
      `br_if 0` continues the loop (no IR emitted at the br_if site);
      inner `br depth ≥ 1` cross-Loop emits `KernelOp.breakOp`.
    * `wif ... welse ... wend` commits the popped cond and emits
      `KernelOp.branch cond thenOps elseOps` (or `... [] elseOps` for
      no-else). Mirrors `RawInstr::If`/`Else`/`End` in `lower.rs`.

    `br depth` lowering:
    * target is Loop and `depth = 0`: emit nothing (continue at
      structured-Loop's natural fall-through). Drops the rest of the
      current scope (dead code in WASM validation).
    * target is non-Loop AND a Loop is between current top and
      target: emit `[.breakOp]`, drop rest.
    * else: refuse with `none` (redirect-chain unsupported in this
      slice).

    `brIf depth` lowering:
    * target is Loop, `depth = 0`: the lowered rest-of-body nests
      INTO the branch's else arm —
      `[.branch cond [] (postOps ++ backedgeEndBreak …)]` — so the
      tail runs exactly when the backedge does not fire (mirrors
      `reconstruct_loop_backedges` in lower.rs). On `cond` the empty
      then-arm falls through the loop end and auto-continues; on
      `!cond` the else arm runs the tail and then Breaks out, unless
      the tail already exits or re-enters the loop. The cond register
      is committed beforehand. With an empty tail this reduces
      byte-for-byte to the historical eager
      `[.branch cond [] [.breakOp]]` (see
      `lowerInstrs_brIf0_loop_empty_tail`).
    * target is non-Loop with Loop between: emit
      `[.branch cond [.breakOp] []]` — break on `cond`. Continues.
    * else: refuse with `none`.

    `fuel : Nat` bounds the recursion depth of the structured arms;
    a kernel with `N` levels of nesting needs `fuel ≥ N`. -/
def lowerInstrs (fuel : Nat) (frames : List FrameKind) (s : LowerState) :
    List WasmInstr → Option (LowerState × List KernelOp)
  | [] => some (s, [])
  | i :: rest =>
      match i with
      | .block _ =>
          match fuel with
          | 0 => none
          | f + 1 =>
              match splitAtEnd rest with
              | none => none
              | some (body, post) => do
                  let (s1, innerOps) ← lowerInstrs f (.block :: frames) s body
                  let (s2, postOps)  ← lowerInstrs f frames s1 post
                  pure (s2, innerOps ++ postOps)
      | .wloop _ =>
          match fuel with
          | 0 => none
          | f + 1 =>
              match splitAtEnd rest with
              | none => none
              | some (body, post) => do
                  -- Snapshot localReg / localTy / currentReg at loop
                  -- entry (mirrors production's `force_locals_to_stable`
                  -- semantics on lower.rs line 546). Inside the body,
                  -- any `localSet` updates currentReg + emits a
                  -- stable-sync Copy; post-loop reads see the entry
                  -- baseline because currentReg is reset to its
                  -- pre-loop value at frame close.
                  let entry_localReg := s.localReg
                  let entry_localTy  := s.localTy
                  let entry_currentReg := s.currentReg
                  let (s1, bodyOps) ← lowerInstrs f (.loopK :: frames) s body
                  -- Restore localReg / localTy / currentReg after the
                  -- loop body. localReg + localTy snapshot/restore
                  -- mirrors production's `merge_locals_post_frame`
                  -- for the Loop arm. currentReg restore clears the
                  -- per-frame post-write bindings so post-loop reads
                  -- fall back to the stable-reg layer (which was kept
                  -- in sync by localSet's dual-Copy).
                  let s1_restored : LowerState :=
                    { s1 with localReg := entry_localReg,
                              localTy  := entry_localTy,
                              currentReg := entry_currentReg }
                  let (s2, postOps) ← lowerInstrs f frames s1_restored post
                  pure (s2, [.loopOp bodyOps] ++ postOps)
      | .wif _ =>
          match fuel with
          | 0 => none
          | f + 1 =>
              match splitAtElseOrEnd rest with
              | none => none
              | some (thenBody, elseBody, post) => do
                  let (svCond, s0) ← s.popSym
                  let (cond, s1, opsCommit) ← s0.commit svCond
                  -- Cast u32 cond to bool (`.branch` requires vBool;
                  -- mirrors the brIf L6 fix).
                  let (cond_bool, s_cast) := s1.alloc
                  -- Snapshot localReg + localTy at If-entry. Mirrors
                  -- production's `snapshot_locals` / `restore_locals_from_snapshot`
                  -- (lower.rs lines 523/534): both branches must lower
                  -- from the same baseline local-binding state so eval's
                  -- "pick one body based on cond" is in sync with the
                  -- lowering's "splice in both bodies" shape.
                  let entry_localReg := s_cast.localReg
                  let entry_localTy  := s_cast.localTy
                  let entry_currentReg := s_cast.currentReg
                  let (s2, thenOps) ← lowerInstrs f (.wif :: frames) s_cast thenBody
                  -- Restore localReg / localTy / currentReg from the
                  -- snapshot before lowering elseBody (matches
                  -- `restore_locals_from_snapshot`). The freshly-bumped
                  -- nextReg from thenBody is kept so elseBody allocates
                  -- non-colliding regs.
                  let s2_restored : LowerState :=
                    { s2 with localReg := entry_localReg,
                              localTy  := entry_localTy,
                              currentReg := entry_currentReg }
                  let (s3, elseOps) ← lowerInstrs f (.wif :: frames) s2_restored elseBody
                  -- Restore again after elseBody (post-If merge: both
                  -- branches' local rebindings are discarded; post-wif
                  -- reads fall back to the stable-reg layer which was
                  -- kept in sync by localSet's dual-Copy per-write).
                  let s3_restored : LowerState :=
                    { s3 with localReg := entry_localReg,
                              localTy  := entry_localTy,
                              currentReg := entry_currentReg }
                  let (s4, postOps) ← lowerInstrs f frames s3_restored post
                  pure (s4, opsCommit
                            ++ [.cast cond_bool cond .u32 .bool,
                                .branch cond_bool thenOps elseOps]
                            ++ postOps)
      | .br depth =>
          -- Code after `br` is dead (WASM validator-rejected if
          -- reached); we simply don't recurse on `rest`.
          match frames.get? depth with
          | none => none
          | some .loopK =>
              -- depth = 0 inside the loop body: continue at fall-through.
              -- depth > 0 with .loopK at that frame: break out of the
              -- *containing* loop (cross-Loop) — emit Break.
              if depth = 0 then some (s, [])
              else if hasLoopAbove frames depth then some (s, [.breakOp])
              else some (s, [])  -- continue an outer loop: still no IR.
          | some k =>
              if hasLoopAbove frames depth then
                if loopsAbove frames depth = 1 ∧ k = .block then
                  -- Production (2026-06-12) lowers this via the
                  -- exit-flag record (`emit_loop_crossing_exit`):
                  -- flag declared at the target frame, `flag := true;
                  -- Break` at the site, wrap at the target's End. Not
                  -- yet modeled — refuse rather than keep the
                  -- label-lossy plain Break production no longer
                  -- emits for this shape.
                  none
                else some (s, [.breakOp])
              else none  -- record-and-wrap (`record_br_at`): not yet modeled.
      | .brIf depth => do
          let (svCond, s0) ← s.popSym
          let (cond, s1, opsCommit) ← s0.commit svCond
          match frames.get? depth with
          | none => none
          | some .loopK =>
              if depth = 0 then do
                -- br_if 0 to Loop: continue if cond, run the rest of
                -- the body on the exit path if !cond. The lowered
                -- tail nests INTO the branch's else arm (mirrors
                -- `reconstruct_loop_backedges` in lower.rs): emitting
                -- it after the branch would run it unconditionally on
                -- every iteration. The else arm gains a trailing exit
                -- `Break` unless the tail already exits or re-enters
                -- the loop (`backedgeEndBreak`). Cast cond from u32
                -- to bool before .branch reads it (the WASM-route cmp
                -- pipeline emits u32 results; `.branch`'s evalOp
                -- expects vBool).
                let (cond_bool, s_cast) := s1.alloc
                let (s2, postOps) ← lowerInstrs fuel frames s_cast rest
                pure (s2,
                  opsCommit
                  ++ [.cast cond_bool cond .u32 .bool,
                      .branch cond_bool []
                        (postOps ++ backedgeEndBreak (tailReenters 0 rest) postOps)])
              else if hasLoopAbove frames depth then do
                -- br_if to outer Loop with another Loop between:
                -- break the inner loop on cond.
                let (cond_bool, s_cast) := s1.alloc
                let (s2, postOps) ← lowerInstrs fuel frames s_cast rest
                pure (s2,
                  opsCommit
                  ++ [.cast cond_bool cond .u32 .bool,
                      .branch cond_bool [.breakOp] []]
                  ++ postOps)
              else do
                -- Continue an outer loop directly: no IR (cond
                -- is unused on the KOps side; the loop's natural
                -- wrap-around handles iteration).
                let (s2, postOps) ← lowerInstrs fuel frames s1 rest
                pure (s2, opsCommit ++ postOps)
          | some k =>
              if hasLoopAbove frames depth then
                if loopsAbove frames depth = 1 ∧ k = .block then
                  -- Exit-flag record in production — not yet modeled
                  -- (see the `br` arm above).
                  none
                else do
                  let (cond_bool, s_cast) := s1.alloc
                  let (s2, postOps) ← lowerInstrs fuel frames s_cast rest
                  pure (s2,
                    opsCommit
                    ++ [.cast cond_bool cond .u32 .bool,
                        .branch cond_bool [.breakOp] []]
                    ++ postOps)
              else none  -- record-and-wrap (`record_br_at`): not yet modeled.
      | .wreturn =>
          -- Production (`RawInstr::Return`): inside a frame with no
          -- Loop open it emits nothing and keeps lowering — the
          -- post-return ops are WASM dead code; production lowers
          -- them too (they sit after the function's wrap or are
          -- simply unreachable). With a Loop open production emits
          -- `Break`; at function top level it records an always-true
          -- wrap. Neither of those shapes occurs on the audited
          -- kernel surface (2026-06-12) — refuse them rather than
          -- model semantics nothing exercises.
          if frames.any (· = .loopK) ∨ frames.isEmpty then none
          else do
            let (s2, postOps) ← lowerInstrs fuel frames s rest
            pure (s2, postOps)
      | _ => do
          let (s1, ops1) ← lowerInstr s i
          let (s2, ops2) ← lowerInstrs fuel frames s1 rest
          pure (s2, ops1 ++ ops2)

/-- Empty-tail reduction: when the backedge `br_if 0` is the loop
    body's last instruction (the ubiquitous rustc `while`-loop shape
    — every previously-modeled kernel), the nested emission reduces
    byte-for-byte to the historical eager shape
    `opsCommit ++ [.cast …, .branch cond_bool [] [.breakOp]]`:
    the tail lowers to no ops, `tailReenters`/`endsInBreak` are both
    false, and `backedgeEndBreak` supplies exactly the old `[.breakOp]`
    else arm. Previously-proven kernels are therefore unaffected by
    the tail-nesting re-sync. -/
theorem lowerInstrs_brIf0_loop_empty_tail
    (fuel : Nat) (frames : List FrameKind) (s : LowerState)
    (h_target : frames.get? 0 = some .loopK) :
    lowerInstrs fuel frames s [.brIf 0] =
      (do
        let (svCond, s0) ← s.popSym
        let (cond, s1, opsCommit) ← s0.commit svCond
        let (cond_bool, s_cast) := s1.alloc
        pure (s_cast,
          opsCommit ++ [.cast cond_bool cond .u32 .bool,
                        .branch cond_bool [] [.breakOp]])) := by
  simp only [lowerInstrs]
  rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
  · rfl
  simp only [Option.bind_eq_bind, Option.some_bind]
  rcases hcommit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rfl
  simp only [Option.some_bind]
  rw [h_target]
  simp [lowerInstrs, LowerState.alloc, backedgeEndBreak, endsInBreak]

-- ════════════════════════════════════════════════════════════════════
-- Backedge tail-nesting pins
--
-- Behavioral pins for the depth-0 loop-backedge arm's nested emission
-- (the `reconstruct_loop_backedges` re-sync). One pin per facet of
-- the end-of-body Break rule; all decided by evaluation.
-- ════════════════════════════════════════════════════════════════════

section BackedgePins

/-- `KernelOp` is a nested inductive, so `DecidableEq` cannot derive;
    compare through `Repr` (injective on these first-order values). -/
private def pinEq {α : Type} [Repr α] (a b : α) : Bool :=
  toString (repr a) == toString (repr b)

/-- Fall-through tail: the lowered rest-of-body (`localSet`'s
    zero-init + dual-Copy) nests INTO the else arm, and the exit
    `Break` is appended after it (the tail would otherwise fall off
    the wasm loop end into the structured Loop's auto-continue). -/
example :
    pinEq
      (lowerInstrs 4 [.loopK] LowerState.empty
        [.i32Const 1, .brIf 0, .i32Const 5, .localSet 0])
      (some ({ LowerState.empty with
                nextReg := 5,
                localReg := [(0, 4)], localTy := [(0, .i32)],
                currentReg := [(0, 3)] },
        [.const 0 (.i32 1),
         .cast 1 0 .u32 .bool,
         .branch 1 []
           [.const 2 (.i32 5), .const 3 (.i32 0), .copy 3 2, .copy 4 3,
            .breakOp]])) = true := by native_decide

/-- Explicit continue: the tail ends in `br 0` (production records
    `continue_pos`), so its fall-through re-enters the loop — no exit
    `Break` is appended. -/
example :
    pinEq
      (lowerInstrs 4 [.loopK] LowerState.empty
        [.i32Const 1, .brIf 0, .i32Const 5, .localSet 0, .br 0])
      (some ({ LowerState.empty with
                nextReg := 5,
                localReg := [(0, 4)], localTy := [(0, .i32)],
                currentReg := [(0, 3)] },
        [.const 0 (.i32 1),
         .cast 1 0 .u32 .bool,
         .branch 1 []
           [.const 2 (.i32 5), .const 3 (.i32 0), .copy 3 2, .copy 4 3]]))
      = true := by native_decide

/-- Later backedge: a second `br_if 0` in the tail makes the FIRST
    record non-last (production's reverse `last_record` walk) — the
    outer else arm terminates in the inner composite and gets no
    `Break` of its own; the inner (last) record gets the usual one. -/
example :
    pinEq
      (lowerInstrs 4 [.loopK] LowerState.empty
        [.i32Const 1, .brIf 0, .i32Const 1, .brIf 0])
      (some ({ LowerState.empty with nextReg := 4 },
        [.const 0 (.i32 1),
         .cast 1 0 .u32 .bool,
         .branch 1 []
           [.const 2 (.i32 1),
            .cast 3 2 .u32 .bool,
            .branch 3 [] [.breakOp]]])) = true := by native_decide

/-- Tail already exits: a multi-loop-crossing `br` lowers to
    `[.breakOp]`, so the wrapped tail ends in an unconditional Break
    (`endsInBreak`) and no second exit `Break` is appended. -/
example :
    pinEq
      (lowerInstrs 4 [.loopK, .loopK, .block] LowerState.empty
        [.i32Const 1, .brIf 0, .br 2])
      (some ({ LowerState.empty with nextReg := 2 },
        [.const 0 (.i32 1),
         .cast 1 0 .u32 .bool,
         .branch 1 [] [.breakOp]])) = true := by native_decide

end BackedgePins

end Quanta.Wasm
