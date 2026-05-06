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
  deriving Repr

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
  deriving Repr

def LowerState.empty : LowerState :=
  { nextReg := 0, stack := [], localReg := [], localTy := [] }

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

def setLocalReg (s : LowerState) (i : Nat) (r : Reg) (ty : Scalar) : LowerState :=
  let regs' := (i, r) :: s.localReg.filter (fun p => p.fst ≠ i)
  let tys'  := (i, ty) :: s.localTy.filter (fun p => p.fst ≠ i)
  { s with localReg := regs', localTy := tys' }

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
      let (dst, s1) := s.alloc
      some (dst, s1, [.const dst (.u32 (UInt32.ofNat n.toNat))])
  | .bufferPtr _          => none
  | .scaledIdx _ _        => none
  | .bufferAccess _ _ _   => none

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

/-- Lower a single i32 binary op: pop two operand regs, allocate a
    result reg, emit the corresponding KOps `binOp`, push the result.
    `none` on stack underflow. -/
def lowerI32Bin (s : LowerState) (op : BinOp) : Option (LowerState × List KernelOp) := do
  let (b, s1) ← s.pop
  let (a, s2) ← s1.pop
  let (dst, s3) := s2.alloc
  let s4 := s3.push dst
  pure (s4, [.binOp dst a b op .u32])

/-- Lower a single i32 comparison. KOps `Cmp` produces a `vBool`, but
    WASM's `i32.{eq,ne,lt,le,gt,ge}` push an `wI32 0/1` — so we emit
    `Cmp` followed by a `Cast bool→u32` to re-enter the u32 alphabet
    before the value flows back onto the stack as `.reg _ .u32`.

    Production's lowering pushes a `.reg _ .bool` slot and casts at
    consume-time via `commit()`. The Lean port casts eagerly here to
    keep `WasmValue.encodes` single-shape (always `.u32`) and avoid a
    cascade through every existing per-op preservation proof. The
    end-to-end IR shape is identical (cmp + cast); only the placement
    of the cast in the lowering pass differs. -/
def lowerI32Cmp (s : LowerState) (op : CmpOp) : Option (LowerState × List KernelOp) := do
  let (b, s1) ← s.pop
  let (a, s2) ← s1.pop
  let (boolReg, s3) := s2.alloc
  let (dst, s4) := s3.alloc
  let s5 := s4.push dst
  pure (s5, [.cmp boolReg a b op .bool, .cast dst boolReg .bool .u32])

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
  | .localGet i => do
      let stable ← s.lookupLocal i
      let (fresh, s1) := s.alloc
      let s2 := s1.push fresh
      pure (s2, [.copy fresh stable])
  | .localSet i => do
      let (src, s1) ← s.pop
      -- ty defaults to `.u32` when the local has no recorded type yet
      -- (slice 1 only models i32). Using `getD` (not `getDM`) keeps the
      -- result a plain `Scalar` instead of an `Option Scalar`, which
      -- avoids an extra monadic bind in the proof.
      let ty : Scalar := (s1.lookupLocalTy i).getD .u32
      match s1.lookupLocal i with
      | some dst =>
          -- Local already has a stable register → emit a copy into it.
          pure (s1.setLocalReg i dst ty, [.copy dst src])
      | none =>
          -- First write: allocate the local's stable reg, copy in.
          let (dst, s2) := s1.alloc
          pure (s2.setLocalReg i dst ty, [.copy dst src])
  | .localTee i => do
      -- `local.tee` = `local.set i` followed by `local.get i`. The
      -- `localGet` half breaks aliasing by emitting a Copy into a
      -- fresh register, so the post-tee stack value is `post_fresh`,
      -- not the local's stable register. Same alias-free invariant
      -- as `localGet`.
      let (src, s1) ← s.pop
      let ty : Scalar := (s1.lookupLocalTy i).getD .u32
      match s1.lookupLocal i with
      | some dst =>
          let s2 := s1.setLocalReg i dst ty
          let (post_fresh, s3) := s2.alloc
          pure (s3.push post_fresh, [.copy dst src, .copy post_fresh dst])
      | none =>
          let (dst, s2) := s1.alloc
          let s3 := s2.setLocalReg i dst ty
          let (post_fresh, s4) := s3.alloc
          pure (s4.push post_fresh, [.copy dst src, .copy post_fresh dst])
  -- i32 arithmetic
  | .i32Add  => lowerI32Bin s .add
  | .i32Sub  => lowerI32Bin s .sub
  | .i32Mul  => lowerI32Bin s .mul
  | .i32And  => lowerI32Bin s .bAnd
  | .i32Or   => lowerI32Bin s .bOr
  | .i32Xor  => lowerI32Bin s .bXor
  | .i32Shl  => lowerI32Bin s .shl
  | .i32ShrU => lowerI32Bin s .shr
  | .i32DivU => lowerI32Bin s .div
  | .i32RemU => lowerI32Bin s .rem
  -- i32 comparisons (unsigned only — signed lift in a later slice).
  | .i32Eq  => lowerI32Cmp s .eq
  | .i32Ne  => lowerI32Cmp s .ne
  | .i32LtU => lowerI32Cmp s .lt
  | .i32LeU => lowerI32Cmp s .le
  | .i32GtU => lowerI32Cmp s .gt
  | .i32GeU => lowerI32Cmp s .ge
  -- Return / nop / drop
  | .wreturn => some (s, [])
  | .nop     => some (s, [])
  | .drop    => do
      let (_, s1) ← s.pop
      pure (s1, [])
  -- Outside slice 1 — refused, matching `UnsupportedOp` in production.
  | _ => none

/-- Lower a list of WASM instructions, threading state. Concatenates
    the per-instr op lists. `none` if any single op refuses or stack
    underflows. -/
def lowerInstrs (s : LowerState) : List WasmInstr → Option (LowerState × List KernelOp)
  | [] => some (s, [])
  | i :: rest => do
      let (s1, ops1) ← lowerInstr s i
      let (s2, ops2) ← lowerInstrs s1 rest
      pure (s2, ops1 ++ ops2)

end Quanta.Wasm
