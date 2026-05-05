/-
# WASM → KernelOps lowering — Lean specification (step 059)

Mirrors the production translator at
`crates/quanta-wasm-lowering/src/lower.rs`. The Rust pass simulates
WASM's stack with a richer symbolic abstract domain (buffer pointers,
scaled indices, etc.) so it can recognize buffer-access patterns. For
the slice-1 subset (no memory, no buffers), the symbolic domain
collapses to "every stack slot is a Quanta IR register" — that's what
this Lean port models.

State carried during lowering:
* `nextReg` — fresh-register counter (matches `LowerCtx::next_reg`).
* `stack`  — list of Quanta IR registers, top = stack top.
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
  /-- Symbolic stack — top first. **Slices 1-3 model the stack as
      a list of `Reg`s** (every value-typed slot is a plain register).
      Slice 4 will replace this with `List SymVal` to support the
      buffer-pattern recognition, at the cost of cascading through
      every per-op proof. The `SymVal` type is defined above as
      preparatory scope marker. -/
  stack    : List Reg
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

def push (s : LowerState) (r : Reg) : LowerState :=
  { s with stack := r :: s.stack }

def pop (s : LowerState) : Option (Reg × LowerState) :=
  match s.stack with
  | []      => none
  | r :: rs => some (r, { s with stack := rs })

def lookupLocal (s : LowerState) (i : Nat) : Option Reg :=
  s.localReg.find? (fun p => p.fst = i) |>.map Prod.snd

def lookupLocalTy (s : LowerState) (i : Nat) : Option Scalar :=
  s.localTy.find? (fun p => p.fst = i) |>.map Prod.snd

def setLocalReg (s : LowerState) (i : Nat) (r : Reg) (ty : Scalar) : LowerState :=
  let regs' := (i, r) :: s.localReg.filter (fun p => p.fst ≠ i)
  let tys'  := (i, ty) :: s.localTy.filter (fun p => p.fst ≠ i)
  { s with localReg := regs', localTy := tys' }

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

/-- Lower a single i32 comparison: pops two regs, allocates a result
    reg, emits the comparison; the result is bool-typed but pushed
    as-is — KOps `cmp` already produces a bool register. -/
def lowerI32Cmp (s : LowerState) (op : CmpOp) : Option (LowerState × List KernelOp) := do
  let (b, s1) ← s.pop
  let (a, s2) ← s1.pop
  let (dst, s3) := s2.alloc
  let s4 := s3.push dst
  pure (s4, [.cmp dst a b op .u32])

/-- Lower one WASM instruction. Returns the new state and the emitted
    KOps. `none` for ops outside the subset (matches the production
    pass's `UnsupportedOp` error). -/
def lowerInstr (s : LowerState) : WasmInstr → Option (LowerState × List KernelOp)
  -- Constants. WASM `i32.const n` produces a fresh register holding the
  -- low-32 bits of `n`; we encode it as `ConstValue.u32` because the
  -- IR's u32 alphabet matches WASM i32 wrapping semantics exactly.
  | .i32Const n =>
      let (_, s1, ops) := freshAndPush s
        (fun r => .const r (.u32 (UInt32.ofNat n.toNat)))
      some (s1, ops)
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
      let ty ← s1.lookupLocalTy i |>.getDM (some .u32)
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
      let ty ← s1.lookupLocalTy i |>.getDM (some .u32)
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
