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
  deriving Repr

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
  let s6 := s5.push dst
  pure (s6, opsA ++ opsB ++ [.binOp dst ra rb op .u32])

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

/-- Lower `i32.add`. Production fast-path: if the popped operands are
    `<bufferPtr slot> <scaledIdx base scale>` (in either order), fold
    to `SymVal.bufferAccess slot base scale` with no IR emitted. The
    typed `KernelOp.Load`/`Store` consumes the `bufferAccess` in the
    next memory op. Otherwise fall through to `lowerI32Bin .add`.
    Mirrors `RawInstr::I32Add` in `lower.rs`. -/
def lowerI32Add (s : LowerState) : Option (LowerState × List KernelOp) :=
  match s.stack with
  | .scaledIdx base scale :: .bufferPtr slot :: rest =>
      some ({ s with stack := .bufferAccess slot base scale :: rest }, [])
  | .bufferPtr slot :: .scaledIdx base scale :: rest =>
      some ({ s with stack := .bufferAccess slot base scale :: rest }, [])
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
          let stable ← s.lookupLocal i
          let (fresh, s1) := s.alloc
          let s2 := s1.push fresh
          pure (s2, [.copy fresh stable])
  | .localSet i => do
      -- popSym + commit (matches binop/cmp/localTee): a popped
      -- `.i32ConstSym` materializes via a const-op prefix, while
      -- buffer SymVals refuse at `commit` (and never reach localSet
      -- in well-formed code — the buffer-pattern arms intercept
      -- them earlier).
      let (sv, s1) ← s.popSym
      let (src, s2, opsCommit) ← s1.commit sv
      -- ty defaults to `.u32` when the local has no recorded type yet
      -- (slice 1 only models i32). Using `getD` (not `getDM`) keeps the
      -- result a plain `Scalar` instead of an `Option Scalar`, which
      -- avoids an extra monadic bind in the proof.
      let ty : Scalar := (s2.lookupLocalTy i).getD .u32
      match s2.lookupLocal i with
      | some dst =>
          -- Local already has a stable register → emit a copy into it.
          pure (s2.setLocalReg i dst ty, opsCommit ++ [.copy dst src])
      | none =>
          -- First write: allocate the local's stable reg, copy in.
          let (dst, s3) := s2.alloc
          pure (s3.setLocalReg i dst ty, opsCommit ++ [.copy dst src])
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
      let ty : Scalar := (s2.lookupLocalTy i).getD .u32
      match s2.lookupLocal i with
      | some dst =>
          let s3 := s2.setLocalReg i dst ty
          let (post_fresh, s4) := s3.alloc
          pure (s4.push post_fresh,
                opsCommit ++ [.copy dst src, .copy post_fresh dst])
      | none =>
          let (dst, s3) := s2.alloc
          let s4 := s3.setLocalReg i dst ty
          let (post_fresh, s5) := s4.alloc
          pure (s5.push post_fresh,
                opsCommit ++ [.copy dst src, .copy post_fresh dst])
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
  -- Return / nop / drop
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
    * target is Loop, `depth = 0`: emit
      `[.branch cond [] [.breakOp]]` — break on `!cond`. The cond
      register is committed beforehand. Continues with `rest`.
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
                  -- Snapshot localReg / localTy at loop entry (mirrors
                  -- production's `force_locals_to_stable` semantics
                  -- on lower.rs line 546). Inside the body, any
                  -- `localSet` may rebind locals, but post-loop
                  -- reads see the entry baseline — the per-iteration
                  -- IR runs against fresh bindings each time.
                  let entry_localReg := s.localReg
                  let entry_localTy  := s.localTy
                  let (s1, bodyOps) ← lowerInstrs f (.loopK :: frames) s body
                  -- Restore localReg / localTy after the loop body
                  -- (matches `merge_locals_post_frame` for the Loop
                  -- arm — post-loop reads see the pre-loop bindings
                  -- via the unchanged register layer). Production's
                  -- richer merge emits Copy ops; the Lean spec port
                  -- defers that propagation.
                  let s1_restored : LowerState :=
                    { s1 with localReg := entry_localReg,
                              localTy  := entry_localTy }
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
                  let (s2, thenOps) ← lowerInstrs f (.wif :: frames) s_cast thenBody
                  -- Restore localReg / localTy from the snapshot before
                  -- lowering elseBody (matches `restore_locals_from_snapshot`).
                  -- The freshly-bumped nextReg from thenBody is kept so
                  -- elseBody allocates non-colliding regs.
                  let s2_restored : LowerState :=
                    { s2 with localReg := entry_localReg,
                              localTy  := entry_localTy }
                  let (s3, elseOps) ← lowerInstrs f (.wif :: frames) s2_restored elseBody
                  -- Restore again after elseBody (post-If merge: both
                  -- branches' local rebindings are discarded; reads after
                  -- the wif see the pre-If state via the unchanged
                  -- stable-register layer).
                  -- Note: this is a simplification of production's
                  -- `merge_locals_post_frame` which emits Copy ops to
                  -- propagate the modified value. The Lean spec port
                  -- defers that propagation until cons_wloop needs it.
                  let s3_restored : LowerState :=
                    { s3 with localReg := entry_localReg,
                              localTy  := entry_localTy }
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
          | some _ =>
              if hasLoopAbove frames depth then some (s, [.breakOp])
              else none  -- redirect-chain: not supported in this slice.
      | .brIf depth => do
          let (svCond, s0) ← s.popSym
          let (cond, s1, opsCommit) ← s0.commit svCond
          match frames.get? depth with
          | none => none
          | some .loopK =>
              if depth = 0 then do
                -- br_if 0 to Loop: continue if cond, break if !cond.
                -- Cast cond from u32 to bool before .branch reads it
                -- (the WASM-route cmp pipeline emits u32 results;
                -- `.branch`'s evalOp expects vBool).
                let (cond_bool, s_cast) := s1.alloc
                let (s2, postOps) ← lowerInstrs fuel frames s_cast rest
                pure (s2,
                  opsCommit
                  ++ [.cast cond_bool cond .u32 .bool,
                      .branch cond_bool [] [.breakOp]]
                  ++ postOps)
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
          | some _ =>
              if hasLoopAbove frames depth then do
                let (cond_bool, s_cast) := s1.alloc
                let (s2, postOps) ← lowerInstrs fuel frames s_cast rest
                pure (s2,
                  opsCommit
                  ++ [.cast cond_bool cond .u32 .bool,
                      .branch cond_bool [.breakOp] []]
                  ++ postOps)
              else none
      | _ => do
          let (s1, ops1) ← lowerInstr s i
          let (s2, ops2) ← lowerInstrs fuel frames s1 rest
          pure (s2, ops1 ++ ops2)

end Quanta.Wasm
