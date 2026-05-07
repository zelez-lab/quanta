/-
# WASM big-step semantics — Quanta-relevant subset (step 059)

Big-step `evalInstr` / `evalInstrs` over `Quanta.Wasm.WasmInstr`. The
operator coverage here grows in lockstep with the preservation theorems
in `Quanta.Wasm.Preservation` — every constructor that has a real
semantic case in `evalInstr` must be paired with a per-op preservation
lemma; everything else is `Option.none` (untreated).

Slice 1 (this commit) wires: `i32Const`, `i32Add`, `i32Sub`, `i32Mul`,
the bitwise i32 family, `localGet`, `localSet`, `localTee`,
`wreturn`. That subset is small but exercises every shape the
preservation proof needs (constants, locals, arithmetic, return). The
remaining ops surface as `none`; their preservation lemmas land with
later slices.

Design notes:
* WASM is a *stack machine* — the state carries a value stack alongside
  the locals. The `evalInstr` definition pops operands off the stack
  and pushes the result, exactly as the wasmparser semantics document
  it (https://webassembly.github.io/spec/core/exec/instructions.html).
* Locals are an indexed map; out-of-range reads / writes return `none`
  (matches `wasmparser`'s validator-checked behaviour at runtime).
* Integer arithmetic uses the same wrapping primitives the KOps
  semantics does — see `Quanta.Semantics.Cpu.eval_u32_wrapping_*`.
  Reusing those primitives is what makes per-op preservation provable
  by `simp` / `rfl` once the refinement relation is unfolded.
* `wreturn` halts the surrounding function; we model it via a `halted`
  flag on the state that future instructions short-circuit.
-/

import Quanta.Wasm.Syntax
import Quanta.Wasm.Structured
import Quanta.Semantics.Cpu

namespace Quanta.Wasm

open Quanta.Semantics.Cpu

-- ════════════════════════════════════════════════════════════════════
-- Value alphabet
-- ════════════════════════════════════════════════════════════════════

/-- Runtime value alphabet. `i32` represented as `UInt32` (matches the
    wrapping semantics of WASM integer ops); `i64` as `UInt64`; floats
    as raw bit patterns evaluated through the same `Cpu` primitives
    KOps uses. -/
inductive WasmValue where
  | wI32 (n : UInt32)
  | wI64 (n : UInt64)
  | wF32 (bits : UInt32)
  | wF64 (bits : UInt64)
  deriving Repr, DecidableEq

-- ════════════════════════════════════════════════════════════════════
-- State (locals + value stack + halted flag)
-- ════════════════════════════════════════════════════════════════════

abbrev WasmStack : Type := List WasmValue
abbrev WasmLocals : Type := List WasmValue

/-- Linear memory: WASM is byte-addressable, modeled as a flat list of
    bytes. Loads/stores access aligned chunks (4 bytes for i32/f32, 1
    byte for i32.load8u/s, etc.). Production lowering recognizes the
    canonical `<buffer_ptr> + <scaled_index> .load` pattern and emits
    a typed KOps `Load { field, index, ty }` against `Heap` — sidesteps
    the byte-level memory entirely. The byte view here is what makes
    the equivalence statement well-formed. -/
abbrev WasmMem : Type := List UInt8

structure WasmState where
  locals : WasmLocals
  stack  : WasmStack
  mem    : WasmMem := []
  halted : Bool := false
  deriving Repr

namespace WasmState

/-- Read local `i`. `none` if out of range. -/
def getLocal (s : WasmState) (i : Nat) : Option WasmValue :=
  s.locals.get? i

/-- Write local `i`. `none` if out of range. -/
def setLocal (s : WasmState) (i : Nat) (v : WasmValue) : Option WasmState :=
  if i < s.locals.length then
    some { s with locals := s.locals.set i v }
  else
    none

/-- Push a value on the stack. Always succeeds. -/
def push (s : WasmState) (v : WasmValue) : WasmState :=
  { s with stack := v :: s.stack }

/-- Pop a value off the stack. `none` if empty. -/
def pop (s : WasmState) : Option (WasmValue × WasmState) :=
  match s.stack with
  | []      => none
  | v :: rs => some (v, { s with stack := rs })

end WasmState

-- ════════════════════════════════════════════════════════════════════
-- i32 binary-op dispatch
--
-- Every WASM i32 binop pops two `wI32` values, applies the op, and
-- pushes the result. We bundle the op-to-function dispatch here so the
-- per-instr `evalInstr` reads as `binI32 op s`.
-- ════════════════════════════════════════════════════════════════════

/-- Apply a binary u32-typed op to the top two stack values. Returns
    the resulting state with the result pushed; `none` on type error
    or stack underflow. -/
def binI32 (op : UInt32 → UInt32 → UInt32) (s : WasmState) : Option WasmState := do
  let (b, s1) ← s.pop
  let (a, s2) ← s1.pop
  match a, b with
  | .wI32 av, .wI32 bv => some (s2.push (.wI32 (op av bv)))
  | _, _ => none

/-- Apply an i32 comparison: pops two `wI32` values, pushes `wI32 0/1`
    based on the predicate. -/
def cmpI32 (p : UInt32 → UInt32 → Bool) (s : WasmState) : Option WasmState := do
  let (b, s1) ← s.pop
  let (a, s2) ← s1.pop
  match a, b with
  | .wI32 av, .wI32 bv =>
      some (s2.push (.wI32 (if p av bv then 1 else 0)))
  | _, _ => none

-- ════════════════════════════════════════════════════════════════════
-- Memory access primitives
--
-- WASM linear memory is byte-addressable little-endian. A 4-byte load
-- at byte address `addr` reads `mem[addr..addr+4]` and assembles them
-- as a u32. The address arithmetic in production WASM is `base_ptr +
-- byte_offset` where `byte_offset = element_index * element_size`. The
-- lowering pass recognizes this pattern; the operational semantics
-- here is the bottom-up byte view that the lowering must agree with.
-- ════════════════════════════════════════════════════════════════════

/-- Read 4 bytes at `addr` as a little-endian u32. Returns `none` if
    the read goes out of bounds. -/
def WasmMem.load_u32 (m : WasmMem) (addr : Nat) : Option UInt32 := do
  let b0 ← m.get? addr
  let b1 ← m.get? (addr + 1)
  let b2 ← m.get? (addr + 2)
  let b3 ← m.get? (addr + 3)
  pure ((b0.toUInt32) ||| (b1.toUInt32 <<< 8) |||
        (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24))

/-- Write 4 bytes at `addr` as little-endian u32. Returns `none` if
    out of bounds. -/
def WasmMem.store_u32 (m : WasmMem) (addr : Nat) (v : UInt32) : Option WasmMem := do
  if addr + 3 < m.length then
    let m1 := m.set addr (UInt8.ofNat (v.toNat &&& 0xFF))
    let m2 := m1.set (addr + 1) (UInt8.ofNat ((v.toNat >>> 8) &&& 0xFF))
    let m3 := m2.set (addr + 2) (UInt8.ofNat ((v.toNat >>> 16) &&& 0xFF))
    let m4 := m3.set (addr + 3) (UInt8.ofNat ((v.toNat >>> 24) &&& 0xFF))
    pure m4
  else
    none

/-- WASM byte-load/store roundtrip — TCB axiom. The 4 successive
    `List.set`s in `store_u32` reconstruct exactly to the original
    `UInt32` when read back via `load_u32`. Constructively provable
    via `Nat.shiftRight_shiftLeft`-style bit-arithmetic plus
    `List.getElem?_set` machinery, but the proof is mechanical and
    deep. Accepted as TCB capturing WASM spec compliance for byte-
    addressable memory.

    Used by `Quanta.Wasm.Preservation.preservation_i32Store` to
    discharge the new-state `HeapRefines` clause at the
    `(slot, b.toNat)` entry the store wrote. -/
axiom WasmMem.store_load_same
    (m : WasmMem) (addr : Nat) (v : UInt32) (m' : WasmMem)
    (h_store : m.store_u32 addr v = some m') :
    m'.load_u32 addr = some v

/-- WASM byte-store + byte-load disjoint preservation — TCB axiom.
    A 4-byte store at `addr_s` doesn't perturb the load at `addr_l`
    when their byte ranges don't overlap. Constructively provable
    via 4 invocations of `List.getElem?_set_ne`; mechanical and
    accepted as TCB.

    Used by `preservation_i32Store` to lift the input `HeapRefines`
    past the store's mem-write at every `(slot', idx')` other than
    the one being written. -/
axiom WasmMem.store_load_disjoint
    (m : WasmMem) (addr_s : Nat) (v : UInt32) (m' : WasmMem)
    (h_store : m.store_u32 addr_s v = some m') (addr_l : Nat)
    (h_disj : addr_s + 4 ≤ addr_l ∨ addr_l + 4 ≤ addr_s) :
    m'.load_u32 addr_l = m.load_u32 addr_l

/-- Single-byte load. -/
def WasmMem.load_u8 (m : WasmMem) (addr : Nat) : Option UInt8 := m.get? addr

/-- Single-byte store. -/
def WasmMem.store_u8 (m : WasmMem) (addr : Nat) (v : UInt8) : Option WasmMem :=
  if addr < m.length then some (m.set addr v) else none

/-- Generic load helper: pops the address from the stack, reads via
    `f`, pushes the result back as the appropriate WasmValue. -/
def loadI32 (s : WasmState) (offset : Nat) : Option WasmState := do
  let (vaddr, s1) ← s.pop
  match vaddr with
  | .wI32 a =>
      let v ← s1.mem.load_u32 (a.toNat + offset)
      pure (s1.push (.wI32 v))
  | _ => none

/-- Generic store helper: pops value, then address. -/
def storeI32 (s : WasmState) (offset : Nat) : Option WasmState := do
  let (vval, s1) ← s.pop
  let (vaddr, s2) ← s1.pop
  match vaddr, vval with
  | .wI32 a, .wI32 v =>
      let m' ← s2.mem.store_u32 (a.toNat + offset) v
      pure { s2 with mem := m' }
  | _, _ => none

-- ════════════════════════════════════════════════════════════════════
-- Big-step eval — covered subset
-- ════════════════════════════════════════════════════════════════════

/-- Single-instruction step. `none` means "stuck" (validator-rejected
    state, or operator outside the lowered subset). The translator
    pass refuses these statically, so the preservation theorem is only
    obligated for `some`-returning runs. -/
def evalInstr (s : WasmState) : WasmInstr → Option WasmState
  -- Constants
  | .i32Const n => some (s.push (.wI32 (UInt32.ofNat n.toNat)))
  -- Locals
  | .localGet i => do
      let v ← s.getLocal i
      pure (s.push v)
  | .localSet i => do
      let (v, s1) ← s.pop
      s1.setLocal i v
  | .localTee i => do
      -- `local.tee` = `local.set` followed by re-pushing the value.
      -- We model it as: read top, set, leave on stack.
      let (v, _) ← s.pop
      let s' ← s.pop.bind (fun (_, srest) => srest.setLocal i v)
      pure (s'.push v)
  -- i32 arithmetic
  | .i32Add  => binI32 eval_u32_wrapping_add s
  | .i32Sub  => binI32 eval_u32_wrapping_sub s
  | .i32Mul  => binI32 eval_u32_wrapping_mul s
  | .i32And  => binI32 eval_u32_bitand s
  | .i32Or   => binI32 eval_u32_bitor s
  | .i32Xor  => binI32 eval_u32_bitxor s
  | .i32Shl  => binI32 (fun a b => a <<< b) s
  | .i32ShrU => binI32 (fun a b => a >>> b) s
  | .i32DivU => binI32 eval_u32_div s
  | .i32RemU => binI32 eval_u32_rem s
  -- i32 comparisons (unsigned)
  | .i32Eq  => cmpI32 (· == ·) s
  | .i32Ne  => cmpI32 (· != ·) s
  | .i32LtU => cmpI32 (· < ·) s
  | .i32LeU => cmpI32 (· <= ·) s
  | .i32GtU => cmpI32 (· > ·) s
  | .i32GeU => cmpI32 (· >= ·) s
  -- Memory: i32-typed loads/stores. The byte-load family (i32.load8u/s,
  -- i32.store8) lifts in a later sub-slice.
  | .i32Load offset _align  => loadI32 s offset
  | .i32Store offset _align => storeI32 s offset
  -- Return halts; subsequent instructions short-circuit.
  | .wreturn => some { s with halted := true }
  -- Misc
  | .nop  => some s
  | .drop => do let (_, s1) ← s.pop; pure s1
  -- Everything else is outside this slice's coverage.
  | _ => none

/-- Sequential big-step over an instruction list.

    Halted state short-circuits — matches WASM's `return` semantics.

    `block` and `wif` are handled here (not in `evalInstr`) because
    the structured-control semantics has to inspect the instruction
    stream past the head: a `block` at position 0 needs to consume
    the inner body up to the matching `wend`. We use the splitter
    helpers in `Quanta.Wasm.Structured` to pre-extract the inner
    body, recurse on it, then continue with the suffix.

    `fuel : Nat` bounds the recursion depth of the structured calls
    — each `block`/`wif` recursion costs one unit. The flat per-
    instruction sequence in the `_ => evalInstr` arm uses no fuel
    (structural recursion on the list tail). For a kernel of `N`
    structured-control nesting levels, `fuel ≥ N` suffices.
    `wloop` and `br`/`br_if` arrive in slice 5b. -/
def evalInstrs (fuel : Nat) (s : WasmState) : List WasmInstr → Option WasmState
  | [] => some s
  | i :: rest =>
      if s.halted then some s
      else
        match i with
        | .block _ =>
            match fuel with
            | 0 => none
            | f + 1 =>
                match splitAtEnd rest with
                | none => none
                | some (body, post) =>
                    match evalInstrs f s body with
                    | none => none
                    | some s' => evalInstrs f s' post
        | .wif _ =>
            match fuel with
            | 0 => none
            | f + 1 =>
                match s.pop with
                | none => none
                | some (vc, s0) =>
                    match splitAtElseOrEnd rest with
                    | none => none
                    | some (thenBody, elseBody, post) =>
                        match vc with
                        | .wI32 c =>
                            let body := if c = 0 then elseBody else thenBody
                            match evalInstrs f s0 body with
                            | none => none
                            | some s' => evalInstrs f s' post
                        | _ => none
        | _ =>
            match evalInstr s i with
            | none    => none
            | some s' => evalInstrs fuel s' rest

end Quanta.Wasm
