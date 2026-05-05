/-
# WASM operator subset — Lean syntactic view (step 059)

This module mirrors the `RawInstr` enum from
`crates/quanta-wasm-lowering/src/lib.rs` — the owned form of
`wasmparser::Operator` for the operator surface the Quanta lowering
pass actually consumes. Adding a WASM op the lowering pass should
handle means adding a constructor here, a case in
`Quanta.Wasm.Semantics.evalInstr`, and an arm in
`Quanta.Wasm.Translate.lowerInstr`.

This file is **syntax only** — operational semantics live in
`Quanta.Wasm.Semantics`, the lowering pass in `Quanta.Wasm.Translate`,
and the preservation theorems in `Quanta.Wasm.Preservation`.

The WASM type alphabet here is `WasmTy` (i32 / i64 / f32 / f64), the
same four-way split rustc emits for `wasm32-unknown-unknown`. Vector
and reference types are not in the subset (kernels never touch them).
-/

namespace Quanta.Wasm

inductive WasmTy where
  | i32 | i64 | f32 | f64
  deriving Repr, DecidableEq

/-- Owned form of `wasmparser::Operator` for the subset Quanta lowers.

    Constructor names match `RawInstr` 1:1 so the cross-language audit
    is mechanical (port one line from `lib.rs:207-321`, drop one line
    here). -/
inductive WasmInstr where
  -- Locals
  | localGet (idx : Nat)
  | localSet (idx : Nat)
  | localTee (idx : Nat)
  -- Constants
  | i32Const (n : Int)
  | i64Const (n : Int)
  | f32Const (bits : UInt32)
  | f64Const (bits : UInt64)
  -- Integer arithmetic (i32 only — the wide-int variants enter when
  -- a slice extends to i64).
  | i32Add | i32Sub | i32Mul
  | i32DivS | i32DivU | i32RemS | i32RemU
  | i32And | i32Or  | i32Xor
  | i32Shl | i32ShrS | i32ShrU
  | i32Eq  | i32Ne
  | i32LtS | i32LtU | i32GtS | i32GtU
  | i32LeS | i32LeU | i32GeS | i32GeU
  | i32Eqz
  -- Float arithmetic (f32 only).
  | f32Add | f32Sub | f32Mul | f32Div
  | f32Eq  | f32Ne  | f32Lt  | f32Gt | f32Le | f32Ge
  | f32Neg | f32Abs | f32Sqrt
  | f32Min | f32Max
  -- Conversions
  | i32WrapI64
  | f32ConvertI32S | f32ConvertI32U
  | i32TruncF32S   | i32TruncF32U
  | f32ReinterpretI32 | i32ReinterpretF32
  -- Memory (offset / align as raw u32/u64 — the proof project only
  -- inspects them through the abstract heap projection in
  -- `Quanta.Wasm.Translate`).
  | i32Load   (offset : Nat) (align : Nat)
  | i32Store  (offset : Nat) (align : Nat)
  | f32Load   (offset : Nat) (align : Nat)
  | f32Store  (offset : Nat) (align : Nat)
  | i32Load8U (offset : Nat) (align : Nat)
  | i32Load8S (offset : Nat) (align : Nat)
  | i32Store8 (offset : Nat) (align : Nat)
  -- Control flow. Block-typed ops carry their result arity (0 for
  -- the kernel subset; rustc emits 0-arity blocks for control flow).
  | block (arity : Nat)
  | wloop (arity : Nat)
  | wif   (arity : Nat)
  | welse
  | wend
  | br    (depth : Nat)
  | brIf  (depth : Nat)
  | wreturn
  -- Calls (intrinsics + user-defined).
  | call  (idx : Nat)
  -- Misc
  | drop
  | wselect
  | unreachable
  | nop
  /-- Catch-all for ops outside the lowered subset — execution traps
      under the operational semantics, lowering refuses with
      `UnsupportedOp`. Mirrors `RawInstr::Unsupported`. -/
  | unsupported (tag : String)
  deriving Repr

end Quanta.Wasm
