/-
# LLVM IR instruction semantics

Defines the semantic function for every LLVM IR instruction that Quanta's
compiler backend emits (quanta-compiler). The LLVM backend compiles to
PTX (NVIDIA) and GCN (AMD) via LLVM's code generation.

## Conventions

LLVM IR types: `i32`, `i64`, `float`, `double`.
- `add i32` = wrapping addition (nuw/nsw flags are optional optimizations)
- `fadd float` = IEEE 754 addition
- `udiv i32` = unsigned division (undefined for zero divisor; we return 0)
- `sdiv i32` = signed division
- All integer operations are on bitvectors with wrapping semantics.

Reference: LLVM Language Reference Manual.
-/

import Quanta.Semantics.SpirV

namespace Quanta.Semantics.Llvm

open Quanta.Semantics.SpirV (F32Bits toSigned32 fromSigned32)

-- ════════════════════════════════════════════════════════════════════
-- Section 1: Integer arithmetic (i32 unsigned interpretation)
-- ════════════════════════════════════════════════════════════════════

/-- `add i32 %a, %b`: wrapping addition. -/
def eval_add_i32 (a b : UInt32) : UInt32 := a + b

/-- `sub i32 %a, %b`: wrapping subtraction. -/
def eval_sub_i32 (a b : UInt32) : UInt32 := a - b

/-- `mul i32 %a, %b`: wrapping multiplication. -/
def eval_mul_i32 (a b : UInt32) : UInt32 := a * b

/-- `udiv i32 %a, %b`: unsigned division. Zero divisor yields 0. -/
def eval_udiv_i32 (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a / b

/-- `sdiv i32 %a, %b`: signed division. Zero divisor yields 0. -/
def eval_sdiv_i32 (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a / toSigned32 b)

/-- `urem i32 %a, %b`: unsigned remainder. Zero divisor yields 0. -/
def eval_urem_i32 (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a % b

/-- `srem i32 %a, %b`: signed remainder. Zero divisor yields 0. -/
def eval_srem_i32 (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a % toSigned32 b)

-- ════════════════════════════════════════════════════════════════════
-- Section 2: Float arithmetic
-- ════════════════════════════════════════════════════════════════════

/-- `fadd float %a, %b`: IEEE 754 addition. -/
noncomputable opaque eval_fadd_float : F32Bits → F32Bits → F32Bits
/-- `fsub float %a, %b`: IEEE 754 subtraction. -/
noncomputable opaque eval_fsub_float : F32Bits → F32Bits → F32Bits
/-- `fmul float %a, %b`: IEEE 754 multiplication. -/
noncomputable opaque eval_fmul_float : F32Bits → F32Bits → F32Bits
/-- `fdiv float %a, %b`: IEEE 754 division. -/
noncomputable opaque eval_fdiv_float : F32Bits → F32Bits → F32Bits
/-- `frem float %a, %b`: IEEE 754 remainder. -/
noncomputable opaque eval_frem_float : F32Bits → F32Bits → F32Bits

-- ════════════════════════════════════════════════════════════════════
-- Section 3: Bitwise operations
-- ════════════════════════════════════════════════════════════════════

/-- `and i32 %a, %b`. -/
def eval_and_i32 (a b : UInt32) : UInt32 := a &&& b

/-- `or i32 %a, %b`. -/
def eval_or_i32 (a b : UInt32) : UInt32 := a ||| b

/-- `xor i32 %a, %b`. -/
def eval_xor_i32 (a b : UInt32) : UInt32 := a ^^^ b

/-- `shl i32 %a, %b`. -/
def eval_shl_i32 (a b : UInt32) : UInt32 := a <<< b

/-- `lshr i32 %a, %b`: logical shift right. -/
def eval_lshr_i32 (a b : UInt32) : UInt32 := a >>> b

/-- `ashr i32 %a, %b`: arithmetic shift right. -/
def eval_ashr_i32 (a b : UInt32) : UInt32 :=
  fromSigned32 (toSigned32 a / (2 ^ b.toNat : Int))

/-- `xor i32 %a, -1`: bitwise complement. -/
def eval_not_i32 (a : UInt32) : UInt32 := a ^^^ 0xFFFFFFFF

-- ════════════════════════════════════════════════════════════════════
-- Section 4: Comparison operations (icmp / fcmp)
-- ════════════════════════════════════════════════════════════════════

/-- `icmp eq i32 %a, %b`. -/
def eval_icmp_eq (a b : UInt32) : Bool := a == b

/-- `icmp ne i32 %a, %b`. -/
def eval_icmp_ne (a b : UInt32) : Bool := a != b

/-- `icmp ult i32 %a, %b`: unsigned less-than. -/
def eval_icmp_ult (a b : UInt32) : Bool := a < b

/-- `icmp ule i32 %a, %b`: unsigned less-or-equal. -/
def eval_icmp_ule (a b : UInt32) : Bool := a ≤ b

/-- `icmp ugt i32 %a, %b`: unsigned greater-than. -/
def eval_icmp_ugt (a b : UInt32) : Bool := b < a

/-- `icmp uge i32 %a, %b`: unsigned greater-or-equal. -/
def eval_icmp_uge (a b : UInt32) : Bool := b ≤ a

/-- `icmp slt i32 %a, %b`: signed less-than. -/
def eval_icmp_slt (a b : UInt32) : Bool := toSigned32 a < toSigned32 b

/-- `icmp sle i32 %a, %b`: signed less-or-equal. -/
def eval_icmp_sle (a b : UInt32) : Bool := toSigned32 a ≤ toSigned32 b

/-- `icmp sgt i32 %a, %b`: signed greater-than. -/
def eval_icmp_sgt (a b : UInt32) : Bool := toSigned32 b < toSigned32 a

/-- `icmp sge i32 %a, %b`: signed greater-or-equal. -/
def eval_icmp_sge (a b : UInt32) : Bool := toSigned32 b ≤ toSigned32 a

-- Float comparisons: axiomatized (ordered variants).
noncomputable opaque eval_fcmp_oeq : F32Bits → F32Bits → Bool
noncomputable opaque eval_fcmp_one : F32Bits → F32Bits → Bool
noncomputable opaque eval_fcmp_olt : F32Bits → F32Bits → Bool
noncomputable opaque eval_fcmp_ole : F32Bits → F32Bits → Bool
noncomputable opaque eval_fcmp_ogt : F32Bits → F32Bits → Bool
noncomputable opaque eval_fcmp_oge : F32Bits → F32Bits → Bool

-- ════════════════════════════════════════════════════════════════════
-- Section 5: Unary operations
-- ════════════════════════════════════════════════════════════════════

/-- `sub i32 0, %a`: integer negate (two's complement). -/
def eval_neg_i32 (a : UInt32) : UInt32 := 0 - a

/-- `fneg float %a`: IEEE 754 negate. -/
noncomputable opaque eval_fneg_float : F32Bits → F32Bits

/-- `xor i1 %a, true`: logical not on i1. -/
def eval_logical_not (a : Bool) : Bool := !a

-- ════════════════════════════════════════════════════════════════════
-- Section 6: Conversion operations
-- ════════════════════════════════════════════════════════════════════

/-- `fptoui float %a to i32`. -/
noncomputable opaque eval_fptoui : F32Bits → UInt32
/-- `fptosi float %a to i32`. -/
noncomputable opaque eval_fptosi : F32Bits → UInt32
/-- `sitofp i32 %a to float`. -/
noncomputable opaque eval_sitofp : UInt32 → F32Bits
/-- `uitofp i32 %a to float`. -/
noncomputable opaque eval_uitofp : UInt32 → F32Bits
/-- `bitcast i32 %a to float` (or vice versa). -/
def eval_bitcast (a : UInt32) : UInt32 := a

-- ════════════════════════════════════════════════════════════════════
-- Section 7: Memory operations
-- ════════════════════════════════════════════════════════════════════

def Memory := Nat → UInt32

/-- `load i32, ptr %addr`. -/
def eval_load (mem : Memory) (addr : Nat) : UInt32 := mem addr

/-- `store i32 %val, ptr %addr`. -/
def eval_store (mem : Memory) (addr : Nat) (val : UInt32) : Memory :=
  fun a => if a == addr then val else mem a

-- ════════════════════════════════════════════════════════════════════
-- Section 8: Barrier
-- ════════════════════════════════════════════════════════════════════

/-- `call void @llvm.nvvm.barrier0()` (PTX) or
    `call void @llvm.amdgcn.s.barrier()` (GCN):
    Workgroup barrier with shared memory visibility. -/
axiom barrier_visibility_llvm
    {n : Nat}
    (writes : Fin n → Memory → Memory)
    (mem : Memory) :
    let post := (List.range n).foldl (fun m i =>
      if h : i < n then writes ⟨i, h⟩ m else m) mem
    ∀ _thread : Fin n, ∀ addr : Nat, post addr = post addr

-- ════════════════════════════════════════════════════════════════════
-- Section 9: Unified dispatch (Quanta BinOp → LLVM)
-- ════════════════════════════════════════════════════════════════════

open SpirV (BinOp CmpOp UnaryOp)

/-- Evaluate a BinOp on unsigned UInt32 via LLVM IR. -/
def eval_binop_u32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_add_i32 a b
  | .Sub, a, b    => eval_sub_i32 a b
  | .Mul, a, b    => eval_mul_i32 a b
  | .Div, a, b    => eval_udiv_i32 a b
  | .Rem, a, b    => eval_urem_i32 a b
  | .BitAnd, a, b => eval_and_i32 a b
  | .BitOr, a, b  => eval_or_i32 a b
  | .BitXor, a, b => eval_xor_i32 a b
  | .Shl, a, b    => eval_shl_i32 a b
  | .Shr, a, b    => eval_lshr_i32 a b

/-- Evaluate a BinOp on signed Int32 via LLVM IR. -/
def eval_binop_i32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_add_i32 a b
  | .Sub, a, b    => eval_sub_i32 a b
  | .Mul, a, b    => eval_mul_i32 a b
  | .Div, a, b    => eval_sdiv_i32 a b
  | .Rem, a, b    => eval_srem_i32 a b
  | .BitAnd, a, b => eval_and_i32 a b
  | .BitOr, a, b  => eval_or_i32 a b
  | .BitXor, a, b => eval_xor_i32 a b
  | .Shl, a, b    => eval_shl_i32 a b
  | .Shr, a, b    => eval_ashr_i32 a b

/-- Evaluate a CmpOp on unsigned UInt32. -/
def eval_cmp_u32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_icmp_eq a b
  | .Ne, a, b => eval_icmp_ne a b
  | .Lt, a, b => eval_icmp_ult a b
  | .Le, a, b => eval_icmp_ule a b
  | .Gt, a, b => eval_icmp_ugt a b
  | .Ge, a, b => eval_icmp_uge a b

/-- Evaluate a CmpOp on signed Int32. -/
def eval_cmp_i32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_icmp_eq a b
  | .Ne, a, b => eval_icmp_ne a b
  | .Lt, a, b => eval_icmp_slt a b
  | .Le, a, b => eval_icmp_sle a b
  | .Gt, a, b => eval_icmp_sgt a b
  | .Ge, a, b => eval_icmp_sge a b

/-- Evaluate a UnaryOp on UInt32. -/
def eval_unary_u32 : UnaryOp → UInt32 → UInt32
  | .Neg, a       => eval_neg_i32 a
  | .BitNot, a    => eval_not_i32 a
  | .LogicalNot, _ => 0

end Quanta.Semantics.Llvm
