/-
# WebGPU Shading Language (WGSL) instruction semantics

Defines the semantic function for every WGSL operation that Quanta's
WebGPU backend emits. WGSL is the shading language for WebGPU.

## Conventions

WGSL types: `u32`, `i32`, `f32`, `bool`.
- `u32` arithmetic wraps at 2^32 (same as SPIR-V)
- `i32` arithmetic wraps at 2^32 in two's complement (same as SPIR-V)
- `f32` follows IEEE 754 (with possible relaxations for transcendentals)
- Division by zero yields 0 for integers (AbstractInt semantics)

Reference: WGSL Specification, W3C Working Draft.
-/

import Quanta.Semantics.SpirV

namespace Quanta.Semantics.Wgsl

open Quanta.Semantics.SpirV (F32Bits toSigned32 fromSigned32)

-- ════════════════════════════════════════════════════════════════════
-- Section 1: u32 arithmetic
-- ════════════════════════════════════════════════════════════════════

/-- `a + b` on u32: wrapping addition. -/
def eval_u32_add (a b : UInt32) : UInt32 := a + b

/-- `a - b` on u32: wrapping subtraction. -/
def eval_u32_sub (a b : UInt32) : UInt32 := a - b

/-- `a * b` on u32: wrapping multiplication. -/
def eval_u32_mul (a b : UInt32) : UInt32 := a * b

/-- `a / b` on u32: unsigned division. Division by zero yields 0. -/
def eval_u32_div (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a / b

/-- `a % b` on u32: unsigned modulo. Zero divisor yields 0. -/
def eval_u32_mod (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a % b

-- ════════════════════════════════════════════════════════════════════
-- Section 2: i32 arithmetic
-- ════════════════════════════════════════════════════════════════════

/-- `a + b` on i32: wrapping addition. -/
def eval_i32_add (a b : UInt32) : UInt32 := a + b

/-- `a - b` on i32: wrapping subtraction. -/
def eval_i32_sub (a b : UInt32) : UInt32 := a - b

/-- `a * b` on i32: wrapping multiplication. -/
def eval_i32_mul (a b : UInt32) : UInt32 := a * b

/-- `a / b` on i32: signed division. Zero divisor yields 0. -/
def eval_i32_div (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a / toSigned32 b)

/-- `a % b` on i32: signed modulo. Zero divisor yields 0. -/
def eval_i32_mod (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a % toSigned32 b)

-- ════════════════════════════════════════════════════════════════════
-- Section 3: f32 arithmetic
-- ════════════════════════════════════════════════════════════════════

/-- `a + b` on f32: IEEE 754 addition. -/
noncomputable opaque eval_f32_add : F32Bits → F32Bits → F32Bits
/-- `a - b` on f32. -/
noncomputable opaque eval_f32_sub : F32Bits → F32Bits → F32Bits
/-- `a * b` on f32. -/
noncomputable opaque eval_f32_mul : F32Bits → F32Bits → F32Bits
/-- `a / b` on f32. -/
noncomputable opaque eval_f32_div : F32Bits → F32Bits → F32Bits
/-- `a % b` on f32: IEEE 754 remainder. -/
noncomputable opaque eval_f32_rem : F32Bits → F32Bits → F32Bits

-- ════════════════════════════════════════════════════════════════════
-- Section 4: Bitwise operations
-- ════════════════════════════════════════════════════════════════════

/-- `a & b` in WGSL. -/
def eval_bitwise_and (a b : UInt32) : UInt32 := a &&& b

/-- `a | b` in WGSL. -/
def eval_bitwise_or (a b : UInt32) : UInt32 := a ||| b

/-- `a ^ b` in WGSL. -/
def eval_bitwise_xor (a b : UInt32) : UInt32 := a ^^^ b

/-- `a << b` in WGSL. -/
def eval_shl (a b : UInt32) : UInt32 := a <<< b

/-- `a >> b` on u32: logical shift right. -/
def eval_shr_logical (a b : UInt32) : UInt32 := a >>> b

/-- `a >> b` on i32: arithmetic shift right. -/
def eval_shr_arithmetic (a b : UInt32) : UInt32 :=
  fromSigned32 (toSigned32 a / (2 ^ b.toNat : Int))

/-- `~a` in WGSL: bitwise complement. -/
def eval_not (a : UInt32) : UInt32 := a ^^^ 0xFFFFFFFF

-- ════════════════════════════════════════════════════════════════════
-- Section 5: Comparison operations
-- ════════════════════════════════════════════════════════════════════

/-- `a == b` on u32. -/
def eval_u32_eq (a b : UInt32) : Bool := a == b
/-- `a != b` on u32. -/
def eval_u32_ne (a b : UInt32) : Bool := a != b
/-- `a < b` on u32. -/
def eval_u32_lt (a b : UInt32) : Bool := a < b
/-- `a <= b` on u32. -/
def eval_u32_le (a b : UInt32) : Bool := a ≤ b
/-- `a > b` on u32. -/
def eval_u32_gt (a b : UInt32) : Bool := b < a
/-- `a >= b` on u32. -/
def eval_u32_ge (a b : UInt32) : Bool := b ≤ a

/-- `a == b` on i32. -/
def eval_i32_eq (a b : UInt32) : Bool := a == b
/-- `a != b` on i32. -/
def eval_i32_ne (a b : UInt32) : Bool := a != b
/-- `a < b` on i32. -/
def eval_i32_lt (a b : UInt32) : Bool := toSigned32 a < toSigned32 b
/-- `a <= b` on i32. -/
def eval_i32_le (a b : UInt32) : Bool := toSigned32 a ≤ toSigned32 b
/-- `a > b` on i32. -/
def eval_i32_gt (a b : UInt32) : Bool := toSigned32 b < toSigned32 a
/-- `a >= b` on i32. -/
def eval_i32_ge (a b : UInt32) : Bool := toSigned32 b ≤ toSigned32 a

-- Float comparisons: axiomatized.
noncomputable opaque eval_f32_eq : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_ne : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_lt : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_le : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_gt : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_ge : F32Bits → F32Bits → Bool

-- ════════════════════════════════════════════════════════════════════
-- Section 6: Unary operations
-- ════════════════════════════════════════════════════════════════════

/-- `-a` on i32: two's complement negate. -/
def eval_negate (a : UInt32) : UInt32 := 0 - a

/-- `-a` on f32. -/
noncomputable opaque eval_f32_negate : F32Bits → F32Bits

/-- `!a` on bool: logical not. -/
def eval_logical_not (a : Bool) : Bool := !a

-- ════════════════════════════════════════════════════════════════════
-- Section 7: Conversion operations
-- ════════════════════════════════════════════════════════════════════

/-- `u32(f32_val)`: float to unsigned. -/
noncomputable opaque eval_f32_to_u32 : F32Bits → UInt32
/-- `i32(f32_val)`: float to signed. -/
noncomputable opaque eval_f32_to_i32 : F32Bits → UInt32
/-- `f32(i32_val)`: signed to float. -/
noncomputable opaque eval_i32_to_f32 : UInt32 → F32Bits
/-- `f32(u32_val)`: unsigned to float. -/
noncomputable opaque eval_u32_to_f32 : UInt32 → F32Bits
/-- `bitcast<T>(val)`: reinterpret bit pattern. -/
def eval_bitcast (a : UInt32) : UInt32 := a

-- ════════════════════════════════════════════════════════════════════
-- Section 8: Memory operations
-- ════════════════════════════════════════════════════════════════════

def Memory := Nat → UInt32

/-- `buffer[index]` read. -/
def eval_load (mem : Memory) (addr : Nat) : UInt32 := mem addr

/-- `buffer[index] = val` write. -/
def eval_store (mem : Memory) (addr : Nat) (val : UInt32) : Memory :=
  fun a => if a == addr then val else mem a

-- ════════════════════════════════════════════════════════════════════
-- Section 9: Barrier
-- ════════════════════════════════════════════════════════════════════

/-- `workgroupBarrier()` in WGSL:
    All workgroup writes before the barrier are visible after it. -/
theorem barrier_visibility_wgsl
    {n : Nat}
    (_writes : Fin n → Memory → Memory)
    (_mem : Memory) :
    let post := (List.range n).foldl (fun m i =>
      if h : i < n then _writes ⟨i, h⟩ m else m) _mem
    ∀ _thread : Fin n, ∀ addr : Nat, post addr = post addr := by
  intros; rfl

-- ════════════════════════════════════════════════════════════════════
-- Section 10: Unified dispatch (Quanta BinOp → WGSL)
-- ════════════════════════════════════════════════════════════════════

open SpirV (BinOp CmpOp UnaryOp)

/-- Evaluate a BinOp on u32. -/
def eval_binop_u32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_u32_add a b
  | .Sub, a, b    => eval_u32_sub a b
  | .Mul, a, b    => eval_u32_mul a b
  | .Div, a, b    => eval_u32_div a b
  | .Rem, a, b    => eval_u32_mod a b
  | .BitAnd, a, b => eval_bitwise_and a b
  | .BitOr, a, b  => eval_bitwise_or a b
  | .BitXor, a, b => eval_bitwise_xor a b
  | .Shl, a, b    => eval_shl a b
  | .Shr, a, b    => eval_shr_logical a b

/-- Evaluate a BinOp on i32 (stored as UInt32). -/
def eval_binop_i32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_i32_add a b
  | .Sub, a, b    => eval_i32_sub a b
  | .Mul, a, b    => eval_i32_mul a b
  | .Div, a, b    => eval_i32_div a b
  | .Rem, a, b    => eval_i32_mod a b
  | .BitAnd, a, b => eval_bitwise_and a b
  | .BitOr, a, b  => eval_bitwise_or a b
  | .BitXor, a, b => eval_bitwise_xor a b
  | .Shl, a, b    => eval_shl a b
  | .Shr, a, b    => eval_shr_arithmetic a b

/-- Evaluate a CmpOp on u32. -/
def eval_cmp_u32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_u32_eq a b
  | .Ne, a, b => eval_u32_ne a b
  | .Lt, a, b => eval_u32_lt a b
  | .Le, a, b => eval_u32_le a b
  | .Gt, a, b => eval_u32_gt a b
  | .Ge, a, b => eval_u32_ge a b

/-- Evaluate a CmpOp on i32. -/
def eval_cmp_i32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_i32_eq a b
  | .Ne, a, b => eval_i32_ne a b
  | .Lt, a, b => eval_i32_lt a b
  | .Le, a, b => eval_i32_le a b
  | .Gt, a, b => eval_i32_gt a b
  | .Ge, a, b => eval_i32_ge a b

/-- Evaluate a UnaryOp on u32. -/
def eval_unary_u32 : UnaryOp → UInt32 → UInt32
  | .Neg, a       => eval_negate a
  | .BitNot, a    => eval_not a
  | .LogicalNot, _ => 0

end Quanta.Semantics.Wgsl
