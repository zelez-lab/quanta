/-
# CPU executor instruction semantics

Defines the semantic function for every operation in the Quanta CPU
executor (driver/cpu/eval.rs). This is the reference implementation
that all GPU backends must agree with.

## Conventions

The CPU executor uses Rust's wrapping arithmetic:
- `a.wrapping_add(b)` on u32 → UInt32 wrapping add
- `a.wrapping_sub(b)` on u32 → UInt32 wrapping sub
- Division by zero returns 0 (not a trap)
- Signed operations use `i32::wrapping_*` methods
- Cast operations use Rust's `as` semantics

Reference: src/driver/cpu/eval.rs
-/

import Quanta.Semantics.SpirV

namespace Quanta.Semantics.Cpu

open Quanta.Semantics.SpirV (F32Bits toSigned32 fromSigned32)

-- ════════════════════════════════════════════════════════════════════
-- Section 1: Unsigned integer binary operations (matches eval_binop for U32)
-- ════════════════════════════════════════════════════════════════════

/-- `va.wrapping_add(vb)` in Rust. -/
def eval_u32_wrapping_add (a b : UInt32) : UInt32 := a + b

/-- `va.wrapping_sub(vb)` in Rust. -/
def eval_u32_wrapping_sub (a b : UInt32) : UInt32 := a - b

/-- `va.wrapping_mul(vb)` in Rust. -/
def eval_u32_wrapping_mul (a b : UInt32) : UInt32 := a * b

/-- `if vb == 0 { 0 } else { va / vb }` in Rust (unsigned). -/
def eval_u32_div (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a / b

/-- `if vb == 0 { 0 } else { va % vb }` in Rust (unsigned). -/
def eval_u32_rem (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a % b

/-- `va & vb` in Rust. -/
def eval_u32_bitand (a b : UInt32) : UInt32 := a &&& b

/-- `va | vb` in Rust. -/
def eval_u32_bitor (a b : UInt32) : UInt32 := a ||| b

/-- `va ^ vb` in Rust. -/
def eval_u32_bitxor (a b : UInt32) : UInt32 := a ^^^ b

/-- `va.wrapping_shl(vb)` in Rust. -/
def eval_u32_shl (a b : UInt32) : UInt32 := a <<< b

/-- `va.wrapping_shr(vb)` in Rust (logical shift for unsigned). -/
def eval_u32_shr (a b : UInt32) : UInt32 := a >>> b

-- ════════════════════════════════════════════════════════════════════
-- Section 2: Signed integer binary operations (matches eval_binop for I32)
-- ════════════════════════════════════════════════════════════════════

/-- `va.wrapping_add(vb)` on i32 — same bit pattern as unsigned. -/
def eval_i32_wrapping_add (a b : UInt32) : UInt32 := a + b

/-- `va.wrapping_sub(vb)` on i32. -/
def eval_i32_wrapping_sub (a b : UInt32) : UInt32 := a - b

/-- `va.wrapping_mul(vb)` on i32. -/
def eval_i32_wrapping_mul (a b : UInt32) : UInt32 := a * b

/-- `va.wrapping_div(vb)` on i32 with zero check. -/
def eval_i32_div (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a / toSigned32 b)

/-- `va.wrapping_rem(vb)` on i32 with zero check. -/
def eval_i32_rem (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a % toSigned32 b)

/-- `va & vb` on i32 — same as unsigned bitwise AND. -/
def eval_i32_bitand (a b : UInt32) : UInt32 := a &&& b

/-- `va | vb` on i32. -/
def eval_i32_bitor (a b : UInt32) : UInt32 := a ||| b

/-- `va ^ vb` on i32. -/
def eval_i32_bitxor (a b : UInt32) : UInt32 := a ^^^ b

/-- `va.wrapping_shl(vb as u32)` on i32. -/
def eval_i32_shl (a b : UInt32) : UInt32 := a <<< b

/-- `va.wrapping_shr(vb as u32)` on i32 — arithmetic shift. -/
def eval_i32_shr (a b : UInt32) : UInt32 :=
  fromSigned32 (toSigned32 a / (2 ^ b.toNat : Int))

-- ════════════════════════════════════════════════════════════════════
-- Section 3: Float operations (axiomatized)
-- ════════════════════════════════════════════════════════════════════

/-- `va + vb` on f32 (Rust `+` operator). -/
noncomputable opaque eval_f32_add : F32Bits → F32Bits → F32Bits
/-- `va - vb` on f32. -/
noncomputable opaque eval_f32_sub : F32Bits → F32Bits → F32Bits
/-- `va * vb` on f32. -/
noncomputable opaque eval_f32_mul : F32Bits → F32Bits → F32Bits
/-- `va / vb` on f32. -/
noncomputable opaque eval_f32_div : F32Bits → F32Bits → F32Bits
/-- `va % vb` on f32. -/
noncomputable opaque eval_f32_rem : F32Bits → F32Bits → F32Bits

-- ════════════════════════════════════════════════════════════════════
-- Section 4: Comparison operations (matches eval_cmp)
-- ════════════════════════════════════════════════════════════════════

/-- `va == vb` on u32. -/
def eval_u32_eq (a b : UInt32) : Bool := a == b
/-- `va != vb` on u32. -/
def eval_u32_ne (a b : UInt32) : Bool := a != b
/-- `va < vb` on u32. -/
def eval_u32_lt (a b : UInt32) : Bool := a < b
/-- `va <= vb` on u32. -/
def eval_u32_le (a b : UInt32) : Bool := a ≤ b
/-- `va > vb` on u32. -/
def eval_u32_gt (a b : UInt32) : Bool := b < a
/-- `va >= vb` on u32. -/
def eval_u32_ge (a b : UInt32) : Bool := b ≤ a

/-- `va == vb` on i32. -/
def eval_i32_eq (a b : UInt32) : Bool := a == b
/-- `va != vb` on i32. -/
def eval_i32_ne (a b : UInt32) : Bool := a != b
/-- `va < vb` on i32 (signed). -/
def eval_i32_lt (a b : UInt32) : Bool := toSigned32 a < toSigned32 b
/-- `va <= vb` on i32. -/
def eval_i32_le (a b : UInt32) : Bool := toSigned32 a ≤ toSigned32 b
/-- `va > vb` on i32. -/
def eval_i32_gt (a b : UInt32) : Bool := toSigned32 b < toSigned32 a
/-- `va >= vb` on i32. -/
def eval_i32_ge (a b : UInt32) : Bool := toSigned32 b ≤ toSigned32 a

-- Float comparisons: axiomatized.
noncomputable opaque eval_f32_eq : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_ne : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_lt : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_le : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_gt : F32Bits → F32Bits → Bool
noncomputable opaque eval_f32_ge : F32Bits → F32Bits → Bool

-- ════════════════════════════════════════════════════════════════════
-- Section 5: Unary operations (matches eval_unary)
-- ════════════════════════════════════════════════════════════════════

/-- `a.as_u32().wrapping_neg()` — two's complement negate. -/
def eval_u32_neg (a : UInt32) : UInt32 := 0 - a

/-- `!a.as_u32()` — bitwise complement. -/
def eval_u32_bitnot (a : UInt32) : UInt32 := a ^^^ 0xFFFFFFFF

/-- `!a.as_bool()` — logical not. -/
def eval_logical_not (a : Bool) : Bool := !a

/-- `-a.as_f32()` — float negate. -/
noncomputable opaque eval_f32_neg : F32Bits → F32Bits

-- ════════════════════════════════════════════════════════════════════
-- Section 6: Cast operations (matches eval_cast)
-- ════════════════════════════════════════════════════════════════════

/-- `val.as_f32() as u32` — float to unsigned via Rust `as`. -/
noncomputable opaque eval_f32_to_u32 : F32Bits → UInt32
/-- `val.as_f32() as i32` (stored as UInt32). -/
noncomputable opaque eval_f32_to_i32 : F32Bits → UInt32
/-- `val.as_i32() as f32`. -/
noncomputable opaque eval_i32_to_f32 : UInt32 → F32Bits
/-- `val.as_u32() as f32`. -/
noncomputable opaque eval_u32_to_f32 : UInt32 → F32Bits
/-- Identity cast for same-width integer types. -/
def eval_bitcast (a : UInt32) : UInt32 := a

-- ════════════════════════════════════════════════════════════════════
-- Section 7: Memory operations (matches KernelOp::Load/Store)
-- ════════════════════════════════════════════════════════════════════

def Memory := Nat → UInt32

/-- `read_scalar(buf, idx, ty)` — read from buffer. -/
def eval_load (mem : Memory) (addr : Nat) : UInt32 := mem addr

/-- `write_scalar(buf, idx, val, ty)` — write to buffer. -/
def eval_store (mem : Memory) (addr : Nat) (val : UInt32) : Memory :=
  fun a => if a == addr then val else mem a

-- ════════════════════════════════════════════════════════════════════
-- Section 8: Control flow (matches KernelOp::Branch)
-- ════════════════════════════════════════════════════════════════════

/-- `if cv.as_bool() { then_ops } else { else_ops }` — conditional branch.
    Returns the value produced by the taken branch. -/
def eval_branch (cond : Bool) (then_val else_val : UInt32) : UInt32 :=
  if cond then then_val else else_val

-- ════════════════════════════════════════════════════════════════════
-- Section 9: Unified dispatch (Quanta BinOp → CPU)
-- ════════════════════════════════════════════════════════════════════

open SpirV (BinOp CmpOp UnaryOp)

/-- Evaluate a BinOp on unsigned UInt32 (mirrors eval_binop for U32). -/
def eval_binop_u32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_u32_wrapping_add a b
  | .Sub, a, b    => eval_u32_wrapping_sub a b
  | .Mul, a, b    => eval_u32_wrapping_mul a b
  | .Div, a, b    => eval_u32_div a b
  | .Rem, a, b    => eval_u32_rem a b
  | .BitAnd, a, b => eval_u32_bitand a b
  | .BitOr, a, b  => eval_u32_bitor a b
  | .BitXor, a, b => eval_u32_bitxor a b
  | .Shl, a, b    => eval_u32_shl a b
  | .Shr, a, b    => eval_u32_shr a b

/-- Evaluate a BinOp on signed Int32. -/
def eval_binop_i32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_i32_wrapping_add a b
  | .Sub, a, b    => eval_i32_wrapping_sub a b
  | .Mul, a, b    => eval_i32_wrapping_mul a b
  | .Div, a, b    => eval_i32_div a b
  | .Rem, a, b    => eval_i32_rem a b
  | .BitAnd, a, b => eval_i32_bitand a b
  | .BitOr, a, b  => eval_i32_bitor a b
  | .BitXor, a, b => eval_i32_bitxor a b
  | .Shl, a, b    => eval_i32_shl a b
  | .Shr, a, b    => eval_i32_shr a b

/-- Evaluate a CmpOp on unsigned UInt32. -/
def eval_cmp_u32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_u32_eq a b
  | .Ne, a, b => eval_u32_ne a b
  | .Lt, a, b => eval_u32_lt a b
  | .Le, a, b => eval_u32_le a b
  | .Gt, a, b => eval_u32_gt a b
  | .Ge, a, b => eval_u32_ge a b

/-- Evaluate a CmpOp on signed Int32. -/
def eval_cmp_i32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_i32_eq a b
  | .Ne, a, b => eval_i32_ne a b
  | .Lt, a, b => eval_i32_lt a b
  | .Le, a, b => eval_i32_le a b
  | .Gt, a, b => eval_i32_gt a b
  | .Ge, a, b => eval_i32_ge a b

/-- Evaluate a UnaryOp on unsigned UInt32. -/
def eval_unary_u32 : UnaryOp → UInt32 → UInt32
  | .Neg, a       => eval_u32_neg a
  | .BitNot, a    => eval_u32_bitnot a
  | .LogicalNot, _ => 0

end Quanta.Semantics.Cpu
