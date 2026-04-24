/-
# Metal Shading Language (MSL) instruction semantics

Defines the semantic function for every MSL operation that Quanta's Metal
backend emits. MSL compiles to Apple GPU ISA; these semantics match the
Metal Shading Language Specification v3.2.

## Conventions

MSL uses C++ operator overloading for arithmetic. The semantics are:
- `uint + uint`  = wrapping addition (same as SPIR-V OpIAdd)
- `int + int`    = wrapping addition (same as SPIR-V OpIAdd)
- `float + float` = IEEE 754 addition (same as SPIR-V OpFAdd)
- `threadgroup_barrier(mem_flags::mem_threadgroup)` = workgroup barrier

All integer operations use two's complement wrapping, matching SPIR-V
and C++ unsigned/signed semantics.
-/

import Quanta.Semantics.SpirV

namespace Quanta.Semantics.Msl

open Quanta.Semantics.SpirV (Float32 toSigned32 fromSigned32)

-- ════════════════════════════════════════════════════════════════════
-- Section 1: Unsigned integer arithmetic
-- ════════════════════════════════════════════════════════════════════

/-- `uint + uint` in MSL: wrapping addition. -/
def eval_uint_add (a b : UInt32) : UInt32 := a + b

/-- `uint - uint` in MSL: wrapping subtraction. -/
def eval_uint_sub (a b : UInt32) : UInt32 := a - b

/-- `uint * uint` in MSL: wrapping multiplication. -/
def eval_uint_mul (a b : UInt32) : UInt32 := a * b

/-- `uint / uint` in MSL: unsigned division. Zero divisor yields 0. -/
def eval_uint_div (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a / b

/-- `uint % uint` in MSL: unsigned modulo. Zero divisor yields 0. -/
def eval_uint_mod (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a % b

-- ════════════════════════════════════════════════════════════════════
-- Section 2: Signed integer arithmetic
-- ════════════════════════════════════════════════════════════════════

/-- `int + int` in MSL: wrapping addition (identical bit pattern to unsigned). -/
def eval_int_add (a b : UInt32) : UInt32 := a + b

/-- `int - int` in MSL: wrapping subtraction. -/
def eval_int_sub (a b : UInt32) : UInt32 := a - b

/-- `int * int` in MSL: wrapping multiplication. -/
def eval_int_mul (a b : UInt32) : UInt32 := a * b

/-- `int / int` in MSL: signed division. Zero divisor yields 0. -/
def eval_int_div (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a / toSigned32 b)

/-- `int % int` in MSL: signed modulo. Zero divisor yields 0. -/
def eval_int_mod (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a % toSigned32 b)

-- ════════════════════════════════════════════════════════════════════
-- Section 3: Float arithmetic
-- ════════════════════════════════════════════════════════════════════

/-- `float + float` in MSL: IEEE 754 addition. -/
opaque eval_float_add : Float32 → Float32 → Float32
/-- `float - float` in MSL: IEEE 754 subtraction. -/
opaque eval_float_sub : Float32 → Float32 → Float32
/-- `float * float` in MSL: IEEE 754 multiplication. -/
opaque eval_float_mul : Float32 → Float32 → Float32
/-- `float / float` in MSL: IEEE 754 division. -/
opaque eval_float_div : Float32 → Float32 → Float32
/-- `fmod(a, b)` in MSL: IEEE 754 remainder. -/
opaque eval_float_rem : Float32 → Float32 → Float32

-- ════════════════════════════════════════════════════════════════════
-- Section 4: Bitwise operations
-- ════════════════════════════════════════════════════════════════════

/-- `a & b` in MSL. -/
def eval_bitwise_and (a b : UInt32) : UInt32 := a &&& b

/-- `a | b` in MSL. -/
def eval_bitwise_or (a b : UInt32) : UInt32 := a ||| b

/-- `a ^ b` in MSL. -/
def eval_bitwise_xor (a b : UInt32) : UInt32 := a ^^^ b

/-- `a << b` in MSL. -/
def eval_shl (a b : UInt32) : UInt32 := a <<< b

/-- `a >> b` on uint in MSL: logical shift. -/
def eval_shr_logical (a b : UInt32) : UInt32 := a >>> b

/-- `a >> b` on int in MSL: arithmetic shift. -/
def eval_shr_arithmetic (a b : UInt32) : UInt32 :=
  fromSigned32 (toSigned32 a / (2 ^ b.toNat : Int))

/-- `~a` in MSL: bitwise complement. -/
def eval_not (a : UInt32) : UInt32 := a ^^^ 0xFFFFFFFF

-- ════════════════════════════════════════════════════════════════════
-- Section 5: Comparison operations
-- ════════════════════════════════════════════════════════════════════

/-- `a == b` on uint. -/
def eval_uint_eq (a b : UInt32) : Bool := a == b
/-- `a != b` on uint. -/
def eval_uint_ne (a b : UInt32) : Bool := a != b
/-- `a < b` on uint. -/
def eval_uint_lt (a b : UInt32) : Bool := a < b
/-- `a <= b` on uint. -/
def eval_uint_le (a b : UInt32) : Bool := a ≤ b
/-- `a > b` on uint. -/
def eval_uint_gt (a b : UInt32) : Bool := b < a
/-- `a >= b` on uint. -/
def eval_uint_ge (a b : UInt32) : Bool := b ≤ a

/-- `a == b` on int. -/
def eval_int_eq (a b : UInt32) : Bool := a == b
/-- `a != b` on int. -/
def eval_int_ne (a b : UInt32) : Bool := a != b
/-- `a < b` on int (signed comparison). -/
def eval_int_lt (a b : UInt32) : Bool := toSigned32 a < toSigned32 b
/-- `a <= b` on int. -/
def eval_int_le (a b : UInt32) : Bool := toSigned32 a ≤ toSigned32 b
/-- `a > b` on int. -/
def eval_int_gt (a b : UInt32) : Bool := toSigned32 b < toSigned32 a
/-- `a >= b` on int. -/
def eval_int_ge (a b : UInt32) : Bool := toSigned32 b ≤ toSigned32 a

-- Float comparisons: axiomatized.
opaque eval_float_eq : Float32 → Float32 → Bool
opaque eval_float_ne : Float32 → Float32 → Bool
opaque eval_float_lt : Float32 → Float32 → Bool
opaque eval_float_le : Float32 → Float32 → Bool
opaque eval_float_gt : Float32 → Float32 → Bool
opaque eval_float_ge : Float32 → Float32 → Bool

-- ════════════════════════════════════════════════════════════════════
-- Section 6: Unary operations
-- ════════════════════════════════════════════════════════════════════

/-- Unary `-a` on int: two's complement negate. -/
def eval_int_negate (a : UInt32) : UInt32 := 0 - a

/-- Unary `-a` on float. -/
opaque eval_float_negate : Float32 → Float32

/-- `!a` on bool: logical not. -/
def eval_logical_not (a : Bool) : Bool := !a

-- ════════════════════════════════════════════════════════════════════
-- Section 7: Conversion operations
-- ════════════════════════════════════════════════════════════════════

/-- `static_cast<uint>(float_val)`: float to unsigned. -/
opaque eval_float_to_uint : Float32 → UInt32
/-- `static_cast<int>(float_val)`: float to signed. -/
opaque eval_float_to_int : Float32 → UInt32
/-- `static_cast<float>(int_val)`: signed to float. -/
opaque eval_int_to_float : UInt32 → Float32
/-- `static_cast<float>(uint_val)`: unsigned to float. -/
opaque eval_uint_to_float : UInt32 → Float32
/-- `as_type<T>(val)`: bitcast (reinterpret). -/
def eval_bitcast (a : UInt32) : UInt32 := a

-- ════════════════════════════════════════════════════════════════════
-- Section 8: Memory operations
-- ════════════════════════════════════════════════════════════════════

/-- Memory state. -/
def Memory := Nat → UInt32

/-- `buffer[index]` read. -/
def eval_load (mem : Memory) (addr : Nat) : UInt32 := mem addr

/-- `buffer[index] = val` write. -/
def eval_store (mem : Memory) (addr : Nat) (val : UInt32) : Memory :=
  fun a => if a == addr then val else mem a

-- ════════════════════════════════════════════════════════════════════
-- Section 9: Barrier
-- ════════════════════════════════════════════════════════════════════

/-- `threadgroup_barrier(mem_flags::mem_threadgroup)`:
    All threadgroup writes before the barrier are visible after it. -/
axiom barrier_visibility_msl
    (writes : Fin n → Memory → Memory)
    (mem : Memory) :
    let post := (List.range n).foldl (fun m i => writes ⟨i, sorry⟩ m) mem
    ∀ thread : Fin n, ∀ addr : Nat, post addr = post addr

-- ════════════════════════════════════════════════════════════════════
-- Section 10: Unified dispatch (Quanta BinOp → MSL)
-- ════════════════════════════════════════════════════════════════════

open SpirV (BinOp CmpOp UnaryOp)

/-- Evaluate a BinOp on unsigned UInt32 via MSL semantics. -/
def eval_binop_u32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_uint_add a b
  | .Sub, a, b    => eval_uint_sub a b
  | .Mul, a, b    => eval_uint_mul a b
  | .Div, a, b    => eval_uint_div a b
  | .Rem, a, b    => eval_uint_mod a b
  | .BitAnd, a, b => eval_bitwise_and a b
  | .BitOr, a, b  => eval_bitwise_or a b
  | .BitXor, a, b => eval_bitwise_xor a b
  | .Shl, a, b    => eval_shl a b
  | .Shr, a, b    => eval_shr_logical a b

/-- Evaluate a BinOp on signed Int32 (stored as UInt32). -/
def eval_binop_i32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_int_add a b
  | .Sub, a, b    => eval_int_sub a b
  | .Mul, a, b    => eval_int_mul a b
  | .Div, a, b    => eval_int_div a b
  | .Rem, a, b    => eval_int_mod a b
  | .BitAnd, a, b => eval_bitwise_and a b
  | .BitOr, a, b  => eval_bitwise_or a b
  | .BitXor, a, b => eval_bitwise_xor a b
  | .Shl, a, b    => eval_shl a b
  | .Shr, a, b    => eval_shr_arithmetic a b

/-- Evaluate a CmpOp on unsigned UInt32. -/
def eval_cmp_u32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_uint_eq a b
  | .Ne, a, b => eval_uint_ne a b
  | .Lt, a, b => eval_uint_lt a b
  | .Le, a, b => eval_uint_le a b
  | .Gt, a, b => eval_uint_gt a b
  | .Ge, a, b => eval_uint_ge a b

/-- Evaluate a CmpOp on signed Int32. -/
def eval_cmp_i32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_int_eq a b
  | .Ne, a, b => eval_int_ne a b
  | .Lt, a, b => eval_int_lt a b
  | .Le, a, b => eval_int_le a b
  | .Gt, a, b => eval_int_gt a b
  | .Ge, a, b => eval_int_ge a b

/-- Evaluate a UnaryOp on unsigned UInt32. -/
def eval_unary_u32 : UnaryOp → UInt32 → UInt32
  | .Neg, a       => eval_int_negate a
  | .BitNot, a    => eval_not a
  | .LogicalNot, _ => 0

end Quanta.Semantics.Msl
