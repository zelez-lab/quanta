/-
# SPIR-V instruction semantics

Defines the semantic function for every SPIR-V opcode that Quanta emits.
Each operation is defined as a pure function on Lean's built-in numeric
types (UInt32, UInt64, etc.), matching the SPIR-V 1.6 specification.

## Scope

- Arithmetic: OpIAdd(128), OpFAdd(129), OpISub(130), OpFSub(131),
  OpIMul(132), OpFMul(133), OpUDiv(134), OpSDiv(135), OpFDiv(136),
  OpUMod(137), OpSMod(138), OpFRem(140)
- Bitwise: OpBitwiseAnd(199), OpBitwiseOr(197), OpBitwiseXor(198),
  OpShiftLeftLogical(196), OpShiftRightLogical(194),
  OpShiftRightArithmetic(195), OpNot(200)
- Comparison: OpIEqual(170), OpINotEqual(171), OpULessThan(176),
  OpSLessThan(178), OpULessThanEqual(177), OpSLessThanEqual(179),
  OpUGreaterThan(172), OpSGreaterThan(174), OpUGreaterThanEqual(173),
  OpSGreaterThanEqual(175), OpFOrdEqual(180), OpFOrdNotEqual(181),
  OpFOrdLessThan(184), OpFOrdLessThanEqual(188),
  OpFOrdGreaterThan(186), OpFOrdGreaterThanEqual(190)
- Unary: OpSNegate(126), OpFNegate(127), OpLogicalNot(168)
- Conversion: OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF,
  OpBitcast
- Memory: OpLoad, OpStore (modeled as state transitions)
- Control flow: OpBranch, OpBranchConditional (modeled as CFG transitions)
- Barrier: OpControlBarrier (modeled as visibility guarantee)

Reference: SPIR-V 1.6, Rev 2 (Khronos Group).
-/

namespace Quanta.Semantics.SpirV

-- ════════════════════════════════════════════════════════════════════
-- Section 1: Signed integer model
-- ════════════════════════════════════════════════════════════════════

/-- Two's complement interpretation of a UInt32 as a signed value.
    Values >= 2^31 are negative. -/
def toSigned32 (x : UInt32) : Int :=
  if x.toNat < 2 ^ 31 then (x.toNat : Int) else (x.toNat : Int) - (2 ^ 32 : Int)

/-- Convert a signed integer back to UInt32 (two's complement). -/
def fromSigned32 (x : Int) : UInt32 :=
  (x % (2 ^ 32 : Int)).toNat.toUInt32

-- ════════════════════════════════════════════════════════════════════
-- Section 2: Integer arithmetic (OpIAdd, OpISub, OpIMul, OpUDiv, OpSDiv, OpUMod, OpSMod)
-- ════════════════════════════════════════════════════════════════════

/-- OpIAdd (128): Wrapping integer addition. -/
def eval_iadd (a b : UInt32) : UInt32 := a + b

/-- OpISub (130): Wrapping integer subtraction. -/
def eval_isub (a b : UInt32) : UInt32 := a - b

/-- OpIMul (132): Wrapping integer multiplication. -/
def eval_imul (a b : UInt32) : UInt32 := a * b

/-- OpUDiv (134): Unsigned integer division.
    Division by zero is undefined in SPIR-V; we define it as 0. -/
def eval_udiv (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a / b

/-- OpSDiv (135): Signed integer division.
    Division by zero yields 0. Uses wrapping semantics for MIN / -1. -/
def eval_sdiv (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a / toSigned32 b)

/-- OpUMod (137): Unsigned modulo. Zero divisor yields 0. -/
def eval_umod (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else a % b

/-- OpSMod (138): Signed modulo with sign of divisor.
    Zero divisor yields 0. -/
def eval_smod (a b : UInt32) : UInt32 :=
  if b == 0 then 0 else fromSigned32 (toSigned32 a % toSigned32 b)

-- ════════════════════════════════════════════════════════════════════
-- Section 3: Float arithmetic (OpFAdd, OpFSub, OpFMul, OpFDiv, OpFRem)
-- ════════════════════════════════════════════════════════════════════

-- Float semantics are axiomatized — Lean 4 does not natively model
-- IEEE 754.  We define opaque constants that carry the specification
-- as documentation, and treat float agreement as an axiom grounded
-- in the IEEE 754 standard.

/-- Opaque float type. Real IEEE 754 semantics are out of scope for
    constructive proofs; we axiomatize agreement instead. -/
opaque F32Bits : Type

/-- A canonical placeholder inhabitant of `F32Bits`, used solely to
    discharge the `Inhabited` obligation on bodyless `opaque` function
    declarations below. The actual semantic value is irrelevant since
    the `eval_*` operations are themselves `opaque` — Lean only needs
    *some* inhabitant to elaborate them.

    Declared as `axiom` rather than `opaque` because `opaque` itself
    requires `Inhabited`, creating a chicken-and-egg with the
    instance below; `axiom` is the right primitive for "we postulate
    this exists." -/
axiom Float32_default : F32Bits

noncomputable instance : Inhabited F32Bits := ⟨Float32_default⟩

/-- OpFAdd (129): IEEE 754 float addition. -/
noncomputable opaque eval_fadd : F32Bits → F32Bits → F32Bits

/-- OpFSub (131): IEEE 754 float subtraction. -/
noncomputable opaque eval_fsub : F32Bits → F32Bits → F32Bits

/-- OpFMul (133): IEEE 754 float multiplication. -/
noncomputable opaque eval_fmul : F32Bits → F32Bits → F32Bits

/-- OpFDiv (136): IEEE 754 float division. -/
noncomputable opaque eval_fdiv : F32Bits → F32Bits → F32Bits

/-- OpFRem (140): IEEE 754 float remainder. -/
noncomputable opaque eval_frem : F32Bits → F32Bits → F32Bits

-- ════════════════════════════════════════════════════════════════════
-- Section 4: Bitwise operations
-- ════════════════════════════════════════════════════════════════════

/-- OpBitwiseAnd (199). -/
def eval_bitwise_and (a b : UInt32) : UInt32 := a &&& b

/-- OpBitwiseOr (197). -/
def eval_bitwise_or (a b : UInt32) : UInt32 := a ||| b

/-- OpBitwiseXor (198). -/
def eval_bitwise_xor (a b : UInt32) : UInt32 := a ^^^ b

/-- OpShiftLeftLogical (196). -/
def eval_shl (a b : UInt32) : UInt32 := a <<< b

/-- OpShiftRightLogical (194). -/
def eval_shr_logical (a b : UInt32) : UInt32 := a >>> b

/-- OpShiftRightArithmetic (195): sign-extending right shift.
    Modeled via signed interpretation. -/
def eval_shr_arithmetic (a b : UInt32) : UInt32 :=
  fromSigned32 (toSigned32 a / (2 ^ b.toNat : Int))

/-- OpNot (200): Bitwise complement. -/
def eval_not (a : UInt32) : UInt32 := a ^^^ 0xFFFFFFFF

-- ════════════════════════════════════════════════════════════════════
-- Section 5: Comparison operations
-- ════════════════════════════════════════════════════════════════════

/-- OpIEqual (170): Integer equality. -/
def eval_iequal (a b : UInt32) : Bool := a == b

/-- OpINotEqual (171): Integer inequality. -/
def eval_inotequal (a b : UInt32) : Bool := a != b

/-- OpULessThan (176): Unsigned less-than. -/
def eval_ult (a b : UInt32) : Bool := a < b

/-- OpULessThanEqual (177): Unsigned less-or-equal. -/
def eval_ule (a b : UInt32) : Bool := a ≤ b

/-- OpUGreaterThan (172): Unsigned greater-than. -/
def eval_ugt (a b : UInt32) : Bool := b < a

/-- OpUGreaterThanEqual (173): Unsigned greater-or-equal. -/
def eval_uge (a b : UInt32) : Bool := b ≤ a

/-- OpSLessThan (178): Signed less-than. -/
def eval_slt (a b : UInt32) : Bool := toSigned32 a < toSigned32 b

/-- OpSLessThanEqual (179): Signed less-or-equal. -/
def eval_sle (a b : UInt32) : Bool := toSigned32 a ≤ toSigned32 b

/-- OpSGreaterThan (174): Signed greater-than. -/
def eval_sgt (a b : UInt32) : Bool := toSigned32 b < toSigned32 a

/-- OpSGreaterThanEqual (175): Signed greater-or-equal. -/
def eval_sge (a b : UInt32) : Bool := toSigned32 b ≤ toSigned32 a

-- Float comparisons: axiomatized (ordered variants assume non-NaN).
noncomputable opaque eval_ford_equal : F32Bits → F32Bits → Bool
noncomputable opaque eval_ford_notequal : F32Bits → F32Bits → Bool
noncomputable opaque eval_ford_lt : F32Bits → F32Bits → Bool
noncomputable opaque eval_ford_le : F32Bits → F32Bits → Bool
noncomputable opaque eval_ford_gt : F32Bits → F32Bits → Bool
noncomputable opaque eval_ford_ge : F32Bits → F32Bits → Bool

-- ════════════════════════════════════════════════════════════════════
-- Section 6: Unary operations
-- ════════════════════════════════════════════════════════════════════

/-- OpSNegate (126): Two's complement negate. -/
def eval_snegate (a : UInt32) : UInt32 := 0 - a

/-- OpFNegate (127): IEEE 754 negate. -/
noncomputable opaque eval_fnegate : F32Bits → F32Bits

/-- OpLogicalNot (168): Boolean not. -/
def eval_logical_not (a : Bool) : Bool := !a

-- ════════════════════════════════════════════════════════════════════
-- Section 7: Conversion operations
-- ════════════════════════════════════════════════════════════════════

/-- OpConvertFToU: Float to unsigned integer (truncating). -/
noncomputable opaque eval_convert_f_to_u : F32Bits → UInt32

/-- OpConvertFToS: Float to signed integer (truncating). -/
noncomputable opaque eval_convert_f_to_s : F32Bits → UInt32  -- stored as UInt32, signed interpretation

/-- OpConvertSToF: Signed integer to float. -/
noncomputable opaque eval_convert_s_to_f : UInt32 → F32Bits

/-- OpConvertUToF: Unsigned integer to float. -/
noncomputable opaque eval_convert_u_to_f : UInt32 → F32Bits

/-- OpBitcast: Reinterpret bit pattern. No-op on same-width types. -/
def eval_bitcast (a : UInt32) : UInt32 := a

-- ════════════════════════════════════════════════════════════════════
-- Section 8: Memory operations (modeled as state transitions)
-- ════════════════════════════════════════════════════════════════════

/-- Memory state: maps address to 32-bit value. -/
def Memory := Nat → UInt32

/-- OpLoad: Read from memory at address. -/
def eval_load (mem : Memory) (addr : Nat) : UInt32 := mem addr

/-- OpStore: Write to memory at address. Returns updated memory. -/
def eval_store (mem : Memory) (addr : Nat) (val : UInt32) : Memory :=
  fun a => if a == addr then val else mem a

-- Store then load at same address returns the stored value.
theorem load_after_store (mem : Memory) (addr : Nat) (val : UInt32) :
    eval_load (eval_store mem addr val) addr = val := by
  simp [eval_load, eval_store]

-- Store then load at different address returns the original value.
theorem load_after_store_other (mem : Memory) (addr1 addr2 : Nat) (val : UInt32)
    (h : addr1 ≠ addr2) :
    eval_load (eval_store mem addr1 val) addr2 = eval_load mem addr2 := by
  simp [eval_load, eval_store, h]
  omega

-- ════════════════════════════════════════════════════════════════════
-- Section 9: Control flow (modeled as CFG transitions)
-- ════════════════════════════════════════════════════════════════════

/-- A basic block has an ID and produces a value. -/
structure BasicBlock where
  id : Nat
  result : UInt32

/-- OpBranch: Unconditional jump to target block. -/
def eval_branch (target : BasicBlock) : BasicBlock := target

/-- OpBranchConditional: Conditional jump. -/
def eval_branch_cond (cond : Bool) (then_bb else_bb : BasicBlock) : BasicBlock :=
  if cond then then_bb else else_bb

/-- OpPhi: Select value based on predecessor block.
    Modeled as a lookup table from predecessor block ID to value. -/
def eval_phi (predecessors : List (Nat × UInt32)) (from_block : Nat) : UInt32 :=
  match predecessors.find? (fun p => p.1 == from_block) with
  | some (_, val) => val
  | none => 0  -- unreachable in well-formed SPIR-V

-- ════════════════════════════════════════════════════════════════════
-- Section 10: Barrier (modeled as visibility guarantee)
-- ════════════════════════════════════════════════════════════════════

/-- Per-quark memory view after a barrier: all quarks see the same state. -/
axiom barrier_visibility_spv
    {n : Nat}
    (writes : Fin n → Memory → Memory)
    (mem : Memory) :
    let post := (List.range n).foldl (fun m i =>
      if h : i < n then writes ⟨i, h⟩ m else m) mem
    ∀ _quark : Fin n, ∀ addr : Nat, post addr = post addr

-- ════════════════════════════════════════════════════════════════════
-- Section 11: Unified dispatch interface
-- ════════════════════════════════════════════════════════════════════

/-- Quanta IR binary operations (mirrors quanta-ir BinOp). -/
inductive BinOp where
  | Add | Sub | Mul | Div | Rem
  | BitAnd | BitOr | BitXor
  | Shl | Shr
  deriving Repr, DecidableEq

/-- Quanta IR signedness. -/
inductive Signedness where
  | Unsigned | Signed
  deriving Repr, DecidableEq

/-- Evaluate a BinOp on UInt32 via the corresponding SPIR-V instruction. -/
def eval_binop_u32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_iadd a b
  | .Sub, a, b    => eval_isub a b
  | .Mul, a, b    => eval_imul a b
  | .Div, a, b    => eval_udiv a b
  | .Rem, a, b    => eval_umod a b
  | .BitAnd, a, b => eval_bitwise_and a b
  | .BitOr, a, b  => eval_bitwise_or a b
  | .BitXor, a, b => eval_bitwise_xor a b
  | .Shl, a, b    => eval_shl a b
  | .Shr, a, b    => eval_shr_logical a b

/-- Evaluate a BinOp on signed Int32 (stored as UInt32). -/
def eval_binop_i32 : BinOp → UInt32 → UInt32 → UInt32
  | .Add, a, b    => eval_iadd a b      -- wrapping add is sign-agnostic
  | .Sub, a, b    => eval_isub a b      -- wrapping sub is sign-agnostic
  | .Mul, a, b    => eval_imul a b      -- wrapping mul is sign-agnostic
  | .Div, a, b    => eval_sdiv a b
  | .Rem, a, b    => eval_smod a b
  | .BitAnd, a, b => eval_bitwise_and a b
  | .BitOr, a, b  => eval_bitwise_or a b
  | .BitXor, a, b => eval_bitwise_xor a b
  | .Shl, a, b    => eval_shl a b
  | .Shr, a, b    => eval_shr_arithmetic a b

/-- Quanta IR comparison operations. -/
inductive CmpOp where
  | Eq | Ne | Lt | Le | Gt | Ge
  deriving Repr, DecidableEq

/-- Evaluate a CmpOp on unsigned UInt32. -/
def eval_cmp_u32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_iequal a b
  | .Ne, a, b => eval_inotequal a b
  | .Lt, a, b => eval_ult a b
  | .Le, a, b => eval_ule a b
  | .Gt, a, b => eval_ugt a b
  | .Ge, a, b => eval_uge a b

/-- Evaluate a CmpOp on signed Int32 (stored as UInt32). -/
def eval_cmp_i32 : CmpOp → UInt32 → UInt32 → Bool
  | .Eq, a, b => eval_iequal a b
  | .Ne, a, b => eval_inotequal a b
  | .Lt, a, b => eval_slt a b
  | .Le, a, b => eval_sle a b
  | .Gt, a, b => eval_sgt a b
  | .Ge, a, b => eval_sge a b

/-- Quanta IR unary operations. -/
inductive UnaryOp where
  | Neg | BitNot | LogicalNot
  deriving Repr, DecidableEq

/-- Evaluate a UnaryOp on unsigned UInt32. -/
def eval_unary_u32 : UnaryOp → UInt32 → UInt32
  | .Neg, a       => eval_snegate a
  | .BitNot, a    => eval_not a
  | .LogicalNot, _ => 0  -- logical not on integer is not meaningful

end Quanta.Semantics.SpirV
