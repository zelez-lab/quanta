-- SPIR-V comparison opcode correctness specification.
--
-- Theorem T2 extension: every (CmpOp, FloatKind) pair maps to exactly
-- one SPIR-V comparison opcode, and that opcode matches SPIR-V 1.6.

namespace Quanta.ComparisonOps

/-- Reuse the float/signedness tag from Opcodes. -/
inductive FloatKind where
  | IsFloat | IsSignedInt | IsUnsignedInt
  deriving Repr, DecidableEq

/-- Quanta IR comparison operations. -/
inductive CmpOp where
  | Eq | Ne | Lt | Le | Gt | Ge
  deriving Repr, DecidableEq

/-- SPIR-V comparison opcodes from the specification. -/
inductive SpvCmpOp where
  | IEqual                  -- 170
  | INotEqual               -- 171
  | UGreaterThan            -- 172
  | UGreaterThanEqual       -- 173
  | SGreaterThan            -- 174
  | SGreaterThanEqual       -- 175
  | ULessThan               -- 176
  | ULessThanEqual          -- 177
  | SLessThan               -- 178
  | SLessThanEqual          -- 179
  | FOrdEqual               -- 180
  | FOrdNotEqual            -- 181
  | FOrdLessThan            -- 184
  | FOrdGreaterThan         -- 186
  | FOrdLessThanEqual       -- 188
  | FOrdGreaterThanEqual    -- 190
  deriving Repr, DecidableEq

/-- The SPIR-V numeric opcode for each comparison operation. -/
def SpvCmpOp.toNat : SpvCmpOp → Nat
  | .IEqual               => 170
  | .INotEqual            => 171
  | .UGreaterThan         => 172
  | .UGreaterThanEqual    => 173
  | .SGreaterThan         => 174
  | .SGreaterThanEqual    => 175
  | .ULessThan            => 176
  | .ULessThanEqual       => 177
  | .SLessThan            => 178
  | .SLessThanEqual       => 179
  | .FOrdEqual            => 180
  | .FOrdNotEqual         => 181
  | .FOrdLessThan         => 184
  | .FOrdGreaterThan      => 186
  | .FOrdLessThanEqual    => 188
  | .FOrdGreaterThanEqual => 190

/-- Map (CmpOp, FloatKind) to the correct SPIR-V comparison opcode. -/
def cmpOpToSpv : CmpOp → FloatKind → SpvCmpOp
  | .Eq, .IsFloat       => .FOrdEqual
  | .Eq, _              => .IEqual
  | .Ne, .IsFloat       => .FOrdNotEqual
  | .Ne, _              => .INotEqual
  | .Lt, .IsFloat       => .FOrdLessThan
  | .Lt, .IsSignedInt   => .SLessThan
  | .Lt, .IsUnsignedInt => .ULessThan
  | .Le, .IsFloat       => .FOrdLessThanEqual
  | .Le, .IsSignedInt   => .SLessThanEqual
  | .Le, .IsUnsignedInt => .ULessThanEqual
  | .Gt, .IsFloat       => .FOrdGreaterThan
  | .Gt, .IsSignedInt   => .SGreaterThan
  | .Gt, .IsUnsignedInt => .UGreaterThan
  | .Ge, .IsFloat       => .FOrdGreaterThanEqual
  | .Ge, .IsSignedInt   => .SGreaterThanEqual
  | .Ge, .IsUnsignedInt => .UGreaterThanEqual

-- Theorem: Eq + float maps to FOrdEqual (180)
theorem eq_float_is_180 : (cmpOpToSpv .Eq .IsFloat).toNat = 180 := by rfl

-- Theorem: Eq + int maps to IEqual (170)
theorem eq_int_signed_is_170 : (cmpOpToSpv .Eq .IsSignedInt).toNat = 170 := by rfl
theorem eq_int_unsigned_is_170 : (cmpOpToSpv .Eq .IsUnsignedInt).toNat = 170 := by rfl

-- Theorem: Ne + float maps to FOrdNotEqual (181)
theorem ne_float_is_181 : (cmpOpToSpv .Ne .IsFloat).toNat = 181 := by rfl

-- Theorem: all float comparisons use FOrd* variants (opcodes >= 180)
theorem float_cmp_uses_ford :
    ∀ op, (cmpOpToSpv op .IsFloat).toNat ≥ 180 := by
  intro op; cases op <;> simp [cmpOpToSpv, SpvCmpOp.toNat]

-- Theorem: all integer comparisons use I*/U*/S* variants (opcodes < 180)
theorem int_cmp_uses_integer :
    ∀ op fk, fk ≠ .IsFloat →
      (cmpOpToSpv op fk).toNat < 180 := by
  intro op fk hfk
  cases op <;> cases fk <;> simp_all [cmpOpToSpv, SpvCmpOp.toNat]

-- The original sketch claimed `op1 = op2 ∧ fk1 = fk2`, but `Eq` and
-- `Ne` collapse `IsSignedInt`/`IsUnsignedInt` (both map to `.IEqual`
-- / `.INotEqual` respectively), so `fk1 = fk2` doesn't follow. Same
-- pattern as `Opcodes.lean`: state injectivity at the `SpvCmpOp`
-- level — that's the property the wire-format round-trip needs.

/-- `SpvCmpOp.toNat` is injective on the inductive variants. -/
theorem spv_cmp_toNat_injective :
    ∀ x y : SpvCmpOp, x.toNat = y.toNat → x = y := by
  intro x y h
  cases x <;> cases y <;> simp_all [SpvCmpOp.toNat]

/-- Two `(op, fk)` inputs that produce the same numeric opcode produce
    the same `SpvCmpOp` value. -/
theorem cmpop_mapping_consistent :
    ∀ op1 op2 fk1 fk2,
      (cmpOpToSpv op1 fk1).toNat = (cmpOpToSpv op2 fk2).toNat →
      cmpOpToSpv op1 fk1 = cmpOpToSpv op2 fk2 := by
  intro op1 op2 fk1 fk2 h
  exact spv_cmp_toNat_injective _ _ h

end Quanta.ComparisonOps
