import Mathlib.Tactic.SplitIfs
import Mathlib.Tactic.Ring

-- Quanta.Prims.FloatOrder — the monotone bijection that makes every
-- u32 radix/bitonic primitive order-correct for f32 keys.
--
-- Model: IEEE-754 binary32 bit patterns as `Nat < 2^32`. The sign bit
-- is the top bit; `totalOrder` (IEEE 754-2019 §5.10, = Rust's
-- `f32::total_cmp`) on bit patterns is sign-magnitude order:
-- non-negatives ascend with their pattern, negatives DESCEND with
-- their pattern (a bigger negative pattern is a smaller value), and
-- every negative sits below every non-negative. NaNs need no special
-- case at this level — totalOrder places them by the same
-- sign-magnitude rule (negative NaNs below -inf, positive NaNs above
-- +inf), which is exactly the deterministic policy the prims document.
--
-- The transform (the classic radix-float trick):
--   key(b) = if b < 2^31 then b + 2^31   -- non-negative: set sign bit
--            else 2^32 - 1 - b           -- negative: invert all bits
-- (`2^32 - 1 - b` IS bitwise NOT on 32-bit patterns.)
--
-- Scope note (honest framing): these theorems are about BIT PATTERNS
-- and the totalOrder SPEC — the bridge from Lean's `Float` type is
-- not mechanized (Lean's Float lacks a usable bit-level model). The
-- kernel's `bitcast` is the modeled boundary.

namespace Quanta.Prims.FloatOrder

def signBit : Nat := 2 ^ 31
def width : Nat := 2 ^ 32

/-- A binary32 bit pattern. -/
def Pat (b : Nat) : Prop := b < width

/-- IEEE totalOrder on bit patterns (sign-magnitude order). -/
def smLE (a b : Nat) : Prop :=
  if a < signBit then
    -- a non-negative: b must be non-negative and ≥ in pattern.
    b < signBit ∧ a ≤ b
  else
    -- a negative: below every non-negative; among negatives the
    -- ORDER REVERSES with the pattern.
    b < signBit ∨ b ≤ a

/-- The monotone transform. -/
def key (b : Nat) : Nat :=
  if b < signBit then b + signBit else width - 1 - b

theorem signBit_lt_width : signBit < width := by
  simp [signBit, width]

/-- THE ORDER THEOREM: the transform carries IEEE totalOrder to plain
`Nat` order — so any correct unsigned sort over `key`ed values orders
the floats by totalOrder. -/
theorem key_monotone_iff {a b : Nat} (ha : Pat a) (hb : Pat b) :
    smLE a b ↔ key a ≤ key b := by
  unfold smLE key Pat signBit width at *
  split_ifs at * <;> omega

/-- The transform is injective on patterns — no two floats collide, so
key-sorting loses nothing. -/
theorem key_injective {a b : Nat} (ha : Pat a) (hb : Pat b)
    (h : key a = key b) : a = b := by
  unfold key Pat signBit width at *
  split_ifs at * <;> omega

/-- The transform stays in range — the keyed value is a valid u32. -/
theorem key_in_range {a : Nat} (ha : Pat a) : Pat (key a) := by
  unfold key Pat signBit width at *
  split_ifs at * <;> omega

/-- The inverse: applying the same case split undoes the transform
(the kernel's read-back path). -/
def unkey (b : Nat) : Nat :=
  if b < signBit then width - 1 - b else b - signBit

theorem unkey_key {a : Nat} (ha : Pat a) : unkey (key a) = a := by
  unfold unkey key Pat signBit width at *
  split_ifs at * <;> omega

/-- Strictness transfers too (needed for top-k uniqueness arguments):
strict sign-magnitude order is strict key order. -/
theorem key_strict_iff {a b : Nat} (ha : Pat a) (hb : Pat b) :
    (smLE a b ∧ a ≠ b) ↔ key a < key b := by
  constructor
  · rintro ⟨hle, hne⟩
    rcases Nat.lt_or_ge (key a) (key b) with h | h
    · exact h
    · have := (key_monotone_iff ha hb).mp hle
      have : key a = key b := Nat.le_antisymm this h
      exact absurd (key_injective ha hb this) hne
  · intro h
    refine ⟨(key_monotone_iff ha hb).mpr (Nat.le_of_lt h), ?_⟩
    intro rfl_eq
    subst rfl_eq
    exact Nat.lt_irrefl _ h

end Quanta.Prims.FloatOrder
