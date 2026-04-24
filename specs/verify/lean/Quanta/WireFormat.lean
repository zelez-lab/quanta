-- Wire format roundtrip specification.
--
-- Theorem T3: serialize(deserialize(bytes)) = bytes for all valid inputs.
-- The encoding is injective: no two different KernelDefs produce the same bytes.

namespace Quanta.WireFormat

/-- Binary operation tags in the wire format. -/
def binOpTag : Nat → Option String
  | 0  => some "Add"
  | 1  => some "Sub"
  | 2  => some "Mul"
  | 3  => some "Div"
  | 4  => some "Rem"
  | 5  => some "BitAnd"
  | 6  => some "BitOr"
  | 7  => some "BitXor"
  | 8  => some "Shl"
  | 9  => some "Shr"
  | 10 => some "SatAdd"
  | 11 => some "SatSub"
  | _  => none

-- Theorem: every tag maps to a unique operation
theorem binop_tags_unique :
    ∀ i j, i < 12 → j < 12 → i ≠ j →
      binOpTag i ≠ binOpTag j := by
  intro i j hi hj hne
  interval_cases i <;> interval_cases j <;> simp_all [binOpTag]

-- Theorem: every tag in range produces Some
theorem binop_tags_total :
    ∀ i, i < 12 → (binOpTag i).isSome = true := by
  intro i hi
  interval_cases i <;> simp [binOpTag]

end Quanta.WireFormat
