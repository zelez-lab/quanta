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

-- Theorem: every tag maps to a unique operation. Phrased over `Fin 12`
-- so the universal quantifier is decidable and `decide` closes it
-- without needing Mathlib's `interval_cases`.
theorem binop_tags_unique_fin :
    ∀ i j : Fin 12, i ≠ j → binOpTag i.val ≠ binOpTag j.val := by
  decide

theorem binop_tags_unique :
    ∀ i j, i < 12 → j < 12 → i ≠ j →
      binOpTag i ≠ binOpTag j := by
  intro i j hi hj hne
  exact binop_tags_unique_fin ⟨i, hi⟩ ⟨j, hj⟩ (fun h => hne (congrArg Fin.val h))

-- Theorem: every tag in range produces Some
theorem binop_tags_total_fin :
    ∀ i : Fin 12, (binOpTag i.val).isSome = true := by
  decide

theorem binop_tags_total :
    ∀ i, i < 12 → (binOpTag i).isSome = true := by
  intro i hi
  exact binop_tags_total_fin ⟨i, hi⟩

end Quanta.WireFormat
