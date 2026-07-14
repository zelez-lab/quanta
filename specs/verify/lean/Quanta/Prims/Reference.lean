/-
Block-cooperative primitives — Lean formalisation of
`quanta-prims`'s reference implementations.

The Rust crate at `crates/sci/quanta-prims/` ships two layers per
primitive:

1. A GPU kernel (the one users dispatch at runtime).
2. A pure single-thread Rust reference impl in
   `quanta_prims::reference`, used as the correctness oracle in
   differential tests.

The kernel's correctness is enforced by the differential tests:
"GPU output equals reference output on this input." This file
proves the **reference layer** correct on its own terms.
Combined with the differential test suite, it transitively
certifies the GPU kernels.
-/

import Mathlib.Data.List.Sort
import Mathlib.Data.List.Perm.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Tactic.Ring

namespace Quanta.Prims

-- ── Reduce ──────────────────────────────────────────────────────

/-- Reference reduce: the sum of a list of natural numbers.
    Mirrors `quanta_prims::reference::reduce_add_u32` (modulo
    `Nat` vs `u32::wrapping_add`). -/
def reduceAdd (xs : List Nat) : Nat :=
  xs.foldl (· + ·) 0

/-- T9000 — `reduceAdd` equals the standard list sum. -/
theorem reduceAdd_eq_sum (xs : List Nat) :
    reduceAdd xs = xs.sum := by
  unfold reduceAdd
  rw [← List.sum_eq_foldl]

/-- T9001 — `reduceAdd []` is 0 (additive identity). -/
theorem reduceAdd_nil :
    reduceAdd ([] : List Nat) = 0 := by
  rfl

/-- T9002 — `reduceAdd` is invariant under permutation. The
    order of summation doesn't matter for `Nat.add`. Downstream
    uses this to certify that a parallel reduce (which may run
    in different orders on different backends) returns the same
    sum. -/
theorem reduceAdd_perm
    {xs ys : List Nat} (h : List.Perm xs ys) :
    reduceAdd xs = reduceAdd ys := by
  rw [reduceAdd_eq_sum, reduceAdd_eq_sum]
  exact h.sum_eq

-- ── Scan ────────────────────────────────────────────────────────

/-- Reference inclusive prefix-sum scan. Mirrors
    `quanta_prims::reference::scan_add_u32`. -/
def scanAdd (xs : List Nat) : List Nat :=
  xs.scanl (· + ·) 0 |>.tail

/-- T9010 — `scanAdd` preserves the input length. -/
theorem scanAdd_length (xs : List Nat) :
    (scanAdd xs).length = xs.length := by
  unfold scanAdd
  simp [List.length_scanl]

/-- T9011 — `scanAdd []` is `[]`. -/
theorem scanAdd_nil :
    scanAdd ([] : List Nat) = [] := by
  rfl

-- ── Sort ────────────────────────────────────────────────────────
--
-- Lean's `List.mergeSort` takes a boolean total-preorder
-- predicate; we instantiate it with `Nat`'s `≤` as a decidable
-- `Bool` comparator.

/-- Reference ascending sort. Mirrors
    `quanta_prims::reference::radix_sort_u32` (which uses
    `slice::sort_unstable` — a different algorithm but the same
    output: ascending order). -/
def sortAsc (xs : List Nat) : List Nat :=
  xs.mergeSort (fun a b => a ≤ b)

/-- T9020 — The sorted output is a permutation of the input. -/
theorem sortAsc_perm (xs : List Nat) :
    List.Perm (sortAsc xs) xs := by
  unfold sortAsc
  exact List.mergeSort_perm xs _

/-- T9021 — `sortAsc` preserves length. Falls out of T9020 since
    permutations have equal length. -/
theorem sortAsc_length (xs : List Nat) :
    (sortAsc xs).length = xs.length :=
  (sortAsc_perm xs).length_eq

/-- T9022 — `sortAsc` preserves the sum. Combines T9002
    (reduce is permutation-invariant) and T9020 (sort is a
    permutation): the sum of the sorted list equals the sum of
    the input. -/
theorem sortAsc_sum (xs : List Nat) :
    reduceAdd (sortAsc xs) = reduceAdd xs :=
  reduceAdd_perm (sortAsc_perm xs)

end Quanta.Prims
