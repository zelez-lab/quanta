/-
Online (streaming / FlashAttention-style) softmax — Lean proof foundation
for `quanta-nn`'s fused attention kernel.

The fused attention kernel wants to process the key/value sequence in
*blocks*, carrying a running maximum, a running normaliser, and a running
weighted accumulator, and NEVER materialising the full N² score matrix.
This file proves that this online recurrence computes exactly the same
result as the textbook two-pass softmax-weighted sum, and that it is
numerically stable (every `exp` argument the kernel evaluates is ≤ 0).

## The two forms

Scores `x : List ℝ` (nonempty) paired with values `v : List ℝ` of the same
length — modelled here as a single `List (ℝ × ℝ)` of `(score, value)` pairs
(scalar values suffice; the kernel replays this per output dimension).

* **Direct (two-pass).** `m* = max x`; `l* = Σⱼ exp(xⱼ − m*)`;
  `out* = (Σⱼ exp(xⱼ − m*)·vⱼ) / l*`. Equivalently `out* = Σⱼ softmax(x)ⱼ·vⱼ`.

* **Online (one-pass fold).** Carry `(m, l, acc)`. On a new pair `(xᵢ, vᵢ)`
  with running max `m`:
  ```
  m'   = max m xᵢ
  l'   = l   · exp(m − m') + exp(xᵢ − m')
  acc' = acc · exp(m − m') + exp(xᵢ − m')·vᵢ
  ```
  initialised from the first element: `(x₁, 1, v₁)` (since `exp(x₁−x₁) = 1`).

## What we prove (T9200–T9209)

The load-bearing result is a *generalised fold invariant* stated relative to
an arbitrary already-summarised prefix (`online_step` / `fold_absorb`): folding
`step` over `ps` starting from the exact summary of `pre` yields the exact
summary of `pre ++ ps`. This single lemma is simultaneously
  * the prefix invariant (T9203, with `pre = [p]`),
  * the block-merge / append law (T9206, splitting a list at an arbitrary
    point — which licenses *any* block schedule), and
  * the equivalence with the two-pass form (T9204).
Stability (T9207/T9208) and the softmax basics — sums-to-one (T9200) and
shift-invariance (T9201) — round out the obligations.

Everything is exact over `ℝ` (finite lists, no rounding model): ZERO axioms,
ZERO sorries. Mathlib's `Real.exp` API (`exp_add`, `exp_sub`, `exp_zero`,
`exp_pos`, `exp_le_one`) and `Finset`/`List` big-operator lemmas do the work.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace Quanta.Nn

open Real

/-! ## Softmax basics (T9200 / T9201) -/

/-- Unnormalised softmax weight of score `xᵢ` against a reference `m`:
    `exp(xᵢ − m)`.  With `m = max x` this is the numerically-stable form the
    kernel actually evaluates. -/
noncomputable def wt (m x : ℝ) : ℝ := Real.exp (x - m)

/-- Normaliser of a score list against reference `m`: `Σⱼ exp(xⱼ − m)`. -/
noncomputable def sumWt (m : ℝ) (xs : List ℝ) : ℝ :=
  (xs.map (fun x => wt m x)).sum

/-- softmax coordinate `i`: `exp(xᵢ − m) / Σⱼ exp(xⱼ − m)`.  Reference `m` is a
    free shift; the value is independent of `m` (T9201). -/
noncomputable def softmaxCoord (m xi : ℝ) (xs : List ℝ) : ℝ :=
  wt m xi / sumWt m xs

/-- `sumWt` is strictly positive on a nonempty list (every summand is a
    positive `exp`), so it is a legal denominator. -/
theorem sumWt_pos {m : ℝ} {xs : List ℝ} (h : xs ≠ []) : 0 < sumWt m xs := by
  unfold sumWt wt
  induction xs with
  | nil => exact absurd rfl h
  | cons a t ih =>
    simp only [List.map_cons, List.sum_cons]
    cases t with
    | nil => simpa using Real.exp_pos (a - m)
    | cons b t' =>
      have hpos : (0 : ℝ) < Real.exp (a - m) := Real.exp_pos _
      have htail : 0 < (((b :: t').map (fun x => Real.exp (x - m))).sum) :=
        ih (by simp)
      linarith

/-- Sum of a list of quotients with a fixed denominator: `Σ (fⱼ / c) = (Σ fⱼ)/c`.
    A pure `List.sum` fact used to pull the softmax normaliser out of a sum. -/
theorem sum_map_div_const {α : Type} (c : ℝ) (f : α → ℝ) (xs : List α) :
    (xs.map (fun x => f x / c)).sum = (xs.map (fun x => f x)).sum / c := by
  induction xs with
  | nil => simp
  | cons a t ih => simp only [List.map_cons, List.sum_cons, ih]; ring

/-- Reference change factors through the sum: `Σ exp(xⱼ − m) = exp(m'−m)·Σ exp(xⱼ − m')`.
    Isolates the `Real.exp_add` step used by both `t9200` and `t9201`. -/
theorem sumWt_shift (m m' : ℝ) (xs : List ℝ) :
    sumWt m xs = Real.exp (m' - m) * sumWt m' xs := by
  unfold sumWt wt
  induction xs with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.sum_cons, ih]
    have h1 : Real.exp (a - m) = Real.exp (m' - m) * Real.exp (a - m') := by
      rw [← Real.exp_add]; ring_nf
    rw [h1]; ring

/-- **T9201 — softmax is shift-invariant.** Shifting every score by a constant
    `c` (equivalently: choosing any reference `m`) leaves each softmax
    coordinate unchanged.  This is why the online form may subtract the running
    max: the running max is just a shift, and softmax ignores shifts.  Formally
    the softmax coordinate does not depend on the reference `m`. -/
theorem t9201_softmax_shift_invariant (m m' xi : ℝ) (xs : List ℝ) :
    softmaxCoord m xi xs = softmaxCoord m' xi xs := by
  unfold softmaxCoord
  -- numerator: wt m xi = exp(m'-m) · wt m' xi ; denominator: same factor.
  have hnum : wt m xi = Real.exp (m' - m) * wt m' xi := by
    unfold wt; rw [← Real.exp_add]; ring_nf
  rw [hnum, sumWt_shift m m' xs]
  have hc : Real.exp (m' - m) ≠ 0 := ne_of_gt (Real.exp_pos _)
  rw [mul_div_mul_left _ _ hc]

/-- **T9200 — softmax sums to one.** `Σᵢ softmax(x)ᵢ = 1` for a nonempty score
    list.  The normaliser is exactly the sum of the numerators. -/
theorem t9200_softmax_sum_one {m : ℝ} {xs : List ℝ} (h : xs ≠ []) :
    (xs.map (fun xi => softmaxCoord m xi xs)).sum = 1 := by
  have hpos : sumWt m xs ≠ 0 := ne_of_gt (sumWt_pos (m := m) h)
  -- Σ (wt xi / S) = (Σ wt xi) / S = S / S = 1.
  have hrw : (xs.map (fun xi => softmaxCoord m xi xs)).sum
      = (xs.map (fun xi => wt m xi)).sum / sumWt m xs :=
    sum_map_div_const (sumWt m xs) (fun xi => wt m xi) xs
  rw [hrw]
  have : (xs.map (fun xi => wt m xi)).sum = sumWt m xs := rfl
  rw [this, div_self hpos]

/-! ## Online fold state and the streaming recurrence -/

/-- The running online-softmax state: `(m, l, acc)` — running max, running
    normaliser `Σ exp(xⱼ − m)`, running weighted accumulator
    `Σ exp(xⱼ − m)·vⱼ`. -/
structure State where
  m : ℝ
  l : ℝ
  acc : ℝ

/-- The streaming step: absorb one `(score, value)` pair `p` into the running
    state.  This is the exact recurrence the fused kernel runs per element. -/
noncomputable def step (s : State) (p : ℝ × ℝ) : State :=
  let m' := max s.m p.1
  { m   := m'
    l   := s.l * Real.exp (s.m - m') + Real.exp (p.1 - m')
    acc := s.acc * Real.exp (s.m - m') + Real.exp (p.1 - m') * p.2 }

/-- The block-merge combine: fold two independently-summarised states
    (`s₁` over block A, `s₂` over block B) into the state of `A ++ B`.  The
    running max is `max m₁ m₂`; each block's sums are rescaled by
    `exp(mᵢ − m)`.  This is `step` generalised from a singleton second block to
    an arbitrary one — it is what licenses arbitrary block partitioning. -/
noncomputable def merge (s₁ s₂ : State) : State :=
  let m := max s₁.m s₂.m
  { m   := m
    l   := s₁.l * Real.exp (s₁.m - m) + s₂.l * Real.exp (s₂.m - m)
    acc := s₁.acc * Real.exp (s₁.m - m) + s₂.acc * Real.exp (s₂.m - m) }

/-! ## The spec: what a state *should* be for a given list -/

/-- Running max of a nonempty pair-list's scores.  Defined by `foldl max` over
    the tail seeded at the head score; total, and equal to the online `m`. -/
def specMax : List (ℝ × ℝ) → ℝ
  | [] => 0
  | p :: ps => ps.foldl (fun acc q => max acc q.1) p.1

/-- Reference normaliser for reference `m`: `Σⱼ exp(xⱼ − m)`. -/
noncomputable def specL (m : ℝ) (ps : List (ℝ × ℝ)) : ℝ :=
  (ps.map (fun q => Real.exp (q.1 - m))).sum

/-- Reference weighted accumulator for reference `m`: `Σⱼ exp(xⱼ − m)·vⱼ`. -/
noncomputable def specAcc (m : ℝ) (ps : List (ℝ × ℝ)) : ℝ :=
  (ps.map (fun q => Real.exp (q.1 - m) * q.2)).sum

/-- A state `s` *summarises* the pair-list `ps` (against its own running max
    `s.m`) iff `s.l`, `s.acc` are the reference normaliser/accumulator taken at
    reference `s.m`. The running max is tracked explicitly as `s.m = specMax`. -/
def Summarises (s : State) (ps : List (ℝ × ℝ)) : Prop :=
  s.m = specMax ps ∧ s.l = specL s.m ps ∧ s.acc = specAcc s.m ps

/-- Initial state built from the head element: `(x₁, 1, v₁)`.  Since
    `exp(x₁ − x₁) = 1`, this is `Summarises … [p]`. -/
noncomputable def init (p : ℝ × ℝ) : State :=
  { m := p.1, l := 1, acc := p.2 }

/-- The online fold over a nonempty pair-list: seed from the head, `step` the
    tail.  This is the whole streaming algorithm. -/
noncomputable def online : List (ℝ × ℝ) → State
  | [] => { m := 0, l := 0, acc := 0 }
  | p :: ps => ps.foldl step (init p)

/-! ## Rescaling: specL / specAcc under a change of reference -/

/-- **T9202 — normaliser/accumulator rescale.** Lowering the reference from `m`
    to `m'` multiplies both `specL` and `specAcc` by `exp(m − m')`.  This is the
    algebraic heart of the merge: the `exp(mᵢ − m)` factors are exactly this
    rescale.  (`specAcc` case bundled in the same lemma.) -/
theorem t9202_spec_rescale (m m' : ℝ) (ps : List (ℝ × ℝ)) :
    specL m' ps = specL m ps * Real.exp (m - m') ∧
    specAcc m' ps = specAcc m ps * Real.exp (m - m') := by
  constructor
  · unfold specL
    induction ps with
    | nil => simp
    | cons a t ih =>
      simp only [List.map_cons, List.sum_cons, ih]
      have h1 : Real.exp (a.1 - m') = Real.exp (a.1 - m) * Real.exp (m - m') := by
        rw [← Real.exp_add]; ring_nf
      rw [h1]; ring
  · unfold specAcc
    induction ps with
    | nil => simp
    | cons a t ih =>
      simp only [List.map_cons, List.sum_cons, ih]
      have h1 : Real.exp (a.1 - m') = Real.exp (a.1 - m) * Real.exp (m - m') := by
        rw [← Real.exp_add]; ring_nf
      rw [h1]; ring

/-- `specMax` of a cons, unfolded one step: `specMax (p :: ps)` threads the head
    score through the `foldl max`. Convenience for the fold proof. -/
theorem specMax_cons (p : ℝ × ℝ) (ps : List (ℝ × ℝ)) :
    specMax (p :: ps) = ps.foldl (fun acc q => max acc q.1) p.1 := rfl

/-- `foldl max` pulls its seed out: `foldl max s ps = max s (foldl max s' ps)`
    modulo the seed — precisely `foldl (max ·) (max a s) ps = max a (foldl (max ·) s ps)`.
    The seed `a` factors through every `max` monotonically. -/
theorem foldl_max_seed (a : ℝ) (s : ℝ) (ps : List (ℝ × ℝ)) :
    ps.foldl (fun acc q => max acc q.1) (max a s)
      = max a (ps.foldl (fun acc q => max acc q.1) s) := by
  induction ps generalizing s with
  | nil => rfl
  | cons b t ih =>
    simp only [List.foldl_cons]
    rw [max_assoc a s b.1, ih (max s b.1)]

/-- `specMax` distributes over append (both sides nonempty):
    `specMax (as ++ bs) = max (specMax as) (specMax bs)`.  The append law for the
    running max — the missing algebraic fact behind block-merge. -/
theorem specMax_append (a : ℝ × ℝ) (as' : List (ℝ × ℝ))
    (b : ℝ × ℝ) (bs' : List (ℝ × ℝ)) :
    specMax ((a :: as') ++ (b :: bs'))
      = max (specMax (a :: as')) (specMax (b :: bs')) := by
  simp only [specMax, List.cons_append, List.foldl_append, List.foldl_cons]
  -- LHS folds `b :: bs'` seeded at `foldl max as' a.1`; pull that seed out.
  rw [foldl_max_seed (as'.foldl (fun acc q => max acc q.1) a.1) b.1 bs']

/-- Specialisation to a singleton second block: `specMax (pre ++ [q])
    = max (specMax pre) q.1`.  Used by the single-step fold invariant. -/
theorem specMax_append_singleton (p : ℝ × ℝ) (ps : List (ℝ × ℝ)) (q : ℝ × ℝ) :
    specMax ((p :: ps) ++ [q]) = max (specMax (p :: ps)) q.1 := by
  have := specMax_append p ps q []
  simpa using this

/-! ## The generalised fold invariant (T9203) — the load-bearing lemma -/

/-- **T9203 — online step preserves the summary (generalised fold invariant).**
    If `s` exactly summarises a nonempty prefix `pre`, then `step s q` exactly
    summarises `pre ++ [q]`.  Proven directly from the rescale law: `step` is
    `merge s (init q)` in disguise, and both `l` and `acc` pick up the same
    `exp(s.m − m')` / `exp(q.1 − m')` factors that the reference change
    demands.  This one contract, iterated, gives the whole equivalence. -/
theorem t9203_step_summarises {s : State} {p : ℝ × ℝ} {ps : List (ℝ × ℝ)}
    (h : Summarises s (p :: ps)) (q : ℝ × ℝ) :
    Summarises (step s q) ((p :: ps) ++ [q]) := by
  obtain ⟨hm, hl, hacc⟩ := h
  refine ⟨?_, ?_, ?_⟩
  · -- running max threads through
    show max s.m q.1 = specMax ((p :: ps) ++ [q])
    rw [specMax_append_singleton, hm]
  · -- normaliser
    show s.l * Real.exp (s.m - max s.m q.1) + Real.exp (q.1 - max s.m q.1)
        = specL (max s.m q.1) ((p :: ps) ++ [q])
    have hmax : max s.m q.1 = specMax ((p :: ps) ++ [q]) := by
      rw [specMax_append_singleton, hm]
    -- specL over an append splits as a sum
    have hsplit : specL (max s.m q.1) ((p :: ps) ++ [q])
        = specL (max s.m q.1) (p :: ps) + Real.exp (q.1 - max s.m q.1) := by
      unfold specL
      rw [List.map_append, List.sum_append]
      simp
    rw [hsplit]
    have hre := (t9202_spec_rescale s.m (max s.m q.1) (p :: ps)).1
    rw [hre, ← hl]
  · -- accumulator (same shape)
    show s.acc * Real.exp (s.m - max s.m q.1) + Real.exp (q.1 - max s.m q.1) * q.2
        = specAcc (max s.m q.1) ((p :: ps) ++ [q])
    have hsplit : specAcc (max s.m q.1) ((p :: ps) ++ [q])
        = specAcc (max s.m q.1) (p :: ps) + Real.exp (q.1 - max s.m q.1) * q.2 := by
      unfold specAcc
      rw [List.map_append, List.sum_append]
      simp
    rw [hsplit]
    have hre := (t9202_spec_rescale s.m (max s.m q.1) (p :: ps)).2
    rw [hre, ← hacc]

/-- The initial state summarises the singleton list.  Base case of the fold. -/
theorem init_summarises (p : ℝ × ℝ) : Summarises (init p) [p] := by
  refine ⟨rfl, ?_, ?_⟩
  · show (1 : ℝ) = specL p.1 [p]; unfold specL; simp
  · show p.2 = specAcc p.1 [p]; unfold specAcc; simp

/-- **T9203′ — the fold invariant, iterated.** Folding `step` over any tail `ps`
    starting from a state that summarises the nonempty prefix `pre` yields a
    state summarising `pre ++ ps`.  The induction is on `ps`; each step is
    `t9203_step_summarises`. -/
theorem fold_summarises (ps : List (ℝ × ℝ)) :
    ∀ {s : State} {pre : List (ℝ × ℝ)},
      pre ≠ [] → Summarises s pre → Summarises (ps.foldl step s) (pre ++ ps) := by
  induction ps with
  | nil => intro s pre _ h; simpa using h
  | cons q qs ih =>
    intro s pre hpre h
    -- summarise pre ++ [q] first, then recurse on qs
    obtain ⟨p, ps', rfl⟩ : ∃ p ps', pre = p :: ps' := by
      cases pre with
      | nil => exact absurd rfl hpre
      | cons a t => exact ⟨a, t, rfl⟩
    have hstep : Summarises (step s q) ((p :: ps') ++ [q]) :=
      t9203_step_summarises h q
    have := ih (s := step s q) (pre := (p :: ps') ++ [q]) (by simp) hstep
    simpa [List.append_assoc] using this

/-- **T9204 — the online fold summarises the whole list.** For a nonempty
    pair-list, `online ps` exactly summarises `ps`: `l = Σ exp(xⱼ − m*)`,
    `acc = Σ exp(xⱼ − m*)·vⱼ`, `m = max x`.  Corollary of the fold invariant
    with `pre = [head]`. -/
theorem t9204_online_summarises {p : ℝ × ℝ} {ps : List (ℝ × ℝ)} :
    Summarises (online (p :: ps)) (p :: ps) := by
  show Summarises (ps.foldl step (init p)) (p :: ps)
  have := fold_summarises ps (s := init p) (pre := [p]) (by simp) (init_summarises p)
  simpa using this

/-! ## Equivalence with the direct two-pass form (T9205) -/

/-- The direct two-pass softmax-weighted sum: `out* = (Σ exp(xⱼ − m*)·vⱼ) / l*`,
    with `m* = max x`, `l* = Σ exp(xⱼ − m*)`. -/
noncomputable def directOut (ps : List (ℝ × ℝ)) : ℝ :=
  let m := specMax ps
  specAcc m ps / specL m ps

/-- The online result: `acc / l` from the final fold state. -/
noncomputable def onlineOut (ps : List (ℝ × ℝ)) : ℝ :=
  (online ps).acc / (online ps).l

/-- **T9205 — online ≡ direct.** For a nonempty pair-list, the online fold's
    `acc / l` equals the direct two-pass softmax-weighted sum.  Immediate from
    T9204: both numerator and denominator agree entrywise. -/
theorem t9205_online_eq_direct {p : ℝ × ℝ} {ps : List (ℝ × ℝ)} :
    onlineOut (p :: ps) = directOut (p :: ps) := by
  obtain ⟨hm, hl, hacc⟩ := t9204_online_summarises (p := p) (ps := ps)
  unfold onlineOut directOut
  rw [hacc, hl, hm]

/-- **T9205′ — direct ≡ softmax-weighted sum.** The two-pass output is exactly
    `Σⱼ softmax(x)ⱼ · vⱼ` — closing the loop to the textbook definition.  Both
    sides are `(Σ wt·v)/S`; `softmax(x)ⱼ = wtⱼ/S`, and pulling the common `/S`
    out of the sum gives the claim. -/
theorem t9205'_direct_eq_softmax_weighted {p : ℝ × ℝ} {ps : List (ℝ × ℝ)} :
    directOut (p :: ps)
      = ((p :: ps).map
          (fun q => softmaxCoord (specMax (p :: ps)) q.1 ((p :: ps).map Prod.fst)
                      * q.2)).sum := by
  -- Denominator identity: the softmax normaliser over the score list equals specL.
  have hsum : sumWt (specMax (p :: ps)) ((p :: ps).map Prod.fst)
      = specL (specMax (p :: ps)) (p :: ps) := by
    unfold sumWt wt specL; rw [List.map_map]; rfl
  have hpos : specL (specMax (p :: ps)) (p :: ps) ≠ 0 :=
    ne_of_gt (by rw [← hsum]; exact sumWt_pos (m := specMax (p :: ps)) (by simp))
  -- Each RHS summand: softmaxCoord · v = (exp(x-m) · v) / specL.
  have hterm : ∀ q : ℝ × ℝ,
      softmaxCoord (specMax (p :: ps)) q.1 ((p :: ps).map Prod.fst) * q.2
        = (Real.exp (q.1 - specMax (p :: ps)) * q.2) / specL (specMax (p :: ps)) (p :: ps) := by
    intro q; unfold softmaxCoord wt; rw [hsum]; ring
  rw [List.map_congr_left (fun q _ => hterm q)]
  -- Σ (fⱼ / S) = (Σ fⱼ)/S = specAcc/specL = directOut.
  rw [sum_map_div_const]
  show _ = specAcc (specMax (p :: ps)) (p :: ps) / specL (specMax (p :: ps)) (p :: ps)
  rfl

/-! ## Block-merge / append law (T9206) — arbitrary block schedules -/

/-- **T9206 — merge = append.** Merging the summaries of two nonempty blocks
    yields the summary of their concatenation.  This is the block-partition
    license: a fused kernel may process K/V in blocks of any size, in any
    left-to-right grouping, and combining the partial `(m, l, acc)` states via
    `merge` reproduces the single-pass result exactly.  Because `merge` is
    equivalent to appending the raw lists, it is associative-in-effect —
    any parenthesisation / block schedule gives the same state. -/
theorem t9206_merge_summarises {s₁ s₂ : State} {as bs : List (ℝ × ℝ)}
    (ha : as ≠ []) (hb : bs ≠ [])
    (h₁ : Summarises s₁ as) (h₂ : Summarises s₂ bs) :
    Summarises (merge s₁ s₂) (as ++ bs) := by
  obtain ⟨hm₁, hl₁, hacc₁⟩ := h₁
  obtain ⟨hm₂, hl₂, hacc₂⟩ := h₂
  -- destructure both blocks as nonempty conses
  obtain ⟨a, as', rfl⟩ : ∃ a as', as = a :: as' := by
    cases as with
    | nil => exact absurd rfl ha
    | cons a t => exact ⟨a, t, rfl⟩
  obtain ⟨b, bs', rfl⟩ : ∃ b bs', bs = b :: bs' := by
    cases bs with
    | nil => exact absurd rfl hb
    | cons b t => exact ⟨b, t, rfl⟩
  have hmaxapp : specMax ((a :: as') ++ (b :: bs'))
      = max (specMax (a :: as')) (specMax (b :: bs')) := specMax_append a as' b bs'
  -- let m be the merged max
  have hmeq : max s₁.m s₂.m = specMax ((a :: as') ++ (b :: bs')) := by
    rw [hmaxapp, hm₁, hm₂]
  refine ⟨hmeq, ?_, ?_⟩
  · -- normaliser: rescale each block's specL to the common reference, then add
    show s₁.l * Real.exp (s₁.m - max s₁.m s₂.m)
          + s₂.l * Real.exp (s₂.m - max s₁.m s₂.m)
        = specL (max s₁.m s₂.m) ((a :: as') ++ (b :: bs'))
    have hsplit : specL (max s₁.m s₂.m) ((a :: as') ++ (b :: bs'))
        = specL (max s₁.m s₂.m) (a :: as') + specL (max s₁.m s₂.m) (b :: bs') := by
      unfold specL; rw [List.map_append, List.sum_append]
    rw [hsplit]
    rw [(t9202_spec_rescale s₁.m (max s₁.m s₂.m) (a :: as')).1,
        (t9202_spec_rescale s₂.m (max s₁.m s₂.m) (b :: bs')).1, ← hl₁, ← hl₂]
  · -- accumulator: same shape
    show s₁.acc * Real.exp (s₁.m - max s₁.m s₂.m)
          + s₂.acc * Real.exp (s₂.m - max s₁.m s₂.m)
        = specAcc (max s₁.m s₂.m) ((a :: as') ++ (b :: bs'))
    have hsplit : specAcc (max s₁.m s₂.m) ((a :: as') ++ (b :: bs'))
        = specAcc (max s₁.m s₂.m) (a :: as') + specAcc (max s₁.m s₂.m) (b :: bs') := by
      unfold specAcc; rw [List.map_append, List.sum_append]
    rw [hsplit]
    rw [(t9202_spec_rescale s₁.m (max s₁.m s₂.m) (a :: as')).2,
        (t9202_spec_rescale s₂.m (max s₁.m s₂.m) (b :: bs')).2, ← hacc₁, ← hacc₂]

/-! ## Stability: every evaluated `exp` argument is ≤ 0 (T9207 / T9208) -/

/-- **T9207 — step stability.** In every `step`, the two `exp` arguments the
    kernel evaluates are `≤ 0`: `s.m − m' ≤ 0` (rescaling the running state) and
    `p.1 − m' ≤ 0` (the fresh weight), where `m' = max s.m p.1`.  Hence every
    `exp` the online form evaluates lies in `(0, 1]` — no overflow.  This is the
    numerical-stability property that motivates subtracting the running max. -/
theorem t9207_step_exp_args_nonpos (s : State) (p : ℝ × ℝ) :
    s.m - max s.m p.1 ≤ 0 ∧ p.1 - max s.m p.1 ≤ 0 := by
  constructor
  · have : s.m ≤ max s.m p.1 := le_max_left _ _
    linarith
  · have : p.1 ≤ max s.m p.1 := le_max_right _ _
    linarith

/-- **T9208 — every evaluated weight is in `(0, 1]`.** Corollary of T9207 via
    `Real.exp_pos` and `Real.exp_le_one` (`exp x ≤ 1 ↔ x ≤ 0`): the two weights
    `exp(s.m − m')` and `exp(p.1 − m')` the online step multiplies by satisfy
    `0 < w ≤ 1`.  This is the concrete no-overflow bound. -/
theorem t9208_step_weights_unit (s : State) (p : ℝ × ℝ) :
    (0 < Real.exp (s.m - max s.m p.1) ∧ Real.exp (s.m - max s.m p.1) ≤ 1) ∧
    (0 < Real.exp (p.1 - max s.m p.1) ∧ Real.exp (p.1 - max s.m p.1) ≤ 1) := by
  obtain ⟨h1, h2⟩ := t9207_step_exp_args_nonpos s p
  refine ⟨⟨Real.exp_pos _, ?_⟩, ⟨Real.exp_pos _, ?_⟩⟩
  · exact Real.exp_le_one_iff.mpr h1
  · exact Real.exp_le_one_iff.mpr h2

/-- **T9209 — merge stability.** The block-merge combine also evaluates only
    `exp` arguments `≤ 0`: `m₁ − m ≤ 0` and `m₂ − m ≤ 0` with `m = max m₁ m₂`.
    The per-block rescale factors are likewise in `(0, 1]`. -/
theorem t9209_merge_exp_args_nonpos (s₁ s₂ : State) :
    s₁.m - max s₁.m s₂.m ≤ 0 ∧ s₂.m - max s₁.m s₂.m ≤ 0 := by
  refine ⟨?_, ?_⟩
  · have : s₁.m ≤ max s₁.m s₂.m := le_max_left _ _
    linarith
  · have : s₂.m ≤ max s₁.m s₂.m := le_max_right _ _
    linarith

end Quanta.Nn
