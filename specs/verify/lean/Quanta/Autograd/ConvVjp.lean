/-
conv2d VJP correctness — via the im2col / col2im adjoint.

The Rust `quanta-autograd` computes `conv2d` as `im2col → matmul → reshape`,
and its backward reuses the matmul VJP (`MatmulVjp.lean`) for `∂cols`/`∂w`,
then recovers `∂x` with `col2im`. The one fact that is *not* matmul — the fact
that makes the `∂x` step correct — is that **col2im is the transpose of
im2col**:

  ⟨im2col x, y⟩ = ⟨x, col2im y⟩      (the adjoint / reverse-mode duality)

So if `cols = im2col x`, the reverse-mode gradient flowing into `x` is exactly
`col2im (∂cols)`. The `tests/conv.rs` `im2col_col2im_adjoint` test checks this
numerically on the real NCHW gather; here we prove it holds for **any** gather
map, which is the structural reason the kernel's index arithmetic is irrelevant
to the VJP's correctness.

Abstraction. `im2col` reads, for each patch slot `p`, at most one input pixel —
a partial map `g : P → Option I` (`none` = zero-padding, out of bounds). Then:

  im2col x  =  fun p => (g p).elim 0 x          -- pull `x` along `g`
  col2im y  =  fun i => ∑ p with g p = some i, y p   -- push `y` along `g`

and the adjoint is a double-sum rearrangement, true for every `g`. The concrete
NCHW `g` (with stride/pad) is the tested Rust kernel; its *shape* is what this
file abstracts over.
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Real.Basic

namespace Quanta.Autograd

open scoped BigOperators

variable {P I : Type*} [Fintype P] [Fintype I] [DecidableEq I]

/-- `im2col` as a pull along the gather `g`: patch slot `p` reads input `x i`
    when `g p = some i`, and `0` when `g p = none` (zero-padding / OOB). -/
def im2col (g : P → Option I) (x : I → ℝ) : P → ℝ :=
  fun p => (g p).elim 0 x

/-- `col2im` as the push along `g`: input pixel `i` accumulates `y p` over every
    patch slot `p` that gathered from it (`g p = some i`). The fold of overlapping
    patches. -/
def col2im (g : P → Option I) (y : P → ℝ) : I → ℝ :=
  fun i => ∑ p ∈ Finset.univ.filter (fun p => g p = some i), y p

omit [Fintype P] in
/-- A single patch slot contributes its value to exactly the input pixel it
    gathered from: `im2col x p * y p = Σᵢ [g p = some i] · x i · y p`. The RHS is
    `0` when `g p = none`, and `x i · y p` for the unique `i` otherwise. -/
private theorem slot_split (g : P → Option I) (x : I → ℝ) (y : P → ℝ) (p : P) :
    im2col g x p * y p
      = ∑ i, (if g p = some i then x i * y p else 0) := by
  unfold im2col
  cases hp : g p with
  | none => simp [hp]
  | some i =>
      -- Only the i-th summand survives, giving x i * y p.
      simp only [Option.elim_some]
      rw [Finset.sum_eq_single i]
      · simp [hp]
      · intro b _ hb
        simp [hp, Ne.symm hb]
      · intro h; exact absurd (Finset.mem_univ i) h

/-- **The im2col / col2im adjoint.** For all `x, y`:
    `⟨im2col x, y⟩ = ⟨x, col2im y⟩`. Hence in reverse mode the gradient into the
    convolution input is `col2im (∂cols)` — the correctness of the `∂x` step. -/
theorem im2col_col2im_adjoint (g : P → Option I) (x : I → ℝ) (y : P → ℝ) :
    (∑ p, im2col g x p * y p) = ∑ i, x i * col2im g y i := by
  -- Expand each slot as a sum over inputs, swap the order, and fold the inner
  -- sum back into col2im.
  have h1 : (∑ p, im2col g x p * y p)
      = ∑ p, ∑ i, (if g p = some i then x i * y p else 0) := by
    apply Finset.sum_congr rfl
    intro p _; exact slot_split g x y p
  rw [h1, Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  -- ∑ p, [g p = some i] · x i · y p  =  x i · ∑ p with g p = some i, y p.
  -- Push `x i` into the col2im sum (sum_filter on both) and match termwise.
  unfold col2im
  rw [Finset.sum_filter, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro p _
  by_cases hp : g p = some i
  · simp [hp, mul_comm]
  · simp [hp]

/-- Specialised orientation matching `vjp::conv2d`: the gradient into `x` is the
    push of the cols-gradient `gc` through `col2im`, and it pairs against `x`
    exactly as `im2col x` pairs against `gc`. (Same statement, named for the
    Rust call site.) -/
theorem conv2d_dx_eq_col2im (g : P → Option I) (x : I → ℝ) (gc : P → ℝ) :
    (∑ p, im2col g x p * gc p) = ∑ i, x i * col2im g gc i :=
  im2col_col2im_adjoint g x gc

end Quanta.Autograd
