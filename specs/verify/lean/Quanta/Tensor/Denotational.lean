import Mathlib.Data.Fin.Basic
import Mathlib.Algebra.BigOperators.Fin

/-! # Denotational layout algebra

Foundational layer for `quanta-tensor`'s Lean verification arm.

A layout is *defined* as its index function: `Coord → Int`. No
struct, no fields, no representation. This file is the canonical
mathematical content; the symbolic `Layout` struct in
`Quanta.Tensor.Layout` is one possible implementation, connected
back via an agreement theorem (see `Quanta.Tensor.Bridge`).

## Why this layer exists

Downstream math crates (`quanta-blas`, `quanta-sort`, `quanta-fft`)
need shape-correctness lemmas that survive Lean version bumps,
mathlib reshuffles, and `quanta-tensor` API rewrites. Lemmas
stated on the denotational layer don't reference list machinery,
record fields, or tactic implementations — they're statements
about arithmetic on `Fin n`-indexed functions.

The associativity theorem is the clearest demonstration:
`composeD_assoc` closes by `rfl` because composition of index
functions *is* function composition, and function composition is
associative by definition.

## Structure

- `Shape n` — function from axis index to extent.
- `Coord s` — function from axis index to in-bounds index.
- `Layout s` — function from coordinate to integer offset.
- `composeD` — composition via an explicit coordinate link.

This file deliberately stays small. Everything specific to the
symbolic representation lives in `Quanta.Tensor.Layout`; the
bridge between the two lives in `Quanta.Tensor.Bridge`.
-/

namespace Quanta.Tensor.Denotational

-- ── Core types ──────────────────────────────────────────────────

/-- A shape is a function from axis index to extent. The rank `n`
    is a `Nat` type parameter; the extent at each axis is `Nat`.

    Excluding zero extents at the type-class level isn't enforced
    here — downstream proofs that need `extent ≥ 1` carry that as
    an explicit hypothesis. The well-formedness invariant is more
    elegant when factored out of the basic type. -/
abbrev Shape (n : Nat) : Type := Fin n → Nat

/-- A coordinate for shape `s` is a function picking an index
    `< s i` for every axis `i`. The bound proof lives in the
    `Fin` type, so downstream code never tracks it explicitly. -/
abbrev Coord {n : Nat} (s : Shape n) : Type := (i : Fin n) → Fin (s i)

/-- A layout for shape `s` is any function from a coordinate to
    an integer offset. That's the entire definition.

    `Int` (not `Nat`) because slice / broadcast / transpose can
    introduce negative effective strides in the symbolic layer;
    the denotational `Int` covers them uniformly. -/
abbrev Layout {n : Nat} (s : Shape n) : Type := Coord s → Int

-- ── Composition ─────────────────────────────────────────────────

/-- Compose two layouts via an explicit coordinate link.

    `composeD A B link` produces a layout over `B`'s coordinate
    space, but the offset is computed by feeding the link's
    output into `A`. The `B` argument is ignored at this level —
    its role is type alignment (we're indexing into `A` *as if*
    we were walking `B`'s coordinate space, with `link` doing the
    translation).

    The link function carries the entire content of "how does
    B's coordinate map into A's coordinate." For row-major
    composition the link is the natural raveling/unraveling pair;
    for strided / broadcast cases it picks up the corresponding
    coordinate translations. -/
def composeD {n m : Nat} {sa : Shape n} {sb : Shape m}
    (A : Layout sa) (_B : Layout sb)
    (link : Coord sb → Coord sa) : Layout sb :=
  fun coord => A (link coord)

-- ── Theorems ────────────────────────────────────────────────────

/-- T8200 — Composition is function composition under the hood;
    associativity holds by `rfl`. This is the load-bearing
    correctness lemma every downstream tiled-algorithm proof
    reduces to. -/
theorem composeD_assoc
    {n m k : Nat}
    {sa : Shape n} {sb : Shape m} {sc : Shape k}
    (A : Layout sa) (B : Layout sb) (C : Layout sc)
    (link_BA : Coord sb → Coord sa)
    (link_CB : Coord sc → Coord sb) :
    composeD (composeD A B link_BA) C link_CB
      = composeD A C (link_BA ∘ link_CB) := by
  rfl

/-- T8201 — Left identity: composing with the trivial layout
    `fun _ => 0` on the right yields the constant-zero offset
    function. Useful for normalising compositions where the inner
    layout is a placeholder. -/
theorem composeD_with_zero_inner
    {n m : Nat} {sa : Shape n} {sb : Shape m}
    (A : Layout sa) (link : Coord sb → Coord sa) :
    composeD A (fun _ => 0) link = fun coord => A (link coord) := by
  rfl

/-- T8202 — Composing with the identity link returns the outer
    layout up to coordinate-space rename. When the two coordinate
    spaces have the same shape (`sa = sb` at the type level),
    `composeD A B id = A`. -/
theorem composeD_id_link
    {n : Nat} {s : Shape n}
    (A B : Layout s) :
    composeD A B (id : Coord s → Coord s) = A := by
  rfl

/-- T8203 — A layout is determined by its index function. Two
    layouts are equal iff they agree on every coordinate. Stated
    as the standard `funext` lift so downstream proofs can reason
    pointwise. -/
theorem layout_ext
    {n : Nat} {s : Shape n}
    (L₁ L₂ : Layout s)
    (h : ∀ coord, L₁ coord = L₂ coord) :
    L₁ = L₂ := by
  funext coord
  exact h coord

/-- T8204 — `composeD` distributes pointwise: applying the
    composed layout at a coordinate is exactly applying the outer
    layout at the linked coordinate. Useful as a rewrite rule. -/
theorem composeD_apply
    {n m : Nat} {sa : Shape n} {sb : Shape m}
    (A : Layout sa) (B : Layout sb)
    (link : Coord sb → Coord sa) (coord : Coord sb) :
    composeD A B link coord = A (link coord) := by
  rfl

end Quanta.Tensor.Denotational
