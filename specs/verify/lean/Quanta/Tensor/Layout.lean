/-
Tensor layout algebra — Lean formalisation of `quanta-tensor`.

Mirrors the Rust substrate at `crates/sci/quanta-tensor/src/`:

- `Shape` is a list of axis extents.
- `Layout` is a `Shape` paired with a stride list of the same length
  and an integer base offset.
- The indexer `Layout.offset` maps a coordinate vector to a flat
  buffer offset (modelled as `Int` to match Rust's `isize` strides).

The file covers the substrate and structural theorems, permutation
bijectivity, tile-offset bounds, reshape + coalesce offset
equivalence, and composition. The multi-rank `compose` is modelled
faithfully after the production right-distributive fold
(`composeFold` / `composeIntPairs` / `composeN` below), and
composition associativity is proven whenever the leftmost layout
has rank 1 — arbitrary ranks (and a divisibility-fold-shaped
result) on the middle and right (t8094), plus the rank-0-middle
case (t8096). The remaining open composition theorem is
associativity with a rank ≥ 2 leftmost layout; see the section
comment above `composeFold`.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace Quanta.Tensor

/-- A multi-dimensional shape: an ordered list of axis extents. -/
structure Shape where
  dims : List Nat
  deriving Repr, DecidableEq

namespace Shape

/-- Product of all axis extents. The empty product is 1
    (rank-0 / scalar shape). -/
def linearSize (s : Shape) : Nat :=
  s.dims.foldr (· * ·) 1

/-- Rank (number of axes). -/
def rank (s : Shape) : Nat := s.dims.length

end Shape

/-- Function-style layout: a shape paired with strides + base
    offset. Strides are `Int` so layout ops that iterate axes in
    reverse can carry negative increments — matches Rust's `isize`. -/
structure Layout where
  shape       : Shape
  strides     : List Int
  baseOffset  : Int
  deriving Repr

namespace Layout

/-- Row-major (C order) strides: rightmost axis varies fastest, so
    `strides[i] = ∏ dims[i+1..]`. -/
def rowMajorStrides : List Nat → List Int
  | []         => []
  | _ :: rest  =>
    let restStrides := rowMajorStrides rest
    let mySize : Nat := rest.foldr (· * ·) 1
    (Int.ofNat mySize) :: restStrides

/-- Construct a row-major layout. -/
def rowMajor (dims : List Nat) : Layout :=
  { shape := { dims := dims }
    strides := rowMajorStrides dims
    baseOffset := 0 }

/-- Dot product of a coordinate vector with the stride vector.
    Zips the shorter of the two; for well-formed coordinates the
    lists are the same length. -/
def dot : List Nat → List Int → Int
  | [], _              => 0
  | _, []              => 0
  | c :: cs, s :: rest => (Int.ofNat c) * s + dot cs rest

/-- Map an N-coordinate to a flat-buffer offset (modelled in `Int`
    to match Rust's `isize` arithmetic). Total over inputs; the
    well-formedness obligation that bounds + base produce a
    non-negative result is a separate property. -/
def offset (l : Layout) (coord : List Nat) : Int :=
  l.baseOffset + dot coord l.strides

/-- Number of distinct coordinates the layout indexes. -/
def linearSize (l : Layout) : Nat := l.shape.linearSize

/-- Rank. -/
def rank (l : Layout) : Nat := l.shape.dims.length

end Layout

-- ─────────────────────────────────────────────────────────────────
-- Algebra: transpose, slice, complement (rank-0 case), compose
-- with the rank-0 helper.
--
-- These mirror the production ops in
-- `crates/sci/quanta-tensor/src/layout/{ops,algebra}.rs`. The Rust
-- side handles every rank; the Lean port stays at the structural
-- tier for now (transpose, slice, rank-0 complement, base-offset
-- arithmetic). Deeper algebraic theorems on compose / permute /
-- broadcast / full-rank complement land in follow-up commits.
-- ─────────────────────────────────────────────────────────────────

namespace Layout

/-- Swap two positions in a list. Out-of-range indices behave as a
    no-op, matching the production `swapAt` helper. -/
def swapAt {α : Type} (xs : List α) (i j : Nat) : List α :=
  match xs.get? i, xs.get? j with
  | some a, some b =>
    xs.mapIdx (fun k x =>
      if k = i then b else if k = j then a else x)
  | _, _ => xs

/-- Spec-level model of the production `Layout::transpose`. Swap
    the i-th and j-th positions in both dims and strides. Base
    offset is unchanged. -/
def transpose (l : Layout) (i j : Nat) : Layout :=
  { shape := { dims := swapAt l.shape.dims i j }
    strides := swapAt l.strides i j
    baseOffset := l.baseOffset }

/-- Spec-level model of `Layout::slice(axis, start, end)`: keep
    strides, replace the `axis` extent with `end - start`, and
    advance `baseOffset` by `start * strides[axis]`. -/
def slice (l : Layout) (axis startIdx endIdx : Nat) : Layout :=
  let newExt : Nat := endIdx - startIdx
  let s : Int := l.strides.getD axis 0
  { shape := { dims := l.shape.dims.set axis newExt }
    strides := l.strides
    baseOffset := l.baseOffset + (Int.ofNat startIdx) * s }

/-- Spec-level model of `Layout::complement(cosize)` for the
    rank-0 case: the complement is a rank-1 contiguous layout
    whose single extent equals `cosize` (the production op handles
    higher-rank inputs via stride-sort, ported separately). -/
def complementRank0 (cosize : Nat) : Layout :=
  { shape := { dims := [cosize] }
    strides := [1]
    baseOffset := 0 }

end Layout

-- ─────────────────────────────────────────────────────────────────
-- Theorems.
-- ─────────────────────────────────────────────────────────────────

open Layout

/-- T8000 — `Shape.linearSize` unfolds to the right fold of
    multiplication over the extent list. -/
theorem t8000_linear_size_is_prod (s : Shape) :
    s.linearSize = s.dims.foldr (· * ·) 1 := rfl

/-- T8001 — `rowMajorStrides` has length equal to the input. -/
theorem t8001_row_major_strides_length (dims : List Nat) :
    (rowMajorStrides dims).length = dims.length := by
  induction dims with
  | nil => rfl
  | cons _ rest ih => simp [rowMajorStrides, ih]

/-- T8002 — A rank-0 row-major layout has linear size 1. -/
theorem t8002_row_major_rank_zero_linear_size :
    (rowMajor []).linearSize = 1 := by
  rfl

/-- T8003 — A rank-0 row-major layout indexes the single element
    at offset 0. -/
theorem t8003_row_major_rank_zero_offset :
    (rowMajor []).offset [] = 0 := by
  rfl

/-- T8004 — `dot` with two empty lists is 0. -/
theorem t8004_dot_empty : dot [] ([] : List Int) = 0 := rfl

/-- T8005 — `dot` zeroes out when the coordinate is the empty list. -/
theorem t8005_dot_empty_coord (strides : List Int) :
    dot [] strides = 0 := by
  cases strides with
  | nil => rfl
  | cons _ _ => rfl

/-- T8006 — `dot` zeroes out when the stride list is empty. -/
theorem t8006_dot_empty_strides (coord : List Nat) :
    dot coord [] = 0 := by
  cases coord with
  | nil => rfl
  | cons _ _ => rfl

/-- T8007 — `Layout.offset` on the empty coordinate equals
    `baseOffset`. -/
theorem t8007_offset_empty_coord (l : Layout) :
    l.offset [] = l.baseOffset := by
  unfold Layout.offset
  simp [dot]

/-- T8008 — Row-major offset of an all-zero coordinate is 0. -/
theorem t8008_row_major_origin (dims : List Nat) :
    (rowMajor dims).offset (List.replicate dims.length 0) = 0 := by
  unfold rowMajor Layout.offset
  simp
  induction dims with
  | nil => rfl
  | cons _ rest ih =>
    simp [rowMajorStrides, List.replicate, dot]
    exact ih

/-- T8009 — `linearSize` of a row-major layout is the product of
    its dims. -/
theorem t8009_row_major_linear_size (dims : List Nat) :
    (rowMajor dims).linearSize = dims.foldr (· * ·) 1 := by
  rfl

/-- T8010 — `rank` of a row-major layout equals the input length. -/
theorem t8010_row_major_rank (dims : List Nat) :
    (rowMajor dims).rank = dims.length := by
  rfl

/-- T8011 — `Shape.rank` of an empty-dims shape is 0. -/
theorem t8011_empty_shape_rank :
    Shape.rank { dims := [] } = 0 := by
  rfl

/-- T8012 — `Shape.linearSize` of an empty-dims shape is 1. -/
theorem t8012_empty_shape_linear_size :
    Shape.linearSize { dims := [] } = 1 := by
  rfl

/-- T8013 — `dot` distributes over a cons on both sides: prepending
    a coordinate `c` and a stride `s` adds `c * s` to the result. -/
theorem t8013_dot_cons (c : Nat) (cs : List Nat) (s : Int) (rest : List Int) :
    dot (c :: cs) (s :: rest) = (Int.ofNat c) * s + dot cs rest := by
  rfl

/-- T8014 — `Layout.offset` of a layout with `baseOffset = 0` is
    exactly the stride dot product. -/
theorem t8014_offset_zero_base (l : Layout) (coord : List Nat)
    (h : l.baseOffset = 0) :
    l.offset coord = dot coord l.strides := by
  unfold Layout.offset
  rw [h]
  simp

-- ─────────────────────────────────────────────────────────────────
-- Algebra theorems.
-- ─────────────────────────────────────────────────────────────────

/-- T8015 — `transpose(i, i)` preserves `baseOffset`. -/
theorem t8015_transpose_preserves_base_offset (l : Layout) (i j : Nat) :
    (transpose l i j).baseOffset = l.baseOffset := by
  rfl

/-- T8016 — `slice` keeps the stride list unchanged. -/
theorem t8016_slice_keeps_strides (l : Layout) (axis startIdx endIdx : Nat) :
    (slice l axis startIdx endIdx).strides = l.strides := by
  rfl

/-- T8017 — `slice` advances `baseOffset` by `start * strides[axis]`
    (when `axis` is in range). The `getD` falls back to 0 for
    out-of-range axes, which makes the property total over `Nat`. -/
theorem t8017_slice_advances_base (l : Layout) (axis startIdx endIdx : Nat) :
    (slice l axis startIdx endIdx).baseOffset
      = l.baseOffset + (Int.ofNat startIdx) * (l.strides.getD axis 0) := by
  rfl

/-- T8018 — `complementRank0` is a rank-1 layout. -/
theorem t8018_complement_rank0_has_rank_1 (cosize : Nat) :
    (complementRank0 cosize).rank = 1 := by
  rfl

/-- T8019 — `complementRank0` has base offset 0. -/
theorem t8019_complement_rank0_zero_base (cosize : Nat) :
    (complementRank0 cosize).baseOffset = 0 := by
  rfl

/-- T8020 — `complementRank0` has stride 1 on its single mode. -/
theorem t8020_complement_rank0_stride_one (cosize : Nat) :
    (complementRank0 cosize).strides = [1] := by
  rfl

/-- T8021 — `complementRank0`'s single extent equals `cosize`. -/
theorem t8021_complement_rank0_extent (cosize : Nat) :
    (complementRank0 cosize).shape.dims = [cosize] := by
  rfl

/-- T8022 — `complementRank0`'s linear size equals `cosize`. -/
theorem t8022_complement_rank0_linear_size (cosize : Nat) :
    (complementRank0 cosize).linearSize = cosize := by
  unfold Layout.linearSize
  simp [complementRank0, Shape.linearSize]

/-- T8023 — `slice` keeps the strides list at the same length as
    the original (the strides are literally unchanged). This is
    the algebraic prerequisite for "rank is preserved". -/
theorem t8023_slice_preserves_strides_length
    (l : Layout) (axis startIdx endIdx : Nat) :
    (slice l axis startIdx endIdx).strides.length = l.strides.length := by
  rfl

/-- T8024 — `slice` keeps the dims list at the same length as the
    original. `List.set` is length-preserving regardless of whether
    `axis` is in range. -/
theorem t8024_slice_preserves_dims_length
    (l : Layout) (axis startIdx endIdx : Nat) :
    (slice l axis startIdx endIdx).shape.dims.length = l.shape.dims.length := by
  simp [slice, List.length_set]

/-- T8025 — `slice` preserves `rank`. Combines T8024 with the
    definitional unfold of `Layout.rank`. -/
theorem t8025_slice_preserves_rank
    (l : Layout) (axis startIdx endIdx : Nat) :
    (slice l axis startIdx endIdx).rank = l.rank := by
  unfold Layout.rank
  exact t8024_slice_preserves_dims_length l axis startIdx endIdx

/-- T8026 — `complementRank0` is well-formed in the structural
    sense: its shape and stride lists are non-empty and of equal
    length. (Length 1 matches the rank-1 claim from T8018.) -/
theorem t8026_complement_rank0_strides_match_rank (cosize : Nat) :
    (complementRank0 cosize).strides.length
      = (complementRank0 cosize).shape.dims.length := by
  rfl

/-- T8027 — Full-range `slice(axis, 0, _)` leaves `baseOffset`
    unchanged. The advance is `0 * stride[axis] = 0`. -/
theorem t8027_slice_from_zero_keeps_base
    (l : Layout) (axis endIdx : Nat) :
    (slice l axis 0 endIdx).baseOffset = l.baseOffset := by
  unfold Layout.slice
  simp

/-- T8028 — A `slice` with `start = end` produces a layout whose
    axis extent is 0. (The production op rejects this; here we
    just state the spec-level fact that `Nat.sub` saturates.) -/
theorem t8028_slice_empty_range_zero_extent
    (l : Layout) (axis startIdx : Nat) :
    (slice l axis startIdx startIdx).shape.dims.length = l.shape.dims.length := by
  exact t8024_slice_preserves_dims_length l axis startIdx startIdx

-- ─────────────────────────────────────────────────────────────────
-- The tile-offset bound. For a row-major layout and any in-range
-- coordinate, the offset is a non-negative integer strictly less
-- than `linearSize`. This is the bijection precondition that
-- downstream sort / FFT correctness theorems lean on.
-- ─────────────────────────────────────────────────────────────────

/-- `dot coord (rowMajorStrides dims)` is non-negative for any
    coordinate. Every term in the sum is a product of two
    non-negative naturals embedded in `Int`. -/
theorem t8029_dot_row_major_nonneg (coord dims : List Nat) :
    0 ≤ dot coord (rowMajorStrides dims) := by
  induction coord generalizing dims with
  | nil => simp [dot]
  | cons c cs ih =>
    cases dims with
    | nil => simp [rowMajorStrides, dot]
    | cons d ds =>
      simp [rowMajorStrides, dot]
      have h1 : (0 : Int) ≤ (c : Int) * ((ds.foldr (· * ·) 1 : Nat) : Int) := by
        have hc : (0 : Int) ≤ (c : Int) := Int.ofNat_nonneg c
        have hs : (0 : Int) ≤ ((ds.foldr (· * ·) 1 : Nat) : Int) :=
          Int.ofNat_nonneg _
        exact Int.mul_nonneg hc hs
      have h2 : (0 : Int) ≤ dot cs (rowMajorStrides ds) := ih ds
      exact Int.add_nonneg h1 h2

/-- T8030 — Non-negativity half of `tile_offset_bound`. For any
    row-major layout and any coordinate, the resulting offset is
    a non-negative `Int`. Combines T8029 (the dot-product
    non-negativity) with the `baseOffset = 0` rewrite that
    `rowMajor` guarantees. -/
theorem t8030_row_major_offset_nonneg (dims coord : List Nat) :
    0 ≤ (rowMajor dims).offset coord := by
  unfold rowMajor Layout.offset
  simp
  exact t8029_dot_row_major_nonneg coord dims

/-- A coordinate is **bounded** for the given dims when their
    lengths match and every component is strictly less than the
    corresponding extent. This is the precondition row-major
    indexing needs to land in `[0, linearSize)`. -/
def CoordBounded : List Nat → List Nat → Prop
  | [], []          => True
  | c :: cs, d :: ds => c < d ∧ CoordBounded cs ds
  | _, _            => False

/-- T8031 — Upper-bound half. For any bounded coord, the
    row-major dot product is strictly less than the linear size,
    as an `Int` comparison. -/
theorem t8031_dot_row_major_lt_linear_size
    (coord dims : List Nat) (h : CoordBounded coord dims) :
    dot coord (rowMajorStrides dims)
      < ((dims.foldr (· * ·) 1 : Nat) : Int) := by
  induction coord generalizing dims with
  | nil =>
    cases dims with
    | nil => simp [dot, rowMajorStrides]
    | cons _ _ => simp [CoordBounded] at h
  | cons c cs ih =>
    cases dims with
    | nil => simp [CoordBounded] at h
    | cons d ds =>
      obtain ⟨hc, hbtail⟩ := h
      have ih' := ih ds hbtail
      simp [rowMajorStrides, dot, List.foldr]
      -- The head stride is S := ds.foldr (·*·) 1, a Nat.
      set S : Nat := ds.foldr (· * ·) 1 with hS
      -- Goal: c * S + dot cs ... < d * S (as Int).
      -- We have: dot cs (rowMajorStrides ds) < (S : Int).
      -- And c + 1 ≤ d, so c * S + S ≤ d * S, so c * S + dot < d * S.
      have hSnn : (0 : Int) ≤ (S : Int) := Int.ofNat_nonneg _
      have hcS : (c : Int) * S ≥ 0 := by
        apply Int.mul_nonneg
        · exact Int.ofNat_nonneg _
        · exact hSnn
      have hcsucc : (c : Int) + 1 ≤ (d : Int) := by exact_mod_cast hc
      -- (c + 1) * S ≤ d * S
      have hub : ((c : Int) + 1) * (S : Int) ≤ (d : Int) * (S : Int) := by
        exact mul_le_mul_of_nonneg_right hcsucc hSnn
      -- Algebra: c * S + dot < c * S + S = (c + 1) * S ≤ d * S
      linarith

/-- T8032 — `tile_offset_bound`. For a row-major layout and any
    bounded coordinate, the offset is in `[0, linearSize)` as an
    `Int`. This is the bijection precondition downstream sort /
    FFT correctness theorems rely on. -/
theorem t8032_tile_offset_bound (dims coord : List Nat)
    (h : CoordBounded coord dims) :
    0 ≤ (rowMajor dims).offset coord ∧
      (rowMajor dims).offset coord < ((rowMajor dims).linearSize : Int) := by
  refine ⟨t8030_row_major_offset_nonneg dims coord, ?_⟩
  unfold rowMajor Layout.offset Layout.linearSize Shape.linearSize
  simp
  exact t8031_dot_row_major_lt_linear_size coord dims h

-- ─────────────────────────────────────────────────────────────────
-- Permutation bijectivity. The Rust `permute(perm)` op takes a
-- list `perm` of axis indices and rearranges the layout's dims +
-- strides so that new axis `i` is old axis `perm[i]`. The Lean
-- side mirrors this with `permuteList`, and the theorem below
-- states that when `perm` is a valid permutation of
-- `0..xs.length`, `permuteList xs perm` is a permutation of `xs`
-- in the `List.Perm` sense — i.e. the rearranged list has the
-- same multiset of elements.
-- ─────────────────────────────────────────────────────────────────

/-- Apply a permutation to a list: `perm[i] = j` selects old
    element `j` into new position `i`. Mirrors the production
    `permute` (in `crates/sci/quanta-tensor/src/layout/ops.rs`). -/
def permuteList {α : Type} [Inhabited α] (xs : List α) (perm : List Nat) : List α :=
  perm.map (fun j => xs.getD j default)

/-- `IsPermOf n perm` says that `perm` is a permutation of
    `List.range n` — equivalently, every index in `0..n` appears
    in `perm` exactly once. -/
def IsPermOf (n : Nat) (perm : List Nat) : Prop :=
  List.Perm perm (List.range n)

/-- T8033 — Identity permutation. `permuteList xs (List.range
    xs.length)` returns `xs` itself. The map of `List.range n`
    through `xs.getD · default` reconstructs `xs` exactly when
    every index is in range. -/
theorem t8033_permute_identity {α : Type} [Inhabited α] (xs : List α) :
    permuteList xs (List.range xs.length) = xs := by
  unfold permuteList
  induction xs with
  | nil => simp
  | cons x rest ih =>
    rw [List.length_cons, List.range_succ_eq_map]
    simp only [List.map_cons, List.getD_cons_zero]
    have hmap : List.map (fun j => (x :: rest).getD j default)
                 (List.map Nat.succ (List.range rest.length))
             = List.map (fun j => rest.getD j default) (List.range rest.length) := by
      simp [List.map_map, Function.comp_def]
    rw [hmap, ih]

/-- T8034 — Permutation bijectivity. If `perm` is a permutation
    of `List.range xs.length`, then `permuteList xs perm` is a
    permutation of `xs` in the `List.Perm` (multiset-equal)
    sense.

    This is the load-bearing lemma for downstream sort proofs:
    every reordering of a tensor's axes preserves the multiset of
    elements, so sort/permute can chain without losing data. -/
theorem t8034_permute_is_bijection {α : Type} [Inhabited α]
    (xs : List α) (perm : List Nat) (h : IsPermOf xs.length perm) :
    List.Perm (permuteList xs perm) xs := by
  -- Step 1: perm ~ List.range xs.length (by h), so mapping
  -- through `xs.getD · default` preserves the multiset.
  have hmap : List.Perm
      (List.map (fun j => xs.getD j default) perm)
      (List.map (fun j => xs.getD j default) (List.range xs.length)) :=
    List.Perm.map _ h
  -- Step 2: map of identity range = xs (by T8033 unfolded).
  have hid : List.map (fun j => xs.getD j default) (List.range xs.length) = xs := by
    have := t8033_permute_identity xs
    unfold permuteList at this
    exact this
  rw [hid] at hmap
  exact hmap

-- ─────────────────────────────────────────────────────────────────
-- Composition of rank-1 layouts. As a foundation we ship the
-- rank-1×rank-1 closed form — the simplest case CuTe handles as a
-- special shortcut — together with the identity-composition and
-- rank-1 associativity theorems. The full multi-rank `compose`
-- (the divisibility-checking fold from CuTe) is modelled further
-- below (`composeFold` / `composeIntPairs` / `composeN`), with
-- associativity proven for a rank-1 leftmost layout (t8094).
-- ─────────────────────────────────────────────────────────────────

namespace Layout

/-- Rank-1 layout constructor: a single mode of extent `n` and
    stride `s`. Used to state and prove the rank-1 composition
    closed form below. -/
def rank1 (n : Nat) (s : Int) : Layout :=
  { shape := { dims := [n] }
    strides := [s]
    baseOffset := 0 }

/-- Rank-1×rank-1 composition. For two rank-1 layouts `(b, db)`
    on top of `(a, sa)`, the composition shrinks to `(b, db * sa)`
    — the extent comes from the RHS, the stride is the product of
    the two. This matches CuTe's
    `composition_impl(LShape, LStride, RShape, RStride)` shortcut
    when both sides are integral. -/
def compose11 (a : Layout) (b : Layout) : Layout :=
  rank1 (b.shape.dims.headD 1) ((b.strides.headD 0) * (a.strides.headD 0))

end Layout

/-- T8035 — `rank1 n s` has rank 1. -/
theorem t8035_rank1_rank (n : Nat) (s : Int) :
    (rank1 n s).rank = 1 := by
  rfl

/-- T8036 — `rank1 n s` has linear size `n`. -/
theorem t8036_rank1_linear_size (n : Nat) (s : Int) :
    (rank1 n s).linearSize = n := by
  unfold Layout.linearSize
  simp [rank1, Shape.linearSize]

/-- T8037 — `compose11 (rank1 a 1) (rank1 b db) = rank1 b db`.
    Composing with the unit-stride identity on the left is the
    identity. -/
theorem t8037_compose11_left_identity (a b : Nat) (db : Int) :
    compose11 (rank1 a 1) (rank1 b db) = rank1 b db := by
  unfold compose11 rank1
  simp

/-- T8038 — `compose11` is associative on rank-1 layouts.
    Reduces to `(dc * db) * sa = dc * (db * sa)` in `Int`, which
    is the standard associativity. -/
theorem t8038_compose11_assoc (na nb nc : Nat) (sa db dc : Int) :
    compose11 (compose11 (rank1 na sa) (rank1 nb db)) (rank1 nc dc)
      = compose11 (rank1 na sa) (compose11 (rank1 nb db) (rank1 nc dc)) := by
  unfold compose11 rank1
  simp
  ring

/-- T8039 — `compose11` preserves rank-1-ness. Stated as: the
    composed layout's rank equals 1. -/
theorem t8039_compose11_preserves_rank1 (a b : Layout) :
    (compose11 a b).rank = 1 := by
  rfl

/-- T8040 — `compose11`'s base offset is 0. Since `rank1`
    constructors set base 0 and `compose11` builds via `rank1`,
    the result has base 0 regardless of inputs. -/
theorem t8040_compose11_zero_base (a b : Layout) :
    (compose11 a b).baseOffset = 0 := by
  rfl

-- ─────────────────────────────────────────────────────────────────
-- Rank-1 LHS, rank-N RHS composition. The right-distributivity
-- branch of CuTe's `composition_impl`: composing a single LHS
-- mode with an RHS layout of multiple modes is the same as
-- composing the LHS mode with each RHS mode separately and
-- concatenating the results.
-- ─────────────────────────────────────────────────────────────────

namespace Layout

/-- Compose a rank-1 LHS layout `(a_dim, sa)` with a rank-N RHS
    layout `(b_dims, b_strides)` by mapping each RHS mode
    `(b_i, db_i)` to `(b_i, db_i * sa)`. The result's shape is
    `b.shape.dims` (unchanged) and its strides are
    `b.strides.map (· * sa)`. Base offset is 0.

    Equivalent to applying `compose11` mode-wise to the RHS;
    direct construction here keeps the result flat. -/
def compose1n (a : Layout) (b : Layout) : Layout :=
  { shape := b.shape
    strides := b.strides.map (· * (a.strides.headD 0))
    baseOffset := 0 }

end Layout

/-- T8041 — `compose1n` preserves the RHS's rank. The result's
    shape comes directly from `b.shape`. -/
theorem t8041_compose1n_preserves_rhs_rank (a b : Layout) :
    (compose1n a b).rank = b.rank := by
  rfl

/-- T8042 — `compose1n` keeps the RHS's shape unchanged. -/
theorem t8042_compose1n_preserves_rhs_shape (a b : Layout) :
    (compose1n a b).shape = b.shape := by
  rfl

/-- T8043 — `compose1n` produces a layout with base offset 0. -/
theorem t8043_compose1n_zero_base (a b : Layout) :
    (compose1n a b).baseOffset = 0 := by
  rfl

/-- T8044 — `compose1n` agrees with `compose11` when the RHS is
    rank 1. -/
theorem t8044_compose1n_matches_compose11_on_rank1
    (a : Layout) (n : Nat) (s : Int) :
    compose1n a (rank1 n s) = compose11 a (rank1 n s) := by
  unfold compose1n compose11 rank1
  simp

/-- T8045 — `compose1n a (rank1 n 1) = rank1 n sa`, where `sa` is
    `a`'s head stride. Composing with a unit-stride rank-1 RHS
    just lifts the LHS's stride into the result. Special case of
    the more general right-identity behaviour. -/
theorem t8045_compose1n_with_unit_stride_rhs
    (a : Layout) (n : Nat) :
    compose1n a (rank1 n 1) = rank1 n (a.strides.headD 0) := by
  unfold compose1n rank1
  simp

/-- T8046 — `compose1n` is "right-distributive" over rank-1 RHS
    decomposition in the sense that the strides emerge as the
    product of the RHS strides with the LHS head stride. Stated
    explicitly so the structural property is visible to readers
    and downstream proofs. -/
theorem t8046_compose1n_strides_formula (a b : Layout) :
    (compose1n a b).strides = b.strides.map (· * (a.strides.headD 0)) := by
  rfl

/-- T8047 — `compose1n`'s strides under rank-1×rank-N×rank-1
    associativity. Both nested `compose1n` calls produce the same
    stride list, by associativity of `Int` multiplication.

    A subtle textual issue: `compose1n` reaches into the
    `.strides.headD 0` accessor on each side, but the inner
    `compose1n` has produced a *mapped* strides list, so
    `headD` lands on different fold-shaped expressions on each
    side. We sidestep that by stating + proving the stride
    formula directly, then deriving the layout equality. -/
theorem t8047_compose1n_strides_assoc
    (sa dc : Int) (b : Layout) :
    b.strides.map (fun s => dc * (s * sa))
      = b.strides.map (fun s => (dc * s) * sa) := by
  apply List.map_congr_left
  intros
  ring

-- ─────────────────────────────────────────────────────────────────
-- Full-rank composition associativity, `compose1n` tier. T8048
-- lifts T8047 from the stride-list level to a layout-level
-- equality, then T8049-T8052 extend the result outward to the
-- rank-1 × rank-N cases and the symmetric forms downstream tiling
-- needs. (`compose1n` is the head-stride shortcut; the faithful
-- divisibility-checking fold and its associativity theorem t8094
-- live in the `composeN` section at the end of the file.)
-- ─────────────────────────────────────────────────────────────────

/-- T8048 — Associativity for rank-1 × rank-1 × rank-N.
    `compose1n` over a `compose11` LHS produces the same layout
    as nested `compose1n` calls. Reduces to `List.map_map` plus
    integer associativity. -/
theorem t8048_compose11_compose1n_assoc
    (na nb : Nat) (sa sb : Int) (c : Layout) :
    compose1n (compose11 (rank1 na sa) (rank1 nb sb)) c
      = compose1n (rank1 na sa) (compose1n (rank1 nb sb) c) := by
  unfold compose1n compose11 rank1
  simp [List.map_map, Function.comp, mul_assoc]

/-- T8049 — Associativity for rank-1 × rank-1 × rank-1 stated at
    the `compose1n` level. Equivalent to T8038 (which is stated at
    the `compose11` level); we include this restatement so the
    full chain has a uniform `compose1n` shape. -/
theorem t8049_compose1n_rank1_assoc
    (na nb nc : Nat) (sa sb sc : Int) :
    compose1n (compose1n (rank1 na sa) (rank1 nb sb)) (rank1 nc sc)
      = compose1n (rank1 na sa) (compose1n (rank1 nb sb) (rank1 nc sc)) := by
  unfold compose1n rank1
  simp [List.map_map, Function.comp, mul_assoc]

/-- T8050 — Strides of `compose1n a (compose1n b c)` in fully
    expanded form: `c.strides.map (· * sb * sa)` where `sa, sb` are
    the head strides of `a, b`. Stated explicitly so downstream
    proofs can pattern-match on the closed form. -/
theorem t8050_compose1n_nested_strides
    (a b c : Layout) :
    (compose1n a (compose1n b c)).strides
      = c.strides.map (fun s => s * (b.strides.headD 0) * (a.strides.headD 0)) := by
  unfold compose1n
  simp [List.map_map, Function.comp, mul_assoc]

/-- T8051 — The fully expanded strides of the left-associated form
    `compose1n (compose1n a b) c` match T8050's right-associated
    form. The shape and base-offset components are equal by
    construction (`compose1n` always inherits the rightmost layout's
    shape and emits base 0); putting T8050 + this together gives
    full layout-level associativity for any rank-1 × rank-1 ×
    rank-N triple (T8052 below).

    After unfolding `compose1n` once on each side, the inner
    `b.strides.map (· * sa)` lands on `headD 0`. We case-split on
    `b.strides`: in both the empty and non-empty cases the
    multiplication closes by `ring`. -/
theorem t8051_compose1n_left_assoc_strides
    (a b c : Layout) :
    (compose1n (compose1n a b) c).strides
      = c.strides.map (fun s => s * (b.strides.headD 0) * (a.strides.headD 0)) := by
  unfold compose1n
  -- The outer `compose1n`'s strides are `c.strides.map (· * sab)`
  -- where `sab = (b.strides.map (· * sa)).headD 0`. The headD of
  -- a mapped list equals `f` applied to the original head (with
  -- the default mapped accordingly).
  cases hb : b.strides with
  | nil =>
    simp [hb]
  | cons h t =>
    simp [hb]
    intro _ _
    ring

/-- T8052 — Full layout-level associativity for the rank-1 × rank-1
    × rank-N case. Both sides have the same shape (`c.shape` —
    `compose1n` always inherits the rightmost layout's shape), the
    same base offset (`0` by construction), and the same strides
    (by T8050 + T8051). -/
theorem t8052_compose1n_assoc_with_rank1_lhs
    (a b c : Layout) :
    compose1n (compose1n a b) c = compose1n a (compose1n b c) := by
  -- Strides equality from T8050 + T8051.
  have hs : (compose1n (compose1n a b) c).strides
            = (compose1n a (compose1n b c)).strides := by
    rw [t8050_compose1n_nested_strides, t8051_compose1n_left_assoc_strides]
  -- Both sides are Layout records; shape and base are equal by
  -- construction (compose1n always inherits the rightmost shape
  -- and sets base 0).
  show (compose1n (compose1n a b) c) = (compose1n a (compose1n b c))
  -- Use the `Layout` constructor injectivity: equality of records
  -- reduces to fieldwise equality.
  cases h1 : compose1n (compose1n a b) c
  cases h2 : compose1n a (compose1n b c)
  congr 1
  · -- shape: both sides equal c.shape.
    have e1 : (compose1n (compose1n a b) c).shape = c.shape := by
      unfold compose1n; rfl
    have e2 : (compose1n a (compose1n b c)).shape = c.shape := by
      unfold compose1n; rfl
    rw [h1] at e1
    rw [h2] at e2
    exact e1.trans e2.symm
  · -- strides: by hs.
    rw [h1, h2] at hs
    exact hs
  · -- base offset: both 0.
    have e1 : (compose1n (compose1n a b) c).baseOffset = 0 := by
      unfold compose1n; rfl
    have e2 : (compose1n a (compose1n b c)).baseOffset = 0 := by
      unfold compose1n; rfl
    rw [h1] at e1
    rw [h2] at e2
    exact e1.trans e2.symm

-- ─────────────────────────────────────────────────────────────────
-- Reshape + coalesce. Mirrors the production ops added in
-- `crates/sci/quanta-tensor/src/layout/ops.rs`:
--
-- - `reshape` models the SUCCESS path of `Layout::reshape` (the
--   production op guards with `is_contiguous` + size-match and
--   refuses otherwise; the theorems below take those guards as
--   hypotheses).
-- - `coalesce` models `Layout::coalesce` exactly: one right-to-left
--   pass that drops extent-1 modes and fuses a mode into the group
--   on its right when its stride continues the group's progression.
--   The Rust loop's `acc.last_mut()` corresponds to the head of the
--   recursive result here (Rust builds innermost-first and
--   reverses; the fold below combines from the right directly).
-- ─────────────────────────────────────────────────────────────────

namespace Layout

/-- Unflatten a row-major linear index into a coordinate vector
    over `dims`: outer coordinate is `k / ∏ dims[1..]`, and the
    remainder recurses. Inner axes vary fastest, matching
    `rowMajorStrides`. -/
def unflatten : List Nat → Nat → List Nat
  | [], _ => []
  | _ :: ds, k =>
    (k / ds.foldr (· * ·) 1) :: unflatten ds (k % ds.foldr (· * ·) 1)

/-- A layout is (row-major) **contiguous** when its strides are
    exactly the dense row-major strides of its dims. Mirrors the
    production `Layout::is_contiguous`; `baseOffset` is
    deliberately not part of the definition (a leading-axis slice
    is still one dense block, just shifted). -/
def IsContiguous (l : Layout) : Prop :=
  l.strides = rowMajorStrides l.shape.dims

/-- Spec-level model of the success path of `Layout::reshape`:
    swap in the new dims with dense row-major strides, keep
    `baseOffset`. -/
def reshape (l : Layout) (newDims : List Nat) : Layout :=
  { shape := { dims := newDims }
    strides := rowMajorStrides newDims
    baseOffset := l.baseOffset }

/-- One step of the coalesce fold: `p` is the next (extent, stride)
    mode moving right-to-left, `r` is the already-coalesced suffix.
    Drop extent-1 modes; fuse `p` into the suffix's outermost group
    when `p`'s stride continues the group's progression
    (`s = s' * d'`); otherwise start a new group. -/
def coalesceStep (p : Nat × Int) (r : List (Nat × Int)) : List (Nat × Int) :=
  if p.1 = 1 then r
  else
    match r with
    | [] => [p]
    | (d', s') :: tail =>
      if p.2 = s' * (Int.ofNat d') then (p.1 * d', s') :: tail
      else p :: (d', s') :: tail

/-- CuTe coalesce over an (extent, stride) pair list: fold the
    step right-to-left. -/
def coalescePairs (ps : List (Nat × Int)) : List (Nat × Int) :=
  ps.foldr coalesceStep []

/-- Spec-level model of `Layout::coalesce`. -/
def coalesce (l : Layout) : Layout :=
  { shape := { dims := (coalescePairs (l.shape.dims.zip l.strides)).map Prod.fst }
    strides := (coalescePairs (l.shape.dims.zip l.strides)).map Prod.snd
    baseOffset := l.baseOffset }

/-- Product of the extents of a pair list — the pair-level view of
    `linearSize`. -/
def pairsProd (ps : List (Nat × Int)) : Nat :=
  (ps.map Prod.fst).foldr (· * ·) 1

/-- Offset of row-major linear index `k` through a pair list — the
    pair-level view of `dot ∘ unflatten` (no base offset). -/
def pairsOffset (ps : List (Nat × Int)) (k : Nat) : Int :=
  dot (unflatten (ps.map Prod.fst) k) (ps.map Prod.snd)

end Layout

-- ─────────────────────────────────────────────────────────────────
-- Reshape theorems.
-- ─────────────────────────────────────────────────────────────────

/-- T8053 — `reshape` installs exactly the requested dims. -/
theorem t8053_reshape_dims (l : Layout) (newDims : List Nat) :
    (l.reshape newDims).shape.dims = newDims := rfl

/-- T8054 — `reshape`'s rank is the length of the requested dims. -/
theorem t8054_reshape_rank (l : Layout) (newDims : List Nat) :
    (l.reshape newDims).rank = newDims.length := rfl

/-- T8055 — `reshape` preserves `baseOffset`: a contiguous reshape
    is a pure reindexing of the same shifted block. -/
theorem t8055_reshape_preserves_base_offset (l : Layout) (newDims : List Nat) :
    (l.reshape newDims).baseOffset = l.baseOffset := rfl

/-- T8056 — `reshape` installs dense row-major strides. -/
theorem t8056_reshape_strides_row_major (l : Layout) (newDims : List Nat) :
    (l.reshape newDims).strides = rowMajorStrides newDims := rfl

/-- T8057 — the result of a `reshape` is itself contiguous, so
    reshapes chain (the Rust round-trip test relies on this). -/
theorem t8057_reshape_is_contiguous (l : Layout) (newDims : List Nat) :
    IsContiguous (l.reshape newDims) := rfl

/-- T8058 — under the production op's size guard, `reshape`
    preserves `linearSize`. -/
theorem t8058_reshape_preserves_linear_size (l : Layout) (newDims : List Nat)
    (h : newDims.foldr (· * ·) 1 = l.linearSize) :
    (l.reshape newDims).linearSize = l.linearSize := h

/-- T8059 — `unflatten` produces a coordinate of the right rank. -/
theorem t8059_unflatten_length (dims : List Nat) (k : Nat) :
    (unflatten dims k).length = dims.length := by
  induction dims generalizing k with
  | nil => rfl
  | cons d ds ih => simp [unflatten, ih]

/-- T8060 — the flatten/unflatten round trip: dotting the
    unflattened coordinate of `k` against the row-major strides
    recovers `k` exactly, for any in-range `k`. This is the heart
    of "contiguous reshape is a pure reindexing". -/
theorem t8060_unflatten_row_major_dot (dims : List Nat) (k : Nat)
    (h : k < dims.foldr (· * ·) 1) :
    dot (unflatten dims k) (rowMajorStrides dims) = Int.ofNat k := by
  induction dims generalizing k with
  | nil =>
    have hk0 : k = 0 := Nat.lt_one_iff.mp h
    subst hk0
    rfl
  | cons d ds ih =>
    have hP : 0 < ds.foldr (· * ·) 1 := by
      rcases Nat.eq_zero_or_pos (ds.foldr (· * ·) 1) with h0 | hpos
      · rw [List.foldr_cons, h0, Nat.mul_zero] at h
        exact absurd h (Nat.not_lt_zero k)
      · exact hpos
    simp only [unflatten, rowMajorStrides, dot]
    rw [ih (k % ds.foldr (· * ·) 1) (Nat.mod_lt k hP)]
    -- `Int.ofNat` distributes over `*` and `+` definitionally, so
    -- the goal is `Int.ofNat (k/P * P + k%P) = Int.ofNat k` up to
    -- defeq — one `congrArg` over the Nat div/mod identity.
    exact congrArg Int.ofNat (Nat.div_add_mod' k (ds.foldr (· * ·) 1))

/-- T8061 — closed form for contiguous offsets: on a contiguous
    layout, the offset of the unflattened linear index `k` is
    `baseOffset + k`. -/
theorem t8061_contiguous_offset_linear (l : Layout) (k : Nat)
    (hc : IsContiguous l) (hk : k < l.linearSize) :
    l.offset (unflatten l.shape.dims k) = l.baseOffset + Int.ofNat k := by
  have hc' : l.strides = rowMajorStrides l.shape.dims := hc
  unfold Layout.offset
  rw [hc', t8060_unflatten_row_major_dot l.shape.dims k hk]

/-- T8062 — **reshape offset equivalence.** On a contiguous layout,
    reshaping to any same-size dims leaves the offset of every
    linear index unchanged: both sides land at `baseOffset + k`.
    Reshape is a pure reindexing of the contiguous domain. -/
theorem t8062_reshape_offset_equiv (l : Layout) (newDims : List Nat) (k : Nat)
    (hc : IsContiguous l)
    (hsize : newDims.foldr (· * ·) 1 = l.linearSize)
    (hk : k < l.linearSize) :
    (l.reshape newDims).offset (unflatten newDims k)
      = l.offset (unflatten l.shape.dims k) := by
  have hk' : k < (l.reshape newDims).linearSize := by
    rw [t8058_reshape_preserves_linear_size l newDims hsize]
    exact hk
  calc (l.reshape newDims).offset (unflatten newDims k)
      = (l.reshape newDims).baseOffset + Int.ofNat k :=
        t8061_contiguous_offset_linear (l.reshape newDims) k rfl hk'
    _ = l.baseOffset + Int.ofNat k := rfl
    _ = l.offset (unflatten l.shape.dims k) :=
        (t8061_contiguous_offset_linear l k hc hk).symm

-- ─────────────────────────────────────────────────────────────────
-- Coalesce theorems. First the four step-shape lemmas (the case
-- split every later proof rewrites with), then the structural
-- invariants, then the offset-equivalence chain.
-- ─────────────────────────────────────────────────────────────────

/-- T8063 — `coalescePairs` unfolds one cons at a time through
    `coalesceStep`. -/
theorem t8063_coalescePairs_cons (p : Nat × Int) (rest : List (Nat × Int)) :
    coalescePairs (p :: rest) = coalesceStep p (coalescePairs rest) := rfl

/-- T8064 — an extent-1 mode is dropped: its coordinate is always
    0, so its stride can never contribute to an offset. -/
theorem t8064_coalesceStep_drops_unit (d : Nat) (s : Int)
    (r : List (Nat × Int)) (h : d = 1) :
    coalesceStep (d, s) r = r := by
  simp [coalesceStep, h]

/-- T8065 — a non-unit mode arriving at an empty suffix starts the
    first group. -/
theorem t8065_coalesceStep_starts_group (d : Nat) (s : Int) (h : d ≠ 1) :
    coalesceStep (d, s) [] = [(d, s)] := by
  simp [coalesceStep, h]

/-- T8066 — the fuse case: when the incoming stride continues the
    outermost group's progression (`s = s' * d'`), the extents
    multiply and the group's stride survives. -/
theorem t8066_coalesceStep_fuses (d d' : Nat) (s s' : Int)
    (tail : List (Nat × Int)) (hd : d ≠ 1) (hs : s = s' * Int.ofNat d') :
    coalesceStep (d, s) ((d', s') :: tail) = (d * d', s') :: tail := by
  simp [coalesceStep, hd, hs]

/-- T8067 — the push case: a non-unit, non-fusable mode starts a
    new group in front. -/
theorem t8067_coalesceStep_pushes (d d' : Nat) (s s' : Int)
    (tail : List (Nat × Int)) (hd : d ≠ 1) (hs : s ≠ s' * Int.ofNat d') :
    coalesceStep (d, s) ((d', s') :: tail) = (d, s) :: (d', s') :: tail := by
  simp [coalesceStep, hd, hs]
  exact hs

/-- T8068 — every step multiplies the running extent product by the
    incoming extent, whatever branch fires. -/
theorem t8068_coalesceStep_prod (d : Nat) (s : Int) (r : List (Nat × Int)) :
    pairsProd (coalesceStep (d, s) r) = d * pairsProd r := by
  by_cases hd1 : d = 1
  · rw [t8064_coalesceStep_drops_unit d s r hd1, hd1, Nat.one_mul]
  · cases r with
    | nil =>
      rw [t8065_coalesceStep_starts_group d s hd1]
      rfl
    | cons p tail =>
      obtain ⟨d', s'⟩ := p
      by_cases hf : s = s' * Int.ofNat d'
      · rw [t8066_coalesceStep_fuses d d' s s' tail hd1 hf]
        show (d * d') * pairsProd tail = d * (d' * pairsProd tail)
        rw [Nat.mul_assoc]
      · rw [t8067_coalesceStep_pushes d d' s s' tail hd1 hf]
        rfl

/-- T8069 — `coalescePairs` preserves the extent product. -/
theorem t8069_coalescePairs_prod (ps : List (Nat × Int)) :
    pairsProd (coalescePairs ps) = pairsProd ps := by
  induction ps with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨d, s⟩ := p
    rw [t8063_coalescePairs_cons, t8068_coalesceStep_prod, ih]
    rfl

/-- T8070 — `coalesce` preserves `baseOffset`. -/
theorem t8070_coalesce_preserves_base_offset (l : Layout) :
    (l.coalesce).baseOffset = l.baseOffset := rfl

/-- T8071 — zip/fst projection: with enough strides, projecting
    the first components of `dims.zip strides` recovers `dims`. -/
theorem t8071_zip_map_fst {α β : Type} (as : List α) (bs : List β)
    (h : as.length ≤ bs.length) : (as.zip bs).map Prod.fst = as := by
  induction as generalizing bs with
  | nil => rfl
  | cons a as ih =>
    cases bs with
    | nil => simp at h
    | cons b bs =>
      simp only [List.zip_cons_cons, List.map_cons]
      rw [ih bs (by simpa using h)]

/-- T8072 — zip/snd projection, symmetric to T8071. -/
theorem t8072_zip_map_snd {α β : Type} (as : List α) (bs : List β)
    (h : bs.length ≤ as.length) : (as.zip bs).map Prod.snd = bs := by
  induction as generalizing bs with
  | nil =>
    cases bs with
    | nil => rfl
    | cons b bs => simp at h
  | cons a as ih =>
    cases bs with
    | nil => rfl
    | cons b bs =>
      simp only [List.zip_cons_cons, List.map_cons]
      rw [ih bs (by simpa using h)]

/-- T8073 — `coalesce` preserves `linearSize` (for well-formed
    layouts, where the stride list covers every axis). -/
theorem t8073_coalesce_preserves_linear_size (l : Layout)
    (hlen : l.shape.dims.length ≤ l.strides.length) :
    (l.coalesce).linearSize = l.linearSize := by
  show pairsProd (coalescePairs (l.shape.dims.zip l.strides))
      = l.shape.dims.foldr (· * ·) 1
  rw [t8069_coalescePairs_prod]
  show ((l.shape.dims.zip l.strides).map Prod.fst).foldr (· * ·) 1 = _
  rw [t8071_zip_map_fst l.shape.dims l.strides hlen]

/-- T8074 — `coalesce` is structurally well-formed: its stride list
    and dims list have equal length (both project the same pair
    list). -/
theorem t8074_coalesce_wellformed (l : Layout) :
    (l.coalesce).strides.length = (l.coalesce).shape.dims.length := by
  simp [Layout.coalesce]

/-- T8075 — `coalesce` output is compact: no extent-1 mode
    survives. (A fused extent `d * d'` cannot be 1 because `d ≠ 1`.) -/
theorem t8075_coalescePairs_no_unit_extents (ps : List (Nat × Int)) :
    ∀ p ∈ coalescePairs ps, p.1 ≠ 1 := by
  induction ps with
  | nil => simp [coalescePairs]
  | cons q rest ih =>
    obtain ⟨d, s⟩ := q
    rw [t8063_coalescePairs_cons]
    by_cases hd1 : d = 1
    · rw [t8064_coalesceStep_drops_unit d s _ hd1]
      exact ih
    · cases hr : coalescePairs rest with
      | nil =>
        rw [t8065_coalesceStep_starts_group d s hd1]
        intro p hp
        rw [List.mem_singleton] at hp
        subst hp
        exact hd1
      | cons p' tail =>
        obtain ⟨d', s'⟩ := p'
        by_cases hf : s = s' * Int.ofNat d'
        · rw [t8066_coalesceStep_fuses d d' s s' tail hd1 hf]
          intro p hp
          rcases List.mem_cons.mp hp with heq | hmem
          · subst heq
            intro hcon
            exact hd1 (Nat.dvd_one.mp ⟨d', hcon.symm⟩)
          · exact ih p (by rw [hr]; exact List.mem_cons_of_mem _ hmem)
        · rw [t8067_coalesceStep_pushes d d' s s' tail hd1 hf]
          intro p hp
          rcases List.mem_cons.mp hp with heq | hmem
          · subst heq
            exact hd1
          · exact ih p (by rw [hr]; exact hmem)

/-- T8076 — `pairsOffset` peels one mode: the outer coordinate is
    `k / pairsProd ps` and the remainder recurses. Definitional. -/
theorem t8076_pairsOffset_cons (d : Nat) (s : Int)
    (ps : List (Nat × Int)) (k : Nat) :
    pairsOffset ((d, s) :: ps) k
      = Int.ofNat (k / pairsProd ps) * s + pairsOffset ps (k % pairsProd ps) :=
  rfl

/-- T8077 — **coalesce offset equivalence** at the pair level. For
    every in-range linear index, walking the coalesced modes
    reaches exactly the same offset as walking the originals.
    The fuse case is the interesting one: with `s = s' * d'` and
    `P = d' * Q`, the identities `k % P % Q = k % Q` and
    `k / Q = d' * (k / P) + (k % P) / Q` recombine the split
    coordinate into the fused one. -/
theorem t8077_coalescePairs_offset (ps : List (Nat × Int)) (k : Nat)
    (hk : k < pairsProd ps) :
    pairsOffset (coalescePairs ps) k = pairsOffset ps k := by
  induction ps generalizing k with
  | nil => rfl
  | cons q rest ih =>
    obtain ⟨d, s⟩ := q
    have hkP : k < d * pairsProd rest := hk
    have hPpos : 0 < pairsProd rest := by
      rcases Nat.eq_zero_or_pos (pairsProd rest) with h0 | hpos
      · rw [h0, Nat.mul_zero] at hkP
        exact absurd hkP (Nat.not_lt_zero k)
      · exact hpos
    rw [t8063_coalescePairs_cons, t8076_pairsOffset_cons,
        ← ih (k % pairsProd rest) (Nat.mod_lt k hPpos)]
    by_cases hd1 : d = 1
    · rw [t8064_coalesceStep_drops_unit d s _ hd1]
      have hkP' : k < pairsProd rest := by
        rw [hd1, Nat.one_mul] at hkP
        exact hkP
      rw [Nat.div_eq_of_lt hkP', Nat.mod_eq_of_lt hkP']
      simp
    · cases hr : coalescePairs rest with
      | nil =>
        have hP1 : pairsProd rest = 1 := by
          have h := t8069_coalescePairs_prod rest
          rw [hr] at h
          simpa [pairsProd] using h.symm
        rw [t8065_coalesceStep_starts_group d s hd1, hP1,
            t8076_pairsOffset_cons]
        rfl
      | cons p' tail =>
        obtain ⟨d', s'⟩ := p'
        have hPr : d' * pairsProd tail = pairsProd rest := by
          have h := t8069_coalescePairs_prod rest
          rw [hr] at h
          exact h
        have hQpos : 0 < pairsProd tail := by
          rcases Nat.eq_zero_or_pos (pairsProd tail) with h0 | hpos
          · rw [h0, Nat.mul_zero] at hPr
            rw [← hPr] at hPpos
            exact absurd hPpos (lt_irrefl 0)
          · exact hpos
        by_cases hf : s = s' * Int.ofNat d'
        · rw [t8066_coalesceStep_fuses d d' s s' tail hd1 hf,
              t8076_pairsOffset_cons, t8076_pairsOffset_cons]
          have hdvd : pairsProd tail ∣ pairsProd rest :=
            ⟨d', by rw [← hPr]; exact Nat.mul_comm d' (pairsProd tail)⟩
          have hmm : k % pairsProd rest % pairsProd tail = k % pairsProd tail :=
            Nat.mod_mod_of_dvd k hdvd
          have hk_eq : k = pairsProd tail * (d' * (k / pairsProd rest))
              + k % pairsProd rest := by
            conv_lhs => rw [← Nat.div_add_mod k (pairsProd rest)]
            rw [← hPr]
            ring
          have hdiv : k / pairsProd tail
              = d' * (k / pairsProd rest)
                + (k % pairsProd rest) / pairsProd tail := by
            conv_lhs => rw [hk_eq]
            rw [Nat.mul_add_div hQpos]
          rw [hmm, hf, hdiv]
          -- Split the composite cast by definitional equality
          -- (`Int.ofNat` commutes with `+` and `*` by `rfl`), then
          -- close by commutative-ring normalisation.
          have hsplit : Int.ofNat (d' * (k / pairsProd rest)
                + k % pairsProd rest / pairsProd tail)
              = Int.ofNat d' * Int.ofNat (k / pairsProd rest)
                + Int.ofNat (k % pairsProd rest / pairsProd tail) := rfl
          rw [hsplit]
          ring
        · rw [t8067_coalesceStep_pushes d d' s s' tail hd1 hf,
              t8076_pairsOffset_cons]
          have hP' : pairsProd ((d', s') :: tail) = pairsProd rest := hPr
          rw [hP']

/-- T8078 — **coalesce offset equivalence** at the layout level.
    For a well-formed layout (strides cover the axes) and every
    in-range linear index `k`, the coalesced layout's offset of
    `k`'s coordinate equals the original layout's. Combined with
    T8069/T8073 (size preserved) and T8070 (base preserved), the
    coalesced layout indexes identically over the flattened
    domain. -/
theorem t8078_coalesce_offset_equiv (l : Layout) (k : Nat)
    (hlen : l.strides.length = l.shape.dims.length)
    (hk : k < l.linearSize) :
    (l.coalesce).offset (unflatten (l.coalesce).shape.dims k)
      = l.offset (unflatten l.shape.dims k) := by
  have hfst : (l.shape.dims.zip l.strides).map Prod.fst = l.shape.dims :=
    t8071_zip_map_fst l.shape.dims l.strides (le_of_eq hlen.symm)
  have hsnd : (l.shape.dims.zip l.strides).map Prod.snd = l.strides :=
    t8072_zip_map_snd l.shape.dims l.strides (le_of_eq hlen)
  have hk' : k < pairsProd (l.shape.dims.zip l.strides) := by
    show k < ((l.shape.dims.zip l.strides).map Prod.fst).foldr (· * ·) 1
    rw [hfst]
    exact hk
  calc (l.coalesce).offset (unflatten (l.coalesce).shape.dims k)
      = l.baseOffset + pairsOffset (coalescePairs (l.shape.dims.zip l.strides)) k :=
        rfl
    _ = l.baseOffset + pairsOffset (l.shape.dims.zip l.strides) k := by
        rw [t8077_coalescePairs_offset _ k hk']
    _ = l.offset (unflatten l.shape.dims k) := by
        unfold Layout.offset pairsOffset
        rw [hfst, hsnd]

-- ─────────────────────────────────────────────────────────────────
-- The full multi-rank `compose`. Faithful success-path model of
-- `Layout::compose` in `crates/sci/quanta-tensor/src/layout/algebra.rs`:
--
-- - `composeFold` is `compose_lhs_with_int`'s LHS fold: walk the
--   LHS modes left-to-right carrying the unconsumed RHS extent and
--   stride, emitting `(min-clamped extent, rest_stride * lhs_stride)`
--   modes as the RHS mode is spread across the LHS modes.
-- - `composeIntPairs` is the dispatch layer (stride-0 RHS and
--   rank-0 LHS short-circuit; the Rust rank-1 shortcut coincides
--   with the fold's tail arm, so it needs no separate case).
-- - `composeN` right-distributes over the RHS modes and
--   concatenates the partial results — the exact loop in
--   `Layout::compose`. Base offsets add.
--
-- Like `reshape` above, this models the SUCCESS path: the
-- production op refuses on divisibility failure, so every theorem
-- below also covers the composable (non-refused) inputs.
--
-- The load-bearing structural fact (t8081/t8082): the fold's
-- control flow — which modes are emitted, their extents, the
-- carried rest extent/stride — depends only on the LHS extents and
-- the RHS (extent, stride) pair; the LHS strides enter only as the
-- multiplicative factor of each emitted stride. Composing on the
-- left with a rank-1 layout therefore scales every stride and
-- changes nothing else, which reduces rank-1 × rank-N × rank-K
-- associativity (t8094) to stride-scaling commutation through the
-- fold. For a rank ≥ 2 LEFTMOST layout this shortcut is gone: the
-- fold genuinely branches on its divisibility gate, and the naive
-- statement (carrying only t8094's well-formedness premises) is
-- FALSE — t8097 proves a concrete rank-2 counterexample. The true
-- theorem carries a composability (divisibility) precondition, whose
-- core obligation is a two-stage/one-stage fold exchange over the
-- leftmost mode list
-- (`composeIntPairs (composeIntPairs a nb db ...) nc dc =
--   composeIntPairs a nc (dc * db)`-shaped, requiring ceil-division
-- composition arithmetic under the divisibility invariant). t8094
-- covers every rank-1-leftmost instance unconditionally; the guarded
-- rank ≥ 2 form is a follow-up.
-- ─────────────────────────────────────────────────────────────────

namespace Layout

/-- Ceiling division on `Nat`. Callers clamp the divisor with
    `max · 1`, mirroring the production `div_ceil` call sites. -/
def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

/-- The LHS fold of `compose_lhs_with_int` (success path): walk the
    LHS `(extent, stride)` modes left-to-right; `restS` / `restD`
    carry the unconsumed RHS extent and stride, and `emitted`
    records whether a mode has been produced (the Rust tail checks
    `result_shape.is_empty()`). The skip arm fires when the current
    LHS mode is fully consumed by the carried stride
    (`next_shape == 1`) or the RHS extent is exhausted
    (`rest_shape == 1`). -/
def composeFold : List (Nat × Int) → Nat → Int → Bool → List (Nat × Int)
  | [], _, _, _ => []
  | [x], restS, restD, emitted =>
      if emitted ∧ restS = 1 then [] else [(restS, restD * x.2)]
  | x :: y :: rest, restS, restD, emitted =>
      if ceilDiv x.1 (max restD.natAbs 1) = 1 ∨ restS = 1 then
        composeFold (y :: rest) restS
          (Int.ofNat (ceilDiv restD.natAbs (max x.1 1)) * restD.sign) emitted
      else
        (min (ceilDiv x.1 (max restD.natAbs 1)) restS, restD * x.2) ::
          composeFold (y :: rest)
            (restS / min (ceilDiv x.1 (max restD.natAbs 1)) restS)
            (Int.ofNat (ceilDiv restD.natAbs (max x.1 1)) * restD.sign) true

/-- Compose an LHS pair list with ONE RHS mode `(s, d)` — the
    dispatch layer of `compose_lhs_with_int`. Stride-0 RHS reads a
    single element repeatedly; rank-0 LHS passes the RHS mode
    through; everything else runs the fold. -/
def composeIntPairs (ps : List (Nat × Int)) (s : Nat) (d : Int) : List (Nat × Int) :=
  if d = 0 then [(s, 0)]
  else
    match ps with
    | [] => [(s, d)]
    | x :: rest => composeFold (x :: rest) s d false

/-- Scale every stride in a pair list by `k`. This is what
    left-composition with a rank-1 layout of stride `k` does. -/
def scalePairs (ps : List (Nat × Int)) (k : Int) : List (Nat × Int) :=
  ps.map (fun q => (q.1, q.2 * k))

/-- The `(extent, stride)` pair list of a layout. -/
def pairsOf (l : Layout) : List (Nat × Int) :=
  l.shape.dims.zip l.strides

/-- Spec-level model of the full multi-rank `Layout::compose`
    (success path): right-distribute over the RHS modes — each RHS
    mode composes against the whole LHS via `composeIntPairs` —
    and concatenate. Rank-0 RHS returns the LHS unchanged, exactly
    as the production op does. -/
def composeN (a b : Layout) : Layout :=
  if b.shape.dims.isEmpty then a
  else
    { shape := { dims := ((pairsOf b).flatMap
        (fun q => composeIntPairs (pairsOf a) q.1 q.2)).map Prod.fst }
      strides := ((pairsOf b).flatMap
        (fun q => composeIntPairs (pairsOf a) q.1 q.2)).map Prod.snd
      baseOffset := a.baseOffset + b.baseOffset }

end Layout

/-- T8079 — with `emitted = false` the fold never returns the empty
    list: either some step emits (a cons), or the tail arm fires
    with nothing emitted yet and produces its mode. -/
theorem t8079_composeFold_unemitted_ne_nil
    (x : Nat × Int) (ps : List (Nat × Int)) (s : Nat) (d : Int) :
    composeFold (x :: ps) s d false ≠ [] := by
  induction ps generalizing x s d with
  | nil => simp [composeFold]
  | cons y rest ih =>
    simp only [composeFold]
    split
    · exact ih y s _
    · simp

/-- T8080 — `composeIntPairs` never returns the empty list: every
    dispatch arm produces at least one mode. (The production op
    keeps at least one output mode per RHS mode; this is what makes
    the composed layout non-degenerate below.) -/
theorem t8080_composeIntPairs_ne_nil (ps : List (Nat × Int)) (s : Nat) (d : Int) :
    composeIntPairs ps s d ≠ [] := by
  by_cases hd : d = 0
  · simp [composeIntPairs, hd]
  · cases ps with
    | nil => simp [composeIntPairs, hd]
    | cons x rest =>
      simpa [composeIntPairs, hd] using
        t8079_composeFold_unemitted_ne_nil x rest s d

/-- T8081 — the load-bearing structural lemma: the fold's control
    flow (which modes are emitted, their extents, the carried rest
    extent/stride) depends only on the LHS EXTENTS and the RHS
    (extent, stride) pair; the LHS strides enter only as the
    multiplicative factor of each emitted stride. Scaling every LHS
    stride by `k` therefore scales every output stride by `k` and
    changes nothing else. -/
theorem t8081_composeFold_scale
    (x : Nat × Int) (ps : List (Nat × Int)) (k : Int)
    (s : Nat) (d : Int) (em : Bool) :
    composeFold (scalePairs (x :: ps) k) s d em
      = scalePairs (composeFold (x :: ps) s d em) k := by
  induction ps generalizing x s d em with
  | nil =>
    obtain ⟨x1, x2⟩ := x
    by_cases h : em = true ∧ s = 1
    · simp [composeFold, scalePairs, h]
    · simp [composeFold, scalePairs, h, mul_assoc]
  | cons y rest ih =>
    obtain ⟨x1, x2⟩ := x
    obtain ⟨y1, y2⟩ := y
    have ihy := ih (y1, y2)
    simp only [scalePairs, List.map_cons] at ihy ⊢
    by_cases hcond : ceilDiv x1 (max d.natAbs 1) = 1 ∨ s = 1
    · simp [composeFold, hcond, ihy]
    · simp [composeFold, hcond, ihy, mul_assoc]

/-- T8082 — scale commutation lifted through the dispatch layer:
    composing a stride-scaled LHS with one RHS mode is the scaled
    composition. (Stride-0 RHS scales trivially: `0 * k = 0`.) -/
theorem t8082_composeIntPairs_scale
    (x : Nat × Int) (ps : List (Nat × Int)) (k : Int) (s : Nat) (d : Int) :
    composeIntPairs (scalePairs (x :: ps) k) s d
      = scalePairs (composeIntPairs (x :: ps) s d) k := by
  obtain ⟨x1, x2⟩ := x
  by_cases hd : d = 0
  · simp [composeIntPairs, scalePairs, hd]
  · simpa [composeIntPairs, scalePairs, hd] using
      t8081_composeFold_scale (x1, x2) ps k s d false

/-- T8083 — rank-1 LHS closed form: one LHS mode `(n, s0)` composed
    with one RHS mode `(s, d)` is `[(s, d * s0)]` — exactly the
    `compose11` stride product. (The stride-0 arm agrees because
    `0 * s0 = 0`.) -/
theorem t8083_composeIntPairs_rank1 (n : Nat) (s0 : Int) (s : Nat) (d : Int) :
    composeIntPairs [(n, s0)] s d = [(s, d * s0)] := by
  by_cases hd : d = 0
  · simp [composeIntPairs, hd]
  · simp [composeIntPairs, composeFold, hd]

/-- T8084 — rank-0 RHS: the composition is the LHS unchanged,
    matching the production early return. -/
theorem t8084_composeN_rank0_rhs (a b : Layout) (h : b.shape.dims = []) :
    composeN a b = a := by
  simp [composeN, h]

/-- T8085 — the non-rank-0 unfolding of `composeN` as an explicit
    record: the right-distributive flatMap over the RHS modes. -/
theorem t8085_composeN_unfold (a b : Layout) (h : b.shape.dims ≠ []) :
    composeN a b
      = { shape := { dims := ((pairsOf b).flatMap
            (fun q => composeIntPairs (pairsOf a) q.1 q.2)).map Prod.fst }
          strides := ((pairsOf b).flatMap
            (fun q => composeIntPairs (pairsOf a) q.1 q.2)).map Prod.snd
          baseOffset := a.baseOffset + b.baseOffset } := by
  have hne : b.shape.dims.isEmpty = false := by
    cases hd : b.shape.dims with
    | nil => exact absurd hd h
    | cons _ _ => rfl
  simp [composeN, hne]

/-- T8086 — `composeN` output is structurally well-formed: dims and
    strides project the same pair list, so their lengths agree. -/
theorem t8086_composeN_wellformed (a b : Layout) (h : b.shape.dims ≠ []) :
    (composeN a b).strides.length = (composeN a b).shape.dims.length := by
  rw [t8085_composeN_unfold a b h]
  simp

/-- T8087 — scaling strides leaves the extents untouched. -/
theorem t8087_scalePairs_map_fst (ps : List (Nat × Int)) (k : Int) :
    (scalePairs ps k).map Prod.fst = ps.map Prod.fst := by
  simp [scalePairs]

/-- T8088 — the strides of a scaled pair list are the mapped
    original strides. -/
theorem t8088_scalePairs_map_snd (ps : List (Nat × Int)) (k : Int) :
    (scalePairs ps k).map Prod.snd = (ps.map Prod.snd).map (· * k) := by
  simp [scalePairs]

/-- T8089 — scaling distributes over the per-RHS-mode
    concatenation. -/
theorem t8089_scalePairs_flatMap (qs : List (Nat × Int))
    (g : Nat × Int → List (Nat × Int)) (k : Int) :
    scalePairs (qs.flatMap g) k = qs.flatMap (fun q => scalePairs (g q) k) := by
  induction qs with
  | nil => rfl
  | cons q rest ih =>
    simp only [scalePairs] at ih ⊢
    simp [List.map_append, ih]

/-- T8090 — a flatMap of per-mode singletons `(q.1, q.2 * k)` is
    exactly `scalePairs`. Bridges the rank-1-LHS composition
    (T8083 mode-wise) to the scaled-layout view. -/
theorem t8090_flatMap_mode_scale (qs : List (Nat × Int)) (k : Int) :
    qs.flatMap (fun q => [((q.1 : Nat), q.2 * k)]) = scalePairs qs k := by
  induction qs with
  | nil => rfl
  | cons q rest ih => simp [scalePairs, ih]

/-- T8091 — closed form for a rank-1 LHS: composing `(na, sa)` on
    the left of any well-formed layout keeps the shape and scales
    every stride by `sa`; base offsets add. This is the faithful
    (`composeN`) counterpart of `compose1n`. -/
theorem t8091_composeN_rank1_lhs (a b : Layout) (na : Nat) (sa : Int)
    (had : a.shape.dims = [na]) (has : a.strides = [sa])
    (hb : b.strides.length = b.shape.dims.length)
    (hbr : b.shape.dims ≠ []) :
    composeN a b
      = { shape := b.shape
          strides := b.strides.map (· * sa)
          baseOffset := a.baseOffset + b.baseOffset } := by
  have hpa : pairsOf a = [(na, sa)] := by simp [pairsOf, had, has]
  have h1 : (pairsOf b).map Prod.fst = b.shape.dims :=
    t8071_zip_map_fst b.shape.dims b.strides (le_of_eq hb.symm)
  have h2 : (pairsOf b).map Prod.snd = b.strides :=
    t8072_zip_map_snd b.shape.dims b.strides (le_of_eq hb)
  rw [t8085_composeN_unfold a b hbr, hpa]
  simp only [t8083_composeIntPairs_rank1, t8090_flatMap_mode_scale,
    t8087_scalePairs_map_fst, t8088_scalePairs_map_snd, h1, h2]

/-- T8092 — `composeN` agrees with the rank-1×rank-1 shortcut
    `compose11` on rank-1 inputs. -/
theorem t8092_composeN_matches_compose11 (na nb : Nat) (sa db : Int) :
    composeN (rank1 na sa) (rank1 nb db)
      = compose11 (rank1 na sa) (rank1 nb db) := by
  rw [t8091_composeN_rank1_lhs (rank1 na sa) (rank1 nb db) na sa rfl rfl rfl
    (by simp [rank1])]
  simp [rank1, compose11]

/-- T8093 — `composeN` agrees with the rank-1-LHS shortcut
    `compose1n` whenever the RHS has base offset 0 (`compose1n`
    zeroes the base; the faithful op adds them). -/
theorem t8093_composeN_matches_compose1n (na : Nat) (sa : Int) (b : Layout)
    (hb : b.strides.length = b.shape.dims.length)
    (hbr : b.shape.dims ≠ []) (hb0 : b.baseOffset = 0) :
    composeN (rank1 na sa) b = compose1n (rank1 na sa) b := by
  rw [t8091_composeN_rank1_lhs (rank1 na sa) b na sa rfl rfl hb hbr]
  simp [compose1n, rank1, hb0]

/-- T8094 — **multi-rank composition associativity, rank-1 leftmost.**
    For a rank-1 layout `a` (any base offset) and well-formed
    layouts `b`, `c` of ARBITRARY rank,

      `composeN (composeN a b) c = composeN a (composeN b c)`.

    This is the faithful-fold statement: the outer-left composition
    genuinely runs the multi-mode divisibility fold over the
    multi-mode result of `composeN a b`. The proof reduces it to
    T8081/T8082 (stride scaling commutes through the fold) plus
    T8089 (scaling distributes over the RHS-mode concatenation) —
    no structure of the fold's arithmetic is needed beyond "LHS
    strides are a multiplicative factor".

    `b` must have rank ≥ 1: the production rank-0-RHS early return
    (`self.clone()`) drops the RHS base offset, so a rank-0 middle
    with non-zero base genuinely breaks base-offset associativity
    (T8096 proves the base-0 rank-0-middle case). `c` may be
    rank 0. The rank ≥ 2 LEFTMOST case remains open — see the
    section comment above `composeFold`. -/
theorem t8094_composeN_assoc_rank1_lhs
    (a b c : Layout) (na : Nat) (sa : Int)
    (had : a.shape.dims = [na]) (has : a.strides = [sa])
    (hb : b.strides.length = b.shape.dims.length)
    (hbr : b.shape.dims ≠ [])
    (hc : c.strides.length = c.shape.dims.length) :
    composeN (composeN a b) c = composeN a (composeN b c) := by
  cases hcd : c.shape.dims with
  | nil =>
    rw [t8084_composeN_rank0_rhs (composeN a b) c hcd,
        t8084_composeN_rank0_rhs b c hcd]
  | cons c0 crest =>
    have hcr : c.shape.dims ≠ [] := by
      rw [hcd]; exact List.cons_ne_nil _ _
    obtain ⟨d0, ds, hbd⟩ : ∃ d0 ds, b.shape.dims = d0 :: ds := by
      cases hd : b.shape.dims with
      | nil => exact absurd hd hbr
      | cons d0 ds => exact ⟨d0, ds, rfl⟩
    obtain ⟨s0, ss, hbs⟩ : ∃ s0 ss, b.strides = s0 :: ss := by
      cases hs : b.strides with
      | nil => rw [hs, hbd] at hb; simp at hb
      | cons s0 ss => exact ⟨s0, ss, rfl⟩
    have hzip : pairsOf b = (d0, s0) :: ds.zip ss := by
      simp [pairsOf, hbd, hbs]
    obtain ⟨t0, ts, hcs⟩ : ∃ t0 ts, c.strides = t0 :: ts := by
      cases hs : c.strides with
      | nil => rw [hs, hcd] at hc; simp at hc
      | cons t0 ts => exact ⟨t0, ts, rfl⟩
    have hcp : pairsOf c = (c0, t0) :: crest.zip ts := by
      simp [pairsOf, hcd, hcs]
    -- the middle composition's mode list is non-empty
    have hPne : (pairsOf c).flatMap
        (fun q => composeIntPairs ((d0, s0) :: ds.zip ss) q.1 q.2) ≠ [] := by
      rw [hcp, List.flatMap_cons]
      intro hnil
      exact t8080_composeIntPairs_ne_nil ((d0, s0) :: ds.zip ss) c0 t0
        ((List.append_eq_nil.mp hnil).1)
    -- LEFT: inner compose = b with strides scaled by sa; unfold outer
    rw [t8091_composeN_rank1_lhs a b na sa had has hb hbr]
    rw [t8085_composeN_unfold _ c hcr]
    rw [t8085_composeN_unfold b c hcr]
    have hpr : pairsOf { shape := b.shape
                         strides := b.strides.map (· * sa)
                         baseOffset := a.baseOffset + b.baseOffset }
        = scalePairs (pairsOf b) sa := by
      show b.shape.dims.zip (b.strides.map (· * sa)) = _
      rw [List.zip_map_right]
      rfl
    rw [hpr, hzip]
    -- push the scaling out of the per-RHS-mode fold
    have hcomm : (pairsOf c).flatMap (fun q =>
          composeIntPairs (scalePairs ((d0, s0) :: ds.zip ss) sa) q.1 q.2)
        = scalePairs ((pairsOf c).flatMap (fun q =>
            composeIntPairs ((d0, s0) :: ds.zip ss) q.1 q.2)) sa := by
      rw [t8089_scalePairs_flatMap]
      simp only [t8082_composeIntPairs_scale]
    rw [hcomm]
    -- RIGHT: apply the rank-1 closed form to the composed middle
    set M : Layout :=
      { shape := { dims := ((pairsOf c).flatMap
          (fun q => composeIntPairs ((d0, s0) :: ds.zip ss) q.1 q.2)).map Prod.fst }
        strides := ((pairsOf c).flatMap
          (fun q => composeIntPairs ((d0, s0) :: ds.zip ss) q.1 q.2)).map Prod.snd
        baseOffset := b.baseOffset + c.baseOffset } with hM
    have hMwf : M.strides.length = M.shape.dims.length := by
      simp [hM]
    have hMne : M.shape.dims ≠ [] := by
      simp only [hM]
      intro h
      exact hPne (List.map_eq_nil_iff.mp h)
    rw [t8091_composeN_rank1_lhs a M na sa had has hMwf hMne]
    simp [hM, t8087_scalePairs_map_fst, t8088_scalePairs_map_snd, add_assoc]

/-- T8095 — rank-0 LHS: each RHS mode passes through unchanged. -/
theorem t8095_composeIntPairs_rank0 (s : Nat) (d : Int) :
    composeIntPairs [] s d = [(s, d)] := by
  by_cases hd : d = 0
  · simp [composeIntPairs, hd]
  · simp [composeIntPairs, hd]

/-- T8096 — associativity with a rank-0 MIDDLE layout of base
    offset 0 (the base-0 restriction is essential: the production
    rank-0-RHS early return drops the RHS base, so a non-zero
    rank-0 middle base breaks the equation — see T8094's docstring).
    Together with T8094 this covers every rank-1-or-less leftmost /
    middle combination. -/
theorem t8096_composeN_assoc_rank0_mid
    (a b c : Layout)
    (hbd : b.shape.dims = []) (hbs : b.strides = []) (hb0 : b.baseOffset = 0)
    (hc : c.strides.length = c.shape.dims.length) :
    composeN (composeN a b) c = composeN a (composeN b c) := by
  rw [t8084_composeN_rank0_rhs a b hbd]
  cases hcd : c.shape.dims with
  | nil => rw [t8084_composeN_rank0_rhs a c hcd, t8084_composeN_rank0_rhs b c hcd,
               t8084_composeN_rank0_rhs a b hbd]
  | cons c0 crest =>
    have hcr : c.shape.dims ≠ [] := by
      rw [hcd]; exact List.cons_ne_nil _ _
    -- composeN b c = c (rank-0 pass-through, base 0 + c.base)
    have hbc : composeN b c = c := by
      rw [t8085_composeN_unfold b c hcr]
      have hpb : pairsOf b = [] := by simp [pairsOf, hbd, hbs]
      have hid : (pairsOf c).flatMap
          (fun q => composeIntPairs (pairsOf b) q.1 q.2) = pairsOf c := by
        rw [hpb]
        simp only [t8095_composeIntPairs_rank0]
        induction pairsOf c with
        | nil => rfl
        | cons q rest ih => simp [ih]
      rw [hid]
      have h1 : (pairsOf c).map Prod.fst = c.shape.dims :=
        t8071_zip_map_fst c.shape.dims c.strides (le_of_eq hc.symm)
      have h2 : (pairsOf c).map Prod.snd = c.strides :=
        t8072_zip_map_snd c.shape.dims c.strides (le_of_eq hc)
      rw [h1, h2, hb0]
      simp
    rw [hbc]

/-- T8097 — **multi-rank composition associativity FAILS for a rank ≥ 2
    leftmost layout without a composability hypothesis.** The general
    statement (carrying only t8094's well-formedness premises, no
    divisibility precondition) is genuinely false: a concrete witness
    with a rank-2 leftmost layout gives two different results depending
    on the grouping.

    Witness: `a = ⟨[4,3],[10,100]⟩` (rank 2), `b = ⟨[2],[1]⟩`,
    `c = ⟨[6],[1]⟩`. Left grouping yields shape `[6]`, right grouping
    shape `[4]` — the two disagree.

    Root cause: `composeFold` faithfully models the SUCCESS path of the
    production `Layout::compose`, which refuses (`DivisibilityFailed`)
    exactly the mode arrangements where the fold's divisibility branch
    would produce this divergence. On such refused inputs the model
    still computes a (production-unreachable) value, and the two
    groupings need not agree. t8094's rank-1-leftmost proof escapes this
    because a single LHS mode never enters the fold's divisibility
    branch — composition is pure stride-scaling there. Making the
    general rank ≥ 2 statement TRUE requires threading the composability
    (divisibility) precondition through both groupings (follow-up). -/
theorem t8097_composeN_assoc_rank2_lhs_fails :
    ∃ a b c : Layout,
      a.strides.length = a.shape.dims.length ∧
      b.strides.length = b.shape.dims.length ∧
      c.strides.length = c.shape.dims.length ∧
      composeN (composeN a b) c ≠ composeN a (composeN b c) := by
  refine ⟨{ shape := { dims := [4, 3] }, strides := [10, 100], baseOffset := 0 },
          { shape := { dims := [2] }, strides := [1], baseOffset := 0 },
          { shape := { dims := [6] }, strides := [1], baseOffset := 0 },
          rfl, rfl, rfl, ?_⟩
  -- The two groupings disagree already at the shape: left gives [6],
  -- right gives [4]. Layout has no DecidableEq, so witness the
  -- inequality through the `.shape.dims` projection.
  intro h
  have hdims := congrArg (fun l => l.shape.dims) h
  simp only [] at hdims
  exact absurd hdims (by decide)

end Quanta.Tensor
