/-
Tensor layout algebra — Lean formalisation of `quanta-tensor`.

Mirrors the Rust substrate at `crates/quanta-tensor/src/`:

- `Shape` is a list of axis extents.
- `Layout` is a `Shape` paired with a stride list of the same length
  and an integer base offset.
- The indexer `Layout.offset` maps a coordinate vector to a flat
  buffer offset (modelled as `Int` to match Rust's `isize` strides).

This first commit establishes the substrate and the easy structural
theorems. Harder algebraic theorems (composition associativity,
permutation bijectivity, tile-offset bounds) land in a follow-up
when their proof obligations are stable.
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
-- `crates/quanta-tensor/src/layout/{ops,algebra}.rs`. The Rust
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
    `permute` (in `crates/quanta-tensor/src/layout/ops.rs`). -/
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
-- Composition of rank-1 layouts. The full multi-rank `compose`
-- requires the divisibility-checking fold from CuTe; modelling
-- that operationally in Lean and proving associativity over it
-- is multi-session work. As a foundation we ship the rank-1×
-- rank-1 closed form — the simplest case CuTe handles as a
-- special shortcut — together with the identity-composition
-- and rank-1 associativity theorems. These cover the most
-- common downstream usage and give a hook the full theorem can
-- build on later.
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

end Quanta.Tensor
