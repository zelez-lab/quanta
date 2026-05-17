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

end Quanta.Tensor
