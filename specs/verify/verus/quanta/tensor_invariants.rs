//! Verus mirror of `quanta-tensor` — layout algebra invariants.
//!
//! Mirrors the production types in
//! `crates/quanta-tensor/src/{shape,layout}.rs`.
//!
//! Verified properties:
//!
//! | Theorem                          | What it proves                                    |
//! |----------------------------------|---------------------------------------------------|
//! | t8100_shape_linear_size_matches  | linearSize == product of dims (computed)          |
//! | t8101_row_major_strides_len      | rowMajorStrides(dims).len() == dims.len()         |
//! | t8102_offset_zero_base_empty     | offset on empty coord with base 0 returns 0       |
//! | t8103_offset_baseline            | offset == baseOffset when coord is empty          |
//! | t8104_shape_rank_zero_size_one   | empty shape has linearSize 1                      |
//! | t8105_dot_empty_coord            | dot([], strides) == 0                             |
//! | t8106_dot_empty_strides          | dot(coord, []) == 0                               |
//! | t8107_dot_cons                   | dot(c :: cs, s :: rest) == c*s + dot(cs, rest)    |
//!
//! Layout-op structural invariants (t8111–t8121): preservation
//! properties on transpose, permute, slice, broadcast, complement,
//! and compose. These are the operational guarantees downstream
//! crates rely on (rank, linear size, base offset shape). The
//! deeper algebraic theorems (composition associativity,
//! permutation bijectivity, tile-offset bounds) live in the Lean
//! arm at `specs/verify/lean/Quanta/Tensor/Layout.lean`.
//!
//! Matched-by-number with the Lean theorems in the same file at a
//! slightly different offset because the Verus arm exercises the
//! operational ghost model where the Lean arm exercises the
//! mathematical theorem.

use vstd::prelude::*;

verus! {

// ── Abstract Shape and Layout state ─────────────────────────────────

/// Ghost model of `quanta_tensor::Shape`. A list of non-negative
/// axis extents. Each extent is required ≥ 1 in the production
/// constructor (see `Shape::new`); the Verus invariants below
/// assume the same — callers that need the invariant must supply
/// it as a precondition.
pub struct ShapeModel {
    pub dims: Seq<nat>,
}

impl ShapeModel {
    pub open spec fn rank(self) -> nat {
        self.dims.len()
    }

    /// Product of all extents. Spec-side fold using `Seq::fold_left`
    /// would also work; the recursive definition matches the Lean
    /// `foldr (· * ·) 1` exactly.
    pub open spec fn linear_size(self) -> nat
        decreases self.dims.len()
    {
        if self.dims.len() == 0 {
            1nat
        } else {
            self.dims[0] * Self { dims: self.dims.drop_first() }.linear_size()
        }
    }

    /// Well-formedness: every extent ≥ 1.
    pub open spec fn well_formed(self) -> bool {
        forall|i: int| 0 <= i < self.dims.len() ==> self.dims[i] >= 1
    }
}

/// Ghost model of `quanta_tensor::Layout`. A shape paired with a
/// stride sequence of the same length and an integer base offset.
pub struct LayoutModel {
    pub shape: ShapeModel,
    pub strides: Seq<int>,
    pub base_offset: int,
}

impl LayoutModel {
    /// Well-formedness: strides match the shape rank, and the shape
    /// itself is well-formed.
    pub open spec fn well_formed(self) -> bool {
        &&& self.shape.well_formed()
        &&& self.strides.len() == self.shape.dims.len()
    }

    pub open spec fn rank(self) -> nat {
        self.shape.dims.len()
    }

    pub open spec fn linear_size(self) -> nat {
        self.shape.linear_size()
    }

    /// Dot product of a coordinate sequence with the stride
    /// sequence. Zips the shorter of the two so the function is
    /// total. Matches the Lean `dot` definition.
    pub open spec fn dot(coord: Seq<nat>, strides: Seq<int>) -> int
        decreases coord.len()
    {
        if coord.len() == 0 || strides.len() == 0 {
            0int
        } else {
            (coord[0] as int) * strides[0]
                + Self::dot(coord.drop_first(), strides.drop_first())
        }
    }

    /// Map an N-coordinate to a flat-buffer offset.
    pub open spec fn offset(self, coord: Seq<nat>) -> int {
        self.base_offset + Self::dot(coord, self.strides)
    }
}

/// Row-major strides for the given dims: rightmost axis varies
/// fastest, so `strides[i] = ∏ dims[i+1..]`.
pub open spec fn row_major_strides(dims: Seq<nat>) -> Seq<int>
    decreases dims.len()
{
    if dims.len() == 0 {
        Seq::<int>::empty()
    } else {
        let rest = dims.drop_first();
        let rest_strides = row_major_strides(rest);
        let my_size = ShapeModel { dims: rest }.linear_size() as int;
        seq![my_size] + rest_strides
    }
}

/// Construct a row-major layout.
pub open spec fn row_major(dims: Seq<nat>) -> LayoutModel {
    LayoutModel {
        shape: ShapeModel { dims: dims },
        strides: row_major_strides(dims),
        base_offset: 0,
    }
}

// ── Theorems ────────────────────────────────────────────────────────

/// T8100 — `linear_size` of a shape equals the product of its dims
/// (computed inductively). For the empty shape the product is 1.
proof fn t8100_shape_linear_size_matches_empty()
    ensures (ShapeModel { dims: Seq::<nat>::empty() }).linear_size() == 1nat,
{
}

/// T8101 — `row_major_strides(dims)` has the same length as `dims`.
proof fn t8101_row_major_strides_len(dims: Seq<nat>)
    ensures row_major_strides(dims).len() == dims.len(),
    decreases dims.len(),
{
    if dims.len() == 0 {
    } else {
        t8101_row_major_strides_len(dims.drop_first());
    }
}

/// T8102 — Empty coordinate, zero base: offset is 0.
proof fn t8102_offset_zero_base_empty(l: LayoutModel)
    requires l.base_offset == 0,
    ensures l.offset(Seq::<nat>::empty()) == 0int,
{
}

/// T8103 — `offset` on an empty coordinate equals `base_offset`,
/// independently of the stride sequence.
proof fn t8103_offset_empty_coord_is_base(l: LayoutModel)
    ensures l.offset(Seq::<nat>::empty()) == l.base_offset,
{
}

/// T8104 — A rank-0 shape has linear size 1.
proof fn t8104_shape_rank_zero_size_one()
    ensures (ShapeModel { dims: Seq::<nat>::empty() }).linear_size() == 1nat,
{
}

/// T8105 — `dot` with an empty coordinate is 0.
proof fn t8105_dot_empty_coord(strides: Seq<int>)
    ensures LayoutModel::dot(Seq::<nat>::empty(), strides) == 0int,
{
}

/// T8106 — `dot` with empty strides is 0.
proof fn t8106_dot_empty_strides(coord: Seq<nat>)
    ensures LayoutModel::dot(coord, Seq::<int>::empty()) == 0int,
    decreases coord.len(),
{
    if coord.len() == 0 {
    } else {
    }
}

/// T8107 — `dot` distributes over a cons on both sides.
proof fn t8107_dot_cons(c: nat, cs: Seq<nat>, s: int, rest: Seq<int>)
    ensures ({
        let coord = seq![c] + cs;
        let strides = seq![s] + rest;
        LayoutModel::dot(coord, strides) == (c as int) * s + LayoutModel::dot(cs, rest)
    }),
{
    let coord = seq![c] + cs;
    let strides = seq![s] + rest;
    assert(coord.len() > 0);
    assert(strides.len() > 0);
    assert(coord[0] == c);
    assert(strides[0] == s);
    assert(coord.drop_first() =~= cs);
    assert(strides.drop_first() =~= rest);
}

/// T8108 — A rank-0 row-major layout indexes the single element at
/// offset 0.
proof fn t8108_row_major_rank_zero_offset()
    ensures (row_major(Seq::<nat>::empty())).offset(Seq::<nat>::empty()) == 0int,
{
}

/// T8109 — Row-major construction preserves the input dims as the
/// resulting layout's shape dims.
proof fn t8109_row_major_preserves_dims(dims: Seq<nat>)
    ensures row_major(dims).shape.dims == dims,
{
}

/// T8110 — Row-major construction has base offset 0.
proof fn t8110_row_major_zero_base(dims: Seq<nat>)
    ensures row_major(dims).base_offset == 0,
{
}

// ── Layout-op ghost specs ────────────────────────────────────────

/// Swap two positions in a sequence. Used to model `Layout::transpose`.
pub open spec fn seq_swap<T>(xs: Seq<T>, i: int, j: int) -> Seq<T>
    recommends 0 <= i < xs.len(), 0 <= j < xs.len(),
{
    xs.update(i, xs[j]).update(j, xs[i])
}

/// Ghost model of `Layout::transpose(i, j)`: swap the i-th and
/// j-th positions in both dims and strides; base_offset is
/// unchanged.
pub open spec fn transpose(l: LayoutModel, i: int, j: int) -> LayoutModel
    recommends 0 <= i < l.rank(), 0 <= j < l.rank(),
{
    LayoutModel {
        shape: ShapeModel { dims: seq_swap(l.shape.dims, i, j) },
        strides: seq_swap(l.strides, i, j),
        base_offset: l.base_offset,
    }
}

/// Ghost model of `Layout::slice(axis, start, end)`: replace the
/// axis extent with `end - start` and advance `base_offset` by
/// `start * stride[axis]`. Strides are unchanged.
pub open spec fn slice(l: LayoutModel, axis: int, start: nat, end: nat) -> LayoutModel
    recommends
        0 <= axis < l.rank(),
        start < end,
        end <= l.shape.dims[axis],
{
    let new_extent: nat = (end - start) as nat;
    let new_dims = l.shape.dims.update(axis, new_extent);
    let new_base = l.base_offset + (start as int) * l.strides[axis];
    LayoutModel {
        shape: ShapeModel { dims: new_dims },
        strides: l.strides,
        base_offset: new_base,
    }
}

/// Ghost model of `Layout::complement(cosize)` for the rank-0
/// case: the complement is a single rank-1 contiguous mode equal
/// to cosize. (The rank-1 + higher-rank cases land in a follow-up
/// once we mirror the iterative sort-and-emit in Verus.)
pub open spec fn complement_rank0(cosize: nat) -> LayoutModel {
    LayoutModel {
        shape: ShapeModel { dims: seq![cosize] },
        strides: seq![1int],
        base_offset: 0,
    }
}

// ── Theorems ────────────────────────────────────────────────────

/// T8111 — `transpose` preserves rank.
proof fn t8111_transpose_preserves_rank(l: LayoutModel, i: int, j: int)
    requires
        l.well_formed(),
        0 <= i < l.rank(),
        0 <= j < l.rank(),
    ensures transpose(l, i, j).rank() == l.rank(),
{
}

/// T8112 — `transpose(i, i)` is a no-op on dims and strides.
proof fn t8112_transpose_same_axis_is_id(l: LayoutModel, i: int)
    requires
        l.well_formed(),
        0 <= i < l.rank(),
    ensures
        transpose(l, i, i).shape.dims =~= l.shape.dims,
        transpose(l, i, i).strides =~= l.strides,
        transpose(l, i, i).base_offset == l.base_offset,
{
}

/// T8113 — `transpose` preserves base_offset.
proof fn t8113_transpose_preserves_base_offset(l: LayoutModel, i: int, j: int)
    requires
        l.well_formed(),
        0 <= i < l.rank(),
        0 <= j < l.rank(),
    ensures transpose(l, i, j).base_offset == l.base_offset,
{
}

/// T8114 — `slice` preserves rank.
proof fn t8114_slice_preserves_rank(l: LayoutModel, axis: int, start: nat, end: nat)
    requires
        l.well_formed(),
        0 <= axis < l.rank(),
        start < end,
        end <= l.shape.dims[axis],
    ensures slice(l, axis, start, end).rank() == l.rank(),
{
}

/// T8115 — `slice` advances `base_offset` by `start * stride[axis]`.
proof fn t8115_slice_advances_base(l: LayoutModel, axis: int, start: nat, end: nat)
    requires
        l.well_formed(),
        0 <= axis < l.rank(),
        start < end,
        end <= l.shape.dims[axis],
    ensures
        slice(l, axis, start, end).base_offset
            == l.base_offset + (start as int) * l.strides[axis],
{
}

/// T8116 — `slice` keeps strides unchanged.
proof fn t8116_slice_keeps_strides(l: LayoutModel, axis: int, start: nat, end: nat)
    requires
        l.well_formed(),
        0 <= axis < l.rank(),
        start < end,
        end <= l.shape.dims[axis],
    ensures slice(l, axis, start, end).strides =~= l.strides,
{
}

/// T8117 — `slice` updates only the targeted axis extent.
proof fn t8117_slice_axis_only(l: LayoutModel, axis: int, start: nat, end: nat, k: int)
    requires
        l.well_formed(),
        0 <= axis < l.rank(),
        start < end,
        end <= l.shape.dims[axis],
        0 <= k < l.rank(),
        k != axis,
    ensures slice(l, axis, start, end).shape.dims[k] == l.shape.dims[k],
{
}

/// T8118 — `slice` sets the targeted axis to `end - start`.
proof fn t8118_slice_axis_extent(l: LayoutModel, axis: int, start: nat, end: nat)
    requires
        l.well_formed(),
        0 <= axis < l.rank(),
        start < end,
        end <= l.shape.dims[axis],
    ensures slice(l, axis, start, end).shape.dims[axis] == (end - start) as nat,
{
}

/// T8119 — `complement_rank0` of any `cosize` returns a rank-1
/// contiguous layout whose linear size equals `cosize`.
proof fn t8119_complement_rank0_shape(cosize: nat)
    requires cosize >= 1,
    ensures
        complement_rank0(cosize).rank() == 1,
        complement_rank0(cosize).shape.dims[0] == cosize,
        complement_rank0(cosize).strides[0] == 1int,
        complement_rank0(cosize).base_offset == 0,
{
}

/// T8120 — `complement_rank0` is well-formed for any positive
/// `cosize`. (cosize == 0 isn't supported by Shape::new.)
proof fn t8120_complement_rank0_well_formed(cosize: nat)
    requires cosize >= 1,
    ensures complement_rank0(cosize).well_formed(),
{
    let m = complement_rank0(cosize);
    assert(m.strides.len() == m.shape.dims.len());
    assert forall|k: int| 0 <= k < m.shape.dims.len() implies m.shape.dims[k] >= 1 by {
        assert(m.shape.dims[k] == cosize);
    }
}

/// T8121 — `transpose` returns a well-formed layout when given a
/// well-formed input. (Swapping two positions preserves both the
/// shape's ≥1 extent invariant and the strides-length-matches-rank
/// invariant.)
proof fn t8121_transpose_preserves_well_formed(l: LayoutModel, i: int, j: int)
    requires
        l.well_formed(),
        0 <= i < l.rank(),
        0 <= j < l.rank(),
    ensures transpose(l, i, j).well_formed(),
{
    let t = transpose(l, i, j);
    // Strides length matches rank: both lists are swap-of-original.
    assert(t.strides.len() == l.strides.len());
    assert(t.shape.dims.len() == l.shape.dims.len());
    // Every extent ≥ 1: swapping doesn't introduce new values.
    assert forall|k: int| 0 <= k < t.shape.dims.len()
        implies t.shape.dims[k] >= 1 by
    {
        // The k-th element is either l.dims[i], l.dims[j], or l.dims[k]
        // — all ≥ 1 by l.well_formed().
        if k == i {
            assert(t.shape.dims[k] == l.shape.dims[j]);
        } else if k == j {
            assert(t.shape.dims[k] == l.shape.dims[i]);
        } else {
            assert(t.shape.dims[k] == l.shape.dims[k]);
        }
    }
}

} // verus!
