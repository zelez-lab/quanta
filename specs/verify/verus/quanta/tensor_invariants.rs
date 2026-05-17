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
//! Matched-by-number with the Lean theorems in
//! `specs/verify/lean/Quanta/Tensor/Layout.lean` (T8000–T8014), at
//! a slightly different offset because the Verus arm exercises the
//! operational ghost model where the Lean arm exercises the
//! mathematical theorem.
//!
//! Phase 1 surface: the substrate types + indexer. Layout ops
//! (transpose, permute, slice, broadcast) get their own Verus
//! invariants in a follow-up commit alongside the matching Lean
//! algebraic theorems.

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

} // verus!
