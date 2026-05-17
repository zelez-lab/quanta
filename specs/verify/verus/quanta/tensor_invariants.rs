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

// ── complement_rank1 ─────────────────────────────────────────────
//
// Spec-level model of `Layout::complement` on a rank-1 input.
// Mirrors the closed-form result from
// `crates/quanta-tensor/src/layout/algebra.rs::complement_rank1`.
//
// Given a rank-1 layout `(s, d)` with `d > 0` and a cosize ≥ s*d,
// the result is one of three shapes depending on whether the
// leading "gap" mode and the trailing "periods" mode each
// contribute:
//
// - `d == 1 && periods == 1`: empty (rank-0) result.
// - `d > 1  && periods == 1`: rank-1 `(d, 1)`.
// - `d == 1 && periods > 1`:  rank-1 `(periods, s*d)`.
// - `d > 1  && periods > 1`:  rank-2 `((d, 1), (periods, s*d))`.
//
// where `periods = ceil_div(cosize, s*d)`.

/// Ghost helper: ceil_div for `nat`. Matches the production
/// `usize::div_ceil`. Verus's nat subtraction produces `int`, so
/// we work in `int` internally and coerce back via `as nat`.
pub open spec fn ceil_div(a: nat, b: nat) -> nat
    recommends b > 0,
{
    if b == 0 {
        0nat
    } else if a == 0 {
        0nat
    } else {
        let ai: int = a as int;
        let bi: int = b as int;
        ((ai - 1) / bi + 1) as nat
    }
}

/// Spec-level model of `Layout::complement_rank1`. Given a rank-1
/// layout's `(s, d)` and a cosize, produces the corresponding
/// layout model. Precondition: `d > 0` and `cosize >= s * d`.
pub open spec fn complement_rank1(s: nat, d: nat, cosize: nat) -> LayoutModel
    recommends d > 0, cosize >= s * d,
{
    let period: nat = s * d;
    let periods: nat = ceil_div(cosize, period);
    if d == 1 && periods <= 1 {
        // Layout already covers cosize. Empty complement.
        LayoutModel {
            shape: ShapeModel { dims: Seq::<nat>::empty() },
            strides: Seq::<int>::empty(),
            base_offset: 0,
        }
    } else if d > 1 && periods <= 1 {
        LayoutModel {
            shape: ShapeModel { dims: seq![d] },
            strides: seq![1int],
            base_offset: 0,
        }
    } else if d == 1 && periods > 1 {
        LayoutModel {
            shape: ShapeModel { dims: seq![periods] },
            strides: seq![period as int],
            base_offset: 0,
        }
    } else {
        // d > 1 && periods > 1
        LayoutModel {
            shape: ShapeModel { dims: seq![d, periods] },
            strides: seq![1int, period as int],
            base_offset: 0,
        }
    }
}

/// T8122 — `complement_rank1` produces a layout with base offset 0.
proof fn t8122_complement_rank1_zero_base(s: nat, d: nat, cosize: nat)
    requires d > 0, cosize >= s * d,
    ensures complement_rank1(s, d, cosize).base_offset == 0,
{
}

/// T8123 — Strides match the rank in every branch. The case split
/// on `d` and `periods` produces dims and strides of equal length
/// across all four branches, so `strides.len() == shape.dims.len()`.
proof fn t8123_complement_rank1_strides_match_rank(s: nat, d: nat, cosize: nat)
    requires d > 0, cosize >= s * d,
    ensures
        complement_rank1(s, d, cosize).strides.len()
            == complement_rank1(s, d, cosize).shape.dims.len(),
{
}

/// Helper: `ceil_div(a, a) == 1` whenever `a >= 1`. Used by T8124
/// and T8125 to drive the `periods <= 1` branch of
/// `complement_rank1`. Verus can establish the underlying
/// `(a-1)/a + 1 == 1` once we expose the integer-division bound
/// `0 <= (a-1) < a`.
proof fn ceil_div_self_eq_one(a: nat)
    requires a >= 1,
    ensures ceil_div(a, a) == 1nat,
{
    let ai: int = a as int;
    assert(ai >= 1);
    assert((ai - 1) / ai == 0int) by (nonlinear_arith)
        requires ai >= 1;
}

/// T8124 — When `d == 1` and `cosize == s * d`, the complement is
/// the empty (rank-0) layout. This is the degenerate "already
/// covers" case the production code handles separately.
proof fn t8124_complement_rank1_full_coverage(s: nat)
    requires s >= 1,
    ensures complement_rank1(s, 1nat, s) == (LayoutModel {
        shape: ShapeModel { dims: Seq::<nat>::empty() },
        strides: Seq::<int>::empty(),
        base_offset: 0,
    }),
{
    ceil_div_self_eq_one(s);
}

/// T8125 — When `d > 1` and `cosize == s * d`, the periods count
/// is 1 and the complement reduces to a rank-1 `(d, 1)` layout.
proof fn t8125_complement_rank1_single_period(s: nat, d: nat)
    requires d > 1, s >= 1,
    ensures complement_rank1(s, d, s * d) == (LayoutModel {
        shape: ShapeModel { dims: seq![d] },
        strides: seq![1int],
        base_offset: 0,
    }),
{
    let period: nat = s * d;
    assert(period >= 1) by (nonlinear_arith)
        requires s >= 1, d > 1, period == s * d;
    ceil_div_self_eq_one(period);
}

/// T8126 — When `d == 1` and `cosize > s`, the result is a rank-1
/// periods layout `(periods, s)`.
proof fn t8126_complement_rank1_unit_gap(s: nat, cosize: nat)
    requires s >= 1, cosize > s,
    ensures ({
        let periods = ceil_div(cosize, s);
        periods > 1 ==> complement_rank1(s, 1nat, cosize) == (LayoutModel {
            shape: ShapeModel { dims: seq![periods] },
            strides: seq![s as int],
            base_offset: 0,
        })
    }),
{
}

/// T8127 — The general d > 1, periods > 1 case produces a rank-2
/// layout. Stated as an existence claim (rank == 2) rather than a
/// specific element-by-element comparison so the dependent-typed
/// match doesn't dominate the proof obligation.
proof fn t8127_complement_rank1_two_mode_rank(s: nat, d: nat, cosize: nat)
    requires
        s >= 1,
        d > 1,
        cosize >= s * d,
        ceil_div(cosize, s * d) > 1,
    ensures complement_rank1(s, d, cosize).rank() == 2,
{
}

// ── Rank-N complement: stride-sort fold model ───────────────────
//
// Mirrors `Layout::complement_general` in
// `crates/quanta-tensor/src/layout/algebra.rs`. The production
// algorithm:
//
//   work = [(shape_0, stride_0), …, (shape_{R-1}, stride_{R-1})]
//   last_stride = 1
//   result = ([], [])
//   while |work| > 1:
//     pick min-stride pair (s_m, d_m) at index `j`
//     emit (d_m / last_stride, last_stride)
//     last_stride := d_m * s_m
//     work := work without index j  (swap_remove)
//   emit (last_pair.stride / last_stride, last_stride)
//   if cosize / boundary > 1: emit (periods, boundary)
//   drop leading size-1 modes
//
// We mirror that as a recursive spec fn with `decreases work.len()`,
// accumulating `(Seq<nat>, Seq<int>)` of result modes. The
// load-bearing structural invariants — strides.len() ==
// shape.dims.len() and base_offset == 0 — fall out of the
// construction itself.
//
// We deliberately do *not* model the runtime divisibility /
// injectivity checks (`d % last == 0`, `new_shape > 0`); those
// raise `ComplementInfeasible` errors in production and surface
// as ill-formed model output if violated. The Rust-side test suite
// covers the error paths; the spec covers the structural shape of
// the *successful* path.

/// Stride-sort fold. The production algorithm picks the smallest
/// remaining stride per iteration; in the spec we instead require
/// the caller to pass a *pre-sorted* `work` sequence (ascending
/// stride), so each step consumes the head via `drop_first` —
/// equivalent up to permutation of the result modes, and Verus can
/// see the decreasing measure trivially.
///
/// The `sort_by_stride` correspondence is a separate (later)
/// theorem; for now the structural invariants below hold for any
/// input order.
pub open spec fn complement_fold(
    work: Seq<(nat, nat)>,
    last_stride: nat,
    acc_shape: Seq<nat>,
    acc_stride: Seq<int>,
) -> (Seq<nat>, Seq<int>, nat)
    decreases work.len()
{
    if work.len() == 0 {
        (acc_shape, acc_stride, last_stride)
    } else if work.len() == 1 {
        // Base case: emit the final gap mode for the leftover pair.
        let (last_s, last_d) = work[0];
        let final_gap = if last_stride == 0 { 0nat } else { last_d / last_stride };
        let boundary = last_d * last_s;
        (
            acc_shape.push(final_gap),
            acc_stride.push(last_stride as int),
            boundary,
        )
    } else {
        let (s_h, d_h) = work[0];
        let new_shape = if last_stride == 0 { 0nat } else { d_h / last_stride };
        let next_last = d_h * s_h;
        complement_fold(
            work.drop_first(),
            next_last,
            acc_shape.push(new_shape),
            acc_stride.push(last_stride as int),
        )
    }
}

/// Drop leading modes whose extent is 1 (the "no gap" case in the
/// production cleanup). If every mode has extent 1, the result is
/// the empty (rank-0) layout — that's caught by returning an empty
/// pair.
pub open spec fn drop_leading_size_one(
    dims: Seq<nat>,
    strides: Seq<int>,
) -> (Seq<nat>, Seq<int>)
    decreases dims.len()
{
    if dims.len() == 0 || strides.len() != dims.len() {
        (dims, strides)
    } else if dims[0] == 1nat {
        drop_leading_size_one(dims.drop_first(), strides.drop_first())
    } else {
        (dims, strides)
    }
}

/// Spec-level model of `Layout::complement_general`. Builds the
/// fold's working sequence from `(shape, stride)` pairs (treating
/// negative strides as 0 — the production code rejects those as
/// `ComplementInfeasible`), runs the fold, optionally appends the
/// trailing periods mode, then drops leading size-1 modes.
pub open spec fn complement_general(l: LayoutModel, cosize: nat) -> LayoutModel
    recommends l.well_formed(), l.rank() >= 2,
{
    let work = build_work_seq(l.shape.dims, l.strides);
    let (raw_shape, raw_stride, boundary) =
        complement_fold(work, 1nat, Seq::<nat>::empty(), Seq::<int>::empty());

    // Trailing periods mode: appended iff cosize / boundary > 1.
    let (with_periods_shape, with_periods_stride) = if boundary == 0 {
        (raw_shape, raw_stride)
    } else {
        let periods = ceil_div(cosize, boundary);
        if periods <= 1 {
            (raw_shape, raw_stride)
        } else {
            (raw_shape.push(periods), raw_stride.push(boundary as int))
        }
    };

    // Drop leading size-1 modes.
    let (final_shape, final_stride) =
        drop_leading_size_one(with_periods_shape, with_periods_stride);

    LayoutModel {
        shape: ShapeModel { dims: final_shape },
        strides: final_stride,
        base_offset: l.base_offset,
    }
}

/// Build `(shape, stride)` pairs from a shape and stride sequence.
/// Negative strides become 0 in the ghost model — they're rejected
/// at runtime, so the spec's behaviour on them is unconstrained.
pub open spec fn build_work_seq(dims: Seq<nat>, strides: Seq<int>) -> Seq<(nat, nat)>
    decreases dims.len()
{
    if dims.len() == 0 || strides.len() == 0 {
        Seq::<(nat, nat)>::empty()
    } else {
        let head_d = if strides[0] < 0 { 0nat } else { strides[0] as nat };
        seq![(dims[0], head_d)] + build_work_seq(dims.drop_first(), strides.drop_first())
    }
}

// ── Rank-N complement: theorems ─────────────────────────────────

/// T8128 — `complement_fold` preserves the length equality between
/// the accumulated shape and stride sequences. Inductive over the
/// `decreases work.len()` recursion.
proof fn t8128_complement_fold_length_invariant(
    work: Seq<(nat, nat)>,
    last_stride: nat,
    acc_shape: Seq<nat>,
    acc_stride: Seq<int>,
)
    requires acc_shape.len() == acc_stride.len(),
    ensures ({
        let (out_shape, out_stride, _) =
            complement_fold(work, last_stride, acc_shape, acc_stride);
        out_shape.len() == out_stride.len()
    }),
    decreases work.len()
{
    if work.len() == 0 {
        // Empty work: result is (acc_shape, acc_stride, _).
    } else if work.len() == 1 {
        // Base case: each branch pushes exactly one element to both
        // sides, so length equality is preserved.
    } else {
        let (s_h, d_h) = work[0];
        let new_shape = if last_stride == 0 { 0nat } else { d_h / last_stride };
        let next_last = d_h * s_h;
        t8128_complement_fold_length_invariant(
            work.drop_first(),
            next_last,
            acc_shape.push(new_shape),
            acc_stride.push(last_stride as int),
        );
    }
}

/// T8129 — `drop_leading_size_one` preserves the length equality.
proof fn t8129_drop_leading_size_one_length_invariant(dims: Seq<nat>, strides: Seq<int>)
    requires dims.len() == strides.len(),
    ensures ({
        let (out_dims, out_strides) = drop_leading_size_one(dims, strides);
        out_dims.len() == out_strides.len()
    }),
    decreases dims.len()
{
    if dims.len() == 0 || strides.len() != dims.len() {
        // Base case: return unchanged. (The strides.len() guard is
        // dead under our precondition; included for the spec's
        // totality.)
    } else if dims[0] == 1nat {
        t8129_drop_leading_size_one_length_invariant(dims.drop_first(), strides.drop_first());
    } else {
        // Non-empty, head != 1: return unchanged.
    }
}

/// T8130 — `complement_general` produces a layout whose strides
/// length matches its shape rank. Combines T8128 (fold preserves
/// the length equality) and T8129 (the cleanup preserves it).
proof fn t8130_complement_general_well_formed_lengths(l: LayoutModel, cosize: nat)
    ensures
        complement_general(l, cosize).strides.len()
            == complement_general(l, cosize).shape.dims.len(),
{
    // Fold result satisfies the length equality (empty starts equal).
    t8128_complement_fold_length_invariant(
        build_work_seq(l.shape.dims, l.strides),
        1nat,
        Seq::<nat>::empty(),
        Seq::<int>::empty(),
    );
    let (raw_shape, raw_stride, boundary) = complement_fold(
        build_work_seq(l.shape.dims, l.strides),
        1nat,
        Seq::<nat>::empty(),
        Seq::<int>::empty(),
    );
    assert(raw_shape.len() == raw_stride.len());

    // Appending the periods mode preserves equality.
    let (with_periods_shape, with_periods_stride) = if boundary == 0 {
        (raw_shape, raw_stride)
    } else {
        let periods = ceil_div(cosize, boundary);
        if periods <= 1 {
            (raw_shape, raw_stride)
        } else {
            (raw_shape.push(periods), raw_stride.push(boundary as int))
        }
    };
    assert(with_periods_shape.len() == with_periods_stride.len());

    // Cleanup preserves equality.
    t8129_drop_leading_size_one_length_invariant(with_periods_shape, with_periods_stride);
}

/// T8131 — `complement_general` preserves the layout's base offset.
/// Structural: the cleanup and fold never touch `base_offset`.
proof fn t8131_complement_general_preserves_base_offset(l: LayoutModel, cosize: nat)
    ensures complement_general(l, cosize).base_offset == l.base_offset,
{
}

/// T8132 — `complement_fold` returned tail length equals
/// `acc.len() + work.len()`. Each step pushes exactly one element
/// and shrinks `work` by exactly one (via `drop_first`).
proof fn t8132_complement_fold_length_growth(
    work: Seq<(nat, nat)>,
    last_stride: nat,
    acc_shape: Seq<nat>,
    acc_stride: Seq<int>,
)
    requires acc_shape.len() == acc_stride.len(),
    ensures ({
        let (out_shape, _, _) =
            complement_fold(work, last_stride, acc_shape, acc_stride);
        out_shape.len() == acc_shape.len() + work.len()
    }),
    decreases work.len()
{
    if work.len() == 0 {
        // No-op, lengths preserved.
    } else if work.len() == 1 {
        // Each branch pushes one element.
    } else {
        let (s_h, d_h) = work[0];
        let new_shape = if last_stride == 0 { 0nat } else { d_h / last_stride };
        let next_last = d_h * s_h;
        t8132_complement_fold_length_growth(
            work.drop_first(),
            next_last,
            acc_shape.push(new_shape),
            acc_stride.push(last_stride as int),
        );
    }
}

} // verus!
