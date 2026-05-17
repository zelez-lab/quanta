//! Unit tests for the layout algebra (`compose`, `complement`,
//! `logical_divide`, `tiled_divide`). Companion file per the
//! project's testing guideline — `algebra.rs` stays test-free.

use crate::layout::{Layout, LayoutError};
use crate::shape::Shape;

// ── complement ───────────────────────────────────────────────────

#[test]
fn complement_rank1_basic() {
    // Layout (4, 1) — 4 contiguous elements. Cosize 12: there are
    // 3 "periods" of 4. Footprint = 4*1 = 4 == period. Since d ==
    // 1, the first mode is dropped; result has only the period.
    let l = Layout::from_parts(Shape::from_dims_unchecked(vec![4]), vec![1], 0);
    let c = l.complement(12).unwrap();
    assert_eq!(c.shape().dims(), &[3]);
    assert_eq!(c.strides(), &[4]);
}

#[test]
fn complement_rank1_with_gap() {
    // Layout (4, 2) — 4 elements spaced 2 apart. Cosize 16. First
    // mode walks gaps of size 2 (d = 2 → mode (2, 1)). Second mode
    // walks 16/8 = 2 periods of length 8.
    let l = Layout::from_parts(Shape::from_dims_unchecked(vec![4]), vec![2], 0);
    let c = l.complement(16).unwrap();
    assert_eq!(c.shape().dims(), &[2, 2]);
    assert_eq!(c.strides(), &[1, 8]);
}

#[test]
fn complement_higher_rank_unsupported() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert!(matches!(
        l.complement(36),
        Err(LayoutError::UnsupportedRank {
            op: "complement",
            rank: 2
        })
    ));
}

#[test]
fn complement_cosize_smaller_than_footprint_rejects() {
    let l = Layout::from_parts(Shape::from_dims_unchecked(vec![4]), vec![2], 0);
    // Footprint 8; cosize 4 — infeasible.
    assert!(matches!(
        l.complement(4),
        Err(LayoutError::ComplementInfeasible { .. })
    ));
}

// ── compose ──────────────────────────────────────────────────────

#[test]
fn compose_rank1_with_rank1_integral() {
    // self = (12, 1) (flat 12-element layout).
    // rhs  = (4, 1)  (read 4 contiguous elements).
    let lhs = Layout::row_major(&[12]).unwrap();
    let rhs = Layout::row_major(&[4]).unwrap();
    let out = lhs.compose(&rhs).unwrap();
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.strides(), &[1]);
}

#[test]
fn compose_with_stride_zero_rhs_repeats() {
    // self = (12, 1), rhs = (4, 0) — read element 0 four times.
    let lhs = Layout::row_major(&[12]).unwrap();
    let rhs = Layout::from_parts(Shape::from_dims_unchecked(vec![4]), vec![0], 0);
    let out = lhs.compose(&rhs).unwrap();
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.strides(), &[0]);
}

#[test]
fn compose_rank2_lhs_rank1_rhs_basic() {
    // self = (2×3) row-major (strides [3, 1]).
    // rhs  = (6, 1) — walk the 6 elements in order.
    //
    // Composition of a rank-2 LHS with a rank-1 RHS unfolds the
    // linear RHS index into the LHS modes — equivalent to
    // re-emitting the LHS as its own layout. Out has shape [2, 3]
    // with strides [3, 1], and out.at([row, col]) == lhs.at([row,
    // col]).
    let lhs = Layout::row_major(&[2, 3]).unwrap();
    assert_eq!(lhs.strides(), &[3, 1]);
    let rhs = Layout::from_parts(Shape::from_dims_unchecked(vec![6]), vec![1], 0);
    let out = lhs.compose(&rhs).unwrap();
    assert_eq!(out.shape().dims(), &[2, 3]);
    assert_eq!(out.strides(), &[3, 1]);
    for row in 0..2usize {
        for col in 0..3usize {
            let from_compose = out.at(&[row, col]).unwrap();
            let from_lhs = lhs.at(&[row, col]).unwrap();
            assert_eq!(from_compose, from_lhs, "row={row}, col={col}");
        }
    }
}

// ── logical_divide ───────────────────────────────────────────────

#[test]
fn logical_divide_partitions_rank1() {
    // self = (12, 1) — 12 contiguous elements.
    // tiler = (4, 1) — tile size 4.
    // Result: ((4 elements per tile), (3 tiles)) flattened to
    // shape [4, 3] with strides [1, 4].
    let self_layout = Layout::row_major(&[12]).unwrap();
    let tiler = Layout::from_parts(Shape::from_dims_unchecked(vec![4]), vec![1], 0);
    let out = self_layout.logical_divide(&tiler).unwrap();
    assert_eq!(out.shape().dims(), &[4, 3]);
    assert_eq!(out.strides(), &[1, 4]);
    assert_eq!(out.at(&[0, 0]).unwrap(), 0);
    assert_eq!(out.at(&[3, 2]).unwrap(), 11);
}

#[test]
fn tiled_divide_equals_logical_divide_in_flat_form() {
    let self_layout = Layout::row_major(&[12]).unwrap();
    let tiler = Layout::from_parts(Shape::from_dims_unchecked(vec![4]), vec![1], 0);
    let a = self_layout.logical_divide(&tiler).unwrap();
    let b = self_layout.tiled_divide(&tiler).unwrap();
    assert_eq!(a, b);
}
