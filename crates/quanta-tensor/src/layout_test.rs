//! Unit tests for `Layout`. Companion file per the project's
//! testing guideline — production code in `layout.rs`,
//! `layout/ops.rs`, and `layout/strides.rs` stays test-free.

use crate::layout::{Layout, LayoutError};

#[test]
fn row_major_strides_2_3_4() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    // Outer to inner: 3*4, 4, 1.
    assert_eq!(l.strides(), &[12, 4, 1]);
    assert_eq!(l.linear_size(), 24);
}

#[test]
fn column_major_strides_2_3_4() {
    let l = Layout::column_major(&[2, 3, 4]).unwrap();
    // Inner to outer: 1, 2, 2*3.
    assert_eq!(l.strides(), &[1, 2, 6]);
    assert_eq!(l.linear_size(), 24);
}

#[test]
fn row_major_indexing_round_trip() {
    // 2×3 matrix in row-major: offset(i,j) = i*3 + j.
    let l = Layout::row_major(&[2, 3]).unwrap();
    let cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([0, 2], 2),
        ([1, 0], 3),
        ([1, 1], 4),
        ([1, 2], 5),
    ];
    for (coord, expected) in cases {
        assert_eq!(l.at(&coord).unwrap(), expected, "at({:?})", coord);
    }
}

#[test]
fn column_major_indexing_round_trip() {
    // 2×3 matrix in column-major: offset(i,j) = i + j*2.
    let l = Layout::column_major(&[2, 3]).unwrap();
    let cases = [
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 2),
        ([1, 1], 3),
        ([0, 2], 4),
        ([1, 2], 5),
    ];
    for (coord, expected) in cases {
        assert_eq!(l.at(&coord).unwrap(), expected, "at({:?})", coord);
    }
}

#[test]
fn at_rejects_wrong_rank() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert_eq!(
        l.at(&[0]),
        Err(LayoutError::RankMismatch {
            got: 1,
            expected: 2,
        })
    );
}

#[test]
fn at_rejects_out_of_bounds() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert_eq!(
        l.at(&[2, 0]),
        Err(LayoutError::OutOfBounds {
            axis: 0,
            got: 2,
            max: 1,
        })
    );
    assert_eq!(
        l.at(&[0, 3]),
        Err(LayoutError::OutOfBounds {
            axis: 1,
            got: 3,
            max: 2,
        })
    );
}

#[test]
fn rank_zero_layout_indexes_single_element() {
    let l = Layout::row_major(&[]).unwrap();
    assert_eq!(l.rank(), 0);
    assert_eq!(l.linear_size(), 1);
    assert_eq!(l.at(&[]).unwrap(), 0);
}

/// Every distinct coordinate of a row-major layout maps to a
/// distinct offset, and the offsets cover [0, linear_size).
/// This is the bijection property that downstream proofs lean on.
#[test]
fn row_major_is_bijective_over_linear_range() {
    let dims = [2, 3, 4];
    let l = Layout::row_major(&dims).unwrap();
    let n = l.linear_size();
    let mut seen = vec![false; n];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let off = l.at(&[i, j, k]).unwrap();
                assert!(off < n, "offset {} out of range for n={}", off, n);
                assert!(!seen[off], "duplicate offset {}", off);
                seen[off] = true;
            }
        }
    }
    assert!(seen.iter().all(|&b| b));
}

// ── transpose ────────────────────────────────────────────────────────

#[test]
fn transpose_2d_matches_manual_offsets() {
    // 2×3 row-major: row stride 3, col stride 1.
    let l = Layout::row_major(&[2, 3]).unwrap();
    let t = l.transpose(0, 1).unwrap();
    // After transpose: shape is 3×2 with strides [1, 3]. So at[i,j]
    // in the transposed layout reads the offset that at[j,i] would
    // have produced in the original.
    assert_eq!(t.shape().dims(), &[3, 2]);
    assert_eq!(t.strides(), &[1, 3]);
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(t.at(&[i, j]).unwrap(), l.at(&[j, i]).unwrap());
        }
    }
}

#[test]
fn transpose_same_axis_is_noop() {
    let l = Layout::row_major(&[4, 5]).unwrap();
    let t = l.transpose(1, 1).unwrap();
    assert_eq!(t, l);
}

#[test]
fn transpose_rejects_out_of_range_axis() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert!(matches!(
        l.transpose(0, 5),
        Err(LayoutError::AxisOutOfRange { axis: 5, rank: 2 })
    ));
}

// ── permute ──────────────────────────────────────────────────────────

#[test]
fn permute_3d_reverses_axes() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    let p = l.permute(&[2, 1, 0]).unwrap();
    assert_eq!(p.shape().dims(), &[4, 3, 2]);
    // Original strides [12, 4, 1] become [1, 4, 12].
    assert_eq!(p.strides(), &[1, 4, 12]);
    // Indexing equivalence: p.at(&[k,j,i]) == l.at(&[i,j,k]).
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(p.at(&[k, j, i]).unwrap(), l.at(&[i, j, k]).unwrap());
            }
        }
    }
}

#[test]
fn permute_identity_is_noop() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    let p = l.permute(&[0, 1, 2]).unwrap();
    assert_eq!(p, l);
}

#[test]
fn permute_rejects_duplicate() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    assert!(matches!(
        l.permute(&[0, 0, 2]),
        Err(LayoutError::InvalidPermutation { .. })
    ));
}

#[test]
fn permute_rejects_out_of_range() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    assert!(matches!(
        l.permute(&[0, 1, 5]),
        Err(LayoutError::InvalidPermutation { .. })
    ));
}

#[test]
fn permute_rejects_wrong_rank() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    assert!(matches!(
        l.permute(&[0, 1]),
        Err(LayoutError::InvalidPermutation { .. })
    ));
}

// ── slice ────────────────────────────────────────────────────────────

#[test]
fn slice_axis_1_clips_columns() {
    // 2×4 row-major: row stride 4, col stride 1.
    let l = Layout::row_major(&[2, 4]).unwrap();
    let s = l.slice(1, 1, 3).unwrap();
    assert_eq!(s.shape().dims(), &[2, 2]);
    assert_eq!(s.strides(), &[4, 1]);
    // base_offset = 0 + 1 * 1 = 1 (start of column 1).
    assert_eq!(s.base_offset(), 1);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(s.at(&[i, j]).unwrap(), l.at(&[i, j + 1]).unwrap());
        }
    }
}

#[test]
fn slice_axis_0_clips_rows() {
    let l = Layout::row_major(&[4, 3]).unwrap();
    let s = l.slice(0, 1, 3).unwrap();
    assert_eq!(s.shape().dims(), &[2, 3]);
    // base_offset = 0 + 1 * 3 = 3 (start of row 1).
    assert_eq!(s.base_offset(), 3);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(s.at(&[i, j]).unwrap(), l.at(&[i + 1, j]).unwrap());
        }
    }
}

#[test]
fn slice_rejects_empty_range() {
    let l = Layout::row_major(&[4, 3]).unwrap();
    assert!(matches!(
        l.slice(0, 2, 2),
        Err(LayoutError::InvalidSlice { .. })
    ));
}

#[test]
fn slice_rejects_out_of_range_end() {
    let l = Layout::row_major(&[4, 3]).unwrap();
    assert!(matches!(
        l.slice(0, 0, 5),
        Err(LayoutError::InvalidSlice { .. })
    ));
}

// ── broadcast ────────────────────────────────────────────────────────

#[test]
fn broadcast_size_1_axis_becomes_zero_stride() {
    // 1×3 row-major broadcast to 4×3: leading axis gets stride 0,
    // trailing axis keeps its stride.
    let l = Layout::row_major(&[1, 3]).unwrap();
    let b = l.broadcast(&[4, 3]).unwrap();
    assert_eq!(b.shape().dims(), &[4, 3]);
    assert_eq!(b.strides(), &[0, 1]);
    for i in 0..4 {
        for j in 0..3 {
            assert_eq!(b.at(&[i, j]).unwrap(), l.at(&[0, j]).unwrap());
        }
    }
}

#[test]
fn broadcast_adds_leading_axes_with_zero_stride() {
    // Rank-1 vector of length 3 broadcast to 2×3: prepend a size-2
    // axis with stride 0.
    let l = Layout::row_major(&[3]).unwrap();
    let b = l.broadcast(&[2, 3]).unwrap();
    assert_eq!(b.shape().dims(), &[2, 3]);
    assert_eq!(b.strides(), &[0, 1]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(b.at(&[i, j]).unwrap(), l.at(&[j]).unwrap());
        }
    }
}

#[test]
fn broadcast_rejects_size_mismatch_on_nonunit_axis() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert!(matches!(
        l.broadcast(&[4, 3]),
        Err(LayoutError::BroadcastIncompatible { .. })
    ));
}

#[test]
fn broadcast_rejects_self_rank_greater_than_target() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert!(matches!(
        l.broadcast(&[3]),
        Err(LayoutError::BroadcastIncompatible { .. })
    ));
}
