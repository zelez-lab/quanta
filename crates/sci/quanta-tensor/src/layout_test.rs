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

// ─────────────────────────────────────────────────────────────────
// is_contiguous
// ─────────────────────────────────────────────────────────────────

/// Unflatten a row-major linear index `k` into a coordinate over
/// `dims`. Test-side twin of the Lean `unflatten` spec function.
fn unflatten(dims: &[usize], mut k: usize) -> Vec<usize> {
    let mut coord = vec![0usize; dims.len()];
    for i in (0..dims.len()).rev() {
        coord[i] = k % dims[i];
        k /= dims[i];
    }
    coord
}

#[test]
fn is_contiguous_row_major() {
    assert!(Layout::row_major(&[2, 3, 4]).unwrap().is_contiguous());
    // Rank 0 and rank 1 are trivially dense.
    assert!(Layout::row_major(&[]).unwrap().is_contiguous());
    assert!(Layout::row_major(&[7]).unwrap().is_contiguous());
}

#[test]
fn is_contiguous_rejects_column_major_rank2() {
    // Rank ≥ 2 column-major is not row-major dense…
    assert!(!Layout::column_major(&[2, 3]).unwrap().is_contiguous());
    // …but rank-1 column-major coincides with row-major.
    assert!(Layout::column_major(&[5]).unwrap().is_contiguous());
}

#[test]
fn is_contiguous_rejects_transpose() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert!(!l.transpose(0, 1).unwrap().is_contiguous());
}

#[test]
fn is_contiguous_leading_axis_slice_stays_contiguous() {
    // Clipping the outermost axis keeps one dense block, shifted
    // by base_offset — still contiguous by our definition.
    let l = Layout::row_major(&[4, 3]).unwrap();
    let s = l.slice(0, 1, 3).unwrap();
    assert!(s.is_contiguous());
    assert_eq!(s.base_offset(), 3);
}

#[test]
fn is_contiguous_inner_axis_slice_is_not() {
    let l = Layout::row_major(&[4, 3]).unwrap();
    let s = l.slice(1, 0, 2).unwrap();
    assert!(!s.is_contiguous());
}

#[test]
fn is_contiguous_broadcast_zero_stride_is_not() {
    // broadcast pads leading axes with stride 0; strict density
    // rejects that even though the axis has extent 1.
    let b = Layout::row_major(&[3]).unwrap().broadcast(&[1, 3]).unwrap();
    assert!(!b.is_contiguous());
}

// ─────────────────────────────────────────────────────────────────
// reshape
// ─────────────────────────────────────────────────────────────────

#[test]
fn reshape_contiguous_ok() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    let r = l.reshape(&[6, 4]).unwrap();
    assert_eq!(r.shape().dims(), &[6, 4]);
    assert_eq!(r.strides(), &[4, 1]);
    assert_eq!(r.linear_size(), 24);
    assert_eq!(r.base_offset(), 0);
    // Pure reindexing: linear index k lands at offset k both ways.
    for k in 0..24 {
        assert_eq!(l.at(&unflatten(&[2, 3, 4], k)).unwrap(), k);
        assert_eq!(r.at(&unflatten(&[6, 4], k)).unwrap(), k);
    }
}

#[test]
fn reshape_rank_collapse_to_1d_matches_linear_index() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    let flat = l.reshape(&[24]).unwrap();
    for k in 0..24 {
        assert_eq!(flat.at(&[k]).unwrap(), k);
    }
}

#[test]
fn reshape_preserves_base_offset() {
    // Leading-axis slice keeps contiguity but shifts the origin;
    // reshape must carry the shift through.
    let l = Layout::row_major(&[4, 3]).unwrap();
    let s = l.slice(0, 2, 4).unwrap();
    assert_eq!(s.base_offset(), 6);
    let r = s.reshape(&[6]).unwrap();
    assert_eq!(r.base_offset(), 6);
    for k in 0..6 {
        assert_eq!(r.at(&[k]).unwrap(), 6 + k);
    }
}

#[test]
fn reshape_rejects_non_contiguous() {
    let t = Layout::row_major(&[2, 3]).unwrap().transpose(0, 1).unwrap();
    assert!(matches!(
        t.reshape(&[6]),
        Err(LayoutError::NonContiguousReshape { .. })
    ));
}

#[test]
fn reshape_rejects_size_mismatch() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert_eq!(
        l.reshape(&[7]),
        Err(LayoutError::ReshapeSizeMismatch {
            from_size: 6,
            to_size: 7,
        })
    );
}

#[test]
fn reshape_rejects_zero_extent() {
    let l = Layout::row_major(&[2, 3]).unwrap();
    assert!(matches!(l.reshape(&[0, 6]), Err(LayoutError::Shape(_))));
}

#[test]
fn reshape_round_trip_is_identity() {
    let l = Layout::row_major(&[2, 3, 4]).unwrap();
    let back = l.reshape(&[6, 4]).unwrap().reshape(&[2, 3, 4]).unwrap();
    assert_eq!(back, l);
}

// ─────────────────────────────────────────────────────────────────
// coalesce
// ─────────────────────────────────────────────────────────────────

#[test]
fn coalesce_fuses_row_major_to_rank1() {
    let c = Layout::row_major(&[2, 3, 4]).unwrap().coalesce();
    assert_eq!(c.shape().dims(), &[24]);
    assert_eq!(c.strides(), &[1]);
}

#[test]
fn coalesce_drops_size_one_axes() {
    let l = Layout::row_major(&[1, 4, 1]).unwrap();
    let c = l.coalesce();
    assert_eq!(c.shape().dims(), &[4]);
    assert_eq!(c.strides(), &[1]);
    assert_eq!(c.linear_size(), l.linear_size());
}

#[test]
fn coalesce_identity_on_non_fusable() {
    // Transposed 2-D layout: nothing to drop, nothing to fuse.
    let t = Layout::row_major(&[2, 3]).unwrap().transpose(0, 1).unwrap();
    let c = t.coalesce();
    assert_eq!(c, t);
}

#[test]
fn coalesce_all_ones_gives_rank_zero() {
    let c = Layout::row_major(&[1, 1]).unwrap().coalesce();
    assert_eq!(c.rank(), 0);
    assert_eq!(c.linear_size(), 1);
    assert_eq!(c.at(&[]).unwrap(), 0);
}

#[test]
fn coalesce_mixed_drop_and_fuse() {
    // slice(0, 0, 1) leaves dims [1, 3, 4] with strides [12, 4, 1]:
    // the leading axis drops, the trailing two fuse.
    let l = Layout::row_major(&[2, 3, 4])
        .unwrap()
        .slice(0, 0, 1)
        .unwrap();
    let c = l.coalesce();
    assert_eq!(c.shape().dims(), &[12]);
    assert_eq!(c.strides(), &[1]);
}

#[test]
fn coalesce_preserves_offsets_over_flat_domain() {
    // Offset-equivalence: walking both layouts in row-major
    // coordinate order visits the same flat offsets.
    let cases = vec![
        Layout::row_major(&[2, 3, 4]).unwrap(),
        Layout::column_major(&[2, 3, 4]).unwrap(),
        Layout::row_major(&[1, 4, 1]).unwrap(),
        Layout::row_major(&[4, 3]).unwrap().slice(1, 0, 2).unwrap(),
        Layout::row_major(&[2, 3, 4])
            .unwrap()
            .slice(0, 1, 2)
            .unwrap(),
        Layout::row_major(&[3]).unwrap().broadcast(&[2, 3]).unwrap(),
        Layout::row_major(&[2, 3]).unwrap().transpose(0, 1).unwrap(),
    ];
    for l in cases {
        let c = l.coalesce();
        assert_eq!(c.linear_size(), l.linear_size(), "{:?}", l);
        assert_eq!(c.base_offset(), l.base_offset(), "{:?}", l);
        let ld = l.shape().dims().to_vec();
        let cd = c.shape().dims().to_vec();
        for k in 0..l.linear_size() {
            assert_eq!(
                c.at(&unflatten(&cd, k)).unwrap(),
                l.at(&unflatten(&ld, k)).unwrap(),
                "layout {:?} diverges from its coalesce at linear index {}",
                l,
                k
            );
        }
    }
}

#[test]
fn coalesce_unblocks_reshape_after_broadcast_padding() {
    // The strictness escape hatch documented on is_contiguous:
    // broadcast's stride-0 padding axis blocks reshape, coalesce
    // drops it and reshape goes through.
    let b = Layout::row_major(&[6]).unwrap().broadcast(&[1, 6]).unwrap();
    assert!(b.reshape(&[2, 3]).is_err());
    let r = b.coalesce().reshape(&[2, 3]).unwrap();
    assert_eq!(r.shape().dims(), &[2, 3]);
    assert_eq!(r.strides(), &[3, 1]);
}
