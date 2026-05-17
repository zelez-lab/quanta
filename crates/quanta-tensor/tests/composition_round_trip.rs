//! Integration test for the roadmap acceptance criterion: chained
//! layout ops compose correctly on a 4-D tensor and the resulting
//! indexer agrees with a manual offset computation.
//!
//! The roadmap's literal phrasing is
//! `permute(transpose(reshape(t)))`, but `reshape` is more subtle
//! than the other ops (it only works on contiguous layouts) and
//! lands in a later crate. This test instead exercises the same
//! spirit — composing three different ops on a non-trivial tensor
//! and asserting offset equivalence — by chaining transpose →
//! permute → slice.

use quanta_tensor::Layout;

/// Compute the row-major offset of a coordinate by hand. The
/// independent oracle for the test.
fn row_major_offset(coord: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        offset += coord[i] * stride;
        stride *= shape[i];
    }
    offset
}

#[test]
fn transpose_then_permute_then_slice_on_4d_tensor() {
    // Source: a 2×3×4×5 row-major tensor. linear_size = 120.
    let src_shape = [2usize, 3, 4, 5];
    let src = Layout::row_major(&src_shape).unwrap();
    assert_eq!(src.linear_size(), 120);

    // Step 1: transpose axes 1 and 2. New shape 2×4×3×5.
    let after_transpose = src.transpose(1, 2).unwrap();
    assert_eq!(after_transpose.shape().dims(), &[2, 4, 3, 5]);

    // Step 2: cyclic permute (3, 0, 1, 2) — last axis to front.
    // New shape: 5×2×4×3.
    let after_permute = after_transpose.permute(&[3, 0, 1, 2]).unwrap();
    assert_eq!(after_permute.shape().dims(), &[5, 2, 4, 3]);

    // Step 3: slice axis 0 to [1, 4). New shape: 3×2×4×3.
    let sliced = after_permute.slice(0, 1, 4).unwrap();
    assert_eq!(sliced.shape().dims(), &[3, 2, 4, 3]);

    // For every coordinate (a, b, c, d) of the sliced layout, the
    // offset must equal the manual row-major offset of the
    // corresponding coordinate in the original.
    //
    // Coordinate mapping (composing the three ops in reverse):
    //   sliced.at[(a,b,c,d)]
    //     == after_permute.at[(a+1, b, c, d)]
    //     == after_transpose.at[(b, c, d, a+1)]     // un-permute [3,0,1,2]
    //     == src.at[(b, d, c, a+1)]                  // un-transpose axes 1↔2
    //
    // We compare both numbers via the independent oracle.
    let mut count = 0usize;
    for a in 0..3 {
        for b in 0..2 {
            for c in 0..4 {
                for d in 0..3 {
                    let from_composed = sliced.at(&[a, b, c, d]).unwrap();
                    let oracle = row_major_offset(&[b, d, c, a + 1], &src_shape);
                    assert_eq!(
                        from_composed, oracle,
                        "mismatch at ({a},{b},{c},{d}): composed={from_composed}, oracle={oracle}"
                    );
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, sliced.linear_size());
}

#[test]
fn broadcast_after_slice_preserves_source_offsets() {
    // 1×3 source. Slice axis 1 to [0, 2). Broadcast leading axis to 4.
    let src = Layout::row_major(&[1, 3]).unwrap();
    let sliced = src.slice(1, 0, 2).unwrap();
    assert_eq!(sliced.shape().dims(), &[1, 2]);
    let bcast = sliced.broadcast(&[4, 2]).unwrap();
    assert_eq!(bcast.shape().dims(), &[4, 2]);

    // Every broadcast row reads the same two sliced elements.
    for i in 0..4 {
        for j in 0..2 {
            assert_eq!(bcast.at(&[i, j]).unwrap(), sliced.at(&[0, j]).unwrap());
        }
    }
}

#[test]
fn double_transpose_is_identity_on_offsets() {
    // Composition law: transpose(d0, d1) ∘ transpose(d0, d1) = id.
    let src = Layout::row_major(&[3, 5, 7]).unwrap();
    let twice = src.transpose(0, 2).unwrap().transpose(0, 2).unwrap();
    assert_eq!(twice, src);
}

#[test]
fn permute_inverse_returns_original_layout() {
    // Cyclic [1, 2, 0] inverted by [2, 0, 1].
    let src = Layout::row_major(&[2, 3, 4]).unwrap();
    let forward = src.permute(&[1, 2, 0]).unwrap();
    let back = forward.permute(&[2, 0, 1]).unwrap();
    assert_eq!(back, src);
}
