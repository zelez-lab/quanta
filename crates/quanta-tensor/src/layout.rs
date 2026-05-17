//! Layout — function-style indexing into a flat buffer.
//!
//! `Layout` pairs a [`Shape`] with a stride vector and exposes the
//! indexer `at(&[usize]) -> usize`. The strides are deliberately
//! `isize` so layout ops like `slice` and `permute` that iterate
//! axes in reverse can carry negative stride increments without
//! losing the offset.
//!
//! Each `Layout` also carries a `base_offset: isize` that the
//! indexer adds to the stride dot product. Constructors set it to
//! `0`; `slice` may bump it forward to the first slice element.
//!
//! Invariants (held by every constructor + every op in this crate):
//! - `strides.len() == shape.rank()`.
//! - `shape` has no zero extents (enforced by `Shape::new`).
//! - For every valid coordinate `c`, `at(c)` returns a non-negative
//!   value that fits in `usize`. (The dot product is computed in
//!   `isize`; we panic on negative results, which would indicate a
//!   bad `base_offset` / stride combo.)

use core::fmt;

use crate::shape::{Shape, ShapeError};

/// Errors raised by `Layout` constructors, the indexer, and the
/// layout ops (`transpose`, `permute`, `slice`, `broadcast`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// Underlying shape construction failed.
    Shape(ShapeError),
    /// Indexer received the wrong number of coordinates.
    RankMismatch {
        /// Number of coordinates the indexer was called with.
        got: usize,
        /// Number of axes the layout has.
        expected: usize,
    },
    /// Indexer received a coordinate that's outside the shape.
    OutOfBounds {
        /// Axis index where the violation occurred.
        axis: usize,
        /// Supplied coordinate.
        got: usize,
        /// Inclusive upper bound (`shape.dims()[axis] - 1`).
        max: usize,
    },
    /// An op referenced an axis index ≥ the layout's rank.
    AxisOutOfRange {
        /// Axis the op asked about.
        axis: usize,
        /// Rank of the layout the op was called on.
        rank: usize,
    },
    /// `permute` was given an array that isn't a valid permutation
    /// of `0..rank` (duplicate index or out-of-range index).
    InvalidPermutation {
        /// The supplied permutation.
        perm: Vec<usize>,
    },
    /// `slice` was given an empty or reversed range, or one whose
    /// end exceeds the axis extent.
    InvalidSlice {
        /// Axis the slice was applied to.
        axis: usize,
        /// Requested half-open range.
        start: usize,
        /// End of the requested range.
        end: usize,
        /// Extent of the axis at the time of the call.
        extent: usize,
    },
    /// `broadcast` was given a target shape incompatible with
    /// `self`. A self-axis with extent ≠ 1 cannot be broadcast to
    /// a different extent.
    BroadcastIncompatible {
        /// Axis in the *aligned* (right-justified) layout where the
        /// mismatch occurred.
        axis: usize,
        /// Extent of that axis in `self` (after right-alignment).
        self_extent: usize,
        /// Extent of that axis in the target shape.
        target_extent: usize,
    },
}

impl From<ShapeError> for LayoutError {
    fn from(e: ShapeError) -> Self {
        LayoutError::Shape(e)
    }
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::Shape(e) => write!(f, "{}", e),
            LayoutError::RankMismatch { got, expected } => write!(
                f,
                "rank mismatch in Layout::at: got {} coordinates, expected {}",
                got, expected
            ),
            LayoutError::OutOfBounds { axis, got, max } => write!(
                f,
                "coordinate {} out of bounds at axis {} (max is {})",
                got, axis, max
            ),
            LayoutError::AxisOutOfRange { axis, rank } => {
                write!(f, "axis {} out of range for layout of rank {}", axis, rank)
            }
            LayoutError::InvalidPermutation { perm } => write!(
                f,
                "{:?} is not a valid permutation of 0..{}",
                perm,
                perm.len()
            ),
            LayoutError::InvalidSlice {
                axis,
                start,
                end,
                extent,
            } => write!(
                f,
                "invalid slice {}..{} on axis {} of extent {}",
                start, end, axis, extent
            ),
            LayoutError::BroadcastIncompatible {
                axis,
                self_extent,
                target_extent,
            } => write!(
                f,
                "broadcast incompatible at aligned axis {}: self extent {} vs target extent {}",
                axis, self_extent, target_extent
            ),
        }
    }
}

/// Function-style layout: maps an N-coordinate to a flat-buffer
/// offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<isize>,
    base_offset: isize,
}

impl Layout {
    /// Row-major (C order) layout: the last axis varies fastest.
    ///
    /// For `shape = [d0, d1, d2]` this produces strides
    /// `[d1*d2, d2, 1]` so coordinate `(i, j, k)` lands at
    /// `i*d1*d2 + j*d2 + k`.
    pub fn row_major(shape: &[usize]) -> Result<Self, LayoutError> {
        let shape = Shape::new(shape)?;
        let strides = compute_row_major_strides(shape.dims());
        Ok(Self {
            shape,
            strides,
            base_offset: 0,
        })
    }

    /// Column-major (Fortran order) layout: the first axis varies
    /// fastest.
    ///
    /// For `shape = [d0, d1, d2]` this produces strides
    /// `[1, d0, d0*d1]` so coordinate `(i, j, k)` lands at
    /// `i + j*d0 + k*d0*d1`.
    pub fn column_major(shape: &[usize]) -> Result<Self, LayoutError> {
        let shape = Shape::new(shape)?;
        let strides = compute_column_major_strides(shape.dims());
        Ok(Self {
            shape,
            strides,
            base_offset: 0,
        })
    }

    /// Construct a `Layout` directly from a shape + strides + base
    /// offset. Internal use by layout ops that compute their own
    /// stride vectors (transpose, permute, slice, broadcast).
    ///
    /// Panics in debug mode if `strides.len() != shape.rank()`.
    pub(crate) fn from_parts(shape: Shape, strides: Vec<isize>, base_offset: isize) -> Self {
        debug_assert_eq!(
            strides.len(),
            shape.rank(),
            "Layout::from_parts: strides.len() ({}) != shape.rank() ({})",
            strides.len(),
            shape.rank(),
        );
        Self {
            shape,
            strides,
            base_offset,
        }
    }

    /// Borrow the layout's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Borrow the strides (in axis order). Signed to allow ops
    /// that iterate in reverse; for stock row/column-major layouts
    /// all values are non-negative.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Base offset added to every indexer result. `0` for layouts
    /// returned by `row_major` / `column_major`; non-zero for the
    /// output of `slice`.
    pub fn base_offset(&self) -> isize {
        self.base_offset
    }

    /// Rank (number of axes).
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Number of distinct coordinates the layout can index.
    /// `shape.linear_size()`.
    pub fn linear_size(&self) -> usize {
        self.shape.linear_size()
    }

    /// Map an N-coordinate to a flat-buffer offset.
    ///
    /// `coord.len()` must equal `self.rank()`, and each coordinate
    /// must be strictly less than the corresponding shape extent.
    pub fn at(&self, coord: &[usize]) -> Result<usize, LayoutError> {
        if coord.len() != self.rank() {
            return Err(LayoutError::RankMismatch {
                got: coord.len(),
                expected: self.rank(),
            });
        }
        let dims = self.shape.dims();
        let mut offset: isize = self.base_offset;
        for (axis, (&c, (&d, &s))) in coord
            .iter()
            .zip(dims.iter().zip(self.strides.iter()))
            .enumerate()
        {
            if c >= d {
                return Err(LayoutError::OutOfBounds {
                    axis,
                    got: c,
                    max: d - 1,
                });
            }
            offset += (c as isize) * s;
        }
        debug_assert!(
            offset >= 0,
            "Layout::at produced a negative offset {} — bad strides/base_offset combination",
            offset
        );
        Ok(offset as usize)
    }
}

// ── Composable layout ops ────────────────────────────────────────────
//
// Each op returns a new `Layout` without touching the underlying
// buffer. They are pure functions of the input layout.

impl Layout {
    /// Swap two axes. The buffer is untouched; both extents and
    /// strides exchange positions at `d0` and `d1`. `transpose(i, i)`
    /// is a no-op.
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Self, LayoutError> {
        let rank = self.rank();
        if d0 >= rank {
            return Err(LayoutError::AxisOutOfRange { axis: d0, rank });
        }
        if d1 >= rank {
            return Err(LayoutError::AxisOutOfRange { axis: d1, rank });
        }
        let mut new_dims = self.shape.dims().to_vec();
        let mut new_strides = self.strides.clone();
        new_dims.swap(d0, d1);
        new_strides.swap(d0, d1);
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(new_dims),
            new_strides,
            self.base_offset,
        ))
    }

    /// General axis permutation. `perm[i] = j` means new axis `i`
    /// is old axis `j`. The supplied slice must be a valid
    /// permutation of `0..rank` (each index appears exactly once).
    pub fn permute(&self, perm: &[usize]) -> Result<Self, LayoutError> {
        let rank = self.rank();
        if perm.len() != rank {
            return Err(LayoutError::InvalidPermutation {
                perm: perm.to_vec(),
            });
        }
        let mut seen = vec![false; rank];
        for &p in perm {
            if p >= rank || seen[p] {
                return Err(LayoutError::InvalidPermutation {
                    perm: perm.to_vec(),
                });
            }
            seen[p] = true;
        }
        let old_dims = self.shape.dims();
        let new_dims: Vec<usize> = perm.iter().map(|&j| old_dims[j]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&j| self.strides[j]).collect();
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(new_dims),
            new_strides,
            self.base_offset,
        ))
    }

    /// Clip one axis to a half-open `[start, end)` range. The new
    /// layout has the same rank, the same strides, and a `base_offset`
    /// advanced by `start * strides[axis]` so that the indexer's
    /// origin sits at the first slice element.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Self, LayoutError> {
        let rank = self.rank();
        if axis >= rank {
            return Err(LayoutError::AxisOutOfRange { axis, rank });
        }
        let extent = self.shape.dims()[axis];
        if start >= end || end > extent {
            return Err(LayoutError::InvalidSlice {
                axis,
                start,
                end,
                extent,
            });
        }
        let mut new_dims = self.shape.dims().to_vec();
        new_dims[axis] = end - start;
        let new_base = self.base_offset + (start as isize) * self.strides[axis];
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(new_dims),
            self.strides.clone(),
            new_base,
        ))
    }

    /// Broadcast `self` to `target`. The result has rank
    /// `target.len()`. Axes are aligned to the **right** (last axis
    /// matches last axis); missing leading axes are filled in with
    /// extent `1`. For each aligned axis:
    ///
    /// - If self's extent equals target's, the stride is copied.
    /// - If self's extent is 1 and target's is ≥ 1, the stride is
    ///   set to 0 (so every coordinate on this axis reads the same
    ///   source element).
    /// - Any other mismatch returns `BroadcastIncompatible`.
    ///
    /// `base_offset` is preserved.
    pub fn broadcast(&self, target: &[usize]) -> Result<Self, LayoutError> {
        let target_shape = Shape::new(target)?;
        let target_rank = target_shape.rank();
        let self_rank = self.rank();
        if self_rank > target_rank {
            // Right-align: self can't have more axes than target.
            return Err(LayoutError::BroadcastIncompatible {
                axis: 0,
                self_extent: self.shape.dims().first().copied().unwrap_or(1),
                target_extent: target.first().copied().unwrap_or(1),
            });
        }
        let pad = target_rank - self_rank;
        let mut new_strides = vec![0isize; target_rank];
        for (i, &t) in target.iter().enumerate() {
            if i < pad {
                // New leading axis: extent goes from target, stride
                // is 0 (every coordinate hits the same source).
                new_strides[i] = 0;
            } else {
                let self_axis = i - pad;
                let s = self.shape.dims()[self_axis];
                if s == t {
                    new_strides[i] = self.strides[self_axis];
                } else if s == 1 {
                    new_strides[i] = 0;
                } else {
                    return Err(LayoutError::BroadcastIncompatible {
                        axis: i,
                        self_extent: s,
                        target_extent: t,
                    });
                }
            }
        }
        Ok(Layout::from_parts(
            target_shape,
            new_strides,
            self.base_offset,
        ))
    }
}

fn compute_row_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    let mut running: isize = 1;
    for i in (0..dims.len()).rev() {
        strides[i] = running;
        running *= dims[i] as isize;
    }
    strides
}

fn compute_column_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    let mut running: isize = 1;
    for i in 0..dims.len() {
        strides[i] = running;
        running *= dims[i] as isize;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

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
                expected: 2
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
                max: 1
            })
        );
        assert_eq!(
            l.at(&[0, 3]),
            Err(LayoutError::OutOfBounds {
                axis: 1,
                got: 3,
                max: 2
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

    // ── transpose ────────────────────────────────────────────────

    #[test]
    fn transpose_2d_matches_manual_offsets() {
        // 2×3 row-major: row stride 3, col stride 1.
        let l = Layout::row_major(&[2, 3]).unwrap();
        let t = l.transpose(0, 1).unwrap();
        // After transpose: shape is 3×2 with strides [1, 3].
        // So at[i,j] in the transposed layout reads the offset that
        // at[j,i] would have produced in the original.
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

    // ── permute ──────────────────────────────────────────────────

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

    // ── slice ────────────────────────────────────────────────────

    #[test]
    fn slice_axis_1_clips_columns() {
        // 2×4 row-major: row stride 4, col stride 1.
        let l = Layout::row_major(&[2, 4]).unwrap();
        let s = l.slice(1, 1, 3).unwrap();
        assert_eq!(s.shape().dims(), &[2, 2]);
        assert_eq!(s.strides(), &[4, 1]);
        // base_offset = 0 + 1 * 1 = 1 (start of column 1).
        assert_eq!(s.base_offset(), 1);
        // s.at(&[i, j]) should equal l.at(&[i, j + 1]).
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

    // ── broadcast ────────────────────────────────────────────────

    #[test]
    fn broadcast_size_1_axis_becomes_zero_stride() {
        // 1×3 row-major broadcast to 4×3: leading axis gets
        // stride 0, trailing axis keeps its stride.
        let l = Layout::row_major(&[1, 3]).unwrap();
        let b = l.broadcast(&[4, 3]).unwrap();
        assert_eq!(b.shape().dims(), &[4, 3]);
        assert_eq!(b.strides(), &[0, 1]);
        // Every row reads the same source row.
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(b.at(&[i, j]).unwrap(), l.at(&[0, j]).unwrap());
            }
        }
    }

    #[test]
    fn broadcast_adds_leading_axes_with_zero_stride() {
        // Rank-1 vector of length 3 broadcast to 2×3: prepend a
        // size-2 axis with stride 0.
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
        // Can't broadcast a 2-sized axis to size 4.
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
}
