//! Composable layout ops on `Layout`.
//!
//! Each op returns a new `Layout` without touching the underlying
//! buffer. They are pure functions of the input layout.

use crate::shape::Shape;

use super::{Layout, LayoutError};

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
        let mut new_dims = self.shape().dims().to_vec();
        let mut new_strides = self.strides().to_vec();
        new_dims.swap(d0, d1);
        new_strides.swap(d0, d1);
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(new_dims),
            new_strides,
            self.base_offset(),
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
        let old_dims = self.shape().dims();
        let old_strides = self.strides();
        let new_dims: Vec<usize> = perm.iter().map(|&j| old_dims[j]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&j| old_strides[j]).collect();
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(new_dims),
            new_strides,
            self.base_offset(),
        ))
    }

    /// Clip one axis to a half-open `[start, end)` range. The new
    /// layout has the same rank, the same strides, and a `base_offset`
    /// advanced by `start * strides[axis]` so the indexer's origin
    /// sits at the first slice element.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Self, LayoutError> {
        let rank = self.rank();
        if axis >= rank {
            return Err(LayoutError::AxisOutOfRange { axis, rank });
        }
        let extent = self.shape().dims()[axis];
        if start >= end || end > extent {
            return Err(LayoutError::InvalidSlice {
                axis,
                start,
                end,
                extent,
            });
        }
        let mut new_dims = self.shape().dims().to_vec();
        new_dims[axis] = end - start;
        let new_base = self.base_offset() + (start as isize) * self.strides()[axis];
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(new_dims),
            self.strides().to_vec(),
            new_base,
        ))
    }

    /// Broadcast `self` to `target`. The result has rank
    /// `target.len()`. Axes are aligned to the right (last axis
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
            return Err(LayoutError::BroadcastIncompatible {
                axis: 0,
                self_extent: self.shape().dims().first().copied().unwrap_or(1),
                target_extent: target.first().copied().unwrap_or(1),
            });
        }
        let pad = target_rank - self_rank;
        let self_dims = self.shape().dims();
        let self_strides = self.strides();
        let mut new_strides = vec![0isize; target_rank];
        for (i, &t) in target.iter().enumerate() {
            if i < pad {
                new_strides[i] = 0;
            } else {
                let self_axis = i - pad;
                let s = self_dims[self_axis];
                if s == t {
                    new_strides[i] = self_strides[self_axis];
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
            self.base_offset(),
        ))
    }
}
