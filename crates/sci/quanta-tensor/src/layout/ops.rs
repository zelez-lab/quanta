//! Composable layout ops on `Layout`.
//!
//! Each op returns a new `Layout` without touching the underlying
//! buffer. They are pure functions of the input layout.

use crate::shape::Shape;

use super::strides::compute_row_major_strides;
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

    /// Reinterpret the layout under a new shape — the O(1) rank
    /// fold/unfold every downstream rank-collapsing consumer
    /// (GEMM's `M×K` flattening, FFT's radix splits) needs.
    ///
    /// Preconditions:
    /// - `self.is_contiguous()` — the strides must be row-major
    ///   dense. For general strided layouts a reshape is not
    ///   always expressible as a view (CuTe-inherited fact), so we
    ///   refuse with [`LayoutError::NonContiguousReshape`] instead
    ///   of guessing.
    /// - `new_shape` must have the same element count as `self`
    ///   ([`LayoutError::ReshapeSizeMismatch`] otherwise) and no
    ///   zero extents (`LayoutError::Shape`).
    ///
    /// On success the result is `Layout::row_major(new_shape)`
    /// with `self`'s `base_offset` carried over: the offset of
    /// linear index `k` is `base_offset + k` before and after, so
    /// reshape is a pure reindexing of the same flat block.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, LayoutError> {
        if !self.is_contiguous() {
            return Err(LayoutError::NonContiguousReshape {
                strides: self.strides().to_vec(),
            });
        }
        let shape = Shape::new(new_shape)?;
        if shape.linear_size() != self.linear_size() {
            return Err(LayoutError::ReshapeSizeMismatch {
                from_size: self.linear_size(),
                to_size: shape.linear_size(),
            });
        }
        let new_strides = compute_row_major_strides(shape.dims());
        Ok(Layout::from_parts(shape, new_strides, self.base_offset()))
    }

    /// CuTe-style `coalesce`: the most compact layout that indexes
    /// identically over the row-major-flattened domain.
    ///
    /// Two simplifications, applied right-to-left in one pass:
    /// - **drop** every extent-1 axis (its coordinate is always 0,
    ///   so its stride never contributes to an offset);
    /// - **fuse** an axis into the group on its right when
    ///   `stride[i] == stride[i+1] * extent[i+1]` — the axis just
    ///   continues the same arithmetic progression.
    ///
    /// `linear_size` and `base_offset` are preserved, and for every
    /// linear index `k` the offset through the coalesced layout
    /// equals the offset through `self` (both walk coordinates in
    /// row-major order). Coalescing everything away (all extents 1)
    /// yields the rank-0 layout. Infallible — the identity layout
    /// is always a valid result.
    pub fn coalesce(&self) -> Self {
        let dims = self.shape().dims();
        let strides = self.strides();
        // Built innermost-first: `acc.last()` is the outermost
        // mode of the already-processed suffix.
        let mut acc: Vec<(usize, isize)> = Vec::new();
        for (&d, &s) in dims.iter().zip(strides.iter()).rev() {
            if d == 1 {
                continue;
            }
            match acc.last_mut() {
                Some((group_d, group_s)) if s == *group_s * (*group_d as isize) => {
                    *group_d *= d;
                }
                _ => acc.push((d, s)),
            }
        }
        acc.reverse();
        let (new_dims, new_strides): (Vec<usize>, Vec<isize>) = acc.into_iter().unzip();
        Layout::from_parts(
            Shape::from_dims_unchecked(new_dims),
            new_strides,
            self.base_offset(),
        )
    }
}
