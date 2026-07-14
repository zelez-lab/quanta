//! Shape — multi-dimensional extents.
//!
//! `Shape` is a thin wrapper over `Vec<usize>` that enforces "every
//! extent is at least 1." Zero-size axes are degenerate and don't
//! flow cleanly through stride math (`stride * 0 == 0`, fine; but
//! `linear_size == 0` is a foot-gun for downstream allocators), so
//! we reject them at construction.
//!
//! Rank-0 (scalar) shapes are allowed — they have an empty extent
//! list and `linear_size == 1`.

use core::fmt;

/// Errors that can occur when constructing a `Shape`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// One of the supplied axis extents was zero. Zero-size axes
    /// are rejected at construction; if you need an empty tensor,
    /// use a rank-0 shape with no extents (linear size 1) or just
    /// allocate zero bytes upstream of this type.
    ZeroExtent {
        /// Index of the offending axis in the supplied extent list.
        axis: usize,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::ZeroExtent { axis } => {
                write!(f, "axis {} has extent 0", axis)
            }
        }
    }
}

/// A multi-dimensional shape — an ordered list of axis extents.
///
/// Each extent is ≥ 1. The rank (number of axes) is `dims().len()`.
/// `linear_size()` is the product of all extents (1 for rank 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Construct a shape from a slice of axis extents. Returns
    /// `ShapeError::ZeroExtent` if any extent is 0.
    pub fn new(dims: &[usize]) -> Result<Self, ShapeError> {
        for (axis, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(ShapeError::ZeroExtent { axis });
            }
        }
        Ok(Self {
            dims: dims.to_vec(),
        })
    }

    /// Construct a shape without checking the no-zero-extent
    /// invariant. Useful for layout-op outputs where the invariant
    /// is preserved structurally — e.g. transpose just permutes
    /// extents, slice clips a range whose length is already ≥ 1.
    ///
    /// Callers must guarantee no extent is zero.
    pub(crate) fn from_dims_unchecked(dims: Vec<usize>) -> Self {
        debug_assert!(
            dims.iter().all(|&d| d > 0),
            "from_dims_unchecked given a zero extent: {:?}",
            dims
        );
        Self { dims }
    }

    /// Axis extents in declaration order.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Product of all extents. 1 for a rank-0 shape.
    pub fn linear_size(&self) -> usize {
        self.dims.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_positive_extents() {
        let s = Shape::new(&[2, 3, 4]).unwrap();
        assert_eq!(s.dims(), &[2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.linear_size(), 24);
    }

    #[test]
    fn new_rejects_zero_extent() {
        assert_eq!(
            Shape::new(&[2, 0, 4]),
            Err(ShapeError::ZeroExtent { axis: 1 })
        );
    }

    #[test]
    fn rank_zero_is_scalar() {
        let s = Shape::new(&[]).unwrap();
        assert_eq!(s.rank(), 0);
        // Empty product = 1: scalar holds one element.
        assert_eq!(s.linear_size(), 1);
    }
}
