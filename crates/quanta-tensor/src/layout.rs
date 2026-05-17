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

pub mod algebra;
pub mod ops;
mod strides;

use strides::{compute_column_major_strides, compute_row_major_strides};

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
    /// `complement` / `compose` was called on a higher-rank layout
    /// that the dynamic-stride port doesn't yet handle. CuTe's
    /// complement supports rank > 1 via stride sorting, which
    /// requires per-call dynamic checks we haven't ported.
    UnsupportedRank {
        /// Op the caller invoked.
        op: &'static str,
        /// Rank that was supplied.
        rank: usize,
    },
    /// A divisibility check inside `compose` failed: the layout
    /// algebra requires either `rest_stride % curr_shape == 0` or
    /// `rest_stride < curr_shape`. Neither held.
    DivisibilityFailed {
        /// "stride" or "shape" — which divisibility law was violated.
        kind: &'static str,
        /// LHS extent / stride at the failure point.
        lhs: usize,
        /// RHS extent / stride at the failure point.
        rhs: usize,
    },
    /// `complement` cannot produce a layout for the given inputs
    /// (zero stride combined with a non-trivial layout, or
    /// non-injective layout fed in).
    ComplementInfeasible {
        /// Short reason string.
        reason: &'static str,
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
            LayoutError::UnsupportedRank { op, rank } => write!(
                f,
                "{} is not yet supported for layouts of rank {} \
                 (dynamic-stride port handles rank 1 only)",
                op, rank
            ),
            LayoutError::DivisibilityFailed { kind, lhs, rhs } => write!(
                f,
                "layout composition violated the {} divisibility law: lhs={} rhs={}",
                kind, lhs, rhs
            ),
            LayoutError::ComplementInfeasible { reason } => {
                write!(f, "complement is undefined: {}", reason)
            }
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
