//! Global algebra on `Layout`: composition, complement, divide.
//!
//! These are the *load-bearing* ops for downstream math crates.
//! Where the local ops in `super::ops` are transformations of a
//! single layout (transpose, slice, ...), these ops produce a new
//! layout from two inputs and carry the algebraic identities
//! GEMM-style tiling depends on:
//!
//! - `compose(A, B)`: apply A to the output of B, with stride and
//!   shape divisibility checks. Foundation for tiled-divide.
//! - `complement(A, cosize)`: the layout that fills the "remaining"
//!   space after A within `cosize`, used to construct the residual
//!   modes of a tile.
//! - `logical_divide(A, tiler)`: split A by a tiler into a layout
//!   with two modes — the tile modes and the residual modes.
//! - `tiled_divide(A, tiler)`: convenience flat-tuple form of
//!   `logical_divide` (the tile mode + each residual mode
//!   separately).
//!
//! The Rust port is derived from CUTLASS CuTe's
//! `include/cute/layout.hpp` (composition_impl ~L1033, complement
//! ~L1180, logical_divide ~L1559). CuTe uses compile-time integer
//! tuples; we use runtime `Vec<usize>` + `Vec<isize>`, so the
//! divisibility checks become runtime `Result`s rather than
//! `static_assert`s.

use crate::shape::Shape;

use super::{Layout, LayoutError};

impl Layout {
    /// Build the complement of a layout within a `cosize`.
    ///
    /// Returns a layout that, together with `self`, covers
    /// `0..cosize` without overlap. The result's modes walk the
    /// "gaps" in `self`'s footprint, followed by any "periods"
    /// needed to reach `cosize`.
    ///
    /// **Rank handling**:
    /// - **Rank 0**: trivial. The complement is `(cosize, 1)` —
    ///   a contiguous rank-1 layout walking every position.
    /// - **Rank 1** (`(s, d)`): closed form
    ///   `((d, 1), (ceil_div(cosize, s*d), s*d))`, with the
    ///   leading `d` mode dropped when `d == 1` and the trailing
    ///   periods mode dropped when there's only one period.
    /// - **Rank ≥ 2**: port of CuTe's stride-sort algorithm
    ///   (`cute/layout.hpp` ~L1180). Iteratively picks the
    ///   smallest remaining stride, emits a complement mode
    ///   `(min_stride / last_result_stride, last_result_stride)`,
    ///   and advances `last_result_stride` to `min_stride *
    ///   shape[min_idx]`. After `R-1` iterations the final
    ///   mode is appended, plus a periods mode for the leftover
    ///   `cosize / last_result_stride`.
    ///
    /// # Errors
    /// - `ComplementInfeasible { reason }` for degenerate inputs
    ///   (stride-0 with a non-trivial shape, negative strides,
    ///   cosize smaller than the layout's footprint, or a
    ///   non-injective layout detected mid-algorithm).
    pub fn complement(&self, cosize: usize) -> Result<Layout, LayoutError> {
        let rank = self.rank();
        if rank == 0 {
            // Complement of a rank-0 layout: the cosize as a
            // contiguous rank-1 layout.
            return Ok(Layout::from_parts(
                Shape::from_dims_unchecked(vec![cosize]),
                vec![1],
                self.base_offset(),
            ));
        }
        if rank == 1 {
            return self.complement_rank1(cosize);
        }
        // Rank ≥ 2: full CuTe-style stride-sort.
        self.complement_general(cosize)
    }

    /// Rank-1 closed-form complement. Kept as a fast path so the
    /// most common downstream pattern doesn't pay the
    /// sort-and-fold cost.
    fn complement_rank1(&self, cosize: usize) -> Result<Layout, LayoutError> {
        let s = self.shape().dims()[0];
        let d = self.strides()[0];
        if d == 0 {
            return Err(LayoutError::ComplementInfeasible {
                reason: "stride-0 layout has no complement",
            });
        }
        if d < 0 {
            return Err(LayoutError::ComplementInfeasible {
                reason: "negative-stride complement is not implemented",
            });
        }
        let d = d as usize;
        let footprint = s.saturating_mul(d);
        if cosize < footprint {
            return Err(LayoutError::ComplementInfeasible {
                reason: "cosize is smaller than the layout's footprint",
            });
        }
        let period = footprint;
        let periods = cosize.div_ceil(period);
        if periods <= 1 && d == 1 {
            return Ok(Layout::from_parts(
                Shape::from_dims_unchecked(vec![]),
                vec![],
                self.base_offset(),
            ));
        }
        let mut dims: Vec<usize> = Vec::new();
        let mut strides: Vec<isize> = Vec::new();
        if d > 1 {
            dims.push(d);
            strides.push(1);
        }
        if periods > 1 {
            dims.push(periods);
            strides.push(period as isize);
        }
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(dims),
            strides,
            self.base_offset(),
        ))
    }

    /// Multi-rank complement. Ports CuTe's stride-sort fold.
    ///
    /// Working set: a list of `(shape_i, stride_i)` pairs. Each
    /// iteration removes the smallest-stride pair and emits one
    /// result mode. After all `R` pairs are consumed, one
    /// trailing "periods" mode is appended for the
    /// `cosize / last_result_stride` factor.
    fn complement_general(&self, cosize: usize) -> Result<Layout, LayoutError> {
        // Snapshot working (shape, stride) pairs as unsigned.
        // Rejects zero or negative strides up-front — they can't
        // sort or feed the divide step.
        let mut work: Vec<(usize, usize)> = Vec::with_capacity(self.rank());
        for (&s, &d) in self.shape().dims().iter().zip(self.strides().iter()) {
            if d <= 0 {
                return Err(LayoutError::ComplementInfeasible {
                    reason: "complement requires every stride to be positive",
                });
            }
            work.push((s, d as usize));
        }

        let mut result_shape: Vec<usize> = Vec::with_capacity(self.rank() + 1);
        let mut result_stride: Vec<isize> = Vec::with_capacity(self.rank() + 1);
        // The accumulator that CuTe carries as `result_stride`'s
        // last element. Starts at 1.
        let mut last_stride: usize = 1;

        // Run the fold over R-1 pairs; the final pair is handled
        // below to produce the closing "gap" mode separately.
        while work.len() > 1 {
            // Find the index of the smallest stride.
            let min_idx = work
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, d))| *d)
                .map(|(i, _)| i)
                .expect("work non-empty in loop guard");
            let (min_shape, min_stride) = work[min_idx];

            // new_shape = min_stride / last_stride. Must be > 0
            // (CuTe asserts; we return an error).
            if last_stride == 0
                || min_stride < last_stride
                || !min_stride.is_multiple_of(last_stride)
            {
                return Err(LayoutError::ComplementInfeasible {
                    reason: "non-injective layout: stride not divisible by previous step",
                });
            }
            let new_shape = min_stride / last_stride;
            if new_shape == 0 {
                return Err(LayoutError::ComplementInfeasible {
                    reason: "non-injective layout detected in complement",
                });
            }
            result_shape.push(new_shape);
            result_stride.push(last_stride as isize);
            last_stride = min_stride.saturating_mul(min_shape);

            // Remove the consumed mode from the working set.
            work.swap_remove(min_idx);
            // `swap_remove` reorders the tail; that's fine because
            // the algorithm only cares about the multiset of
            // remaining (shape, stride) pairs, not their order.
        }

        // Final mode: handle the last remaining (shape, stride)
        // pair. CuTe appends `new_shape = min_stride / last_stride`
        // and uses `min_stride * min_shape` as the boundary
        // between the gap modes and the trailing periods.
        let (last_shape_in, last_stride_in) = *work.last().expect("rank >= 2 guarantees one left");
        if last_stride == 0
            || last_stride_in < last_stride
            || !last_stride_in.is_multiple_of(last_stride)
        {
            return Err(LayoutError::ComplementInfeasible {
                reason: "non-injective layout: final stride not divisible by previous step",
            });
        }
        let final_gap = last_stride_in / last_stride;
        if final_gap == 0 {
            return Err(LayoutError::ComplementInfeasible {
                reason: "non-injective layout detected at final mode",
            });
        }
        result_shape.push(final_gap);
        result_stride.push(last_stride as isize);
        let boundary = last_stride_in.saturating_mul(last_shape_in);

        // Trailing periods: ceil_div(cosize, boundary) periods of
        // length `boundary`. Dropped if there's only one period.
        if cosize < boundary {
            return Err(LayoutError::ComplementInfeasible {
                reason: "cosize is smaller than the layout's footprint",
            });
        }
        let periods = cosize.div_ceil(boundary);
        if periods > 1 {
            result_shape.push(periods);
            result_stride.push(boundary as isize);
        }

        // Drop any leading size-1 mode (the "no gap" case).
        if let Some((idx, _)) = result_shape.iter().enumerate().find(|&(_, &s)| s != 1) {
            if idx > 0 {
                result_shape.drain(0..idx);
                result_stride.drain(0..idx);
            }
        } else {
            // All modes have size 1 — collapse to rank-0.
            result_shape.clear();
            result_stride.clear();
        }

        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(result_shape),
            result_stride,
            self.base_offset(),
        ))
    }

    /// Compose two layouts: apply `self` to the output of `rhs`.
    ///
    /// Both operands are flat (Quanta's `Layout` is always flat;
    /// CuTe's nested-tuple form maps to our flat form by
    /// concatenating modes). The result has rank equal to
    /// `rhs.rank() + leftover_from_self` after the fold.
    ///
    /// # Errors
    /// - `DivisibilityFailed` when the stride or shape divisibility
    ///   conditions don't hold on a given fold step.
    /// - `UnsupportedRank` for cases the port doesn't yet handle
    ///   (we cover the common cases — see tests).
    pub fn compose(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        let rhs_rank = rhs.rank();
        if rhs_rank == 0 {
            // Rank-0 RHS: the composition is just self.
            return Ok(self.clone());
        }
        // Right-distributivity: compose with each RHS dim
        // independently, then concatenate the results.
        //
        // For each (s_r, d_r) in rhs, run the LHS-fold below to
        // produce a partial layout; concatenate all partials into
        // the final result.
        let mut out_dims: Vec<usize> = Vec::new();
        let mut out_strides: Vec<isize> = Vec::new();
        for axis in 0..rhs_rank {
            let s_r = rhs.shape().dims()[axis];
            let d_r = rhs.strides()[axis];
            let part = compose_lhs_with_int(self, s_r, d_r)?;
            out_dims.extend(part.shape().dims().iter().copied());
            out_strides.extend(part.strides().iter().copied());
        }
        Ok(Layout::from_parts(
            Shape::from_dims_unchecked(out_dims),
            out_strides,
            self.base_offset() + rhs.base_offset(),
        ))
    }

    /// Logical divide: split `self` by a `tiler` layout into a
    /// rank-2-ish layout `(tile_modes, residual_modes)`.
    ///
    /// Equivalent to `self.compose(make_layout(tiler, complement(tiler, self.linear_size())))`
    /// — the tiler walks one tile, the complement walks the tiles.
    pub fn logical_divide(&self, tiler: &Layout) -> Result<Layout, LayoutError> {
        let tiler_complement = tiler.complement(self.linear_size())?;
        // Build a combined layout whose first modes are the tiler
        // and whose later modes are the complement. CuTe's
        // make_layout(tiler, complement) returns a hierarchical
        // tuple; we flatten by concatenating dim/stride lists.
        let mut combined_dims = tiler.shape().dims().to_vec();
        combined_dims.extend(tiler_complement.shape().dims().iter().copied());
        let mut combined_strides = tiler.strides().to_vec();
        combined_strides.extend(tiler_complement.strides().iter().copied());
        let combined = Layout::from_parts(
            Shape::from_dims_unchecked(combined_dims),
            combined_strides,
            tiler.base_offset() + tiler_complement.base_offset(),
        );
        self.compose(&combined)
    }

    /// Tiled divide: like `logical_divide`, but the tiler position
    /// is preserved at the front and each complement mode is its
    /// own axis afterwards. The result reads as
    /// `(tile, r1, r2, …)`.
    pub fn tiled_divide(&self, tiler: &Layout) -> Result<Layout, LayoutError> {
        // For our flat representation, `tiled_divide == logical_divide`
        // because we never nested the tile modes into a single
        // hierarchical axis in the first place. Kept as a separate
        // method so the call site reads correctly and so we can
        // add nesting back without changing callers if/when we do.
        self.logical_divide(tiler)
    }
}

/// Compose `lhs` (a full Layout) with a single (s, d) pair on the
/// right. Direct port of CuTe's `composition_impl` for the
/// `lhs: tuple, rhs: integral` case (the load-bearing branch).
///
/// The fold walks lhs left-to-right, accumulating an output
/// (shape, stride) and carrying `(rest_shape, rest_stride)` —
/// the as-yet-unconsumed portion of the RHS — forward.
fn compose_lhs_with_int(
    lhs: &Layout,
    rhs_shape: usize,
    rhs_stride: isize,
) -> Result<Layout, LayoutError> {
    let r = lhs.rank();
    // RHS stride 0: the composition reads the same element of
    // self.at(0) for every rhs coordinate — just one mode (rhs_shape,
    // 0).
    if rhs_stride == 0 {
        return Ok(Layout::from_parts(
            Shape::from_dims_unchecked(vec![rhs_shape]),
            vec![0],
            0,
        ));
    }
    // LHS rank 0: trivial.
    if r == 0 {
        return Ok(Layout::from_parts(
            Shape::from_dims_unchecked(vec![rhs_shape]),
            vec![rhs_stride],
            0,
        ));
    }
    // LHS integral (rank 1) shortcut: result is `(rhs_shape, rhs_stride * lhs_stride[0])`.
    if r == 1 {
        let l_stride = lhs.strides()[0];
        return Ok(Layout::from_parts(
            Shape::from_dims_unchecked(vec![rhs_shape]),
            vec![rhs_stride * l_stride],
            0,
        ));
    }

    // General case: fold over lhs[0..r-1], handle lhs[r-1] at the end.
    let l_shape = lhs.shape().dims();
    let l_stride = lhs.strides();
    let mut result_shape: Vec<usize> = Vec::new();
    let mut result_stride: Vec<isize> = Vec::new();
    let mut rest_shape: usize = rhs_shape;
    let mut rest_stride: isize = rhs_stride;

    for i in 0..r - 1 {
        let curr_shape = l_shape[i];
        let curr_stride = l_stride[i];

        // Stride divisibility: rest_stride % curr_shape == 0 || |rest_stride| < curr_shape.
        let rs_abs = rest_stride.unsigned_abs();
        let cs = curr_shape;
        if rs_abs >= cs && !rs_abs.is_multiple_of(cs) {
            return Err(LayoutError::DivisibilityFailed {
                kind: "stride",
                lhs: cs,
                rhs: rs_abs,
            });
        }

        // next_shape = ceil_div(curr_shape, |rest_stride|).
        let next_shape = cs.div_ceil(rs_abs.max(1));
        // next_stride = ceil_div(|rest_stride|, curr_shape) * sign(rest_stride).
        let next_stride_abs = rs_abs.div_ceil(cs.max(1));
        let next_stride = (next_stride_abs as isize) * rest_stride.signum();

        if next_shape == 1 || rest_shape == 1 {
            // No mode emitted; just advance rest_stride.
            rest_stride = next_stride;
            continue;
        }

        let new_shape = next_shape.min(rest_shape);
        // Shape divisibility: rest_shape % new_shape == 0.
        if !rest_shape.is_multiple_of(new_shape) {
            return Err(LayoutError::DivisibilityFailed {
                kind: "shape",
                lhs: new_shape,
                rhs: rest_shape,
            });
        }
        result_shape.push(new_shape);
        result_stride.push(rest_stride * curr_stride);
        rest_shape /= new_shape;
        rest_stride = next_stride;
    }

    // Tail: handle the last lhs mode.
    if result_shape.is_empty() {
        return Ok(Layout::from_parts(
            Shape::from_dims_unchecked(vec![rest_shape]),
            vec![rest_stride * l_stride[r - 1]],
            0,
        ));
    }
    if rest_shape == 1 {
        return Ok(Layout::from_parts(
            Shape::from_dims_unchecked(result_shape),
            result_stride,
            0,
        ));
    }
    result_shape.push(rest_shape);
    result_stride.push(rest_stride * l_stride[r - 1]);
    Ok(Layout::from_parts(
        Shape::from_dims_unchecked(result_shape),
        result_stride,
        0,
    ))
}
