//! `concat_axis0` — join tensors along axis 0.
//!
//! Built from the pieces we already have and trust: each input is scattered
//! into a full-size zero tensor at its row offset with
//! [`pad_axis0`](crate::Array::pad_axis0), then the (disjoint) placements are
//! summed. No new kernel, and the adjoint is just a `narrow` per input — so the
//! differentiable [`Var`](../../quanta_autograd) version needs no new VJP math.
//!
//! Axis-0 only (matching the `narrow`/`pad_axis0` family). The trailing shape
//! (`dims[1..]`) must match across all inputs.

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

impl<T: ArrayScalar> Array<T> {
    /// Concatenate `parts` along axis 0. Every part must share the same
    /// trailing shape (`dims[1..]`); the result's axis-0 extent is the sum of
    /// the parts' axis-0 extents, in order.
    pub fn concat_axis0(parts: &[&Array<T>]) -> Result<Array<T>, ArrayError> {
        if parts.is_empty() {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "concat_axis0: need at least one array",
            )));
        }
        let rest = parts[0].shape()[1..].to_vec();
        for p in parts {
            if p.shape().is_empty() {
                return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                    "concat_axis0: parts must have at least one axis",
                )));
            }
            if p.shape()[1..] != rest[..] {
                return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                    "concat_axis0: parts must share the trailing shape",
                )));
            }
        }
        let total: usize = parts.iter().map(|p| p.shape()[0]).sum();

        // Scatter each part into a full-height zero tensor at its offset, then
        // sum the disjoint placements. `pad_axis0` gives a contiguous
        // `[total, rest…]` tensor, so the sums are plain elementwise adds.
        let mut offset = 0usize;
        let mut acc: Option<Array<T>> = None;
        for p in parts {
            let placed = p.pad_axis0(total, offset)?;
            acc = Some(match acc {
                None => placed,
                Some(a) => a.add(&placed)?,
            });
            offset += p.shape()[0];
        }
        Ok(acc.expect("parts is non-empty"))
    }

    /// Stack `parts` along a new leading axis: `N` tensors of shape `[d…]`
    /// become one of shape `[N, d…]`. Each part is reshaped to `[1, d…]` and
    /// concatenated — so every part must be contiguous and share the same
    /// shape.
    pub fn stack_axis0(parts: &[&Array<T>]) -> Result<Array<T>, ArrayError> {
        if parts.is_empty() {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "stack_axis0: need at least one array",
            )));
        }
        let shape = parts[0].shape().to_vec();
        for p in parts {
            if p.shape() != shape.as_slice() {
                return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                    "stack_axis0: parts must all share the same shape",
                )));
            }
        }
        let mut with_axis = [1usize].to_vec();
        with_axis.extend_from_slice(&shape);
        let reshaped: Result<Vec<Array<T>>, ArrayError> = parts
            .iter()
            .map(|p| p.contiguous_or_self()?.reshape(&with_axis))
            .collect();
        let reshaped = reshaped?;
        let refs: Vec<&Array<T>> = reshaped.iter().collect();
        Array::concat_axis0(&refs)
    }
}
