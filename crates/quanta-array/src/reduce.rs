//! Whole-array reductions.
//!
//! `sum` / `min` / `max` wrap `quanta-prims` device-wide reduce kernels;
//! `mean` derives from `sum`. These reduce the **entire** array to a
//! scalar (numpy `arr.sum()` with no axis). Per-axis reductions are a
//! later increment (they need a segmented/strided reduce shape).
//!
//! The prims reduce takes a host slice (it uploads internally), so the
//! array's logical row-major data is materialized first via `to_vec`.
//! `prod` is deferred — `quanta-prims` has add/min/max device reduces but
//! no multiply-reduce yet.

use crate::array::Array;
use crate::error::ArrayError;

impl Array<f32> {
    /// Sum of all elements (numpy `arr.sum()`). Tree-reduction order, so
    /// expect a few ULP of drift vs a sequential fold for large arrays.
    pub fn sum(&self) -> Result<f32, ArrayError> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Ok(0.0);
        }
        Ok(quanta_prims::device_reduce_add_f32(self.gpu(), &data)?)
    }

    /// Arithmetic mean of all elements (`sum / len`).
    pub fn mean(&self) -> Result<f32, ArrayError> {
        let n = self.len();
        if n == 0 {
            return Ok(f32::NAN);
        }
        Ok(self.sum()? / n as f32)
    }

    /// Minimum element (numpy `arr.min()`).
    pub fn min(&self) -> Result<f32, ArrayError> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Ok(f32::INFINITY);
        }
        Ok(quanta_prims::device_reduce_min_f32(self.gpu(), &data)?)
    }

    /// Maximum element (numpy `arr.max()`).
    pub fn max(&self) -> Result<f32, ArrayError> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Ok(f32::NEG_INFINITY);
        }
        Ok(quanta_prims::device_reduce_max_f32(self.gpu(), &data)?)
    }
}
