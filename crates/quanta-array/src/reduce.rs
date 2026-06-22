//! Whole-array reductions.
//!
//! `sum` / `min` / `max` wrap `quanta-prims` device-wide reduce kernels
//! (generic over [`ReduceScalar`] — f32 / i32 / u32); `mean` derives from
//! `sum` and is floating-point only. These reduce the **entire** array to a
//! scalar (numpy `arr.sum()` with no axis). Per-axis reductions are a later
//! increment (they need a segmented/strided reduce shape).
//!
//! The prims reduce takes a host slice (it uploads internally), so the
//! array's logical row-major data is materialized first via `to_vec`.
//! `prod` is deferred — prims has add/min/max device reduces but no
//! multiply-reduce yet.

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::{FloatScalar, ReduceScalar};

impl<T: ReduceScalar> Array<T> {
    /// Sum of all elements (numpy `arr.sum()`). Float sums use tree
    /// reduction order, so expect a few ULP of drift vs a sequential fold.
    pub fn sum(&self) -> Result<T, ArrayError> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Ok(T::ZERO);
        }
        Ok(T::reduce_add(self.gpu(), &data)?)
    }

    /// Minimum element (numpy `arr.min()`). Errors on an empty array.
    pub fn min(&self) -> Result<T, ArrayError> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "min of an empty array",
            )));
        }
        Ok(T::reduce_min(self.gpu(), &data)?)
    }

    /// Maximum element (numpy `arr.max()`). Errors on an empty array.
    pub fn max(&self) -> Result<T, ArrayError> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "max of an empty array",
            )));
        }
        Ok(T::reduce_max(self.gpu(), &data)?)
    }
}

impl<T: FloatScalar + ReduceScalar + ToF64> Array<T> {
    /// Arithmetic mean of all elements (`sum / len`). Floating-point only.
    pub fn mean(&self) -> Result<T, ArrayError> {
        let n = self.len();
        if n == 0 {
            return Ok(T::from_f64(f64::NAN));
        }
        // Sum + divide in f64, then convert back — keeps `mean` generic
        // without putting arithmetic bounds on the scalar traits.
        Ok(T::from_f64(self.sum()?.to_f64() / n as f64))
    }
}

/// Convert a float scalar to `f64` (for `mean`'s division).
pub trait ToF64 {
    fn to_f64(self) -> f64;
}
impl ToF64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
