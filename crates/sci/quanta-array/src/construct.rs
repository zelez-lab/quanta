//! Array construction — numpy-style builders.
//!
//! Generic builders (`from_slice`, `from_vec`, `full`) work for any
//! [`GpuType`]. The numeric builders (`zeros`/`ones`/`arange`/`linspace`/
//! `eye`) are generic over [`ArrayScalar`] (f32/f64/i32/u32/i64/u64).

use quanta_core::{Gpu, GpuType};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

impl<T: GpuType> Array<T> {
    /// Build a contiguous row-major array from a host slice + shape. The
    /// slice length must equal the shape's element count.
    pub fn from_slice(gpu: &Gpu, data: &[T], shape: &[usize]) -> Result<Array<T>, ArrayError> {
        let layout = Layout::row_major(shape)?;
        let n = layout.linear_size();
        if data.len() != n {
            return Err(ArrayError::LengthMismatch {
                expected: n,
                got: data.len(),
            });
        }
        let field = gpu.field::<T>(n)?;
        field.write(data)?;
        Ok(Array::from_parts(gpu.clone(), field, layout))
    }

    /// Build from an owned `Vec` + shape.
    pub fn from_vec(gpu: &Gpu, data: Vec<T>, shape: &[usize]) -> Result<Array<T>, ArrayError> {
        Array::from_slice(gpu, &data, shape)
    }

    /// A contiguous array of `shape` filled with `value`.
    pub fn full(gpu: &Gpu, value: T, shape: &[usize]) -> Result<Array<T>, ArrayError> {
        let layout = Layout::row_major(shape)?;
        let n = layout.linear_size();
        let data = vec![value; n];
        let field = gpu.field::<T>(n)?;
        field.write(&data)?;
        Ok(Array::from_parts(gpu.clone(), field, layout))
    }
}

impl<T: ArrayScalar> Array<T> {
    /// `shape`-shaped array of zeros.
    pub fn zeros(gpu: &Gpu, shape: &[usize]) -> Result<Array<T>, ArrayError> {
        Array::full(gpu, T::ZERO, shape)
    }

    /// `shape`-shaped array of ones.
    pub fn ones(gpu: &Gpu, shape: &[usize]) -> Result<Array<T>, ArrayError> {
        Array::full(gpu, T::ONE, shape)
    }

    /// 1-D array `[start, start+step, …)` of length `n` (numpy `arange`
    /// with an explicit count + step). Step math is done in `f64` and
    /// converted to `T` (truncating for integers).
    pub fn arange(gpu: &Gpu, start: f64, step: f64, n: usize) -> Result<Array<T>, ArrayError> {
        let data: Vec<T> = (0..n)
            .map(|i| T::from_f64(start + step * i as f64))
            .collect();
        Array::from_slice(gpu, &data, &[n])
    }

    /// 1-D array of `n` evenly-spaced values from `start` to `stop`
    /// inclusive (numpy `linspace`). `n == 1` yields `[start]`.
    pub fn linspace(gpu: &Gpu, start: f64, stop: f64, n: usize) -> Result<Array<T>, ArrayError> {
        let data: Vec<T> = if n == 0 {
            Vec::new()
        } else if n == 1 {
            vec![T::from_f64(start)]
        } else {
            let denom = (n - 1) as f64;
            (0..n)
                .map(|i| T::from_f64(start + (stop - start) * (i as f64 / denom)))
                .collect()
        };
        Array::from_slice(gpu, &data, &[n])
    }

    /// `n × n` identity matrix.
    pub fn eye(gpu: &Gpu, n: usize) -> Result<Array<T>, ArrayError> {
        let mut data = vec![T::ZERO; n * n];
        for (i, slot) in data.iter_mut().enumerate().step_by(n + 1).take(n) {
            let _ = i;
            *slot = T::ONE;
        }
        Array::from_slice(gpu, &data, &[n, n])
    }
}
