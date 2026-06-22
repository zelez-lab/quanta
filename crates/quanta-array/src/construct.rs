//! Array construction — numpy-style builders.
//!
//! Generic builders (`from_slice`, `from_vec`, `full`) work for any
//! [`GpuType`]. The numeric builders (`zeros`/`ones`/`arange`/`linspace`/
//! `eye`) are provided for `f32` first (the dtype matrix follows once the
//! ufunc path is generalized); `zeros`/`ones` also have generic forms
//! gated on a small scalar helper.

use quanta::{Gpu, GpuType};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;

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

impl Array<f32> {
    /// `shape`-shaped array of zeros.
    pub fn zeros(gpu: &Gpu, shape: &[usize]) -> Result<Array<f32>, ArrayError> {
        Array::full(gpu, 0.0, shape)
    }

    /// `shape`-shaped array of ones.
    pub fn ones(gpu: &Gpu, shape: &[usize]) -> Result<Array<f32>, ArrayError> {
        Array::full(gpu, 1.0, shape)
    }

    /// 1-D array `[start, start+step, …)` of length `n` (numpy `arange`
    /// with an explicit count + step).
    pub fn arange(gpu: &Gpu, start: f32, step: f32, n: usize) -> Result<Array<f32>, ArrayError> {
        let data: Vec<f32> = (0..n).map(|i| start + step * i as f32).collect();
        Array::from_slice(gpu, &data, &[n])
    }

    /// 1-D array of `n` evenly-spaced values from `start` to `stop`
    /// inclusive (numpy `linspace`). `n == 1` yields `[start]`.
    pub fn linspace(gpu: &Gpu, start: f32, stop: f32, n: usize) -> Result<Array<f32>, ArrayError> {
        let data: Vec<f32> = if n == 0 {
            Vec::new()
        } else if n == 1 {
            vec![start]
        } else {
            let denom = (n - 1) as f32;
            (0..n)
                .map(|i| start + (stop - start) * (i as f32 / denom))
                .collect()
        };
        Array::from_slice(gpu, &data, &[n])
    }

    /// `n × n` identity matrix.
    pub fn eye(gpu: &Gpu, n: usize) -> Result<Array<f32>, ArrayError> {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Array::from_slice(gpu, &data, &[n, n])
    }
}
