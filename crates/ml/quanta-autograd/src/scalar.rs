//! The element trait for differentiable arrays.
//!
//! [`Tape`](crate::Tape) is generic over `T: DiffScalar`. In practice the only
//! type that is both `FloatScalar` (transcendentals) and `ReduceScalar`
//! (device sums, needed for the `sum` loss) is `f32` — so `DiffScalar` is f32
//! today. It exists so the matmul VJP (which calls the f32-only
//! `Array::matmul`) can be reached from the otherwise type-generic backward
//! pass without scattering `T == f32` assumptions through the engine.

use quanta_array::{Array, ArrayError, FloatScalar, ReduceScalar};
use quanta_core::{Field, Gpu};

/// A scalar that supports the full autograd op set, including the linear-algebra
/// ops whose VJPs are themselves matmuls.
pub trait DiffScalar: FloatScalar + ReduceScalar {
    /// 2-D matrix multiply `a (m×k) · b (k×n)` — the forward op and the
    /// building block of the matmul VJP (`Gᵀ`-flavoured products).
    fn array_matmul(a: &Array<Self>, b: &Array<Self>) -> Result<Array<Self>, ArrayError>;

    /// This array seen as `f32` — `Some` only when `Self` IS `f32`
    /// (identity, zero-copy). The hook fused f32 kernels use to bind
    /// an input's own device buffer instead of staging a converted
    /// copy; a non-f32 scalar returns `None` and takes the staged
    /// path.
    fn as_f32_array(a: &Array<Self>) -> Option<&Array<f32>> {
        let _ = a;
        None
    }

    /// Adopt an `f32` field a fused kernel wrote as an `Array<Self>`
    /// of `shape`. `f32` adopts the buffer zero-copy
    /// ([`Array::from_field`]); any other scalar converts through the
    /// host (a completion point — the read finishes pending producers
    /// of the field).
    fn array_from_f32_field(
        gpu: &Gpu,
        field: Field<f32>,
        shape: &[usize],
    ) -> Result<Array<Self>, ArrayError> {
        let host = field.read().map_err(ArrayError::Gpu)?;
        let converted: Vec<Self> = host.iter().map(|&x| Self::from_f64(x as f64)).collect();
        Array::from_slice(gpu, &converted, shape)
    }

    /// Adopt an `Array<f32>` an f32 device pipeline produced as an
    /// `Array<Self>` — the inverse of [`DiffScalar::as_f32_array`].
    /// Identity (zero-copy, no completion point) when `Self` IS `f32`;
    /// any other scalar converts through the host (a completion point,
    /// like [`DiffScalar::array_from_f32_field`]).
    fn array_from_f32(a: Array<f32>) -> Result<Array<Self>, ArrayError> {
        let shape = a.shape().to_vec();
        let host = a.to_vec()?;
        let converted: Vec<Self> = host.iter().map(|&x| Self::from_f64(x as f64)).collect();
        Array::from_slice(a.gpu(), &converted, &shape)
    }
}

impl DiffScalar for f32 {
    fn array_matmul(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        a.matmul(b)
    }

    fn as_f32_array(a: &Array<f32>) -> Option<&Array<f32>> {
        Some(a)
    }

    fn array_from_f32_field(
        gpu: &Gpu,
        field: Field<f32>,
        shape: &[usize],
    ) -> Result<Array<f32>, ArrayError> {
        Array::from_field(gpu, field, shape)
    }

    fn array_from_f32(a: Array<f32>) -> Result<Array<f32>, ArrayError> {
        Ok(a)
    }
}
