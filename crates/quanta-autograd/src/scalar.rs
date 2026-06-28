//! The element trait for differentiable arrays.
//!
//! [`Tape`](crate::Tape) is generic over `T: DiffScalar`. In practice the only
//! type that is both `FloatScalar` (transcendentals) and `ReduceScalar`
//! (device sums, needed for the `sum` loss) is `f32` — so `DiffScalar` is f32
//! today. It exists so the matmul VJP (which calls the f32-only
//! `Array::matmul`) can be reached from the otherwise type-generic backward
//! pass without scattering `T == f32` assumptions through the engine.

use quanta_array::{Array, ArrayError, FloatScalar, ReduceScalar};

/// A scalar that supports the full autograd op set, including the linear-algebra
/// ops whose VJPs are themselves matmuls.
pub trait DiffScalar: FloatScalar + ReduceScalar {
    /// 2-D matrix multiply `a (m×k) · b (k×n)` — the forward op and the
    /// building block of the matmul VJP (`Gᵀ`-flavoured products).
    fn array_matmul(a: &Array<Self>, b: &Array<Self>) -> Result<Array<Self>, ArrayError>;
}

impl DiffScalar for f32 {
    fn array_matmul(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        a.matmul(b)
    }
}
