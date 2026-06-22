//! Scalar-element traits for `Array<T>`.
//!
//! `GpuType` (from `quanta`) says a type can be a kernel scalar, but it
//! carries no numeric values or operations. quanta-array layers two small
//! traits on top:
//!
//! - [`ArrayScalar`] — numeric elements with `ZERO`/`ONE` (the all-dtype
//!   surface: construction + arithmetic ufuncs).
//! - [`FloatScalar`] — floating-point elements only; gates the
//!   transcendental math ufuncs (`sqrt`/`exp`/`sin`/…) which are
//!   float-only on every backend. Calling them on an integer array is a
//!   compile error, not a silent garbage result.

use quanta::GpuType;

/// A numeric array element: has additive/multiplicative identities and a
/// host-side conversion from `f64` (for `arange`/`linspace` step math).
pub trait ArrayScalar: GpuType {
    const ZERO: Self;
    const ONE: Self;
    /// Convert from an `f64` index/step value (truncating for integers).
    fn from_f64(v: f64) -> Self;
}

/// A floating-point array element — gates the math-function ufuncs.
pub trait FloatScalar: ArrayScalar {}

/// An array element with device-wide reductions (`sum`/`min`/`max`),
/// dispatched to the matching `quanta-prims` reduce kernel. Implemented for
/// the types prims provides reduces for: f32 / i32 / u32.
pub trait ReduceScalar: ArrayScalar {
    fn reduce_add(gpu: &quanta::Gpu, data: &[Self]) -> Result<Self, quanta::QuantaError>;
    fn reduce_min(gpu: &quanta::Gpu, data: &[Self]) -> Result<Self, quanta::QuantaError>;
    fn reduce_max(gpu: &quanta::Gpu, data: &[Self]) -> Result<Self, quanta::QuantaError>;
}

macro_rules! reduce_scalar {
    ($t:ty, $add:ident, $min:ident, $max:ident) => {
        impl ReduceScalar for $t {
            fn reduce_add(gpu: &quanta::Gpu, data: &[Self]) -> Result<Self, quanta::QuantaError> {
                quanta_prims::$add(gpu, data)
            }
            fn reduce_min(gpu: &quanta::Gpu, data: &[Self]) -> Result<Self, quanta::QuantaError> {
                quanta_prims::$min(gpu, data)
            }
            fn reduce_max(gpu: &quanta::Gpu, data: &[Self]) -> Result<Self, quanta::QuantaError> {
                quanta_prims::$max(gpu, data)
            }
        }
    };
}

reduce_scalar!(
    f32,
    device_reduce_add_f32,
    device_reduce_min_f32,
    device_reduce_max_f32
);
reduce_scalar!(
    i32,
    device_reduce_add_i32,
    device_reduce_min_i32,
    device_reduce_max_i32
);
reduce_scalar!(
    u32,
    device_reduce_add_u32,
    device_reduce_min_u32,
    device_reduce_max_u32
);

macro_rules! int_scalar {
    ($($t:ty),*) => {$(
        impl ArrayScalar for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            fn from_f64(v: f64) -> Self { v as $t }
        }
    )*};
}
macro_rules! float_scalar {
    ($($t:ty),*) => {$(
        impl ArrayScalar for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            fn from_f64(v: f64) -> Self { v as $t }
        }
        impl FloatScalar for $t {}
    )*};
}

int_scalar!(i32, u32, i64, u64);
float_scalar!(f32, f64);
