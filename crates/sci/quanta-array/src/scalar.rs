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

use quanta_core::GpuType;

/// A numeric array element: has additive/multiplicative identities and a
/// host-side conversion from `f64` (for `arange`/`linspace` step math).
///
/// Implemented for the ten element types: `f32` / `f64`, `i32` / `u32`,
/// `i64` / `u64`, and the narrow ints `u8` / `i8` / `u16` / `i16`.
/// Integer arithmetic is two's-complement wrapping (mod 2^w) at every
/// store and cast, whatever the width.
pub trait ArrayScalar: GpuType {
    const ZERO: Self;
    const ONE: Self;
    /// Convert from an `f64` index/step value. For integers this is Rust
    /// `as`: truncate toward zero, **saturating** at the type's range
    /// (`Array::<u8>::arange` stepping past 255 clamps to 255).
    fn from_f64(v: f64) -> Self;
}

/// A floating-point array element — gates the math-function ufuncs.
pub trait FloatScalar: ArrayScalar {}

/// An array element with device-wide reductions (`sum`/`min`/`max`),
/// dispatched to the matching `quanta-prims` reduce kernel. Implemented for
/// the types prims provides reduces for: f32 / i32 / u32.
///
/// The other integer dtypes (i64/u64 and the narrow u8/i8/u16/i16) have no
/// whole-array reduce on purpose — a same-width `sum` of a `u8` array wraps
/// at 255, a wrong answer for the type's usual population. The spelling is
/// an explicit widen: `a.astype::<u32>()?.sum()`, which makes the
/// accumulator the caller's choice (numpy's `sum(dtype=…)`). In-dtype
/// whole-array min/max exist via `a.reshape(&[1, n])?.max_axis_last()`.
///
/// The reductions take an on-device [`Field`](quanta_core::Field) so the data
/// never round-trips through host memory — the array stays GPU-resident.
/// The `_resident` forms go one step further: the reduced scalar itself
/// stays on the device as a 1-element field (no readback, no deferred-lane
/// flush), bit-equal to the host-returning forms.
pub trait ReduceScalar: ArrayScalar {
    fn reduce_add(
        gpu: &quanta_core::Gpu,
        data: &quanta_core::Field<Self>,
        n: usize,
    ) -> Result<Self, quanta_core::QuantaError>;
    fn reduce_min(
        gpu: &quanta_core::Gpu,
        data: &quanta_core::Field<Self>,
        n: usize,
    ) -> Result<Self, quanta_core::QuantaError>;
    fn reduce_max(
        gpu: &quanta_core::Gpu,
        data: &quanta_core::Field<Self>,
        n: usize,
    ) -> Result<Self, quanta_core::QuantaError>;
    fn reduce_add_resident(
        gpu: &quanta_core::Gpu,
        data: &quanta_core::Field<Self>,
        n: usize,
    ) -> Result<quanta_core::Field<Self>, quanta_core::QuantaError>;
    fn reduce_min_resident(
        gpu: &quanta_core::Gpu,
        data: &quanta_core::Field<Self>,
        n: usize,
    ) -> Result<quanta_core::Field<Self>, quanta_core::QuantaError>;
    fn reduce_max_resident(
        gpu: &quanta_core::Gpu,
        data: &quanta_core::Field<Self>,
        n: usize,
    ) -> Result<quanta_core::Field<Self>, quanta_core::QuantaError>;
}

macro_rules! reduce_scalar {
    ($t:ty, $add:ident, $min:ident, $max:ident, $add_res:ident, $min_res:ident,
     $max_res:ident) => {
        impl ReduceScalar for $t {
            fn reduce_add(
                gpu: &quanta_core::Gpu,
                data: &quanta_core::Field<Self>,
                n: usize,
            ) -> Result<Self, quanta_core::QuantaError> {
                quanta_prims::$add(gpu, data, n)
            }
            fn reduce_min(
                gpu: &quanta_core::Gpu,
                data: &quanta_core::Field<Self>,
                n: usize,
            ) -> Result<Self, quanta_core::QuantaError> {
                quanta_prims::$min(gpu, data, n)
            }
            fn reduce_max(
                gpu: &quanta_core::Gpu,
                data: &quanta_core::Field<Self>,
                n: usize,
            ) -> Result<Self, quanta_core::QuantaError> {
                quanta_prims::$max(gpu, data, n)
            }
            fn reduce_add_resident(
                gpu: &quanta_core::Gpu,
                data: &quanta_core::Field<Self>,
                n: usize,
            ) -> Result<quanta_core::Field<Self>, quanta_core::QuantaError> {
                quanta_prims::$add_res(gpu, data, n)
            }
            fn reduce_min_resident(
                gpu: &quanta_core::Gpu,
                data: &quanta_core::Field<Self>,
                n: usize,
            ) -> Result<quanta_core::Field<Self>, quanta_core::QuantaError> {
                quanta_prims::$min_res(gpu, data, n)
            }
            fn reduce_max_resident(
                gpu: &quanta_core::Gpu,
                data: &quanta_core::Field<Self>,
                n: usize,
            ) -> Result<quanta_core::Field<Self>, quanta_core::QuantaError> {
                quanta_prims::$max_res(gpu, data, n)
            }
        }
    };
}

reduce_scalar!(
    f32,
    device_reduce_add_f32_field,
    device_reduce_min_f32_field,
    device_reduce_max_f32_field,
    device_reduce_add_f32_resident,
    device_reduce_min_f32_resident,
    device_reduce_max_f32_resident
);
reduce_scalar!(
    i32,
    device_reduce_add_i32_field,
    device_reduce_min_i32_field,
    device_reduce_max_i32_field,
    device_reduce_add_i32_resident,
    device_reduce_min_i32_resident,
    device_reduce_max_i32_resident
);
reduce_scalar!(
    u32,
    device_reduce_add_u32_field,
    device_reduce_min_u32_field,
    device_reduce_max_u32_field,
    device_reduce_add_u32_resident,
    device_reduce_min_u32_resident,
    device_reduce_max_u32_resident
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

int_scalar!(i32, u32, i64, u64, u8, i8, u16, i16);
float_scalar!(f32, f64);
