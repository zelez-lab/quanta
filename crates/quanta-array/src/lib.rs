//! # quanta-array — the NumPy-equivalent GPU array
//!
//! `Array<T>` is a host-side N-dimensional array backed by GPU memory: it
//! owns a [`quanta::Field`] and carries a [`quanta_tensor::Layout`]
//! (shape and strides). It exposes numpy-style construction, broadcasting
//! ufuncs, reductions, and zero-copy shape manipulation — without the user
//! writing kernels.
//!
//! ## Design (step 084.2)
//!
//! - **Shape/stride algebra** is delegated to `quanta-tensor` (proven in
//!   Lean), so `reshape` / `permute` / `transpose` / `broadcast` are pure
//!   host operations on the layout that produce zero-copy views over the
//!   same `Field`.
//! - **Elementwise ufuncs** build a [`quanta_ir::KernelDef`] at runtime and
//!   dispatch it through `quanta`'s JIT path (`wave_jit`). Broadcasting is
//!   lowered to strided indexing in the generated kernel.
//! - **Reductions** wrap `quanta-prims` device-wide reduce kernels.
//! - It is a **compute-only** consumer of `quanta` (render off — the
//!   headless shape from step 085).
//!
//! ```ignore
//! use quanta_array::Array;
//! let gpu = quanta::init_cpu();
//! let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
//! let b = Array::ones(&gpu, &[2, 2])?;
//! let c = a.add(&b)?;          // broadcasting elementwise add
//! let s = c.sum()?;           // whole-array reduction
//! assert_eq!(c.to_vec()?, vec![2.0, 3.0, 4.0, 5.0]);
//! ```
//!
//! ## Dtype safety
//!
//! Construction and arithmetic are generic over every numeric dtype, but the
//! transcendental math ufuncs (`sqrt`/`exp`/`sin`/…) are floating-point only.
//! Calling one on an integer array is a **compile error**, not a silent wrong
//! result:
//!
//! ```compile_fail
//! let g = quanta::init_cpu();
//! let a = quanta_array::Array::from_slice(&g, &[4i32, 9, 16], &[3]).unwrap();
//! let _ = a.sqrt(); // ERROR: `i32: FloatScalar` is not satisfied
//! ```

mod array;
mod broadcast;
mod construct;
mod conv;
mod error;
mod gather;
mod index;
mod linalg;
mod pad;
mod pool;
mod reduce;
mod reduce_axis;
mod scalar;
mod ufunc;

pub use array::Array;
pub use conv::conv_out;
pub use error::ArrayError;
pub use scalar::{ArrayScalar, FloatScalar, ReduceScalar};

// Re-export the layout vocabulary so users name shapes through quanta-array.
pub use quanta_tensor::{Layout, Shape};
