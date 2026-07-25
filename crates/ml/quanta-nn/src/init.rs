//! Named weight initializers — the standalone family behind every
//! layer's `init`, and the way to seed custom parameter trees.
//!
//! Deterministic per [`Key`]: the same key always yields the same
//! tensor, on every backend (the dropout discipline — no global RNG,
//! no init-order hazard). Each sample CONSUMES its key.
//!
//! The scaled variants derive their fans from the tensor shape by the
//! standard convention (see [`fans`]): rank-2 `[in, out]` reads the
//! dims directly, rank-4 `[Cout, Cin, kh, kw]` folds the receptive
//! field into both fans — exactly the formulas [`crate::layer::Linear`]
//! and [`crate::conv::Conv2d`] shipped with, which now delegate here.
//! Layers keep their defaults; a custom scheme is applied by building
//! the params struct yourself:
//!
//! ```ignore
//! let w = Init::XavierNormal.sample(&gpu, kw, &[in_dim, out_dim])?;
//! let params = LinearParams { w, b: None };
//! ```

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar};
use quanta_core::{Gpu, QuantaError};

use crate::layer::Key;

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(QuantaError::invalid_param(msg)))
}

/// A named initialization scheme. `sample` draws a tensor;
/// [`Init::sample_with_fans`] overrides the shape-derived fans for
/// exotic layouts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Init {
    Zeros,
    Ones,
    /// Uniform in `[lo, hi)`, unscaled.
    Uniform {
        lo: f32,
        hi: f32,
    },
    /// Gaussian via Box-Muller, unscaled.
    Normal {
        mean: f32,
        std: f32,
    },
    /// Glorot/Xavier uniform: `±√(6 / (fan_in + fan_out))` — keeps
    /// forward AND backward variance flat through linear+tanh-era
    /// stacks.
    XavierUniform,
    /// Glorot/Xavier normal: `std = √(2 / (fan_in + fan_out))`.
    XavierNormal,
    /// He/Kaiming uniform: `±√(6 / fan_in)` — the ReLU-family gain
    /// (√2) folded in; the default `Linear`/`Conv2d` scheme.
    KaimingUniform,
    /// He/Kaiming normal: `std = √(2 / fan_in)`.
    KaimingNormal,
}

/// The `(fan_in, fan_out)` a shape implies, by the standard
/// convention:
///
/// - rank 1 `[n]`      → `(n, n)`
/// - rank 2 `[in, out]` → `(in, out)` (this crate's Linear layout:
///   `y = x @ w`)
/// - rank ≥ 3 `[Cout, Cin, k…]` → `(Cin·∏k, Cout·∏k)` (the NCHW conv
///   weight layout — the receptive field multiplies both fans)
pub fn fans(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (1, 1),
        [n] => (*n, *n),
        [i, o] => (*i, *o),
        [cout, cin, rest @ ..] => {
            let rf: usize = rest.iter().product::<usize>().max(1);
            (cin * rf, cout * rf)
        }
    }
}

impl Init {
    /// Draw a `shape` tensor, deriving fans from the shape (see
    /// [`fans`]). Consumes `key`.
    pub fn sample<T: DiffScalar + ToF64>(
        &self,
        gpu: &Gpu,
        key: Key,
        shape: &[usize],
    ) -> Result<Array<T>, AutogradError> {
        let (fan_in, fan_out) = fans(shape);
        self.sample_with_fans(gpu, key, shape, fan_in, fan_out)
    }

    /// Draw a `shape` tensor with explicit fans — for layouts the
    /// shape convention cannot see (grouped/transposed weights).
    /// Consumes `key`.
    pub fn sample_with_fans<T: DiffScalar + ToF64>(
        &self,
        gpu: &Gpu,
        key: Key,
        shape: &[usize],
        fan_in: usize,
        fan_out: usize,
    ) -> Result<Array<T>, AutogradError> {
        let n: usize = shape.iter().product();
        if n == 0 {
            return Err(bad("init: empty shape"));
        }
        if fan_in == 0 || fan_out == 0 {
            return Err(bad("init: fans must be nonzero"));
        }
        let host: Vec<f32> = match *self {
            Init::Zeros => vec![0.0; n],
            Init::Ones => vec![1.0; n],
            Init::Uniform { lo, hi } => key.uniform(n, lo, hi),
            Init::Normal { mean, std } => key.normal(n, mean, std),
            Init::XavierUniform => {
                let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
                key.uniform(n, -bound, bound)
            }
            Init::XavierNormal => {
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                key.normal(n, 0.0, std)
            }
            Init::KaimingUniform => {
                let bound = (6.0 / fan_in as f32).sqrt();
                key.uniform(n, -bound, bound)
            }
            Init::KaimingNormal => {
                let std = (2.0 / fan_in as f32).sqrt();
                key.normal(n, 0.0, std)
            }
        };
        let t_host: Vec<T> = host.iter().map(|&v| T::from_f64(v as f64)).collect();
        Array::from_slice(gpu, &t_host, shape).map_err(AutogradError::from)
    }
}
