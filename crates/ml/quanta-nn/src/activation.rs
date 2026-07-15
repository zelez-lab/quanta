//! Fused activations — rowwise softmax / log-softmax, GeLU, SwiGLU — plus
//! the zero-parameter module forms of the whole activation family.
//!
//! The rowwise pair is max-stabilized: T9223 says the subtraction is exact,
//! not an approximation, and the backward formulas are the proven adjoints
//! (T9224 `dx = p⊙(g − ⟨g,p⟩)`, T9225 `dx = g − (∑g)·p`). GeLU is the
//! tanh-approximation (GPT-2 form, matching the composed [`Var::gelu`]
//! oracle); its backward reuses the tanh the forward saved — T9227's sech²
//! identity is why no cosh is ever evaluated. SwiGLU gates the first half
//! of the features by the silu of the second-half pair (`[N, 2H] → [N, H]`);
//! its backward derives σ′ from the forward's sigmoid via T9226.
//!
//! Module forms are layers with `Params = ()` — they occupy a slot in a
//! tuple stack, contribute no leaves to the parameter tree, and consume no
//! keys.

use crate::functional::{f32_field_to_array, lift, to_f32_host};
use crate::layer::{Key, Layer};
use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
pub(crate) mod dsl {
    use quanta_core::*;

    /// Rowwise softmax stats: one thread per row streams the C channels
    /// and writes `stats[i*2] = m` (row max — seeded from the first
    /// element, never a sentinel), `stats[i*2+1] = l = Σ exp(x−m)`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn sm_stats(x: &[f32], stats: &mut [f32], n: u32, c: u32) {
        let i = quark_id();
        let row = if i < n { i } else { 0u32 };
        let base = row * c;

        let mut m: f32 = x[base as usize];
        let mut p: u32 = 1u32;
        while p < c {
            m = fmax(m, x[(base + p) as usize]);
            p = p + 1u32;
        }
        let mut l: f32 = 0.0f32;
        let mut q: u32 = 0u32;
        while q < c {
            l = l + exp(fmin(x[(base + q) as usize] - m, 0.0f32));
            q = q + 1u32;
        }
        if i < n {
            stats[(row * 2u32) as usize] = m;
            stats[(row * 2u32 + 1u32) as usize] = l;
        }
    }

    /// Rowwise softmax / log-softmax forward, elementwise from the stats:
    /// `logf = 0` → `exp(x−m)/l`; `logf = 1` → `(x−m) − ln l` (T9223).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn sm_fwd(x: &[f32], stats: &[f32], out: &mut [f32], n: u32, c: u32, logf: f32) {
        let i = quark_id();
        let total = n * c;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / c;
        let m = stats[(row * 2u32) as usize];
        let l = stats[(row * 2u32 + 1u32) as usize];
        let e = fmin(x[idx as usize] - m, 0.0f32);
        let y = logf * (e - ln(l)) + (1.0f32 - logf) * (exp(e) / l);
        if i < total {
            out[idx as usize] = y;
        }
    }

    /// Backward row reduction over the saved forward `y`:
    /// `logf = 0` → `r = ⟨g, y⟩` (softmax, `y = p`);
    /// `logf = 1` → `r = Σ g` (log-softmax).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn sm_bwd_rows(g: &[f32], y: &[f32], r: &mut [f32], n: u32, c: u32, logf: f32) {
        let i = quark_id();
        let row = if i < n { i } else { 0u32 };
        let base = row * c;
        let mut dot: f32 = 0.0f32;
        let mut sum: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < c {
            dot = dot + g[(base + p) as usize] * y[(base + p) as usize];
            sum = sum + g[(base + p) as usize];
            p = p + 1u32;
        }
        if i < n {
            r[row as usize] = logf * sum + (1.0f32 - logf) * dot;
        }
    }

    /// Backward elementwise, the proven adjoints:
    /// `logf = 0` → `dx = y·(g − r)` (T9224, `y = p`, `r = ⟨g,p⟩`);
    /// `logf = 1` → `dx = g − r·exp(y)` (T9225, `exp(y) = p`, `r = Σg`).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn sm_bwd_dx(y: &[f32], g: &[f32], r: &[f32], dx: &mut [f32], n: u32, c: u32, logf: f32) {
        let i = quark_id();
        let total = n * c;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / c;
        let rv = r[row as usize];
        let gv = g[idx as usize];
        let yv = y[idx as usize];
        let d = logf * (gv - rv * exp(fmin(yv, 0.0f32))) + (1.0f32 - logf) * (yv * (gv - rv));
        if i < total {
            dx[idx as usize] = d;
        }
    }

    /// GeLU forward (tanh approximation, GPT-2 form). Saves `tanh(u)` for
    /// the backward. `u` is clamped to ±15 where tanh is ±1 to f32
    /// precision, so the `exp` can never overflow.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn gelu_fwd(x: &[f32], out: &mut [f32], tsave: &mut [f32], n: u32) {
        let i = quark_id();
        let idx = if i < n { i } else { 0u32 };
        let xv = x[idx as usize];
        let u = 0.7978845608028654f32 * (xv + 0.044715f32 * xv * xv * xv);
        let uc = fmin(fmax(u, -15.0f32), 15.0f32);
        let e2 = exp(2.0f32 * uc);
        let t = (e2 - 1.0f32) / (e2 + 1.0f32);
        if i < n {
            out[idx as usize] = 0.5f32 * xv * (1.0f32 + t);
            tsave[idx as usize] = t;
        }
    }

    /// GeLU backward from the saved tanh: `sech²u = 1 − t²` (T9227), so
    /// `dx = g·(0.5(1+t) + 0.5x·(1−t²)·u′)` with
    /// `u′ = √(2/π)·(1 + 3·0.044715·x²)`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn gelu_bwd(x: &[f32], tsave: &[f32], g: &[f32], dx: &mut [f32], n: u32) {
        let i = quark_id();
        let idx = if i < n { i } else { 0u32 };
        let xv = x[idx as usize];
        let t = tsave[idx as usize];
        let du = 0.7978845608028654f32 * (1.0f32 + 0.134145f32 * xv * xv);
        let sech2 = 1.0f32 - t * t;
        let d = 0.5f32 * (1.0f32 + t) + 0.5f32 * xv * sech2 * du;
        if i < n {
            dx[idx as usize] = g[idx as usize] * d;
        }
    }

    /// SwiGLU forward: `out[·, j] = silu(a)·b` with `a = x[·, j]`,
    /// `b = x[·, h + j]` (`x` is `[N, 2H]`, `out` is `[N, H]`). The exp
    /// argument is clamped ≤ 30 so saturation stays finite.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn swiglu_fwd(x: &[f32], out: &mut [f32], n: u32, h: u32) {
        let i = quark_id();
        let total = n * h;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / h;
        let col = idx % h;
        let base = row * 2u32 * h;
        let a = x[(base + col) as usize];
        let b = x[(base + h + col) as usize];
        let s = 1.0f32 / (1.0f32 + exp(fmin(-a, 30.0f32)));
        if i < total {
            out[idx as usize] = a * s * b;
        }
    }

    /// SwiGLU backward: `da = g·b·s·(1 + a·(1−s))` (σ′ from the forward's
    /// sigmoid, T9226), `db = g·a·s`. One thread writes both halves of the
    /// `[N, 2H]` gradient.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn swiglu_bwd(x: &[f32], g: &[f32], dx: &mut [f32], n: u32, h: u32) {
        let i = quark_id();
        let total = n * h;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / h;
        let col = idx % h;
        let base = row * 2u32 * h;
        let a = x[(base + col) as usize];
        let b = x[(base + h + col) as usize];
        let gv = g[idx as usize];
        let s = 1.0f32 / (1.0f32 + exp(fmin(-a, 30.0f32)));
        let da = gv * b * s * (1.0f32 + a * (1.0f32 - s));
        let db = gv * a * s;
        if i < total {
            dx[(base + col) as usize] = da;
            dx[(base + h + col) as usize] = db;
        }
    }
}

fn check(cond: bool, msg: &'static str) -> Result<(), QuantaError> {
    if cond {
        Ok(())
    } else {
        Err(QuantaError::invalid_param(msg))
    }
}

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(QuantaError::invalid_param(msg)))
}

/// Host dispatch: rowwise softmax (`logf = 0`) / log-softmax (`logf = 1`)
/// forward over `n×c`, writing the output and the `(m, l)` stats.
pub fn softmax_forward(
    gpu: &Gpu,
    n: u32,
    c: u32,
    logf: f32,
    x: &Field<f32>,
    out: &Field<f32>,
    stats: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, cu) = (n as usize, c as usize);
    check(x.len() == nu * cu, "softmax: X length must be n*c")?;
    check(out.len() == nu * cu, "softmax: OUT length must be n*c")?;
    check(stats.len() == nu * 2, "softmax: STATS length must be n*2")?;
    if n == 0 || c == 0 {
        return Ok(());
    }
    let mut w = dsl::sm_stats(gpu)?;
    w.bind(0, x);
    w.bind(1, stats);
    w.set_value(2, n);
    w.set_value(3, c);
    gpu.dispatch(&w, n)?.wait()?;

    let mut w = dsl::sm_fwd(gpu)?;
    w.bind(0, x);
    w.bind(1, stats);
    w.bind(2, out);
    w.set_value(3, n);
    w.set_value(4, c);
    w.set_value(5, logf);
    gpu.dispatch(&w, n * c)?.wait()?;
    Ok(())
}

/// Host dispatch: the proven-adjoint backward from the saved forward `y`.
#[allow(clippy::too_many_arguments)]
pub fn softmax_backward(
    gpu: &Gpu,
    n: u32,
    c: u32,
    logf: f32,
    y: &Field<f32>,
    g: &Field<f32>,
    r: &Field<f32>,
    dx: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, cu) = (n as usize, c as usize);
    check(
        y.len() == nu * cu && g.len() == nu * cu && dx.len() == nu * cu,
        "softmax_bwd: Y/G/DX length must be n*c",
    )?;
    check(r.len() == nu, "softmax_bwd: R length must be n")?;
    if n == 0 || c == 0 {
        return Ok(());
    }
    let mut w = dsl::sm_bwd_rows(gpu)?;
    w.bind(0, g);
    w.bind(1, y);
    w.bind(2, r);
    w.set_value(3, n);
    w.set_value(4, c);
    w.set_value(5, logf);
    gpu.dispatch(&w, n)?.wait()?;

    let mut w = dsl::sm_bwd_dx(gpu)?;
    w.bind(0, y);
    w.bind(1, g);
    w.bind(2, r);
    w.bind(3, dx);
    w.set_value(4, n);
    w.set_value(5, c);
    w.set_value(6, logf);
    gpu.dispatch(&w, n * c)?.wait()?;
    Ok(())
}

// ── Tape-differentiable wrappers ─────────────────────────────────────────

fn upload(gpu: &Gpu, host: &[f32]) -> Result<Field<f32>, AutogradError> {
    let f = gpu.field::<f32>(host.len()).map_err(lift)?;
    f.write(host).map_err(lift)?;
    Ok(f)
}

fn softmax_var_impl<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
    logf: f32,
) -> Result<Var<T>, AutogradError> {
    let xs = x.value().shape().to_vec();
    if xs.len() != 2 {
        return Err(bad("softmax_var: input must be 2-D [N, C]"));
    }
    let (n, c) = (xs[0], xs[1]);
    let gpu = x.value().gpu().clone();
    let x_f32 = to_f32_host(&x.value())?;

    let y_f32 = {
        let xf = upload(&gpu, &x_f32)?;
        let of = gpu.field::<f32>(n * c).map_err(lift)?;
        let sf = gpu.field::<f32>(n.max(1) * 2).map_err(lift)?;
        softmax_forward(&gpu, n as u32, c as u32, logf, &xf, &of, &sf).map_err(lift)?;
        of.read().map_err(lift)?
    };

    let out_t: Vec<T> = y_f32.iter().map(|&v| T::from_f64(v as f64)).collect();
    let out_arr = Array::from_slice(&gpu, &out_t, &[n, c]).map_err(AutogradError::from)?;

    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let g_f32 = to_f32_host(g)?;
        let yf = upload(&gpu_b, &y_f32)?;
        let gf = upload(&gpu_b, &g_f32)?;
        let rf = gpu_b.field::<f32>(n.max(1)).map_err(lift)?;
        let dxf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        softmax_backward(&gpu_b, n as u32, c as u32, logf, &yf, &gf, &rf, &dxf).map_err(lift)?;
        let dx = f32_field_to_array::<T>(&gpu_b, &dxf, &[n, c])?;
        Ok(vec![dx])
    };

    Ok(tape.custom_vjp(&[x], out_arr, backward))
}

/// Fused rowwise softmax over `[N, C]` (T9223 stabilization, T9224
/// backward). The composed [`Var::softmax`] is the differential oracle.
pub fn softmax_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
) -> Result<Var<T>, AutogradError> {
    softmax_var_impl(tape, x, 0.0)
}

/// Fused rowwise log-softmax over `[N, C]` (T9223, T9225). The composed
/// [`Var::log_softmax`] is the differential oracle.
pub fn log_softmax_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
) -> Result<Var<T>, AutogradError> {
    softmax_var_impl(tape, x, 1.0)
}

/// Fused GeLU (tanh approximation), any shape. The backward reuses the
/// forward's saved tanh (T9227). The composed [`Var::gelu`] — the same
/// GPT-2 form — is the differential oracle.
pub fn gelu_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
) -> Result<Var<T>, AutogradError> {
    let shape = x.value().shape().to_vec();
    let gpu = x.value().gpu().clone();
    let x_f32 = to_f32_host(&x.value())?;
    let n = x_f32.len();
    if n == 0 {
        return Err(bad("gelu_var: empty input"));
    }

    let (y_f32, t_f32) = {
        let xf = upload(&gpu, &x_f32)?;
        let of = gpu.field::<f32>(n).map_err(lift)?;
        let tf = gpu.field::<f32>(n).map_err(lift)?;
        let mut w = dsl::gelu_fwd(&gpu).map_err(lift)?;
        w.bind(0, &xf);
        w.bind(1, &of);
        w.bind(2, &tf);
        w.set_value(3, n as u32);
        gpu.dispatch(&w, n as u32)
            .map_err(lift)?
            .wait()
            .map_err(lift)?;
        (of.read().map_err(lift)?, tf.read().map_err(lift)?)
    };

    let out_t: Vec<T> = y_f32.iter().map(|&v| T::from_f64(v as f64)).collect();
    let out_arr = Array::from_slice(&gpu, &out_t, &shape).map_err(AutogradError::from)?;

    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let g_f32 = to_f32_host(g)?;
        let xf = upload(&gpu_b, &x_f32)?;
        let tf = upload(&gpu_b, &t_f32)?;
        let gf = upload(&gpu_b, &g_f32)?;
        let dxf = gpu_b.field::<f32>(n).map_err(lift)?;
        let mut w = dsl::gelu_bwd(&gpu_b).map_err(lift)?;
        w.bind(0, &xf);
        w.bind(1, &tf);
        w.bind(2, &gf);
        w.bind(3, &dxf);
        w.set_value(4, n as u32);
        gpu_b
            .dispatch(&w, n as u32)
            .map_err(lift)?
            .wait()
            .map_err(lift)?;
        let dx = f32_field_to_array::<T>(&gpu_b, &dxf, g.shape())?;
        Ok(vec![dx])
    };

    Ok(tape.custom_vjp(&[x], out_arr, backward))
}

/// Fused SwiGLU gate: `[N, 2H] → [N, H]`, `out = silu(x[:, :H]) ⊙ x[:, H:]`.
/// The backward derives σ′ from the forward's sigmoid (T9226). The
/// composed silu/mul path over split halves is the differential oracle.
pub fn swiglu_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
) -> Result<Var<T>, AutogradError> {
    let xs = x.value().shape().to_vec();
    if xs.len() != 2 || !xs[1].is_multiple_of(2) || xs[1] == 0 {
        return Err(bad("swiglu_var: input must be [N, 2H] with H ≥ 1"));
    }
    let (n, h) = (xs[0], xs[1] / 2);
    let gpu = x.value().gpu().clone();
    let x_f32 = to_f32_host(&x.value())?;

    let y_f32 = {
        let xf = upload(&gpu, &x_f32)?;
        let of = gpu.field::<f32>((n * h).max(1)).map_err(lift)?;
        let mut w = dsl::swiglu_fwd(&gpu).map_err(lift)?;
        w.bind(0, &xf);
        w.bind(1, &of);
        w.set_value(2, n as u32);
        w.set_value(3, h as u32);
        gpu.dispatch(&w, (n * h) as u32)
            .map_err(lift)?
            .wait()
            .map_err(lift)?;
        of.read().map_err(lift)?
    };

    let out_t: Vec<T> = y_f32.iter().map(|&v| T::from_f64(v as f64)).collect();
    let out_arr = Array::from_slice(&gpu, &out_t, &[n, h]).map_err(AutogradError::from)?;

    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let g_f32 = to_f32_host(g)?;
        let xf = upload(&gpu_b, &x_f32)?;
        let gf = upload(&gpu_b, &g_f32)?;
        let dxf = gpu_b.field::<f32>((n * 2 * h).max(1)).map_err(lift)?;
        let mut w = dsl::swiglu_bwd(&gpu_b).map_err(lift)?;
        w.bind(0, &xf);
        w.bind(1, &gf);
        w.bind(2, &dxf);
        w.set_value(3, n as u32);
        w.set_value(4, h as u32);
        gpu_b
            .dispatch(&w, (n * h) as u32)
            .map_err(lift)?
            .wait()
            .map_err(lift)?;
        let dx = f32_field_to_array::<T>(&gpu_b, &dxf, &[n, 2 * h])?;
        Ok(vec![dx])
    };

    Ok(tape.custom_vjp(&[x], out_arr, backward))
}

// ── Module forms ─────────────────────────────────────────────────────────

macro_rules! zero_param_layer {
    ($(#[$doc:meta])* $name:ident, |$tape:ident, $x:ident| $body:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl<T: DiffScalar + ToF64> Layer<T> for $name {
            type Params = ();

            fn in_dim(&self) -> Option<usize> {
                None
            }
            fn out_dim(&self, in_dim: usize) -> usize {
                in_dim
            }
            fn init(&self, _gpu: &Gpu, _key: Key) -> Result<(), AutogradError> {
                Ok(())
            }
            fn apply(
                &self,
                $tape: &Tape<T>,
                _p: &(),
                $x: &Var<T>,
            ) -> Result<Var<T>, AutogradError> {
                $body
            }
        }
    };
}

zero_param_layer!(
    /// ReLU as a layer (composed [`Var::relu`], proven per-op VJP).
    Relu, |_tape, x| x.relu()
);
zero_param_layer!(
    /// GeLU as a layer (fused kernels, T9227 backward).
    Gelu, |tape, x| gelu_var(tape, x)
);
zero_param_layer!(
    /// SiLU / swish as a layer (composed `x·σ(x)`).
    Silu, |_tape, x| x.silu()
);
zero_param_layer!(
    /// Sigmoid as a layer (composed [`Var::sigmoid`]).
    Sigmoid, |_tape, x| x.sigmoid()
);
zero_param_layer!(
    /// Tanh as a layer (composed [`Var::tanh`]).
    Tanh, |_tape, x| x.tanh()
);
zero_param_layer!(
    /// Rowwise softmax as a layer (fused, T9224 backward). 2-D input.
    Softmax, |tape, x| softmax_var(tape, x)
);
zero_param_layer!(
    /// Rowwise log-softmax as a layer (fused, T9225 backward). 2-D input.
    LogSoftmax, |tape, x| log_softmax_var(tape, x)
);

/// SwiGLU as a layer: halves the feature width (`[N, 2H] → [N, H]`) —
/// the width contract propagates the halving through tuple stacks.
#[derive(Debug, Clone, Copy, Default)]
pub struct SwiGlu;

impl<T: DiffScalar + ToF64> Layer<T> for SwiGlu {
    type Params = ();

    fn in_dim(&self) -> Option<usize> {
        None
    }
    fn out_dim(&self, in_dim: usize) -> usize {
        in_dim / 2
    }
    fn init(&self, _gpu: &Gpu, _key: Key) -> Result<(), AutogradError> {
        Ok(())
    }
    fn apply(&self, tape: &Tape<T>, _p: &(), x: &Var<T>) -> Result<Var<T>, AutogradError> {
        swiglu_var(tape, x)
    }
}
