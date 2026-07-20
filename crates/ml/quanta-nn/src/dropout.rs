//! Key-based dropout — one fused kernel, both directions (f32, any shape).
//!
//! The mask is a **pure function of (key, element index)**: each element
//! draws one Philox4×32-10 word with counter `(i, stream_lo, stream_hi, 0)`
//! and key `(seed_lo, seed_hi)`, and is KEPT iff `t ≤ u` where
//! `t = ⌊rate · 2³²⌋`. Kept elements scale by `2³² / (2³² − t)` (inverted
//! dropout). What the proofs in
//! `specs/verify/lean/Quanta/Nn/DropoutVjp.lean` establish:
//!
//! * **T9231** — averaged over all 2³² equally-likely words the masked
//!   scale is exactly the identity: unbiased at the rate the kernel
//!   implements.
//! * **T9232** — the mask-scale map is diagonal, hence self-adjoint: the
//!   VJP is the SAME masked scaling applied to the cotangent. The backward
//!   therefore regenerates the mask from the key and reruns the forward
//!   kernel — **no mask is ever stored**.
//! * **T9233** — the floor threshold undershoots the requested rate by
//!   less than `2⁻³²` and never exceeds it.
//!
//! Determinism is the point: same key, same shape → same mask, on every
//! backend (Philox is counter-based and bit-exact across CPU/GPU — the
//! quanta-rand contract). No global RNG exists anywhere in this path; the
//! [`Key`](crate::layer::Key) is the whole effect (decision D4).

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::{Field, Gpu, QuantaError};

use crate::functional::{f32_field_to_array, lift, to_f32_host};
use crate::layer::{Key, Layer};

#[allow(unused_imports)]
mod dsl {
    use quanta_core::*;

    /// Masked scale: `out[i] = x[i] · keep(i) · inv_keep` with
    /// `keep(i) = (philox(key, i) ≥ thresh)`. Elementwise, loop-free,
    /// single-flag store guard. Serves forward AND backward (T9232) —
    /// the backward binds `g` as `x`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn dropout_mask_scale(
        x: &[f32],
        out: &mut [f32],
        n: u32,
        thresh: u32,
        inv_keep: f32,
        k_lo: u32,
        k_hi: u32,
        s_lo: u32,
        s_hi: u32,
    ) {
        let i = quark_id();
        let idx = if i < n { i } else { 0u32 };
        let r = quanta_rand::philox4x32_10_first_u32_kernel(idx, s_lo, s_hi, 0u32, k_lo, k_hi);
        let keep = (r >= thresh) as u32 as f32;
        if i < n {
            out[idx as usize] = x[idx as usize] * keep * inv_keep;
        }
    }
}

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(QuantaError::invalid_param(msg)))
}

/// The kernel-facing halves of a [`Key`]: Philox key words from the seed,
/// counter words 1–2 from the stream (counter word 0 is the element index).
fn key_words(key: Key) -> (u32, u32, u32, u32) {
    let (seed, stream) = key.raw();
    (
        seed as u32,
        (seed >> 32) as u32,
        stream as u32,
        (stream >> 32) as u32,
    )
}

/// `⌊rate · 2³²⌋` (T9233) and the inverted-dropout scale `1 / (1 − t/2³²)`.
fn threshold_and_scale(rate: f32) -> (u32, f32) {
    let t = ((rate as f64) * 4294967296.0).floor() as u64;
    let t = t.min(u32::MAX as u64) as u32;
    let keep = 1.0 - (t as f64) / 4294967296.0;
    (t, (1.0 / keep) as f32)
}

/// Host reference for the kernel's per-element keep decision — the same
/// Philox word, drawn on the CPU (bit-exact by the quanta-rand contract).
/// Public for differential tests and downstream mask inspection.
pub fn keep_mask_host(key: Key, rate: f32, n: usize) -> Vec<bool> {
    let (k_lo, k_hi, s_lo, s_hi) = key_words(key);
    let (thresh, _) = threshold_and_scale(rate);
    (0..n)
        .map(|i| {
            quanta_rand::philox4x32::philox4x32_10_first_u32(i as u32, s_lo, s_hi, 0, k_lo, k_hi)
                >= thresh
        })
        .collect()
}

/// Host dispatch: the masked scale over a flat length-`n` buffer.
#[allow(clippy::too_many_arguments)]
pub fn dropout_mask_scale(
    gpu: &Gpu,
    n: u32,
    thresh: u32,
    inv_keep: f32,
    key: Key,
    x: &Field<f32>,
    out: &Field<f32>,
) -> Result<(), QuantaError> {
    if x.len() != n as usize || out.len() != n as usize {
        return Err(QuantaError::invalid_param(
            "dropout: X and OUT lengths must equal n",
        ));
    }
    if n == 0 {
        return Ok(());
    }
    let (k_lo, k_hi, s_lo, s_hi) = key_words(key);
    let mut w = dsl::dropout_mask_scale(gpu)?;
    w.bind(0, x);
    w.bind(1, out);
    w.set_value(2, n);
    w.set_value(3, thresh);
    w.set_value(4, inv_keep);
    w.set_value(5, k_lo);
    w.set_value(6, k_hi);
    w.set_value(7, s_lo);
    w.set_value(8, s_hi);
    gpu.dispatch(&w, n)?.wait()?;
    Ok(())
}

/// Tape-differentiable key-based dropout, any shape, elementwise.
///
/// `rate ∈ [0, 1]`: `0` is the identity (returns the input `Var`
/// unchanged — no tape node), `1` zeroes everything (gradient included).
/// In between, the fused kernel masks-and-scales; the backward reruns the
/// SAME kernel on the cotangent with the mask regenerated from `key`
/// (T9232) — nothing is stored. The key is CONSUMED (decision D2): one
/// key, one mask.
pub fn dropout_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
    rate: f32,
    key: Key,
) -> Result<Var<T>, AutogradError> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(bad("dropout_var: rate must be in [0, 1]"));
    }
    if rate == 0.0 {
        // Identity — a same-shape reshape is the tape's identity node
        // (`Var` is not `Clone`; the key is consumed unused, like a
        // zero-init Linear bias).
        let shape = x.value().shape().to_vec();
        return x.reshape(&shape);
    }
    let shape = x.value().shape().to_vec();
    let n: usize = shape.iter().product();
    let gpu = x.value().gpu().clone();

    if rate >= 1.0 {
        // Degenerate but well-defined: everything dropped, forward and
        // backward both zero (the keep-probability is 0 — no scale can
        // recover it).
        let zeros: Vec<T> = (0..n).map(|_| T::from_f64(0.0)).collect();
        let out = Array::from_slice(&gpu, &zeros, &shape).map_err(AutogradError::from)?;
        let gpu_b = gpu.clone();
        let shape_b = shape.clone();
        let backward = move |_g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
            let zeros: Vec<T> = (0..n).map(|_| T::from_f64(0.0)).collect();
            Ok(vec![
                Array::from_slice(&gpu_b, &zeros, &shape_b).map_err(AutogradError::from)?,
            ])
        };
        return Ok(tape.custom_vjp(&[x], out, backward));
    }

    let (thresh, inv_keep) = threshold_and_scale(rate);
    let x_f32 = to_f32_host(&x.value())?;

    let out_f32 = {
        let xf = gpu.field::<f32>(n).map_err(lift)?;
        let of = gpu.field::<f32>(n).map_err(lift)?;
        xf.write(&x_f32).map_err(lift)?;
        dropout_mask_scale(&gpu, n as u32, thresh, inv_keep, key, &xf, &of).map_err(lift)?;
        of.read().map_err(lift)?
    };
    let out_t: Vec<T> = out_f32.iter().map(|&v| T::from_f64(v as f64)).collect();
    let out_arr = Array::from_slice(&gpu, &out_t, &shape).map_err(AutogradError::from)?;

    let gpu_b = gpu.clone();
    let shape_b = shape.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let g_f32 = to_f32_host(g)?;
        let gf = gpu_b.field::<f32>(n).map_err(lift)?;
        let df = gpu_b.field::<f32>(n).map_err(lift)?;
        gf.write(&g_f32).map_err(lift)?;
        // T9232: the VJP IS the forward map — same kernel, same key.
        dropout_mask_scale(&gpu_b, n as u32, thresh, inv_keep, key, &gf, &df).map_err(lift)?;
        Ok(vec![f32_field_to_array::<T>(&gpu_b, &df, &shape_b)?])
    };

    Ok(tape.custom_vjp(&[x], out_arr, backward))
}

/// Dropout as a layer. `apply` (eval) is the identity — there is no mode
/// flag anywhere; training is the separate, key-consuming
/// [`Layer::apply_train`] path, which this layer overrides: it splits the
/// incoming key, masks with one half, and passes the other on down the
/// stack (D2's state-passing style).
pub struct Dropout {
    pub rate: f32,
}

impl<T: DiffScalar + ToF64> Layer<T> for Dropout {
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
    fn apply(&self, _tape: &Tape<T>, _p: &(), x: &Var<T>) -> Result<Var<T>, AutogradError> {
        // Eval semantics: the identity (inverted dropout needs no
        // inference-time rescale). Same-shape reshape = the identity node.
        let shape = x.value().shape().to_vec();
        x.reshape(&shape)
    }
    fn apply_train(
        &self,
        tape: &Tape<T>,
        _p: &(),
        x: &Var<T>,
        key: Key,
    ) -> Result<(Var<T>, Key), AutogradError> {
        let (k_mask, k_rest) = key.split();
        Ok((dropout_var(tape, x, self.rate, k_mask)?, k_rest))
    }
}
