//! Losses — fused stable cross-entropy plus the composed family (MSE, L1,
//! Huber, BCE, BCE-with-logits).
//!
//! Cross-entropy is the verified-track one: the fused forward computes the
//! stable form `lse(x) − x_y` per row (nonnegative by T9228, reusing the
//! max-stabilized stats kernel from the softmax family), and the backward
//! is one elementwise kernel producing `scale·(softmax − onehot)` — the
//! N×C log-softmax intermediate exists on *neither* pass. The composed
//! [`Var::cross_entropy`] is the differential oracle.
//!
//! The rest compose from per-op-proven VJPs. BCE-with-logits uses the
//! overflow-free spelling `max(x,0) − x·y + log(1 + e^{−|x|})`, equal to
//! the textbook form by T9229 — σ is never evaluated near 0 or 1. Huber's
//! branches carry the knee constants T9230 pins down.

use crate::activation::dsl as act_dsl;
use crate::functional::{f32_field_to_array, lift, to_f32_host};
use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::QuantaError;

#[allow(unused_imports)]
mod dsl {
    use quanta_core::*;

    /// Stable per-row cross-entropy from the softmax stats:
    /// `rows[i] = (m + ln l) − x[i, y_i]` (T9228: never negative).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn ce_rows(x: &[f32], stats: &[f32], labels: &[u32], rows: &mut [f32], n: u32, c: u32) {
        let i = quark_id();
        let row = if i < n { i } else { 0u32 };
        let m = stats[(row * 2u32) as usize];
        let l = stats[(row * 2u32 + 1u32) as usize];
        let y = labels[row as usize];
        let yc = if y < c { y } else { 0u32 };
        let v = m + ln(l) - x[(row * c + yc) as usize];
        if i < n {
            rows[row as usize] = v;
        }
    }

    /// Cross-entropy backward, elementwise: `dx = scale·(p − onehot)` with
    /// `p = exp(x−m)/l` reconstructed from the stats.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn ce_bwd(
        x: &[f32],
        stats: &[f32],
        labels: &[u32],
        dx: &mut [f32],
        n: u32,
        c: u32,
        scale: f32,
    ) {
        let i = quark_id();
        let total = n * c;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / c;
        let col = idx % c;
        let m = stats[(row * 2u32) as usize];
        let l = stats[(row * 2u32 + 1u32) as usize];
        let p = exp(fmin(x[idx as usize] - m, 0.0f32)) / l;
        let ind = if col == labels[row as usize] {
            1.0f32
        } else {
            0.0f32
        };
        if i < total {
            dx[idx as usize] = scale * (p - ind);
        }
    }
}

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(QuantaError::invalid_param(msg)))
}

/// How a loss collapses to its scalar: averaged over elements or summed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    Mean,
    Sum,
}

/// A constant leaf var of `like`'s shape filled with `v`.
fn const_like<T: DiffScalar>(
    tape: &Tape<T>,
    like: &Var<T>,
    v: f64,
) -> Result<Var<T>, AutogradError> {
    let shape = like.value().shape().to_vec();
    let a = Array::full(like.value().gpu(), T::from_f64(v), &[1])
        .and_then(|a| a.broadcast_to(&shape))
        .and_then(|a| a.contiguous())
        .map_err(AutogradError::from)?;
    Ok(tape.var(a))
}

/// Collapse an elementwise loss to its scalar per `reduction`.
fn reduce_all<T: DiffScalar>(
    tape: &Tape<T>,
    elems: &Var<T>,
    reduction: Reduction,
) -> Result<Var<T>, AutogradError> {
    let count = elems.value().shape().iter().product::<usize>().max(1);
    let s = elems.sum()?;
    match reduction {
        Reduction::Sum => Ok(s),
        Reduction::Mean => s.mul(&const_like(tape, &s, 1.0 / count as f64)?),
    }
}

/// `|z|` composed as `relu(z) + relu(−z)` (per-op-proven VJPs; the
/// subgradient at 0 is 0, matching the convention everywhere else).
fn abs_var<T: DiffScalar>(z: &Var<T>) -> Result<Var<T>, AutogradError> {
    z.relu()?.add(&z.neg()?.relu()?)
}

/// Mean-squared error `(pred − target)²`, reduced.
pub fn mse_loss<T: DiffScalar>(
    tape: &Tape<T>,
    pred: &Var<T>,
    target: &Var<T>,
    reduction: Reduction,
) -> Result<Var<T>, AutogradError> {
    let d = pred.sub(target)?;
    reduce_all(tape, &d.mul(&d)?, reduction)
}

/// Mean-absolute error `|pred − target|`, reduced.
pub fn l1_loss<T: DiffScalar>(
    tape: &Tape<T>,
    pred: &Var<T>,
    target: &Var<T>,
    reduction: Reduction,
) -> Result<Var<T>, AutogradError> {
    reduce_all(tape, &abs_var(&pred.sub(target)?)?, reduction)
}

/// Huber loss: `z²/2` inside `|z| ≤ δ`, `δ(|z| − δ/2)` outside — the
/// branches meet at the knee and the gradient is globally
/// `clamp(z, −δ, δ)` (T9230). The branch mask is a detached constant
/// (locally constant in `z` away from the knee).
pub fn huber_loss<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    pred: &Var<T>,
    target: &Var<T>,
    delta: f64,
    reduction: Reduction,
) -> Result<Var<T>, AutogradError> {
    if delta <= 0.0 {
        return Err(bad("huber_loss: delta must be positive"));
    }
    let d = pred.sub(target)?;
    let az = abs_var(&d)?;
    let quad = d.mul(&d)?.mul(&const_like(tape, &d, 0.5)?)?;
    let lin = az.mul(&const_like(tape, &az, delta)?)?.sub(&const_like(
        tape,
        &az,
        delta * delta / 2.0,
    )?)?;
    // Detached {1, 0} mask over |z| ≤ δ.
    let az_host = az.value().to_vec().map_err(AutogradError::from)?;
    let mask_host: Vec<T> = az_host
        .iter()
        .map(|v| T::from_f64(if v.to_f64() <= delta { 1.0 } else { 0.0 }))
        .collect();
    let mask = Array::from_slice(d.value().gpu(), &mask_host, d.value().shape())
        .map_err(AutogradError::from)?;
    reduce_all(tape, &quad.where_mask(&mask, &lin)?, reduction)
}

/// Binary cross-entropy over probabilities in `(0, 1)`:
/// `−(y·log p + (1−y)·log(1−p))`, reduced. Prefer
/// [`bce_with_logits_loss`] when you have logits — it cannot saturate.
pub fn bce_loss<T: DiffScalar>(
    tape: &Tape<T>,
    probs: &Var<T>,
    target: &Var<T>,
    reduction: Reduction,
) -> Result<Var<T>, AutogradError> {
    let one = const_like(tape, probs, 1.0)?;
    let pos = target.mul(&probs.log()?)?;
    let neg = one.sub(target)?.mul(&one.sub(probs)?.log()?)?;
    reduce_all(tape, &pos.add(&neg)?.neg()?, reduction)
}

/// Binary cross-entropy from logits, in the overflow-free spelling
/// `max(x,0) − x·y + log(1 + e^{−|x|})` — equal to the textbook form for
/// every logit (T9229), finite even at `x = ±100`.
pub fn bce_with_logits_loss<T: DiffScalar>(
    tape: &Tape<T>,
    logits: &Var<T>,
    target: &Var<T>,
    reduction: Reduction,
) -> Result<Var<T>, AutogradError> {
    let one = const_like(tape, logits, 1.0)?;
    let softplus = one.add(&abs_var(logits)?.neg()?.exp()?)?.log()?;
    let elems = logits.relu()?.sub(&logits.mul(target)?)?.add(&softplus)?;
    reduce_all(tape, &elems, reduction)
}

/// Fused stable cross-entropy from logits `[N, C]` and integer labels
/// (`labels[i] < C`). Forward is `lse(x) − x_y` per row off the shared
/// max-stabilized stats (T9223/T9228); backward is one elementwise kernel
/// `scale·(softmax − onehot)`. The composed [`Var::cross_entropy`] (mean
/// reduction) is the differential oracle.
pub fn cross_entropy_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    logits: &Var<T>,
    labels: &[u32],
    reduction: Reduction,
) -> Result<Var<T>, AutogradError> {
    let xs = logits.value().shape().to_vec();
    if xs.len() != 2 {
        return Err(bad("cross_entropy: logits must be 2-D [N, C]"));
    }
    let (n, c) = (xs[0], xs[1]);
    if n == 0 || c == 0 {
        return Err(bad("cross_entropy: empty logits"));
    }
    if labels.len() != n {
        return Err(bad("cross_entropy: labels length must be N"));
    }
    if labels.iter().any(|&y| y as usize >= c) {
        return Err(bad("cross_entropy: label out of range"));
    }
    let gpu = logits.value().gpu().clone();
    let x_f32 = to_f32_host(&logits.value())?;

    let (rows, stats_f32) = {
        let xf = gpu.field::<f32>(n * c).map_err(lift)?;
        xf.write(&x_f32).map_err(lift)?;
        let sf = gpu.field::<f32>(n * 2).map_err(lift)?;
        let lf = gpu.field::<u32>(n).map_err(lift)?;
        lf.write(labels).map_err(lift)?;
        let rf = gpu.field::<f32>(n).map_err(lift)?;

        let mut w = act_dsl::sm_stats(&gpu).map_err(lift)?;
        w.bind(0, &xf);
        w.bind(1, &sf);
        w.set_value(2, n as u32);
        w.set_value(3, c as u32);
        gpu.dispatch(&w, n as u32)
            .map_err(lift)?
            .wait()
            .map_err(lift)?;

        let mut w = dsl::ce_rows(&gpu).map_err(lift)?;
        w.bind(0, &xf);
        w.bind(1, &sf);
        w.bind(2, &lf);
        w.bind(3, &rf);
        w.set_value(4, n as u32);
        w.set_value(5, c as u32);
        gpu.dispatch(&w, n as u32)
            .map_err(lift)?
            .wait()
            .map_err(lift)?;
        (rf.read().map_err(lift)?, sf.read().map_err(lift)?)
    };

    let red_scale = match reduction {
        Reduction::Mean => 1.0 / n as f64,
        Reduction::Sum => 1.0,
    };
    let total: f64 = rows.iter().map(|&v| v as f64).sum::<f64>() * red_scale;
    let out_arr =
        Array::from_slice(&gpu, &[T::from_f64(total)], &[1]).map_err(AutogradError::from)?;

    let labels_own: Vec<u32> = labels.to_vec();
    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let g0 = to_f32_host(g)?[0];
        let scale = g0 * red_scale as f32;
        let xf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        xf.write(&x_f32).map_err(lift)?;
        let sf = gpu_b.field::<f32>(n * 2).map_err(lift)?;
        sf.write(&stats_f32).map_err(lift)?;
        let lf = gpu_b.field::<u32>(n).map_err(lift)?;
        lf.write(&labels_own).map_err(lift)?;
        let dxf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        let mut w = dsl::ce_bwd(&gpu_b).map_err(lift)?;
        w.bind(0, &xf);
        w.bind(1, &sf);
        w.bind(2, &lf);
        w.bind(3, &dxf);
        w.set_value(4, n as u32);
        w.set_value(5, c as u32);
        w.set_value(6, scale);
        gpu_b
            .dispatch(&w, (n * c) as u32)
            .map_err(lift)?
            .wait()
            .map_err(lift)?;
        let dx = f32_field_to_array::<T>(&gpu_b, &dxf, &[n, c])?;
        Ok(vec![dx])
    };

    Ok(tape.custom_vjp(&[logits], out_arr, backward))
}
