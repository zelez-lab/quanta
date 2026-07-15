//! Optimizers as tree operations — fused SGD/Adam/AdamW over [`ParamTree`],
//! LR schedules, and gradient clipping.
//!
//! The architecture record's optimizer story made concrete: an optimizer is
//! a small `Copy` configuration plus a **state tree mirroring the parameter
//! tree** (leaf-indexed in `flatten` order). `step` is state-passing —
//! it CONSUMES the state and returns the successor, the same ownership
//! discipline as [`Key`](crate::layer::Key): a stale optimizer state cannot
//! be reused by accident (decision D2). Parameters stay borrowable values;
//! keeping an old tree around (checkpoint, comparison) is legitimate.
//!
//! Each leaf updates in ONE kernel dispatch: `sgd_step` folds weight decay,
//! the momentum recurrence (T9219), and the nesterov/classical direction
//! into a single elementwise pass; `adam_step` folds both moment
//! recurrences, the exact bias correction (T9220 — the `1/(1−βᵗ)` factors
//! are host scalars), and BOTH weight-decay spellings — coupled L2 enters
//! the effective gradient, decoupled (AdamW) shrinks the parameter directly;
//! T9221 is why one kernel serves both.
//!
//! The learning rate is plain data on a `Copy` struct: schedule by
//! rebuilding, `Sgd { lr: sched.lr(t), ..opt }` — no callbacks, no
//! registration, nothing ambient (decision D4).

use crate::functional::{f32_field_to_array, lift, to_f32_host};
use crate::layer::ParamTree;
use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar};
use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod dsl {
    use quanta_core::*;

    /// One fused SGD step per element: weight decay into the effective
    /// gradient, the velocity recurrence `v ← μ·v + gₑ` (T9219), then the
    /// classical (`−lr·v`) or nesterov (`−lr·(gₑ + μ·v)`) step, selected
    /// by indicator arithmetic on `nesterov ∈ {0, 1}`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn sgd_step(
        p: &[f32],
        g: &[f32],
        v: &[f32],
        p_out: &mut [f32],
        v_out: &mut [f32],
        n: u32,
        lr: f32,
        mu: f32,
        wd: f32,
        nesterov: f32,
    ) {
        let i = quark_id();
        let idx = if i < n { i } else { 0u32 };
        let geff = g[idx as usize] + wd * p[idx as usize];
        let vnew = mu * v[idx as usize] + geff;
        let dir = nesterov * (geff + mu * vnew) + (1.0f32 - nesterov) * vnew;
        let pnew = p[idx as usize] - lr * dir;
        if i < n {
            p_out[idx as usize] = pnew;
            v_out[idx as usize] = vnew;
        }
    }

    /// One fused Adam/AdamW step per element: both moment recurrences,
    /// exact bias correction via host-computed `1/(1−βᵗ)` factors (T9220),
    /// and both weight-decay spellings — `wd_c` folds into the effective
    /// gradient (coupled L2), `wd_d` shrinks the parameter directly
    /// (decoupled, AdamW; T9221). At most one of the two is nonzero.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn adam_step(
        p: &[f32],
        g: &[f32],
        m: &[f32],
        v: &[f32],
        p_out: &mut [f32],
        m_out: &mut [f32],
        v_out: &mut [f32],
        n: u32,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        bc1_inv: f32,
        bc2_inv: f32,
        wd_c: f32,
        wd_d: f32,
    ) {
        let i = quark_id();
        let idx = if i < n { i } else { 0u32 };
        let geff = g[idx as usize] + wd_c * p[idx as usize];
        let mnew = b1 * m[idx as usize] + (1.0f32 - b1) * geff;
        let vnew = b2 * v[idx as usize] + (1.0f32 - b2) * geff * geff;
        let mhat = mnew * bc1_inv;
        let vhat = vnew * bc2_inv;
        let pnew = p[idx as usize] - lr * (mhat / (sqrt(vhat) + eps) + wd_d * p[idx as usize]);
        if i < n {
            p_out[idx as usize] = pnew;
            m_out[idx as usize] = mnew;
            v_out[idx as usize] = vnew;
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

/// Host dispatch: fused SGD step over `n` elements. `nesterov` is the
/// `{0, 1}` indicator.
#[allow(clippy::too_many_arguments)]
pub fn sgd_step_field(
    gpu: &Gpu,
    n: u32,
    lr: f32,
    mu: f32,
    wd: f32,
    nesterov: f32,
    p: &Field<f32>,
    g: &Field<f32>,
    v: &Field<f32>,
    p_out: &Field<f32>,
    v_out: &Field<f32>,
) -> Result<(), QuantaError> {
    let nu = n as usize;
    check(
        p.len() == nu && g.len() == nu && v.len() == nu,
        "sgd_step: P/G/V length must be n",
    )?;
    check(
        p_out.len() == nu && v_out.len() == nu,
        "sgd_step: output length must be n",
    )?;
    if n == 0 {
        return Ok(());
    }
    let mut w = dsl::sgd_step(gpu)?;
    w.bind(0, p);
    w.bind(1, g);
    w.bind(2, v);
    w.bind(3, p_out);
    w.bind(4, v_out);
    w.set_value(5, n);
    w.set_value(6, lr);
    w.set_value(7, mu);
    w.set_value(8, wd);
    w.set_value(9, nesterov);
    gpu.dispatch(&w, n)?.wait()?;
    Ok(())
}

/// Host dispatch: fused Adam/AdamW step over `n` elements. `bc1_inv` /
/// `bc2_inv` are the exact bias-correction factors `1/(1−βᵗ)`; `wd_c` /
/// `wd_d` the coupled / decoupled weight decays (at most one nonzero).
#[allow(clippy::too_many_arguments)]
pub fn adam_step_field(
    gpu: &Gpu,
    n: u32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    bc1_inv: f32,
    bc2_inv: f32,
    wd_c: f32,
    wd_d: f32,
    p: &Field<f32>,
    g: &Field<f32>,
    m: &Field<f32>,
    v: &Field<f32>,
    p_out: &Field<f32>,
    m_out: &Field<f32>,
    v_out: &Field<f32>,
) -> Result<(), QuantaError> {
    let nu = n as usize;
    check(
        p.len() == nu && g.len() == nu && m.len() == nu && v.len() == nu,
        "adam_step: P/G/M/V length must be n",
    )?;
    check(
        p_out.len() == nu && m_out.len() == nu && v_out.len() == nu,
        "adam_step: output length must be n",
    )?;
    if n == 0 {
        return Ok(());
    }
    let mut w = dsl::adam_step(gpu)?;
    w.bind(0, p);
    w.bind(1, g);
    w.bind(2, m);
    w.bind(3, v);
    w.bind(4, p_out);
    w.bind(5, m_out);
    w.bind(6, v_out);
    w.set_value(7, n);
    w.set_value(8, lr);
    w.set_value(9, b1);
    w.set_value(10, b2);
    w.set_value(11, eps);
    w.set_value(12, bc1_inv);
    w.set_value(13, bc2_inv);
    w.set_value(14, wd_c);
    w.set_value(15, wd_d);
    gpu.dispatch(&w, n)?.wait()?;
    Ok(())
}

// ── Leaf plumbing ────────────────────────────────────────────────────────

/// Upload a leaf as an f32 field.
fn upload<T: DiffScalar + ToF64>(gpu: &Gpu, a: &Array<T>) -> Result<Field<f32>, AutogradError> {
    let host = to_f32_host(a)?;
    let f = gpu.field::<f32>(host.len()).map_err(lift)?;
    f.write(&host).map_err(lift)?;
    Ok(f)
}

fn zeros_like<T: DiffScalar>(a: &Array<T>) -> Result<Array<T>, AutogradError> {
    Array::<T>::zeros(a.gpu(), a.shape()).map_err(AutogradError::from)
}

fn same_shape<T: DiffScalar>(
    a: &Array<T>,
    b: &Array<T>,
    msg: &'static str,
) -> Result<(), AutogradError> {
    if a.shape() == b.shape() {
        Ok(())
    } else {
        Err(bad(msg))
    }
}

// ── SGD ──────────────────────────────────────────────────────────────────

/// SGD configuration: momentum, nesterov, and (coupled) weight decay.
/// Plain data — schedule the learning rate by rebuilding:
/// `Sgd { lr: sched.lr(t), ..opt }`.
#[derive(Debug, Clone, Copy)]
pub struct Sgd {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

/// SGD state: one velocity leaf per parameter leaf, `flatten` order.
pub struct SgdState<T: DiffScalar> {
    pub velocity: Vec<Array<T>>,
}

impl Sgd {
    /// Plain SGD (no momentum, no decay).
    pub fn new(lr: f32) -> Self {
        Sgd {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Classical momentum SGD.
    pub fn momentum(lr: f32, mu: f32) -> Self {
        Sgd {
            momentum: mu,
            ..Sgd::new(lr)
        }
    }

    /// Zero-velocity state shaped like `params`.
    pub fn init<T: DiffScalar, P: ParamTree<T>>(
        &self,
        params: &P,
    ) -> Result<SgdState<T>, AutogradError> {
        Ok(SgdState {
            velocity: params
                .flatten()
                .iter()
                .map(zeros_like)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// One optimizer step: `(params, grads, state) → (new_params, new_state)`.
    /// Consumes the state (D2 — a stale state cannot be replayed).
    pub fn step<T: DiffScalar + ToF64, P: ParamTree<T>>(
        &self,
        params: &P,
        grads: &P,
        state: SgdState<T>,
    ) -> Result<(P, SgdState<T>), AutogradError> {
        let p_leaves = params.flatten();
        let g_leaves = grads.flatten();
        if p_leaves.len() != g_leaves.len() || p_leaves.len() != state.velocity.len() {
            return Err(bad("sgd: params/grads/state leaf counts differ"));
        }
        let mut new_p = Vec::with_capacity(p_leaves.len());
        let mut new_v = Vec::with_capacity(p_leaves.len());
        for ((p, g), v) in p_leaves.iter().zip(&g_leaves).zip(&state.velocity) {
            same_shape(p, g, "sgd: param/grad leaf shapes differ")?;
            same_shape(p, v, "sgd: param/state leaf shapes differ")?;
            let gpu = p.gpu().clone();
            let n = p.to_vec().map_err(AutogradError::from)?.len();
            let pf = upload(&gpu, p)?;
            let gf = upload(&gpu, g)?;
            let vf = upload(&gpu, v)?;
            let pof = gpu.field::<f32>(n).map_err(lift)?;
            let vof = gpu.field::<f32>(n).map_err(lift)?;
            sgd_step_field(
                &gpu,
                n as u32,
                self.lr,
                self.momentum,
                self.weight_decay,
                if self.nesterov { 1.0 } else { 0.0 },
                &pf,
                &gf,
                &vf,
                &pof,
                &vof,
            )
            .map_err(lift)?;
            new_p.push(f32_field_to_array::<T>(&gpu, &pof, p.shape())?);
            new_v.push(f32_field_to_array::<T>(&gpu, &vof, p.shape())?);
        }
        let params = params.unflatten(&mut new_p.into_iter())?;
        Ok((params, SgdState { velocity: new_v }))
    }
}

// ── Adam / AdamW ─────────────────────────────────────────────────────────

/// Adam configuration. `decoupled = true` makes `weight_decay` the AdamW
/// decay (shrinks the parameter directly, outside the moments — T9221);
/// `false` folds it into the gradient as coupled L2.
#[derive(Debug, Clone, Copy)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub decoupled: bool,
}

/// Adam state: first/second moment leaves (`flatten` order) plus the step
/// counter driving the exact bias correction (T9220).
pub struct AdamState<T: DiffScalar> {
    pub m: Vec<Array<T>>,
    pub v: Vec<Array<T>>,
    pub t: u64,
}

impl Adam {
    /// Adam with the usual defaults (β₁ = 0.9, β₂ = 0.999, ε = 1e-8).
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled: false,
        }
    }

    /// AdamW: decoupled weight decay (the transformer-training default).
    pub fn adamw(lr: f32, weight_decay: f32) -> Self {
        Adam {
            weight_decay,
            decoupled: true,
            ..Adam::new(lr)
        }
    }

    /// Zero-moment state shaped like `params`, `t = 0`.
    pub fn init<T: DiffScalar, P: ParamTree<T>>(
        &self,
        params: &P,
    ) -> Result<AdamState<T>, AutogradError> {
        let zeros = |()| -> Result<Vec<Array<T>>, AutogradError> {
            params
                .flatten()
                .iter()
                .map(zeros_like)
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(AdamState {
            m: zeros(())?,
            v: zeros(())?,
            t: 0,
        })
    }

    /// One optimizer step: `(params, grads, state) → (new_params, new_state)`.
    /// Consumes the state; the successor carries `t + 1`.
    pub fn step<T: DiffScalar + ToF64, P: ParamTree<T>>(
        &self,
        params: &P,
        grads: &P,
        state: AdamState<T>,
    ) -> Result<(P, AdamState<T>), AutogradError> {
        let p_leaves = params.flatten();
        let g_leaves = grads.flatten();
        if p_leaves.len() != g_leaves.len()
            || p_leaves.len() != state.m.len()
            || p_leaves.len() != state.v.len()
        {
            return Err(bad("adam: params/grads/state leaf counts differ"));
        }
        let t = state.t + 1;
        let bc1_inv = 1.0 / (1.0 - self.beta1.powi(t as i32));
        let bc2_inv = 1.0 / (1.0 - self.beta2.powi(t as i32));
        let (wd_c, wd_d) = if self.decoupled {
            (0.0, self.weight_decay)
        } else {
            (self.weight_decay, 0.0)
        };
        let mut new_p = Vec::with_capacity(p_leaves.len());
        let mut new_m = Vec::with_capacity(p_leaves.len());
        let mut new_v = Vec::with_capacity(p_leaves.len());
        for (((p, g), m), v) in p_leaves.iter().zip(&g_leaves).zip(&state.m).zip(&state.v) {
            same_shape(p, g, "adam: param/grad leaf shapes differ")?;
            same_shape(p, m, "adam: param/state leaf shapes differ")?;
            same_shape(p, v, "adam: param/state leaf shapes differ")?;
            let gpu = p.gpu().clone();
            let n = p.to_vec().map_err(AutogradError::from)?.len();
            let pf = upload(&gpu, p)?;
            let gf = upload(&gpu, g)?;
            let mf = upload(&gpu, m)?;
            let vf = upload(&gpu, v)?;
            let pof = gpu.field::<f32>(n).map_err(lift)?;
            let mof = gpu.field::<f32>(n).map_err(lift)?;
            let vof = gpu.field::<f32>(n).map_err(lift)?;
            adam_step_field(
                &gpu, n as u32, self.lr, self.beta1, self.beta2, self.eps, bc1_inv, bc2_inv, wd_c,
                wd_d, &pf, &gf, &mf, &vf, &pof, &mof, &vof,
            )
            .map_err(lift)?;
            new_p.push(f32_field_to_array::<T>(&gpu, &pof, p.shape())?);
            new_m.push(f32_field_to_array::<T>(&gpu, &mof, p.shape())?);
            new_v.push(f32_field_to_array::<T>(&gpu, &vof, p.shape())?);
        }
        let params = params.unflatten(&mut new_p.into_iter())?;
        Ok((
            params,
            AdamState {
                m: new_m,
                v: new_v,
                t,
            },
        ))
    }
}

// ── LR schedules ─────────────────────────────────────────────────────────

/// A learning-rate schedule: a pure function of the 0-based step index.
/// Feed it back into the optimizer by rebuilding the config:
/// `Adam { lr: sched.lr(t), ..opt }`.
#[derive(Debug, Clone, Copy)]
pub enum Schedule {
    /// `lr` forever.
    Constant { lr: f32 },
    /// `lr · gammaᵏ` on the `k`-th interval of `every` steps.
    Step { lr: f32, gamma: f32, every: u64 },
    /// Linear ramp `lr·(t+1)/warmup` for `t < warmup`, then `lr`.
    LinearWarmup { lr: f32, warmup: u64 },
    /// Linear warmup to `base`, then cosine decay to `min_lr` at `total`
    /// (clamped at `min_lr` beyond).
    Cosine {
        base: f32,
        min_lr: f32,
        warmup: u64,
        total: u64,
    },
}

impl Schedule {
    /// The learning rate at (0-based) step `t`.
    pub fn lr(&self, t: u64) -> f32 {
        match *self {
            Schedule::Constant { lr } => lr,
            Schedule::Step { lr, gamma, every } => {
                let k = t.checked_div(every).unwrap_or(0);
                lr * gamma.powi(k.min(i32::MAX as u64) as i32)
            }
            Schedule::LinearWarmup { lr, warmup } => {
                if t < warmup {
                    lr * (t + 1) as f32 / warmup as f32
                } else {
                    lr
                }
            }
            Schedule::Cosine {
                base,
                min_lr,
                warmup,
                total,
            } => {
                if t < warmup {
                    base * (t + 1) as f32 / warmup as f32
                } else if t >= total {
                    min_lr
                } else {
                    let span = (total - warmup) as f32;
                    let progress = (t - warmup) as f32 / span;
                    min_lr + 0.5 * (base - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
                }
            }
        }
    }
}

// ── Gradient clipping ────────────────────────────────────────────────────

/// Clamp every gradient element into `[−max_abs, max_abs]`.
pub fn clip_grad_value<T: DiffScalar + ToF64, P: ParamTree<T>>(
    grads: &P,
    max_abs: f32,
) -> Result<P, AutogradError> {
    grads.map(&mut |leaf: &Array<T>| {
        let host = to_f32_host(leaf)?;
        let clipped: Vec<T> = host
            .iter()
            .map(|&v| T::from_f64(v.clamp(-max_abs, max_abs) as f64))
            .collect();
        Array::from_slice(leaf.gpu(), &clipped, leaf.shape()).map_err(AutogradError::from)
    })
}

/// Scale the WHOLE gradient tree so its global L2 norm (over all leaves
/// concatenated) is at most `max_norm`. Returns the scaled tree and the
/// pre-clip norm; under the threshold the tree passes through unscaled.
pub fn clip_grad_norm<T: DiffScalar + ToF64, P: ParamTree<T>>(
    grads: &P,
    max_norm: f32,
) -> Result<(P, f32), AutogradError> {
    let mut sq_sum = 0.0f64;
    for leaf in grads.flatten() {
        for v in to_f32_host(&leaf)? {
            sq_sum += (v as f64) * (v as f64);
        }
    }
    let norm = sq_sum.sqrt() as f32;
    if norm <= max_norm {
        let identity = grads.map(&mut |leaf: &Array<T>| Ok(leaf.shallow_clone()))?;
        return Ok((identity, norm));
    }
    let scale = max_norm / norm;
    let scaled = grads.map(&mut |leaf: &Array<T>| {
        let host = to_f32_host(leaf)?;
        let s: Vec<T> = host
            .iter()
            .map(|&v| T::from_f64((v * scale) as f64))
            .collect();
        Array::from_slice(leaf.gpu(), &s, leaf.shape()).map_err(AutogradError::from)
    })?;
    Ok((scaled, norm))
}
