//! Fused LayerNorm / RMSNorm kernels (forward + backward, f32, `[N, C]`).
//!
//! The backward kernels implement the closed-form three-term gradients
//! proven to be the adjoints of the normalization linearizations in
//! `specs/verify/lean/Quanta/Nn/NormVjp.lean` (T9210–T9215):
//!
//! * LayerNorm: `dx = rstd · (h − mean(h) − x̂ · mean(h∘x̂))`
//! * RMSNorm:   `dx = rrms · (h − x̂ · mean(h∘x̂))`
//!
//! with `h = g∘γ` the upstream gradient pulled through the scale and `x̂`
//! the normalized row. T9213/T9214 bound the divisors below by `√ε`, which
//! is why no kernel here guards a division.
//!
//! ## Thread layouts (the SDPA patterns, reused verbatim)
//!
//! * **Row-stats kernels** (`*_stats`, `*_bwd_rowstats`) — one thread per
//!   row, streaming the C dimension: the `sdpa_bwd_delta` precompute shape.
//!   Forward stats save `(μ, rstd)` (RMS: `rrms`) exactly like SDPA saves
//!   `(m, l)`; the backward consumes them instead of recomputing.
//! * **Elementwise kernels** (`*_fwd`, `*_bwd_dx`) — one thread per `(row,
//!   channel)` entry, loop-free bodies, single-flag store guards.
//! * **Parameter-gradient kernels** (`*_bwd_dparams`) — one thread per
//!   *column*, streaming the N rows (`dγ_c = Σᵢ gᵢ_c·x̂ᵢ_c`, `dβ_c = Σᵢ
//!   gᵢ_c`): the column-major twin of the row-stats shape.
//!
//! All loops are top-level with clamped indices (never nested inside a
//! bounds `if` — the loop-guard lowering rule); masks and guards are single
//! precomputed flags.

use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod dsl {
    use quanta_core::*;

    /// LayerNorm forward stats: one thread per row `i` streams the C
    /// channels and writes `stats[i*2] = μ`, `stats[i*2+1] = rstd`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn ln_stats(x: &[f32], stats: &mut [f32], n: u32, c: u32, eps: f32) {
        let i = quark_id();
        let row = if i < n { i } else { 0u32 };
        let base = row * c;
        let cf = c as f32;

        let mut sum: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < c {
            sum = sum + x[(base + p) as usize];
            p = p + 1u32;
        }
        let mu = sum / cf;

        let mut sq: f32 = 0.0f32;
        let mut q: u32 = 0u32;
        while q < c {
            let d = x[(base + q) as usize] - mu;
            sq = sq + d * d;
            q = q + 1u32;
        }
        let rstd = 1.0f32 / sqrt(sq / cf + eps);

        if i < n {
            stats[(row * 2u32) as usize] = mu;
            stats[(row * 2u32 + 1u32) as usize] = rstd;
        }
    }

    /// LayerNorm forward, elementwise: `out = (x − μ)·rstd·γ + β`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn ln_fwd(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        stats: &[f32],
        out: &mut [f32],
        n: u32,
        c: u32,
    ) {
        let i = quark_id();
        let total = n * c;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / c;
        let col = idx % c;
        let mu = stats[(row * 2u32) as usize];
        let rstd = stats[(row * 2u32 + 1u32) as usize];
        let xh = (x[idx as usize] - mu) * rstd;
        if i < total {
            out[idx as usize] = xh * gamma[col as usize] + beta[col as usize];
        }
    }

    /// LayerNorm backward row-stats: one thread per row streams C and
    /// writes the two row means the T9210 formula needs —
    /// `bstats[i*2] = mean(h)`, `bstats[i*2+1] = mean(h∘x̂)`, `h = g∘γ`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn ln_bwd_rowstats(
        x: &[f32],
        gamma: &[f32],
        stats: &[f32],
        g: &[f32],
        bstats: &mut [f32],
        n: u32,
        c: u32,
    ) {
        let i = quark_id();
        let row = if i < n { i } else { 0u32 };
        let base = row * c;
        let cf = c as f32;
        let mu = stats[(row * 2u32) as usize];
        let rstd = stats[(row * 2u32 + 1u32) as usize];

        let mut s1: f32 = 0.0f32;
        let mut s2: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < c {
            let h = g[(base + p) as usize] * gamma[p as usize];
            let xh = (x[(base + p) as usize] - mu) * rstd;
            s1 = s1 + h;
            s2 = s2 + h * xh;
            p = p + 1u32;
        }
        if i < n {
            bstats[(row * 2u32) as usize] = s1 / cf;
            bstats[(row * 2u32 + 1u32) as usize] = s2 / cf;
        }
    }

    /// LayerNorm backward dx, elementwise — the proven three-term formula:
    /// `dx = rstd · (h − mean(h) − x̂·mean(h∘x̂))` (T9210).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn ln_bwd_dx(
        x: &[f32],
        gamma: &[f32],
        stats: &[f32],
        bstats: &[f32],
        g: &[f32],
        dx: &mut [f32],
        n: u32,
        c: u32,
    ) {
        let i = quark_id();
        let total = n * c;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / c;
        let col = idx % c;
        let mu = stats[(row * 2u32) as usize];
        let rstd = stats[(row * 2u32 + 1u32) as usize];
        let m1 = bstats[(row * 2u32) as usize];
        let m2 = bstats[(row * 2u32 + 1u32) as usize];
        let h = g[idx as usize] * gamma[col as usize];
        let xh = (x[idx as usize] - mu) * rstd;
        if i < total {
            dx[idx as usize] = rstd * (h - m1 - xh * m2);
        }
    }

    /// LayerNorm parameter gradients: one thread per column `c` streams the
    /// N rows — `dγ_c = Σᵢ g·x̂`, `dβ_c = Σᵢ g`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn ln_bwd_dparams(
        x: &[f32],
        stats: &[f32],
        g: &[f32],
        dgamma: &mut [f32],
        dbeta: &mut [f32],
        n: u32,
        c: u32,
    ) {
        let i = quark_id();
        let col = if i < c { i } else { 0u32 };

        let mut sg: f32 = 0.0f32;
        let mut sgx: f32 = 0.0f32;
        let mut r: u32 = 0u32;
        while r < n {
            let idx = r * c + col;
            let mu = stats[(r * 2u32) as usize];
            let rstd = stats[(r * 2u32 + 1u32) as usize];
            let gv = g[idx as usize];
            let xh = (x[idx as usize] - mu) * rstd;
            sg = sg + gv;
            sgx = sgx + gv * xh;
            r = r + 1u32;
        }
        if i < c {
            dgamma[col as usize] = sgx;
            dbeta[col as usize] = sg;
        }
    }

    /// RMSNorm forward stats: `stats[i] = rrms = 1/√(mean(x²) + ε)`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn rms_stats(x: &[f32], stats: &mut [f32], n: u32, c: u32, eps: f32) {
        let i = quark_id();
        let row = if i < n { i } else { 0u32 };
        let base = row * c;
        let cf = c as f32;

        let mut sq: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < c {
            let v = x[(base + p) as usize];
            sq = sq + v * v;
            p = p + 1u32;
        }
        let rrms = 1.0f32 / sqrt(sq / cf + eps);
        if i < n {
            stats[row as usize] = rrms;
        }
    }

    /// RMSNorm forward, elementwise: `out = x·rrms·γ`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn rms_fwd(x: &[f32], gamma: &[f32], stats: &[f32], out: &mut [f32], n: u32, c: u32) {
        let i = quark_id();
        let total = n * c;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / c;
        let col = idx % c;
        let rrms = stats[row as usize];
        if i < total {
            out[idx as usize] = x[idx as usize] * rrms * gamma[col as usize];
        }
    }

    /// RMSNorm backward row-stats: `bstats[i] = mean(h∘x̂)` (the single
    /// mean T9211 needs — the centering term drops).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn rms_bwd_rowstats(
        x: &[f32],
        gamma: &[f32],
        stats: &[f32],
        g: &[f32],
        bstats: &mut [f32],
        n: u32,
        c: u32,
    ) {
        let i = quark_id();
        let row = if i < n { i } else { 0u32 };
        let base = row * c;
        let cf = c as f32;
        let rrms = stats[row as usize];

        let mut s2: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < c {
            let h = g[(base + p) as usize] * gamma[p as usize];
            let xh = x[(base + p) as usize] * rrms;
            s2 = s2 + h * xh;
            p = p + 1u32;
        }
        if i < n {
            bstats[row as usize] = s2 / cf;
        }
    }

    /// RMSNorm backward dx: `dx = rrms · (h − x̂·mean(h∘x̂))` (T9211).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn rms_bwd_dx(
        x: &[f32],
        gamma: &[f32],
        stats: &[f32],
        bstats: &[f32],
        g: &[f32],
        dx: &mut [f32],
        n: u32,
        c: u32,
    ) {
        let i = quark_id();
        let total = n * c;
        let idx = if i < total { i } else { 0u32 };
        let row = idx / c;
        let col = idx % c;
        let rrms = stats[row as usize];
        let m2 = bstats[row as usize];
        let h = g[idx as usize] * gamma[col as usize];
        let xh = x[idx as usize] * rrms;
        if i < total {
            dx[idx as usize] = rrms * (h - xh * m2);
        }
    }

    /// RMSNorm γ gradient: one thread per column streams rows,
    /// `dγ_c = Σᵢ g·x̂`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn rms_bwd_dgamma(x: &[f32], stats: &[f32], g: &[f32], dgamma: &mut [f32], n: u32, c: u32) {
        let i = quark_id();
        let col = if i < c { i } else { 0u32 };

        let mut sgx: f32 = 0.0f32;
        let mut r: u32 = 0u32;
        while r < n {
            let idx = r * c + col;
            let rrms = stats[r as usize];
            sgx = sgx + g[idx as usize] * (x[idx as usize] * rrms);
            r = r + 1u32;
        }
        if i < c {
            dgamma[col as usize] = sgx;
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

/// Host dispatch: fused LayerNorm forward. `x`/`out` are `n×c` row-major,
/// `gamma`/`beta` are `c`, `stats` is `n×2` and receives `(μ, rstd)` per row
/// (the backward's input, exactly like SDPA's `(m, l)`).
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_forward(
    gpu: &Gpu,
    n: u32,
    c: u32,
    eps: f32,
    x: &Field<f32>,
    gamma: &Field<f32>,
    beta: &Field<f32>,
    out: &Field<f32>,
    stats: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, cu) = (n as usize, c as usize);
    check(x.len() == nu * cu, "layer_norm: X length must be n*c")?;
    check(gamma.len() == cu, "layer_norm: GAMMA length must be c")?;
    check(beta.len() == cu, "layer_norm: BETA length must be c")?;
    check(out.len() == nu * cu, "layer_norm: OUT length must be n*c")?;
    check(
        stats.len() == nu * 2,
        "layer_norm: STATS length must be n*2",
    )?;
    if n == 0 || c == 0 {
        return Ok(());
    }

    let mut w = dsl::ln_stats(gpu)?;
    w.bind(0, x);
    w.bind(1, stats);
    w.set_value(2, n);
    w.set_value(3, c);
    w.set_value(4, eps);
    gpu.dispatch(&w, n)?.wait()?;

    let mut w = dsl::ln_fwd(gpu)?;
    w.bind(0, x);
    w.bind(1, gamma);
    w.bind(2, beta);
    w.bind(3, stats);
    w.bind(4, out);
    w.set_value(5, n);
    w.set_value(6, c);
    gpu.dispatch(&w, n * c)?.wait()?;
    Ok(())
}

/// Host dispatch: fused LayerNorm backward — the T9210 three-term formula
/// plus the column-reduced parameter gradients.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward(
    gpu: &Gpu,
    n: u32,
    c: u32,
    x: &Field<f32>,
    gamma: &Field<f32>,
    stats: &Field<f32>,
    g: &Field<f32>,
    bstats: &Field<f32>,
    dx: &Field<f32>,
    dgamma: &Field<f32>,
    dbeta: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, cu) = (n as usize, c as usize);
    check(x.len() == nu * cu, "layer_norm bwd: X length must be n*c")?;
    check(gamma.len() == cu, "layer_norm bwd: GAMMA length must be c")?;
    check(
        stats.len() == nu * 2,
        "layer_norm bwd: STATS length must be n*2",
    )?;
    check(g.len() == nu * cu, "layer_norm bwd: G length must be n*c")?;
    check(
        bstats.len() == nu * 2,
        "layer_norm bwd: BSTATS length must be n*2",
    )?;
    check(dx.len() == nu * cu, "layer_norm bwd: DX length must be n*c")?;
    check(
        dgamma.len() == cu,
        "layer_norm bwd: DGAMMA length must be c",
    )?;
    check(dbeta.len() == cu, "layer_norm bwd: DBETA length must be c")?;
    if n == 0 || c == 0 {
        return Ok(());
    }

    let mut w = dsl::ln_bwd_rowstats(gpu)?;
    w.bind(0, x);
    w.bind(1, gamma);
    w.bind(2, stats);
    w.bind(3, g);
    w.bind(4, bstats);
    w.set_value(5, n);
    w.set_value(6, c);
    gpu.dispatch(&w, n)?.wait()?;

    let mut w = dsl::ln_bwd_dx(gpu)?;
    w.bind(0, x);
    w.bind(1, gamma);
    w.bind(2, stats);
    w.bind(3, bstats);
    w.bind(4, g);
    w.bind(5, dx);
    w.set_value(6, n);
    w.set_value(7, c);
    gpu.dispatch(&w, n * c)?.wait()?;

    let mut w = dsl::ln_bwd_dparams(gpu)?;
    w.bind(0, x);
    w.bind(1, stats);
    w.bind(2, g);
    w.bind(3, dgamma);
    w.bind(4, dbeta);
    w.set_value(5, n);
    w.set_value(6, c);
    gpu.dispatch(&w, c)?.wait()?;
    Ok(())
}

/// Host dispatch: fused RMSNorm forward. `stats` is `n` and receives `rrms`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_forward(
    gpu: &Gpu,
    n: u32,
    c: u32,
    eps: f32,
    x: &Field<f32>,
    gamma: &Field<f32>,
    out: &Field<f32>,
    stats: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, cu) = (n as usize, c as usize);
    check(x.len() == nu * cu, "rms_norm: X length must be n*c")?;
    check(gamma.len() == cu, "rms_norm: GAMMA length must be c")?;
    check(out.len() == nu * cu, "rms_norm: OUT length must be n*c")?;
    check(stats.len() == nu, "rms_norm: STATS length must be n")?;
    if n == 0 || c == 0 {
        return Ok(());
    }

    let mut w = dsl::rms_stats(gpu)?;
    w.bind(0, x);
    w.bind(1, stats);
    w.set_value(2, n);
    w.set_value(3, c);
    w.set_value(4, eps);
    gpu.dispatch(&w, n)?.wait()?;

    let mut w = dsl::rms_fwd(gpu)?;
    w.bind(0, x);
    w.bind(1, gamma);
    w.bind(2, stats);
    w.bind(3, out);
    w.set_value(4, n);
    w.set_value(5, c);
    gpu.dispatch(&w, n * c)?.wait()?;
    Ok(())
}

/// Host dispatch: fused RMSNorm backward — the T9211 formula (no centering
/// term, no β).
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward(
    gpu: &Gpu,
    n: u32,
    c: u32,
    x: &Field<f32>,
    gamma: &Field<f32>,
    stats: &Field<f32>,
    g: &Field<f32>,
    bstats: &Field<f32>,
    dx: &Field<f32>,
    dgamma: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, cu) = (n as usize, c as usize);
    check(x.len() == nu * cu, "rms_norm bwd: X length must be n*c")?;
    check(gamma.len() == cu, "rms_norm bwd: GAMMA length must be c")?;
    check(stats.len() == nu, "rms_norm bwd: STATS length must be n")?;
    check(g.len() == nu * cu, "rms_norm bwd: G length must be n*c")?;
    check(bstats.len() == nu, "rms_norm bwd: BSTATS length must be n")?;
    check(dx.len() == nu * cu, "rms_norm bwd: DX length must be n*c")?;
    check(dgamma.len() == cu, "rms_norm bwd: DGAMMA length must be c")?;
    if n == 0 || c == 0 {
        return Ok(());
    }

    let mut w = dsl::rms_bwd_rowstats(gpu)?;
    w.bind(0, x);
    w.bind(1, gamma);
    w.bind(2, stats);
    w.bind(3, g);
    w.bind(4, bstats);
    w.set_value(5, n);
    w.set_value(6, c);
    gpu.dispatch(&w, n)?.wait()?;

    let mut w = dsl::rms_bwd_dx(gpu)?;
    w.bind(0, x);
    w.bind(1, gamma);
    w.bind(2, stats);
    w.bind(3, bstats);
    w.bind(4, g);
    w.bind(5, dx);
    w.set_value(6, n);
    w.set_value(7, c);
    gpu.dispatch(&w, n * c)?.wait()?;

    let mut w = dsl::rms_bwd_dgamma(gpu)?;
    w.bind(0, x);
    w.bind(1, stats);
    w.bind(2, g);
    w.bind(3, dgamma);
    w.set_value(4, n);
    w.set_value(5, c);
    gpu.dispatch(&w, c)?.wait()?;
    Ok(())
}

// ── Tape integration ─────────────────────────────────────────────────────

use crate::functional::{f32_field_to_array, lift, to_f32_host};
use quanta_array::Array;
use quanta_array::ToF64;
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(quanta_array::ArrayError::Gpu(QuantaError::invalid_param(
        msg,
    )))
}

/// Tape-differentiable fused LayerNorm over a `[N, C]` input with `[C]`
/// scale/shift. Forward runs the fused kernels (saving `(μ, rstd)`); the
/// backward is the proven T9210 three-term formula plus the column-reduced
/// parameter gradients, registered through [`Tape::custom_vjp`]. The
/// composed [`Var::layer_norm`] remains the differential-test oracle.
pub fn layer_norm_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
    gamma: &Var<T>,
    beta: &Var<T>,
    eps: f32,
) -> Result<Var<T>, AutogradError> {
    let xs = x.value().shape().to_vec();
    if xs.len() != 2 {
        return Err(bad("layer_norm_var: input must be 2-D [N, C]"));
    }
    let (n, c) = (xs[0], xs[1]);
    if gamma.value().shape() != [c] || beta.value().shape() != [c] {
        return Err(bad("layer_norm_var: gamma/beta must be [C]"));
    }
    let gpu = x.value().gpu().clone();

    let x_f32 = to_f32_host(&x.value())?;
    let ga_f32 = to_f32_host(&gamma.value())?;
    let be_f32 = to_f32_host(&beta.value())?;

    // Fused forward.
    let (out_f32, stats_f32) = {
        let xf = gpu.field::<f32>(n * c).map_err(lift)?;
        let gf = gpu.field::<f32>(c).map_err(lift)?;
        let bf = gpu.field::<f32>(c).map_err(lift)?;
        let of = gpu.field::<f32>(n * c).map_err(lift)?;
        let sf = gpu.field::<f32>(n * 2).map_err(lift)?;
        xf.write(&x_f32).map_err(lift)?;
        gf.write(&ga_f32).map_err(lift)?;
        bf.write(&be_f32).map_err(lift)?;
        layer_norm_forward(&gpu, n as u32, c as u32, eps, &xf, &gf, &bf, &of, &sf).map_err(lift)?;
        (of.read().map_err(lift)?, sf.read().map_err(lift)?)
    };

    let out_t: Vec<T> = out_f32.iter().map(|&v| T::from_f64(v as f64)).collect();
    let out_arr = Array::from_slice(&gpu, &out_t, &[n, c]).map_err(AutogradError::from)?;

    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let g_f32 = to_f32_host(g)?;
        let xf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        let gaf = gpu_b.field::<f32>(c).map_err(lift)?;
        let sf = gpu_b.field::<f32>(n * 2).map_err(lift)?;
        let gf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        let bsf = gpu_b.field::<f32>(n * 2).map_err(lift)?;
        let dxf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        let dgf = gpu_b.field::<f32>(c).map_err(lift)?;
        let dbf = gpu_b.field::<f32>(c).map_err(lift)?;
        xf.write(&x_f32).map_err(lift)?;
        gaf.write(&ga_f32).map_err(lift)?;
        sf.write(&stats_f32).map_err(lift)?;
        gf.write(&g_f32).map_err(lift)?;
        layer_norm_backward(
            &gpu_b, n as u32, c as u32, &xf, &gaf, &sf, &gf, &bsf, &dxf, &dgf, &dbf,
        )
        .map_err(lift)?;
        let dx = f32_field_to_array::<T>(&gpu_b, &dxf, &[n, c])?;
        let dgamma = f32_field_to_array::<T>(&gpu_b, &dgf, &[c])?;
        let dbeta = f32_field_to_array::<T>(&gpu_b, &dbf, &[c])?;
        Ok(vec![dx, dgamma, dbeta])
    };

    Ok(tape.custom_vjp(&[x, gamma, beta], out_arr, backward))
}

/// Tape-differentiable fused RMSNorm over `[N, C]` with `[C]` scale — the
/// T9211 backward (no centering term, no shift). The composed
/// [`Var::rms_norm`] remains the oracle.
pub fn rms_norm_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
    gamma: &Var<T>,
    eps: f32,
) -> Result<Var<T>, AutogradError> {
    let xs = x.value().shape().to_vec();
    if xs.len() != 2 {
        return Err(bad("rms_norm_var: input must be 2-D [N, C]"));
    }
    let (n, c) = (xs[0], xs[1]);
    if gamma.value().shape() != [c] {
        return Err(bad("rms_norm_var: gamma must be [C]"));
    }
    let gpu = x.value().gpu().clone();

    let x_f32 = to_f32_host(&x.value())?;
    let ga_f32 = to_f32_host(&gamma.value())?;

    let (out_f32, stats_f32) = {
        let xf = gpu.field::<f32>(n * c).map_err(lift)?;
        let gf = gpu.field::<f32>(c).map_err(lift)?;
        let of = gpu.field::<f32>(n * c).map_err(lift)?;
        let sf = gpu.field::<f32>(n).map_err(lift)?;
        xf.write(&x_f32).map_err(lift)?;
        gf.write(&ga_f32).map_err(lift)?;
        rms_norm_forward(&gpu, n as u32, c as u32, eps, &xf, &gf, &of, &sf).map_err(lift)?;
        (of.read().map_err(lift)?, sf.read().map_err(lift)?)
    };

    let out_t: Vec<T> = out_f32.iter().map(|&v| T::from_f64(v as f64)).collect();
    let out_arr = Array::from_slice(&gpu, &out_t, &[n, c]).map_err(AutogradError::from)?;

    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let g_f32 = to_f32_host(g)?;
        let xf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        let gaf = gpu_b.field::<f32>(c).map_err(lift)?;
        let sf = gpu_b.field::<f32>(n).map_err(lift)?;
        let gf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        let bsf = gpu_b.field::<f32>(n).map_err(lift)?;
        let dxf = gpu_b.field::<f32>(n * c).map_err(lift)?;
        let dgf = gpu_b.field::<f32>(c).map_err(lift)?;
        xf.write(&x_f32).map_err(lift)?;
        gaf.write(&ga_f32).map_err(lift)?;
        sf.write(&stats_f32).map_err(lift)?;
        gf.write(&g_f32).map_err(lift)?;
        rms_norm_backward(
            &gpu_b, n as u32, c as u32, &xf, &gaf, &sf, &gf, &bsf, &dxf, &dgf,
        )
        .map_err(lift)?;
        let dx = f32_field_to_array::<T>(&gpu_b, &dxf, &[n, c])?;
        let dgamma = f32_field_to_array::<T>(&gpu_b, &dgf, &[c])?;
        Ok(vec![dx, dgamma])
    };

    Ok(tape.custom_vjp(&[x, gamma], out_arr, backward))
}

/// Tape-differentiable GroupNorm over `[N, C]` with `[C]` scale/shift and
/// `C % groups == 0`: each row's channels split into `groups` segments,
/// each segment normalized independently, then the per-CHANNEL affine.
///
/// Composed over the proven LayerNorm core (T9210's backward) via the
/// reshape `[N, C] → [N·groups, C/groups]` — the normalization runs the
/// fused kernels with a unit/zero inner affine (their gradients are
/// computed and discarded; the per-channel γ/β affine happens outside,
/// through ordinary per-op VJPs). No new kernel, no new proof
/// obligation. GroupNorm(1) is LayerNorm-without-per-channel-stats-fusion;
/// GroupNorm(C) is InstanceNorm-per-channel.
pub fn group_norm_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
    gamma: &Var<T>,
    beta: &Var<T>,
    groups: usize,
    eps: f32,
) -> Result<Var<T>, AutogradError> {
    let xs = x.value().shape().to_vec();
    if xs.len() != 2 {
        return Err(bad("group_norm_var: input must be 2-D [N, C]"));
    }
    let (n, c) = (xs[0], xs[1]);
    if groups == 0 || c % groups != 0 {
        return Err(bad("group_norm_var: C must be divisible by groups"));
    }
    if gamma.value().shape() != [c] || beta.value().shape() != [c] {
        return Err(bad("group_norm_var: gamma/beta must be [C]"));
    }
    let cg = c / groups;
    let gpu = x.value().gpu().clone();

    let ones_host: Vec<T> = (0..cg).map(|_| T::from_f64(1.0)).collect();
    let zeros_host: Vec<T> = (0..cg).map(|_| T::from_f64(0.0)).collect();
    let ones = tape.var(Array::from_slice(&gpu, &ones_host, &[cg]).map_err(AutogradError::from)?);
    let zeros = tape.var(Array::from_slice(&gpu, &zeros_host, &[cg]).map_err(AutogradError::from)?);

    let xg = x.reshape(&[n * groups, cg])?;
    let xn = layer_norm_var(tape, &xg, &ones, &zeros, eps)?;
    xn.reshape(&[n, c])?
        .mul(&gamma.reshape(&[1, c])?)?
        .add(&beta.reshape(&[1, c])?)
}
