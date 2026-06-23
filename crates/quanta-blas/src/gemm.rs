//! Level-3 BLAS GEMM (f32) — `C ← α·A·B + β·C`, all row-major.
//!
//! This is the **naive** generic kernel: one thread per output element
//! `C[m,n]`, looping over the shared dimension `k`. It is correct on every
//! backend and establishes the proven contract (Higham §3.5, see
//! `specs/verify/lean/Quanta/Blas/Gemm.lean`). The tiled / shared-memory /
//! cooperative-matrix paths that close the perf gap are a later increment —
//! per the recipe, generic first, fork only when a bench shows ≥2× loss.
//!
//! Dimensions and scalars (`m, n, k, α, β`) are passed as kernel scalar
//! params (`set_value` at dispatch). The 1-D dispatch of `m·n` threads is
//! mapped to the 2-D output: thread `i` → row `i / n`, col `i % n`.

use quanta::{Field, Gpu, QuantaError};

/// The naive GEMM kernel. Thread `i` computes one output entry
/// `C[i/n, i%n] = α·Σₖ A[row,k]·B[k,col] + β·C[row,col]`, all row-major.
#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    #[quanta::kernel(workgroup = [256])]
    pub fn gemm_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
    ) {
        let i = quark_id();
        let total = m * n;
        // Clamp the working index so over-dispatched lanes (Vulkan rounds the
        // grid up to a multiple of the workgroup size) compute a valid — if
        // redundant — entry; the store is guarded below. Keeping the loop at
        // the top level (not nested in an `if`) avoids the structured-control
        // lowering hazard with loop-carried accumulators.
        let idx = if i < total { i } else { 0u32 };
        let row = idx / n;
        let col = idx % n;

        let mut acc: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < k {
            let av = a[(row * k + p) as usize];
            let bv = b[(p * n + col) as usize];
            acc = acc + av * bv;
            p = p + 1u32;
        }

        let cv = c[idx as usize];
        let result = alpha * acc + beta * cv;
        if i < total {
            c[idx as usize] = result;
        }
    }
}

/// `gemm`: `C ← α·A·B + β·C`, all row-major. `a` is `m×k`, `b` is `k×n`,
/// `c` is `m×n` (read for `β·C`, overwritten with the result, in place).
///
/// Returns an error on a shape mismatch between the declared dimensions and
/// the field lengths.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    gpu: &Gpu,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<f32>,
    b: &Field<f32>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    if a.len() != mu * ku {
        return Err(QuantaError::invalid_param("gemm: A length must be m*k"));
    }
    if b.len() != ku * nu {
        return Err(QuantaError::invalid_param("gemm: B length must be k*n"));
    }
    if c.len() != mu * nu {
        return Err(QuantaError::invalid_param("gemm: C length must be m*n"));
    }
    let total = mu * nu;
    if total == 0 || ku == 0 {
        return Ok(());
    }

    let mut wave = kernel::gemm_f32(gpu)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c); // in place: C is both read (β·C) and written
    wave.set_value(3, m);
    wave.set_value(4, n);
    wave.set_value(5, k);
    wave.set_value(6, alpha);
    wave.set_value(7, beta);
    gpu.dispatch(&wave, total as u32)?.wait()?;
    Ok(())
}
