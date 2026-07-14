//! Pure-Rust Level-1 BLAS reference — the differential-test oracle.
//!
//! Slow but provably-correct (see the Lean forward-error bounds in
//! `specs/verify/lean/Quanta/Blas/Reference.lean`). Every GPU op is
//! validated against these on a fixed corpus. `dot`/`nrm2` accumulate in
//! `f64` so the oracle is tighter than any `f32` summation order — the
//! GPU result is compared to it within a relative tolerance, matching the
//! Higham bound stated against the exact real sum.

/// `scal`: scale each element by `alpha`, in place (`x[i] ← alpha·x[i]`).
pub fn scal(alpha: f32, x: &mut [f32]) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

/// `axpy`: `y[i] ← alpha·x[i] + y[i]`, in place into `y`. `x` and `y`
/// must have the same length.
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "axpy: length mismatch");
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// `dot`: inner product `Σ x[i]·y[i]`, accumulated in `f64` for a tight
/// reference, returned as `f32`. `x` and `y` must have the same length.
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "dot: length mismatch");
    let mut acc = 0.0f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        acc += (xi as f64) * (yi as f64);
    }
    acc as f32
}

/// `nrm2`: Euclidean norm `sqrt(Σ x[i]²)`, accumulated in `f64`.
pub fn nrm2(x: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for &xi in x.iter() {
        acc += (xi as f64) * (xi as f64);
    }
    acc.sqrt() as f32
}

/// `gemm`: `C ← alpha·A·B + beta·C`, all row-major.
///
/// `a` is `m×k`, `b` is `k×n`, `c` is `m×n` (read for the `beta·C` term and
/// overwritten with the result). The inner product accumulates in `f64` for
/// a tight reference. Mirrors `quanta_blas::level1`-style contract; this is
/// the differential oracle for `gemm_f32`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "gemm: A must be m×k");
    assert_eq!(b.len(), k * n, "gemm: B must be k×n");
    assert_eq!(c.len(), m * n, "gemm: C must be m×n");
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += (a[row * k + p] as f64) * (b[p * n + col] as f64);
            }
            let cval = c[row * n + col];
            c[row * n + col] = (alpha as f64 * acc + beta as f64 * cval as f64) as f32;
        }
    }
}

/// `gemv`: `y ← alpha·A·x + beta·y`, A row-major `m×n`.
///
/// `a` is `m×n`, `x` is length `n`, `y` is length `m` (read for the `beta·y`
/// term and overwritten with the result). The inner product accumulates in
/// `f64` for a tight reference — the differential oracle for `gemv_f32`. A
/// gemv row is a gemm entry (`alpha·dot(row, x) + beta·y[i]`), so this mirrors
/// the gemm reference with `n = 1` output columns.
pub fn gemv(m: usize, n: usize, alpha: f32, a: &[f32], x: &[f32], beta: f32, y: &mut [f32]) {
    assert_eq!(a.len(), m * n, "gemv: A must be m×n");
    assert_eq!(x.len(), n, "gemv: x must be length n");
    assert_eq!(y.len(), m, "gemv: y must be length m");
    for row in 0..m {
        let mut acc = 0.0f64;
        for j in 0..n {
            acc += (a[row * n + j] as f64) * (x[j] as f64);
        }
        let yval = y[row];
        y[row] = (alpha as f64 * acc + beta as f64 * yval as f64) as f32;
    }
}

/// Mixed-precision GEMM oracle parameterised by the narrow→f32 load
/// conversion. A,B are raw `u16` bit patterns of the narrow dtype, C is f32.
/// Mirrors the GPU `gemm_mixed` kernel exactly: each A/B element is converted
/// to f32 on load via `to_f32`, the inner product accumulates **in f32
/// left-to-right** (the kernel's order), and the result is the f32 `α·acc +
/// β·C`. Accumulating in f32 (not f64) makes this the kernel's exact numerical
/// twin, so the differential test is a tight match, not a tolerance band.
#[allow(clippy::too_many_arguments)]
fn gemm_narrow<E: Copy>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[E],
    b: &[E],
    beta: f32,
    c: &mut [f32],
    to_f32: impl Fn(E) -> f32,
) {
    assert_eq!(a.len(), m * k, "gemm_narrow: A must be m×k");
    assert_eq!(b.len(), k * n, "gemm_narrow: B must be k×n");
    assert_eq!(c.len(), m * n, "gemm_narrow: C must be m×n");
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = to_f32(a[row * k + p]);
                let bv = to_f32(b[p * n + col]);
                acc += av * bv;
            }
            let cval = c[row * n + col];
            c[row * n + col] = alpha * acc + beta * cval;
        }
    }
}

/// `gemm_bf16`: mixed-precision GEMM oracle for bf16 inputs (the differential
/// twin of `gemm_mixed(GemmInputType::Bf16, …)`). `a` is `m×k`, `b` is `k×n`,
/// `c` is `m×n` (read for `β·C`, overwritten).
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u16],
    b: &[u16],
    beta: f32,
    c: &mut [f32],
) {
    gemm_narrow(m, n, k, alpha, a, b, beta, c, quanta_ir::dtype::bf16_to_f32);
}

/// `gemm_f16`: mixed-precision GEMM oracle for IEEE-half inputs (the
/// differential twin of `gemm_mixed(GemmInputType::F16, …)`).
#[allow(clippy::too_many_arguments)]
pub fn gemm_f16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u16],
    b: &[u16],
    beta: f32,
    c: &mut [f32],
) {
    gemm_narrow(m, n, k, alpha, a, b, beta, c, quanta_ir::dtype::f16_to_f32);
}

/// `gemm_fp8_e5m2`: mixed-precision GEMM oracle for fp8 E5M2 inputs (raw `u8`
/// bit patterns), the differential twin of
/// `gemm_mixed8(GemmInputType::Fp8E5M2, …)`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_e5m2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u8],
    b: &[u8],
    beta: f32,
    c: &mut [f32],
) {
    let (eb, mb) = quanta_ir::dtype::E5M2;
    gemm_narrow(m, n, k, alpha, a, b, beta, c, |x| {
        quanta_ir::dtype::fp8_to_f32(x, eb, mb)
    });
}

/// `gemm_fp8_e4m3`: mixed-precision GEMM oracle for fp8 E4M3 inputs.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_e4m3(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u8],
    b: &[u8],
    beta: f32,
    c: &mut [f32],
) {
    let (eb, mb) = quanta_ir::dtype::E4M3;
    gemm_narrow(m, n, k, alpha, a, b, beta, c, |x| {
        quanta_ir::dtype::fp8_to_f32(x, eb, mb)
    });
}

/// `gemm_q8_sym`: per-tensor symmetric int8 quantized GEMM oracle — the
/// differential twin of `gemm_quant(GemmQuantType::Q8Symmetric, …)`. A,B are
/// int8 codes (as `i32`), C is f32. Dequantisation folds into the effective
/// scale: the kernel accumulates `Σ(qa·qb)` in f32 (codes cast to f32) and
/// scales the whole sum by `α·sa·sb`, so this mirrors that order exactly.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q8_sym(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_scale: f32,
    b_scale: f32,
    a: &[i32],
    b: &[i32],
    beta: f32,
    c: &mut [f32],
) {
    let alpha_eff = alpha * a_scale * b_scale;
    gemm_narrow(m, n, k, alpha_eff, a, b, beta, c, |q| q as f32);
}

/// `gemm_q4_sym`: per-tensor symmetric int4 quantized GEMM oracle — the
/// differential twin of `gemm_quant4(GemmQuantType::Q4Symmetric, …)`. A,B are
/// int4 codes packed 8 per `u32` word (`a` is `ceil(m·k/8)` words, `b` is
/// `ceil(k·n/8)`); C is f32. Unpacks each logical code with `int4_unpack`
/// (word `idx/8`, nibble `idx%8`), exactly as the kernel's `Load { ty: I4 }`
/// does, then runs the same fold-into-alpha int8 path.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_sym(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_scale: f32,
    b_scale: f32,
    a: &[u32],
    b: &[u32],
    beta: f32,
    c: &mut [f32],
) {
    use quanta_ir::dtype::int4_unpack;
    let unpack = |packed: &[u32], idx: usize| int4_unpack(packed[idx / 8], (idx % 8) as u32) as f32;
    assert_eq!(a.len(), (m * k).div_ceil(8), "gemm_q4_sym: A packed words");
    assert_eq!(b.len(), (k * n).div_ceil(8), "gemm_q4_sym: B packed words");
    assert_eq!(c.len(), m * n, "gemm_q4_sym: C must be m×n");
    let alpha_eff = alpha * a_scale * b_scale;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += unpack(a, row * k + p) * unpack(b, p * n + col);
            }
            let cval = c[row * n + col];
            c[row * n + col] = alpha_eff * acc + beta * cval;
        }
    }
}

use crate::params::{Diag, Side, Trans, Uplo, trsm_plan};

/// Solve one substitution lane in `f64`: the effective triangular matrix is
/// `M[i,p] = a[i·rs + p·cs]` (see [`trsm_plan`]); `forward` picks the sweep
/// direction; `unit` skips the diagonal divide. `lane` holds the RHS values
/// and is overwritten with the solution; `alpha` scales each RHS entry as it
/// is first touched — the same order the GPU kernel uses.
#[allow(clippy::too_many_arguments)]
fn tri_lane_solve_f64(
    nt: usize,
    rs: usize,
    cs: usize,
    forward: bool,
    unit: bool,
    alpha: f64,
    a: &[f32],
    lane: &mut [f64],
) {
    let step = |i: usize, lane: &mut [f64]| {
        let mut acc = alpha * lane[i];
        let (lo, hi) = if forward { (0, i) } else { (i + 1, nt) };
        for (p, xp) in lane.iter().enumerate().take(hi).skip(lo) {
            acc -= (a[i * rs + p * cs] as f64) * xp;
        }
        lane[i] = if unit {
            acc
        } else {
            acc / (a[i * (rs + cs)] as f64)
        };
    };
    if forward {
        for i in 0..nt {
            step(i, lane);
        }
    } else {
        for i in (0..nt).rev() {
            step(i, lane);
        }
    }
}

/// `trsm`: solve `op(A)·X = α·B` (`side = Left`) or `X·op(A) = α·B`
/// (`side = Right`) for `X`, in place on `b`. `A` is triangular (`na×na`
/// with `na = m` for Left, `na = n` for Right, row-major, only the `uplo`
/// triangle referenced — and for `Diag::Unit`, the diagonal is not read
/// either); `b` is `m×n` row-major. Substitution runs in `f64` per lane
/// (column for Left, row for Right) — the differential oracle for the GPU
/// `trsm`. All `side`/`uplo`/`trans`/`diag` combinations are supported.
#[allow(clippy::too_many_arguments)]
pub fn trsm(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &mut [f32],
) {
    let na = match side {
        Side::Left => m,
        Side::Right => n,
    };
    assert_eq!(a.len(), na * na, "trsm: A must be na×na");
    assert_eq!(b.len(), m * n, "trsm: B must be m×n");
    let (rs, cs, forward) = trsm_plan(side, uplo, trans, na);
    let unit = diag == Diag::Unit;
    match side {
        Side::Left => {
            // Each RHS column is an independent solve of length m.
            for j in 0..n {
                let mut lane: Vec<f64> = (0..m).map(|t| b[t * n + j] as f64).collect();
                tri_lane_solve_f64(m, rs, cs, forward, unit, alpha as f64, a, &mut lane);
                for (t, v) in lane.iter().enumerate() {
                    b[t * n + j] = *v as f32;
                }
            }
        }
        Side::Right => {
            // Each RHS row is an independent solve of length n.
            for i in 0..m {
                let mut lane: Vec<f64> = (0..n).map(|t| b[i * n + t] as f64).collect();
                tri_lane_solve_f64(n, rs, cs, forward, unit, alpha as f64, a, &mut lane);
                for (t, v) in lane.iter().enumerate() {
                    b[i * n + t] = *v as f32;
                }
            }
        }
    }
}

/// `trsv`: solve `op(A)·x = b` for `x`, in place on `x` (which starts as
/// `b`). `A` is `n×n` triangular, row-major. Exactly `trsm` with a single
/// RHS column — the differential oracle for the GPU `trsv`.
pub fn trsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[f32], x: &mut [f32]) {
    assert_eq!(x.len(), n, "trsv: x must be length n");
    trsm(Side::Left, uplo, trans, diag, n, 1, 1.0, a, x);
}

/// `syrk`: symmetric rank-k update `C ← α·op(A)·op(A)ᵀ + β·C`, updating
/// **only** the `uplo` triangle of `C` (the opposite triangle is untouched).
/// `Trans::NoTrans` takes `A` as `n×k` (`C = α·A·Aᵀ + β·C`);
/// `Trans::Trans` takes `A` as `k×n` (`C = α·Aᵀ·A + β·C`). `C` is `n×n`
/// row-major. The inner product accumulates in `f64` — the differential
/// oracle for the GPU `syrk`.
#[allow(clippy::too_many_arguments)]
pub fn syrk(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    assert_eq!(a.len(), n * k, "syrk: A must be n×k (or k×n for Trans)");
    assert_eq!(c.len(), n * n, "syrk: C must be n×n");
    // op(A)[r,p] = a[r·ars + p·acs] for r in 0..n, p in 0..k.
    let (ars, acs) = match trans {
        Trans::NoTrans => (k, 1),
        Trans::Trans => (1, n),
    };
    for i in 0..n {
        for j in 0..n {
            let in_tri = match uplo {
                Uplo::Lower => j <= i,
                Uplo::Upper => j >= i,
            };
            if !in_tri {
                continue;
            }
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += (a[i * ars + p * acs] as f64) * (a[j * ars + p * acs] as f64);
            }
            let cval = c[i * n + j];
            c[i * n + j] = (alpha as f64 * acc + beta as f64 * cval as f64) as f32;
        }
    }
}

/// `potrf`: Cholesky factorisation of a symmetric positive-definite `n×n`
/// row-major matrix `a`, **in place**. With `uplo = Lower` it computes the
/// lower factor `L` (so `A = L·Lᵀ`) into `a`'s lower triangle; with
/// `uplo = Upper` the upper factor `U` (so `A = Uᵀ·U`) into the upper
/// triangle. The opposite (unreferenced) triangle is left untouched — the
/// caller's original entries survive there, exactly as LAPACK leaves them.
///
/// The right-looking column algorithm, accumulated in `f64` — the
/// differential oracle for the GPU `cholesky`. No positive-definiteness
/// check (as in LAPACK's error path): a non-SPD input yields a `nan` from
/// the diagonal `sqrt`, which propagates.
pub fn potrf(uplo: Uplo, n: usize, a: &mut [f32]) {
    assert_eq!(a.len(), n * n, "potrf: A must be n×n");
    // Work in f64 over the referenced triangle, then write back.
    match uplo {
        Uplo::Lower => {
            // L[j][j] = sqrt(A[j][j] − Σ_{p<j} L[j][p]²)
            // L[i][j] = (A[i][j] − Σ_{p<j} L[i][p]·L[j][p]) / L[j][j], i>j
            for j in 0..n {
                let mut d = a[j * n + j] as f64;
                for p in 0..j {
                    let ljp = a[j * n + p] as f64;
                    d -= ljp * ljp;
                }
                let ljj = d.sqrt();
                a[j * n + j] = ljj as f32;
                for i in (j + 1)..n {
                    let mut s = a[i * n + j] as f64;
                    for p in 0..j {
                        s -= (a[i * n + p] as f64) * (a[j * n + p] as f64);
                    }
                    a[i * n + j] = (s / ljj) as f32;
                }
            }
        }
        Uplo::Upper => {
            // U[j][j] = sqrt(A[j][j] − Σ_{p<j} U[p][j]²)
            // U[j][i] = (A[j][i] − Σ_{p<j} U[p][j]·U[p][i]) / U[j][j], i>j
            for j in 0..n {
                let mut d = a[j * n + j] as f64;
                for p in 0..j {
                    let upj = a[p * n + j] as f64;
                    d -= upj * upj;
                }
                let ujj = d.sqrt();
                a[j * n + j] = ujj as f32;
                for i in (j + 1)..n {
                    let mut s = a[j * n + i] as f64;
                    for p in 0..j {
                        s -= (a[p * n + j] as f64) * (a[p * n + i] as f64);
                    }
                    a[j * n + i] = (s / ujj) as f32;
                }
            }
        }
    }
}

/// `getrf`: LU factorisation with partial (row) pivoting of an `n×n`
/// row-major matrix `a`, in place. On return the strict lower triangle holds
/// the unit-lower multipliers `L` (implicit unit diagonal) and the upper
/// triangle (with diagonal) holds `U`, such that `P·A = L·U`. `ipiv[k]` is
/// the row swapped with row `k` at step `k` (LAPACK convention). Works in
/// f64 internally — the differential oracle for the GPU `lu`.
pub fn getrf(n: usize, a: &mut [f32], ipiv: &mut [usize]) {
    assert_eq!(a.len(), n * n, "getrf: A must be n×n");
    assert_eq!(ipiv.len(), n, "getrf: ipiv must have length n");
    // Work in an f64 scratch, then write back.
    let mut w: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    for k in 0..n {
        // Partial pivot: row r ≥ k with the largest |w[r,k]|.
        let mut best_row = k;
        let mut best_abs = w[k * n + k].abs();
        for r in (k + 1)..n {
            let v = w[r * n + k].abs();
            if v > best_abs {
                best_abs = v;
                best_row = r;
            }
        }
        ipiv[k] = best_row;
        if best_row != k {
            for j in 0..n {
                w.swap(k * n + j, best_row * n + j);
            }
        }
        let akk = w[k * n + k];
        for i in (k + 1)..n {
            let m = w[i * n + k] / akk;
            w[i * n + k] = m;
            for j in (k + 1)..n {
                w[i * n + j] -= m * w[k * n + j];
            }
        }
    }
    for (dst, &v) in a.iter_mut().zip(w.iter()) {
        *dst = v as f32;
    }
}

/// `geqrf`: Householder QR of an `m×n` (`m ≥ n`) row-major matrix, in f64.
/// On return the upper triangle of `a` holds `R`, the strict lower part the
/// essential reflector tails (`v_i`, `i > k`, with implicit `v_k = 1` under
/// the LAPACK scaling — here we store the *un-normalised* `v` so it matches
/// the GPU kernel's scratch layout: `v_k = x_k − α`), and `tau[k] = 2/(vᵀv)`.
pub fn geqrf(m: usize, n: usize, a: &mut [f32], tau: &mut [f32]) {
    assert_eq!(a.len(), m * n, "geqrf: A must be m×n");
    assert_eq!(tau.len(), n, "geqrf: tau must have length n");
    assert!(m >= n, "geqrf: requires m >= n");
    // Work in f64; keep a full reflector buffer V (m×n) as the GPU path does.
    let mut af: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let mut vv: Vec<f64> = vec![0.0; m * n];
    for k in 0..n {
        // ‖x‖² over rows i in [k, m).
        let mut nrm2 = 0.0f64;
        for i in k..m {
            let xi = af[i * n + k];
            nrm2 += xi * xi;
        }
        let xnorm = nrm2.sqrt();
        let xk = af[k * n + k];
        let sgn = if xk < 0.0 { -1.0 } else { 1.0 };
        let alpha = -sgn * xnorm;
        let vk = xk - alpha;
        let tail = nrm2 - xk * xk;
        let vtv = vk * vk + tail;
        let tau_k = if vtv > 0.0 { 2.0 / vtv } else { 0.0 };
        // Store v into the scratch column and reflector tail below R.
        for i in k..m {
            let vi = if i == k { vk } else { af[i * n + k] };
            vv[i * n + k] = vi;
        }
        af[k * n + k] = alpha;
        for i in (k + 1)..m {
            af[i * n + k] = vv[i * n + k];
        }
        tau[k] = tau_k as f32;
        // Apply H = I − τ·v·vᵀ to the trailing columns j > k.
        for j in (k + 1)..n {
            let mut dot = 0.0f64;
            for p in k..m {
                dot += vv[p * n + k] * af[p * n + j];
            }
            let scale = tau_k * dot;
            for i in k..m {
                af[i * n + j] -= scale * vv[i * n + k];
            }
        }
    }
    for (dst, &src) in a.iter_mut().zip(af.iter()) {
        *dst = src as f32;
    }
}

/// Reconstruct the explicit `m×m` orthogonal factor `Q = H_0·H_1···H_{n-1}`
/// from the reflectors that [`geqrf`] left in `a` (below-diagonal tails, with
/// `v_k = a[k,k]_pre` recovered as `x_k − α`) and `tau`. Returned row-major
/// `m×m`. Used by the differential test to check `Q·R ≈ A`.
///
/// The reflector `v` for column `k` is recomputed the same way `geqrf`
/// stored it. `v_k` is not recoverable from the packed `a` alone (its
/// diagonal now holds `α`), so this helper takes the **original** matrix
/// `a0` and re-runs the factorisation to rebuild each `v`, exactly mirroring
/// `geqrf`. It is a test oracle, not a production routine.
pub fn form_q(m: usize, n: usize, a0: &[f32]) -> Vec<f32> {
    assert_eq!(a0.len(), m * n);
    // Re-run the factorisation to recover the reflector vectors, then
    // accumulate Q = H_0···H_{n-1} applied to the identity.
    let mut af: Vec<f64> = a0.iter().map(|&x| x as f64).collect();
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(n); // v[k] full length m
    let mut taus: Vec<f64> = Vec::with_capacity(n);
    for k in 0..n {
        let mut nrm2 = 0.0f64;
        for i in k..m {
            let xi = af[i * n + k];
            nrm2 += xi * xi;
        }
        let xnorm = nrm2.sqrt();
        let xk = af[k * n + k];
        let sgn = if xk < 0.0 { -1.0 } else { 1.0 };
        let alpha = -sgn * xnorm;
        let vk = xk - alpha;
        let tail = nrm2 - xk * xk;
        let vtv = vk * vk + tail;
        let tau_k = if vtv > 0.0 { 2.0 / vtv } else { 0.0 };
        let mut v = vec![0.0f64; m];
        for i in k..m {
            v[i] = if i == k { vk } else { af[i * n + k] };
        }
        // Apply to trailing columns to advance af (so column k+1 sees the update).
        for j in (k + 1)..n {
            let mut dot = 0.0f64;
            for p in k..m {
                dot += v[p] * af[p * n + j];
            }
            let scale = tau_k * dot;
            for i in k..m {
                af[i * n + j] -= scale * v[i];
            }
        }
        af[k * n + k] = alpha;
        vs.push(v);
        taus.push(tau_k);
    }
    // Q = H_0·(H_1·(···)). Build by applying H_k (k = n-1 … 0) to identity.
    let mut q = vec![0.0f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    for k in (0..n).rev() {
        let v = &vs[k];
        let tau_k = taus[k];
        // Q ← H_k · Q : for each column j, Q[:,j] −= τ·v·(vᵀ Q[:,j]).
        for j in 0..m {
            let mut dot = 0.0f64;
            for p in k..m {
                dot += v[p] * q[p * m + j];
            }
            let scale = tau_k * dot;
            for i in k..m {
                q[i * m + j] -= scale * v[i];
            }
        }
    }
    q.iter().map(|&x| x as f32).collect()
}

/// `symm`: symmetric matrix-multiply reference. `side = Left`:
/// `C ← α·A·B + β·C` with `A` a `d×d` (`d = m`) symmetric matrix whose full
/// value is reconstructed from its `uplo` triangle. `side = Right`:
/// `C ← α·B·A + β·C` (`d = n`). `B`, `C` are `m×n` row-major. f64 accumulate.
#[allow(clippy::too_many_arguments)]
pub fn symm(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    let d = match side {
        Side::Left => m,
        Side::Right => n,
    };
    assert_eq!(a.len(), d * d, "symm: A must be d×d");
    assert_eq!(b.len(), m * n, "symm: B must be m×n");
    assert_eq!(c.len(), m * n, "symm: C must be m×n");
    // Asym[r,cc] from the stored triangle.
    let asym = |r: usize, cc: usize| -> f64 {
        let in_tri = match uplo {
            Uplo::Lower => cc <= r,
            Uplo::Upper => cc >= r,
        };
        let (rr, kk) = if in_tri { (r, cc) } else { (cc, r) };
        a[rr * d + kk] as f64
    };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..d {
                let (av, bv) = match side {
                    Side::Left => (asym(i, p), b[p * n + j] as f64),
                    Side::Right => (b[i * d + p] as f64, asym(p, j)),
                };
                acc += av * bv;
            }
            let cval = c[i * n + j];
            c[i * n + j] = (alpha as f64 * acc + beta as f64 * cval as f64) as f32;
        }
    }
}

/// `syr2k`: symmetric rank-2k update reference
/// `C ← α·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + β·C`, `C` symmetric, only the
/// `uplo` triangle written. `op(X)[r,p] = x[r·rs + p·cs]` with the same
/// stride convention as `syrk`. f64 accumulate.
#[allow(clippy::too_many_arguments)]
pub fn syr2k(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    assert_eq!(a.len(), n * k, "syr2k: A must be n×k (or k×n for Trans)");
    assert_eq!(b.len(), n * k, "syr2k: B must be n×k (or k×n for Trans)");
    assert_eq!(c.len(), n * n, "syr2k: C must be n×n");
    let (rs, cs) = match trans {
        Trans::NoTrans => (k, 1),
        Trans::Trans => (1, n),
    };
    for i in 0..n {
        for j in 0..n {
            let in_tri = match uplo {
                Uplo::Lower => j <= i,
                Uplo::Upper => j >= i,
            };
            if !in_tri {
                continue;
            }
            let mut acc = 0.0f64;
            for p in 0..k {
                let ai = a[i * rs + p * cs] as f64;
                let bj = b[j * rs + p * cs] as f64;
                let bi = b[i * rs + p * cs] as f64;
                let aj = a[j * rs + p * cs] as f64;
                acc += ai * bj + bi * aj;
            }
            let cval = c[i * n + j];
            c[i * n + j] = (alpha as f64 * acc + beta as f64 * cval as f64) as f32;
        }
    }
}

/// `trmm`: triangular matrix-multiply reference. `side = Left`:
/// `B ← α·op(A)·B` (`A` is `m×m`); `side = Right`: `B ← α·B·op(A)` (`A` is
/// `n×n`). `A` triangular; only the `uplo` triangle read, diagonal implicit
/// 1 under `Diag::Unit`. `B` is `m×n` row-major, overwritten. f64 accumulate.
#[allow(clippy::too_many_arguments)]
pub fn trmm(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &mut [f32],
) {
    let na = match side {
        Side::Left => m,
        Side::Right => n,
    };
    assert_eq!(a.len(), na * na, "trmm: A must be na×na");
    assert_eq!(b.len(), m * n, "trmm: B must be m×n");
    let (rs, cs, forward) = trsm_plan(side, uplo, trans, na);
    // M[i,p] = a[i·rs + p·cs]; M is lower-triangular iff `forward`.
    let mval = |i: usize, p: usize| -> f64 {
        if diag == Diag::Unit && i == p {
            1.0
        } else {
            a[i * rs + p * cs] as f64
        }
    };
    // Compute into a fresh buffer, then copy back (reference clarity; the GPU
    // kernel does it in place with a direction-ordered walk).
    let mut out = vec![0.0f32; m * n];
    let nt = na;
    let (nlanes, lb, ts) = match side {
        Side::Left => (n, 1usize, n),
        Side::Right => (m, n, 1usize),
    };
    for lane in 0..nlanes {
        let base = lane * lb;
        for i in 0..nt {
            // triangle of i: lower M -> p in [0, i]; upper M -> p in [i, nt).
            let (lo, hi) = if forward { (0, i + 1) } else { (i, nt) };
            let mut acc = 0.0f64;
            for p in lo..hi {
                acc += mval(i, p) * (b[base + p * ts] as f64);
            }
            out[base + i * ts] = (alpha as f64 * acc) as f32;
        }
    }
    b.copy_from_slice(&out);
}

/// `syev`: symmetric eigendecomposition reference via cyclic Jacobi, in f64.
/// Reads the `uplo` triangle of the `n×n` row-major `a`, reconstructs the
/// full symmetric matrix, and rotates it to diagonal form. Returns the
/// eigenvalues (ascending) in `w` and the orthonormal eigenvectors as the
/// columns of the returned `n×n` row-major matrix (column `j` ↔ `w[j]`).
///
/// The differential-test ground truth for [`crate::eigh`]. Iterative: it
/// runs cyclic sweeps until the off-diagonal norm falls below `tol` or a
/// sweep cap is hit — the same algorithm the GPU path uses, in f64.
pub fn syev(uplo: Uplo, n: usize, a: &[f32], w: &mut [f32]) -> Vec<f32> {
    assert_eq!(a.len(), n * n, "syev: A must be n×n");
    assert_eq!(w.len(), n, "syev: w must have length n");

    // Reconstruct the full symmetric matrix in f64.
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let src = match uplo {
                Uplo::Lower => {
                    if j <= i {
                        a[i * n + j]
                    } else {
                        a[j * n + i]
                    }
                }
                Uplo::Upper => {
                    if j >= i {
                        a[i * n + j]
                    } else {
                        a[j * n + i]
                    }
                }
            };
            m[i * n + j] = src as f64;
        }
    }
    // Eigenvector accumulator V = I.
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n <= 1 {
        for i in 0..n {
            w[i] = m[i * n + i] as f32;
        }
        return v.iter().map(|&x| x as f32).collect();
    }

    let tol = 1e-14;
    for _ in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() > off {
                    off = apq.abs();
                }
                if apq.abs() <= tol {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Row pass: rows p,q ← Jᵀ·rows.
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                // Column pass: cols p,q ← cols·J.
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                // Accumulate V ← V·J.
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
        if off <= tol {
            break;
        }
    }

    // Sort eigenvalues ascending; permute eigenvector columns.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        m[i * n + i]
            .partial_cmp(&m[j * n + j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut vout = vec![0.0f32; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        w[new_col] = m[old_col * n + old_col] as f32;
        for row in 0..n {
            vout[row * n + new_col] = v[row * n + old_col] as f32;
        }
    }
    vout
}

/// `gesvd`: economy singular value decomposition of an `m×n` (`m ≥ n`)
/// row-major matrix via one-sided Jacobi, in f64. Orthogonalises the
/// columns of `A` by right Givens rotations (`A ← A·J`), accumulating the
/// right factor `V`; on convergence the column norms are the singular
/// values and the normalised columns are `U`. Returns `(U, s, V)`:
/// `U` is `m×n` (orthonormal columns), `s` the `n` singular values
/// (descending), `V` the `n×n` right-singular matrix (orthonormal). The
/// differential-test ground truth for [`crate::svd`]. Iterative: cyclic
/// sweeps until every column pair is orthogonal to tolerance.
pub fn gesvd(m: usize, n: usize, a: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(a.len(), m * n, "gesvd: A must be m×n");
    assert!(m >= n, "gesvd: requires m >= n (economy SVD)");
    // Working column-orthogonalisation target `w = A` (m×n) and `V = I` (n×n),
    // both in f64.
    let mut w: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let mut vv = vec![0.0f64; n * n];
    for i in 0..n {
        vv[i * n + i] = 1.0;
    }
    const MAX_SWEEPS: usize = 60;
    const TOL: f64 = 1e-14;
    for _ in 0..MAX_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                // Column inner products of the current w.
                let mut alpha = 0.0; // ⟨w_p, w_p⟩
                let mut beta = 0.0; // ⟨w_q, w_q⟩
                let mut gamma = 0.0; // ⟨w_p, w_q⟩
                for i in 0..m {
                    let wip = w[i * n + p];
                    let wiq = w[i * n + q];
                    alpha += wip * wip;
                    beta += wiq * wiq;
                    gamma += wip * wiq;
                }
                if gamma.abs() > off {
                    off = gamma.abs();
                }
                if gamma.abs() <= TOL * (alpha * beta).sqrt().max(f64::MIN_POSITIVE) {
                    continue;
                }
                // One-sided Jacobi angle orthogonalising columns (p, q).
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (zeta * zeta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Apply the right rotation to w columns (p, q) and to V columns.
                for i in 0..m {
                    let wip = w[i * n + p];
                    let wiq = w[i * n + q];
                    w[i * n + p] = c * wip - s * wiq;
                    w[i * n + q] = s * wip + c * wiq;
                }
                for i in 0..n {
                    let vip = vv[i * n + p];
                    let viq = vv[i * n + q];
                    vv[i * n + p] = c * vip - s * viq;
                    vv[i * n + q] = s * vip + c * viq;
                }
            }
        }
        if off <= TOL {
            break;
        }
    }
    // Column norms of w are the singular values; normalise to get U columns.
    let mut sig = vec![0.0f64; n];
    for j in 0..n {
        let mut nrm = 0.0;
        for i in 0..m {
            nrm += w[i * n + j] * w[i * n + j];
        }
        sig[j] = nrm.sqrt();
    }
    // Sort descending; permute U columns and V columns to match.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        sig[j]
            .partial_cmp(&sig[i])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut u_out = vec![0.0f32; m * n];
    let mut s_out = vec![0.0f32; n];
    let mut v_out = vec![0.0f32; n * n];
    for (newc, &oldc) in order.iter().enumerate() {
        s_out[newc] = sig[oldc] as f32;
        let denom = if sig[oldc] > 1e-300 { sig[oldc] } else { 1.0 };
        for i in 0..m {
            u_out[i * n + newc] = (w[i * n + oldc] / denom) as f32;
        }
        for i in 0..n {
            v_out[i * n + newc] = vv[i * n + oldc] as f32;
        }
    }
    (u_out, s_out, v_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scal_basic() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        scal(2.0, &mut x);
        assert_eq!(x, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn axpy_basic() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![10.0f32, 20.0, 30.0];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn nrm2_basic() {
        assert_eq!(nrm2(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn gemm_basic() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemv_basic() {
        // [[1,2],[3,4]] · [5,6] = [1·5+2·6, 3·5+4·6] = [17, 39]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![5.0f32, 6.0];
        let mut y = vec![0.0f32; 2];
        gemv(2, 2, 1.0, &a, &x, 0.0, &mut y);
        assert_eq!(y, vec![17.0, 39.0]);
    }

    #[test]
    fn gemv_alpha_beta() {
        // y starts at 1; alpha=2, beta=3 → 2·(A·x) + 3·y
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let x = vec![5.0f32, 6.0];
        let mut y = vec![1.0f32; 2];
        gemv(2, 2, 2.0, &a, &x, 3.0, &mut y);
        // 2·[5,6] + 3·[1,1] = [13, 15]
        assert_eq!(y, vec![13.0, 15.0]);
    }

    #[test]
    fn gemm_bf16_basic() {
        use quanta_ir::dtype::f32_to_bf16;
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]] — exact in bf16
        // (all integers < 256 are bf16-representable, products exact in f32).
        let a: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| f32_to_bf16(x))
            .collect();
        let b: Vec<u16> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .map(|&x| f32_to_bf16(x))
            .collect();
        let mut c = vec![0.0f32; 4];
        gemm_bf16(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_bf16_rounds_inputs() {
        use quanta_ir::dtype::{bf16_to_f32, f32_to_bf16};
        // A value that is NOT bf16-representable gets rounded on the way in,
        // and the oracle uses the rounded value — confirming it models the
        // narrow-storage path, not exact f32.
        let x = 1.0001f32;
        let xb = f32_to_bf16(x);
        let xr = bf16_to_f32(xb); // the value the kernel actually multiplies
        let a = vec![xb];
        let b = vec![f32_to_bf16(1.0)];
        let mut c = vec![0.0f32];
        gemm_bf16(1, 1, 1, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c[0], xr);
        assert_ne!(c[0], x, "input must be quantised to bf16, not exact f32");
    }

    #[test]
    fn gemm_f16_basic() {
        use quanta_ir::dtype::f32_to_f16;
        // Same exact-integer case as bf16 — all representable in f16.
        let a: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| f32_to_f16(x))
            .collect();
        let b: Vec<u16> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .map(|&x| f32_to_f16(x))
            .collect();
        let mut c = vec![0.0f32; 4];
        gemm_f16(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_fp8_basic() {
        use quanta_ir::dtype::{E4M3, f32_to_fp8};
        // Small integers, exactly representable in E4M3:
        // [[1,2],[2,1]] · [[1,0],[0,2]] = [[1,4],[2,2]].
        let (eb, mb) = E4M3;
        let enc = |x: f32| f32_to_fp8(x, eb, mb);
        let a: Vec<u8> = [1.0f32, 2.0, 2.0, 1.0].iter().map(|&x| enc(x)).collect();
        let b: Vec<u8> = [1.0f32, 0.0, 0.0, 2.0].iter().map(|&x| enc(x)).collect();
        let mut c = vec![0.0f32; 4];
        gemm_fp8_e4m3(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c, vec![1.0, 4.0, 2.0, 2.0]);
    }

    #[test]
    fn gemm_q8_sym_basic() {
        // Codes [[2,4],[6,8]] · [[1,0],[0,1]] with sa=0.5, sb=0.25.
        // Dequantised A = [[1,2],[3,4]], B = identity → A·I = A.
        let a = vec![2i32, 4, 6, 8];
        let b = vec![1i32, 0, 0, 1];
        let mut c = vec![0.0f32; 4];
        gemm_q8_sym(2, 2, 2, 1.0, 0.5, 0.25, &a, &b, 0.0, &mut c);
        // α·sa·sb·(codes·codes) = 0.125 · [[2,4],[6,8]] = [[0.25,0.5],[0.75,1.0]]
        assert_eq!(c, vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn gemm_q4_sym_basic() {
        use quanta_ir::dtype::int4_pack;
        // 2×2 · 2×2, codes A=[[1,2],[3,-1]], B=identity, sa=1, sb=2.
        // Pack each matrix's 4 codes into one u32 word (nibbles 0..4).
        let pack = |codes: &[i32]| {
            let mut w = 0u32;
            for (i, &q) in codes.iter().enumerate() {
                w = int4_pack(w, i as u32, q);
            }
            vec![w]
        };
        let a = pack(&[1, 2, 3, -1]);
        let b = pack(&[1, 0, 0, 1]); // identity
        let mut c = vec![0.0f32; 4];
        gemm_q4_sym(2, 2, 2, 1.0, 1.0, 2.0, &a, &b, 0.0, &mut c);
        // dequant A=[[1,2],[3,-1]], B scaled by 2 → A·(2I) = 2A = [[2,4],[6,-2]]
        assert_eq!(c, vec![2.0, 4.0, 6.0, -2.0]);
    }

    #[test]
    fn trsv_lower_basic() {
        // L = [[2,0],[3,4]], b = [4, 22] → x = [2, 4]  (3·2 + 4·4 = 22).
        // Upper-triangle slot holds garbage — must never be read.
        let a = vec![2.0f32, 99.0, 3.0, 4.0];
        let mut x = vec![4.0f32, 22.0];
        trsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, 2, &a, &mut x);
        assert_eq!(x, vec![2.0, 4.0]);
    }

    #[test]
    fn trsv_upper_basic() {
        // U = [[2,3],[0,4]], b = [16, 8] → x₁ = 2, x₀ = (16−3·2)/2 = 5.
        let a = vec![2.0f32, 3.0, 99.0, 4.0];
        let mut x = vec![16.0f32, 8.0];
        trsv(Uplo::Upper, Trans::NoTrans, Diag::NonUnit, 2, &a, &mut x);
        assert_eq!(x, vec![5.0, 2.0]);
    }

    #[test]
    fn trsv_lower_transpose() {
        // Lᵀ·x = b with L = [[2,0],[3,4]] ⇒ [[2,3],[0,4]]·x = [16,8] → [5,2].
        let a = vec![2.0f32, 99.0, 3.0, 4.0];
        let mut x = vec![16.0f32, 8.0];
        trsv(Uplo::Lower, Trans::Trans, Diag::NonUnit, 2, &a, &mut x);
        assert_eq!(x, vec![5.0, 2.0]);
    }

    #[test]
    fn trsv_unit_diag_ignores_diagonal() {
        // Unit lower: implicit 1s on the diagonal; stored diagonal is trash.
        // [[1,0],[3,1]]·x = [2, 10] → x = [2, 4].
        let a = vec![777.0f32, 99.0, 3.0, 555.0];
        let mut x = vec![2.0f32, 10.0];
        trsv(Uplo::Lower, Trans::NoTrans, Diag::Unit, 2, &a, &mut x);
        assert_eq!(x, vec![2.0, 4.0]);
    }

    #[test]
    fn trsm_left_lower_multi_rhs() {
        // L = [[2,0],[3,4]], B = [[4,2],[22,7]], α = 1 → each column solved
        // independently: col₀ [4,22] → [2,4]; col₁ [2,7] → [1,(7−3)/4 = 1].
        let a = vec![2.0f32, 99.0, 3.0, 4.0];
        let mut b = vec![4.0f32, 2.0, 22.0, 7.0];
        trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            2,
            2,
            1.0,
            &a,
            &mut b,
        );
        assert_eq!(b, vec![2.0, 1.0, 4.0, 1.0]);
    }

    #[test]
    fn trsm_right_upper_rows() {
        // X·U = B with U = [[2,3],[0,4]], B = [[2,7]] (1×2).
        // x₀·2 = 2 → x₀ = 1; x₀·3 + x₁·4 = 7 → x₁ = 1.
        let a = vec![2.0f32, 3.0, 99.0, 4.0];
        let mut b = vec![2.0f32, 7.0];
        trsm(
            Side::Right,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1,
            2,
            1.0,
            &a,
            &mut b,
        );
        assert_eq!(b, vec![1.0, 1.0]);
    }

    #[test]
    fn trsm_alpha_scales_rhs() {
        // A = 2·I, B = [[4],[8]], α = 0.5 → X = 0.5·B / 2 = [[1],[2]].
        let a = vec![2.0f32, 99.0, 0.0, 2.0];
        let mut b = vec![4.0f32, 8.0];
        trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            2,
            1,
            0.5,
            &a,
            &mut b,
        );
        assert_eq!(b, vec![1.0, 2.0]);
    }

    #[test]
    fn syrk_lower_basic() {
        // A = [[1,2],[3,4]] → A·Aᵀ = [[5,11],[11,25]]. Lower only: the upper
        // slot of C keeps its initial value.
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![7.0f32; 4];
        syrk(Uplo::Lower, Trans::NoTrans, 2, 2, 1.0, &a, 0.0, &mut c);
        assert_eq!(c, vec![5.0, 7.0, 11.0, 25.0]);
    }

    #[test]
    fn syrk_upper_transposed() {
        // Same A read as Trans (A is k×n = 2×2 here): C = AᵀA = [[10,14],[14,20]].
        // Upper only; β = 1 keeps and adds to the initial C.
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![1.0f32; 4];
        syrk(Uplo::Upper, Trans::Trans, 2, 2, 1.0, &a, 1.0, &mut c);
        assert_eq!(c, vec![11.0, 15.0, 1.0, 21.0]);
    }

    #[test]
    fn gemm_alpha_beta() {
        // C starts at 1; alpha=2, beta=3 → 2·(A·B) + 3·C
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![1.0f32; 4];
        gemm(2, 2, 2, 2.0, &a, &b, 3.0, &mut c);
        // 2·B + 3·1 = [13,15,17,19]
        assert_eq!(c, vec![13.0, 15.0, 17.0, 19.0]);
    }
}
