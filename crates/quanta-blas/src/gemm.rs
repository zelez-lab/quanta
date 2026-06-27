//! Level-3 BLAS GEMM (f32) — `C ← α·A·B + β·C`, all row-major.
//!
//! Two kernels behind one [`gemm`] entry point:
//!
//! - **tiled** (`gemm_f32_tiled`, the default): stages `TILE×TILE` blocks of
//!   A and B into workgroup-shared memory so each global load is reused
//!   `TILE` times — the standard GEMM bandwidth win. One workgroup of `256`
//!   threads computes one `16×16` output tile.
//! - **naive** (`gemm_f32_naive`): one thread per output entry, reads A/B
//!   straight from global memory. Used as a fallback for sub-tile problems
//!   (all dims ≤ `TILE`), where tiling/shared-mem + barriers buy nothing.
//!
//! Both satisfy the same proven contract (Higham §3.5, see
//! `specs/verify/lean/Quanta/Blas/Gemm.lean`); the differential tests pin
//! that each implementation meets it. Cooperative-matrix / tensor-core paths
//! are a later per-backend fork.
//!
//! Dimensions and scalars (`m, n, k, α, β`) are passed as kernel scalar
//! params (`set_value` at dispatch).

use quanta::{Field, Gpu, QuantaError};

/// Output tile edge. Workgroup is `TILE*TILE = 256` threads; each computes
/// one `C` entry within a `TILE×TILE` output block.
const TILE: u32 = 16;

#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    // `barrier()` shim (name resolution for the kernel body; the macro
    // lowers it to KernelOp::Barrier). Indexed `#[quanta::shared]` array
    // access (`tile[idx]` / `tile[idx] = v`) is rewritten by the kernel
    // macro into the shared load/store IR ops automatically — no manual
    // intrinsics needed.
    #[cfg(target_arch = "wasm32")]
    #[link(wasm_import_module = "quanta")]
    unsafe extern "C" {
        fn barrier();
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    fn barrier() {}

    /// Naive GEMM. Thread `i` computes one output entry
    /// `C[i/n, i%n] = α·Σₖ A[row,k]·B[k,col] + β·C[row,col]`, row-major.
    #[quanta::kernel(workgroup = [256])]
    pub fn gemm_f32_naive(
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
        // Clamp the working index so over-dispatched lanes compute a valid —
        // if redundant — entry; the store is guarded below. Keeping the loop
        // top-level (not nested in an `if`) avoids the structured-control
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

    /// Tiled shared-memory GEMM. Workgroup `256 = 16×16`; block `gid/256`
    /// owns one `16×16` output tile, thread `gid%256` owns entry `(tr,tc)`
    /// within it. K is swept one `16`-wide tile at a time: cooperatively load
    /// an A-tile and B-tile into shared memory (clamped to 0 past the matrix
    /// edge), barrier, accumulate from shared, barrier. The K-tile loop is
    /// top-level and only the final store is guarded — the loop-carried `acc`
    /// must not sit inside a bounds `if` (structured-control lowering hazard).
    #[quanta::kernel(workgroup = [256])]
    pub fn gemm_f32_tiled(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
    ) {
        #[quanta::shared]
        let a_tile: [f32; 256]; // 16×16, slot 0
        #[quanta::shared]
        let b_tile: [f32; 256]; // slot 1

        let gid = quark_id();
        let block = gid / 256u32;
        let tid = gid % 256u32;
        let tr = tid / 16u32; // local row in tile
        let tc = tid % 16u32; // local col in tile
        let local = tr * 16u32 + tc; // this thread's slot in each tile

        // Number of output tiles across N, and this block's tile coords.
        let tiles_n = (n + 15u32) / 16u32;
        let block_row = block / tiles_n;
        let block_col = block % tiles_n;

        // Global output coordinate this thread is responsible for.
        let row = block_row * 16u32 + tr;
        let col = block_col * 16u32 + tc;

        // Loop-invariant row/col validity masks (do NOT depend on the K-tile
        // counter), computed once. The per-tile K-edge mask is folded in via a
        // clamped index + value mask without a short-circuit `&&` (which the
        // WASM route mis-lowers when the second operand is loop-dependent).
        let row_ok = if row < m { 1.0f32 } else { 0.0f32 };
        let col_ok = if col < n { 1.0f32 } else { 0.0f32 };
        let row_c = if row < m { row } else { 0u32 };
        let col_c = if col < n { col } else { 0u32 };

        let num_k_tiles = (k + 15u32) / 16u32;
        let mut acc: f32 = 0.0f32;
        let mut kt: u32 = 0u32;
        while kt < num_k_tiles {
            // Cooperative load into shared. Indices are clamped in-bounds so
            // the buffer load is unconditional; out-of-range entries are zeroed
            // by multiplying 0/1 masks (indicator arithmetic, no branch / no
            // `&&`). The K-edge mask is a single comparison.
            let a_col = kt * 16u32 + tc;
            let a_kmask = if a_col < k { 1.0f32 } else { 0.0f32 };
            let a_col_c = if a_col < k { a_col } else { 0u32 };
            let a_val = a[(row_c * k + a_col_c) as usize] * (row_ok * a_kmask);
            a_tile[local as usize] = a_val;

            let b_row = kt * 16u32 + tr;
            let b_kmask = if b_row < k { 1.0f32 } else { 0.0f32 };
            let b_row_c = if b_row < k { b_row } else { 0u32 };
            let b_val = b[(b_row_c * n + col_c) as usize] * (b_kmask * col_ok);
            b_tile[local as usize] = b_val;

            unsafe { barrier() };

            // Accumulate this tile's contribution from shared memory.
            let mut p: u32 = 0u32;
            while p < 16u32 {
                let av = a_tile[(tr * 16u32 + p) as usize];
                let bv = b_tile[(p * 16u32 + tc) as usize];
                acc = acc + av * bv;
                p = p + 1u32;
            }

            unsafe { barrier() };
            kt = kt + 1u32;
        }

        // Store, guarded by a SINGLE precomputed bounds flag (mirrors the
        // naive kernel's `if i < total { … }` shape, which lowers correctly).
        // A `row < m && col < n` guard — and even nested ifs — collapsed in
        // the WASM-route lowering, leaving the store UNCONDITIONAL, so an
        // out-of-range lane (col = n) wrote to `row*n+col`, aliasing a valid
        // C entry. Folding the two bounds into one `u32` flag and guarding on
        // a single comparison keeps the guard intact through lowering.
        let in_bounds = row_ok * col_ok; // 1.0 iff row<m AND col<n, else 0.0
        let cv = c[(row * n + col) as usize];
        let result = alpha * acc + beta * cv;
        if in_bounds > 0.5f32 {
            c[(row * n + col) as usize] = result;
        }
    }
}

/// `gemm`: `C ← α·A·B + β·C`, all row-major. `a` is `m×k`, `b` is `k×n`,
/// `c` is `m×n` (read for `β·C`, overwritten with the result, in place).
///
/// Uses the tiled shared-memory kernel; sub-tile problems (all dims ≤ 16)
/// route to the naive kernel, which skips the barrier overhead where tiling
/// has no benefit. Errors on a shape mismatch between the declared
/// dimensions and the field lengths.
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
    if mu * nu == 0 || ku == 0 {
        return Ok(());
    }

    // Tensor-core path: when the device has cooperative matrices and the
    // problem fits the contract (C += A·B, i.e. α = β = 1, with m/n a multiple
    // of 32 and k a multiple of 8), the 4×4 register-blocked simdgroup_matrix
    // kernel beats the tiled kernel (~1.5× at N=512). The N≥512 threshold keeps
    // it to where the 32×32 tiles fill the GPU; smaller problems under-occupy
    // and the tiled kernel wins, so they stay on it.
    if alpha == 1.0
        && beta == 1.0
        && m >= 512
        && n >= 512
        && m.is_multiple_of(32)
        && n.is_multiple_of(32)
        && k.is_multiple_of(8)
        && gpu.supports_cooperative_matrix()
        && crate::mixed_tc::gemm_f32_tc(gpu, m, n, k, a, b, c).is_ok()
    {
        return Ok(());
    }

    // Sub-tile problems route to the naive kernel — it avoids the shared-mem
    // + double-barrier overhead where there's nothing to tile.
    if m <= TILE && n <= TILE && k <= TILE {
        dispatch_naive(gpu, m, n, k, alpha, a, b, beta, c)
    } else {
        dispatch_tiled(gpu, m, n, k, alpha, a, b, beta, c)
    }
}

/// Dispatch the naive kernel (one thread per output entry, `m·n` threads).
/// No shape validation — `gemm` does it.
#[allow(clippy::too_many_arguments)]
fn dispatch_naive(
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
    let mut wave = kernel::gemm_f32_naive(gpu)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c);
    wave.set_value(3, m);
    wave.set_value(4, n);
    wave.set_value(5, k);
    wave.set_value(6, alpha);
    wave.set_value(7, beta);
    gpu.dispatch(&wave, m * n)?.wait()?;
    Ok(())
}

/// Dispatch the tiled kernel — one 256-thread workgroup per 16×16 output
/// tile. No shape validation — `gemm` does it.
#[allow(clippy::too_many_arguments)]
fn dispatch_tiled(
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
    let total_threads = m.div_ceil(TILE) * n.div_ceil(TILE) * (TILE * TILE);
    let mut wave = kernel::gemm_f32_tiled(gpu)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c); // in place: C is read (β·C) and written
    wave.set_value(3, m);
    wave.set_value(4, n);
    wave.set_value(5, k);
    wave.set_value(6, alpha);
    wave.set_value(7, beta);
    gpu.dispatch(&wave, total_threads)?.wait()?;
    Ok(())
}

/// Benchmark/regression shim: force the **naive** kernel regardless of shape.
/// Not part of the stable surface — exists so `benches/gemm.rs` can compare
/// naive vs tiled on the same problem. Validates shapes like `gemm`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_naive(
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
    if m * n == 0 || k == 0 {
        return Ok(());
    }
    dispatch_naive(gpu, m, n, k, alpha, a, b, beta, c)
}

/// Benchmark/regression shim: force the **tiled** kernel regardless of shape.
/// See [`gemm_naive`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_tiled(
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
    if m * n == 0 || k == 0 {
        return Ok(());
    }
    dispatch_tiled(gpu, m, n, k, alpha, a, b, beta, c)
}
