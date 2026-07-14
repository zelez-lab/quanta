//! Linear algebra on `Array<f32>` — `matmul`, `dot`, `norm`.
//!
//! These delegate to `quanta-blas` (the verified Level-1 / GEMM ops), so the
//! numerical contract is the Higham-style forward-error bound proven in
//! `specs/verify/lean/Quanta/Blas/`. quanta-array sits *above* quanta-blas in
//! the stack: it owns the shape/layout, materializes contiguous operands, and
//! calls *down* into blas — blas never depends on the array surface.
//!
//! f32-only for this increment (blas Level-1 + GEMM are f32). The functional
//! API returns a fresh `Array`; the blas ops underneath are device-resident,
//! so no host round-trip happens for the math itself.

use quanta_blas::{
    Uplo, cholesky as blas_cholesky, dot as blas_dot, eigh as blas_eigh, gemm as blas_gemm,
    lstsq as blas_lstsq, lu_inv as blas_lu_inv, lu_solve as blas_lu_solve, nrm2 as blas_nrm2,
    qr as blas_qr, svd as blas_svd,
};

use crate::array::Array;
use crate::error::ArrayError;

/// The `(U, s, V)` triple returned by [`Array::svd`].
type SvdResult = Result<(Array<f32>, Array<f32>, Array<f32>), ArrayError>;

/// Build a fresh, device-resident `Array<f32>` holding a contiguous copy of
/// `src`'s logical data. The blas factorizations mutate their matrix argument
/// in place, so the array API — where an `Array` is logically immutable —
/// copies into a throwaway field first, leaving the caller's array untouched.
fn owned_copy(src: &Array<f32>) -> Result<Array<f32>, ArrayError> {
    let host = src.contiguous_or_self()?.to_vec()?;
    Array::<f32>::from_slice(src.gpu(), &host, src.shape())
}

/// Broadcast two batch-dim lists (numpy rules: right-aligned, each pair must be
/// equal or one of them 1). Returns the broadcast batch shape.
fn broadcast_batch(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        if da == db || da == 1 || db == 1 {
            out[i] = da.max(db);
        } else {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "matmul: batch dimensions are not broadcast-compatible",
            )));
        }
    }
    Ok(out)
}

/// Per-batch-dim stride (in matrix-slice units) for indexing an operand's flat
/// slices from an output-batch coordinate: the operand's own row-major batch
/// strides, right-aligned into the output batch rank, with a 0 wherever the
/// operand's extent is 1 (broadcast) so that dim is held at slice 0.
fn broadcast_batch_strides(operand_batch: &[usize], out_batch: &[usize]) -> Vec<usize> {
    let rank = out_batch.len();
    let off = rank - operand_batch.len();
    // Row-major strides over the operand's own batch dims.
    let mut own = vec![0usize; operand_batch.len()];
    let mut acc = 1usize;
    for i in (0..operand_batch.len()).rev() {
        own[i] = acc;
        acc *= operand_batch[i];
    }
    let mut out = vec![0usize; rank];
    for (i, slot) in out.iter_mut().enumerate() {
        if i >= off {
            let j = i - off;
            // dim absent (i < off) or extent 1 → broadcast, stride 0.
            *slot = if operand_batch[j] == 1 { 0 } else { own[j] };
        }
    }
    out
}

/// Row-major unravel of a flat index into a coordinate over `dims`.
fn unravel(mut flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut coord = vec![0usize; dims.len()];
    for i in (0..dims.len()).rev() {
        coord[i] = flat % dims[i];
        flat /= dims[i];
    }
    coord
}

/// Dot product of a coordinate with a stride list.
fn dot_usize(coord: &[usize], stride: &[usize]) -> usize {
    coord.iter().zip(stride).map(|(c, s)| c * s).sum()
}

/// Validate that `a` is a square 2-D matrix and return its dimension.
fn square_dim(a: &Array<f32>, ctx: &str) -> Result<usize, ArrayError> {
    if a.rank() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
            match ctx {
                "solve" => "solve: A must be a square 2-D matrix",
                "inv" => "inv: A must be a square 2-D matrix",
                "cholesky" => "cholesky: A must be a square 2-D matrix",
                "eigh" => "eigh: A must be a square 2-D matrix",
                _ => "linalg: A must be a square 2-D matrix",
            },
        )));
    }
    Ok(a.shape()[0])
}

impl Array<f32> {
    /// Matrix multiply (numpy `a @ b` / `np.matmul`).
    ///
    /// - **2-D × 2-D**: `self (m×k) · rhs (k×n) → (m×n)`, the plain matrix
    ///   product.
    /// - **Batched (rank ≥ 2)**: the last two dims are the matrix; the leading
    ///   dims are batch dims, **broadcast** between the two operands per numpy
    ///   rules. `A (…batchA, m, k) · B (…batchB, k, n) → (…batch, m, n)` with
    ///   `…batch = broadcast(…batchA, …batchB)`. E.g. `(B,m,k)·(k,n) → (B,m,n)`
    ///   and `(B,H,m,k)·(B,H,k,n) → (B,H,m,n)`.
    ///
    /// Returns a fresh contiguous row-major `Array`.
    pub fn matmul(&self, rhs: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        if self.rank() < 2 || rhs.rank() < 2 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "matmul: both operands must be at least 2-D",
            )));
        }
        // Plain 2-D product — the fast path (single gemm, no host round-trip).
        if self.rank() == 2 && rhs.rank() == 2 {
            return self.matmul_2d(rhs);
        }
        self.matmul_batched(rhs)
    }

    /// The 2-D matrix product `self (m×k) · rhs (k×n) → (m×n)` via one gemm.
    fn matmul_2d(&self, rhs: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = rhs.shape()[0];
        let n = rhs.shape()[1];
        if k != k2 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "matmul: inner dimensions disagree (A is m×k, B must be k×n)",
            )));
        }
        let a = self.contiguous_or_self()?;
        let b = rhs.contiguous_or_self()?;
        let c = Array::<f32>::zeros(self.gpu(), &[m, n])?;
        blas_gemm(
            self.gpu(),
            m as u32,
            n as u32,
            k as u32,
            1.0,
            a.field_ref(),
            b.field_ref(),
            0.0,
            c.field_ref(),
        )?;
        Ok(c)
    }

    /// Batched matmul: broadcast the leading (batch) dims, multiply the trailing
    /// 2-D matrices. Runs one gemm per output batch slice (correctness-first; a
    /// single strided/batched dispatch is a perf follow-up).
    fn matmul_batched(&self, rhs: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        let ash = self.shape();
        let bsh = rhs.shape();
        let (m, ka) = (ash[ash.len() - 2], ash[ash.len() - 1]);
        let (kb, n) = (bsh[bsh.len() - 2], bsh[bsh.len() - 1]);
        if ka != kb {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "matmul: inner dimensions disagree (A is …m×k, B must be …k×n)",
            )));
        }
        let k = ka;
        let a_batch = &ash[..ash.len() - 2];
        let b_batch = &bsh[..bsh.len() - 2];
        let out_batch = broadcast_batch(a_batch, b_batch)?;

        // Host-side batch orchestration over device gemms. Materialize both
        // operands to contiguous host data, gemm each broadcast slice, collect
        // the result, upload once.
        let a_host = self.contiguous_or_self()?.to_vec()?;
        let b_host = rhs.contiguous_or_self()?.to_vec()?;
        let batch_count: usize = out_batch.iter().product();
        let (a_mat, b_mat) = (m * k, k * n);
        let out_mat = m * n;
        let mut out_host = vec![0.0f32; batch_count * out_mat];

        // Strides (in matrix units) over each operand's own batch dims, with a
        // 0 stride wherever that operand's dim was broadcast (extent 1).
        let a_bstride = broadcast_batch_strides(a_batch, &out_batch);
        let b_bstride = broadcast_batch_strides(b_batch, &out_batch);

        let g = self.gpu();
        for flat in 0..batch_count {
            let coord = unravel(flat, &out_batch);
            let ai = dot_usize(&coord, &a_bstride);
            let bi = dot_usize(&coord, &b_bstride);
            let a_slice = &a_host[ai * a_mat..ai * a_mat + a_mat];
            let b_slice = &b_host[bi * b_mat..bi * b_mat + b_mat];
            let af = Array::<f32>::from_slice(g, a_slice, &[m, k])?;
            let bf = Array::<f32>::from_slice(g, b_slice, &[k, n])?;
            let cf = Array::<f32>::zeros(g, &[m, n])?;
            blas_gemm(
                g,
                m as u32,
                n as u32,
                k as u32,
                1.0,
                af.field_ref(),
                bf.field_ref(),
                0.0,
                cf.field_ref(),
            )?;
            let c_slice = cf.to_vec()?;
            out_host[flat * out_mat..flat * out_mat + out_mat].copy_from_slice(&c_slice);
        }

        let mut out_shape = out_batch;
        out_shape.push(m);
        out_shape.push(n);
        Array::<f32>::from_slice(g, &out_host, &out_shape)
    }

    /// Inner product of two 1-D arrays of equal length (numpy `np.dot` /
    /// `a @ b` for vectors). Device-resident reduction.
    pub fn dot(&self, rhs: &Array<f32>) -> Result<f32, ArrayError> {
        if self.rank() != 1 || rhs.rank() != 1 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "dot: both operands must be 1-D",
            )));
        }
        if self.len() != rhs.len() {
            return Err(ArrayError::LengthMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }
        let a = self.contiguous_or_self()?;
        let b = rhs.contiguous_or_self()?;
        Ok(blas_dot(self.gpu(), a.field_ref(), b.field_ref())?)
    }

    /// Euclidean (L2) norm of all elements, `√(Σ xᵢ²)` (numpy
    /// `np.linalg.norm`). Flattens any shape; device-resident reduction.
    pub fn norm(&self) -> Result<f32, ArrayError> {
        let a = self.contiguous_or_self()?;
        Ok(blas_nrm2(self.gpu(), a.field_ref())?)
    }

    // ── Factorization-backed solves (numpy.linalg parity) ───────────────
    //
    // Each copies its matrix argument into a throwaway field (the blas
    // factorizations are in-place), calls down into the verified `quanta-blas`
    // routine, and wraps the result as a fresh `Array`. The numerical contract
    // is the one proven in `specs/verify/lean/Quanta/Blas/`.

    /// Solve the general linear system `A · X = B` for `X` (numpy
    /// `np.linalg.solve`). `A` is square `n×n`, `B` is `n×nrhs` (a 1-D `B`
    /// of length `n` is treated as a single right-hand side and the result
    /// returned as an `n×1`). Uses LU with partial pivoting.
    pub fn solve(&self, b: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        let n = square_dim(self, "solve")?;
        let (brows, nrhs) = match b.rank() {
            1 => (b.shape()[0], 1usize),
            2 => (b.shape()[0], b.shape()[1]),
            _ => {
                return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                    "solve: B must be 1-D or 2-D",
                )));
            }
        };
        if brows != n {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "solve: B row count must match A's dimension",
            )));
        }
        let g = self.gpu();
        // Solve into a scratch copy of B. `lu_solve` factors A in place
        // internally (and fills `ipiv`), so A is passed raw — no separate
        // `lu` call.
        let a = owned_copy(self)?;
        let x = owned_copy(b)?;
        let ipiv = g.field::<u32>(n)?;
        ipiv.write(&vec![0u32; n])?;
        blas_lu_solve(
            g,
            n as u32,
            nrhs as u32,
            a.field_ref(),
            &ipiv,
            x.field_ref(),
        )?;
        x.reshape(&[n, nrhs])
    }

    /// Matrix inverse `A⁻¹` (numpy `np.linalg.inv`). `A` is square `n×n`.
    /// Uses LU with partial pivoting.
    pub fn inv(&self) -> Result<Array<f32>, ArrayError> {
        let n = square_dim(self, "inv")?;
        let g = self.gpu();
        // `lu_inv` factors A in place internally (via `lu_solve`), so A is
        // passed raw and `ipiv` is filled by the call.
        let a = owned_copy(self)?;
        let ipiv = g.field::<u32>(n)?;
        ipiv.write(&vec![0u32; n])?;
        let out = Array::<f32>::zeros(g, &[n, n])?;
        blas_lu_inv(g, n as u32, a.field_ref(), &ipiv, out.field_ref())?;
        Ok(out)
    }

    /// Cholesky factor of a symmetric positive-definite matrix, returning the
    /// **lower** triangular `L` with `A = L·Lᵀ` (numpy `np.linalg.cholesky`).
    /// The strict upper triangle of the result is zero.
    pub fn cholesky(&self) -> Result<Array<f32>, ArrayError> {
        let n = square_dim(self, "cholesky")?;
        let g = self.gpu();
        let a = owned_copy(self)?;
        blas_cholesky(g, Uplo::Lower, n as u32, a.field_ref())?;
        // blas leaves the input triangle untouched in the other half; zero it
        // so the returned L is a clean lower-triangular matrix (numpy shape).
        let mut host = a.to_vec()?;
        for r in 0..n {
            for c in (r + 1)..n {
                host[r * n + c] = 0.0;
            }
        }
        Array::<f32>::from_slice(g, &host, &[n, n])
    }

    /// Least-squares solution of `A·X ≈ B` (numpy `np.linalg.lstsq`), for an
    /// overdetermined or square system (`m ≥ n`). Returns `X` (`n×nrhs`)
    /// minimizing `‖A·X − B‖`, via the verified QR path in `quanta-blas`.
    ///
    /// `blas::lstsq` factors `A` and applies `Qᵀ`/back-substitutes in place on
    /// the `B` buffer, leaving the solution in its first `n` rows; we allocate
    /// a full `m×nrhs` scratch for `B`, run the solve, and slice off the top
    /// `n×nrhs` result.
    pub fn lstsq(&self, b: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        if self.rank() != 2 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "lstsq: A must be a 2-D matrix",
            )));
        }
        let (m, n) = (self.shape()[0], self.shape()[1]);
        if m < n {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "lstsq: requires m >= n (overdetermined or square)",
            )));
        }
        let (brows, nrhs) = match b.rank() {
            1 => (b.shape()[0], 1usize),
            2 => (b.shape()[0], b.shape()[1]),
            _ => {
                return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                    "lstsq: B must be 1-D or 2-D",
                )));
            }
        };
        if brows != m {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "lstsq: B row count must match A's row count",
            )));
        }
        let g = self.gpu();
        // `lstsq` factors A and solves in place on the B buffer (solution in
        // its first n rows), so A is passed raw and tau is filled by the call.
        let a = owned_copy(self)?;
        let rhs = owned_copy(b)?;
        let tau = g.field::<f32>(n)?;
        tau.write(&vec![0.0f32; n])?;
        blas_lstsq(
            g,
            m as u32,
            n as u32,
            nrhs as u32,
            a.field_ref(),
            &tau,
            rhs.field_ref(),
        )?;
        // The solution occupies the first n rows of the m×nrhs buffer.
        let full = rhs.to_vec()?;
        let mut sol = vec![0.0f32; n * nrhs];
        for i in 0..n {
            for j in 0..nrhs {
                sol[i * nrhs + j] = full[i * nrhs + j];
            }
        }
        Array::<f32>::from_slice(g, &sol, &[n, nrhs])
    }

    /// Reduced QR factorization `A = Q·R` of an `m×n` (`m ≥ n`) matrix
    /// (numpy `np.linalg.qr`, `mode="reduced"`): `Q` is `m×n` with
    /// orthonormal columns and `R` is `n×n` upper-triangular.
    ///
    /// `Q` is formed on the host from the Householder reflectors that
    /// `quanta-blas` leaves packed below `R`; the factorization itself
    /// (reflectors + `tau`) is computed on-device by the verified `qr` kernel.
    pub fn qr(&self) -> Result<(Array<f32>, Array<f32>), ArrayError> {
        if self.rank() != 2 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "qr: A must be a 2-D matrix",
            )));
        }
        let (m, n) = (self.shape()[0], self.shape()[1]);
        if m < n {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "qr: requires m >= n",
            )));
        }
        let g = self.gpu();
        let a = owned_copy(self)?;
        let tau = g.field::<f32>(n)?;
        tau.write(&vec![0.0f32; n])?;
        blas_qr(g, m as u32, n as u32, a.field_ref(), &tau)?;

        // Read the packed factorization back: `packed` holds R in its upper
        // triangle and the essential reflector tails (v_i, i>k) below the
        // diagonal of column k; `tau_h[k]` is 2/(vᵀv).
        let packed = a.to_vec()?;
        let tau_h = tau.read()?;

        // R: upper triangle of the m×n packed buffer, as an n×n matrix.
        let mut r = vec![0.0f32; n * n];
        for i in 0..n {
            for j in i..n {
                r[i * n + j] = packed[i * n + j];
            }
        }

        // Q = H_0·H_1···H_{n-1}, H_k = I − tau_k·v_k·v_kᵀ, applied to I (m×n).
        // v_k has v_k[k]=1, v_k[i]=packed[i*n+k] for i>k, 0 below. Accumulate
        // in f64 for accuracy, then downcast.
        let mut q = vec![0.0f64; m * n];
        for (i, qi) in q.iter_mut().enumerate().take(m * n) {
            let (row, col) = (i / n, i % n);
            *qi = if row == col { 1.0 } else { 0.0 };
        }
        for k in (0..n).rev() {
            let tk = tau_h[k] as f64;
            if tk == 0.0 {
                continue;
            }
            // Reflector v_k (length m). blas stores the *un-normalised* v: the
            // tail v_i (i>k) sits below the diagonal in `packed`, while the
            // diagonal itself was overwritten by α (= R[k,k]). Recover the
            // diagonal entry v_k from τ_k = 2/(v_kᵀv_k): v_k² = 2/τ_k − Σ_{i>k}v_i².
            // Its sign is that of (x_k − α) = −sign(α) (α = −sgn·‖x‖).
            let mut v = vec![0.0f64; m];
            let mut tail = 0.0f64;
            for i in (k + 1)..m {
                let vi = packed[i * n + k] as f64;
                v[i] = vi;
                tail += vi * vi;
            }
            let vk2 = (2.0 / tk - tail).max(0.0);
            let alpha = packed[k * n + k] as f64;
            let sign = if alpha < 0.0 { 1.0 } else { -1.0 };
            v[k] = sign * vk2.sqrt();
            // Q ← (I − tk·v·vᵀ)·Q : for each column c, w = vᵀ·Q[:,c]; Q[:,c] −= tk·w·v
            for c in 0..n {
                let mut w = 0.0f64;
                for i in 0..m {
                    w += v[i] * q[i * n + c];
                }
                let f = tk * w;
                for i in 0..m {
                    q[i * n + c] -= f * v[i];
                }
            }
        }
        let q_f32: Vec<f32> = q.iter().map(|&x| x as f32).collect();
        let q_arr = Array::<f32>::from_slice(g, &q_f32, &[m, n])?;
        let r_arr = Array::<f32>::from_slice(g, &r, &[n, n])?;
        Ok((q_arr, r_arr))
    }

    /// Eigendecomposition of a **symmetric** matrix, GPU-native (numpy
    /// `np.linalg.eigh`): returns `(eigenvalues [n], eigenvectors [n, n])`
    /// with eigenvalues **ascending** and column `j` of the eigenvector
    /// matrix pairing with `eigenvalues[j]`, so `A · V ≈ V · diag(λ)`.
    ///
    /// Reads only the lower triangle of `A` (the matrix is assumed symmetric).
    /// The decomposition runs entirely on-device via the verified `quanta-blas`
    /// cyclic-Jacobi kernel. (The host-driven [`Array::eigh_symmetric`], which
    /// sorts descending, remains for callers that want that convention.)
    pub fn eigh(&self) -> Result<(Array<f32>, Array<f32>), ArrayError> {
        let n = square_dim(self, "eigh")?;
        let g = self.gpu();
        let a = owned_copy(self)?;
        let w = g.field::<f32>(n)?;
        w.write(&vec![0.0f32; n])?;
        let v = g.field::<f32>(n * n)?;
        v.write(&vec![0.0f32; n * n])?;
        blas_eigh(g, Uplo::Lower, n as u32, a.field_ref(), &w, &v)?;
        let w_arr = Array::<f32>::from_slice(g, &w.read()?[..n], &[n])?;
        let v_arr = Array::<f32>::from_slice(g, &v.read()?[..n * n], &[n, n])?;
        Ok((w_arr, v_arr))
    }

    /// Economy singular value decomposition `A = U · diag(s) · Vᵀ` of an
    /// `m×n` (`m ≥ n`) matrix (numpy `np.linalg.svd`, `full_matrices=False`):
    /// returns `(U [m, n], s [n], V [n, n])` with the singular values `s`
    /// **descending** and `U`, `V` orthonormal. Uses one-sided Jacobi.
    ///
    /// Note the third return is `V` (not `Vᵀ`): reconstruct with
    /// `U · diag(s) · Vᵀ`.
    pub fn svd(&self) -> SvdResult {
        if self.rank() != 2 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "svd: A must be a 2-D matrix",
            )));
        }
        let (m, n) = (self.shape()[0], self.shape()[1]);
        if m < n {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "svd: requires m >= n (economy SVD)",
            )));
        }
        let g = self.gpu();
        let a = owned_copy(self)?;
        let u = g.field::<f32>(m * n)?;
        u.write(&vec![0.0f32; m * n])?;
        let s = g.field::<f32>(n)?;
        s.write(&vec![0.0f32; n])?;
        let v = g.field::<f32>(n * n)?;
        v.write(&vec![0.0f32; n * n])?;
        blas_svd(g, m as u32, n as u32, a.field_ref(), &u, &s, &v)?;
        let u_arr = Array::<f32>::from_slice(g, &u.read()?[..m * n], &[m, n])?;
        let s_arr = Array::<f32>::from_slice(g, &s.read()?[..n], &[n])?;
        let v_arr = Array::<f32>::from_slice(g, &v.read()?[..n * n], &[n, n])?;
        Ok((u_arr, s_arr, v_arr))
    }
}
