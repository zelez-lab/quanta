# BLAS (linear algebra on the GPU)

`quanta::sci::linalg` is the classic linear-algebra surface — Level-1/2/3 BLAS
plus the LAPACK-style factorisations — running on whatever backend you compiled
for. It works on raw device `Field<f32>` buffers you allocate and fill
yourself, which is what you want inside a hot loop; the friendly path is
`Array`'s own methods ([Array math](arrays-numpy.md)). What distinguishes it
is that the numerical contract is **mechanically proven**: Higham-style
`(1+δ)` forward-error bounds for the BLAS entries, formalised in Lean — the
[verification page](../../verification/index.md) says exactly what the chain
covers and what is still flagged follow-up. This page is a task-by-task
recipe.

```toml
[dependencies]
quanta = { version = "0.1", features = ["sci", "metal"] } # vulkan / software
```

## Setup and conventions

Everything is **f32 and row-major**. Dimensions are passed as `u32`; the
buffers are `Field`s whose length must match (a mismatch is an error, never a
silent misread). Results are written **in place** into a buffer you pass —
there is no allocation behind your back.

```rust,ignore
use quanta::sci::linalg::{self, Diag, Side, Trans, Uplo};

let gpu = quanta::init_cpu();    // real GPU: quanta::init()

let n: u32 = 512;                // dimensions are u32 …
let len = (n * n) as usize;      // … field lengths are usize
let a = gpu.field::<f32>(len)?;
let b = gpu.field::<f32>(len)?;
let c = gpu.field::<f32>(len)?;
a.write(&host_a)?;
b.write(&host_b)?;
c.write(&vec![0.0f32; len])?;
```

Four enums select the classic BLAS variants, and every routine that takes them
supports every combination:

| enum | values | meaning |
|---|---|---|
| `Uplo` | `Lower` / `Upper` | which triangle of `A` is read (or written) |
| `Trans` | `NoTrans` / `Trans` | use `A` or `Aᵀ` |
| `Diag` | `NonUnit` / `Unit` | `Unit` never reads the stored diagonal — it is implicitly 1 |
| `Side` | `Left` / `Right` | `A` multiplies/solves from the left or the right |

## Level 1 — vectors

```rust,ignore
linalg::scal(&gpu, 2.0, &x)?;            // x ← α·x        (in place)
linalg::axpy(&gpu, 2.0, &x, &y)?;        // y ← α·x + y    (in place on y)
let d = linalg::dot(&gpu, &x, &y)?;      // Σ xᵢ·yᵢ
let nrm = linalg::nrm2(&gpu, &x)?;       // ‖x‖₂
```

`scal`/`axpy` mutate their target because these ops are memory-bandwidth-bound
— avoiding a second buffer *is* the optimisation. `dot`/`nrm2` multiply into a
temp field on the device and reduce there, so the vector never leaves the GPU;
only the scalar comes back.

## Level 2 — matrix-vector

```rust,ignore
// y ← α·A·x + β·y, A row-major m×n, in place on y
linalg::gemv(&gpu, m, n, 1.0, &a, &x, 0.0, &y)?;

// solve op(A)·x = b for x, A n×n triangular — in place on x, which starts as b
linalg::trsv(&gpu, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, &a, &x)?;
```

`gemv` *is* `gemm` with one output column (`gemm(m, 1, n, …)`), so it runs the
same kernel and inherits the same proven bound. `trsv` is likewise `trsm` with
a single right-hand side; the substitution runs serially in one lane — correct
everywhere, and a parallel single-vector solver is a later optimisation.

## Level 3 — GEMM

```rust,ignore
// C ← α·A·B + β·C, A m×k, B k×n, C m×n — all row-major, in place on C
linalg::gemm(&gpu, m, n, k, 1.0, &a, &b, 0.0, &c)?;
```

One call, three kernels behind it, picked for you:

- **tensor cores** when the device enumerates a cooperative-matrix shape and
  the problem fits the contract — `C += A·B` (`α = β = 1`), `m`/`n` at least
  512 and multiples of 32, `k` a multiple of 8;
- the **naive** kernel for sub-tile problems (every dimension ≤ 16), where
  shared memory and its barriers buy nothing;
- the **tiled shared-memory** kernel for everything else.

The two ends of that range have pages of their own: [Matrix
multiply](matrix-multiply.md) writes the tiled kernel out as a
`#[quanta::kernel]` you can read and modify, and [Cooperative matrices
(tensor cores)](cooperative-matrix.md) covers the shape-is-a-hardware-fact
contract, `gemm_tc::<In, Acc>` and `tc_shape_for`. Reach for those when you
want to *choose* a kernel; `gemm` is the routing default.

## Level 3 — symmetric and triangular

The rest of Level-3 exists so you never pay for structure you already know
about — a symmetric operand is read from one triangle, a symmetric result is
written to one triangle:

| call | computes |
|---|---|
| `symm(&gpu, side, uplo, m, n, α, &a, &b, β, &c)` | `C ← α·A·B + β·C` (or `α·B·A`), `A` symmetric, only its `uplo` triangle read |
| `syrk(&gpu, uplo, trans, n, k, α, &a, β, &c)` | `C ← α·A·Aᵀ + β·C` (or `α·Aᵀ·A`), only `C`'s `uplo` triangle written |
| `syr2k(&gpu, uplo, trans, n, k, α, &a, &b, β, &c)` | `C ← α·(A·Bᵀ + B·Aᵀ) + β·C`, same one-triangle rule |
| `trmm(&gpu, side, uplo, trans, diag, m, n, α, &a, &b)` | `B ← α·op(A)·B` (or `α·B·op(A)`), `A` triangular, in place on `B` |
| `trsm(&gpu, side, uplo, trans, diag, m, n, α, &a, &b)` | solve `op(A)·X = α·B` (or `X·op(A) = α·B`), in place on `B` |

```rust,ignore
// Gram matrix of an n×k data block, lower triangle only.
linalg::syrk(&gpu, Uplo::Lower, Trans::NoTrans, n, k, 1.0, &a, 0.0, &c)?;
```

`trsm` runs one thread per independent right-hand-side lane — a column for
`Side::Left`, a row for `Side::Right` — so there are no barriers and no
cross-thread communication. As in BLAS, no singularity check is performed: a
zero diagonal with `Diag::NonUnit` yields `inf`/`nan`, it does not error.

## Mixed precision and quantized GEMM

The narrow-dtype entries store `A` and `B` in a narrow format, convert each
element to f32 on load, and **accumulate in f32** — the standard ML
mixed-precision path. The storage width picks the entry point:

```rust,ignore
use quanta::sci::linalg::{GemmInputType, GemmQuantType};

// bf16 / f16 — two bytes per element, in a Field<u16>
linalg::gemm_mixed(&gpu, GemmInputType::Bf16, m, n, k, 1.0, &a16, &b16, 0.0, &c)?;

// fp8 E5M2 / E4M3 — one byte per element, in a Field<u8>
linalg::gemm_mixed8(&gpu, GemmInputType::Fp8E4M3, m, n, k, 1.0, &a8, &b8, 0.0, &c)?;

// int8 codes + per-tensor scales, in a Field<i32>
linalg::gemm_quant(&gpu, GemmQuantType::Q8Symmetric, m, n, k, 1.0, sa, sb, &a_q, &b_q, 0.0, &c)?;

// int4 codes, 8 packed per Field<u32> word
linalg::gemm_quant4(&gpu, GemmQuantType::Q4Symmetric, m, n, k, 1.0, sa, sb, &a_q4, &b_q4, 0.0, &c)?;
```

Each has a `gemv_*` sibling (`gemv_mixed`, `gemv_mixed8`, `gemv_quant`,
`gemv_quant4`) for the one-column case. `C` is always `Field<f32>`. The
quantized forms fold dequantisation into the effective alpha —
`(sa·A)·(sb·B) = sa·sb·(A·B)` — so the same kernel runs over the raw codes.

The output contract does not change: the dtype is an implementation detail of
*how* the entry is computed, and the forward-error bound splits into the proven
f32 GEMM error over the narrow-rounded inputs plus the input-quantisation
error. For loading real int8/int4 checkpoints, see [Run a quantized
checkpoint](quantized-checkpoints.md).

## Factorisations

All three factor **in place** and destroy their input. That is the LAPACK
convention (`posv`, `gesv`, `gels`), and it is what makes them allocation-free
— pass a copy when the original `A` must survive. Like LAPACK, none of them
check for a singular or non-definite matrix: you get `inf`/`nan`, not an error.

```rust,ignore
// Cholesky: A = L·Lᵀ (Lower) or Uᵀ·U (Upper); the other triangle is untouched
linalg::cholesky(&gpu, Uplo::Lower, n, &a)?;
linalg::chol_solve(&gpu, Uplo::Lower, n, nrhs, &a, &b)?;   // A·X = B, SPD, in place on b

// LU with partial pivoting: P·A = L·U; ipiv[k] is the row swapped with row k
let ipiv = gpu.field::<u32>(n)?;
linalg::lu(&gpu, n, &a, &ipiv)?;
linalg::lu_solve(&gpu, n, nrhs, &a, &ipiv, &b)?;           // A·X = B, general
linalg::lu_inv(&gpu, n, &a, &ipiv, &inv)?;                 // A⁻¹ into `inv`

// QR by Householder (m ≥ n): R in the upper triangle, reflector tails below
let tau = gpu.field::<f32>(n)?;
linalg::qr(&gpu, m, n, &a, &tau)?;
linalg::lstsq(&gpu, m, n, nrhs, &a, &tau, &b)?;            // min‖A·X − B‖
```

`lstsq` takes `b` as `m×nrhs` and leaves the solution in its **top `n×nrhs`
block**; the remaining rows are scratch. Each factorisation issues `n`
sequential column steps — the inherent critical path — with the trailing update
of each step running in parallel on the device. Blocked-panel versions are a
performance increment, not a correctness one.

## Symmetric eigenvalues and SVD

These two do *not* destroy their input — they rotate an internal device copy:

```rust,ignore
// A = V·Λ·Vᵀ. Only the `uplo` triangle of `a` is read.
let w = gpu.field::<f32>(n)?;         // eigenvalues, ascending
let evecs = gpu.field::<f32>(n * n)?; // column j is the eigenvector for w[j]
linalg::eigh(&gpu, Uplo::Lower, n, &a, &w, &evecs)?;

// Economy SVD, A = U·diag(s)·Vᵀ, m ≥ n (m < n returns NotSupported)
let u = gpu.field::<f32>(m * n)?;     // m×n, orthonormal columns
let s = gpu.field::<f32>(n)?;         // singular values, descending
let v = gpu.field::<f32>(n * n)?;     // n×n, orthonormal — V, not Vᵀ
linalg::svd(&gpu, m, n, &a, &u, &s, &v)?;
```

Both are iterative (cyclic Jacobi for `eigh`, one-sided Jacobi for `svd`), so
their contract is a convergence tolerance rather than a fixed operation count.

## Checking against the reference

The crate is a pure-Rust reference library before it is a GPU library — the
oracle the differential tests run against, always available, no `gpu` feature
and no device needed:

```rust,ignore
use quanta::sci::linalg::{reference, Uplo};

let mut y = host_y.clone();
reference::gemv(m, n, 1.0, &host_a, &host_x, 0.0, &mut y);   // host-side truth
reference::potrf(Uplo::Lower, n, &mut host_a);               // Cholesky
let (u, s, v) = reference::gesvd(m, n, &host_a);
```

## Performance

The honest numbers — per-kernel GFLOP/s, backend coverage, and the strategies
that were tried and measured *slower* — live in
[`crates/sci/quanta-blas/PERFORMANCE.md`](https://github.com/zelez-lab/quanta/blob/main/crates/sci/quanta-blas/PERFORMANCE.md).
The short version, on a real M1 Pro: the tiled GEMM reaches 372 GFLOP/s at
N=512 against naive's 85, and the tensor-core kernel 553 — about 11% of the
part's ~5 TFLOP/s fp32 peak. Level-1 ops are bandwidth-bound and already near
the memory roofline, so their figure of merit is GB/s, not GFLOP/s.

## Notes

- **f32, row-major, dense.** No complex, no banded/packed storage, no
  column-major flag — pass `Trans::Trans` for a transposed operand.
- **The one deferred op is the general (non-symmetric) eigendecomposition**
  (`np.linalg.eig` on an arbitrary matrix). It is a deliberate gap: it needs
  complex eigenvalues and a Hessenberg → Francis-QR iteration that is
  research-grade to make robust, for an op whose consumers overwhelmingly want
  the symmetric case that `eigh` covers.
- All backends are equivalent — `init_cpu()` runs the software lane (what the
  tests use), `init()` picks a real GPU.
- Related: [Linear algebra](../tutorials/linear-algebra.md) for the gentler
  `Array`-level tour, and [FFT](fft.md) for the other verified numerical
  companion.
