# quanta-blas

Verified-numerics BLAS for Quanta. The headline claim: **the only BLAS
that ships a mechanically-proven forward-error bound for every op** —
Higham-style `(1+δ)` bounds formalised in Lean
(`specs/verify/lean/Quanta/Blas/Reference.lean`).

Cross-backend by construction (Metal / Vulkan / WebGPU / CPU), built on
`quanta-tensor` (shape proofs) and `quanta-prims` (device-resident
reductions).

## Status — Level-1 + GEMV + tiled GEMM (f32) + mixed-precision (bf16/f16/fp8/int8/int4)

| op | signature | notes |
|----|-----------|-------|
| `scal` | `scal(gpu, α, &x)` | `x ← α·x`, in place |
| `axpy` | `axpy(gpu, α, &x, &y)` | `y ← α·x + y`, in place |
| `dot`  | `dot(gpu, &x, &y) -> f32` | `Σ xᵢ·yᵢ`, device-resident reduction |
| `nrm2` | `nrm2(gpu, &x) -> f32` | `√(Σ xᵢ²)` |
| `gemv` | `gemv(gpu, m, n, α, &a, &x, β, &y)` | `y ← α·A·x + β·y`, A row-major `m×n`, in place on y |
| `gemm` | `gemm(gpu, m, n, k, α, &a, &b, β, &c)` | `C ← α·A·B + β·C`, row-major, in place on C (auto-routes to the tensor-core path when it fits) |
| `gemm_f32_tc` | `gemm_f32_tc(gpu, m, n, k, &a, &b, &c)` | `C ← A·B + C` via Metal `simdgroup_matrix` (4×4 blocked); needs cooperative-matrix support, m/n mult 32, k mult 8 |
| `gemm_mixed` | `gemm_mixed(gpu, dtype, …, &a: Field<u16>, …)` | mixed-precision, 2-byte inputs (bf16/f16), C f32 |
| `gemm_mixed8` | `gemm_mixed8(gpu, dtype, …, &a: Field<u8>, …)` | mixed-precision, 1-byte inputs (fp8 E5M2/E4M3), C f32 |
| `gemv_mixed` / `gemv_mixed8` | `gemv_mixed(gpu, dtype, m, n, α, &a, &x, β, &y)` | mixed-precision GEMV (via `gemm_mixed*` N=1) |
| `gemm_quant` | `gemm_quant(gpu, qty, …, sa, sb, &a: Field<i32>, …)` | int8 (Q8 symmetric) codes + per-tensor scales, C f32 |
| `gemm_quant4` | `gemm_quant4(gpu, qty, …, sa, sb, &a: Field<u32>, …)` | int4 (Q4 symmetric), 8 codes packed per word, C f32 |
| `gemv_quant` / `gemv_quant4` | `gemv_quant(gpu, qty, m, n, α, sa, sx, &a, &x, β, &y)` | quantized GEMV (via `gemm_quant*` N=1) |

`scal`/`axpy` mutate in place (these ops are memory-bandwidth-bound;
avoiding a second buffer is the win). `dot`/`nrm2` multiply into a temp
field on the GPU and reduce there, so the vector never leaves the device.
`gemv` is a GEMM with one output column (`gemm(m, 1, n, …)`) — a gemv entry
*is* a gemm entry, so it reuses the gemm kernel and the same proven bound.
`gemm` uses the **tiled shared-memory** kernel (sub-tile problems route to a
naive kernel that skips the barrier overhead) — correct on every backend and
matching the proven Higham §3.5 contract.

The mixed-precision entries store A,B in a narrow dtype, convert each element
to f32 on load, and **accumulate in f32** — the standard ML mixed-precision
path. 2-byte dtypes (**bf16**, **f16**) ride a `Field<u16>` via `gemm_mixed`;
1-byte dtypes (**fp8 E5M2 / E4M3**) ride a `Field<u8>` via `gemm_mixed8` (the
storage width is intrinsic to the dtype, so it picks the entry — passing the
wrong one errors). The output contract is the same real-arithmetic GEMM entry;
the dtype is an implementation detail of *how* the entry is computed. The
forward-error bound splits into the proven f32 GEMM error over the
narrow-rounded inputs plus the input-quantisation error
(`Quanta.Blas.gemmEntry_narrow_error_split`, with per-dtype instances), so each
dtype reuses the GEMM proof.

The quantized entries take per-tensor symmetric integer codes plus scales
`sa`/`sb`: **int8** (Q8) one code per `Field<i32>` slot via `gemm_quant`; **int4**
(Q4) packed 8 codes per `Field<u32>` word via `gemm_quant4`. Dequantisation
folds into the effective alpha — `(sa·A)·(sb·B) = sa·sb·(A·B)` — so the same
kernel runs over the raw codes and the same split bound applies (a quantized
entry is the real GEMM entry over the dequantised inputs). int4's nibble
unpacking happens in the `I4` load on every backend.

The crate is a pure-Rust reference library (`quanta_blas::reference`, the
differential-test oracle) until you enable `gpu` + a backend:

```toml
quanta-blas = { version = "0.1", features = ["gpu-metal"] } # or gpu-vulkan
```

The dtype matrix is complete (f32 + bf16/f16/fp8/int8/int4) and `gemm` has a
**tensor-core** path (`gemm_f32_tc`, Metal `simdgroup_matrix`). Coming next:
deeper fragment reuse for GEMM and the Vulkan cooperative-matrix path.

## Performance (honest framing)

On a real M1 Pro (Metal): the tiled GEMM beats naive (**372 vs 85 GFLOP/s at
N=512**), and the **tensor-core kernel beats tiled — 553 GFLOP/s at N=512,
~1.5×** — using `simdgroup_matrix` with 4×4 fragment register-blocking (8
fragment loads feed 16 MMAs per K-step). That is ~11% of the M1 Pro's ~5 TFLOP/s
fp32 peak (tiled was ~7.5%). `gemm()` routes to the tensor-core path
automatically when the device supports cooperative matrices and the problem fits
(`C += A·B`, N≥512, m/n multiple of 32, k multiple of 8), else the tiled kernel.
The remaining vendor gap (~80% of Accelerate/cuBLAS) needs threadgroup-shared
staging of the A/B tiles on top of the register blocking. The Vulkan
`VK_KHR_cooperative_matrix` path is not yet wired (Metal-only today). Level-1
ops are bandwidth-bound — the generic kernel is already near memory roofline.

We never hide where we lose — full numbers, backend coverage, and the gaps
are in [`PERFORMANCE.md`](PERFORMANCE.md).
