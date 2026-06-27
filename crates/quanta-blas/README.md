# quanta-blas

Verified-numerics BLAS for Quanta. The headline claim: **the only BLAS
that ships a mechanically-proven forward-error bound for every op** —
Higham-style `(1+δ)` bounds formalised in Lean
(`specs/verify/lean/Quanta/Blas/Reference.lean`).

Cross-backend by construction (Metal / Vulkan / WebGPU / CPU), built on
`quanta-tensor` (shape proofs) and `quanta-prims` (device-resident
reductions).

## Status — Level-1 + GEMV + tiled GEMM (f32) + mixed-precision GEMM (bf16/f16/fp8)

| op | signature | notes |
|----|-----------|-------|
| `scal` | `scal(gpu, α, &x)` | `x ← α·x`, in place |
| `axpy` | `axpy(gpu, α, &x, &y)` | `y ← α·x + y`, in place |
| `dot`  | `dot(gpu, &x, &y) -> f32` | `Σ xᵢ·yᵢ`, device-resident reduction |
| `nrm2` | `nrm2(gpu, &x) -> f32` | `√(Σ xᵢ²)` |
| `gemv` | `gemv(gpu, m, n, α, &a, &x, β, &y)` | `y ← α·A·x + β·y`, A row-major `m×n`, in place on y |
| `gemm` | `gemm(gpu, m, n, k, α, &a, &b, β, &c)` | `C ← α·A·B + β·C`, row-major, in place on C |
| `gemm_mixed` | `gemm_mixed(gpu, dtype, …, &a: Field<u16>, …)` | mixed-precision, 2-byte inputs (bf16/f16), C f32 |
| `gemm_mixed8` | `gemm_mixed8(gpu, dtype, …, &a: Field<u8>, …)` | mixed-precision, 1-byte inputs (fp8 E5M2/E4M3), C f32 |
| `gemv_mixed` / `gemv_mixed8` | `gemv_mixed(gpu, dtype, m, n, α, &a, &x, β, &y)` | mixed-precision GEMV (via `gemm_mixed*` N=1) |

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
dtype reuses the GEMM proof. int8 / int4 land next.

The crate is a pure-Rust reference library (`quanta_blas::reference`, the
differential-test oracle) until you enable `gpu` + a backend:

```toml
quanta-blas = { version = "0.1", features = ["gpu-metal"] } # or gpu-vulkan
```

Coming next: the cooperative-matrix / tensor-core `gemm` path (the vendor
perf gap) and the int8 / int4 quantized dtype matrix.

## Performance (honest framing)

The tiled GEMM is a real win over the naive kernel — **3.8× at N=512** on an
M1 Pro (369 vs 97 GFLOP/s), the speedup growing with size as the shared-memory
reuse compounds. But that is still only ~7% of the M1 Pro's fp32 peak: the
generic tiled kernel closes the *easy* gap over naive, not the vendor gap. The
strategic targets (~80% of vendor BLAS on tier-2 Apple-Silicon GPUs) are
reached by the **cooperative-matrix / tensor-core path** (`simdgroup_matrix`,
`VK_KHR_cooperative_matrix`), the next GEMM increment. Level-1 ops are
bandwidth-bound — the generic kernel is already near memory roofline.

We never hide where we lose — full numbers, backend coverage, and the gaps
are in [`PERFORMANCE.md`](PERFORMANCE.md).
