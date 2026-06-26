# quanta-blas

Verified-numerics BLAS for Quanta. The headline claim: **the only BLAS
that ships a mechanically-proven forward-error bound for every op** —
Higham-style `(1+δ)` bounds formalised in Lean
(`specs/verify/lean/Quanta/Blas/Reference.lean`).

Cross-backend by construction (Metal / Vulkan / WebGPU / CPU), built on
`quanta-tensor` (shape proofs) and `quanta-prims` (device-resident
reductions).

## Status — Level-1 + Level-2 GEMV + tiled GEMM (f32)

| op | signature | notes |
|----|-----------|-------|
| `scal` | `scal(gpu, α, &x)` | `x ← α·x`, in place |
| `axpy` | `axpy(gpu, α, &x, &y)` | `y ← α·x + y`, in place |
| `dot`  | `dot(gpu, &x, &y) -> f32` | `Σ xᵢ·yᵢ`, device-resident reduction |
| `nrm2` | `nrm2(gpu, &x) -> f32` | `√(Σ xᵢ²)` |
| `gemv` | `gemv(gpu, m, n, α, &a, &x, β, &y)` | `y ← α·A·x + β·y`, A row-major `m×n`, in place on y |
| `gemm` | `gemm(gpu, m, n, k, α, &a, &b, β, &c)` | `C ← α·A·B + β·C`, row-major, in place on C |

`scal`/`axpy` mutate in place (these ops are memory-bandwidth-bound;
avoiding a second buffer is the win). `dot`/`nrm2` multiply into a temp
field on the GPU and reduce there, so the vector never leaves the device.
`gemv` is a GEMM with one output column (`gemm(m, 1, n, …)`) — a gemv entry
*is* a gemm entry, so it reuses the gemm kernel and the same proven bound.
`gemm` uses the **tiled shared-memory** kernel (sub-tile problems route to a
naive kernel that skips the barrier overhead) — correct on every backend and
matching the proven Higham §3.5 contract.

The crate is a pure-Rust reference library (`quanta_blas::reference`, the
differential-test oracle) until you enable `gpu` + a backend:

```toml
quanta-blas = { version = "0.1", features = ["gpu-metal"] } # or gpu-vulkan
```

Coming next: the cooperative-matrix / tensor-core `gemm` path (the vendor
perf gap) and the f16/bf16/i8 dtype matrix.

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
