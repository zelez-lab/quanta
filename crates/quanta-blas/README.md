# quanta-blas

Verified-numerics BLAS for Quanta. The headline claim: **the only BLAS
that ships a mechanically-proven forward-error bound for every op** —
Higham-style `(1+δ)` bounds formalised in Lean
(`specs/verify/lean/Quanta/Blas/Reference.lean`).

Cross-backend by construction (Metal / Vulkan / WebGPU / CPU), built on
`quanta-tensor` (shape proofs) and `quanta-prims` (device-resident
reductions).

## Status — Level-1 (f32)

| op | signature | notes |
|----|-----------|-------|
| `scal` | `scal(gpu, α, &x)` | `x ← α·x`, in place |
| `axpy` | `axpy(gpu, α, &x, &y)` | `y ← α·x + y`, in place |
| `dot`  | `dot(gpu, &x, &y) -> f32` | `Σ xᵢ·yᵢ`, device-resident reduction |
| `nrm2` | `nrm2(gpu, &x) -> f32` | `√(Σ xᵢ²)` |

`scal`/`axpy` mutate in place (these ops are memory-bandwidth-bound;
avoiding a second buffer is the win). `dot`/`nrm2` multiply into a temp
field on the GPU and reduce there, so the vector never leaves the device.

The crate is a pure-Rust reference library (`quanta_blas::reference`, the
differential-test oracle) until you enable `gpu` + a backend:

```toml
quanta-blas = { version = "0.1", features = ["gpu-metal"] } # or gpu-vulkan
```

Coming next: Level-2 (`gemv`), Level-3 (`gemm`) with the cooperative-matrix
tensor-core paths, and the f16/bf16/i8 dtype matrix.

## Performance (honest framing)

quanta-blas v0.1 targets **~50% of vendor BLAS** on tier-1 datacentre GPUs
(H100, MI300X), **~80%** on tier-2 consumer / Apple-Silicon GPUs, and is
the **only** option where vendor BLAS doesn't exist (WebGPU, mobile).
Level-1 ops are bandwidth-bound, so the generic cross-backend kernel is
already near memory roofline. The GEMM tensor-core work is where the tuned
per-backend paths and the bigger competitive gap-closing happen. We never
hide where we lose — see `PERFORMANCE.md`.
