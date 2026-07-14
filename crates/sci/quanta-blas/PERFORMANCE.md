# quanta-blas — performance

Honest, reproducible numbers. We never hide where we lose. Re-run with:

```sh
cargo bench -p quanta-blas --features gpu-metal --bench gemm    # GEMM
cargo bench -p quanta-blas --features gpu-metal --bench level1  # Level-1
```

(or `--features gpu-vulkan` for a Vulkan backend).

## GEMM (f32, square M=N=K) — naive vs tiled vs tensor-core

Apple **M1 Pro**, real Metal device (Metal Toolchain installed). GFLOP/s =
`2·M·N·K / time`. The `tc` column is the cooperative-matrix (`simdgroup_matrix`)
path; `tc/tiled` is its speedup over the tiled kernel.

| N | naive GFLOP/s | tiled GFLOP/s | tc GFLOP/s | tc/tiled |
|---|--------------:|--------------:|-----------:|---------:|
| 64  | 2.28  | 2.31   | 1.46   | 0.63× |
| 128 | 13.57 | 17.60  | 12.55  | 0.71× |
| 256 | 61.14 | 88.72  | 88.64  | 1.00× |
| 512 | 85.43 | 371.80 | 552.55 | 1.49× |

The tiled kernel (one 256-thread workgroup per 16×16 output tile, A/B blocks
staged in shared memory) pulls away from naive as N grows. The **tensor-core**
kernel then beats *tiled* at large N: each subgroup owns a 32×32 tile as a 4×4
grid of 8×8 `simdgroup_matrix` accumulators, so per K-step 8 global fragment
loads feed 16 MMAs (each fragment reused 4×) — the arithmetic intensity that
carries it to **~1.5× over tiled at N=512** (553 vs 372 GFLOP/s). At N≤256 the
32×32 tiles under-occupy the GPU (and launch overhead dominates the tiny sizes),
so `gemm()` only routes to the tensor-core path at **N≥512** (and only for
`C += A·B`, m/n multiples of 32, k a multiple of 8); smaller problems stay on
the tiled kernel. The bench cross-checks every kernel against the reference, so
a perf run doubles as a correctness check.

## Where we are vs the target

The strategy is ~80% of vendor BLAS on tier-2 (Apple-Silicon) GPUs.

- M1 Pro fp32 peak is roughly ~5 TFLOP/s; the tensor-core GEMM tops out near
  ~550 GFLOP/s at N=512 — order ~11% of peak (tiled was ~7.5%). The 4×4
  register-blocked kernel is the **Apple-optimal** path.
- **Shared-staging was tried and is slower on Apple Silicon** (a 64×64-tile,
  4-subgroup kernel that stages A/B through threadgroup memory measured
  ~432 GFLOP/s at N=512 vs the register kernel's ~557). On unified memory the
  bandwidth a staged tile would save is mostly already served by the system
  cache, so the two `barrier()`s per K-step the staging needs cost more than
  they save — the register kernel runs its subgroups barrier-free. The
  shared-source fragment load (`CooperativeMatrixLoad { from_shared }`) is kept
  and validated (`tc_shared_load_probe`) because shared-staging is the right
  strategy on **discrete** GPUs (NVIDIA-style, where global memory is far); it
  is reserved for the Vulkan path.
- Vulkan `VK_KHR_cooperative_matrix` is **not yet wired** — the Metal
  `simdgroup_matrix` path is the validated one; the SPIR-V emitter still falls
  back to scalar (so `supports_cooperative_matrix()` is Metal-only today).
- No vendor comparison (Apple Accelerate / cuBLAS) is wired in yet; the numbers
  above are quanta-internal only.

## Backend coverage

GEMV runs on the GEMM kernel (`gemm(m, 1, n, …)`), so it inherits GEMM's
backend coverage exactly — every row below applies to `gemv` too.

| Backend | GEMM status |
|---------|-------------|
| Software (CPU) | correct (differential tests; gemv 9/9; mixed bf16/f16/fp8 11/11; quant int8+int4 11/11). TC: `NotSupported` (CPU lane), gate-tested. |
| Metal (M1 Pro) | correct + benched on real hardware; gemv 9/9; mixed bf16/f16/fp8 11/11; quant int8+int4 11/11; **tensor-core 6/6** (incl. 128³). |
| Vulkan (lavapipe, RPi 5 Mesa LLVM 20) | **correct — 15/15 gemm tests pass** (all tiled + partial-tail cases). The tiled kernel's shared-memory + barriers lower to SPIR-V lavapipe accepts. (Note: lavapipe *does* reject subgroup reduce/scan ops — prims block kernels fail there with `VkResult -13` — but GEMM uses neither, only shared memory.) |
| WebGPU | not yet wired |

## Level-1 (f32)

Bandwidth-bound (1–2 flops/element), so the figure of merit is GB/s vs the
memory roofline, not GFLOP/s. Run `--bench level1` for current numbers; these
ops are already near roofline on the generic kernel (no per-backend tuning
needed).
