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
| 64  | 1.94  | 2.18   | 1.80   | 0.83× |
| 128 | 14.09 | 15.94  | 14.20  | 0.89× |
| 256 | 57.11 | 84.87  | 98.66  | 1.16× |
| 512 | 89.82 | 375.84 | 498.28 | 1.33× |

The tiled kernel (one 256-thread workgroup per 16×16 output tile, A/B blocks
staged in shared memory) pulls away from naive as N grows. The **tensor-core**
kernel then beats *tiled* at N≥256: each subgroup owns a 16×16 tile as a 2×2
grid of 8×8 `simdgroup_matrix` accumulators, so every loaded fragment feeds two
MMAs (2× arithmetic intensity), and the MMA units carry it to **1.33× over
tiled at N=512** (498 vs 376 GFLOP/s). At N≤128 launch overhead dominates and
tiled wins, so `gemm()` only routes to the tensor-core path at N≥256 (and only
for `C += A·B`, m/n multiples of 16, k a multiple of 8). The bench cross-checks
every kernel against the reference, so a perf run doubles as a correctness
check.

## Where we are vs the target

The strategy is ~80% of vendor BLAS on tier-2 (Apple-Silicon) GPUs.

- M1 Pro fp32 peak is roughly ~5 TFLOP/s; the tensor-core GEMM tops out near
  ~500 GFLOP/s at N=512 — order ~10% of peak (tiled was ~7.5%). The 2×2
  register-blocked kernel is the first real step past the SIMT tiled path;
  more fragment reuse (4×4 blocking + threadgroup-shared staging of the A/B
  tiles) is the next lever toward the vendor gap.
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
