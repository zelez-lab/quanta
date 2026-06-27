# quanta-blas — performance

Honest, reproducible numbers. We never hide where we lose. Re-run with:

```sh
cargo bench -p quanta-blas --features gpu-metal --bench gemm    # GEMM
cargo bench -p quanta-blas --features gpu-metal --bench level1  # Level-1
```

(or `--features gpu-vulkan` for a Vulkan backend).

## GEMM (f32, square M=N=K) — naive vs tiled

Apple **M1 Pro** (Metal), 2026-06-26. GFLOP/s = `2·M·N·K / time`.

| N | naive ms | naive GFLOP/s | tiled ms | tiled GFLOP/s | speedup |
|---|---------:|--------------:|---------:|--------------:|--------:|
| 64  | 0.281 | 1.87  | 0.248 | 2.12   | 1.13× |
| 128 | 0.316 | 13.26 | 0.239 | 17.55  | 1.32× |
| 256 | 0.556 | 60.32 | 0.392 | 85.65  | 1.42× |
| 512 | 2.767 | 97.01 | 0.728 | 368.51 | 3.80× |

The tiled kernel (one 256-thread workgroup per 16×16 output tile, A/B blocks
staged in shared memory and reused `TILE` times) pulls away from the naive
kernel as N grows — the shared-memory reuse compounds. At N=512 it is **3.8×
faster** (369 vs 97 GFLOP/s). The bench cross-checks naive ≡ tiled on every
shape, so a perf run doubles as a correctness smoke test.

## Where we are vs the target

The strategy is ~80% of vendor BLAS on tier-2 (Apple-Silicon) GPUs. We are
**not there yet** with the generic tiled kernel:

- M1 Pro fp32 peak is roughly ~5 TFLOP/s; this tiled GEMM tops out near ~370
  GFLOP/s at N=512 — order ~7% of peak. The generic tiled kernel closes the
  *easy* gap over naive but not the vendor gap.
- The vendor gap is closed by the **cooperative-matrix / tensor-core path**
  (`simdgroup_matrix` on Metal, `VK_KHR_cooperative_matrix` on Vulkan), which
  is the next GEMM increment — not yet implemented.
- No vendor comparison (Apple Accelerate / cuBLAS) is wired in yet; that is a
  later increment. The numbers above are quanta-internal only.

## Backend coverage

GEMV runs on the GEMM kernel (`gemm(m, 1, n, …)`), so it inherits GEMM's
backend coverage exactly — every row below applies to `gemv` too.

| Backend | GEMM status |
|---------|-------------|
| Software (CPU) | correct (differential tests; gemv 9/9; mixed bf16/f16/fp8 11/11) |
| Metal (M1 Pro) | correct + benched (above); gemv 9/9; mixed bf16/f16/fp8 11/11 |
| Vulkan (lavapipe, RPi 5 Mesa LLVM 20) | **correct — 15/15 gemm tests pass** (all tiled + partial-tail cases). The tiled kernel's shared-memory + barriers lower to SPIR-V lavapipe accepts. (Note: lavapipe *does* reject subgroup reduce/scan ops — prims block kernels fail there with `VkResult -13` — but GEMM uses neither, only shared memory.) |
| WebGPU | not yet wired |

## Level-1 (f32)

Bandwidth-bound (1–2 flops/element), so the figure of merit is GB/s vs the
memory roofline, not GFLOP/s. Run `--bench level1` for current numbers; these
ops are already near roofline on the generic kernel (no per-backend tuning
needed).
