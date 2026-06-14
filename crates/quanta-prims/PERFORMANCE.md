# quanta-prims performance

Honest perf numbers as of v0.1.0-alpha.2. Numbers are from a
single device — your mileage will vary.

## What's measured

Wall-clock time of the **top-level convenience kernels** end to
end: write inputs to GPU, dispatch, wait, read outputs. This is
the path a user dispatching one primitive at a time will see.
For kernels chained inside a larger pipeline that keeps data
resident on the GPU, the practical speedup over CPU is higher
than reported here — the memcpy contribution drops.

The benches live in `examples/bench_throughput.rs` (sweep over
N) and `examples/bench_vs_cpu.rs` (head-to-head). Reproduce
with:

```sh
cargo run -p quanta-prims --features gpu-metal --release \
    --example bench_throughput
```

## Apple M1 Pro

### Throughput by N (median over 10 iter, 2 warmup) — Tier 1

| N       | reduce  | scan    | sort    |
| ------- | ------- | ------- | ------- |
| 256     | ~8.0 ms | ~8.0 ms | ~8.0 ms |
| 2 048   | ~8.0 ms | ~8.0 ms | ~8.0 ms |
| 16 384  | ~8.0 ms | ~8.0 ms | ~8.0 ms |
| 65 536  | ~8.0 ms | ~8.0 ms | ~8.0 ms |
| 262 144 | ~8.0 ms | ~8.0 ms | ~8.0 ms |

### Throughput by N — Tier 2

| N       | compact | histogram | top-k (k=16) |
| ------- | ------- | --------- | ------------ |
| 256     | ~8.0 ms | ~8.0 ms   | ~8.0 ms      |
| 2 048   | ~8.0 ms | ~8.0 ms   | ~8.0 ms      |
| 16 384  | ~8.0 ms | ~8.0 ms   | ~8.0 ms      |
| 65 536  | ~8.0 ms | ~8.0 ms   | ~8.0 ms      |
| 262 144 | ~8.0 ms | ~8.0 ms   | ~8.0 ms      |

The ~8 ms wall is **fixed dispatch + sync overhead on M1 Pro**,
not the primitive's compute time. Until that overhead drops
below the per-element work, throughput is bandwidth-bound by
the host↔device round trip, not by the kernel.

At N = 262 144 the effective throughput is ~33 M elements/sec
across all primitives — Tier 1 and Tier 2 alike. The kernels
themselves take microseconds; the wall is the dispatch + wait
+ readback path.

Tier 2 caveats:

- **compact** uploads a per-element predicate buffer once per
  call; build + upload is included in the wall time.
- **histogram** likewise uploads a per-element bucket index
  buffer.
- **top-k** runs an inlined bitonic sort plus a conditional
  write — it does NOT share `radix_sort`'s LSD body, so the two
  lines can drift apart as the algorithms evolve.

### GPU vs CPU head-to-head at N = 16 384

| Primitive       | GPU (ms) | CPU 1-thr (ms) | N-core est (ms) |
| --------------- | -------- | -------------- | --------------- |
| reduce_add_u32  | 1.4      | 0.001          | 0.0001          |
| scan_add_u32    | 1.4      | 0.006          | 0.0006          |
| radix_sort_u32  | 2.0      | 0.137          | 0.014           |

(Numbers re-measured after the LSD-radix body swap; the sort
line is the 16-pass radix, not the original bitonic.)

At 16 384 elements the single-thread CPU reference beats the
GPU by 100×+ on reduce/scan and ~14× on sort, because the
fixed dispatch overhead dominates. The CPU compiler turns the
reference impls into vectorised inner loops that absolutely
crush the GPU on small N.

## What this means for users

- **GPU primitives in this crate are valuable when data is
  already on the GPU** and the alternative would be a host
  round-trip to compute the same thing. That's the typical
  context for cooperative primitives inside larger kernels.
- **Don't reach for these to accelerate a stand-alone reduce
  / scan / sort on small data.** The dispatch overhead is real
  and is a property of the platform, not this crate. For
  CPU-resident data, `iter::sum`, `slice::sort_unstable`, and
  hand-rolled prefix-sum loops will outperform.
- **The bench harness is shipped so users can verify the
  numbers on their own hardware.** Different GPUs (NVIDIA
  H100, AMD MI300X, integrated mobile) have very different
  dispatch profiles; on hardware with low launch overhead the
  crossover N is much smaller.

## Optimisation pipeline

For v0.1 we ship correctness across Tier 1 + Tier 2; perf
optimisation is queued:

- **Reduce dispatch overhead via wave reuse.** Quanta's Wave
  object already supports repeated dispatch without rebuild;
  the bench harness above redundantly creates / drops the Wave
  inside the timed loop, simulating worst-case user code.
- **Multi-block tile size.** Currently each `*_buffer` kernel
  uses workgroup_size = 256, one workgroup per 256-element
  block. Tuning to 512 or 1024 on Apple Silicon (which has
  bigger SIMD groups) may help.
- **Cooperative-matrix paths** (only for reduce). Apple
  simdgroup matrix instructions could implement the warp-level
  reduction in one op instead of N shuffles. Out of v0.1 scope.

None of these are needed for **correctness**; they're perf
follow-ups. The current primitives are the right thing to
build downstream crates against — when downstream perf demands
go up, the kernels swap underneath without API change.
