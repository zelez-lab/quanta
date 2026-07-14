# Quanta performance regression suite (step 069)

Benchmarks run on every PR via CI; regressions ≥25% block merge.

## Layout

```
bench/
  baselines/
    macos-aarch64.json    # Apple M1 Pro reference (M-series macOS, arm64)
    linux-x86_64.json     # (TODO: record on a Linux+Vulkan reference)
  README.md               # this file

crates/tools/quanta-bench/      # the harness binary
```

## Run locally

```sh
just bench           # run, print JSON to stdout
just bench-record    # run, overwrite committed baseline (do NOT commit unless intentional)
just bench-check     # run + compare against committed baseline; gate on ±5%
just bench-smoke     # run at smallest sizes (CI smoke check, no gate)
```

Direct invocation:

```sh
cargo run --release -p quanta-bench -- run --out /tmp/cur.json
cargo run --release -p quanta-bench -- compare \
    --baseline bench/baselines/macos-aarch64.json \
    --current /tmp/cur.json \
    --threshold 5
```

## Workloads

| Bench               | What                                                                  |
|---------------------|-----------------------------------------------------------------------|
| `heavy_compute`     | 1000 iterations of sin/cos/sqrt per element; 1k → 1M elements         |
| `add_one_dispatch`  | Dispatch overhead: 64× the same `data[i] += 1` over 1M elements       |
| `mandelbrot`        | 4K (3840×2160) Mandelbrot, up to 1000 iterations per pixel            |

## Threshold policy

- **Local:** ±5% by default. Tight enough to catch real regressions on a
  quiet workstation.
- **CI (macos-14 GitHub runner):** ±25%. Shared runners have neighbor noise;
  tightening below 25% produces flaky failures.
- **Improvements ≥threshold also fail.** Legitimate optimizations land with
  a baseline update in the same PR. This forces every speedup to be
  consciously committed, not silently masked by future regressions.

## Updating the baseline

Improvements ≥threshold fail by design — the same PR must update the
baseline:

```sh
just bench-record       # overwrite baseline JSON
git diff bench/baselines/macos-aarch64.json
git add bench/baselines/macos-aarch64.json
```

The PR description should explain *why* the change in numbers — which
optimization landed, which workload moved, and ideally a flame graph.

## Known limitations (future work)

- **No median-of-N.** Each `run` does one warmup + one measured run per
  workload. GPU dispatch jitter on shared CI runners produces ±10-20%
  noise; the 25% CI threshold absorbs this. A `--runs N --aggregate median`
  flag would give a tighter signal at the cost of longer CI time.
- **No CPU-execution smoke.** The CPU software backend has IR-coverage
  gaps for some kernels (e.g., `while` loops in mandelbrot trigger an
  unset-register error). Linux CI (no GPU) currently runs build-only;
  full smoke execution requires fixing the CPU executor first.
- **No Linux baseline.** `linux-x86_64.json` does not exist yet — needs a
  Vulkan-capable Linux runner (RPi 5 is one candidate, see step 053).
- **No public dashboard.** `perf.quanta.rs` / GitHub Pages with historical
  trend lines is described in the roadmap but not built yet.
