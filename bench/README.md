# Quanta performance regression suite (step 069)

Benchmarks run in CI on the `run-perf` PR label or a manual dispatch (the
Linux lane is dispatch-only); a ≥25% move in either direction fails the job.

## Layout

```
bench/
  baselines/
    macos-aarch64.json    # Apple M1 Pro reference (M-series macOS, arm64)
    linux-x86_64.json     # lavapipe (llvmpipe) reference, recorded on the ubuntu CI runner
    windows-x86_64.json   # Intel Iris Xe reference (self-hosted Windows rig)
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
- **CI (shared GitHub runners):** ±25%, on the macos-14 lane and on
  `perf-regression-linux`'s lavapipe alike. Shared runners have neighbor
  noise; tightening below 25% produces flaky failures.
- **Windows (Iris Xe rig):** no CI job yet — `just bench-check` on the rig
  picks `windows-x86_64.json` up by host OS.
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
  unset-register error), so no lane runs the suite on the `software`
  backend. The Linux numbers come from lavapipe — a CPU *Vulkan*
  implementation, not Quanta's own CPU executor: the `perf-regression-linux`
  job (`workflow_dispatch` only) runs the smoke and the full release suite
  on the ubuntu runner's lavapipe and compares against
  `linux-x86_64.json` at a 25% threshold. Smoke on the `software` backend
  still requires fixing the CPU executor first.
- **No public dashboard.** `perf.quanta.rs` / GitHub Pages with historical
  trend lines is described in the roadmap but not built yet.
