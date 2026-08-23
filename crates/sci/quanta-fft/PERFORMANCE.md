# quanta-fft — performance

Honest, reproducible numbers. We never hide where we lose. Re-run with:

```sh
cargo bench -p quanta-fft --features gpu-metal --bench fft    # 1-D: radix-2, Bluestein, rfft
cargo bench -p quanta-fft --features gpu-metal --bench fft2   # 2-D
```

(or `--features gpu-vulkan` for a Vulkan backend). Both benches cross-check
their output against the CPU baseline before timing it, so a perf run doubles
as a differential correctness check.

## The FLOP model

GFLOP/s below is `5·N·log2(N) / time`. Radix-2 Cooley-Tukey runs
`(N/2)·log2(N)` butterflies and a butterfly is one complex multiply-add — 10
flops — so `5·N·log2(N)` is the arithmetic the *algorithm* owes. It is a
model, not an instruction count; its value is that two implementations of the
same transform are directly comparable under it. Separability extends it to
2-D unchanged: `H` row transforms of length `W` plus `W` column transforms of
length `H` is `5·H·W·log2(H·W)`, i.e. the same formula with `N = H·W`.

**Bluestein does not fit the model.** Its cost is three length-`M` transforms
at `M = next_pow2(2N−1)` plus O(N) chirp work, so that table reports
milliseconds and spells out `M`.

Every time below includes the **host round trip**: `fft` and
`FftPlan::execute` take host slices and return host `Vec`s, so the upload and
the download are part of the measurement. That is the API's real cost, not a
harness artefact.

## Radix-2 (complex f32, 1-D)

Apple **M1 Pro**, real Metal device. Baseline is the CPU software device
(`quanta::init_cpu()`) in the same process. `fft` is the one-shot entry point
(plan built per call); `plan` is a reused `FftPlan`. `speedup` is `cpu / plan`.

| N | fft ms | GFLOP/s | plan ms | GFLOP/s | cpu ms | GFLOP/s | speedup |
|---|-------:|--------:|--------:|--------:|-------:|--------:|--------:|
| 1024      |  57.949 | 0.001 |  38.093 | 0.001 |     10.563 | 0.005 |   0.3× |
| 2048      |  72.006 | 0.002 |  52.991 | 0.002 |     21.340 | 0.005 |   0.4× |
| 4096      |  71.990 | 0.003 |  62.960 | 0.004 |     47.270 | 0.005 |   0.8× |
| 8192      |  64.014 | 0.008 |  87.977 | 0.006 |    101.079 | 0.005 |   1.1× |
| 16384     |  89.933 | 0.013 |  85.926 | 0.013 |    219.879 | 0.005 |   2.6× |
| 32768     | 103.972 | 0.024 | 111.048 | 0.022 |    473.848 | 0.005 |   4.3× |
| 65536     | 119.971 | 0.044 | 119.574 | 0.044 |   1023.067 | 0.005 |   8.6× |
| 131072    | 135.971 | 0.082 | 137.319 | 0.081 |   2202.458 | 0.005 |  16.0× |
| 262144    | 143.984 | 0.164 | 142.105 | 0.166 |   4647.083 | 0.005 |  32.7× |
| 524288    | 168.089 | 0.296 | 159.921 | 0.311 |   9500.323 | 0.005 |  59.4× |
| 1048576   | 172.080 | 0.609 | 178.335 | 0.588 |  20112.544 | 0.005 | 112.8× |

Read the wall-clock column, not the GFLOP/s column, and the shape of this
library falls out: **from 2^10 to 2^20 the work grows 2048× and the time grows
2.97×.** The transform is not compute-bound on this device — it is bound by
what happens around the arithmetic. Concretely, one `execute` is

- one dispatch for the bit-reversal permutation and one per butterfly stage,
  so `log2(N) + 1` dispatches in all — 11 at N=1024, 21 at N=2^20;
- four fresh device fields allocated per call (input and work buffers, re and
  im); and
- a full host round trip: the input is written up and the spectrum is read
  back on every call.

That is why the small-N rows are nearly flat: at N=1024 the arithmetic is
51 kflop and the 58 ms is essentially 11 dispatches plus two transfers. It is
also why **the GPU loses to the CPU below N≈8192** — 0.3× at 1024 — and only
crosses over once there is enough data per dispatch to pay for the dispatch.
At the top of the sweep the picture inverts: 112.8× the CPU at N=2^20.

The absolute figure is the part we are not going to dress up. 0.6 GFLOP/s
against an M1 Pro's ~5 TFLOP/s fp32 peak is order **0.01% of peak**. Nothing
here is a claim about how fast an FFT can be on this hardware; it is a
measurement of the current implementation, whose per-call structure (host
round trip, one dispatch per stage, four allocations) has never been tuned.

**Plan reuse currently buys nothing measurable.** `FftPlan` exists to skip the
per-call kernel JIT and twiddle upload, and it does — but those are not what
the call costs. The `fft` and `plan` columns agree to within run-to-run noise
at every size. Holding a plan is still the right shape for repeated
same-size work, and it will start paying the moment the per-call overheads
come down; today it is not where the time goes.

## Bluestein chirp-z (non-power-of-2 N)

| N | M | device ms | cpu ms | speedup |
|---|---|----------:|-------:|--------:|
| 1000   |   2048 | 303.753 |    68.335 |  0.2× |
| 3000   |   8192 | 335.953 |   318.633 |  0.9× |
| 10000  |  32768 | 396.750 |  1445.386 |  3.6× |
| 30000  |  65536 | 430.686 |  3055.007 |  7.1× |
| 100000 | 262144 | 495.927 | 13704.326 | 27.6× |

The cost model checks out against the radix-2 table above: a Bluestein
transform at `M` costs **3.4× to 5.2× a plain power-of-2 transform of length
`M`** (495.927 vs 143.984 at M=262144 is 3.4×; 303.753 vs 72.006 at M=2048 is
4.2×), converging on the predicted three-transforms-plus-chirp as N grows and
the fixed per-call costs are amortised. The practical advice the crate docs
give — pick a power-of-2 length when you can choose — is confirmed: it is
worth a factor of three to five, on top of `M` being up to 2× larger than `N`.

The crossover against the CPU sits near N=3000, later than the radix-2 path's
because Bluestein pays three transforms for one.

## Real input: `rfft`

`rfft` transforms a length-N real signal through one half-size complex plan
plus an O(N) split pass. Compared against transforming the same signal as
complex-with-zero-imaginary:

| N | rfft ms | complex ms | speedup |
|---|--------:|-----------:|--------:|
| 4096    |  87.943 |  95.991 | 1.09× |
| 16384   | 110.757 | 118.222 | 1.07× |
| 65536   | 108.936 | 127.561 | 1.17× |
| 262144  | 135.885 | 152.024 | 1.12× |
| 1048576 | 175.952 | 183.934 | 1.05× |

**The crate docs claim "~2× the throughput"; measured, it is 1.05–1.17×.** The
claim describes the arithmetic honestly — half the transform length really is
about half the work — but the arithmetic is not what a call costs here, so
halving it buys ~10%. The *memory* half of the claim does hold and is
unaffected by any of this: `rfft` allocates and moves half the device data,
and never materialises the zero imaginary array. When the per-call overheads
come down, this row should move toward the 2× the algorithm promises; until
then the docs overstate the throughput win and this table is the number to
trust.

## 2-D: `fft2`

This is the one where we lose outright, at every size measured.

| H×W | device ms | GFLOP/s | cpu ms | GFLOP/s | speedup |
|-----|----------:|--------:|-------:|--------:|--------:|
| 64×64   |  7631.001 | 0.000 |  146.993 | 0.002 | 0.02× |
| 128×128 | 17399.178 | 0.000 |  446.698 | 0.003 | 0.03× |
| 256×256 | 38673.219 | 0.000 | 1534.533 | 0.003 | 0.04× |
| 512×512 | 84708.976 | 0.000 | 5989.207 | 0.004 | 0.07× |

**A 512×512 2-D transform takes 85 seconds on the GPU and 6 on the CPU** — the
device is 14× slower there, and 52× slower at 64×64. This is not a subtle
effect and it has a single cause: `fft2` is the separable row-column method
built on the 1-D engine, so it issues `H` row `execute`s and then `W` column
`execute`s — **`H + W` separate 1-D calls, each paying the full per-call cost
from the table above**. Divide it out and the arithmetic is exactly that:
84709 ms / 1024 calls = 82.7 ms per pass at 512×512, 7631 / 128 = 59.6 ms at
64×64 — the same 40–80 ms per-call floor the 1-D sweep measures, multiplied by
`H + W`. The CPU baseline runs the identical decomposition; its per-call floor
is simply microseconds.

The fix is structural, not a tuning knob: the row pass is `H` independent
length-`W` transforms and belongs in **one** batched dispatch, not `H` of
them, with the transpose staying on the device between passes. Until that
lands, `fft2` is correct (it is differential-tested against the direct 2-D DFT
on every backend) and it is the wrong tool for anything time-sensitive — run
2-D work on the CPU reference, or decompose it yourself and batch the rows.

## Backend coverage

| Backend | FFT status |
|---------|------------|
| Software (CPU) | correct — the CI companion lane runs the full suite (`cargo test -p quanta-fft --features gpu` under `QUANTA_BACKEND=cpu`): radix-2, Bluestein, `rfft`, `fft2`, and the inverse round trips, all against the direct-DFT oracle. |
| Metal (M1 Pro) | correct + benched here; differential-tested against the direct DFT on real hardware. Every number on this page is this device. |
| Vulkan | builds and type-checks under `gpu-vulkan`; **no device run recorded**, so there are no Vulkan numbers to quote and none are invented here. |
| WebGPU | not wired — the crate has no `webgpu` feature. |

## Where we are vs the target

- **No vendor comparison is wired in.** There is no VkFFT / cuFFT / Accelerate
  `vDSP` link in this repo, so every number above is quanta-internal. A vendor
  FFT on this part would be orders of magnitude faster than 0.6 GFLOP/s; the
  gap is not a mystery to be measured, it is the per-call structure described
  above, and closing it is implementation work that has not been done.
- **The three things standing between here and a competitive FFT**, in the
  order they matter: keep the data on the device across a transform (a
  `Field`-in / `Field`-out surface next to the host-slice one, so a pipeline
  of transforms does not round-trip per stage); fuse the small stages so the
  dispatch count stops tracking `log2(N)`; and batch `fft2`'s row and column
  passes into one dispatch each instead of one per row.
- **Correctness is not what is being traded away here.** Radix-2 is
  *mechanically proven equal to the direct DFT* in Lean (0 sorry — see
  [`docs/verification/index.md`](../../../docs/verification/index.md)); the
  Bluestein path is differential-tested against the same oracle. The
  performance story on this page is about the runtime around the kernels, not
  about the transform.

## Measurement conditions

Every figure is the **fastest of several repeats**, not a mean: the host is a
shared developer workstation, and a mean measures whatever else was running.
Timings were taken with other work live on the machine, which matters
asymmetrically — the small-N rows are host-overhead-bound and therefore the
load-sensitive ones (N=1024 has been observed between 19 ms and 58 ms across
runs on this Mac), while the large-N rows are stable to a few percent. Treat
the top of the sweep as solid and the bottom as an upper bound.
