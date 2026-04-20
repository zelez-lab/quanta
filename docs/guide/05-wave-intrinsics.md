# Wave Intrinsics

A proton (warp/wave) is a group of quarks that execute in lockstep. On NVIDIA,
a proton is 32 quarks. On AMD, 64 quarks. On Apple, 32 quarks.

Wave intrinsics let quarks within a proton exchange data and evaluate conditions
collectively -- without shared memory or atomics. They compile to single hardware
instructions.

## wave_shuffle

Read a value from another quark in the same proton:

```rust
#[quanta::kernel]
fn shuffle_example(data: &mut [f32]) {
    let i = quark_id();
    let my_val = data[i];

    // Read the value from the quark `delta` positions away
    let neighbor = wave_shuffle(my_val, 1);  // get value from quark i+1

    data[i] = my_val + neighbor;
}
```

`wave_shuffle(value, delta)` returns the `value` held by the quark at relative
offset `delta` within the proton. The lane index wraps within the proton width.

No memory access occurs. Data moves through register lanes.

## wave_ballot

Ask "which quarks in my proton satisfy a condition?"

```rust
#[quanta::kernel]
fn ballot_example(data: &[f32], count: &mut [u32]) {
    let i = quark_id();
    let active = data[i] > 0.0;

    // Returns a bitmask: bit N is set if quark N has active == true
    let mask = wave_ballot(active);

    // Count bits to find how many quarks passed
    if local_id() == 0 {
        atomic_add(&mut count[0], popcount(mask));
    }
}
```

`wave_ballot(predicate)` returns a bitmask where bit N is set if quark N in the
proton has `predicate == true`.

## wave_any / wave_all

Collective boolean tests:

```rust
#[quanta::kernel]
fn early_exit(data: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let val = data[i];

    // True if ANY quark in the proton has val > threshold
    if wave_any(val > 100.0) {
        // At least one quark found a large value
        output[i] = val;
    }

    // True only if ALL quarks satisfy the condition
    if wave_all(val >= 0.0) {
        // Every quark in the proton has non-negative data
        output[i] = val;
    }
}
```

- `wave_any(pred)` -- returns `true` for **all** quarks if at least one quark
  has `pred == true`.
- `wave_all(pred)` -- returns `true` for **all** quarks only if every quark
  has `pred == true`.

These are uniform within a proton: every quark gets the same answer.

## Example: warp-level reduction

Sum all values in a proton without shared memory (assuming 32-quark proton):

```rust
#[quanta::kernel]
fn warp_reduce(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let lid = local_id();
    let mut val = input[i];

    // Tree reduction using shuffle
    val = val + wave_shuffle(val, 16);
    val = val + wave_shuffle(val, 8);
    val = val + wave_shuffle(val, 4);
    val = val + wave_shuffle(val, 2);
    val = val + wave_shuffle(val, 1);

    // Lane 0 has the sum of all 32 values
    if lid % 32 == 0 {
        output[i / 32] = val;
    }
}
```

This is faster than a shared memory reduction because:
- No memory writes/reads (registers only)
- No barriers needed
- Compiles to 5 hardware instructions

The downside: it only works within a single proton (32 or 64 quarks). For larger
reductions, combine with [shared memory](03-shared-memory.md):
1. Warp-reduce within each proton (wave intrinsics)
2. Write proton results to shared memory
3. Final reduction in shared memory (one proton's worth of data)

## When to use wave intrinsics

| Pattern             | Intrinsic     | Why                                    |
|---------------------|---------------|----------------------------------------|
| Small reduction     | `wave_shuffle`| Fastest possible sum/min/max for <=64 elements |
| Prefix sum (scan)   | `wave_shuffle`| Build inclusive scan with log2(N) shuffles |
| Early termination   | `wave_all`    | Skip work when all lanes are done      |
| Sparse processing   | `wave_ballot` | Count active lanes, compute offsets    |
| Broadcast           | `wave_shuffle`| Share one quark's value with all others |

## Hardware differences

| GPU       | Proton width | Implication                          |
|-----------|-------------|---------------------------------------|
| NVIDIA    | 32          | 5 shuffle steps for full reduction    |
| AMD       | 64          | 6 shuffle steps for full reduction    |
| Apple     | 32          | Same as NVIDIA                        |

Write kernels that work for both widths. Use `quark_per_proton()` or compile
two variants if you need maximum performance on both.

## Next

- [Textures](06-textures.md) -- image data and sampling
- [Rendering](07-rendering.md) -- the graphics pipeline
