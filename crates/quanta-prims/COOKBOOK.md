# quanta-prims cookbook

Recipe catalogue: patterns that come up over and over in
downstream GPU kernels. Each recipe states the workload, the
primitive it uses, and a minimal kernel that solves it.

> Every snippet uses the GPU primitives from `quanta_prims` and
> the `#[quanta::kernel]` machinery from `quanta`. All snippets
> assume the `gpu` feature is enabled on `quanta-prims`.

---

## Sum a buffer (block-local)

**Use case:** compute the per-block sum of a u32 buffer.
Foundational for any cooperative reduction.

**Pattern:**

```rust,ignore
use quanta_prims::block_reduce_add_u32_buffer;

let mut wave = block_reduce_add_u32_buffer(&gpu)?;
wave.bind(0, &input);   // [u32; 256 * num_blocks]
wave.bind(1, &output);  // [u32; num_blocks]
gpu.dispatch(&wave, (256 * num_blocks) as u32)?.wait()?;
```

`output[b]` holds the sum of `input[b*256..(b+1)*256]`. Use the
identical shape for `min`, `max`, `i32`, `f32`.

---

## Compute a dot product (inside your own kernel)

**Use case:** dot product `x · y = Σ x[i] * y[i]` cooperatively
within a block. The classic case for "primitive inside a
kernel."

**Pattern:**

```rust,ignore
use quanta::*;
use quanta_prims::block_reduce_add_u32_kernel;

#[quanta::kernel(workgroup_size = [256, 1, 1])]
fn dot_block(x: &[u32], y: &[u32], partials: &mut [u32]) {
    #[quanta::shared] let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 { scratch[lane] = 0u32; }
    barrier();

    let product = x[i as usize] * y[i as usize];
    let block_dot = block_reduce_add_u32_kernel(product);

    if lane == 0u32 {
        partials[block as usize] = block_dot;
    }
}
```

To get the global dot product, dispatch this kernel then sum
the `partials` buffer with a second pass (or a CPU reduce for
small enough output).

---

## Find the max-magnitude element

**Use case:** numerical stability — find `max(|x[i]|)` so the
caller can re-scale `x` before a follow-up op.

**Pattern:** combine `block_reduce_max_f32` with an `abs`
preprocessing step:

```rust,ignore
use quanta::*;
use quanta_prims::block_reduce_max_f32_kernel;

#[quanta::kernel(workgroup_size = [256, 1, 1])]
fn block_max_abs(data: &[f32], out: &mut [f32]) {
    #[quanta::shared] let scratch: [f32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 { scratch[lane] = -1.0e38f32; }
    barrier();

    let v = data[i as usize];
    let mag = abs(v);
    let block_max = block_reduce_max_f32_kernel(mag);

    if lane == 0u32 {
        out[block as usize] = block_max;
    }
}
```

The `abs` intrinsic is part of Quanta's f32 math; it doesn't
involve `quanta-prims`.

---

## Per-block histogram (small bin count)

**Use case:** count occurrences of each value in a small range.
Foundation for radix-sort variants and image processing.

**Pattern:** one reduce per bin. Works well for ≤ 16 bins.

```rust,ignore
use quanta::*;
use quanta_prims::block_reduce_add_u32_kernel;

#[quanta::kernel(workgroup_size = [256, 1, 1])]
fn histogram_4_bins(data: &[u32], counts: &mut [u32]) {
    #[quanta::shared] let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    let v = data[i as usize];

    // For each bin, every lane contributes 1 if its value
    // falls in the bin, else 0. The block reduce sums those
    // indicators -> the bin count.
    let mut bin: u32 = 0u32;
    while bin < 4u32 {
        if lane < 32u32 { scratch[lane] = 0u32; }
        barrier();

        let in_bin = if v == bin { 1u32 } else { 0u32 };
        let bin_count = block_reduce_add_u32_kernel(in_bin);

        if lane == 0u32 {
            counts[(block * 4u32 + bin) as usize] = bin_count;
        }
        barrier();

        bin = bin + 1u32;
    }
}
```

Output has shape `[num_blocks * 4]` — block-major, bin-minor.
For larger bin counts use a single shared `[u32; BINS]` array
and atomic adds instead.

---

## Compute output positions (compaction / filter)

**Use case:** keep only elements satisfying a predicate, packed
into a dense output buffer. Foundation for stream compaction.

**Pattern:** indicator (1 if keep, 0 if drop) → inclusive scan
→ output position is `(scan[i] - 1)`. Lane only writes if its
indicator is 1.

```rust,ignore
use quanta::*;
use quanta_prims::block_scan_add_u32_kernel;

#[quanta::kernel(workgroup_size = [256, 1, 1])]
fn compact_even(data: &[u32], out: &mut [u32], counts: &mut [u32]) {
    #[quanta::shared] let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 { scratch[lane] = 0u32; }
    barrier();

    let v = data[i as usize];
    let keep = if v % 2u32 == 0u32 { 1u32 } else { 0u32 };
    let inc_rank = block_scan_add_u32_kernel(keep);

    // If we keep this element, write it at position
    // (block_offset + inc_rank - 1) of the output.
    if keep == 1u32 {
        let pos = block * 256u32 + (inc_rank - 1u32);
        out[pos as usize] = v;
    }

    // Lane 255 writes the block's kept-count (needed to know
    // the valid range of `out` for this block).
    if lane == 255u32 {
        counts[block as usize] = inc_rank;
    }
}
```

The output for block `b` is valid from `b*256` through
`b*256 + counts[b] - 1`.

---

## Sort the block-local working set

**Use case:** every workgroup processes a 256-element tile that
needs to be sorted before further work (e.g. inside a tiled
sort or a histogram bucketing pass).

**Pattern:** call `block_radix_sort_u32_buffer` as a stand-alone
dispatch:

```rust,ignore
use quanta_prims::block_radix_sort_u32_buffer;

let mut wave = block_radix_sort_u32_buffer(&gpu)?;
wave.bind(0, &input);   // [u32; 256 * num_blocks]
wave.bind(1, &output);  // same shape
gpu.dispatch(&wave, (256 * num_blocks) as u32)?.wait()?;
// Each 256-element block is now sorted ascending.
```

Each block sorts independently. For a globally-sorted output,
use the Tier-3 `device_sort_u32` wrapper below.

---

## Tier-2 convenience kernels

The "compute output positions" and "per-block histogram"
recipes above show the patterns from scratch — useful when you
want to fuse the primitive into a larger kernel. If you just
need the standalone operation on a buffer, three Tier-2
convenience kernels do the work directly.

### `block_compact_u32_buffer`

```rust,ignore
use quanta_prims::block_compact_u32_buffer;

let mut wave = block_compact_u32_buffer(&gpu)?;
wave.bind(0, &predicates); // [u32; n], non-zero = keep
wave.bind(1, &data);       // [u32; n]
wave.bind(2, &out);        // [u32; n]
wave.bind(3, &counts);     // [u32; n / 256], per-block kept count
gpu.dispatch(&wave, n as u32)?.wait()?;
```

Kept entries from block `b` end up at `out[b*256..b*256+counts[b]]`.

### `block_histogram_u32_buffer`

```rust,ignore
use quanta_prims::block_histogram_u32_buffer;

// Caller pre-computes bucket indices (values in 0..256).
let mut wave = block_histogram_u32_buffer(&gpu)?;
wave.bind(0, &buckets);  // [u32; n], each value in 0..256
wave.bind(1, &counts);   // [u32; n], block-major: counts[b*256 + bucket]
gpu.dispatch(&wave, n as u32)?.wait()?;
```

Shared-memory atomics today emit only on Metal; other backends
return `NotSupported`. The kernel hard-codes 256 buckets per
block (= workgroup size).

### `block_top_k_u32_buffer`

```rust,ignore
use quanta_prims::block_top_k_u32_buffer;

let k: u32 = 16;
let mut wave = block_top_k_u32_buffer(&gpu)?;
wave.bind(0, &data);        // [u32; n]
wave.bind(1, &top_k_out);   // [u32; n / 256 * k]
wave.set_value(2, k);
gpu.dispatch(&wave, n as u32)?.wait()?;
```

`top_k_out[b * k + i]` is the i-th-largest value of block `b`,
with i=0 the maximum. `k <= 256`.

---

## Tier-3 device-wide wrappers

The block kernels above leave you with *per-block* results. When
you want the whole-buffer answer and your data starts on the
host, the Tier-3 wrappers handle upload, padding, multi-pass
orchestration, and readback in one call:

```rust,ignore
use quanta_prims::{device_reduce_add_u32, device_sort_u32};

// Any length ≥ 1; multi-pass block reduce under the hood.
let total: u32 = device_reduce_add_u32(&gpu, &data)?;

// Any length; device-wide bitonic network under the hood.
let sorted: Vec<u32> = device_sort_u32(&gpu, &data)?;
```

The full reduce family is `device_reduce_{add,min,max}_{u32,i32,f32}`.
These are convenience demos: each call round-trips host ↔ GPU,
so inside a larger pipeline prefer the `block_*` kernels and
keep intermediates resident.

---

## Get a CPU oracle for any primitive

**Use case:** write a test that compares GPU output against a
known-correct CPU result.

**Pattern:** every device fn has a sibling in
`quanta_prims::reference`:

```rust
use quanta_prims::reference;

let xs = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
assert_eq!(reference::reduce_add_u32(&xs), 31);
assert_eq!(reference::scan_add_u32(&xs), vec![3, 4, 8, 9, 14, 23, 25, 31]);
assert_eq!(reference::reduce_min_u32(&xs), 1);
assert_eq!(reference::reduce_max_u32(&xs), 9);
assert_eq!(reference::radix_sort_u32(&xs), vec![1, 1, 2, 3, 4, 5, 6, 9]);
```

The reference module is feature-flag-free; it's available
whether or not the `gpu` feature is on. Use it in
`#[cfg(test)]` modules in downstream crates as the oracle for
your differential tests.

---

## Where to go next

- **[README.md](README.md)** — full crate overview + status.
- **[GETTING_STARTED.md](GETTING_STARTED.md)** — first
  walkthrough.
- **`tests/block_reduce_family.rs`** — every reduce variant
  exercised end-to-end. Good template for writing your own
  differential tests.
- **`tests/block_scan.rs`** — scan correctness, including the
  ramp / uniform / first-output / last-output checks.
- **`tests/block_sort.rs`** — sort across pseudo-random,
  ties, extreme values, multi-block independence.
- **`tests/block_compact.rs`**, **`tests/block_histogram.rs`**,
  **`tests/block_top_k.rs`** — Tier-2 convenience-kernel
  differential tests.
