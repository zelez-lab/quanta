# Histogram

Count occurrences of values using atomic operations.

## Kernel

```rust
const NUM_BINS: u32 = 256;

#[quanta::kernel]
fn histogram(input: &[u32], bins: &mut [u32], count: u32) {
    let i = quark_id();
    if i < count {
        let value = input[i];
        let bin = value % NUM_BINS;
        atomic_add(&mut bins[bin], 1u32);
    }
}
```

## Host code

```rust
fn main() {
    let gpu = quanta::init().unwrap();

    let count: usize = 4_000_000;
    let num_bins: usize = 256;

    // Generate random-ish input data
    let data: Vec<u32> = (0..count).map(|i| ((i * 7 + 13) % 1000) as u32).collect();

    let input = gpu.field::<u32>(count).unwrap();
    let bins = gpu.field::<u32>(num_bins).unwrap();

    input.write(&data).unwrap();
    // Zero-initialize bins
    bins.write(&vec![0u32; num_bins]).unwrap();

    let mut wave = histogram(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &bins);
    wave.set_value(2, count as u32);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    let result = bins.read().unwrap();
    let total: u32 = result.iter().sum();
    assert_eq!(total, count as u32);

    println!("Bin 0: {} occurrences", result[0]);
    println!("Bin 1: {} occurrences", result[1]);
    println!("Total: {total}");
}
```

## How atomics work

`atomic_add(&mut bins[bin], 1u32)` translates to a hardware atomic increment.
When multiple quarks hit the same bin simultaneously, the hardware serializes
their updates — no data is lost.

Available atomic operations in Quanta kernels:

| Function | Effect |
|----------|--------|
| `atomic_add(dst, val)` | `*dst += val`, returns old value |
| `atomic_sub(dst, val)` | `*dst -= val`, returns old value |
| `atomic_min(dst, val)` | `*dst = min(*dst, val)`, returns old value |
| `atomic_max(dst, val)` | `*dst = max(*dst, val)`, returns old value |
| `atomic_and(dst, val)` | `*dst &= val`, returns old value |
| `atomic_or(dst, val)` | `*dst |= val`, returns old value |
| `atomic_xor(dst, val)` | `*dst ^= val`, returns old value |
| `atomic_exchange(dst, val)` | `*dst = val`, returns old value |
| `atomic_compare_exchange(dst, expected, desired)` | CAS |

## Performance notes

- Atomic contention reduces throughput. If many quarks write the same bin,
  they serialize.
- For high-contention histograms, use shared-memory privatization: each workgroup
  builds a local histogram in shared memory, then atomically merges to global.
