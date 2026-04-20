# Atomics

Atomic operations perform read-modify-write on a single memory location without
races. Multiple quarks can atomically update the same address concurrently.

## Available atomic operations

| Function                         | Operation                    | Supported types     |
|----------------------------------|------------------------------|---------------------|
| `atomic_add(&mut field[i], val)` | `field[i] += val`            | u32, i32, f32       |
| `atomic_sub(&mut field[i], val)` | `field[i] -= val`            | u32, i32            |
| `atomic_min(&mut field[i], val)` | `field[i] = min(field[i], val)` | u32, i32         |
| `atomic_max(&mut field[i], val)` | `field[i] = max(field[i], val)` | u32, i32         |
| `atomic_and(&mut field[i], val)` | `field[i] &= val`           | u32, i32            |
| `atomic_or(&mut field[i], val)`  | `field[i] |= val`           | u32, i32            |
| `atomic_xor(&mut field[i], val)` | `field[i] ^= val`           | u32, i32            |
| `atomic_exchange(&mut field[i], val)` | swap, return old       | u32, i32            |
| `atomic_cas(&mut field[i], expected, desired)` | CAS, return old | u32, i32        |

All atomics return the **previous** value at the memory location.

## Example: histogram

Count how many elements fall into each of 256 bins:

```rust
#[quanta::kernel]
fn histogram(data: &[u32], bins: &mut [u32]) {
    let i = quark_id();
    let value = data[i];
    let bin = value % 256;
    atomic_add(&mut bins[bin], 1u32);
}
```

Usage:

```rust
let data = gpu.compute_field::<u32>(1_000_000)?;
let bins = gpu.compute_field::<u32>(256)?;

// Zero the bins
gpu.write_field(&bins, &vec![0u32; 256])?;
gpu.write_field(&data, &input_data)?;

let mut wave = histogram(&gpu)?;
wave.bind(0, &data);
wave.bind(1, &bins);

let mut pulse = gpu.dispatch(&wave, 1_000_000)?;
gpu.wait(&mut pulse)?;

let counts = gpu.read_field::<u32>(&bins)?;
```

## Example: global counter

Assign unique IDs to elements that pass a filter:

```rust
#[quanta::kernel]
fn compact(input: &[f32], output: &mut [f32], count: &mut [u32], threshold: f32) {
    let i = quark_id();
    if input[i] > threshold {
        let slot = atomic_add(&mut count[0], 1u32);
        output[slot] = input[i];
    }
}
```

Each quark that passes the filter gets a unique slot via `atomic_add`. The
final value of `count[0]` is the total number of passing elements.

## Compare-and-swap (CAS)

`atomic_cas` is the most general atomic. It reads the current value, compares
it to `expected`, and writes `desired` only if they match:

```rust
#[quanta::kernel]
fn cas_increment(counter: &mut [u32]) {
    let mut old = counter[0];
    loop {
        let new_val = old + 1;
        let prev = atomic_cas(&mut counter[0], old, new_val);
        if prev == old {
            break;
        }
        old = prev;
    }
}
```

CAS enables lock-free algorithms: linked lists, queues, custom reduction
patterns. Use `atomic_add` when possible -- it is faster than a CAS loop.

## Performance considerations

- Atomics to the **same** address serialize. If all quarks atomically update
  `bins[0]`, they execute one at a time. Spread contention across addresses.

- Shared memory atomics (within a workgroup) are faster than global atomics.
  Use a two-phase approach: accumulate locally in shared memory, then atomically
  merge into global memory once per workgroup.

- On AMD, atomics operate at the wave (64-lane) level. On NVIDIA, at the warp
  (32-lane) level. Same-address atomics within a proton are coalesced into a
  single hardware operation on some GPUs.

## When to use atomics

| Pattern            | Why atomics                                    |
|--------------------|------------------------------------------------|
| Histogram          | Multiple quarks update the same bin            |
| Stream compaction  | Allocate output slots without knowing order    |
| Reduction (simple) | Single-pass sum when shared memory is overkill |
| Locks              | Last resort -- usually avoidable               |

Avoid atomics in hot inner loops. Prefer [shared memory](03-shared-memory.md)
reductions or [wave intrinsics](05-wave-intrinsics.md) when the communication
is within a workgroup or proton.

## Next

- [Wave intrinsics](05-wave-intrinsics.md) -- warp-level communication without memory
- [Textures](06-textures.md) -- image data on the GPU
