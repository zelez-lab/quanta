# Parallel Reduce

Sum an array using shared memory and tree reduction.

## Kernel

```rust
const BLOCK_SIZE: u32 = 256;

#[quanta::kernel]
fn reduce_sum(input: &[f32], output: &mut [f32], count: u32) {
    #[quanta::shared]
    let local: [f32; 256];

    let lid = local_id();
    let gid = quark_id();

    // Load from global memory into shared memory
    if gid < count {
        local[lid] = input[gid];
    } else {
        local[lid] = 0.0f32;
    }
    barrier();

    // Tree reduction in shared memory
    let mut stride = 128u32;
    while stride > 0u32 {
        if lid < stride {
            local[lid] = local[lid] + local[lid + stride];
        }
        barrier();
        stride = stride / 2u32;
    }

    // First thread writes the block result
    if lid == 0u32 {
        let block_id = group_id();
        output[block_id] = local[0u32];
    }
}
```

## Host code

```rust
fn main() {
    let gpu = quanta::init().unwrap();

    let count: usize = 1_048_576;
    let block_size: u32 = 256;
    let num_blocks = ((count as u32) + block_size - 1) / block_size;

    let data: Vec<f32> = (0..count).map(|i| (i % 7) as f32).collect();

    let input = gpu.compute_field::<f32>(count).unwrap();
    let partial = gpu.compute_field::<f32>(num_blocks as usize).unwrap();

    gpu.write_field(&input, &data).unwrap();

    let mut wave = reduce_sum(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &partial);
    wave.set_value(2, count as u32);

    // Dispatch with group size = BLOCK_SIZE
    let mut pulse = gpu.wave_dispatch(&wave, [num_blocks, 1, 1]).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // Read partial sums and finish on CPU
    let partials = gpu.read_field(&partial).unwrap();
    let total: f32 = partials.iter().sum();

    let expected: f32 = data.iter().sum();
    assert!((total - expected).abs() < 1.0);
    println!("Sum = {total}");
}
```

## Reduction pattern

```
Step 0:  [a0 a1 a2 a3 a4 a5 a6 a7]  (stride = 4)
Step 1:  [a0+a4  a1+a5  a2+a6  a3+a7  _ _ _ _]  (stride = 2)
Step 2:  [s0+s2  s1+s3  _ _ _ _ _ _]  (stride = 1)
Step 3:  [final  _ _ _ _ _ _ _]
```

Each `barrier()` ensures all quarks in the group have finished writing before
the next round reads. Without barriers, quarks would read stale shared memory.

## Key concepts

- `#[quanta::shared]` declares workgroup-local memory (fast, ~100x faster than global)
- `local_id()` is the thread index within the workgroup (0..255)
- `group_id()` is the workgroup index
- `barrier()` synchronizes all quarks in the workgroup
- For large arrays, dispatch multiple groups and reduce partial sums on CPU
