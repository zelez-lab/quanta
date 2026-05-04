# Matrix Multiply

Tiled matrix multiply using shared memory for coalesced access.

## Data layout

```rust
#[derive(quanta::Fields)]
struct MatmulData {
    a: Vec<f32>,       // M x K input matrix
    b: Vec<f32>,       // K x N input matrix
    c: Vec<f32>,       // M x N output matrix
    m: u32,            // push constant: rows of A / rows of C
    n: u32,            // push constant: cols of B / cols of C
    k: u32,            // push constant: cols of A / rows of B
}
```

Three `Vec<f32>` fields map to GPU storage buffer slots 0, 1, 2. Three `u32`
scalars become push constants at slots 3, 4, 5.

## Kernel

```rust
const TILE: u32 = 16;

#[quanta::kernel]
fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    #[quanta::shared]
    let tile_a: [f32; 256]; // TILE * TILE

    #[quanta::shared]
    let tile_b: [f32; 256]; // TILE * TILE

    let row = quark_id() / n;
    let col = quark_id() % n;
    let lid = local_id();
    let local_row = lid / TILE;
    let local_col = lid % TILE;

    let mut sum = 0.0f32;
    let num_tiles = (k + TILE - 1u32) / TILE;

    for t in 0..num_tiles {
        // Load tile of A into shared memory
        let a_col = t * TILE + local_col;
        if row < m && a_col < k {
            tile_a[local_row * TILE + local_col] = a[row * k + a_col];
        } else {
            tile_a[local_row * TILE + local_col] = 0.0f32;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE + local_row;
        if b_row < k && col < n {
            tile_b[local_row * TILE + local_col] = b[b_row * n + col];
        } else {
            tile_b[local_row * TILE + local_col] = 0.0f32;
        }

        barrier();

        // Compute partial dot product from this tile
        for i in 0..TILE {
            sum += tile_a[local_row * TILE + i] * tile_b[i * TILE + local_col];
        }

        barrier();
    }

    if row < m && col < n {
        c[row * n + col] = sum;
    }
}
```

## Host code

```rust
fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;

    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;
    let tile: u32 = 16;

    // Initialize matrices
    let a_data: Vec<f32> = (0..(m * k) as usize)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..(k * n) as usize)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();

    let a = gpu.field::<f32>((m * k) as usize)?;
    let b = gpu.field::<f32>((k * n) as usize)?;
    let c = gpu.field::<f32>((m * n) as usize)?;

    a.write(&a_data)?;
    b.write(&b_data)?;

    let mut wave = matmul(&gpu)?;
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &c);
    wave.set_value(3, m);
    wave.set_value(4, n);
    wave.set_value(5, k);

    // Dispatch with TILE x TILE workgroups covering the output matrix
    let groups_x = (n + tile - 1) / tile;
    let groups_y = (m + tile - 1) / tile;
    let mut pulse = gpu.wave_dispatch(&wave, [groups_x * groups_y, 1, 1])?;
    pulse.wait()?;

    let result = c.read()?;
    println!("C[0,0] = {}", result[0]);
    println!("C[511,511] = {}", result[511 * n as usize + 511]);
    Ok(())
}
```

## Why tiling matters

Without tiling, each quark reads an entire row of A and column of B from global
memory. For 1024x1024 matrices, that is 2048 global loads per output element.

With TILE=16 tiling:
- Load a 16x16 block of A and B into shared memory (fast, ~100x lower latency)
- All 256 quarks in the tile reuse the same shared data
- Global memory reads drop from 2048 to 128 per element (16 tiles x 8 bytes each)

The two `barrier()` calls ensure:
1. All quarks have finished loading before computation begins
2. All quarks have finished computing before the next tile overwrites shared memory
