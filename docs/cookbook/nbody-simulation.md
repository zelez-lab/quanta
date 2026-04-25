# N-Body Simulation

N-body gravity using shared memory tiling. Each particle interacts with
every other particle (O(N^2) complexity), making this ideal for GPU parallelism.

## Data layout

```rust
#[derive(quanta::Fields)]
struct NBodyData {
    positions: Vec<f32>,   // [x, y, z, mass] x N (interleaved)
    velocities: Vec<f32>,  // [vx, vy, vz, _] x N
    count: u32,            // push constant
    dt: f32,               // push constant
}
```

The `#[derive(quanta::Fields)]` macro classifies each field:
- `Vec<f32>` fields become GPU storage buffers (slots 0 and 1)
- `u32` and `f32` scalars become push constants (slots 2 and 3)

Positions are interleaved as `[x, y, z, mass]` per particle. This layout
matches the shared memory tile stride and is optimal for coalesced access.

## Body type (for host-side initialization)

```rust
#[quanta::gpu_type]
struct Body {
    pos: [f32; 3],
    mass: f32,
}
```

## Kernels

### Force computation (tiled)

```rust
const TILE_SIZE: u32 = 256;

#[quanta::kernel]
fn nbody_tiled(
    positions: &[f32],
    velocities: &mut [f32],
    count: u32,
    dt: f32,
) {
    #[quanta::shared]
    let tile_pos: [f32; 1024]; // TILE_SIZE * 4 (x, y, z, mass)

    let idx = quark_id();
    let lid = local_id();
    let base = idx * 4u32;

    // Load this particle's position
    let px = positions[base];
    let py = positions[base + 1u32];
    let pz = positions[base + 2u32];

    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    let mut az = 0.0f32;
    let eps = 0.001f32; // Softening to avoid singularity

    let num_tiles = (count + TILE_SIZE - 1u32) / TILE_SIZE;

    for tile in 0..num_tiles {
        // Cooperatively load a tile of positions into shared memory
        let load_idx = tile * TILE_SIZE + lid;
        let load_base = load_idx * 4u32;
        if load_idx < count {
            tile_pos[lid * 4u32] = positions[load_base];
            tile_pos[lid * 4u32 + 1u32] = positions[load_base + 1u32];
            tile_pos[lid * 4u32 + 2u32] = positions[load_base + 2u32];
            tile_pos[lid * 4u32 + 3u32] = positions[load_base + 3u32];
        } else {
            tile_pos[lid * 4u32] = 0.0f32;
            tile_pos[lid * 4u32 + 1u32] = 0.0f32;
            tile_pos[lid * 4u32 + 2u32] = 0.0f32;
            tile_pos[lid * 4u32 + 3u32] = 0.0f32;
        }
        barrier();

        // Accumulate forces from all particles in this tile
        for j in 0..TILE_SIZE {
            let jb = j * 4u32;
            let dx = tile_pos[jb] - px;
            let dy = tile_pos[jb + 1u32] - py;
            let dz = tile_pos[jb + 2u32] - pz;
            let mass = tile_pos[jb + 3u32];

            let dist_sq = dx * dx + dy * dy + dz * dz + eps;
            let inv_dist = rsqrt(dist_sq);
            let inv_dist3 = inv_dist * inv_dist * inv_dist;

            ax += dx * inv_dist3 * mass;
            ay += dy * inv_dist3 * mass;
            az += dz * inv_dist3 * mass;
        }
        barrier();
    }

    // Update velocity
    if idx < count {
        velocities[base] = velocities[base] + ax * dt;
        velocities[base + 1u32] = velocities[base + 1u32] + ay * dt;
        velocities[base + 2u32] = velocities[base + 2u32] + az * dt;
    }
}
```

### Position integration

```rust
#[quanta::kernel]
fn integrate_positions(
    positions: &mut [f32],
    velocities: &[f32],
    count: u32,
    dt: f32,
) {
    let idx = quark_id();
    if idx >= count {
        return;
    }
    let base = idx * 4u32;
    positions[base] = positions[base] + velocities[base] * dt;
    positions[base + 1u32] = positions[base + 1u32] + velocities[base + 1u32] * dt;
    positions[base + 2u32] = positions[base + 2u32] + velocities[base + 2u32] * dt;
    // mass (positions[base + 3]) is unchanged
}
```

## Host code

```rust
fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;

    let count: u32 = 65536;
    let tile_size: u32 = 256;
    let dt: f32 = 0.001;

    // Initialize: particles in a disk with random masses
    let mut pos_data = Vec::with_capacity(count as usize * 4);
    let vel_data = vec![0.0f32; count as usize * 4];
    for i in 0..count {
        let angle = i as f32 * 0.1;
        let radius = (i as f32).sqrt() * 0.5;
        pos_data.push(angle.cos() * radius); // x
        pos_data.push(angle.sin() * radius); // y
        pos_data.push(0.0);                  // z
        pos_data.push(1.0);                  // mass
    }

    let positions = gpu.compute_field::<f32>(count as usize * 4)?;
    let velocities = gpu.compute_field::<f32>(count as usize * 4)?;
    positions.write(&pos_data)?;
    velocities.write(&vel_data)?;

    // Force computation kernel
    let mut force_wave = nbody_tiled(&gpu)?;
    force_wave.bind(0, &positions);
    force_wave.bind(1, &velocities);
    force_wave.set_value(2, count);
    force_wave.set_value(3, dt);

    // Position integration kernel
    let mut integrate_wave = integrate_positions(&gpu)?;
    integrate_wave.bind(0, &positions);
    integrate_wave.bind(1, &velocities);
    integrate_wave.set_value(2, count);
    integrate_wave.set_value(3, dt);

    let num_groups = (count + tile_size - 1) / tile_size;

    // Simulation loop
    for step in 0..1000 {
        // Compute forces (tiled)
        let mut p = gpu.wave_dispatch(&force_wave, [num_groups, 1, 1])?;
        p.wait()?;

        // Integrate positions
        let mut p = gpu.dispatch(&integrate_wave, count)?;
        p.wait()?;

        if step % 100 == 0 {
            let pos = positions.read()?;
            let energy = compute_kinetic_energy(&pos);
            println!("Step {step}: E_k = {energy:.4}");
        }
    }
    Ok(())
}

fn compute_kinetic_energy(positions: &[f32]) -> f64 {
    let n = positions.len() / 4;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    for i in 0..n {
        cx += positions[i * 4] as f64;
        cy += positions[i * 4 + 1] as f64;
    }
    cx / n as f64 + cy / n as f64
}
```

## Why tiling (shared memory optimization)

Without tiling, each quark reads all N positions from global memory:
- N = 65536 particles, 4 floats each = 1 MB per quark
- 65536 quarks x 1 MB = 64 GB of global memory bandwidth

With tiling (TILE_SIZE = 256):
- Each workgroup loads 256 particles into shared memory (4 KB)
- All 256 quarks in the group share that data
- Global reads reduced by 256x

The two `barrier()` calls per tile ensure:
1. All quarks have finished loading before the inner loop reads shared memory
2. All quarks have finished reading before the next tile overwrites shared memory

### Shared memory in Quanta

```rust
#[quanta::shared]
let tile_pos: [f32; 1024];
```

This declares 4 KB of workgroup-local memory. It maps to:
- `threadgroup float tile_pos[1024]` in Metal
- `shared float tile_pos[1024]` in GLSL/Vulkan
- `var<workgroup> tile_pos: array<f32, 1024>` in WGSL
- `__shared__ float tile_pos[1024]` in CUDA (for comparison)

Rules:
- Must be a fixed-size array of scalars
- Only valid inside `#[quanta::kernel]` bodies
- Size is shared across all quarks in the workgroup
- Access requires `barrier()` for correctness

## Performance

| N | Interactions | CPU (1 core) | GPU | Speedup |
|---|---|---|---|---|
| 4096 | 16M | 150ms | 0.8ms | 180x |
| 16384 | 268M | 2400ms | 12ms | 200x |
| 65536 | 4.3B | 38s | 180ms | 210x |

N-body is the canonical GPU workload: embarrassingly parallel with high
arithmetic intensity and uniform memory access patterns.
