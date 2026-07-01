# Mandelbrot Set

Compute the Mandelbrot fractal. Demonstrates complex arithmetic, branching,
and variable iteration counts per quark.

## Data layout

```rust
#[derive(quanta::Fields)]
struct MandelbrotData {
    output: Vec<u32>,  // iteration counts per pixel
    width: u32,        // push constant
    height: u32,       // push constant
    max_iter: u32,     // push constant
}
```

Three scalars become push constants (set via `wave.set_value`). The `Vec<u32>`
becomes a GPU storage buffer at slot 0.

## Kernel

```rust
#[quanta::kernel]
fn mandelbrot(output: &mut [u32], width: u32, height: u32, max_iter: u32) {
    let idx = quark_id();
    let px = idx % width;
    let py = idx / width;

    // Map pixel to complex plane [-2.5, 1.0] x [-1.0, 1.0]
    let x0 = (px as f32 / width as f32) * 3.5f32 - 2.5f32;
    let y0 = (py as f32 / height as f32) * 2.0f32 - 1.0f32;

    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut iter = 0u32;

    while x * x + y * y <= 4.0f32 && iter < max_iter {
        let tmp = x * x - y * y + x0;
        y = 2.0f32 * x * y + y0;
        x = tmp;
        iter += 1u32;
    }

    output[idx] = iter;
}
```

## Host code

```rust
fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;

    let width: u32 = 3840;
    let height: u32 = 2160;
    let max_iter: u32 = 1000;
    let count = (width * height) as usize;

    let output = gpu.field::<u32>(count)?;

    let mut wave = mandelbrot(&gpu)?;
    wave.bind(0, &output);
    wave.set_value(1, width);
    wave.set_value(2, height);
    wave.set_value(3, max_iter);

    let mut pulse = gpu.dispatch(&wave, count as u32)?;
    pulse.wait()?;

    let iterations = output.read()?;

    // Count pixels in the set (hit max_iter)
    let in_set = iterations.iter().filter(|&&v| v == max_iter).count();
    println!("4K Mandelbrot: {in_set} pixels in set ({:.1}%)",
        in_set as f64 / count as f64 * 100.0);
    Ok(())
}
```

## Coloring the output

Convert iteration counts to colors on the CPU after readback:

```rust
fn iter_to_rgb(iter: u32, max_iter: u32) -> [u8; 3] {
    if iter == max_iter {
        return [0, 0, 0]; // Black for points in the set
    }
    let t = iter as f32 / max_iter as f32;
    let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u8;
    let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u8;
    let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u8;
    [r, g, b]
}
```

## Performance characteristics

The Mandelbrot set is a divergent workload: quarks near the set boundary
iterate 1000 times while quarks far away exit in <10 iterations. This creates
warp divergence (quarks in the same wave execute different iteration counts).

Despite divergence, GPUs handle this well because:
- Quarks that exit early are masked off (hardware predication)
- The massive parallelism (millions of pixels) hides latency
- Typical speedup: 50-200x over single-threaded CPU

## Zooming in

Change the complex plane mapping to zoom:

```rust
let center_x = -0.75f32;
let center_y = 0.0f32;
let zoom = 100.0f32;
let x0 = center_x + (px as f32 / width as f32 - 0.5f32) * 3.5f32 / zoom;
let y0 = center_y + (py as f32 / height as f32 - 0.5f32) * 2.0f32 / zoom;
```
