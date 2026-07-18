# Image Blur

Box blur on a packed-RGBA8 image, entirely in a compute kernel. Demonstrates the
storage-texture read/write intrinsics (`texture_load_2d` / `texture_write_2d`)
and the `pack_unorm4x8` / `unpack_unorm4x8_*` channel intrinsics.

A colour image is an **RGBA8 storage texture**, which the kernel sees as
`&mut Texture2D<u32>`: each texel crosses the boundary as one packed
`0xAABBGGRR` `u32` (byte order R, G, B, A). There is **no read-only RGBA8
spelling** — the source you only read is still `&mut Texture2D<u32>`, read with
`texture_load_2d` (a storage read). See the
[`#[quanta::kernel]` texture parameters](../../reference/macros.md#quantakernel).

## Kernel

```rust
#[quanta::kernel]
fn box_blur(
    input: &mut Texture2D<u32>,
    output: &mut Texture2D<u32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;

    let mut sum_r = 0.0f32;
    let mut sum_g = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut count = 0u32;

    for dy in 0..(2u32 * radius + 1u32) {
        for dx in 0..(2u32 * radius + 1u32) {
            let sx = x + dx - radius; // wraps below 0 in u32; the guard filters it
            let sy = y + dy - radius;
            if sx < width && sy < height {
                // Unpack the packed RGBA8 texel to unorm floats in [0, 1].
                let texel = texture_load_2d(input, sx, sy);
                sum_r += unpack_unorm4x8_r(texel);
                sum_g += unpack_unorm4x8_g(texel);
                sum_b += unpack_unorm4x8_b(texel);
                count += 1u32;
            }
        }
    }

    let inv = 1.0f32 / (count as f32);
    // Repack the averaged channels into one RGBA8 texel (opaque alpha).
    let out = pack_unorm4x8(sum_r * inv, sum_g * inv, sum_b * inv, 1.0f32);
    texture_write_2d(output, x, y, out);
}
```

The texture dimensions arrive as the `width` / `height` scalars (push
constants). The `texture_size` intrinsic exists only on the CPU reference
device, so pass the dimensions explicitly for a portable kernel.

## Host code

```rust
use quanta::{Format, TextureDesc, TextureUsage};

fn main() {
    let gpu = quanta::init().unwrap();

    // RGBA8 read-write storage textures need Tier-2 support on Metal; guard for
    // it (native Vulkan and the CPU device always support them).
    if !gpu.supports_compute_textures() {
        eprintln!("compute storage textures unsupported on this device");
        return;
    }

    let width: u32 = 1920;
    let height: u32 = 1080;
    let pixel_count = (width * height) as usize;

    // Both textures are read-write storage images (RGBA8). The input is only
    // read in the kernel, but an RGBA8 source has no read-only spelling, so it
    // still needs SHADER_WRITE usage.
    let usage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);
    let input_tex = gpu.create_texture(
        &TextureDesc::new(width, height, Format::RGBA8).with_usage(usage),
    ).unwrap();
    let output_tex = gpu.create_texture(
        &TextureDesc::new(width, height, Format::RGBA8).with_usage(usage),
    ).unwrap();

    // Generate a gradient test image: 4 bytes per texel, R, G, B, A order.
    let mut pixels = vec![0u8; pixel_count * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            pixels[idx]     = (255.0 * x as f32 / width as f32) as u8;  // R
            pixels[idx + 1] = (255.0 * y as f32 / height as f32) as u8; // G
            pixels[idx + 2] = 128;                                      // B
            pixels[idx + 3] = 255;                                      // A
        }
    }
    input_tex.write(&pixels).unwrap();

    // Create and dispatch the blur kernel.
    let mut wave = box_blur(&gpu).unwrap();
    wave.bind_texture(0, &input_tex);
    wave.bind_texture(1, &output_tex);
    wave.set_value(2, width);
    wave.set_value(3, height);
    wave.set_value(4, 3u32); // radius = 3 -> 7x7 box

    let mut pulse = gpu.dispatch(&wave, pixel_count as u32).unwrap();
    pulse.wait().unwrap();

    // Read the blurred result back as RGBA8 bytes.
    let result = output_tex.read().unwrap();
    println!("Blur complete, {} bytes output", result.len());
}
```

## Texture operations in kernels

The compute (kernel) texture intrinsics — a different set from the shader
`sample` intrinsic. The full contract is in the
[`#[quanta::kernel]` reference](../../reference/macros.md#quantakernel).

| Function | Description |
|----------|-------------|
| `texture_load_2d(tex, x, y)` | Storage read of texel `(x, y)`. On `&mut Texture2D<u32>`, the whole RGBA8 texel as a packed `0xAABBGGRR` u32; on `&mut/&Texture2D<f32>`, the R channel as `f32` |
| `texture_write_2d(tex, x, y, v)` | Write texel `(x, y)`. On `&mut Texture2D<u32>`, `v: u32` packed RGBA8; on `&mut Texture2D<f32>`, `v: f32` into the R channel |
| `texture_sample_2d(tex, x, y)` | Sampled read of a `&Texture2D` slot through the fixed compute sampler (nearest, clamp-to-edge, unnormalized texel coords); returns the R channel as `f32` |
| `pack_unorm4x8(r, g, b, a)` | Pack four unorm `f32` channels into a packed RGBA8 `u32` |
| `unpack_unorm4x8_r/_g/_b/_a(v)` | Unpack one channel of a packed RGBA8 `u32` as `f32` in `[0, 1]` |

> **Note.** `texture_load_2d` / `texture_write_2d` on a `&mut Texture2D<f32>`
> (R32Float) storage texture carry a **single** channel (R). To blur colour,
> use the packed-RGBA8 (`&mut Texture2D<u32>`) form above and unpack/repack the
> four channels.

## Separable optimization

For production use, split into horizontal and vertical passes. Each pass reads
the packed RGBA8 texel, averages along one axis, and writes back:

```rust
#[quanta::kernel]
fn blur_horizontal(
    input: &mut Texture2D<u32>,
    output: &mut Texture2D<u32>,
    width: u32,
    radius: u32,
) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;

    let mut sum_r = 0.0f32;
    let mut sum_g = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut count = 0u32;

    for dx in 0..(2u32 * radius + 1u32) {
        let sx = x + dx - radius;
        if sx < width {
            let texel = texture_load_2d(input, sx, y);
            sum_r += unpack_unorm4x8_r(texel);
            sum_g += unpack_unorm4x8_g(texel);
            sum_b += unpack_unorm4x8_b(texel);
            count += 1u32;
        }
    }

    let inv = 1.0f32 / (count as f32);
    let out = pack_unorm4x8(sum_r * inv, sum_g * inv, sum_b * inv, 1.0f32);
    texture_write_2d(output, x, y, out);
}
```

This reduces complexity from O(radius²) to O(radius) per pixel: run
`blur_horizontal` into a scratch texture, then a matching `blur_vertical`
(iterating `dy` over the column) into the destination.
