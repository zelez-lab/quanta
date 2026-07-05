# Image Blur

Box blur on a 2D texture. Demonstrates texture read/write operations.

## Kernel

```rust
#[quanta::kernel]
fn box_blur(input: &Texture2D<f32>, output: &mut Texture2D<f32>, radius: u32) {
    let x = quark_id() % texture_width(input);
    let y = quark_id() / texture_width(input);
    let width = texture_width(input);
    let height = texture_height(input);

    let mut sum_r = 0.0f32;
    let mut sum_g = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut count = 0u32;

    for dy in 0..(2u32 * radius + 1u32) {
        for dx in 0..(2u32 * radius + 1u32) {
            let sx = x + dx - radius;
            let sy = y + dy - radius;
            if sx < width && sy < height {
                let pixel = texture_read(input, sx, sy);
                sum_r += pixel.x;
                sum_g += pixel.y;
                sum_b += pixel.z;
                count += 1u32;
            }
        }
    }

    let inv = 1.0f32 / (count as f32);
    texture_write(output, x, y, vec4(sum_r * inv, sum_g * inv, sum_b * inv, 1.0f32));
}
```

## Host code

```rust
use quanta::{Format, TextureDesc, TextureUsage};

fn main() {
    let gpu = quanta::init().unwrap();

    let width: u32 = 1920;
    let height: u32 = 1080;
    let pixel_count = (width * height) as usize;

    // Create input/output textures
    let input_tex = gpu.create_texture(
        &TextureDesc::new(width, height, Format::RGBA32Float)
            .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE)),
    ).unwrap();

    let output_tex = gpu.create_texture(
        &TextureDesc::new(width, height, Format::RGBA32Float)
            .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE)),
    ).unwrap();

    // Generate gradient test image (RGBA f32)
    let mut pixels = vec![0.0f32; pixel_count * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            pixels[idx] = x as f32 / width as f32;       // R
            pixels[idx + 1] = y as f32 / height as f32;  // G
            pixels[idx + 2] = 0.5;                        // B
            pixels[idx + 3] = 1.0;                        // A
        }
    }

    // Upload pixel data
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
    };
    input_tex.write(bytes).unwrap();

    // Create and dispatch the blur kernel
    let mut wave = box_blur(&gpu).unwrap();
    wave.bind_texture(0, &input_tex);
    wave.bind_texture(1, &output_tex);
    wave.set_value(2, 3u32); // radius = 3 -> 7x7 box

    let mut pulse = gpu.dispatch(&wave, pixel_count as u32).unwrap();
    pulse.wait().unwrap();

    // Read blurred result
    let result = output_tex.read().unwrap();
    println!("Blur complete, {} bytes output", result.len());
}
```

## Texture operations in kernels

| Function | Description |
|----------|-------------|
| `texture_read(tex, x, y)` | Read RGBA from 2D texture at integer coordinates |
| `texture_write(tex, x, y, value)` | Write RGBA to 2D texture |
| `texture_sample(tex, u, v)` | Sample with bilinear filtering (normalized coords) |
| `texture_width(tex)` | Get texture width in pixels |
| `texture_height(tex)` | Get texture height in pixels |

## Separable optimization

For production use, split into horizontal and vertical passes:

```rust
#[quanta::kernel]
fn blur_horizontal(input: &Texture2D<f32>, output: &mut Texture2D<f32>, radius: u32) {
    let x = quark_id() % texture_width(input);
    let y = quark_id() / texture_width(input);
    let width = texture_width(input);
    let mut sum = vec4(0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut count = 0u32;

    for dx in 0..(2u32 * radius + 1u32) {
        let sx = x + dx - radius;
        if sx < width {
            sum += texture_read(input, sx, y);
            count += 1u32;
        }
    }

    texture_write(output, x, y, sum / (count as f32));
}
```

This reduces complexity from O(radius^2) to O(radius) per pixel.
