//! Texture conformance tests — create, write, read, formats.

use quanta::{Format, Gpu, TextureDesc, TextureUsage};

/// RGBA8 texture round-trip — write pixels, read back, verify.
pub fn rgba8_write_read(gpu: &Gpu) {
    let w = 64;
    let h = 64;
    let tex = gpu
        .create_texture(&TextureDesc {
            width: w,
            height: h,
            format: Format::RGBA8,
            usage: TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
            ..TextureDesc::default()
        })
        .unwrap();

    // Checkerboard pattern
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if (x + y) % 2 == 0 {
                pixels[i] = 255; // R
                pixels[i + 1] = 0;
                pixels[i + 2] = 0;
                pixels[i + 3] = 255; // A
            } else {
                pixels[i] = 0;
                pixels[i + 1] = 0;
                pixels[i + 2] = 255; // B
                pixels[i + 3] = 255;
            }
        }
    }

    gpu.texture_write(&tex, &pixels).unwrap();
    let result = gpu.texture_read(&tex).unwrap();

    assert_eq!(pixels.len(), result.len(), "texture size mismatch");
    assert_eq!(pixels, result, "texture data mismatch");
}

/// R8 texture (single channel — glyph atlas format).
pub fn r8_write_read(gpu: &Gpu) {
    let w = 32;
    let h = 32;
    let tex = gpu
        .create_texture(&TextureDesc {
            width: w,
            height: h,
            format: Format::R8,
            usage: TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
            ..TextureDesc::default()
        })
        .unwrap();

    let pixels: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
    gpu.texture_write(&tex, &pixels).unwrap();
    let result = gpu.texture_read(&tex).unwrap();
    assert_eq!(pixels, result, "R8 texture data mismatch");
}

/// Render target creation — verify no errors.
pub fn render_target_create(gpu: &Gpu) {
    let _tex = gpu.render_target(1920, 1080, Format::BGRA8).unwrap();
    let _tex2 = gpu.render_target(800, 600, Format::RGBA16Float).unwrap();
}

/// MSAA render target creation.
pub fn msaa_target_create(gpu: &Gpu) {
    let _tex = gpu.msaa_target(1920, 1080, Format::BGRA8, 4).unwrap();
}

/// Texture with mipmaps.
pub fn mipmap_create(gpu: &Gpu) {
    let tex = gpu
        .create_texture(&TextureDesc {
            width: 256,
            height: 256,
            format: Format::RGBA8,
            mip_levels: 0, // auto-calculate
            usage: TextureUsage::SHADER_READ
                .union(TextureUsage::SHADER_WRITE)
                .union(TextureUsage::RENDER_TARGET),
            ..TextureDesc::default()
        })
        .unwrap();

    // Write base level
    let pixels = vec![128u8; 256 * 256 * 4];
    gpu.texture_write(&tex, &pixels).unwrap();

    // Generate mipmaps
    gpu.generate_mipmaps(&tex).unwrap();
}

/// All format creation — verify each format allocates without error.
pub fn all_formats(gpu: &Gpu) {
    let formats = [
        Format::R8,
        Format::RGBA8,
        Format::BGRA8,
        Format::R16Float,
        Format::R32Float,
        Format::RG32Float,
        Format::RGBA16Float,
        Format::RGBA32Float,
    ];

    for fmt in &formats {
        let _tex = gpu
            .create_texture(&TextureDesc {
                width: 16,
                height: 16,
                format: *fmt,
                ..TextureDesc::default()
            })
            .unwrap();
    }
}

/// Run all texture tests.
pub fn run_all(gpu: &Gpu) {
    rgba8_write_read(gpu);
    r8_write_read(gpu);
    render_target_create(gpu);
    msaa_target_create(gpu);
    mipmap_create(gpu);
    all_formats(gpu);
}
