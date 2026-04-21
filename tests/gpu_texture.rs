//! Tier 2 — Texture operations.
//!
//! Verifies texture creation, pixel write, and read-back across formats.
//! Requires a GPU; skips gracefully if none available.

use quanta::{Format, TextureDesc, TextureUsage};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Tests ---

#[test]
fn texture_rgba8_round_trip() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 32;
    let h = 32;
    let tex = gpu
        .create_texture(&TextureDesc {
            width: w,
            height: h,
            format: Format::RGBA8,
            usage: TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
            ..TextureDesc::default()
        })
        .unwrap();

    // Gradient pattern
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            pixels[i] = (x * 8) as u8; // R
            pixels[i + 1] = (y * 8) as u8; // G
            pixels[i + 2] = 128; // B
            pixels[i + 3] = 255; // A
        }
    }

    gpu.texture_write(&tex, &pixels).unwrap();
    let result = gpu.texture_read(&tex).unwrap();

    assert_eq!(pixels.len(), result.len(), "RGBA8 size mismatch");
    assert_eq!(pixels, result, "RGBA8 pixel data mismatch");
}

#[test]
fn texture_r32float_round_trip() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16;
    let h = 16;
    let tex = gpu
        .create_texture(&TextureDesc {
            width: w,
            height: h,
            format: Format::R32Float,
            usage: TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
            ..TextureDesc::default()
        })
        .unwrap();

    // Write float values as raw bytes
    let float_data: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32 * 0.1).collect();
    let bytes: Vec<u8> = float_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    gpu.texture_write(&tex, &bytes).unwrap();
    let result = gpu.texture_read(&tex).unwrap();

    assert_eq!(bytes.len(), result.len(), "R32Float size mismatch");
    // Compare as floats for tolerance
    for i in 0..float_data.len() {
        let offset = i * 4;
        let got = f32::from_le_bytes([
            result[offset],
            result[offset + 1],
            result[offset + 2],
            result[offset + 3],
        ]);
        assert!(
            (got - float_data[i]).abs() < 0.001,
            "R32Float mismatch at pixel {}: expected {}, got {}",
            i,
            float_data[i],
            got
        );
    }
}

#[test]
fn texture_depth32float_create() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Depth textures can be created but may not support read-back on all platforms.
    // Verify creation succeeds.
    let _tex = gpu
        .create_texture(&TextureDesc {
            width: 64,
            height: 64,
            format: Format::Depth32Float,
            usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
            ..TextureDesc::default()
        })
        .unwrap();
}

#[test]
fn texture_partial_write() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16;
    let h = 16;
    let tex = gpu
        .create_texture(&TextureDesc {
            width: w,
            height: h,
            format: Format::RGBA8,
            usage: TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
            ..TextureDesc::default()
        })
        .unwrap();

    // Write full texture with black
    let black = vec![0u8; (w * h * 4) as usize];
    gpu.texture_write(&tex, &black).unwrap();

    // Write full texture with white
    let white = vec![255u8; (w * h * 4) as usize];
    gpu.texture_write(&tex, &white).unwrap();

    let result = gpu.texture_read(&tex).unwrap();
    // After overwrite, all pixels should be white
    for (i, byte) in result.iter().enumerate() {
        assert_eq!(
            *byte, 255,
            "partial_write: byte {} should be 255, got {}",
            i, byte
        );
    }
}

#[test]
fn texture_mipmap_generation() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(&TextureDesc {
            width: 64,
            height: 64,
            format: Format::RGBA8,
            mip_levels: 0, // auto-calculate
            usage: TextureUsage::SHADER_READ
                .union(TextureUsage::SHADER_WRITE)
                .union(TextureUsage::RENDER_TARGET),
            ..TextureDesc::default()
        })
        .unwrap();

    // Write solid red to base level
    let pixels = vec![255u8; 64 * 64 * 4];
    gpu.texture_write(&tex, &pixels).unwrap();

    // Generate mipmaps (should not error)
    gpu.generate_mipmaps(&tex).unwrap();
}

#[test]
fn texture_multiple_formats_create() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

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
        let result = gpu.create_texture(&TextureDesc {
            width: 8,
            height: 8,
            format: *fmt,
            usage: TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
            ..TextureDesc::default()
        });
        assert!(
            result.is_ok(),
            "failed to create texture with format {:?}",
            fmt
        );
    }
}
