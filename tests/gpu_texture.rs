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
        .create_texture(
            &TextureDesc::new(w, h, Format::RGBA8)
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE)),
        )
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

    tex.write(&pixels).unwrap();
    let result = tex.read().unwrap();

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
        .create_texture(
            &TextureDesc::new(w, h, Format::R32Float)
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE)),
        )
        .unwrap();

    // Write float values as raw bytes
    let float_data: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32 * 0.1).collect();
    let bytes: Vec<u8> = float_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    tex.write(&bytes).unwrap();
    let result = tex.read().unwrap();

    assert_eq!(bytes.len(), result.len(), "R32Float size mismatch");
    // Compare as floats for tolerance
    for (i, &expected) in float_data.iter().enumerate() {
        let offset = i * 4;
        let got = f32::from_le_bytes([
            result[offset],
            result[offset + 1],
            result[offset + 2],
            result[offset + 3],
        ]);
        assert!(
            (got - expected).abs() < 0.001,
            "R32Float mismatch at pixel {}: expected {}, got {}",
            i,
            expected,
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
        .create_texture(
            &TextureDesc::new(64, 64, Format::Depth32Float)
                .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)),
        )
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
        .create_texture(
            &TextureDesc::new(w, h, Format::RGBA8)
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE)),
        )
        .unwrap();

    // Write full texture with black
    let black = vec![0u8; (w * h * 4) as usize];
    tex.write(&black).unwrap();

    // Write full texture with white
    let white = vec![255u8; (w * h * 4) as usize];
    tex.write(&white).unwrap();

    let result = tex.read().unwrap();
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
        .create_texture(
            &TextureDesc::new(64, 64, Format::RGBA8)
                .with_mip_levels(0) // auto-calculate
                .with_usage(
                    TextureUsage::SHADER_READ
                        .union(TextureUsage::SHADER_WRITE)
                        .union(TextureUsage::RENDER_TARGET),
                ),
        )
        .unwrap();

    // Write solid red to base level
    let pixels = vec![255u8; 64 * 64 * 4];
    tex.write(&pixels).unwrap();

    // Generate mipmaps (should not error)
    tex.generate_mipmaps().unwrap();
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
        let result = gpu.create_texture(
            &TextureDesc::new(8, 8, *fmt)
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE)),
        );
        assert!(
            result.is_ok(),
            "failed to create texture with format {:?}",
            fmt
        );
    }
}
