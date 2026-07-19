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
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
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
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
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
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
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

    // Seed the base level from the CPU, then build the chain. The texture
    // does NOT carry RENDER_TARGET usage: render targets are private
    // (GPU-resident) storage on Metal and reject CPU writes, and this test
    // only needs a CPU-seeded, sampleable, mipmapped texture — SHADER_READ |
    // STORAGE keeps it shared so `write` lands.
    let tex = gpu
        .create_texture(
            &TextureDesc::new(64, 64, Format::RGBA8)
                .with_mip_levels(0) // auto-calculate
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
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
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
        );
        assert!(
            result.is_ok(),
            "failed to create texture with format {:?}",
            fmt
        );
    }
}

/// Render targets are private (GPU-resident) storage on Metal, so a CPU
/// write to one must be rejected — not silently dropped. The error is
/// `NotSupported` on Metal (other backends may accept the write, so this
/// assertion only fires there). Covers both a render-target-only texture
/// and the sampled render-target class (RENDER_TARGET | SHADER_READ), since
/// both are private.
#[test]
fn cpu_write_to_render_target_is_rejected_on_metal() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    // The private-storage render-target contract is Metal-specific. This
    // suite is built with the `metal` feature, so `init()` selects Metal on
    // real hardware; skip only the CPU software fallback (QUANTA_CPU=1),
    // which accepts the write.
    if gpu.caps().vendor == quanta::Vendor::Software {
        eprintln!("skipping: render-target private-storage contract is Metal-only");
        return;
    }

    for usage in [
        TextureUsage::RENDER_TARGET,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    ] {
        let tex = gpu
            .create_texture(&TextureDesc::new(16, 16, Format::RGBA8).with_usage(usage))
            .unwrap();
        let pixels = vec![0u8; 16 * 16 * 4];
        let err = tex
            .write(&pixels)
            .expect_err("CPU write to a render target must be rejected on Metal");
        assert!(
            matches!(err.kind, quanta::QuantaErrorKind::NotSupported(_)),
            "expected NotSupported for a CPU write to a render target, got {:?}",
            err.kind
        );
        // write_region must reject it identically (same contract, same path).
        let patch = vec![0u8; 4 * 4 * 4];
        let err = tex
            .write_region((0, 0), (4, 4), &patch)
            .expect_err("CPU write_region to a render target must be rejected on Metal");
        assert!(
            matches!(err.kind, quanta::QuantaErrorKind::NotSupported(_)),
            "expected NotSupported for write_region to a render target, got {:?}",
            err.kind
        );
    }
}

/// A sampled render target (RENDER_TARGET | SHADER_READ) is private storage
/// on Metal; reading it back must go through the staging-blit path and
/// return a correctly sized buffer (not a `getBytes` failure on private
/// storage). The render suites assert the pixel *values* after a draw; this
/// unit pins that the readback mechanism itself covers the sampled-RT class.
#[test]
fn render_target_readback_returns_full_buffer() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let (w, h) = (24u32, 24u32);
    let tex = gpu
        .create_texture(
            &TextureDesc::new(w, h, Format::RGBA8)
                .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)),
        )
        .unwrap();
    let pixels = tex.read().unwrap();
    assert_eq!(
        pixels.len(),
        (w * h * 4) as usize,
        "sampled render-target readback must return the full pixel buffer"
    );
}
