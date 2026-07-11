//! Sub-region texture uploads (`Texture::write_region`).
//!
//! The region path is the glyph-atlas hot path: a few new glyphs per
//! frame must land in a large atlas without re-uploading the whole
//! texture. Verifies the region lands exactly where addressed, the
//! rest of the texture is preserved, and the API-level validation
//! rejects out-of-bounds and mis-sized writes.
//! Requires a GPU; skips gracefully if none available.

use quanta::{Format, TextureDesc, TextureUsage};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// Full round-trip on any backend: base fill, region write, readback.
fn region_roundtrip(gpu: &quanta::Gpu) {
    if !gpu.supports_texture_write_region() {
        eprintln!("skipping: write_region not supported on this backend");
        return;
    }

    let w = 8u32;
    let h = 8u32;
    let tex = gpu
        .create_texture(
            &TextureDesc::new(w, h, Format::RGBA8).with_usage(TextureUsage::SHADER_READ),
        )
        .unwrap();

    // Base fill: every pixel (10, 20, 30, 40).
    let base: Vec<u8> = (0..w * h).flat_map(|_| [10u8, 20, 30, 40]).collect();
    tex.write(&base).unwrap();

    // Region write: 3x2 block at (2, 3), every pixel (200, 100, 50, 255).
    let (ox, oy, rw, rh) = (2u32, 3u32, 3u32, 2u32);
    let patch: Vec<u8> = (0..rw * rh).flat_map(|_| [200u8, 100, 50, 255]).collect();
    tex.write_region((ox, oy), (rw, rh), &patch).unwrap();

    let pixels = tex.read().unwrap();
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let got = &pixels[i..i + 4];
            let inside = x >= ox && x < ox + rw && y >= oy && y < oy + rh;
            let want: [u8; 4] = if inside {
                [200, 100, 50, 255]
            } else {
                [10, 20, 30, 40]
            };
            assert_eq!(
                got,
                want,
                "pixel ({x},{y}) {} the region has wrong contents",
                if inside { "inside" } else { "outside" }
            );
        }
    }
}

#[test]
fn write_region_updates_only_the_region() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    region_roundtrip(&gpu);
}

#[test]
fn write_region_r8_atlas_shape() {
    // R8 with odd offsets exercises the 1-byte-per-pixel row math —
    // the exact shape of a glyph atlas upload.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_texture_write_region() {
        eprintln!("skipping: write_region not supported on this backend");
        return;
    }

    let w = 16u32;
    let h = 16u32;
    let tex = gpu
        .create_texture(&TextureDesc::new(w, h, Format::R8).with_usage(TextureUsage::SHADER_READ))
        .unwrap();
    tex.write(&vec![7u8; (w * h) as usize]).unwrap();

    let (ox, oy, rw, rh) = (5u32, 7u32, 3u32, 4u32);
    let patch = vec![251u8; (rw * rh) as usize];
    tex.write_region((ox, oy), (rw, rh), &patch).unwrap();

    let pixels = tex.read().unwrap();
    for y in 0..h {
        for x in 0..w {
            let inside = x >= ox && x < ox + rw && y >= oy && y < oy + rh;
            let want = if inside { 251 } else { 7 };
            assert_eq!(
                pixels[(y * w + x) as usize],
                want,
                "texel ({x},{y}) has wrong value"
            );
        }
    }
}

#[test]
fn write_region_validates_bounds_and_length() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(
            &TextureDesc::new(8, 8, Format::RGBA8).with_usage(TextureUsage::SHADER_READ),
        )
        .unwrap();

    // Region spills past the right edge.
    let data = vec![0u8; 4 * 4 * 4];
    assert!(
        tex.write_region((6, 0), (4, 4), &data).is_err(),
        "out-of-bounds region must be rejected"
    );
    // Region spills past the bottom edge.
    assert!(
        tex.write_region((0, 6), (4, 4), &data).is_err(),
        "out-of-bounds region must be rejected"
    );
    // Data length does not match the region.
    assert!(
        tex.write_region((0, 0), (4, 4), &data[..15]).is_err(),
        "mis-sized data must be rejected"
    );
    // Zero-size region is a no-op, not an error.
    tex.write_region((0, 0), (0, 4), &[]).unwrap();
}
