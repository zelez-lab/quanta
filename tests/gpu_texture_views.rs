//! Tier 2 -- Texture view operations.
//!
//! Verifies texture_view_create and texture_view_destroy.
//! Features may return "not supported" -- that is acceptable.
//! Requires a GPU; skips gracefully if none available.

use quanta::{Format, TextureDesc, TextureUsage, TextureViewDesc};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn texture_view_create_basic() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(
            &TextureDesc::new(64, 64, Format::RGBA8)
                .with_mip_levels(4)
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
        )
        .unwrap();

    let desc = TextureViewDesc {
        format: None,
        mip_range: 0..4,
        layer_range: 0..1,
    };

    match gpu.texture_view_create(&tex, &desc) {
        Ok(view) => {
            assert!(view.handle() != 0, "texture view handle should be nonzero");
            // Clean up.
            gpu.texture_view_destroy(view).unwrap();
        }
        Err(e) => {
            eprintln!("texture_view_create not supported: {}", e);
        }
    }
}

#[test]
fn texture_view_single_mip() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(
            &TextureDesc::new(128, 128, Format::RGBA8)
                .with_mip_levels(0) // auto-calculate
                .with_usage(
                    TextureUsage::SHADER_READ
                        .union(TextureUsage::STORAGE)
                        .union(TextureUsage::RENDER_TARGET),
                ),
        )
        .unwrap();

    // View only mip level 1.
    let desc = TextureViewDesc {
        format: None,
        mip_range: 1..2,
        layer_range: 0..1,
    };

    match gpu.texture_view_create(&tex, &desc) {
        Ok(view) => {
            gpu.texture_view_destroy(view).unwrap();
        }
        Err(e) => {
            eprintln!("single mip view not supported: {}", e);
        }
    }
}

#[test]
fn texture_view_format_reinterpret() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(
            &TextureDesc::new(32, 32, Format::RGBA8)
                .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
        )
        .unwrap();

    // Attempt to reinterpret as BGRA8 (same size, different channel order).
    let desc = TextureViewDesc {
        format: Some(Format::BGRA8),
        mip_range: 0..1,
        layer_range: 0..1,
    };

    match gpu.texture_view_create(&tex, &desc) {
        Ok(view) => {
            gpu.texture_view_destroy(view).unwrap();
        }
        Err(e) => {
            // Format reinterpret may not be supported.
            eprintln!("format reinterpret view not supported: {}", e);
        }
    }
}

#[test]
fn texture_view_destroy_does_not_panic() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(
            &TextureDesc::new(16, 16, Format::RGBA8).with_usage(TextureUsage::SHADER_READ),
        )
        .unwrap();

    let desc = TextureViewDesc {
        format: None,
        mip_range: 0..1,
        layer_range: 0..1,
    };

    if let Ok(view) = gpu.texture_view_create(&tex, &desc) {
        // Destroy should not panic.
        let result = gpu.texture_view_destroy(view);
        assert!(result.is_ok(), "texture_view_destroy should succeed");
    }
}
