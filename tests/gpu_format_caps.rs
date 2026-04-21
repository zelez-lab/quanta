//! Tier 2 -- Format capability queries.
//!
//! Verifies format_caps returns reasonable results for all common formats.
//! Requires a GPU; skips gracefully if none available.

use quanta::Format;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn query_rgba8_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::RGBA8);
    // RGBA8 should be filterable and renderable on any GPU.
    assert!(caps.filterable, "RGBA8 must be filterable");
    assert!(caps.renderable, "RGBA8 must be renderable");
}

#[test]
fn query_bgra8_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::BGRA8);
    assert!(caps.filterable, "BGRA8 must be filterable");
    assert!(caps.renderable, "BGRA8 must be renderable");
}

#[test]
fn query_r8_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::R8);
    assert!(caps.filterable, "R8 must be filterable");
}

#[test]
fn query_r32float_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::R32Float);
    // R32Float should be usable as storage.
    assert!(caps.storage, "R32Float should support storage");
}

#[test]
fn query_rgba16float_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::RGBA16Float);
    assert!(caps.filterable, "RGBA16Float must be filterable");
    assert!(caps.renderable, "RGBA16Float must be renderable");
}

#[test]
fn query_rgba32float_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::RGBA32Float);
    assert!(caps.storage, "RGBA32Float should support storage");
}

#[test]
fn query_depth32float_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Query should not panic. The default implementation reports depth=false;
    // a driver that overrides format_caps may set depth=true for Depth32Float.
    let _caps = gpu.format_caps(Format::Depth32Float);
}

#[test]
fn query_r16float_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::R16Float);
    assert!(caps.filterable, "R16Float must be filterable");
}

#[test]
fn query_rg32float_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.format_caps(Format::RG32Float);
    assert!(caps.storage, "RG32Float should support storage");
}

#[test]
fn all_formats_return_valid_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Every format should return a FormatCaps without panicking.
    let formats = [
        Format::RGBA8,
        Format::BGRA8,
        Format::R8,
        Format::R16Float,
        Format::R32Float,
        Format::RG32Float,
        Format::RGBA16Float,
        Format::RGBA32Float,
        Format::Depth32Float,
        Format::Bc1Rgba,
        Format::Bc3Rgba,
        Format::Bc5Rg,
        Format::Bc7Rgba,
        Format::Astc4x4,
        Format::Astc6x6,
        Format::Astc8x8,
        Format::Etc2Rgb8,
        Format::Etc2Rgba8,
    ];

    for fmt in &formats {
        // Should not panic.
        let _caps = gpu.format_caps(*fmt);
    }
}
