#![cfg(feature = "render")]
//! Tier 2 -- Sampler creation and configuration.
//!
//! Verifies sampler_create with various configurations.
//! Requires a GPU; skips gracefully if none available.

use quanta::{AddressMode, CompareOp, Filter, Format, SamplerDesc};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn sampler_default_desc() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = SamplerDesc::default();
    let sampler = gpu.sampler(&desc).unwrap();
    // Sampler should have a valid handle.
    assert!(sampler.handle() != 0, "sampler handle should be nonzero");
}

#[test]
fn sampler_nearest_filter() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = SamplerDesc::default()
        .with_filters(Filter::Nearest, Filter::Nearest)
        .with_mip_filter(Filter::Nearest);
    let sampler = gpu.sampler(&desc).unwrap();
    assert!(sampler.handle() != 0);
}

#[test]
fn sampler_linear_filter() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = SamplerDesc::default()
        .with_filters(Filter::Linear, Filter::Linear)
        .with_mip_filter(Filter::Linear);
    let sampler = gpu.sampler(&desc).unwrap();
    assert!(sampler.handle() != 0);
}

#[test]
fn sampler_repeat_address_mode() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = SamplerDesc::default().with_address_modes(AddressMode::Repeat, AddressMode::Repeat);
    let sampler = gpu.sampler(&desc).unwrap();
    assert!(sampler.handle() != 0);
}

#[test]
fn sampler_mirror_repeat_address_mode() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = SamplerDesc::default()
        .with_address_modes(AddressMode::MirrorRepeat, AddressMode::MirrorRepeat);
    let sampler = gpu.sampler(&desc).unwrap();
    assert!(sampler.handle() != 0);
}

#[test]
fn sampler_anisotropy() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = SamplerDesc::default().with_max_anisotropy(16);
    let sampler = gpu.sampler(&desc).unwrap();
    assert!(sampler.handle() != 0);
}

#[test]
fn sampler_comparison_for_shadow() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Comparison sampler for shadow mapping.
    let desc = SamplerDesc::default()
        .with_filters(Filter::Linear, Filter::Linear)
        .with_compare(CompareOp::LessEqual);
    let sampler = gpu.sampler(&desc).unwrap();
    assert!(sampler.handle() != 0);
}

#[test]
fn sampler_used_in_render_pass() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Create a texture and sampler, then bind in a render pass.
    let tex = gpu.texture(16, 16).unwrap();
    let pixels = vec![128u8; 16 * 16 * 4];
    tex.write(&pixels).unwrap();

    let target = gpu.render_target(16, 16, Format::RGBA8).unwrap();

    // Bind sampler and texture into the render pass.
    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .texture(0, &tex)
        .sampler(0, SamplerDesc::default())
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
}
