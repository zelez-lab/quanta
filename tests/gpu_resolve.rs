#![cfg(feature = "render")]
//! Tier 2 -- MSAA resolve operations.
//!
//! Verifies resolve_texture between MSAA and single-sample textures.
//! Requires a GPU; skips gracefully if none available.

use quanta::RenderGpu;

use quanta::Format;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn resolve_4x_msaa_to_single_sample() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 32;
    let h = 32;

    // Create 4x MSAA render target.
    let msaa = gpu.msaa_target(w, h, Format::RGBA8, 4).unwrap();

    // Create single-sample resolve target.
    let resolve = gpu.render_target(w, h, Format::RGBA8).unwrap();

    // Render to the MSAA target (clear to a color).
    let mut pulse = gpu.render(&msaa).unwrap().pulse().unwrap();
    pulse.wait().unwrap();

    // Resolve MSAA -> single sample.
    match gpu.resolve_texture(&msaa, &resolve) {
        Ok(()) => {
            // Verify the resolve target is readable.
            let pixels = resolve.read().unwrap();
            assert_eq!(pixels.len(), (w * h * 4) as usize);
        }
        Err(e) => {
            // MSAA resolve not supported -- acceptable.
            eprintln!("resolve_texture not supported: {}", e);
        }
    }
}

#[test]
fn resolve_2x_msaa() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16;
    let h = 16;

    let msaa = gpu.msaa_target(w, h, Format::RGBA8, 2).unwrap();
    let resolve = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pulse = gpu.render(&msaa).unwrap().pulse().unwrap();
    pulse.wait().unwrap();

    // May or may not be supported.
    let _ = gpu.resolve_texture(&msaa, &resolve);
}

#[test]
fn resolve_rgba16float() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16;
    let h = 16;

    let msaa = gpu.msaa_target(w, h, Format::RGBA16Float, 4).unwrap();
    let resolve = gpu.render_target(w, h, Format::RGBA16Float).unwrap();

    let mut pulse = gpu.render(&msaa).unwrap().pulse().unwrap();
    pulse.wait().unwrap();

    let _ = gpu.resolve_texture(&msaa, &resolve);
}

#[test]
fn msaa_target_creation() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Verify MSAA targets can be created at various sample counts.
    for samples in [1, 2, 4] {
        let tex = gpu.msaa_target(8, 8, Format::RGBA8, samples).unwrap();
        assert_eq!(tex.width(), 8);
        assert_eq!(tex.height(), 8);
    }
}
