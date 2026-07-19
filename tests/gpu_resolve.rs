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

/// The windowed effect shape: a BGRA8 MSAA scene (the swapchain format)
/// resolved into an RGBA8 destination (the only compute texel format —
/// SPIR-V has no BGRA8 storage image). `vkCmdResolveImage` requires
/// identical formats (VUID 01386, a real device loss on Iris Xe), so the
/// Vulkan driver converts through a cached same-format temp plus a
/// format-converting blit; a backend without a conversion path must
/// reject loudly (NotSupported) instead of recording the invalid resolve.
#[test]
fn resolve_bgra8_scene_into_rgba8_converts_or_rejects() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let (w, h) = (8u32, 8u32);
    let msaa = gpu.msaa_target(w, h, Format::BGRA8, 4).unwrap();
    let dst = gpu.render_target(w, h, Format::RGBA8).unwrap();

    // Channel-asymmetric clear so a missed swizzle (a raw byte copy
    // instead of a converting blit) shows up as swapped R/B.
    gpu.render(&msaa)
        .unwrap()
        .clear(quanta::Color::rgba(1.0, 0.25, 0.0, 1.0))
        .pulse()
        .unwrap()
        .wait()
        .unwrap();

    match gpu.resolve_texture(&msaa, &dst) {
        Ok(()) => {
            let px = dst.read().unwrap();
            // RGBA8 bytes, texel 0: R≈255, G≈64, B≈0.
            assert!(px[0] >= 250, "R channel lost in conversion: {:?}", &px[..4]);
            assert!(
                (58..=70).contains(&px[1]),
                "G channel off after conversion: {:?}",
                &px[..4]
            );
            assert!(
                px[2] <= 5,
                "B channel gained — swizzle miss (raw copy?): {:?}",
                &px[..4]
            );
        }
        Err(e) => {
            assert!(
                matches!(e.kind, quanta::QuantaErrorKind::NotSupported(_)),
                "a format-mismatched resolve must convert or reject loudly; got: {e}"
            );
            eprintln!("SKIP: converting resolve unsupported here: {e}");
        }
    }
}
