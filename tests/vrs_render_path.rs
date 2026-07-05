#![cfg(feature = "render")]
//! End-to-end test for the VRS render-encoder path (step 063).
//!
//! Validates slice 1's contract through the typed `RenderBuilder`:
//! when the active backend has wired VRS natively (Vulkan with
//! VK_KHR_fragment_shading_rate, slice 1 commit), the pass submits
//! cleanly. When the backend hasn't wired it (Metal until slice 3,
//! WebGPU never — not in spec), the typed API surfaces NotSupported,
//! never InvalidParam.
//!
//! Run: cargo test --test vrs_render_path --features software
//!
//! Skips quietly when no GPU is available (per the gpu_render.rs
//! convention).

use quanta::RenderGpu;

use quanta::{Format, QuantaErrorKind, ShadingRate};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn render_with_shading_rate_either_succeeds_or_not_supported() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let target = gpu
        .render_target(64, 64, Format::RGBA8)
        .expect("render target alloc");

    let r = gpu
        .render(&target)
        .unwrap()
        .shading_rate(ShadingRate::R2x2)
        .pulse();

    match r {
        Ok(mut pulse) => {
            // Native VRS path is wired and the rate is supported on
            // this hardware. Wait for the pass to complete.
            pulse.wait().expect("VRS render pulse");
        }
        Err(e) => {
            // Backend hasn't wired VRS, the extension is missing,
            // or the rate is unsupported. The category must be
            // NotSupported — never InvalidParam, never Internal.
            assert!(
                matches!(e.kind, QuantaErrorKind::NotSupported(_)),
                "VRS render path returned non-NotSupported error: {:?}",
                e.kind
            );
        }
    }
}

#[test]
fn render_without_shading_rate_unaffected_by_vrs_plumbing() {
    // Sanity check: regular renders (no SetShadingRate op) must
    // continue to work unchanged after slice 1's encoder split.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let target = gpu
        .render_target(64, 64, Format::RGBA8)
        .expect("render target alloc");
    let mut pulse = gpu.render(&target).unwrap().pulse().expect("plain render");
    pulse.wait().expect("plain render pulse");
}
