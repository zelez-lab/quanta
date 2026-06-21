#![cfg(feature = "render")]
//! Tier 2 -- Debug label operations.
//!
//! Verifies debug_push and debug_pop do not panic.
//! Requires a GPU; skips gracefully if none available.

use quanta::Format;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn debug_push_pop_no_panic() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Debug labels are no-ops in most drivers but must not panic.
    gpu.debug_push("test group");
    gpu.debug_pop();
}

#[test]
fn debug_nested_push_pop() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    gpu.debug_push("outer");
    gpu.debug_push("inner");
    gpu.debug_pop();
    gpu.debug_pop();
}

#[test]
fn debug_push_pop_around_dispatch() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 32;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    gpu.debug_push("compute pass");

    // A simple field write/read within a debug scope.
    let data = vec![1.0f32; count];
    field.write(&data).unwrap();
    let result = field.read().unwrap();
    assert_eq!(result, data);

    gpu.debug_pop();
}

#[test]
fn debug_push_pop_in_render_pass() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();

    // Debug labels inside render pass.
    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .debug_push("render section")
        .debug_pop()
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
}

#[test]
fn debug_push_empty_label() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Empty label should not panic.
    gpu.debug_push("");
    gpu.debug_pop();
}
