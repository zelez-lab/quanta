//! Tier 2 -- Multiple render target (MRT) tests.
//!
//! Verifies rendering to multiple color targets simultaneously.
//! Requires a GPU; skips gracefully if none available.

use quanta::render_pass::{ColorTarget, DepthTarget};
use quanta::{Color, Format, LoadOp, StoreOp, TextureDesc, TextureUsage};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn mrt_two_color_targets() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16;
    let h = 16;

    let target0 = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let target1 = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pass = gpu.render_begin(&target0).unwrap();

    // Set up two color targets with different clear colors.
    pass.set_color_targets(vec![
        ColorTarget {
            texture: target0.handle(),
            load_op: LoadOp::Clear(Color::rgb(1.0, 0.0, 0.0)),
            store_op: StoreOp::Store,
        },
        ColorTarget {
            texture: target1.handle(),
            load_op: LoadOp::Clear(Color::rgb(0.0, 1.0, 0.0)),
            store_op: StoreOp::Store,
        },
    ]);

    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // Both textures should be readable without error.
    let pixels0 = gpu.texture_read(&target0).unwrap();
    let pixels1 = gpu.texture_read(&target1).unwrap();

    assert_eq!(pixels0.len(), (w * h * 4) as usize);
    assert_eq!(pixels1.len(), (w * h * 4) as usize);
}

#[test]
fn mrt_clear_different_colors() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 8;
    let h = 8;

    let target0 = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let target1 = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pass = gpu.render_begin(&target0).unwrap();

    pass.set_color_targets(vec![
        ColorTarget {
            texture: target0.handle(),
            load_op: LoadOp::Clear(Color::rgb(0.0, 0.0, 1.0)), // blue
            store_op: StoreOp::Store,
        },
        ColorTarget {
            texture: target1.handle(),
            load_op: LoadOp::Clear(Color::rgb(1.0, 1.0, 0.0)), // yellow
            store_op: StoreOp::Store,
        },
    ]);

    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();
}

#[test]
fn mrt_store_op_dont_care() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 8;
    let h = 8;

    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pass = gpu.render_begin(&target).unwrap();

    // DontCare store -- should not crash.
    pass.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Clear(Color::WHITE),
        store_op: StoreOp::DontCare,
    }]);

    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();
}

#[test]
fn mrt_load_op_load() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 8;
    let h = 8;

    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    // First pass: clear to white.
    let mut pass1 = gpu.render_begin(&target).unwrap();
    pass1.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Clear(Color::WHITE),
        store_op: StoreOp::Store,
    }]);
    let mut pulse1 = gpu.render_end(pass1).unwrap();
    gpu.wait(&mut pulse1).unwrap();

    // Second pass: load existing contents (should not clear).
    let mut pass2 = gpu.render_begin(&target).unwrap();
    pass2.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Load,
        store_op: StoreOp::Store,
    }]);
    let mut pulse2 = gpu.render_end(pass2).unwrap();
    gpu.wait(&mut pulse2).unwrap();

    let pixels = gpu.texture_read(&target).unwrap();
    assert_eq!(pixels.len(), (w * h * 4) as usize);
}

#[test]
fn mrt_load_op_dont_care() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 8;
    let h = 8;

    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::DontCare,
        store_op: StoreOp::Store,
    }]);

    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();
}

#[test]
fn mrt_with_depth_target() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16;
    let h = 16;

    let color = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let depth = gpu
        .create_texture(&TextureDesc {
            width: w,
            height: h,
            format: Format::Depth32Float,
            usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
            ..TextureDesc::default()
        })
        .unwrap();

    let mut pass = gpu.render_begin(&color).unwrap();

    pass.set_color_targets(vec![ColorTarget {
        texture: color.handle(),
        load_op: LoadOp::Clear(Color::BLACK),
        store_op: StoreOp::Store,
    }]);

    pass.set_depth_target(DepthTarget {
        texture: depth.handle(),
        load_op: LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 0.0)),
        store_op: StoreOp::Store,
        stencil_load_op: LoadOp::DontCare,
        stencil_store_op: StoreOp::DontCare,
    });

    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();
}
